"""
SROP entrypoint — called by the message route.

This is the core of the assignment. It ties together:
  - Loading session state from DB
  - Running the ADK orchestrator with that state as context
  - Extracting routing decision and tool calls from ADK events
  - Recording the trace
  - Persisting updated session state to DB

The route calls: result = await pipeline.run(session_id, user_message, db)
It receives: PipelineResult(content, routed_to, trace_id)

Design questions you need to answer:
  1. How do you inject SessionState into the ADK agent so it knows the user's context?
     (system prompt injection vs ADK session state vs re-hydrating from message history)
  2. How do you determine WHICH sub-agent handled the turn from ADK's event stream?
  3. How do you capture tool calls (name, args, result) for the trace?
  4. What is your timeout strategy? (see settings.llm_timeout_seconds)
  5. If the DB write for state fails after the LLM responds, what do you do?

See docs/google-adk-guide.md for ADK event stream patterns.
"""
import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents import orchestrator
from app.api.errors import RateLimitedError, SessionNotFoundError, UpstreamTimeoutError
from app.db.models import AgentTrace, IdempotencyRecord, Message, Session
from app.settings import settings
from app.srop.guardrails import REFUSAL_MESSAGE, is_out_of_scope_query
from app.srop.state import SessionState

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _format_sse(event_name: str, payload: dict) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload)}\n\n"


@dataclass
class PipelineResult:
    content: str
    routed_to: str
    trace_id: str


def _ensure_knowledge_citations(content: str, routed_to: str, chunk_ids: list[str]) -> str:
    if routed_to != "knowledge" or not chunk_ids or "[chunk_" in content:
        return content
    citations = ", ".join(f"[{chunk_id}]" for chunk_id in chunk_ids[:3])
    return f"{content}\n\nSources: {citations}"


def _is_resource_exhausted_error(exc: Exception) -> bool:
    return type(exc).__name__ == "_ResourceExhaustedError" or "RESOURCE_EXHAUSTED" in str(exc)


async def _run_agent_turn(
    user_message: str,
    state: SessionState,
) -> orchestrator.OrchestratorTurnResult:
    return await asyncio.wait_for(
        orchestrator.run_turn(user_message=user_message, state=state),
        timeout=settings.llm_timeout_seconds,
    )


async def run(
    session_id: str,
    user_message: str,
    db: AsyncSession,
    idempotency_key: str | None = None,
) -> PipelineResult:
    trace_id = str(uuid.uuid4())
    session = await db.scalar(select(Session).where(Session.session_id == session_id))
    if session is None:
        raise SessionNotFoundError(f"Session not found: {session_id}")

    if idempotency_key is not None:
        existing_record = await db.scalar(
            select(IdempotencyRecord).where(
                IdempotencyRecord.session_id == session_id,
                IdempotencyRecord.idempotency_key == idempotency_key,
            )
        )
        if existing_record is not None:
            return PipelineResult(
                content=existing_record.reply,
                routed_to=existing_record.routed_to,
                trace_id=existing_record.trace_id,
            )

    state = SessionState.from_db_dict(session.state)
    started_at = time.perf_counter()
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        session_id=session_id,
        trace_id=trace_id,
        user_id=state.user_id,
    )
    log.info("pipeline_started", user_message=user_message, idempotency_key=bool(idempotency_key))

    db.add(
        Message(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role="user",
            content=user_message,
        )
    )

    try:
        if is_out_of_scope_query(user_message):
            turn_result = orchestrator.OrchestratorTurnResult(
                content=REFUSAL_MESSAGE,
                routed_to="smalltalk",
                tool_calls=[],
                retrieved_chunk_ids=[],
            )
        else:
            try:
                turn_result = await _run_agent_turn(user_message=user_message, state=state)
            except TimeoutError as exc:
                raise UpstreamTimeoutError(
                    f"LLM did not respond within {settings.llm_timeout_seconds}s"
                ) from exc
            except Exception as exc:
                if _is_resource_exhausted_error(exc):
                    raise RateLimitedError(
                        "Upstream LLM quota was exhausted; retry later."
                    ) from exc
                raise

        state.last_agent = turn_result.routed_to
        state.turn_count += 1
        turn_result.content = _ensure_knowledge_citations(
            turn_result.content,
            turn_result.routed_to,
            turn_result.retrieved_chunk_ids,
        )
        for tool_call in turn_result.tool_calls:
            if tool_call.get("tool_name") != "create_ticket":
                continue
            result = tool_call.get("result")
            if not isinstance(result, dict):
                continue
            ticket_id = result.get("ticket_id")
            if not isinstance(ticket_id, str) or not ticket_id:
                continue
            state.last_ticket_id = ticket_id
            if ticket_id not in state.open_ticket_ids:
                state.open_ticket_ids.append(ticket_id)
        session.state = state.to_db_dict()

        db.add(
            Message(
                message_id=str(uuid.uuid4()),
                session_id=session_id,
                role="assistant",
                content=turn_result.content,
                trace_id=trace_id,
            )
        )
        db.add(
            AgentTrace(
                trace_id=trace_id,
                session_id=session_id,
                routed_to=turn_result.routed_to,
                tool_calls=turn_result.tool_calls,
                retrieved_chunk_ids=turn_result.retrieved_chunk_ids,
                latency_ms=max(int((time.perf_counter() - started_at) * 1000), 0),
            )
        )
        if idempotency_key is not None:
            db.add(
                IdempotencyRecord(
                    record_id=str(uuid.uuid4()),
                    session_id=session_id,
                    idempotency_key=idempotency_key,
                    reply=turn_result.content,
                    routed_to=turn_result.routed_to,
                    trace_id=trace_id,
                )
            )
        await db.commit()
        log.info(
            "pipeline_completed",
            routed_to=turn_result.routed_to,
            retrieved_chunk_count=len(turn_result.retrieved_chunk_ids),
        )
    finally:
        structlog.contextvars.clear_contextvars()

    return PipelineResult(
        content=turn_result.content,
        routed_to=turn_result.routed_to,
        trace_id=trace_id,
    )


async def run_streaming(
    session_id: str,
    user_message: str,
    db: AsyncSession,
    idempotency_key: str | None = None,
) -> AsyncGenerator[str, None]:
    """SSE streaming variant of run().

    Yields Server-Sent Event strings:
      event: token\ndata: {"text": "..."}\n\n   — zero or more partial text events
      event: done\ndata: {"routed_to": ..., "trace_id": ..., "reply": ...}\n\n  — terminal success
      event: error\ndata: {"code": ..., "detail": ...}\n\n              — terminal failure

    SessionNotFoundError is raised *before* the first yield so the caller
    (the route) can still return a regular 404 HTTP response.
    """
    trace_id = str(uuid.uuid4())
    session = await db.scalar(select(Session).where(Session.session_id == session_id))
    if session is None:
        raise SessionNotFoundError(f"Session not found: {session_id}")

    # Idempotency cache hit — replay as a single token + done.
    if idempotency_key is not None:
        existing_record = await db.scalar(
            select(IdempotencyRecord).where(
                IdempotencyRecord.session_id == session_id,
                IdempotencyRecord.idempotency_key == idempotency_key,
            )
        )
        if existing_record is not None:
            yield _format_sse("token", {"text": existing_record.reply})
            yield _format_sse(
                "done",
                {
                    "routed_to": existing_record.routed_to,
                    "trace_id": existing_record.trace_id,
                    "reply": existing_record.reply,
                },
            )
            return

    state = SessionState.from_db_dict(session.state)
    started_at = time.perf_counter()
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        session_id=session_id,
        trace_id=trace_id,
        user_id=state.user_id,
    )
    log.info("pipeline_streaming_started", user_message=user_message)

    db.add(
        Message(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role="user",
            content=user_message,
        )
    )

    try:
        # --- Guardrail fast path ---
        if is_out_of_scope_query(user_message):
            yield _format_sse("token", {"text": REFUSAL_MESSAGE})
            turn_result = orchestrator.OrchestratorTurnResult(
                content=REFUSAL_MESSAGE,
                routed_to="smalltalk",
                tool_calls=[],
                retrieved_chunk_ids=[],
            )
        else:
            # --- Streaming ADK path with whole-turn timeout ---
            turn_result = None
            try:
                async with asyncio.timeout(settings.llm_timeout_seconds):
                    async for kind, payload in orchestrator.run_turn_streaming(
                        user_message=user_message, state=state
                    ):
                        if kind == "token":
                            yield _format_sse("token", {"text": payload})
                        elif kind == "result":
                            turn_result = payload
            except TimeoutError:
                yield _format_sse(
                    "error",
                    {
                        "code": "UPSTREAM_TIMEOUT",
                        "detail": f"LLM did not respond within {settings.llm_timeout_seconds}s",
                    },
                )
                log.warning("pipeline_streaming_timeout")
                structlog.contextvars.clear_contextvars()
                return
            except Exception as exc:
                if not _is_resource_exhausted_error(exc):
                    raise
                yield _format_sse(
                    "error",
                    {
                        "code": "RATE_LIMITED",
                        "detail": "Upstream LLM quota was exhausted; retry later.",
                    },
                )
                log.warning("pipeline_streaming_rate_limited")
                structlog.contextvars.clear_contextvars()
                return

            if turn_result is None:
                turn_result = orchestrator.OrchestratorTurnResult(
                    content="I could not produce a response for that request.",
                    routed_to="smalltalk",
                    tool_calls=[],
                    retrieved_chunk_ids=[],
                )

        # --- DB writes (same as non-streaming path) ---
        state.last_agent = turn_result.routed_to
        state.turn_count += 1
        turn_result.content = _ensure_knowledge_citations(
            turn_result.content,
            turn_result.routed_to,
            turn_result.retrieved_chunk_ids,
        )
        for tool_call in turn_result.tool_calls:
            if tool_call.get("tool_name") != "create_ticket":
                continue
            result = tool_call.get("result")
            if not isinstance(result, dict):
                continue
            ticket_id = result.get("ticket_id")
            if not isinstance(ticket_id, str) or not ticket_id:
                continue
            state.last_ticket_id = ticket_id
            if ticket_id not in state.open_ticket_ids:
                state.open_ticket_ids.append(ticket_id)
        session.state = state.to_db_dict()

        db.add(
            Message(
                message_id=str(uuid.uuid4()),
                session_id=session_id,
                role="assistant",
                content=turn_result.content,
                trace_id=trace_id,
            )
        )
        db.add(
            AgentTrace(
                trace_id=trace_id,
                session_id=session_id,
                routed_to=turn_result.routed_to,
                tool_calls=turn_result.tool_calls,
                retrieved_chunk_ids=turn_result.retrieved_chunk_ids,
                latency_ms=max(int((time.perf_counter() - started_at) * 1000), 0),
            )
        )
        if idempotency_key is not None:
            db.add(
                IdempotencyRecord(
                    record_id=str(uuid.uuid4()),
                    session_id=session_id,
                    idempotency_key=idempotency_key,
                    reply=turn_result.content,
                    routed_to=turn_result.routed_to,
                    trace_id=trace_id,
                )
            )
        await db.commit()
        log.info(
            "pipeline_streaming_completed",
            routed_to=turn_result.routed_to,
        )

        yield _format_sse(
            "done",
            {
                "routed_to": turn_result.routed_to,
                "trace_id": trace_id,
                "reply": turn_result.content,
            },
        )
    finally:
        structlog.contextvars.clear_contextvars()
