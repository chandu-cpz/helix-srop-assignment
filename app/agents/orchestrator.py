"""
SROP Root Orchestrator — Google ADK agent.

Routes every user turn to KnowledgeAgent or AccountAgent via ADK's AgentTool.
This means the LLM decides which tool to call — you do not parse its output.

Intent → sub-agent:
  knowledge:  "how do I X", "what is X", docs questions
  account:    "show my builds", "my account status", usage questions
  smalltalk:  greetings, thanks — root agent handles inline (no tool call)

See docs/google-adk-guide.md for AgentTool pattern and event extraction.
"""
import uuid
from collections.abc import AsyncGenerator
from dataclasses import asdict, dataclass
from typing import Any, Literal

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

from app.agents.tools.account_tools import get_account_status, get_recent_builds
from app.agents.tools.search_docs import search_docs
from app.agents.tools.ticket_tools import create_ticket
from app.settings import settings
from app.srop.state import SessionState

APP_NAME = "helix_srop"

ROOT_INSTRUCTION = """
You are the Helix Support Concierge — a routing agent.
Call the correct specialist tool based on the user's intent.

Intent → tool:
- HOW to do something, WHAT something is, docs/feature questions → knowledge_agent
- Their account, builds, failed builds, status, usage, plan, limits → account_agent
- Requests for human support, escalation, or a support ticket → escalation_agent
- Greetings or off-topic → respond directly, no tool call

Always call a tool when intent matches. Never answer knowledge or account questions yourself.
User context will be in the system message — use it. Never ask the user for user_id or plan_tier.
"""

KNOWLEDGE_INSTRUCTION = """
You are the Helix Knowledge Agent.
Use the search_docs tool for product and documentation questions.
Answer using the retrieved documentation and cite at least one chunk ID in square brackets.
If the search results are empty, say you could not find a matching document.
"""

ACCOUNT_INSTRUCTION = """
You are the Helix Account Agent.
Use account tools for plan, builds, usage, and account status questions.
Use the current user_id from context; never ask the user to provide it again.
If the user asks for failed builds, call get_recent_builds and summarize failures.
Base your answer on tool results and keep it concise.
"""

ESCALATION_INSTRUCTION = """
You are the Helix Escalation Agent.
When the user asks for human help, a support ticket, or escalation, call create_ticket.
Return the created ticket ID and a brief next-step summary.
"""

_ROUTE_BY_TOOL = {
  "search_docs": "knowledge",
  "get_account_status": "account",
  "get_recent_builds": "account",
  "create_ticket": "escalation",
}


@dataclass
class OrchestratorTurnResult:
  content: str
  routed_to: str
  tool_calls: list[dict]
  retrieved_chunk_ids: list[str]


def _build_root_instruction(state: SessionState) -> str:
  return "\n".join(
    [
      ROOT_INSTRUCTION.strip(),
      "",
      "Current user context:",
      f"- user_id: {state.user_id}",
      f"- plan_tier: {state.plan_tier}",
      f"- last_agent: {state.last_agent or 'none'}",
      f"- turn_count: {state.turn_count}",
    ]
  )


def _extract_text(event: Any) -> str:
  content = getattr(event, "content", None)
  parts = getattr(content, "parts", None) or []
  texts = [getattr(part, "text", "") for part in parts if getattr(part, "text", "")]
  return "\n".join(texts).strip()


def _normalize_route(author: str | None) -> str | None:
  if author in {"knowledge_agent", "knowledge"}:
    return "knowledge"
  if author in {"account_agent", "account"}:
    return "account"
  if author in {"escalation_agent", "escalation"}:
    return "escalation"
  if author in {"srop_root", "root", None, ""}:
    return "smalltalk"
  return None


def _infer_routed_to(tool_calls: list[dict[str, Any]], authors: list[str | None]) -> str:
  for call in tool_calls:
    route = _ROUTE_BY_TOOL.get(call.get("tool_name"))
    if route:
      return route

  for author in reversed(authors):
    route = _normalize_route(author)
    if route:
      return route

  return "smalltalk"


# ---------------------------------------------------------------------------
# Internal: build the ADK agent graph + runner for a given turn.
# Returns (runner, adk_session, tool_calls_list, retrieved_chunk_ids_list, authors_list)
# so that both the streaming and non-streaming paths share the same construction.
# ---------------------------------------------------------------------------

async def _build_runner(
  user_message: str,
  state: SessionState,
) -> tuple[
  "Runner",
  Any,  # ADK session
  list[dict[str, Any]],  # tool_calls (mutated in-place by closures)
  list[str],             # retrieved_chunk_ids (mutated in-place)
  list[str | None],      # authors (mutated in-place)
]:
  """Construct traced tool closures, agents, runner and ADK session."""
  tool_calls: list[dict[str, Any]] = []
  retrieved_chunk_ids: list[str] = []
  authors: list[str | None] = []

  async def traced_search_docs(
    query: str,
    k: int = 5,
    product_area: str | None = None,
  ) -> list[dict[str, Any]]:
    """Search Helix product docs and return matching chunks with chunk IDs."""
    chunks = await search_docs(query=query, k=k, product_area=product_area)
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    retrieved_chunk_ids.extend(
      chunk_id for chunk_id in chunk_ids if chunk_id not in retrieved_chunk_ids
    )
    tool_calls.append(
      {
        "tool_name": "search_docs",
        "args": {"query": query, "k": k, "product_area": product_area},
        "result": {"count": len(chunks), "chunk_ids": chunk_ids},
      }
    )
    return [
      {
        "chunk_id": chunk.chunk_id,
        "score": chunk.score,
        "content": chunk.content,
        "metadata": chunk.metadata,
      }
      for chunk in chunks
    ]

  async def traced_get_account_status(user_id: str) -> dict[str, Any]:
    """Get the user's current Helix account status and plan limits."""
    status = await get_account_status(user_id=user_id)
    payload = asdict(status)
    tool_calls.append(
      {
        "tool_name": "get_account_status",
        "args": {"user_id": user_id},
        "result": payload,
      }
    )
    return payload

  async def traced_get_recent_builds(user_id: str, limit: int = 5) -> list[dict[str, Any]]:
    """Get the user's recent Helix builds, newest first."""
    builds = await get_recent_builds(user_id=user_id, limit=limit)
    payload = [asdict(build) for build in builds]
    tool_calls.append(
      {
        "tool_name": "get_recent_builds",
        "args": {"user_id": user_id, "limit": limit},
        "result": {"count": len(payload), "build_ids": [build["build_id"] for build in payload]},
      }
    )
    return payload

  async def traced_create_ticket(
    user_id: str,
    summary: str,
    priority: str = "medium",
  ) -> dict[str, Any]:
    """Create a Helix support ticket when the user asks for human help."""
    ticket = await create_ticket(user_id=user_id, summary=summary, priority=priority)
    tool_calls.append(
      {
        "tool_name": "create_ticket",
        "args": {"user_id": user_id, "summary": summary, "priority": priority},
        "result": ticket,
      }
    )
    return ticket

  knowledge_agent = LlmAgent(
    name="knowledge_agent",
    model=settings.adk_model,
    instruction=KNOWLEDGE_INSTRUCTION,
    tools=[traced_search_docs],
  )
  account_agent = LlmAgent(
    name="account_agent",
    model=settings.adk_model,
    instruction="\n".join(
      [
        ACCOUNT_INSTRUCTION.strip(),
        "",
        "Current user context:",
        f"- user_id: {state.user_id}",
        f"- plan_tier: {state.plan_tier}",
      ]
    ),
    tools=[traced_get_account_status, traced_get_recent_builds],
  )
  escalation_agent = LlmAgent(
    name="escalation_agent",
    model=settings.adk_model,
    instruction="\n".join(
      [
        ESCALATION_INSTRUCTION.strip(),
        "",
        "Current user context:",
        f"- user_id: {state.user_id}",
        f"- plan_tier: {state.plan_tier}",
      ]
    ),
    tools=[traced_create_ticket],
  )
  root_agent = LlmAgent(
    name="srop_root",
    model=settings.adk_model,
    instruction=_build_root_instruction(state),
    tools=[
      AgentTool(agent=knowledge_agent),
      AgentTool(agent=account_agent),
      AgentTool(agent=escalation_agent),
    ],
  )

  session_service = InMemorySessionService()
  runner = Runner(app_name=APP_NAME, agent=root_agent, session_service=session_service)
  adk_session = await session_service.create_session(
    app_name=APP_NAME,
    user_id=state.user_id,
    state=state.to_db_dict(),
    session_id=str(uuid.uuid4()),
  )
  return runner, adk_session, tool_calls, retrieved_chunk_ids, authors


# Sentinel type for the streaming generator
_StreamItem = tuple[Literal["token", "result"], Any]


async def run_turn_streaming(
  user_message: str,
  state: SessionState,
) -> AsyncGenerator[_StreamItem, None]:
  """Async generator that streams ADK events for one turn.

  Yields:
    ("token", text: str)  — for every non-final text-bearing ADK event.
    ("result", OrchestratorTurnResult)  — exactly once as the terminal item.
  """
  runner, adk_session, tool_calls, retrieved_chunk_ids, authors = await _build_runner(
    user_message, state
  )

  final_text = ""
  async for event in runner.run_async(
    user_id=state.user_id,
    session_id=adk_session.id,
    new_message=types.Content(
      role="user",
      parts=[types.Part.from_text(text=user_message)],
    ),
  ):
    authors.append(getattr(event, "author", None))
    if event.is_final_response():
      final_text = _extract_text(event)
    else:
      partial = _extract_text(event)
      if partial:
        yield "token", partial

  routed_to = _infer_routed_to(tool_calls, authors)
  if not final_text:
    final_text = "I could not produce a response for that request."

  yield "result", OrchestratorTurnResult(
    content=final_text,
    routed_to=routed_to,
    tool_calls=tool_calls,
    retrieved_chunk_ids=retrieved_chunk_ids,
  )


async def run_turn(user_message: str, state: SessionState) -> OrchestratorTurnResult:
  """Non-streaming convenience wrapper around run_turn_streaming."""
  result: OrchestratorTurnResult | None = None
  async for kind, payload in run_turn_streaming(user_message=user_message, state=state):
    if kind == "result":
      result = payload
  assert result is not None
  return result
