"""
POST /v1/chat/{session_id} — send a user message, get assistant reply.
"""
from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.srop import pipeline

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    content: str


class ChatResponse(BaseModel):
    reply: str
    routed_to: str   # which sub-agent handled this turn
    trace_id: str


@router.post("/chat/{session_id}", response_model=ChatResponse)
async def chat(
    session_id: str,
    body: ChatRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> ChatResponse | StreamingResponse:
    """
    Run one turn of the SROP pipeline.

    Supports both JSON and SSE responses:
    - Default (no special Accept header): returns JSON ChatResponse.
    - Accept: text/event-stream: returns an SSE stream of token and done events.

    Error cases:
    - Session not found → 404
    - LLM timeout → 504 (JSON) or event: error (SSE)
    """
    accept = request.headers.get("accept", "")
    if "text/event-stream" in accept:
        # SessionNotFoundError raised here (before streaming starts) → normal 404.
        stream = pipeline.run_streaming(
            session_id, body.content, db, idempotency_key=idempotency_key
        )
        return StreamingResponse(
            stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    result = await pipeline.run(session_id, body.content, db, idempotency_key=idempotency_key)
    return ChatResponse(reply=result.content, routed_to=result.routed_to, trace_id=result.trace_id)
