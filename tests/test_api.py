"""
Integration tests — exercise the full SROP pipeline.
LLM mocked at the ADK boundary (not at the HTTP layer).
"""
import pytest

from app.db.models import AgentTrace


@pytest.mark.asyncio
async def test_create_session(client):
    resp = await client.post("/v1/sessions", json={"user_id": "u_test_001"})
    assert resp.status_code == 200
    assert "session_id" in resp.json()


@pytest.mark.asyncio
async def test_get_trace_returns_structured_trace(client, db):
    db.add(
        AgentTrace(
            trace_id="trace-test-001",
            session_id="session-test-001",
            routed_to="knowledge",
            tool_calls=[
                {
                    "tool_name": "search_docs",
                    "args": {"query": "deploy key"},
                    "result": {"count": 2},
                }
            ],
            retrieved_chunk_ids=["chunk_1", "chunk_2"],
            latency_ms=123,
        )
    )
    await db.commit()

    resp = await client.get("/v1/traces/trace-test-001")

    assert resp.status_code == 200
    body = resp.json()
    assert body["trace_id"] == "trace-test-001"
    assert body["routed_to"] == "knowledge"
    assert body["retrieved_chunk_ids"] == ["chunk_1", "chunk_2"]
    assert body["tool_calls"][0]["tool_name"] == "search_docs"


@pytest.mark.asyncio
async def test_trace_not_found_returns_404(client):
    resp = await client.get("/v1/traces/missing-trace")

    assert resp.status_code == 404
    assert resp.json()["title"] == "TRACE_NOT_FOUND"


@pytest.mark.asyncio
async def test_knowledge_query_routes_correctly(client, mock_adk):
    """
    Core integration test.

    Sends a knowledge question, asserts:
    1. Response contains a reply
    2. routed_to == "knowledge"
    3. trace exists with retrieved chunk IDs
    4. Turn 2 in the same session has access to context from turn 1
       (state persistence — at minimum, plan_tier available without re-asking)

    Implement after pipeline.run() and state persistence are working.
    The mock_adk fixture must patch at the ADK boundary, not at the HTTP layer.
    """
    # Create session
    sess = await client.post("/v1/sessions", json={"user_id": "u_test_002", "plan_tier": "pro"})
    session_id = sess.json()["session_id"]

    # Turn 1 — knowledge query
    r1 = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "How do I rotate a deploy key?"},
    )
    assert r1.status_code == 200
    assert r1.json()["routed_to"] == "knowledge"
    trace_id = r1.json()["trace_id"]

    # Trace must have chunk IDs
    trace = await client.get(f"/v1/traces/{trace_id}")
    assert trace.status_code == 200
    assert len(trace.json()["retrieved_chunk_ids"]) > 0

    # Turn 2 — follow-up in same session
    r2 = await client.post(f"/v1/chat/{session_id}", json={"content": "What is my plan tier?"})
    assert r2.status_code == 200
    # Agent should know plan_tier from state — not re-ask
    assert "pro" in r2.json()["reply"].lower()


@pytest.mark.asyncio
async def test_session_not_found_returns_404(client):
    resp = await client.post("/v1/chat/nonexistent-id", json={"content": "hello"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_chat_idempotency_replays_original_response(client, mock_adk):
    sess = await client.post("/v1/sessions", json={"user_id": "u_test_003", "plan_tier": "pro"})
    session_id = sess.json()["session_id"]
    headers = {"Idempotency-Key": "idem-001"}

    first = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "How do I rotate a deploy key?"},
        headers=headers,
    )
    second = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "How do I rotate a deploy key?"},
        headers=headers,
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json() == first.json()
    assert len(mock_adk.calls) == 1


@pytest.mark.asyncio
async def test_escalation_ticket_id_is_available_in_follow_up(client, mock_adk):
    sess = await client.post("/v1/sessions", json={"user_id": "u_test_004", "plan_tier": "free"})
    session_id = sess.json()["session_id"]

    first = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "Please escalate this to human support"},
    )
    second = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "What is my latest ticket id?"},
    )

    assert first.status_code == 200
    assert first.json()["routed_to"] == "escalation"
    assert "ticket-test-001" in first.json()["reply"]
    assert second.status_code == 200
    assert "ticket-test-001" in second.json()["reply"]


@pytest.mark.asyncio
async def test_sse_stream_returns_done_event_with_trace(client, mock_adk, db):
    """SSE path: stream must yield a terminal 'done' event with routed_to and trace_id,
    and must persist the trace row to the DB.
    """
    import json as _json

    from sqlalchemy import select as _select

    from app.db.models import AgentTrace

    sess = await client.post("/v1/sessions", json={"user_id": "u_test_sse", "plan_tier": "pro"})
    session_id = sess.json()["session_id"]

    resp = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "How do I rotate a deploy key?"},
        headers={"Accept": "text/event-stream"},
    )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    # Parse SSE events from the response body.
    done_payload: dict | None = None
    for line in resp.text.splitlines():
        if line.startswith("event: done"):
            continue
        if line.startswith("data: ") and done_payload is None:
            # Accept the first data line that follows an event: done marker
            # by scanning pairs of (event, data) lines.
            pass

    # Simpler: collect all (event_name, data) pairs
    events: list[tuple[str, dict]] = []
    current_event = ""
    for raw_line in resp.text.splitlines():
        if raw_line.startswith("event: "):
            current_event = raw_line[len("event: "):].strip()
        elif raw_line.startswith("data: ") and current_event:
            events.append((current_event, _json.loads(raw_line[len("data: "):])))
            current_event = ""

    done_events = [(name, data) for name, data in events if name == "done"]
    assert len(done_events) == 1, f"Expected exactly 1 done event, got: {events}"
    _, done_data = done_events[0]
    assert "routed_to" in done_data
    assert "trace_id" in done_data
    assert "reply" in done_data
    assert done_data["routed_to"] == "knowledge"

    # Trace must have been written to DB.
    trace = await db.scalar(_select(AgentTrace).where(AgentTrace.trace_id == done_data["trace_id"]))
    assert trace is not None
    assert trace.routed_to == "knowledge"
