# Helix SROP - AI Support Concierge

Name: Chandrakanth V
Stateful RAG Orchestration Pipeline for Helix support workflows. The service exposes a FastAPI chat API backed by async SQLAlchemy, Google ADK `AgentTool` routing, persistent session state, Chroma document retrieval, and structured per-turn traces.

## Setup

```bash
uv sync
cp .env.example .env
# Set GOOGLE_API_KEY in .env
uv run python -m app.rag.ingest --path docs/
uv run uvicorn app.main:app --reload
```

The ingest command batches and retries embedding calls to stay friendlier to Gemini free-tier
limits. If needed, tune it with `--batch-size` and `--batch-delay`.

Health check:

```bash
curl -fsS http://127.0.0.1:8000/healthz
```

Create a session and chat:

```bash
SESSION=$(curl -s -X POST http://127.0.0.1:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u_demo", "plan_tier": "pro"}' | jq -r .session_id)

curl -s -X POST "http://127.0.0.1:8000/v1/chat/$SESSION" \
  -H "Content-Type: application/json" \
  -d '{"content": "How do I rotate a deploy key?"}' | jq .
```

## Docker

```bash
cp .env.example .env
# Set GOOGLE_API_KEY in .env
docker compose build api
docker compose up -d api
curl -fsS http://127.0.0.1:8000/healthz
```

For knowledge answers in Docker, ingest docs into the mounted volume after the service starts:

```bash
docker compose exec api uv run python -m app.rag.ingest --path docs/
```

## Tests And Lint

```bash
uv run pytest -q
uv run ruff check .
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/sessions` | Create a stateful session from `user_id` and `plan_tier`. |
| `POST` | `/v1/chat/{session_id}` | Run one support-concierge turn. |
| `GET` | `/v1/traces/{trace_id}` | Fetch the structured trace for one turn. |
| `GET` | `/healthz` | Basic service health check. |

`POST /v1/chat/{session_id}` also supports:

- `Idempotency-Key` header for replay-safe retries.
- `Accept: text/event-stream` for SSE token/done events.

## Architecture

```text
POST /v1/chat/{session_id}
        |
        v
SROP pipeline
  - load sessions.state from SQLite
  - inject user_id, plan_tier, last_agent, turn_count into ADK context
  - run root ADK LlmAgent
  - route via AgentTool to specialist agents
  - persist messages, updated state, and agent_traces row
        |
        +--> KnowledgeAgent -> search_docs -> Chroma vector store
        +--> AccountAgent   -> account tools -> async app DB/mock account data
        +--> EscalationAgent -> create_ticket -> tickets table
```

## Design Decisions

### State persistence

I used explicit application-owned session state in the `sessions.state` JSON column. Each turn rehydrates `SessionState`, passes it into ADK session state and the root instruction, then writes the updated state back after the turn. This keeps restart-surviving state small and reviewable while avoiding reliance on in-memory ADK sessions.

Persisted state includes `user_id`, `plan_tier`, `last_agent`, `turn_count`, `open_ticket_ids`, and `last_ticket_id`.

### Agent routing

The root ADK `LlmAgent` uses ADK's `AgentTool` pattern for `KnowledgeAgent`, `AccountAgent`, and `EscalationAgent`. Routing is inferred from ADK tool calls/events for tracing; the application does not ask the model to emit a route string and parse it.

### Chunking strategy

The ingest CLI uses heading-aware Markdown splitting with a fixed-size fallback and overlap for long sections. Chunk IDs are deterministic from source file stem plus chunk index, so repeated ingestion upserts the same chunks instead of creating duplicates.

### Vector store choice

Chroma is used as a local persistent vector store because it fits the assignment constraints, works with SQLite-style local development, and keeps setup under a few minutes.

### Error handling

LLM turns are wrapped with `asyncio.wait_for` or `asyncio.timeout` and converted to `UPSTREAM_TIMEOUT`. Missing sessions return `SESSION_NOT_FOUND`. Domain errors are returned as stable problem-detail JSON responses.

## Trace Shape

Each successful turn writes one `agent_traces` row with:

- `trace_id` and `session_id`
- `routed_to`
- `tool_calls` with tool name, args, and result summary
- `retrieved_chunk_ids`
- `latency_ms`

This makes `/v1/traces/{trace_id}` useful for debugging routing, retrieval, and tool behavior.

## Extensions Completed

- [x] E1: Idempotency via `Idempotency-Key`.
- [x] E2: Escalation agent with persisted tickets and follow-up ticket context.
- [x] E3: Streaming SSE for chat responses.
- [ ] E4: Reranking.
- [x] E5: Guardrails for out-of-scope requests and PII redaction in logs.
- [x] E6: Docker build/compose with healthcheck.
- [ ] E7: Eval harness.

## Known Limitations

- Account tools use deterministic mock build/usage data rather than a real Helix account backend.
- Knowledge quality depends on running `app.rag.ingest` before asking documentation questions.
- Reranking and the eval harness are not implemented.
- Real LLM and embedding calls require a valid `GOOGLE_API_KEY`.(Free tier limits)

## Time Spent

| Phase | Time |
| --- | ---: |
| Setup + DB + FastAPI boilerplate | 2 hours |
| RAG ingest + search_docs | 2.5 hours |
| ADK agents and tool tracing | 3 hours |
| Pipeline + state persistence | 2.5 hours |
| Extensions | 3 hours |
| Tests, Docker, README cleanup | 2 hours |
| **Total** | **15 hours** |
