"""
Test fixtures.

Key fixtures:
- `client`: async test client with in-memory SQLite DB
- `mock_adk`: patches the ADK root agent so tests don't hit the real LLM
- `seeded_db`: DB with a test user and session pre-created
"""
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.agents.orchestrator import OrchestratorTurnResult
from app.db.models import Base
from app.db.session import get_db
from app.main import app

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)


@pytest_asyncio.fixture(autouse=True)
async def setup_test_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db() -> AsyncSession:
    async with TestSessionLocal() as session:
        yield session


@pytest.fixture(autouse=True)
def override_tool_session_factories(monkeypatch):
    monkeypatch.setattr("app.agents.tools.account_tools.session_factory", TestSessionLocal)
    monkeypatch.setattr("app.agents.tools.ticket_tools.session_factory", TestSessionLocal)


@pytest_asyncio.fixture
async def client(db):
    """Async test client with DB overridden to in-memory SQLite."""
    app.dependency_overrides[get_db] = lambda: db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def mock_adk(monkeypatch):
    """
    Patch the ADK pipeline so tests don't call the real LLM.

    TODO for candidate: patch at the ADK boundary (not at the HTTP layer).
    The mock should:
    1. Accept a user message
    2. Return a canned response with a specified routed_to value
    3. Allow tests to assert which sub-agent was called

    Example:
        def mock_run(session_id, message, db):
            if "rotate" in message.lower():
                return PipelineResult(
                    content="To rotate a deploy key...",
                    routed_to="knowledge",
                    trace_id="test-trace-001",
                )
            ...

        monkeypatch.setattr("app.srop.pipeline.run", mock_run)
    """
    async def mock_run_turn(*, user_message: str, state):
        mock_run_turn.calls.append({"user_message": user_message, "user_id": state.user_id})
        lowered = user_message.lower()

        if "plan tier" in lowered:
            return OrchestratorTurnResult(
                content=f"Your current plan tier is {state.plan_tier}.",
                routed_to="account",
                tool_calls=[
                    {
                        "tool_name": "get_account_status",
                        "args": {"user_id": state.user_id},
                        "result": {"plan_tier": state.plan_tier},
                    }
                ],
                retrieved_chunk_ids=[],
            )

        if "ticket" in lowered and state.last_ticket_id:
            return OrchestratorTurnResult(
                content=f"Your latest support ticket is {state.last_ticket_id}.",
                routed_to="escalation",
                tool_calls=[],
                retrieved_chunk_ids=[],
            )

        if "human" in lowered or "escalate" in lowered or "support ticket" in lowered:
            return OrchestratorTurnResult(
                content="I created support ticket ticket-test-001 for a human follow-up.",
                routed_to="escalation",
                tool_calls=[
                    {
                        "tool_name": "create_ticket",
                        "args": {
                            "user_id": state.user_id,
                            "summary": user_message,
                            "priority": "high",
                        },
                        "result": {
                            "ticket_id": "ticket-test-001",
                            "status": "open",
                            "priority": "high",
                        },
                    }
                ],
                retrieved_chunk_ids=[],
            )

        if "deploy key" in lowered or "rotate" in lowered:
            return OrchestratorTurnResult(
                content=(
                    "To rotate a deploy key, open the repository settings, revoke the old "
                    "key, and add a new one. [chunk_deploy_keys_0]"
                ),
                routed_to="knowledge",
                tool_calls=[
                    {
                        "tool_name": "search_docs",
                        "args": {"query": user_message, "k": 5, "product_area": None},
                        "result": {"count": 1, "chunk_ids": ["chunk_deploy_keys_0"]},
                    }
                ],
                retrieved_chunk_ids=["chunk_deploy_keys_0"],
            )

        return OrchestratorTurnResult(
            content="Hello from the mocked orchestrator.",
            routed_to="smalltalk",
            tool_calls=[],
            retrieved_chunk_ids=[],
        )

    monkeypatch.setattr("app.agents.orchestrator.run_turn", mock_run_turn)

    async def mock_run_turn_streaming(*, user_message: str, state):
        """Streaming version of the mock — yields token(s) then the result."""
        result = await mock_run_turn(user_message=user_message, state=state)
        yield "token", result.content
        yield "result", result

    monkeypatch.setattr("app.agents.orchestrator.run_turn_streaming", mock_run_turn_streaming)
    mock_run_turn.calls = []
    return mock_run_turn
