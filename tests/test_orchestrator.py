from app.agents.orchestrator import _build_root_instruction, _infer_routed_to
from app.srop.state import SessionState


def test_build_root_instruction_includes_session_state():
    instruction = _build_root_instruction(
        SessionState(user_id="u_123", plan_tier="pro", last_agent="knowledge", turn_count=2)
    )

    assert "user_id: u_123" in instruction
    assert "plan_tier: pro" in instruction
    assert "last_agent: knowledge" in instruction
    assert "turn_count: 2" in instruction


def test_infer_routed_to_prefers_tool_calls_over_authors():
    tool_calls = [{"tool_name": "search_docs", "args": {}, "result": {"count": 1}}]
    authors = ["srop_root"]

    assert _infer_routed_to(tool_calls, authors) == "knowledge"


def test_infer_routed_to_falls_back_to_smalltalk():
    assert _infer_routed_to([], ["srop_root"]) == "smalltalk"
