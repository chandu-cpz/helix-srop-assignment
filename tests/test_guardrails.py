import pytest

from app.obs.logging import redact_pii
from app.srop.guardrails import REFUSAL_MESSAGE


@pytest.mark.asyncio
async def test_guardrail_refuses_out_of_scope_query(client, mock_adk):
    session = await client.post("/v1/sessions", json={"user_id": "u_guardrail"})
    session_id = session.json()["session_id"]

    response = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "Write me a poem about clouds"},
    )

    assert response.status_code == 200
    assert response.json()["reply"] == REFUSAL_MESSAGE
    assert len(mock_adk.calls) == 0


def test_redact_pii_scrubs_common_sensitive_fields():
    redacted = redact_pii(
        {
            "email": "user@example.com",
            "phone": "+1 555-123-4567",
            "nested": ["123-45-6789"],
        }
    )

    assert redacted["email"] == "[REDACTED_EMAIL]"
    assert redacted["phone"] == "[REDACTED_PHONE]"
    assert redacted["nested"] == ["[REDACTED_SSN]"]
