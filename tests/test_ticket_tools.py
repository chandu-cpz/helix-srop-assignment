import pytest
from sqlalchemy import select

from app.agents.tools.ticket_tools import create_ticket
from app.db.models import Ticket


@pytest.mark.asyncio
async def test_create_ticket_persists_ticket(db):
    result = await create_ticket(
        user_id="u_ticket_001",
        summary="Please have a human investigate repeated build failures.",
        priority="high",
    )

    ticket = await db.scalar(select(Ticket).where(Ticket.ticket_id == result["ticket_id"]))

    assert ticket is not None
    assert ticket.user_id == "u_ticket_001"
    assert ticket.priority == "high"
    assert ticket.status == "open"
