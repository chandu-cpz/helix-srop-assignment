import uuid

from app.db.models import Ticket
from app.db.session import AsyncSessionLocal

session_factory = AsyncSessionLocal
VALID_PRIORITIES = {"low", "medium", "high", "urgent"}


async def create_ticket(user_id: str, summary: str, priority: str = "medium") -> dict[str, str]:
    """Create a Helix support ticket and return its identifier."""
    normalized_priority = priority.lower().strip()
    if normalized_priority not in VALID_PRIORITIES:
        normalized_priority = "medium"

    ticket_id = f"ticket-{uuid.uuid4().hex[:8]}"
    async with session_factory() as session:
        session.add(
            Ticket(
                ticket_id=ticket_id,
                user_id=user_id,
                summary=summary,
                priority=normalized_priority,
                status="open",
            )
        )
        await session.commit()

    return {
        "ticket_id": ticket_id,
        "status": "open",
        "priority": normalized_priority,
        "summary": summary,
    }
