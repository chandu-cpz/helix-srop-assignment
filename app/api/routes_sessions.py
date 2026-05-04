"""
POST /v1/sessions — create a session.
"""
import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Session, User
from app.db.session import get_db
from app.srop.state import SessionState

router = APIRouter(tags=["sessions"])
VALID_PLAN_TIERS = {"free", "pro", "enterprise"}


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str
    plan_tier: str = "free"


class CreateSessionResponse(BaseModel):
    session_id: str
    user_id: str


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(
    body: CreateSessionRequest,
    db: AsyncSession = Depends(get_db),
) -> CreateSessionResponse:
    """
    Create a new session. Upsert the user if not seen before.
    Initialize SessionState and persist to DB.
    """
    if body.plan_tier not in VALID_PLAN_TIERS:
        raise ValueError(f"Unsupported plan tier: {body.plan_tier}")

    user = await db.scalar(select(User).where(User.user_id == body.user_id))
    if user is None:
        user = User(user_id=body.user_id, plan_tier=body.plan_tier)
        db.add(user)
    else:
        user.plan_tier = body.plan_tier

    session_id = str(uuid.uuid4())
    state = SessionState(user_id=body.user_id, plan_tier=body.plan_tier)
    db.add(
        Session(
            session_id=session_id,
            user_id=body.user_id,
            state=state.to_db_dict(),
        )
    )
    await db.commit()

    return CreateSessionResponse(session_id=session_id, user_id=body.user_id)
