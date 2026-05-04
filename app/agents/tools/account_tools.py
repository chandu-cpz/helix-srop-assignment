"""
Account tools — used by AccountAgent.

These tools query the DB for user-specific data.
Mock data is acceptable for the take-home; the integration matters.

TODO for candidate: implement these tools.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256

from sqlalchemy import select

from app.db.models import User
from app.db.session import AsyncSessionLocal


@dataclass
class BuildSummary:
    build_id: str
    pipeline: str
    status: str  # passed | failed | cancelled
    branch: str
    started_at: datetime
    duration_seconds: int


@dataclass
class AccountStatus:
    user_id: str
    plan_tier: str
    concurrent_builds_used: int
    concurrent_builds_limit: int
    storage_used_gb: float
    storage_limit_gb: float


session_factory = AsyncSessionLocal

_PLAN_LIMITS: dict[str, tuple[int, float]] = {
    "free": (1, 10.0),
    "pro": (5, 100.0),
    "enterprise": (20, 1000.0),
}
_BUILD_STATUSES = ("passed", "failed", "cancelled")
_PIPELINES = ("deploy", "release", "smoke", "integration", "nightly")
_BRANCHES = ("main", "develop", "release", "feature/deploy-keys", "feature/rag")


def _stable_seed(value: str) -> int:
    digest = sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


async def _load_plan_tier(user_id: str) -> str:
    async with session_factory() as session:
        user = await session.scalar(select(User).where(User.user_id == user_id))

    if user is None:
        return "free"

    return user.plan_tier


async def get_recent_builds(user_id: str, limit: int = 5) -> list[BuildSummary]:
    """
    Return the most recent builds for a user, newest first.

    For the take-home: returning mock/seeded data is fine.
    The key evaluation point is that this is wired as an ADK tool
    and the agent correctly invokes it when the user asks about builds.
    """
    if limit <= 0:
        return []

    seed = _stable_seed(user_id)
    anchor = datetime.utcnow().replace(hour=12, minute=0, second=0, microsecond=0)
    builds: list[BuildSummary] = []

    for index in range(limit):
        selector = (seed >> (index * 8)) & 0xFF
        builds.append(
            BuildSummary(
                build_id=f"b_{user_id[-8:]}_{index + 1:02d}_{selector:02x}",
                pipeline=_PIPELINES[selector % len(_PIPELINES)],
                status=_BUILD_STATUSES[(selector // 3) % len(_BUILD_STATUSES)],
                branch=_BRANCHES[(selector + index) % len(_BRANCHES)],
                started_at=anchor - timedelta(hours=index * 6 + (selector % 4)),
                duration_seconds=180 + (selector * 7),
            )
        )

    return builds


async def get_account_status(user_id: str) -> AccountStatus:
    """
    Return current account status (plan, usage limits).

    For the take-home: mock data is fine.
    """
    plan_tier = await _load_plan_tier(user_id)
    concurrent_limit, storage_limit = _PLAN_LIMITS.get(plan_tier, _PLAN_LIMITS["free"])
    seed = _stable_seed(user_id)

    return AccountStatus(
        user_id=user_id,
        plan_tier=plan_tier,
        concurrent_builds_used=seed % (concurrent_limit + 1),
        concurrent_builds_limit=concurrent_limit,
        storage_used_gb=round(((seed // 17) % 1000) / 1000 * storage_limit, 1),
        storage_limit_gb=storage_limit,
    )
