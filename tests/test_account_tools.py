import pytest

from app.agents.tools.account_tools import get_account_status, get_recent_builds
from app.db.models import User


@pytest.mark.asyncio
async def test_get_account_status_reads_plan_tier_from_db(db):
    db.add(User(user_id="u_account_pro", plan_tier="pro"))
    await db.commit()

    status = await get_account_status("u_account_pro")

    assert status.user_id == "u_account_pro"
    assert status.plan_tier == "pro"
    assert 0 <= status.concurrent_builds_used <= status.concurrent_builds_limit
    assert 0.0 <= status.storage_used_gb <= status.storage_limit_gb


@pytest.mark.asyncio
async def test_get_recent_builds_is_deterministic_for_same_user():
    builds_one = await get_recent_builds("u_builds_001", limit=3)
    builds_two = await get_recent_builds("u_builds_001", limit=3)

    assert len(builds_one) == 3
    assert [build.build_id for build in builds_one] == [build.build_id for build in builds_two]
    assert [build.pipeline for build in builds_one] == [build.pipeline for build in builds_two]
    assert [build.status for build in builds_one] == [build.status for build in builds_two]
    assert [build.started_at for build in builds_one] == [build.started_at for build in builds_two]
