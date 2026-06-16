"""Repository parity suite against real Postgres (asyncpg).

Catches enum / server_default divergence from the sqlite unit suite. Requires
`make services-up`; excluded from `make test` via the ``integration`` marker.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from screencropnet_yolo.server import db
from screencropnet_yolo.server.config import get_settings

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def pg_session() -> AsyncIterator[AsyncSession]:
    engine = db.make_engine(get_settings().postgres_dsn)
    async with engine.begin() as conn:
        await conn.run_sync(db.Base.metadata.drop_all)
        await conn.run_sync(db.Base.metadata.create_all)
    factory = db.make_sessionmaker(engine)
    async with factory() as session:
        yield session
    await engine.dispose()


async def test_pg_roundtrip_and_transitions(pg_session: AsyncSession) -> None:
    job = await db.create_job(pg_session, original_path="/o/x.png", batch_id="pg")
    if job.status != db.JobStatus.pending:
        raise AssertionError("new job must be pending on Postgres")
    if job.created_at is None or job.updated_at is None:
        raise AssertionError("server_default timestamps must populate on Postgres")

    await db.mark_processing(pg_session, job.job_id)
    await db.mark_done(
        pg_session,
        job.job_id,
        is_twitter=True,
        pred_class="twitter",
        pred_prob=0.9,
        time_for_pred=0.1,
    )
    done = await db.get_job(pg_session, job.job_id)
    if done is None or done.status != db.JobStatus.done or done.is_twitter is not True:
        raise AssertionError("done transition must persist on Postgres")


async def test_pg_status_summary_and_twitter_filter(pg_session: AsyncSession) -> None:
    a = await db.create_job(pg_session, original_path="/o/1.png", batch_id="pg")
    b = await db.create_job(pg_session, original_path="/o/2.png", batch_id="pg")
    await db.create_job(pg_session, original_path="/o/3.png", batch_id="pg")
    await db.mark_done(
        pg_session,
        a.job_id,
        is_twitter=True,
        pred_class="twitter",
        pred_prob=0.9,
        time_for_pred=0.1,
    )
    await db.mark_failed(pg_session, b.job_id, error="x")

    summary = await db.status_summary(pg_session, batch_id="pg")
    if summary.total != 3 or summary.done != 1 or summary.failed != 1 or summary.twitter_count != 1:
        raise AssertionError("status_summary counts diverge on Postgres")

    positives = await db.list_twitter_positive(pg_session, batch_id="pg")
    if [j.original_path for j in positives] != ["/o/1.png"]:
        raise AssertionError("twitter filter must match sqlite behaviour on Postgres")
