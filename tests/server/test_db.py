from __future__ import annotations

from collections.abc import AsyncIterator

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from screencropnet_yolo.server.db import (
    JobStatus,
    create_all,
    create_job,
    get_job,
    list_jobs,
    list_twitter_positive,
    make_engine,
    make_sessionmaker,
    mark_done,
    mark_failed,
    mark_processing,
    status_summary,
)


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    await create_all(engine)
    factory = make_sessionmaker(engine)
    async with factory() as sess:
        yield sess
    await engine.dispose()


async def test_create_job_starts_pending(session: AsyncSession) -> None:
    job = await create_job(session, original_path="/a/x.png", batch_id="b1")
    if job.status != JobStatus.pending:
        raise AssertionError("new jobs must start pending")
    if not job.job_id:
        raise AssertionError("a job_id must be assigned")
    fetched = await get_job(session, job.job_id)
    if fetched is None or fetched.original_path != "/a/x.png":
        raise AssertionError("job should round-trip through the database")


async def test_status_transitions(session: AsyncSession) -> None:
    job = await create_job(session, original_path="/a/x.png", batch_id="b1")

    await mark_processing(session, job.job_id)
    processing = await get_job(session, job.job_id)
    if processing is None or processing.status != JobStatus.processing:
        raise AssertionError("mark_processing should move the job to processing")

    await mark_done(
        session,
        job.job_id,
        is_twitter=True,
        pred_class="twitter",
        pred_prob=0.97,
        time_for_pred=0.12,
    )
    done = await get_job(session, job.job_id)
    if done is None or done.status != JobStatus.done:
        raise AssertionError("mark_done should move the job to done")
    if done.is_twitter is not True or done.pred_class != "twitter":
        raise AssertionError("prediction fields must be persisted")


async def test_mark_failed_records_error(session: AsyncSession) -> None:
    job = await create_job(session, original_path="/a/x.png", batch_id="b1")
    await mark_failed(session, job.job_id, error="boom")
    failed = await get_job(session, job.job_id)
    if failed is None or failed.status != JobStatus.failed or failed.error != "boom":
        raise AssertionError("mark_failed must record failed status and the error string")


async def test_list_twitter_positive_filters_done_and_twitter(session: AsyncSession) -> None:
    twitter_done = await create_job(session, original_path="/p/1.png", batch_id="b1")
    await mark_done(
        session,
        twitter_done.job_id,
        is_twitter=True,
        pred_class="twitter",
        pred_prob=0.9,
        time_for_pred=0.1,
    )
    not_twitter = await create_job(session, original_path="/p/2.png", batch_id="b1")
    await mark_done(
        session,
        not_twitter.job_id,
        is_twitter=False,
        pred_class="tiktok",
        pred_prob=0.8,
        time_for_pred=0.1,
    )
    still_pending = await create_job(session, original_path="/p/3.png", batch_id="b1")
    _ = still_pending

    positives = await list_twitter_positive(session, batch_id="b1")
    if [j.original_path for j in positives] != ["/p/1.png"]:
        raise AssertionError("only done AND is_twitter jobs should be returned")


async def test_list_jobs_filters(session: AsyncSession) -> None:
    await create_job(session, original_path="/p/1.png", batch_id="b1")
    await create_job(session, original_path="/p/2.png", batch_id="b2")
    only_b1 = await list_jobs(session, batch_id="b1")
    if len(only_b1) != 1 or only_b1[0].original_path != "/p/1.png":
        raise AssertionError("list_jobs must filter by batch_id")


async def test_status_summary_counts(session: AsyncSession) -> None:
    a = await create_job(session, original_path="/p/1.png", batch_id="b1")
    b = await create_job(session, original_path="/p/2.png", batch_id="b1")
    await create_job(session, original_path="/p/3.png", batch_id="b1")
    await mark_done(
        session, a.job_id, is_twitter=True, pred_class="twitter", pred_prob=0.9, time_for_pred=0.1
    )
    await mark_failed(session, b.job_id, error="x")

    summary = await status_summary(session, batch_id="b1")
    if summary.total != 3:
        raise AssertionError("total must count all jobs in the batch")
    if summary.done != 1 or summary.failed != 1:
        raise AssertionError("done/failed counts are wrong")
    if summary.twitter_count != 1:
        raise AssertionError("twitter_count must reflect done twitter-positive jobs")
    if summary.counts.get("pending") != 1:
        raise AssertionError("per-status counts must be present")
    if summary.throughput_per_sec < 0:
        raise AssertionError("throughput must be non-negative")
