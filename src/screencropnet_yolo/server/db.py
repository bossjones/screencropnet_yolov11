"""SQLAlchemy 2.0 async models and repository helpers.

Postgres is the production source of truth; the unit suite runs the same code
against aiosqlite. Only portable column types are used so the two stay in sync.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, Float, String, func, select
from sqlalchemy import Enum as SAEnum
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from screencropnet_yolo.server.schemas import StatusSummary


class JobStatus(StrEnum):
    pending = "pending"
    processing = "processing"
    done = "done"
    failed = "failed"


class Base(DeclarativeBase):
    pass


class ClassificationJob(Base):
    __tablename__ = "classification_jobs"

    job_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    batch_id: Mapped[str] = mapped_column(String, index=True)
    original_path: Mapped[str] = mapped_column(String)
    # native_enum=False stores the value as VARCHAR + CHECK, so the DDL is
    # identical on sqlite and Postgres (no native ENUM type to keep in sync).
    status: Mapped[JobStatus] = mapped_column(
        SAEnum(JobStatus, native_enum=False, name="jobstatus"),
        index=True,
        default=JobStatus.pending,
    )
    is_twitter: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    pred_class: Mapped[str | None] = mapped_column(String, nullable=True)
    pred_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    time_for_pred: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


def make_engine(dsn: str) -> AsyncEngine:
    # StaticPool keeps an in-memory sqlite db alive across sessions for tests;
    # it is harmless for a single-process Postgres connection too.
    if dsn.endswith(":memory:"):
        from sqlalchemy.pool import StaticPool

        return create_async_engine(
            dsn, poolclass=StaticPool, connect_args={"check_same_thread": False}
        )
    return create_async_engine(dsn)


def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    # expire_on_commit=False so attributes remain accessible after commit without
    # triggering an (illegal) lazy load on the async session.
    return async_sessionmaker(engine, expire_on_commit=False)


async def create_all(engine: AsyncEngine) -> None:
    """Create all tables (tests only; production uses Alembic)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def create_job(
    session: AsyncSession,
    *,
    original_path: str,
    batch_id: str,
    job_id: str | None = None,
) -> ClassificationJob:
    job = ClassificationJob(
        job_id=job_id or str(uuid4()),
        batch_id=batch_id,
        original_path=original_path,
        status=JobStatus.pending,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


async def mark_processing(session: AsyncSession, job_id: str) -> ClassificationJob | None:
    job = await get_job(session, job_id)
    if job is None:
        return None
    job.status = JobStatus.processing
    await session.commit()
    return job


async def mark_done(
    session: AsyncSession,
    job_id: str,
    *,
    is_twitter: bool,
    pred_class: str,
    pred_prob: float,
    time_for_pred: float,
) -> ClassificationJob | None:
    job = await get_job(session, job_id)
    if job is None:
        return None
    job.status = JobStatus.done
    job.is_twitter = is_twitter
    job.pred_class = pred_class
    job.pred_prob = pred_prob
    job.time_for_pred = time_for_pred
    await session.commit()
    return job


async def mark_failed(
    session: AsyncSession, job_id: str, *, error: str
) -> ClassificationJob | None:
    job = await get_job(session, job_id)
    if job is None:
        return None
    job.status = JobStatus.failed
    job.error = error
    await session.commit()
    return job


async def get_job(session: AsyncSession, job_id: str) -> ClassificationJob | None:
    return await session.get(ClassificationJob, job_id)


async def list_jobs(
    session: AsyncSession,
    batch_id: str | None = None,
    status: JobStatus | None = None,
) -> list[ClassificationJob]:
    stmt = select(ClassificationJob)
    if batch_id is not None:
        stmt = stmt.where(ClassificationJob.batch_id == batch_id)
    if status is not None:
        stmt = stmt.where(ClassificationJob.status == status)
    stmt = stmt.order_by(ClassificationJob.created_at)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def list_twitter_positive(
    session: AsyncSession, batch_id: str | None = None
) -> list[ClassificationJob]:
    stmt = select(ClassificationJob).where(
        ClassificationJob.status == JobStatus.done,
        ClassificationJob.is_twitter.is_(True),
    )
    if batch_id is not None:
        stmt = stmt.where(ClassificationJob.batch_id == batch_id)
    stmt = stmt.order_by(ClassificationJob.created_at)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def status_summary(session: AsyncSession, batch_id: str | None = None) -> StatusSummary:
    jobs = await list_jobs(session, batch_id=batch_id)
    counts: dict[str, int] = {status.value: 0 for status in JobStatus}
    twitter_count = 0
    for job in jobs:
        counts[job.status.value] += 1
        if job.status == JobStatus.done and job.is_twitter:
            twitter_count += 1

    done = counts[JobStatus.done.value]
    failed = counts[JobStatus.failed.value]

    throughput = 0.0
    finished = [j for j in jobs if j.status in (JobStatus.done, JobStatus.failed)]
    if finished:
        start = min(j.created_at for j in jobs)
        end = max(j.updated_at for j in finished)
        span = (end - start).total_seconds()
        if span > 0:
            throughput = len(finished) / span

    return StatusSummary(
        batch_id=batch_id,
        total=len(jobs),
        counts=counts,
        twitter_count=twitter_count,
        done=done,
        failed=failed,
        throughput_per_sec=throughput,
    )
