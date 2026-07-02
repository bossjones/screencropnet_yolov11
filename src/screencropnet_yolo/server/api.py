"""FastAPI app: validate + enqueue submissions, expose job state from Postgres.

This module never imports torch and never runs the classifier; it only writes a
``pending`` job, publishes a small JSON message, and serves job/status reads
straight from the database (the source of truth).
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, Form, HTTPException, Request, Response, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from screencropnet_yolo.server import db
from screencropnet_yolo.server.compression import UploadTooLarge, enforce_max_size
from screencropnet_yolo.server.config import Settings, get_settings
from screencropnet_yolo.server.db import make_engine, make_sessionmaker
from screencropnet_yolo.server.metrics import JOBS_SUBMITTED, metrics_asgi_app
from screencropnet_yolo.server.queue import Publisher, RabbitPublisher
from screencropnet_yolo.server.schemas import (
    ClassifyAccepted,
    JobView,
    QueueMessage,
    StatusSummary,
)


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


async def get_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    async with request.app.state.sessionmaker() as session:
        yield session


def get_publisher(request: Request) -> Publisher:
    return request.app.state.publisher


def _configure_logging(settings: Settings) -> None:
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = settings.logs_dir / "api.log"
    logger = logging.getLogger("screencropnet_yolo.api")
    logger.setLevel(logging.INFO)
    if not any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(log_path)
        for h in logger.handlers
    ):
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)


def _install_profiler(app: FastAPI) -> None:
    """Attach a pyinstrument middleware that renders ``?profile=1`` requests as HTML."""
    from pyinstrument import Profiler
    from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
    from starlette.responses import HTMLResponse

    async def _dispatch(request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not request.query_params.get("profile"):
            return await call_next(request)
        profiler = Profiler(async_mode="enabled")
        profiler.start()
        await call_next(request)
        profiler.stop()
        return HTMLResponse(profiler.output_html())

    # ty misreads starlette's BaseHTTPMiddleware as not a middleware factory
    # (it flags starlette's own code the same way); basedpyright accepts it.
    app.add_middleware(BaseHTTPMiddleware, dispatch=_dispatch)  # ty: ignore[invalid-argument-type]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    _configure_logging(settings)

    app = FastAPI(title="screencropnet ingest/classify")
    app.state.settings = settings
    app.state.sessionmaker = make_sessionmaker(make_engine(settings.postgres_dsn))
    app.state.publisher = RabbitPublisher(settings.rabbit_url, settings.worker_queue_name)
    app.mount("/metrics", metrics_asgi_app())

    # Opt-in flamegraphs: with SCREENCROPNET_PROFILE set, any request carrying
    # ?profile=1 returns a pyinstrument HTML report instead of its normal body.
    # Off by default, so production and the test suite are unaffected.
    if os.environ.get("SCREENCROPNET_PROFILE"):
        _install_profiler(app)

    @app.get("/healthz")
    async def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/classify", status_code=202, response_model=ClassifyAccepted)
    async def classify(
        file: UploadFile,
        original_path: Annotated[str, Form()],
        session: Annotated[AsyncSession, Depends(get_db_session)],
        publisher: Annotated[Publisher, Depends(get_publisher)],
        settings: Annotated[Settings, Depends(get_settings_dep)],
        batch_id: Annotated[str | None, Form()] = None,
    ) -> ClassifyAccepted:
        contents = await file.read()
        settings.compress_tmp_dir.mkdir(parents=True, exist_ok=True)

        # Stage the upload and enforce the size limit before any state is created,
        # so a rejected upload leaves no job row and no queue message behind.
        staged = settings.compress_tmp_dir / f"upload-{uuid4().hex}.webp"
        staged.write_bytes(contents)
        try:
            enforce_max_size(staged, settings.max_upload_bytes)
        except UploadTooLarge as exc:
            staged.unlink(missing_ok=True)
            raise HTTPException(status_code=413, detail="upload exceeds maximum size") from exc

        batch = batch_id or uuid4().hex
        job = await db.create_job(session, original_path=original_path, batch_id=batch)
        compressed_path = settings.compress_tmp_dir / f"{job.job_id}.webp"
        staged.rename(compressed_path)

        await publisher.publish(
            QueueMessage(
                job_id=job.job_id,
                batch_id=batch,
                compressed_path=str(compressed_path),
                original_path=original_path,
            )
        )
        JOBS_SUBMITTED.labels(batch_id=batch).inc()
        return ClassifyAccepted(job_id=job.job_id, batch_id=batch)

    @app.get("/jobs/{job_id}", response_model=JobView)
    async def get_job(
        job_id: str, session: Annotated[AsyncSession, Depends(get_db_session)]
    ) -> JobView:
        job = await db.get_job(session, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return JobView.model_validate(job)

    @app.get("/jobs", response_model=list[JobView])
    async def list_jobs_endpoint(
        session: Annotated[AsyncSession, Depends(get_db_session)],
        batch_id: str | None = None,
        status: str | None = None,
    ) -> list[JobView]:
        job_status = db.JobStatus(status) if status is not None else None
        jobs = await db.list_jobs(session, batch_id=batch_id, status=job_status)
        return [JobView.model_validate(job) for job in jobs]

    @app.get("/twitter", response_model=list[JobView])
    async def list_twitter_endpoint(
        session: Annotated[AsyncSession, Depends(get_db_session)],
        batch_id: str | None = None,
    ) -> list[JobView]:
        jobs = await db.list_twitter_positive(session, batch_id=batch_id)
        return [JobView.model_validate(job) for job in jobs]

    @app.get("/status", response_model=StatusSummary)
    async def status_endpoint(
        session: Annotated[AsyncSession, Depends(get_db_session)],
        batch_id: str | None = None,
    ) -> StatusSummary:
        return await db.status_summary(session, batch_id=batch_id)

    return app
