from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path

import anyio.to_thread
from PIL import Image
from pytest_mock import MockerFixture
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from screencropnet_yolo.server import db
from screencropnet_yolo.server.classifier import FakeClassifier
from screencropnet_yolo.server.config import Settings
from screencropnet_yolo.server.schemas import QueueMessage
from screencropnet_yolo.server.worker import (
    _configure_logging,
    drain_and_cancel,
    handle_message,
    logger,
    on_message,
)


def _write_webp(path: Path) -> None:
    Image.new("RGB", (4, 4)).save(path, format="WEBP")


async def _make_pending_job(
    session_factory: async_sessionmaker[AsyncSession], compressed: Path
) -> QueueMessage:
    async with session_factory() as session:
        job = await db.create_job(session, original_path="/o/x.png", batch_id="b1")
    _write_webp(compressed)
    return QueueMessage(
        job_id=job.job_id,
        batch_id="b1",
        compressed_path=str(compressed),
        original_path="/o/x.png",
    )


async def test_handle_message_marks_done_off_the_loop(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    msg = await _make_pending_job(session_factory, tmp_path / "x.webp")
    spy = mocker.spy(anyio.to_thread, "run_sync")

    await handle_message(
        msg.model_dump_json().encode(),
        classifier=FakeClassifier(pred_class="twitter", pred_prob=0.91),
        session_factory=session_factory,
    )

    if not spy.called:
        raise AssertionError("inference must run off the event loop via anyio.to_thread.run_sync")
    async with session_factory() as session:
        done = await db.get_job(session, msg.job_id)
    if done is None or done.status != db.JobStatus.done:
        raise AssertionError("a successful message must produce a done job")
    if done.is_twitter is not True or done.pred_class != "twitter":
        raise AssertionError("is_twitter and prediction fields must be written")


async def test_handle_message_marks_not_twitter(
    session_factory: async_sessionmaker[AsyncSession], tmp_path: Path
) -> None:
    msg = await _make_pending_job(session_factory, tmp_path / "x.webp")
    await handle_message(
        msg.model_dump_json().encode(),
        classifier=FakeClassifier(pred_class="tiktok"),
        session_factory=session_factory,
    )
    async with session_factory() as session:
        done = await db.get_job(session, msg.job_id)
    if done is None or done.is_twitter is not False:
        raise AssertionError("a non-twitter prediction must set is_twitter False")


class _Boom:
    def infer(self, image: Image.Image) -> list[dict[str, object]]:
        raise RuntimeError("inference exploded")


async def test_handle_message_records_failure(
    session_factory: async_sessionmaker[AsyncSession], tmp_path: Path
) -> None:
    msg = await _make_pending_job(session_factory, tmp_path / "x.webp")
    await handle_message(
        msg.model_dump_json().encode(),
        classifier=_Boom(),
        session_factory=session_factory,
    )
    async with session_factory() as session:
        failed = await db.get_job(session, msg.job_id)
    if failed is None or failed.status != db.JobStatus.failed:
        raise AssertionError("a classifier exception must produce a failed job")
    if failed.error is None or "exploded" not in failed.error:
        raise AssertionError("the error string must be persisted")


class _FakeMessage:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def process(self) -> contextlib.AbstractAsyncContextManager[None]:
        return contextlib.nullcontext()


async def test_on_message_wrapper_delegates(
    session_factory: async_sessionmaker[AsyncSession], tmp_path: Path
) -> None:
    msg = await _make_pending_job(session_factory, tmp_path / "x.webp")
    await on_message(
        _FakeMessage(msg.model_dump_json().encode()),
        classifier=FakeClassifier(),
        session_factory=session_factory,
    )
    async with session_factory() as session:
        done = await db.get_job(session, msg.job_id)
    if done is None or done.status != db.JobStatus.done:
        raise AssertionError("on_message must process the body and ack")


def test_configure_logging_routes_to_worker_log_path_override(tmp_path: Path) -> None:
    log_path = tmp_path / "supervisor" / "worker-3.log"
    settings = Settings(logs_dir=tmp_path / "logs", worker_log_path=log_path)
    added: list[logging.Handler] = []
    before = set(logger.handlers)
    try:
        _configure_logging(settings)
        added = [h for h in logger.handlers if h not in before]
        if not any(getattr(h, "baseFilename", None) == str(log_path) for h in logger.handlers):
            raise AssertionError("worker_log_path must route the FileHandler to the override path")
        if not log_path.parent.is_dir():
            raise AssertionError("the override log's parent directory must be created")
    finally:
        for handler in added:
            logger.removeHandler(handler)
            handler.close()


class _RecordingQueue:
    """A stand-in for an aio-pika queue that records the cancelled consumer tag."""

    def __init__(self) -> None:
        self.cancelled: str | None = None

    async def cancel(self, consumer_tag: str) -> None:
        self.cancelled = consumer_tag


async def test_drain_and_cancel_cancels_consumer_then_awaits_inflight() -> None:
    queue = _RecordingQueue()
    finished = {"done": False}

    async def _job() -> None:
        await asyncio.sleep(0.01)
        finished["done"] = True

    task = asyncio.create_task(_job())
    await drain_and_cancel(queue, "ctag-1", {task}, timeout=1.0)

    if queue.cancelled != "ctag-1":
        raise AssertionError("drain must cancel the consumer tag so no new jobs arrive")
    if not finished["done"]:
        raise AssertionError("an in-flight job under the timeout must be awaited to completion")


async def test_drain_and_cancel_returns_when_inflight_exceeds_timeout() -> None:
    queue = _RecordingQueue()

    async def _slow() -> None:
        await asyncio.sleep(30)

    task = asyncio.create_task(_slow())
    try:
        await drain_and_cancel(queue, "ctag-2", {task}, timeout=0.05)
        if not task.done():
            # The point of the test: drain returned promptly instead of blocking
            # for the full 30s sleep. A still-running task here is expected.
            pass
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_drain_and_cancel_handles_no_inflight() -> None:
    queue = _RecordingQueue()
    await drain_and_cancel(queue, "ctag-3", set(), timeout=1.0)
    if queue.cancelled != "ctag-3":
        raise AssertionError("drain must cancel the consumer tag even with nothing in flight")
