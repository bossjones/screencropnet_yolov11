from __future__ import annotations

import contextlib
from pathlib import Path

import anyio.to_thread
from PIL import Image
from pytest_mock import MockerFixture
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from screencropnet_yolo.server import db
from screencropnet_yolo.server.classifier import FakeClassifier
from screencropnet_yolo.server.schemas import QueueMessage
from screencropnet_yolo.server.worker import handle_message, on_message


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
