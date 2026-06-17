"""End-to-end test of the real classifier through the actual worker code path.

Drives ``handle_message`` directly with a loaded ``ScreenNetClassifier`` over an
in-memory sqlite session factory — no broker, no Postgres, no Docker. Skip-guarded
the same way as ``test_classifier_e2e``; run via ``make test-e2e``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from screencropnet_yolo.server import db
from screencropnet_yolo.server.classifier import ScreenNetClassifier
from screencropnet_yolo.server.compression import compress_lossless_webp
from screencropnet_yolo.server.config import get_settings
from screencropnet_yolo.server.db import JobStatus
from screencropnet_yolo.server.schemas import QueueMessage
from screencropnet_yolo.server.worker import handle_message

pytestmark = [pytest.mark.integration, pytest.mark.e2e]


async def test_real_classifier_through_worker_path(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    settings = get_settings()
    if not settings.weights_path.exists():
        pytest.skip(f"weights not found at {settings.weights_path}; run `make download-weights`")
    if not settings.raw_dataset_dir.is_dir():
        pytest.skip(f"dataset dir not found: {settings.raw_dataset_dir}")

    images = sorted(settings.raw_dataset_dir.glob("*_twitter.PNG"))
    if not images:
        pytest.skip(f"no *_twitter.PNG screenshots under {settings.raw_dataset_dir}")
    screenshot = images[0]

    async with session_factory() as session:
        job = await db.create_job(session, original_path=str(screenshot), batch_id="e2e")
    job_id = job.job_id

    compressed = compress_lossless_webp(screenshot, tmp_path)
    msg = QueueMessage(
        job_id=job_id,
        batch_id="e2e",
        compressed_path=str(compressed),
        original_path=str(screenshot),
    )

    classifier = ScreenNetClassifier(settings)
    classifier.load_model()

    await handle_message(
        msg.model_dump_json().encode(),
        classifier=classifier,
        session_factory=session_factory,
    )

    async with session_factory() as session:
        done = await db.get_job(session, job_id)
    if done is None:
        raise AssertionError("job vanished after handle_message")
    if done.status != JobStatus.done:
        raise AssertionError(f"expected job status done, got {done.status} (error={done.error!r})")
    if done.is_twitter is not True:
        raise AssertionError(f"expected is_twitter True, got {done.is_twitter!r}")
    if done.pred_prob is None:
        raise AssertionError("expected pred_prob to be populated")
    if done.time_for_pred is None:
        raise AssertionError("expected time_for_pred to be populated")
