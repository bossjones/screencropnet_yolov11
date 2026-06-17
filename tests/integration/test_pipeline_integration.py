"""End-to-end pipeline against real Postgres + RabbitMQ.

Publishes 50+ jobs through a real broker and consumes them with the aio-pika
``on_message`` wrapper, injecting ``FakeClassifier`` (no weights/GPU). Requires
`make services-up`; excluded from `make test` via the ``integration`` marker.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import aio_pika
import pytest
from PIL import Image

from screencropnet_yolo.server import db
from screencropnet_yolo.server.classifier import FakeClassifier
from screencropnet_yolo.server.config import get_settings
from screencropnet_yolo.server.queue import RabbitPublisher
from screencropnet_yolo.server.schemas import QueueMessage, StatusSummary
from screencropnet_yolo.server.worker import on_message

pytestmark = pytest.mark.integration

IMAGE_COUNT = 60


async def test_pipeline_processes_all_jobs(tmp_path: Path) -> None:
    settings = get_settings()
    engine = db.make_engine(settings.postgres_dsn)
    async with engine.begin() as conn:
        await conn.run_sync(db.Base.metadata.drop_all)
        await conn.run_sync(db.Base.metadata.create_all)
    session_factory = db.make_sessionmaker(engine)

    # Seed pending jobs + a real webp per image, and publish each to the broker.
    publisher = RabbitPublisher(settings.rabbit_url, settings.worker_queue_name)
    await publisher.connect()
    for i in range(IMAGE_COUNT):
        webp = tmp_path / f"img{i}.webp"
        Image.new("RGB", (8, 8)).save(webp, format="WEBP")
        async with session_factory() as session:
            job = await db.create_job(
                session, original_path=f"/o/img{i}.png", batch_id="integration"
            )
        await publisher.publish(
            QueueMessage(
                job_id=job.job_id,
                batch_id="integration",
                compressed_path=str(webp),
                original_path=f"/o/img{i}.png",
            )
        )
    await publisher.close()

    # Consume with the real aio-pika wrapper + FakeClassifier (all twitter-positive).
    connection = await aio_pika.connect_robust(settings.rabbit_url)
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=settings.rabbit_prefetch_count)
    queue = await channel.declare_queue(settings.worker_queue_name, durable=True)
    classifier = FakeClassifier(pred_class="twitter")

    async def _consume(message: aio_pika.abc.AbstractIncomingMessage) -> None:
        await on_message(message, classifier=classifier, session_factory=session_factory)

    tag = await queue.consume(_consume)

    async def _await_completion() -> StatusSummary:
        while True:
            async with session_factory() as session:
                summary = await db.status_summary(session, batch_id="integration")
            if summary.done + summary.failed >= IMAGE_COUNT:
                return summary
            await asyncio.sleep(0.2)

    try:
        summary = await asyncio.wait_for(_await_completion(), timeout=25)
    finally:
        await queue.cancel(tag)
        await connection.close()
        await engine.dispose()

    if summary.done != IMAGE_COUNT:
        raise AssertionError(f"expected all {IMAGE_COUNT} jobs done, got {summary.done}")
    if summary.failed != 0:
        raise AssertionError(f"no jobs should fail, got {summary.failed}")
    if summary.twitter_count != IMAGE_COUNT:
        raise AssertionError("every job classified twitter must be counted")
    if summary.throughput_per_sec <= 0:
        raise AssertionError("throughput must be positive")
