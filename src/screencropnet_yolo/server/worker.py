"""RabbitMQ worker: classify off the event loop, write results to Postgres.

``handle_message`` is the pure core (bytes in, DB writes out) so it can be
unit-tested with a ``FakeClassifier`` and sqlite, with no broker. ``on_message``
is the thin aio-pika wrapper exercised in integration.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Protocol, cast

import aio_pika
import anyio.to_thread
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from screencropnet_yolo.server import db
from screencropnet_yolo.server.classifier import Classifier, ScreenNetClassifier, is_twitter
from screencropnet_yolo.server.config import Settings, get_settings
from screencropnet_yolo.server.db import make_engine, make_sessionmaker
from screencropnet_yolo.server.metrics import (
    JOBS_IN_PROGRESS,
    JOBS_PROCESSED,
    PRED_LATENCY,
    TWITTER_POSITIVE,
    start_worker_metrics_server,
)
from screencropnet_yolo.server.schemas import QueueMessage


async def handle_message(
    body: bytes,
    *,
    classifier: Classifier,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    msg = QueueMessage.model_validate_json(body)
    async with session_factory() as session:
        await db.mark_processing(session, msg.job_id)
        JOBS_IN_PROGRESS.inc()
        try:
            with Image.open(msg.compressed_path) as image:
                result = await anyio.to_thread.run_sync(classifier.infer, image)
            first = result[0]
            time_for_pred = cast(float, first["time_for_pred"])
            twitter = is_twitter(result)
            await db.mark_done(
                session,
                msg.job_id,
                is_twitter=twitter,
                pred_class=cast(str, first["pred_class"]),
                pred_prob=cast(float, first["pred_prob"]),
                time_for_pred=time_for_pred,
            )
            JOBS_PROCESSED.labels(status="done").inc()
            PRED_LATENCY.observe(time_for_pred)
            if twitter:
                TWITTER_POSITIVE.inc()
        except Exception as exc:
            await db.mark_failed(session, msg.job_id, error=str(exc))
            JOBS_PROCESSED.labels(status="failed").inc()
        finally:
            JOBS_IN_PROGRESS.dec()


class IncomingMessage(Protocol):
    """The slice of an aio-pika incoming message that ``on_message`` needs."""

    body: bytes

    def process(self) -> contextlib.AbstractAsyncContextManager[object]: ...


async def on_message(
    message: IncomingMessage,
    *,
    classifier: Classifier,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with message.process():
        await handle_message(message.body, classifier=classifier, session_factory=session_factory)


async def run_worker(settings: Settings) -> None:
    classifier = ScreenNetClassifier(settings)
    classifier.load_model()
    start_worker_metrics_server(settings.worker_metrics_port)

    session_factory = make_sessionmaker(make_engine(settings.postgres_dsn))
    connection = await aio_pika.connect_robust(settings.rabbit_url)
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=settings.rabbit_prefetch_count)
    queue = await channel.declare_queue(settings.worker_queue_name, durable=True)

    async def _consume(message: aio_pika.abc.AbstractIncomingMessage) -> None:
        await on_message(message, classifier=classifier, session_factory=session_factory)

    await queue.consume(_consume)
    await asyncio.Future()


def main() -> None:
    asyncio.run(run_worker(get_settings()))
