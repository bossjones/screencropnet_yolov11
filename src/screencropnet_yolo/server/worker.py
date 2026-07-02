"""RabbitMQ worker: classify off the event loop, write results to Postgres.

``handle_message`` is the pure core (bytes in, DB writes out) so it can be
unit-tested with a ``FakeClassifier`` and sqlite, with no broker. ``on_message``
is the thin aio-pika wrapper exercised in integration.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
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

logger = logging.getLogger("screencropnet_yolo.worker")


def _configure_logging(settings: Settings) -> None:
    log_path = settings.worker_log_path or settings.logs_dir / "worker.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    if not any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(log_path)
        for h in logger.handlers
    ):
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)


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
            logger.info("job %s done (twitter=%s)", msg.job_id, twitter)
        except Exception as exc:
            await db.mark_failed(session, msg.job_id, error=str(exc))
            JOBS_PROCESSED.labels(status="failed").inc()
            logger.exception("job %s failed: %s", msg.job_id, exc)
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


class _CancellableConsumer(Protocol):
    """The slice of an aio-pika queue that warm shutdown needs."""

    async def cancel(self, consumer_tag: str) -> object: ...


async def drain_and_cancel(
    queue: _CancellableConsumer,
    consumer_tag: str,
    in_flight: set[asyncio.Task[None]],
    *,
    timeout: float,
) -> None:
    """Stop new deliveries, then let in-flight handlers finish (bounded by ``timeout``).

    Cancelling the consumer tag first guarantees no *new* messages arrive while we
    wait; any handler that overruns ``timeout`` is left running (its message stays
    unacked and the broker requeues it for another worker).
    """
    await queue.cancel(consumer_tag)
    if in_flight:
        await asyncio.wait(in_flight, timeout=timeout)


async def run_worker(settings: Settings, *, shutdown_timeout: float = 30.0) -> None:
    _configure_logging(settings)
    classifier = ScreenNetClassifier(settings)
    classifier.load_model()
    start_worker_metrics_server(settings.worker_metrics_port)

    session_factory = make_sessionmaker(make_engine(settings.postgres_dsn))
    connection = await aio_pika.connect_robust(settings.rabbit_url)
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=settings.rabbit_prefetch_count)
    queue = await channel.declare_queue(settings.worker_queue_name, durable=True)
    logger.info("worker consuming from %s", settings.worker_queue_name)

    in_flight: set[asyncio.Task[None]] = set()

    async def _consume(message: aio_pika.abc.AbstractIncomingMessage) -> None:
        task = asyncio.current_task()
        if task is not None:
            in_flight.add(task)
        try:
            await on_message(message, classifier=classifier, session_factory=session_factory)
        finally:
            if task is not None:
                in_flight.discard(task)

    consumer_tag = await queue.consume(_consume)

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()
    logger.info("shutdown requested; draining in-flight jobs up to %.1fs", shutdown_timeout)
    await drain_and_cancel(queue, consumer_tag, in_flight, timeout=shutdown_timeout)
    await connection.close()
    logger.info("worker stopped")


def main() -> None:
    asyncio.run(run_worker(get_settings()))
