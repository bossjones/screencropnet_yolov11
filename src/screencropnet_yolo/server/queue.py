"""RabbitMQ publishing behind a Protocol seam.

The API depends only on the :class:`Publisher` protocol so tests can inject
:class:`FakePublisher` and run without a broker. Messages are small JSON bodies
(``QueueMessage``), not pickled images.
"""

from __future__ import annotations

from typing import Protocol

import aio_pika

from screencropnet_yolo.server.schemas import QueueMessage


class Publisher(Protocol):
    async def publish(self, msg: QueueMessage) -> None: ...


class RabbitPublisher:
    def __init__(self, url: str, queue_name: str) -> None:
        self._url = url
        self._queue_name = queue_name
        self._connection: aio_pika.abc.AbstractRobustConnection | None = None
        self._channel: aio_pika.abc.AbstractChannel | None = None

    async def connect(self) -> None:
        self._connection = await aio_pika.connect_robust(self._url)
        self._channel = await self._connection.channel()
        await self._channel.declare_queue(self._queue_name, durable=True)

    async def publish(self, msg: QueueMessage) -> None:
        if self._channel is None:
            await self.connect()
        assert self._channel is not None
        message = aio_pika.Message(
            body=msg.model_dump_json().encode(),
            headers={"job_id": msg.job_id},
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            content_type="application/json",
        )
        await self._channel.default_exchange.publish(message, routing_key=self._queue_name)

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()


class FakePublisher:
    """In-memory publisher that records messages for assertions."""

    def __init__(self) -> None:
        self.published: list[QueueMessage] = []

    async def publish(self, msg: QueueMessage) -> None:
        self.published.append(msg)


## Tests


async def test_fake_publisher_records_messages() -> None:
    publisher = FakePublisher()
    msg = QueueMessage(
        job_id="j1", batch_id="b1", compressed_path="/tmp/x.webp", original_path="/o/x.png"
    )
    await publisher.publish(msg)
    if publisher.published != [msg]:
        raise AssertionError("FakePublisher must record each published message")
