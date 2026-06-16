from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from screencropnet_yolo.server import db
from screencropnet_yolo.server.config import Settings
from screencropnet_yolo.server.queue import FakePublisher


async def test_classify_accepts_persists_and_publishes_once(
    async_client: httpx.AsyncClient,
    fake_publisher: FakePublisher,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    resp = await async_client.post(
        "/classify",
        files={"file": ("x.webp", b"webp-bytes", "image/webp")},
        data={"original_path": "/originals/x.png", "batch_id": "b1"},
    )
    if resp.status_code != 202:
        raise AssertionError(f"expected 202, got {resp.status_code}: {resp.text}")
    payload = resp.json()
    if not payload.get("job_id") or payload.get("batch_id") != "b1":
        raise AssertionError("response must carry job_id and batch_id")

    if len(fake_publisher.published) != 1:
        raise AssertionError("exactly one queue message must be published")
    message = fake_publisher.published[0]
    if message.original_path != "/originals/x.png" or message.job_id != payload["job_id"]:
        raise AssertionError("published message must reference the job and original path")

    async with session_factory() as session:
        jobs = await db.list_jobs(session, batch_id="b1")
    if len(jobs) != 1 or jobs[0].status != db.JobStatus.pending:
        raise AssertionError("exactly one pending job row must be persisted")


@pytest_asyncio.fixture
async def tiny_limit_client(
    build_app: Callable[[Settings], FastAPI],
    tmp_path: Path,
):
    settings = Settings(
        logs_dir=tmp_path / "logs",
        compress_tmp_dir=tmp_path / "uploads",
        max_upload_bytes=8,
    )
    app = build_app(settings)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def test_classify_oversize_returns_413(
    tiny_limit_client: httpx.AsyncClient,
    fake_publisher: FakePublisher,
) -> None:
    resp = await tiny_limit_client.post(
        "/classify",
        files={"file": ("x.webp", b"x" * 64, "image/webp")},
        data={"original_path": "/originals/x.png", "batch_id": "b1"},
    )
    if resp.status_code != 413:
        raise AssertionError(f"oversize upload must return 413, got {resp.status_code}")
    if fake_publisher.published:
        raise AssertionError("no message should be published for a rejected upload")


async def test_get_job_404_for_unknown(async_client: httpx.AsyncClient) -> None:
    resp = await async_client.get("/jobs/does-not-exist")
    if resp.status_code != 404:
        raise AssertionError("unknown job id must return 404")


async def test_status_shape(async_client: httpx.AsyncClient) -> None:
    await async_client.post(
        "/classify",
        files={"file": ("x.webp", b"webp", "image/webp")},
        data={"original_path": "/o/x.png", "batch_id": "b9"},
    )
    resp = await async_client.get("/status", params={"batch_id": "b9"})
    if resp.status_code != 200:
        raise AssertionError("status endpoint must return 200")
    body = resp.json()
    for key in (
        "batch_id",
        "total",
        "counts",
        "twitter_count",
        "done",
        "failed",
        "throughput_per_sec",
    ):
        if key not in body:
            raise AssertionError(f"status payload missing key {key!r}")
    if body["total"] != 1 or body["counts"].get("pending") != 1:
        raise AssertionError("status must reflect exact persisted counts")


async def test_healthz(async_client: httpx.AsyncClient) -> None:
    resp = await async_client.get("/healthz")
    if resp.status_code != 200 or resp.json() != {"ok": True}:
        raise AssertionError("healthz must return {'ok': true}")


@pytest.mark.parametrize("path", ["/jobs", "/twitter"])
async def test_list_endpoints_return_lists(async_client: httpx.AsyncClient, path: str) -> None:
    await async_client.post(
        "/classify",
        files={"file": ("x.webp", b"webp", "image/webp")},
        data={"original_path": "/o/x.png", "batch_id": "bL"},
    )
    resp = await async_client.get(path, params={"batch_id": "bL"})
    if resp.status_code != 200 or not isinstance(resp.json(), list):
        raise AssertionError(f"{path} must return a JSON list")
