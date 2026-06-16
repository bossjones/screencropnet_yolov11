from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
from PIL import Image
from pytest_mock import MockerFixture

from screencropnet_yolo.client.api_client import ScreenCropClient
from screencropnet_yolo.server.config import Settings
from screencropnet_yolo.server.queue import FakePublisher


def _make_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 10)).save(path, format="PNG")
    return path


async def test_submit_image_uploads_compressed_webp_with_original_path(
    async_client: httpx.AsyncClient,
    fake_publisher: FakePublisher,
    tmp_path: Path,
) -> None:
    settings = Settings(compress_tmp_dir=tmp_path / "client_tmp")
    client = ScreenCropClient(base_url="http://test", client=async_client, settings=settings)
    original = _make_image(tmp_path / "shot.png")

    accepted = await client.submit_image(str(original), batch_id="b1")
    if accepted.batch_id != "b1" or not accepted.job_id:
        raise AssertionError("submit_image must return the accepted job")

    message = fake_publisher.published[0]
    if message.original_path != str(original):
        raise AssertionError("the upload must carry the real original_path")
    with Image.open(message.compressed_path) as uploaded:
        if uploaded.format != "WEBP":
            raise AssertionError("the uploaded payload must be the compressed WebP")


async def test_submit_folder_discovers_recursively(
    async_client: httpx.AsyncClient,
    fake_publisher: FakePublisher,
    tmp_path: Path,
) -> None:
    settings = Settings(compress_tmp_dir=tmp_path / "client_tmp")
    client = ScreenCropClient(base_url="http://test", client=async_client, settings=settings)
    _make_image(tmp_path / "src" / "a.png")
    _make_image(tmp_path / "src" / "nested" / "b.png")
    (tmp_path / "src" / "ignore.txt").write_text("nope")

    results = await client.submit_folder(tmp_path / "src", batch_id="bF", recursive=True)
    if len(results) != 2 or len(fake_publisher.published) != 2:
        raise AssertionError("submit_folder must submit every discovered image once")


async def test_submit_folder_respects_concurrency_cap(
    async_client: httpx.AsyncClient,
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    settings = Settings(compress_tmp_dir=tmp_path / "client_tmp")
    client = ScreenCropClient(
        base_url="http://test", client=async_client, settings=settings, concurrency=2
    )
    for i in range(6):
        _make_image(tmp_path / "src" / f"img{i}.png")

    in_flight = 0
    peak = 0

    async def _instrumented(original_path: str, batch_id: str):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return None

    mocker.patch.object(client, "submit_image", side_effect=_instrumented)
    await client.submit_folder(tmp_path / "src", batch_id="bC", recursive=True)
    if peak > 2:
        raise AssertionError(f"concurrency cap of 2 was exceeded (peak {peak})")


async def test_status_parses_summary(async_client: httpx.AsyncClient, tmp_path: Path) -> None:
    settings = Settings(compress_tmp_dir=tmp_path / "client_tmp")
    client = ScreenCropClient(base_url="http://test", client=async_client, settings=settings)
    original = _make_image(tmp_path / "shot.png")
    await client.submit_image(str(original), batch_id="bS")

    summary = await client.status(batch_id="bS")
    if summary.total != 1 or summary.counts.get("pending") != 1:
        raise AssertionError("status() must parse the StatusSummary with exact counts")
