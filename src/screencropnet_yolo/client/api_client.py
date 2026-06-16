"""Async HTTP client and image discovery for the ingest/classify API.

The client compresses each original to a lossless WebP and uploads *that*,
recording the real ``original_path`` so the server can export the true original
later. Folder submission is bounded by a semaphore so a 500-image run never
floods the API.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from screencropnet_yolo.server.compression import compress_lossless_webp
from screencropnet_yolo.server.config import Settings, get_settings
from screencropnet_yolo.server.schemas import ClassifyAccepted, JobView, StatusSummary

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}


def discover_images(folder: Path | str, recursive: bool = True) -> list[Path]:
    """Return image files under ``folder`` (case-insensitive extension match)."""
    folder = Path(folder)
    entries = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in entries if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


class ScreenCropClient:
    def __init__(
        self,
        base_url: str,
        client: httpx.AsyncClient | None = None,
        concurrency: int = 8,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client or httpx.AsyncClient(base_url=base_url)
        self._owns_client = client is None
        self._concurrency = concurrency

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def submit_image(self, original_path: str, batch_id: str) -> ClassifyAccepted:
        original = Path(original_path)
        webp = compress_lossless_webp(original, self._settings.compress_tmp_dir)
        files = {"file": (webp.name, webp.read_bytes(), "image/webp")}
        data = {"original_path": str(original), "batch_id": batch_id}
        resp = await self._client.post("/classify", files=files, data=data)
        resp.raise_for_status()
        return ClassifyAccepted.model_validate(resp.json())

    async def submit_folder(
        self, folder: Path | str, batch_id: str, recursive: bool = True
    ) -> list[ClassifyAccepted]:
        images = discover_images(folder, recursive=recursive)
        semaphore = asyncio.Semaphore(self._concurrency)

        async def _one(path: Path) -> ClassifyAccepted:
            async with semaphore:
                return await self.submit_image(str(path), batch_id)

        return list(await asyncio.gather(*(_one(path) for path in images)))

    async def get_job(self, job_id: str) -> JobView:
        resp = await self._client.get(f"/jobs/{job_id}")
        resp.raise_for_status()
        return JobView.model_validate(resp.json())

    async def list_jobs(
        self, batch_id: str | None = None, status: str | None = None
    ) -> list[JobView]:
        params: dict[str, str] = {}
        if batch_id is not None:
            params["batch_id"] = batch_id
        if status is not None:
            params["status"] = status
        resp = await self._client.get("/jobs", params=params)
        resp.raise_for_status()
        return [JobView.model_validate(item) for item in resp.json()]

    async def list_twitter(self, batch_id: str | None = None) -> list[JobView]:
        params = {"batch_id": batch_id} if batch_id is not None else {}
        resp = await self._client.get("/twitter", params=params)
        resp.raise_for_status()
        return [JobView.model_validate(item) for item in resp.json()]

    async def status(self, batch_id: str | None = None) -> StatusSummary:
        params = {"batch_id": batch_id} if batch_id is not None else {}
        resp = await self._client.get("/status", params=params)
        resp.raise_for_status()
        return StatusSummary.model_validate(resp.json())
