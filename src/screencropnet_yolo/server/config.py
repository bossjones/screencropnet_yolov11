"""Runtime settings and device selection for the ingest/classify service.

Torch is imported lazily inside :func:`pick_device` so the API and the entire
unit-test suite can run without torch, weights, or a GPU.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SCREENCROPNET_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    postgres_dsn: str = "postgresql+asyncpg://screencrop:screencrop@localhost:5432/screencrop"
    rabbit_url: str = "amqp://guest:guest@localhost:5672/"
    worker_queue_name: str = "screennet_inference_queue"
    rabbit_prefetch_count: int = 8
    class_names: list[str] = ["facebook", "tiktok", "twitter"]
    arch: str = "efficientnet_b0"
    weights_path: Path = Path("~/Documents/my_models/ScreenNetV1.pth").expanduser()
    device_preference: list[str] = ["mps", "cuda", "cpu"]
    max_upload_bytes: int = 25 * 1024 * 1024
    compress_tmp_dir: Path = Path("/tmp/screencropnet_uploads")
    client_concurrency: int = 8
    raw_dataset_dir: Path = Path("scratch/datasets/twitter_screenshots_raw/train_images")
    export_label: str = "twitter"
    export_index_pad: int = 5
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    worker_metrics_port: int = 8001
    logs_dir: Path = Path("logs")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def pick_device(preference: list[str]) -> str:
    """Return the first available device from ``preference``.

    Falls back to ``"cpu"`` whenever torch is unavailable, so callers in
    torch-free environments still get a usable device string.
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    for device in preference:
        if device == "cuda" and torch.cuda.is_available():
            return "cuda"
        if device == "mps" and torch.backends.mps.is_available():
            return "mps"
        if device == "cpu":
            return "cpu"
    return "cpu"
