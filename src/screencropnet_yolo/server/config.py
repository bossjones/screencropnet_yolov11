"""Runtime settings and device selection for the ingest/classify service.

Torch is imported lazily inside :func:`pick_device` so the API and the entire
unit-test suite can run without torch, weights, or a GPU.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
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
    # Repo-local, gitignored default; overridable via SCREENCROPNET_WEIGHTS_PATH.
    # `_expand_weights_path` handles `~` so env overrides like `~/foo.pth` work too.
    weights_path: Path = Path("scratch/models/ScreenNetV1.pth")
    weights_url: str = "https://www.dropbox.com/scl/fi/8a5cc7e1ngcnm78kcqnga/ScreenNetV1.pth?rlkey=sbxats642fui9gpuwj8susha0&dl=1"
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

    # Per-worker log override: when set, a worker routes its FileHandler here
    # instead of the shared `logs_dir/worker.log`. The supervisor assigns a
    # distinct path per child via SCREENCROPNET_WORKER_LOG_PATH so a fleet
    # doesn't interleave into one file.
    worker_log_path: Path | None = None

    # `screencrop-supervisorctl` fleet defaults. State files (PID/port/weights)
    # live under supervisor_state_dir; each worker gets metrics port base+i.
    supervisor_state_dir: Path = Path("logs/supervisor")
    supervisor_metrics_base_port: int = 8001
    supervisor_workers: int = 2

    # `serve --select` fuzzy-picks weights from these roots; `.pth` (ScreenNet) is
    # included alongside the demo's `.pt`/`.onnx` via SERVER_MODEL_EXTS.
    model_search_roots: list[Path] = [Path("runs"), Path("scratch/models")]

    # `doctor` probe targets. Host ports mirror docker-compose.yml (prometheus is
    # remapped 9091->9090, grafana 3001->3000). worker_metrics_url tracks
    # worker_metrics_port by default; override any of these via SCREENCROPNET_* env.
    prometheus_url: str = "http://127.0.0.1:9091/-/healthy"
    grafana_url: str = "http://127.0.0.1:3001/api/health"
    rabbit_mgmt_url: str = "http://127.0.0.1:15672/"
    worker_metrics_url: str = "http://127.0.0.1:8001/"
    doctor_timeout: float = 2.0

    @field_validator("weights_path", mode="after")
    @classmethod
    def _expand_weights_path(cls, v: Path) -> Path:
        return v.expanduser()


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
