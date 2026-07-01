from __future__ import annotations

import sys
import types
from pathlib import Path

from pytest import MonkeyPatch
from pytest_mock import MockerFixture

from screencropnet_yolo.server.config import Settings, get_settings, pick_device


def test_pick_device_falls_back_to_cpu_without_torch(mocker: MockerFixture) -> None:
    # Setting the module to None makes `import torch` raise ImportError.
    mocker.patch.dict(sys.modules, {"torch": None})
    if "cpu" != pick_device(["mps", "cuda", "cpu"]):
        raise AssertionError("expected cpu fallback when torch is unavailable")


def test_pick_device_prefers_first_available_backend(mocker: MockerFixture) -> None:
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
    )
    mocker.patch.dict(sys.modules, {"torch": fake_torch})
    if "mps" != pick_device(["mps", "cuda", "cpu"]):
        raise AssertionError("expected mps when only mps backend is available")


def test_pick_device_skips_unavailable_then_returns_cpu(mocker: MockerFixture) -> None:
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
    )
    mocker.patch.dict(sys.modules, {"torch": fake_torch})
    if "cpu" != pick_device(["mps", "cuda", "cpu"]):
        raise AssertionError("expected cpu when no accelerator backend is available")


def test_settings_defaults() -> None:
    settings = Settings()
    if settings.worker_queue_name != "screennet_inference_queue":
        raise AssertionError("unexpected default queue name")
    if settings.class_names != ["facebook", "tiktok", "twitter"]:
        raise AssertionError("unexpected default class names")
    if settings.max_upload_bytes != 25 * 1024 * 1024:
        raise AssertionError("unexpected default max upload size")
    if settings.export_label != "twitter":
        raise AssertionError("unexpected default export label")


def test_env_overrides(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("SCREENCROPNET_POSTGRES_DSN", "postgresql+asyncpg://x@h/db")
    monkeypatch.setenv("SCREENCROPNET_WEIGHTS_PATH", "/tmp/custom.pth")
    settings = Settings()
    if settings.postgres_dsn != "postgresql+asyncpg://x@h/db":
        raise AssertionError("SCREENCROPNET_POSTGRES_DSN did not override default")
    if settings.weights_path != Path("/tmp/custom.pth"):
        raise AssertionError("SCREENCROPNET_WEIGHTS_PATH did not override default")


def test_weights_path_default_is_repo_local() -> None:
    if Settings().weights_path != Path("scratch/models/ScreenNetV1.pth"):
        raise AssertionError("expected repo-local default weights path")


def test_weights_path_expands_tilde_override(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("SCREENCROPNET_WEIGHTS_PATH", "~/some_dir/ScreenNetV1.pth")
    weights_path = Settings().weights_path
    if str(weights_path).startswith("~"):
        raise AssertionError("`~`-based override should expand to an absolute path")
    if not weights_path.is_absolute():
        raise AssertionError("expanded weights path should be absolute")


def test_weights_url_is_direct_download() -> None:
    if not Settings().weights_url.endswith("dl=1"):
        raise AssertionError("weights_url should be a Dropbox direct-download link (dl=1)")


def test_get_settings_is_cached() -> None:
    if get_settings() is not get_settings():
        raise AssertionError("get_settings() should return a cached singleton")


def test_supervisor_defaults() -> None:
    settings = Settings()
    if settings.worker_log_path is not None:
        raise AssertionError("worker_log_path should default to None (single shared worker.log)")
    if settings.supervisor_state_dir != Path("logs/supervisor"):
        raise AssertionError("unexpected default supervisor_state_dir")
    if settings.supervisor_metrics_base_port != 8001:
        raise AssertionError("unexpected default supervisor_metrics_base_port")
    if settings.supervisor_workers != 2:
        raise AssertionError("unexpected default supervisor_workers")


def test_supervisor_env_overrides(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("SCREENCROPNET_WORKER_LOG_PATH", "/tmp/w7.log")
    monkeypatch.setenv("SCREENCROPNET_SUPERVISOR_STATE_DIR", "/tmp/sup")
    monkeypatch.setenv("SCREENCROPNET_SUPERVISOR_METRICS_BASE_PORT", "9100")
    monkeypatch.setenv("SCREENCROPNET_SUPERVISOR_WORKERS", "5")
    settings = Settings()
    if settings.worker_log_path != Path("/tmp/w7.log"):
        raise AssertionError("SCREENCROPNET_WORKER_LOG_PATH did not override default")
    if settings.supervisor_state_dir != Path("/tmp/sup"):
        raise AssertionError("SCREENCROPNET_SUPERVISOR_STATE_DIR did not override default")
    if settings.supervisor_metrics_base_port != 9100:
        raise AssertionError("SCREENCROPNET_SUPERVISOR_METRICS_BASE_PORT did not override default")
    if settings.supervisor_workers != 5:
        raise AssertionError("SCREENCROPNET_SUPERVISOR_WORKERS did not override default")
