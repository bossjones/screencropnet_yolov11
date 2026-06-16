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


def test_get_settings_is_cached() -> None:
    if get_settings() is not get_settings():
        raise AssertionError("get_settings() should return a cached singleton")
