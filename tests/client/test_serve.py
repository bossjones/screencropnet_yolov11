from __future__ import annotations

import os
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from screencropnet_yolo.client import serve
from screencropnet_yolo.server.config import Settings


def test_resolve_no_select_returns_configured_weights(tmp_path: Path) -> None:
    weights = tmp_path / "ScreenNetV1.pth"
    weights.write_bytes(b"x")
    settings = Settings(weights_path=weights)
    if serve.resolve_serve_weights(select=False, settings=settings) != weights:
        raise AssertionError("without --select, the configured weights_path must be returned")


def test_resolve_select_picks_via_selector(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    models = tmp_path / "models"
    pt = runs / "run" / "weights" / "best.pt"
    pth = models / "ScreenNetV1.pth"
    pt.parent.mkdir(parents=True)
    pth.parent.mkdir(parents=True)
    pt.write_bytes(b"x")
    pth.write_bytes(b"x")
    settings = Settings(model_search_roots=[runs, models])

    def selector(choices: list[str]) -> list[str]:
        return [next(c for c in choices if "ScreenNetV1.pth" in c)]

    chosen = serve.resolve_serve_weights(select=True, settings=settings, selector=selector)
    if chosen != pth:
        raise AssertionError(f"selector's .pth pick must win, got {chosen}")


def test_resolve_select_empty_roots_raises(tmp_path: Path) -> None:
    settings = Settings(model_search_roots=[tmp_path / "nope"])
    with pytest.raises(FileNotFoundError):
        serve.resolve_serve_weights(select=True, settings=settings, selector=lambda c: c[:1])


def test_resolve_select_cancel_raises(tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x")
    settings = Settings(model_search_roots=[tmp_path])
    with pytest.raises(RuntimeError):
        serve.resolve_serve_weights(select=True, settings=settings, selector=lambda _c: [])


def test_apply_weights_env_sets_var_and_clears_cache(mocker: MockerFixture, tmp_path: Path) -> None:
    mocker.patch.dict(os.environ, {}, clear=False)
    cache_clear = mocker.patch.object(serve.get_settings, "cache_clear")
    weights = tmp_path / "w.pth"

    serve.apply_weights_env(weights)

    if os.environ.get(serve.WEIGHTS_ENV) != str(weights):
        raise AssertionError("apply_weights_env must export SCREENCROPNET_WEIGHTS_PATH")
    cache_clear.assert_called_once()


def test_serve_launches_uvicorn_and_worker(mocker: MockerFixture, tmp_path: Path) -> None:
    weights = tmp_path / "w.pth"
    weights.write_bytes(b"x")
    settings = Settings(weights_path=weights, api_host="0.0.0.0", api_port=9000)

    mocker.patch.object(serve, "apply_weights_env")
    popen = mocker.patch("screencropnet_yolo.client.serve.subprocess.Popen")
    uvicorn_run = mocker.patch("uvicorn.run")

    serve.serve(select=False, with_worker=True, settings=settings)

    if popen.call_count != 1:
        raise AssertionError("--with-worker must spawn exactly one worker process")
    uvicorn_run.assert_called_once()
    kwargs = uvicorn_run.call_args.kwargs
    if kwargs.get("factory") is not True or kwargs.get("port") != 9000:
        raise AssertionError(f"uvicorn must launch the factory on the configured port: {kwargs}")
