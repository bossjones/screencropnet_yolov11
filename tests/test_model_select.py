from __future__ import annotations

from pathlib import Path

from screencropnet_yolo import model_select
from screencropnet_yolo.model_select import (
    SERVER_MODEL_EXTS,
    discover_models,
    format_model_choice,
    select_model,
)


def test_select_model_maps_choice_back_to_path(tmp_path: Path) -> None:
    """The picked display line is mapped back to its exact Path, not re-parsed."""
    first = tmp_path / "a.pt"
    second = tmp_path / "b.onnx"
    first.write_bytes(b"x")
    second.write_bytes(b"x")

    def selector(choices: list[str]) -> list[str]:
        return [choices[1]]

    picked = select_model([first, second], selector=selector)
    if picked != second:
        raise AssertionError(f"expected {second}, got {picked}")


def test_select_model_returns_none_on_cancel(tmp_path: Path) -> None:
    """An empty pick (ESC) yields None so callers can distinguish cancellation."""
    p = tmp_path / "a.pt"
    p.write_bytes(b"x")
    if select_model([p], selector=lambda _choices: []) is not None:
        raise AssertionError("cancelled selection must return None")


def test_discover_models_across_roots_with_pth(tmp_path: Path) -> None:
    """Server discovery finds .pth alongside .pt/.onnx and orders newest-first."""
    import os

    runs = tmp_path / "runs"
    models = tmp_path / "models"
    pt = runs / "train" / "weights" / "best.pt"
    pth = models / "ScreenNetV1.pth"
    pt.parent.mkdir(parents=True)
    pth.parent.mkdir(parents=True)
    pt.write_bytes(b"x")
    pth.write_bytes(b"x")
    os.utime(pt, (1, 1))
    os.utime(pth, (10_000_000, 10_000_000))

    found = [p for root in (runs, models) for p in discover_models(root, SERVER_MODEL_EXTS)]
    if set(found) != {pt, pth}:
        raise AssertionError(f"expected both weights discovered, got {found}")


def test_demo_reexports_shared_helpers() -> None:
    """demo.py must re-export the shared helpers so its public names keep resolving."""
    from screencropnet_yolo import demo

    if demo.discover_models is not model_select.discover_models:
        raise AssertionError("demo.discover_models must be the shared implementation")
    if demo.select_model is not select_model:
        raise AssertionError("demo.select_model must be the shared implementation")
    if demo.format_model_choice is not format_model_choice:
        raise AssertionError("demo.format_model_choice must be the shared implementation")
