"""Shared fuzzy model-selection helpers (torch-free; ``fzf`` optional).

Discovery, display formatting, and interactive picking of model weight files.
Kept dependency-light — no torch, no Ultralytics — so importing it costs nothing
and both the ``demo`` YOLO CLI and the ``serve`` classifier launcher can reuse it.

``pyfzf`` (and thus the ``fzf`` binary) is imported lazily inside ``_fzf_select``,
so importing this module never requires ``fzf``; only an actual ``--select`` run
pulls it in.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from screencropnet_yolo.output import human_size

MODEL_EXTS = {".pt", ".onnx"}
"""Weight extensions the YOLO ``demo`` cares about."""

SERVER_MODEL_EXTS = {".pt", ".onnx", ".pth"}
"""Weight extensions the ScreenNet ``serve`` classifier cares about (adds ``.pth``)."""


def discover_models(search_root: Path, exts: set[str] = MODEL_EXTS) -> list[Path]:
    """All model weight files under ``search_root`` matching ``exts``, newest first."""
    if not search_root.is_dir():
        return []
    found = [p for p in search_root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def format_model_choice(path: Path) -> str:
    """Render one fzf line for ``path``: ``'best.pt : /abs/path  [42.0 MB]'``."""
    return f"{path.name} : {path}  [{human_size(path.stat().st_size)}]"


ModelSelector = Callable[[list[str]], list[str]]
"""A picker: given display lines, return the chosen line(s) (empty if cancelled)."""


def _fzf_select(choices: list[str]) -> list[str]:
    """Default selector: hand ``choices`` to fzf via pyfzf, return the chosen line(s).

    ``pyfzf`` is imported lazily so that importing this module never requires the
    ``fzf`` binary — only ``--select`` runs pull it in.
    """
    from pyfzf.pyfzf import FzfPrompt

    # pyfzf ships no type stubs, so .prompt is untyped; the signature here is the contract.
    return FzfPrompt().prompt(choices, "--height=40% --reverse")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


def select_model(candidates: list[Path], *, selector: ModelSelector | None = None) -> Path | None:
    """Present ``candidates`` via a fuzzy picker; return the chosen Path, or None if cancelled.

    The display line is mapped back to its Path through a dict, so the selection is
    recovered exactly rather than re-parsed out of the formatted string.
    """
    choices = {format_model_choice(p): p for p in candidates}
    picked = (selector or _fzf_select)(list(choices))
    if not picked:
        return None
    return choices[picked[0]]


## Tests

import os  # noqa: E402
import tempfile  # noqa: E402


def test_discover_models_finds_pt_and_onnx() -> None:
    with tempfile.TemporaryDirectory() as d:
        runs = Path(d)
        pt = runs / "a" / "train" / "weights" / "best.pt"
        onnx = runs / "b" / "weights" / "best.onnx"
        pt.parent.mkdir(parents=True)
        onnx.parent.mkdir(parents=True)
        pt.write_bytes(b"x")
        onnx.write_bytes(b"x")
        (runs / "notes.txt").write_text("nope")
        os.utime(pt, (1, 1))
        os.utime(onnx, (10_000_000, 10_000_000))
        found = discover_models(runs)
        if found != [onnx, pt]:
            raise AssertionError(f"expected newest-first [onnx, pt], got {found}")


def test_discover_models_empty_when_absent() -> None:
    with tempfile.TemporaryDirectory() as d:
        if discover_models(Path(d) / "missing") != []:
            raise AssertionError("a missing root must yield []")


def test_discover_models_server_exts_include_pth() -> None:
    with tempfile.TemporaryDirectory() as d:
        runs = Path(d)
        pth = runs / "ScreenNetV1.pth"
        pth.write_bytes(b"x")
        if discover_models(runs, SERVER_MODEL_EXTS) != [pth]:
            raise AssertionError("SERVER_MODEL_EXTS must discover .pth weights")
        if discover_models(runs) != []:
            raise AssertionError("default MODEL_EXTS must ignore .pth")


def test_format_model_choice_line() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "best.pt"
        p.write_bytes(b"x" * 1536)  # 1.5 KB
        line = format_model_choice(p)
        if line != f"best.pt : {p}  [1.5 KB]":
            raise AssertionError(f"unexpected choice line: {line!r}")
