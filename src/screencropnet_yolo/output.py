"""
Presentation helpers for the training CLI's user-facing output.

This module is deliberately pure: no Ultralytics/torch imports, no I/O. It owns the
run-configuration banner, the closing artifacts table, a `human_size` byte formatter,
a raw-ANSI `colorize` helper, and a color-aware `logging.Formatter`. Keeping it
dependency-light makes the formatting logic trivially unit-testable.

ANSI is raw escape codes (not `rich`) so callers can render colored output without
pulling a TUI dependency into the training path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from typing_extensions import override

_UNITS = ["B", "KB", "MB", "GB", "TB", "PB"]


class Color:
    """Raw ANSI SGR codes. Use via `colorize` so disabling is a single switch."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"


def colorize(text: str, color: str, *, enabled: bool) -> str:
    """Wrap `text` in an ANSI `color` + reset, or return it unchanged when disabled.

    `enabled` is the resolved decision (CLI flag AND TTY AND no NO_COLOR); an empty
    `color` also returns the text unchanged so callers can pass a conditional code.
    """
    if not enabled or not color:
        return text
    return f"{color}{text}{Color.RESET}"


def human_size(n_bytes: int) -> str:
    """Format a byte count as a short human-readable string (e.g. `1.5 KB`)."""
    if n_bytes < 0:
        raise ValueError(f"byte count must be non-negative, got {n_bytes}")
    size = float(n_bytes)
    for unit in _UNITS:
        if size < 1024 or unit == _UNITS[-1]:
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.1f} {unit}"
        size /= 1024
    # Unreachable: the loop always returns on the last unit.
    return f"{size:.1f} {_UNITS[-1]}"


@dataclass
class Artifact:
    """One row in the closing artifacts table.

    `path`/`size` are None when the artifact wasn't produced (rendered as "missing"
    rather than crashing the summary).
    """

    label: str
    path: str | None
    size: int | None


def format_run_summary(
    *,
    model_size: str,
    arch: str,
    device: str,
    epochs: int,
    batch: int,
    imgsz: int,
    dataset_path: str,
    output_dir: str,
    weights_dir: str,
    best_pt: str,
    export_formats: list[str],
    enabled: bool = False,
) -> str:
    """Render the startup run-configuration banner as a multi-line string."""
    title = colorize("RUN CONFIGURATION", Color.BOLD + Color.CYAN, enabled=enabled)
    rule = "=" * 60
    rows = [
        ("Model", f"{arch} (size={model_size})"),
        ("Device", device),
        ("Epochs", str(epochs)),
        ("Batch", str(batch)),
        ("Image size", str(imgsz)),
        ("Dataset", dataset_path),
        ("Output dir", output_dir),
        ("Weights dir", weights_dir),
        ("Expected best .pt", best_pt),
        ("Export formats", ", ".join(export_formats)),
    ]
    width = max(len(label) for label, _ in rows)
    lines = [rule, title, rule]
    for label, value in rows:
        key = colorize(f"{label:>{width}}", Color.DIM, enabled=enabled)
        lines.append(f"  {key} : {value}")
    lines.append(rule)
    return "\n".join(lines)


def format_artifacts_table(
    rows: list[Artifact],
    *,
    best_epoch: int | None = None,
    best_map: float | None = None,
    enabled: bool = False,
) -> str:
    """Render the closing artifacts table with paths and human-readable sizes."""
    title = colorize("ARTIFACTS", Color.BOLD + Color.CYAN, enabled=enabled)
    rule = "=" * 60
    lines = [rule, title, rule]

    label_width = max((len(r.label) for r in rows), default=0)
    for row in rows:
        label = colorize(f"{row.label:>{label_width}}", Color.DIM, enabled=enabled)
        if row.path is None:
            value = colorize("(missing)", Color.YELLOW, enabled=enabled)
        else:
            size = human_size(row.size) if row.size is not None else "(missing)"
            value = f"{row.path}  [{size}]"
        lines.append(f"  {label} : {value}")

    if best_epoch is not None or best_map is not None:
        lines.append("-" * 60)
        if best_epoch is not None:
            lines.append(f"  Best epoch    : {best_epoch}")
        if best_map is not None:
            lines.append(f"  Best mAP50-95 : {best_map:.4f}")
    lines.append(rule)
    return "\n".join(lines)


class ColorFormatter(logging.Formatter):
    """Logging formatter that colorizes the levelname when `enabled`.

    With `enabled=False` it behaves exactly like a plain `logging.Formatter`, so the
    same format string can drive both the colored stream handler and the plain file
    handler. Only the levelname is colored; the record is restored after formatting so
    other handlers see the original value.
    """

    _LEVEL_COLORS = {
        "DEBUG": Color.DIM,
        "INFO": Color.GREEN,
        "WARNING": Color.YELLOW,
        "ERROR": Color.RED,
        "CRITICAL": Color.BOLD + Color.RED,
    }

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        *,
        enabled: bool = False,
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.enabled = enabled

    @override
    def format(self, record: logging.LogRecord) -> str:
        color = self._LEVEL_COLORS.get(record.levelname) if self.enabled else None
        if not color:
            return super().format(record)
        original = record.levelname
        record.levelname = colorize(record.levelname, color, enabled=True)
        try:
            return super().format(record)
        finally:
            record.levelname = original


## Tests


def test_human_size_edges() -> None:
    cases = {0: "0 B", 1: "1 B", 1024: "1.0 KB", 1536: "1.5 KB"}
    for n, expected in cases.items():
        if human_size(n) != expected:
            raise AssertionError(f"human_size({n}) == {human_size(n)!r}, expected {expected!r}")


def test_colorize_toggle() -> None:
    if colorize("x", Color.GREEN, enabled=False) != "x":
        raise AssertionError("disabled colorize must return text unchanged")
    enabled = colorize("x", Color.GREEN, enabled=True)
    if not (enabled.startswith(Color.GREEN) and enabled.endswith(Color.RESET)):
        raise AssertionError("enabled colorize must wrap with code and reset")
