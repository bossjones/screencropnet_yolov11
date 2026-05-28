"""Baseline pins for the YOLO26 migration: ultralytics version and weight names."""

from __future__ import annotations

import ultralytics

from screencropnet_yolo.model import ModelFactory


def _parse_version(version: str) -> tuple[int, ...]:
    """Parse a dotted version string into a comparable integer tuple."""
    parts: list[int] = []
    for token in version.split("."):
        digits = "".join(c for c in token if c.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def test_ultralytics_version_at_least_8_4_52() -> None:
    """YOLO26 requires Ultralytics 8.4.52 or newer."""
    assert _parse_version(ultralytics.__version__) >= (8, 4, 52), ultralytics.__version__


def test_yolo26_weights_string_present_in_model_sizes() -> None:
    """Every default weight must be a yolo26 checkpoint."""
    values = set(ModelFactory.MODEL_SIZES.values())
    assert all(v.startswith("yolo26") for v in values), values
