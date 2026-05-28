"""Tests for the Label Studio YOLO export → dataset converter."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
import yaml

from scripts.ls_yolo_export_to_dataset import build_dataset


def _make_export_zip(zip_path: Path, n: int) -> None:
    """Build a fake LS YOLO export ZIP with ``n`` image/label pairs."""
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("classes.txt", "tweet_region\n")
        for i in range(n):
            zf.writestr(f"images/img_{i:03d}.png", b"\x89PNG\r\n\x1a\n")
            zf.writestr(f"labels/img_{i:03d}.txt", "0 0.5 0.5 0.4 0.3\n")


def test_export_writes_data_yaml_single_class(tmp_path: Path) -> None:
    export = tmp_path / "ls_export.zip"
    _make_export_zip(export, n=10)
    out_dir = tmp_path / "dataset"

    summary = build_dataset(export, out_dir, val_ratio=0.2, seed=42)

    data = yaml.safe_load((out_dir / "data.yaml").read_text())
    assert data["nc"] == 1
    assert data["names"] == ["tweet_region"]
    assert data["train"] == "train/images"
    assert data["val"] == "val/images"

    assert summary["total"] == 10
    assert summary["train"] == 8
    assert summary["val"] == 2

    assert len(list((out_dir / "train" / "images").glob("*.png"))) == 8
    assert len(list((out_dir / "train" / "labels").glob("*.txt"))) == 8
    assert len(list((out_dir / "val" / "images").glob("*.png"))) == 2
    assert len(list((out_dir / "val" / "labels").glob("*.txt"))) == 2


def test_export_split_is_deterministic(tmp_path: Path) -> None:
    export = tmp_path / "ls_export.zip"
    _make_export_zip(export, n=10)

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    build_dataset(export, out_a, val_ratio=0.2, seed=42)
    build_dataset(export, out_b, val_ratio=0.2, seed=42)

    val_a = sorted(p.name for p in (out_a / "val" / "images").glob("*.png"))
    val_b = sorted(p.name for p in (out_b / "val" / "images").glob("*.png"))
    assert val_a == val_b


def test_export_without_pairs_raises(tmp_path: Path) -> None:
    export = tmp_path / "empty.zip"
    with zipfile.ZipFile(export, "w") as zf:
        zf.writestr("images/.keep", b"")
        zf.writestr("labels/.keep", b"")

    with pytest.raises(ValueError):
        build_dataset(export, tmp_path / "out")
