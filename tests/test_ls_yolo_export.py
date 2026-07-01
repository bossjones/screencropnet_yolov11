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


def _make_labels_only_zip(zip_path: Path, n: int, with_images_dir: bool = True) -> None:
    """A local-files LS export: labels present but the ``images/`` dir is empty.

    Mirrors what "YOLO with images" produces when tasks reference images via
    ``/data/local-files/`` — the labels come out but no image bytes are bundled.
    ``with_images_dir`` toggles whether an (empty) ``images/`` dir entry exists.
    """
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("classes.txt", "tweet_region\n")
        if with_images_dir:
            zf.writestr("images/", b"")
        for i in range(n):
            zf.writestr(f"labels/img_{i:03d}.txt", "0 0.5 0.5 0.4 0.3\n")


def _write_images(dir_path: Path, names: list[str]) -> None:
    """Create placeholder image files (uppercase ``.PNG``, like the staged dir)."""
    dir_path.mkdir(parents=True, exist_ok=True)
    for name in names:
        (dir_path / name).write_bytes(b"\x89PNG\r\n\x1a\n")


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


def test_export_three_way_split_writes_test(tmp_path: Path) -> None:
    export = tmp_path / "ls_export.zip"
    _make_export_zip(export, n=10)
    out_dir = tmp_path / "dataset"

    summary = build_dataset(export, out_dir, val_ratio=0.2, test_ratio=0.1, seed=42)

    assert summary == {
        "train": 7,
        "val": 2,
        "test": 1,
        "total": 10,
        "data_yaml": str(out_dir / "data.yaml"),
    }

    data = yaml.safe_load((out_dir / "data.yaml").read_text())
    assert data["test"] == "test/images"

    assert len(list((out_dir / "test" / "images").glob("*.png"))) == 1
    assert len(list((out_dir / "test" / "labels").glob("*.txt"))) == 1


def test_export_omits_test_when_ratio_zero(tmp_path: Path) -> None:
    export = tmp_path / "ls_export.zip"
    _make_export_zip(export, n=10)
    out_dir = tmp_path / "dataset"

    build_dataset(export, out_dir, val_ratio=0.2, seed=42)

    data = yaml.safe_load((out_dir / "data.yaml").read_text())
    assert "test" not in data
    assert not (out_dir / "test").exists()


def test_export_ratios_summing_above_one_raise(tmp_path: Path) -> None:
    export = tmp_path / "ls_export.zip"
    _make_export_zip(export, n=10)

    with pytest.raises(ValueError):
        build_dataset(export, tmp_path / "out", val_ratio=0.7, test_ratio=0.4)


def test_export_rerun_replaces_stale_files(tmp_path: Path) -> None:
    """Re-running into a populated dir must not leave files from the prior run."""
    out_dir = tmp_path / "dataset"

    big = tmp_path / "big.zip"
    _make_export_zip(big, n=10)
    build_dataset(big, out_dir, val_ratio=0.2, seed=42)

    small = tmp_path / "small.zip"
    _make_export_zip(small, n=4)
    summary = build_dataset(small, out_dir, val_ratio=0.25, seed=42)

    assert summary["total"] == 4
    train_imgs = list((out_dir / "train" / "images").glob("*.png"))
    val_imgs = list((out_dir / "val" / "images").glob("*.png"))
    assert len(train_imgs) + len(val_imgs) == 4


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


def test_export_labels_only_zip_uses_images_root(tmp_path: Path) -> None:
    """A labels-only export pairs against ``images_root`` (uppercase suffixes ok)."""
    export = tmp_path / "labels_only.zip"
    _make_labels_only_zip(export, n=10)
    images_root = tmp_path / "train_images"
    _write_images(images_root, [f"img_{i:03d}.PNG" for i in range(10)])
    out_dir = tmp_path / "dataset"

    summary = build_dataset(
        export, out_dir, val_ratio=0.2, test_ratio=0.1, seed=42, images_root=images_root
    )

    assert summary["total"] == 10
    assert summary["train"] + summary["val"] + summary["test"] == 10
    train_imgs = list((out_dir / "train" / "images").glob("*.PNG"))
    assert len(train_imgs) == summary["train"]


def test_export_no_images_dir_uses_images_root(tmp_path: Path) -> None:
    """Missing ``images/`` dir entirely is tolerated when ``images_root`` is given."""
    export = tmp_path / "no_images_dir.zip"
    _make_labels_only_zip(export, n=6, with_images_dir=False)
    images_root = tmp_path / "train_images"
    _write_images(images_root, [f"img_{i:03d}.PNG" for i in range(6)])

    summary = build_dataset(export, tmp_path / "dataset", val_ratio=0.5, images_root=images_root)

    assert summary["total"] == 6


def test_export_images_root_supplements_partial_zip(tmp_path: Path) -> None:
    """Images bundled in the zip are used; the rest are filled from ``images_root``."""
    export = tmp_path / "partial.zip"
    with zipfile.ZipFile(export, "w") as zf:
        zf.writestr("classes.txt", "tweet_region\n")
        for i in range(10):
            zf.writestr(f"labels/img_{i:03d}.txt", "0 0.5 0.5 0.4 0.3\n")
            if i < 4:
                zf.writestr(f"images/img_{i:03d}.png", b"\x89PNG\r\n\x1a\n")
    images_root = tmp_path / "train_images"
    _write_images(images_root, [f"img_{i:03d}.PNG" for i in range(4, 10)])

    summary = build_dataset(export, tmp_path / "dataset", val_ratio=0.2, images_root=images_root)

    assert summary["total"] == 10


def test_export_labels_only_without_images_root_still_raises(tmp_path: Path) -> None:
    """No ``images_root`` and no bundled images → the original error stands."""
    export = tmp_path / "labels_only.zip"
    _make_labels_only_zip(export, n=10)

    with pytest.raises(ValueError):
        build_dataset(export, tmp_path / "out")
