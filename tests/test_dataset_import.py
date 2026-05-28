"""Tests for the Pascal-VOC CSV -> YOLO single-class import pipeline."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from screencropnet_yolo.dataset_import import (
    convert_csv,
    pascal_row_to_yolo,
    prepare_twitter_dataset,
)


def test_pascal_row_to_yolo_collapses_all_labels_to_tweet_region() -> None:
    """A Pascal-VOC box is normalized to YOLO center form under class 0."""
    row = {
        "xmin": 30,
        "ymin": 391,
        "xmax": 1161,
        "ymax": 752,
        "width": 1179,
        "height": 2556,
        "label": "twitter",
    }

    class_id, x_c, y_c, w, h = pascal_row_to_yolo(row, class_map={"twitter": 0, "tweet_region": 0})

    assert class_id == 0
    assert x_c == pytest.approx(0.505089, abs=1e-5)
    assert y_c == pytest.approx(0.223592, abs=1e-5)
    assert w == pytest.approx(0.959288, abs=1e-5)
    assert h == pytest.approx(0.141236, abs=1e-5)


def test_pascal_row_to_yolo_validates_bounds() -> None:
    """A box with xmax < xmin is rejected."""
    row = {
        "xmin": 100,
        "ymin": 10,
        "xmax": 50,
        "ymax": 80,
        "width": 200,
        "height": 200,
        "label": "twitter",
    }

    with pytest.raises(ValueError, match="bbox"):
        pascal_row_to_yolo(row, class_map={"twitter": 0})


def test_pascal_row_to_yolo_missing_column_raises_valueerror() -> None:
    """A row missing a required column yields a descriptive ValueError, not KeyError."""
    row = {"xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50, "label": "twitter"}  # no width/height

    with pytest.raises(ValueError, match="[Cc]olumn"):
        pascal_row_to_yolo(row, class_map={"twitter": 0})


def _write_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["filename", "width", "height", "label", "xmin", "ymin", "xmax", "ymax"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_convert_csv_writes_one_txt_per_image(tmp_path: Path) -> None:
    """One label file is produced per distinct image, regardless of row count."""
    csv_path = tmp_path / "annotations.csv"
    _write_csv(
        csv_path,
        [
            {
                "filename": "a.png",
                "width": 100,
                "height": 100,
                "label": "twitter",
                "xmin": 10,
                "ymin": 10,
                "xmax": 50,
                "ymax": 50,
            },
            {
                "filename": "a.png",
                "width": 100,
                "height": 100,
                "label": "tweet_region",
                "xmin": 60,
                "ymin": 60,
                "xmax": 90,
                "ymax": 90,
            },
            {
                "filename": "b.png",
                "width": 100,
                "height": 100,
                "label": "twitter",
                "xmin": 5,
                "ymin": 5,
                "xmax": 40,
                "ymax": 40,
            },
            {
                "filename": "c.png",
                "width": 100,
                "height": 100,
                "label": "twitter",
                "xmin": 1,
                "ymin": 1,
                "xmax": 20,
                "ymax": 20,
            },
        ],
    )

    out = tmp_path / "out"
    convert_csv(csv_path, out, class_map={"twitter": 0, "tweet_region": 0})

    label_files = sorted((out / "labels").glob("*.txt"))
    assert len(label_files) == 3
    # a.png had two boxes -> two YOLO lines, both class 0
    a_lines = (out / "labels" / "a.txt").read_text().strip().splitlines()
    assert len(a_lines) == 2
    assert all(line.split()[0] == "0" for line in a_lines)


def test_convert_csv_missing_filename_raises_valueerror(tmp_path: Path) -> None:
    """A CSV lacking the filename column yields a descriptive ValueError, not KeyError."""
    csv_path = tmp_path / "no_filename.csv"
    csv_path.write_text("width,height,label,xmin,ymin,xmax,ymax\n100,100,twitter,10,10,50,50\n")

    with pytest.raises(ValueError, match="filename"):
        convert_csv(csv_path, tmp_path / "out", class_map={"twitter": 0})


def test_prepare_twitter_dataset_creates_train_val_split_and_data_yaml(tmp_path: Path) -> None:
    """The full pipeline builds a YOLO directory tree and a single-class data.yaml."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    rows: list[dict[str, object]] = []
    for i in range(5):
        name = f"img{i}.png"
        cv2.imwrite(str(images_dir / name), np.zeros((10, 10, 3), dtype=np.uint8))
        rows.append(
            {
                "filename": name,
                "width": 10,
                "height": 10,
                "label": "twitter",
                "xmin": 1,
                "ymin": 1,
                "xmax": 8,
                "ymax": 8,
            }
        )

    csv_path = tmp_path / "annotations.csv"
    _write_csv(csv_path, rows)

    out = tmp_path / "dataset"
    prepare_twitter_dataset(images_dir, csv_path, out, val_ratio=0.2, seed=42)

    assert (out / "train" / "images").is_dir()
    assert (out / "train" / "labels").is_dir()
    assert (out / "val" / "images").is_dir()

    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["nc"] == 1
    assert data["names"] == ["tweet_region"]
