"""Tests for the Pascal-VOC CSV → Label Studio tasks converter."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from scripts.pascal_csv_to_ls_tasks import bbox_to_ls_value, load_tasks, row_to_task


def test_row_to_task_normalizes_to_percent() -> None:
    row = {
        "img_path": "train_images/00000_twitter.PNG",
        "xmin": "30",
        "ymin": "391",
        "xmax": "1161",
        "ymax": "752",
        "width": "1179",
        "height": "2556",
        "label": "twitter",
    }
    task = row_to_task(row, images_url_prefix="/data/local-files/?d=train_images")

    assert task["data"]["image"] == "/data/local-files/?d=train_images/00000_twitter.PNG"
    value = task["predictions"][0]["result"][0]["value"]
    assert value["x"] == pytest.approx(2.544, abs=1e-3)
    assert value["y"] == pytest.approx(15.298, abs=1e-3)
    assert value["width"] == pytest.approx(95.929, abs=1e-3)
    assert value["height"] == pytest.approx(14.123, abs=1e-3)
    assert value["rectanglelabels"] == ["tweet_region"]


def test_bbox_to_ls_value_clamps_out_of_bounds() -> None:
    value = bbox_to_ls_value(xmin=-5, ymin=0, xmax=120, ymax=50, width=100, height=100)
    assert value["x"] == pytest.approx(0.0)
    assert value["width"] == pytest.approx(100.0)
    assert value["height"] == pytest.approx(50.0)


def test_bbox_to_ls_value_rejects_degenerate_box() -> None:
    with pytest.raises(ValueError):
        bbox_to_ls_value(xmin=10, ymin=10, xmax=10, ymax=20, width=100, height=100)


def test_load_tasks_raises_on_missing_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        dedent(
            """\
            img_path,xmin,ymin,xmax,ymax,height,label
            train_images/ok.PNG,10,10,90,90,100,twitter
            """
        )
    )

    with pytest.raises(ValueError, match="missing required column"):
        load_tasks(csv_path, images_url_prefix="/img")


def test_load_tasks_skips_malformed_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        dedent(
            """\
            img_path,xmin,ymin,xmax,ymax,width,height,label
            train_images/ok.PNG,10,10,90,90,100,100,twitter
            train_images/bad.PNG,10,10,5,5,100,100,twitter
            """
        )
    )

    tasks, skipped = load_tasks(csv_path, images_url_prefix="/img")

    assert len(tasks) == 1
    assert skipped == 1
    assert tasks[0]["data"]["image"] == "/img/ok.PNG"


def test_load_tasks_skips_missing_image_files(tmp_path: Path) -> None:
    images_root = tmp_path / "images"
    images_root.mkdir()
    (images_root / "present.PNG").write_bytes(b"x")

    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        dedent(
            """\
            img_path,xmin,ymin,xmax,ymax,width,height,label
            train_images/present.PNG,10,10,90,90,100,100,twitter
            train_images/absent.PNG,10,10,90,90,100,100,twitter
            """
        )
    )

    tasks, skipped = load_tasks(csv_path, images_url_prefix="/img", images_root=images_root)

    assert len(tasks) == 1
    assert skipped == 1
    assert tasks[0]["data"]["image"] == "/img/present.PNG"
