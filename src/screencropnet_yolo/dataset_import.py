"""Pascal-VOC CSV -> YOLO single-class import for the ``tweet_region`` dataset.

Bridges externally produced Pascal-VOC bounding-box annotations (one CSV row per
box) into the YOLO training layout, collapsing every source label into a single
``tweet_region`` class. Reuses :class:`DatasetSplitter` and
:func:`create_dataset_yaml` so the resulting tree matches the rest of the
pipeline.
"""

from __future__ import annotations

import csv
import logging
import shutil
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from screencropnet_yolo.dataset_utils import (
    DatasetSplitter,
    DatasetValidator,
    create_dataset_yaml,
)

logger = logging.getLogger(__name__)

TWEET_REGION_CLASS_ID = 0
TWEET_REGION_CLASS_NAME = "tweet_region"

# Every source label collapses to the single tweet_region class.
DEFAULT_CLASS_MAP: dict[str, int] = {
    "twitter": TWEET_REGION_CLASS_ID,
    TWEET_REGION_CLASS_NAME: TWEET_REGION_CLASS_ID,
}


def pascal_row_to_yolo(
    row: Mapping[str, Any], class_map: Mapping[str, int]
) -> tuple[int, float, float, float, float]:
    """Convert one Pascal-VOC annotation row to a YOLO ``(cls, xc, yc, w, h)`` tuple.

    Coordinates are normalized against the image ``width``/``height``. Unknown
    labels collapse to :data:`TWEET_REGION_CLASS_ID` so any source taxonomy maps
    onto the single-class target.
    """
    try:
        xmin = float(row["xmin"])
        ymin = float(row["ymin"])
        xmax = float(row["xmax"])
        ymax = float(row["ymax"])
        width = float(row["width"])
        height = float(row["height"])
    except KeyError as e:
        raise ValueError(f"Missing required column in CSV row: {e}") from e
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid numeric value in CSV row: {e}") from e

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions in bbox row: {width}x{height}")
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(
            f"Invalid bbox: expected xmin<xmax and ymin<ymax, got ({xmin}, {ymin}, {xmax}, {ymax})"
        )

    class_id = class_map.get(str(row["label"]), TWEET_REGION_CLASS_ID)
    x_center = (xmin + xmax) / 2 / width
    y_center = (ymin + ymax) / 2 / height
    box_w = (xmax - xmin) / width
    box_h = (ymax - ymin) / height

    return class_id, x_center, y_center, box_w, box_h


def convert_csv(
    csv_path: str | Path,
    output_dir: str | Path,
    class_map: Mapping[str, int] | None = None,
) -> Path:
    """Convert a Pascal-VOC CSV into one YOLO ``.txt`` per image under ``labels/``.

    Rows are grouped by their ``filename`` column; every box for an image becomes
    a line in that image's label file. Returns the ``labels/`` directory.
    """
    resolved_map = dict(class_map) if class_map is not None else dict(DEFAULT_CLASS_MAP)
    labels_dir = Path(output_dir) / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    lines_by_image: dict[str, list[str]] = defaultdict(list)
    with Path(csv_path).open(newline="") as f:
        for raw in csv.DictReader(f):
            row = dict(raw)
            if "label" not in row and "class" in row:
                row["label"] = row["class"]
            if "filename" not in row:
                raise ValueError("CSV row is missing the required 'filename' column.")
            class_id, x_c, y_c, w, h = pascal_row_to_yolo(row, resolved_map)
            stem = Path(str(row["filename"])).stem
            lines_by_image[stem].append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    for stem, lines in lines_by_image.items():
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")

    logger.info(f"Wrote {len(lines_by_image)} YOLO label files to: {labels_dir}")
    return labels_dir


def prepare_twitter_dataset(
    images_dir: str | Path,
    csv_path: str | Path,
    output_dir: str | Path,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
    class_map: Mapping[str, int] | None = None,
) -> Path:
    """Build a single-class YOLO dataset (train/val split + ``data.yaml``).

    Stages images alongside their converted labels, splits them with
    :class:`DatasetSplitter` (``test_ratio=0``), and emits a ``data.yaml`` pinned
    to ``nc: 1, names: [tweet_region]``. Returns the path to ``data.yaml``.
    """
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    staging = output_dir / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    for image_path in _iter_images(images_dir):
        shutil.copy2(image_path, staging / image_path.name)

    convert_csv(csv_path, staging, class_map=class_map)

    splitter = DatasetSplitter(
        source_path=str(staging),
        output_path=str(output_dir),
        train_ratio=1.0 - val_ratio,
        val_ratio=val_ratio,
        test_ratio=0.0,
        seed=seed,
    )
    splitter.split()

    yaml_path = create_dataset_yaml(output_dir, [TWEET_REGION_CLASS_NAME], output_dir / "data.yaml")

    shutil.rmtree(staging)
    return Path(yaml_path)


def _iter_images(directory: Path) -> list[Path]:
    """Collect supported image files directly under ``directory``."""
    images: list[Path] = []
    for ext in DatasetValidator.SUPPORTED_IMAGE_FORMATS:
        images.extend(directory.glob(f"*{ext}"))
        images.extend(directory.glob(f"*{ext.upper()}"))
    return images
