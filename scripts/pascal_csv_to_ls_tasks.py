#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Convert a Pascal-VOC CSV into a Label Studio ``tasks.json``.

Each existing pixel-space box becomes a pre-annotation so the 341 already-labelled
screenshots open in Label Studio with a ``tweet_region`` rectangle pre-drawn for
verification. The CSV already carries each image's ``width``/``height``, so no
image decoding is needed here — only stdlib.

CSV columns: ``img_path,xmin,ymin,xmax,ymax,width,height,label``.

Run: ``uv run scripts/pascal_csv_to_ls_tasks.py --csv ... --images-url-prefix ... --out ...``
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

LABEL = "tweet_region"
MODEL_VERSION = "pascal_csv_seed"
FROM_NAME = "label"
TO_NAME = "image"


def expanded_path(value: str) -> Path:
    """Resolve ``~`` and ``$VAR`` references in a user-supplied path argument."""
    return Path(os.path.expandvars(value)).expanduser()


def bbox_to_ls_value(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    width: float,
    height: float,
) -> dict[str, Any]:
    """Convert pixel-space Pascal-VOC coords to Label Studio's percent schema.

    Label Studio rectangle ``value`` uses ``x``/``y``/``width``/``height`` as
    percentages of the image, with the origin at the top-left corner.

    Coordinates are clamped to the image boundaries before conversion so minor
    rounding/annotation overruns survive instead of dropping the row (matching
    the ML backend's clamping). Genuinely degenerate boxes still raise.

    Raises:
        ValueError: if the image dims are non-positive or the box collapses to
            zero area after clamping.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"non-positive image size: {width}x{height}")

    x1 = max(0.0, min(width, xmin))
    y1 = max(0.0, min(height, ymin))
    x2 = max(0.0, min(width, xmax))
    y2 = max(0.0, min(height, ymax))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"degenerate box: ({xmin},{ymin},{xmax},{ymax})")

    return {
        "x": x1 / width * 100.0,
        "y": y1 / height * 100.0,
        "width": (x2 - x1) / width * 100.0,
        "height": (y2 - y1) / height * 100.0,
        "rotation": 0,
        "rectanglelabels": [LABEL],
    }


def row_to_task(row: Mapping[str, str], images_url_prefix: str) -> dict[str, Any]:
    """Build one Label Studio task (with a seed pre-annotation) from a CSV row.

    Raises:
        ValueError: propagated from :func:`bbox_to_ls_value` for malformed boxes.
    """
    filename = Path(row["img_path"]).name
    value = bbox_to_ls_value(
        xmin=float(row["xmin"]),
        ymin=float(row["ymin"]),
        xmax=float(row["xmax"]),
        ymax=float(row["ymax"]),
        width=float(row["width"]),
        height=float(row["height"]),
    )
    return {
        "data": {"image": f"{images_url_prefix}/{filename}"},
        "predictions": [
            {
                "model_version": MODEL_VERSION,
                "score": 1.0,
                "result": [
                    {
                        "from_name": FROM_NAME,
                        "to_name": TO_NAME,
                        "type": "rectanglelabels",
                        "original_width": int(float(row["width"])),
                        "original_height": int(float(row["height"])),
                        "image_rotation": 0,
                        "value": value,
                    }
                ],
            }
        ],
    }


def load_tasks(
    csv_path: Path,
    images_url_prefix: str,
    images_root: Path | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Read the CSV and return ``(tasks, skipped_count)``.

    Rows with malformed boxes are skipped. If ``images_root`` is given, rows
    whose image file is absent on disk are also skipped.

    Raises:
        ValueError: if the CSV is missing a required column (a structural error
            that would otherwise silently skip every row).
    """
    tasks: list[dict[str, Any]] = []
    skipped = 0
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                if images_root is not None:
                    filename = Path(row["img_path"]).name
                    if not (images_root / filename).is_file():
                        skipped += 1
                        continue
                tasks.append(row_to_task(row, images_url_prefix))
            except KeyError as e:
                raise ValueError(f"CSV is missing required column: {e}") from e
            except ValueError:
                skipped += 1
    return tasks, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=expanded_path, help="Pascal-VOC CSV path")
    parser.add_argument(
        "--images-root",
        type=expanded_path,
        default=None,
        help="optional dir of images; rows with a missing file are skipped",
    )
    parser.add_argument(
        "--images-url-prefix",
        required=True,
        help="URL prefix Label Studio uses to serve images, e.g. /data/local-files/?d=train_images",
    )
    parser.add_argument("--out", required=True, type=expanded_path, help="output tasks.json path")
    args = parser.parse_args()

    tasks, skipped = load_tasks(args.csv, args.images_url_prefix, args.images_root)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(tasks, indent=2))

    print(f"✔︎ wrote {len(tasks)} tasks to {args.out} (skipped {skipped})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
