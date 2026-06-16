#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pyyaml>=6.0",
# ]
# ///
"""Turn a Label Studio YOLO export ZIP into a YOLO26-ready dataset tree.

Label Studio's "YOLO with images" export contains ``images/`` and ``labels/``
sibling dirs (plus ``classes.txt``/``notes.json``). This script pairs each
label ``.txt`` with its image, makes a deterministic train/val(/test) split,
copies the pairs into the layout the YOLO26 trainer expects, and writes
``data.yaml`` with a single ``tweet_region`` class.

The split dirs (``train``/``val``/``test``) under ``--out`` are cleared before
copying, so re-exporting into an existing dataset folder replaces it cleanly
rather than merging stale files across runs.

Output layout::

    <out>/
      data.yaml          # nc: 1, names: [tweet_region]
      train/images/  train/labels/
      val/images/    val/labels/
      test/images/   test/labels/   # only when --test-ratio > 0

Run: ``uv run scripts/ls_yolo_export_to_dataset.py --export ls_export.zip \
--out <dir> --val-ratio 0.2 --test-ratio 0.1``
"""

from __future__ import annotations

import argparse
import random
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import yaml

CLASS_NAME = "tweet_region"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _find_dir(root: Path, name: str) -> Path:
    """Locate a ``name`` dir anywhere under ``root`` (LS nests exports sometimes)."""
    if (root / name).is_dir():
        return root / name
    for candidate in root.rglob(name):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"no '{name}/' dir found inside the export")


def _pair_images_with_labels(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    """Pair every label ``.txt`` with its same-stem image. Unmatched are dropped."""
    images_by_stem = {
        p.stem: p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    }
    pairs: list[tuple[Path, Path]] = []
    for label in sorted(labels_dir.glob("*.txt")):
        image = images_by_stem.get(label.stem)
        if image is not None:
            pairs.append((image, label))
    return pairs


def _split(
    pairs: list[tuple[Path, Path]], val_ratio: float, test_ratio: float, seed: int
) -> dict[str, list[tuple[Path, Path]]]:
    """Deterministically partition pairs into train/val/test.

    Test and val are sliced off the front of the shuffled order; whatever
    remains is train. ``test`` is empty when ``test_ratio`` is 0.
    """
    ordered = sorted(pairs, key=lambda pl: pl[1].stem)
    random.Random(seed).shuffle(ordered)
    n_val = round(len(ordered) * val_ratio)
    n_test = round(len(ordered) * test_ratio)
    return {
        "test": ordered[:n_test],
        "val": ordered[n_test : n_test + n_val],
        "train": ordered[n_test + n_val :],
    }


def _copy_split(pairs: list[tuple[Path, Path]], out_dir: Path, split: str) -> None:
    img_out = out_dir / split / "images"
    lbl_out = out_dir / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for image, label in pairs:
        shutil.copy2(image, img_out / image.name)
        shutil.copy2(label, lbl_out / label.name)


def write_data_yaml(out_dir: Path, has_test: bool = False) -> Path:
    """Write the single-class YOLO ``data.yaml`` and return its path.

    ``path`` is intentionally omitted so YOLO resolves ``train``/``val`` relative
    to the ``data.yaml`` location, keeping the exported dataset portable across
    machines and containers. ``test`` is included only when a test split exists.
    """
    data: dict[str, Any] = {
        "train": "train/images",
        "val": "val/images",
    }
    if has_test:
        data["test"] = "test/images"
    data["nc"] = 1
    data["names"] = [CLASS_NAME]
    out_path = out_dir / "data.yaml"
    out_path.write_text(yaml.safe_dump(data, sort_keys=False))
    return out_path


def build_dataset(
    export_zip: Path,
    out_dir: Path,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Unpack the export, split, copy pairs, and write ``data.yaml``.

    Returns a summary dict with train/val/test counts and the ``data.yaml`` path.
    The ``train``/``val``/``test`` dirs under ``out_dir`` are cleared first, so
    re-running into an existing dataset folder replaces it cleanly.

    Raises:
        ValueError: if ``val_ratio``/``test_ratio`` are negative, their sum
            exceeds 1.0, or the export contains no image/label pairs.
    """
    if val_ratio < 0.0 or test_ratio < 0.0 or val_ratio + test_ratio > 1.0:
        raise ValueError(
            "val_ratio and test_ratio must be >= 0 and sum to <= 1.0, "
            f"got val_ratio={val_ratio}, test_ratio={test_ratio}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        shutil.rmtree(out_dir / split, ignore_errors=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(export_zip) as zf:
            zf.extractall(tmp_path)

        images_dir = _find_dir(tmp_path, "images")
        labels_dir = _find_dir(tmp_path, "labels")
        pairs = _pair_images_with_labels(images_dir, labels_dir)
        if not pairs:
            raise ValueError("export contained no image/label pairs")

        splits = _split(pairs, val_ratio, test_ratio, seed)
        for split, split_pairs in splits.items():
            if split_pairs:
                _copy_split(split_pairs, out_dir, split)

    has_test = bool(splits["test"])
    data_yaml = write_data_yaml(out_dir, has_test=has_test)
    return {
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        "total": len(pairs),
        "data_yaml": str(data_yaml),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True, type=Path, help="Label Studio YOLO export ZIP")
    parser.add_argument("--out", required=True, type=Path, help="output dataset dir")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="validation fraction")
    parser.add_argument(
        "--test-ratio", type=float, default=0.0, help="test fraction (0 disables the test split)"
    )
    parser.add_argument("--seed", type=int, default=42, help="shuffle seed")
    args = parser.parse_args()

    summary = build_dataset(
        args.export, args.out, args.val_ratio, args.test_ratio, args.seed
    )
    print(
        f"✔︎ {summary['total']} pairs → train={summary['train']} val={summary['val']} "
        f"test={summary['test']}; wrote {summary['data_yaml']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
