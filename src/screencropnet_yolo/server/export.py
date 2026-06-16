"""Export twitter-positive originals into the raw YOLO dataset.

Continues the dataset's ``NNNNN_<label>.EXT`` sequence from the largest parsed
index (the set has gaps, so the index is derived from ``max(parsed)``, never the
file count). Copies the *real original* file (never the compressed WebP),
preserving the original extension/case. Idempotent on ``original_path`` via a
sidecar manifest and collision-safe (probes the next free index, never
overwrites).
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from strif import atomic_output_file

from screencropnet_yolo.server.schemas import ExportRecord

_MANIFEST_NAME = ".export_manifest.json"


class _HasOriginalPath(Protocol):
    original_path: str


def _index_pattern(label: str) -> re.Pattern[str]:
    return re.compile(rf"^(\d+)_{re.escape(label)}\.", re.IGNORECASE)


def _used_indices(dataset_dir: Path, label: str) -> set[int]:
    pattern = _index_pattern(label)
    used: set[int] = set()
    for path in dataset_dir.glob(f"*_{label}.*"):
        match = pattern.match(path.name)
        if match:
            used.add(int(match.group(1)))
    return used


def current_max_index(dataset_dir: Path, label: str = "twitter", pad: int = 5) -> int:
    """Return the largest parsed ``NNNNN`` index in ``dataset_dir``, or -1 if none."""
    used = _used_indices(dataset_dir, label)
    return max(used) if used else -1


def next_index(dataset_dir: Path, label: str = "twitter", pad: int = 5) -> int:
    """Return the next index to allocate (``current_max_index + 1``)."""
    return current_max_index(dataset_dir, label=label, pad=pad) + 1


def _load_manifest(manifest_path: Path) -> dict[str, str]:
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text())
    return {}


def _save_manifest(manifest_path: Path, manifest: dict[str, str]) -> None:
    with atomic_output_file(manifest_path, make_parents=True) as tmp:
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def export_originals(
    jobs: Iterable[_HasOriginalPath],
    dataset_dir: Path,
    *,
    label: str = "twitter",
    pad: int = 5,
    dry_run: bool = False,
) -> list[ExportRecord]:
    """Copy each job's original into ``dataset_dir`` as ``NNNNN_<label>.EXT``."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_dir / _MANIFEST_NAME
    manifest = _load_manifest(manifest_path)
    used = _used_indices(dataset_dir, label)
    next_idx = (max(used) + 1) if used else 0

    records: list[ExportRecord] = []
    manifest_dirty = False
    for job in jobs:
        original = Path(job.original_path)
        key = str(original)

        existing = manifest.get(key)
        if existing is not None and (dataset_dir / existing).is_file():
            match = _index_pattern(label).match(existing)
            records.append(
                ExportRecord(
                    original_path=key,
                    dest_path=str(dataset_dir / existing),
                    index=int(match.group(1)) if match else -1,
                    copied=False,
                    reason="already_exported",
                )
            )
            continue

        while next_idx in used:
            next_idx += 1

        dest_name = f"{next_idx:0{pad}d}_{label}{original.suffix}"
        dest = dataset_dir / dest_name
        records.append(
            ExportRecord(
                original_path=key,
                dest_path=str(dest),
                index=next_idx,
                copied=not dry_run,
                reason="dry_run" if dry_run else "copied",
            )
        )
        if not dry_run:
            with atomic_output_file(dest, make_parents=True) as tmp:
                shutil.copyfile(original, tmp)
            manifest[key] = dest_name
            manifest_dirty = True
        used.add(next_idx)
        next_idx += 1

    if manifest_dirty:
        _save_manifest(manifest_path, manifest)
    return records
