from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from screencropnet_yolo.server.export import (
    current_max_index,
    export_originals,
    next_index,
)


@dataclass
class _Job:
    original_path: str


def _seed_dataset(dataset_dir: Path, indices: list[int], ext: str = ".PNG") -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for i in indices:
        (dataset_dir / f"{i:05d}_twitter{ext}").write_bytes(b"existing")


def _job(path: Path) -> _Job:
    return _Job(original_path=str(path))


def test_current_max_index_handles_gaps(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    _seed_dataset(dataset, [0, 2, 50, 1494])
    if current_max_index(dataset, label="twitter", pad=5) != 1494:
        raise AssertionError("max index must come from the largest parsed number, not file count")
    if next_index(dataset, label="twitter", pad=5) != 1495:
        raise AssertionError("next index must be max + 1")


def test_index_on_empty_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    dataset.mkdir()
    if current_max_index(dataset) != -1:
        raise AssertionError("empty dataset should report -1")
    if next_index(dataset) != 0:
        raise AssertionError("first export into empty dataset should be index 0")


def test_export_copies_real_original_preserving_extension_case(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    _seed_dataset(dataset, [0, 1494])
    src_dir = tmp_path / "orig"
    src_dir.mkdir()
    a = src_dir / "a.PNG"
    a.write_bytes(b"AAA-png-bytes")
    b = src_dir / "b.JPG"
    b.write_bytes(b"BBB-jpg-bytes")

    records = export_originals([_job(a), _job(b)], dataset, label="twitter", pad=5)

    dest_a = dataset / "01495_twitter.PNG"
    dest_b = dataset / "01496_twitter.JPG"
    if not dest_a.is_file() or not dest_b.is_file():
        raise AssertionError("exported files must continue the sequence with preserved extensions")
    if dest_a.read_bytes() != b"AAA-png-bytes":
        raise AssertionError("must copy the real original bytes")
    if dest_b.read_bytes() != b"BBB-jpg-bytes":
        raise AssertionError("must copy the real original bytes")
    if not all(r.copied for r in records):
        raise AssertionError("records should report copied=True on a real run")


def test_export_is_idempotent_on_reexport(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    _seed_dataset(dataset, [10])
    src = tmp_path / "orig" / "shot.PNG"
    src.parent.mkdir()
    src.write_bytes(b"data")

    export_originals([_job(src)], dataset, label="twitter", pad=5)
    count_after_first = len(list(dataset.glob("*_twitter.*")))
    records = export_originals([_job(src)], dataset, label="twitter", pad=5)
    count_after_second = len(list(dataset.glob("*_twitter.*")))

    if count_after_first != count_after_second:
        raise AssertionError("re-exporting the same original must not create new files")
    if any(r.copied for r in records):
        raise AssertionError("second export should report copied=False")
    if records[0].reason != "already_exported":
        raise AssertionError("idempotent skip should report reason 'already_exported'")


def test_export_is_collision_safe(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    # 1494 is max, but 1495 is already occupied -> allocator must skip to 1496.
    _seed_dataset(dataset, [1494, 1495])
    src = tmp_path / "orig" / "shot.PNG"
    src.parent.mkdir()
    src.write_bytes(b"payload")

    records = export_originals([_job(src)], dataset, label="twitter", pad=5)

    if (dataset / "01495_twitter.PNG").read_bytes() == b"payload":
        raise AssertionError("must never overwrite an occupied index")
    if records[0].index != 1496:
        raise AssertionError("allocator must probe to the next free index")


def test_export_dry_run_writes_nothing(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    _seed_dataset(dataset, [0])
    src = tmp_path / "orig" / "shot.PNG"
    src.parent.mkdir()
    src.write_bytes(b"data")

    records = export_originals([_job(src)], dataset, label="twitter", pad=5, dry_run=True)

    if list(dataset.glob("00001_twitter.*")):
        raise AssertionError("dry run must not write any files")
    if records[0].copied:
        raise AssertionError("dry run records should report copied=False")
    if records[0].index != 1:
        raise AssertionError("dry run should still report the index it would use")
