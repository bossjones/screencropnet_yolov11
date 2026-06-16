from __future__ import annotations

from pathlib import Path

from screencropnet_yolo.client.api_client import discover_images


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_discover_recursive_case_insensitive(tmp_path: Path) -> None:
    _touch(tmp_path / "a.png")
    _touch(tmp_path / "b.JPG")
    _touch(tmp_path / "notes.txt")
    _touch(tmp_path / "sub" / "d.jpeg")
    _touch(tmp_path / "sub" / "e.gif")

    found = discover_images(tmp_path, recursive=True)
    names = sorted(p.name for p in found)
    if names != ["a.png", "b.JPG", "d.jpeg", "e.gif"]:
        raise AssertionError(f"recursive discovery must match images case-insensitively: {names}")


def test_discover_flat_ignores_subdirs(tmp_path: Path) -> None:
    _touch(tmp_path / "a.png")
    _touch(tmp_path / "sub" / "d.jpeg")
    found = discover_images(tmp_path, recursive=False)
    names = [p.name for p in found]
    if names != ["a.png"]:
        raise AssertionError(f"flat discovery must not descend into subdirs: {names}")
