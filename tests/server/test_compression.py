from __future__ import annotations

from pathlib import Path

from PIL import Image
from pytest import raises

from screencropnet_yolo.server.compression import (
    UploadTooLarge,
    compress_lossless_webp,
    enforce_max_size,
)


def _make_png(path: Path, mode: str = "RGB", size: tuple[int, int] = (64, 48)) -> Path:
    color = (10, 20, 30, 40) if mode == "RGBA" else (10, 20, 30)
    Image.new(mode, size, color).save(path, format="PNG")
    return path


def test_compress_produces_valid_full_res_webp(tmp_path: Path) -> None:
    src = _make_png(tmp_path / "shot.png", size=(80, 60))
    dst_dir = tmp_path / "out"

    result = compress_lossless_webp(src, dst_dir)

    if result.suffix != ".webp":
        raise AssertionError("compressed file must be a .webp")
    if not result.is_file():
        raise AssertionError("compressed file was not written")
    with Image.open(result) as reopened:
        if reopened.size != (80, 60):
            raise AssertionError("compression must preserve full resolution")


def test_compress_preserves_rgba_alpha(tmp_path: Path) -> None:
    src = _make_png(tmp_path / "shot.png", mode="RGBA")
    result = compress_lossless_webp(src, tmp_path / "out")
    with Image.open(result) as reopened:
        if reopened.mode != "RGBA":
            raise AssertionError("WebP output must preserve the RGBA alpha channel")


def test_compress_leaves_original_bytes_unchanged(tmp_path: Path) -> None:
    src = _make_png(tmp_path / "shot.png")
    before = src.read_bytes()
    compress_lossless_webp(src, tmp_path / "out")
    if src.read_bytes() != before:
        raise AssertionError("the original file must never be modified")


def test_enforce_max_size_allows_small_file(tmp_path: Path) -> None:
    src = _make_png(tmp_path / "shot.png")
    enforce_max_size(src, max_bytes=10 * 1024 * 1024)


def test_enforce_max_size_raises_when_over_limit(tmp_path: Path) -> None:
    src = _make_png(tmp_path / "shot.png", size=(256, 256))
    with raises(UploadTooLarge):
        enforce_max_size(src, max_bytes=8)
