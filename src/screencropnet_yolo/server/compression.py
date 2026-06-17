"""Lossless WebP compression for fast upload while preserving true originals.

The client compresses to a full-resolution lossless WebP under a temp dir and
uploads that; the export step later copies the *real original* file, never the
compressed WebP.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


class UploadTooLarge(Exception):
    """Raised when a file exceeds the configured maximum upload size."""


def compress_lossless_webp(src: Path, dst_dir: Path) -> Path:
    """Write a full-resolution lossless WebP copy of ``src`` into ``dst_dir``.

    The source file's bytes are never modified. RGBA images keep their alpha
    channel (WebP supports alpha). Returns the path to the new ``.webp`` file.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{src.stem}.webp"
    with Image.open(src) as image:
        image.save(dst, format="WEBP", lossless=True, quality=100, method=6)
    return dst


def enforce_max_size(path: Path, max_bytes: int) -> None:
    """Raise :class:`UploadTooLarge` if ``path`` is larger than ``max_bytes``."""
    size = path.stat().st_size
    if size > max_bytes:
        raise UploadTooLarge(f"{path} is {size} bytes, exceeds limit of {max_bytes}")
