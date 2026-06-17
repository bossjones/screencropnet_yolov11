#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.27",
#     "strif>=3",
# ]
# ///
"""Download the ScreenNetV1 EfficientNet-B0 checkpoint to a local, overridable path.

Mirrors ``Settings`` defaults (``scratch/models/ScreenNetV1.pth`` and the Dropbox
``dl=1`` direct link) without importing the package, so it runs under ``uv run``'s
isolated PEP 723 environment. ``SCREENCROPNET_WEIGHTS_PATH`` overrides the destination
(``~`` is expanded), matching the env override honoured by ``Settings.weights_path``.

The download follows redirects (Dropbox 302s to ``dl.dropboxusercontent.com``), is
idempotent (skips an existing checkpoint unless ``--force``), writes atomically, and
refuses HTML payloads so a Dropbox error page never masquerades as weights.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import httpx
from strif import atomic_output_file

# Kept in sync with screencropnet_yolo.server.config.Settings.
DEFAULT_URL = "https://www.dropbox.com/scl/fi/8a5cc7e1ngcnm78kcqnga/ScreenNetV1.pth?rlkey=sbxats642fui9gpuwj8susha0&dl=1"
DEFAULT_DEST = Path("scratch/models/ScreenNetV1.pth")
# Floor to distinguish a real checkpoint from a Dropbox HTML error page; the EfficientNet-B0
# weights are comfortably larger. This is a sanity gate, not exact-size validation (use
# --sha256 for that).
MIN_VALID_BYTES = 1 * 1024 * 1024


def expanded_path(value: str) -> Path:
    """Resolve ``~`` and ``$VAR`` references in a user-supplied path argument."""
    return Path(os.path.expandvars(value)).expanduser()


def _default_dest() -> Path:
    env = os.environ.get("SCREENCROPNET_WEIGHTS_PATH")
    return expanded_path(env) if env else DEFAULT_DEST


def _looks_like_html(first_chunk: bytes, content_type: str) -> bool:
    if "text/html" in content_type.lower():
        return True
    head = first_chunk.lstrip()[:512].lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html")


def _download(url: str, dest: Path) -> int:
    written = 0
    # atomic_output_file(make_parents=True) creates dest's parent dirs for us.
    with atomic_output_file(dest, make_parents=True) as tmp:
        with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            with tmp.open("wb") as fh:
                first = True
                for chunk in response.iter_bytes():
                    if first:
                        if _looks_like_html(chunk, content_type):
                            raise ValueError(
                                f"refusing to save HTML payload from {url} "
                                "(content-type or body looks like an error page, not a .pth)"
                            )
                        first = False
                    fh.write(chunk)
                    written += len(chunk)
        if written < MIN_VALID_BYTES:
            raise ValueError(
                f"downloaded only {written} bytes from {url}; expected a .pth "
                f"of at least {MIN_VALID_BYTES} bytes (likely an error page)"
            )
    return written


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=expanded_path,
        default=_default_dest(),
        help="destination path for the checkpoint (default honours SCREENCROPNET_WEIGHTS_PATH)",
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="download URL (Dropbox dl=1 direct link)")
    parser.add_argument(
        "--force", action="store_true", help="re-download even if a valid checkpoint already exists"
    )
    parser.add_argument("--sha256", default=None, help="optional expected sha256 to verify the file")
    args = parser.parse_args()

    dest: Path = args.dest

    if dest.exists() and dest.stat().st_size >= MIN_VALID_BYTES and not args.force:
        print(f"✔︎ weights already present: {dest} ({dest.stat().st_size} bytes); use --force to re-download")
        return 0

    print(f"∆ downloading {args.url}\n  → {dest}")
    try:
        size = _download(args.url, dest)
    except (httpx.HTTPError, ValueError) as exc:
        print(f"✘ download failed: {exc}", file=sys.stderr)
        return 1

    if args.sha256:
        actual = _sha256(dest)
        if actual != args.sha256:
            dest.unlink(missing_ok=True)
            print(f"✘ sha256 mismatch: expected {args.sha256}, got {actual}", file=sys.stderr)
            return 1
        print(f"✔︎ sha256 verified: {actual}")

    print(f"✔︎ saved {dest} ({size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
