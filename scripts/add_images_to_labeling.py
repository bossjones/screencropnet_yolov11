#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Copy a folder of images into the Label Studio staging directory, renaming to
the project naming convention (``NNNNN_<suffix>.<EXT>``).

Scans ``--staging-dir`` for existing files to auto-detect the next available
index, so it is safe to run multiple times — each run starts after the highest
index already present.

If a destination file already exists, the script prompts for confirmation before
overwriting unless ``--force`` is given.

Run: ``uv run scripts/add_images_to_labeling.py --source-dir /path/to/images``
Or:  ``make labeling-add-images IMAGE_DIR=/path/to/images``

See docs/adding-images-to-label-studio.md for full workflow instructions.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
)

DEFAULT_STAGING_DIR = Path("scratch/datasets/twitter_screenshots_raw/train_images")
DEFAULT_SUFFIX = "twitter"
DEFAULT_EXT = "PNG"
INDEX_WIDTH = 5


def detect_next_index(staging_dir: Path, suffix: str, ext: str) -> int:
    """Scan staging_dir for files matching the naming convention and return max index + 1.

    Returns 0 when the directory is empty or no file matches the pattern.
    The scan is case-insensitive on the extension to handle mixed-case existing
    files (e.g. uppercase .PNG alongside lowercase .png).
    """
    pattern = re.compile(
        r"^(\d{" + str(INDEX_WIDTH) + r",})_" + re.escape(suffix) + r"\." + re.escape(ext) + r"$",
        re.IGNORECASE,
    )
    max_index = -1
    if staging_dir.is_dir():
        for entry in staging_dir.iterdir():
            m = pattern.match(entry.name)
            if m:
                idx = int(m.group(1))
                if idx > max_index:
                    max_index = idx
    return max_index + 1


def collect_images(source_dir: Path) -> list[Path]:
    """Return sorted list of image files in source_dir (non-recursive).

    Sorting by filename makes the index assignment deterministic and reproducible.
    """
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source directory not found: {source_dir}")
    return [
        p
        for p in sorted(source_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]


def _confirm_overwrite(dest_name: str) -> bool:
    """Prompt user to confirm overwriting an existing destination file.

    Defaults to skip (N) on empty input or non-interactive (piped) context.
    """
    try:
        answer = input(f"  overwrite {dest_name}? [y/N] ").strip().lower()
        return answer in ("y", "yes")
    except EOFError:
        return False


def add_images(
    source_dir: Path,
    staging_dir: Path,
    suffix: str,
    start_index: int | None,
    ext: str,
    force: bool,
    dry_run: bool,
    verbose: bool,
) -> dict[str, object]:
    """Rename and copy images from source_dir into staging_dir.

    Returns a summary dict: first_index, last_index, count, skipped.
    """
    images = collect_images(source_dir)
    if not images:
        return {"first_index": None, "last_index": None, "count": 0, "skipped": 0}

    next_index = start_index if start_index is not None else detect_next_index(staging_dir, suffix, ext)

    if not dry_run:
        staging_dir.mkdir(parents=True, exist_ok=True)

    first_index = next_index
    count = 0
    skipped = 0

    for src in images:
        dest_name = f"{next_index:0{INDEX_WIDTH}d}_{suffix}.{ext}"
        dest = staging_dir / dest_name

        if dest.exists():
            if force:
                if verbose or dry_run:
                    label = "[dry-run] " if dry_run else ""
                    print(f"  {label}overwrite {src.name} → {dest_name}")
            elif dry_run:
                print(f"  [dry-run] would prompt: overwrite {dest_name}? (use --force to skip prompt)")
            elif not _confirm_overwrite(dest_name):
                print(f"  skipped {dest_name}", file=sys.stderr)
                skipped += 1
                next_index += 1
                continue
        elif verbose or dry_run:
            label = "[dry-run] " if dry_run else ""
            print(f"  {label}{src.name} → {dest_name}")

        if not dry_run:
            shutil.copy2(src, dest)
        count += 1
        next_index += 1

    return {
        "first_index": first_index,
        "last_index": next_index - 1,
        "count": count,
        "skipped": skipped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="folder of images to add (jpg/jpeg/png/webp/bmp/tif, non-recursive)",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=DEFAULT_STAGING_DIR,
        metavar="DIR",
        help=f"destination staging directory (default: {DEFAULT_STAGING_DIR})",
    )
    parser.add_argument(
        "--suffix",
        default=DEFAULT_SUFFIX,
        metavar="WORD",
        help=f"middle part of the output filename NNNNN_<suffix>.EXT (default: {DEFAULT_SUFFIX!r})",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        metavar="N",
        help="override the starting counter (default: auto-detect from highest existing index + 1)",
    )
    parser.add_argument(
        "--ext",
        default=DEFAULT_EXT,
        metavar="EXT",
        help=f"output file extension without dot, uppercase for convention consistency (default: {DEFAULT_EXT!r})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing destination files without prompting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would happen without writing any files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="log each file copy (implied by --dry-run)",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("[dry-run] no files will be written")

    try:
        summary = add_images(
            source_dir=args.source_dir,
            staging_dir=args.staging_dir,
            suffix=args.suffix,
            start_index=args.start_index,
            ext=args.ext,
            force=args.force,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    except FileNotFoundError as exc:
        print(f"✘ {exc}", file=sys.stderr)
        return 1

    if summary["count"] == 0 and summary["skipped"] == 0:
        print(f"✘ no supported image files found in {args.source_dir}", file=sys.stderr)
        return 1

    verb = "would copy" if args.dry_run else "copied"
    first = f"{summary['first_index']:0{INDEX_WIDTH}d}"
    last = f"{summary['last_index']:0{INDEX_WIDTH}d}"
    skipped_note = f"; skipped {summary['skipped']} existing" if summary["skipped"] else ""
    print(
        f"✔︎ {verb} {summary['count']} image(s) to {args.staging_dir} "
        f"(indices {first}–{last}_{args.suffix}.{args.ext}{skipped_note})"
    )

    if not args.dry_run and summary["count"] > 0:
        csv_path = args.staging_dir.parent / "labels_pascal_temp.csv"
        print()
        print("Next steps:")
        print("  Case A (no labels): go to Label Studio → Settings → Cloud Storage → Sync")
        print("  Case B (with CSV labels):")
        print(f"    1. Append rows to {csv_path}")
        print("    2. make labeling-tasks")
        print("    3. make labeling-setup-project  (or upload tasks.json via UI → Import)")
        print()
        print("See docs/adding-images-to-label-studio.md for full instructions.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
