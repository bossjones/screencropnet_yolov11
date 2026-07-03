#!/usr/bin/env -S uv run python
"""Render profiler output into a ``docs/perf/`` markdown report.

Shared by ``run_scalene.py`` and ``run_torch_profiler.py``; also runnable
standalone to re-render a report from a raw artifact. All pure logic (parsing,
rendering, naming) lives in :mod:`screencropnet_yolo.perf`; this module only
adds the date stamp and the filesystem writes. Run with ``uv run`` (project
environment) so ``screencropnet_yolo`` is importable — not as an isolated PEP 723
script, which would not have the package.
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
from pathlib import Path

from screencropnet_yolo import perf

REPO_ROOT = Path(__file__).resolve().parents[4]
PERF_DOCS = REPO_ROOT / "docs" / "perf"


def today_str() -> str:
    return datetime.date.today().isoformat()


def write_scalene_report(
    target: str,
    command: list[str],
    scalene_json: Path,
    out_root: Path = PERF_DOCS,
) -> Path:
    data = json.loads(scalene_json.read_text())
    rows = perf.scalene_rows(data)
    date = today_str()
    dest = out_root / "scalene" / perf.report_filename(date, perf.label_from_target(target))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(perf.render_scalene_markdown(target, command, rows, date))
    return dest


def write_torch_report(
    target: str,
    command: list[str],
    events_json: Path,
    trace_src: Path | None,
    out_root: Path = PERF_DOCS,
) -> Path:
    payload = json.loads(events_json.read_text())
    rows = perf.torch_rows(payload.get("events", []))
    cuda = bool(payload.get("cuda", False))
    date = today_str()
    label = perf.label_from_target(target)
    dest_dir = out_root / "pytorch_profiler"
    dest_dir.mkdir(parents=True, exist_ok=True)

    trace_rel = "(trace not exported)"
    if trace_src is not None and trace_src.exists():
        trace_dest = dest_dir / f"{date}-{label}.trace.json"
        shutil.copyfile(trace_src, trace_dest)
        trace_rel = str(trace_dest.relative_to(REPO_ROOT))

    dest = dest_dir / perf.report_filename(date, label)
    dest.write_text(perf.render_torch_markdown(target, command, rows, date, trace_rel, cuda))
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a docs/perf report from a raw artifact.")
    parser.add_argument("backend", choices=["scalene", "torch"])
    parser.add_argument("raw", type=Path, help="scalene JSON or torch events JSON")
    parser.add_argument("--target", required=True)
    parser.add_argument("--trace", type=Path, default=None, help="torch Chrome trace to copy")
    args = parser.parse_args()

    if args.backend == "scalene":
        dest = write_scalene_report(args.target, ["(re-rendered)"], args.raw)
    else:
        dest = write_torch_report(args.target, ["(re-rendered)"], args.raw, args.trace)
    print(dest)


if __name__ == "__main__":
    main()
