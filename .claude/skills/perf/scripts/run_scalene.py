#!/usr/bin/env -S uv run python
"""Profile a pytest target with Scalene and write a ``docs/perf/scalene`` report.

Scalene installs ephemerally (``uv run --with scalene``), so this adds nothing to
``uv.lock``. The profiled pytest run strips coverage + the thread-timeout via the
command built in :func:`screencropnet_yolo.perf.scalene_command`. Run this with
``uv run`` (project environment), e.g.::

    uv run .claude/skills/perf/scripts/run_scalene.py tests/test_inference_e2e.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import render_report
from rich.console import Console

from screencropnet_yolo import perf

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="pytest target: file, file::test, or node id")
    parser.add_argument("--marker", default="integration", help="pytest marker to select")
    parser.add_argument("-v", "--verbose", action="store_true", help="echo the resolved command")
    args = parser.parse_args()

    runner = str(Path(__file__).with_name("_pytest_runner.py"))
    with tempfile.TemporaryDirectory() as tmp:
        out_json = Path(tmp) / "scalene.json"
        cmd = perf.scalene_command(runner, args.target, str(out_json), marker=args.marker)
        if args.verbose:
            console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
        console.print(f"[bold]Scalene[/bold] profiling [cyan]{args.target}[/cyan] …")
        subprocess.run(cmd, check=False)  # pytest may exit non-zero; Scalene still writes JSON

        if not out_json.exists():
            console.print("[red]Scalene produced no JSON — did the run start?[/red]")
            return 1
        dest = render_report.write_scalene_report(args.target, cmd, out_json)

    console.print(f"[green]✔︎ wrote[/green] {dest.relative_to(render_report.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
