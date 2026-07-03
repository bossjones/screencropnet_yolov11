#!/usr/bin/env -S uv run python
"""Profile a pytest target with the PyTorch Profiler and write a report.

Sets ``SCREENCROPNET_TORCH_PROFILE_DIR`` so the ``tests/conftest.py`` fixture
wraps the ``@pytest.mark.pytorch_profile`` test in ``torch.profiler`` and dumps
its ``key_averages`` events + Chrome trace there, then renders a
``docs/perf/pytorch_profiler`` report. torch comes from the optional ``worker``
group (``uv run --group worker``). Run with ``uv run`` (project environment)::

    uv run .claude/skills/perf/scripts/run_torch_profiler.py tests/test_inference_e2e.py
"""

from __future__ import annotations

import argparse
import os
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
    parser.add_argument("-v", "--verbose", action="store_true", help="echo the resolved command")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        cmd = perf.torch_pytest_command(args.target)
        env = {**os.environ, "SCREENCROPNET_TORCH_PROFILE_DIR": tmp}
        if args.verbose:
            console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
        console.print(f"[bold]PyTorch Profiler[/bold] profiling [cyan]{args.target}[/cyan] …")
        subprocess.run(cmd, check=False, env=env)  # pytest may exit non-zero

        events = sorted(Path(tmp).glob("*.events.json"))
        if not events:
            console.print(
                "[red]No profile produced.[/red] Is torch installed "
                "(`uv sync --group worker`) and does the target carry "
                "`@pytest.mark.pytorch_profile`?"
            )
            return 1
        events_json = events[0]
        trace = events_json.with_name(events_json.name.replace(".events.json", ".trace.json"))
        dest = render_report.write_torch_report(
            args.target, cmd, events_json, trace if trace.exists() else None
        )

    console.print(f"[green]✔︎ wrote[/green] {dest.relative_to(render_report.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
