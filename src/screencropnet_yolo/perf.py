"""Torch-free, subprocess-free core for the ``perf`` profiling skill.

The skill runs the repo's real-model inference tests under two profilers —
Scalene (line-level CPU/GPU/memory) and the PyTorch Profiler (operator-level
traces) — and writes durable markdown reports under ``docs/perf/``. This module
holds the pure building blocks that are worth unit-testing in isolation:

- pytest argument-vector construction with the mandatory coverage/timeout strip,
- the two profiler command builders,
- report naming,
- parsing of each profiler's raw output into ranked rows,
- markdown rendering of those rows.

The IO edges (running the subprocesses, the ``torch.profiler`` fixture) live in
the skill scripts and ``tests/conftest.py`` and import from here. Keeping this
core torch-free means importing it never requires the optional ``worker`` group.

Why the strip flags are non-negotiable: ``pyproject.toml`` bakes
``--cov=src --cov-branch`` and ``--timeout=30 --timeout_method=thread`` into every
pytest run. Coverage's ``sys.settrace`` badly distorts both profilers and a
profiled test easily blows the 30s thread-timeout, so every command neutralizes
them with ``-p no:cov -o addopts= --timeout=0`` rather than editing the config.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, NamedTuple

# Flags that must appear on every profiled pytest invocation. ``-o addopts=``
# clears the injected coverage + default ``-m "not integration"`` selection, so
# each builder supplies its own ``-m`` (the last ``-m`` wins). Clearing addopts
# also drops the repo's ``--import-mode=importlib``, which it needs to avoid
# duplicate-basename collection errors, so we restore it explicitly.
_STRIP_FLAGS: tuple[str, ...] = (
    "-p", "no:cov",
    "-o", "addopts=",
    "--timeout=0",
    "--import-mode=importlib",
)  # fmt: skip


def build_pytest_argv(
    target: str,
    marker: str = "integration",
    extra: Sequence[str] | None = None,
) -> list[str]:
    """Build a ``pytest`` argument vector for a profiled run of ``target``.

    Always neutralizes coverage and the thread-timeout, and selects ``marker``
    (which replaces the injected ``-m "not integration"`` because the last
    ``-m`` expression wins).
    """
    argv = ["pytest", target, *_STRIP_FLAGS, "-m", marker]
    if extra:
        argv.extend(extra)
    _assert_stripped(argv)
    return argv


def scalene_command(
    program: str, target: str, outfile: str, marker: str = "integration"
) -> list[str]:
    """Build the Scalene command that profiles ``pytest`` and writes JSON.

    Scalene is installed ephemerally via ``uv run --with scalene`` (no lockfile
    churn). Scalene 2.3's ``run`` subcommand profiles a program *file* (it has no
    ``-m module`` form), so ``program`` is a tiny runner that calls
    ``pytest.main``; everything after ``---`` is that runner's args — the target
    plus the profiled selection. Output is JSON at ``outfile``.
    """
    cmd = [
        "uv", "run", "--with", "scalene",
        "python", "-m", "scalene", "run", "-o", outfile,
        program, "---", target, *_STRIP_FLAGS, "-m", marker,
    ]  # fmt: skip
    _assert_stripped(cmd)
    return cmd


def torch_pytest_command(target: str) -> list[str]:
    """Build the ``pytest`` command for the in-process PyTorch Profiler run.

    torch lives in the optional ``worker`` dependency group, so the run happens
    under ``uv run --group worker``. The ``pytorch_profile`` marker selects the
    tests whose ``conftest`` fixture wraps them in ``torch.profiler``.
    """
    cmd = ["uv", "run", "--group", "worker", *build_pytest_argv(target, marker="pytorch_profile")]
    return cmd


def _assert_stripped(argv: Sequence[str]) -> None:
    """Refuse any command that would let coverage/timeout corrupt the profile."""
    joined = " ".join(argv)
    missing = [flag for flag in ("no:cov", "addopts=", "--timeout=0") if flag not in joined]
    if missing:
        raise AssertionError(f"profiling command is missing strip flags: {missing}")


def label_from_target(target: str) -> str:
    """Derive a filename slug from a pytest target.

    A node id (``file.py::test_x``) slugs its test name; a plain file slugs its
    stem. ``tests/test_inference_e2e.py`` -> ``test-inference-e2e``.
    """
    tail = target.rsplit("::", 1)[-1] if "::" in target else target.rsplit("/", 1)[-1]
    if tail.endswith(".py"):
        tail = tail[: -len(".py")]
    return slugify(tail)


def slugify(text: str) -> str:
    """Lowercase, collapse any run of non-alphanumerics to a single hyphen."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def report_filename(date_str: str, label: str) -> str:
    """``<YYYY-MM-DD>-<label>.md`` — date-prefixed per the docs conventions."""
    return f"{date_str}-{label}.md"


class ScaleneRow(NamedTuple):
    file: str
    lineno: int
    code: str
    cpu_pct: float
    py_pct: float
    c_pct: float
    peak_mb: float
    gpu_pct: float


def scalene_rows(data: dict[str, Any], top: int = 15) -> list[ScaleneRow]:
    """Flatten Scalene's per-file JSON into the hottest lines by total CPU %.

    ``cpu_pct`` sums the Python and native (C) shares, which is how Scalene
    attributes where wall time actually goes.
    """
    rows: list[ScaleneRow] = []
    for path, file_data in data.get("files", {}).items():
        for line in file_data.get("lines", []):
            py = float(line.get("n_cpu_percent_python", 0.0))
            c = float(line.get("n_cpu_percent_c", 0.0))
            peak = float(line.get("n_peak_mb", 0.0))
            gpu = float(line.get("n_gpu_percent", 0.0))
            # Skip lines Scalene attributed no cost to — otherwise the report is
            # padded with comment/import lines that drown out the hotspots.
            if py + c == 0.0 and peak == 0.0 and gpu == 0.0:
                continue
            rows.append(
                ScaleneRow(
                    file=path,
                    lineno=int(line.get("lineno", 0)),
                    code=str(line.get("line", "")).rstrip("\n"),
                    cpu_pct=py + c,
                    py_pct=py,
                    c_pct=c,
                    peak_mb=peak,
                    gpu_pct=gpu,
                )
            )
    rows.sort(key=lambda r: r.cpu_pct, reverse=True)
    return rows[:top]


class TorchRow(NamedTuple):
    name: str
    calls: int
    self_cpu_us: float
    cpu_us: float
    self_cuda_us: float
    cpu_mem: int
    cuda_mem: int


def torch_rows(events: Sequence[dict[str, Any]], top: int = 15) -> list[TorchRow]:
    """Rank ``key_averages`` events by self device time, falling back to CPU.

    When any op has CUDA self-time the ranking uses it (that's the bottleneck on
    a GPU box); on CPU/MPS everything falls back to ``self_cpu_time_total``.
    """
    rows = [
        TorchRow(
            name=str(e.get("name", "")),
            calls=int(e.get("count", 0)),
            self_cpu_us=float(e.get("self_cpu_time_total", 0.0)),
            cpu_us=float(e.get("cpu_time_total", 0.0)),
            self_cuda_us=float(e.get("self_cuda_time_total", 0.0)),
            cpu_mem=int(e.get("cpu_memory_usage", 0)),
            cuda_mem=int(e.get("cuda_memory_usage", 0)),
        )
        for e in events
    ]
    if any(r.self_cuda_us > 0 for r in rows):
        rows.sort(key=lambda r: r.self_cuda_us, reverse=True)
    else:
        rows.sort(key=lambda r: r.self_cpu_us, reverse=True)
    return rows[:top]


def render_scalene_markdown(
    target: str,
    command: Sequence[str],
    rows: Sequence[ScaleneRow],
    date_str: str,
) -> str:
    """Render a Scalene run into a markdown report body."""
    header = [
        f"# Scalene profile — `{target}`",
        "",
        f"- Date: {date_str}",
        f"- Target: `{target}`",
        "- Backend: Scalene (line-level CPU / native / memory / GPU)",
        "",
        "## Command",
        "",
        "```bash",
        " ".join(command),
        "```",
        "",
        "## Hottest lines",
        "",
        "| File | Line | CPU % | Python % | Native % | Peak MB | GPU % | Code |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    body = [
        f"| `{r.file}` | {r.lineno} | {r.cpu_pct:.1f} | {r.py_pct:.1f} | {r.c_pct:.1f} "
        f"| {r.peak_mb:.1f} | {r.gpu_pct:.1f} | `{_clip(r.code)}` |"
        for r in rows
    ] or ["| _(no lines attributed)_ | | | | | | | |"]
    footer = [
        "",
        "## Optimization candidates",
        "",
        _candidates_note(),
    ]
    return "\n".join([*header, *body, *footer]) + "\n"


def render_torch_markdown(
    target: str,
    command: Sequence[str],
    rows: Sequence[TorchRow],
    date_str: str,
    trace_path: str,
    cuda_available: bool,
) -> str:
    """Render a PyTorch Profiler run into a markdown report body."""
    device = (
        "CPU + CUDA" if cuda_available else "CPU / MPS (CUDA unavailable — memory snapshot skipped)"
    )
    header = [
        f"# PyTorch Profiler — `{target}`",
        "",
        f"- Date: {date_str}",
        f"- Target: `{target}`",
        f"- Activities: {device}",
        f"- Chrome trace: `{trace_path}` (open in `chrome://tracing` or Perfetto)",
        "",
        "## Command",
        "",
        "```bash",
        " ".join(command),
        "```",
        "",
        "## Hottest operators",
        "",
        "| Operator | Calls | Self CPU (µs) | CPU total (µs) | Self CUDA (µs) | CPU mem (B) | CUDA mem (B) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    body = [
        f"| `{r.name}` | {r.calls} | {r.self_cpu_us:.1f} | {r.cpu_us:.1f} "
        f"| {r.self_cuda_us:.1f} | {r.cpu_mem} | {r.cuda_mem} |"
        for r in rows
    ] or ["| _(no operators recorded)_ | | | | | | |"]
    footer = [
        "",
        "## Optimization candidates",
        "",
        _candidates_note(),
    ]
    return "\n".join([*header, *body, *footer]) + "\n"


def _candidates_note() -> str:
    return (
        "Read the hottest rows above against `src/screencropnet_yolo/inference.py` "
        "(`InferencePipeline.predict_image/predict_batch/predict_video`) and "
        "`model.py`. Cross-check the numbers with `evaluation.py::benchmark_model`. "
        "List concrete, testable changes here before touching code."
    )


def _clip(code: str, width: int = 60) -> str:
    code = code.replace("|", "\\|").strip()
    return code if len(code) <= width else code[: width - 1] + "…"
