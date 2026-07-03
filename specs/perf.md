# Spec: `perf` skill — Scalene + PyTorch Profiler harness for the inference hot path

> Implementation is **TDD** — drive every unit with `superpowers:test-driven-development` (RED → GREEN → refactor). Write the failing test first, then the minimal implementation. Pure helpers (arg-vector builders, JSON→markdown row mappers) land with inline `## Tests`; IO/subprocess/fixture wiring lands in `tests/`.

## Purpose

The repo already samples call stacks with `pyinstrument` (`make profile-*`, the `SCREENCROPNET_PROFILE` middleware in `server/api.py`, `benchmark_model()` in `evaluation.py`). What it lacks is **line-level CPU/GPU/memory attribution** (which source line burns the time and allocates the bytes) and **PyTorch operator-level traces** (which aten op dominates a `model.predict()` call). Those are exactly the two questions you need answered before optimizing the YOLO inference path.

The `perf` skill (`.claude/skills/perf/`) is a repeatable, skill-driven workflow that runs the repo's real-model inference tests under **Scalene** and the **PyTorch Profiler**, and lands durable markdown reports under `docs/perf/`. A later session reads those reports and turns the hotspots into concrete changes in `inference.py` / `model.py`. The skill is scaffolded with `skill-creator`; the source material is distilled into `references/` from three upstream docs:

- Scalene — <https://github.com/plasma-umass/scalene>
- PyTorch Profiler — <https://pytorch.org/blog/introducing-pytorch-profiler-the-new-and-improved-performance-tool/>
- Understanding GPU Memory (part 1) — <https://pytorch.org/blog/understanding-gpu-memory-1/>

## Usage

```bash
# Scalene backend (process-level, no code changes, ephemeral install):
uv run .claude/skills/perf/scripts/run_scalene.py tests/test_inference_e2e.py

# PyTorch Profiler backend (in-process, needs the marker + worker torch):
uv run .claude/skills/perf/scripts/run_torch_profiler.py tests/test_inference_e2e.py

# Makefile shortcuts (mirror the existing profile-* targets):
make profile-scalene TARGET=tests/test_inference_e2e.py
make profile-torch   TARGET=tests/test_inference_e2e.py
```

Each run writes one markdown report:

```text
docs/perf/scalene/2026-07-02-test-inference-e2e.md
docs/perf/pytorch_profiler/2026-07-02-test-inference-e2e.md
```

The skill triggers on: "profile", "scalene", "pytorch profiler", "why is X slow", "performance", "gpu memory", "cpu hotspot", "memory hotspot".

### Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `target` (positional) | — | A pytest target: file, `file::test`, or a node-id. The test(s) to profile. |
| `--out-dir PATH` | `docs/perf/<backend>/` | Where the markdown report is written. |
| `--label TEXT` | derived from target | Slug used in the report filename (`<date>-<label>.md`). |
| `--keep-raw` | off | Keep the raw profiler artifacts (scalene JSON, Chrome trace) next to the report. |
| `--open` | off | Open the report (and any HTML/trace) after writing. |
| `-v` | off | Echo the fully-resolved profiler command before running. |

## Coverage & timeout stripping (non-negotiable)

`pyproject.toml` `addopts` bakes `--cov=src --cov-branch` and `--timeout=30 --timeout_method=thread` into **every** pytest invocation. Coverage's `sys.settrace` and the thread-timeout both corrupt profiler measurements (and a profiled run easily exceeds 30s). Every profiling invocation therefore strips them per-command — **never by editing `pyproject.toml`** (CLAUDE.md forbids touching the injected cov flags):

```bash
-p no:cov -o addopts="" --timeout=0 -m pytorch_profile   # torch backend selects the marked tests
```

Because `addopts=""` also drops the default `-m "not integration"`, the runner supplies its own `-m` (last `-m` wins). The scripts hard-guard: if the resolved command lacks the strip flags, they refuse to run.

## Scalene backend

Process-level, zero production-code changes. `run_scalene.py` builds and runs:

```bash
uv run --with scalene python -m scalene --cli --json --outfile <tmp>.json \
  --- -m pytest <target> -p no:cov -o addopts="" --timeout=0 -m integration
```

Scalene installs **ephemerally** via `uv run --with scalene` (no lockfile churn). It reports CPU (Python vs native/C), peak per-line memory, memory growth, and GPU utilization line-by-line. `render_report.py` parses the JSON and emits, per hot file: the top-N lines by CPU, the native-vs-Python split, per-line peak memory, and GPU % where present — plus a "suspected hotspots" list pointing at `inference.py` line ranges.

## PyTorch Profiler backend

In-process, operator-level. Requires torch, which lives in the optional `worker` dependency group (`uv sync --group worker`).

- A new **`pytorch_profile`** pytest marker tags the real-inference tests.
- A `tests/conftest.py` fixture (`pytorch_profiler`) wraps the marked test body in
  `torch.profiler.profile(activities=[CPU(, CUDA)], record_shapes=True, profile_memory=True, with_stack=True, on_trace_ready=…)`.
- On teardown it exports a Chrome trace (`export_chrome_trace`) and the `key_averages()` table (sorted by `self_cpu_time_total`, or `self_cuda_time_total` when CUDA is present), to a temp dir the renderer consumes.
- `render_report.py` renders `docs/perf/pytorch_profiler/<date>-<test>.md`: the top ops by self-time, input shapes, allocated/peak memory, and the trace path (openable in `chrome://tracing` / Perfetto).

### CPU/MPS vs CUDA

This is a Mac (no CUDA). Local runs profile on **CPU/MPS** with `profile_memory=True`. The *Understanding GPU Memory* snapshot workflow (`torch.cuda.memory._record_memory_history()` → run → `_dump_snapshot(<pickle>)`, viewed at pytorch.org/memory_viz) is **CUDA-only**; the fixture emits it **only when `torch.cuda.is_available()`** and silently skips it otherwise. The CUDA path stays documented in `references/gpu-memory.md` for a later run on a GPU box / CI.

## Report layout

```text
docs/perf/
├── README.md                     # what the two backends are; how to act on a report
├── scalene/
│   └── <YYYY-MM-DD>-<label>.md
└── pytorch_profiler/
    └── <YYYY-MM-DD>-<label>.md
```

Filenames are lowercase-hyphenated with a date prefix (per `.claude/rules/documentation.md`). Reports are ATX-headed, fenced-with-language, and pass `rumdl`. Each report ends with an **"Optimization candidates"** section — the actionable output a later session reads to change the codebase.

## Reused building blocks

- `pyinstrument` targets in the `Makefile` (`profile`, `profile-demo`, `profile-doctor`, `profile-open`) — mirror their shape for `profile-scalene` / `profile-torch`; the new backends complement, not replace, the sampling profiler.
- `evaluation.py:519 benchmark_model(...)` — existing inference-speed benchmark (CUDA warmup/synchronize, per-batch-size timing). Cross-reference its numbers in reports; do not duplicate it.
- `inference.py InferencePipeline.predict_image / predict_batch / predict_video` — the hot path reports point at; each calls `self.model.predict(...)`.
- `tests/test_inference_e2e.py` (real `YOLO("yolo26n.pt").predict()` on CPU) and `tests/integration/test_classifier_e2e.py` — the profiling targets.

## Dependencies

- **Scalene** — ephemeral, `uv run --with scalene`. Not added to any dependency group.
- **torch / torchvision** — already in the optional `worker` group; the torch backend runs under `uv sync --group worker` / `uv run --group worker`.
- **rich** — already available; used for the scripts' console summaries (per `.claude/rules/python-scripts.md`).
- No new pinned dependencies and no `uv.lock` changes.

## Testing (TDD)

- **Pure helpers** (inline `## Tests`, no pytest import): the pytest arg-vector builder (asserts `-p no:cov`, `addopts=""`, `--timeout=0`, and a `-m` selector are always present; the guard rejects a vector missing any strip flag); the scalene-JSON → markdown row mapper; the `key_averages` → markdown row mapper; the `<date>-<label>.md` filename builder.
- **`tests/` unit tests** (`mocker` fixture, never `unittest.mock`): subprocess calls patched — assert the exact resolved command; assert the renderer writes to the correct `docs/perf/<backend>/` path; assert torch-absent → clean skip, CUDA-absent → snapshot step skipped not errored.
- **Marker test**: `uv run pytest --collect-only -m pytorch_profile` lists the marked tests with no "unknown marker" warning.
- The real profiled runs are `integration` / `pytorch_profile` (local-only, excluded by the default `-m "not integration"`), so `make test` in CI stays fast.

## Step by Step Tasks

IMPORTANT: Execute every step in order, top to bottom.

### 1. Register the `pytorch_profile` marker

- Add to `pyproject.toml` `markers`: `"pytorch_profile: marks tests to run under torch.profiler (local/GPU only; deselect with '-m \"not pytorch_profile\"')"`.

### 2. Create the `docs/perf/` tree

- `docs/perf/scalene/.gitkeep`, `docs/perf/pytorch_profiler/.gitkeep`, and `docs/perf/README.md` (passes `rumdl`).

### 3. Scaffold the skill with `skill-creator`

- Generate `.claude/skills/perf/` (SKILL.md + `references/` + `scripts/`). SKILL.md documents the workflow: pick target → run backend → read report → propose changes.
- Write `references/scalene.md`, `references/pytorch-profiler.md`, `references/gpu-memory.md` (distilled notes + the three links above).

### 4. Write `scripts/render_report.py` (PEP 723)

- One renderer, two input modes (scalene JSON | torch key_averages + trace) → fixed markdown template → correct `docs/perf/<backend>/` file. `rich` console summary.

### 5. Write `scripts/run_scalene.py` (PEP 723)

- Build the guarded `uv run --with scalene python -m scalene … --- -m pytest …` command; run; hand the JSON to the renderer.

### 6. Add the conftest fixture + `scripts/run_torch_profiler.py`

- `pytorch_profiler` fixture in `tests/conftest.py` (profile wrap, trace + `key_averages` export, CUDA-gated snapshot).
- `run_torch_profiler.py` invokes `uv run --group worker pytest -m pytorch_profile <target> -p no:cov -o addopts="" --timeout=0`, then the renderer.

### 7. Mark the heavy tests

- Add `@pytest.mark.pytorch_profile` alongside the existing `integration` mark on `tests/test_inference_e2e.py` and `tests/integration/test_classifier_e2e.py`.

### 8. Add Makefile targets

- `profile-scalene` / `profile-torch` mirroring the `profile*` block, delegating to the skill scripts.

### 9. Generate seed reports

- Run each backend once against `tests/test_inference_e2e.py`; confirm reports land under `docs/perf/*`.

## Acceptance criteria

- `.claude/skills/perf/` exists with SKILL.md, `references/` (three source links), and runnable `scripts/`.
- `pytorch_profile` marker registered; `uv run pytest --collect-only -m pytorch_profile` lists the marked tests with no unknown-marker warning.
- Each backend produces a markdown report in the correct `docs/perf/` subtree with coverage + timeout stripped.
- CUDA snapshot auto-skips on this Mac without error; torch-absent yields a clean skip.
- `make lint`, `make test`, and `rumdl docs/` are clean.

## Validation commands

- `uv run pytest --collect-only -m pytorch_profile -q` — marker resolves, lists heavy tests, no unknown-marker warning.
- `uv run --with scalene python -m scalene --version` — Scalene reachable ephemerally.
- `uv run --group worker python -c "import torch.profiler; print('ok')"` — torch.profiler importable with the worker group.
- `uv run .claude/skills/perf/scripts/run_scalene.py tests/test_inference_e2e.py && ls docs/perf/scalene/` — seed Scalene report present.
- `uv run .claude/skills/perf/scripts/run_torch_profiler.py tests/test_inference_e2e.py && ls docs/perf/pytorch_profiler/` — seed torch report present.
- `make lint` — codespell + ruff + basedpyright clean.
- `make test` — suite green (heavy tests excluded by default).
- `rumdl docs/` — new docs conform.

## Notes

- Never edit the injected coverage `addopts`; strip per-invocation with `-p no:cov -o addopts="" --timeout=0` (CLAUDE.md).
- Scalene stays ephemeral; torch already lives in the `worker` group — no new pinned deps, no `uv.lock` churn.
- The *Understanding GPU Memory* snapshot workflow is CUDA-only; local Mac runs rely on `profile_memory=True` + MPS. Keep the CUDA path documented for a later GPU/CI run.
- If an audit agent is run against the new skill, follow `.claude/rules/audit-protocol.md`: pass only the file path, no context.
