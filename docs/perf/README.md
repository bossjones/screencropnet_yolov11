# Performance reports

Line-level and operator-level profiles of the YOLO inference hot path, produced
by the [`perf` skill](../../.claude/skills/perf/SKILL.md). These complement the
sampling profiles from `make profile-*` (pyinstrument) — they answer *which
source line* and *which operator* actually costs time and memory.

## Backends

- **`scalene/`** — [Scalene](https://github.com/plasma-umass/scalene) line-level
  CPU (Python vs native), peak memory, and GPU utilization. Process-level, no
  code changes, installed ephemerally via `uv run --with scalene`.
- **`pytorch_profiler/`** — the
  [PyTorch Profiler](https://pytorch.org/blog/introducing-pytorch-profiler-the-new-and-improved-performance-tool/)
  operator-level CPU/CUDA time, input shapes, and memory, plus a Chrome trace.
  In-process via the `pytorch_profile` pytest marker + a `tests/conftest.py`
  fixture. On this Mac it profiles CPU/MPS; the CUDA memory-history snapshot
  ([Understanding GPU Memory](https://pytorch.org/blog/understanding-gpu-memory-1/))
  is emitted only when CUDA is available.

## Generating reports

```bash
make profile-scalene TARGET=tests/test_inference_e2e.py
make profile-torch   TARGET=tests/test_inference_e2e.py
```

Each run writes one date-prefixed markdown file (`<YYYY-MM-DD>-<label>.md`) into
the matching subdirectory. Every run strips the injected coverage and 30s
thread-timeout, which would otherwise corrupt the measurements.

## Acting on a report

Each report ends with an **Optimization candidates** section. Read the hottest
rows against `src/screencropnet_yolo/inference.py`
(`InferencePipeline.predict_image` / `predict_batch` / `predict_video`) and
`model.py`, cross-check against `evaluation.py::benchmark_model`, then turn the
findings into concrete, test-backed changes.
