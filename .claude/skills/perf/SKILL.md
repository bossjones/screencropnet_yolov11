---
name: perf
description: >-
  Profile the YOLO inference hot path in this repo with Scalene (line-level
  CPU/GPU/memory) and the PyTorch Profiler (operator-level traces), writing
  durable markdown reports under docs/perf/. Use when the user says "profile",
  "scalene", "pytorch profiler", "why is X slow", "performance", "cpu hotspot",
  "memory hotspot", "gpu memory", or wants to find and fix an inference/training
  bottleneck. Not for pyinstrument sampling (that's `make profile-*`).
---

# perf — Scalene + PyTorch Profiler harness

Run the repo's real-model inference tests under two deep profilers and land the
results as markdown under `docs/perf/`, so the hotspots can be turned into
concrete code changes. This complements the existing pyinstrument sampling
(`make profile-*`) with line-level and operator-level attribution.

## When to use

- "Why is inference slow?" / "profile predict_image" / "find the CPU hotspot".
- "What's allocating memory in the model load?" / "gpu memory".
- Before optimizing `inference.py` / `model.py` — generate evidence first.

For a quick flame graph, prefer `make profile-demo` (pyinstrument) instead.

## The two backends

| Backend | Granularity | How it runs | Report |
| --- | --- | --- | --- |
| Scalene | per source line: CPU (Python vs native), peak MB, GPU % | `uv run --with scalene` around pytest (ephemeral, no code change) | `docs/perf/scalene/<date>-<label>.md` |
| PyTorch Profiler | per operator: self CPU/CUDA time, shapes, memory + Chrome trace | `pytorch_profile` marker + `tests/conftest.py` fixture, torch from the `worker` group | `docs/perf/pytorch_profiler/<date>-<label>.md` |

## Workflow

1. **Pick a target.** Default to `tests/test_inference_e2e.py` (real
   `yolo26n.pt` inference on CPU). Real-model targets carry `@pytest.mark.integration`
   and are excluded from `make test`; the PyTorch backend also needs
   `@pytest.mark.pytorch_profile` (already on the two heavy tests). Adding a new
   target? Mark it and confirm with `uv run pytest --collect-only -m pytorch_profile`.

2. **Run a backend.**

   ```bash
   uv run .claude/skills/perf/scripts/run_scalene.py tests/test_inference_e2e.py
   uv run .claude/skills/perf/scripts/run_torch_profiler.py tests/test_inference_e2e.py
   # or: make profile-scalene TARGET=... / make profile-torch TARGET=...
   ```

   Both strip coverage + the 30s thread-timeout (`-p no:cov -o addopts= --timeout=0
   --import-mode=importlib`); the builders in `screencropnet_yolo.perf` refuse to
   run a command that doesn't. Scalene installs on demand; torch comes from
   `uv sync --group worker`. On this Mac the torch backend profiles CPU/MPS and
   skips the CUDA-only memory snapshot automatically.

3. **Read the report.** Open the generated `docs/perf/<backend>/<date>-<label>.md`.
   The **Hottest lines / operators** table names the cost centers; the
   **Optimization candidates** section is where you write concrete, testable
   changes.

4. **Improve the code.** Map the hotspots to
   `src/screencropnet_yolo/inference.py`
   (`InferencePipeline.predict_image/predict_batch/predict_video`) and `model.py`.
   Cross-check timings against `evaluation.py::benchmark_model`. Make changes
   test-first (`superpowers:test-driven-development`), then re-profile to confirm
   the win.

## Architecture

- `screencropnet_yolo.perf` — torch-free, subprocess-free pure core: command
  builders (with the mandatory strip guard), report naming, profiler-output
  parsing, and markdown rendering. Unit-tested in `tests/test_perf.py`.
- `scripts/run_scalene.py`, `scripts/run_torch_profiler.py`,
  `scripts/render_report.py` — thin IO wrappers that run the profilers and write
  the reports. Run them with `uv run` (project env), not as isolated scripts.
- `tests/conftest.py::_pytorch_profiler` — the marker-gated profiler fixture.

## References

- `references/scalene.md` — Scalene install-free usage, JSON output, reading it.
- `references/pytorch-profiler.md` — `torch.profiler` API, `key_averages`, traces.
- `references/gpu-memory.md` — the CUDA memory-history snapshot workflow (remote/GPU only).

## Guardrails

- Never edit the injected coverage `addopts` in `pyproject.toml`; strip
  per-invocation (CLAUDE.md).
- Scalene stays ephemeral; torch already lives in `worker` — add no pinned deps.
- Auditing this skill? Follow `.claude/rules/audit-protocol.md`: pass only the path.
