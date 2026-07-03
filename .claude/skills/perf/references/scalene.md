# Scalene reference

Source: <https://github.com/plasma-umass/scalene>

Scalene is a high-performance CPU, GPU, and memory profiler for Python that
attributes cost to individual **source lines** and separates **Python time from
native (C) time** — the distinction that tells you whether a hotspot is
optimizable Python or an unavoidable library call (e.g. torch/ONNX kernels).

## Install-free usage in this repo

Scalene is not a project dependency. Run it ephemerally so it never touches
`uv.lock`:

```bash
uv run --with scalene python -m scalene --cli --json --outfile out.json \
  --- -m pytest <target> -p no:cov -o addopts= --timeout=0 --import-mode=importlib -m integration
```

- `--cli --json --outfile` — machine-readable output the skill parses (no browser UI).
- `---` — everything after is the program Scalene launches (here, the pytest module).
- The trailing strip flags are mandatory: coverage's `sys.settrace` and the 30s
  thread-timeout both corrupt Scalene's sampling.

`screencropnet_yolo.perf.scalene_command(target, outfile)` builds this exact
vector; `scalene_rows()` + `render_scalene_markdown()` turn the JSON into a report.

## Reading the JSON

Top-level `files` maps each path to `{ "lines": [ … ] }`. Each line record the
skill uses:

- `n_cpu_percent_python` / `n_cpu_percent_c` — CPU % in Python vs native code.
  A line dominated by native % (e.g. `self.model.predict(...)`) is spending its
  time inside torch/ONNX; optimize by changing *how often / how much* you call
  it, not the Python around it.
- `n_peak_mb` — peak memory attributed to the line (allocation hotspots).
- `n_gpu_percent` — GPU utilization (0 on a CPU/MPS-only Mac).

## Interpreting results here

- High **native %** on a `predict` line → batch more, resize smaller, or export
  to a faster format; the Python is already thin.
- High **Python %** on a pre/post-processing line → that's yours to optimize
  (vectorize, avoid copies, move work out of the loop).
- High **peak MB** on model construction → look at load/warmup and dtype.
