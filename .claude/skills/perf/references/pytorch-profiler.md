# PyTorch Profiler reference

Source: <https://pytorch.org/blog/introducing-pytorch-profiler-the-new-and-improved-performance-tool/>

`torch.profiler` records **per-operator** CPU and CUDA time, input shapes,
memory, and a stack — the operator-level view Scalene's line view can't give.
torch lives in the optional `worker` dependency group here, so all of this runs
under `uv sync --group worker` / `uv run --group worker`.

## The fixture

`tests/conftest.py::_pytorch_profiler` is autouse but inert unless a test carries
`@pytest.mark.pytorch_profile` **and** the runner set
`SCREENCROPNET_TORCH_PROFILE_DIR`. When active it wraps the test body in:

```python
from torch.profiler import ProfilerActivity, profile

activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(ProfilerActivity.CUDA)

with profile(activities=activities, record_shapes=True,
             profile_memory=True, with_stack=True) as prof:
    ...  # the test runs here
```

On teardown it exports:

- `prof.key_averages()` → a JSON list of events (name, count, self/total CPU
  time, self CUDA time, CPU/CUDA memory) that `perf.torch_rows()` ranks.
- `prof.export_chrome_trace(...)` → a `.trace.json` openable in `chrome://tracing`
  or [Perfetto](https://ui.perfetto.dev) for a timeline view.

## Reading key_averages

- **Self CPU time** — time in the op itself, excluding children. This is the
  ranking key on CPU/MPS; the top rows are your real cost centers (typically
  `aten::conv2d`, `aten::_convolution`, `aten::mm`).
- **Self CUDA time** — used as the ranking key when any op has CUDA time (a GPU
  box). Zero on this Mac.
- **Input shapes** (`record_shapes=True`) — same op, different shapes = different
  cost; tells you whether resizing inputs helps.
- **Memory** (`profile_memory=True`) — allocation per op; spikes point at
  intermediates worth reducing.

## Interpreting results here

- Convolution/matmul ops dominating self time → the model itself is the cost;
  wins come from smaller `imgsz`, batching, half precision, or export.
- Non-op time between ops in the trace → Python/pre-processing overhead; profile
  it with the Scalene backend instead.
