# Understanding GPU memory reference

Source: <https://pytorch.org/blog/understanding-gpu-memory-1/>

PyTorch can record a **memory-history snapshot** — every CUDA allocation/free
with its stack — and render it as an interactive timeline at
<https://pytorch.org/memory_viz>. This is the definitive tool for CUDA OOMs and
fragmentation, showing exactly which tensors are alive and where they came from.

## CUDA-only — skipped on this Mac

The snapshot API is **CUDA-specific**; there is no MPS/CPU equivalent. This
machine has no CUDA, so `tests/conftest.py::_pytorch_profiler` guards it behind
`torch.cuda.is_available()` and skips it locally. Local memory insight comes from
the profiler's `profile_memory=True` (per-op allocations) instead. Keep this path
for a later run on a GPU box or CI.

## The snapshot workflow (GPU box / CI only)

```python
import torch

# 1. Start recording before the workload.
torch.cuda.memory._record_memory_history(max_entries=100_000)

# 2. Run the model (load, warmup, inference/training).
...

# 3. Dump the snapshot, then stop recording.
torch.cuda.memory._dump_snapshot("memory-snapshot.pickle")
torch.cuda.memory._record_memory_history(enabled=None)
```

Then drag `memory-snapshot.pickle` onto <https://pytorch.org/memory_viz> to see
the allocation timeline, stacks, and fragmentation.

The fixture already runs these three steps automatically around a
`pytorch_profile` test when CUDA is present, writing the pickle next to the
report's raw artifacts.

## Reading a snapshot

- **Active memory timeline** — total allocated over time; a rising sawtooth that
  never returns to baseline is a leak or accumulation.
- **Allocation stacks** — click a block to see the Python stack that allocated
  it; maps a spike back to a specific line in `model.py` / `inference.py`.
- **Fragmentation** — reserved-but-unallocated gaps; large gaps mean the
  allocator can't reuse freed blocks, a cause of "OOM with free memory".
