# PyTorch Profiler — `tests/test_inference_e2e.py`

- Date: 2026-07-02
- Target: `tests/test_inference_e2e.py`
- Activities: CPU / MPS (CUDA unavailable — memory snapshot skipped)
- Chrome trace: `docs/perf/pytorch_profiler/2026-07-02-test-inference-e2e.trace.json` (open in `chrome://tracing` or Perfetto)

## Command

```bash
uv run --group worker pytest tests/test_inference_e2e.py -p no:cov -o addopts= --timeout=0 --import-mode=importlib -m pytorch_profile
```

## Hottest operators

| Operator | Calls | Self CPU (µs) | CPU total (µs) | Self CUDA (µs) | CPU mem (B) | CUDA mem (B) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `aten::_slow_conv2d_forward` | 94 | 15566.9 | 19959.7 | 0.0 | 64057600 | 0 |
| `aten::copy_` | 1009 | 5811.9 | 5811.9 | 0.0 | 3008 | 0 |
| `aten::silu_` | 87 | 4287.2 | 4287.2 | 0.0 | 0 | 0 |
| `aten::cat` | 29 | 2640.4 | 2792.0 | 0.0 | 35036000 | 0 |
| `aten::mm` | 192 | 1581.9 | 1632.5 | 0.0 | 9557152 | 0 |
| `aten::add` | 315 | 1055.0 | 1488.8 | 0.0 | 6724220 | 0 |
| `aten::_convolution` | 102 | 924.7 | 21005.0 | 0.0 | 70022400 | 0 |
| `aten::max_pool2d_with_indices` | 3 | 776.3 | 776.3 | 0.0 | 1843200 | 0 |
| `aten::_to_copy` | 806 | 578.0 | 2407.5 | 0.0 | 15358720 | 0 |
| `aten::fill_` | 19 | 431.1 | 431.1 | 0.0 | 0 | 0 |
| `aten::empty` | 1068 | 426.8 | 426.8 | 0.0 | 127640644 | 0 |
| `aten::view` | 649 | 414.3 | 414.3 | 0.0 | 0 | 0 |
| `aten::upsample_nearest2d` | 2 | 401.9 | 710.9 | 0.0 | 4915200 | 0 |
| `aten::bmm` | 4 | 389.9 | 420.2 | 0.0 | 2969600 | 0 |
| `aten::empty_strided` | 806 | 365.7 | 365.7 | 0.0 | 13796164 | 0 |

## Optimization candidates

Read the hottest rows above against `src/screencropnet_yolo/inference.py` (`InferencePipeline.predict_image/predict_batch/predict_video`) and `model.py`. Cross-check the numbers with `evaluation.py::benchmark_model`. List concrete, testable changes here before touching code.
