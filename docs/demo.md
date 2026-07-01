# Demo tool

`screencrop-demo` is a lightweight visual smoke test for a detector. Point it at a
directory of screenshots and it samples a handful at random, runs inference on
*copies* (the originals are never touched), draws the detected boxes, and stitches
the annotated copies into a single `contact_sheet.png` montage that opens
automatically on macOS.

It is deliberately smaller than the full training pipeline
([usage.md](usage.md)) and the async classify service
([quickstart.md](quickstart.md)): no dataset validation, no metrics, no export —
just "point at some images and see what the model detects".

## Demo CLI

Two equivalent invocation forms:

```bash
# Console script (installed from pyproject.toml)
screencrop-demo <images_dir> [options]

# Module form (no console script needed)
uv run python -m screencropnet_yolo.demo <images_dir> [options]
```

### Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `images_dir` | | (required) | Directory of images to sample from (positional) |
| `--count` | `-n` | `10` | Number of images to sample |
| `--model` | | config default | Explicit model weights (`.pt`/`.onnx`) |
| `--latest` | | off | Use the newest trained `best.pt` under `runs/` |
| `--select` / `--fuzzy` | | off | Interactively pick a `.pt`/`.onnx` model under `runs/` with `fzf`; highest priority |
| `--seed` | | none | Seed the sampler for a reproducible selection |
| `--conf` | | config / `0.25` | Confidence threshold (overrides config) |
| `--recursive` / `--no-recursive` | | on | Recurse into subdirectories when discovering images |
| `--device` | | config / `auto` | Device: `auto`, `cpu`, `cuda`, `mps`, or an index |
| `--output` | `-o` | fresh `/tmp` dir | Output directory |
| `--config` | `-c` | packaged `config.yaml` | Config file (for class names, default model, thresholds) |
| `--color` | | off | Colorize the banner/table/log (auto-disabled off a TTY or under `NO_COLOR`) |
| `--no-open` | | off | Do not open the contact sheet (for headless/CI) |
| `--json` | | off | Also write a `detections.json` alongside the montage |
| `--concurrency` | | `8` | Max concurrent inference threads |

### Model resolution

The model is chosen by the first rule that matches (`resolve_model`):

1. `--select` / `--fuzzy` — scan `runs/` recursively for every `.pt`/`.onnx`
   weight, list them newest-first through `fzf` (each line is
   `name : /abs/path  [size]`), and load the picked file. This is the only source
   that surfaces exported `.onnx` artifacts. Errors if no candidates are found;
   cancelling the picker (ESC) exits cleanly with a non-zero status.
2. `--model PATH` — an explicit checkpoint (`.pt` or `.onnx`). Errors if the path
   does not exist.
3. `--latest` — the newest `best.pt` by modification time under `runs/` (matches
   `runs/*/train/weights/best.pt` and `runs/*/weights/best.pt`). Errors if none
   is found.
4. Config default — `model.weights` from the config file if set, otherwise the
   base pretrained checkpoint for `model.size` (e.g. `yolo26m.pt`).

`--select`, `--model`, and `--latest` are mutually exclusive. The run banner names
the chosen source (e.g. `selected (best.onnx)`) so the output is never ambiguous.
`.onnx` selections run inference unchanged — `InferencePipeline` hands the path to
`YOLO(...)`, which loads it via `onnxruntime`.

> **Prerequisite for `--select`:** the [`fzf`](https://github.com/junegunn/fzf)
> binary must be on your `PATH` (`brew install fzf`). It is a system dependency,
> not a Python package — the `pyfzf` wrapper only shells out to it, and it is
> imported lazily so the rest of the demo never requires it.

### Output layout

Everything is written under the output dir (a fresh
`/tmp/screencropnet_demo_*` dir by default, or `--output`):

```text
<output_dir>/
├── annotated/            # one annotated copy per sampled image
├── contact_sheet.png     # the montage (auto-opened on macOS)
├── detections.json       # only with --json
└── demo_YYYYMMDD_HHMMSS.log
```

### Examples

```bash
# Quickest check: newest trained model against a screenshots folder
screencrop-demo ~/screenshots --latest

# Reproducible sample of 6 images with a JSON dump of detections
screencrop-demo ~/screenshots -n 6 --seed 42 --json

# Interactively fuzzy-pick any .pt/.onnx under runs/ (requires fzf)
screencrop-demo ~/screenshots --select -n 4

# An explicit checkpoint at a custom confidence, on CPU
screencrop-demo ~/screenshots --model runs/twitter_detect/train/weights/best.pt --conf 0.4 --device cpu

# Headless / CI: skip the auto-open, keep everything under a known dir
screencrop-demo ~/screenshots --latest --no-open -o ./demo_out
```

### How it works

Inference is fanned out with bounded concurrency: each sampled image is handled by
an async worker gated on `--concurrency`, and the blocking YOLO call runs off the
event loop via `anyio.to_thread.run_sync`. A shared lock serializes the single
Ultralytics instance (it is not thread-safe). The finished montage is opened with
`open` on macOS and is skipped on other platforms or when `--no-open` is passed.

For the underlying inference API (`InferencePipeline`, `draw_detections`,
`ResultExporter`) see [usage.md](usage.md) and [api-reference.md](api-reference.md).
