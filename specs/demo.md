# Spec: `demo` CLI — async inference on random images with an annotated contact sheet

## Purpose

Give a human a fast, visual way to *prove a model works* on real screenshots. The
existing paths — the full `train.py` pipeline and the async server/worker (RabbitMQ +
Postgres) — are heavyweight. `screencropnet_yolo.demo` is the "point at a folder, draw
boxes, look at the result" tool: sample N random images, run inference, draw bounding
boxes onto **copies** (originals untouched), stitch them into one contact-sheet montage,
and open it so detections are immediately visible.

## Usage

```bash
uv run python -m screencropnet_yolo.demo <images_dir> [options]
# or, once installed:
screencrop-demo <images_dir> [options]
```

Example:

```bash
uv run python -m screencropnet_yolo.demo \
  datasets/twitter_screenshots_localization_dataset/test/images -n 6
```

### Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `images_dir` (positional) | — | Directory to sample images from. |
| `-n`, `--count` | `10` | How many images to sample and annotate. |
| `--model PATH` | — | Explicit model weights (`.pt`/`.onnx`). |
| `--latest` | off | Use the newest trained `best.pt` discovered under `runs/`. |
| `--select` / `--fuzzy` | off | Interactively pick a `.pt`/`.onnx` model discovered under `runs/` with `fzf`. Highest precedence; mutually exclusive with `--model`/`--latest`. |
| `--seed INT` | — | Seed the sampler for a reproducible selection. Omit for random-each-run. |
| `--conf FLOAT` | config `inference.confidence` | Confidence threshold. |
| `--recursive` / `--no-recursive` | `--recursive` | Recurse into subdirectories when discovering images. |
| `--device STR` | config `device.type` | `auto`/`cpu`/`cuda`/`mps`/index. Resolved via `resolve_device`. |
| `-o`, `--output DIR` | fresh `/tmp` dir | Where copies, montage, log, and JSON land. |
| `-c`, `--config PATH` | packaged `config.yaml` | Config file. |
| `--color` | off | Colorize banner/table/log (TTY- and `NO_COLOR`-aware). |
| `--no-open` | off | Do not open the montage (headless/CI). |
| `--json` | off | Also write a detections JSON next to the montage. |

## Model resolution

Four sources, in precedence order (first match wins):

1. `--select` / `--fuzzy` — scan `runs/` recursively for every `.pt`/`.onnx` weight
   file, list them newest-first through `fzf` (each line is
   `name : /abs/path  [size]`), and load the picked file. This is the only source
   that surfaces exported `.onnx` artifacts. Error if no candidates are found;
   cancelling the picker (ESC) is a clean non-zero exit.
2. `--model PATH` — explicit path (`.pt` or `.onnx`). Error if it does not exist.
3. `--latest` — newest `runs/*/train/weights/best.pt` (also `runs/*/weights/best.pt`)
   by modification time. Error if no run is found.
4. Default ("current" model) — config `model.weights` if set, else the base pretrained
   checkpoint for `model.size` (e.g. `"m"` → `yolo26m.pt` via `ModelFactory.MODEL_SIZES`).

`--select`, `--model`, and `--latest` are mutually exclusive (argparse rejects any
combination). The run banner names which source was used (e.g. `selected (best.onnx)`)
so the output is never ambiguous.

`.onnx` selections run inference unchanged: `InferencePipeline` hands the path to
`YOLO(model_path)`, and ultralytics loads it via `onnxruntime` (already a pinned
dependency).

Note: the base checkpoint (e.g. `yolo26m.pt`) is an off-the-shelf COCO model, so on a
fresh checkout with no trained weights it may draw few or no `tweet_region` boxes — that
is expected, not a broken model. Use `--latest` or `--model` to see the trained detector.

## Async-first design

YOLO inference is CPU/GPU-bound and blocking, so it is not faked as coroutine-native.
The orchestration is async and offloads the blocking work off the event loop, mirroring
`server/worker.py`'s `anyio.to_thread.run_sync(classifier.infer, ...)`:

- `main` calls `asyncio.run(run_demo(...))`.
- `run_demo` bounds fan-out with an `asyncio.Semaphore(concurrency)` and, per image,
  `await anyio.to_thread.run_sync(annotate_one, pipeline, src, out_dir)`.
- `annotate_one` (the blocking body) reads the image, runs `pipeline.predict_image`,
  draws boxes with `pipeline.draw_detections`, and writes the annotated **copy**.

No blocking call runs on the event loop; N images process concurrently across threads.

## Output layout

Everything for a run lands under one directory (`/tmp/screencropnet_demo_*` unless
`--output` overrides):

```
<output_dir>/
  annotated/          # one annotated copy per sampled image (originals untouched)
  contact_sheet.png   # the montage that gets opened
  detections.json     # only with --json
  demo_<timestamp>.log
```

## Display

The montage is a single contact sheet: each annotated copy is resized-with-pad into a
square tile (aspect preserved) and tiled into a grid. It is opened on macOS via
`subprocess.Popen(["open", <montage>])`, detached. Display is skipped under `--no-open`,
on non-macOS platforms, and off a TTY, so CI and Linux never spawn a viewer.

## Reused building blocks

- `inference.InferencePipeline` — `predict_image`, `draw_detections` (promoted from the
  former private `_draw_detections`), and `ResultExporter.to_json`.
- `model.resolve_device`, `model.ModelFactory.MODEL_SIZES`.
- `output.format_run_summary`, `output.format_artifacts_table`, `output.Artifact`,
  `output.human_size`, `output.colorize`, `output.Color`, `output.ColorFormatter`.
- `train.load_config` for config loading; a small local `setup_logging` writing a
  timestamped log into the run's output dir.
- `pyfzf` (`FzfPrompt`) for `--select`, imported lazily so importing the module
  never requires the `fzf` binary. `fzf` is a **system prerequisite**
  (`brew install fzf`), not a pip-installable dependency — `pyfzf` only shells out
  to it. The selector is injectable, so unit tests never spawn a real `fzf`.

## Edge cases

- Empty directory / only non-image files → clear message, exit non-zero, no crash.
- `--count` greater than available images → sample all of them (clamp).
- `--count 0` → nothing to do; graceful message.
- `--latest` with no runs found → error naming the searched `runs/` dir.
- `--model` path missing → error.
- `--select` with no `.pt`/`.onnx` under `runs/` → error naming the searched dir, exit non-zero.
- `--select` cancelled (ESC in the picker) → "Model selection cancelled", exit non-zero.
- `fzf` not installed / not on PATH → clear error, exit non-zero, no crash.
- Unreadable image → warn and skip that one; do not abort the whole run.
- Non-macOS / headless → montage still written, viewer not spawned.

## Testing (TDD)

Tests are written first, watched fail, then implemented.

**Inline `## Tests`** in `demo.py` (pure, torch-free; no `import pytest`):

- `discover_images` — empty dir → `[]`; filters non-images; recursive vs non-recursive.
- `sample_images` — clamps when `count > len`; deterministic under a seeded
  `random.Random`; `count == 0` → `[]`.
- `find_latest_run` — picks newest by mtime; `None` when absent.
- `resolve_model` — full precedence ladder; missing `--model` path errors; `--latest`
  with no runs errors; `--select` uses the injected selector, errors with no
  candidates, and errors on cancel.
- `discover_models` — finds `.pt`+`.onnx` newest-first, excludes non-models, `[]` when absent.
- `format_model_choice` — exact `name : path  [size]` line.
- `build_contact_sheet` — output shape for N tiles at a given column count.

**`tests/test_demo.py`** (`pytest-mock` `mocker`, `pytest-asyncio` auto mode):

- `annotate_one` — mocked cv2 + pipeline; writes a copy, returns `(path, result)`.
- `run_demo` — mocked pipeline; one copy per image, results collected, bounded
  concurrency.
- `open_paths` — asserts mocked `subprocess.Popen` args; skipped under `--no-open` /
  non-macOS.
- `select_model` — a fake selector's chosen line maps back to the exact `Path`;
  an empty selection (cancel) yields `None`. No real `fzf` subprocess runs.
- `main` — happy path (everything below the pipeline mocked): exit 0, montage written,
  banner printed; empty-directory path: graceful non-zero, no crash; `--select` path
  routes through a patched `_fzf_select` (exit 0, picker called once) and exits
  non-zero when no models are discoverable.

## Acceptance criteria

- Sampling is random each run (different set) unless `--seed` is given.
- Originals are never modified; only copies are annotated.
- `-n/--count` changes the sample size; each model source selectable and named in the
  banner.
- Inference orchestration is async with off-thread offload and a bounded semaphore.
- `make lint`, `make check`, and `make test` are clean; new tests were written test-first.
