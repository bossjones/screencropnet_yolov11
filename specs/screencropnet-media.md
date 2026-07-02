# Spec: `screencrop-media` CLI — async crop + solid-background resize for images & video

> Implementation is **TDD** — drive every unit with `superpowers:test-driven-development` (RED → GREEN → refactor). Write the failing test first, then the minimal implementation. Pure core lands with inline `## Tests`; IO/async/CLI land in `tests/`.

## Purpose

Two Discord cogs in `cerebro-bot` — `autocrop.py` and `autoresize.py` — do genuinely useful media work: crop a Twitter screenshot down to its content region, then letterbox-resize it onto a solid-color background at a share-ready size. But they're welded to Discord (attachments, `ctx.send`, temp dirs), split across two incompatible stacks (an old EfficientNet-B0 regressor for images, raw ffmpeg for video), and can't touch a local file or folder.

`screencrop-media` is the standalone, batchable, async-first version of that capability, built on **this** repo's YOLO model. Point it at an image, a video, or a directory of them; optionally `--autocrop` (YOLO-detect → crop to the content region), optionally `--autoresize` (fit onto a solid background at a target size), do either, both, or neither; write artifacts to a **separate** output directory without ever mutating or deleting the inputs. It reuses the repo's standard `--select/--fuzzy` fzf model picker and the `demo` async-inference discipline, so it feels like the rest of the toolchain.

## Usage

```bash
uv run python -m screencropnet_yolo.media <input> [flags]
# installed console script:
screencrop-media <input> [flags]
```

```bash
# Crop + resize a single screenshot with the newest trained model:
screencrop-media ~/Downloads/tweet.png --autocrop --autoresize --latest

# Crop every screenshot in a folder (one level deep), pick the model interactively:
screencrop-media ~/Downloads/screenshots --autocrop --depth 1 --select

# Crop + square-resize a video clip:
screencrop-media ~/Downloads/clip.mp4 --autocrop --autoresize --square --latest

# Just report metadata, no processing:
screencrop-media ~/Downloads --info --json
```

`~` (and `$VAR`) in the input path are expanded, so `~/Downloads/clip.mp4` resolves to `/Users/you/Downloads/clip.mp4`.

### Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `input` (positional) | — | An image, a video, or a directory. `~`/env-var expanded. |
| `--autocrop` | off | YOLO-detect the content region and crop to the highest-confidence box. |
| `--autoresize` | off | Fit the (cropped) media onto a solid background at the target size. |
| `--output-dir PATH` | `./screencrop-media-out/` | Where artifacts go. Input subtree is mirrored under it. Never the input dir. |
| `--select` / `--fuzzy` | off | Interactively pick a `.pt`/`.onnx` model under `runs/` via `fzf`. Highest precedence. |
| `--model PATH` | — | Explicit model checkpoint. |
| `--latest` | off | Use the newest `best.pt` under `runs/`. |
| `--depth N` | `0` | Directory recursion depth (`0` = the top level only). |
| `--size WxH` | — | Explicit resize target (e.g. `1080x1350`). Overrides the default. |
| `--width N` / `--height N` | — | Resize target components (either or both). |
| `--square` | off | Shortcut for `1080x1080`. |
| `--bg-color HEX` | auto | Background fill (e.g. `#16202A`). Default: auto-detect white vs dark from the top-left pixel. |
| `--conf FLOAT` | config `inference.confidence` | Detection confidence threshold. |
| `--threads N` | `8` | Worker count bounding the async fan-out. |
| `--info` | off | Print a metadata table for the inputs and exit. No processing. |
| `--json` | off | With `--info`, emit JSON instead of a table. |
| `--color` / `--no-color` | auto | Force/disable ANSI color (auto-off off a TTY or under `NO_COLOR`). |
| `-v` … `-vvvv` | warnings | Increase verbosity: `-v` INFO, `-vv` DEBUG, `-vvv`/`-vvvv` TRACE. |
| `--help` | — | Show help and exit. |

Default resize target when `--autoresize` is set and no size flag is given: **1080×1350** (portrait 4:5). `--square` → **1080×1080**. `--size`/`--width`/`--height` override; passing both `--size` and `--width`/`--height` is an error.

## Model resolution

A model is only needed when `--autocrop` is set. Four sources, in precedence order (first match wins), mirroring `demo.resolve_model`:

1. `--select` / `--fuzzy` — interactive `fzf` pick of any `.pt`/`.onnx` under `runs/`.
2. `--model PATH` — explicit checkpoint.
3. `--latest` — newest `best.pt` under `runs/` by mtime (`demo.find_latest_run`).
4. config `model.weights`, else the base pretrained checkpoint for `model.size` (`ModelFactory.MODEL_SIZES`).

`--select`, `--model`, and `--latest` are mutually exclusive; the run banner names the chosen source. `fzf` is a **system prerequisite** for `--select` (`brew install fzf`), imported lazily so no other path needs it.

## Input discovery & path handling

- Expand `~` and `$VAR` with `Path.expanduser()`/`os.path.expandvars` before anything else.
- A file input is processed directly; a directory input is walked to `--depth` (0 = top level, N = N levels down).
- Files are classified by extension:
  - images: `.png` `.jpg` `.jpeg` `.webp` `.bmp`
  - video: `.mp4` `.mov` `.mkv` `.webm` `.gif`
  - anything else is ignored (and counted).
- Path resolution/statting fans out concurrently (async) — per the RealPython concurrency guidance, IO-bound directory walks and `stat`s overlap rather than block serially.

## Processing model

Per input, the pipeline runs `--autocrop` then `--autoresize`; each is independent, so all four combinations are valid:

- **`--autocrop`** — run YOLO detection, take the **single highest-confidence** `Detection.bbox` (x1,y1,x2,y2), clamp it to the image bounds, and crop (`img[y1:y2, x1:x2]`). No detections → skip the crop, log a warning, continue.
- **`--autoresize`** — fit the (possibly cropped) media onto a solid background at the target size: aspect-preserving scale (LANCZOS/`INTER_AREA`), centered, padded with the background color. Adapts `demo._fit_tile` / `DataPreprocessor.letterbox(auto=False)`.
- **neither** — passthrough copy into the output tree unchanged.

**Background color** is auto-detected from the top-left pixel: pure white → `#FFFFFF`; otherwise the Discord dark-mode `#16202A` (opaque — the old cog's alpha `1` was a bug). `--bg-color HEX` overrides.

**Video** uses a **single static crop rect** for the whole clip: sample a few frames, run detection on each, **union** the boxes, and apply one fixed rectangle — stable and fast, no per-frame jitter. Processing shells out to **ffmpeg**:

```
crop=w:h:x:y,scale=W:H:force_original_aspect_ratio=decrease,pad=width=W:height=H:x=-1:y=-1:color=0xRRGGBB
```

x264 encode with the audio track preserved (mirrors the old `autoresize` filter chain). `--autocrop` alone omits the `scale`/`pad`; `--autoresize` alone omits the `crop`.

## Async-first design

YOLO inference and cv2/ffmpeg work are CPU/GPU-bound and blocking, so the orchestration mirrors `demo.run_demo`:

- `asyncio.run(main())` from a sync entry point.
- `asyncio.Semaphore(threads)` bounds concurrent inputs.
- Each blocking unit (cv2 decode/crop/encode, ffmpeg subprocess, `pipeline.predict_image`) runs via `await anyio.to_thread.run_sync(...)`.
- A shared `threading.Lock` (`predict_lock`) serializes `.predict` on the single shared model — ultralytics' lazy fuse/predictor setup is **not** thread-safe (see `demo.annotate_one`); only predict is serialized, decode/draw/encode stay concurrent.
- Path resolution fans out with `asyncio.gather`.

## Output layout

```
screencrop-media-out/                 # --output-dir (default ./screencrop-media-out/)
  <mirrored input subdirs>/           # input tree structure preserved
    cropped-<stem>.<ext>              # --autocrop only
    resized-<stem>.<ext>              # --autoresize only
    cropped-resized-<stem>.<ext>      # both
    <stem>.<ext>                      # neither (passthrough copy)
```

Inputs are never modified or deleted. If `--output-dir` resolves to (or inside) the input directory, the tool refuses with a clear error. Writes are atomic (`strif.atomic_output_file`).

## Display

- A `rich.progress` bar advances over inputs.
- A closing `rich.table.Table` summarizes each artifact (name, kind, `human_size`, `✔︎`/`✘`), plus totals (processed / skipped / failed).
- Color is TTY- and `NO_COLOR`-aware; `--color`/`--no-color` force it.

## `--info`

Prints per-input metadata and exits without processing:

- images: path, kind, dimensions (via cv2), byte size.
- video: path, kind, dimensions, duration, fps, codec, bitrate (via `ffprobe`), byte size.

Rendered as a `rich` table, or as JSON with `--json`.

## Verbosity

`-v` INFO, `-vv` DEBUG, `-vvv`/`-vvvv` TRACE (a custom sub-DEBUG level). TRACE logs every stage: path resolved, model chosen, frames sampled, box selected, ffmpeg argv, each write. Uses `output.ColorFormatter`.

## Reused building blocks

- `model_select.discover_models` / `select_model` / `format_model_choice` / `_fzf_select` / `ModelSelector` / `MODEL_EXTS` — the `--select/--fuzzy` picker (injectable `selector` for torch-free tests).
- `inference.InferencePipeline` (`predict_image` → `InferenceResult`), `inference.Detection` (`.bbox` is the crop source), `inference.ResultExporter` (JSON metadata).
- `demo.resolve_model` / `find_latest_run` (model precedence + newest run), `demo.run_demo` / `annotate_one` / `_fit_tile` (async offload + `predict_lock`; resize-onto-bg template), `demo.setup_logging`.
- `model.resolve_device` (`"auto"` → `0`/`"mps"`/`"cpu"`).
- `output.human_size` / `Artifact` / `format_artifacts_table` / `Color` / `ColorFormatter`.
- `dataset_utils.DataPreprocessor.letterbox(auto=False)` — exact fit-onto-solid-color reference.
- `strif.atomic_output_file` — safe artifact writes.

## Edge cases

- No detections in a file → skip crop, log warning, still resize/passthrough; exit non-zero only if **every** input fails.
- Unreadable/corrupt file → skip, count it, continue.
- `fzf` not on PATH → clean non-zero; zero model candidates → error naming the searched dir; ESC-cancel → "Model selection cancelled", non-zero.
- `--output-dir` == (or inside) input dir → refuse, non-zero.
- `ffmpeg`/`ffprobe` missing → clear error naming the missing binary.
- Empty directory (no media at the given depth) → error naming the dir.
- `--size` together with `--width`/`--height` → error.

## Dependencies

- Promote `rich` from `[dependency-groups] dev` to `[project] dependencies` (the tool is a shipped entry point that imports rich at runtime).
- Video uses the `ffmpeg`/`ffprobe` **binaries** directly via plain `subprocess` (argv built by `media_core` for unit-testability) — **no `ffmpeg-python` library** is added. The binaries are **system prerequisites** (`brew install ffmpeg`), like `fzf` — not pip-installable.
- Reuse the existing `anyio.to_thread` idiom for async offload; no `aiofiles`.

## Testing (TDD)

Tests are written first, watched fail, then implemented. The suite exercises the **real** checked-in `tests/fixtures/` assets — the `00007`/`00012`/`00222_twitter.PNG` screenshots (note the uppercase `.PNG` → extensions match case-insensitively) and the `AlertesInfos-…-img1.mp4` clip — wherever a wrapper can run on real bytes without torch or a spawned model. Split by purity:

**Inline `## Tests`** in `media_core.py` (pure, torch-free; no `import pytest`):
- `classify_kind` / `discover_inputs` — run against `tests/fixtures/` (real `.PNG` case-insensitive → image, `.mp4` → video), plus a synthetic `tmp_path` tree for the `--depth` bound; single-file vs dir; deterministic sort.
- `pick_crop_box` / `clamp_box` — highest-confidence selection (inject a `NamedTuple` detection), clamp to bounds, reject inverted boxes; `union_boxes` for the static video rect.
- `resolve_target_size` / `parse_size` — `--size`/`--width`/`--height`/`--square` precedence + conflict errors.
- `resolve_bg_color` / `parse_hex_color` / `to_ffmpeg_color` — top-left white vs `#16202A`, `--bg-color` override, hex parse.
- `fit_geometry` — scaled size + centered offset (mirrors `_fit_tile`).
- `output_path_for` / `output_would_clobber_input` — mirrored subtree, `cropped-`/`resized-`/`cropped-resized-`/passthrough names, refuse output==input.
- `build_ffmpeg_filter` / `build_ffmpeg_crop_resize_args` / `parse_ffprobe_json` / `image_info_row` — assert on argv lists and fed JSON, no subprocess.
- `verbosity_to_level` — `-v…-vvvv` → level ints incl. TRACE.

**`tests/test_media.py`** (`pytest-mock` `mocker`, `pytest-asyncio` auto mode, `typer.testing.CliRunner`; `fixtures_dir`/`sample_image`/`sample_video` fixtures):
- IO wrappers on **real bytes**: `read_image` / `crop_image` / `resize_onto_bg` / `write_image` run on a real `sample_image` (assert real dimensions, crop shape, the letterbox canvas size + solid-bg corner pixel); `top_left_rgb` on a constructed array. Only ffmpeg (`run_ffmpeg`/`probe_media`) and the YOLO pipeline are patched.
- `run_media` async fan-out: `Semaphore(threads)` bound + `anyio.to_thread` offload over real fixture copies; a corrupt input fails without sinking the batch.
- `process_image`/`process_video`: passthrough copies bytes exactly; autocrop uses the highest-confidence box (fake pipeline); no-detections leaves the image uncropped; video builds the expected ffmpeg `crop`+`pad` filter (patched `run_ffmpeg`).
- Typer CLI via `CliRunner`: `--help`, `--info --json` on the fixtures dir, empty-dir → exit 1, output-inside-input refused, `--size`+`--width` conflict → exit 1, an end-to-end `--autoresize --square` run (real cv2) that asserts a 1080×1080 artifact and untouched input, and `--select` propagation to `resolve_model`.

The default suite spawns no torch, ffmpeg, or fzf (they're patched). A single tool-gated **`@pytest.mark.integration`** test (`shutil.which("ffmpeg")`-guarded, deselected by `addopts = -m "not integration"`) runs real ffmpeg on the fixture clip.

## Step by Step Tasks

1. **Dependencies & wiring** — promote `rich` to `[project] dependencies`; `uv add ffmpeg-python`; add `screencrop-media = "screencropnet_yolo.media:main"` to `[project.scripts]`; `uv sync`; confirm `ffmpeg`/`ffprobe` on PATH.
2. **Pure core: discovery & path handling (TDD)** — `expand_input`, `discover_inputs`, `classify_kind`, `MEDIA_IMAGE_EXTS`/`MEDIA_VIDEO_EXTS`.
3. **Pure core: geometry & naming (TDD)** — `pick_crop_box`, `clamp_box`, `resolve_target_size`, `resolve_bg_color`, `fit_geometry`, `output_path_for`.
4. **Pure core: ffmpeg/ffprobe argv + info rows (TDD)** — builders + `InfoRow`.
5. **Pure core: verbosity mapping (TDD)** — `verbosity_to_level`.
6. **IO wrappers (TDD, patched)** — `read_image`, `write_image`, `crop_image`, `resize_onto_bg`, `run_ffmpeg`, `probe_media`, `sample_video_frames`.
7. **Async orchestration (TDD)** — `process_one` (crop→resize→passthrough) + `run_media` mirroring `demo.run_demo`.
8. **Typer CLI shell (TDD, `CliRunner`)** — `app`/commands, `resolve_model`-style precedence, rich progress + summary table, `--info`, `main()`.
9. **Validate** — `make lint`, `make check`, `make test`; live smoke; confirm inputs untouched.

## Acceptance criteria

- Processes a file, a video, and a directory (`--depth`-bounded) with `~` expansion, writing only under the output dir and never mutating/deleting inputs.
- `--autocrop`/`--autoresize` work independently, together, and neither (passthrough); crop uses the highest-confidence box; resize letterboxes onto the resolved background at the target size.
- `--select/--fuzzy` picks a model via `fzf` with the documented precedence; `--info` prints a metadata table (+`--json`); `-v…-vvvv` scales logging; `--threads` bounds the async fan-out; rich progress + summary render and honor `--no-color`/`NO_COLOR`/non-TTY.
- Video uses ffmpeg with a single static crop rect and preserves audio.
- `make lint`, `make check`, and `make test` are clean; new tests were written test-first.

## Validation commands

- `uv run python -c "import screencropnet_yolo.media_core"` — torch-free import smoke.
- `uv run python -m py_compile src/screencropnet_yolo/media.py src/screencropnet_yolo/media_core.py`
- `make lint` / `make check` / `make test`
- Live smoke (needs `fzf`, `ffmpeg`):
  - `uv run screencrop-media ~/Downloads/tweet.png --autocrop --autoresize --latest`
  - `uv run screencrop-media ./datasets/twitter_screenshots_localization_dataset/test/images --autocrop --depth 1 --latest`
  - `uv run screencrop-media ~/Downloads/clip.mp4 --autocrop --autoresize --square --select`
  - `uv run screencrop-media ~/Downloads --info --json`

## Notes

- **Old-code fidelity:** dark-mode background is `#16202A` (opaque; the old cog's alpha `1` was a bug); default targets 1080×1350 (portrait) and 1080×1080 (square) are preserved.
- **Torch quarantine:** keep `media_core.py` torch-free; only `media.py` touches `InferencePipeline`/`resolve_device`. Avoid importing `model.py` unless the base-checkpoint fallback is needed.
- **Reach goals (documented, not built now):**
  - `--profile` → wrap the inference/processing path in `torch.profiler` (per the PyTorch profiler + GPU-memory blog posts), emitting a chrome trace, gated behind the flag.
  - scalene → a documented `uv run scalene -m screencropnet_yolo.media …` recipe to find CPU/IO hotspots in path resolution + image IO, then propose speedups (batch cv2 decode, tune `--threads`, a process pool for the pure-CPU letterbox).
