# Plan: Training Output UX Overhaul

## Context

Why this change: a full `make train` run (100 epochs) currently emits a flat
stream of `INFO` log lines that bury the information an operator actually needs.
Three concrete problems prompted this work:

1. **You can't tell how far along training is.** Each line says `Epoch 1:`,
   `Epoch 2:` … with no total, so there's no sense of "12 of 100".
2. **The exported artifact you care about — the `.pt` — is silently dropped.**
   `config.yaml` lists `pytorch` *and* `onnx` as export formats, yet the run
   reported `Exported models: ['onnx']`. The PyTorch branch in `ModelExporter`
   looks for `{output_dir}/best.pt`, but Ultralytics writes the real best
   weights to `{output_dir}/train/weights/best.pt`, so the existence check fails
   and PyTorch is skipped without a warning.
3. **The "best model" path shown during training is misleading.** The
   `CheckpointCallback` logs `Saved new best model` after copying `last.pt` to
   `{output_dir}/weights/best.pt`, but the model that actually gets evaluated and
   exported is Ultralytics' own `{output_dir}/train/weights/best.pt` — a
   *different* file. The user is never told the real path the exported `.pt`
   lives at.

Intended outcome: a run that opens with a readable config banner (model, device,
**epoch count**, dataset, output paths), shows `Epoch N/total` progress, treats
`.pt` as the guaranteed primary artifact while ONNX stays best-effort, and closes
with an artifacts summary table listing the **real** paths and file sizes of
every output. An opt-in `--color` flag adds ANSI coloring. Built test-first.

## Objective

When complete, `uv run python -m screencropnet_yolo.train` produces:
- A startup **run-configuration banner** (model size/arch, device, total epochs,
  batch, imgsz, dataset path, output dir, weights dir, export formats).
- Per-epoch lines reading `Epoch N/TOTAL: …`.
- A guaranteed PyTorch (`.pt`) export, reported **first**, with its real path;
  ONNX (and any other format) failures logged as `WARNING` and skipped, never
  fatal, never silent.
- A closing **artifacts summary** table: best `.pt`, `.onnx` (if any),
  `training_history.json`, `evaluation_results.json`, visualizations dir — each
  with absolute path and human-readable size, plus best epoch / best mAP.
- An optional `--color` flag (default off) that colorizes banners, the summary,
  and log levels; honors `NO_COLOR` and non-TTY stdout.

## Problem Statement

The training CLI's value is hard to extract from its output. Progress is
unanchored (no total epochs), the most important deliverable (`.pt`) is reported
incorrectly (or not at all), and the paths shown for the "best model" don't match
the file that's actually exported. Operators must read source to learn where
their trained weights landed.

## Solution Approach

Three threads, all behind small, pure, unit-testable helper functions so the work
can be done test-first:

1. **Export correctness (`model.py`, `train.py`).** Fix the PyTorch branch to
   resolve the *actual* loaded checkpoint path instead of guessing
   `{output_dir}/best.pt`; copy it to a stable, reported location. Order the
   export loop so `pytorch` runs first. Wrap each non-PyTorch format in
   try/except that logs a `WARNING` and continues.
2. **Progress legibility (`training.py`).** Thread the total epoch count into
   `MetricsLogger` and render `Epoch N/TOTAL`. Make `CheckpointCallback` log the
   path it actually wrote, and align the "best model" notion with the file that
   gets exported.
3. **Framing + color (`train.py`, new `output.py`).** Add a run-config banner and
   an artifacts-summary table built from pure formatting functions in a new
   `output.py` module, plus a `colorize()` helper and a color-aware logging
   formatter gated on a new `--color` CLI flag.

## Relevant Files

Use these files to complete the task:

- `src/screencropnet_yolo/train.py` — CLI entry/orchestration. Adds `--color` and
  (optionally) `--no-onnx` argparse args, emits the run-config banner before
  training and the artifacts summary after export, wires color into logging
  setup (`setup_logging`, ~line 56), and owns `export_model()` (lines 286-312)
  where format ordering + warning-on-failure lives.
- `src/screencropnet_yolo/model.py` — `ModelExporter.export()` (lines 309-373).
  Fix the `pytorch` branch (lines 337-342) so it reports the real checkpoint
  path; ensure pytorch is emitted first and other formats fail soft.
- `src/screencropnet_yolo/training.py` — `MetricsLogger.on_epoch_end()` (per-epoch
  line, ~214-223), `CheckpointCallback.on_epoch_end()` (best-model log + path,
  ~321-334), `Trainer._setup_default_callbacks()` (~502-507, where MetricsLogger
  is built — pass total epochs here), `Trainer.train()` completion logs
  (~613-621). Total epochs come from `self.config.get("epochs", 100)`.
- `src/screencropnet_yolo/config/config.yaml` — `export.formats` already lists
  `pytorch` then `onnx` (lines 147-154); keep as-is (the bug is in code, not
  config). `training.epochs` at line 46.

### New Files

- `src/screencropnet_yolo/output.py` — pure presentation helpers: `colorize()`,
  `Color` constants / a tiny enabled-flag palette, `format_run_summary(...)`,
  `format_artifacts_table(...)`, `human_size(n_bytes)`, and a `ColorFormatter`
  (subclass of `logging.Formatter`). No I/O, no Ultralytics imports — keeps them
  trivially testable with inline `## Tests` and a `tests/test_output.py`.
- `tests/test_output.py` — unit tests for the formatting/color helpers.
- `tests/test_export_reporting.py` — tests that PyTorch is reported first and
  that an ONNX export failure degrades to a warning (using `mocker`).

## Implementation Phases

### Phase 1: Foundation
Create `output.py` with the pure helpers (`colorize`, `human_size`,
`format_run_summary`, `format_artifacts_table`, `ColorFormatter`) — written
test-first. These have no dependencies on training and can be fully covered by
unit tests.

### Phase 2: Core Implementation
Fix the export path bug and ordering in `model.py`/`train.py`; thread total
epochs into `MetricsLogger`; correct the `CheckpointCallback` path logging.

### Phase 3: Integration & Polish
Wire the banner + summary + `--color` flag into `train.py`'s `main()` and
`setup_logging`, then run the end-to-end smoke train to confirm real output.

## Step by Step Tasks

IMPORTANT: Execute every step in order, top to bottom. Follow the
`superpowers:test-driven-development` skill — write the failing test first, watch
it fail, then implement. Inline tests go under a `## Tests` comment in the source
module (no pytest import); longer tests go in `tests/test_*.py` using the
`mocker` fixture (never `unittest.mock`). Run `make lint` and `make test` clean
before declaring done.

### 1. Save this spec
- Write this document to `specs/ux-training.md` (the canonical location the user
  asked for; this plan file is a working copy).

### 2. (TDD) Build the `output.py` presentation helpers
- Write `tests/test_output.py` first, asserting:
  - `human_size(1536)` → `"1.5 KB"`, `human_size(0)` → `"0 B"`, MB/GB rollover.
  - `colorize("x", Color.GREEN, enabled=False)` returns `"x"` unchanged;
    `enabled=True` wraps with the ANSI code and a reset.
  - `format_run_summary(...)` output contains the total epoch count, model size,
    device, dataset path, output dir, and the export formats — assert on
    substrings, not exact layout.
  - `format_artifacts_table([...])` renders one row per artifact with its path
    and size, and omits/labels missing artifacts gracefully.
- Implement `src/screencropnet_yolo/output.py` to pass. Keep functions pure
  (take plain args/dataclasses, return `str`); `from __future__ import
  annotations`, absolute imports, modern type syntax.
- Add a couple of inline `## Tests` for `human_size`/`colorize` edge cases.

### 3. (TDD) Color-aware logging formatter + `--color` flag
- In `tests/test_output.py`, assert `ColorFormatter(enabled=False)` produces no
  ANSI escapes and `enabled=True` colors the levelname; assert it honors a
  passed `enabled` resolved from `--color` AND `NO_COLOR`/non-TTY.
- Implement `ColorFormatter` in `output.py`.
- In `train.py`: add `--color` (store_true, default False) to argparse; compute
  effective enable = `args.color and sys.stdout.isatty() and "NO_COLOR" not in
  os.environ`; pass it into `setup_logging` (~line 56) so the stream handler uses
  `ColorFormatter`. File-log handler stays plain (no ANSI in the logfile).

### 4. (TDD) Guarantee + correctly report the PyTorch export
- Write `tests/test_export_reporting.py` (using `mocker`) asserting:
  - `export(["onnx", "pytorch"])` returns a dict whose **first** reported/logged
    format is `pytorch` and whose `pytorch` value is the real checkpoint path
    (not a non-existent `{output_dir}/best.pt`).
  - When the underlying ONNX `model.export(...)` raises, `export()` logs a
    `WARNING`, omits `onnx` from the result, and still returns `pytorch` — it
    does not propagate the exception.
- In `model.py` `ModelExporter.export()`:
  - Reorder so `pytorch` is handled before other formats (or sort formats with
    pytorch first).
  - In the `pytorch` branch, resolve the actual source weights the exporter was
    constructed with (the best `.pt` passed from `train.py`), copy it to a stable
    reported path under the run dir if needed, and always include it in the
    result. Do not gate on a guessed `{output_dir}/best.pt` existing.
  - Wrap each non-pytorch format's export in try/except → `logger.warning(...)`
    + `continue`.

### 5. Per-epoch `N/total` progress
- Pass total epochs into `MetricsLogger` (constructor arg, sourced from
  `self.config.get("epochs", 100)` in `Trainer._setup_default_callbacks`,
  ~lines 502-507; or read `trainer.epochs` inside the callback if cleaner).
- Update the `MetricsLogger.on_epoch_end()` format string (~214-223) to
  `Epoch {epoch}/{total}: …`.
- Add/adjust a focused test (inline or in `tests/`) for the formatted line given
  a known epoch/total.

### 6. Correct the "best model" path logging
- In `CheckpointCallback.on_epoch_end()` (~328-334), include the written path in
  the log: `Saved new best model (mAP50-95: X) -> {best_path}`.
- Reconcile the dual best-`.pt` confusion: make the final artifacts summary point
  at the **exported** best `.pt` (the one returned by `export_model`/Ultralytics
  `trainer.best`), since that's what users consume. If the manual
  `{output_dir}/weights/best.pt` copy is redundant with Ultralytics'
  `train/weights/best.pt`, note it in the summary or prefer the exported path.
  (Do not silently delete the existing copy behavior; surface the canonical one.)

### 7. Run-config banner at startup
- In `train.py` `main()`, after config load/merge and before `train_model()`
  (~line 504), call `format_run_summary(...)` and log it. Include: model size +
  resolved arch label, device, **total epochs**, batch, imgsz, dataset/data.yaml
  path, output dir, weights dir + expected best `.pt`, and the export formats
  list. Replace or precede the existing bare `STARTING YOLO 26 TRAINING` banner.

### 8. Artifacts summary at the end
- In `train.py` `main()`, after export + visualizations and before the final
  `TRAINING COMPLETE` banner (~529-533), gather the produced artifacts (best
  `.pt` from export result first, `.onnx` if present, `training_history.json`,
  `evaluation_results.json`, `visualizations/`), compute sizes via
  `Path.stat().st_size` + `human_size`, and log `format_artifacts_table(...)`.
  Include best epoch and best mAP@50-95 in the summary.

### 9. Validate
- Run `make lint` and `make test` until clean.
- Run the end-to-end smoke train (Step "Validation Commands" below) and eyeball
  the banner, `Epoch N/total` lines, `.pt`-first export report, and artifacts
  table. Run once with `--color` and once without.

## Testing Strategy

- **Unit (pure helpers):** `tests/test_output.py` covers `human_size`,
  `colorize`, `ColorFormatter`, `format_run_summary`, `format_artifacts_table`.
  These are deterministic string functions — assert on substrings/structure, not
  brittle exact layouts.
- **Behavioral (export):** `tests/test_export_reporting.py` uses `mocker` to stub
  the Ultralytics `model.export` call: assert pytorch-first ordering, real
  `.pt` path, and warning-not-raise on ONNX failure. No real model download.
- **Inline:** small `## Tests` blocks in `output.py` for `human_size`/`colorize`
  edge cases (must not import pytest).
- **Edge cases:** `NO_COLOR` set; stdout not a TTY (color auto-disabled);
  `--color` not passed (plain output); ONNX export raising; an artifact file
  missing when building the summary (row labeled, not a crash); epoch total of 0
  or 1 in the progress string.
- **E2E:** the smoke command below exercises the full train→eval→export→summary
  path against the local dataset.

## Acceptance Criteria

- Per-epoch lines read `Epoch N/TOTAL: …`.
- A run-config banner appears before training with model, device, total epochs,
  dataset, output dir, weights dir, and export formats.
- `Exported models` reports `pytorch` **first** with a path that exists on disk;
  `.pt` is never silently dropped.
- An ONNX export failure produces a `WARNING` and the run still completes with the
  `.pt` exported.
- A closing artifacts table lists best `.pt`, `.onnx` (if produced),
  `training_history.json`, `evaluation_results.json`, and the visualizations dir,
  each with an absolute path and human-readable size, plus best epoch / best mAP.
- `--color` colorizes banners/summary/log levels when stdout is a TTY and
  `NO_COLOR` is unset; output is plain otherwise and by default.
- The "Saved new best model" log includes the path written; the artifacts summary
  points at the actually-exported best `.pt`.
- `make lint` and `make test` are clean.

## Validation Commands

Execute these to validate the task is complete:

- `make lint` — codespell + ruff + basedpyright clean.
- `make test` — full suite (incl. new `tests/test_output.py`,
  `tests/test_export_reporting.py`, and inline `## Tests`) passes.
- `make check` — `ty` type check clean.
- `uv run pytest -s tests/test_output.py tests/test_export_reporting.py` — see the
  new tests' output directly.
- `uv run python -m screencropnet_yolo.train --epochs 2 --model-size n --batch 4 --imgsz 320 --output ./runs/smoke`
  — confirm banner, `Epoch 1/2`/`Epoch 2/2`, `.pt`-first export with real path,
  and the artifacts table.
- `uv run python -m screencropnet_yolo.train --epochs 2 --model-size n --batch 4 --imgsz 320 --output ./runs/smoke-color --color`
  — confirm ANSI coloring on a TTY.

## Notes

- No new dependencies: `--color` uses raw ANSI codes in the new `output.py`, not
  a TUI library (the user chose option 1, not "full rich UI"). Do **not** add
  `rich`.
- Keep `output.py` free of Ultralytics/torch imports so its tests stay fast and
  import-light.
- The dual best-`.pt` situation (manual `{output_dir}/weights/best.pt` copy vs.
  Ultralytics `{output_dir}/train/weights/best.pt`) is pre-existing; this work
  surfaces the canonical exported path rather than refactoring the checkpoint
  pipeline. If the manual copy is dead weight, flag it to the user rather than
  removing it unprompted (per the project's "no unprompted compat/behavior
  changes" rule).
- Follow project conventions: absolute imports, `from __future__ import
  annotations`, `pathlib.Path`, modern union types, no `Optional`. Suppress
  basedpyright only with targeted `# pyright: ignore[ruleName]`.
