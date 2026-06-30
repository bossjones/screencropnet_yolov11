# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Heads-up: CLAUDE.md is hand-edited and tracked in git

This file is the source of truth for project memory — edit it directly. The `agent-rules` `Makefile` target that previously regenerated `CLAUDE.md` (and `AGENTS.md`) from `.cursor/rules/*.mdc` is currently **commented out** to prevent clobbering hand edits, so `make` and `make clean` no longer touch this file.

If you want to regenerate from `.cursor/rules/` again, uncomment the `agent-rules`, `CLAUDE.md`, `AGENTS.md`, and matching `clean` lines in `Makefile`. The canonical source rules still live in `.cursor/rules/general.mdc` and `.cursor/rules/python.mdc` and are worth a read when updating conventions here.

## Commands

All Python execution must go through `uv` — never invoke `pip` or bare `python`.

```bash
make install        # uv sync --all-extras
make lint           # codespell + ruff check --fix + ruff format + basedpyright (devtools/lint.py)
make test           # uv run pytest (with coverage, junit, 30s timeout)
make check          # uv run ty check (separate, faster type checker — additional to basedpyright)
make                # agent-rules + install + lint + test
make build          # uv build
make upgrade        # uv sync --upgrade --all-extras --dev
make clean          # removes dist/, .venv/, caches, CLAUDE.md, AGENTS.md
```

Pytest specifics:

- `pytest` is configured (in `pyproject.toml`) to collect from both `src/` and `tests/` — inline tests under `## Tests` comments in source modules are picked up automatically.
- `make test` swallows stdout. To see output for a single test: `uv run pytest -s tests/test_model.py::test_something`.
- Custom markers: `e2e`, `fast`, `integration`, `slow`, `unittest`. Deselect with e.g. `-m "not slow"`.
- Tests have a 30s thread-based timeout by default.
- Coverage is on by default and writes to `cov.xml`, `htmlcov/`, `cov_annotate/`, and `junit/test-results.xml`. Open the HTML report with `make open-coverage`.
- Don't override coverage (`-p no:cov` or a custom `--cov`) — `addopts` injects cov flags and overriding them errors; run `uv run pytest <target>` as-is.

Type-stub generation from runtime traces:

```bash
make autotype       # monkeytype-create (trace via pytest) + monkeytype-apply (apply stubs)
```

## Architecture

This is a YOLO 26 training/inference pipeline for detecting and classifying bounding boxes in Twitter screenshots, built on Ultralytics. The package lives at `src/screencropnet_yolo/` (src-layout, hatchling build, dynamic version from git via `uv-dynamic-versioning`).

Pipeline modules (each is a self-contained stage; the train script wires them together):

- `dataset_utils.py` — `RoboflowLoader`, `DatasetSplitter`, `DatasetValidator`, `create_dataset_yaml`, `check_class_imbalance`, `display_dataset_stats`. Handles dataset acquisition, train/val/test splits, and YOLO-format `data.yaml` generation.
- `dataset_import.py` — `prepare_twitter_dataset`, `convert_csv`, `pascal_row_to_yolo`. Bridges externally produced Pascal-VOC CSV annotations into the single-class `tweet_region` YOLO layout, reusing `DatasetSplitter`/`create_dataset_yaml`.
- `model.py` — `ModelConfig`, `AugmentationConfig`, `ModelFactory`, `ModelExporter`, `ModelQuantizer`. Wraps Ultralytics `YOLO` with config dataclasses; centralizes hyperparameters, device selection, multi-GPU setup, and export formats. `ModelExporter` exports `pytorch` first (copying the real `train/weights/best.pt` via `source_weights`); other formats fail soft.
- `training.py` — `Trainer`, `TrainingHistory`, `create_ablation_study`. Owns the training loop and produces history artifacts; ablation helper compares hyperparameter variants.
- `evaluation.py` — `Evaluator`, `EvaluationResults`. Computes metrics on val/test sets after training.
- `inference.py` — runtime prediction on new images.
- `visualization.py` — `TrainingVisualizer`, `ConfusionMatrixVisualizer`, `ResultsDashboard`. Plot helpers used by both training and evaluation.
- `output.py` — `format_run_summary`, `format_artifacts_table`, `human_size`, `colorize`, `Color`, `Artifact`, `ColorFormatter`. Pure presentation helpers for the CLI's run banner, artifacts table, and color-aware logging; no Ultralytics/torch imports, raw ANSI instead of `rich`. Color is opt-in via `--color` (auto-disabled off a TTY or under `NO_COLOR`).
- `train.py` — CLI entry point that orchestrates the above (loads config, splits data, builds model, trains, evaluates, visualizes). Prints a RUN CONFIGURATION banner before training and an ARTIFACTS table after export. Logs to a timestamped file under the run's output dir.
- `screencropnet_yolo.py` — package entrypoint exposed via `[project.scripts]` as the `screencropnet_yolo` command (currently a stub `main()`).
- `config/config.yaml` — default training config consumed by `train.py --config`.

Key external dependencies: `ultralytics` (YOLO 26), `torch`, `opencv-python`, `wandb` (experiment tracking), `matplotlib`/`seaborn` (plots), `albumentationsx` (augmentation, dev-only).

ONNX export deps (`onnx`, `onnxslim`, `onnxruntime`) are pinned in main `dependencies`: ultralytics' export tries to `pip install` them on demand, which fails in the uv venv (no `pip`), so they must stay declared — add deps with `uv add`, never rely on auto-install.

End-to-end smoke (full train→eval→export→visualize against the local default dataset; Roboflow is disabled by default in `config.yaml`):

```bash
uv run python -m screencropnet_yolo.train --epochs 2 --model-size n --batch 4 --imgsz 320 --output ./runs/smoke
```

The default dataset lives at `datasets/twitter_screenshots_localization_dataset` (train/val/test + `data.yaml`); all `runs/` output is gitignored.

## Project conventions

These come from `.cursor/rules/python.mdc` (full text is in the generated `CLAUDE.md`/`AGENTS.md`); the high-impact items:

- **Imports**: always absolute (`from screencropnet_yolo.module import ...`), never relative. Import `Callable` etc. from `collections.abc`; use `typing_extensions` for `@override` (we still support 3.11). Add `from __future__ import annotations` where types appear.
- **Types**: modern union syntax (`str | None`, `list[str]`, `dict[str, X]`). Never import `Optional`. Use `@override` when overriding base methods.
- **Files**: prefer `pathlib.Path` and `Path(...).read_text()` over `with open(...)`. Use `strif.atomic_output_file` for writes.
- **Linting bar**: `make lint` and `make test` must be clean before considering work complete. Suppress basedpyright only with a specific `# pyright: ignore[ruleName]` and only when the warning is genuinely not a real problem. Globally disabling a rule in `pyproject.toml` requires explicit user confirmation.
- **Tests**: long tests in `tests/test_*.py`; small focused tests inline in the source file under a `## Tests` comment (inline tests must not import pytest). Use `pytest-mock`'s `mocker` fixture — never `unittest.mock`. Don't write trivial assertions or `assert False` (use `raise AssertionError(...)`).
- **Comments/docstrings**: explain *why*, never restate the code. No step-numbering, no decorative headers, no emojis except the project's existing `✔︎`/`✘`/`∆`/`‼︎` conventions for user output. Use `dedent()` for multi-line literals.
- **Backward compatibility**: don't add compat shims/aliases unless the user explicitly asks; do flag breaking API changes.

## Project-local Claude rules

`.claude/rules/` contains additional rules that apply project-wide:

- `audit-protocol.md` — when invoking audit/review agents, pass only the file path. No context, no hints about what was just changed, no "verify"/"check" language. Tainted context biases the audit.
- `python-scripts.md` — standalone scripts use PEP 723 inline metadata (`#!/usr/bin/env -S uv run` + `# /// script` block) so `uv run script.py` auto-installs deps.
- `documentation.md`, `plugin-structure.md`, `skill-development.md`, `visual-testing.md` — domain-specific conventions; consult when working in those areas.
