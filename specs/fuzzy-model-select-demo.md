# Plan: fzf fuzzy model selection for the `demo` CLI

## Task Description

Add an interactive fuzzy-finder model picker to `screencropnet_yolo.demo`. A new
flag (`--select`, aliased `--fuzzy`) scans the filesystem (default: `runs/`) for
model weight files — both `.pt` and `.onnx` — and presents them through
[`fzf`](https://github.com/junegunn/fzf) (via [`pyfzf`](https://github.com/nk412/pyfzf)).
Each candidate is shown as a single readable line:

```
   best.pt : /Users/bossjones/dev/bossjones/screencropnet_yolov11/runs/twitter_detect/best.pt  [42.0 MB]
 best.onnx : /Users/bossjones/dev/bossjones/screencropnet_yolov11/runs/twitter_detect/train/weights/best.onnx  [77.9 MB]
```

The path the user picks becomes the model loaded for that demo run, and the run
banner names it as the model source. `specs/demo.md` is updated to document the
new flag, the model-source precedence change, and the new edge cases.

## Objective

`uv run python -m screencropnet_yolo.demo <images_dir> --select` opens an fzf
picker listing every discovered `.pt`/`.onnx` under `runs/`, and loads the chosen
file for inference — with clean, tested failure modes when `fzf` is missing, no
candidates exist, or the user cancels the picker. `make lint`, `make check`, and
`make test` stay green; new logic is written test-first.

## Problem Statement

Today the demo picks a model from three non-interactive sources: `--model PATH`,
`--latest` (newest `best.pt`), or the configured/base checkpoint. To point the
demo at any *other* trained artifact — an older run, a specific ablation, or an
exported `.onnx` — the user must hand-type a long path. There is no way to
browse what has actually been produced under `runs/`, and `.onnx` exports are not
discoverable at all (`find_latest_run` only looks for `best.pt`). This is friction
exactly when a human is trying to eyeball "does *this* model work."

## Solution Approach

Introduce fzf-backed interactive selection as a first-class, highest-precedence
model source, layered onto the existing `resolve_model` ladder without disturbing
the pure/torch-free helpers that are already inline-tested:

1. **Discovery** — a pure `discover_models(root)` walks `root` recursively for
   `.pt`/`.onnx` files, newest-first. Reuses the same rglob style as
   `discover_images`/`find_latest_run`.
2. **Presentation** — a pure `format_model_choice(path)` renders one fzf line
   using the existing `output.human_size`. Selection maps display-line → `Path`
   via a dict so we never fragile-parse the line back.
3. **Interaction** — `select_model(candidates, selector=...)` calls an injectable
   `selector` callable (defaulting to a thin `pyfzf` wrapper). Injection keeps the
   subprocess/TTY-bound bit out of unit tests.
4. **Integration** — `resolve_model` gains a `select: bool` branch at the top of
   the precedence ladder; `--select`/`--model`/`--latest` become a mutually
   exclusive argparse group. The banner label becomes `selected (best.onnx)`.

ONNX "just works" for inference: `InferencePipeline.__init__` does
`self.model = YOLO(model_path)` (`inference.py:132`), and ultralytics loads
`.onnx` via `onnxruntime`, which is already a pinned dependency.

## Relevant Files

Use these files to complete the task:

- `src/screencropnet_yolo/demo.py` — add `discover_models`, `format_model_choice`,
  `select_model`, the default fzf selector, wire `--select`/`--fuzzy` into
  `parse_args`, extend `resolve_model`, and add inline `## Tests`. This is the
  primary edit surface.
- `tests/test_demo.py` — add mocked tests for the interactive selector and for
  `main --select` (the fzf call is always mocked; no real subprocess in tests).
- `src/screencropnet_yolo/output.py` — reuse `human_size` for the `[42.0 MB]`
  suffix (no change needed; read-only reference at `output.py:47`).
- `src/screencropnet_yolo/inference.py` — confirm `YOLO(model_path)` accepts the
  chosen path incl. `.onnx` (`inference.py:112`,`:132`; read-only reference).
- `specs/demo.md` — update the flags table, "Model resolution" precedence list,
  "Edge cases", and "Testing (TDD)" sections to include fuzzy selection.
- `pyproject.toml` — add the `pyfzf` runtime dependency (via `uv add`).
- `CLAUDE.md` — the demo command reference is not currently listed there; no edit
  required, but note the `fzf` system-binary prerequisite in `specs/demo.md`.

### New Files

None. All code lands in existing modules.

## Implementation Phases

### Phase 1: Foundation
Add the `pyfzf` dependency and the pure discovery/formatting helpers with inline
tests. No interactivity yet — everything here is deterministic and torch-free.

### Phase 2: Core Implementation
Add the injectable `select_model` + default fzf selector, extend `resolve_model`
with the `select` branch, and wire the mutually exclusive `--select`/`--fuzzy`
argparse group. Add mocked tests in `tests/test_demo.py`.

### Phase 3: Integration & Polish
Handle failure modes (missing `fzf` binary, zero candidates, user cancel), update
`specs/demo.md`, and run the full `lint`/`check`/`test` gate plus a manual smoke.

## Step by Step Tasks

IMPORTANT: Execute every step in order, top to bottom. Follow TDD — write the
failing test first, watch it fail, then implement (per the `superpowers:test-driven-development`
and `prefers-tdd` conventions).

### 1. Add the `pyfzf` dependency
- Run `uv add pyfzf` so it lands in `[project].dependencies` and `uv.lock`.
- Confirm the `fzf` binary is available (`command -v fzf`); it is already at
  `/opt/homebrew/bin/fzf` on this machine. `pyfzf` shells out to it and does
  **not** bundle it — this is a documented system prerequisite, not a pip dep.

### 2. Write inline tests for `discover_models` and `format_model_choice` (TDD, fail first)
- In `demo.py`'s `## Tests` block, add:
  - `test_discover_models_finds_pt_and_onnx` — a temp `runs/` with
    `a/train/weights/best.pt` and `b/weights/best.onnx` (+ a `.txt` decoy) →
    both models returned, decoy excluded, newest-first ordering asserted via
    `os.utime` (mirror `test_find_latest_run_picks_newest`).
  - `test_discover_models_empty_when_absent` — non-existent/empty root → `[]`.
  - `test_format_model_choice_line` — asserts the exact shape
    `f"{name} : {path}  [{size}]"` for a known byte size (e.g. 42 MB → `42.0 MB`).
- Run `uv run pytest src/screencropnet_yolo/demo.py -q` and watch them fail.

### 3. Implement `discover_models` and `format_model_choice`
- Add near `find_latest_run`:
  ```python
  MODEL_EXTS = {".pt", ".onnx"}

  def discover_models(search_root: Path) -> list[Path]:
      """All model weight files (.pt/.onnx) under search_root, newest first."""
      if not search_root.is_dir():
          return []
      found = [
          p for p in search_root.rglob("*")
          if p.is_file() and p.suffix.lower() in MODEL_EXTS
      ]
      return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)

  def format_model_choice(path: Path) -> str:
      """One fzf line: 'best.pt : /abs/path  [42.0 MB]'."""
      return f"{path.name} : {path}  [{human_size(path.stat().st_size)}]"
  ```
- Import `human_size` from `screencropnet_yolo.output` (extend the existing import
  on `demo.py:39`).
- Rerun the inline tests — now green.

### 4. Write mocked tests for `select_model` (TDD, fail first)
- In `tests/test_demo.py`, add a `TestSelectModel` class:
  - `test_returns_picked_path` — build two temp files, pass a fake `selector`
    that returns the display line for the second candidate; assert
    `select_model(...)` returns that `Path` exactly (no parsing fragility —
    proves the display→Path dict mapping).
  - `test_returns_none_on_cancel` — fake `selector` returns `[]` (fzf cancel/ESC)
    → `select_model` returns `None`.
- These reference `demo.select_model`, which does not exist yet → import/attr
  error is the expected first failure.

### 5. Implement `select_model` and the default fzf selector
- Define the injectable selector type and default:
  ```python
  from collections.abc import Callable

  ModelSelector = Callable[[list[str]], list[str]]

  def _fzf_select(choices: list[str]) -> list[str]:
      """Default selector: hand choices to fzf via pyfzf, return the chosen line(s)."""
      from pyfzf.pyfzf import FzfPrompt  # local import: fzf is only needed for --select
      return FzfPrompt().prompt(choices, "--height=40% --reverse")

  def select_model(
      candidates: list[Path], *, selector: ModelSelector | None = None
  ) -> Path | None:
      """Present candidates via fzf; return the chosen Path, or None if cancelled."""
      choices = {format_model_choice(p): p for p in candidates}
      picked = (selector or _fzf_select)(list(choices))
      if not picked:
          return None
      return choices[picked[0]]
  ```
- Local-import `pyfzf` inside `_fzf_select` so importing `demo` never requires the
  `fzf` binary (keeps inline tests and non-`--select` runs dependency-light).
- Rerun `tests/test_demo.py::TestSelectModel` — green.

### 6. Write tests for the `resolve_model` `select` branch (TDD, fail first)
- Add inline tests in `demo.py`:
  - `test_resolve_model_select_uses_selector` — pass `select=True` plus a fake
    selector returning a known candidate line; assert the returned ref is that
    file's path and the label starts with `selected`.
  - `test_resolve_model_select_no_candidates_errors` — empty `runs_dir` +
    `select=True` → `FileNotFoundError` mentioning `.pt/.onnx`.
  - `test_resolve_model_select_cancel_errors` — selector returns `[]` →
    a clear error (e.g. `RuntimeError("Model selection cancelled")`).
- Watch fail (new `select`/`selector` params don't exist yet).

### 7. Extend `resolve_model` with the `select` branch
- Update the signature and add the branch at the **top** of the precedence ladder:
  ```python
  def resolve_model(
      *,
      model: str | None,
      latest: bool,
      select: bool = False,
      config: dict[str, Any],
      runs_dir: Path,
      selector: ModelSelector | None = None,
  ) -> tuple[str, str]:
      if select:
          candidates = discover_models(runs_dir)
          if not candidates:
              raise FileNotFoundError(
                  f"No model files (.pt/.onnx) found under {runs_dir}"
              )
          chosen = select_model(candidates, selector=selector)
          if chosen is None:
              raise RuntimeError("Model selection cancelled")
          return str(chosen), f"selected ({chosen.name})"
      # ... existing model / latest / config / base ladder unchanged ...
  ```
- Update the docstring precedence line to mention fuzzy selection first.
- Rerun inline tests — green.

### 8. Wire `--select`/`--fuzzy` into `parse_args`
- Replace the standalone `--model`/`--latest` args with a mutually exclusive
  group so only one model source can be chosen at a time:
  ```python
  source = parser.add_mutually_exclusive_group()
  source.add_argument("--model", default=None, help="Explicit model weights (.pt); highest priority")
  source.add_argument("--latest", action="store_true", help="Use the newest trained best.pt under runs/")
  source.add_argument(
      "--select", "--fuzzy", dest="select", action="store_true",
      help="Interactively pick a .pt/.onnx model under runs/ via fzf",
  )
  ```
- Confirm `argparse` accepts the two long option strings as one action (`--fuzzy`
  is an alias of `--select`; both set `args.select`).

### 9. Thread `select` through `main`
- Update the `resolve_model(...)` call in `main` (`demo.py:310`) to pass
  `select=args.select` (leave `selector=None` so production uses real fzf).
- No other `main` change is needed: the returned `(model_ref, model_source)` flows
  into the banner and `InferencePipeline` exactly as today. ONNX paths load via
  `YOLO(model_path)` unchanged.

### 10. Add a mocked `main --select` test
- In `tests/test_demo.py::TestMain`, add `test_select_flag_invokes_picker`:
  - Create an images dir with one file and a `runs/` tree containing a `best.pt`.
  - `mocker.patch("screencropnet_yolo.demo._fzf_select", return_value=[<line>])`
    where `<line>` is `demo.format_model_choice(<the best.pt>)`.
  - Mock `InferencePipeline`, `resolve_device`, `build_contact_sheet`,
    `open_paths`, and `run_demo` as the existing happy-path test does.
  - Assert `rc == 0`, the contact sheet exists, and `_fzf_select` was called once.
- Optionally add `test_select_no_models_returns_nonzero`: `--select` with an empty
  `runs/` → `rc == 1` (the `FileNotFoundError` is caught by `main`'s try/except).

### 11. Update `specs/demo.md`
- **Flags table**: add a `--select` / `--fuzzy` row — "Interactively pick a
  `.pt`/`.onnx` model discovered under `runs/` using fzf. Mutually exclusive with
  `--model`/`--latest`."
- **Model resolution**: renumber precedence so fuzzy selection is source #1
  (highest), then explicit `--model`, `--latest`, config weights, base checkpoint.
  Note that selection is the only source that surfaces `.onnx` exports.
- **Reused building blocks**: add `output.human_size` (already listed) and note
  the new `pyfzf` dependency + `fzf` binary prerequisite.
- **Edge cases**: add — `fzf` not on PATH → clear error, exit non-zero; no
  `.pt`/`.onnx` under `runs/` → error naming the searched dir; user cancels the
  picker (ESC) → "Model selection cancelled", exit non-zero.
- **Testing (TDD)**: add the inline `discover_models`/`format_model_choice`/
  `resolve_model(select=...)` cases and the mocked `select_model` / `main --select`
  cases.

### 12. Validate
- Run the full gate (see Validation Commands). Fix any lint/type findings.
- Manual smoke on this machine (fzf present): run `--select`, confirm the picker
  lists `runs/` models with sizes, pick one, confirm the banner reads
  `selected (<name>)` and the contact sheet opens.

## Testing Strategy

- **Pure helpers** (`discover_models`, `format_model_choice`, `resolve_model`
  select branch) → inline `## Tests` in `demo.py`, torch-free, no `import pytest`,
  temp dirs + `os.utime` for deterministic mtime ordering (matching the existing
  `find_latest_run`/`resolve_model` inline tests).
- **Interactive selection** (`select_model`, `main --select`) → `tests/test_demo.py`
  with `pytest-mock`'s `mocker`. The fzf call is **always** injected/patched — a
  real `fzf` subprocess never runs in the suite, so tests stay deterministic and
  CI-safe (no TTY).
- **Edge cases covered**: zero candidates (`FileNotFoundError`), user cancel
  (`None` → `RuntimeError`, caught by `main` → exit 1), missing `fzf` binary
  (documented; `pyfzf` raises on `FzfPrompt()` init — surfaced through `main`'s
  try/except as a non-zero exit; assert via a `selector` that raises if you want
  explicit coverage).
- **No new visual assertions** — selection does not change montage output, so the
  existing contact-sheet tests suffice (per `visual-testing.md`, montage shape is
  already covered by `test_tile_images_shape`).

## Acceptance Criteria

- `--select` (and its `--fuzzy` alias) lists every `.pt` and `.onnx` under `runs/`
  as `name : /abs/path  [size]`, newest first, and loads the picked file.
- The run banner names the source as `selected (<filename>)`.
- `--select`, `--model`, and `--latest` are mutually exclusive (argparse errors if
  combined).
- No candidates → non-zero exit naming the searched dir; user cancel → non-zero
  exit with "Model selection cancelled"; missing `fzf` → non-zero exit, no crash.
- Importing `screencropnet_yolo.demo` does not require the `fzf` binary (pyfzf is
  imported lazily inside the selector).
- `.onnx` selections run inference unchanged via `YOLO(model_path)`.
- New logic is written test-first; `make lint`, `make check`, `make test` are clean.

## Validation Commands

Execute these to validate the task is complete:

- `uv add pyfzf` — dependency is declared and locked (run once, in step 1).
- `uv run pytest src/screencropnet_yolo/demo.py -q` — inline pure-helper tests pass.
- `uv run pytest tests/test_demo.py -q` — mocked selector/main tests pass.
- `make lint` — codespell + ruff + basedpyright clean.
- `make check` — `ty` type check clean.
- `make test` — full suite (coverage/junit) green.
- Manual (fzf present): `uv run python -m screencropnet_yolo.demo datasets/twitter_screenshots_localization_dataset/test/images --select -n 4 --no-open`
  — picker lists `runs/` models with sizes; selecting one runs the demo and the
  banner reads `Model: selected (<name>)`.

## Notes

- **New dependency**: `uv add pyfzf` (runtime). `pyfzf` is a thin wrapper that
  shells out to the `fzf` binary — `fzf` must be installed on PATH
  (`brew install fzf`); it is **not** a Python-installable dependency. Document
  this prerequisite in `specs/demo.md`.
- **Search root**: default is `runs/` (`DEFAULT_RUNS_DIR`), which covers both
  example layouts (`runs/*/best.pt` and `runs/*/train/weights/best.onnx`) via
  `rglob`. If a broader search is later wanted, add a `--select-root PATH` flag —
  out of scope for this plan; avoid scanning the whole filesystem by default.
- **Precedence choice**: fuzzy selection sits at the top of the ladder because it
  is an explicit interactive user action; the mutually exclusive argparse group
  makes the "which source" question unambiguous rather than relying on ordering.
- **Why inject the selector**: keeping `select_model` selector-injectable is what
  lets the whole feature be unit-tested without a TTY or subprocess, consistent
  with how `resolve_model` is already pure and inline-tested.
- **ONNX confirmation**: `InferencePipeline.__init__` → `self.model = YOLO(model_path)`
  (`inference.py:132`); ultralytics uses `onnxruntime` (already pinned) to run
  `.onnx`. No inference-path changes required.
