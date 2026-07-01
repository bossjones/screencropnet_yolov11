# Plan: Fix training pipeline — resolve `device="auto"` in evaluation (and harden post-training stages)

## Task Description

`make train` previously crashed in three places that have already been fixed on
`feature-retrain`:
1. `'dict' object has no attribute 'box'` — callbacks read `trainer.metrics` as a dict.
2. `'DetectionTrainer' object has no attribute 'save'` — `CheckpointCallback` copies
   `trainer.last` instead.
3. `FileNotFoundError: .../train/weights/last.pt` — the trained-weights path is now read
   from `model.trainer.best/last` and the run dir is made absolute.

Training now runs to completion and the pipeline advances into evaluation, where it hits the
**next** latent bug:

```
ValueError: Invalid CUDA 'device=auto' requested. Use 'device=cpu' ...
  evaluation.py:159  self.model.val(..., device=self.device)
```

This spec fixes that bug and verifies the remaining post-training stages (export,
visualization) so a default `make train` completes end-to-end.

## Objective

A short `make train` run proceeds cleanly through **train → evaluate → export → visualize**
with no exceptions, writing `evaluation_results.json` and the `visualizations/` artifacts.

## Problem Statement

`config["device"]["type"]` is `"auto"`. The **training** path resolves `"auto"` to a concrete
device (`mps`/`cuda:0`/`cpu`) via `ModelConfig._setup_device()` (`src/screencropnet_yolo/model.py:79-113`).
The **evaluation** path does not: `evaluate_model()` (`src/screencropnet_yolo/train.py:262-267`)
passes the raw `"auto"` into `Evaluator`, which forwards it to `model.val(device="auto")`
(`src/screencropnet_yolo/evaluation.py:159`). Ultralytics' `val()`/`predict()` reject `"auto"`
(only `train()` accepts it), so evaluation raises `ValueError`.

The device-resolution logic exists but is trapped inside `ModelConfig` and not reused outside
training. Root fix: extract a small reusable resolver and apply it wherever a concrete device
is required.

## Solution Approach

Add a module-level `resolve_device()` in `model.py` (mockable via the module's `torch`
reference, matching the existing `tests/test_model.py` mocking style) and resolve `"auto"` at
the single safe choke point — `Evaluator.__init__` — so any caller passing `"auto"` is handled.
Leave `ModelConfig._setup_device()` (and its passing tests) untouched; optionally have it
delegate the simple branch to the new helper without changing its multi-GPU behavior.

Export and ablation need no device fix: `ModelExporter.export` wraps each format in
`try/except` and continues (`model.py:354-355`), and `run_ablation_study` builds a
`ModelConfig`, which resolves `"auto"` itself.

## Relevant Files

- `src/screencropnet_yolo/model.py` — add `resolve_device()`; `ModelConfig._setup_device()`
  (`:79-113`) is the existing logic to mirror/reuse.
- `src/screencropnet_yolo/evaluation.py` — `Evaluator.__init__` (`:111-124`) resolves
  `device`; `evaluate()` (`:152-163`) forwards it to `model.val`.
- `src/screencropnet_yolo/train.py` — `evaluate_model()` (`:249-283`) constructs the
  `Evaluator`; no change required once `Evaluator` self-resolves, but verify.
- `tests/test_model.py` — has `mock_torch_cuda` fixture (`:24+`) to drive device branches.
- `tests/test_evaluation.py` — existing `Evaluator` tests to extend.
- `src/screencropnet_yolo/config/config.yaml` — `device.type: "auto"` (`:77-79`),
  `export.formats` (`:148-156`).

## Step by Step Tasks

IMPORTANT: Execute every step in order, top to bottom. Use TDD — write the failing test
first, watch it fail, then implement (per `superpowers:test-driven-development`).

### 1. Add `resolve_device()` to `model.py` (RED → GREEN)
- In `tests/test_model.py`, add tests using the existing `mock_torch_cuda` fixture:
  - `resolve_device("auto")` → `"mps"` when only MPS is available; → `0` when CUDA is
    available; → `"cpu"` when neither.
  - `resolve_device("cpu")`, `resolve_device("mps")`, `resolve_device(0)` → returned
    unchanged (passthrough; never returns `"auto"`).
- Implement in `model.py` (module-level, so `screencropnet_yolo.model.torch` patches apply):
  ```python
  def resolve_device(device: str | int | list[int]) -> str | int | list[int]:
      """Resolve an 'auto' device spec to a concrete torch/ultralytics device.

      ultralytics' val()/predict()/export() reject device='auto' (only train()
      accepts it), so callers outside training must resolve it first.
      """
      if device != "auto":
          return device
      if torch.cuda.is_available():
          return 0
      if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
          return "mps"
      return "cpu"
  ```

### 2. Resolve device inside `Evaluator` (RED → GREEN)
- In `tests/test_evaluation.py`, add a test: `Evaluator(model, data_yaml, class_names,
  device="auto")` sets `evaluator.device` to a concrete value (not `"auto"`), and `evaluate()`
  calls `model.val(...)` with that resolved device. Mock `model.val` to return a stub metrics
  object with `.box.map50/.map/.mp/.mr` and `.speed`; patch torch so resolution is
  deterministic (e.g. CPU-only → `"cpu"`).
- In `evaluation.py`, import and apply the resolver:
  ```python
  from screencropnet_yolo.model import resolve_device
  ...
  self.device = resolve_device(device)
  ```
  (Confirm no import cycle: `model.py` does not import `evaluation.py`.)

### 3. Optionally delegate `ModelConfig._setup_device()` to the helper
- Only if it keeps `tests/test_model.py` green: have the non-multi-GPU `"auto"` branch return
  `resolve_device("auto")`, preserving multi-GPU (`gpu_ids`) and logging. If this perturbs
  existing tests, leave `_setup_device()` as-is — the standalone helper is the source of truth
  for non-training callers.

### 4. Run the unit suites
- `uv run pytest tests/test_model.py tests/test_evaluation.py tests/test_train.py
  tests/test_training.py -q` — all green.

### 5. Lint and full test suite
- `make lint` (0 errors) and `make test` (all pass).

### 6. End-to-end smoke verification
- Run a short pipeline and confirm it advances through every stage:
  ```bash
  uv run python -m screencropnet_yolo.train --epochs 2 --model-size n --batch 4 \
      --imgsz 320 --output ./runs/fix_training_smoke
  ```
- Confirm: no `ValueError`/`FileNotFoundError`; logs show `EVALUATING MODEL` →
  `EXPORTING MODEL` → `Creating visualizations...` → process exits 0.
- Confirm artifacts exist: `runs/fix_training_smoke/evaluation_results.json` and
  `runs/fix_training_smoke/visualizations/training_curves.png`.
- If a further latent bug surfaces in export/visualization, fix it with the same TDD loop and
  note it. Known non-blockers: ONNX export logs `Failed to export to onnx` when `onnx` is
  absent and continues — acceptable (see Notes for enabling it).

## Testing Strategy

- Unit: `resolve_device` across all three backends + passthrough (mocked torch); `Evaluator`
  resolves `"auto"` and forwards the concrete device to `model.val`.
- Regression: existing `tests/test_model.py` device tests and `tests/test_evaluation.py` stay
  green (no behavior change for explicit devices).
- Integration/E2E: the smoke run in step 6 is the real proof the full chain completes.

## Acceptance Criteria

- `resolve_device("auto")` returns a concrete device; never returns `"auto"`; explicit values
  pass through unchanged.
- `Evaluator` constructed with `device="auto"` calls `model.val` with a concrete device.
- `make lint` and `make test` pass.
- The step-6 smoke run completes through evaluate → export → visualize with exit code 0 and
  writes `evaluation_results.json` and `visualizations/`.

## Validation Commands

- `uv run pytest tests/test_model.py tests/test_evaluation.py -q` — device resolver + evaluator.
- `make lint` — codespell + ruff + basedpyright, 0 errors.
- `make test` — full suite green.
- `uv run python -m screencropnet_yolo.train --epochs 2 --model-size n --batch 4 --imgsz 320 --output ./runs/fix_training_smoke` — end-to-end; must exit 0.
- `ls runs/fix_training_smoke/evaluation_results.json runs/fix_training_smoke/visualizations/` — artifacts present.

## Notes

- **Import direction:** `evaluation.py` → `model.py` is safe (no cycle).
- **Optional ONNX export (not a blocker):** default config requests `onnx`, but `onnx` is not
  installed, so export logs a failure and skips it. To enable real ONNX export, run
  `uv add onnx onnxslim` (requires user sign-off on new deps) — out of scope for the blocking
  fix.
- **No backward-compat shims** (project rule); `Evaluator`'s public signature is unchanged.
- The three earlier fixes (callback metrics dict, `CheckpointCallback` copy, weights-path
  capture) are already on `feature-retrain` and are prerequisites for reaching this bug.
