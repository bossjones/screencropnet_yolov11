# Plan: Fix all `make check` (ty) diagnostics

## Task Description
`make check` runs `uv run ty check` (Astral's `ty` type checker, currently v0.0.6) as a second, faster type checker alongside `make lint`'s `basedpyright`. It has never been made clean: it currently reports **45 diagnostics** across the repo. Fix all of them — either with real code fixes or, where the cause is a documented `ty` limitation (stub imprecision, invariant generics, subscript-narrowing gaps), a scoped inline suppression matching the project's existing `# pyright: ignore[rule]` convention.

## Objective
`uv run ty check` (and therefore `make check`) exits 0 with zero diagnostics, without weakening `basedpyright`/`ruff` coverage or changing runtime behavior.

## Problem Statement
Investigation (`uv run ty check`, then re-run scoped with `-c 'src.include=["src","tests","devtools"]'` to test hypotheses) shows the 45 diagnostics fall into two very different buckets:

1. **21 diagnostics are false-positive noise from scope, not real bugs.** Unlike `basedpyright` (configured with `include = ["src", "tests", "devtools"]` in `pyproject.toml:184`), `ty` has no `[tool.ty]` section at all, so it defaults to scanning the **entire project root** — including `.claude/hooks/**` (Claude Code hook scripts with their own PEP 723 inline deps like `anthropic`, `openai`, `elevenlabs`, `pyttsx3` — see [python-scripts.md](.claude/rules/python-scripts.md)), `scripts/**` (one-off tooling scripts, not in `basedpyright`'s `include` either), and `tools/labeling/ml_backend/**` (a separate Label Studio ML backend meant to run in its own environment with `label_studio_ml`/`timm`, no local `pyproject.toml`/venv of its own). None of these are installed in this project's `.venv`, so `ty` reports `unresolved-import` (17x), plus two knock-on diagnostics in `scripts/` files. Confirmed by testing `uv run ty check -c 'src.include=["src","tests","devtools"]'`: diagnostic count drops from 45 → 24 with those 21 gone and nothing else changing.

2. **24 diagnostics are inside `src/` and `tests/`** — the code `ty` is actually meant to check. Per [typing-faq](https://docs.astral.sh/ty/reference/typing-faq/) and [rules reference](https://docs.astral.sh/ty/reference/rules/), most of these trace back to two `ty`-specific behaviors that diverge from `basedpyright`:
   - **Invariant generics + imprecise third-party stubs**: `ty` treats `numpy.ndarray`'s generic parameters as invariant (no covariance for mutable containers, confirmed in the typing FAQ). `cv2`'s bundled stubs return `ndarray[Any, dtype[integer[Any] | floating[Any]]]` from image ops (`cvtColor`, `resize`, `copyMakeBorder`, `fastNlMeansDenoisingColored`, `VideoCapture.read`) instead of preserving `uint8`, so assigning/returning/passing the result where our code declares `npt.NDArray[np.uint8]` fails even though the runtime dtype is always `uint8`. This is exactly the class of stub-precision gap the ["coming from pyright"](https://docs.astral.sh/ty/coming-from-mypy-or-pyright/) doc flags as a source of new diagnostics when migrating. 8 of our diagnostics are this.
   - **Subscript-expression narrowing gap**: `ty`'s `isinstance`/`hasattr` narrowing (["reachability based on types"](https://docs.astral.sh/ty/features/type-system/#reachability-based-on-types)) narrows bound names/attributes reliably, but does **not** retain narrowing across repeated evaluations of the same subscript expression (`batch[j]` evaluated fresh in each branch). 3 diagnostics in `inference.py` are this — and it has a genuine code fix (bind to a local variable once), not a suppression.
   - `hasattr`-narrowed callables on untyped third-party attributes (`ultralytics.YOLO.model`) resolve to a non-callable synthetic protocol type — 6 `call-non-callable` diagnostics, all already carrying `basedpyright` ignores for the same underlying untyped-library gap.
   - 4 diagnostics are genuine, fixable type errors (a numpy-scalar-vs-`float` return, and 3 test fixture calls that construct numpy-typed tuples where a plain-`float` tuple is declared).

## Solution Approach
Two-pronged, matching the two buckets above:

1. **Scope `ty` to the same file set `basedpyright` already checks.** Add a `[tool.ty.src]` section to `pyproject.toml` mirroring `[tool.basedpyright]`'s `include`. Verified via `uv run ty check -c 'src.include=["src","tests","devtools"]'` that this key is accepted and eliminates exactly the 21 out-of-scope diagnostics with no side effects.
2. **Fix or suppress the remaining 24**, file by file:
   - Where `ty` is catching a real gap in our code (numpy scalar leaking into a `float`-typed return; subscript narrowing that a bound local fixes) — fix the code.
   - Where the cause is 100%-attributable to third-party stub imprecision or an untyped library (`cv2`, `ultralytics`) that `basedpyright` already carries a `# pyright: ignore[...]` for — add a co-located `# ty: ignore[<rule>]` per [suppression](https://docs.astral.sh/ty/suppression/) docs (ty uses its own `# ty: ignore[rule]` syntax; PEP 484 `# type: ignore[ty:rule]` also works but the codebase's existing convention is bare `# pyright: ignore[...]`, so mirror it 1:1 with a second comment rather than switching styles).
   - One legitimate optional dependency (`roboflow`, imported lazily inside `try/except ImportError`, matching the `pyfzf` lazy-import pattern in `demo.py`) gets the same suppression treatment — it's intentionally not a project dependency.

No `[tool.ty.rules]` blanket rule-level severity downgrades are used — every suppression is a targeted inline comment on the exact line, per the linting bar in `CLAUDE.md` ("Suppress ... only with a specific rule ... and only when the warning is genuinely not a real problem").

## Relevant Files
- [pyproject.toml](pyproject.toml) — add `[tool.ty.src]` scoping section near the existing `[tool.basedpyright]` block (`pyproject.toml:176-205`).
- [src/screencropnet_yolo/dataset_utils.py](src/screencropnet_yolo/dataset_utils.py) — `roboflow` lazy import (line 401); `cv2` dtype-imprecision suppressions in `preprocess_screenshot`/`_reduce_compression_artifacts`/`_normalize_colors`/`letterbox` (lines 447, 456, 460, 474, 525, 529).
- [src/screencropnet_yolo/inference.py](src/screencropnet_yolo/inference.py) — subscript-narrowing fix in `predict_batch` (lines ~270-282); `cv2.VideoWriter_fourcc` stub-gap suppression (line 359); `cv2` dtype suppressions on `predict_image`/`draw_detections` calls in `predict_video` (lines 371, 376).
- [src/screencropnet_yolo/model.py](src/screencropnet_yolo/model.py) — `_setup_device` return-type fix via `gpu_ids` narrowing (line 130); `get_model_info` hasattr-narrowed-callable suppression (lines 488-489).
- [src/screencropnet_yolo/evaluation.py](src/screencropnet_yolo/evaluation.py) — `find_optimal_confidence` numpy-scalar return fix (line 516); `benchmark_model` `model.model(...)` call suppression (lines 556, 566).
- [tests/test_inference.py](tests/test_inference.py) — 3 `create_detection(...)` call sites intentionally passing numpy-float32-derived tuples (lines 656-657, 672) get targeted suppressions (these tests exist specifically to guard numpy-type JSON serialization, so the argument types must stay numpy-derived at runtime — casting to `float()` in the test would silently defeat the regression test).

## Step by Step Tasks

### 1. Scope `ty` to `src`, `tests`, `devtools`
- Add to `pyproject.toml` (near `[tool.basedpyright]`):
  ```toml
  [tool.ty.src]
  include = ["src", "tests", "devtools"]
  ```
- Re-run `uv run ty check`; confirm diagnostic count drops from 45 to 24 and the only remaining diagnostics are the ones enumerated below (no new ones appear).

### 2. Suppress the lazy optional `roboflow` import
- In `dataset_utils.py` (`RoboflowLoader.download`, line ~401), extend the existing comment:
  `from roboflow import Roboflow  # pyright: ignore[reportMissingImports]  # ty: ignore[unresolved-import]`

### 3. Fix the `inference.py` subscript-narrowing gap in `predict_batch`
- Rebind `batch[j]` to a local (`item = batch[j]`) once per loop iteration, then narrow on `item` instead of re-evaluating `batch[j]` inside each `isinstance` check. This resolves all 3 diagnostics on these lines as a genuine fix (`ty` narrows bound locals reliably) and should let the existing `# pyright: ignore[...]` comments on `img_arr = batch[j]` be dropped too — verify with `basedpyright` in the validation step.

### 4. Suppress remaining `cv2`/numpy stub-imprecision diagnostics
Add a co-located `# ty: ignore[<rule>]` next to each existing `# pyright: ignore[...]`:
- `dataset_utils.py:447` (`cvtColor` BGR2RGB) — `invalid-assignment`
- `dataset_utils.py:456` (`cvtColor` RGB2BGR) — `invalid-assignment`
- `dataset_utils.py:460` (`fastNlMeansDenoisingColored` return) — `invalid-return-type`
- `dataset_utils.py:474` (`cvtColor` LAB2RGB return) — `invalid-return-type`
- `dataset_utils.py:525` (`cv2.resize`) — `invalid-assignment`
- `dataset_utils.py:529` (`cv2.copyMakeBorder`) — `invalid-assignment`
- `inference.py:359` (`cv2.VideoWriter_fourcc`, real attribute missing from `ty`'s bundled stub) — `unresolved-attribute`
- `inference.py:371` (`self.predict_image(frame, ...)` in `predict_video`) — `invalid-argument-type`
- `inference.py:376` (`self.draw_detections(frame, ...)` in `predict_video`) — `invalid-argument-type`

### 5. Fix `model.py::_setup_device` return-type mismatch
- The `None` branch is real: `ModelConfig.gpu_ids: list[int] | None` (default `None`, filled to `[0, 1]` in `__post_init__`), so `device = self.config.gpu_ids` in the multi-GPU branch is statically `list[int] | None` even though the invariant (never `None` post-construction) always holds at runtime.
- Add `assert self.config.gpu_ids is not None` immediately before that assignment (documents the `__post_init__` invariant, zero behavior change).
- Annotate the local up front — `device: str | list[int] | int` at first assignment — so each branch is checked against the exact declared union individually rather than producing one combined mismatch at the `return`.
- Re-run `ty` on this function; if a residual mismatch remains (e.g. the `str & ~Literal["auto"]` narrowing artifact), add a scoped `# ty: ignore[invalid-return-type]` only if it's a genuine `ty`-only artifact — confirm with `basedpyright` that it doesn't need the same treatment before assuming so.

### 6. Suppress `model.py::get_model_info` hasattr-narrowed-callable diagnostics
- Lines 488-489 (`model.model.parameters()`), already carrying `# pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]`. Add `# ty: ignore[call-non-callable]` alongside — this is `ty`'s `hasattr` narrowing producing a non-callable synthetic protocol type for an attribute on the untyped `ultralytics.YOLO` class, not a real bug (`basedpyright`'s `allowedUntypedLibraries` config already treats `ultralytics` leniently for the same reason).

### 7. Suppress `evaluation.py::benchmark_model` `model.model(...)` call diagnostics
- Lines 556 and 566 (already `# pyright: ignore[reportCallIssue, reportOptionalCall]`). Add `# ty: ignore[call-non-callable]` — `ultralytics.YOLO.model`'s installed (partially-typed) attribute type includes `str | None` in addition to the loaded `nn.Module`, which `ty` picks up directly from the library source since there's no first-party stub to smooth it over.

### 8. Fix `evaluation.py::find_optimal_confidence` return-type mismatch (line 516)
- `best_conf` originates from `np.linspace(...)[0]` (a numpy scalar, not a `float` subtype in `ty`'s view) and `best_value` can be `Literal[0]` or an untyped ultralytics metric value. Declared return is `tuple[float, float]`.
- Fix: `return float(best_conf), float(best_value)` — genuine correctness improvement (callers get plain Python floats, not numpy scalars), no suppression needed.

### 9. Suppress the 3 numpy-tuple `create_detection` test calls
- `tests/test_inference.py:656-657` and `:672` intentionally pass `tuple(np.array([...], dtype=np.float32))` into a `tuple[float, float, float, float]`-typed parameter, specifically to regression-test that `ResultExporter.to_json`/`to_coco` can serialize real numpy-float32-derived detections (see the tests' own docstrings). Do **not** cast to `float()` here — that would defeat the test. Add `# ty: ignore[invalid-argument-type]` on each of the 3 call sites instead.

### 10. Full validation pass
- Run `make check` (must show `Found 0 diagnostics` / exit 0).
- Run `make lint` (basedpyright + ruff) to confirm none of the edits (steps 3, 5, 8) introduced new `basedpyright` warnings, and that dropped `# pyright: ignore` comments (step 3) don't trigger `reportUnnecessaryTypeIgnoreComment`-style regressions (already disabled per `pyproject.toml:189`, but confirm no other new warning appears).
- Run `make test` to confirm the `inference.py` narrowing refactor (step 3) and `evaluation.py` float-cast (step 8) don't change behavior.

## Acceptance Criteria
- `uv run ty check` reports `Found 0 diagnostics` and exits 0.
- `make lint` and `make test` remain clean (no new failures introduced by the fixes).
- No blanket `[tool.ty.rules]` severity downgrades — every suppression is an inline, rule-scoped `# ty: ignore[...]` comment co-located with the code it covers.
- `.claude/hooks/**`, `scripts/**`, `tools/labeling/**` are excluded from `ty`'s scan (matching `basedpyright`'s existing `include`), not silenced via per-line suppressions.
- The 4 genuine bugs (subscript-narrowing in `predict_batch`, `_setup_device`'s `gpu_ids` invariant, `find_optimal_confidence`'s numpy-scalar return) are fixed in code, not suppressed.

## Validation Commands
- `uv run ty check` — must print `All checks passed!` / 0 diagnostics.
- `make check` — same, via the Makefile target.
- `make lint` — confirm `basedpyright` still passes with no new diagnostics.
- `make test` — confirm no behavior regressions from the `inference.py`/`evaluation.py` code changes.

## Notes
- `ty` v0.0.6 is pre-1.0 and under active development; some of these diagnostics (particularly the `cv2`/numpy stub-precision ones and the `hasattr`-narrowed-callable gap) may simply disappear in a future `ty` release as its bundled/vendored third-party stubs and narrowing improve. When upgrading `ty` in the future, it's worth re-running `uv run ty check` with the suppression comments temporarily removed to see which are still needed (per the [suppression](https://docs.astral.sh/ty/suppression/) docs, `ty` has a separate `unused-ignore-comment` rule that would flag stale ones — but note it's the one rule that a *bare* `# ty: ignore` can't suppress, so don't reach for a bare ignore anywhere in this plan).
- No new dependencies needed.
