# Plan: Migrate `screencropnet_yolov11` → `screencropnet_yolo` (default model: YOLO26), Strict TDD

## Task Description
Migrate the current package from YOLO v11 to YOLO v26, rename it to `screencropnet_yolo` (version-agnostic name; `yolo26` becomes the default model variant), and reduce class count to a single `tweet_region` class so it consumes the dataset produced by `01-label-studio-tweet-region-annotation.md`. Follow strict red-green-refactor TDD throughout, using the existing `pytest-mock` `MockerFixture` patterns.

## Objective
- `ultralytics>=8.4.52` pinned; `yolo26{n,s,m,l,x}.pt` are the new defaults.
- Package importable as `screencropnet_yolo`; old name no longer resolves.
- `config/config.yaml` defaults to `nc: 1, names: [tweet_region]`.
- All existing tests pass; new tests cover v26-specific behavior (model name strings, NMS-free inference shape, version pin).
- `make lint && make test` clean.

## Problem Statement
YOLO v11 is the prior generation; v26 (Ultralytics, GA 8.4.52) brings NMS-free end-to-end inference, DFL removal, and CPU-side speedups. The package name `screencropnet_yolov11` will become misleading as soon as we switch, so we rename it now to `screencropnet_yolo` and treat the YOLO version as a config concern, not a structural one.

## Solution Approach
Phased migration with rename **deferred to the last phase** so each prior phase's red/green signals stay clean (a rename-first approach would mass-fail every test on `ImportError`). Each phase: write failing tests for the next behavior change, make minimal source edits to pass, refactor. Use existing test-file conventions (`tests/test_<module>.py`, `pytest_mock.MockerFixture`, no `conftest.py`).

## Relevant Files
Use these files to complete the task:

- `src/screencropnet_yolov11/model.py` lines 54–65 — `MODEL_SIZES` dict hardcodes `yolo11n.pt`…`yolo11x.pt`. Becomes `yolo26{n,s,m,l,x}.pt`.
- `src/screencropnet_yolov11/model.py` lines 132, 138 — log strings mention `"YOLO11"`.
- `src/screencropnet_yolov11/config/config.yaml` — class list (currently 12 classes; collapse to one), `roboflow.format`, `wandb.experiment_name`.
- `src/screencropnet_yolov11/train.py` lines 176, 461 — log banners say `"YOLO 11"`.
- `src/screencropnet_yolov11/dataset_utils.py` line 367 — `format="yolov11"` default in `RoboflowLoader` (leave the literal; Roboflow has no `"yolov26"` export and label format is identical).
- `src/screencropnet_yolov11/inference.py` line 384, 564 — string `"YOLO 11 Twitter Detection"` and YOLO-format conversion (still valid for v26).
- `pyproject.toml` — `[project].name`, `[project.scripts]`, `[tool.hatch.build.targets.wheel].packages`, `[tool.hatch.build.targets.sdist].packages`, dependency `ultralytics>=8.3.240` → `>=8.4.52`.
- `tests/test_model.py`, `tests/test_dataset_utils.py`, `tests/test_training.py`, `tests/test_evaluation.py`, `tests/test_inference.py`, `tests/test_train.py`, `tests/test_visualization.py` — every `mocker.patch("screencropnet_yolov11.<x>")` becomes `screencropnet_yolo.<x>`; mock weight names update.
- `Makefile` — `agent-rules` regenerates `CLAUDE.md`/`AGENTS.md`.
- `.cursor/rules/general.mdc`, `.cursor/rules/python.mdc` — no edits needed.
- `src/screencropnet_yolov11/screencropnet_yolov11.py` (the script-entry stub) — gets renamed in Phase 5.

### New Files
- `tests/test_baseline.py` — pins ultralytics version and YOLO26 model name resolution.
- `tests/test_packaging.py` — pins the new package name and entry point.
- `tests/test_config.py` — pins config schema (single class, experiment name).
- `tests/test_dataset_import.py` — covers the Pascal→YOLO converter pipeline (the import-side helper used to bridge the annotation plan's output if needed without Label Studio).
- `src/screencropnet_yolov11/dataset_import.py` (renamed in Phase 5) — Pascal-VOC CSV → YOLO `.txt` converter; single-class collapse.

## Implementation Phases

### Phase 1: Foundation — version pin + dataset import helper (additive)
Bump `ultralytics`, add `dataset_import.py` for the single-class Pascal→YOLO converter. No existing behavior touched.

### Phase 2: Model swap (still as `screencropnet_yolov11`)
Update `MODEL_SIZES` and log strings to `yolo26`. Update tests in lockstep.

### Phase 3: Config to single class
Collapse `model.class_names` to `["tweet_region"]`, `model.num_classes = 1`; switch `wandb.experiment_name` to `"twitter_yolo26"`. Update banners in `train.py`.

### Phase 4: NMS-free inference verification
Pin the Results-object schema; add an integration test (skipped by default) that downloads `yolo26n.pt` and runs against a sample image.

### Phase 5: Atomic package rename `screencropnet_yolov11` → `screencropnet_yolo`
Single commit. `git mv`, batch sed across `src/`, `tests/`, `pyproject.toml`. Re-run suite.

### Phase 6: Docs regen + cleanup
`make agent-rules` to regen `CLAUDE.md`/`AGENTS.md`; update `README.md` references.

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom. Every task follows **R (write failing test) → G (minimal code) → Refactor**.

### 1. Phase 0 baseline — ultralytics version pin
- **R**: add `tests/test_baseline.py::test_ultralytics_version_at_least_8_4_52` parsing `ultralytics.__version__` and asserting `>= (8, 4, 52)`. Verify it fails against current `>=8.3.240`.
- **G**: bump `ultralytics>=8.4.52` in `pyproject.toml`; `uv sync`. Re-run; test green.
- **R**: add `tests/test_baseline.py::test_yolo26_weights_string_present_in_model_sizes` — currently expected to fail.
- **Refactor**: none yet.

### 2. Phase 1 — Pascal-VOC → YOLO single-class converter (additive)
- **R**: `tests/test_dataset_import.py::test_pascal_row_to_yolo_collapses_all_labels_to_tweet_region` — feed `(xmin=30,ymin=391,xmax=1161,ymax=752,width=1179,height=2556,label="twitter")`, expect `(0, x_c≈0.5054, y_c≈0.2235, w≈0.9593, h≈0.1412)` (6dp).
- **G**: implement `dataset_import.pascal_row_to_yolo(row, class_map={"twitter":0, "tweet_region":0})`.
- **R**: `test_pascal_row_to_yolo_validates_bounds` — bbox with `xmax<xmin` raises `ValueError`.
- **R**: `test_convert_csv_writes_one_txt_per_image` — write a tiny `tmp_path` CSV with rows for 3 distinct images; assert 3 `.txt` files appear under `labels/`.
- **R**: `test_prepare_twitter_dataset_creates_train_val_split_and_data_yaml` — given 5 dummy PNGs (`cv2.imwrite(np.zeros((10,10,3)))`), call `prepare_twitter_dataset(...)` with `val_ratio=0.2, seed=42`, assert directory tree + `data.yaml` with `nc:1, names:[tweet_region]`.
- **G**: implement using existing `DatasetSplitter` patterns and `create_dataset_yaml` (already class-name-parameterizable).
- **Refactor**: extract `TWEET_REGION_CLASS_ID = 0` module-level constant.

### 3. Phase 2 — MODEL_SIZES → yolo26
- **R**: edit `tests/test_model.py::TestModelFactory::test_model_sizes_constants` to expect `"yolo26n.pt"`…`"yolo26x.pt"`. Confirm red.
- **R**: `test_create_model_loads_yolo26_weights_by_default` — patch `screencropnet_yolov11.model.YOLO`, call `ModelFactory(ModelConfig(size="l")).create_model()`, assert `mock_yolo.assert_called_once_with("yolo26l.pt")`.
- **R**: `test_create_model_log_says_yolo26` — capture `caplog`, assert `"YOLO26"` substring present.
- **G**: update `MODEL_SIZES` dict (lines 54–65) and log f-strings (lines 132, 138) in `model.py`. Re-run; green.
- **R/G**: keep `test_create_model_accepts_custom_weights_path` green (no change).
- **Refactor**: keep `MODEL_SIZES` as the single source of truth; add a docstring noting v26 weights.

### 4. Phase 3 — Config to single class + banners
- **R**: `tests/test_config.py::test_config_yaml_has_single_class_tweet_region` — load `config/config.yaml` via PyYAML; assert `model.class_names == ["tweet_region"]` and `model.num_classes == 1`. Confirm red against the current 12-class list (lines 39–52).
- **R**: `test_config_yaml_experiment_name_is_twitter_yolo26`.
- **R**: `test_config_yaml_roboflow_format_yolov11_preserved` — pin the literal so a careless rename doesn't break Roboflow loads.
- **G**: edit `config.yaml`: collapse class list to `["tweet_region"]`, set `num_classes: 1`, set `wandb.experiment_name: "twitter_yolo26"`. Re-run; green.
- **R**: `tests/test_train.py::test_log_banner_mentions_yolo26` — patch logger, invoke the banner emitter, assert `"YOLO 26"` substring. Red against existing `"STARTING YOLO 11 TRAINING"`.
- **G**: update `train.py` lines 176, 461 and argparse `description=`. Update `inference.py` line 384 banner.
- **Refactor**: extract a module-level `BANNER = "YOLO 26 Twitter Screenshot Detection"`.

### 5. Phase 4 — NMS-free inference & Results schema
- **Inspection (read-only)**: `rg -n "len\(.*\.boxes|nms|max_det" tests/test_inference.py` to spot tests that assume specific post-NMS counts.
- **R**: `tests/test_inference.py::test_inference_no_explicit_nms_call` — assert `InferencePipeline.predict` does not call `torchvision.ops.nms` (v26 handles internally).
- **R**: `test_inference_handles_e2e_results_shape` — feed a mock `Results` object with v26's expected `.boxes.xyxy / .conf / .cls` attributes, assert downstream parsing returns the right `Detection` list shape.
- **G**: usually no source change required; if a test fails, fix the parser, not the test.
- **Integration test** (skipped by default; marker `@pytest.mark.integration`): `tests/test_inference_e2e.py::test_real_yolo26_predicts_on_sample_image` — actually downloads `yolo26n.pt`, predicts on a sample PNG, asserts non-empty `.boxes`.

### 6. Phase 5 — Atomic package rename
- **R**: `tests/test_packaging.py::test_package_importable_as_screencropnet_yolo` — `importlib.import_module("screencropnet_yolo")` succeeds; `import screencropnet_yolov11` raises `ModuleNotFoundError`. Red.
- **R**: `test_pyproject_project_name_is_screencropnet_yolo` — `tomllib.loads(...)` assert `[project].name == "screencropnet_yolo"`.
- **R**: `test_pyproject_script_entrypoint` — `[project.scripts]` declares the renamed entry point.
- **R**: `test_hatch_wheel_packages_point_to_new_src` — assert `[tool.hatch.build.targets.wheel].packages == ["src/screencropnet_yolo"]`.
- **G (single atomic batch)**:
  1. `git mv src/screencropnet_yolov11 src/screencropnet_yolo`.
  2. `git mv src/screencropnet_yolo/screencropnet_yolov11.py src/screencropnet_yolo/screencropnet_yolo.py`.
  3. Enumerate references: `rg -l "screencropnet_yolov11"`.
  4. Mechanical replace `screencropnet_yolov11` → `screencropnet_yolo` across all matches (`src/`, `tests/`, `pyproject.toml`, `.copier-answers.yml` if present, `Makefile`, `README.md`).
  5. `uv sync` to rebuild editable install metadata.
  6. Full `uv run pytest` — every test must be green.
- **Refactor**: if the script-entry module exported a stub `main()`, ensure the new `screencropnet_yolo.py` still does.

### 7. Phase 6 — Documentation regeneration
- `make agent-rules` to regenerate `CLAUDE.md` and `AGENTS.md` from `.cursor/rules/*.mdc`. Verify diff is only the auto-generated header/footer; cursor rules don't mention the package name.
- Update `README.md` references to the new package name and YOLO26 banner.
- Confirm `make` (the `default` target = `agent-rules + install + lint + test`) passes end to end.

### 8. Final validation
- `make` — full pipeline green.
- `make check` — `ty` type checker green.
- `make build` — `screencropnet_yolo-<version>-py3-none-any.whl` produced.
- Smoke training run (1 epoch, mocked dataset) — `uv run python -c "from screencropnet_yolo.model import ModelFactory, ModelConfig; ..."`.

## Testing Strategy
- **Unit tests**: every behavior change has a preceding red test. Mock `ultralytics.YOLO` with `mocker.patch("screencropnet_yolo.model.YOLO")` (post-rename) to avoid network. Use `tmp_path` for filesystem tests, `caplog` for log assertions.
- **Integration tests**: marked `@pytest.mark.integration`, deselected from default `make test`. Actually exercise YOLO26 weight download and `predict()` against a sample image.
- **Configuration tests**: PyYAML + `tomllib` assertions on `config/config.yaml` and `pyproject.toml` to catch accidental drift.
- **Regression gate**: after Phase 5 rename, full suite must remain green. Any failure post-rename is mechanical (missed module-path string) and fixable by re-running `rg "screencropnet_yolov11"` and patching the survivors.

## Acceptance Criteria
- `ultralytics>=8.4.52` pinned and installed.
- `MODEL_SIZES` returns `yolo26{n,s,m,l,x}.pt`; `ModelFactory` instantiates `YOLO("yolo26m.pt")` by default.
- `config/config.yaml` declares `num_classes: 1, class_names: ["tweet_region"], wandb.experiment_name: "twitter_yolo26"`.
- `import screencropnet_yolo` works; `import screencropnet_yolov11` raises `ModuleNotFoundError`.
- `pyproject.toml`: `[project].name == "screencropnet_yolo"`; script entry, hatch packages updated.
- `make lint && make test && make check` all green.
- Every new behavior in this plan has at least one test that was red before its implementing commit.
- No backwards-compat shims (`screencropnet_yolov11` is not aliased).

## Validation Commands
- `uv run pytest -v tests/test_baseline.py tests/test_packaging.py tests/test_config.py tests/test_dataset_import.py` — new tests pass.
- `uv run pytest -v` — full suite green.
- `uv run pytest -v -m integration` (optional, network) — YOLO26 integration test passes.
- `uv run python -c "import screencropnet_yolo; print(screencropnet_yolo.__name__)"` — prints `screencropnet_yolo`.
- `uv run python -c "import screencropnet_yolov11" 2>&1 | grep ModuleNotFoundError` — confirms old name gone.
- `uv run python -c "from screencropnet_yolo.model import MODEL_SIZES; assert all(v.startswith('yolo26') for v in MODEL_SIZES.values()), MODEL_SIZES"` — weights are v26.
- `uv run python -c "import yaml; d=yaml.safe_load(open('src/screencropnet_yolo/config/config.yaml')); assert d['model']['num_classes']==1 and d['model']['class_names']==['tweet_region']"` — config matches.
- `make lint` — codespell + ruff + basedpyright clean.
- `make test` — full pytest green with coverage.
- `make check` — `ty` clean.
- `make build` — wheel + sdist artifacts named `screencropnet_yolo-*`.

## Notes
- No new deps beyond bumping `ultralytics>=8.4.52`.
- Existing `dataset_utils.RoboflowLoader` `format="yolov11"` default is preserved on purpose — Roboflow's export format string is identical to YOLO's label format and no `"yolov26"` literal is advertised. Test pins the value to catch accidental change.
- `make agent-rules` rewrites `CLAUDE.md` and `AGENTS.md` from `.cursor/rules/*.mdc`. Do not hand-edit `CLAUDE.md`; edit the `.cursor/rules/*.mdc` source.
- Do not add `screencropnet_yolov11` re-export shims; per project conventions (CLAUDE.md), back-compat shims require explicit user request.
- All scripts that are standalone (not part of the package) should use PEP 723 inline metadata per `.claude/rules/python-scripts.md`.
- Per `.claude/rules/audit-protocol.md`, any audit/review agents called for code review should receive only file paths — no context, no hints.

## Risk Register (carry into execution)
| Risk | Mitigation |
|---|---|
| YOLO26 weight filename differs (`yolov26n.pt` vs `yolo26n.pt`) | Run one `@pytest.mark.integration` test before Phase 2 to confirm the exact string `ultralytics` resolves. |
| NMS-free changes `.boxes` schema | Phase 4 mock-schema test pins expected attrs; integration test surfaces real-world drift. |
| Mass mock-string rename in Phase 5 introduces typos | `rg --files-with-matches "screencropnet_yolov11"` → batch replace → full suite. |
| Accidental commit of `scratch/` images | `.gitignore` already excludes `scratch/`; verify before each commit. |
| `ty` type checker stricter than basedpyright on v26 typing | If `make check` reports issues only `ty` sees, add a targeted `# ty: ignore[ruleName]` per CLAUDE.md guidance. |
