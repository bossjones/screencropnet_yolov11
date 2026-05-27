# Plan: Label Studio Annotation Pipeline for Tweet-Region Bounding Boxes

## Task Description
Set up a reproducible, `uvx`-driven Label Studio workflow that lets the user annotate Twitter screenshots with a **single class — `tweet_region`** — and exports YOLO-format labels ready for use by the YOLO26 trainer (see `02-yolo26-migration-with-tdd.md`). Reuse the existing EfficientNet-B0 single-bbox regression weights at `/Users/bossjones/dev/bossjones/pytorch-lab/screencropnet/models/` as a Label Studio ML backend that pre-predicts the outer tweet bounding box; annotators only need to confirm/adjust.

## Objective
- Stand up Label Studio (`uvx`) and a YOLO-style ML backend powered by the existing EfficientNet checkpoint.
- Provide one ergonomic command to import the 376 source screenshots (Pascal VOC CSV) into a Label Studio project with the model's predictions pre-populated as `tweet_region` rectangles.
- Export annotations as YOLO-format files in the layout YOLO26 training expects (`images/`, `labels/`, `data.yaml`).
- Do **not** modify the source dataset at `/Users/bossjones/dev/bossjones/pytorch-lab/scratch/datasets/twitter_screenshots_localization_dataset/`. Copy what's needed into this repo under `scratch/datasets/twitter_screenshots/` (already covered by `.gitignore`'s `scratch` entry).

## Problem Statement
The existing dataset is annotated in Pascal VOC pixel coordinates as a single class (`twitter`). Annotation was ad-hoc; there is no UI for verifying/correcting boxes or for adding new images. The user also wants a model-in-the-loop workflow that reuses existing trained weights to accelerate annotation.

## Solution Approach
- Use **Label Studio** (open-source UI) launched via `uvx label-studio` for the annotation surface — no global install required.
- Use the existing `ScreenCropNetV1_378_epochs.pth` (EfficientNet-B0 regressing `[xmin, ymin, xmax, ymax]`) as a **Label Studio ML backend** so each newly-imported image gets a pre-prediction the annotator only needs to nudge.
- Provide a converter script (`scripts/pascal_csv_to_ls_tasks.py`) that turns the existing Pascal VOC CSV into Label Studio JSON `tasks.json` (with pre-annotations baked in) — so the existing 376 boxes appear pre-drawn for verification.
- Provide an export step that converts Label Studio's YOLO export into the `data.yaml`-driven directory layout YOLO26 expects.

## Relevant Files
Use these files / resources to complete the task:

- `/Users/bossjones/dev/bossjones/pytorch-lab/screencropnet/arch.py` — `ObjLocModel` (EfficientNet-B0, 4 regression outputs). Import as-is into the ML backend module.
- `/Users/bossjones/dev/bossjones/pytorch-lab/screencropnet/models/ScreenCropNetV1_378_epochs.pth` — production checkpoint to copy into this repo at `scratch/checkpoints/screencropnet_efficientnet_b0_378.pth`.
- `/Users/bossjones/dev/bossjones/pytorch-lab/screencropnet/try_predict.py` — reference for preprocessing (resize to 224×224, albumentations normalize) and post-processing (raw output is pixel-space xyxy).
- `/Users/bossjones/dev/bossjones/pytorch-lab/scratch/datasets/twitter_screenshots_localization_dataset/labels_pascal_temp.csv` — source CSV (read-only); columns `filename,width,height,class,xmin,ymin,xmax,ymax`.
- `/Users/bossjones/dev/bossjones/pytorch-lab/scratch/datasets/twitter_screenshots_localization_dataset/train_images/` — 376 PNGs (read-only).
- `.gitignore` — already excludes `scratch/`; verify before adding large image dirs.

### New Files
- `tools/labeling/label_config.xml` — Label Studio labeling-interface XML (single `tweet_region` RectangleLabel).
- `tools/labeling/ml_backend/model.py` — `LabelStudioMLBase` subclass loading the EfficientNet checkpoint and returning rectangle predictions.
- `tools/labeling/ml_backend/_wsgi.py` — standard Label Studio ML backend WSGI entry.
- `tools/labeling/ml_backend/requirements.txt` — `label-studio-ml`, `torch`, `timm`, `albumentations`, `opencv-python-headless`, `Pillow`.
- `tools/labeling/ml_backend/pyproject.toml` — optional PEP 723 inline-metadata form so `uvx --from . label-studio-ml start ./ml_backend` works.
- `scripts/pascal_csv_to_ls_tasks.py` — PEP 723 inline-deps script; converts Pascal VOC CSV to Label Studio `tasks.json` with pre-annotations.
- `scripts/ls_yolo_export_to_dataset.py` — PEP 723 script; takes a Label Studio YOLO export ZIP and produces the YOLO26-ready directory under `scratch/datasets/twitter_screenshots/{train,val}/{images,labels}` plus `data.yaml` with `nc: 1, names: [tweet_region]`.
- `tools/labeling/README.md` — single-page operator runbook.

## Implementation Phases

### Phase 1: Foundation
- Copy the EfficientNet checkpoint into `scratch/checkpoints/` (gitignored).
- Copy the 376 PNGs and the CSV into `scratch/datasets/twitter_screenshots_raw/` (gitignored). Use `shutil.copy2` or `cp -R`; never edit the originals.
- Confirm `.gitignore` covers `scratch/` (it does).

### Phase 2: Core Implementation
- Write `tools/labeling/label_config.xml` with one `<RectangleLabels>` → one `<Label value="tweet_region" .../>`.
- Build the ML backend (`tools/labeling/ml_backend/model.py`) that:
  1. Loads `ObjLocModel` and the `.pth` weights at startup.
  2. On `predict(tasks)`: for each task, fetch the image, resize to 224×224, run forward, rescale predicted pixel xyxy back to the original image dims, convert to Label Studio's percent-of-image schema, return as `rectanglelabels` predictions with `model_version="screencropnet_efficientnet_b0_378"`.
- Build `scripts/pascal_csv_to_ls_tasks.py`: PEP 723 inline-deps script that reads the CSV and writes a `tasks.json` where each task has `data.image` plus `predictions[0].result` containing the existing Pascal box as a `tweet_region` rectangle (this is the "verify existing labels" path; the ML backend is for fresh images).
- Build `scripts/ls_yolo_export_to_dataset.py`: takes a Label Studio YOLO export ZIP → unpacks → splits into train/val (80/20, seed=42) → writes `data.yaml` with `nc: 1, names: [tweet_region]`.

### Phase 3: Integration & Polish
- Operator runbook (`tools/labeling/README.md`) covering: launch sequence, environment vars, troubleshooting (CORS, host.docker.internal, port 9090).
- Smoke test: run the ML backend, hit `/health` and `/predict` with a tiny mocked task, confirm JSON shape matches the LS schema.
- Verify round-trip on 3-5 sample images: import → predict → annotate → export → confirm the produced YOLO label files parse cleanly and bbox coords land within `[0, 1]`.

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

### 1. Stage the source artifacts inside this repo
- Verify `.gitignore` already lists `scratch` (it does — line in `.gitignore`).
- `mkdir -p scratch/datasets/twitter_screenshots_raw/train_images scratch/checkpoints`.
- Copy CSV: `cp /Users/bossjones/dev/bossjones/pytorch-lab/scratch/datasets/twitter_screenshots_localization_dataset/labels_pascal_temp.csv scratch/datasets/twitter_screenshots_raw/labels_pascal_temp.csv`.
- Copy images: `cp -R /Users/bossjones/dev/bossjones/pytorch-lab/scratch/datasets/twitter_screenshots_localization_dataset/train_images/ scratch/datasets/twitter_screenshots_raw/train_images/`.
- Copy weights: `cp /Users/bossjones/dev/bossjones/pytorch-lab/screencropnet/models/ScreenCropNetV1_378_epochs.pth scratch/checkpoints/screencropnet_efficientnet_b0_378.pth`.

### 2. Author the Label Studio labeling interface XML
- Create `tools/labeling/label_config.xml`:
  ```xml
  <View>
    <Image name="image" value="$image"/>
    <RectangleLabels name="label" toName="image">
      <Label value="tweet_region" background="#1da1f2"/>
    </RectangleLabels>
  </View>
  ```

### 3. Author the Pascal-VOC → Label Studio tasks converter
- Create `scripts/pascal_csv_to_ls_tasks.py` with PEP 723 inline metadata (`#!/usr/bin/env -S uv run`, deps: `pandas`, `Pillow`).
- CLI: `--csv`, `--images-root`, `--images-url-prefix` (so Label Studio can serve them), `--out tasks.json`.
- For each row: read image width/height (already in CSV), construct Label Studio task JSON:
  ```json
  {
    "data": {"image": "<prefix>/<filename>"},
    "predictions": [{
      "model_version": "pascal_csv_seed",
      "score": 1.0,
      "result": [{
        "from_name": "label",
        "to_name": "image",
        "type": "rectanglelabels",
        "value": {
          "x": (xmin / width) * 100,
          "y": (ymin / height) * 100,
          "width": ((xmax - xmin) / width) * 100,
          "height": ((ymax - ymin) / height) * 100,
          "rotation": 0,
          "rectanglelabels": ["tweet_region"]
        }
      }]
    }]
  }
  ```
- Skip rows where bbox is malformed; log skipped count.

### 4. Build the EfficientNet-B0 ML backend
- Create `tools/labeling/ml_backend/` with `model.py`, `_wsgi.py`, `requirements.txt`, and a `Dockerfile` (optional, for containerized runs).
- `model.py` defines `class TweetRegionModel(LabelStudioMLBase)`:
  - In `__init__`: load `ObjLocModel(num_classes=4)` from `scratch/checkpoints/`; set `self.model.eval()`.
  - In `predict(tasks, context)`: for each task fetch `image_url`, download, run preprocess (224×224 resize + ImageNet-style normalize), forward, rescale predicted xyxy back to original dims using original width/height, convert to percent-coords schema, return one prediction with `from_name="label"`, `to_name="image"`, `type="rectanglelabels"`, `rectanglelabels=["tweet_region"]`, `score=0.9`.
- Reuse preprocessing code patterns from `pytorch-lab/screencropnet/try_predict.py`.

### 5. Wire `uvx`-driven runbook
- Document the launch order in `tools/labeling/README.md`:
  ```bash
  # Terminal 1: ML backend
  cd tools/labeling/ml_backend
  uvx --from label-studio-ml --with torch --with timm --with albumentations \
      label-studio-ml start . --port 9090

  # Terminal 2: Label Studio UI
  uvx label-studio start --port 8080
  ```
- In the UI: create project → paste `label_config.xml` → Settings → Machine Learning → Add Backend `http://localhost:9090`.
- Import data: either upload `scratch/datasets/twitter_screenshots_raw/train_images/` directly, or import the pre-built `tasks.json` from step 3.

### 6. Annotation pass
- Annotators open each task. Label Studio shows the pre-prediction; they correct or accept.
- For tasks with no pre-prediction, the ML backend auto-predicts on open (interactive prediction setting).

### 7. Export & convert to YOLO26 dataset layout
- In Label Studio: Export → YOLO with images.
- Run `uv run scripts/ls_yolo_export_to_dataset.py --export ./ls_export.zip --out scratch/datasets/twitter_screenshots/ --val-ratio 0.2 --seed 42`.
- Script writes:
  ```
  scratch/datasets/twitter_screenshots/
  ├── data.yaml          # nc: 1, names: [tweet_region]
  ├── train/images/      # 80%
  ├── train/labels/      # YOLO txt: "0 x_c y_c w h"
  ├── val/images/        # 20%
  └── val/labels/
  ```

### 8. Validate end-to-end
- Open 5 random YOLO label files; assert format `0 <float> <float> <float> <float>` with all floats in `[0, 1]`.
- Load `data.yaml` via PyYAML and assert `nc == 1` and `names == ["tweet_region"]`.
- Run a 1-epoch dry train using YOLO26 (or current YOLO11) against this `data.yaml` to confirm the trainer loads the dataset without errors.

## Testing Strategy
- **Converter scripts**: pytest unit tests with `tmp_path` fixture. `tests/test_pascal_to_ls_tasks.py::test_row_to_task_normalizes_to_percent` — feed `(xmin=30, ymin=391, xmax=1161, ymax=752, width=1179, height=2556)`, assert `value.x ≈ 2.544`, `value.y ≈ 15.298`, `value.width ≈ 95.929`, `value.height ≈ 14.123`. `tests/test_ls_yolo_export.py::test_export_writes_data_yaml_single_class`.
- **ML backend**: pytest with `pytest-mock` `mocker` to patch torch model load; assert `.predict(tasks)` returns the expected JSON shape for a tiny mocked output `[100., 200., 500., 700.]` on a `(1080, 1920)` image.
- **Integration smoke test** (marked `@pytest.mark.integration`, skipped in CI): runs the backend against a real image and asserts a non-zero bbox is returned.

## Acceptance Criteria
- `uvx label-studio` launches the UI on port 8080 without additional installs.
- A Label Studio project with `label_config.xml` can connect to the ML backend at `http://localhost:9090`.
- Opening any task in the project shows a pre-predicted `tweet_region` rectangle.
- Importing `tasks.json` (from `pascal_csv_to_ls_tasks.py`) yields 376 tasks each with one pre-annotation.
- Exporting the project as YOLO and running `ls_yolo_export_to_dataset.py` produces a valid YOLO directory tree that YOLO26 (see `02-yolo26-migration-with-tdd.md`) accepts.
- Source dataset at `/Users/bossjones/dev/bossjones/pytorch-lab/...` is untouched (verify with `git status` over there).

## Validation Commands
- `uvx label-studio --version` — confirm uvx-bootstrap works.
- `uvx --from label-studio-ml label-studio-ml --help` — confirm ML backend CLI accessible via uvx.
- `uv run scripts/pascal_csv_to_ls_tasks.py --csv scratch/datasets/twitter_screenshots_raw/labels_pascal_temp.csv --images-root scratch/datasets/twitter_screenshots_raw/train_images --images-url-prefix /data/local-files/?d=train_images --out scratch/labeling/tasks.json` — produces a non-empty `tasks.json` with 282–376 tasks.
- `uv run pytest tests/test_pascal_to_ls_tasks.py tests/test_ls_yolo_export.py -v` — converter tests pass.
- `python -c "import yaml; d=yaml.safe_load(open('scratch/datasets/twitter_screenshots/data.yaml')); assert d['nc']==1 and d['names']==['tweet_region']"` — exported dataset matches YOLO26 expectations.

## Notes
- `uvx label-studio` is supported because Label Studio is a regular PyPI package; uvx will create an ephemeral env and run the entry point.
- The EfficientNet model outputs raw pixel-xyxy in the 224×224 space — remember to rescale to original image dims before sending to Label Studio.
- New deps (only inside `tools/labeling/ml_backend`): `label-studio-ml`, `torch`, `timm`, `albumentations`, `Pillow`. Keep them inside the ML backend's own venv (or use `uvx --with`).
- If the user later wants a multi-class taxonomy back, only `label_config.xml` and the converter's `rectanglelabels` lists need updating; the rest of the pipeline (ML backend, exporter) is class-count agnostic.
