# Tweet-region annotation with Label Studio

Annotate Twitter screenshots with a single `tweet_region` class and export
YOLO-format labels for the YOLO26 trainer. An EfficientNet-B0 ML backend
pre-predicts the outer tweet box so annotators only confirm/adjust.

All paths below are relative to the repo root. Everything under `scratch/` is
gitignored.

## 0. Stage source artifacts (once)

```bash
mkdir -p scratch/datasets/twitter_screenshots_raw scratch/checkpoints scratch/labeling
SRC=/Users/bossjones/dev/bossjones/pytorch-lab
cp "$SRC/scratch/datasets/twitter_screenshots_localization_dataset/labels_pascal_temp.csv" \
   scratch/datasets/twitter_screenshots_raw/labels_pascal_temp.csv
cp -R "$SRC/scratch/datasets/twitter_screenshots_localization_dataset/train_images" \
   scratch/datasets/twitter_screenshots_raw/train_images
cp "$SRC/screencropnet/models/ScreenCropNetV1_378_epochs.pth" \
   scratch/checkpoints/screencropnet_efficientnet_b0_378.pth
```

The source dataset is read-only — never edit the originals.

## 1. Pre-build tasks from the existing labels (optional)

Turn the 341 already-labelled rows into a `tasks.json` with the boxes pre-drawn
for verification:

```bash
uv run scripts/pascal_csv_to_ls_tasks.py \
  --csv scratch/datasets/twitter_screenshots_raw/labels_pascal_temp.csv \
  --images-root scratch/datasets/twitter_screenshots_raw/train_images \
  --images-url-prefix "/data/local-files/?d=train_images" \
  --out scratch/labeling/tasks.json
```

## 2. Launch the ML backend (terminal 1)

```bash
cd tools/labeling/ml_backend
uvx --from label-studio-ml --with torch --with timm --with albumentations \
    --with opencv-python-headless label-studio-ml start . --port 9090
```

The backend reads the checkpoint from
`scratch/checkpoints/screencropnet_efficientnet_b0_378.pth` by default; override
with `CHECKPOINT_PATH`. Device is auto-selected (cuda > mps > cpu); override with
`DEVICE`.

### Or run it in Docker

A `docker-compose.yml` lives next to the `Dockerfile`. It maps port 9090 and
mounts repo-root `scratch/checkpoints/` read-only into the container, so stage
the checkpoint (step 0) first. From the repo root:

```bash
make ml-backend-build   # build the image
make ml-backend-up-d    # start detached (daemonized)
make ml-backend-down    # stop and remove
make ml-backend-up      # or run in the foreground
```

From inside `tools/labeling/ml_backend/` the same targets are `build`, `up-d`,
`down`, and `up`. The container forces `DEVICE=cpu` (the `python:3.11-slim` base
has no CUDA/MPS); override via the `DEVICE` env var. `CHECKPOINT_PATH` is already
set to the mounted checkpoint. Confirm it's up with
`curl -s http://localhost:9090/health`.

## 3. Launch Label Studio (terminal 2)

```bash
# Allow Label Studio to serve images straight from disk.
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$PWD/scratch/datasets/twitter_screenshots_raw"
uvx label-studio start --port 8080
```

In the UI:

1. Create a project; paste `tools/labeling/label_config.xml` into
   **Settings → Labeling Interface → Code**.
2. **Settings → Cloud Storage → Add Local Storage**, point it at
   `scratch/datasets/twitter_screenshots_raw/train_images`, enable "treat every
   bucket object as a source file", and sync.
   (Or import `scratch/labeling/tasks.json` directly to get the pre-drawn boxes.)
3. **Settings → Machine Learning → Add Model**: `http://localhost:9090`. Enable
   "Use for interactive preannotations" so fresh images auto-predict on open.

## 4. Annotate

Open each task. Accept or nudge the pre-drawn `tweet_region` rectangle. Tasks
with no seed annotation get an ML-backend prediction on open.

## 5. Export to the YOLO26 dataset layout

In Label Studio: **Export → YOLO with images** (downloads a ZIP). Then:

```bash
uv run scripts/ls_yolo_export_to_dataset.py \
  --export ./ls_export.zip \
  --out scratch/datasets/twitter_screenshots/ \
  --val-ratio 0.2 --seed 42
```

Produces:

```
scratch/datasets/twitter_screenshots/
├── data.yaml          # nc: 1, names: [tweet_region]
├── train/images/  train/labels/
└── val/images/    val/labels/
```

Point the YOLO26 trainer at `scratch/datasets/twitter_screenshots/data.yaml`.

## Troubleshooting

- **CORS / backend unreachable**: confirm the backend health check at
  `http://localhost:9090/health` returns `{"status": "UP"}`. If Label Studio
  runs in Docker, use `http://host.docker.internal:9090` instead of `localhost`.
- **Images don't load**: local file serving must be enabled
  (`LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true`) and the document root must
  contain the image dir referenced by `--images-url-prefix`.
- **Port already in use**: change `--port` (backend default 9090, UI default
  8080) and update the ML model URL in project settings.
- **Wrong-looking boxes**: the checkpoint expects a 224×224 ImageNet-normalized
  input (`MODEL_INPUT_SIZE`); verify the checkpoint matches `arch.ObjLocModel`.
