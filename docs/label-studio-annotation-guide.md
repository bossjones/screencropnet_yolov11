# Label Studio annotation guide

Annotate Twitter screenshots with `tweet_region` bounding boxes and export a
YOLO26-ready dataset. By the end you will have a `data.yaml` + train/val/test
split ready to hand to the trainer.

> **You may not need this.** A ready-to-train dataset already exists at
> `datasets/twitter_screenshots_localization_dataset/` (272 train / 68 val / 1
> test, all `tweet_region`), and it is the default `dataset.path` in
> `src/screencropnet_yolo/config/config.yaml`. To just train, run
> `uv run python -m screencropnet_yolo.train` (or `make train`) — no annotation
> needed. Follow this guide only to **re-annotate** those images or **add new
> ones**; the export step writes back into that same canonical folder.

> **Adding new images?** See [Adding images to Label Studio](adding-images-to-label-studio.md)
> for the rename-and-copy workflow, file-naming conventions, and the
> `scripts/add_images_to_labeling.py` utility that automates index assignment.

## How it works

An EfficientNet-B0 ML backend pre-predicts the outer tweet card boundary for
each image. You only confirm or nudge the box — far faster than drawing from
scratch. Fresh images with no seed annotation get a live prediction the moment
you open the task.

```text
pytorch-lab images + CSV
     ↓ scripts/pascal_csv_to_ls_tasks.py
tasks.json  (boxes pre-drawn)
     ↓ import into Label Studio (port 8080)
annotated tasks  ←  EfficientNet-B0 ML backend (port 9090)
     ↓ Export → YOLO with images
ls_export.zip
     ↓ scripts/ls_yolo_export_to_dataset.py
datasets/twitter_screenshots_localization_dataset/  (data.yaml + train/ val/ test/)
     ↓
uv run python -m screencropnet_yolo.train
```

## Prerequisites

- `uv` — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Read access to `pytorch-lab` at `/Users/bossjones/dev/bossjones/pytorch-lab`
- ~2 GB free disk (images + checkpoint)
- Docker Desktop (optional — only for the Docker backend path)

---

## Step 1 — Stage source artifacts (once)

Everything under `scratch/` is gitignored. Pull the checkpoint and images from
the read-only `pytorch-lab` source:

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

Equivalent: `make labeling-stage` (override the source with
`make labeling-stage PYTORCH_LAB=/path/to/pytorch-lab`).

Never edit files under `scratch/datasets/twitter_screenshots_raw/` — treat them
as read-only originals.

---

## Step 2 — Pre-build tasks from existing labels

If you have an existing Pascal-VOC CSV (341 labelled rows from `pytorch-lab`),
turn it into a Label Studio `tasks.json` with boxes pre-drawn for verification:

```bash
uv run scripts/pascal_csv_to_ls_tasks.py \
  --csv scratch/datasets/twitter_screenshots_raw/labels_pascal_temp.csv \
  --images-root scratch/datasets/twitter_screenshots_raw/train_images \
  --images-url-prefix "/data/local-files/?d=train_images" \
  --out scratch/labeling/tasks.json
```

Equivalent: `make labeling-tasks`.

`scripts/pascal_csv_to_ls_tasks.py` is a PEP 723 script (stdlib-only, no install
needed). It writes a JSON array of Label Studio tasks, each with a
`rectanglelabels` pre-annotation for `tweet_region`.

Skip this step if you are annotating brand-new images with no prior labels — the
ML backend will predict on the fly.

---

## Step 3 — Launch the ML backend

Open a dedicated terminal and leave it running while you annotate.

### Option A — Native (recommended on Apple Silicon)

Runs with MPS acceleration; auto-selects cuda > mps > cpu:

```bash
make ml-backend
```

Or directly from the backend directory:

```bash
cd tools/labeling/ml_backend
uvx --python 3.11 \
    --from "git+https://github.com/HumanSignal/label-studio-ml-backend.git" \
    --with torch --with timm --with albumentations \
    --with opencv-python-headless --with redis --with rq \
    label-studio-ml start . --port 9090
```

### Option B — Docker (CPU-only on macOS)

Docker Desktop on macOS has no Metal/MPS passthrough, so this is CPU-only.
Stage the checkpoint first (Step 1), then:

```bash
make ml-backend-build   # build the image once
make ml-backend-up-d    # start detached
make ml-backend-down    # stop when done
```

### Confirm the backend is up

```bash
curl -s http://localhost:9090/health
# → {"status": "UP"}
```

**Environment overrides** (native or Docker):

| Variable | Default | Notes |
|----------|---------|-------|
| `CHECKPOINT_PATH` | `scratch/checkpoints/screencropnet_efficientnet_b0_378.pth` | Path to `.pth` weights |
| `DEVICE` | auto (cuda > mps > cpu) | Override with `export DEVICE=cpu` |
| `MODEL_INPUT_SIZE` | `224` | Square input edge; must match checkpoint |

---

## Step 4 — Launch Label Studio

Open a second terminal. The `label-studio-local` make target sets the required
environment variables and launches on port 8080:

```bash
make label-studio-local
```

Equivalent manual launch:

```bash
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$PWD/scratch/datasets/twitter_screenshots_raw"
uvx label-studio start --port 8080
```

**First run only:** Label Studio asks you to create a local account (email +
password). These credentials are stored locally — no internet connection is
required and nothing is sent upstream.

Open <http://localhost:8080> in your browser.

To use the scripted setup in the next step, copy a token from **Account &
Settings → Access Token** and export it:

```bash
export LABEL_STUDIO_API_KEY=<token from Account & Settings → Access Token>
```

---

## Step 5 — Create and configure your project

### Automated (recommended)

With Label Studio running (Step 4), the ML backend running (Step 3), and
`LABEL_STUDIO_API_KEY` exported, configure everything in one command:

```bash
make labeling-setup-project
```

This wraps `scripts/setup_ls_project.py`, which:

- creates (or reuses) a project titled `screencropnet`,
- applies the `tweet_region` labeling interface,
- imports `scratch/labeling/tasks.json` (the pre-drawn boxes from Step 2), and
- connects the ML backend at `http://localhost:9090`.

It is safe to re-run: an existing `screencropnet` project is reused, tasks are
not re-imported (pass `--force-import` to override), and the ML backend is not
added twice. Useful flags: `--title`, `--ml-backend-url`, `--no-ml-backend`,
`--no-reuse`, `--force-import`. This also performs Step 6, so skip Step 6 if you
used it.

> The script runs via `uv run` and pulls `label-studio-sdk` into an isolated
> environment, so it is intentionally not a project dependency.

### Manual (UI)

1. Click **Create Project** and give it a name (e.g. "Twitter Screenshot Annotation").

2. Go to **Settings → Labeling Interface → Code** and paste the label config:

   ```xml
   <View>
     <Image name="image" value="$image"/>
     <RectangleLabels name="label" toName="image">
       <Label value="tweet_region" background="#1da1f2"/>
     </RectangleLabels>
   </View>
   ```

   The source file is `tools/labeling/label_config.xml`. Click **Save**.

3. Connect the ML backend:
   - **Settings → Machine Learning → Add Model**
   - URL: `http://localhost:9090`
   - Enable **"Use for interactive preannotations"**
   - Click **Validate and Save** — the status indicator turns green when the
     backend responds

   Docker users: use `http://host.docker.internal:9090` instead of `localhost`
   because Label Studio inside Docker cannot reach the host's loopback directly.

---

## Step 6 — Import tasks

If you ran `make labeling-setup-project` in Step 5, tasks are already imported and
a Local Storage is registered for you — this step is the manual equivalent.

### With pre-drawn boxes (recommended)

If you ran Step 2, import the generated file:

- **Import → Upload Files** → select `scratch/labeling/tasks.json`

Tasks appear in the project list. Each one has a gold prediction rectangle
already drawn at the correct `tweet_region` boundary.

The task images are referenced by `/data/local-files/?d=train_images/...` URLs,
which only resolve once a Local Storage whose path is under the document root is
registered. `make labeling-setup-project` registers that storage automatically
and **does not sync it** — syncing would create a second, duplicate set of tasks
from the directory and break the seed-prediction pairing. If you set this up
manually, add the storage but do not click **Sync**:

- **Settings → Cloud Storage → Add Local Storage**
- Path: `scratch/datasets/twitter_screenshots_raw/train_images`
- Save without syncing

### Without pre-existing labels (fresh images)

- **Settings → Cloud Storage → Add Local Storage**
- Path: `scratch/datasets/twitter_screenshots_raw/train_images`
- Enable **"Treat every bucket object as a source file"** → **Sync**

The ML backend predicts the box live when you open each task. There are no
pre-drawn boxes, so the first open may take a second while the model runs.

---

## Step 7 — Annotate

### What `tweet_region` covers

The bounding box should tightly wrap the entire tweet card: profile picture,
display name, handle, tweet body, and the metadata row (timestamp, like/retweet
counts). It should not include unrelated UI chrome above or below the card.

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| **W** | Accept the current annotation and move to the next task |
| **Tab** / **Shift+Tab** | Next / previous task |
| Click box, then **Backspace** | Delete the selected box |
| Click + drag corner/edge | Resize the box |

### When to accept vs. adjust

- **Accept** if the box captures the full tweet card with no clipping
- **Adjust** if the box cuts into the tweet text, misses the author row, or
  includes a second tweet above/below
- **Draw from scratch** (drag to create a new rectangle) if the prediction is
  completely wrong — this is rare with a good checkpoint

### Edge cases

- **Overlapping tweets**: draw separate boxes for each card
- **Partial cards** at image edges: box the visible portion only
- **Corrupt or unreadable images**: use **Skip** to defer without annotating

---

## Step 8 — Export and convert to YOLO format

### Export from Label Studio

In the project view: **Export → YOLO with images**

Label Studio downloads a ZIP (e.g. `project-1-at-2025-06-15.zip`). Rename it
for clarity:

```bash
mv ~/Downloads/project-1-at-*.zip ./ls_export.zip
```

### Convert to the canonical YOLO26 dataset

This writes back into the canonical dataset folder (the `config.yaml` default).
`--test-ratio` reproduces the train/val/test layout the trainer expects.

> **Labels-only export — this is normal.** Because the images are served from
> Local Storage by reference (`/data/local-files/?d=…`), Label Studio's "YOLO
> with images" export bundles the `labels/` but leaves `images/` **empty** — it
> does not copy referenced files into the ZIP. The converter therefore pulls each
> label's image from `--images-root` (the staged `train_images/` dir) by matching
> filenames. `make labeling-export` passes this for you.

> **Heads-up:** the converter clears the `train/`, `val/`, and `test/` subdirs of
> `--out` before copying, so this **replaces** the existing canonical dataset.

```bash
uv run scripts/ls_yolo_export_to_dataset.py \
  --export ./ls_export.zip \
  --out datasets/twitter_screenshots_localization_dataset/ \
  --images-root scratch/datasets/twitter_screenshots_raw/train_images \
  --val-ratio 0.2 \
  --test-ratio 0.1 \
  --seed 42
```

Equivalent: `make labeling-export LS_EXPORT=./ls_export.zip`.

> **Why not just Sync the storage to get one self-contained ZIP?** Syncing
> doesn't help: it still won't pack the referenced images into the export, and it
> *adds a duplicate task per file* on top of the imported `tasks.json` (which is
> why Step 6 says add the storage but don't sync — Label Studio's own docs say to
> choose **Save**, not **Save & Sync**, for reference-based tasks). A single
> self-contained ZIP would require uploading the image files directly into Label
> Studio, which gives up the pre-drawn seed-box workflow. The labels-only export
> plus `--images-root` rejoin is the intended path.

Output:

```text
datasets/twitter_screenshots_localization_dataset/
├── data.yaml          # nc: 1, names: [tweet_region]
├── train/
│   ├── images/        # 70% of annotated pairs
│   └── labels/        # YOLO format: "0 x_c y_c w h" (normalized)
├── val/
│   ├── images/        # 20% of annotated pairs
│   └── labels/
└── test/
    ├── images/        # 10% of annotated pairs
    └── labels/
```

### Validate before training

The packaged config already points `dataset.path` at this folder, so no flags
are needed:

```bash
uv run python -m screencropnet_yolo.train --validate-only
```

Equivalent: `make dataset-validate`.

This runs `DatasetValidator` and logs stats without starting a training run.
Fix any reported errors before proceeding.

### Train

```bash
uv run python -m screencropnet_yolo.train
```

Equivalent: `make train`. Add `-d <path>` only to train on a different dataset:

```bash
uv run python -m screencropnet_yolo.train \
  -d datasets/twitter_screenshots_localization_dataset
```

---

## Troubleshooting

See `tools/labeling/README.md` for the full troubleshooting reference. The most
common issues:

- **Backend unreachable / CORS error**: check `curl -s http://localhost:9090/health`.
  If Label Studio runs in Docker, use `http://host.docker.internal:9090`.
- **Images don't load** (`/data/local-files/` errors in the server log):
  - **`403`** → serving is disabled. Launch with `make label-studio-local` so
    `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true` and the document root are set.
  - **`404`** → serving is on but no Local Storage is registered whose path is a
    prefix of the requested file (or the file is missing). Re-run
    `make labeling-setup-project` (it reuses the project, skips re-import, and
    registers the storage), and confirm the document root contains the
    `train_images/` directory referenced by `--images-url-prefix`.
- **Port conflict**: change `--port` on the backend (default 9090) or Label
  Studio (default 8080) and update the ML model URL in project settings.
- **Wrong-looking boxes / 500 on prediction**: the checkpoint must match
  `arch.ObjLocModel` (EfficientNet-B0, 4 regression outputs). A shape mismatch
  at the final linear layer means the `.pth` file is from a different
  architecture. Verify `MODEL_INPUT_SIZE=224`.
