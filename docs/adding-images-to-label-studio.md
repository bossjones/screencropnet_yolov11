# Adding images to Label Studio

This guide covers how to add new screenshots to the Label Studio annotation
project. For the full end-to-end setup (launching Label Studio, the ML backend,
and annotating from scratch), see
[label-studio-annotation-guide.md](label-studio-annotation-guide.md).

## Overview

Two workflows apply depending on whether the new images already have bounding-box
coordinates:

- **Case A — fresh images (no prior labels)**: rename and copy images into the
  staging directory, then sync via Label Studio's Local Storage UI. The ML
  backend predicts the `tweet_region` box live when each task is opened.
- **Case B — images with existing Pascal-VOC CSV labels**: rename and copy
  images, append rows to the staging CSV, regenerate `tasks.json`, and
  re-import into Label Studio with pre-drawn boxes.

Use `scripts/add_images_to_labeling.py` for the rename-and-copy step in either
case. It auto-detects the next available index, so running it multiple times is
safe.

---

## File naming convention

All images in the staging directory follow this pattern:

```text
NNNNN_twitter.PNG
```

- `NNNNN` — five-digit zero-padded integer (e.g. `00000`, `01494`)
- `_twitter` — literal suffix identifying the project
- `.PNG` — uppercase extension (matches the pytorch-lab originals)

Examples: `00000_twitter.PNG`, `00070_twitter.PNG`, `01494_twitter.PNG`

### Finding the current maximum index

```bash
ls scratch/datasets/twitter_screenshots_raw/train_images/ | sort | tail -3
```

The utility script auto-detects this and starts from `max + 1`, so you do not
normally need to check manually.

---

## Using the utility script

`scripts/add_images_to_labeling.py` scans the staging directory for existing
files, determines the next available index, renames each source image to the
convention, and copies it into place.

### Quick start

```bash
# Preview without writing (recommended first run)
uv run scripts/add_images_to_labeling.py \
  --source-dir /path/to/new/screenshots \
  --dry-run --verbose

# Live run
uv run scripts/add_images_to_labeling.py \
  --source-dir /path/to/new/screenshots
```

Or via make:

```bash
make labeling-add-images IMAGE_DIR=/path/to/new/screenshots
```

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--source-dir` | required | Folder of images to add (non-recursive) |
| `--staging-dir` | `scratch/datasets/twitter_screenshots_raw/train_images` | Destination |
| `--suffix` | `twitter` | Middle part of the filename — `NNNNN_<suffix>.PNG` |
| `--start-index` | auto-detected | Override the starting counter |
| `--ext` | `PNG` | Output file extension (uppercase for convention consistency) |
| `--force` | off | Overwrite existing destination files without prompting |
| `--dry-run` | off | Print what would happen; no files written |
| `--verbose` | off | Log each file copy (implied by `--dry-run`) |

Accepted input formats: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tif`,
`.tiff` (case-insensitive).

### Collision safety

If a destination file already exists, the script prompts before overwriting:

```text
  overwrite 01495_twitter.PNG? [y/N]
```

Pass `--force` to skip all prompts and overwrite unconditionally. In
`--dry-run` mode the script shows what it would ask, without writing anything.

---

## Case A — Fresh images (no prior labels)

Use this when you have new screenshots with no existing bounding-box coordinates.

### Step 1 — Copy and rename images

```bash
uv run scripts/add_images_to_labeling.py \
  --source-dir /path/to/new/screenshots \
  --verbose
```

Note the index range printed (e.g. `indices 01495–01510_twitter.PNG`).

### Step 2 — Confirm the files landed

```bash
ls scratch/datasets/twitter_screenshots_raw/train_images/ | sort | tail -5
```

### Step 3 — Sync in Label Studio UI

In your browser at <http://localhost:8080>:

**Settings → Cloud Storage → Local Storage → (select the existing storage) → Sync Now**

New tasks appear in the project list immediately. Each task has no pre-drawn
box; the ML backend predicts it live when you open the task for the first time.

### Step 4 — Annotate

Open tasks in Label Studio, accept or adjust the predicted box, and submit.
See [label-studio-annotation-guide.md — Step 7](label-studio-annotation-guide.md#step-7----annotate)
for keyboard shortcuts and edge-case guidance.

### Step 5 — Export and convert

```bash
# Export from Label Studio UI: Export → YOLO with images → rename the zip
mv ~/Downloads/project-1-at-*.zip ./ls_export.zip

# Convert to the canonical YOLO26 dataset
make labeling-export LS_EXPORT=./ls_export.zip
```

---

## Case B — Images with existing Pascal-VOC CSV labels

Use this when you have bounding-box coordinates already recorded in Pascal-VOC
format (columns: `img_path,xmin,ymin,xmax,ymax,width,height,label`).

### Step 1 — Copy and rename images

```bash
uv run scripts/add_images_to_labeling.py \
  --source-dir /path/to/new/screenshots \
  --verbose
```

Note the index range printed (e.g. `indices 01495–01510_twitter.PNG`).

### Step 2 — Append rows to the staging CSV

Open `scratch/datasets/twitter_screenshots_raw/labels_pascal_temp.csv` and
append one row per bounding box using the new filenames. Required columns:

```text
img_path,xmin,ymin,xmax,ymax,width,height,label
train_images/01495_twitter.PNG,30,391,1161,752,1179,2556,twitter
train_images/01496_twitter.PNG,20,300,1137,1836,1170,2532,twitter
```

All coordinates are in pixels. `xmin`/`ymin` are the top-left corner;
`xmax`/`ymax` are the bottom-right corner. `width`/`height` are the full
image dimensions (not the box dimensions).

### Step 3 — Regenerate tasks.json

```bash
make labeling-tasks
```

This runs `scripts/pascal_csv_to_ls_tasks.py` over the updated CSV and
produces `scratch/labeling/tasks.json` with pre-drawn boxes for every row.

### Step 4 — Re-import into Label Studio

**First time setting up the project** (or to reset it):

```bash
make labeling-setup-project
```

**Adding tasks to an existing project** (force re-import):

```bash
uv run scripts/setup_ls_project.py \
  --title screencropnet \
  --tasks scratch/labeling/tasks.json \
  --force-import
```

Or manually via the UI: **Import → Upload Files → select `scratch/labeling/tasks.json`**.

### Step 5 — Annotate

Confirm or adjust the pre-drawn boxes. Boxes from the CSV are treated as seed
predictions; the ML backend is available for interactive re-prediction if a box
needs to be redrawn.

### Step 6 — Export and convert

```bash
mv ~/Downloads/project-1-at-*.zip ./ls_export.zip
make labeling-export LS_EXPORT=./ls_export.zip
```

---

## After annotation — validate and train

These steps are identical for both cases once annotation is complete.

```bash
# Validate the dataset split before training
make dataset-validate

# Train
make train
```

Full export and training details:
[label-studio-annotation-guide.md — Step 8](label-studio-annotation-guide.md#step-8----export-and-convert-to-yolo-format).

---

## Environment prerequisites

- Label Studio running on port 8080: `make label-studio-local`
- `LABEL_STUDIO_API_KEY` exported (needed for `make labeling-setup-project`)
- ML backend running on port 9090 (for live predictions): `make ml-backend`

---

## See also

- [label-studio-annotation-guide.md](label-studio-annotation-guide.md) — full
  end-to-end setup guide (launch Label Studio, ML backend, annotate, export)
- [dataset-preparation.md](dataset-preparation.md) — YOLO dataset layout and
  alternative ingestion methods
- `scripts/add_images_to_labeling.py` — utility script described in this guide
- `scripts/pascal_csv_to_ls_tasks.py` — converts Pascal-VOC CSV to Label Studio tasks
- `scripts/ls_yolo_export_to_dataset.py` — converts Label Studio export to YOLO dataset
