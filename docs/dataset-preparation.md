# Dataset preparation

The trainer expects a YOLO-format dataset with a `data.yaml`. This doc covers the
layout and the four ways to produce one: validate an existing dataset, pull from
Roboflow, convert Pascal-VOC CSV labels, or annotate from scratch in Label
Studio.

## Expected layout

```
dataset/
├── data.yaml          # nc: 1, names: [tweet_region]
├── train/
│   ├── images/
│   └── labels/        # one .txt per image, YOLO format
├── val/
│   ├── images/
│   └── labels/
└── test/              # optional
    ├── images/
    └── labels/
```

Each label `.txt` holds one detection per line as
`class_id x_center y_center width height`, all normalized to `[0, 1]`. This
project is single-class, so `class_id` is always `0`.

## Validate an existing dataset

`--validate-only` runs structural checks, reports stats, and warns on class
imbalance without training:

```bash
uv run python -m screencropnet_yolo.train -d ./datasets/twitter --validate-only
```

Programmatically:

```python
from screencropnet_yolo.dataset_utils import (
    DatasetValidator, check_class_imbalance, display_dataset_stats, create_dataset_yaml,
)

validator = DatasetValidator("./datasets/twitter", class_names=["tweet_region"])
is_valid, stats, errors = validator.validate()
display_dataset_stats(stats)
for w in check_class_imbalance(stats.class_distribution):
    print(w)

create_dataset_yaml("./datasets/twitter", ["tweet_region"], "./datasets/twitter/data.yaml")
```

## Auto-split

Set `dataset.auto_split: true` and `split_ratios` in the config, or use the
splitter directly:

```python
from screencropnet_yolo.dataset_utils import DatasetSplitter

counts = DatasetSplitter(
    source_path="./datasets/twitter_raw",
    output_path="./datasets/twitter",
    train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42,
).split()
print(counts)  # {'train': ..., 'val': ..., 'test': ...}
```

## Roboflow

Enable in the config (`dataset.roboflow.enabled: true`) and set the workspace,
project, and version. Provide the API key via the `ROBOFLOW_API_KEY` environment
variable rather than committing it:

```bash
export ROBOFLOW_API_KEY=...   # never hardcode in config.yaml
uv run python -m screencropnet_yolo.train -c config/config.yaml
```

Or use the loader directly:

```python
from screencropnet_yolo.dataset_utils import RoboflowLoader

path = RoboflowLoader(
    api_key="...", workspace="my-workspace", project="twitter-screens",
    version=1, output_path="./datasets/twitter", format="yolov11",
).download()
```

## Convert Pascal-VOC CSV labels

If you already have Pascal-VOC annotations in a CSV, build a single-class YOLO
dataset (train/val split + `data.yaml`) in one call:

```python
from screencropnet_yolo.dataset_import import prepare_twitter_dataset

data_yaml = prepare_twitter_dataset(
    images_dir="./raw/train_images",
    csv_path="./raw/labels_pascal_temp.csv",
    # Write straight to the config's default dataset.path so a later
    # `train.py -c config.yaml` (no -d) finds it automatically.
    output_dir="./datasets/twitter_screenshots_localization_dataset",
    val_ratio=0.2, seed=42,
)
```

### Expected CSV columns

One row per bounding box, pixel-space coordinates:

```text
img_path,xmin,ymin,xmax,ymax,width,height,label
train_images/00000_twitter.PNG,30,391,1161,752,1179,2556,twitter
```

- The image column may be named `img_path` **or** `filename`; any directory
  prefix and extension are stripped to match the image file by stem
  (`train_images/00000_twitter.PNG` → label `00000_twitter.txt`).
- The class column may be named `label` **or** `class`. Every value collapses to
  the single `tweet_region` class (id `0`), so any source taxonomy maps onto the
  single-class target.
- `width`/`height` are the source image dimensions and are required to normalize
  the box.

### What the conversion does

- **Drops unlabeled images.** Only images that have at least one CSV row are
  staged; images with no annotation are skipped (and the count is logged) so the
  validator does not later reject the dataset for a missing label.
- **Splits and emits `data.yaml`.** Images are split with `DatasetSplitter`
  (`test_ratio=0`) into `train/` and `val/`, and a `data.yaml` pinned to
  `nc: 1, names: [tweet_region]` is written. Image discovery is
  case-insensitive, so uppercase extensions like `.PNG`/`.JPG` are handled.

To convert just the labels (no split), use `convert_csv(csv_path, output_dir)`,
which writes YOLO `.txt` files under `<output_dir>/labels/` and returns that
directory.

Then validate and train against the generated tree:

```bash
# If output_dir matches dataset.path in config.yaml, no -d is needed:
uv run python -m screencropnet_yolo.train -c config/config.yaml --validate-only
```

## Annotate from scratch (Label Studio)

For new screenshots with no labels, use the Label Studio workflow. An
EfficientNet-B0 ML backend pre-predicts the tweet box so annotators only confirm
or nudge it, then you export straight to the YOLO 26 layout.

Full getting-started tutorial: [docs/label-studio-annotation-guide.md](label-studio-annotation-guide.md).
Quick command reference: [tools/labeling/README.md](../tools/labeling/README.md).

Two helper scripts (PEP 723, run with `uv run`) support this flow:

- `scripts/pascal_csv_to_ls_tasks.py` — turn an existing Pascal-VOC CSV into a
  Label Studio `tasks.json` with boxes pre-drawn for verification.
- `scripts/ls_yolo_export_to_dataset.py` — convert a Label Studio "YOLO with
  images" export ZIP into a YOLO 26 dataset tree with `data.yaml`.

```bash
uv run scripts/ls_yolo_export_to_dataset.py \
  --export ./ls_export.zip \
  --out ./datasets/twitter \
  --val-ratio 0.2 --seed 42
```

Then point the trainer at the generated `data.yaml`:

```bash
uv run python -m screencropnet_yolo.train -d ./datasets/twitter
```
