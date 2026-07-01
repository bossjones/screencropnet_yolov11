# screencropnet_yolo

A YOLO 26 training, evaluation, and inference pipeline for detecting **tweet
regions** in Twitter/X screenshots. Built on [Ultralytics](https://docs.ultralytics.com/)
YOLO 26, it wraps the full lifecycle — dataset acquisition, validation, training,
metrics, visualization, and multi-format export — behind a configurable CLI and a
set of composable Python modules.

The detector is single-class (`tweet_region`): it localizes the outer bounding
box of a tweet inside a screenshot so downstream tooling can crop it cleanly.

> Status: Beta. Package import name is `screencropnet_yolo`; the GitHub repo is
> `screencropnet_yolov11` (a holdover from the YOLOv11 era — the code targets
> YOLO 26 now). Not yet published to PyPI; install from source.

## Features

- **Config-driven training** — one YAML file (`config/config.yaml`) controls the
  model size, hyperparameters, augmentation, device selection, and export
  formats; any field can be overridden on the command line.
- **Dataset tooling** — validate YOLO-format datasets, auto-split into
  train/val/test, generate `data.yaml`, pull from Roboflow, or convert
  Pascal-VOC CSV annotations to YOLO labels.
- **Label Studio workflow** — annotate screenshots with an EfficientNet-B0 ML
  backend that pre-predicts the tweet box, then export straight to a YOLO 26
  dataset (see [docs/dataset-preparation.md](docs/dataset-preparation.md)).
- **Evaluation & metrics** — mAP@50, mAP@50-95, precision, recall, F1, per-class
  metrics, confusion matrix, and confidence-threshold tuning.
- **Inference** — single image, batch, and video, with COCO/YOLO/JSON export and
  drawn-detection rendering.
- **Visualization** — training curves, loss components, confusion matrices, and a
  combined results dashboard.
- **Multi-GPU, AMP, and export** — PyTorch, ONNX, and (optionally) TensorRT /
  CoreML / TFLite, plus INT8/FP16 quantization helpers.

## Requirements

- Python 3.11–3.13
- [uv](https://docs.astral.sh/uv/) for dependency management (this project does
  **not** use `pip`/`poetry` directly)
- A trained-from or pretrained YOLO 26 weights file (Ultralytics downloads
  `yolo26{n,s,m,l,x}.pt` automatically on first use)

## Installation

```bash
git clone https://github.com/bossjones/screencropnet_yolov11.git
cd screencropnet_yolov11
make install        # uv sync --all-extras
```

If you don't have `uv` yet, see [docs/installation.md](docs/installation.md).

## Quick start

Training is driven by the `screencropnet_yolo.train` module. Point it at a
dataset and a config:

```bash
# Train with the bundled default config
uv run python -m screencropnet_yolo.train \
  --config src/screencropnet_yolo/config/config.yaml

# Override the dataset and a few hyperparameters on the CLI
uv run python -m screencropnet_yolo.train \
  --data ./datasets/twitter_screenshots \
  --model-size m --epochs 100 --batch 16 --imgsz 640

# Validate a dataset without training
uv run python -m screencropnet_yolo.train --config config/config.yaml --validate-only

# Evaluate or export an existing checkpoint only
uv run python -m screencropnet_yolo.train --eval-only   runs/twitter_detect/train/weights/best.pt
uv run python -m screencropnet_yolo.train --export-only runs/twitter_detect/train/weights/best.pt

# Resume from a checkpoint
uv run python -m screencropnet_yolo.train --resume runs/twitter_detect/train/weights/last.pt
```

A full training run validates the dataset, trains, evaluates the best
checkpoint, exports the model, and writes visualizations under the configured
output directory (default `./runs/twitter_detect`).

> The `screencropnet_yolo` console script declared in `pyproject.toml` is
> currently a stub. Run training via the module form shown above.

### Inference in Python

```python
from screencropnet_yolo.inference import InferencePipeline

pipeline = InferencePipeline(
    model_path="runs/twitter_detect/train/weights/best.pt",
    class_names=["tweet_region"],
)
result = pipeline.predict_image("screenshot.png")
for det in result.detections:
    print(det.class_name, det.confidence, det.bbox)
```

See [docs/usage.md](docs/usage.md) for evaluation, batch/video inference, export,
and more Python API examples.

### Async classify pipeline

The repo has two halves. The training pipeline above builds the YOLO 26
tweet-region **detector**. The other half is an async **classify/ingest** service
(FastAPI + RabbitMQ worker + Postgres + CLI) that triages a folder of screenshots
as twitter / not-twitter and exports the twitter-positive originals into the raw
dataset. To get it running end to end, follow
[docs/quickstart.md](docs/quickstart.md); for the architecture, endpoints, and
metrics, see [docs/screencrop-pipeline.md](docs/screencrop-pipeline.md).

## Documentation

| Doc | What's inside |
|-----|---------------|
| [docs/quickstart.md](docs/quickstart.md) | Get the async classify pipeline running end to end in ~10 minutes |
| [docs/screencrop-pipeline.md](docs/screencrop-pipeline.md) | Classify pipeline deep dive: architecture, endpoints, metrics, export semantics, config |
| [docs/installation.md](docs/installation.md) | Installing `uv` and Python |
| [docs/usage.md](docs/usage.md) | CLI reference, training/eval/inference/export workflows, Python API |
| [docs/demo.md](docs/demo.md) | Quick visual smoke test via the `screencrop-demo` tool |
| [docs/configuration.md](docs/configuration.md) | Complete `config.yaml` field reference |
| [docs/dataset-preparation.md](docs/dataset-preparation.md) | Dataset layout, validation, splitting, Roboflow, Pascal-VOC conversion, Label Studio |
| [docs/architecture.md](docs/architecture.md) | Pipeline modules and how they fit together |
| [docs/api-reference.md](docs/api-reference.md) | Public classes and functions per module |
| [docs/development.md](docs/development.md) | Dev workflows, linting, testing, type checking |
| [docs/publishing.md](docs/publishing.md) | Publishing releases to PyPI |
| [tools/labeling/README.md](tools/labeling/README.md) | Label Studio annotation runbook |

## Development

```bash
make install   # uv sync --all-extras
make lint       # codespell + ruff + basedpyright
make test       # pytest with coverage
make check      # ty type check
make            # install + lint + test
```

See [docs/development.md](docs/development.md) for the full workflow.

## License

MIT — see [LICENSE](LICENSE).
