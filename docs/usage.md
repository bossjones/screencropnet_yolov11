# Usage

This covers the training CLI and the most common Python APIs. For dataset
preparation see [dataset-preparation.md](dataset-preparation.md); for every
config field see [configuration.md](configuration.md).

## Training CLI

Training runs through the `screencropnet_yolo.train` module:

```bash
uv run python -m screencropnet_yolo.train [options]
```

> The `screencropnet_yolo` console script in `pyproject.toml` is currently a
> stub. Use the module form above.

### Options

| Flag | Short | Description |
|------|-------|-------------|
| `--config` | `-c` | Path to the YAML config (default `config/config.yaml`) |
| `--data` | `-d` | Dataset path (overrides `dataset.path`) |
| `--epochs` | `-e` | Number of epochs |
| `--batch` | `-b` | Batch size |
| `--imgsz` | | Image size |
| `--workers` | `-w` | Data-loader workers |
| `--model-size` | `-m` | Model size: `n`, `s`, `m`, `l`, `x` |
| `--device` | | Device: `auto`, `cpu`, `cuda`, `0`, `1`, … |
| `--output` | `-o` | Output directory |
| `--resume` | `-r` | Resume from a checkpoint path |
| `--validate-only` | | Validate the dataset, then exit |
| `--eval-only MODEL_PATH` | | Evaluate the given checkpoint, then exit |
| `--export-only MODEL_PATH` | | Export the given checkpoint, then exit |

### What a full run does

1. Optionally download from Roboflow (`dataset.roboflow.enabled`).
2. Optionally auto-split into train/val/test (`dataset.auto_split`).
3. Validate the dataset and report class imbalance; abort on failure.
4. Build the model (`ModelFactory`) and train (`Trainer`).
5. Evaluate the best checkpoint (`Evaluator`).
6. Export the model (`ModelExporter`) to the configured formats.
7. Write visualizations and, if enabled, run an ablation study.

### Examples

```bash
# Default config
uv run python -m screencropnet_yolo.train -c src/screencropnet_yolo/config/config.yaml

# Quick override of dataset + hyperparameters
uv run python -m screencropnet_yolo.train -d ./datasets/twitter -m s -e 50 -b 32

# Validate only
uv run python -m screencropnet_yolo.train -c config/config.yaml --validate-only

# Evaluate / export an existing checkpoint
uv run python -m screencropnet_yolo.train --eval-only   runs/twitter_detect/train/weights/best.pt
uv run python -m screencropnet_yolo.train --export-only runs/twitter_detect/train/weights/best.pt

# Resume
uv run python -m screencropnet_yolo.train -r runs/twitter_detect/train/weights/last.pt
```

## Python API

### Train programmatically

```python
from screencropnet_yolo.model import ModelConfig, ModelFactory, AugmentationConfig
from screencropnet_yolo.training import Trainer

config = ModelConfig(size="m", epochs=100, batch_size=16, image_size=640, device="auto")
model = ModelFactory(config).create_model()

trainer = Trainer(
    model=model,
    data_yaml="runs/twitter_detect/dataset.yaml",
    output_dir="runs/twitter_detect",
    config={"epochs": 100, "augmentation": AugmentationConfig.get_augmentation("twitter")},
)
history = trainer.train()
print("best mAP50-95:", history.best_mAP50_95)
history.save("runs/twitter_detect/history.json")
```

Add custom callbacks before training:

```python
from screencropnet_yolo.training import EarlyStopping

trainer.add_callback(EarlyStopping(patience=20, monitor="mAP50_95", mode="max"))
```

### Evaluate

```python
from ultralytics import YOLO
from screencropnet_yolo.evaluation import Evaluator

model = YOLO("runs/twitter_detect/train/weights/best.pt")
evaluator = Evaluator(model, data_yaml="runs/twitter_detect/dataset.yaml",
                      class_names=["tweet_region"])
results = evaluator.evaluate(split="val", conf=0.25, iou=0.45)
print(results.mAP50, results.mAP50_95)
results.save("evaluation_results.json")
```

Tune the confidence threshold:

```python
from screencropnet_yolo.evaluation import find_optimal_confidence

best_conf, best_f1 = find_optimal_confidence(
    model, data_yaml="runs/twitter_detect/dataset.yaml", metric="f1"
)
```

### Inference

```python
from screencropnet_yolo.inference import InferencePipeline, ResultExporter

pipeline = InferencePipeline(
    model_path="runs/twitter_detect/train/weights/best.pt",
    class_names=["tweet_region"],
    conf_threshold=0.25,
    iou_threshold=0.45,
)

# Single image
result = pipeline.predict_image("screenshot.png")
for det in result.detections:
    print(det.class_name, round(det.confidence, 3), det.bbox)

# Batch
results = pipeline.predict_batch(["a.png", "b.png", "c.png"], batch_size=16)

# Video
frames = pipeline.predict_video("clip.mp4", output_path="clip_annotated.mp4")

# Export
ResultExporter.to_json(results, "detections.json")
ResultExporter.to_coco(results, "detections_coco.json", class_names=["tweet_region"])
ResultExporter.to_yolo(results, "yolo_labels/")
```

`InferenceResult` supports `.filter_by_confidence(min_conf)` and
`.filter_by_class(class_ids)`, which return new filtered results.

### Export & quantize

```python
from ultralytics import YOLO
from screencropnet_yolo.model import ModelExporter, ModelQuantizer

model = YOLO("runs/twitter_detect/train/weights/best.pt")
exported = ModelExporter(model, "runs/twitter_detect").export(
    formats=["pytorch", "onnx"], image_size=640, simplify=True, opset=12
)

quant = ModelQuantizer("runs/twitter_detect/train/weights/best.pt")
quant.quantize_fp16("model_fp16.onnx")
```

### Visualization

```python
from screencropnet_yolo.visualization import TrainingVisualizer, ResultsDashboard

viz = TrainingVisualizer("runs/twitter_detect/visualizations")
viz.plot_training_curves(history.to_dict(), "training_curves.png")

dashboard = ResultsDashboard("runs/twitter_detect/visualizations")
dashboard.create_dashboard(history.to_dict(), results.to_dict(),
                           class_names=["tweet_region"], save_path="dashboard.png")
```

See [api-reference.md](api-reference.md) for the full list of classes and
functions.
