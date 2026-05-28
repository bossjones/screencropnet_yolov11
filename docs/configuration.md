# Configuration reference

Training is driven by a YAML config. The bundled default lives at
`src/screencropnet_yolo/config/config.yaml`; pass your own with `--config`.

CLI flags override the file at runtime (see [usage.md](usage.md) for the full
list). The YAML is the baseline; flags are targeted overrides.

## `dataset`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `path` | str | `./datasets/twitter_screenshots_localization_dataset` | Dataset root, YOLO format expected |
| `auto_split` | bool | `false` | If true, split into train/val/test using `split_ratios` |
| `split_ratios.train` / `.val` / `.test` | float | `0.7` / `0.2` / `0.1` | Split fractions (must sum to 1.0) |
| `seed` | int | `42` | Random seed for splitting / reproducibility |
| `roboflow.enabled` | bool | `false` | Pull the dataset from Roboflow before training |
| `roboflow.api_key` | str | `""` | Roboflow key; prefer the `ROBOFLOW_API_KEY` env var |
| `roboflow.workspace` | str | `""` | Roboflow workspace slug |
| `roboflow.project` | str | `""` | Roboflow project slug |
| `roboflow.version` | int | `1` | Dataset version |
| `roboflow.format` | str | `yolov11` | Export format requested from Roboflow |

## `model`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `size` | str | `m` | Model size: `n`, `s`, `m`, `l`, `x` (nano→xlarge) |
| `weights` | str \| null | `null` | Path to weights; `null` uses Ultralytics' pretrained `yolo26<size>.pt` |
| `num_classes` | int | `1` | Number of classes |
| `class_names` | list[str] | `[tweet_region]` | Class names (length should match `num_classes`) |

## `training`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `epochs` | int | `100` | Training epochs |
| `batch_size` | int | `16` | Batch size |
| `image_size` | int | `640` | Input image size (square) |
| `learning_rate` | float | `0.01` | Initial learning rate |
| `lr_scheduler` | str | `cosine` | `cosine`, `linear`, or `constant` |
| `warmup_epochs` | int | `3` | Warmup epochs |
| `warmup_momentum` | float | `0.8` | Warmup momentum |
| `warmup_bias_lr` | float | `0.1` | Warmup bias learning rate |
| `optimizer` | str | `SGD` | `SGD`, `Adam`, or `AdamW` |
| `momentum` | float | `0.937` | Optimizer momentum |
| `weight_decay` | float | `0.0005` | Weight decay |
| `patience` | int | `20` | Early-stopping patience (epochs) |
| `min_delta` | float | `0.001` | Minimum improvement to reset patience |
| `save_period` | int | `10` | Save a checkpoint every N epochs |
| `save_best` | bool | `true` | Keep the best checkpoint |
| `workers` | int | `8` | Data-loader workers |
| `amp` | bool | `true` | Mixed-precision (AMP) training |

## `device`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | str | `auto` | `auto`, `cpu`, `cuda`, `mps`, or a GPU index (`0`, `1`, …) |
| `multi_gpu` | bool | `false` | Enable multi-GPU training |
| `gpu_ids` | list[int] | `[0, 1]` | GPUs to use when `multi_gpu` is true |

## `augmentation`

Tuned for screenshots — low rotation/shear to keep text readable.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `mosaic` | float | `1.0` | Mosaic probability |
| `mixup` | float | `0.0` | MixUp probability |
| `copy_paste` | float | `0.0` | Copy-paste probability |
| `degrees` | float | `0.0` | Rotation range (± degrees) |
| `translate` | float | `0.1` | Translation (± fraction) |
| `scale` | float | `0.5` | Scale gain (±) |
| `shear` | float | `0.0` | Shear (± degrees) |
| `perspective` | float | `0.0` | Perspective (± fraction) |
| `flipud` | float | `0.0` | Vertical flip probability |
| `fliplr` | float | `0.5` | Horizontal flip probability |
| `hsv_h` / `hsv_s` / `hsv_v` | float | `0.015` / `0.7` / `0.4` | Hue / saturation / value jitter |
| `erasing` | float | `0.4` | Random-erasing probability |

## `inference`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `confidence` | float | `0.25` | Confidence threshold |
| `iou_threshold` | float | `0.45` | IoU threshold for NMS |
| `max_detections` | int | `300` | Max detections per image |
| `augment` | bool | `false` | Test-time augmentation (TTA) |
| `half` | bool | `false` | Half-precision inference |

## `validation`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `val_period` | int | `1` | Validate every N epochs |
| `metrics` | list[str] | `[mAP50, mAP50-95, precision, recall, f1]` | Metrics to track |
| `plots` | bool | `true` | Generate plots |
| `confusion_matrix` | bool | `true` | Generate a confusion matrix |

## `export`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `formats` | list[str] | `[pytorch, onnx]` | Export formats (also `tensorrt`, `coreml`, `tflite`) |
| `onnx.opset` | int | `12` | ONNX opset version |
| `onnx.simplify` | bool | `true` | Run ONNX simplifier |
| `onnx.dynamic` | bool | `false` | Dynamic input shapes |
| `quantization.enabled` | bool | `false` | Enable quantization on export |
| `quantization.type` | str | `int8` | `int8` or `fp16` |

## `logging`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `level` | str | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `output_dir` | str | `./runs/twitter_detect` | Where runs, logs, and artifacts are written |
| `experiment_name` | str | `twitter_yolo26` | Experiment label |
| `tensorboard` | bool | `true` | Enable TensorBoard logging |
| `wandb.enabled` | bool | `false` | Enable Weights & Biases logging |
| `wandb.project` | str | `twitter-screenshot-detection` | W&B project |
| `wandb.entity` | str | `""` | W&B entity/team |

## `ablation`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `false` | Run an ablation study after training |
| `parameters.model_sizes` | list[str] | `[n, s, m]` | Model sizes to compare |
| `parameters.image_sizes` | list[int] | `[416, 640, 832]` | Image sizes to compare |
| `parameters.batch_sizes` | list[int] | `[8, 16, 32]` | Batch sizes to compare |
