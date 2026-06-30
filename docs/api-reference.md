# API reference

Public classes and functions per module. Import everything absolutely from
`screencropnet_yolo.<module>`. For runnable examples see [usage.md](usage.md).

## `dataset_utils`

### Classes

- `DatasetStats` (dataclass) — container for counts, class distribution, bbox
  stats, image sizes, and lists of corrupt images / missing annotations.
- `DatasetValidator(dataset_path, class_names)` — `.validate() -> (is_valid,
  DatasetStats, errors)`.
- `DatasetSplitter(source_path, output_path, train_ratio=0.7, val_ratio=0.2,
  test_ratio=0.1, seed=42)` — `.split() -> dict[str, int]`.
- `RoboflowLoader(api_key, workspace, project, version, output_path,
  format="yolov11")` — `.download() -> Path`.
- `TwitterScreenshotPreprocessor(target_size=640)` — `.preprocess(image)`,
  `.letterbox(image, new_shape=(640, 640), color=(114, 114, 114), auto=True,
  scale_fill=False, scaleup=True)`.

### Functions

- `create_dataset_yaml(dataset_path, class_names, output_path) -> str`
- `display_dataset_stats(stats) -> None`
- `check_class_imbalance(class_distribution, threshold=0.1) -> list[str]`

## `dataset_import`

Constants: `TWEET_REGION_CLASS_ID = 0`, `TWEET_REGION_CLASS_NAME =
"tweet_region"`, `DEFAULT_CLASS_MAP`.

- `pascal_row_to_yolo(row, class_map) -> tuple[int, float, float, float, float]`
- `convert_csv(csv_path, output_dir, class_map=None) -> Path` — writes one YOLO
  `.txt` per image under `<output_dir>/labels/` and returns that dir. Accepts
  the image column as `img_path` or `filename`, and the class column as `label`
  or `class`.
- `prepare_twitter_dataset(images_dir, csv_path, output_dir, *, val_ratio=0.2,
  seed=42, class_map=None) -> Path` — full pipeline (convert + train/val split +
  `data.yaml`). Stages only images that have a CSV annotation; images with no
  label are skipped so the dataset validates cleanly.

## `model`

### Classes

- `ModelConfig` (dataclass) — `size="m"`, `weights=None`, `num_classes=None`,
  `device="auto"`, `multi_gpu=False`, `gpu_ids=None`, `epochs=100`,
  `batch_size=16`, `image_size=640`, `learning_rate=0.01`, `optimizer="SGD"`,
  `momentum=0.937`, `weight_decay=0.0005`, `warmup_epochs=3`, `patience=20`,
  `amp=True`, `workers=8`.
- `ModelFactory(config)` — `.create_model() -> YOLO`,
  `.get_training_args(data_yaml, output_dir) -> dict`.
- `AugmentationConfig` — `get_augmentation(strategy="twitter") -> dict`
  (strategies: `twitter`, `conservative`, `aggressive`).
- `ModelExporter(model, output_dir, source_weights=None)` — `.export(formats,
  image_size=640, half=False, dynamic=False, simplify=True, opset=12) ->
  dict[str, str]`. `source_weights` is the real `.pt` checkpoint backing `model`
  (Ultralytics writes it to `{run}/train/weights/best.pt`, not `{output_dir}`);
  the `pytorch` format reports/copies that file instead of guessing. PyTorch is
  exported first; other formats (e.g. `onnx`) fail soft — a failed export logs
  a warning and is skipped rather than aborting the run.
- `ModelQuantizer(model_path)` — `.quantize_int8(calibration_data, output_path)`,
  `.quantize_fp16(output_path)`.

### Functions

- `get_model_info(model) -> dict`
- `compare_models(model_paths, test_image) -> dict`

## `output`

Pure presentation helpers for the training CLI (no Ultralytics/torch imports,
raw ANSI rather than `rich`). Drives the run-configuration banner, the closing
artifacts table, and color-aware logging.

### Classes

- `Color` — raw ANSI SGR code constants (`RESET`, `BOLD`, `DIM`, `RED`,
  `GREEN`, `YELLOW`, `BLUE`, `CYAN`).
- `Artifact(label, path, size)` (dataclass) — one row in the artifacts table;
  `path`/`size` are `None` when the artifact wasn't produced.
- `ColorFormatter(fmt=None, datefmt=None, *, enabled=False)` — a
  `logging.Formatter` that colorizes the levelname when `enabled`; behaves as a
  plain formatter otherwise, so the same format string drives both the colored
  stream handler and the plain file handler.

### Functions

- `colorize(text, color, *, enabled) -> str` — wrap `text` in an ANSI code +
  reset, or return it unchanged when `enabled` is false or `color` is empty.
- `human_size(n_bytes) -> str` — format a byte count as e.g. `1.5 KB`.
- `format_run_summary(*, model_size, arch, device, epochs, batch, imgsz,
  dataset_path, output_dir, weights_dir, best_pt, export_formats, enabled=False)
  -> str` — render the startup `RUN CONFIGURATION` banner.
- `format_artifacts_table(rows, *, best_epoch=None, best_map=None,
  enabled=False) -> str` — render the closing `ARTIFACTS` table from a list of
  `Artifact`s.

## `training`

### Classes

- `TrainingMetrics` (dataclass) — per-epoch losses and metrics; `.to_dict()`.
- `TrainingHistory` (dataclass) — `.add_metrics(m)`, `.to_dict()`, `.save(path)`;
  fields `best_epoch`, `best_mAP50`, `best_mAP50_95`, `training_time`.
- `Trainer(model, data_yaml, output_dir, config)` — `.train() -> TrainingHistory`,
  `.resume(checkpoint_path) -> TrainingHistory`, `.add_callback(callback)`.
- Callbacks (subclass `TrainingCallback`): `MetricsLogger(history,
  log_interval=1)`, `EarlyStopping(patience=20, min_delta=0.001,
  monitor="mAP50_95", mode="max")`, `CheckpointCallback(save_dir, save_period=10,
  save_best=True)`, `TensorBoardCallback(log_dir)`, `WandbCallback(project,
  entity=None, config=None)`.

### Functions

- `create_ablation_study(model_factory, data_yaml, output_dir, ablation_config)
  -> dict[str, TrainingHistory]`

## `evaluation`

### Classes

- `ClassMetrics` (dataclass) — per-class precision/recall/f1/ap50/ap50_95/support.
- `EvaluationResults` (dataclass) — aggregate metrics, `class_metrics`,
  `confusion_matrix`, timings; `.to_dict()`, `.save(path)`.
- `Evaluator(model, data_yaml, class_names, device="auto")` — `.evaluate(split="val",
  conf=0.25, iou=0.45, batch_size=16, image_size=640, verbose=True) ->
  EvaluationResults`.

### Functions

- `calculate_iou(box1, box2) -> float`
- `calculate_precision_recall_curve(predictions, ground_truths,
  iou_threshold=0.5) -> (precision, recall, thresholds)`
- `calculate_average_precision(precision, recall) -> float`
- `generate_confusion_matrix(predictions, ground_truths, num_classes,
  iou_threshold=0.5) -> ndarray`
- `analyze_errors(model, test_images, class_names, conf=0.25) -> dict`
- `find_optimal_confidence(model, data_yaml, conf_range=(0.1, 0.9), num_steps=9,
  metric="f1") -> (optimal_conf, best_value)`
- `benchmark_model(model, image_size=640, batch_sizes=None, device="cuda",
  warmup_runs=10, test_runs=100) -> dict`

## `inference`

### Classes

- `Detection` (dataclass) — `class_id`, `class_name`, `confidence`, `bbox`
  (x1,y1,x2,y2), `bbox_normalized`; `.to_dict()`.
- `InferenceResult` (dataclass) — `image_path`, `image_size`, `detections`,
  `inference_time`; `.to_dict()`, `.filter_by_confidence(min_conf)`,
  `.filter_by_class(class_ids)`.
- `InferencePipeline(model_path, class_names, device="auto", conf_threshold=0.25,
  iou_threshold=0.45, max_detections=300)` — `.predict_image(image, conf=None,
  iou=None, augment=False)`, `.predict_batch(images, conf=None, iou=None,
  batch_size=16)`, `.predict_video(video_path, output_path=None, conf=None,
  iou=None, show=False, save_frames=False)`.
- `ResultExporter` — static `to_json(results, output_path)`, `to_coco(results,
  output_path, class_names)`, `to_yolo(results, output_dir)`.

### Functions

- `apply_nms(detections, iou_threshold=0.45, class_agnostic=False) ->
  list[Detection]`

## `visualization`

All plot methods return a Matplotlib `Figure` and accept `save_path=None`.

- `TrainingVisualizer(output_dir)` — `.plot_training_curves(history, save_path)`,
  `.plot_loss_components(history, save_path)`.
- `ConfusionMatrixVisualizer` — static `plot_confusion_matrix(matrix, class_names,
  normalize=True, save_path=None)`.
- `DetectionVisualizer(class_names)` — `.draw_detections(image, detections,
  show_confidence=True, line_width=2)`, `.plot_detection_grid(images, results,
  cols=3, figsize=(15, 10), save_path=None)`.
- `DatasetVisualizer` — static `plot_class_distribution`,
  `plot_bbox_size_distribution`, `plot_image_size_distribution`.
- `ResultsDashboard(output_dir)` — `.create_dashboard(training_history,
  evaluation_results, class_names, save_path=None)`.
- `create_comparison_plot(results, metric="mAP50_95", save_path=None)` —
  compare ablation-study results.
