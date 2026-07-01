"""
Evaluation module for YOLO 11 Twitter Screenshot Detection.

This module handles:
- Model validation
- Metrics calculation (mAP, precision, recall, F1)
- Confusion matrix generation
- Per-class analysis
- Results export
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from ultralytics import YOLO

from screencropnet_yolo.model import resolve_device

logger = logging.getLogger(__name__)


@dataclass
class ClassMetrics:
    """Metrics for a single class."""

    class_name: str
    class_id: int
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    ap50: float = 0.0
    ap50_95: float = 0.0
    support: int = 0  # Number of ground truth instances


@dataclass
class EvaluationResults:
    """Container for evaluation results."""

    # Overall metrics
    mAP50: float = 0.0
    mAP50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    # Per-class metrics
    class_metrics: list[ClassMetrics] = field(default_factory=list)

    # Confusion matrix
    confusion_matrix: npt.NDArray[np.int64] | None = None

    # Speed metrics
    preprocess_time: float = 0.0
    inference_time: float = 0.0
    postprocess_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "overall": {
                "mAP50": self.mAP50,
                "mAP50_95": self.mAP50_95,
                "precision": self.precision,
                "recall": self.recall,
                "f1": self.f1,
            },
            "per_class": [
                {
                    "class_name": m.class_name,
                    "class_id": m.class_id,
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                    "ap50": m.ap50,
                    "ap50_95": m.ap50_95,
                    "support": m.support,
                }
                for m in self.class_metrics
            ],
            "speed": {
                "preprocess_ms": self.preprocess_time,
                "inference_ms": self.inference_time,
                "postprocess_ms": self.postprocess_time,
                "total_ms": self.preprocess_time + self.inference_time + self.postprocess_time,
            },
        }

        if self.confusion_matrix is not None:
            result["confusion_matrix"] = self.confusion_matrix.tolist()

        return result

    def save(self, path: str) -> None:
        """Save results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class Evaluator:
    """
    Evaluator class for YOLO 11 Twitter Screenshot Detection.

    Performs comprehensive model evaluation with detailed metrics.
    """

    def __init__(self, model: YOLO, data_yaml: str, class_names: list[str], device: str = "auto"):
        """
        Initialize evaluator.

        Args:
            model: YOLO model instance
            data_yaml: Path to dataset YAML file
            class_names: List of class names
            device: Device to run evaluation on
        """
        self.model = model
        self.data_yaml = data_yaml
        self.class_names = class_names
        # ultralytics' val() rejects device='auto'; resolve it to a concrete device.
        self.device = resolve_device(device)

    def evaluate(
        self,
        split: str = "val",
        conf: float = 0.25,
        iou: float = 0.45,
        batch_size: int = 16,
        image_size: int = 640,
        verbose: bool = True,
    ) -> EvaluationResults:
        """
        Run evaluation on dataset.

        Args:
            split: Dataset split to evaluate ('val' or 'test')
            conf: Confidence threshold
            iou: IoU threshold for NMS
            batch_size: Batch size for evaluation
            image_size: Image size for evaluation
            verbose: Print detailed results

        Returns:
            EvaluationResults with all metrics
        """
        logger.info(f"Starting evaluation on {split} set...")

        # Run validation
        results = self.model.val(
            data=self.data_yaml,
            split=split,
            batch=batch_size,
            imgsz=image_size,
            conf=conf,
            iou=iou,
            device=self.device,
            plots=True,
            save_json=True,
            verbose=verbose,
        )

        # Extract metrics
        eval_results = EvaluationResults(
            mAP50=float(results.box.map50),
            mAP50_95=float(results.box.map),
            precision=float(results.box.mp),
            recall=float(results.box.mr),
        )

        # Calculate F1
        if eval_results.precision + eval_results.recall > 0:
            eval_results.f1 = (
                2
                * eval_results.precision
                * eval_results.recall
                / (eval_results.precision + eval_results.recall)
            )

        # Extract per-class metrics
        if hasattr(results.box, "ap50") and results.box.ap50 is not None:
            ap50_per_class = results.box.ap50
            ap_per_class = results.box.ap

            for i, class_name in enumerate(self.class_names):
                if i < len(ap50_per_class):
                    class_metrics = ClassMetrics(
                        class_name=class_name,
                        class_id=i,
                        ap50=float(ap50_per_class[i]),
                        ap50_95=float(ap_per_class[i]) if i < len(ap_per_class) else 0.0,
                    )
                    eval_results.class_metrics.append(class_metrics)

        # Get confusion matrix if available
        if hasattr(results, "confusion_matrix") and results.confusion_matrix is not None:
            eval_results.confusion_matrix = results.confusion_matrix.matrix

        # Speed metrics
        if hasattr(results, "speed"):
            eval_results.preprocess_time = results.speed.get("preprocess", 0.0)
            eval_results.inference_time = results.speed.get("inference", 0.0)
            eval_results.postprocess_time = results.speed.get("postprocess", 0.0)

        if verbose:
            self._print_results(eval_results)

        return eval_results

    def _print_results(self, results: EvaluationResults) -> None:
        """Print formatted evaluation results."""
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)

        print("\nOverall Metrics:")
        print(f"  mAP@50:      {results.mAP50:.4f}")
        print(f"  mAP@50-95:   {results.mAP50_95:.4f}")
        print(f"  Precision:   {results.precision:.4f}")
        print(f"  Recall:      {results.recall:.4f}")
        print(f"  F1 Score:    {results.f1:.4f}")

        if results.class_metrics:
            print("\nPer-Class Metrics:")
            print(f"{'Class':<20} {'AP@50':>8} {'AP@50-95':>10} {'P':>8} {'R':>8} {'F1':>8}")
            print("-" * 70)

            for m in results.class_metrics:
                print(
                    f"{m.class_name:<20} "
                    f"{m.ap50:>8.4f} "
                    f"{m.ap50_95:>10.4f} "
                    f"{m.precision:>8.4f} "
                    f"{m.recall:>8.4f} "
                    f"{m.f1:>8.4f}"
                )

        print("\nSpeed Metrics:")
        print(f"  Preprocess:  {results.preprocess_time:.2f} ms/img")
        print(f"  Inference:   {results.inference_time:.2f} ms/img")
        print(f"  Postprocess: {results.postprocess_time:.2f} ms/img")
        total_time = results.preprocess_time + results.inference_time + results.postprocess_time
        print(f"  Total:       {total_time:.2f} ms/img ({1000 / total_time:.1f} FPS)")

        print("=" * 70 + "\n")


def calculate_iou(
    box1: npt.NDArray[np.floating[Any]], box2: npt.NDArray[np.floating[Any]]
) -> float:
    """
    Calculate IoU between two boxes.

    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]

    Returns:
        IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def calculate_precision_recall_curve(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> tuple[
    npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]
]:
    """
    Calculate precision-recall curve.

    Args:
        predictions: List of prediction dictionaries
        ground_truths: List of ground truth dictionaries
        iou_threshold: IoU threshold for matching

    Returns:
        Tuple of (precision, recall, thresholds)
    """
    # Sort predictions by confidence
    sorted_preds = sorted(predictions, key=lambda x: x["confidence"], reverse=True)

    total_gt = len(ground_truths)
    tp = np.zeros(len(sorted_preds))
    fp = np.zeros(len(sorted_preds))

    matched_gt = set()

    for i, pred in enumerate(sorted_preds):
        best_iou = 0
        best_gt_idx = -1

        for j, gt in enumerate(ground_truths):
            if j in matched_gt:
                continue
            if pred["class"] != gt["class"]:
                continue

            iou = calculate_iou(pred["box"], gt["box"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_threshold:
            tp[i] = 1
            matched_gt.add(best_gt_idx)
        else:
            fp[i] = 1

    # Calculate cumulative TP and FP
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    # Calculate precision and recall
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / total_gt if total_gt > 0 else np.zeros_like(tp_cumsum)

    # Get confidence thresholds
    thresholds = np.array([p["confidence"] for p in sorted_preds])

    return precision, recall, thresholds


def calculate_average_precision(
    precision: npt.NDArray[np.floating[Any]], recall: npt.NDArray[np.floating[Any]]
) -> float:
    """
    Calculate Average Precision using 101-point interpolation.

    Args:
        precision: Precision values
        recall: Recall values

    Returns:
        Average Precision value
    """
    # Add sentinel values
    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[1], precision, [0]])

    # Make precision monotonically decreasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # Calculate area under curve (101-point interpolation)
    recall_levels = np.linspace(0, 1, 101)
    ap = 0

    for r in recall_levels:
        precision_at_r = precision[recall >= r]
        ap += precision_at_r.max() if len(precision_at_r) > 0 else 0

    ap /= 101

    return ap


def generate_confusion_matrix(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> npt.NDArray[np.int64]:
    """
    Generate confusion matrix for object detection.

    Args:
        predictions: List of prediction dictionaries
        ground_truths: List of ground truth dictionaries
        num_classes: Number of classes
        iou_threshold: IoU threshold for matching

    Returns:
        Confusion matrix as numpy array
    """
    # Include background class
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    matched_gt = set()

    for pred in predictions:
        best_iou = 0
        best_gt_idx = -1
        best_gt_class = num_classes  # Background

        for j, gt in enumerate(ground_truths):
            if j in matched_gt:
                continue

            iou = calculate_iou(pred["box"], gt["box"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
                best_gt_class = gt["class"]

        if best_iou >= iou_threshold:
            matrix[best_gt_class, pred["class"]] += 1
            matched_gt.add(best_gt_idx)
        else:
            # False positive - predicted but no matching GT
            matrix[num_classes, pred["class"]] += 1

    # Handle missed ground truths (false negatives)
    for j, gt in enumerate(ground_truths):
        if j not in matched_gt:
            matrix[gt["class"], num_classes] += 1

    return matrix


def analyze_errors(
    model: YOLO, test_images: list[str], class_names: list[str], conf: float = 0.25
) -> dict[str, Any]:
    """
    Analyze common error patterns in model predictions.

    Args:
        model: YOLO model
        test_images: List of test image paths
        class_names: List of class names
        conf: Confidence threshold

    Returns:
        Dictionary of error analysis results
    """
    errors = {
        "false_positives_by_class": {name: 0 for name in class_names},
        "false_negatives_by_class": {name: 0 for name in class_names},
        "confusion_pairs": [],  # Most common class confusions
        "low_confidence_correct": 0,  # Correct predictions with low confidence
        "high_confidence_wrong": 0,  # Wrong predictions with high confidence
        "small_object_misses": 0,  # Missed small objects
        "large_object_misses": 0,  # Missed large objects
    }

    # Run inference on test images
    model.predict(test_images, conf=conf, verbose=False)

    # Analysis would require ground truth annotations
    # This is a placeholder for the analysis logic

    logger.info("Error analysis complete")

    return errors


def find_optimal_confidence(
    model: YOLO,
    data_yaml: str,
    conf_range: tuple[float, float] = (0.1, 0.9),
    num_steps: int = 9,
    metric: str = "f1",
) -> tuple[float, float]:
    """
    Find optimal confidence threshold by testing multiple values.

    Args:
        model: YOLO model
        data_yaml: Path to dataset YAML
        conf_range: Range of confidence values to test
        num_steps: Number of steps in the range
        metric: Metric to optimize ('f1', 'precision', 'recall', 'map50')

    Returns:
        Tuple of (optimal_conf, best_metric_value)
    """
    conf_values = np.linspace(conf_range[0], conf_range[1], num_steps)
    best_conf = conf_values[0]
    best_value = 0

    results_log = []

    for conf in conf_values:
        results = model.val(data=data_yaml, conf=conf, verbose=False)

        if metric == "f1":
            p = results.box.mp
            r = results.box.mr
            value = 2 * p * r / (p + r) if (p + r) > 0 else 0
        elif metric == "precision":
            value = results.box.mp
        elif metric == "recall":
            value = results.box.mr
        elif metric == "map50":
            value = results.box.map50
        else:
            value = results.box.map

        results_log.append({"conf": conf, metric: value})

        if value > best_value:
            best_value = value
            best_conf = conf

    logger.info(f"Optimal confidence threshold: {best_conf:.2f} ({metric}={best_value:.4f})")

    return best_conf, best_value


def benchmark_model(
    model: YOLO,
    image_size: int = 640,
    batch_sizes: list[int] | None = None,
    device: str = "cuda",
    warmup_runs: int = 10,
    test_runs: int = 100,
) -> dict[str, dict[str, float]]:
    """
    Benchmark model inference speed.

    Args:
        model: YOLO model
        image_size: Image size for benchmarking
        batch_sizes: List of batch sizes to test
        device: Device to run benchmark on
        warmup_runs: Number of warmup runs
        test_runs: Number of test runs

    Returns:
        Dictionary of benchmark results per batch size
    """
    if batch_sizes is None:
        batch_sizes = [1, 8, 16, 32]

    import time

    import torch

    results = {}

    for batch_size in batch_sizes:
        # Create dummy input
        dummy_input = torch.randn(batch_size, 3, image_size, image_size).to(device)

        # Warmup
        for _ in range(warmup_runs):
            _ = model.model(dummy_input)  # pyright: ignore[reportCallIssue, reportOptionalCall]

        # Synchronize if CUDA
        if device == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(test_runs):
            start = time.perf_counter()
            _ = model.model(dummy_input)  # pyright: ignore[reportCallIssue, reportOptionalCall]
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times) * 1000  # Convert to ms
        std_time = np.std(times) * 1000

        results[f"batch_{batch_size}"] = {
            "avg_ms": avg_time,
            "std_ms": std_time,
            "fps": batch_size / (avg_time / 1000),
            "ms_per_image": avg_time / batch_size,
        }

        logger.info(
            f"Batch {batch_size}: {avg_time:.2f}±{std_time:.2f}ms "
            f"({results[f'batch_{batch_size}']['fps']:.1f} FPS)"
        )

    return results
