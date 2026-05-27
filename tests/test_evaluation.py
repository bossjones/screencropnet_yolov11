"""Tests for evaluation module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pytest_mock import MockerFixture

from screencropnet_yolov11.evaluation import (
    ClassMetrics,
    EvaluationResults,
    Evaluator,
    analyze_errors,
    benchmark_model,
    calculate_average_precision,
    calculate_iou,
    calculate_precision_recall_curve,
    find_optimal_confidence,
    generate_confusion_matrix,
)

# --- Helper Functions ---


def create_mock_yolo(mocker: MockerFixture) -> Any:
    """Create a mock YOLO model with common attributes."""
    mock_model = mocker.MagicMock()
    mock_model.task = "detect"
    mock_model.type = "v11"

    # Mock internal model
    mock_model.model = mocker.MagicMock()

    # Mock predict method
    mock_result = mocker.MagicMock()
    mock_result.boxes = [mocker.MagicMock()] * 3
    mock_model.predict.return_value = [mock_result]

    return mock_model


def create_mock_validation_results(mocker: MockerFixture) -> Any:
    """Create mock validation results from model.val()."""
    mock_results = mocker.MagicMock()

    # Box metrics
    mock_results.box.map50 = 0.85
    mock_results.box.map = 0.72
    mock_results.box.mp = 0.88
    mock_results.box.mr = 0.80
    mock_results.box.ap50 = np.array([0.90, 0.80])
    mock_results.box.ap = np.array([0.75, 0.68])

    # Confusion matrix
    mock_results.confusion_matrix.matrix = np.array([[10, 2], [1, 15]])

    # Speed metrics
    mock_results.speed = {
        "preprocess": 1.5,
        "inference": 10.2,
        "postprocess": 2.3,
    }

    return mock_results


# --- TestClassMetrics ---


class TestClassMetrics:
    """Tests for ClassMetrics dataclass."""

    def test_default_values(self) -> None:
        """Default field values are initialized correctly."""
        metrics = ClassMetrics(class_name="test_class", class_id=0)

        assert metrics.class_name == "test_class"
        assert metrics.class_id == 0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0
        assert metrics.ap50 == 0.0
        assert metrics.ap50_95 == 0.0
        assert metrics.support == 0

    def test_custom_initialization(self) -> None:
        """Custom values are properly assigned."""
        metrics = ClassMetrics(
            class_name="twitter_post",
            class_id=1,
            precision=0.95,
            recall=0.88,
            f1=0.91,
            ap50=0.92,
            ap50_95=0.78,
            support=150,
        )

        assert metrics.class_name == "twitter_post"
        assert metrics.class_id == 1
        assert metrics.precision == 0.95
        assert metrics.recall == 0.88
        assert metrics.f1 == 0.91
        assert metrics.ap50 == 0.92
        assert metrics.ap50_95 == 0.78
        assert metrics.support == 150


# --- TestEvaluationResults ---


class TestEvaluationResults:
    """Tests for EvaluationResults dataclass."""

    def test_default_values(self) -> None:
        """Default field values are initialized correctly."""
        results = EvaluationResults()

        assert results.mAP50 == 0.0
        assert results.mAP50_95 == 0.0
        assert results.precision == 0.0
        assert results.recall == 0.0
        assert results.f1 == 0.0
        assert results.class_metrics == []
        assert results.confusion_matrix is None
        assert results.preprocess_time == 0.0
        assert results.inference_time == 0.0
        assert results.postprocess_time == 0.0

    def test_to_dict(self) -> None:
        """to_dict returns correct dictionary structure."""
        results = EvaluationResults(
            mAP50=0.85,
            mAP50_95=0.72,
            precision=0.88,
            recall=0.80,
            f1=0.84,
            preprocess_time=1.5,
            inference_time=10.2,
            postprocess_time=2.3,
        )

        result_dict = results.to_dict()

        assert result_dict["overall"]["mAP50"] == 0.85
        assert result_dict["overall"]["mAP50_95"] == 0.72
        assert result_dict["overall"]["precision"] == 0.88
        assert result_dict["overall"]["recall"] == 0.80
        assert result_dict["overall"]["f1"] == 0.84
        assert result_dict["speed"]["preprocess_ms"] == 1.5
        assert result_dict["speed"]["inference_ms"] == 10.2
        assert result_dict["speed"]["postprocess_ms"] == 2.3
        assert result_dict["speed"]["total_ms"] == 14.0
        assert result_dict["per_class"] == []
        assert "confusion_matrix" not in result_dict

    def test_to_dict_with_class_metrics(self) -> None:
        """to_dict includes nested class metrics."""
        results = EvaluationResults(
            mAP50=0.85,
            class_metrics=[
                ClassMetrics(class_name="class_a", class_id=0, ap50=0.90),
                ClassMetrics(class_name="class_b", class_id=1, ap50=0.80),
            ],
        )

        result_dict = results.to_dict()

        assert len(result_dict["per_class"]) == 2
        assert result_dict["per_class"][0]["class_name"] == "class_a"
        assert result_dict["per_class"][0]["ap50"] == 0.90
        assert result_dict["per_class"][1]["class_name"] == "class_b"
        assert result_dict["per_class"][1]["ap50"] == 0.80

    def test_to_dict_with_confusion_matrix(self) -> None:
        """to_dict includes confusion matrix when present."""
        confusion_mat = np.array([[10, 2], [1, 15]], dtype=np.int64)
        results = EvaluationResults(confusion_matrix=confusion_mat)

        result_dict = results.to_dict()

        assert "confusion_matrix" in result_dict
        assert result_dict["confusion_matrix"] == [[10, 2], [1, 15]]

    def test_save(self, tmp_path: Path) -> None:
        """save creates valid JSON file."""
        results = EvaluationResults(
            mAP50=0.85,
            mAP50_95=0.72,
            precision=0.88,
            recall=0.80,
            f1=0.84,
        )
        output_file = tmp_path / "results.json"

        results.save(str(output_file))

        assert output_file.exists()
        with open(output_file) as f:
            saved_data = json.load(f)
        assert saved_data["overall"]["mAP50"] == 0.85


# --- TestCalculateIou ---


class TestCalculateIou:
    """Tests for calculate_iou function."""

    def test_identical_boxes(self) -> None:
        """Identical boxes have IoU of 1.0."""
        box = np.array([10, 10, 50, 50], dtype=np.float64)

        iou = calculate_iou(box, box)

        assert iou == 1.0

    def test_no_overlap(self) -> None:
        """Non-overlapping boxes have IoU of 0.0."""
        box1 = np.array([0, 0, 10, 10], dtype=np.float64)
        box2 = np.array([20, 20, 30, 30], dtype=np.float64)

        iou = calculate_iou(box1, box2)

        assert iou == 0.0

    def test_partial_overlap(self) -> None:
        """Partially overlapping boxes have correct IoU."""
        box1 = np.array([0, 0, 20, 20], dtype=np.float64)
        box2 = np.array([10, 10, 30, 30], dtype=np.float64)

        iou = calculate_iou(box1, box2)

        # Intersection: 10x10 = 100
        # Union: 400 + 400 - 100 = 700
        expected_iou = 100 / 700
        assert abs(iou - expected_iou) < 1e-6

    def test_one_box_inside_other(self) -> None:
        """Smaller box inside larger box."""
        box1 = np.array([0, 0, 100, 100], dtype=np.float64)
        box2 = np.array([25, 25, 75, 75], dtype=np.float64)

        iou = calculate_iou(box1, box2)

        # Intersection: 50x50 = 2500
        # Union: 10000 + 2500 - 2500 = 10000
        expected_iou = 2500 / 10000
        assert abs(iou - expected_iou) < 1e-6

    def test_edge_touching(self) -> None:
        """Boxes sharing only an edge have IoU of 0.0."""
        box1 = np.array([0, 0, 10, 10], dtype=np.float64)
        box2 = np.array([10, 0, 20, 10], dtype=np.float64)

        iou = calculate_iou(box1, box2)

        assert iou == 0.0

    def test_zero_area_box(self) -> None:
        """Zero area box results in zero IoU."""
        box1 = np.array([0, 0, 0, 0], dtype=np.float64)
        box2 = np.array([0, 0, 10, 10], dtype=np.float64)

        iou = calculate_iou(box1, box2)

        assert iou == 0.0


# --- TestCalculatePrecisionRecallCurve ---


class TestCalculatePrecisionRecallCurve:
    """Tests for calculate_precision_recall_curve function."""

    def test_empty_predictions(self) -> None:
        """Empty predictions return empty arrays."""
        ground_truths = [{"class": 0, "box": np.array([10, 10, 50, 50], dtype=np.float64)}]

        precision, recall, thresholds = calculate_precision_recall_curve([], ground_truths)

        assert len(precision) == 0
        assert len(recall) == 0
        assert len(thresholds) == 0

    def test_empty_ground_truths(self) -> None:
        """Empty ground truths result in all false positives."""
        predictions = [
            {"class": 0, "confidence": 0.9, "box": np.array([10, 10, 50, 50], dtype=np.float64)}
        ]

        precision, recall, thresholds = calculate_precision_recall_curve(predictions, [])

        # With no GTs, recall is 0 (division by zero handled)
        assert len(precision) == 1
        assert len(recall) == 1

    def test_all_true_positives(self) -> None:
        """Perfect predictions have precision=1 at all thresholds."""
        box = np.array([10, 10, 50, 50], dtype=np.float64)
        predictions = [
            {"class": 0, "confidence": 0.9, "box": box},
            {"class": 0, "confidence": 0.8, "box": np.array([60, 60, 100, 100], dtype=np.float64)},
        ]
        ground_truths = [
            {"class": 0, "box": box},
            {"class": 0, "box": np.array([60, 60, 100, 100], dtype=np.float64)},
        ]

        precision, recall, thresholds = calculate_precision_recall_curve(predictions, ground_truths)

        assert np.all(precision == 1.0)
        assert recall[-1] == 1.0

    def test_all_false_positives(self) -> None:
        """All wrong predictions have precision=0."""
        predictions = [
            {"class": 0, "confidence": 0.9, "box": np.array([10, 10, 20, 20], dtype=np.float64)},
        ]
        ground_truths = [
            {"class": 0, "box": np.array([100, 100, 150, 150], dtype=np.float64)},
        ]

        precision, recall, thresholds = calculate_precision_recall_curve(predictions, ground_truths)

        assert precision[0] == 0.0
        assert recall[0] == 0.0

    def test_mixed_predictions(self) -> None:
        """Mixed TP/FP predictions have decreasing precision."""
        predictions = [
            {
                "class": 0,
                "confidence": 0.9,
                "box": np.array([10, 10, 50, 50], dtype=np.float64),
            },  # TP
            {
                "class": 0,
                "confidence": 0.7,
                "box": np.array([200, 200, 250, 250], dtype=np.float64),
            },  # FP
        ]
        ground_truths = [
            {"class": 0, "box": np.array([10, 10, 50, 50], dtype=np.float64)},
        ]

        precision, recall, thresholds = calculate_precision_recall_curve(predictions, ground_truths)

        assert precision[0] == 1.0  # First is TP: 1/1
        assert precision[1] == 0.5  # Second includes FP: 1/2

    def test_confidence_ordering(self) -> None:
        """Predictions are sorted by confidence in descending order."""
        predictions = [
            {"class": 0, "confidence": 0.5, "box": np.array([10, 10, 50, 50], dtype=np.float64)},
            {"class": 0, "confidence": 0.9, "box": np.array([10, 10, 50, 50], dtype=np.float64)},
            {"class": 0, "confidence": 0.7, "box": np.array([10, 10, 50, 50], dtype=np.float64)},
        ]
        ground_truths = [
            {"class": 0, "box": np.array([10, 10, 50, 50], dtype=np.float64)},
        ]

        _, _, thresholds = calculate_precision_recall_curve(predictions, ground_truths)

        # Thresholds should be in descending order
        assert thresholds[0] == 0.9
        assert thresholds[1] == 0.7
        assert thresholds[2] == 0.5


# --- TestCalculateAveragePrecision ---


class TestCalculateAveragePrecision:
    """Tests for calculate_average_precision function."""

    def test_perfect_precision(self) -> None:
        """All precision=1 gives AP close to 1."""
        precision = np.array([1.0, 1.0, 1.0, 1.0])
        recall = np.array([0.25, 0.5, 0.75, 1.0])

        ap = calculate_average_precision(precision, recall)

        assert ap > 0.99

    def test_zero_precision(self) -> None:
        """All precision=0 gives AP of 0."""
        precision = np.array([0.0, 0.0, 0.0, 0.0])
        recall = np.array([0.25, 0.5, 0.75, 1.0])

        ap = calculate_average_precision(precision, recall)

        # Due to sentinel values (precision starts with 1), AP is not exactly 0
        assert ap < 0.02

    def test_monotonic_enforcement(self) -> None:
        """Non-monotonic precision is corrected."""
        precision = np.array([0.9, 0.8, 0.95, 0.7])  # Non-monotonic
        recall = np.array([0.25, 0.5, 0.75, 1.0])

        ap = calculate_average_precision(precision, recall)

        # Should complete without error and give reasonable AP
        assert 0.0 <= ap <= 1.0

    def test_typical_curve(self) -> None:
        """Typical decreasing precision curve gives expected AP range."""
        precision = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
        recall = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

        ap = calculate_average_precision(precision, recall)

        # Should be between 0.6 and 0.9 for this curve
        assert 0.6 <= ap <= 0.9

    def test_empty_arrays(self) -> None:
        """Empty arrays are handled gracefully."""
        precision = np.array([])
        recall = np.array([])

        ap = calculate_average_precision(precision, recall)

        # With sentinel values [1] and [0], AP calculation should work
        assert ap >= 0.0


# --- TestGenerateConfusionMatrix ---


class TestGenerateConfusionMatrix:
    """Tests for generate_confusion_matrix function."""

    def test_empty_inputs(self) -> None:
        """Empty inputs return zero matrix."""
        matrix = generate_confusion_matrix([], [], num_classes=2)

        # Shape: (num_classes + 1, num_classes + 1) for background
        assert matrix.shape == (3, 3)
        assert np.sum(matrix) == 0

    def test_single_class_correct(self) -> None:
        """Single correct prediction updates diagonal."""
        box = np.array([10, 10, 50, 50], dtype=np.float64)
        predictions = [{"class": 0, "box": box}]
        ground_truths = [{"class": 0, "box": box}]

        matrix = generate_confusion_matrix(predictions, ground_truths, num_classes=2)

        assert matrix[0, 0] == 1  # Correct prediction
        assert np.sum(matrix) == 1

    def test_multi_class(self) -> None:
        """Multiple classes with correct predictions."""
        predictions = [
            {"class": 0, "box": np.array([10, 10, 50, 50], dtype=np.float64)},
            {"class": 1, "box": np.array([60, 60, 100, 100], dtype=np.float64)},
        ]
        ground_truths = [
            {"class": 0, "box": np.array([10, 10, 50, 50], dtype=np.float64)},
            {"class": 1, "box": np.array([60, 60, 100, 100], dtype=np.float64)},
        ]

        matrix = generate_confusion_matrix(predictions, ground_truths, num_classes=2)

        assert matrix[0, 0] == 1
        assert matrix[1, 1] == 1

    def test_false_positives_as_background(self) -> None:
        """Unmatched predictions are counted as false positives."""
        predictions = [
            {"class": 0, "box": np.array([200, 200, 250, 250], dtype=np.float64)},  # No matching GT
        ]
        ground_truths = [
            {"class": 0, "box": np.array([10, 10, 50, 50], dtype=np.float64)},
        ]

        matrix = generate_confusion_matrix(predictions, ground_truths, num_classes=2)

        # FP: background row, predicted class column
        assert matrix[2, 0] == 1  # Background predicting class 0
        # FN: GT class, background column
        assert matrix[0, 2] == 1  # Missed GT

    def test_false_negatives_as_background(self) -> None:
        """Unmatched ground truths are counted as false negatives."""
        predictions: list[dict[str, Any]] = []
        ground_truths = [
            {"class": 1, "box": np.array([10, 10, 50, 50], dtype=np.float64)},
        ]

        matrix = generate_confusion_matrix(predictions, ground_truths, num_classes=2)

        # FN: GT class row, background column
        assert matrix[1, 2] == 1

    def test_class_confusion(self) -> None:
        """Wrong class prediction records confusion."""
        box = np.array([10, 10, 50, 50], dtype=np.float64)
        predictions = [{"class": 1, "box": box}]  # Predicted class 1
        ground_truths = [{"class": 0, "box": box}]  # Actual class 0

        matrix = generate_confusion_matrix(predictions, ground_truths, num_classes=2)

        assert matrix[0, 1] == 1  # GT class 0, predicted class 1


# --- TestEvaluator ---


class TestEvaluator:
    """Tests for Evaluator class."""

    def test_init(self, mocker: MockerFixture) -> None:
        """Evaluator initialization stores parameters."""
        mock_model = create_mock_yolo(mocker)

        evaluator = Evaluator(
            model=mock_model,
            data_yaml="/path/to/data.yaml",
            class_names=["class_a", "class_b"],
            device="cuda:0",
        )

        assert evaluator.model is mock_model
        assert evaluator.data_yaml == "/path/to/data.yaml"
        assert evaluator.class_names == ["class_a", "class_b"]
        assert evaluator.device == "cuda:0"

    def test_evaluate(self, mocker: MockerFixture) -> None:
        """evaluate extracts metrics from model.val() results."""
        mock_model = create_mock_yolo(mocker)
        mock_results = create_mock_validation_results(mocker)
        mock_model.val.return_value = mock_results

        evaluator = Evaluator(
            model=mock_model,
            data_yaml="/path/to/data.yaml",
            class_names=["class_a", "class_b"],
        )
        results = evaluator.evaluate(verbose=False)

        assert results.mAP50 == 0.85
        assert results.mAP50_95 == 0.72
        assert results.precision == 0.88
        assert results.recall == 0.80
        # F1 = 2 * 0.88 * 0.80 / (0.88 + 0.80) = 0.8380952...
        assert abs(results.f1 - 0.838095) < 0.001
        assert len(results.class_metrics) == 2
        assert results.class_metrics[0].ap50 == 0.90
        assert results.preprocess_time == 1.5
        assert results.inference_time == 10.2

    def test_evaluate_calls_model_val_with_params(self, mocker: MockerFixture) -> None:
        """evaluate passes correct parameters to model.val()."""
        mock_model = create_mock_yolo(mocker)
        mock_results = create_mock_validation_results(mocker)
        mock_model.val.return_value = mock_results

        evaluator = Evaluator(
            model=mock_model,
            data_yaml="/path/to/data.yaml",
            class_names=["class_a"],
            device="cuda:1",
        )
        evaluator.evaluate(
            split="test",
            conf=0.5,
            iou=0.6,
            batch_size=32,
            image_size=1280,
            verbose=False,
        )

        mock_model.val.assert_called_once_with(
            data="/path/to/data.yaml",
            split="test",
            batch=32,
            imgsz=1280,
            conf=0.5,
            iou=0.6,
            device="cuda:1",
            plots=True,
            save_json=True,
            verbose=False,
        )

    def test_evaluate_with_missing_confusion_matrix(self, mocker: MockerFixture) -> None:
        """evaluate handles missing confusion matrix gracefully."""
        mock_model = create_mock_yolo(mocker)
        mock_results = create_mock_validation_results(mocker)
        del mock_results.confusion_matrix  # Remove attribute

        mock_model.val.return_value = mock_results

        evaluator = Evaluator(
            model=mock_model,
            data_yaml="/path/to/data.yaml",
            class_names=["class_a"],
        )
        results = evaluator.evaluate(verbose=False)

        assert results.confusion_matrix is None

    def test_evaluate_with_missing_speed(self, mocker: MockerFixture) -> None:
        """evaluate handles missing speed metrics gracefully."""
        mock_model = create_mock_yolo(mocker)
        mock_results = create_mock_validation_results(mocker)
        del mock_results.speed

        mock_model.val.return_value = mock_results

        evaluator = Evaluator(
            model=mock_model,
            data_yaml="/path/to/data.yaml",
            class_names=["class_a"],
        )
        results = evaluator.evaluate(verbose=False)

        assert results.preprocess_time == 0.0
        assert results.inference_time == 0.0
        assert results.postprocess_time == 0.0

    def test_print_results(self, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]) -> None:
        """_print_results outputs formatted results."""
        mock_model = create_mock_yolo(mocker)
        evaluator = Evaluator(
            model=mock_model,
            data_yaml="/path/to/data.yaml",
            class_names=["class_a"],
        )

        results = EvaluationResults(
            mAP50=0.85,
            mAP50_95=0.72,
            precision=0.88,
            recall=0.80,
            f1=0.84,
            preprocess_time=1.5,
            inference_time=10.2,
            postprocess_time=2.3,
        )

        evaluator._print_results(results)

        captured = capsys.readouterr()
        assert "EVALUATION RESULTS" in captured.out
        assert "mAP@50:" in captured.out
        assert "0.8500" in captured.out
        assert "Precision:" in captured.out
        assert "Speed Metrics:" in captured.out


# --- TestAnalyzeErrors ---


class TestAnalyzeErrors:
    """Tests for analyze_errors function."""

    def test_basic_analysis(self, mocker: MockerFixture) -> None:
        """analyze_errors returns expected structure."""
        mock_model = create_mock_yolo(mocker)

        errors = analyze_errors(
            model=mock_model,
            test_images=["/path/to/img1.jpg", "/path/to/img2.jpg"],
            class_names=["class_a", "class_b"],
            conf=0.3,
        )

        assert "false_positives_by_class" in errors
        assert "false_negatives_by_class" in errors
        assert "confusion_pairs" in errors
        assert "low_confidence_correct" in errors
        assert "high_confidence_wrong" in errors
        assert "small_object_misses" in errors
        assert "large_object_misses" in errors
        assert errors["false_positives_by_class"]["class_a"] == 0
        assert errors["false_positives_by_class"]["class_b"] == 0

    def test_calls_model_predict(self, mocker: MockerFixture) -> None:
        """analyze_errors calls model.predict with correct params."""
        mock_model = create_mock_yolo(mocker)
        test_images = ["/path/to/img1.jpg"]

        analyze_errors(
            model=mock_model,
            test_images=test_images,
            class_names=["class_a"],
            conf=0.4,
        )

        mock_model.predict.assert_called_once_with(test_images, conf=0.4, verbose=False)

    def test_empty_images_list(self, mocker: MockerFixture) -> None:
        """analyze_errors handles empty image list."""
        mock_model = create_mock_yolo(mocker)

        errors = analyze_errors(
            model=mock_model,
            test_images=[],
            class_names=["class_a"],
        )

        assert errors is not None
        mock_model.predict.assert_called_once_with([], conf=0.25, verbose=False)


# --- TestFindOptimalConfidence ---


class TestFindOptimalConfidence:
    """Tests for find_optimal_confidence function."""

    def test_find_optimal_f1(self, mocker: MockerFixture) -> None:
        """find_optimal_confidence finds best F1 threshold."""
        mock_model = create_mock_yolo(mocker)

        # Create results with varying metrics for different conf values
        results_list = []
        for i in range(5):
            mock_result = mocker.MagicMock()
            # Simulate F1 peaking at conf=0.5
            conf = 0.1 + i * 0.2
            if conf == 0.5:
                mock_result.box.mp = 0.9
                mock_result.box.mr = 0.9
            else:
                mock_result.box.mp = 0.7
                mock_result.box.mr = 0.7
            mock_result.box.map50 = 0.8
            mock_result.box.map = 0.7
            results_list.append(mock_result)

        mock_model.val.side_effect = results_list

        best_conf, best_value = find_optimal_confidence(
            model=mock_model,
            data_yaml="/path/to/data.yaml",
            conf_range=(0.1, 0.9),
            num_steps=5,
            metric="f1",
        )

        assert 0.1 <= best_conf <= 0.9
        assert best_value > 0

    def test_find_optimal_precision(self, mocker: MockerFixture) -> None:
        """find_optimal_confidence finds best precision threshold."""
        mock_model = create_mock_yolo(mocker)

        mock_result = mocker.MagicMock()
        mock_result.box.mp = 0.95
        mock_result.box.mr = 0.70
        mock_result.box.map50 = 0.85
        mock_result.box.map = 0.72
        mock_model.val.return_value = mock_result

        best_conf, best_value = find_optimal_confidence(
            model=mock_model,
            data_yaml="/path/to/data.yaml",
            num_steps=3,
            metric="precision",
        )

        assert best_value == 0.95

    def test_find_optimal_recall(self, mocker: MockerFixture) -> None:
        """find_optimal_confidence finds best recall threshold."""
        mock_model = create_mock_yolo(mocker)

        mock_result = mocker.MagicMock()
        mock_result.box.mp = 0.85
        mock_result.box.mr = 0.92
        mock_result.box.map50 = 0.80
        mock_result.box.map = 0.70
        mock_model.val.return_value = mock_result

        best_conf, best_value = find_optimal_confidence(
            model=mock_model,
            data_yaml="/path/to/data.yaml",
            num_steps=3,
            metric="recall",
        )

        assert best_value == 0.92

    def test_find_optimal_map50(self, mocker: MockerFixture) -> None:
        """find_optimal_confidence finds best mAP50 threshold."""
        mock_model = create_mock_yolo(mocker)

        mock_result = mocker.MagicMock()
        mock_result.box.mp = 0.85
        mock_result.box.mr = 0.80
        mock_result.box.map50 = 0.88
        mock_result.box.map = 0.75
        mock_model.val.return_value = mock_result

        best_conf, best_value = find_optimal_confidence(
            model=mock_model,
            data_yaml="/path/to/data.yaml",
            num_steps=3,
            metric="map50",
        )

        assert best_value == 0.88

    def test_find_optimal_default_metric(self, mocker: MockerFixture) -> None:
        """find_optimal_confidence defaults to mAP50-95."""
        mock_model = create_mock_yolo(mocker)

        mock_result = mocker.MagicMock()
        mock_result.box.mp = 0.85
        mock_result.box.mr = 0.80
        mock_result.box.map50 = 0.88
        mock_result.box.map = 0.75
        mock_model.val.return_value = mock_result

        best_conf, best_value = find_optimal_confidence(
            model=mock_model,
            data_yaml="/path/to/data.yaml",
            num_steps=3,
            metric="other",  # Unknown metric defaults to map
        )

        assert best_value == 0.75


# --- TestBenchmarkModel ---


class TestBenchmarkModel:
    """Tests for benchmark_model function."""

    def test_benchmark_with_cuda(self, mocker: MockerFixture) -> None:
        """benchmark_model runs on CUDA with synchronization."""
        mock_model = create_mock_yolo(mocker)

        # Mock torch module (imported inside the function)
        mock_torch = mocker.MagicMock()
        mock_torch.randn.return_value = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"torch": mock_torch})

        # Mock time.perf_counter
        mocker.patch("time.perf_counter", side_effect=[i * 0.01 for i in range(1000)])

        results = benchmark_model(
            model=mock_model,
            batch_sizes=[1],
            device="cuda",
            warmup_runs=2,
            test_runs=5,
        )

        assert "batch_1" in results
        assert "avg_ms" in results["batch_1"]
        assert "std_ms" in results["batch_1"]
        assert "fps" in results["batch_1"]
        assert "ms_per_image" in results["batch_1"]
        # CUDA sync should be called: 1 (after warmup) + 5 (test runs)
        assert mock_torch.cuda.synchronize.call_count == 6

    def test_benchmark_cpu_no_sync(self, mocker: MockerFixture) -> None:
        """benchmark_model on CPU skips CUDA synchronization."""
        mock_model = create_mock_yolo(mocker)

        # Mock torch module
        mock_torch = mocker.MagicMock()
        mock_torch.randn.return_value = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"torch": mock_torch})

        # Mock time.perf_counter
        mocker.patch("time.perf_counter", side_effect=[i * 0.01 for i in range(1000)])

        benchmark_model(
            model=mock_model,
            batch_sizes=[1],
            device="cpu",
            warmup_runs=2,
            test_runs=5,
        )

        mock_torch.cuda.synchronize.assert_not_called()

    def test_benchmark_multiple_batch_sizes(self, mocker: MockerFixture) -> None:
        """benchmark_model tests all specified batch sizes."""
        mock_model = create_mock_yolo(mocker)

        # Mock torch module
        mock_torch = mocker.MagicMock()
        mock_torch.randn.return_value = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"torch": mock_torch})

        # Mock time.perf_counter
        mocker.patch("time.perf_counter", side_effect=[i * 0.01 for i in range(10000)])

        results = benchmark_model(
            model=mock_model,
            batch_sizes=[1, 8, 16],
            device="cuda",
            warmup_runs=2,
            test_runs=5,
        )

        assert "batch_1" in results
        assert "batch_8" in results
        assert "batch_16" in results

    def test_benchmark_default_batch_sizes(self, mocker: MockerFixture) -> None:
        """benchmark_model uses default batch sizes when None."""
        mock_model = create_mock_yolo(mocker)

        # Mock torch module
        mock_torch = mocker.MagicMock()
        mock_torch.randn.return_value = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"torch": mock_torch})

        # Mock time.perf_counter
        mocker.patch("time.perf_counter", side_effect=[i * 0.01 for i in range(10000)])

        results = benchmark_model(
            model=mock_model,
            batch_sizes=None,  # Use default
            device="cuda",
            warmup_runs=1,
            test_runs=2,
        )

        # Default is [1, 8, 16, 32]
        assert "batch_1" in results
        assert "batch_8" in results
        assert "batch_16" in results
        assert "batch_32" in results

    def test_benchmark_timing_positive(self, mocker: MockerFixture) -> None:
        """benchmark_model returns positive timing values."""
        mock_model = create_mock_yolo(mocker)

        # Mock torch module
        mock_torch = mocker.MagicMock()
        mock_torch.randn.return_value = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"torch": mock_torch})

        # Simulate 10ms per inference
        mocker.patch("time.perf_counter", side_effect=[i * 0.01 for i in range(1000)])

        results = benchmark_model(
            model=mock_model,
            batch_sizes=[1],
            device="cuda",
            warmup_runs=2,
            test_runs=5,
        )

        assert results["batch_1"]["avg_ms"] > 0
        assert results["batch_1"]["fps"] > 0
        assert results["batch_1"]["ms_per_image"] > 0
