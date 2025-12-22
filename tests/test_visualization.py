"""Tests for visualization module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from screencropnet_yolov11.visualization import (
    ConfusionMatrixVisualizer,
    DatasetVisualizer,
    DetectionVisualizer,
    ResultsDashboard,
    TrainingVisualizer,
    create_comparison_plot,
)


def create_sample_training_history(epochs: int = 10) -> dict[str, list[float]]:
    """Create sample training history with all metrics."""
    return {
        "train_loss": [1.0 - (i * 0.08) for i in range(epochs)],
        "val_loss": [1.1 - (i * 0.07) for i in range(epochs)],
        "mAP50": [0.5 + (i * 0.04) for i in range(epochs)],
        "mAP50_95": [0.4 + (i * 0.03) for i in range(epochs)],
        "precision": [0.6 + (i * 0.03) for i in range(epochs)],
        "recall": [0.5 + (i * 0.04) for i in range(epochs)],
        "f1": [0.55 + (i * 0.035) for i in range(epochs)],
        "learning_rate": [0.01 * (0.9**i) for i in range(epochs)],
    }


def create_sample_loss_history(epochs: int = 10) -> dict[str, list[float]]:
    """Create sample loss component history."""
    return {
        "box_loss": [0.5 - (i * 0.04) for i in range(epochs)],
        "cls_loss": [0.3 - (i * 0.02) for i in range(epochs)],
        "dfl_loss": [0.2 - (i * 0.015) for i in range(epochs)],
    }


def create_sample_detections(n: int = 3) -> list[dict[str, Any]]:
    """Create sample detection dictionaries."""
    return [
        {
            "class_id": i % 2,
            "confidence": 0.9 - (i * 0.1),
            "bbox": [10 + i * 20, 10 + i * 10, 100 + i * 20, 100 + i * 10],
        }
        for i in range(n)
    ]


def create_sample_evaluation_results(
    num_classes: int = 3,
) -> dict[str, Any]:
    """Create sample evaluation results dictionary."""
    class_names = [f"class_{i}" for i in range(num_classes)]
    return {
        "per_class": [
            {
                "class_name": name,
                "ap50": 0.8 + (i * 0.05),
                "precision": 0.75 + (i * 0.05),
                "recall": 0.7 + (i * 0.05),
            }
            for i, name in enumerate(class_names)
        ],
        "overall": {
            "mAP50": 0.85,
            "mAP50_95": 0.72,
            "precision": 0.80,
            "recall": 0.75,
            "f1": 0.77,
        },
        "confusion_matrix": np.array([[10, 2, 1], [1, 15, 2], [0, 1, 12]]),
    }


def create_test_image_array(width: int = 200, height: int = 150, channels: int = 3) -> Any:
    """Create a simple test image array (BGR format)."""
    return np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)


class TestTrainingVisualizer:
    """Tests for TrainingVisualizer class."""

    def test_init_creates_output_directory(self, tmp_path: Path) -> None:
        """Output directory is created on initialization."""
        output_dir = tmp_path / "training_plots"
        visualizer = TrainingVisualizer(str(output_dir))

        assert output_dir.exists()
        assert visualizer.output_dir == output_dir
        plt.close("all")

    def test_init_existing_directory(self, tmp_path: Path) -> None:
        """Existing output directory is handled correctly."""
        output_dir = tmp_path / "existing"
        output_dir.mkdir()
        visualizer = TrainingVisualizer(str(output_dir))

        assert output_dir.exists()
        assert visualizer.output_dir == output_dir
        plt.close("all")

    def test_plot_training_curves_all_metrics(self, tmp_path: Path) -> None:
        """Training curves are plotted with all metrics."""
        visualizer = TrainingVisualizer(str(tmp_path))
        history = create_sample_training_history()

        fig = visualizer.plot_training_curves(history)

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 4
        plt.close("all")

    def test_plot_training_curves_minimal_metrics(self, tmp_path: Path) -> None:
        """Training curves work with minimal metrics."""
        visualizer = TrainingVisualizer(str(tmp_path))
        history = {"train_loss": [1.0, 0.8, 0.6]}

        fig = visualizer.plot_training_curves(history)

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 4
        plt.close("all")

    def test_plot_training_curves_single_epoch(self, tmp_path: Path) -> None:
        """Training curves handle single epoch data."""
        visualizer = TrainingVisualizer(str(tmp_path))
        history = {"train_loss": [0.5], "val_loss": [0.6]}

        fig = visualizer.plot_training_curves(history)

        assert isinstance(fig, Figure)
        plt.close("all")

    def test_plot_training_curves_empty_history(self, tmp_path: Path) -> None:
        """Training curves handle empty history."""
        visualizer = TrainingVisualizer(str(tmp_path))
        history: dict[str, list[float]] = {}

        fig = visualizer.plot_training_curves(history)

        assert isinstance(fig, Figure)
        plt.close("all")

    def test_plot_training_curves_saves_figure(self, tmp_path: Path) -> None:
        """Training curves are saved when save_path is provided."""
        visualizer = TrainingVisualizer(str(tmp_path))
        history = create_sample_training_history()
        save_path = tmp_path / "training_curves.png"

        fig = visualizer.plot_training_curves(history, save_path=str(save_path))

        assert save_path.exists()
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_plot_loss_components_all_losses(self, tmp_path: Path) -> None:
        """Loss components are plotted with all loss types."""
        visualizer = TrainingVisualizer(str(tmp_path))
        history = create_sample_loss_history()

        fig = visualizer.plot_loss_components(history)

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1
        plt.close("all")

    def test_plot_loss_components_partial_losses(self, tmp_path: Path) -> None:
        """Loss components work with partial loss data."""
        visualizer = TrainingVisualizer(str(tmp_path))
        history = {"box_loss": [0.5, 0.4, 0.3]}

        fig = visualizer.plot_loss_components(history)

        assert isinstance(fig, Figure)
        plt.close("all")

    def test_plot_loss_components_saves_figure(self, tmp_path: Path) -> None:
        """Loss components are saved when save_path is provided."""
        visualizer = TrainingVisualizer(str(tmp_path))
        history = create_sample_loss_history()
        save_path = tmp_path / "loss_components.png"

        fig = visualizer.plot_loss_components(history, save_path=str(save_path))

        assert save_path.exists()
        assert isinstance(fig, Figure)
        plt.close("all")


class TestConfusionMatrixVisualizer:
    """Tests for ConfusionMatrixVisualizer class."""

    def test_plot_confusion_matrix_normalized(self) -> None:
        """Confusion matrix is normalized correctly."""
        matrix = np.array([[10, 2], [3, 15]])
        class_names = ["cat", "dog"]

        fig = ConfusionMatrixVisualizer.plot_confusion_matrix(matrix, class_names, normalize=True)

        assert isinstance(fig, Figure)
        plt.close("all")

    def test_plot_confusion_matrix_unnormalized(self) -> None:
        """Confusion matrix displays raw counts when normalize=False."""
        matrix = np.array([[10, 2], [3, 15]])
        class_names = ["cat", "dog"]

        fig = ConfusionMatrixVisualizer.plot_confusion_matrix(matrix, class_names, normalize=False)

        assert isinstance(fig, Figure)
        plt.close("all")

    def test_plot_confusion_matrix_zero_row(self) -> None:
        """Confusion matrix handles rows with all zeros (division by zero)."""
        matrix = np.array([[10, 2], [0, 0], [1, 5]])
        class_names = ["cat", "dog", "bird"]

        fig = ConfusionMatrixVisualizer.plot_confusion_matrix(matrix, class_names, normalize=True)

        assert isinstance(fig, Figure)
        plt.close("all")

    def test_plot_confusion_matrix_single_class(self) -> None:
        """Confusion matrix handles single class."""
        matrix = np.array([[10]])
        class_names = ["object"]

        fig = ConfusionMatrixVisualizer.plot_confusion_matrix(matrix, class_names)

        assert isinstance(fig, Figure)
        plt.close("all")

    def test_plot_confusion_matrix_saves_figure(self, tmp_path: Path) -> None:
        """Confusion matrix is saved when save_path is provided."""
        matrix = np.array([[10, 2], [3, 15]])
        class_names = ["cat", "dog"]
        save_path = tmp_path / "confusion_matrix.png"

        fig = ConfusionMatrixVisualizer.plot_confusion_matrix(
            matrix, class_names, save_path=str(save_path)
        )

        assert save_path.exists()
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_plot_confusion_matrix_large(self) -> None:
        """Confusion matrix handles larger matrices."""
        n = 10
        matrix = np.random.randint(0, 100, (n, n))
        class_names = [f"class_{i}" for i in range(n)]

        fig = ConfusionMatrixVisualizer.plot_confusion_matrix(matrix, class_names)

        assert isinstance(fig, Figure)
        plt.close("all")


class TestDetectionVisualizer:
    """Tests for DetectionVisualizer class."""

    def test_init_generates_colors(self) -> None:
        """Colors are generated for each class."""
        class_names = ["cat", "dog", "bird"]
        visualizer = DetectionVisualizer(class_names)

        assert len(visualizer.colors) == 3
        assert visualizer.class_names == class_names
        plt.close("all")

    def test_generate_colors_distinct(self) -> None:
        """Generated colors are distinct."""
        visualizer = DetectionVisualizer(["a", "b", "c", "d", "e"])
        colors = visualizer.colors

        assert len(colors) == 5
        assert len(set(colors)) == 5
        plt.close("all")

    def test_draw_detections_single_box(self) -> None:
        """Single detection is drawn correctly."""
        visualizer = DetectionVisualizer(["cat", "dog"])
        image = create_test_image_array()
        detections = [{"class_id": 0, "confidence": 0.9, "bbox": [10, 10, 100, 100]}]

        result = visualizer.draw_detections(image, detections)

        assert result.shape == image.shape
        assert not np.array_equal(result, image)
        plt.close("all")

    def test_draw_detections_multiple_boxes(self) -> None:
        """Multiple detections are drawn correctly."""
        visualizer = DetectionVisualizer(["cat", "dog"])
        image = create_test_image_array()
        detections = create_sample_detections(3)

        result = visualizer.draw_detections(image, detections)

        assert result.shape == image.shape
        assert not np.array_equal(result, image)
        plt.close("all")

    def test_draw_detections_out_of_bounds_class_id(self) -> None:
        """Out of bounds class_id falls back to class_N label."""
        visualizer = DetectionVisualizer(["cat"])
        image = create_test_image_array()
        detections = [{"class_id": 5, "confidence": 0.8, "bbox": [10, 10, 50, 50]}]

        result = visualizer.draw_detections(image, detections)

        assert result.shape == image.shape
        plt.close("all")

    def test_draw_detections_no_confidence(self) -> None:
        """Detections are drawn without confidence when show_confidence=False."""
        visualizer = DetectionVisualizer(["cat", "dog"])
        image = create_test_image_array()
        detections = [{"class_id": 0, "confidence": 0.9, "bbox": [10, 10, 100, 100]}]

        result = visualizer.draw_detections(image, detections, show_confidence=False)

        assert result.shape == image.shape
        plt.close("all")

    def test_draw_detections_empty_list(self) -> None:
        """Empty detections list returns original image."""
        visualizer = DetectionVisualizer(["cat", "dog"])
        image = create_test_image_array()

        result = visualizer.draw_detections(image, [])

        assert np.array_equal(result, image)
        plt.close("all")

    def test_draw_detections_custom_line_width(self) -> None:
        """Custom line width is applied."""
        visualizer = DetectionVisualizer(["cat"])
        image = create_test_image_array()
        detections = [{"class_id": 0, "confidence": 0.9, "bbox": [10, 10, 100, 100]}]

        result = visualizer.draw_detections(image, detections, line_width=5)

        assert result.shape == image.shape
        assert not np.array_equal(result, image)
        plt.close("all")

    def test_plot_detection_grid_basic(self) -> None:
        """Detection grid is created correctly."""
        visualizer = DetectionVisualizer(["cat", "dog"])
        images = [create_test_image_array() for _ in range(4)]
        results = [create_sample_detections(2) for _ in range(4)]

        fig = visualizer.plot_detection_grid(images, results)

        assert isinstance(fig, Figure)
        plt.close("all")

    def test_plot_detection_grid_custom_cols(self) -> None:
        """Detection grid respects custom column count."""
        visualizer = DetectionVisualizer(["cat", "dog"])
        images = [create_test_image_array() for _ in range(6)]
        results = [create_sample_detections(1) for _ in range(6)]

        fig = visualizer.plot_detection_grid(images, results, cols=2)

        assert isinstance(fig, Figure)
        plt.close("all")

    def test_plot_detection_grid_saves_figure(self, tmp_path: Path) -> None:
        """Detection grid is saved when save_path is provided."""
        visualizer = DetectionVisualizer(["cat", "dog"])
        images = [create_test_image_array() for _ in range(2)]
        results = [create_sample_detections(1) for _ in range(2)]
        save_path = tmp_path / "detection_grid.png"

        fig = visualizer.plot_detection_grid(images, results, save_path=str(save_path))

        assert save_path.exists()
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_plot_detection_grid_mismatched_lengths(self) -> None:
        """Mismatched images and results lists raise ValueError."""
        visualizer = DetectionVisualizer(["cat", "dog"])
        images = [create_test_image_array() for _ in range(3)]
        results = [create_sample_detections(1) for _ in range(2)]

        with pytest.raises(ValueError):
            visualizer.plot_detection_grid(images, results)
        plt.close("all")

    def test_plot_detection_grid_single_image(self) -> None:
        """Detection grid handles single image."""
        visualizer = DetectionVisualizer(["cat"])
        images = [create_test_image_array()]
        results = [[{"class_id": 0, "confidence": 0.9, "bbox": [10, 10, 50, 50]}]]

        fig = visualizer.plot_detection_grid(images, results)

        assert isinstance(fig, Figure)
        plt.close("all")


class TestDatasetVisualizer:
    """Tests for DatasetVisualizer class."""

    def test_plot_class_distribution_basic(self) -> None:
        """Class distribution bar chart is created."""
        class_counts = {"cat": 100, "dog": 150, "bird": 80}

        fig = DatasetVisualizer.plot_class_distribution(class_counts)

        assert isinstance(fig, Figure)
        plt.close("all")

    def test_plot_class_distribution_single_class(self) -> None:
        """Class distribution handles single class."""
        class_counts = {"object": 50}

        fig = DatasetVisualizer.plot_class_distribution(class_counts)

        assert isinstance(fig, Figure)
        plt.close("all")

    def test_plot_class_distribution_saves_figure(self, tmp_path: Path) -> None:
        """Class distribution is saved when save_path is provided."""
        class_counts = {"cat": 100, "dog": 150}
        save_path = tmp_path / "class_distribution.png"

        fig = DatasetVisualizer.plot_class_distribution(class_counts, save_path=str(save_path))

        assert save_path.exists()
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_plot_bbox_size_distribution_basic(self) -> None:
        """Bbox size distribution plots all three histograms."""
        widths = [50.0, 100.0, 150.0, 80.0, 120.0]
        heights = [40.0, 90.0, 140.0, 70.0, 110.0]

        fig = DatasetVisualizer.plot_bbox_size_distribution(widths, heights)

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 3
        plt.close("all")

    def test_plot_bbox_size_distribution_zero_height(self) -> None:
        """Aspect ratio calculation handles zero height."""
        widths = [50.0, 100.0, 150.0]
        heights = [40.0, 0.0, 100.0]

        fig = DatasetVisualizer.plot_bbox_size_distribution(widths, heights)

        assert isinstance(fig, Figure)
        plt.close("all")

    def test_plot_bbox_size_distribution_saves_figure(self, tmp_path: Path) -> None:
        """Bbox size distribution is saved when save_path is provided."""
        widths = [50.0, 100.0, 150.0]
        heights = [40.0, 90.0, 140.0]
        save_path = tmp_path / "bbox_distribution.png"

        fig = DatasetVisualizer.plot_bbox_size_distribution(
            widths, heights, save_path=str(save_path)
        )

        assert save_path.exists()
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_plot_bbox_size_distribution_mismatched_lengths(self) -> None:
        """Mismatched widths and heights raise ValueError."""
        widths = [50.0, 100.0, 150.0]
        heights = [40.0, 90.0]

        with pytest.raises(ValueError):
            DatasetVisualizer.plot_bbox_size_distribution(widths, heights)
        plt.close("all")

    def test_plot_image_size_distribution_basic(self) -> None:
        """Image size distribution creates 2D histogram."""
        sizes = [(640, 480), (800, 600), (1920, 1080), (640, 480)]

        fig = DatasetVisualizer.plot_image_size_distribution(sizes)

        assert isinstance(fig, Figure)
        plt.close("all")

    def test_plot_image_size_distribution_saves_figure(self, tmp_path: Path) -> None:
        """Image size distribution is saved when save_path is provided."""
        sizes = [(640, 480), (800, 600), (1920, 1080)]
        save_path = tmp_path / "image_sizes.png"

        fig = DatasetVisualizer.plot_image_size_distribution(sizes, save_path=str(save_path))

        assert save_path.exists()
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_plot_image_size_distribution_single_size(self) -> None:
        """Image size distribution handles single image size."""
        sizes = [(640, 480)]

        fig = DatasetVisualizer.plot_image_size_distribution(sizes)

        assert isinstance(fig, Figure)
        plt.close("all")


class TestResultsDashboard:
    """Tests for ResultsDashboard class."""

    def test_init_creates_output_directory(self, tmp_path: Path) -> None:
        """Output directory is created on initialization."""
        output_dir = tmp_path / "dashboard"
        dashboard = ResultsDashboard(str(output_dir))

        assert output_dir.exists()
        assert dashboard.output_dir == output_dir
        plt.close("all")

    def test_create_dashboard_full_data(self, tmp_path: Path) -> None:
        """Dashboard is created with complete data."""
        dashboard = ResultsDashboard(str(tmp_path))
        training_history = create_sample_training_history()
        evaluation_results = create_sample_evaluation_results()
        class_names = ["class_0", "class_1", "class_2"]

        fig = dashboard.create_dashboard(training_history, evaluation_results, class_names)

        assert isinstance(fig, Figure)
        plt.close("all")

    def test_create_dashboard_saves_figure(self, tmp_path: Path) -> None:
        """Dashboard is saved when save_path is provided."""
        dashboard = ResultsDashboard(str(tmp_path))
        training_history = create_sample_training_history()
        evaluation_results = create_sample_evaluation_results()
        class_names = ["class_0", "class_1", "class_2"]
        save_path = tmp_path / "dashboard.png"

        fig = dashboard.create_dashboard(
            training_history, evaluation_results, class_names, save_path=str(save_path)
        )

        assert save_path.exists()
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_create_dashboard_minimal_data(self, tmp_path: Path) -> None:
        """Dashboard handles minimal training history."""
        dashboard = ResultsDashboard(str(tmp_path))
        training_history = {"train_loss": [0.5, 0.4]}
        evaluation_results: dict[str, Any] = {}
        class_names = ["class_0"]

        fig = dashboard.create_dashboard(training_history, evaluation_results, class_names)

        assert isinstance(fig, Figure)
        plt.close("all")

    def test_create_dashboard_no_confusion_matrix(self, tmp_path: Path) -> None:
        """Dashboard handles missing confusion matrix."""
        dashboard = ResultsDashboard(str(tmp_path))
        training_history = create_sample_training_history()
        evaluation_results = {
            "per_class": [{"class_name": "cat", "ap50": 0.9, "precision": 0.8, "recall": 0.7}],
            "overall": {"mAP50": 0.85, "precision": 0.8, "recall": 0.75, "f1": 0.77},
        }
        class_names = ["cat"]

        fig = dashboard.create_dashboard(training_history, evaluation_results, class_names)

        assert isinstance(fig, Figure)
        plt.close("all")


class TestCreateComparisonPlot:
    """Tests for create_comparison_plot function."""

    def test_comparison_plot_single_config(self) -> None:
        """Comparison plot handles single configuration."""
        results = {"baseline": {"mAP50_95": 0.75}}

        fig = create_comparison_plot(results)

        assert isinstance(fig, Figure)
        plt.close("all")

    def test_comparison_plot_multiple_configs(self) -> None:
        """Comparison plot handles multiple configurations."""
        results = {
            "baseline": {"mAP50_95": 0.75},
            "augmented": {"mAP50_95": 0.82},
            "pretrained": {"mAP50_95": 0.88},
        }

        fig = create_comparison_plot(results)

        assert isinstance(fig, Figure)
        plt.close("all")

    def test_comparison_plot_highlights_best(self) -> None:
        """Best value is highlighted in gold."""
        results = {
            "config_a": {"mAP50_95": 0.5},
            "config_b": {"mAP50_95": 0.9},
            "config_c": {"mAP50_95": 0.7},
        }

        fig = create_comparison_plot(results)
        ax = fig.axes[0]
        bars = ax.patches

        best_bar = bars[1]
        assert best_bar.get_facecolor()[:3] == pytest.approx((1.0, 0.843, 0.0), abs=0.01)
        plt.close("all")

    def test_comparison_plot_custom_metric(self) -> None:
        """Custom metric name is used."""
        results = {
            "config_a": {"precision": 0.85},
            "config_b": {"precision": 0.90},
        }

        fig = create_comparison_plot(results, metric="precision")

        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert "precision" in ax.get_ylabel().lower()
        plt.close("all")

    def test_comparison_plot_saves_figure(self, tmp_path: Path) -> None:
        """Comparison plot is saved when save_path is provided."""
        results = {"config_a": {"mAP50_95": 0.8}}
        save_path = tmp_path / "comparison.png"

        fig = create_comparison_plot(results, save_path=str(save_path))

        assert save_path.exists()
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_comparison_plot_missing_metric(self) -> None:
        """Missing metric defaults to 0."""
        results = {
            "config_a": {"mAP50": 0.8},
            "config_b": {},
        }

        fig = create_comparison_plot(results, metric="mAP50_95")

        assert isinstance(fig, Figure)
        plt.close("all")

    def test_comparison_plot_many_configs(self) -> None:
        """Comparison plot handles many configurations."""
        results = {f"config_{i}": {"mAP50_95": 0.5 + i * 0.05} for i in range(10)}

        fig = create_comparison_plot(results)

        assert isinstance(fig, Figure)
        plt.close("all")
