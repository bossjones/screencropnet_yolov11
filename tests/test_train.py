"""Tests for train module."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from pytest_mock import MockerFixture

from screencropnet_yolov11.train import (
    create_visualizations,
    evaluate_model,
    export_model,
    load_config,
    load_dataset_from_roboflow,
    merge_config_with_args,
    parse_args,
    run_ablation_study,
    setup_logging,
    split_dataset_if_needed,
    train_model,
    validate_dataset,
)

# --- Helper Functions ---


def create_sample_config() -> dict[str, Any]:
    """Create a minimal valid configuration dict."""
    return {
        "dataset": {
            "path": "/data/dataset",
            "roboflow": {"enabled": False},
            "auto_split": False,
            "split_ratios": {"train": 0.7, "val": 0.2, "test": 0.1},
            "seed": 42,
        },
        "model": {
            "size": "m",
            "class_names": ["tweet", "retweet"],
            "weights": None,
        },
        "device": {
            "type": "auto",
            "multi_gpu": False,
            "gpu_ids": [0, 1],
        },
        "training": {
            "epochs": 100,
            "batch_size": 16,
            "image_size": 640,
            "learning_rate": 0.01,
            "optimizer": "SGD",
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3,
            "patience": 20,
            "amp": True,
            "workers": 8,
        },
        "logging": {
            "output_dir": "/output",
            "level": "INFO",
            "tensorboard": True,
            "wandb": {},
        },
        "inference": {
            "confidence": 0.25,
            "iou_threshold": 0.45,
        },
        "export": {
            "formats": ["pytorch", "onnx"],
            "onnx": {"dynamic": False, "simplify": True, "opset": 12},
            "quantization": {"enabled": False, "type": "fp16"},
        },
        "ablation": {"enabled": False},
    }


def create_mock_args(**overrides: Any) -> argparse.Namespace:
    """Create mock argparse.Namespace with defaults."""
    defaults = {
        "config": "config/config.yaml",
        "data": None,
        "epochs": None,
        "batch": None,
        "imgsz": None,
        "device": None,
        "workers": None,
        "output": None,
        "model_size": None,
        "resume": None,
        "validate_only": False,
        "eval_only": None,
        "export_only": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def create_mock_training_history(mocker: MockerFixture) -> MagicMock:
    """Create mock TrainingHistory with metrics."""
    mock_metrics = mocker.MagicMock()
    mock_metrics.train_loss = 0.5
    mock_metrics.val_loss = 0.4
    mock_metrics.mAP50 = 0.8
    mock_metrics.mAP50_95 = 0.6
    mock_metrics.precision = 0.85
    mock_metrics.recall = 0.82
    mock_metrics.f1 = 0.83
    mock_metrics.learning_rate = 0.01
    mock_metrics.box_loss = 0.3
    mock_metrics.cls_loss = 0.1
    mock_metrics.dfl_loss = 0.1

    mock_history = mocker.MagicMock()
    mock_history.metrics = [mock_metrics]
    mock_history.best_mAP50_95 = 0.6
    mock_history.best_mAP50 = 0.8
    mock_history.best_epoch = 1
    mock_history.training_time = 100.0
    return mock_history


def create_mock_evaluation_results(mocker: MockerFixture) -> MagicMock:
    """Create mock EvaluationResults."""
    mock_results = mocker.MagicMock()
    mock_results.mAP50 = 0.8
    mock_results.mAP50_95 = 0.6
    mock_results.precision = 0.85
    mock_results.recall = 0.82
    mock_results.f1 = 0.83
    mock_results.confusion_matrix = None
    mock_results.to_dict.return_value = {
        "mAP50": 0.8,
        "mAP50_95": 0.6,
        "precision": 0.85,
        "recall": 0.82,
        "f1": 0.83,
    }
    return mock_results


# --- TestSetupLogging ---


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_creates_log_file_in_output_dir(self, tmp_path: Path) -> None:
        """Log file is created in the output directory."""
        setup_logging(str(tmp_path), "INFO")

        log_files = list(tmp_path.glob("training_*.log"))
        assert len(log_files) == 1

    def test_creates_output_dir_if_not_exists(self, tmp_path: Path) -> None:
        """Output directory is created when it doesn't exist."""
        output_dir = tmp_path / "logs" / "nested"
        setup_logging(str(output_dir), "INFO")

        assert output_dir.exists()

    def test_log_level_info_default(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """INFO log level is set correctly."""
        mock_basic_config = mocker.patch("screencropnet_yolov11.train.logging.basicConfig")

        setup_logging(str(tmp_path), "INFO")

        call_kwargs = mock_basic_config.call_args[1]
        assert call_kwargs["level"] == logging.INFO

    def test_log_level_debug(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """DEBUG log level is set correctly."""
        mock_basic_config = mocker.patch("screencropnet_yolov11.train.logging.basicConfig")

        setup_logging(str(tmp_path), "DEBUG")

        call_kwargs = mock_basic_config.call_args[1]
        assert call_kwargs["level"] == logging.DEBUG

    def test_suppresses_ultralytics_logging(self, tmp_path: Path) -> None:
        """Ultralytics logger is set to WARNING level."""
        setup_logging(str(tmp_path), "INFO")

        ultra_logger = logging.getLogger("ultralytics")
        assert ultra_logger.level == logging.WARNING


# --- TestLoadConfig ---


class TestLoadConfig:
    """Tests for load_config function."""

    def test_loads_valid_yaml_file(self, tmp_path: Path) -> None:
        """Valid YAML file is loaded correctly."""
        config_path = tmp_path / "config.yaml"
        config_data = {"model": {"size": "m"}, "training": {"epochs": 100}}
        config_path.write_text(yaml.dump(config_data))

        result = load_config(str(config_path))

        assert result["model"]["size"] == "m"
        assert result["training"]["epochs"] == 100

    def test_raises_file_not_found_error(self, tmp_path: Path) -> None:
        """FileNotFoundError raised for non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "nonexistent.yaml"))

    def test_raises_yaml_error_on_invalid(self, tmp_path: Path) -> None:
        """YAML parsing error for malformed content."""
        config_path = tmp_path / "invalid.yaml"
        config_path.write_text("invalid: yaml: content: [")

        with pytest.raises(yaml.YAMLError):
            load_config(str(config_path))


# --- TestMergeConfigWithArgs ---


class TestMergeConfigWithArgs:
    """Tests for merge_config_with_args function."""

    def test_cli_args_override_config(self) -> None:
        """CLI arguments override config file values."""
        config = create_sample_config()
        args = create_mock_args(epochs=50, batch=32)

        result = merge_config_with_args(config, args)

        assert result["training"]["epochs"] == 50
        assert result["training"]["batch_size"] == 32

    def test_none_args_do_not_override(self) -> None:
        """None CLI args leave config values unchanged."""
        config = create_sample_config()
        args = create_mock_args(epochs=None, batch=None)

        result = merge_config_with_args(config, args)

        assert result["training"]["epochs"] == 100
        assert result["training"]["batch_size"] == 16

    def test_partial_override(self) -> None:
        """Only specified args override config values."""
        config = create_sample_config()
        args = create_mock_args(epochs=50, batch=None, imgsz=1280)

        result = merge_config_with_args(config, args)

        assert result["training"]["epochs"] == 50
        assert result["training"]["batch_size"] == 16
        assert result["training"]["image_size"] == 1280

    def test_all_eight_parameters_merged(self) -> None:
        """All 8 mergeable parameters are handled."""
        config = create_sample_config()
        args = create_mock_args(
            data="/new/data",
            epochs=200,
            batch=64,
            imgsz=1280,
            device="cuda:0",
            workers=16,
            output="/new/output",
            model_size="l",
        )

        result = merge_config_with_args(config, args)

        assert result["dataset"]["path"] == "/new/data"
        assert result["training"]["epochs"] == 200
        assert result["training"]["batch_size"] == 64
        assert result["training"]["image_size"] == 1280
        assert result["device"]["type"] == "cuda:0"
        assert result["training"]["workers"] == 16
        assert result["logging"]["output_dir"] == "/new/output"
        assert result["model"]["size"] == "l"


# --- TestValidateDataset ---


class TestValidateDataset:
    """Tests for validate_dataset function."""

    def test_valid_dataset_returns_true(self, mocker: MockerFixture) -> None:
        """Valid dataset returns True."""
        mock_validator = mocker.MagicMock()
        mock_stats = mocker.MagicMock()
        mock_stats.class_distribution = {"tweet": 100, "retweet": 100}
        mock_validator.validate.return_value = (True, mock_stats, [])

        mocker.patch("screencropnet_yolov11.train.DatasetValidator", return_value=mock_validator)
        mocker.patch("screencropnet_yolov11.train.display_dataset_stats")
        mocker.patch("screencropnet_yolov11.train.check_class_imbalance", return_value=[])

        config = create_sample_config()
        result = validate_dataset(config)

        assert result is True

    def test_invalid_dataset_returns_false(self, mocker: MockerFixture) -> None:
        """Invalid dataset returns False."""
        mock_validator = mocker.MagicMock()
        mock_stats = mocker.MagicMock()
        mock_validator.validate.return_value = (
            False,
            mock_stats,
            ["Error 1", "Error 2"],
        )

        mocker.patch("screencropnet_yolov11.train.DatasetValidator", return_value=mock_validator)
        mocker.patch("screencropnet_yolov11.train.display_dataset_stats")

        config = create_sample_config()
        result = validate_dataset(config)

        assert result is False

    def test_logs_validation_errors(self, mocker: MockerFixture) -> None:
        """Validation errors are logged."""
        errors = ["Missing image", "Invalid annotation", "Corrupt file"]
        mock_validator = mocker.MagicMock()
        mock_stats = mocker.MagicMock()
        mock_validator.validate.return_value = (False, mock_stats, errors)

        mocker.patch("screencropnet_yolov11.train.DatasetValidator", return_value=mock_validator)
        mocker.patch("screencropnet_yolov11.train.display_dataset_stats")
        mock_logger = mocker.patch("screencropnet_yolov11.train.logger")

        config = create_sample_config()
        validate_dataset(config)

        # Check errors were logged
        assert mock_logger.error.call_count >= len(errors)

    def test_class_imbalance_warning(self, mocker: MockerFixture) -> None:
        """Class imbalance warnings are logged."""
        mock_validator = mocker.MagicMock()
        mock_stats = mocker.MagicMock()
        mock_stats.class_distribution = {"tweet": 900, "retweet": 100}
        mock_validator.validate.return_value = (True, mock_stats, [])

        mocker.patch("screencropnet_yolov11.train.DatasetValidator", return_value=mock_validator)
        mocker.patch("screencropnet_yolov11.train.display_dataset_stats")
        mocker.patch(
            "screencropnet_yolov11.train.check_class_imbalance",
            return_value=["Class 'retweet' is underrepresented"],
        )
        mock_logger = mocker.patch("screencropnet_yolov11.train.logger")

        config = create_sample_config()
        validate_dataset(config)

        mock_logger.warning.assert_called()


# --- TestLoadDatasetFromRoboflow ---


class TestLoadDatasetFromRoboflow:
    """Tests for load_dataset_from_roboflow function."""

    def test_disabled_returns_original_path(self) -> None:
        """When disabled, returns original dataset path."""
        config = create_sample_config()
        config["dataset"]["roboflow"]["enabled"] = False

        result = load_dataset_from_roboflow(config)

        assert result == config["dataset"]["path"]

    def test_enabled_downloads_dataset(self, mocker: MockerFixture) -> None:
        """When enabled, downloads dataset from Roboflow."""
        mock_loader = mocker.MagicMock()
        mock_loader.download.return_value = Path("/downloaded/dataset")
        mocker.patch("screencropnet_yolov11.train.RoboflowLoader", return_value=mock_loader)

        config = create_sample_config()
        config["dataset"]["roboflow"] = {
            "enabled": True,
            "api_key": "test_key",
            "workspace": "test_workspace",
            "project": "test_project",
            "version": 1,
            "format": "yolov11",
        }

        result = load_dataset_from_roboflow(config)

        assert result == "/downloaded/dataset"
        mock_loader.download.assert_called_once()

    def test_api_key_from_config(self, mocker: MockerFixture) -> None:
        """API key from config is used."""
        mock_loader_class = mocker.patch("screencropnet_yolov11.train.RoboflowLoader")
        mock_loader_class.return_value.download.return_value = Path("/data")

        config = create_sample_config()
        config["dataset"]["roboflow"] = {
            "enabled": True,
            "api_key": "config_api_key",
            "workspace": "ws",
            "project": "proj",
            "version": 1,
        }

        load_dataset_from_roboflow(config)

        call_kwargs = mock_loader_class.call_args[1]
        assert call_kwargs["api_key"] == "config_api_key"

    def test_api_key_from_env_fallback(self, mocker: MockerFixture) -> None:
        """Falls back to environment variable when config key is missing."""
        mock_loader_class = mocker.patch("screencropnet_yolov11.train.RoboflowLoader")
        mock_loader_class.return_value.download.return_value = Path("/data")
        mocker.patch.dict("os.environ", {"ROBOFLOW_API_KEY": "env_api_key"})

        config = create_sample_config()
        config["dataset"]["roboflow"] = {
            "enabled": True,
            "api_key": None,
            "workspace": "ws",
            "project": "proj",
            "version": 1,
        }

        load_dataset_from_roboflow(config)

        call_kwargs = mock_loader_class.call_args[1]
        assert call_kwargs["api_key"] == "env_api_key"

    def test_missing_api_key_raises_value_error(self, mocker: MockerFixture) -> None:
        """ValueError raised when API key is not found."""
        mocker.patch.dict("os.environ", {}, clear=True)

        config = create_sample_config()
        config["dataset"]["roboflow"] = {
            "enabled": True,
            "api_key": None,
            "workspace": "ws",
            "project": "proj",
            "version": 1,
        }

        with pytest.raises(ValueError, match="Roboflow API key not provided"):
            load_dataset_from_roboflow(config)


# --- TestSplitDatasetIfNeeded ---


class TestSplitDatasetIfNeeded:
    """Tests for split_dataset_if_needed function."""

    def test_skips_when_auto_split_disabled(self, mocker: MockerFixture) -> None:
        """No split when auto_split is False."""
        mock_splitter_class = mocker.patch("screencropnet_yolov11.train.DatasetSplitter")

        config = create_sample_config()
        config["dataset"]["auto_split"] = False

        split_dataset_if_needed(config)

        mock_splitter_class.assert_not_called()

    def test_calls_splitter_with_ratios(self, mocker: MockerFixture) -> None:
        """Splitter is called with correct ratios when enabled."""
        mock_splitter = mocker.MagicMock()
        mock_splitter.split.return_value = {"train": 70, "val": 20, "test": 10}
        mock_splitter_class = mocker.patch(
            "screencropnet_yolov11.train.DatasetSplitter", return_value=mock_splitter
        )

        config = create_sample_config()
        config["dataset"]["auto_split"] = True
        config["dataset"]["split_ratios"] = {"train": 0.7, "val": 0.2, "test": 0.1}
        config["dataset"]["seed"] = 123

        split_dataset_if_needed(config)

        call_kwargs = mock_splitter_class.call_args[1]
        assert call_kwargs["train_ratio"] == 0.7
        assert call_kwargs["val_ratio"] == 0.2
        assert call_kwargs["test_ratio"] == 0.1
        assert call_kwargs["seed"] == 123


# --- TestTrainModel ---


class TestTrainModel:
    """Tests for train_model function."""

    def test_creates_dataset_yaml(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Dataset YAML file is created."""
        mock_create_yaml = mocker.patch(
            "screencropnet_yolov11.train.create_dataset_yaml",
            return_value=str(tmp_path / "dataset.yaml"),
        )
        mocker.patch("screencropnet_yolov11.train.ModelFactory")
        mock_trainer = mocker.MagicMock()
        mock_trainer.train.return_value = create_mock_training_history(mocker)
        mocker.patch("screencropnet_yolov11.train.Trainer", return_value=mock_trainer)
        mocker.patch("screencropnet_yolov11.train.AugmentationConfig.get_augmentation")

        config = create_sample_config()
        config["logging"]["output_dir"] = str(tmp_path)

        train_model(config)

        mock_create_yaml.assert_called_once()
        call_kwargs = mock_create_yaml.call_args[1]
        assert call_kwargs["class_names"] == ["tweet", "retweet"]

    def test_creates_model_config(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """ModelConfig is created with correct parameters."""
        mocker.patch("screencropnet_yolov11.train.create_dataset_yaml")
        mock_model_config = mocker.patch("screencropnet_yolov11.train.ModelConfig")
        mocker.patch("screencropnet_yolov11.train.ModelFactory")
        mock_trainer = mocker.MagicMock()
        mock_trainer.train.return_value = create_mock_training_history(mocker)
        mocker.patch("screencropnet_yolov11.train.Trainer", return_value=mock_trainer)
        mocker.patch("screencropnet_yolov11.train.AugmentationConfig.get_augmentation")

        config = create_sample_config()
        config["logging"]["output_dir"] = str(tmp_path)
        config["training"]["epochs"] = 50

        train_model(config)

        call_kwargs = mock_model_config.call_args[1]
        assert call_kwargs["size"] == "m"
        assert call_kwargs["epochs"] == 50

    def test_creates_trainer_and_trains(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Trainer is created and train() is called."""
        mocker.patch("screencropnet_yolov11.train.create_dataset_yaml")
        mocker.patch("screencropnet_yolov11.train.ModelFactory")
        mock_trainer = mocker.MagicMock()
        mock_history = create_mock_training_history(mocker)
        mock_trainer.train.return_value = mock_history
        mock_trainer_class = mocker.patch(
            "screencropnet_yolov11.train.Trainer", return_value=mock_trainer
        )
        mocker.patch("screencropnet_yolov11.train.AugmentationConfig.get_augmentation")

        config = create_sample_config()
        config["logging"]["output_dir"] = str(tmp_path)

        result = train_model(config)

        mock_trainer_class.assert_called_once()
        mock_trainer.train.assert_called_once()
        assert result is mock_history

    def test_resume_calls_resume_method(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Resume path triggers resume() instead of train()."""
        mocker.patch("screencropnet_yolov11.train.create_dataset_yaml")
        mocker.patch("screencropnet_yolov11.train.ModelFactory")
        mock_trainer = mocker.MagicMock()
        mock_history = create_mock_training_history(mocker)
        mock_trainer.resume.return_value = mock_history
        mocker.patch("screencropnet_yolov11.train.Trainer", return_value=mock_trainer)
        mocker.patch("screencropnet_yolov11.train.AugmentationConfig.get_augmentation")

        config = create_sample_config()
        config["logging"]["output_dir"] = str(tmp_path)

        train_model(config, resume_path="/checkpoint/last.pt")

        mock_trainer.train.assert_not_called()
        mock_trainer.resume.assert_called_once_with("/checkpoint/last.pt")

    def test_returns_training_history(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Returns TrainingHistory from trainer."""
        mocker.patch("screencropnet_yolov11.train.create_dataset_yaml")
        mocker.patch("screencropnet_yolov11.train.ModelFactory")
        mock_trainer = mocker.MagicMock()
        mock_history = create_mock_training_history(mocker)
        mock_trainer.train.return_value = mock_history
        mocker.patch("screencropnet_yolov11.train.Trainer", return_value=mock_trainer)
        mocker.patch("screencropnet_yolov11.train.AugmentationConfig.get_augmentation")

        config = create_sample_config()
        config["logging"]["output_dir"] = str(tmp_path)

        result = train_model(config)

        assert result.best_mAP50_95 == 0.6

    def test_augmentation_config_merged(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Custom augmentation config is merged."""
        mocker.patch("screencropnet_yolov11.train.create_dataset_yaml")
        mocker.patch("screencropnet_yolov11.train.ModelFactory")
        mock_trainer = mocker.MagicMock()
        mock_trainer.train.return_value = create_mock_training_history(mocker)
        mock_trainer_class = mocker.patch(
            "screencropnet_yolov11.train.Trainer", return_value=mock_trainer
        )
        mock_aug = {"mosaic": 1.0, "fliplr": 0.5}
        mocker.patch(
            "screencropnet_yolov11.train.AugmentationConfig.get_augmentation",
            return_value=mock_aug.copy(),
        )

        config = create_sample_config()
        config["logging"]["output_dir"] = str(tmp_path)
        config["augmentation"] = {"custom_key": "custom_value"}

        train_model(config)

        call_kwargs = mock_trainer_class.call_args[1]
        assert "augmentation" in call_kwargs["config"]


# --- TestEvaluateModel ---


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_loads_model_and_evaluates(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Model is loaded and evaluated."""
        mock_yolo = mocker.MagicMock()
        mocker.patch("ultralytics.YOLO", return_value=mock_yolo)
        mock_evaluator = mocker.MagicMock()
        mock_results = create_mock_evaluation_results(mocker)
        mock_evaluator.evaluate.return_value = mock_results
        mocker.patch("screencropnet_yolov11.train.Evaluator", return_value=mock_evaluator)

        config = create_sample_config()

        result = evaluate_model(config, "/model/best.pt", str(tmp_path))

        assert result is mock_results
        mock_evaluator.evaluate.assert_called_once()

    def test_saves_results_to_json(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Evaluation results are saved to JSON file."""
        mock_yolo = mocker.MagicMock()
        mocker.patch("ultralytics.YOLO", return_value=mock_yolo)
        mock_evaluator = mocker.MagicMock()
        mock_results = create_mock_evaluation_results(mocker)
        mock_evaluator.evaluate.return_value = mock_results
        mocker.patch("screencropnet_yolov11.train.Evaluator", return_value=mock_evaluator)

        config = create_sample_config()

        evaluate_model(config, "/model/best.pt", str(tmp_path))

        expected_path = str(tmp_path / "evaluation_results.json")
        mock_results.save.assert_called_once_with(expected_path)

    def test_uses_config_parameters(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Evaluation uses config parameters."""
        mock_yolo = mocker.MagicMock()
        mocker.patch("ultralytics.YOLO", return_value=mock_yolo)
        mock_evaluator = mocker.MagicMock()
        mock_results = create_mock_evaluation_results(mocker)
        mock_evaluator.evaluate.return_value = mock_results
        mocker.patch("screencropnet_yolov11.train.Evaluator", return_value=mock_evaluator)

        config = create_sample_config()
        config["inference"]["confidence"] = 0.5
        config["inference"]["iou_threshold"] = 0.6

        evaluate_model(config, "/model/best.pt", str(tmp_path))

        call_kwargs = mock_evaluator.evaluate.call_args[1]
        assert call_kwargs["conf"] == 0.5
        assert call_kwargs["iou"] == 0.6


# --- TestExportModel ---


class TestExportModel:
    """Tests for export_model function."""

    def test_exports_to_formats(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Model is exported to specified formats."""
        mock_yolo = mocker.MagicMock()
        mocker.patch("ultralytics.YOLO", return_value=mock_yolo)
        mock_exporter = mocker.MagicMock()
        mock_exporter.export.return_value = {
            "pytorch": "/model.pt",
            "onnx": "/model.onnx",
        }
        mocker.patch("screencropnet_yolov11.train.ModelExporter", return_value=mock_exporter)

        config = create_sample_config()
        config["export"]["formats"] = ["pytorch", "onnx"]

        result = export_model(config, "/model/best.pt", str(tmp_path))

        assert "pytorch" in result
        assert "onnx" in result
        mock_exporter.export.assert_called_once()

    def test_exports_to_onnx_with_options(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """ONNX export includes format-specific options."""
        mock_yolo = mocker.MagicMock()
        mocker.patch("ultralytics.YOLO", return_value=mock_yolo)
        mock_exporter = mocker.MagicMock()
        mock_exporter.export.return_value = {"onnx": "/model.onnx"}
        mocker.patch("screencropnet_yolov11.train.ModelExporter", return_value=mock_exporter)

        config = create_sample_config()
        config["export"]["formats"] = ["onnx"]
        config["export"]["onnx"] = {"dynamic": True, "simplify": False, "opset": 17}

        export_model(config, "/model/best.pt", str(tmp_path))

        call_kwargs = mock_exporter.export.call_args[1]
        assert call_kwargs["dynamic"] is True
        assert call_kwargs["simplify"] is False
        assert call_kwargs["opset"] == 17

    def test_returns_export_paths_dict(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Returns dictionary of exported paths."""
        mock_yolo = mocker.MagicMock()
        mocker.patch("ultralytics.YOLO", return_value=mock_yolo)
        mock_exporter = mocker.MagicMock()
        expected = {"pytorch": "/model.pt", "onnx": "/model.onnx"}
        mock_exporter.export.return_value = expected
        mocker.patch("screencropnet_yolov11.train.ModelExporter", return_value=mock_exporter)

        config = create_sample_config()

        result = export_model(config, "/model/best.pt", str(tmp_path))

        assert result == expected


# --- TestCreateVisualizations ---


class TestCreateVisualizations:
    """Tests for create_visualizations function."""

    def test_creates_visualizations_directory(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Visualizations directory is created."""
        mock_viz = mocker.MagicMock()
        mocker.patch("screencropnet_yolov11.train.TrainingVisualizer", return_value=mock_viz)
        mocker.patch("screencropnet_yolov11.train.ConfusionMatrixVisualizer")
        mocker.patch("screencropnet_yolov11.train.ResultsDashboard")

        config = create_sample_config()
        history = create_mock_training_history(mocker)
        results = create_mock_evaluation_results(mocker)

        create_visualizations(config, history, results, str(tmp_path))

        viz_dir = tmp_path / "visualizations"
        assert viz_dir.exists()

    def test_creates_training_curves(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Training curves are plotted."""
        mock_viz = mocker.MagicMock()
        mocker.patch("screencropnet_yolov11.train.TrainingVisualizer", return_value=mock_viz)
        mocker.patch("screencropnet_yolov11.train.ConfusionMatrixVisualizer")
        mocker.patch("screencropnet_yolov11.train.ResultsDashboard")

        config = create_sample_config()
        history = create_mock_training_history(mocker)
        results = create_mock_evaluation_results(mocker)

        create_visualizations(config, history, results, str(tmp_path))

        mock_viz.plot_training_curves.assert_called_once()
        mock_viz.plot_loss_components.assert_called_once()

    def test_creates_confusion_matrix(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Confusion matrix is plotted when available."""
        mocker.patch("screencropnet_yolov11.train.TrainingVisualizer")
        mock_cm_viz = mocker.patch("screencropnet_yolov11.train.ConfusionMatrixVisualizer")
        mocker.patch("screencropnet_yolov11.train.ResultsDashboard")

        config = create_sample_config()
        history = create_mock_training_history(mocker)
        results = create_mock_evaluation_results(mocker)
        results.confusion_matrix = [[10, 2], [3, 15]]

        create_visualizations(config, history, results, str(tmp_path))

        mock_cm_viz.plot_confusion_matrix.assert_called_once()

    def test_skips_confusion_matrix_when_none(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Confusion matrix is skipped when None."""
        mocker.patch("screencropnet_yolov11.train.TrainingVisualizer")
        mock_cm_viz = mocker.patch("screencropnet_yolov11.train.ConfusionMatrixVisualizer")
        mocker.patch("screencropnet_yolov11.train.ResultsDashboard")

        config = create_sample_config()
        history = create_mock_training_history(mocker)
        results = create_mock_evaluation_results(mocker)
        results.confusion_matrix = None

        create_visualizations(config, history, results, str(tmp_path))

        mock_cm_viz.plot_confusion_matrix.assert_not_called()

    def test_creates_dashboard(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Results dashboard is created."""
        mocker.patch("screencropnet_yolov11.train.TrainingVisualizer")
        mocker.patch("screencropnet_yolov11.train.ConfusionMatrixVisualizer")
        mock_dashboard = mocker.MagicMock()
        mocker.patch("screencropnet_yolov11.train.ResultsDashboard", return_value=mock_dashboard)

        config = create_sample_config()
        history = create_mock_training_history(mocker)
        results = create_mock_evaluation_results(mocker)

        create_visualizations(config, history, results, str(tmp_path))

        mock_dashboard.create_dashboard.assert_called_once()


# --- TestRunAblationStudy ---


class TestRunAblationStudy:
    """Tests for run_ablation_study function."""

    def test_runs_when_enabled(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Ablation study runs when enabled."""
        mocker.patch("screencropnet_yolov11.train.ModelFactory")
        mock_create_ablation = mocker.patch("screencropnet_yolov11.train.create_ablation_study")

        config = create_sample_config()
        config["logging"]["output_dir"] = str(tmp_path)
        config["ablation"] = {"enabled": True, "parameters": {"epochs": [50, 100]}}

        run_ablation_study(config)

        mock_create_ablation.assert_called_once()

    def test_skips_when_disabled(self, mocker: MockerFixture) -> None:
        """Ablation study skipped when disabled."""
        mock_create_ablation = mocker.patch("screencropnet_yolov11.train.create_ablation_study")

        config = create_sample_config()
        config["ablation"]["enabled"] = False

        run_ablation_study(config)

        mock_create_ablation.assert_not_called()


# --- TestParseArgs ---


class TestParseArgs:
    """Tests for parse_args function."""

    def test_default_arguments(self, mocker: MockerFixture) -> None:
        """Default argument values."""
        mocker.patch("sys.argv", ["train.py"])

        args = parse_args()

        assert args.config == "config/config.yaml"
        assert args.data is None
        assert args.epochs is None
        assert args.validate_only is False

    def test_config_path_argument(self, mocker: MockerFixture) -> None:
        """Config path can be specified."""
        mocker.patch("sys.argv", ["train.py", "--config", "/custom/config.yaml"])

        args = parse_args()

        assert args.config == "/custom/config.yaml"

    def test_model_size_choices(self, mocker: MockerFixture) -> None:
        """Model size accepts valid choices."""
        mocker.patch("sys.argv", ["train.py", "--model-size", "l"])

        args = parse_args()

        assert args.model_size == "l"

    def test_flag_arguments(self, mocker: MockerFixture) -> None:
        """Flag arguments are parsed correctly."""
        mocker.patch("sys.argv", ["train.py", "--validate-only"])

        args = parse_args()

        assert args.validate_only is True


# --- TestMain ---


class TestMain:
    """Tests for main function."""

    def test_full_pipeline_success(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Full pipeline completes successfully."""
        config = create_sample_config()
        config["logging"]["output_dir"] = str(tmp_path)

        mocker.patch(
            "screencropnet_yolov11.train.parse_args",
            return_value=create_mock_args(config=str(tmp_path / "config.yaml")),
        )
        mocker.patch("screencropnet_yolov11.train.load_config", return_value=config)
        mocker.patch("screencropnet_yolov11.train.merge_config_with_args", return_value=config)
        mocker.patch("screencropnet_yolov11.train.setup_logging")
        mocker.patch(
            "screencropnet_yolov11.train.load_dataset_from_roboflow",
            return_value="/data",
        )
        mocker.patch("screencropnet_yolov11.train.split_dataset_if_needed")
        mocker.patch("screencropnet_yolov11.train.validate_dataset", return_value=True)
        mock_history = create_mock_training_history(mocker)
        mocker.patch("screencropnet_yolov11.train.train_model", return_value=mock_history)

        # Create fake model weights
        weights_dir = tmp_path / "train" / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "best.pt").touch()

        mock_results = create_mock_evaluation_results(mocker)
        mocker.patch("screencropnet_yolov11.train.evaluate_model", return_value=mock_results)
        mocker.patch("screencropnet_yolov11.train.export_model")
        mocker.patch("screencropnet_yolov11.train.create_visualizations")
        mocker.patch("screencropnet_yolov11.train.run_ablation_study")

        from screencropnet_yolov11.train import main

        result = main()

        assert result == 0

    def test_validate_only_exits_early(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """--validate-only exits after validation."""
        config = create_sample_config()
        config["logging"]["output_dir"] = str(tmp_path)

        mocker.patch(
            "screencropnet_yolov11.train.parse_args",
            return_value=create_mock_args(validate_only=True),
        )
        mocker.patch("screencropnet_yolov11.train.load_config", return_value=config)
        mocker.patch("screencropnet_yolov11.train.merge_config_with_args", return_value=config)
        mocker.patch("screencropnet_yolov11.train.setup_logging")
        mocker.patch(
            "screencropnet_yolov11.train.load_dataset_from_roboflow",
            return_value="/data",
        )
        mocker.patch("screencropnet_yolov11.train.split_dataset_if_needed")
        mocker.patch("screencropnet_yolov11.train.validate_dataset", return_value=True)
        mock_train = mocker.patch("screencropnet_yolov11.train.train_model")

        from screencropnet_yolov11.train import main

        result = main()

        assert result == 0
        mock_train.assert_not_called()

    def test_eval_only_skips_training(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """--eval-only skips training."""
        config = create_sample_config()
        config["logging"]["output_dir"] = str(tmp_path)

        mocker.patch(
            "screencropnet_yolov11.train.parse_args",
            return_value=create_mock_args(eval_only="/model/best.pt"),
        )
        mocker.patch("screencropnet_yolov11.train.load_config", return_value=config)
        mocker.patch("screencropnet_yolov11.train.merge_config_with_args", return_value=config)
        mocker.patch("screencropnet_yolov11.train.setup_logging")
        mocker.patch(
            "screencropnet_yolov11.train.load_dataset_from_roboflow",
            return_value="/data",
        )
        mocker.patch("screencropnet_yolov11.train.split_dataset_if_needed")
        mocker.patch("screencropnet_yolov11.train.validate_dataset", return_value=True)
        mock_train = mocker.patch("screencropnet_yolov11.train.train_model")
        mock_eval = mocker.patch(
            "screencropnet_yolov11.train.evaluate_model",
            return_value=create_mock_evaluation_results(mocker),
        )

        from screencropnet_yolov11.train import main

        result = main()

        assert result == 0
        mock_train.assert_not_called()
        mock_eval.assert_called_once()

    def test_export_only_skips_training(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """--export-only skips training."""
        config = create_sample_config()
        config["logging"]["output_dir"] = str(tmp_path)

        mocker.patch(
            "screencropnet_yolov11.train.parse_args",
            return_value=create_mock_args(export_only="/model/best.pt"),
        )
        mocker.patch("screencropnet_yolov11.train.load_config", return_value=config)
        mocker.patch("screencropnet_yolov11.train.merge_config_with_args", return_value=config)
        mocker.patch("screencropnet_yolov11.train.setup_logging")
        mocker.patch(
            "screencropnet_yolov11.train.load_dataset_from_roboflow",
            return_value="/data",
        )
        mocker.patch("screencropnet_yolov11.train.split_dataset_if_needed")
        mocker.patch("screencropnet_yolov11.train.validate_dataset", return_value=True)
        mock_train = mocker.patch("screencropnet_yolov11.train.train_model")
        mock_export = mocker.patch("screencropnet_yolov11.train.export_model")

        from screencropnet_yolov11.train import main

        result = main()

        assert result == 0
        mock_train.assert_not_called()
        mock_export.assert_called_once()

    def test_validation_failure_returns_one(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Returns 1 when validation fails."""
        config = create_sample_config()
        config["logging"]["output_dir"] = str(tmp_path)

        mocker.patch(
            "screencropnet_yolov11.train.parse_args",
            return_value=create_mock_args(),
        )
        mocker.patch("screencropnet_yolov11.train.load_config", return_value=config)
        mocker.patch("screencropnet_yolov11.train.merge_config_with_args", return_value=config)
        mocker.patch("screencropnet_yolov11.train.setup_logging")
        mocker.patch(
            "screencropnet_yolov11.train.load_dataset_from_roboflow",
            return_value="/data",
        )
        mocker.patch("screencropnet_yolov11.train.split_dataset_if_needed")
        mocker.patch("screencropnet_yolov11.train.validate_dataset", return_value=False)

        from screencropnet_yolov11.train import main

        result = main()

        assert result == 1

    def test_exception_logged_returns_one(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Returns 1 when exception occurs."""
        config = create_sample_config()
        config["logging"]["output_dir"] = str(tmp_path)

        mocker.patch(
            "screencropnet_yolov11.train.parse_args",
            return_value=create_mock_args(),
        )
        mocker.patch("screencropnet_yolov11.train.load_config", return_value=config)
        mocker.patch("screencropnet_yolov11.train.merge_config_with_args", return_value=config)
        mocker.patch("screencropnet_yolov11.train.setup_logging")
        mocker.patch(
            "screencropnet_yolov11.train.load_dataset_from_roboflow",
            side_effect=Exception("Network error"),
        )
        mock_logger = mocker.patch("screencropnet_yolov11.train.logger")

        from screencropnet_yolov11.train import main

        result = main()

        assert result == 1
        mock_logger.exception.assert_called()

    def test_model_path_fallback_best_to_last(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Falls back to last.pt when best.pt doesn't exist."""
        config = create_sample_config()
        config["logging"]["output_dir"] = str(tmp_path)

        mocker.patch(
            "screencropnet_yolov11.train.parse_args",
            return_value=create_mock_args(),
        )
        mocker.patch("screencropnet_yolov11.train.load_config", return_value=config)
        mocker.patch("screencropnet_yolov11.train.merge_config_with_args", return_value=config)
        mocker.patch("screencropnet_yolov11.train.setup_logging")
        mocker.patch(
            "screencropnet_yolov11.train.load_dataset_from_roboflow",
            return_value="/data",
        )
        mocker.patch("screencropnet_yolov11.train.split_dataset_if_needed")
        mocker.patch("screencropnet_yolov11.train.validate_dataset", return_value=True)
        mocker.patch(
            "screencropnet_yolov11.train.train_model",
            return_value=create_mock_training_history(mocker),
        )

        # Create only last.pt, not best.pt
        weights_dir = tmp_path / "train" / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "last.pt").touch()

        mock_eval = mocker.patch(
            "screencropnet_yolov11.train.evaluate_model",
            return_value=create_mock_evaluation_results(mocker),
        )
        mocker.patch("screencropnet_yolov11.train.export_model")
        mocker.patch("screencropnet_yolov11.train.create_visualizations")
        mocker.patch("screencropnet_yolov11.train.run_ablation_study")

        from screencropnet_yolov11.train import main

        main()

        # Check that last.pt was used
        call_args = mock_eval.call_args[0]
        assert "last.pt" in call_args[1]
