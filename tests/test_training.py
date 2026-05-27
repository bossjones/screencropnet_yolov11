"""Tests for the training module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from screencropnet_yolov11.training import (
    CheckpointCallback,
    EarlyStopping,
    MetricsLogger,
    TensorBoardCallback,
    Trainer,
    TrainingCallback,
    TrainingHistory,
    TrainingMetrics,
    WandbCallback,
    create_ablation_study,
)

# Helper functions


def create_mock_trainer(mocker: MockerFixture) -> MagicMock:
    """Create a mock ultralytics trainer with standard attributes."""
    mock_trainer = mocker.MagicMock()
    mock_trainer.epoch = 4  # 0-indexed, so epoch 5
    mock_trainer.stop = False

    # Loss attributes
    mock_loss = mocker.MagicMock()
    mock_loss.mean.return_value = 0.5
    mock_trainer.loss = mock_loss
    mock_trainer.loss_items = [0.1, 0.2, 0.3]  # box, cls, dfl

    # Metrics attributes
    mock_metrics = mocker.MagicMock()
    mock_metrics.box.map50 = 0.85
    mock_metrics.box.map = 0.75
    mock_metrics.box.mp = 0.80  # precision
    mock_metrics.box.mr = 0.70  # recall
    mock_trainer.metrics = mock_metrics

    # Optimizer attributes
    mock_trainer.optimizer.param_groups = [{"lr": 0.001}]

    return mock_trainer


def create_mock_yolo_model(mocker: MockerFixture) -> MagicMock:
    """Create a mock YOLO model for training tests."""
    mock_model = mocker.MagicMock()
    mock_model.task = "detect"
    mock_model.add_callback = mocker.MagicMock()
    mock_model.train = mocker.MagicMock()
    return mock_model


def create_sample_training_metrics(
    epoch: int = 1,
    train_loss: float = 0.5,
    mAP50: float = 0.85,
    mAP50_95: float = 0.75,
) -> TrainingMetrics:
    """Create sample TrainingMetrics for testing."""
    return TrainingMetrics(
        epoch=epoch,
        train_loss=train_loss,
        val_loss=0.4,
        box_loss=0.1,
        cls_loss=0.2,
        dfl_loss=0.3,
        mAP50=mAP50,
        mAP50_95=mAP50_95,
        precision=0.8,
        recall=0.7,
        f1=0.746,
        learning_rate=0.001,
    )


# Test Classes


class TestTrainingMetrics:
    """Tests for TrainingMetrics dataclass."""

    def test_default_values(self) -> None:
        """Default field values are zeros."""
        metrics = TrainingMetrics()
        assert metrics.epoch == 0
        assert metrics.train_loss == 0.0
        assert metrics.val_loss == 0.0
        assert metrics.box_loss == 0.0
        assert metrics.cls_loss == 0.0
        assert metrics.dfl_loss == 0.0
        assert metrics.mAP50 == 0.0
        assert metrics.mAP50_95 == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0
        assert metrics.learning_rate == 0.0

    def test_to_dict_returns_all_fields(self) -> None:
        """to_dict() returns all fields."""
        metrics = create_sample_training_metrics()
        result = metrics.to_dict()

        expected_keys = {
            "epoch",
            "train_loss",
            "val_loss",
            "box_loss",
            "cls_loss",
            "dfl_loss",
            "mAP50",
            "mAP50_95",
            "precision",
            "recall",
            "f1",
            "learning_rate",
        }
        assert set(result.keys()) == expected_keys
        assert result["epoch"] == 1
        assert result["train_loss"] == 0.5
        assert result["mAP50"] == 0.85

    def test_to_dict_with_custom_values(self) -> None:
        """to_dict() correctly serializes custom values."""
        metrics = TrainingMetrics(
            epoch=10,
            train_loss=0.123,
            mAP50=0.95,
            mAP50_95=0.88,
        )
        result = metrics.to_dict()
        assert result["epoch"] == 10
        assert result["train_loss"] == 0.123
        assert result["mAP50"] == 0.95
        assert result["mAP50_95"] == 0.88


class TestTrainingHistory:
    """Tests for TrainingHistory dataclass."""

    def test_default_values(self) -> None:
        """Default values are empty/zero."""
        history = TrainingHistory()
        assert history.metrics == []
        assert history.best_epoch == 0
        assert history.best_mAP50 == 0.0
        assert history.best_mAP50_95 == 0.0
        assert history.training_time == 0.0

    def test_add_metrics_updates_list(self) -> None:
        """add_metrics() appends to metrics list."""
        history = TrainingHistory()
        m1 = create_sample_training_metrics(epoch=1, mAP50_95=0.5)
        m2 = create_sample_training_metrics(epoch=2, mAP50_95=0.6)

        history.add_metrics(m1)
        assert len(history.metrics) == 1
        assert history.metrics[0] == m1

        history.add_metrics(m2)
        assert len(history.metrics) == 2
        assert history.metrics[1] == m2

    def test_add_metrics_updates_best_mAP50_95(self) -> None:
        """add_metrics() tracks best mAP50-95."""
        history = TrainingHistory()

        m1 = create_sample_training_metrics(epoch=1, mAP50=0.7, mAP50_95=0.5)
        history.add_metrics(m1)
        assert history.best_epoch == 1
        assert history.best_mAP50_95 == 0.5
        assert history.best_mAP50 == 0.7

        # Better metric should update
        m2 = create_sample_training_metrics(epoch=2, mAP50=0.8, mAP50_95=0.7)
        history.add_metrics(m2)
        assert history.best_epoch == 2
        assert history.best_mAP50_95 == 0.7
        assert history.best_mAP50 == 0.8

        # Worse metric should not update
        m3 = create_sample_training_metrics(epoch=3, mAP50=0.75, mAP50_95=0.6)
        history.add_metrics(m3)
        assert history.best_epoch == 2
        assert history.best_mAP50_95 == 0.7

    def test_to_dict_serialization(self) -> None:
        """to_dict() serializes full history."""
        history = TrainingHistory()
        history.training_time = 100.5
        history.add_metrics(create_sample_training_metrics(epoch=1, mAP50_95=0.6))
        history.add_metrics(create_sample_training_metrics(epoch=2, mAP50_95=0.7))

        result = history.to_dict()

        assert result["best_epoch"] == 2
        assert result["best_mAP50_95"] == 0.7
        assert result["training_time"] == 100.5
        assert len(result["metrics"]) == 2
        assert result["metrics"][0]["epoch"] == 1
        assert result["metrics"][1]["epoch"] == 2

    def test_save_writes_json_file(self, tmp_path: Path) -> None:
        """save() writes history to JSON file."""
        history = TrainingHistory()
        history.training_time = 50.0
        history.add_metrics(create_sample_training_metrics(epoch=1, mAP50_95=0.65))

        json_path = tmp_path / "history.json"
        history.save(str(json_path))

        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)

        assert data["best_epoch"] == 1
        assert data["training_time"] == 50.0
        assert len(data["metrics"]) == 1


class TestTrainingCallback:
    """Tests for TrainingCallback base class."""

    def test_all_hooks_are_no_ops(self, mocker: MockerFixture) -> None:
        """Base class methods don't raise errors."""
        callback = TrainingCallback()
        mock_trainer = mocker.MagicMock()

        # All methods should execute without error
        callback.on_train_start(mock_trainer)
        callback.on_train_end(mock_trainer)
        callback.on_epoch_start(mock_trainer)
        callback.on_epoch_end(mock_trainer)
        callback.on_batch_start(mock_trainer)
        callback.on_batch_end(mock_trainer)
        callback.on_val_start(mock_trainer)
        callback.on_val_end(mock_trainer)


class TestMetricsLogger:
    """Tests for MetricsLogger callback."""

    def test_init_default_values(self) -> None:
        """Default log_interval is 1."""
        history = TrainingHistory()
        logger = MetricsLogger(history)
        assert logger.history is history
        assert logger.log_interval == 1
        assert logger.epoch_start_time is None

    def test_init_custom_log_interval(self) -> None:
        """Custom log_interval is set."""
        history = TrainingHistory()
        logger = MetricsLogger(history, log_interval=5)
        assert logger.log_interval == 5

    def test_on_epoch_start_records_time(self, mocker: MockerFixture) -> None:
        """on_epoch_start() records start time."""
        mocker.patch("time.time", return_value=1000.0)
        history = TrainingHistory()
        logger = MetricsLogger(history)
        mock_trainer = mocker.MagicMock()

        logger.on_epoch_start(mock_trainer)

        assert logger.epoch_start_time == 1000.0

    def test_on_epoch_end_extracts_metrics(self, mocker: MockerFixture) -> None:
        """on_epoch_end() extracts metrics from trainer."""
        mocker.patch("time.time", return_value=1100.0)
        history = TrainingHistory()
        logger = MetricsLogger(history)
        logger.epoch_start_time = 1000.0
        mock_trainer = create_mock_trainer(mocker)

        logger.on_epoch_end(mock_trainer)

        assert len(history.metrics) == 1
        metrics = history.metrics[0]
        assert metrics.epoch == 5  # trainer.epoch (4) + 1
        assert metrics.train_loss == 0.5
        assert metrics.box_loss == 0.1
        assert metrics.cls_loss == 0.2
        assert metrics.dfl_loss == 0.3
        assert metrics.mAP50 == 0.85
        assert metrics.mAP50_95 == 0.75
        assert metrics.precision == 0.80
        assert metrics.recall == 0.70
        assert metrics.learning_rate == 0.001

    def test_on_epoch_end_handles_none_loss(self, mocker: MockerFixture) -> None:
        """on_epoch_end() handles None loss gracefully."""
        mocker.patch("time.time", return_value=1100.0)
        history = TrainingHistory()
        logger = MetricsLogger(history)
        logger.epoch_start_time = 1000.0

        mock_trainer = create_mock_trainer(mocker)
        mock_trainer.loss = None
        mock_trainer.loss_items = None

        logger.on_epoch_end(mock_trainer)

        metrics = history.metrics[0]
        assert metrics.train_loss == 0.0
        assert metrics.box_loss == 0.0
        assert metrics.cls_loss == 0.0
        assert metrics.dfl_loss == 0.0

    def test_on_epoch_end_handles_missing_metrics(self, mocker: MockerFixture) -> None:
        """on_epoch_end() handles missing metrics attribute."""
        mocker.patch("time.time", return_value=1100.0)
        history = TrainingHistory()
        logger = MetricsLogger(history)
        logger.epoch_start_time = 1000.0

        mock_trainer = create_mock_trainer(mocker)
        mock_trainer.metrics = None

        logger.on_epoch_end(mock_trainer)

        metrics = history.metrics[0]
        assert metrics.mAP50 == 0.0
        assert metrics.mAP50_95 == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0

    def test_on_epoch_end_handles_missing_optimizer(self, mocker: MockerFixture) -> None:
        """on_epoch_end() handles missing optimizer."""
        mocker.patch("time.time", return_value=1100.0)
        history = TrainingHistory()
        logger = MetricsLogger(history)
        logger.epoch_start_time = 1000.0

        mock_trainer = create_mock_trainer(mocker)
        del mock_trainer.optimizer

        logger.on_epoch_end(mock_trainer)

        metrics = history.metrics[0]
        assert metrics.learning_rate == 0.0

    def test_f1_calculation(self, mocker: MockerFixture) -> None:
        """F1 score is calculated correctly."""
        mocker.patch("time.time", return_value=1100.0)
        history = TrainingHistory()
        logger = MetricsLogger(history)
        logger.epoch_start_time = 1000.0

        mock_trainer = create_mock_trainer(mocker)
        # P=0.8, R=0.7 -> F1 = 2*0.8*0.7/(0.8+0.7) = 1.12/1.5 = 0.7466...
        mock_trainer.metrics.box.mp = 0.8
        mock_trainer.metrics.box.mr = 0.7

        logger.on_epoch_end(mock_trainer)

        metrics = history.metrics[0]
        expected_f1 = 2 * (0.8 * 0.7) / (0.8 + 0.7)
        assert abs(metrics.f1 - expected_f1) < 0.001

    def test_f1_avoids_division_by_zero(self, mocker: MockerFixture) -> None:
        """F1 stays zero when P+R=0."""
        mocker.patch("time.time", return_value=1100.0)
        history = TrainingHistory()
        logger = MetricsLogger(history)
        logger.epoch_start_time = 1000.0

        mock_trainer = create_mock_trainer(mocker)
        mock_trainer.metrics.box.mp = 0.0
        mock_trainer.metrics.box.mr = 0.0

        logger.on_epoch_end(mock_trainer)

        metrics = history.metrics[0]
        assert metrics.f1 == 0.0


class TestEarlyStopping:
    """Tests for EarlyStopping callback."""

    def test_init_default_values(self) -> None:
        """Default values are set correctly."""
        es = EarlyStopping()
        assert es.patience == 20
        assert es.min_delta == 0.001
        assert es.monitor == "mAP50_95"
        assert es.mode == "max"
        assert es.best_value == float("-inf")
        assert es.counter == 0
        assert es.best_epoch == 0
        assert es.should_stop is False

    def test_init_custom_values(self) -> None:
        """Custom values are set correctly."""
        es = EarlyStopping(patience=10, min_delta=0.01, monitor="mAP50", mode="min")
        assert es.patience == 10
        assert es.min_delta == 0.01
        assert es.monitor == "mAP50"
        assert es.mode == "min"
        assert es.best_value == float("inf")

    def test_on_epoch_end_no_stop_when_improving(self, mocker: MockerFixture) -> None:
        """Counter stays 0 when metric improves."""
        es = EarlyStopping(patience=5)
        mock_trainer = create_mock_trainer(mocker)

        # First epoch
        mock_trainer.metrics.box.map = 0.5
        es.on_epoch_end(mock_trainer)
        assert es.counter == 0
        assert es.best_value == 0.5
        assert es.should_stop is False

        # Improvement
        mock_trainer.metrics.box.map = 0.6
        es.on_epoch_end(mock_trainer)
        assert es.counter == 0
        assert es.best_value == 0.6

    def test_on_epoch_end_increments_counter_no_improvement(self, mocker: MockerFixture) -> None:
        """Counter increments when no improvement."""
        es = EarlyStopping(patience=5)
        mock_trainer = create_mock_trainer(mocker)

        # Set initial best
        mock_trainer.metrics.box.map = 0.5
        es.on_epoch_end(mock_trainer)

        # No improvement
        mock_trainer.metrics.box.map = 0.5
        es.on_epoch_end(mock_trainer)
        assert es.counter == 1

        mock_trainer.metrics.box.map = 0.49
        es.on_epoch_end(mock_trainer)
        assert es.counter == 2

    def test_on_epoch_end_stops_after_patience(self, mocker: MockerFixture) -> None:
        """Training stops after patience epochs without improvement."""
        es = EarlyStopping(patience=3)
        mock_trainer = create_mock_trainer(mocker)

        # Set initial best
        mock_trainer.metrics.box.map = 0.5
        es.on_epoch_end(mock_trainer)

        # No improvement for patience epochs
        for _ in range(3):
            mock_trainer.metrics.box.map = 0.5
            es.on_epoch_end(mock_trainer)

        assert es.counter == 3
        assert es.should_stop is True
        assert mock_trainer.stop is True

    def test_mode_min_detects_improvement(self, mocker: MockerFixture) -> None:
        """Mode 'min' correctly detects improvement (lower is better)."""
        es = EarlyStopping(patience=5, mode="min", monitor="mAP50")
        mock_trainer = create_mock_trainer(mocker)

        # Initial value
        mock_trainer.metrics.box.map50 = 0.5
        es.on_epoch_end(mock_trainer)
        assert es.best_value == 0.5

        # Lower value is improvement
        mock_trainer.metrics.box.map50 = 0.4
        es.on_epoch_end(mock_trainer)
        assert es.counter == 0
        assert es.best_value == 0.4

        # Higher value is not improvement
        mock_trainer.metrics.box.map50 = 0.45
        es.on_epoch_end(mock_trainer)
        assert es.counter == 1

    def test_min_delta_threshold(self, mocker: MockerFixture) -> None:
        """Improvement must exceed min_delta."""
        es = EarlyStopping(patience=5, min_delta=0.01)
        mock_trainer = create_mock_trainer(mocker)

        # Initial
        mock_trainer.metrics.box.map = 0.5
        es.on_epoch_end(mock_trainer)

        # Small improvement below threshold
        mock_trainer.metrics.box.map = 0.505
        es.on_epoch_end(mock_trainer)
        assert es.counter == 1  # Not enough improvement

        # Sufficient improvement
        mock_trainer.metrics.box.map = 0.52
        es.on_epoch_end(mock_trainer)
        assert es.counter == 0
        assert es.best_value == 0.52

    def test_on_epoch_end_handles_missing_metrics(self, mocker: MockerFixture) -> None:
        """on_epoch_end() handles missing metrics gracefully."""
        es = EarlyStopping(patience=5)
        mock_trainer = mocker.MagicMock()
        mock_trainer.metrics = None

        # Should not raise or change state
        es.on_epoch_end(mock_trainer)
        assert es.counter == 0
        assert es.should_stop is False

    def test_on_epoch_end_handles_unknown_monitor(self, mocker: MockerFixture) -> None:
        """on_epoch_end() handles unknown monitor metric."""
        es = EarlyStopping(patience=5, monitor="unknown_metric")
        mock_trainer = create_mock_trainer(mocker)

        # Should not raise or change state
        es.on_epoch_end(mock_trainer)
        assert es.counter == 0


class TestCheckpointCallback:
    """Tests for CheckpointCallback."""

    def test_init_creates_checkpoint_dir(self, tmp_path: Path) -> None:
        """Checkpoint directory is created on init."""
        save_dir = tmp_path / "checkpoints"
        CheckpointCallback(str(save_dir))
        assert save_dir.exists()

    def test_init_default_values(self, tmp_path: Path) -> None:
        """Default values are set correctly."""
        cb = CheckpointCallback(str(tmp_path))
        assert cb.save_period == 10
        assert cb.save_best is True
        assert cb.best_map == 0.0

    def test_on_epoch_end_saves_at_interval(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Checkpoints saved at periodic intervals."""
        cb = CheckpointCallback(str(tmp_path), save_period=5, save_best=False)
        mock_trainer = create_mock_trainer(mocker)

        # Epoch 4 (0-indexed) -> epoch 5, should save periodic checkpoint
        mock_trainer.epoch = 4
        cb.on_epoch_end(mock_trainer)
        mock_trainer.save.assert_called_once()

        # Epoch 5 (0-indexed) -> epoch 6, should not save
        mock_trainer.save.reset_mock()
        mock_trainer.epoch = 5
        cb.on_epoch_end(mock_trainer)
        mock_trainer.save.assert_not_called()

        # Epoch 9 (0-indexed) -> epoch 10, should save
        mock_trainer.epoch = 9
        cb.on_epoch_end(mock_trainer)
        mock_trainer.save.assert_called_once()

    def test_on_epoch_end_saves_best_model(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Best model is saved when mAP improves."""
        cb = CheckpointCallback(str(tmp_path), save_period=100, save_best=True)
        mock_trainer = create_mock_trainer(mocker)
        mock_trainer.epoch = 0

        # First epoch, should save as best
        mock_trainer.metrics.box.map = 0.5
        cb.on_epoch_end(mock_trainer)
        assert cb.best_map == 0.5
        assert mock_trainer.save.call_count == 1

        # Better mAP, should save again
        mock_trainer.save.reset_mock()
        mock_trainer.metrics.box.map = 0.6
        cb.on_epoch_end(mock_trainer)
        assert cb.best_map == 0.6
        assert mock_trainer.save.call_count == 1

        # Worse mAP, should not save
        mock_trainer.save.reset_mock()
        mock_trainer.metrics.box.map = 0.55
        cb.on_epoch_end(mock_trainer)
        assert mock_trainer.save.call_count == 0

    def test_on_epoch_end_handles_missing_metrics(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """on_epoch_end() handles missing metrics."""
        cb = CheckpointCallback(str(tmp_path), save_period=100, save_best=True)
        mock_trainer = mocker.MagicMock()
        mock_trainer.epoch = 0
        mock_trainer.metrics = None

        # Should not raise
        cb.on_epoch_end(mock_trainer)
        assert cb.best_map == 0.0


class TestTensorBoardCallback:
    """Tests for TensorBoardCallback."""

    def test_init_with_tensorboard_available(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """SummaryWriter is created when tensorboard is available."""
        mock_writer = mocker.MagicMock()
        # Direct instantiation test bypassing __init__
        cb = TensorBoardCallback.__new__(TensorBoardCallback)
        cb.log_dir = tmp_path
        cb.writer = mock_writer

        assert cb.writer is mock_writer

    def test_init_without_tensorboard(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Writer is None when tensorboard not installed."""
        # Simulate ImportError
        mocker.patch.dict("sys.modules", {"torch.utils.tensorboard": None})

        cb = TensorBoardCallback.__new__(TensorBoardCallback)
        cb.log_dir = tmp_path
        cb.writer = None

        assert cb.writer is None

    def test_on_epoch_end_returns_early_if_no_writer(self, mocker: MockerFixture) -> None:
        """on_epoch_end() returns early if writer is None."""
        cb = TensorBoardCallback.__new__(TensorBoardCallback)
        cb.writer = None
        mock_trainer = create_mock_trainer(mocker)

        # Should not raise
        cb.on_epoch_end(mock_trainer)

    def test_on_epoch_end_logs_scalars(self, mocker: MockerFixture) -> None:
        """on_epoch_end() logs all scalars."""
        cb = TensorBoardCallback.__new__(TensorBoardCallback)
        cb.writer = mocker.MagicMock()
        mock_trainer = create_mock_trainer(mocker)

        cb.on_epoch_end(mock_trainer)

        # Check that add_scalar was called for losses and metrics
        calls = cb.writer.add_scalar.call_args_list
        call_names = [c[0][0] for c in calls]

        assert "Loss/box" in call_names
        assert "Loss/cls" in call_names
        assert "Loss/dfl" in call_names
        assert "Loss/total" in call_names
        assert "Metrics/mAP50" in call_names
        assert "Metrics/mAP50-95" in call_names
        assert "Metrics/precision" in call_names
        assert "Metrics/recall" in call_names
        assert "Learning_Rate" in call_names

    def test_on_train_end_closes_writer(self, mocker: MockerFixture) -> None:
        """on_train_end() closes the writer."""
        cb = TensorBoardCallback.__new__(TensorBoardCallback)
        cb.writer = mocker.MagicMock()
        mock_trainer = mocker.MagicMock()

        cb.on_train_end(mock_trainer)

        cb.writer.close.assert_called_once()

    def test_on_train_end_handles_none_writer(self, mocker: MockerFixture) -> None:
        """on_train_end() handles None writer gracefully."""
        cb = TensorBoardCallback.__new__(TensorBoardCallback)
        cb.writer = None
        mock_trainer = mocker.MagicMock()

        # Should not raise
        cb.on_train_end(mock_trainer)


class TestWandbCallback:
    """Tests for WandbCallback."""

    def test_init_stores_config(self, mocker: MockerFixture) -> None:
        """Configuration values are stored."""
        cb = WandbCallback.__new__(WandbCallback)
        cb.project = "test-project"
        cb.entity = "test-entity"
        cb.config = {"epochs": 100}
        cb.run = None
        cb.wandb = mocker.MagicMock()  # pyright: ignore[reportAttributeAccessIssue]

        assert cb.project == "test-project"
        assert cb.entity == "test-entity"
        assert cb.config == {"epochs": 100}

    def test_on_epoch_end_returns_early_if_no_run(self, mocker: MockerFixture) -> None:
        """on_epoch_end() returns early if run is None."""
        cb = WandbCallback.__new__(WandbCallback)
        cb.run = None
        mock_trainer = create_mock_trainer(mocker)

        # Should not raise
        cb.on_epoch_end(mock_trainer)

    def test_on_epoch_end_logs_metrics(self, mocker: MockerFixture) -> None:
        """on_epoch_end() logs metrics via wandb."""
        cb = WandbCallback.__new__(WandbCallback)
        cb.run = mocker.MagicMock()
        cb.wandb = mocker.MagicMock()
        mock_trainer = create_mock_trainer(mocker)

        cb.on_epoch_end(mock_trainer)

        cb.wandb.log.assert_called_once()
        log_dict = cb.wandb.log.call_args[0][0]

        assert log_dict["epoch"] == 5
        assert "loss/box" in log_dict
        assert "loss/cls" in log_dict
        assert "loss/dfl" in log_dict
        assert "loss/total" in log_dict
        assert "metrics/mAP50" in log_dict
        assert "metrics/mAP50-95" in log_dict

    def test_on_train_end_finishes_run(self, mocker: MockerFixture) -> None:
        """on_train_end() finishes the W&B run."""
        cb = WandbCallback.__new__(WandbCallback)
        cb.run = mocker.MagicMock()
        mock_trainer = mocker.MagicMock()

        cb.on_train_end(mock_trainer)

        cb.run.finish.assert_called_once()

    def test_on_train_end_handles_none_run(self, mocker: MockerFixture) -> None:
        """on_train_end() handles None run gracefully."""
        cb = WandbCallback.__new__(WandbCallback)
        cb.run = None
        mock_trainer = mocker.MagicMock()

        # Should not raise
        cb.on_train_end(mock_trainer)


class TestTrainer:
    """Tests for Trainer class."""

    def test_init_creates_output_dir(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Output directory is created on init."""
        mocker.patch("screencropnet_yolov11.training.YOLO")
        mock_model = create_mock_yolo_model(mocker)
        output_dir = tmp_path / "training_output"

        # Mock TensorBoardCallback to avoid tensorboard import
        mocker.patch("screencropnet_yolov11.training.TensorBoardCallback")

        Trainer(mock_model, "data.yaml", str(output_dir), {})

        assert output_dir.exists()

    def test_init_sets_history_and_callbacks(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """History and callbacks are initialized."""
        mock_model = create_mock_yolo_model(mocker)
        mocker.patch("screencropnet_yolov11.training.TensorBoardCallback")

        trainer = Trainer(mock_model, "data.yaml", str(tmp_path), {})

        assert isinstance(trainer.history, TrainingHistory)
        assert len(trainer.callbacks) > 0

    def test_setup_default_callbacks_with_early_stopping(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Early stopping callback added when patience > 0."""
        mock_model = create_mock_yolo_model(mocker)
        mocker.patch("screencropnet_yolov11.training.TensorBoardCallback")

        trainer = Trainer(mock_model, "data.yaml", str(tmp_path), {"patience": 10})

        early_stopping_callbacks = [cb for cb in trainer.callbacks if isinstance(cb, EarlyStopping)]
        assert len(early_stopping_callbacks) == 1
        assert early_stopping_callbacks[0].patience == 10

    def test_setup_default_callbacks_without_early_stopping(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """No early stopping when patience is 0."""
        mock_model = create_mock_yolo_model(mocker)
        mocker.patch("screencropnet_yolov11.training.TensorBoardCallback")

        trainer = Trainer(mock_model, "data.yaml", str(tmp_path), {"patience": 0})

        early_stopping_callbacks = [cb for cb in trainer.callbacks if isinstance(cb, EarlyStopping)]
        assert len(early_stopping_callbacks) == 0

    def test_setup_default_callbacks_tensorboard_disabled(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """TensorBoard callback not added when disabled."""
        mock_model = create_mock_yolo_model(mocker)
        tb_mock = mocker.patch("screencropnet_yolov11.training.TensorBoardCallback")

        Trainer(mock_model, "data.yaml", str(tmp_path), {"tensorboard": False})

        tb_mock.assert_not_called()

    def test_setup_default_callbacks_wandb_enabled(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """WandB callback added when enabled."""
        mock_model = create_mock_yolo_model(mocker)
        mocker.patch("screencropnet_yolov11.training.TensorBoardCallback")
        wandb_mock = mocker.patch("screencropnet_yolov11.training.WandbCallback")

        config = {"wandb": {"enabled": True, "project": "test-proj"}}
        Trainer(mock_model, "data.yaml", str(tmp_path), config)

        wandb_mock.assert_called_once()

    def test_add_callback_appends(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """add_callback() appends to callbacks list."""
        mock_model = create_mock_yolo_model(mocker)
        mocker.patch("screencropnet_yolov11.training.TensorBoardCallback")

        trainer = Trainer(mock_model, "data.yaml", str(tmp_path), {})
        initial_count = len(trainer.callbacks)

        custom_callback = TrainingCallback()
        trainer.add_callback(custom_callback)

        assert len(trainer.callbacks) == initial_count + 1
        assert trainer.callbacks[-1] is custom_callback

    def test_register_callbacks_hooks_to_model(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """_register_callbacks() registers hooks with model."""
        mock_model = create_mock_yolo_model(mocker)
        mocker.patch("screencropnet_yolov11.training.TensorBoardCallback")

        trainer = Trainer(mock_model, "data.yaml", str(tmp_path), {})
        trainer._register_callbacks()

        # Verify add_callback was called for each hook type
        call_args = [call[0][0] for call in mock_model.add_callback.call_args_list]
        assert "on_train_start" in call_args
        assert "on_train_end" in call_args
        assert "on_train_epoch_start" in call_args
        assert "on_train_epoch_end" in call_args
        assert "on_val_start" in call_args
        assert "on_val_end" in call_args

    def test_train_calls_model_train(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """train() calls model.train() with correct args."""
        mock_model = create_mock_yolo_model(mocker)
        mocker.patch("screencropnet_yolov11.training.TensorBoardCallback")
        mocker.patch("time.time", return_value=1000.0)

        config = {
            "epochs": 50,
            "batch_size": 32,
            "image_size": 640,
            "device": "cpu",
        }
        trainer = Trainer(mock_model, "data.yaml", str(tmp_path), config)
        trainer.train()

        mock_model.train.assert_called_once()
        call_kwargs = mock_model.train.call_args[1]
        assert call_kwargs["data"] == "data.yaml"
        assert call_kwargs["epochs"] == 50
        assert call_kwargs["batch"] == 32
        assert call_kwargs["imgsz"] == 640
        assert call_kwargs["device"] == "cpu"

    def test_train_saves_history(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """train() saves history JSON file."""
        mock_model = create_mock_yolo_model(mocker)
        mocker.patch("screencropnet_yolov11.training.TensorBoardCallback")
        mocker.patch("time.time", return_value=1000.0)

        trainer = Trainer(mock_model, "data.yaml", str(tmp_path), {})
        trainer.train()

        history_path = tmp_path / "training_history.json"
        assert history_path.exists()

    def test_train_returns_history(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """train() returns TrainingHistory instance."""
        mock_model = create_mock_yolo_model(mocker)
        mocker.patch("screencropnet_yolov11.training.TensorBoardCallback")
        mocker.patch("time.time", return_value=1000.0)

        trainer = Trainer(mock_model, "data.yaml", str(tmp_path), {})
        result = trainer.train()

        assert isinstance(result, TrainingHistory)

    def test_train_handles_exception(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """train() logs and re-raises exceptions."""
        mock_model = create_mock_yolo_model(mocker)
        mock_model.train.side_effect = RuntimeError("Training failed")
        mocker.patch("screencropnet_yolov11.training.TensorBoardCallback")

        trainer = Trainer(mock_model, "data.yaml", str(tmp_path), {})

        with pytest.raises(RuntimeError, match="Training failed"):
            trainer.train()

    def test_resume_loads_checkpoint(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """resume() loads model from checkpoint and continues training."""
        mock_model = create_mock_yolo_model(mocker)
        mock_yolo_class = mocker.patch("screencropnet_yolov11.training.YOLO")
        mock_yolo_class.return_value = mock_model
        mocker.patch("screencropnet_yolov11.training.TensorBoardCallback")
        mocker.patch("time.time", return_value=1000.0)

        trainer = Trainer(mock_model, "data.yaml", str(tmp_path), {})
        trainer.resume("/path/to/checkpoint.pt")

        # YOLO should be called with checkpoint path
        mock_yolo_class.assert_called_with("/path/to/checkpoint.pt")


class TestCreateAblationStudy:
    """Tests for create_ablation_study function."""

    def test_single_parameter_ablation(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Single parameter ablation runs correct number of trials."""
        mock_model = create_mock_yolo_model(mocker)
        mock_factory = mocker.MagicMock()
        mock_factory.create_model.return_value = mock_model
        mock_factory.config = mocker.MagicMock()

        mocker.patch("screencropnet_yolov11.training.TensorBoardCallback")
        mocker.patch("time.time", return_value=1000.0)

        ablation_config = {"learning_rate": [0.01, 0.001, 0.0001]}

        results = create_ablation_study(
            mock_factory,
            "data.yaml",
            str(tmp_path),
            ablation_config,
        )

        assert len(results) == 3
        assert mock_factory.create_model.call_count == 3

    def test_multiple_parameter_combinations(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Multiple parameters create Cartesian product of combinations."""
        mock_model = create_mock_yolo_model(mocker)
        mock_factory = mocker.MagicMock()
        mock_factory.create_model.return_value = mock_model
        mock_factory.config = mocker.MagicMock()

        mocker.patch("screencropnet_yolov11.training.TensorBoardCallback")
        mocker.patch("time.time", return_value=1000.0)

        ablation_config = {
            "learning_rate": [0.01, 0.001],
            "batch_size": [16, 32],
        }

        results = create_ablation_study(
            mock_factory,
            "data.yaml",
            str(tmp_path),
            ablation_config,
        )

        # 2 * 2 = 4 combinations
        assert len(results) == 4
        assert mock_factory.create_model.call_count == 4

    def test_saves_results_summary(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Ablation summary JSON is saved."""
        mock_model = create_mock_yolo_model(mocker)
        mock_factory = mocker.MagicMock()
        mock_factory.create_model.return_value = mock_model
        mock_factory.config = mocker.MagicMock()

        mocker.patch("screencropnet_yolov11.training.TensorBoardCallback")
        mocker.patch("time.time", return_value=1000.0)

        ablation_config = {"epochs": [10, 20]}

        create_ablation_study(
            mock_factory,
            "data.yaml",
            str(tmp_path),
            ablation_config,
        )

        summary_path = tmp_path / "ablation_summary.json"
        assert summary_path.exists()

        with open(summary_path) as f:
            summary = json.load(f)

        assert len(summary) == 2

    def test_results_contain_training_history(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Results dictionary contains TrainingHistory objects."""
        mock_model = create_mock_yolo_model(mocker)
        mock_factory = mocker.MagicMock()
        mock_factory.create_model.return_value = mock_model
        mock_factory.config = mocker.MagicMock()

        mocker.patch("screencropnet_yolov11.training.TensorBoardCallback")
        mocker.patch("time.time", return_value=1000.0)

        ablation_config = {"lr": [0.01]}

        results = create_ablation_study(
            mock_factory,
            "data.yaml",
            str(tmp_path),
            ablation_config,
        )

        for history in results.values():
            assert isinstance(history, TrainingHistory)
