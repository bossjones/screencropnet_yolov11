"""
Training module for YOLO 11 Twitter Screenshot Detection.

This module handles:
- Training loop with callbacks
- Metrics tracking and logging
- Early stopping
- Checkpoint management
- TensorBoard integration
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""

    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    box_loss: float = 0.0
    cls_loss: float = 0.0
    dfl_loss: float = 0.0
    mAP50: float = 0.0
    mAP50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    learning_rate: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "box_loss": self.box_loss,
            "cls_loss": self.cls_loss,
            "dfl_loss": self.dfl_loss,
            "mAP50": self.mAP50,
            "mAP50_95": self.mAP50_95,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "learning_rate": self.learning_rate,
        }


@dataclass
class TrainingHistory:
    """Container for full training history."""

    metrics: list[TrainingMetrics] = field(default_factory=list)
    best_epoch: int = 0
    best_mAP50: float = 0.0
    best_mAP50_95: float = 0.0
    training_time: float = 0.0

    def add_metrics(self, metrics: TrainingMetrics) -> None:
        """Add epoch metrics to history."""
        self.metrics.append(metrics)

        # Track best metrics
        if metrics.mAP50_95 > self.best_mAP50_95:
            self.best_epoch = metrics.epoch
            self.best_mAP50 = metrics.mAP50
            self.best_mAP50_95 = metrics.mAP50_95

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metrics": [m.to_dict() for m in self.metrics],
            "best_epoch": self.best_epoch,
            "best_mAP50": self.best_mAP50,
            "best_mAP50_95": self.best_mAP50_95,
            "training_time": self.training_time,
        }

    def save(self, path: str) -> None:
        """Save history to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class TrainingCallback:
    """Base class for training callbacks."""

    def on_train_start(self, trainer: Any) -> None:
        """Called when training starts."""
        pass

    def on_train_end(self, trainer: Any) -> None:
        """Called when training ends."""
        pass

    def on_epoch_start(self, trainer: Any) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, trainer: Any) -> None:
        """Called at the end of each epoch."""
        pass

    def on_batch_start(self, trainer: Any) -> None:
        """Called at the start of each batch."""
        pass

    def on_batch_end(self, trainer: Any) -> None:
        """Called at the end of each batch."""
        pass

    def on_val_start(self, trainer: Any) -> None:
        """Called when validation starts."""
        pass

    def on_val_end(self, trainer: Any) -> None:
        """Called when validation ends."""
        pass


class MetricsLogger(TrainingCallback):
    """Callback for logging training metrics."""

    def __init__(self, history: TrainingHistory, log_interval: int = 1):
        """
        Initialize metrics logger.

        Args:
            history: TrainingHistory instance to update
            log_interval: Epochs between detailed logging
        """
        self.history = history
        self.log_interval = log_interval
        self.epoch_start_time = None

    def on_epoch_start(self, trainer: Any) -> None:
        """Record epoch start time."""
        import time

        self.epoch_start_time = time.time()

    def on_epoch_end(self, trainer: Any) -> None:
        """Log metrics at end of epoch."""
        import time

        epoch = trainer.epoch + 1

        # Extract metrics from trainer
        metrics = TrainingMetrics(
            epoch=epoch,
            train_loss=float(trainer.loss.mean()) if trainer.loss is not None else 0.0,
            box_loss=float(trainer.loss_items[0]) if trainer.loss_items is not None else 0.0,
            cls_loss=float(trainer.loss_items[1]) if trainer.loss_items is not None else 0.0,
            dfl_loss=float(trainer.loss_items[2]) if trainer.loss_items is not None else 0.0,
        )

        # Get validation metrics if available
        if hasattr(trainer, "metrics") and trainer.metrics is not None:
            metrics.mAP50 = float(trainer.metrics.box.map50)
            metrics.mAP50_95 = float(trainer.metrics.box.map)
            metrics.precision = float(trainer.metrics.box.mp)
            metrics.recall = float(trainer.metrics.box.mr)
            # Calculate F1
            if metrics.precision + metrics.recall > 0:
                metrics.f1 = (
                    2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall)
                )

        # Get learning rate
        if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
            metrics.learning_rate = trainer.optimizer.param_groups[0]["lr"]

        self.history.add_metrics(metrics)

        # Log to console
        if epoch % self.log_interval == 0:
            epoch_time = time.time() - self.epoch_start_time
            logger.info(
                f"Epoch {epoch}: "
                f"loss={metrics.train_loss:.4f}, "
                f"mAP50={metrics.mAP50:.4f}, "
                f"mAP50-95={metrics.mAP50_95:.4f}, "
                f"P={metrics.precision:.4f}, "
                f"R={metrics.recall:.4f}, "
                f"F1={metrics.f1:.4f}, "
                f"time={epoch_time:.1f}s"
            )


class EarlyStopping(TrainingCallback):
    """Early stopping callback based on validation metrics."""

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.001,
        monitor: str = "mAP50_95",
        mode: str = "max",
    ):
        """
        Initialize early stopping.

        Args:
            patience: Epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            monitor: Metric to monitor
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.best_epoch = 0
        self.should_stop = False

    def on_epoch_end(self, trainer: Any) -> None:
        """Check if training should stop."""
        # Get current metric value
        if hasattr(trainer, "metrics") and trainer.metrics is not None:
            if self.monitor == "mAP50":
                current = trainer.metrics.box.map50
            elif self.monitor == "mAP50_95":
                current = trainer.metrics.box.map
            else:
                return
        else:
            return

        # Check for improvement
        improved = False
        if self.mode == "max":
            improved = current > self.best_value + self.min_delta
        else:
            improved = current < self.best_value - self.min_delta

        if improved:
            self.best_value = current
            self.best_epoch = trainer.epoch + 1
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            logger.info(
                f"Early stopping triggered. "
                f"Best {self.monitor}: {self.best_value:.4f} at epoch {self.best_epoch}"
            )
            self.should_stop = True
            trainer.stop = True


class CheckpointCallback(TrainingCallback):
    """Callback for saving model checkpoints."""

    def __init__(self, save_dir: str, save_period: int = 10, save_best: bool = True):
        """
        Initialize checkpoint callback.

        Args:
            save_dir: Directory for saving checkpoints
            save_period: Save checkpoint every N epochs
            save_best: Save best model separately
        """
        self.save_dir = Path(save_dir)
        self.save_period = save_period
        self.save_best = save_best
        self.best_map = 0.0

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, trainer: Any) -> None:
        """Save checkpoint if needed."""
        epoch = trainer.epoch + 1

        # Save periodic checkpoint
        if epoch % self.save_period == 0:
            checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
            trainer.save(checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if self.save_best and hasattr(trainer, "metrics") and trainer.metrics is not None:
            current_map = trainer.metrics.box.map
            if current_map > self.best_map:
                self.best_map = current_map
                best_path = self.save_dir / "best.pt"
                trainer.save(best_path)
                logger.info(f"Saved new best model (mAP50-95: {current_map:.4f})")


class TensorBoardCallback(TrainingCallback):
    """Callback for TensorBoard logging."""

    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard callback.

        Args:
            log_dir: Directory for TensorBoard logs
        """
        self.log_dir = Path(log_dir)
        self.writer = None

        try:
            from torch.utils.tensorboard import SummaryWriter

            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            logger.info(f"TensorBoard logging enabled: {self.log_dir}")
        except ImportError:
            logger.warning("TensorBoard not installed. Install with: pip install tensorboard")

    def on_epoch_end(self, trainer: Any) -> None:
        """Log metrics to TensorBoard."""
        if self.writer is None:
            return

        epoch = trainer.epoch + 1

        # Log losses
        if trainer.loss_items is not None:
            self.writer.add_scalar("Loss/box", trainer.loss_items[0], epoch)
            self.writer.add_scalar("Loss/cls", trainer.loss_items[1], epoch)
            self.writer.add_scalar("Loss/dfl", trainer.loss_items[2], epoch)
            self.writer.add_scalar("Loss/total", trainer.loss.mean(), epoch)

        # Log validation metrics
        if hasattr(trainer, "metrics") and trainer.metrics is not None:
            self.writer.add_scalar("Metrics/mAP50", trainer.metrics.box.map50, epoch)
            self.writer.add_scalar("Metrics/mAP50-95", trainer.metrics.box.map, epoch)
            self.writer.add_scalar("Metrics/precision", trainer.metrics.box.mp, epoch)
            self.writer.add_scalar("Metrics/recall", trainer.metrics.box.mr, epoch)

        # Log learning rate
        if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
            lr = trainer.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("Learning_Rate", lr, epoch)

    def on_train_end(self, trainer: Any) -> None:
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()


class WandbCallback(TrainingCallback):
    """Callback for Weights & Biases logging."""

    def __init__(self, project: str, entity: str = None, config: dict = None):
        """
        Initialize W&B callback.

        Args:
            project: W&B project name
            entity: W&B entity/team name
            config: Configuration dictionary to log
        """
        self.project = project
        self.entity = entity
        self.config = config
        self.run = None

        try:
            import wandb

            self.wandb = wandb
            self.run = wandb.init(project=project, entity=entity, config=config)
            logger.info(f"W&B logging enabled: {project}")
        except ImportError:
            logger.warning("wandb not installed. Install with: pip install wandb")

    def on_epoch_end(self, trainer: Any) -> None:
        """Log metrics to W&B."""
        if self.run is None:
            return

        log_dict = {"epoch": trainer.epoch + 1}

        # Log losses
        if trainer.loss_items is not None:
            log_dict.update(
                {
                    "loss/box": float(trainer.loss_items[0]),
                    "loss/cls": float(trainer.loss_items[1]),
                    "loss/dfl": float(trainer.loss_items[2]),
                    "loss/total": float(trainer.loss.mean()),
                }
            )

        # Log validation metrics
        if hasattr(trainer, "metrics") and trainer.metrics is not None:
            log_dict.update(
                {
                    "metrics/mAP50": trainer.metrics.box.map50,
                    "metrics/mAP50-95": trainer.metrics.box.map,
                    "metrics/precision": trainer.metrics.box.mp,
                    "metrics/recall": trainer.metrics.box.mr,
                }
            )

        self.wandb.log(log_dict)

    def on_train_end(self, trainer: Any) -> None:
        """Finish W&B run."""
        if self.run is not None:
            self.run.finish()


class Trainer:
    """
    Main trainer class for YOLO 11 Twitter Screenshot Detection.

    Wraps Ultralytics training with custom callbacks and logging.
    """

    def __init__(self, model: YOLO, data_yaml: str, output_dir: str, config: dict[str, Any]):
        """
        Initialize trainer.

        Args:
            model: YOLO model instance
            data_yaml: Path to dataset YAML file
            output_dir: Output directory for training results
            config: Training configuration dictionary
        """
        self.model = model
        self.data_yaml = data_yaml
        self.output_dir = Path(output_dir)
        self.config = config

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tracking
        self.history = TrainingHistory()
        self.callbacks: list[TrainingCallback] = []

        # Set up default callbacks
        self._setup_default_callbacks()

    def _setup_default_callbacks(self) -> None:
        """Set up default training callbacks."""
        # Metrics logger
        self.callbacks.append(MetricsLogger(self.history))

        # Early stopping
        if self.config.get("patience", 0) > 0:
            self.callbacks.append(
                EarlyStopping(
                    patience=self.config.get("patience", 20),
                    min_delta=self.config.get("min_delta", 0.001),
                )
            )

        # Checkpointing
        self.callbacks.append(
            CheckpointCallback(
                save_dir=str(self.output_dir / "weights"),
                save_period=self.config.get("save_period", 10),
            )
        )

        # TensorBoard
        if self.config.get("tensorboard", True):
            self.callbacks.append(TensorBoardCallback(log_dir=str(self.output_dir / "tensorboard")))

        # W&B
        wandb_config = self.config.get("wandb", {})
        if wandb_config.get("enabled", False):
            self.callbacks.append(
                WandbCallback(
                    project=wandb_config.get("project", "twitter-detection"),
                    entity=wandb_config.get("entity"),
                    config=self.config,
                )
            )

    def add_callback(self, callback: TrainingCallback) -> None:
        """Add custom callback."""
        self.callbacks.append(callback)

    def _register_callbacks(self) -> None:
        """Register callbacks with YOLO trainer."""

        def create_hook(method_name):
            def hook(trainer):
                for callback in self.callbacks:
                    method = getattr(callback, method_name, None)
                    if method:
                        method(trainer)

            return hook

        # Register hooks
        self.model.add_callback("on_train_start", create_hook("on_train_start"))
        self.model.add_callback("on_train_end", create_hook("on_train_end"))
        self.model.add_callback("on_train_epoch_start", create_hook("on_epoch_start"))
        self.model.add_callback("on_train_epoch_end", create_hook("on_epoch_end"))
        self.model.add_callback("on_val_start", create_hook("on_val_start"))
        self.model.add_callback("on_val_end", create_hook("on_val_end"))

    def train(self) -> TrainingHistory:
        """
        Run the training loop.

        Returns:
            TrainingHistory with all metrics
        """
        import time

        logger.info("Starting training...")
        logger.info(f"Output directory: {self.output_dir}")

        # Register callbacks
        self._register_callbacks()

        # Get augmentation config
        augmentation = self.config.get("augmentation", {})

        # Build training arguments
        train_args = {
            "data": self.data_yaml,
            "epochs": self.config.get("epochs", 100),
            "batch": self.config.get("batch_size", 16),
            "imgsz": self.config.get("image_size", 640),
            "device": self.config.get("device", "auto"),
            "workers": self.config.get("workers", 8),
            "patience": self.config.get("patience", 20),
            "project": str(self.output_dir),
            "name": "train",
            "exist_ok": True,
            "pretrained": True,
            "optimizer": self.config.get("optimizer", "SGD"),
            "lr0": self.config.get("learning_rate", 0.01),
            "momentum": self.config.get("momentum", 0.937),
            "weight_decay": self.config.get("weight_decay", 0.0005),
            "warmup_epochs": self.config.get("warmup_epochs", 3),
            "amp": self.config.get("amp", True),
            "save": True,
            "save_period": self.config.get("save_period", 10),
            "plots": True,
            "verbose": True,
            # Augmentation
            **augmentation,
        }

        # Start training
        start_time = time.time()

        try:
            self.model.train(**train_args)

            self.history.training_time = time.time() - start_time

            logger.info(f"Training completed in {self.history.training_time:.1f}s")
            logger.info(f"Best mAP50: {self.history.best_mAP50:.4f}")
            logger.info(f"Best mAP50-95: {self.history.best_mAP50_95:.4f}")
            logger.info(f"Best epoch: {self.history.best_epoch}")

            # Save training history
            history_path = self.output_dir / "training_history.json"
            self.history.save(str(history_path))
            logger.info(f"Training history saved to: {history_path}")

            return self.history

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def resume(self, checkpoint_path: str) -> TrainingHistory:
        """
        Resume training from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            TrainingHistory with all metrics
        """
        logger.info(f"Resuming training from: {checkpoint_path}")

        # Load checkpoint
        self.model = YOLO(checkpoint_path)

        # Continue training
        return self.train()


def create_ablation_study(
    model_factory: Any, data_yaml: str, output_dir: str, ablation_config: dict[str, list]
) -> dict[str, TrainingHistory]:
    """
    Run ablation study with different configurations.

    Args:
        model_factory: ModelFactory instance
        data_yaml: Path to dataset YAML
        output_dir: Base output directory
        ablation_config: Dictionary of parameter lists to test

    Returns:
        Dictionary mapping config name to training history
    """
    from itertools import product

    results = {}
    output_base = Path(output_dir)

    # Generate all combinations
    param_names = list(ablation_config.keys())
    param_values = list(ablation_config.values())
    combinations = list(product(*param_values))

    logger.info(f"Running ablation study with {len(combinations)} configurations")

    for combo in combinations:
        # Create config name
        config_name = "_".join(f"{n}={v}" for n, v in zip(param_names, combo, strict=True))
        logger.info(f"\nTesting configuration: {config_name}")

        # Update config
        test_config = model_factory.config.__dict__.copy()
        for name, value in zip(param_names, combo, strict=True):
            if hasattr(model_factory.config, name):
                setattr(model_factory.config, name, value)

        # Create model and train
        model = model_factory.create_model()

        trainer = Trainer(
            model=model,
            data_yaml=data_yaml,
            output_dir=str(output_base / config_name),
            config=test_config,
        )

        history = trainer.train()
        results[config_name] = history

        # Log comparison
        logger.info(f"Configuration {config_name}: mAP50-95={history.best_mAP50_95:.4f}")

    # Save ablation results
    summary = {
        name: {
            "best_mAP50_95": h.best_mAP50_95,
            "best_mAP50": h.best_mAP50,
            "best_epoch": h.best_epoch,
            "training_time": h.training_time,
        }
        for name, h in results.items()
    }

    summary_path = output_base / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Ablation study complete. Summary saved to: {summary_path}")

    return results
