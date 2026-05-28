#!/usr/bin/env python3
"""
YOLO 26 Twitter Screenshot Detection - Main Training Script

Production-ready training script for detecting and classifying bounding boxes
in Twitter screenshots using Ultralytics YOLO 26.

Usage:
    python train.py --config config/config.yaml
    python train.py --data ./datasets/twitter --epochs 100 --batch 16
    python train.py --resume runs/twitter_detect/train/weights/last.pt

Author: Generated for AI Foundations Team
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Add src to path
# sys.path.insert(0, str(Path(__file__).parent / 'src'))
from .dataset_utils import (
    DatasetSplitter,
    DatasetValidator,
    RoboflowLoader,
    check_class_imbalance,
    create_dataset_yaml,
    display_dataset_stats,
)
from .evaluation import EvaluationResults, Evaluator
from .model import AugmentationConfig, ModelConfig, ModelExporter, ModelFactory
from .training import Trainer, TrainingHistory, create_ablation_study
from .visualization import (
    ConfusionMatrixVisualizer,
    ResultsDashboard,
    TrainingVisualizer,
)


# Configure logging
def setup_logging(output_dir: str, log_level: str = "INFO") -> None:
    """Configure logging to file and console."""
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    # Reduce verbosity of some loggers
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)

BANNER = "YOLO 26 Twitter Screenshot Detection"


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def merge_config_with_args(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Merge command line arguments with config file."""
    # Override config with command line args
    if args.data:
        config["dataset"]["path"] = args.data
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch:
        config["training"]["batch_size"] = args.batch
    if args.imgsz:
        config["training"]["image_size"] = args.imgsz
    if args.device:
        config["device"]["type"] = args.device
    if args.workers:
        config["training"]["workers"] = args.workers
    if args.output:
        config["logging"]["output_dir"] = args.output
    if args.model_size:
        config["model"]["size"] = args.model_size

    return config


def validate_dataset(config: dict[str, Any]) -> bool:
    """Validate dataset integrity."""
    logger.info("Validating dataset...")

    dataset_path = config["dataset"]["path"]
    class_names = config["model"]["class_names"]

    validator = DatasetValidator(dataset_path, class_names)
    is_valid, stats, errors = validator.validate()

    display_dataset_stats(stats)

    if not is_valid:
        logger.error("Dataset validation failed:")
        for error in errors[:10]:  # Show first 10 errors
            logger.error(f"  - {error}")
        if len(errors) > 10:
            logger.error(f"  ... and {len(errors) - 10} more errors")
        return False

    # Check for class imbalance
    warnings = check_class_imbalance(stats.class_distribution)
    for warning in warnings:
        logger.warning(warning)

    logger.info("Dataset validation passed!")
    return True


def load_dataset_from_roboflow(config: dict[str, Any]) -> str:
    """Load dataset from Roboflow if configured."""
    rf_config = config["dataset"]["roboflow"]

    if not rf_config.get("enabled", False):
        return config["dataset"]["path"]

    api_key = rf_config.get("api_key") or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError(
            "Roboflow API key not provided. Set ROBOFLOW_API_KEY environment variable."
        )

    loader = RoboflowLoader(
        api_key=api_key,
        workspace=rf_config["workspace"],
        project=rf_config["project"],
        version=rf_config["version"],
        output_path=config["dataset"]["path"],
        format=rf_config.get("format", "yolov11"),
    )

    return str(loader.download())


def split_dataset_if_needed(config: dict[str, Any]) -> None:
    """Split dataset into train/val/test if needed."""
    if not config["dataset"].get("auto_split", False):
        return

    logger.info("Performing automatic train/val/test split...")

    splitter = DatasetSplitter(
        source_path=config["dataset"]["path"],
        output_path=config["dataset"]["path"],
        train_ratio=config["dataset"]["split_ratios"]["train"],
        val_ratio=config["dataset"]["split_ratios"]["val"],
        test_ratio=config["dataset"]["split_ratios"]["test"],
        seed=config["dataset"].get("seed", 42),
    )

    counts = splitter.split()
    logger.info(f"Dataset split complete: {counts}")


def train_model(config: dict[str, Any], resume_path: str | None = None) -> TrainingHistory:
    """Run model training."""
    logger.info("=" * 60)
    logger.info("STARTING YOLO 26 TRAINING")
    logger.info("=" * 60)

    # Create output directory
    output_dir = Path(config["logging"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset YAML
    data_yaml = create_dataset_yaml(
        dataset_path=config["dataset"]["path"],
        class_names=config["model"]["class_names"],
        output_path=str(output_dir / "dataset.yaml"),
    )

    # Initialize model
    model_config = ModelConfig(
        size=config["model"]["size"],
        weights=config["model"].get("weights"),
        num_classes=len(config["model"]["class_names"]),
        device=config["device"]["type"],
        multi_gpu=config["device"].get("multi_gpu", False),
        gpu_ids=config["device"].get("gpu_ids", [0, 1]),
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        image_size=config["training"]["image_size"],
        learning_rate=config["training"]["learning_rate"],
        optimizer=config["training"]["optimizer"],
        momentum=config["training"]["momentum"],
        weight_decay=config["training"]["weight_decay"],
        warmup_epochs=config["training"]["warmup_epochs"],
        patience=config["training"]["patience"],
        amp=config["training"]["amp"],
        workers=config["training"]["workers"],
    )

    factory = ModelFactory(model_config)
    model = factory.create_model()

    # Get training config
    train_config = {
        **config["training"],
        "device": factory.device,
        "tensorboard": config["logging"].get("tensorboard", True),
        "wandb": config["logging"].get("wandb", {}),
        "augmentation": AugmentationConfig.get_augmentation("twitter"),
    }

    # Override with custom augmentation if provided
    if "augmentation" in config:
        augmentation_cfg = train_config["augmentation"]
        if isinstance(augmentation_cfg, dict):
            augmentation_cfg.update(config["augmentation"])

    # Initialize trainer
    trainer = Trainer(
        model=model, data_yaml=data_yaml, output_dir=str(output_dir), config=train_config
    )

    # Train or resume
    if resume_path:
        history = trainer.resume(resume_path)
    else:
        history = trainer.train()

    return history


def evaluate_model(config: dict[str, Any], model_path: str, output_dir: str) -> EvaluationResults:
    """Evaluate trained model."""
    logger.info("=" * 60)
    logger.info("EVALUATING MODEL")
    logger.info("=" * 60)

    from ultralytics import YOLO

    model = YOLO(model_path)

    # Create data yaml path
    data_yaml = str(Path(output_dir) / "dataset.yaml")

    evaluator = Evaluator(
        model=model,
        data_yaml=data_yaml,
        class_names=config["model"]["class_names"],
        device=config["device"]["type"],
    )

    # Run evaluation
    results = evaluator.evaluate(
        split="val",
        conf=config["inference"]["confidence"],
        iou=config["inference"]["iou_threshold"],
        batch_size=config["training"]["batch_size"],
        image_size=config["training"]["image_size"],
    )

    # Save results
    results_path = Path(output_dir) / "evaluation_results.json"
    results.save(str(results_path))
    logger.info(f"Evaluation results saved to: {results_path}")

    return results


def export_model(config: dict[str, Any], model_path: str, output_dir: str) -> dict[str, str]:
    """Export model to various formats."""
    logger.info("=" * 60)
    logger.info("EXPORTING MODEL")
    logger.info("=" * 60)

    from ultralytics import YOLO

    model = YOLO(model_path)

    exporter = ModelExporter(model, output_dir)

    export_config = config.get("export", {})
    formats = export_config.get("formats", ["pytorch", "onnx"])

    exported = exporter.export(
        formats=formats,
        image_size=config["training"]["image_size"],
        half=export_config.get("quantization", {}).get("enabled", False)
        and export_config.get("quantization", {}).get("type") == "fp16",
        dynamic=export_config.get("onnx", {}).get("dynamic", False),
        simplify=export_config.get("onnx", {}).get("simplify", True),
        opset=export_config.get("onnx", {}).get("opset", 12),
    )

    logger.info(f"Exported models: {list(exported.keys())}")
    return exported


def create_visualizations(
    config: dict[str, Any], history: TrainingHistory, results: EvaluationResults, output_dir: str
) -> None:
    """Create training visualizations."""
    logger.info("Creating visualizations...")

    viz_dir = Path(output_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Training curves
    training_viz = TrainingVisualizer(str(viz_dir))

    # Convert history to dict format
    history_dict = {
        "train_loss": [m.train_loss for m in history.metrics],
        "val_loss": [m.val_loss for m in history.metrics],
        "mAP50": [m.mAP50 for m in history.metrics],
        "mAP50_95": [m.mAP50_95 for m in history.metrics],
        "precision": [m.precision for m in history.metrics],
        "recall": [m.recall for m in history.metrics],
        "f1": [m.f1 for m in history.metrics],
        "learning_rate": [m.learning_rate for m in history.metrics],
        "box_loss": [m.box_loss for m in history.metrics],
        "cls_loss": [m.cls_loss for m in history.metrics],
        "dfl_loss": [m.dfl_loss for m in history.metrics],
    }

    training_viz.plot_training_curves(history_dict, str(viz_dir / "training_curves.png"))
    training_viz.plot_loss_components(history_dict, str(viz_dir / "loss_components.png"))

    # Confusion matrix
    if results.confusion_matrix is not None:
        ConfusionMatrixVisualizer.plot_confusion_matrix(
            results.confusion_matrix,
            config["model"]["class_names"],
            save_path=str(viz_dir / "confusion_matrix.png"),
        )

    # Results dashboard
    dashboard = ResultsDashboard(str(viz_dir))
    dashboard.create_dashboard(
        training_history=history_dict,
        evaluation_results=results.to_dict(),
        class_names=config["model"]["class_names"],
        save_path=str(viz_dir / "results_dashboard.png"),
    )

    logger.info(f"Visualizations saved to: {viz_dir}")


def run_ablation_study(config: dict[str, Any]) -> None:
    """Run ablation study if configured."""
    ablation_config = config.get("ablation", {})

    if not ablation_config.get("enabled", False):
        return

    logger.info("=" * 60)
    logger.info("RUNNING ABLATION STUDY")
    logger.info("=" * 60)

    output_dir = Path(config["logging"]["output_dir"]) / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model factory
    model_config = ModelConfig(
        size=config["model"]["size"],
        device=config["device"]["type"],
    )
    factory = ModelFactory(model_config)

    # Run ablation
    create_ablation_study(
        model_factory=factory,
        data_yaml=str(Path(config["logging"]["output_dir"]) / "dataset.yaml"),
        output_dir=str(output_dir),
        ablation_config=ablation_config.get("parameters", {}),
    )

    logger.info(f"Ablation study complete. Results saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=f"{BANNER} Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config config/config.yaml
  %(prog)s --data ./datasets/twitter --epochs 100
  %(prog)s --resume runs/twitter_detect/train/weights/last.pt
        """,
    )

    # Config file
    parser.add_argument(
        "--config", "-c", type=str, default="config/config.yaml", help="Path to config file"
    )

    # Dataset
    parser.add_argument("--data", "-d", type=str, help="Dataset path")

    # Training
    parser.add_argument("--epochs", "-e", type=int, help="Number of epochs")
    parser.add_argument("--batch", "-b", type=int, help="Batch size")
    parser.add_argument("--imgsz", type=int, help="Image size")
    parser.add_argument("--workers", "-w", type=int, help="Data loader workers")

    # Model
    parser.add_argument(
        "--model-size",
        "-m",
        type=str,
        choices=["n", "s", "m", "l", "x"],
        help="Model size (nano/small/medium/large/xlarge)",
    )

    # Device
    parser.add_argument("--device", type=str, help="Device (auto/cpu/cuda/0/1/...)")

    # Output
    parser.add_argument("--output", "-o", type=str, help="Output directory")

    # Resume training
    parser.add_argument("--resume", "-r", type=str, help="Path to checkpoint to resume from")

    # Actions
    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate dataset, do not train"
    )
    parser.add_argument(
        "--eval-only", type=str, metavar="MODEL_PATH", help="Only evaluate model, do not train"
    )
    parser.add_argument(
        "--export-only", type=str, metavar="MODEL_PATH", help="Only export model, do not train"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Load config
    config = load_config(args.config)
    config = merge_config_with_args(config, args)

    # Setup logging
    setup_logging(config["logging"]["output_dir"], config["logging"].get("level", "INFO"))

    logger.info(BANNER)
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {config['logging']['output_dir']}")

    try:
        # Load from Roboflow if configured
        config["dataset"]["path"] = load_dataset_from_roboflow(config)

        # Split dataset if needed
        split_dataset_if_needed(config)

        # Validate dataset
        if not validate_dataset(config):
            logger.error("Dataset validation failed. Exiting.")
            return 1

        if args.validate_only:
            logger.info("Validation complete. Exiting (--validate-only)")
            return 0

        output_dir = config["logging"]["output_dir"]

        # Evaluate only mode
        if args.eval_only:
            results = evaluate_model(config, args.eval_only, output_dir)
            return 0

        # Export only mode
        if args.export_only:
            export_model(config, args.export_only, output_dir)
            return 0

        # Train model
        history = train_model(config, args.resume)

        # Find best model
        best_model = Path(output_dir) / "train" / "weights" / "best.pt"
        if not best_model.exists():
            best_model = Path(output_dir) / "train" / "weights" / "last.pt"

        # Evaluate
        results = evaluate_model(config, str(best_model), output_dir)

        # Export
        export_model(config, str(best_model), output_dir)

        # Create visualizations
        create_visualizations(config, history, results, output_dir)

        # Run ablation study if configured
        run_ablation_study(config)

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Best mAP@50-95: {history.best_mAP50_95:.4f}")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.exception(f"Training failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
