"""
Model initialization and configuration for YOLO 26 Twitter Screenshot Detection.

This module handles:
- Model loading and initialization
- Hyperparameter configuration
- Multi-GPU setup
- Model architecture selection
"""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def resolve_device(device: str | int | list[int]) -> str | int | list[int]:
    """Resolve an 'auto' device spec to a concrete torch/ultralytics device.

    ultralytics' val()/predict()/export() reject device='auto' (only train()
    accepts it), so callers outside training must resolve it first. Explicit
    devices pass through unchanged.
    """
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class ModelConfig:
    """Configuration container for YOLO model."""

    size: str = "m"
    weights: str | None = None
    num_classes: int | None = None
    device: str = "auto"
    multi_gpu: bool = False
    gpu_ids: list[int] | None = None

    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    learning_rate: float = 0.01
    optimizer: str = "SGD"
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: int = 3
    patience: int = 20
    amp: bool = True
    workers: int = 8

    def __post_init__(self):
        if self.gpu_ids is None:
            self.gpu_ids = [0, 1]


class ModelFactory:
    """Factory class for creating and configuring YOLO models."""

    # YOLO26 (Ultralytics >=8.4.52) detection checkpoints. Single source of
    # truth for default weight selection by size alias.
    MODEL_SIZES = {
        "n": "yolo26n.pt",
        "nano": "yolo26n.pt",
        "s": "yolo26s.pt",
        "small": "yolo26s.pt",
        "m": "yolo26m.pt",
        "medium": "yolo26m.pt",
        "l": "yolo26l.pt",
        "large": "yolo26l.pt",
        "x": "yolo26x.pt",
        "xlarge": "yolo26x.pt",
    }

    def __init__(self, config: ModelConfig):
        """
        Initialize model factory.

        Args:
            config: Model configuration object
        """
        self.config = config
        self.device = self._setup_device()

    def _setup_device(self) -> str | list[int] | int:
        """
        Detect and configure the compute device.

        Returns:
            Device string for PyTorch
        """
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                gpu_count = torch.cuda.device_count()
                logger.info(f"CUDA available: {gpu_count} GPU(s) detected")

                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                    logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

                if self.config.multi_gpu and gpu_count > 1:
                    device = self.config.gpu_ids
                    logger.info(f"Multi-GPU training enabled on devices: {device}")
                else:
                    device = 0

            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
                logger.info("Apple Silicon MPS device detected")
            else:
                device = "cpu"
                logger.info("No GPU detected, using CPU")
        else:
            device = self.config.device
            logger.info(f"Using specified device: {device}")

        return device  # pyright: ignore[reportReturnType]

    def create_model(self) -> YOLO:
        """
        Create and initialize YOLO model.

        Returns:
            Initialized YOLO model
        """
        # Determine weights path
        if self.config.weights:
            weights_path = self.config.weights
            logger.info(f"Loading custom weights from: {weights_path}")
        else:
            size_key = self.config.size.lower()
            if size_key not in self.MODEL_SIZES:
                raise ValueError(
                    f"Invalid model size: {self.config.size}. "
                    f"Valid options: {list(self.MODEL_SIZES.keys())}"
                )
            weights_path = self.MODEL_SIZES[size_key]
            logger.info(f"Loading pretrained YOLO26 {self.config.size} model")

        # Initialize model
        model = YOLO(weights_path)

        logger.info("Model initialized successfully")
        logger.info(f"  Architecture: YOLO26-{self.config.size.upper()}")
        logger.info(f"  Device: {self.device}")

        return model

    def get_training_args(self, data_yaml: str, output_dir: str) -> dict[str, Any]:
        """
        Get training arguments dictionary for YOLO.train().

        Args:
            data_yaml: Path to dataset YAML file
            output_dir: Output directory for training results

        Returns:
            Dictionary of training arguments
        """
        args = {
            "data": data_yaml,
            "epochs": self.config.epochs,
            "batch": self.config.batch_size,
            "imgsz": self.config.image_size,
            "device": self.device,
            "workers": self.config.workers,
            "patience": self.config.patience,
            "project": output_dir,
            "name": "train",
            "exist_ok": True,
            "pretrained": True,
            "optimizer": self.config.optimizer,
            "lr0": self.config.learning_rate,
            "momentum": self.config.momentum,
            "weight_decay": self.config.weight_decay,
            "warmup_epochs": self.config.warmup_epochs,
            "amp": self.config.amp,
            "save": True,
            "save_period": 10,
            "plots": True,
            "verbose": True,
        }

        return args


class AugmentationConfig:
    """Configuration for data augmentation strategies."""

    # Default augmentation for Twitter screenshots
    TWITTER_AUGMENTATION = {
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "degrees": 0.0,  # No rotation for text readability
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,  # No shear for text
        "perspective": 0.0,
        "flipud": 0.0,  # No vertical flip
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "erasing": 0.4,
    }

    # Conservative augmentation for high-precision tasks
    CONSERVATIVE_AUGMENTATION = {
        "mosaic": 0.5,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "degrees": 0.0,
        "translate": 0.05,
        "scale": 0.2,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "hsv_h": 0.01,
        "hsv_s": 0.3,
        "hsv_v": 0.2,
        "erasing": 0.2,
    }

    # Aggressive augmentation for small datasets
    AGGRESSIVE_AUGMENTATION = {
        "mosaic": 1.0,
        "mixup": 0.2,
        "copy_paste": 0.3,
        "degrees": 5.0,
        "translate": 0.2,
        "scale": 0.9,
        "shear": 2.0,
        "perspective": 0.001,
        "flipud": 0.0,
        "fliplr": 0.5,
        "hsv_h": 0.02,
        "hsv_s": 0.9,
        "hsv_v": 0.5,
        "erasing": 0.5,
    }

    @classmethod
    def get_augmentation(cls, strategy: str = "twitter") -> dict[str, float]:
        """
        Get augmentation configuration by strategy name.

        Args:
            strategy: Augmentation strategy name

        Returns:
            Dictionary of augmentation parameters
        """
        strategies = {
            "twitter": cls.TWITTER_AUGMENTATION,
            "conservative": cls.CONSERVATIVE_AUGMENTATION,
            "aggressive": cls.AGGRESSIVE_AUGMENTATION,
        }

        if strategy not in strategies:
            raise ValueError(
                f"Unknown augmentation strategy: {strategy}. "
                f"Valid options: {list(strategies.keys())}"
            )

        return strategies[strategy]


class ModelExporter:
    """Handles model export to various formats."""

    SUPPORTED_FORMATS = [
        "pytorch",
        "torchscript",
        "onnx",
        "openvino",
        "tensorrt",
        "coreml",
        "tflite",
        "paddle",
        "ncnn",
    ]

    def __init__(self, model: YOLO, output_dir: str, source_weights: str | Path | None = None):
        """
        Initialize model exporter.

        Args:
            model: Trained YOLO model
            output_dir: Output directory for exported models
            source_weights: Path to the actual ``.pt`` checkpoint backing ``model``.
                Ultralytics writes the real best weights to
                ``{run}/train/weights/best.pt``, not ``{output_dir}/best.pt``; the
                ``pytorch`` export reports/copies this file instead of guessing.
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.source_weights = Path(source_weights) if source_weights else None
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        formats: list[str],
        image_size: int = 640,
        half: bool = False,
        dynamic: bool = False,
        simplify: bool = True,
        opset: int = 12,
    ) -> dict[str, str]:
        """
        Export model to multiple formats.

        Args:
            formats: List of export formats
            image_size: Input image size for export
            half: Use FP16 precision
            dynamic: Dynamic input shapes (ONNX)
            simplify: Simplify ONNX model
            opset: ONNX opset version

        Returns:
            Dictionary mapping format to export path
        """
        exported: dict[str, str] = {}

        # PyTorch is the guaranteed primary artifact: export it first and never let a
        # failure of an optional format (ONNX, etc.) drop or precede it.
        ordered = sorted(formats, key=lambda f: f.lower() != "pytorch")

        for format_name in ordered:
            format_lower = format_name.lower()

            if format_lower == "pytorch":
                pt_path = self._resolve_pytorch_path()
                if pt_path is not None:
                    exported["pytorch"] = str(pt_path)
                    logger.info(f"  PyTorch weights: {pt_path}")
                else:
                    logger.warning("PyTorch export requested but no source weights were found")
                continue

            if format_lower not in self.SUPPORTED_FORMATS:
                logger.warning(f"Unsupported export format: {format_name}")
                continue

            try:
                logger.info(f"Exporting model to {format_name} format...")

                export_args: dict[str, Any] = {
                    "format": format_lower,
                    "imgsz": image_size,
                    "half": half,
                }

                if format_lower == "onnx":
                    export_args.update(
                        {
                            "dynamic": dynamic,
                            "simplify": simplify,
                            "opset": opset,
                        }
                    )

                export_path = self.model.export(**export_args)
                exported[format_lower] = str(export_path)
                logger.info(f"  Exported to: {export_path}")

            except Exception as e:
                # Optional formats fail soft: warn and continue so the run still
                # completes with the PyTorch artifact intact.
                logger.warning(f"Failed to export to {format_name}, skipping: {e}")

        return exported

    def _resolve_pytorch_path(self) -> Path | None:
        """Resolve the real best ``.pt`` and copy it to a stable, reported location.

        Returns the path under ``output_dir`` the caller can rely on, or None if no
        source checkpoint is available. Falls back to ``{output_dir}/best.pt`` /
        ``train/weights/best.pt`` for legacy callers that don't pass ``source_weights``.
        """
        source = self.source_weights
        if source is None or not source.exists():
            for candidate in (
                self.output_dir / "best.pt",
                self.output_dir / "train" / "weights" / "best.pt",
                self.output_dir / "train" / "weights" / "last.pt",
            ):
                if candidate.exists():
                    source = candidate
                    break
            else:
                return None

        target = self.output_dir / "best.pt"
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
        return target


class ModelQuantizer:
    """Handles model quantization for deployment optimization."""

    def __init__(self, model_path: str):
        """
        Initialize quantizer.

        Args:
            model_path: Path to model file
        """
        self.model_path = Path(model_path)

    def quantize_int8(self, calibration_data: str, output_path: str) -> str:
        """
        Quantize model to INT8 precision.

        Args:
            calibration_data: Path to calibration dataset
            output_path: Output path for quantized model

        Returns:
            Path to quantized model
        """
        logger.info("Quantizing model to INT8...")

        # Load model
        model = YOLO(str(self.model_path))

        # Export with INT8 quantization
        export_path = model.export(format="onnx", int8=True, data=calibration_data)

        logger.info(f"INT8 quantized model saved to: {export_path}")
        return str(export_path)

    def quantize_fp16(self, output_path: str) -> str:
        """
        Quantize model to FP16 precision.

        Args:
            output_path: Output path for quantized model

        Returns:
            Path to quantized model
        """
        logger.info("Quantizing model to FP16...")

        # Load model
        model = YOLO(str(self.model_path))

        # Export with FP16
        export_path = model.export(format="onnx", half=True)

        logger.info(f"FP16 quantized model saved to: {export_path}")
        return str(export_path)


def get_model_info(model: YOLO) -> dict[str, Any]:
    """
    Get comprehensive model information.

    Args:
        model: YOLO model instance

    Returns:
        Dictionary of model information
    """
    info = {
        "task": model.task,
        "model_type": model.type,
    }

    # Get model parameters
    if hasattr(model, "model") and model.model is not None and hasattr(model.model, "parameters"):
        total_params = sum(p.numel() for p in model.model.parameters())  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]

        info["total_parameters"] = total_params
        info["trainable_parameters"] = trainable_params
        info["parameters_mb"] = total_params * 4 / 1e6  # Assuming float32

    return info


def compare_models(model_paths: list[str], test_image: str) -> dict[str, dict[str, Any]]:
    """
    Compare multiple models on the same test image.

    Args:
        model_paths: List of paths to model files
        test_image: Path to test image

    Returns:
        Comparison results dictionary
    """
    import time

    results: dict[str, dict[str, Any]] = {}

    for model_path in model_paths:
        model_name = Path(model_path).stem
        model = YOLO(model_path)

        # Warmup
        _ = model.predict(test_image, verbose=False)

        # Timed inference
        start = time.time()
        pred = None
        for _ in range(10):
            pred = model.predict(test_image, verbose=False)
        avg_time = (time.time() - start) / 10

        detection_count = 0
        if pred and pred[0].boxes is not None:
            detection_count = len(pred[0].boxes)

        results[model_name] = {
            "inference_time_ms": avg_time * 1000,
            "detections": detection_count,
            "model_info": get_model_info(model),
        }

    return results
