"""
Model initialization and configuration for YOLO 11 Twitter Screenshot Detection.

This module handles:
- Model loading and initialization
- Hyperparameter configuration
- Multi-GPU setup
- Model architecture selection
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

import torch
from ultralytics import YOLO


logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration container for YOLO model."""
    size: str = "m"
    weights: Optional[str] = None
    num_classes: Optional[int] = None
    device: str = "auto"
    multi_gpu: bool = False
    gpu_ids: List[int] = None
    
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
    
    MODEL_SIZES = {
        'n': 'yolo11n.pt',
        'nano': 'yolo11n.pt',
        's': 'yolo11s.pt',
        'small': 'yolo11s.pt',
        'm': 'yolo11m.pt',
        'medium': 'yolo11m.pt',
        'l': 'yolo11l.pt',
        'large': 'yolo11l.pt',
        'x': 'yolo11x.pt',
        'xlarge': 'yolo11x.pt',
    }
    
    def __init__(self, config: ModelConfig):
        """
        Initialize model factory.
        
        Args:
            config: Model configuration object
        """
        self.config = config
        self.device = self._setup_device()
    
    def _setup_device(self) -> str:
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
                    
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                logger.info("Apple Silicon MPS device detected")
            else:
                device = "cpu"
                logger.info("No GPU detected, using CPU")
        else:
            device = self.config.device
            logger.info(f"Using specified device: {device}")
        
        return device
    
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
            logger.info(f"Loading pretrained YOLO11 {self.config.size} model")
        
        # Initialize model
        model = YOLO(weights_path)
        
        logger.info(f"Model initialized successfully")
        logger.info(f"  Architecture: YOLO11-{self.config.size.upper()}")
        logger.info(f"  Device: {self.device}")
        
        return model
    
    def get_training_args(self, data_yaml: str, output_dir: str) -> Dict[str, Any]:
        """
        Get training arguments dictionary for YOLO.train().
        
        Args:
            data_yaml: Path to dataset YAML file
            output_dir: Output directory for training results
            
        Returns:
            Dictionary of training arguments
        """
        args = {
            'data': data_yaml,
            'epochs': self.config.epochs,
            'batch': self.config.batch_size,
            'imgsz': self.config.image_size,
            'device': self.device,
            'workers': self.config.workers,
            'patience': self.config.patience,
            'project': output_dir,
            'name': 'train',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': self.config.optimizer,
            'lr0': self.config.learning_rate,
            'momentum': self.config.momentum,
            'weight_decay': self.config.weight_decay,
            'warmup_epochs': self.config.warmup_epochs,
            'amp': self.config.amp,
            'save': True,
            'save_period': 10,
            'plots': True,
            'verbose': True,
        }
        
        return args


class AugmentationConfig:
    """Configuration for data augmentation strategies."""
    
    # Default augmentation for Twitter screenshots
    TWITTER_AUGMENTATION = {
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'degrees': 0.0,  # No rotation for text readability
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,  # No shear for text
        'perspective': 0.0,
        'flipud': 0.0,  # No vertical flip
        'fliplr': 0.5,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'erasing': 0.4,
    }
    
    # Conservative augmentation for high-precision tasks
    CONSERVATIVE_AUGMENTATION = {
        'mosaic': 0.5,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'degrees': 0.0,
        'translate': 0.05,
        'scale': 0.2,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.0,
        'hsv_h': 0.01,
        'hsv_s': 0.3,
        'hsv_v': 0.2,
        'erasing': 0.2,
    }
    
    # Aggressive augmentation for small datasets
    AGGRESSIVE_AUGMENTATION = {
        'mosaic': 1.0,
        'mixup': 0.2,
        'copy_paste': 0.3,
        'degrees': 5.0,
        'translate': 0.2,
        'scale': 0.9,
        'shear': 2.0,
        'perspective': 0.001,
        'flipud': 0.0,
        'fliplr': 0.5,
        'hsv_h': 0.02,
        'hsv_s': 0.9,
        'hsv_v': 0.5,
        'erasing': 0.5,
    }
    
    @classmethod
    def get_augmentation(cls, strategy: str = "twitter") -> Dict[str, float]:
        """
        Get augmentation configuration by strategy name.
        
        Args:
            strategy: Augmentation strategy name
            
        Returns:
            Dictionary of augmentation parameters
        """
        strategies = {
            'twitter': cls.TWITTER_AUGMENTATION,
            'conservative': cls.CONSERVATIVE_AUGMENTATION,
            'aggressive': cls.AGGRESSIVE_AUGMENTATION,
        }
        
        if strategy not in strategies:
            raise ValueError(
                f"Unknown augmentation strategy: {strategy}. "
                f"Valid options: {list(strategies.keys())}"
            )
        
        return strategies[strategy]


class ModelExporter:
    """Handles model export to various formats."""
    
    SUPPORTED_FORMATS = ['pytorch', 'torchscript', 'onnx', 'openvino', 
                         'tensorrt', 'coreml', 'tflite', 'paddle', 'ncnn']
    
    def __init__(self, model: YOLO, output_dir: str):
        """
        Initialize model exporter.
        
        Args:
            model: Trained YOLO model
            output_dir: Output directory for exported models
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export(
        self,
        formats: List[str],
        image_size: int = 640,
        half: bool = False,
        dynamic: bool = False,
        simplify: bool = True,
        opset: int = 12
    ) -> Dict[str, str]:
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
        exported = {}
        
        for format_name in formats:
            format_lower = format_name.lower()
            
            if format_lower == 'pytorch':
                # PyTorch format is already saved during training
                pt_path = self.output_dir / 'best.pt'
                if pt_path.exists():
                    exported['pytorch'] = str(pt_path)
                continue
            
            if format_lower not in self.SUPPORTED_FORMATS:
                logger.warning(f"Unsupported export format: {format_name}")
                continue
            
            try:
                logger.info(f"Exporting model to {format_name} format...")
                
                export_args = {
                    'format': format_lower,
                    'imgsz': image_size,
                    'half': half,
                }
                
                if format_lower == 'onnx':
                    export_args.update({
                        'dynamic': dynamic,
                        'simplify': simplify,
                        'opset': opset,
                    })
                
                export_path = self.model.export(**export_args)
                exported[format_lower] = str(export_path)
                logger.info(f"  Exported to: {export_path}")
                
            except Exception as e:
                logger.error(f"Failed to export to {format_name}: {str(e)}")
        
        return exported


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
        export_path = model.export(
            format='onnx',
            int8=True,
            data=calibration_data
        )
        
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
        export_path = model.export(
            format='onnx',
            half=True
        )
        
        logger.info(f"FP16 quantized model saved to: {export_path}")
        return str(export_path)


def get_model_info(model: YOLO) -> Dict[str, Any]:
    """
    Get comprehensive model information.
    
    Args:
        model: YOLO model instance
        
    Returns:
        Dictionary of model information
    """
    info = {
        'task': model.task,
        'model_type': model.type,
    }
    
    # Get model parameters
    if hasattr(model, 'model'):
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(
            p.numel() for p in model.model.parameters() if p.requires_grad
        )
        
        info['total_parameters'] = total_params
        info['trainable_parameters'] = trainable_params
        info['parameters_mb'] = total_params * 4 / 1e6  # Assuming float32
    
    return info


def compare_models(model_paths: List[str], test_image: str) -> Dict[str, Dict]:
    """
    Compare multiple models on the same test image.
    
    Args:
        model_paths: List of paths to model files
        test_image: Path to test image
        
    Returns:
        Comparison results dictionary
    """
    import time
    
    results = {}
    
    for model_path in model_paths:
        model_name = Path(model_path).stem
        model = YOLO(model_path)
        
        # Warmup
        _ = model.predict(test_image, verbose=False)
        
        # Timed inference
        start = time.time()
        for _ in range(10):
            pred = model.predict(test_image, verbose=False)
        avg_time = (time.time() - start) / 10
        
        results[model_name] = {
            'inference_time_ms': avg_time * 1000,
            'detections': len(pred[0].boxes) if pred else 0,
            'model_info': get_model_info(model)
        }
    
    return results
