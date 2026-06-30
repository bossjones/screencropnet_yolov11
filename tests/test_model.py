"""Tests for model module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pytest_mock import MockerFixture

from screencropnet_yolo.model import (
    AugmentationConfig,
    ModelConfig,
    ModelExporter,
    ModelFactory,
    ModelQuantizer,
    compare_models,
    get_model_info,
    resolve_device,
)

# --- Helper Functions ---


def mock_torch_cuda(
    mocker: MockerFixture,
    cuda_available: bool = False,
    device_count: int = 0,
    mps_available: bool = False,
    has_mps_attr: bool = True,
) -> None:
    """Configure torch mocks for device detection tests."""
    mocker.patch(
        "screencropnet_yolo.model.torch.cuda.is_available",
        return_value=cuda_available,
    )
    mocker.patch(
        "screencropnet_yolo.model.torch.cuda.device_count",
        return_value=device_count,
    )

    if cuda_available and device_count > 0:
        mocker.patch(
            "screencropnet_yolo.model.torch.cuda.get_device_name",
            side_effect=lambda i: f"NVIDIA GPU {i}",  # pyright: ignore[reportUnknownLambdaType]
        )
        mock_props = mocker.MagicMock()
        mock_props.total_memory = 16 * 1e9  # 16GB
        mocker.patch(
            "screencropnet_yolo.model.torch.cuda.get_device_properties",
            return_value=mock_props,
        )

    if has_mps_attr:
        mock_backends = mocker.MagicMock()
        mock_backends.mps.is_available.return_value = mps_available
        mocker.patch("screencropnet_yolo.model.torch.backends", mock_backends)
    else:
        mock_backends = mocker.MagicMock(spec=["cuda"])
        mocker.patch("screencropnet_yolo.model.torch.backends", mock_backends)


def create_mock_yolo(mocker: MockerFixture) -> Any:
    """Create a mock YOLO model with common attributes."""
    mock_model = mocker.MagicMock()
    mock_model.task = "detect"
    mock_model.type = "v26"

    # Mock internal model with parameters
    mock_param = mocker.MagicMock()
    mock_param.numel.return_value = 1000000
    mock_param.requires_grad = True
    mock_model.model.parameters.return_value = [mock_param] * 10

    # Mock export method
    mock_model.export.return_value = "/tmp/exported_model.onnx"

    # Mock predict method
    mock_result = mocker.MagicMock()
    mock_result.boxes = [mocker.MagicMock()] * 3
    mock_model.predict.return_value = [mock_result]

    return mock_model


# --- TestModelConfig ---


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_values(self) -> None:
        """Default configuration uses expected values."""
        config = ModelConfig()

        assert config.size == "m"
        assert config.weights is None
        assert config.num_classes is None
        assert config.device == "auto"
        assert config.multi_gpu is False
        assert config.epochs == 100
        assert config.batch_size == 16
        assert config.image_size == 640
        assert config.learning_rate == 0.01
        assert config.optimizer == "SGD"
        assert config.momentum == 0.937
        assert config.weight_decay == 0.0005
        assert config.warmup_epochs == 3
        assert config.patience == 20
        assert config.amp is True
        assert config.workers == 8

    def test_post_init_sets_default_gpu_ids(self) -> None:
        """__post_init__ sets gpu_ids to [0, 1] when None."""
        config = ModelConfig()

        assert config.gpu_ids == [0, 1]

    def test_post_init_preserves_custom_gpu_ids(self) -> None:
        """__post_init__ does not overwrite explicitly set gpu_ids."""
        config = ModelConfig(gpu_ids=[0, 2, 3])

        assert config.gpu_ids == [0, 2, 3]

    def test_custom_values(self) -> None:
        """Custom values are properly assigned."""
        config = ModelConfig(
            size="x",
            weights="/path/to/weights.pt",
            num_classes=5,
            device="cuda:0",
            multi_gpu=True,
            epochs=50,
            batch_size=32,
        )

        assert config.size == "x"
        assert config.weights == "/path/to/weights.pt"
        assert config.num_classes == 5
        assert config.device == "cuda:0"
        assert config.multi_gpu is True
        assert config.epochs == 50
        assert config.batch_size == 32


# --- TestModelFactory ---


class TestModelFactory:
    """Tests for ModelFactory class."""

    def test_model_sizes_mapping(self) -> None:
        """MODEL_SIZES contains expected size mappings."""
        expected = {
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

        assert ModelFactory.MODEL_SIZES == expected

    def test_setup_device_auto_cuda_single_gpu(self, mocker: MockerFixture) -> None:
        """Auto device with single CUDA GPU returns device 0."""
        mock_torch_cuda(mocker, cuda_available=True, device_count=1)

        config = ModelConfig(device="auto", multi_gpu=False)
        factory = ModelFactory(config)

        assert factory.device == 0

    def test_setup_device_auto_cuda_multi_gpu_enabled(self, mocker: MockerFixture) -> None:
        """Auto device with multi_gpu=True returns gpu_ids list."""
        mock_torch_cuda(mocker, cuda_available=True, device_count=4)

        config = ModelConfig(device="auto", multi_gpu=True, gpu_ids=[0, 1, 2])
        factory = ModelFactory(config)

        assert factory.device == [0, 1, 2]

    def test_setup_device_auto_cuda_multi_gpu_single_available(self, mocker: MockerFixture) -> None:
        """Multi-GPU not activated when only one GPU available."""
        mock_torch_cuda(mocker, cuda_available=True, device_count=1)

        config = ModelConfig(device="auto", multi_gpu=True)
        factory = ModelFactory(config)

        assert factory.device == 0

    def test_setup_device_auto_mps(self, mocker: MockerFixture) -> None:
        """Auto device on Apple Silicon returns 'mps'."""
        mock_torch_cuda(mocker, cuda_available=False, mps_available=True)

        config = ModelConfig(device="auto")
        factory = ModelFactory(config)

        assert factory.device == "mps"

    def test_setup_device_auto_cpu_fallback(self, mocker: MockerFixture) -> None:
        """Auto device falls back to CPU when no GPU available."""
        mock_torch_cuda(mocker, cuda_available=False, mps_available=False, has_mps_attr=False)

        config = ModelConfig(device="auto")
        factory = ModelFactory(config)

        assert factory.device == "cpu"

    def test_setup_device_explicit(self, mocker: MockerFixture) -> None:
        """Explicit device specification bypasses auto-detection."""
        mock_torch_cuda(mocker, cuda_available=True, device_count=2)

        config = ModelConfig(device="cuda:1")
        factory = ModelFactory(config)

        assert factory.device == "cuda:1"

    def test_create_model_with_custom_weights(self, mocker: MockerFixture) -> None:
        """Custom weights path is used when specified."""
        mock_torch_cuda(mocker, cuda_available=False, has_mps_attr=False)
        mock_yolo = mocker.patch("screencropnet_yolo.model.YOLO")

        config = ModelConfig(weights="/path/to/custom.pt")
        factory = ModelFactory(config)
        factory.create_model()

        mock_yolo.assert_called_once_with("/path/to/custom.pt")

    def test_create_model_with_size(self, mocker: MockerFixture) -> None:
        """Pretrained weights selected based on size."""
        mock_torch_cuda(mocker, cuda_available=False, has_mps_attr=False)
        mock_yolo = mocker.patch("screencropnet_yolo.model.YOLO")

        config = ModelConfig(size="large")
        factory = ModelFactory(config)
        factory.create_model()

        mock_yolo.assert_called_once_with("yolo26l.pt")

    def test_create_model_loads_yolo26_weights_by_default(self, mocker: MockerFixture) -> None:
        """The default medium size resolves to the yolo26m checkpoint."""
        mock_torch_cuda(mocker, cuda_available=False, has_mps_attr=False)
        mock_yolo = mocker.patch("screencropnet_yolo.model.YOLO")

        factory = ModelFactory(ModelConfig())
        factory.create_model()

        mock_yolo.assert_called_once_with("yolo26m.pt")

    def test_create_model_log_says_yolo26(
        self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Model-load logging identifies the architecture as YOLO26."""
        import logging

        mock_torch_cuda(mocker, cuda_available=False, has_mps_attr=False)
        mocker.patch("screencropnet_yolo.model.YOLO")

        factory = ModelFactory(ModelConfig(size="s"))
        with caplog.at_level(logging.INFO):
            factory.create_model()

        assert "YOLO26" in caplog.text

    def test_create_model_invalid_size(self, mocker: MockerFixture) -> None:
        """ValueError raised for invalid model size."""
        mock_torch_cuda(mocker, cuda_available=False, has_mps_attr=False)

        config = ModelConfig(size="invalid_size")
        factory = ModelFactory(config)

        with pytest.raises(ValueError, match="Invalid model size"):
            factory.create_model()

    def test_create_model_returns_yolo_instance(self, mocker: MockerFixture) -> None:
        """create_model returns YOLO model instance."""
        mock_torch_cuda(mocker, cuda_available=False, has_mps_attr=False)
        mock_model = create_mock_yolo(mocker)
        mocker.patch("screencropnet_yolo.model.YOLO", return_value=mock_model)

        config = ModelConfig()
        factory = ModelFactory(config)
        result = factory.create_model()

        assert result is mock_model

    def test_get_training_args_structure(self, mocker: MockerFixture) -> None:
        """Training args contain all required keys."""
        mock_torch_cuda(mocker, cuda_available=False, has_mps_attr=False)

        config = ModelConfig()
        factory = ModelFactory(config)
        args = factory.get_training_args("/data/data.yaml", "/output")

        expected_keys = {
            "data",
            "epochs",
            "batch",
            "imgsz",
            "device",
            "workers",
            "patience",
            "project",
            "name",
            "exist_ok",
            "pretrained",
            "optimizer",
            "lr0",
            "momentum",
            "weight_decay",
            "warmup_epochs",
            "amp",
            "save",
            "save_period",
            "plots",
            "verbose",
        }

        assert set(args.keys()) == expected_keys

    def test_get_training_args_values(self, mocker: MockerFixture) -> None:
        """Training args values match config settings."""
        mock_torch_cuda(mocker, cuda_available=False, has_mps_attr=False)

        config = ModelConfig(
            epochs=50,
            batch_size=32,
            image_size=1280,
            learning_rate=0.001,
        )
        factory = ModelFactory(config)
        args = factory.get_training_args("/data.yaml", "/output")

        assert args["data"] == "/data.yaml"
        assert args["epochs"] == 50
        assert args["batch"] == 32
        assert args["imgsz"] == 1280
        assert args["lr0"] == 0.001
        assert args["project"] == "/output"


# --- TestResolveDevice ---


class TestResolveDevice:
    """Tests for the module-level resolve_device helper."""

    def test_auto_resolves_to_cuda(self, mocker: MockerFixture) -> None:
        """'auto' resolves to device 0 when CUDA is available."""
        mock_torch_cuda(mocker, cuda_available=True, device_count=1)

        assert resolve_device("auto") == 0

    def test_auto_resolves_to_mps(self, mocker: MockerFixture) -> None:
        """'auto' resolves to 'mps' when only Apple Silicon is available."""
        mock_torch_cuda(mocker, cuda_available=False, mps_available=True)

        assert resolve_device("auto") == "mps"

    def test_auto_resolves_to_cpu(self, mocker: MockerFixture) -> None:
        """'auto' falls back to 'cpu' when neither CUDA nor MPS is available."""
        mock_torch_cuda(mocker, cuda_available=False, mps_available=False, has_mps_attr=False)

        assert resolve_device("auto") == "cpu"

    def test_explicit_cpu_passes_through(self, mocker: MockerFixture) -> None:
        """Explicit 'cpu' is returned unchanged even when CUDA is available."""
        mock_torch_cuda(mocker, cuda_available=True, device_count=1)

        assert resolve_device("cpu") == "cpu"

    def test_explicit_mps_passes_through(self, mocker: MockerFixture) -> None:
        """Explicit 'mps' is returned unchanged."""
        mock_torch_cuda(mocker, cuda_available=True, device_count=1)

        assert resolve_device("mps") == "mps"

    def test_explicit_int_passes_through(self, mocker: MockerFixture) -> None:
        """Explicit integer device index is returned unchanged."""
        mock_torch_cuda(mocker, cuda_available=False, has_mps_attr=False)

        assert resolve_device(0) == 0


# --- TestAugmentationConfig ---


class TestAugmentationConfig:
    """Tests for AugmentationConfig class."""

    def test_twitter_augmentation_values(self) -> None:
        """Twitter augmentation has text-friendly settings."""
        aug = AugmentationConfig.TWITTER_AUGMENTATION

        assert aug["degrees"] == 0.0
        assert aug["shear"] == 0.0
        assert aug["flipud"] == 0.0
        assert aug["perspective"] == 0.0
        assert aug["mosaic"] == 1.0
        assert aug["fliplr"] == 0.5

    def test_conservative_augmentation_values(self) -> None:
        """Conservative augmentation has minimal transformations."""
        aug = AugmentationConfig.CONSERVATIVE_AUGMENTATION

        assert aug["mosaic"] == 0.5
        assert aug["translate"] == 0.05
        assert aug["scale"] == 0.2
        assert aug["fliplr"] == 0.0

    def test_aggressive_augmentation_values(self) -> None:
        """Aggressive augmentation has maximum variety."""
        aug = AugmentationConfig.AGGRESSIVE_AUGMENTATION

        assert aug["mosaic"] == 1.0
        assert aug["mixup"] == 0.2
        assert aug["copy_paste"] == 0.3
        assert aug["degrees"] == 5.0
        assert aug["scale"] == 0.9

    def test_get_augmentation_twitter(self) -> None:
        """get_augmentation returns twitter config."""
        result = AugmentationConfig.get_augmentation("twitter")

        assert result == AugmentationConfig.TWITTER_AUGMENTATION

    def test_get_augmentation_conservative(self) -> None:
        """get_augmentation returns conservative config."""
        result = AugmentationConfig.get_augmentation("conservative")

        assert result == AugmentationConfig.CONSERVATIVE_AUGMENTATION

    def test_get_augmentation_aggressive(self) -> None:
        """get_augmentation returns aggressive config."""
        result = AugmentationConfig.get_augmentation("aggressive")

        assert result == AugmentationConfig.AGGRESSIVE_AUGMENTATION

    def test_get_augmentation_invalid_strategy(self) -> None:
        """ValueError raised for unknown strategy."""
        with pytest.raises(ValueError, match="Unknown augmentation strategy"):
            AugmentationConfig.get_augmentation("nonexistent")

    def test_get_augmentation_error_lists_valid_options(self) -> None:
        """Error message includes valid strategy names."""
        with pytest.raises(ValueError) as exc_info:
            AugmentationConfig.get_augmentation("invalid")

        assert "twitter" in str(exc_info.value)
        assert "conservative" in str(exc_info.value)
        assert "aggressive" in str(exc_info.value)


# --- TestModelExporter ---


class TestModelExporter:
    """Tests for ModelExporter class."""

    def test_supported_formats(self) -> None:
        """SUPPORTED_FORMATS contains expected formats."""
        expected = [
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

        assert ModelExporter.SUPPORTED_FORMATS == expected

    def test_init_creates_output_dir(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Output directory is created on initialization."""
        mock_model = create_mock_yolo(mocker)
        output_dir = tmp_path / "exports"

        exporter = ModelExporter(mock_model, str(output_dir))

        assert output_dir.exists()
        assert exporter.output_dir == output_dir

    def test_export_pytorch_existing_file(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """PyTorch export returns path when best.pt exists."""
        mock_model = create_mock_yolo(mocker)
        output_dir = tmp_path / "exports"
        output_dir.mkdir()
        (output_dir / "best.pt").touch()

        exporter = ModelExporter(mock_model, str(output_dir))
        result = exporter.export(["pytorch"])

        assert "pytorch" in result
        assert result["pytorch"] == str(output_dir / "best.pt")

    def test_export_pytorch_no_file(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """PyTorch export skipped when best.pt doesn't exist."""
        mock_model = create_mock_yolo(mocker)

        exporter = ModelExporter(mock_model, str(tmp_path))
        result = exporter.export(["pytorch"])

        assert "pytorch" not in result

    def test_export_onnx_with_options(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """ONNX export includes format-specific options."""
        mock_model = create_mock_yolo(mocker)
        mock_model.export.return_value = str(tmp_path / "model.onnx")

        exporter = ModelExporter(mock_model, str(tmp_path))
        exporter.export(["onnx"], image_size=1280, dynamic=True, simplify=False, opset=17)

        mock_model.export.assert_called_once()
        call_kwargs = mock_model.export.call_args[1]
        assert call_kwargs["format"] == "onnx"
        assert call_kwargs["imgsz"] == 1280
        assert call_kwargs["dynamic"] is True
        assert call_kwargs["simplify"] is False
        assert call_kwargs["opset"] == 17

    def test_export_multiple_formats(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Multiple formats can be exported in one call."""
        mock_model = create_mock_yolo(mocker)
        mock_model.export.return_value = "/exported/model"

        exporter = ModelExporter(mock_model, str(tmp_path))
        exporter.export(["onnx", "torchscript", "tflite"])

        assert mock_model.export.call_count == 3

    def test_export_unsupported_format_skipped(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Unsupported formats are skipped with warning."""
        mock_model = create_mock_yolo(mocker)

        exporter = ModelExporter(mock_model, str(tmp_path))
        result = exporter.export(["invalid_format"])

        assert "invalid_format" not in result
        mock_model.export.assert_not_called()

    def test_export_handles_exception(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Export continues after individual format failure."""
        mock_model = create_mock_yolo(mocker)
        mock_model.export.side_effect = [
            Exception("Export failed"),
            "/path/to/model.tflite",
        ]

        exporter = ModelExporter(mock_model, str(tmp_path))
        result = exporter.export(["onnx", "tflite"])

        assert "onnx" not in result
        assert "tflite" in result

    def test_export_half_precision(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """FP16 export passes half=True."""
        mock_model = create_mock_yolo(mocker)
        mock_model.export.return_value = "/model.onnx"

        exporter = ModelExporter(mock_model, str(tmp_path))
        exporter.export(["onnx"], half=True)

        call_kwargs = mock_model.export.call_args[1]
        assert call_kwargs["half"] is True


# --- TestModelQuantizer ---


class TestModelQuantizer:
    """Tests for ModelQuantizer class."""

    def test_init_stores_model_path(self, tmp_path: Path) -> None:
        """Model path is stored as Path object."""
        model_file = tmp_path / "model.pt"
        model_file.touch()

        quantizer = ModelQuantizer(str(model_file))

        assert quantizer.model_path == model_file

    def test_quantize_int8(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """INT8 quantization exports with correct parameters."""
        model_file = tmp_path / "model.pt"
        model_file.touch()

        mock_model = create_mock_yolo(mocker)
        mock_model.export.return_value = str(tmp_path / "model_int8.onnx")
        mocker.patch("screencropnet_yolo.model.YOLO", return_value=mock_model)

        quantizer = ModelQuantizer(str(model_file))
        result = quantizer.quantize_int8("/calibration/data", str(tmp_path / "output"))

        mock_model.export.assert_called_once_with(
            format="onnx", int8=True, data="/calibration/data"
        )
        assert result == str(tmp_path / "model_int8.onnx")

    def test_quantize_fp16(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """FP16 quantization exports with half=True."""
        model_file = tmp_path / "model.pt"
        model_file.touch()

        mock_model = create_mock_yolo(mocker)
        mock_model.export.return_value = str(tmp_path / "model_fp16.onnx")
        mocker.patch("screencropnet_yolo.model.YOLO", return_value=mock_model)

        quantizer = ModelQuantizer(str(model_file))
        result = quantizer.quantize_fp16(str(tmp_path / "output"))

        mock_model.export.assert_called_once_with(format="onnx", half=True)
        assert result == str(tmp_path / "model_fp16.onnx")


# --- TestGetModelInfo ---


class TestGetModelInfo:
    """Tests for get_model_info function."""

    def test_basic_info(self, mocker: MockerFixture) -> None:
        """Returns task and model_type from model."""
        mock_model = mocker.MagicMock()
        mock_model.task = "detect"
        mock_model.type = "yolo26"
        del mock_model.model  # Remove model attribute

        result = get_model_info(mock_model)

        assert result["task"] == "detect"
        assert result["model_type"] == "yolo26"

    def test_with_parameters(self, mocker: MockerFixture) -> None:
        """Computes parameter counts when model.model exists."""
        mock_model = create_mock_yolo(mocker)

        result = get_model_info(mock_model)

        assert "total_parameters" in result
        assert "trainable_parameters" in result
        assert "parameters_mb" in result
        assert result["total_parameters"] == 10_000_000
        assert result["trainable_parameters"] == 10_000_000

    def test_without_internal_model(self, mocker: MockerFixture) -> None:
        """Handles model without .model attribute."""
        mock_model = mocker.MagicMock(spec=["task", "type"])
        mock_model.task = "classify"
        mock_model.type = "resnet"

        result = get_model_info(mock_model)

        assert "total_parameters" not in result
        assert result["task"] == "classify"

    def test_parameter_memory_calculation(self, mocker: MockerFixture) -> None:
        """Parameters MB calculated as float32."""
        mock_param = mocker.MagicMock()
        mock_param.numel.return_value = 250000  # 1 MB in float32
        mock_param.requires_grad = True

        mock_model = mocker.MagicMock()
        mock_model.task = "detect"
        mock_model.type = "v11"
        mock_model.model.parameters.return_value = [mock_param]

        result = get_model_info(mock_model)

        assert result["parameters_mb"] == 1.0


# --- TestCompareModels ---


class TestCompareModels:
    """Tests for compare_models function."""

    def test_compare_models_basic(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Returns comparison dict with timing and detections."""
        mock_model = create_mock_yolo(mocker)
        mocker.patch("screencropnet_yolo.model.YOLO", return_value=mock_model)

        model1 = tmp_path / "model1.pt"
        model1.touch()
        test_image = tmp_path / "test.jpg"
        test_image.touch()

        result = compare_models([str(model1)], str(test_image))

        assert "model1" in result
        assert "inference_time_ms" in result["model1"]
        assert "detections" in result["model1"]
        assert "model_info" in result["model1"]

    def test_compare_models_runs_warmup(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Warmup prediction is run before timing."""
        mock_model = create_mock_yolo(mocker)
        mocker.patch("screencropnet_yolo.model.YOLO", return_value=mock_model)

        model1 = tmp_path / "model.pt"
        model1.touch()
        test_image = tmp_path / "test.jpg"
        test_image.touch()

        compare_models([str(model1)], str(test_image))

        # 1 warmup + 10 timed runs
        assert mock_model.predict.call_count == 11

    def test_compare_models_detection_count(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Detection count extracted from prediction results."""
        mock_result = mocker.MagicMock()
        mock_result.boxes = [mocker.MagicMock()] * 5

        mock_model = mocker.MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_model.task = "detect"
        mock_model.type = "v11"
        del mock_model.model  # Remove for simplicity

        mocker.patch("screencropnet_yolo.model.YOLO", return_value=mock_model)

        model1 = tmp_path / "model.pt"
        model1.touch()
        test_image = tmp_path / "test.jpg"
        test_image.touch()

        result = compare_models([str(model1)], str(test_image))

        assert result["model"]["detections"] == 5

    def test_compare_models_empty_prediction(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Handles empty prediction results."""
        mock_model = mocker.MagicMock()
        mock_model.predict.return_value = []
        mock_model.task = "detect"
        mock_model.type = "v11"
        del mock_model.model

        mocker.patch("screencropnet_yolo.model.YOLO", return_value=mock_model)

        model1 = tmp_path / "model.pt"
        model1.touch()
        test_image = tmp_path / "test.jpg"
        test_image.touch()

        result = compare_models([str(model1)], str(test_image))

        assert result["model"]["detections"] == 0
