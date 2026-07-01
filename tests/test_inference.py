"""Tests for inference module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from pytest_mock import MockerFixture

from screencropnet_yolo.inference import (
    Detection,
    InferencePipeline,
    InferenceResult,
    ResultExporter,
    apply_nms,
)

# --- Helper Functions ---


def create_mock_yolo_model(mocker: MockerFixture) -> Any:
    """Create a mock YOLO model for inference tests."""
    mock_model = mocker.MagicMock()
    mock_model.predict.return_value = []
    return mock_model


def create_mock_boxes(mocker: MockerFixture, num_boxes: int = 3) -> Any:
    """Create mock detection boxes."""
    mock_boxes = mocker.MagicMock()

    # Create tensor-like mocks for box data
    xyxy_data = []
    conf_data = []
    cls_data = []

    for i in range(num_boxes):
        xyxy_mock = mocker.MagicMock()
        xyxy_mock.cpu.return_value.numpy.return_value = np.array(
            [10.0 + i * 100, 20.0 + i * 50, 110.0 + i * 100, 120.0 + i * 50]
        )
        xyxy_data.append(xyxy_mock)

        conf_mock = mocker.MagicMock()
        conf_mock.cpu.return_value.numpy.return_value = 0.9 - i * 0.2
        conf_data.append(conf_mock)

        cls_mock = mocker.MagicMock()
        cls_mock.cpu.return_value.numpy.return_value = i
        cls_data.append(cls_mock)

    mock_boxes.xyxy = xyxy_data
    mock_boxes.conf = conf_data
    mock_boxes.cls = cls_data
    mock_boxes.__len__ = lambda self: num_boxes

    return mock_boxes


def create_detection(
    class_id: int = 0,
    class_name: str = "test_class",
    confidence: float = 0.9,
    bbox: tuple[float, float, float, float] = (10.0, 20.0, 110.0, 120.0),
    bbox_normalized: tuple[float, float, float, float] | None = None,
) -> Detection:
    """Create a Detection instance for testing."""
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        bbox_normalized=bbox_normalized,
    )


def create_inference_result(
    image_path: str = "test.jpg",
    image_size: tuple[int, int] = (640, 480),
    detections: list[Detection] | None = None,
    inference_time: float = 10.0,
) -> InferenceResult:
    """Create an InferenceResult instance for testing."""
    result = InferenceResult(
        image_path=image_path,
        image_size=image_size,
        inference_time=inference_time,
    )
    if detections:
        result.detections = detections
    return result


# --- TestDetection ---


class TestDetection:
    """Tests for Detection dataclass."""

    def test_basic_creation(self) -> None:
        """Detection can be created with required fields."""
        det = Detection(
            class_id=0,
            class_name="profile_info",
            confidence=0.95,
            bbox=(10.0, 20.0, 110.0, 120.0),
        )

        assert det.class_id == 0
        assert det.class_name == "profile_info"
        assert det.confidence == 0.95
        assert det.bbox == (10.0, 20.0, 110.0, 120.0)
        assert det.bbox_normalized is None

    def test_with_normalized_bbox(self) -> None:
        """Detection can include normalized coordinates."""
        det = Detection(
            class_id=1,
            class_name="tweet_text",
            confidence=0.85,
            bbox=(100.0, 200.0, 300.0, 400.0),
            bbox_normalized=(0.1, 0.2, 0.3, 0.4),
        )

        assert det.bbox_normalized == (0.1, 0.2, 0.3, 0.4)

    def test_to_dict_without_normalized(self) -> None:
        """to_dict returns correct structure without normalized coords."""
        det = create_detection()
        result = det.to_dict()

        assert result["class_id"] == 0
        assert result["class_name"] == "test_class"
        assert result["confidence"] == 0.9
        assert result["bbox"] == {"x1": 10.0, "y1": 20.0, "x2": 110.0, "y2": 120.0}
        assert result["bbox_normalized"] is None

    def test_to_dict_with_normalized(self) -> None:
        """to_dict includes normalized coordinates when present."""
        det = create_detection(bbox_normalized=(0.1, 0.2, 0.3, 0.4))
        result = det.to_dict()

        assert result["bbox_normalized"] == {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4}


# --- TestInferenceResult ---


class TestInferenceResult:
    """Tests for InferenceResult dataclass."""

    def test_basic_creation(self) -> None:
        """InferenceResult can be created with required fields."""
        result = InferenceResult(
            image_path="test.jpg",
            image_size=(640, 480),
        )

        assert result.image_path == "test.jpg"
        assert result.image_size == (640, 480)
        assert result.detections == []
        assert result.inference_time == 0.0

    def test_with_detections(self) -> None:
        """InferenceResult holds list of detections."""
        det1 = create_detection(class_id=0, class_name="class_a")
        det2 = create_detection(class_id=1, class_name="class_b")

        result = create_inference_result(detections=[det1, det2])

        assert len(result.detections) == 2
        assert result.detections[0].class_name == "class_a"
        assert result.detections[1].class_name == "class_b"

    def test_to_dict_structure(self) -> None:
        """to_dict returns complete result structure."""
        det = create_detection()
        result = create_inference_result(
            image_path="image.png",
            image_size=(1920, 1080),
            detections=[det],
            inference_time=15.5,
        )

        output = result.to_dict()

        assert output["image_path"] == "image.png"
        assert output["image_size"] == {"width": 1920, "height": 1080}
        assert output["num_detections"] == 1
        assert output["inference_time_ms"] == 15.5
        assert len(output["detections"]) == 1

    def test_filter_by_confidence(self) -> None:
        """filter_by_confidence removes low-confidence detections."""
        det_high = create_detection(confidence=0.9)
        det_medium = create_detection(confidence=0.6)
        det_low = create_detection(confidence=0.3)

        result = create_inference_result(detections=[det_high, det_medium, det_low])
        filtered = result.filter_by_confidence(0.5)

        assert len(filtered.detections) == 2
        assert all(d.confidence >= 0.5 for d in filtered.detections)

    def test_filter_by_confidence_preserves_metadata(self) -> None:
        """Filtered result retains original metadata."""
        result = create_inference_result(
            image_path="original.jpg",
            image_size=(800, 600),
            detections=[create_detection(confidence=0.9)],
            inference_time=20.0,
        )
        filtered = result.filter_by_confidence(0.5)

        assert filtered.image_path == "original.jpg"
        assert filtered.image_size == (800, 600)
        assert filtered.inference_time == 20.0

    def test_filter_by_class(self) -> None:
        """filter_by_class keeps only specified class IDs."""
        det0 = create_detection(class_id=0)
        det1 = create_detection(class_id=1)
        det2 = create_detection(class_id=2)
        det3 = create_detection(class_id=0)

        result = create_inference_result(detections=[det0, det1, det2, det3])
        filtered = result.filter_by_class([0, 2])

        assert len(filtered.detections) == 3
        assert all(d.class_id in [0, 2] for d in filtered.detections)

    def test_filter_by_class_empty_result(self) -> None:
        """filter_by_class can return empty detections."""
        det = create_detection(class_id=0)
        result = create_inference_result(detections=[det])
        filtered = result.filter_by_class([5, 6, 7])

        assert len(filtered.detections) == 0


# --- TestInferencePipeline ---


class TestInferencePipeline:
    """Tests for InferencePipeline class."""

    def test_init_loads_model(self, mocker: MockerFixture) -> None:
        """Initializing pipeline loads YOLO model."""
        mock_yolo = mocker.patch("screencropnet_yolo.inference.YOLO")

        InferencePipeline(
            model_path="/path/to/model.pt",
            class_names=["class_a", "class_b"],
        )

        mock_yolo.assert_called_once_with("/path/to/model.pt")

    def test_init_stores_parameters(self, mocker: MockerFixture) -> None:
        """Pipeline stores configuration parameters."""
        mocker.patch("screencropnet_yolo.inference.YOLO")

        pipeline = InferencePipeline(
            model_path="/model.pt",
            class_names=["a", "b", "c"],
            device="cuda:0",
            conf_threshold=0.3,
            iou_threshold=0.5,
            max_detections=100,
        )

        assert pipeline.class_names == ["a", "b", "c"]
        assert pipeline.device == "cuda:0"
        assert pipeline.conf_threshold == 0.3
        assert pipeline.iou_threshold == 0.5
        assert pipeline.max_detections == 100

    def test_predict_image_with_path(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """predict_image processes image from file path."""
        mock_model = create_mock_yolo_model(mocker)
        mock_boxes = create_mock_boxes(mocker, num_boxes=2)
        mock_result = mocker.MagicMock()
        mock_result.boxes = mock_boxes
        mock_model.predict.return_value = [mock_result]

        mocker.patch("screencropnet_yolo.inference.YOLO", return_value=mock_model)

        # Create a test image
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        image_path = tmp_path / "test_image.jpg"
        mocker.patch("screencropnet_yolo.inference.cv2.imread", return_value=test_img)

        pipeline = InferencePipeline(
            model_path="/model.pt",
            class_names=["class_a", "class_b", "class_c"],
        )
        result = pipeline.predict_image(str(image_path))

        assert result.image_path == str(image_path)
        assert result.image_size == (640, 480)
        assert len(result.detections) == 2

    def test_inference_no_explicit_nms_call(self, mocker: MockerFixture) -> None:
        """YOLO26 is end-to-end NMS-free: the pipeline must not post-process with apply_nms."""
        mock_model = create_mock_yolo_model(mocker)
        mock_boxes = create_mock_boxes(mocker, num_boxes=3)
        mock_result = mocker.MagicMock()
        mock_result.boxes = mock_boxes
        mock_model.predict.return_value = [mock_result]
        mocker.patch("screencropnet_yolo.inference.YOLO", return_value=mock_model)
        nms_spy = mocker.patch("screencropnet_yolo.inference.apply_nms")

        pipeline = InferencePipeline(model_path="/model.pt", class_names=["tweet_region"])
        pipeline.predict_image(np.zeros((480, 640, 3), dtype=np.uint8))

        nms_spy.assert_not_called()

    def test_inference_handles_e2e_results_shape(self, mocker: MockerFixture) -> None:
        """The parser reads YOLO26's Results.boxes (.xyxy/.conf/.cls) into Detections."""
        mock_model = create_mock_yolo_model(mocker)
        mock_boxes = create_mock_boxes(mocker, num_boxes=3)
        mock_result = mocker.MagicMock()
        mock_result.boxes = mock_boxes
        mock_model.predict.return_value = [mock_result]
        mocker.patch("screencropnet_yolo.inference.YOLO", return_value=mock_model)

        pipeline = InferencePipeline(model_path="/model.pt", class_names=["tweet_region"])
        result = pipeline.predict_image(np.zeros((480, 640, 3), dtype=np.uint8))

        assert len(result.detections) == 3
        # create_mock_boxes emits descending confidences 0.9, 0.7, 0.5
        assert [round(d.confidence, 1) for d in result.detections] == [0.9, 0.7, 0.5]
        assert [d.class_id for d in result.detections] == [0, 1, 2]
        first = result.detections[0]
        assert first.bbox == pytest.approx((10.0, 20.0, 110.0, 120.0))
        assert first.bbox_normalized is not None

    def test_predict_image_with_numpy_array(self, mocker: MockerFixture) -> None:
        """predict_image processes numpy array directly."""
        mock_model = create_mock_yolo_model(mocker)
        mock_boxes = create_mock_boxes(mocker, num_boxes=1)
        mock_result = mocker.MagicMock()
        mock_result.boxes = mock_boxes
        mock_model.predict.return_value = [mock_result]

        mocker.patch("screencropnet_yolo.inference.YOLO", return_value=mock_model)

        pipeline = InferencePipeline(
            model_path="/model.pt",
            class_names=["class_a"],
        )

        img_array = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pipeline.predict_image(img_array)

        assert result.image_path == "numpy_array"
        assert result.image_size == (640, 480)

    def test_predict_image_failed_read(self, mocker: MockerFixture) -> None:
        """predict_image raises error for unreadable image."""
        mock_model = create_mock_yolo_model(mocker)
        mocker.patch("screencropnet_yolo.inference.YOLO", return_value=mock_model)
        mocker.patch("screencropnet_yolo.inference.cv2.imread", return_value=None)

        pipeline = InferencePipeline(
            model_path="/model.pt",
            class_names=["class_a"],
        )

        with pytest.raises(ValueError, match="Failed to read image"):
            pipeline.predict_image("/nonexistent.jpg")

    def test_predict_image_uses_custom_thresholds(self, mocker: MockerFixture) -> None:
        """Custom conf and iou thresholds are passed to model."""
        mock_model = create_mock_yolo_model(mocker)
        mocker.patch("screencropnet_yolo.inference.YOLO", return_value=mock_model)

        pipeline = InferencePipeline(
            model_path="/model.pt",
            class_names=["class_a"],
        )

        img_array = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline.predict_image(img_array, conf=0.7, iou=0.3)

        call_kwargs = mock_model.predict.call_args[1]
        assert call_kwargs["conf"] == 0.7
        assert call_kwargs["iou"] == 0.3

    def test_predict_image_no_boxes(self, mocker: MockerFixture) -> None:
        """predict_image handles results with no boxes."""
        mock_model = create_mock_yolo_model(mocker)
        mock_result = mocker.MagicMock()
        mock_result.boxes = None
        mock_model.predict.return_value = [mock_result]

        mocker.patch("screencropnet_yolo.inference.YOLO", return_value=mock_model)

        pipeline = InferencePipeline(
            model_path="/model.pt",
            class_names=["class_a"],
        )

        img_array = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pipeline.predict_image(img_array)

        assert len(result.detections) == 0

    def test_predict_image_empty_results(self, mocker: MockerFixture) -> None:
        """predict_image handles empty results list."""
        mock_model = create_mock_yolo_model(mocker)
        mock_model.predict.return_value = []

        mocker.patch("screencropnet_yolo.inference.YOLO", return_value=mock_model)

        pipeline = InferencePipeline(
            model_path="/model.pt",
            class_names=["class_a"],
        )

        img_array = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pipeline.predict_image(img_array)

        assert len(result.detections) == 0

    def test_predict_image_unknown_class_id(self, mocker: MockerFixture) -> None:
        """predict_image handles class_id beyond class_names."""
        mock_model = create_mock_yolo_model(mocker)

        # Create box with class_id=5 (beyond the single class name)
        mock_boxes = mocker.MagicMock()
        xyxy_mock = mocker.MagicMock()
        xyxy_mock.cpu.return_value.numpy.return_value = np.array([10.0, 20.0, 110.0, 120.0])
        conf_mock = mocker.MagicMock()
        conf_mock.cpu.return_value.numpy.return_value = 0.9
        cls_mock = mocker.MagicMock()
        cls_mock.cpu.return_value.numpy.return_value = 5

        mock_boxes.xyxy = [xyxy_mock]
        mock_boxes.conf = [conf_mock]
        mock_boxes.cls = [cls_mock]
        mock_boxes.__len__ = lambda self: 1

        mock_result = mocker.MagicMock()
        mock_result.boxes = mock_boxes
        mock_model.predict.return_value = [mock_result]

        mocker.patch("screencropnet_yolo.inference.YOLO", return_value=mock_model)

        pipeline = InferencePipeline(
            model_path="/model.pt",
            class_names=["class_a"],  # Only one class
        )

        img_array = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pipeline.predict_image(img_array)

        assert result.detections[0].class_name == "class_5"

    def test_predict_batch(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """predict_batch processes multiple images."""
        mock_model = create_mock_yolo_model(mocker)

        mock_boxes = create_mock_boxes(mocker, num_boxes=1)
        mock_result = mocker.MagicMock()
        mock_result.boxes = mock_boxes
        mock_model.predict.return_value = [mock_result, mock_result]

        mocker.patch("screencropnet_yolo.inference.YOLO", return_value=mock_model)

        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        mocker.patch("screencropnet_yolo.inference.cv2.imread", return_value=test_img)

        pipeline = InferencePipeline(
            model_path="/model.pt",
            class_names=["class_a"],
        )

        image_paths: list[str | npt.NDArray[np.uint8]] = [
            str(tmp_path / "img1.jpg"),
            str(tmp_path / "img2.jpg"),
        ]
        results = pipeline.predict_batch(image_paths)

        assert len(results) == 2

    def test_predict_batch_with_numpy_arrays(self, mocker: MockerFixture) -> None:
        """predict_batch processes numpy arrays."""
        mock_model = create_mock_yolo_model(mocker)

        mock_result = mocker.MagicMock()
        mock_result.boxes = None
        mock_model.predict.return_value = [mock_result, mock_result]

        mocker.patch("screencropnet_yolo.inference.YOLO", return_value=mock_model)

        pipeline = InferencePipeline(
            model_path="/model.pt",
            class_names=["class_a"],
        )

        images: list[str | npt.NDArray[np.uint8]] = [
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros((480, 640, 3), dtype=np.uint8),
        ]
        results = pipeline.predict_batch(images)

        assert len(results) == 2
        assert "image_0" in results[0].image_path
        assert "image_1" in results[1].image_path

    def test_predict_video_opens_capture(self, mocker: MockerFixture) -> None:
        """predict_video opens video capture."""
        mock_model = create_mock_yolo_model(mocker)
        mocker.patch("screencropnet_yolo.inference.YOLO", return_value=mock_model)

        mock_cap = mocker.MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda x: {
            5: 30,  # FPS
            3: 640,  # WIDTH
            4: 480,  # HEIGHT
            7: 10,  # FRAME_COUNT
        }.get(x, 0)
        mock_cap.read.side_effect = [(True, np.zeros((480, 640, 3), dtype=np.uint8))] * 3 + [
            (False, None)
        ]

        mocker.patch("screencropnet_yolo.inference.cv2.VideoCapture", return_value=mock_cap)

        pipeline = InferencePipeline(
            model_path="/model.pt",
            class_names=["class_a"],
        )

        results = pipeline.predict_video("/video.mp4")

        assert len(results) == 3
        mock_cap.release.assert_called_once()

    def test_predict_video_invalid_path(self, mocker: MockerFixture) -> None:
        """predict_video raises error for invalid video."""
        mock_model = create_mock_yolo_model(mocker)
        mocker.patch("screencropnet_yolo.inference.YOLO", return_value=mock_model)

        mock_cap = mocker.MagicMock()
        mock_cap.isOpened.return_value = False
        mocker.patch("screencropnet_yolo.inference.cv2.VideoCapture", return_value=mock_cap)

        pipeline = InferencePipeline(
            model_path="/model.pt",
            class_names=["class_a"],
        )

        with pytest.raises(ValueError, match="Could not open video"):
            pipeline.predict_video("/nonexistent.mp4")

    def test_predict_video_with_output(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """predict_video writes annotated video."""
        mock_model = create_mock_yolo_model(mocker)
        mocker.patch("screencropnet_yolo.inference.YOLO", return_value=mock_model)

        mock_cap = mocker.MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda x: {5: 30, 3: 640, 4: 480, 7: 2}.get(x, 0)
        mock_cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None),
        ]
        mocker.patch("screencropnet_yolo.inference.cv2.VideoCapture", return_value=mock_cap)

        mock_writer = mocker.MagicMock()
        mocker.patch("screencropnet_yolo.inference.cv2.VideoWriter", return_value=mock_writer)
        mocker.patch("screencropnet_yolo.inference.cv2.VideoWriter_fourcc", return_value=1)

        pipeline = InferencePipeline(
            model_path="/model.pt",
            class_names=["class_a"],
        )

        output_path = str(tmp_path / "output.mp4")
        pipeline.predict_video("/video.mp4", output_path=output_path)

        mock_writer.write.assert_called()
        mock_writer.release.assert_called_once()

    def test_draw_detections(self, mocker: MockerFixture) -> None:
        """draw_detections annotates image with boxes."""
        mock_model = create_mock_yolo_model(mocker)
        mocker.patch("screencropnet_yolo.inference.YOLO", return_value=mock_model)

        pipeline = InferencePipeline(
            model_path="/model.pt",
            class_names=["class_a"],
        )

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        det = create_detection()
        result = create_inference_result(detections=[det])

        annotated = pipeline.draw_detections(image, result)

        assert annotated.shape == image.shape
        # Original image should not be modified
        assert np.array_equal(image, np.zeros((480, 640, 3), dtype=np.uint8))

    def test_get_color_palette(self, mocker: MockerFixture) -> None:
        """_get_color_palette returns list of RGB tuples."""
        mock_model = create_mock_yolo_model(mocker)
        mocker.patch("screencropnet_yolo.inference.YOLO", return_value=mock_model)

        pipeline = InferencePipeline(
            model_path="/model.pt",
            class_names=["class_a"],
        )

        colors = pipeline._get_color_palette()

        assert len(colors) == 12
        for color in colors:
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)


# --- TestResultExporter ---


class TestResultExporter:
    """Tests for ResultExporter class."""

    def test_to_json_single_result(self, tmp_path: Path) -> None:
        """to_json exports single result to JSON."""
        det = create_detection()
        result = create_inference_result(detections=[det])

        output_path = str(tmp_path / "results.json")
        returned_path = ResultExporter.to_json(result, output_path)

        assert returned_path == output_path
        assert Path(output_path).exists()

        with open(output_path) as f:
            data = json.load(f)

        assert data["num_images"] == 1
        assert data["total_detections"] == 1
        assert len(data["results"]) == 1

    def test_to_json_serializes_numpy_float_bbox(self, tmp_path: Path) -> None:
        """Real predictions carry numpy float32 bbox coords; to_json must serialize them."""
        det = create_detection(
            confidence=np.array([0.87], dtype=np.float32)[0],
            bbox=tuple(np.array([10.0, 20.0, 110.0, 120.0], dtype=np.float32)),  # ty: ignore[invalid-argument-type]
            bbox_normalized=tuple(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)),  # ty: ignore[invalid-argument-type]
        )
        result = create_inference_result(detections=[det])

        output_path = str(tmp_path / "np.json")
        ResultExporter.to_json(result, output_path)

        data = json.loads(Path(output_path).read_text())
        bbox = data["results"][0]["detections"][0]["bbox"]
        assert bbox["x1"] == 10.0

    def test_to_coco_serializes_numpy_float_bbox(self, tmp_path: Path) -> None:
        """to_coco must serialize numpy float32 coords/scores from real predictions."""
        det = create_detection(
            confidence=np.array([0.87], dtype=np.float32)[0],
            bbox=tuple(np.array([10.0, 20.0, 110.0, 120.0], dtype=np.float32)),  # ty: ignore[invalid-argument-type]
        )
        result = create_inference_result(detections=[det])

        output_path = str(tmp_path / "coco.json")
        ResultExporter.to_coco([result], output_path, ["test_class"])

        data = json.loads(Path(output_path).read_text())
        assert data[0]["bbox"][2] == 100.0  # width = x2 - x1

    def test_to_json_multiple_results(self, tmp_path: Path) -> None:
        """to_json exports list of results."""
        results = [
            create_inference_result(
                image_path="img1.jpg",
                detections=[create_detection(), create_detection()],
            ),
            create_inference_result(
                image_path="img2.jpg",
                detections=[create_detection()],
            ),
        ]

        output_path = str(tmp_path / "results.json")
        ResultExporter.to_json(results, output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert data["num_images"] == 2
        assert data["total_detections"] == 3

    def test_to_coco_format(self, tmp_path: Path) -> None:
        """to_coco exports to COCO format."""
        det = create_detection(
            class_id=0,
            bbox=(100.0, 200.0, 300.0, 400.0),
            confidence=0.85,
        )
        result = create_inference_result(
            image_path="/path/to/image_001.jpg",
            detections=[det],
        )

        output_path = str(tmp_path / "coco.json")
        returned_path = ResultExporter.to_coco([result], output_path, ["class_a"])

        assert returned_path == output_path

        with open(output_path) as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["image_id"] == "image_001"
        assert data[0]["category_id"] == 1  # COCO uses 1-indexed
        assert data[0]["bbox"] == [100.0, 200.0, 200.0, 200.0]  # x, y, w, h
        assert data[0]["score"] == 0.85

    def test_to_yolo_format(self, tmp_path: Path) -> None:
        """to_yolo exports to YOLO format files."""
        det = create_detection(
            class_id=0,
            bbox=(64.0, 48.0, 192.0, 144.0),
            bbox_normalized=(0.1, 0.1, 0.3, 0.3),
        )
        result = create_inference_result(
            image_path="test_image.jpg",
            image_size=(640, 480),
            detections=[det],
        )

        output_dir = str(tmp_path / "yolo_output")
        returned_path = ResultExporter.to_yolo([result], output_dir)

        assert Path(returned_path).exists()

        txt_file = Path(output_dir) / "test_image.txt"
        assert txt_file.exists()

        content = txt_file.read_text().strip()
        parts = content.split()
        assert parts[0] == "0"  # class_id
        # x_center, y_center, width, height
        assert len(parts) == 5

    def test_to_yolo_calculates_normalized_coords(self, tmp_path: Path) -> None:
        """to_yolo calculates normalized coords when not provided."""
        det = create_detection(
            class_id=1,
            bbox=(100.0, 100.0, 300.0, 200.0),
            bbox_normalized=None,
        )
        result = create_inference_result(
            image_path="image.jpg",
            image_size=(1000, 500),
            detections=[det],
        )

        output_dir = str(tmp_path / "yolo_output")
        ResultExporter.to_yolo([result], output_dir)

        txt_file = Path(output_dir) / "image.txt"
        content = txt_file.read_text().strip()
        parts = content.split()

        # x_center = (100/1000 + 300/1000) / 2 = 0.2
        # y_center = (100/500 + 200/500) / 2 = 0.3
        # width = (300 - 100) / 1000 = 0.2
        # height = (200 - 100) / 500 = 0.2
        assert float(parts[1]) == pytest.approx(0.2, abs=0.001)
        assert float(parts[2]) == pytest.approx(0.3, abs=0.001)
        assert float(parts[3]) == pytest.approx(0.2, abs=0.001)
        assert float(parts[4]) == pytest.approx(0.2, abs=0.001)


# --- TestApplyNms ---


class TestApplyNms:
    """Tests for apply_nms function."""

    def test_empty_detections(self) -> None:
        """apply_nms returns empty list for empty input."""
        result = apply_nms([])

        assert result == []

    def test_single_detection(self) -> None:
        """Single detection is kept."""
        det = create_detection()
        result = apply_nms([det])

        assert len(result) == 1
        assert result[0] is det

    def test_non_overlapping_detections(self) -> None:
        """Non-overlapping detections are all kept."""
        det1 = create_detection(bbox=(0.0, 0.0, 100.0, 100.0))
        det2 = create_detection(bbox=(200.0, 200.0, 300.0, 300.0))
        det3 = create_detection(bbox=(400.0, 400.0, 500.0, 500.0))

        result = apply_nms([det1, det2, det3])

        assert len(result) == 3

    def test_overlapping_detections_same_class(self) -> None:
        """Overlapping detections of same class are suppressed."""
        det_high = create_detection(
            class_id=0,
            confidence=0.9,
            bbox=(0.0, 0.0, 100.0, 100.0),
        )
        det_low = create_detection(
            class_id=0,
            confidence=0.7,
            bbox=(10.0, 10.0, 110.0, 110.0),  # High overlap
        )

        result = apply_nms([det_high, det_low], iou_threshold=0.3)

        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_overlapping_detections_different_classes(self) -> None:
        """Overlapping detections of different classes are both kept."""
        det1 = create_detection(
            class_id=0,
            confidence=0.9,
            bbox=(0.0, 0.0, 100.0, 100.0),
        )
        det2 = create_detection(
            class_id=1,
            confidence=0.7,
            bbox=(10.0, 10.0, 110.0, 110.0),
        )

        result = apply_nms([det1, det2], iou_threshold=0.3, class_agnostic=False)

        assert len(result) == 2

    def test_class_agnostic_nms(self) -> None:
        """Class-agnostic NMS suppresses across classes."""
        det1 = create_detection(
            class_id=0,
            confidence=0.9,
            bbox=(0.0, 0.0, 100.0, 100.0),
        )
        det2 = create_detection(
            class_id=1,
            confidence=0.7,
            bbox=(10.0, 10.0, 110.0, 110.0),
        )

        result = apply_nms([det1, det2], iou_threshold=0.3, class_agnostic=True)

        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_iou_threshold_boundary(self) -> None:
        """Detections at threshold boundary are handled correctly."""
        # Two boxes with ~50% overlap
        det1 = create_detection(confidence=0.9, bbox=(0.0, 0.0, 100.0, 100.0))
        det2 = create_detection(confidence=0.8, bbox=(50.0, 0.0, 150.0, 100.0))

        # IoU = 50*100 / (100*100 + 100*100 - 50*100) = 5000/15000 ≈ 0.33
        # With threshold 0.5, both should be kept
        result = apply_nms([det1, det2], iou_threshold=0.5)
        assert len(result) == 2

        # With threshold 0.3, one should be suppressed
        result = apply_nms([det1, det2], iou_threshold=0.3)
        assert len(result) == 1

    def test_multiple_classes_nms(self) -> None:
        """NMS correctly handles multiple classes with overlaps."""
        detections = [
            create_detection(class_id=0, confidence=0.9, bbox=(0.0, 0.0, 100.0, 100.0)),
            create_detection(class_id=0, confidence=0.85, bbox=(5.0, 5.0, 105.0, 105.0)),
            create_detection(class_id=1, confidence=0.8, bbox=(0.0, 0.0, 100.0, 100.0)),
            create_detection(class_id=1, confidence=0.75, bbox=(5.0, 5.0, 105.0, 105.0)),
            create_detection(class_id=2, confidence=0.7, bbox=(200.0, 200.0, 300.0, 300.0)),
        ]

        result = apply_nms(detections, iou_threshold=0.3, class_agnostic=False)

        # Should keep: highest for class 0, highest for class 1, and class 2 (no overlap)
        assert len(result) == 3
        class_ids = [d.class_id for d in result]
        assert 0 in class_ids
        assert 1 in class_ids
        assert 2 in class_ids
