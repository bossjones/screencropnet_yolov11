"""Tests for dataset_utils module."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from pytest_mock import MockerFixture

from screencropnet_yolo.dataset_utils import (
    DatasetSplitter,
    DatasetValidator,
    RoboflowLoader,
    TwitterScreenshotPreprocessor,
    check_class_imbalance,
    create_dataset_yaml,
)

# --- Fixtures ---


def create_test_image(path: Path, width: int = 100, height: int = 100) -> None:
    """Create a simple test image file."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (128, 128, 128)  # Gray image
    cv2.imwrite(str(path), img)


def create_yolo_annotation(path: Path, annotations: list[str]) -> None:
    """Create a YOLO format annotation file."""
    path.write_text("\n".join(annotations))


def create_valid_dataset(root: Path, splits: list[str] | None = None) -> None:
    """Create a valid YOLO dataset structure with images and labels."""
    if splits is None:
        splits = ["train", "val"]

    for split in splits:
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for i in range(3):
            img_path = images_dir / f"image_{i}.jpg"
            label_path = labels_dir / f"image_{i}.txt"
            create_test_image(img_path)
            # Valid YOLO annotation: class_id x_center y_center width height
            create_yolo_annotation(label_path, [f"0 0.5 0.5 0.2 0.{i + 1}"])


# --- DatasetValidator Tests ---


class TestDatasetValidator:
    """Tests for DatasetValidator class."""

    def test_validator_with_valid_dataset(self, tmp_path: Path) -> None:
        """Valid dataset structure passes validation."""
        create_valid_dataset(tmp_path)
        validator = DatasetValidator(str(tmp_path), class_names=["tweet"])

        is_valid, stats, errors = validator.validate()

        assert is_valid
        assert len(errors) == 0
        assert stats.total_images == 6  # 3 train + 3 val

    def test_validator_missing_dataset_path(self, tmp_path: Path) -> None:
        """Non-existent dataset path returns error."""
        validator = DatasetValidator(str(tmp_path / "nonexistent"), class_names=["tweet"])

        is_valid, _, errors = validator.validate()

        assert not is_valid
        assert any("does not exist" in e for e in errors)

    def test_validator_missing_train_dir(self, tmp_path: Path) -> None:
        """Missing train directory returns error."""
        # Create only val directory
        (tmp_path / "val" / "images").mkdir(parents=True)
        (tmp_path / "val" / "labels").mkdir(parents=True)
        validator = DatasetValidator(str(tmp_path), class_names=["tweet"])

        is_valid, _, errors = validator.validate()

        assert not is_valid
        assert any("Required directory missing" in e for e in errors)

    def test_validator_missing_labels(self, tmp_path: Path) -> None:
        """Detects images without corresponding annotations."""
        images_dir = tmp_path / "train" / "images"
        labels_dir = tmp_path / "train" / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        # Create image without label
        create_test_image(images_dir / "orphan.jpg")

        validator = DatasetValidator(str(tmp_path), class_names=["tweet"])
        is_valid, stats, errors = validator.validate()

        assert not is_valid
        assert len(stats.missing_annotations) == 1
        assert any("Missing annotation" in e for e in errors)

    def test_validator_invalid_annotation_format(self, tmp_path: Path) -> None:
        """Catches malformed annotation lines."""
        images_dir = tmp_path / "train" / "images"
        labels_dir = tmp_path / "train" / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        create_test_image(images_dir / "test.jpg")
        # Invalid: missing coordinates
        create_yolo_annotation(labels_dir / "test.txt", ["0 0.5"])

        validator = DatasetValidator(str(tmp_path), class_names=["tweet"])
        is_valid, _, errors = validator.validate()

        assert not is_valid
        assert any("Invalid annotation format" in e for e in errors)

    def test_validator_invalid_class_id(self, tmp_path: Path) -> None:
        """Detects out-of-range class IDs."""
        images_dir = tmp_path / "train" / "images"
        labels_dir = tmp_path / "train" / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        create_test_image(images_dir / "test.jpg")
        # Class ID 5 is invalid for single-class dataset
        create_yolo_annotation(labels_dir / "test.txt", ["5 0.5 0.5 0.2 0.2"])

        validator = DatasetValidator(str(tmp_path), class_names=["tweet"])
        is_valid, _, errors = validator.validate()

        assert not is_valid
        assert any("Invalid class ID" in e for e in errors)

    def test_validator_invalid_coordinates(self, tmp_path: Path) -> None:
        """Catches coordinate values outside 0-1 range."""
        images_dir = tmp_path / "train" / "images"
        labels_dir = tmp_path / "train" / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        create_test_image(images_dir / "test.jpg")
        # x_center > 1.0 is invalid
        create_yolo_annotation(labels_dir / "test.txt", ["0 1.5 0.5 0.2 0.2"])

        validator = DatasetValidator(str(tmp_path), class_names=["tweet"])
        is_valid, _, errors = validator.validate()

        assert not is_valid
        assert any("Invalid x_center" in e for e in errors)

    def test_validator_corrupt_image(self, tmp_path: Path) -> None:
        """Detects unreadable/corrupt image files."""
        images_dir = tmp_path / "train" / "images"
        labels_dir = tmp_path / "train" / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        # Create corrupt image (just random bytes)
        (images_dir / "corrupt.jpg").write_bytes(b"not an image")
        create_yolo_annotation(labels_dir / "corrupt.txt", ["0 0.5 0.5 0.2 0.2"])

        validator = DatasetValidator(str(tmp_path), class_names=["tweet"])
        is_valid, stats, errors = validator.validate()

        assert not is_valid
        assert len(stats.corrupt_images) == 1
        assert any("Corrupt image" in e for e in errors)

    def test_validator_statistics_computed(self, tmp_path: Path) -> None:
        """Verifies statistics are correctly aggregated."""
        create_valid_dataset(tmp_path, splits=["train", "val"])
        validator = DatasetValidator(str(tmp_path), class_names=["tweet", "retweet"])

        _, stats, _ = validator.validate()

        assert stats.images_per_split["train"] == 3
        assert stats.images_per_split["val"] == 3
        assert len(stats.image_sizes) == 6
        assert stats.bbox_stats["avg_width"] == 100.0
        assert stats.bbox_stats["avg_height"] == 100.0

    def test_validator_class_distribution(self, tmp_path: Path) -> None:
        """Class distribution is correctly counted."""
        images_dir = tmp_path / "train" / "images"
        labels_dir = tmp_path / "train" / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        create_test_image(images_dir / "test.jpg")
        # Multiple annotations in one file
        create_yolo_annotation(
            labels_dir / "test.txt",
            ["0 0.5 0.5 0.2 0.2", "1 0.3 0.3 0.1 0.1", "0 0.7 0.7 0.1 0.1"],
        )

        validator = DatasetValidator(str(tmp_path), class_names=["tweet", "retweet"])
        _, stats, _ = validator.validate()

        assert stats.class_distribution["tweet"] == 2
        assert stats.class_distribution["retweet"] == 1


# --- DatasetSplitter Tests ---


class TestDatasetSplitter:
    """Tests for DatasetSplitter class."""

    def test_splitter_ratio_validation(self, tmp_path: Path) -> None:
        """ValueError raised when ratios don't sum to 1.0."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            DatasetSplitter(
                source_path=str(tmp_path),
                output_path=str(tmp_path / "output"),
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.1,  # Sum = 0.9
            )

    def test_splitter_creates_correct_splits(self, tmp_path: Path) -> None:
        """Files are distributed according to ratios."""
        source = tmp_path / "source"
        output = tmp_path / "output"
        source.mkdir()

        # Create 10 images
        for i in range(10):
            create_test_image(source / f"img_{i}.jpg")
            create_yolo_annotation(source / f"img_{i}.txt", ["0 0.5 0.5 0.2 0.2"])

        splitter = DatasetSplitter(
            source_path=str(source),
            output_path=str(output),
            train_ratio=0.6,
            val_ratio=0.3,
            test_ratio=0.1,
            seed=42,
        )

        counts = splitter.split()

        assert counts["train"] == 6
        assert counts["val"] == 3
        assert counts["test"] == 1

    def test_splitter_copies_labels(self, tmp_path: Path) -> None:
        """Labels are copied alongside images."""
        source = tmp_path / "source"
        output = tmp_path / "output"
        source.mkdir()

        create_test_image(source / "test.jpg")
        create_yolo_annotation(source / "test.txt", ["0 0.5 0.5 0.2 0.2"])

        splitter = DatasetSplitter(
            source_path=str(source),
            output_path=str(output),
            train_ratio=1.0,
            val_ratio=0.0,
            test_ratio=0.0,
        )
        splitter.split()

        # Check label was copied to train/labels
        label_files = list((output / "train" / "labels").glob("*.txt"))
        assert len(label_files) == 1

    def test_splitter_reproducible_with_seed(self, tmp_path: Path) -> None:
        """Same seed produces same split."""
        source = tmp_path / "source"
        source.mkdir()

        for i in range(10):
            create_test_image(source / f"img_{i}.jpg")

        output1 = tmp_path / "output1"
        output2 = tmp_path / "output2"

        for output in [output1, output2]:
            splitter = DatasetSplitter(
                source_path=str(source),
                output_path=str(output),
                train_ratio=0.6,
                val_ratio=0.2,
                test_ratio=0.2,
                seed=42,
            )
            splitter.split()

        # Compare file names in train splits
        train1_files = sorted(f.name for f in (output1 / "train" / "images").glob("*.jpg"))
        train2_files = sorted(f.name for f in (output2 / "train" / "images").glob("*.jpg"))

        assert train1_files == train2_files


# --- RoboflowLoader Tests ---


class TestRoboflowLoader:
    """Tests for RoboflowLoader class."""

    def test_roboflow_import_error(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Graceful error when roboflow package not installed."""
        loader = RoboflowLoader(
            api_key="fake_key",
            workspace="test",
            project="test",
            version=1,
            output_path=str(tmp_path),
        )

        # Mock the import to fail
        mocker.patch.dict("sys.modules", {"roboflow": None})

        with pytest.raises(ImportError, match="roboflow package not installed"):
            loader.download()


# --- TwitterScreenshotPreprocessor Tests ---


class TestTwitterScreenshotPreprocessor:
    """Tests for TwitterScreenshotPreprocessor class."""

    def test_preprocess_output_shape(self) -> None:
        """Output maintains input dimensions."""
        preprocessor = TwitterScreenshotPreprocessor(target_size=640)
        # Create BGR image (OpenCV format)
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[:] = (128, 128, 128)

        result = preprocessor.preprocess(image)

        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_letterbox_maintains_aspect_ratio(self) -> None:
        """Letterbox adds padding to maintain aspect ratio."""
        preprocessor = TwitterScreenshotPreprocessor()
        # Wide image: 200x100
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        result, _ratio, padding = preprocessor.letterbox(
            image, new_shape=(640, 640), auto=False, scale_fill=False
        )

        assert result.shape[0] == result.shape[1] == 640
        # Padding should be on top/bottom (dh > 0)
        assert padding[1] > 0  # dh

    def test_letterbox_with_scale_fill(self) -> None:
        """Scale fill mode stretches to fill target shape."""
        preprocessor = TwitterScreenshotPreprocessor()
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        result, _ratio, padding = preprocessor.letterbox(
            image, new_shape=(640, 640), auto=False, scale_fill=True
        )

        assert result.shape[0] == 640
        assert result.shape[1] == 640
        # No padding in scale_fill mode
        assert padding == (0.0, 0.0)

    def test_letterbox_no_scaleup(self) -> None:
        """scaleup=False prevents enlarging small images."""
        preprocessor = TwitterScreenshotPreprocessor()
        # Small image
        image = np.zeros((50, 50, 3), dtype=np.uint8)

        _result, ratio, _padding = preprocessor.letterbox(
            image, new_shape=(640, 640), auto=False, scaleup=False
        )

        # Ratio should be capped at 1.0
        assert ratio[0] == 1.0
        assert ratio[1] == 1.0


# --- create_dataset_yaml Tests ---


class TestCreateDatasetYaml:
    """Tests for create_dataset_yaml function."""

    def test_create_dataset_yaml_content(self, tmp_path: Path) -> None:
        """YAML contains correct structure and values."""
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir()
        output_path = tmp_path / "data.yaml"
        class_names = ["tweet", "retweet", "quote"]

        create_dataset_yaml(dataset_path, class_names, output_path)

        with open(output_path) as f:
            content = yaml.safe_load(f)

        assert content["path"] == str(dataset_path.absolute())
        assert content["train"] == "train/images"
        assert content["val"] == "val/images"
        assert content["test"] == "test/images"
        assert content["nc"] == 3
        assert content["names"] == class_names

    def test_create_dataset_yaml_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Parent directories are created if they don't exist."""
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir()
        output_path = tmp_path / "nested" / "path" / "data.yaml"

        result = create_dataset_yaml(dataset_path, ["class1"], output_path)

        assert Path(result).exists()
        assert output_path.parent.exists()


# --- check_class_imbalance Tests ---


class TestCheckClassImbalance:
    """Tests for check_class_imbalance function."""

    def test_check_class_imbalance_empty(self) -> None:
        """Returns warning for empty distribution."""
        warnings = check_class_imbalance({})

        assert len(warnings) == 1
        assert "No annotations found" in warnings[0]

    def test_check_class_imbalance_balanced(self) -> None:
        """No warnings when classes are balanced."""
        distribution = {"class_a": 100, "class_b": 100, "class_c": 100}

        warnings = check_class_imbalance(distribution, threshold=0.1)

        assert len(warnings) == 0

    def test_check_class_imbalance_detects_underrepresented(self) -> None:
        """Flags classes below threshold."""
        distribution = {"common": 900, "rare": 50, "very_rare": 50}

        warnings = check_class_imbalance(distribution, threshold=0.1)

        assert len(warnings) == 2
        assert any("rare" in w for w in warnings)
        assert any("very_rare" in w for w in warnings)
        assert any("underrepresented" in w for w in warnings)

    def test_check_class_imbalance_threshold(self) -> None:
        """Custom threshold is respected."""
        distribution = {"class_a": 80, "class_b": 20}

        # With 20% threshold, class_b is fine
        warnings = check_class_imbalance(distribution, threshold=0.2)
        assert len(warnings) == 0

        # With 25% threshold, class_b is underrepresented
        warnings = check_class_imbalance(distribution, threshold=0.25)
        assert len(warnings) == 1
