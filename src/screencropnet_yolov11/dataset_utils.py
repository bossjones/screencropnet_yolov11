"""
Dataset utilities for YOLO 11 Twitter Screenshot Detection.

This module handles:
- Dataset validation and integrity checks
- Dataset statistics computation
- Train/val/test splitting
- Roboflow integration
- Data preprocessing for Twitter screenshots
"""

from __future__ import annotations

import logging
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import yaml
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class DatasetStats:
    """Container for dataset statistics."""

    total_images: int = 0
    total_annotations: int = 0
    images_per_split: dict[str, int] = field(default_factory=dict)
    class_distribution: dict[str, int] = field(default_factory=dict)
    bbox_stats: dict[str, Any] = field(default_factory=dict)
    image_sizes: list[tuple[int, int]] = field(default_factory=list)
    corrupt_images: list[str] = field(default_factory=list)
    missing_annotations: list[str] = field(default_factory=list)


class DatasetValidator:
    """Validates YOLO format datasets for integrity and completeness."""

    SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def __init__(self, dataset_path: str, class_names: list[str]):
        """
        Initialize dataset validator.

        Args:
            dataset_path: Root path to the dataset
            class_names: List of class names for validation
        """
        self.dataset_path = Path(dataset_path)
        self.class_names = class_names
        self.stats = DatasetStats()

    def validate(self) -> tuple[bool, DatasetStats, list[str]]:
        """
        Perform comprehensive dataset validation.

        Returns:
            Tuple of (is_valid, stats, error_messages)
        """
        errors = []

        # Check dataset structure
        structure_errors = self._validate_structure()
        errors.extend(structure_errors)

        # Validate images and annotations
        for split in ["train", "val", "test"]:
            split_path = self.dataset_path / split
            if split_path.exists():
                split_errors = self._validate_split(split)
                errors.extend(split_errors)

        # Compute statistics
        self._compute_statistics()

        is_valid = len(errors) == 0
        return is_valid, self.stats, errors

    def _validate_structure(self) -> list[str]:
        """Validate dataset directory structure."""
        errors = []

        if not self.dataset_path.exists():
            errors.append(f"Dataset path does not exist: {self.dataset_path}")
            return errors

        # Check for required directories
        required_dirs = ["train"]
        for dir_name in required_dirs:
            dir_path = self.dataset_path / dir_name
            if not dir_path.exists():
                errors.append(f"Required directory missing: {dir_path}")

        # Check for images and labels subdirectories
        for split in ["train", "val", "test"]:
            split_path = self.dataset_path / split
            if split_path.exists():
                images_path = split_path / "images"
                labels_path = split_path / "labels"

                if not images_path.exists() and not labels_path.exists():
                    # Flat structure - images and labels in same directory
                    pass
                elif not images_path.exists():
                    errors.append(f"Images directory missing in {split}")
                elif not labels_path.exists():
                    errors.append(f"Labels directory missing in {split}")

        return errors

    def _validate_split(self, split: str) -> list[str]:
        """Validate a specific split (train/val/test)."""
        errors = []
        split_path = self.dataset_path / split

        # Determine structure
        images_path = split_path / "images"
        labels_path = split_path / "labels"

        if images_path.exists():
            image_dir = images_path
            label_dir = labels_path
        else:
            image_dir = split_path
            label_dir = split_path

        # Get all images
        images = self._get_image_files(image_dir)
        self.stats.images_per_split[split] = len(images)

        # Validate each image and its annotation
        for image_path in images:
            # Check image integrity
            if not self._is_valid_image(image_path):
                self.stats.corrupt_images.append(str(image_path))
                errors.append(f"Corrupt image: {image_path}")
                continue

            # Get image size
            try:
                img = Image.open(image_path)
                self.stats.image_sizes.append(img.size)
            except Exception:
                pass

            # Check for corresponding annotation
            label_path = self._get_label_path(image_path, label_dir)
            if not label_path.exists():
                self.stats.missing_annotations.append(str(image_path))
                errors.append(f"Missing annotation for: {image_path}")
                continue

            # Validate annotation format
            annotation_errors = self._validate_annotation(label_path)
            errors.extend(annotation_errors)

        return errors

    def _get_image_files(self, directory: Path) -> list[Path]:
        """Get all image files in a directory."""
        images = []
        for ext in self.SUPPORTED_IMAGE_FORMATS:
            images.extend(directory.glob(f"*{ext}"))
            images.extend(directory.glob(f"*{ext.upper()}"))
        return images

    def _is_valid_image(self, image_path: Path) -> bool:
        """Check if image file is valid and not corrupted."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return False
            return img.shape[0] > 0 and img.shape[1] > 0
        except Exception:
            return False

    def _get_label_path(self, image_path: Path, label_dir: Path) -> Path:
        """Get the corresponding label file path for an image."""
        return label_dir / f"{image_path.stem}.txt"

    def _validate_annotation(self, label_path: Path) -> list[str]:
        """Validate YOLO format annotation file."""
        errors = []

        try:
            with open(label_path) as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    errors.append(f"Invalid annotation format in {label_path}, line {line_num}")
                    continue

                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Validate class ID
                    if class_id < 0 or class_id >= len(self.class_names):
                        errors.append(
                            f"Invalid class ID {class_id} in {label_path}, line {line_num}"
                        )
                    else:
                        # Update class distribution
                        class_name = self.class_names[class_id]
                        self.stats.class_distribution[class_name] = (
                            self.stats.class_distribution.get(class_name, 0) + 1
                        )

                    # Validate coordinates (should be normalized 0-1)
                    for val, name in [
                        (x_center, "x_center"),
                        (y_center, "y_center"),
                        (width, "width"),
                        (height, "height"),
                    ]:
                        if val < 0 or val > 1:
                            errors.append(
                                f"Invalid {name} value {val} in {label_path}, line {line_num}"
                            )

                    self.stats.total_annotations += 1

                except ValueError:
                    errors.append(f"Invalid number format in {label_path}, line {line_num}")

        except Exception as e:
            errors.append(f"Error reading {label_path}: {str(e)}")

        return errors

    def _compute_statistics(self) -> None:
        """Compute aggregate dataset statistics."""
        self.stats.total_images = sum(self.stats.images_per_split.values())

        if self.stats.image_sizes:
            widths = [s[0] for s in self.stats.image_sizes]
            heights = [s[1] for s in self.stats.image_sizes]

            self.stats.bbox_stats = {
                "avg_width": np.mean(widths),
                "avg_height": np.mean(heights),
                "min_width": min(widths),
                "max_width": max(widths),
                "min_height": min(heights),
                "max_height": max(heights),
            }


class DatasetSplitter:
    """Handles automatic train/val/test splitting of datasets."""

    def __init__(
        self,
        source_path: str,
        output_path: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        """
        Initialize dataset splitter.

        Args:
            source_path: Path to source images and labels
            output_path: Path for split dataset output
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            seed: Random seed for reproducibility
        """
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    def split(self) -> dict[str, int]:
        """
        Perform the dataset split.

        Returns:
            Dictionary with counts per split
        """
        random.seed(self.seed)

        # Find all images
        images = []
        for ext in DatasetValidator.SUPPORTED_IMAGE_FORMATS:
            images.extend(self.source_path.glob(f"**/*{ext}"))

        # Shuffle images
        random.shuffle(images)

        # Calculate split indices
        n = len(images)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        # Create output directories and copy files
        counts = {}
        for split_name, split_images in splits.items():
            split_dir = self.output_path / split_name
            images_dir = split_dir / "images"
            labels_dir = split_dir / "labels"

            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            for img_path in split_images:
                # Copy image
                shutil.copy2(img_path, images_dir / img_path.name)

                # Copy label if exists
                label_path = img_path.with_suffix(".txt")
                if not label_path.exists():
                    label_path = img_path.parent / "labels" / f"{img_path.stem}.txt"

                if label_path.exists():
                    shutil.copy2(label_path, labels_dir / f"{img_path.stem}.txt")

            counts[split_name] = len(split_images)
            logger.info(f"Created {split_name} split with {counts[split_name]} images")

        return counts


class RoboflowLoader:
    """Handles loading datasets from Roboflow."""

    def __init__(
        self,
        api_key: str,
        workspace: str,
        project: str,
        version: int,
        output_path: str,
        format: str = "yolov11",
    ):
        """
        Initialize Roboflow loader.

        Args:
            api_key: Roboflow API key
            workspace: Roboflow workspace name
            project: Project name
            version: Dataset version
            output_path: Path to save downloaded dataset
            format: Export format
        """
        self.api_key = api_key
        self.workspace = workspace
        self.project = project
        self.version = version
        self.output_path = Path(output_path)
        self.format = format

    def download(self) -> Path:
        """
        Download dataset from Roboflow.

        Returns:
            Path to downloaded dataset
        """
        try:
            from roboflow import Roboflow  # pyright: ignore[reportMissingImports]
        except ImportError as err:
            raise ImportError(
                "roboflow package not installed. Install with: pip install roboflow"
            ) from err

        logger.info(f"Downloading dataset from Roboflow: {self.workspace}/{self.project}")

        rf = Roboflow(api_key=self.api_key)
        project = rf.workspace(self.workspace).project(self.project)
        project.version(self.version).download(self.format, location=str(self.output_path))

        logger.info(f"Dataset downloaded to: {self.output_path}")
        return self.output_path


class TwitterScreenshotPreprocessor:
    """
    Preprocessing utilities specific to Twitter screenshots.

    Handles common issues like:
    - Screenshot artifacts and compression
    - Color space normalization
    - High aspect ratio text regions
    """

    def __init__(self, target_size: int = 640):
        """
        Initialize preprocessor.

        Args:
            target_size: Target image size for preprocessing
        """
        self.target_size = target_size

    def preprocess(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """
        Apply preprocessing pipeline to a Twitter screenshot.

        Args:
            image: Input image as numpy array (BGR)

        Returns:
            Preprocessed image
        """
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Remove compression artifacts
        image = self._reduce_compression_artifacts(image)

        # Normalize color space
        image = self._normalize_colors(image)

        # Convert back to BGR for OpenCV compatibility
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image

    def _reduce_compression_artifacts(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Apply mild denoising to reduce JPEG compression artifacts."""
        return cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)

    def _normalize_colors(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Normalize image colors for consistent processing."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        # Convert back to RGB
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def letterbox(
        self,
        image: npt.NDArray[np.uint8],
        new_shape: tuple[int, int] = (640, 640),
        color: tuple[int, int, int] = (114, 114, 114),
        auto: bool = True,
        scale_fill: bool = False,
        scaleup: bool = True,
    ) -> tuple[npt.NDArray[np.uint8], tuple[float, float], tuple[int, int]]:
        """
        Resize and pad image while maintaining aspect ratio.

        Args:
            image: Input image
            new_shape: Target shape
            color: Padding color
            auto: Minimum rectangle padding
            scale_fill: Stretch to fill
            scaleup: Allow scaling up

        Returns:
            Tuple of (resized_image, ratio, padding)
        """
        shape = image.shape[:2]  # Current shape [height, width]

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # Only scale down
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

        if auto:  # Minimum rectangle
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)
        elif scale_fill:  # Stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

        dw /= 2  # Divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # Resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )

        return image, ratio, (dw, dh)


def create_dataset_yaml(
    dataset_path: str | Path, class_names: list[str], output_path: str | Path
) -> str:
    """
    Create YOLO format dataset.yaml file.

    Args:
        dataset_path: Path to dataset root
        class_names: List of class names
        output_path: Path to save yaml file

    Returns:
        Path to created yaml file
    """
    dataset_path = Path(dataset_path)

    yaml_content = {
        "path": str(dataset_path.absolute()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    logger.info(f"Created dataset YAML at: {output_path}")
    return str(output_path)


def display_dataset_stats(stats: DatasetStats) -> None:
    """Display formatted dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    print(f"\nTotal Images: {stats.total_images}")
    print(f"Total Annotations: {stats.total_annotations}")

    print("\nImages per Split:")
    for split, count in stats.images_per_split.items():
        print(f"  {split}: {count}")

    print("\nClass Distribution:")
    for class_name, count in sorted(
        stats.class_distribution.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {class_name}: {count}")

    if stats.bbox_stats:
        print("\nImage Size Statistics:")
        print(
            f"  Average: {stats.bbox_stats['avg_width']:.0f} x {stats.bbox_stats['avg_height']:.0f}"
        )
        print(f"  Min: {stats.bbox_stats['min_width']} x {stats.bbox_stats['min_height']}")
        print(f"  Max: {stats.bbox_stats['max_width']} x {stats.bbox_stats['max_height']}")

    if stats.corrupt_images:
        print(f"\nCorrupt Images: {len(stats.corrupt_images)}")

    if stats.missing_annotations:
        print(f"Missing Annotations: {len(stats.missing_annotations)}")

    print("=" * 60 + "\n")


def check_class_imbalance(class_distribution: dict[str, int], threshold: float = 0.1) -> list[str]:
    """
    Check for class imbalance issues.

    Args:
        class_distribution: Dictionary of class counts
        threshold: Minimum proportion threshold

    Returns:
        List of warning messages
    """
    warnings = []
    total = sum(class_distribution.values())

    if total == 0:
        return ["No annotations found in dataset"]

    for class_name, count in class_distribution.items():
        proportion = count / total
        if proportion < threshold:
            warnings.append(
                f"Class '{class_name}' is underrepresented ({count} instances, "
                f"{proportion:.1%} of total). Consider data augmentation or oversampling."
            )

    return warnings
