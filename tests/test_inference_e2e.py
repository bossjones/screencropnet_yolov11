"""End-to-end YOLO26 inference test.

Deselected from the default suite (``-m "not integration"`` in pyproject). Run
explicitly with ``uv run pytest -m integration``; it downloads ``yolo26n.pt``
and requires network access.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from screencropnet_yolo.inference import InferencePipeline


@pytest.mark.integration
def test_real_yolo26_predicts_on_sample_image(tmp_path: Path) -> None:
    """A real yolo26n.pt model loads and runs predict() end to end."""
    import cv2

    sample = tmp_path / "sample.png"
    cv2.imwrite(str(sample), np.full((640, 640, 3), 127, dtype=np.uint8))

    pipeline = InferencePipeline(
        model_path="yolo26n.pt",
        class_names=["tweet_region"],
        device="cpu",
    )
    result = pipeline.predict_image(str(sample))

    # A blank image may yield zero detections; the contract is that the call
    # completes and returns a well-formed result with the v26 Results schema.
    assert result.image_size == (640, 640)
    assert isinstance(result.detections, list)
