"""End-to-end test of the real ScreenNet model on known Twitter screenshots.

Skip-guarded so ``make test`` (and CI) stay green without torch or weights:
``importorskip("torch")`` plus a weights-exist / dataset-exist skip. Run it with
``make test-e2e`` after ``make download-weights``.
"""

from __future__ import annotations

from typing import cast

import pytest
from PIL import Image

from screencropnet_yolo.server.classifier import ScreenNetClassifier, is_twitter
from screencropnet_yolo.server.config import get_settings

pytestmark = [pytest.mark.integration, pytest.mark.e2e]

K = 10
MIN_TWITTER_FRACTION = 0.8


def test_real_model_classifies_twitter_screenshots() -> None:
    pytest.importorskip("torch")
    settings = get_settings()
    if not settings.weights_path.exists():
        pytest.skip(f"weights not found at {settings.weights_path}; run `make download-weights`")
    if not settings.raw_dataset_dir.is_dir():
        pytest.skip(f"dataset dir not found: {settings.raw_dataset_dir}")

    images = sorted(settings.raw_dataset_dir.glob("*_twitter.PNG"))[:K]
    if not images:
        pytest.skip(f"no *_twitter.PNG screenshots under {settings.raw_dataset_dir}")

    classifier = ScreenNetClassifier(settings)
    classifier.load_model()

    twitter_hits = 0
    for path in images:
        with Image.open(path) as image:
            result = classifier.infer(image)
        first = result[0]
        if set(first) != {"pred_prob", "pred_class", "time_for_pred"}:
            raise AssertionError(f"unexpected result keys for {path.name}: {sorted(first)}")
        if first["pred_class"] not in settings.class_names:
            raise AssertionError(
                f"pred_class {first['pred_class']!r} not in {settings.class_names}"
            )
        prob = cast(float, first["pred_prob"])
        if not 0.0 <= prob <= 1.0:
            raise AssertionError(f"pred_prob {prob} out of range for {path.name}")
        if is_twitter(result):
            twitter_hits += 1

    fraction = twitter_hits / len(images)
    if fraction < MIN_TWITTER_FRACTION:
        raise AssertionError(
            f"only {twitter_hits}/{len(images)} ({fraction:.0%}) classified as twitter; "
            f"expected ≥ {MIN_TWITTER_FRACTION:.0%}"
        )
