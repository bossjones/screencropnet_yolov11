"""Label Studio ML backend that pre-predicts the outer tweet bounding box.

Loads the existing EfficientNet-B0 ``ScreenCropNetV1_378_epochs.pth`` checkpoint
and, for each task, returns a single ``tweet_region`` rectangle the annotator only
needs to nudge.

Run (from this directory)::

    uvx --from label-studio-ml --with torch --with timm --with albumentations \
        --with opencv-python-headless label-studio-ml start . --port 9090

Environment:
    CHECKPOINT_PATH  Path to the .pth checkpoint
                     (default: <repo>/scratch/checkpoints/screencropnet_efficientnet_b0_378.pth)
    DEVICE           torch device (default: auto — cuda > mps > cpu). GPU requires
                     a native run (mps on Apple Silicon) or a Linux+NVIDIA host;
                     Docker-on-macOS is CPU-only (no Metal/CUDA passthrough).
    MODEL_INPUT_SIZE Square resize edge fed to the model (default: 224)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

from arch import ObjLocModel
from geometry import build_prediction

LABEL = "tweet_region"
FROM_NAME = "label"
TO_NAME = "image"
MODEL_VERSION = "screencropnet_efficientnet_b0_378"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CKPT = _REPO_ROOT / "scratch" / "checkpoints" / "screencropnet_efficientnet_b0_378.pth"


def _select_device() -> str:
    override = os.environ.get("DEVICE")
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TweetRegionModel(LabelStudioMLBase):
    """Predicts one ``tweet_region`` rectangle per image using EfficientNet-B0."""

    def setup(self) -> None:
        self.set("model_version", MODEL_VERSION)
        self.device = _select_device()
        self.input_size = int(os.environ.get("MODEL_INPUT_SIZE", "224"))
        ckpt = Path(os.environ.get("CHECKPOINT_PATH", str(_DEFAULT_CKPT)))
        self.model = ObjLocModel(pretrained=False)
        self.model.load_state_dict(
            torch.load(ckpt, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()
        self.transform = A.Compose([A.Resize(self.input_size, self.input_size), A.Normalize()])

    def _predict_one(self, image_path: str) -> dict[str, Any] | None:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]

        tensor = self.transform(image=image)["image"]
        tensor = torch.from_numpy(tensor).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
        x1, y1, x2, y2 = (float(v) for v in np.asarray(out.detach().cpu()).reshape(-1)[:4])

        return build_prediction(
            (x1, y1, x2, y2),
            model_input_size=float(self.input_size),
            img_w=float(img_w),
            img_h=float(img_h),
            from_name=FROM_NAME,
            to_name=TO_NAME,
            label=LABEL,
            model_version=MODEL_VERSION,
            score=0.9,
        )

    def predict(
        self, tasks: list[dict[str, Any]], context: dict[str, Any] | None = None, **kwargs: Any
    ) -> ModelResponse:
        predictions: list[dict[str, Any]] = []
        for task in tasks:
            try:
                image_url = task["data"].get("image")
                local_path = self.get_local_path(image_url, task_id=task.get("id"))
                prediction = self._predict_one(local_path)
            except Exception:  # noqa: BLE001 - one bad image must not crash the batch
                prediction = None
            predictions.append(prediction or {"model_version": MODEL_VERSION, "result": []})
        return ModelResponse(predictions=predictions)
