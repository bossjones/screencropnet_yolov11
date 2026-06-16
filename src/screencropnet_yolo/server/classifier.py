"""Twitter/not classification via the reference ScreenNet model.

Torch and torchvision are imported only inside :meth:`ScreenNetClassifier.load_model`
and :meth:`ScreenNetClassifier.infer`, behind the :class:`Classifier` protocol, so
the API and the entire unit-test suite run without torch, weights, or a GPU.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Protocol

from PIL import Image

from screencropnet_yolo.server.config import Settings, pick_device


class Classifier(Protocol):
    def infer(self, image: Image.Image) -> list[dict[str, object]]: ...


def is_twitter(result: list[dict[str, object]]) -> bool:
    return bool(result) and result[0]["pred_class"] == "twitter"


class ScreenNetClassifier:
    """EfficientNet-B0 head over classes ``[facebook, tiktok, twitter]``."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._class_names = list(settings.class_names)
        self._weights_path = settings.weights_path
        self._device = pick_device(settings.device_preference)
        # Torch objects; typed Any so this module needs no torch type stubs.
        self._model: Any = None
        self._transforms: Any = None

    def load_model(self) -> None:
        import torch
        from torchvision import models as tvm

        weights = tvm.EfficientNet_B0_Weights.DEFAULT
        self._transforms = weights.transforms()
        model = tvm.efficientnet_b0(weights=weights)
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(in_features=1280, out_features=len(self._class_names), bias=True),
        )
        # Trusted local checkpoint. torch 2.9 defaults weights_only=True, which
        # rejects this pickled EfficientNet-B0 state; weights_only=False is safe
        # for our own file.
        state = torch.load(self._weights_path, map_location=self._device, weights_only=False)
        # Some checkpoints wrap the params under a "state_dict"/"model_state_dict"
        # key rather than being a bare state_dict.
        if isinstance(state, dict):
            state = state.get("state_dict") or state.get("model_state_dict") or state
        try:
            model.load_state_dict(state)
        except (RuntimeError, KeyError) as exc:
            raise RuntimeError(
                f"checkpoint format mismatch loading {self._weights_path}; "
                f"expected an EfficientNet-B0 3-class state_dict ({exc})"
            ) from exc
        model.eval()
        model.to(self._device)
        torch.set_num_threads(1)
        self._model = model

    def infer(self, image: Image.Image) -> list[dict[str, object]]:
        import torch

        if self._model is None or self._transforms is None:
            raise RuntimeError("load_model() must be called before infer()")
        # EfficientNet transforms expect 3-channel RGB; screenshots can be RGBA,
        # palette (P), grayscale (L/LA), or CMYK, so normalize any non-RGB mode.
        if image.mode != "RGB":
            image = image.convert("RGB")

        start = perf_counter()
        tensor = self._transforms(image).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)
            prob, index = torch.max(probs, dim=1)
        pred_class = self._class_names[int(index.item())]
        elapsed = perf_counter() - start
        return [
            {
                "pred_prob": round(float(prob.item()), 4),
                "pred_class": pred_class,
                "time_for_pred": round(elapsed, 4),
            }
        ]


class FakeClassifier:
    """Deterministic, torch-free classifier for tests and integration runs."""

    def __init__(self, pred_class: str = "twitter", pred_prob: float = 0.99) -> None:
        self._pred_class = pred_class
        self._pred_prob = pred_prob

    def infer(self, image: Image.Image) -> list[dict[str, object]]:
        return [
            {
                "pred_prob": self._pred_prob,
                "pred_class": self._pred_class,
                "time_for_pred": 0.0,
            }
        ]
