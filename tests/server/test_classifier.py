from __future__ import annotations

import contextlib
import sys
import types

from PIL import Image
from pytest_mock import MockerFixture

from screencropnet_yolo.server.classifier import (
    FakeClassifier,
    ScreenNetClassifier,
    is_twitter,
)
from screencropnet_yolo.server.config import Settings


class _Value:
    def __init__(self, value: object) -> None:
        self._value = value

    def item(self) -> object:
        return self._value


class _Chain:
    def unsqueeze(self, dim: int) -> _Chain:
        return self

    def to(self, device: object) -> _Chain:
        return self


def _fake_torch() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        inference_mode=contextlib.nullcontext,
        softmax=lambda logits, dim: logits,
        max=lambda probs, dim: (_Value(0.8765), _Value(2)),
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
    )


def test_is_twitter_helper() -> None:
    if not is_twitter([{"pred_class": "twitter", "pred_prob": 0.9, "time_for_pred": 0.1}]):
        raise AssertionError("twitter prediction should be detected")
    if is_twitter([{"pred_class": "tiktok", "pred_prob": 0.9, "time_for_pred": 0.1}]):
        raise AssertionError("non-twitter prediction must be False")
    if is_twitter([]):
        raise AssertionError("empty result must be False")


def test_fake_classifier_is_deterministic() -> None:
    fake = FakeClassifier()
    result = fake.infer(Image.new("RGB", (4, 4)))
    if result[0]["pred_class"] != "twitter" or result[0]["pred_prob"] != 0.99:
        raise AssertionError("default FakeClassifier should predict twitter")
    if "time_for_pred" not in result[0]:
        raise AssertionError("result must include time_for_pred")
    if not is_twitter(result):
        raise AssertionError("default fake should be twitter-positive")

    other = FakeClassifier(pred_class="tiktok")
    if is_twitter(other.infer(Image.new("RGB", (4, 4)))):
        raise AssertionError("non-twitter fake must not be twitter-positive")


def test_screennet_does_not_load_weights_in_init(mocker: MockerFixture) -> None:
    fake_torch = mocker.MagicMock()
    mocker.patch.dict(sys.modules, {"torch": fake_torch})
    classifier = ScreenNetClassifier(Settings())
    if classifier._model is not None:  # pyright: ignore[reportPrivateUsage]
        raise AssertionError("the model must not be loaded in __init__")
    if fake_torch.load.called:
        raise AssertionError("torch.load must not be called until load_model() runs")


def test_load_model_invokes_torch_load(mocker: MockerFixture) -> None:
    fake_torch = mocker.MagicMock()
    fake_tv = mocker.MagicMock()
    mocker.patch.dict(
        sys.modules,
        {"torch": fake_torch, "torchvision": fake_tv, "torchvision.models": fake_tv.models},
    )
    classifier = ScreenNetClassifier(Settings())
    classifier.load_model()
    if not fake_torch.load.called:
        raise AssertionError("load_model() must load the weights via torch.load")
    if classifier._model is None:  # pyright: ignore[reportPrivateUsage]
        raise AssertionError("load_model() must populate the model")


def test_infer_shape_and_rgba_to_rgb(mocker: MockerFixture) -> None:
    mocker.patch.dict(sys.modules, {"torch": _fake_torch()})
    classifier = ScreenNetClassifier(Settings())
    classifier._device = "cpu"  # pyright: ignore[reportPrivateUsage]
    classifier._class_names = ["facebook", "tiktok", "twitter"]  # pyright: ignore[reportPrivateUsage]
    classifier._transforms = lambda image: _Chain()  # pyright: ignore[reportPrivateUsage]
    classifier._model = lambda tensor: "logits"  # pyright: ignore[reportPrivateUsage]

    convert_spy = mocker.spy(Image.Image, "convert")
    result = classifier.infer(Image.new("RGBA", (8, 8)))

    if set(result[0]) != {"pred_prob", "pred_class", "time_for_pred"}:
        raise AssertionError("infer must return exactly the three documented keys")
    if result[0]["pred_class"] != "twitter" or result[0]["pred_prob"] != 0.8765:
        raise AssertionError("infer must map argmax index to the class name and prob")
    convert_calls = [c for c in convert_spy.call_args_list if c.args[1:] == ("RGB",)]
    if not convert_calls:
        raise AssertionError("RGBA images must be converted to RGB")
