"""Pins for the training config schema after the single-class YOLO26 migration."""

from __future__ import annotations

from importlib.resources import files
from typing import Any

import yaml


def _load_config() -> dict[str, Any]:
    cfg = files("screencropnet_yolo").joinpath("config", "config.yaml")
    return yaml.safe_load(cfg.read_text())


def test_config_yaml_has_single_class_tweet_region() -> None:
    """The dataset is collapsed to one tweet_region class."""
    cfg = _load_config()
    assert cfg["model"]["class_names"] == ["tweet_region"]
    assert cfg["model"]["num_classes"] == 1


def test_config_yaml_experiment_name_is_twitter_yolo26() -> None:
    """The experiment name reflects the YOLO26 migration."""
    cfg = _load_config()
    assert cfg["logging"]["experiment_name"] == "twitter_yolo26"


def test_config_yaml_roboflow_format_yolov11_preserved() -> None:
    """Roboflow export format stays 'yolov11' (label format is identical for v26)."""
    cfg = _load_config()
    assert cfg["dataset"]["roboflow"]["format"] == "yolov11"
