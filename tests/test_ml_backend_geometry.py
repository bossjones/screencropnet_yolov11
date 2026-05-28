"""Tests for the ML backend's pure coordinate math."""

from __future__ import annotations

import pytest

from tools.labeling.ml_backend.geometry import build_prediction, rescale_xyxy, xyxy_to_ls_value


def test_rescale_xyxy_maps_model_space_to_original() -> None:
    assert rescale_xyxy((10, 20, 30, 40), src_size=100, dst_w=200, dst_h=400) == (
        20.0,
        80.0,
        60.0,
        160.0,
    )


def test_build_prediction_returns_ls_rectangle_schema() -> None:
    prediction = build_prediction(
        (100.0, 200.0, 500.0, 700.0),
        model_input_size=1000.0,
        img_w=1000.0,
        img_h=1000.0,
        from_name="label",
        to_name="image",
        label="tweet_region",
        model_version="screencropnet_efficientnet_b0_378",
        score=0.9,
    )

    assert prediction["model_version"] == "screencropnet_efficientnet_b0_378"
    assert prediction["score"] == pytest.approx(0.9)
    result = prediction["result"][0]
    assert result["type"] == "rectanglelabels"
    assert result["from_name"] == "label"
    assert result["to_name"] == "image"

    value = result["value"]
    assert value["x"] == pytest.approx(10.0)
    assert value["y"] == pytest.approx(20.0)
    assert value["width"] == pytest.approx(40.0)
    assert value["height"] == pytest.approx(50.0)
    assert value["rectanglelabels"] == ["tweet_region"]


def test_xyxy_to_ls_value_clamps_to_image_bounds() -> None:
    value = xyxy_to_ls_value(
        (50.0, 50.0, 200.0, 200.0), img_w=100.0, img_h=100.0, label="tweet_region"
    )

    assert value["x"] == pytest.approx(50.0)
    assert value["y"] == pytest.approx(50.0)
    assert value["width"] == pytest.approx(50.0)
    assert value["height"] == pytest.approx(50.0)
    assert 0.0 <= value["x"] <= 100.0
    assert value["x"] + value["width"] <= 100.0 + 1e-9
    assert value["y"] + value["height"] <= 100.0 + 1e-9
