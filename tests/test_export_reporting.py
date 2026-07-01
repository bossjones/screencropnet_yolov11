"""Tests for ModelExporter reporting: pytorch-first and fail-soft non-pytorch formats."""

from __future__ import annotations

from pathlib import Path

from pytest_mock import MockerFixture

from screencropnet_yolo.model import ModelExporter


def _fake_weights(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake torch weights")
    return path


class TestPyTorchReporting:
    def test_pytorch_reported_first_with_real_path(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        src = _fake_weights(tmp_path / "train" / "weights" / "best.pt")
        out = tmp_path / "out"
        model = mocker.Mock()
        model.export.return_value = str(out / "best.onnx")

        exporter = ModelExporter(model, str(out), source_weights=str(src))
        # pytorch listed LAST, but must be reported FIRST.
        result = exporter.export(["onnx", "pytorch"])

        assert list(result.keys())[0] == "pytorch"
        assert Path(result["pytorch"]).exists()

    def test_pytorch_included_even_without_guessed_best_pt(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        # The old code gated pytorch on a non-existent {output_dir}/best.pt.
        src = _fake_weights(tmp_path / "real" / "best.pt")
        out = tmp_path / "out"
        model = mocker.Mock()

        exporter = ModelExporter(model, str(out), source_weights=str(src))
        result = exporter.export(["pytorch"])

        assert "pytorch" in result
        assert Path(result["pytorch"]).exists()


class TestFailSoft:
    def test_onnx_failure_warns_and_degrades(self, tmp_path: Path, mocker: MockerFixture) -> None:
        src = _fake_weights(tmp_path / "best.pt")
        out = tmp_path / "out"
        model = mocker.Mock()
        model.export.side_effect = RuntimeError("onnx export blew up")
        warn = mocker.patch("screencropnet_yolo.model.logger.warning")

        exporter = ModelExporter(model, str(out), source_weights=str(src))
        # Must not raise.
        result = exporter.export(["pytorch", "onnx"])

        assert "pytorch" in result
        assert "onnx" not in result
        assert warn.called
