"""Tests for the output presentation helpers."""

from __future__ import annotations

import logging

import pytest

from screencropnet_yolo.output import (
    Artifact,
    Color,
    ColorFormatter,
    colorize,
    format_artifacts_table,
    format_run_summary,
    human_size,
)


class TestHumanSize:
    def test_zero(self) -> None:
        assert human_size(0) == "0 B"

    def test_bytes(self) -> None:
        assert human_size(512) == "512 B"

    def test_kilobytes(self) -> None:
        assert human_size(1536) == "1.5 KB"

    def test_exact_kilobyte(self) -> None:
        assert human_size(1024) == "1.0 KB"

    def test_megabyte_rollover(self) -> None:
        assert human_size(1024 * 1024) == "1.0 MB"

    def test_gigabyte_rollover(self) -> None:
        assert human_size(1024 * 1024 * 1024) == "1.0 GB"

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            human_size(-1)


class TestColorize:
    def test_disabled_returns_unchanged(self) -> None:
        assert colorize("x", Color.GREEN, enabled=False) == "x"

    def test_enabled_wraps_with_code_and_reset(self) -> None:
        result = colorize("x", Color.GREEN, enabled=True)
        assert result.startswith(Color.GREEN)
        assert result.endswith(Color.RESET)
        assert "x" in result

    def test_empty_color_returns_unchanged_even_when_enabled(self) -> None:
        assert colorize("x", "", enabled=True) == "x"


class TestFormatRunSummary:
    def _summary(self, *, enabled: bool = False) -> str:
        return format_run_summary(
            model_size="n",
            arch="YOLO26n",
            device="mps",
            epochs=100,
            batch=4,
            imgsz=320,
            dataset_path="/data/twitter/data.yaml",
            output_dir="/runs/smoke",
            weights_dir="/runs/smoke/train/weights",
            best_pt="/runs/smoke/train/weights/best.pt",
            export_formats=["pytorch", "onnx"],
            enabled=enabled,
        )

    def test_contains_epoch_count(self) -> None:
        assert "100" in self._summary()

    def test_contains_model_and_device(self) -> None:
        summary = self._summary()
        assert "YOLO26n" in summary
        assert "mps" in summary

    def test_contains_paths(self) -> None:
        summary = self._summary()
        assert "/data/twitter/data.yaml" in summary
        assert "/runs/smoke" in summary

    def test_contains_export_formats(self) -> None:
        summary = self._summary()
        assert "pytorch" in summary
        assert "onnx" in summary

    def test_disabled_has_no_ansi(self) -> None:
        assert "\033[" not in self._summary(enabled=False)

    def test_enabled_has_ansi(self) -> None:
        assert "\033[" in self._summary(enabled=True)


class TestFormatArtifactsTable:
    def test_one_row_per_artifact_with_path_and_size(self) -> None:
        rows = [
            Artifact(label="best.pt", path="/runs/smoke/best.pt", size=2048),
            Artifact(label="onnx", path="/runs/smoke/best.onnx", size=4096),
        ]
        table = format_artifacts_table(rows)
        assert "/runs/smoke/best.pt" in table
        assert "/runs/smoke/best.onnx" in table
        assert human_size(2048) in table
        assert human_size(4096) in table

    def test_missing_artifact_labeled_not_crashing(self) -> None:
        rows = [Artifact(label="onnx", path=None, size=None)]
        table = format_artifacts_table(rows)
        assert "onnx" in table
        assert "missing" in table.lower()

    def test_includes_best_epoch_and_map(self) -> None:
        rows = [Artifact(label="best.pt", path="/x/best.pt", size=1024)]
        table = format_artifacts_table(rows, best_epoch=42, best_map=0.873)
        assert "42" in table
        assert "0.873" in table

    def test_disabled_has_no_ansi(self) -> None:
        rows = [Artifact(label="best.pt", path="/x/best.pt", size=1024)]
        assert "\033[" not in format_artifacts_table(rows, enabled=False)


class TestColorFormatter:
    def _record(self, level: int = logging.INFO) -> logging.LogRecord:
        return logging.LogRecord(
            name="test",
            level=level,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )

    def test_disabled_emits_no_ansi(self) -> None:
        fmt = ColorFormatter("%(levelname)s | %(message)s", enabled=False)
        out = fmt.format(self._record())
        assert "\033[" not in out
        assert "INFO" in out

    def test_enabled_colors_levelname(self) -> None:
        fmt = ColorFormatter("%(levelname)s | %(message)s", enabled=True)
        out = fmt.format(self._record(logging.WARNING))
        assert "\033[" in out
        assert "hello" in out

    def test_enabled_does_not_mutate_record_levelname(self) -> None:
        record = self._record(logging.ERROR)
        fmt = ColorFormatter("%(levelname)s | %(message)s", enabled=True)
        fmt.format(record)
        assert record.levelname == "ERROR"
