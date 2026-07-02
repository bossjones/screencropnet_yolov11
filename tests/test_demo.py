"""Tests for the demo CLI: async orchestration, display, and the CLI shell.

Pure helpers (discover/sample/resolve/tile) are covered by inline ``## Tests`` in
``screencropnet_yolo.demo``; this file exercises the pieces that need mocks: the
blocking annotate step, the async fan-out, the macOS display shim, and ``main``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pytest_mock import MockerFixture

from screencropnet_yolo import demo


def _fake_result(num_detections: int = 0) -> object:
    """A stand-in for InferenceResult with just the attributes demo touches."""
    return type("R", (), {"detections": list(range(num_detections))})()


class TestAnnotateOne:
    def test_writes_annotated_copy(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """annotate_one infers, draws, and writes a copy into out_dir."""
        fake_cv2 = mocker.patch("screencropnet_yolo.demo.cv2")
        fake_cv2.imread.return_value = np.zeros((8, 8, 3), dtype=np.uint8)
        fake_cv2.imwrite.return_value = True

        pipeline = mocker.MagicMock()
        pipeline.predict_image.return_value = _fake_result(2)
        pipeline.draw_detections.return_value = np.zeros((8, 8, 3), dtype=np.uint8)

        src = tmp_path / "shot.png"
        dest, result = demo.annotate_one(pipeline, src, tmp_path)

        assert dest.parent == tmp_path
        assert dest.name == "shot_annotated.png"
        assert result.detections == [0, 1]
        fake_cv2.imwrite.assert_called_once()

    def test_raises_on_unreadable_image(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """A None from imread (unreadable file) raises so run_demo can skip it."""
        fake_cv2 = mocker.patch("screencropnet_yolo.demo.cv2")
        fake_cv2.imread.return_value = None

        with pytest.raises(ValueError, match="Failed to read"):
            demo.annotate_one(mocker.MagicMock(), tmp_path / "bad.png", tmp_path)


class TestRunDemo:
    async def test_one_copy_per_image(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """run_demo returns one (path, result) per input image."""

        def fake_annotate(
            _pipeline: object, src: Path, out_dir: Path, **_kwargs: object
        ) -> tuple[Path, object]:
            return out_dir / f"{src.stem}.png", _fake_result(1)

        mocker.patch("screencropnet_yolo.demo.annotate_one", side_effect=fake_annotate)

        images = [tmp_path / f"img{i}.png" for i in range(4)]
        results = await demo.run_demo(mocker.MagicMock(), images, tmp_path, concurrency=2)

        assert len(results) == 4
        assert (tmp_path / "annotated").is_dir()

    async def test_skips_failures(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """An image whose inference raises is dropped, not fatal."""

        def fake_annotate(
            _pipeline: object, src: Path, out_dir: Path, **_kwargs: object
        ) -> tuple[Path, object]:
            if src.stem == "boom":
                raise RuntimeError("inference blew up")
            return out_dir / f"{src.stem}.png", _fake_result(0)

        mocker.patch("screencropnet_yolo.demo.annotate_one", side_effect=fake_annotate)

        images = [tmp_path / "ok.png", tmp_path / "boom.png"]
        results = await demo.run_demo(mocker.MagicMock(), images, tmp_path, concurrency=2)

        assert len(results) == 1
        assert results[0][0].name == "ok.png"


class TestSelectModel:
    def test_returns_picked_path(self, tmp_path: Path) -> None:
        """select_model maps the chosen fzf line back to its exact Path."""
        first = tmp_path / "a.pt"
        second = tmp_path / "b.onnx"
        first.write_bytes(b"x")
        second.write_bytes(b"x")

        # Selector "picks" the second candidate's display line.
        def selector(choices: list[str]) -> list[str]:
            return [choices[1]]

        picked = demo.select_model([first, second], selector=selector)
        assert picked == second

    def test_returns_none_on_cancel(self, tmp_path: Path) -> None:
        """An empty selection (fzf ESC/cancel) yields None."""
        p = tmp_path / "a.pt"
        p.write_bytes(b"x")
        picked = demo.select_model([p], selector=lambda _choices: [])
        assert picked is None


class TestOpenPaths:
    def test_opens_on_macos(self, mocker: MockerFixture, tmp_path: Path) -> None:
        mocker.patch("screencropnet_yolo.demo.sys.platform", "darwin")
        popen = mocker.patch("screencropnet_yolo.demo.subprocess.Popen")

        p = tmp_path / "sheet.png"
        demo.open_paths([p], enabled=True)

        popen.assert_called_once()
        argv = popen.call_args.args[0]
        assert argv[0] == "open"
        assert str(p) in argv

    def test_skips_when_disabled(self, mocker: MockerFixture, tmp_path: Path) -> None:
        mocker.patch("screencropnet_yolo.demo.sys.platform", "darwin")
        popen = mocker.patch("screencropnet_yolo.demo.subprocess.Popen")
        demo.open_paths([tmp_path / "sheet.png"], enabled=False)
        popen.assert_not_called()

    def test_skips_on_non_macos(self, mocker: MockerFixture, tmp_path: Path) -> None:
        mocker.patch("screencropnet_yolo.demo.sys.platform", "linux")
        popen = mocker.patch("screencropnet_yolo.demo.subprocess.Popen")
        demo.open_paths([tmp_path / "sheet.png"], enabled=True)
        popen.assert_not_called()


class TestMain:
    def test_happy_path(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """main wires resolve→infer→montage→report and exits 0."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        (images_dir / "a.jpg").write_bytes(b"not-a-real-jpeg")
        out_dir = tmp_path / "out"

        mocker.patch("screencropnet_yolo.demo.InferencePipeline")
        mocker.patch("screencropnet_yolo.demo.resolve_device", return_value="cpu")
        mocker.patch(
            "screencropnet_yolo.demo.build_contact_sheet",
            return_value=np.zeros((10, 10, 3), dtype=np.uint8),
        )
        mocker.patch("screencropnet_yolo.demo.open_paths")

        async def fake_run_demo(*args: object, **kwargs: object) -> list[tuple[Path, object]]:
            return [(out_dir / "annotated" / "a.png", _fake_result(3))]

        mocker.patch("screencropnet_yolo.demo.run_demo", side_effect=fake_run_demo)

        rc = demo.main([str(images_dir), "-n", "1", "--no-open", "-o", str(out_dir)])

        assert rc == 0
        assert (out_dir / "contact_sheet.png").is_file()

    def test_empty_directory_returns_nonzero(self, tmp_path: Path) -> None:
        """No images → graceful non-zero exit, no crash."""
        empty = tmp_path / "empty"
        empty.mkdir()
        rc = demo.main([str(empty), "-o", str(tmp_path / "out")])
        assert rc == 1

    def test_select_flag_invokes_picker(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """--select routes through the fzf picker and loads the chosen model."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        (images_dir / "a.jpg").write_bytes(b"not-a-real-jpeg")
        out_dir = tmp_path / "out"

        runs = tmp_path / "runs"
        weights = runs / "run" / "weights" / "best.pt"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"x")
        mocker.patch("screencropnet_yolo.demo.DEFAULT_RUNS_DIR", runs)

        picker = mocker.patch(
            "screencropnet_yolo.model_select._fzf_select",
            return_value=[demo.format_model_choice(weights)],
        )
        mocker.patch("screencropnet_yolo.demo.InferencePipeline")
        mocker.patch("screencropnet_yolo.demo.resolve_device", return_value="cpu")
        mocker.patch(
            "screencropnet_yolo.demo.build_contact_sheet",
            return_value=np.zeros((10, 10, 3), dtype=np.uint8),
        )
        mocker.patch("screencropnet_yolo.demo.open_paths")

        async def fake_run_demo(*args: object, **kwargs: object) -> list[tuple[Path, object]]:
            return [(out_dir / "annotated" / "a.png", _fake_result(1))]

        mocker.patch("screencropnet_yolo.demo.run_demo", side_effect=fake_run_demo)

        rc = demo.main([str(images_dir), "-n", "1", "--select", "--no-open", "-o", str(out_dir)])

        assert rc == 0
        assert (out_dir / "contact_sheet.png").is_file()
        picker.assert_called_once()

    def test_select_no_models_returns_nonzero(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """--select with no discoverable models exits non-zero without a crash."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        (images_dir / "a.jpg").write_bytes(b"x")
        mocker.patch("screencropnet_yolo.demo.DEFAULT_RUNS_DIR", tmp_path / "empty_runs")

        rc = demo.main([str(images_dir), "--select", "--no-open", "-o", str(tmp_path / "out")])
        assert rc == 1
