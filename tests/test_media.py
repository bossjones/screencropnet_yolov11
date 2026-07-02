"""IO / async / CLI tests for ``screencrop-media``.

The pure geometry/naming/argv logic is covered by inline ``## Tests`` in
``media_core.py``. Here we exercise the real IO wrappers against the checked-in
``tests/fixtures/`` assets (real ``.PNG`` screenshots + one ``.mp4``), patch the
ffmpeg/torch edges for the fast suite, and drive the Typer CLI with ``CliRunner``.
A single tool-gated ``@pytest.mark.integration`` test runs real ffmpeg on the clip.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pytest_mock import MockerFixture
from rich.console import Console
from rich.logging import RichHandler
from typer.testing import CliRunner

from screencropnet_yolo import media

FIXTURES = Path(__file__).parent / "fixtures"
VIDEO_NAME = "AlertesInfos-2071001868927471827-(20260627_224457)-img1.mp4"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def sample_image() -> Path:
    return FIXTURES / "00007_twitter.PNG"


@pytest.fixture
def sample_video() -> Path:
    return FIXTURES / VIDEO_NAME


def _det(conf: float, box: tuple[float, float, float, float]) -> SimpleNamespace:
    return SimpleNamespace(confidence=conf, bbox=box)


def _fake_pipeline(mocker: MockerFixture, detections: list[SimpleNamespace]) -> Any:
    pipeline = mocker.MagicMock()
    pipeline.predict_image.return_value = SimpleNamespace(detections=detections)
    return pipeline


# ---- IO wrappers on real fixture bytes -----------------------------------------------


def test_read_image_real_fixture(sample_image: Path) -> None:
    img = media.read_image(sample_image)
    if img.ndim != 3 or img.shape[2] != 3 or img.shape[0] <= 0:
        raise AssertionError(f"expected a decoded HxWx3 image, got shape {img.shape}")


def test_read_image_raises_on_garbage(tmp_path: Path) -> None:
    bogus = tmp_path / "bad.png"
    bogus.write_bytes(b"not a real png")
    with pytest.raises(ValueError):
        media.read_image(bogus)


def test_crop_image_slices(sample_image: Path) -> None:
    img = media.read_image(sample_image)
    cropped = media.crop_image(img, (10, 20, 110, 220))
    if cropped.shape[:2] != (200, 100):
        raise AssertionError(f"crop must yield (h=200, w=100), got {cropped.shape[:2]}")


def test_top_left_rgb_reads_bgr_pixel() -> None:
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[0, 0] = (10, 20, 30)  # BGR
    if media.top_left_rgb(img) != (30, 20, 10):
        raise AssertionError("top_left_rgb must return the pixel as RGB")


def test_resize_onto_bg_canvas_and_fill(sample_image: Path) -> None:
    img = media.read_image(sample_image)  # a tall screenshot → horizontal padding
    out = media.resize_onto_bg(img, (300, 300), bg_rgb=(255, 0, 0))
    if out.shape != (300, 300, 3):
        raise AssertionError(f"canvas must be 300x300x3, got {out.shape}")
    corner = tuple(int(c) for c in out[0, 0])
    if corner != (0, 0, 255):  # red in BGR
        raise AssertionError(f"padding corner must be the bg color, got {corner}")


def test_write_image_roundtrips_atomically(sample_image: Path, tmp_path: Path) -> None:
    img = media.read_image(sample_image)
    dst = tmp_path / "nested" / "out.png"
    media.write_image(dst, img)
    if not dst.exists():
        raise AssertionError("write_image must create the file (and parents)")
    if media.read_image(dst).shape != img.shape:
        raise AssertionError("written image must round-trip to the same shape")


# ---- ffmpeg / ffprobe edges (patched) ------------------------------------------------


def test_probe_media_parses_json(mocker: MockerFixture, sample_video: Path) -> None:
    mocker.patch("screencropnet_yolo.media.ensure_binary")
    mocker.patch(
        "screencropnet_yolo.media.subprocess.run",
        return_value=subprocess.CompletedProcess(
            [], 0, stdout='{"streams": [], "format": {}}', stderr=""
        ),
    )
    if media.probe_media(sample_video) != {"streams": [], "format": {}}:
        raise AssertionError("probe_media must return the parsed ffprobe JSON")


def test_run_ffmpeg_raises_on_nonzero(mocker: MockerFixture) -> None:
    mocker.patch(
        "screencropnet_yolo.media.subprocess.run",
        return_value=subprocess.CompletedProcess([], 1, stdout="", stderr="boom"),
    )
    with pytest.raises(RuntimeError, match="boom"):
        media.run_ffmpeg(["ffmpeg", "-i", "x"])


def test_process_video_builds_crop_and_resize(mocker: MockerFixture, tmp_path: Path) -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    mocker.patch("screencropnet_yolo.media.ensure_binary")
    mocker.patch("screencropnet_yolo.media.sample_video_frames", return_value=[frame])
    run = mocker.patch("screencropnet_yolo.media.run_ffmpeg")
    pipeline = _fake_pipeline(mocker, [_det(0.9, (10, 20, 210, 220))])

    media.process_video(
        tmp_path / "in.mp4",
        tmp_path / "out.mp4",
        autocrop=True,
        autoresize=True,
        size=(1080, 1080),
        bg_override=None,
        pipeline=pipeline,  # type: ignore[arg-type]
    )
    args = run.call_args.args[0]
    vf = args[args.index("-vf") + 1]
    if "crop=200:200:10:20" not in vf or "pad=width=1080" not in vf:
        raise AssertionError(f"ffmpeg filter must crop then pad, got {vf}")


def test_process_video_passthrough_copies(tmp_path: Path) -> None:
    src = tmp_path / "in.mp4"
    src.write_bytes(b"video-bytes")
    dst = tmp_path / "out" / "in.mp4"
    media.process_video(
        src,
        dst,
        autocrop=False,
        autoresize=False,
        size=(1080, 1080),
        bg_override=None,
        pipeline=None,
    )
    if dst.read_bytes() != b"video-bytes":
        raise AssertionError("neither flag must passthrough-copy the video unchanged")


def test_process_video_no_detections_copies_without_reencode(
    mocker: MockerFixture, tmp_path: Path
) -> None:
    src = tmp_path / "in.mp4"
    src.write_bytes(b"video-bytes")
    dst = tmp_path / "out" / "in.mp4"
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    mocker.patch("screencropnet_yolo.media.ensure_binary")
    mocker.patch("screencropnet_yolo.media.sample_video_frames", return_value=[frame])
    run = mocker.patch("screencropnet_yolo.media.run_ffmpeg")
    pipeline = _fake_pipeline(mocker, [])  # no detections → empty crop, no resize

    media.process_video(
        src,
        dst,
        autocrop=True,
        autoresize=False,
        size=(1080, 1080),
        bg_override=None,
        pipeline=pipeline,
    )
    if run.called:
        raise AssertionError("an empty filter must skip ffmpeg, not re-encode")
    if dst.read_bytes() != b"video-bytes":
        raise AssertionError("the clip must be copied through unchanged")


# ---- process_image -------------------------------------------------------------------


def test_process_image_passthrough_is_exact(sample_image: Path, tmp_path: Path) -> None:
    dst = tmp_path / "copy.png"
    media.process_image(
        sample_image,
        dst,
        autocrop=False,
        autoresize=False,
        size=(1, 1),
        bg_override=None,
        pipeline=None,
    )
    if dst.read_bytes() != sample_image.read_bytes():
        raise AssertionError("passthrough must copy the original bytes exactly")


def test_process_image_autocrop_uses_highest_box(
    mocker: MockerFixture, sample_image: Path, tmp_path: Path
) -> None:
    pipeline = _fake_pipeline(mocker, [_det(0.3, (0, 0, 5, 5)), _det(0.9, (10, 10, 90, 130))])
    dst = tmp_path / "cropped.png"
    media.process_image(
        sample_image,
        dst,
        autocrop=True,
        autoresize=False,
        size=(1, 1),
        bg_override=None,
        pipeline=pipeline,  # type: ignore[arg-type]
    )
    if media.read_image(dst).shape[:2] != (120, 80):
        raise AssertionError("autocrop must crop to the highest-confidence box (120x80)")


def test_process_image_no_detections_keeps_full(
    mocker: MockerFixture, sample_image: Path, tmp_path: Path
) -> None:
    pipeline = _fake_pipeline(mocker, [])
    dst = tmp_path / "full.png"
    media.process_image(
        sample_image,
        dst,
        autocrop=True,
        autoresize=False,
        size=(1, 1),
        bg_override=None,
        pipeline=pipeline,  # type: ignore[arg-type]
    )
    if media.read_image(dst).shape[:2] != media.read_image(sample_image).shape[:2]:
        raise AssertionError("no detections must leave the image uncropped")


# ---- async orchestration -------------------------------------------------------------


async def test_run_media_processes_all_inputs(sample_image: Path, tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    for name in ("a.png", "b.png"):
        shutil.copy2(sample_image, in_dir / name)
    out_dir = tmp_path / "out"

    results = await media.run_media(
        [in_dir / "a.png", in_dir / "b.png"],
        in_dir,
        out_dir,
        pipeline=None,
        autocrop=False,
        autoresize=True,
        size=(200, 200),
        bg_override=None,
        threads=2,
    )
    if len(results) != 2 or not all(r.ok for r in results):
        raise AssertionError(f"both inputs must succeed, got {results}")
    resized = out_dir / "resized-a.png"
    if media.read_image(resized).shape[:2] != (200, 200):
        raise AssertionError("artifact must be resized to the target and named resized-")


async def test_run_media_skips_failures(sample_image: Path, tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    shutil.copy2(sample_image, in_dir / "ok.png")
    (in_dir / "bad.png").write_bytes(b"garbage")
    out_dir = tmp_path / "out"

    results = await media.run_media(
        [in_dir / "ok.png", in_dir / "bad.png"],
        in_dir,
        out_dir,
        pipeline=None,
        autocrop=False,
        autoresize=True,
        size=(120, 120),
        bg_override=None,
        threads=2,
    )
    ok = {r.src.name: r.ok for r in results}
    if ok != {"ok.png": True, "bad.png": False}:
        raise AssertionError(f"a corrupt input must fail without sinking the batch, got {ok}")


# ---- CLI (CliRunner) -----------------------------------------------------------------


def test_cli_help_lists_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    # CI force-enables color; rich highlights each dash separately, so ANSI codes split
    # "--autocrop" in the raw output. Strip escapes (and widen) before asserting.
    monkeypatch.setenv("COLUMNS", "200")
    result = CliRunner().invoke(media.app, ["--help"])
    clean = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    if result.exit_code != 0 or "--autocrop" not in clean:
        raise AssertionError(
            f"--help must succeed and document --autocrop; "
            f"exit={result.exit_code} exc={result.exception!r} clean={clean!r}"
        )


def test_cli_info_json_on_fixtures(fixtures_dir: Path) -> None:
    result = CliRunner().invoke(media.app, [str(fixtures_dir), "--info", "--json", "--no-color"])
    clean = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    if result.exit_code != 0 or '"kind": "image"' not in clean:
        raise AssertionError(f"--info --json must report image metadata, got:\n{result.output}")


def test_gather_info_survives_probe_failure(mocker: MockerFixture, sample_video: Path) -> None:
    # A missing/broken ffprobe (as on a CI runner without ffmpeg) must degrade, not crash.
    mocker.patch(
        "screencropnet_yolo.media.probe_media",
        side_effect=RuntimeError("Required binary 'ffprobe' not found on PATH"),
    )
    rows = media.gather_info([sample_video])
    if len(rows) != 1 or rows[0].kind != "video":
        raise AssertionError(f"a video row must still be emitted, got {rows}")
    if rows[0].width is not None or rows[0].codec is not None:
        raise AssertionError("a failed probe must leave video fields null")
    if rows[0].size_bytes != sample_video.stat().st_size:
        raise AssertionError("the real byte size must still be reported")


def test_cli_empty_dir_errors(tmp_path: Path) -> None:
    result = CliRunner().invoke(media.app, [str(tmp_path), "--no-color"])
    if result.exit_code != 1 or "No image/video" not in result.stderr:
        raise AssertionError(f"an empty dir must exit 1, got {result.exit_code}: {result.stderr}")


def test_cli_output_inside_input_refused(sample_image: Path, tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    shutil.copy2(sample_image, in_dir / "a.png")
    result = CliRunner().invoke(
        media.app, [str(in_dir), "--autoresize", "--output-dir", str(in_dir / "out"), "--no-color"]
    )
    if result.exit_code != 1 or "refusing" not in result.stderr:
        raise AssertionError(f"output inside input must be refused, got: {result.stderr}")


def test_cli_size_conflict_errors(sample_image: Path, tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    shutil.copy2(sample_image, in_dir / "a.png")
    result = CliRunner().invoke(
        media.app,
        [
            str(in_dir),
            "--autoresize",
            "--size",
            "100x100",
            "--width",
            "50",
            "--output-dir",
            str(tmp_path / "out"),
            "--no-color",
        ],
    )
    if result.exit_code != 1:
        raise AssertionError("--size with --width must exit 1")


def test_cli_end_to_end_resize_no_model(sample_image: Path, tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    shutil.copy2(sample_image, in_dir / "a.png")
    out_dir = tmp_path / "out"
    result = CliRunner().invoke(
        media.app,
        [str(in_dir), "--autoresize", "--square", "--output-dir", str(out_dir), "--no-color"],
    )
    if result.exit_code != 0:
        raise AssertionError(f"resize-only run must succeed, got: {result.output}")
    if media.read_image(out_dir / "resized-a.png").shape[:2] != (1080, 1080):
        raise AssertionError("--square must produce a 1080x1080 artifact")
    if sample_image.read_bytes() != (in_dir / "a.png").read_bytes():
        raise AssertionError("the input file must be left untouched")


def test_build_pipeline_passes_select(mocker: MockerFixture, tmp_path: Path) -> None:
    resolve = mocker.patch(
        "screencropnet_yolo.media.resolve_model", return_value=("m.pt", "selected (m.pt)")
    )
    mocker.patch("screencropnet_yolo.media.load_config", return_value={"model": {}})
    mocker.patch("screencropnet_yolo.media.resolve_device", return_value="cpu")
    mocker.patch("screencropnet_yolo.media.InferencePipeline")

    _pipeline, banner = media.build_pipeline(
        model=None, latest=False, select=True, conf=None, config_path=tmp_path
    )
    if resolve.call_args.kwargs.get("select") is not True:
        raise AssertionError("--select must propagate to resolve_model")
    if "m.pt" not in banner or "selected (m.pt)" not in banner:
        raise AssertionError(f"the banner must name the resolved model, got: {banner!r}")


@pytest.mark.parametrize(
    ("verbose", "expected"),
    [(0, 3), (1, 3), (3, 3), (4, 0)],
)
def test_quiet_onnxruntime_severity_by_verbosity(
    mocker: MockerFixture, verbose: int, expected: int
) -> None:
    """ORT's native logger is muted to Error below -vvvv and opened up at -vvvv."""
    set_sev = mocker.patch("onnxruntime.set_default_logger_severity")
    media._quiet_onnxruntime(verbose)
    set_sev.assert_called_once_with(expected)


@pytest.mark.parametrize(
    ("model_ref", "quieted"),
    [("best.onnx", True), ("best.pt", False)],
)
def test_build_pipeline_quiets_only_onnx(
    mocker: MockerFixture, tmp_path: Path, model_ref: str, quieted: bool
) -> None:
    mocker.patch(
        "screencropnet_yolo.media.resolve_model",
        return_value=(model_ref, f"selected ({model_ref})"),
    )
    mocker.patch("screencropnet_yolo.media.load_config", return_value={"model": {}})
    mocker.patch("screencropnet_yolo.media.resolve_device", return_value="cpu")
    mocker.patch("screencropnet_yolo.media.InferencePipeline")
    quiet = mocker.patch("screencropnet_yolo.media._quiet_onnxruntime")

    media.build_pipeline(
        model=None, latest=False, select=True, conf=None, config_path=tmp_path, verbose=2
    )
    if quiet.called is not quieted:
        raise AssertionError(f"_quiet_onnxruntime called={quiet.called} for {model_ref}")


def test_setup_logging_binds_richhandler_to_console() -> None:
    """Logging must share the console the Progress uses, so the two coordinate."""
    root = logging.getLogger()
    saved = root.handlers[:]
    console = Console(stderr=True)
    try:
        media._setup_logging(logging.INFO, console=console)
        handlers = root.handlers
        if len(handlers) != 1 or not isinstance(handlers[0], RichHandler):
            raise AssertionError(f"expected a single RichHandler, got {handlers}")
        if handlers[0].console is not console:
            raise AssertionError("the RichHandler must render to the shared console")
    finally:
        root.handlers = saved


@pytest.mark.parametrize(
    "model_flag", [["--fuzzy"], ["--select"], ["--latest"], ["--model", "x.pt"]]
)
def test_cli_model_flag_without_autocrop_errors(
    sample_image: Path, tmp_path: Path, model_flag: list[str]
) -> None:
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    shutil.copy2(sample_image, in_dir / "a.png")
    out_dir = tmp_path / "out"
    result = CliRunner().invoke(
        media.app, [str(in_dir), *model_flag, "--output-dir", str(out_dir), "--no-color"]
    )
    if result.exit_code != 1 or "require --autocrop" not in result.stderr:
        raise AssertionError(
            f"{model_flag} without --autocrop must exit 1 with guidance, got "
            f"{result.exit_code}: {result.stderr}"
        )
    if out_dir.exists() and any(out_dir.iterdir()):
        raise AssertionError("the guard must reject before writing any artifact")


# ---- integration (real ffmpeg; deselected by default via -m 'not integration') -------


@pytest.mark.integration
def test_integration_video_resize_real_ffmpeg(sample_video: Path, tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None or not sample_video.exists():
        pytest.skip("ffmpeg or the sample video is unavailable")
    dst = tmp_path / "out.mp4"
    media.process_video(
        sample_video,
        dst,
        autocrop=False,
        autoresize=True,
        size=(480, 480),
        bg_override="#16202A",
        pipeline=None,
    )
    if not dst.exists() or dst.stat().st_size <= 0:
        raise AssertionError("real ffmpeg resize must produce a non-empty mp4")


@pytest.mark.integration
def test_integration_autocrop_real_model_reports_and_writes(
    sample_image: Path, tmp_path: Path
) -> None:
    """End-to-end: a real YOLO model crops a fixture and its identity is surfaced."""
    pytest.importorskip("torch")
    weights = Path(__file__).parent.parent / "yolo26n.pt"
    if not weights.exists():
        pytest.skip("yolo26n.pt checkpoint is unavailable")
    before = sample_image.read_bytes()
    out_dir = tmp_path / "out"
    result = CliRunner().invoke(
        media.app,
        [
            str(sample_image),
            "--autocrop",
            "--model",
            str(weights),
            "--output-dir",
            str(out_dir),
            "--no-color",
        ],
    )
    if result.exit_code != 0:
        raise AssertionError(f"real-model autocrop must succeed, got: {result.stderr}")
    if "Model:" not in result.stderr or "yolo26n" not in result.stderr:
        raise AssertionError(f"the run must name the model in use, got: {result.stderr}")
    artifacts = [p for p in out_dir.rglob("*") if p.is_file()]
    if not artifacts:
        raise AssertionError("autocrop must write at least one artifact")
    if sample_image.read_bytes() != before:
        raise AssertionError("the input must be left untouched")
