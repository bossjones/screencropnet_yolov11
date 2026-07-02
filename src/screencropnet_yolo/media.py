"""``screencrop-media`` CLI: async crop + solid-background resize for images & video.

Point it at an image, a video, or a directory of them; optionally ``--autocrop``
(YOLO-detect → crop to the highest-confidence region) and/or ``--autoresize`` (fit onto
a solid background at a target size), do either/both/neither, and write artifacts to a
separate output directory without touching the inputs.

This module owns the IO edges (cv2 reads/writes, ffmpeg/ffprobe subprocesses, the YOLO
pipeline) and the async orchestration; all pure geometry/naming/argv logic lives in the
torch-free :mod:`screencropnet_yolo.media_core`. See ``specs/screencropnet-media.md``.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio.to_thread
import cv2
import numpy as np
import numpy.typing as npt
import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from strif import atomic_output_file

from screencropnet_yolo import media_core as mc
from screencropnet_yolo.demo import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_RUNS_DIR,
    load_config,
    resolve_model,
)
from screencropnet_yolo.inference import InferencePipeline
from screencropnet_yolo.media_core import Box, InfoRow
from screencropnet_yolo.model import resolve_device
from screencropnet_yolo.output import ColorFormatter, human_size

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("./screencrop-media-out")
VIDEO_SAMPLE_FRAMES = 5
"""How many frames to sample for the static video crop rectangle."""

logging.addLevelName(mc.TRACE_LEVEL, "TRACE")


# ---- IO wrappers (thin; patched in tests) --------------------------------------------


def read_image(path: Path) -> npt.NDArray[np.uint8]:
    """Decode an image file to a BGR array; raise if it can't be read."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return np.asarray(img, dtype=np.uint8)


def write_image(path: Path, img: npt.NDArray[np.uint8]) -> None:
    """Encode ``img`` in ``path``'s format and write it atomically (parents created)."""
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise ValueError(f"Failed to encode image for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output_file(path) as tmp:
        Path(tmp).write_bytes(buf.tobytes())


def top_left_rgb(img: npt.NDArray[np.uint8]) -> tuple[int, int, int]:
    """Return the top-left pixel as ``(r, g, b)`` (cv2 arrays are BGR)."""
    b, g, r = (int(c) for c in img[0, 0][:3])
    return r, g, b


def crop_image(img: npt.NDArray[np.uint8], box: Box) -> npt.NDArray[np.uint8]:
    """Slice ``img`` to ``box`` (already clamped to the image bounds)."""
    x1, y1, x2, y2 = box
    return img[y1:y2, x1:x2]


def resize_onto_bg(
    img: npt.NDArray[np.uint8], size: tuple[int, int], bg_rgb: tuple[int, int, int]
) -> npt.NDArray[np.uint8]:
    """Fit ``img`` aspect-preserved and centered onto a solid ``bg`` canvas of ``size``."""
    dst_w, dst_h = size
    src_h, src_w = img.shape[:2]
    geo = mc.fit_geometry(src_w, src_h, dst_w, dst_h)
    resized = cv2.resize(img[:, :, :3], (geo.scaled_w, geo.scaled_h), interpolation=cv2.INTER_AREA)
    r, g, b = bg_rgb
    canvas = np.full((dst_h, dst_w, 3), (b, g, r), dtype=np.uint8)
    canvas[
        geo.offset_y : geo.offset_y + geo.scaled_h, geo.offset_x : geo.offset_x + geo.scaled_w
    ] = resized
    return canvas


def run_ffmpeg(args: list[str]) -> None:
    """Run an ffmpeg/ffprobe argv, raising with stderr context on failure."""
    logger.log(mc.TRACE_LEVEL, "ffmpeg: %s", " ".join(args))
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{args[0]} failed ({proc.returncode}): {proc.stderr.strip()}")


def probe_media(path: Path) -> dict[str, Any]:
    """Run ffprobe and return its parsed JSON metadata for ``path``."""
    ensure_binary("ffprobe")
    proc = subprocess.run(mc.build_ffprobe_args(path), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def sample_video_frames(
    path: Path, count: int = VIDEO_SAMPLE_FRAMES
) -> list[npt.NDArray[np.uint8]]:
    """Grab up to ``count`` evenly-spaced frames from ``path`` (for the static crop rect)."""
    cap = cv2.VideoCapture(str(path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            ok, frame = cap.read()
            return [np.asarray(frame, dtype=np.uint8)] if ok else []
        step = max(1, total // count)
        frames: list[npt.NDArray[np.uint8]] = []
        for idx in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames.append(np.asarray(frame, dtype=np.uint8))
            if len(frames) >= count:
                break
        return frames
    finally:
        cap.release()


def ensure_binary(name: str) -> None:
    """Raise a clear error if a required system binary isn't on PATH."""
    if shutil.which(name) is None:
        raise RuntimeError(f"Required binary {name!r} not found on PATH (install ffmpeg)")


# ---- Processing ----------------------------------------------------------------------


@dataclass
class MediaResult:
    """Outcome of processing one input."""

    src: Path
    dst: Path
    kind: str
    ok: bool
    error: str | None = None


def _detect_box(
    pipeline: InferencePipeline,
    image: npt.NDArray[np.uint8],
    *,
    predict_lock: threading.Lock | None,
) -> Box | None:
    """Predict on ``image`` and return the highest-confidence box (serialized predict)."""
    with predict_lock or nullcontext():
        result = pipeline.predict_image(image)
    return mc.pick_crop_box(result.detections)


def process_image(
    src: Path,
    dst: Path,
    *,
    autocrop: bool,
    autoresize: bool,
    size: tuple[int, int],
    bg_override: str | None,
    pipeline: InferencePipeline | None,
    predict_lock: threading.Lock | None = None,
) -> None:
    """Crop and/or resize an image ``src`` → ``dst`` (passthrough copy when neither)."""
    if not autocrop and not autoresize:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return

    img = read_image(src)
    if autocrop and pipeline is not None:
        box = _detect_box(pipeline, img, predict_lock=predict_lock)
        if box is None:
            logger.warning("No detections in %s; skipping crop", src.name)
        else:
            h, w = img.shape[:2]
            img = crop_image(img, mc.clamp_box(box, w, h))
    if autoresize:
        bg = mc.resolve_bg_color(bg_override, top_left_rgb(img))
        img = resize_onto_bg(img, size, bg)
    write_image(dst, img)


def process_video(
    src: Path,
    dst: Path,
    *,
    autocrop: bool,
    autoresize: bool,
    size: tuple[int, int],
    bg_override: str | None,
    pipeline: InferencePipeline | None,
    predict_lock: threading.Lock | None = None,
) -> None:
    """Crop (single static rect) and/or resize a video via ffmpeg; passthrough when neither."""
    if not autocrop and not autoresize:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return

    ensure_binary("ffmpeg")
    frames = sample_video_frames(src) if (autocrop or autoresize) else []
    crop: Box | None = None
    bg_rgb = mc.DARK_BG
    if frames:
        bg_rgb = top_left_rgb(frames[0])
    if autocrop and pipeline is not None and frames:
        h, w = frames[0].shape[:2]
        boxes = [
            b for b in (_detect_box(pipeline, f, predict_lock=predict_lock) for f in frames) if b
        ]
        union = mc.union_boxes(boxes)
        if union is None:
            logger.warning("No detections across sampled frames of %s; skipping crop", src.name)
        else:
            crop = mc.clamp_box(union, w, h)

    dst.parent.mkdir(parents=True, exist_ok=True)
    if crop is None and not autoresize:
        logger.info("No crop and no resize for %s; copying through", src.name)
        shutil.copy2(src, dst)
        return

    bg = mc.parse_hex_color(bg_override) if bg_override else bg_rgb
    args = mc.build_ffmpeg_crop_resize_args(
        src, dst, crop=crop, size=size if autoresize else None, bg=bg
    )
    run_ffmpeg(args)


def process_one(
    src: Path,
    dst: Path,
    *,
    autocrop: bool,
    autoresize: bool,
    size: tuple[int, int],
    bg_override: str | None,
    pipeline: InferencePipeline | None,
    predict_lock: threading.Lock | None = None,
) -> MediaResult:
    """Dispatch ``src`` to the image or video path; capture per-input failures."""
    kind = mc.classify_kind(src)
    try:
        handler = process_video if kind == "video" else process_image
        handler(
            src,
            dst,
            autocrop=autocrop,
            autoresize=autoresize,
            size=size,
            bg_override=bg_override,
            pipeline=pipeline,
            predict_lock=predict_lock,
        )
        return MediaResult(src=src, dst=dst, kind=kind, ok=True)
    except Exception as exc:  # noqa: BLE001 — one bad input must not sink the batch
        logger.exception("Failed to process %s", src)
        return MediaResult(src=src, dst=dst, kind=kind, ok=False, error=str(exc))


# ---- Async orchestration -------------------------------------------------------------


async def run_media(
    inputs: list[Path],
    input_root: Path,
    output_dir: Path,
    *,
    pipeline: InferencePipeline | None,
    autocrop: bool,
    autoresize: bool,
    size: tuple[int, int],
    bg_override: str | None,
    threads: int = 8,
    progress: Progress | None = None,
) -> list[MediaResult]:
    """Process ``inputs`` concurrently, each blocking unit offloaded off the event loop."""
    sem = asyncio.Semaphore(max(1, threads))
    predict_lock = threading.Lock()
    task = progress.add_task("processing", total=len(inputs)) if progress else None

    async def _worker(src: Path) -> MediaResult:
        dst = mc.output_path_for(src, input_root, output_dir, cropped=autocrop, resized=autoresize)
        async with sem:
            result = await anyio.to_thread.run_sync(
                functools.partial(
                    process_one,
                    src,
                    dst,
                    autocrop=autocrop,
                    autoresize=autoresize,
                    size=size,
                    bg_override=bg_override,
                    pipeline=pipeline,
                    predict_lock=predict_lock,
                )
            )
        if progress is not None and task is not None:
            progress.advance(task)
        return result

    return list(await asyncio.gather(*(_worker(src) for src in inputs)))


# ---- --info --------------------------------------------------------------------------


def gather_info(inputs: list[Path]) -> list[InfoRow]:
    """Collect per-input metadata rows (cv2 dims for images, ffprobe for video)."""
    rows: list[InfoRow] = []
    for src in inputs:
        size_bytes = src.stat().st_size if src.exists() else None
        kind = mc.classify_kind(src)
        if kind == "video":
            try:
                rows.append(mc.parse_ffprobe_json(src, kind, probe_media(src), size_bytes))
            except Exception as exc:  # noqa: BLE001 — a bad/absent ffprobe must not sink --info
                logger.warning("Could not probe %s (%s); reporting name/size only", src, exc)
                rows.append(InfoRow(path=str(src), kind=kind, size_bytes=size_bytes))
        else:
            img = read_image(src)
            h, w = img.shape[:2]
            rows.append(mc.image_info_row(src, w, h, size_bytes))
    return rows


def render_info_table(rows: list[InfoRow]) -> Table:
    """Render ``--info`` rows as a rich table."""
    table = Table(title="media info")
    for col in ("path", "kind", "size", "dims", "duration", "fps", "codec"):
        table.add_column(col, overflow="fold")
    for r in rows:
        dims = f"{r.width}x{r.height}" if r.width and r.height else ""
        table.add_row(
            r.path,
            r.kind,
            human_size(r.size_bytes) if r.size_bytes is not None else "",
            dims,
            f"{r.duration:.2f}s" if r.duration is not None else "",
            f"{r.fps:.2f}" if r.fps is not None else "",
            r.codec or "",
        )
    return table


def info_to_json(rows: list[InfoRow]) -> str:
    """Serialize ``--info`` rows as JSON."""
    return json.dumps([r.__dict__ for r in rows], indent=2)


# ---- CLI -----------------------------------------------------------------------------


app = typer.Typer(add_completion=False, help="Crop and/or resize images & video with a YOLO model.")


def _resolve_color(color: bool | None) -> bool:
    """Auto-detect color from TTY/NO_COLOR unless the ``--color/--no-color`` flag forces it."""
    if color is not None:
        return color
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _setup_logging(level: int, *, color: bool) -> None:
    """Configure a single colored stream handler at ``level``."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(ColorFormatter("%(levelname)8s | %(name)s | %(message)s", enabled=color))
    logging.basicConfig(level=level, handlers=[handler], force=True)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)


@app.command()
def run(  # noqa: PLR0913 — a CLI surface is inherently wide
    input: str = typer.Argument(..., help="Image, video, or directory (~ and $VARS expanded)."),
    autocrop: bool = typer.Option(False, "--autocrop", help="Crop to the detected region."),
    autoresize: bool = typer.Option(False, "--autoresize", help="Fit onto a solid background."),
    output_dir: str | None = typer.Option(
        None, "--output-dir", help="Artifact dir (default ./screencrop-media-out)."
    ),
    select: bool = typer.Option(False, "--select", "--fuzzy", help="Fuzzy-pick a model via fzf."),
    model: str | None = typer.Option(None, "--model", help="Explicit model checkpoint."),
    latest: bool = typer.Option(False, "--latest", help="Use the newest trained best.pt."),
    depth: int = typer.Option(0, "--depth", help="Directory recursion depth (0 = top level)."),
    size: str | None = typer.Option(None, "--size", help="Resize target WxH (e.g. 1080x1350)."),
    width: int | None = typer.Option(None, "--width", help="Resize target width."),
    height: int | None = typer.Option(None, "--height", help="Resize target height."),
    square: bool = typer.Option(False, "--square", help="Resize target 1080x1080."),
    bg_color: str | None = typer.Option(None, "--bg-color", help="Background hex (e.g. #16202A)."),
    conf: float | None = typer.Option(None, "--conf", help="Detection confidence threshold."),
    threads: int = typer.Option(8, "--threads", help="Concurrent worker count."),
    info: bool = typer.Option(False, "--info", help="Print input metadata and exit."),
    json_out: bool = typer.Option(False, "--json", help="With --info, emit JSON."),
    color: bool | None = typer.Option(None, "--color/--no-color", help="Force/disable color."),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="-v..-vvvv verbosity."),
    config: str | None = typer.Option(None, "--config", help="Config YAML."),
) -> None:
    """Crop and/or resize the input(s), writing artifacts under ``--output-dir``."""
    use_color = _resolve_color(color)
    _setup_logging(mc.verbosity_to_level(verbose), color=use_color)
    console = Console(no_color=not use_color, stderr=False)
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    config_path = Path(config) if config else DEFAULT_CONFIG_PATH

    input_path = mc.expand_input(input)
    input_root = input_path if input_path.is_dir() else input_path.parent
    inputs = mc.discover_inputs(input_path, depth)
    if not inputs:
        console.print(f"[red]✘[/red] No image/video files found under {input_path}")
        raise typer.Exit(code=1)
    logger.info("Discovered %d input(s) under %s", len(inputs), input_path)

    if info:
        rows = gather_info(inputs)
        if json_out:
            console.print_json(info_to_json(rows))
        else:
            console.print(render_info_table(rows))
        raise typer.Exit(code=0)

    if mc.output_would_clobber_input(input_root, out_dir):
        console.print(f"[red]✘[/red] --output-dir {out_dir} is inside the input; refusing")
        raise typer.Exit(code=1)

    try:
        target_size = mc.resolve_target_size(size=size, width=width, height=height, square=square)
        if bg_color is not None:
            mc.parse_hex_color(bg_color)  # validate early
    except ValueError as exc:
        console.print(f"[red]✘[/red] {exc}")
        raise typer.Exit(code=1) from exc

    pipeline = (
        build_pipeline(
            model=model, latest=latest, select=select, conf=conf, config_path=config_path
        )
        if autocrop
        else None
    )

    with Progress(console=console, disable=not sys.stderr.isatty()) as progress:
        results = asyncio.run(
            run_media(
                inputs,
                input_root,
                out_dir,
                pipeline=pipeline,
                autocrop=autocrop,
                autoresize=autoresize,
                size=target_size,
                bg_override=bg_color,
                threads=threads,
                progress=progress,
            )
        )

    console.print(_summary_table(results))
    failures = [r for r in results if not r.ok]
    raise typer.Exit(code=1 if failures and len(failures) == len(results) else 0)


def build_pipeline(
    *, model: str | None, latest: bool, select: bool, conf: float | None, config_path: Path
) -> InferencePipeline:
    """Resolve the model and construct a YOLO inference pipeline (torch import happens here)."""
    config = load_config(str(config_path))
    model_ref, source = resolve_model(
        model=model, latest=latest, select=select, config=config, runs_dir=DEFAULT_RUNS_DIR
    )
    model_cfg = config.get("model", {})
    class_names = model_cfg.get("class_names", ["object"])
    threshold = conf if conf is not None else config.get("inference", {}).get("confidence", 0.25)
    device = str(resolve_device(config.get("device", {}).get("type", "auto")))
    logger.info("Model: %s | device: %s | conf: %s", source, device, threshold)
    return InferencePipeline(
        model_path=model_ref, class_names=class_names, device=device, conf_threshold=threshold
    )


def _summary_table(results: list[MediaResult]) -> Table:
    """Render the closing per-artifact summary table."""
    table = Table(title="screencrop-media")
    table.add_column("status")
    table.add_column("kind")
    table.add_column("artifact", overflow="fold")
    for r in results:
        status = "[green]✔︎[/green]" if r.ok else "[red]✘[/red]"
        table.add_row(status, r.kind, str(r.dst) if r.ok else f"{r.src} ({r.error})")
    return table


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
