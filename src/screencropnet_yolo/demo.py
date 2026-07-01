"""Demo CLI: async inference on a random sample of images with a contact sheet.

Point this at a directory of screenshots and it samples ``--count`` images at
random, runs YOLO inference off the event loop, draws bounding boxes onto
*copies* (originals untouched), stitches them into one contact-sheet montage,
and opens it so detections are immediately visible. Model source is selectable:
the configured/base model (default), the latest trained ``best.pt`` (``--latest``),
or an explicit path (``--model``).

See ``specs/demo.md`` for the full specification.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import logging
import math
import os
import random
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import anyio.to_thread
import cv2
import numpy as np
import numpy.typing as npt
import yaml

from screencropnet_yolo.inference import InferencePipeline, InferenceResult, ResultExporter
from screencropnet_yolo.model import ModelFactory, resolve_device
from screencropnet_yolo.output import (
    Artifact,
    ColorFormatter,
    format_artifacts_table,
    human_size,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MODEL_EXTS = {".pt", ".onnx"}
DEFAULT_COUNT = 10
DEFAULT_RUNS_DIR = Path("runs")
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "config.yaml"


def load_config(config_path: str) -> dict[str, Any]:
    """Load a YAML config file (mirrors ``train.load_config`` without its heavy imports)."""
    return yaml.safe_load(Path(config_path).read_text())


def discover_images(directory: Path, *, recursive: bool = True) -> list[Path]:
    """Return image files under ``directory``, sorted; empty list if none/not a dir."""
    if not directory.is_dir():
        return []
    entries = directory.rglob("*") if recursive else directory.glob("*")
    return sorted(p for p in entries if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)


def sample_images(images: list[Path], count: int, rng: random.Random) -> list[Path]:
    """Randomly pick ``count`` images (clamped to what's available); ``rng`` drives choice."""
    if count <= 0 or not images:
        return []
    return rng.sample(images, min(count, len(images)))


def find_latest_run(runs_dir: Path) -> Path | None:
    """Newest trained ``best.pt`` under ``runs_dir`` by mtime, or None if there is none."""
    if not runs_dir.is_dir():
        return None
    candidates = [
        *runs_dir.glob("*/train/weights/best.pt"),
        *runs_dir.glob("*/weights/best.pt"),
    ]
    existing = [p for p in candidates if p.is_file()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def discover_models(search_root: Path) -> list[Path]:
    """All model weight files (``.pt``/``.onnx``) under ``search_root``, newest first."""
    if not search_root.is_dir():
        return []
    found = [p for p in search_root.rglob("*") if p.is_file() and p.suffix.lower() in MODEL_EXTS]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def format_model_choice(path: Path) -> str:
    """Render one fzf line for ``path``: ``'best.pt : /abs/path  [42.0 MB]'``."""
    return f"{path.name} : {path}  [{human_size(path.stat().st_size)}]"


ModelSelector = Callable[[list[str]], list[str]]
"""A picker: given display lines, return the chosen line(s) (empty if cancelled)."""


def _fzf_select(choices: list[str]) -> list[str]:
    """Default selector: hand ``choices`` to fzf via pyfzf, return the chosen line(s).

    ``pyfzf`` is imported lazily so that importing this module never requires the
    ``fzf`` binary — only ``--select`` runs pull it in.
    """
    from pyfzf.pyfzf import FzfPrompt

    # pyfzf ships no type stubs, so .prompt is untyped; the signature here is the contract.
    return FzfPrompt().prompt(choices, "--height=40% --reverse")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


def select_model(candidates: list[Path], *, selector: ModelSelector | None = None) -> Path | None:
    """Present ``candidates`` via a fuzzy picker; return the chosen Path, or None if cancelled.

    The display line is mapped back to its Path through a dict, so the selection is
    recovered exactly rather than re-parsed out of the formatted string.
    """
    choices = {format_model_choice(p): p for p in candidates}
    picked = (selector or _fzf_select)(list(choices))
    if not picked:
        return None
    return choices[picked[0]]


def resolve_model(
    *,
    model: str | None,
    latest: bool,
    select: bool = False,
    config: dict[str, Any],
    runs_dir: Path,
    selector: ModelSelector | None = None,
) -> tuple[str, str]:
    """Resolve the model reference and a human label for the banner.

    Precedence: interactive ``--select`` pick > explicit ``--model`` path >
    ``--latest`` trained run > config ``model.weights`` > base pretrained
    checkpoint for ``model.size``.
    """
    if select:
        candidates = discover_models(runs_dir)
        if not candidates:
            raise FileNotFoundError(f"No model files (.pt/.onnx) found under {runs_dir}")
        chosen = select_model(candidates, selector=selector)
        if chosen is None:
            raise RuntimeError("Model selection cancelled")
        return str(chosen), f"selected ({chosen.name})"

    if model:
        path = Path(model)
        if not path.is_file():
            raise FileNotFoundError(f"Model weights not found: {model}")
        return str(path), f"explicit ({path.name})"

    if latest:
        found = find_latest_run(runs_dir)
        if found is None:
            raise FileNotFoundError(f"No trained model (best.pt) found under {runs_dir}")
        return str(found), f"latest trained ({found})"

    model_cfg = config.get("model", {})
    weights = model_cfg.get("weights")
    if weights:
        return str(weights), f"config weights ({Path(str(weights)).name})"

    size = model_cfg.get("size", "m")
    checkpoint = ModelFactory.MODEL_SIZES.get(size, ModelFactory.MODEL_SIZES["m"])
    return checkpoint, f"base pretrained ({checkpoint})"


def _fit_tile(img: npt.NDArray[np.uint8], cell: int, bg: int) -> npt.NDArray[np.uint8]:
    """Resize ``img`` to fit a ``cell``x``cell`` square, aspect-preserved, padded with ``bg``."""
    h, w = img.shape[:2]
    scale = cell / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(img[:, :, :3], (nw, nh), interpolation=cv2.INTER_AREA)
    tile = np.full((cell, cell, 3), bg, dtype=np.uint8)
    y0, x0 = (cell - nh) // 2, (cell - nw) // 2
    tile[y0 : y0 + nh, x0 : x0 + nw] = resized
    return tile


def tile_images(
    images: list[npt.NDArray[np.uint8]], *, cols: int = 5, cell: int = 320, bg: int = 30
) -> npt.NDArray[np.uint8]:
    """Tile ``images`` into a single grid of ``cols`` columns (rows as needed)."""
    if not images:
        raise ValueError("no images to tile")
    cols = max(1, min(cols, len(images)))
    rows = math.ceil(len(images) / cols)
    sheet = np.full((rows * cell, cols * cell, 3), bg, dtype=np.uint8)
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        sheet[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell] = _fit_tile(img, cell, bg)
    return sheet


def build_contact_sheet(
    image_paths: list[Path], *, cols: int = 5, cell: int = 320
) -> npt.NDArray[np.uint8]:
    """Read the annotated copies and tile them into one contact sheet."""
    images: list[npt.NDArray[np.uint8]] = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            logger.warning("Skipping unreadable image for contact sheet: %s", p)
            continue
        images.append(np.asarray(img, dtype=np.uint8))
    return tile_images(images, cols=cols, cell=cell)


def make_output_dir(base: Path | None = None) -> Path:
    """Return the run's output dir: ``base`` (created) or a fresh temp dir under /tmp."""
    if base is not None:
        base.mkdir(parents=True, exist_ok=True)
        return base
    return Path(tempfile.mkdtemp(prefix="screencropnet_demo_"))


def annotate_one(
    pipeline: InferencePipeline,
    src: Path,
    out_dir: Path,
    *,
    predict_lock: threading.Lock | None = None,
) -> tuple[Path, InferenceResult]:
    """Blocking: infer on ``src``, draw boxes, write an annotated copy into ``out_dir``.

    The original is never touched — detections are drawn onto a fresh copy. A shared
    ``predict_lock`` serializes the ultralytics call, which is not thread-safe on a
    single model instance (the lazy fuse/predictor setup races otherwise); image decode,
    drawing, and encode still run concurrently across the pool.
    """
    img = cv2.imread(str(src))
    if img is None:
        raise ValueError(f"Failed to read image: {src}")
    with predict_lock or nullcontext():
        result = pipeline.predict_image(str(src))
    annotated = pipeline.draw_detections(np.asarray(img, dtype=np.uint8), result)
    dest = out_dir / f"{src.stem}_annotated{src.suffix or '.png'}"
    cv2.imwrite(str(dest), annotated)
    return dest, result


async def run_demo(
    pipeline: InferencePipeline,
    images: list[Path],
    out_dir: Path,
    *,
    concurrency: int = 8,
) -> list[tuple[Path, InferenceResult]]:
    """Infer over ``images`` concurrently, offloading each blocking call off the loop.

    Inference is CPU/GPU-bound, so each image runs via ``anyio.to_thread.run_sync``
    under a semaphore (mirrors ``server/worker.py``). Failures are logged and skipped.
    """
    annotated_dir = out_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    predict_lock = threading.Lock()

    async def _worker(src: Path) -> tuple[Path, InferenceResult] | None:
        async with sem:
            try:
                dest, result = await anyio.to_thread.run_sync(
                    functools.partial(
                        annotate_one, pipeline, src, annotated_dir, predict_lock=predict_lock
                    )
                )
            except Exception:
                logger.exception("Inference failed for %s; skipping", src)
                return None
        logger.info("%s → %d detection(s)", src.name, len(result.detections))
        return dest, result

    outcomes = await asyncio.gather(*(_worker(src) for src in images))
    return [o for o in outcomes if o is not None]


def open_paths(paths: list[Path], *, enabled: bool = True) -> None:
    """Open ``paths`` in the macOS default viewer, detached. No-op when disabled/off-Mac."""
    if not enabled or not paths:
        return
    if sys.platform != "darwin":
        logger.info("Auto-open skipped on non-macOS platform (%s)", sys.platform)
        return
    subprocess.Popen(
        ["open", *[str(p) for p in paths]],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def setup_logging(output_dir: Path, level: str = "INFO", *, color: bool = False) -> Path:
    """Log to a timestamped file in ``output_dir`` plus a (optionally colored) console."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fmt, datefmt = "%(asctime)s | %(levelname)8s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(ColorFormatter(fmt, datefmt=datefmt, enabled=color))
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        handlers=[file_handler, stream_handler],
        force=True,
    )
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    return log_file


def _artifact(label: str, path: Path) -> Artifact:
    size = path.stat().st_size if path.exists() else None
    return Artifact(label=label, path=str(path), size=size)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the demo command."""
    parser = argparse.ArgumentParser(
        prog="screencropnet_yolo.demo",
        description="Run YOLO inference on a random sample of images and open a contact sheet.",
    )
    parser.add_argument("images_dir", help="Directory of images to sample from")
    parser.add_argument(
        "-n", "--count", type=int, default=DEFAULT_COUNT, help="Images to sample (default: 10)"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--model", default=None, help="Explicit model weights (.pt/.onnx); highest priority"
    )
    source.add_argument(
        "--latest", action="store_true", help="Use the newest trained best.pt under runs/"
    )
    source.add_argument(
        "--select",
        "--fuzzy",
        dest="select",
        action="store_true",
        help="Interactively pick a .pt/.onnx model under runs/ via fzf",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed the sampler for a reproducible selection"
    )
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recurse into subdirectories when discovering images (default: on)",
    )
    parser.add_argument("--device", default=None, help="Device (auto/cpu/cuda/mps/index)")
    parser.add_argument(
        "-o", "--output", default=None, help="Output dir (default: a fresh /tmp dir)"
    )
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG_PATH), help="Config file")
    parser.add_argument("--color", action="store_true", help="Colorize banner/table/log")
    parser.add_argument("--no-open", action="store_true", help="Do not open the contact sheet")
    parser.add_argument("--json", action="store_true", help="Also write a detections JSON")
    parser.add_argument(
        "--concurrency", type=int, default=8, help="Max concurrent inference threads (default: 8)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point: sample images, run async inference, build+open a contact sheet."""
    args = parse_args(argv)
    color = bool(args.color) and sys.stdout.isatty() and "NO_COLOR" not in os.environ
    output_dir = make_output_dir(Path(args.output) if args.output else None)
    log_file = setup_logging(output_dir, color=color)

    try:
        config = load_config(args.config)
        model_ref, model_source = resolve_model(
            model=args.model,
            latest=args.latest,
            select=args.select,
            config=config,
            runs_dir=DEFAULT_RUNS_DIR,
        )

        model_cfg = config.get("model", {})
        class_names = model_cfg.get("class_names", ["object"])
        conf = (
            args.conf
            if args.conf is not None
            else config.get("inference", {}).get("confidence", 0.25)
        )
        device = str(resolve_device(args.device or config.get("device", {}).get("type", "auto")))

        images = discover_images(Path(args.images_dir), recursive=args.recursive)
        if not images:
            logger.error("No images found in %s", args.images_dir)
            return 1

        rng = random.Random(args.seed) if args.seed is not None else random.Random()
        sample = sample_images(images, args.count, rng)
        logger.info(
            "Model: %s | device: %s | sampled %d of %d image(s)",
            model_source,
            device,
            len(sample),
            len(images),
        )

        pipeline = InferencePipeline(
            model_path=model_ref, class_names=class_names, device=device, conf_threshold=conf
        )
        results = asyncio.run(run_demo(pipeline, sample, output_dir, concurrency=args.concurrency))
        if not results:
            logger.error("No images were successfully annotated")
            return 1

        annotated_paths = [dest for dest, _ in results]
        montage_path = output_dir / "contact_sheet.png"
        cv2.imwrite(str(montage_path), build_contact_sheet(annotated_paths))

        artifacts = [
            _artifact("contact sheet", montage_path),
            _artifact("annotated copies", output_dir / "annotated"),
            _artifact("log", log_file),
        ]
        if args.json:
            json_path = output_dir / "detections.json"
            ResultExporter.to_json([r for _, r in results], str(json_path))
            artifacts.append(_artifact("detections json", json_path))

        open_paths([montage_path], enabled=not args.no_open)

        total_det = sum(len(r.detections) for _, r in results)
        logger.info(
            "Done: %d image(s), %d detection(s) total; contact sheet at %s",
            len(results),
            total_det,
            montage_path,
        )
        print(format_artifacts_table(artifacts, enabled=color))
        return 0
    except Exception:
        logger.exception("Demo failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())


## Tests


def test_discover_images_filters_and_sorts() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "b.png").write_bytes(b"x")
        (root / "a.jpg").write_bytes(b"x")
        (root / "note.txt").write_text("nope")
        names = [p.name for p in discover_images(root, recursive=False)]
        if names != ["a.jpg", "b.png"]:
            raise AssertionError(f"expected sorted image names, got {names}")


def test_discover_images_recursive_toggle() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        sub = root / "sub"
        sub.mkdir()
        (sub / "deep.png").write_bytes(b"x")
        if discover_images(root, recursive=False):
            raise AssertionError("non-recursive discovery must not descend into subdirs")
        if len(discover_images(root, recursive=True)) != 1:
            raise AssertionError("recursive discovery must find the nested image")


def test_discover_images_empty_dir() -> None:
    with tempfile.TemporaryDirectory() as d:
        if discover_images(Path(d), recursive=True) != []:
            raise AssertionError("empty directory must yield []")


def test_sample_images_clamps_and_is_seedable() -> None:
    imgs = [Path(f"{i}.jpg") for i in range(5)]
    if len(sample_images(imgs, 10, random.Random(0))) != 5:
        raise AssertionError("count greater than available must clamp to len")
    if sample_images(imgs, 0, random.Random(0)) != []:
        raise AssertionError("count of 0 must yield []")
    a = sample_images(imgs, 3, random.Random(42))
    b = sample_images(imgs, 3, random.Random(42))
    if a != b:
        raise AssertionError("same seed must produce the same sample")


def test_find_latest_run_picks_newest() -> None:
    with tempfile.TemporaryDirectory() as d:
        runs = Path(d)
        old_weights = runs / "old" / "train" / "weights"
        new_weights = runs / "new" / "train" / "weights"
        old_weights.mkdir(parents=True)
        new_weights.mkdir(parents=True)
        old = old_weights / "best.pt"
        new = new_weights / "best.pt"
        old.write_bytes(b"x")
        new.write_bytes(b"x")
        os.utime(old, (1, 1))
        os.utime(new, (10_000_000, 10_000_000))
        if find_latest_run(runs) != new:
            raise AssertionError("must select the newest best.pt by mtime")


def test_find_latest_run_none_when_absent() -> None:
    with tempfile.TemporaryDirectory() as d:
        if find_latest_run(Path(d)) is not None:
            raise AssertionError("absence of any run must yield None")


def test_resolve_model_precedence() -> None:
    config = {"model": {"size": "m", "weights": None}}
    with tempfile.TemporaryDirectory() as d:
        runs = Path(d)
        ref, label = resolve_model(model=None, latest=False, config=config, runs_dir=runs)
        if ref != "yolo26m.pt":
            raise AssertionError(f"default must be the base checkpoint, got {ref!r}")
        if "base" not in label:
            raise AssertionError(f"default label must mention the base source, got {label!r}")
        weights = runs / "custom.pt"
        weights.write_bytes(b"x")
        ref2, _ = resolve_model(model=str(weights), latest=False, config=config, runs_dir=runs)
        if ref2 != str(weights):
            raise AssertionError("an explicit --model path must take precedence")


def test_resolve_model_errors() -> None:
    config = {"model": {"size": "m"}}
    with tempfile.TemporaryDirectory() as d:
        runs = Path(d)
        try:
            resolve_model(model="/no/such/model.pt", latest=False, config=config, runs_dir=runs)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("a missing --model path must raise FileNotFoundError")
        try:
            resolve_model(model=None, latest=True, config=config, runs_dir=runs)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("--latest with no runs must raise FileNotFoundError")


def test_resolve_model_select_uses_selector() -> None:
    config = {"model": {"size": "m"}}
    with tempfile.TemporaryDirectory() as d:
        runs = Path(d)
        weights = runs / "run" / "weights" / "best.onnx"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"x")

        def selector(choices: list[str]) -> list[str]:
            return choices[:1]

        ref, label = resolve_model(
            model=None, latest=False, select=True, config=config, runs_dir=runs, selector=selector
        )
        if ref != str(weights):
            raise AssertionError(f"--select must return the picked path, got {ref!r}")
        if not label.startswith("selected"):
            raise AssertionError(f"--select label must start with 'selected', got {label!r}")


def test_resolve_model_select_no_candidates_errors() -> None:
    config = {"model": {"size": "m"}}
    with tempfile.TemporaryDirectory() as d:
        try:
            resolve_model(model=None, latest=False, select=True, config=config, runs_dir=Path(d))
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("--select with no candidates must raise FileNotFoundError")


def test_resolve_model_select_cancel_errors() -> None:
    config = {"model": {"size": "m"}}
    with tempfile.TemporaryDirectory() as d:
        runs = Path(d)
        weights = runs / "run" / "weights" / "best.pt"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"x")
        try:
            resolve_model(
                model=None,
                latest=False,
                select=True,
                config=config,
                runs_dir=runs,
                selector=lambda _choices: [],
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("cancelling --select must raise RuntimeError")


def test_tile_images_shape() -> None:
    imgs = [np.zeros((10, 20, 3), dtype=np.uint8) for _ in range(3)]
    sheet = tile_images(imgs, cols=2, cell=32)
    if sheet.shape != (64, 64, 3):
        raise AssertionError(f"unexpected contact-sheet shape {sheet.shape}")


def test_make_output_dir_uses_base() -> None:
    with tempfile.TemporaryDirectory() as d:
        base = Path(d) / "out"
        result = make_output_dir(base)
        if result != base or not result.is_dir():
            raise AssertionError("make_output_dir(base) must create and return the base dir")


def test_discover_models_finds_pt_and_onnx() -> None:
    with tempfile.TemporaryDirectory() as d:
        runs = Path(d)
        pt = runs / "a" / "train" / "weights" / "best.pt"
        onnx = runs / "b" / "weights" / "best.onnx"
        pt.parent.mkdir(parents=True)
        onnx.parent.mkdir(parents=True)
        pt.write_bytes(b"x")
        onnx.write_bytes(b"x")
        (runs / "notes.txt").write_text("nope")
        os.utime(pt, (1, 1))
        os.utime(onnx, (10_000_000, 10_000_000))
        found = discover_models(runs)
        if found != [onnx, pt]:
            raise AssertionError(f"expected newest-first [onnx, pt], got {found}")


def test_discover_models_empty_when_absent() -> None:
    with tempfile.TemporaryDirectory() as d:
        if discover_models(Path(d) / "missing") != []:
            raise AssertionError("a missing root must yield []")


def test_format_model_choice_line() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "best.pt"
        p.write_bytes(b"x" * 1536)  # 1.5 KB
        line = format_model_choice(p)
        if line != f"best.pt : {p}  [1.5 KB]":
            raise AssertionError(f"unexpected choice line: {line!r}")
