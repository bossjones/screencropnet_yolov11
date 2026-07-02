"""Torch-free pure core for the ``screencrop-media`` CLI.

Everything here is deterministic and dependency-light — no torch, no Ultralytics,
no cv2, no ffmpeg subprocess. Input discovery, geometry, output-path mapping,
ffmpeg/ffprobe argv construction, and metadata parsing all live here so they can be
unit-tested directly (inline ``## Tests`` below) without spawning a process or a model.

The IO/async/CLI shell that actually reads pixels, runs ffmpeg, and drives the YOLO
pipeline lives in ``media.py``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

MEDIA_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
"""Image extensions (compared case-insensitively via ``suffix.lower()``)."""

MEDIA_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".gif"}
"""Video extensions (compared case-insensitively)."""

DEFAULT_SIZE: tuple[int, int] = (1080, 1350)
"""Default resize target: portrait 4:5 (matches the old ``autoresize`` large path)."""

SQUARE_SIZE: tuple[int, int] = (1080, 1080)
"""``--square`` resize target."""

DARK_BG: tuple[int, int, int] = (22, 32, 42)
"""Discord dark-mode background ``#16202A`` (opaque; the old cog's alpha 1 was a bug)."""

WHITE_BG: tuple[int, int, int] = (255, 255, 255)
"""Light background used when the source's top-left pixel is pure white."""

TRACE_LEVEL = 5
"""Custom sub-DEBUG log level for ``-vvv``/``-vvvv``."""

Box = tuple[int, int, int, int]
"""A pixel-space bounding box ``(x1, y1, x2, y2)``."""


class HasBox(Protocol):
    """Structural type for anything with a confidence + xyxy box (e.g. ``inference.Detection``)."""

    @property
    def confidence(self) -> float: ...

    @property
    def bbox(self) -> tuple[float, float, float, float]: ...


def expand_input(raw: str) -> Path:
    """Expand ``~`` and ``$VAR`` in ``raw`` and return a ``Path`` (not resolved to absolute)."""
    return Path(os.path.expandvars(os.path.expanduser(raw)))


def classify_kind(path: Path) -> str:
    """Classify ``path`` as ``"image"``, ``"video"``, or ``"other"`` by (lowercased) extension."""
    suffix = path.suffix.lower()
    if suffix in MEDIA_IMAGE_EXTS:
        return "image"
    if suffix in MEDIA_VIDEO_EXTS:
        return "video"
    return "other"


def is_media(path: Path) -> bool:
    """True if ``path`` is a recognized image or video by extension."""
    return classify_kind(path) != "other"


def discover_inputs(root: Path, depth: int = 0) -> list[Path]:
    """Media files under ``root``, sorted by path.

    A file ``root`` yields ``[root]`` when it's media, else ``[]``. A directory is
    walked to ``depth`` levels below it (``depth=0`` → direct children only), keeping
    only image/video files. Deduplicated and deterministically sorted.
    """
    if root.is_file():
        return [root] if is_media(root) else []
    if not root.is_dir():
        return []
    found: set[Path] = set()
    for p in root.rglob("*"):
        if not p.is_file() or not is_media(p):
            continue
        if len(p.relative_to(root).parts) - 1 <= depth:
            found.add(p)
    return sorted(found, key=lambda p: str(p))


def parse_size(spec: str) -> tuple[int, int]:
    """Parse a ``"WxH"`` size spec (e.g. ``"1080x1350"``) into ``(width, height)``."""
    lowered = spec.lower().replace(" ", "")
    if "x" not in lowered:
        raise ValueError(f"size must look like WxH (e.g. 1080x1350), got {spec!r}")
    w_str, _, h_str = lowered.partition("x")
    try:
        width, height = int(w_str), int(h_str)
    except ValueError as exc:
        raise ValueError(f"size must be two integers WxH, got {spec!r}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"size dimensions must be positive, got {spec!r}")
    return width, height


def resolve_target_size(
    *,
    size: str | None = None,
    width: int | None = None,
    height: int | None = None,
    square: bool = False,
) -> tuple[int, int]:
    """Resolve the resize target from the mutually-constrained size flags.

    ``--size`` may not be combined with ``--width``/``--height``/``--square``. With no
    flags the default is ``DEFAULT_SIZE``; ``--square`` is ``SQUARE_SIZE``; a lone
    ``--width``/``--height`` fills the missing side from ``DEFAULT_SIZE``.
    """
    dims_given = width is not None or height is not None
    if size is not None and (dims_given or square):
        raise ValueError("--size cannot be combined with --width/--height/--square")
    if square and dims_given:
        raise ValueError("--square cannot be combined with --width/--height")
    if size is not None:
        return parse_size(size)
    if square:
        return SQUARE_SIZE
    if dims_given:
        return (
            width if width is not None else DEFAULT_SIZE[0],
            height if height is not None else DEFAULT_SIZE[1],
        )
    return DEFAULT_SIZE


def parse_hex_color(spec: str) -> tuple[int, int, int]:
    """Parse ``#RRGGBB`` / ``0xRRGGBB`` / ``RRGGBB`` into an ``(r, g, b)`` tuple."""
    s = spec.strip().lower()
    if s.startswith("#"):
        s = s[1:]
    elif s.startswith("0x"):
        s = s[2:]
    if len(s) != 6:
        raise ValueError(f"hex color must be 6 digits (RRGGBB), got {spec!r}")
    try:
        r, g, b = (int(s[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError as exc:
        raise ValueError(f"invalid hex color {spec!r}") from exc
    return r, g, b


def pick_bg_from_pixel(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Choose a background from the source's top-left pixel: white → white, else dark."""
    return WHITE_BG if rgb == WHITE_BG else DARK_BG


def resolve_bg_color(
    override: str | None, top_left_rgb: tuple[int, int, int]
) -> tuple[int, int, int]:
    """The background fill: an explicit ``--bg-color`` override, else auto from the pixel."""
    if override is not None:
        return parse_hex_color(override)
    return pick_bg_from_pixel(top_left_rgb)


def to_ffmpeg_color(rgb: tuple[int, int, int]) -> str:
    """Render an ``(r, g, b)`` tuple as ffmpeg's ``0xRRGGBB`` color literal."""
    r, g, b = rgb
    return f"0x{r:02X}{g:02X}{b:02X}"


def pick_crop_box(detections: Sequence[HasBox]) -> Box | None:
    """Return the highest-confidence detection's box as ints, or ``None`` if empty."""
    if not detections:
        return None
    best = max(detections, key=lambda d: d.confidence)
    x1, y1, x2, y2 = best.bbox
    return int(x1), int(y1), int(x2), int(y2)


def union_boxes(boxes: Sequence[Box]) -> Box | None:
    """The smallest box enclosing all ``boxes`` (used to fix a static crop over video)."""
    if not boxes:
        return None
    xs1, ys1, xs2, ys2 = zip(*boxes, strict=True)
    return min(xs1), min(ys1), max(xs2), max(ys2)


def clamp_box(box: Box, width: int, height: int) -> Box:
    """Clamp ``box`` to the ``width``×``height`` image; raise if it has no positive area."""
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), width))
    y1 = max(0, min(int(y1), height))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"crop box has no area after clamping: {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


@dataclass(frozen=True)
class FitGeometry:
    """How a source fits, aspect-preserved and centered, onto a destination canvas."""

    scaled_w: int
    scaled_h: int
    offset_x: int
    offset_y: int


def fit_geometry(src_w: int, src_h: int, dst_w: int, dst_h: int) -> FitGeometry:
    """Aspect-preserving fit of ``src`` into ``dst``: scaled size + centered offset."""
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"source dimensions must be positive, got {(src_w, src_h)}")
    scale = min(dst_w / src_w, dst_h / src_h)
    scaled_w = max(1, round(src_w * scale))
    scaled_h = max(1, round(src_h * scale))
    return FitGeometry(scaled_w, scaled_h, (dst_w - scaled_w) // 2, (dst_h - scaled_h) // 2)


def output_prefix(*, cropped: bool, resized: bool) -> str:
    """Artifact filename prefix for the crop/resize combination."""
    if cropped and resized:
        return "cropped-resized-"
    if cropped:
        return "cropped-"
    if resized:
        return "resized-"
    return ""


def output_path_for(
    input_path: Path, input_root: Path, output_dir: Path, *, cropped: bool, resized: bool
) -> Path:
    """Map an input to its artifact path, mirroring the subtree under ``output_dir``.

    ``input_root`` is the directory the walk started from (the file's parent for a
    single-file input); the input's path relative to it is preserved under ``output_dir``,
    with the basename prefixed by the crop/resize combination.
    """
    rel = input_path.relative_to(input_root)
    prefix = output_prefix(cropped=cropped, resized=resized)
    return output_dir / rel.parent / f"{prefix}{input_path.name}"


def output_would_clobber_input(input_root: Path, output_dir: Path) -> bool:
    """True if ``output_dir`` is the input dir or nested inside it (would risk the source)."""
    out = output_dir.expanduser().resolve()
    root = input_root.expanduser().resolve()
    return out == root or root in out.parents


def verbosity_to_level(count: int) -> int:
    """Map a ``-v`` repeat count to a logging level: 0→WARNING … 3+→TRACE."""
    if count <= 0:
        return logging.WARNING
    if count == 1:
        return logging.INFO
    if count == 2:
        return logging.DEBUG
    return TRACE_LEVEL


def build_ffmpeg_filter(
    *, crop: Box | None, size: tuple[int, int] | None, bg: tuple[int, int, int]
) -> str:
    """Build the ffmpeg ``-vf`` filter chain for an optional crop then optional fit-and-pad."""
    parts: list[str] = []
    if crop is not None:
        x1, y1, x2, y2 = crop
        parts.append(f"crop={x2 - x1}:{y2 - y1}:{x1}:{y1}")
    if size is not None:
        w, h = size
        color = to_ffmpeg_color(bg)
        parts.append(f"scale={w}:{h}:force_original_aspect_ratio=decrease")
        parts.append(f"pad=width={w}:height={h}:x=-1:y=-1:color={color}")
    return ",".join(parts)


def build_ffmpeg_crop_resize_args(
    src: Path,
    dst: Path,
    *,
    crop: Box | None,
    size: tuple[int, int] | None,
    bg: tuple[int, int, int],
) -> list[str]:
    """Full ffmpeg argv to crop/resize ``src`` → ``dst`` (x264 video, audio copied)."""
    vf = build_ffmpeg_filter(crop=crop, size=size, bg=bg)
    args = ["ffmpeg", "-y", "-i", str(src)]
    if vf:
        args += ["-vf", vf]
    args += [
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        str(dst),
    ]
    return args


def build_ffprobe_args(src: Path) -> list[str]:
    """ffprobe argv that emits stream + format metadata as JSON."""
    return [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=width,height,codec_name,codec_type,duration,bit_rate,r_frame_rate",
        "-show_entries",
        "format=duration,bit_rate,size",
        "-of",
        "json",
        str(src),
    ]


@dataclass(frozen=True)
class InfoRow:
    """One ``--info`` row: metadata about a single input file."""

    path: str
    kind: str
    width: int | None = None
    height: int | None = None
    duration: float | None = None
    fps: float | None = None
    codec: str | None = None
    bit_rate: int | None = None
    size_bytes: int | None = None


def _parse_fps(rate: str | None) -> float | None:
    """Parse an ffprobe ``r_frame_rate`` like ``"30000/1001"`` into frames per second."""
    if not rate or rate == "0/0":
        return None
    num, _, den = rate.partition("/")
    try:
        n, d = float(num), float(den) if den else 1.0
    except ValueError:
        return None
    return n / d if d else None


def parse_ffprobe_json(
    path: Path, kind: str, data: dict[str, Any], size_bytes: int | None
) -> InfoRow:
    """Turn ffprobe's parsed JSON into an :class:`InfoRow` (first video stream wins)."""
    streams: list[dict[str, Any]] = data.get("streams") or []
    fmt: dict[str, Any] = data.get("format") or {}
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        video = streams[0] if streams else {}
    duration = video.get("duration") or fmt.get("duration")
    bit_rate = video.get("bit_rate") or fmt.get("bit_rate")
    return InfoRow(
        path=str(path),
        kind=kind,
        width=int(video["width"]) if video.get("width") is not None else None,
        height=int(video["height"]) if video.get("height") is not None else None,
        duration=float(duration) if duration is not None else None,
        fps=_parse_fps(video.get("r_frame_rate")),
        codec=video.get("codec_name"),
        bit_rate=int(bit_rate) if bit_rate is not None else None,
        size_bytes=size_bytes,
    )


def image_info_row(path: Path, width: int, height: int, size_bytes: int | None) -> InfoRow:
    """Build an :class:`InfoRow` for an image (dimensions come from a cv2 read upstream)."""
    return InfoRow(path=str(path), kind="image", width=width, height=height, size_bytes=size_bytes)


## Tests

import tempfile  # noqa: E402
from typing import NamedTuple  # noqa: E402

_FIXTURES = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures"


class _FakeDet(NamedTuple):
    confidence: float
    bbox: tuple[float, float, float, float]


def test_classify_kind_is_case_insensitive_on_real_fixtures() -> None:
    png = _FIXTURES / "00007_twitter.PNG"
    mp4 = _FIXTURES / "AlertesInfos-2071001868927471827-(20260627_224457)-img1.mp4"
    if not png.exists() or not mp4.exists():
        raise AssertionError(f"expected fixtures under {_FIXTURES}")
    if classify_kind(png) != "image":
        raise AssertionError("uppercase .PNG must classify as image (case-insensitive)")
    if classify_kind(mp4) != "video":
        raise AssertionError(".mp4 must classify as video")
    if classify_kind(Path("notes.txt")) != "other":
        raise AssertionError(".txt must classify as other")


def test_discover_inputs_on_fixtures_dir() -> None:
    found = discover_inputs(_FIXTURES, depth=0)
    names = {p.name for p in found}
    if "00007_twitter.PNG" not in names:
        raise AssertionError(f"fixtures discovery missed the .PNG screenshots: {names}")
    if not any(p.suffix.lower() == ".mp4" for p in found):
        raise AssertionError("fixtures discovery missed the .mp4 clip")


def test_discover_inputs_single_file() -> None:
    png = _FIXTURES / "00012_twitter.PNG"
    if discover_inputs(png) != [png]:
        raise AssertionError("a single media file must yield [itself]")
    if discover_inputs(Path(__file__)) != []:
        raise AssertionError("a non-media file must yield []")


def test_discover_inputs_respects_depth() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "top.png").write_bytes(b"x")
        nested = root / "a" / "b"
        nested.mkdir(parents=True)
        (nested / "deep.png").write_bytes(b"x")
        shallow = {p.name for p in discover_inputs(root, depth=0)}
        if shallow != {"top.png"}:
            raise AssertionError(f"depth=0 must only see direct children, got {shallow}")
        deep = {p.name for p in discover_inputs(root, depth=2)}
        if deep != {"top.png", "deep.png"}:
            raise AssertionError(f"depth=2 must reach the nested image, got {deep}")


def test_resolve_target_size_precedence_and_conflicts() -> None:
    if resolve_target_size() != DEFAULT_SIZE:
        raise AssertionError("no flags must give the default portrait size")
    if resolve_target_size(square=True) != SQUARE_SIZE:
        raise AssertionError("--square must give 1080x1080")
    if resolve_target_size(size="800x600") != (800, 600):
        raise AssertionError("--size must parse WxH")
    if resolve_target_size(width=720) != (720, DEFAULT_SIZE[1]):
        raise AssertionError("lone --width must fill height from the default")
    conflicts = [
        lambda: resolve_target_size(size="1x1", width=2),
        lambda: resolve_target_size(size="1x1", square=True),
        lambda: resolve_target_size(square=True, height=3),
    ]
    for call in conflicts:
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("conflicting size flags must raise")


def test_parse_size_rejects_garbage() -> None:
    for bad in ("1080", "axb", "10x0", "-5x10"):
        try:
            parse_size(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"parse_size({bad!r}) must raise")


def test_resolve_bg_color_pixel_and_override() -> None:
    if resolve_bg_color(None, WHITE_BG) != WHITE_BG:
        raise AssertionError("a white top-left pixel must select the white background")
    if resolve_bg_color(None, (10, 20, 30)) != DARK_BG:
        raise AssertionError("a non-white pixel must select the dark background")
    if resolve_bg_color("#16202A", WHITE_BG) != DARK_BG:
        raise AssertionError("an explicit --bg-color must override the pixel auto-detect")
    if parse_hex_color("0xFF0080") != (255, 0, 128):
        raise AssertionError("parse_hex_color must accept the 0x prefix")


def test_to_ffmpeg_color() -> None:
    if to_ffmpeg_color(DARK_BG) != "0x16202A":
        raise AssertionError(f"dark bg must render as 0x16202A, got {to_ffmpeg_color(DARK_BG)}")


def test_pick_and_clamp_crop_box() -> None:
    dets = [
        _FakeDet(0.4, (0, 0, 10, 10)),
        _FakeDet(0.9, (5, 6, 40, 50)),
        _FakeDet(0.7, (1, 1, 2, 2)),
    ]
    if pick_crop_box(dets) != (5, 6, 40, 50):
        raise AssertionError("pick_crop_box must return the highest-confidence box as ints")
    if pick_crop_box([]) is not None:
        raise AssertionError("pick_crop_box([]) must be None")
    if clamp_box((-5, -5, 30, 30), 20, 25) != (0, 0, 20, 25):
        raise AssertionError("clamp_box must clamp to image bounds")
    try:
        clamp_box((10, 10, 5, 5), 100, 100)
    except ValueError:
        pass
    else:
        raise AssertionError("an inverted box must raise after clamping")


def test_union_boxes() -> None:
    if union_boxes([(10, 10, 20, 20), (5, 30, 12, 40)]) != (5, 10, 20, 40):
        raise AssertionError("union_boxes must enclose all boxes")
    if union_boxes([]) is not None:
        raise AssertionError("union_boxes([]) must be None")


def test_fit_geometry_centers_and_preserves_aspect() -> None:
    geo = fit_geometry(200, 100, 1080, 1350)
    if geo.scaled_w != 1080 or geo.scaled_h != 540:
        raise AssertionError(f"wide source must fit width, got {geo}")
    if geo.offset_x != 0 or geo.offset_y != (1350 - 540) // 2:
        raise AssertionError(f"fit must center vertically, got {geo}")
    try:
        fit_geometry(0, 10, 100, 100)
    except ValueError:
        pass
    else:
        raise AssertionError("a zero source dimension must raise")


def test_output_path_for_mirrors_and_prefixes() -> None:
    root = Path("/in")
    out = Path("/out")
    src = root / "sub" / "a.PNG"
    both = output_path_for(src, root, out, cropped=True, resized=True)
    if both != out / "sub" / "cropped-resized-a.PNG":
        raise AssertionError(f"both flags must mirror + prefix, got {both}")
    crop = output_path_for(src, root, out, cropped=True, resized=False)
    if crop != out / "sub" / "cropped-a.PNG":
        raise AssertionError(f"crop-only prefix wrong, got {crop}")
    passthrough = output_path_for(src, root, out, cropped=False, resized=False)
    if passthrough != out / "sub" / "a.PNG":
        raise AssertionError(f"passthrough must keep the name, got {passthrough}")


def test_output_would_clobber_input() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "input"
        root.mkdir()
        if not output_would_clobber_input(root, root):
            raise AssertionError("output == input must be flagged")
        if not output_would_clobber_input(root, root / "out"):
            raise AssertionError("output nested inside input must be flagged")
        if output_would_clobber_input(root, Path(d) / "elsewhere"):
            raise AssertionError("a sibling output dir must be allowed")


def test_verbosity_to_level() -> None:
    if verbosity_to_level(0) != logging.WARNING:
        raise AssertionError("0 → WARNING")
    if verbosity_to_level(1) != logging.INFO:
        raise AssertionError("1 → INFO")
    if verbosity_to_level(2) != logging.DEBUG:
        raise AssertionError("2 → DEBUG")
    if verbosity_to_level(4) != TRACE_LEVEL:
        raise AssertionError("4 → TRACE")


def test_build_ffmpeg_filter_and_args() -> None:
    vf = build_ffmpeg_filter(crop=(10, 20, 110, 220), size=(1080, 1080), bg=DARK_BG)
    if (
        vf
        != "crop=100:200:10:20,scale=1080:1080:force_original_aspect_ratio=decrease,pad=width=1080:height=1080:x=-1:y=-1:color=0x16202A"
    ):
        raise AssertionError(f"unexpected filter chain: {vf}")
    if build_ffmpeg_filter(crop=None, size=None, bg=DARK_BG) != "":
        raise AssertionError("no crop and no size must yield an empty filter")
    args = build_ffmpeg_crop_resize_args(
        Path("/in.mp4"), Path("/out.mp4"), crop=(0, 0, 10, 10), size=None, bg=DARK_BG
    )
    if args[0] != "ffmpeg" or "-vf" not in args or args[-1] != "/out.mp4":
        raise AssertionError(f"ffmpeg argv malformed: {args}")
    if "copy" not in args:
        raise AssertionError('audio must be copied ("-c:a copy")')


def test_parse_ffprobe_json() -> None:
    data = {
        "streams": [
            {"codec_type": "audio", "codec_name": "aac"},
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1280,
                "height": 720,
                "duration": "12.5",
                "bit_rate": "4200000",
                "r_frame_rate": "30000/1001",
            },
        ],
        "format": {"duration": "12.5", "size": "999"},
    }
    row = parse_ffprobe_json(Path("/clip.mp4"), "video", data, size_bytes=999)
    if (row.width, row.height, row.codec) != (1280, 720, "h264"):
        raise AssertionError(f"video stream not parsed: {row}")
    if row.fps is None or abs(row.fps - 29.97) > 0.01:
        raise AssertionError(f"fps must parse 30000/1001 ≈ 29.97, got {row.fps}")
    if row.bit_rate != 4200000 or row.size_bytes != 999:
        raise AssertionError(f"bit_rate/size not parsed: {row}")


def test_image_info_row() -> None:
    row = image_info_row(Path("/a.png"), 800, 600, 1234)
    if row.kind != "image" or (row.width, row.height, row.size_bytes) != (800, 600, 1234):
        raise AssertionError(f"image_info_row wrong: {row}")
