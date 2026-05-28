"""Pure-stdlib coordinate math for the Label Studio ML backend.

Kept dependency-free (no torch / label-studio-ml) so the conversion from the
model's pixel output to Label Studio's percent schema is unit-testable without
loading the heavy ML stack.
"""

from __future__ import annotations

from typing import Any

Xyxy = tuple[float, float, float, float]


def rescale_xyxy(xyxy: Xyxy, src_size: float, dst_w: float, dst_h: float) -> Xyxy:
    """Rescale a box from the square ``src_size`` model space to ``dst_w``x``dst_h``.

    The EfficientNet backbone regresses pixel coordinates in the resized
    ``src_size``x``src_size`` input space; this maps them back onto the original
    image dimensions.
    """
    sx = dst_w / src_size
    sy = dst_h / src_size
    x1, y1, x2, y2 = xyxy
    return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def xyxy_to_ls_value(xyxy: Xyxy, img_w: float, img_h: float, label: str) -> dict[str, Any]:
    """Convert a pixel-space xyxy box (original image dims) to LS percent schema.

    Coordinates are clamped to ``[0, 100]`` so a slightly out-of-frame prediction
    still yields a valid rectangle.
    """
    x1, y1, x2, y2 = xyxy
    x = _clamp(x1 / img_w * 100.0, 0.0, 100.0)
    y = _clamp(y1 / img_h * 100.0, 0.0, 100.0)
    w = _clamp((x2 - x1) / img_w * 100.0, 0.0, 100.0 - x)
    h = _clamp((y2 - y1) / img_h * 100.0, 0.0, 100.0 - y)
    return {
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "rotation": 0,
        "rectanglelabels": [label],
    }


def build_prediction(
    xyxy_model: Xyxy,
    *,
    model_input_size: float,
    img_w: float,
    img_h: float,
    from_name: str,
    to_name: str,
    label: str,
    model_version: str,
    score: float,
) -> dict[str, Any]:
    """Build a complete Label Studio prediction dict from a raw model box.

    ``xyxy_model`` is in the ``model_input_size``-square space; it is rescaled to
    the original ``img_w``x``img_h`` image before percent conversion.
    """
    xyxy = rescale_xyxy(xyxy_model, model_input_size, img_w, img_h)
    value = xyxy_to_ls_value(xyxy, img_w, img_h, label)
    return {
        "model_version": model_version,
        "score": score,
        "result": [
            {
                "from_name": from_name,
                "to_name": to_name,
                "type": "rectanglelabels",
                "original_width": int(img_w),
                "original_height": int(img_h),
                "image_rotation": 0,
                "value": value,
            }
        ],
    }
