"""Binary-mask helpers: RLE codec, dilation, IoU, box of a mask."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from pycocotools import mask as mask_util

BoolMask = npt.NDArray[np.bool_]


def encode_rle(mask: BoolMask) -> dict[str, Any]:
    """COCO RLE of a HxW bool mask, JSON serialisable."""
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": list(rle["size"]), "counts": rle["counts"].decode()}


def decode_rle(rle: dict[str, Any]) -> BoolMask:
    """Inverse of :func:`encode_rle`."""
    coco = {"size": rle["size"], "counts": rle["counts"].encode()}
    out: BoolMask = mask_util.decode(coco).astype(bool)
    return out


def mask_box(mask: BoolMask) -> tuple[int, int, int, int] | None:
    """Tight ``(x0, y0, x1, y1)`` box, exclusive end; None if empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def fill_box(mask: BoolMask) -> BoolMask:
    """The mask's tight box, filled (the mask itself if empty)."""
    box = mask_box(mask)
    if box is None:
        return mask.copy()
    x0, y0, x1, y1 = box
    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = True
    return out


def scale_mask(mask: BoolMask, factor: float) -> BoolMask:
    """The mask together with a copy scaled by ``factor`` about the
    centre of its box (clipped to the image); never smaller."""
    box = mask_box(mask)
    if box is None or factor <= 1.0:
        return mask.copy()
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    nw, nh = max(1, round(w * factor)), max(1, round(h * factor))
    big = cv2.resize(
        mask[y0:y1, x0:x1].astype(np.uint8),
        (nw, nh),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    left = round((x0 + x1 - nw) / 2)
    top = round((y0 + y1 - nh) / 2)
    out = mask.copy()
    img_h, img_w = mask.shape
    cx0, cy0 = max(0, left), max(0, top)
    cx1, cy1 = min(img_w, left + nw), min(img_h, top + nh)
    if cx1 > cx0 and cy1 > cy0:
        out[cy0:cy1, cx0:cx1] |= big[
            cy0 - top : cy1 - top, cx0 - left : cx1 - left
        ]
    return out


def dilate(mask: BoolMask, radius: int) -> BoolMask:
    """Dilate with an elliptic kernel of the given pixel radius."""
    if radius <= 0:
        return mask.copy()
    size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    out: BoolMask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    return out


def mask_iou(a: BoolMask, b: BoolMask) -> float:
    """Intersection over union of two equally sized masks."""
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum()) / union


def coverage(inner: BoolMask, outer: BoolMask) -> float:
    """Fraction of ``inner`` pixels that are also in ``outer``."""
    n = int(inner.sum())
    if n == 0:
        return 1.0
    return float(np.logical_and(inner, outer).sum()) / n
