"""Plot helpers for the notebooks."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import numpy.typing as npt

from blur_data.utils.masks import BoolMask

U8 = npt.NDArray[np.uint8]

#: one colour per overlay layer (RGB). The six region kinds use the
#: categorical slots 1-6 of the dataviz reference palette in fixed order
#: (validated adjacent-pair CVD safe), so charts and overlays agree.
PALETTE = {
    "plate": (0x2A, 0x78, 0xD6),  # blue
    "head": (0xEB, 0x68, 0x34),  # orange
    "head_fallback": (0x1B, 0xAF, 0x7A),  # aqua
    "face": (0xED, 0xA1, 0x00),  # yellow
    "occupant": (0xE8, 0x7B, 0xA4),  # magenta (slot 5)
    "window": (0x00, 0x83, 0x00),  # green (slot 6)
    "away": (230, 0, 0),
    "vehicle": (180, 180, 180),
    "person": (255, 0, 255),
}


def overlay(
    img: U8,
    masks: Sequence[BoolMask],
    color: tuple[int, int, int],
    alpha: float = 0.45,
    outline: int = 3,
) -> U8:
    """Tint masks and draw their outline."""
    out = img.copy()
    for m in masks:
        out[m] = (
            (1 - alpha) * out[m] + alpha * np.asarray(color, np.float32)
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, color, outline)
    return out


def draw_boxes(
    img: U8,
    boxes: Sequence[Sequence[float]],
    color: tuple[int, int, int],
    labels: Sequence[str] | None = None,
    thickness: int = 3,
) -> U8:
    """Rectangles with optional text labels."""
    out = img.copy()
    for i, b in enumerate(boxes):
        p0 = (int(b[0]), int(b[1]))
        p1 = (int(b[2]), int(b[3]))
        cv2.rectangle(out, p0, p1, color, thickness)
        if labels:
            cv2.putText(
                out,
                labels[i],
                (p0[0], max(20, p0[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )
    return out


def crop_box(
    shape: tuple[int, ...], box: Sequence[float], pad: float, square: bool
) -> tuple[int, int, int, int]:
    """Integer window of :func:`crop` (x0, y0, x1, y1)."""
    h, w = shape[:2]
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = (x1 - x0) * (1 + 2 * pad), (y1 - y0) * (1 + 2 * pad)
    if square:
        bw = bh = max(bw, bh, 32)
    return (
        int(max(0, cx - bw / 2)),
        int(max(0, cy - bh / 2)),
        int(min(w, cx + bw / 2)),
        int(min(h, cy + bh / 2)),
    )


def crop(img: U8, box: Sequence[float], pad: float, square: bool) -> U8:
    """Padded crop around a box, optionally squared."""
    a0, b0, a1, b1 = crop_box(img.shape, box, pad, square)
    out: U8 = img[b0:b1, a0:a1]
    return out
