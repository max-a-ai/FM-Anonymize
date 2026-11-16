"""Detections of a text prompt and the tiling / merging around them.

SAM3 resizes every input to a 1008x1008 square, so a 2200x1200 frame
is squashed and small objects lose pixels. The segmenter therefore runs
on the full frame *and* on square tiles; the helpers here place tile
results back into the frame and merge duplicates. Pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blur_data.utils.masks import BoolMask, coverage, mask_iou

Box = tuple[float, float, float, float]
Window = tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive)


@dataclass
class Detection:
    """One SAM3 instance in full-frame coordinates."""

    prompt: str
    score: float
    box: Box
    mask: BoolMask
    source: str = "full"  # "full", "tile<i>" or "crop"


def square_tiles(width: int, height: int) -> list[Window]:
    """Overlapping height x height tiles covering a wide frame.

    A frame that is not wider than tall gets no tiles. The tiles are
    spread evenly, so neighbours overlap whenever the width is not a
    multiple of the height.
    """
    if width <= height:
        return []
    n = int(np.ceil(width / height))
    starts = np.linspace(0, width - height, n).round().astype(int)
    return [(int(s), 0, int(s) + height, height) for s in starts]


def touches_inner_edge(
    box: Box, window: Window, frame: tuple[int, int], tol: float = 3.0
) -> bool:
    """True if the box is cut by a window edge that is not a frame edge.

    Such a detection is a truncated object; the full-frame pass or the
    overlapping tile sees it whole, so it is dropped.
    """
    w, h = frame
    x0, y0, x1, y1 = box
    wx0, wy0, wx1, wy1 = window
    return (
        (wx0 > 0 and x0 - wx0 < tol)
        or (wy0 > 0 and y0 - wy0 < tol)
        or (wx1 < w and wx1 - x1 < tol)
        or (wy1 < h and wy1 - y1 < tol)
    )


def place(
    det_mask: BoolMask, window: Window, frame: tuple[int, int]
) -> BoolMask:
    """Paste a window-sized mask into an empty full-frame mask."""
    w, h = frame
    x0, y0, x1, y1 = window
    full = np.zeros((h, w), dtype=bool)
    full[y0:y1, x0:x1] = det_mask
    return full


def dedupe(
    dets: list[Detection], iou_thr: float, inside_thr: float = 0.8
) -> list[Detection]:
    """Greedy mask suppression, highest score first.

    A detection is dropped if its mask overlaps a kept one with IoU
    >= ``iou_thr`` or lies >= ``inside_thr`` inside it (a partial
    duplicate from a tile or crop). Masks, not boxes, are compared so
    two people where one occludes the other both survive.
    """
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.score):
        if all(
            not boxes_overlap(d.box, k.box)
            or (
                mask_iou(d.mask, k.mask) < iou_thr
                and coverage(d.mask, k.mask) < inside_thr
            )
            for k in kept
        ):
            kept.append(d)
    return kept


def boxes_overlap(a: Box, b: Box) -> bool:
    """Do two boxes share any area (1 px slack for rounding)?"""
    return (
        a[0] <= b[2] + 1
        and b[0] <= a[2] + 1
        and a[1] <= b[3] + 1
        and b[1] <= a[3] + 1
    )


def box_inside(inner: Box, outer: Box, pad: float) -> bool:
    """Is the centre of ``inner`` inside ``outer`` grown by ``pad``?"""
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    ow, oh = outer[2] - outer[0], outer[3] - outer[1]
    return (
        outer[0] - pad * ow <= cx <= outer[2] + pad * ow
        and outer[1] - pad * oh <= cy <= outer[3] + pad * oh
    )
