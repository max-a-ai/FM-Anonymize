"""Full-resolution review sheets: original (regions marked) | blurred.

A sheet is one image: a header bar with the frame id and the region
counts, then the original frame with every region that gets blurred
tinted, outlined and labelled (kind + score + edit rank), a gap, and the
blurred frame, both at native resolution.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import cv2
import numpy as np

from blur_data.anonymize.regions import KINDS, Region
from blur_data.utils.viz import PALETTE, U8

HEADER_PX = 70
GAP_PX = 16
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(
    img: U8,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    opacity: float = 0.6,
) -> None:
    """Small text on a see-through box in ``color``, kept on-image.

    The box is blended, so the scene under a label stays visible in
    crowded frames.
    """
    scale, thick = 0.5, 1
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    h, w = img.shape[:2]
    x = int(np.clip(org[0], 0, max(0, w - tw - 4)))
    y = int(np.clip(org[1], th + base + 4, h - 1))
    y0 = y - th - base - 4
    patch = img[y0:y, x : x + tw + 4].astype(np.float32)
    tint = np.asarray(color, np.float32)
    img[y0:y, x : x + tw + 4] = (
        (1 - opacity) * patch + opacity * tint
    ).astype(np.uint8)
    cv2.putText(
        img, text, (x + 2, y - base - 1), FONT, scale, (0, 0, 0), thick
    )


def highlight(img: U8, regions: Sequence[Region], alpha: float = 0.3) -> U8:
    """Tint, outline and label every region on a copy of ``img``.

    ``regions`` must be in edit order; the label shows the 1-based rank
    (the ``X`` of notebook 03 at which the region gets blurred).
    """
    out = img.copy()
    masks = [r.mask() for r in regions]
    for r, m in zip(regions, masks, strict=True):
        tint = np.asarray(PALETTE[r.kind], np.float32)
        out[m] = ((1 - alpha) * out[m] + alpha * tint).astype(np.uint8)
    for r, m in zip(regions, masks, strict=True):
        contours, _ = cv2.findContours(
            m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, PALETTE[r.kind], 2)
    # reverse rank order: the most confident edit's label ends on top
    for rank in range(len(regions), 0, -1):
        r = regions[rank - 1]
        label = f"#{rank} {r.kind} {r.score:.2f}"
        _label(out, label, (r.box[0], r.box[1] - 3), PALETTE[r.kind])
    return out


def review_sheet(
    original: U8, blurred: U8, regions: Sequence[Region], title: str
) -> U8:
    """Header + highlighted original | gap | blurred, native resolution."""
    if original.shape != blurred.shape:
        raise ValueError("original and blurred differ in shape")
    h, w = original.shape[:2]
    sheet = np.full((HEADER_PX + h, 2 * w + GAP_PX, 3), 255, np.uint8)
    sheet[HEADER_PX:, :w] = highlight(original, regions)
    sheet[HEADER_PX:, w + GAP_PX :] = blurred
    counts = Counter(r.kind for r in regions)
    summary = ", ".join(f"{k} {n}" for k, n in sorted(counts.items()))
    cv2.putText(
        sheet,
        f"{title}   |   {len(regions)} regions: {summary or 'none'}",
        (12, 30),
        FONT,
        0.9,
        (0, 0, 0),
        2,
    )
    cv2.putText(
        sheet,
        "ORIGINAL (marked = blurred)",
        (12, 62),
        FONT,
        0.8,
        (80, 80, 80),
        2,
    )
    cv2.putText(
        sheet, "ANONYMISED", (w + GAP_PX + 12, 62), FONT, 0.8, (80, 80, 80), 2
    )
    x = w - 12
    for kind in reversed(KINDS):
        (tw, _), _ = cv2.getTextSize(kind, FONT, 0.8, 2)
        x -= tw + 40
        cv2.rectangle(sheet, (x, 46), (x + 22, 66), PALETTE[kind], -1)
        cv2.putText(sheet, kind, (x + 28, 64), FONT, 0.8, (0, 0, 0), 2)
    return sheet
