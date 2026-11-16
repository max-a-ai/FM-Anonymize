"""Blur regions into an image.

Each region is blurred with a Gaussian whose sigma scales with the
region's size, so a plate and a near head are both unreadable. Inside
the mask the result is fully blurred; a thin feather is added only
*outside* the mask, so coverage never shrinks. Plates can instead be
filled with a solid colour (``plate_fill``): a fill cannot be
de-blurred.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image, JpegImagePlugin

from blur_data.anonymize.regions import Region
from blur_data.utils.masks import BoolMask, mask_box

U8 = npt.NDArray[np.uint8]
RGB = tuple[int, int, int]

#: Plate fill of FM-Anonymize (``AnonConfig.plate_fill == "white"``).
WHITE: RGB = (255, 255, 255)


def blur_sigma(box: tuple[int, int, int, int], scale: float) -> float:
    """Sigma for a region box: ``scale`` x its longer side, >= 4 px."""
    x0, y0, x1, y1 = box
    return max(4.0, scale * max(x1 - x0, y1 - y0))


def blur_mask(img: U8, mask: BoolMask, scale: float) -> U8:
    """Return ``img`` with ``mask`` blurred (HxWx3 uint8, any order)."""
    box = mask_box(mask)
    if box is None:
        return img
    sigma = blur_sigma(box, scale)
    pad = int(3 * sigma) + 2
    h, w = mask.shape
    x0, y0 = max(0, box[0] - pad), max(0, box[1] - pad)
    x1, y1 = min(w, box[2] + pad), min(h, box[3] + pad)
    crop = img[y0:y1, x0:x1].astype(np.float32)
    blurred = cv2.GaussianBlur(crop, (0, 0), sigma)
    m = mask[y0:y1, x0:x1].astype(np.float32)
    feather = cv2.GaussianBlur(m, (0, 0), max(1.0, sigma / 8))
    alpha = np.maximum(m, feather)[..., None]
    mixed = alpha * blurred + (1.0 - alpha) * crop
    out = img.copy()
    out[y0:y1, x0:x1] = np.clip(np.round(mixed), 0, 255).astype(np.uint8)
    return out


def fill_mask(img: U8, mask: BoolMask, color: RGB) -> U8:
    """Return ``img`` with ``mask`` painted in ``color``."""
    out = img.copy()
    out[mask] = color
    return out


def apply_regions(
    img: U8,
    regions: Sequence[Region],
    limit: int | None = None,
    scale: float = 0.3,
    plate_fill: RGB | None = None,
) -> U8:
    """Blur the first ``limit`` regions (all if None) in given order;
    ``plate`` regions are filled with ``plate_fill`` if it is given."""
    chosen = regions if limit is None else regions[: max(0, limit)]
    out = img
    for region in chosen:
        if plate_fill is not None and region.kind == "plate":
            out = fill_mask(out, region.mask(), plate_fill)
        else:
            out = blur_mask(out, region.mask(), scale)
    return out


def plate_fill_of(cfg: dict[str, object]) -> RGB | None:
    """The plate fill a run's ``config.json`` asks for: white unless it
    says ``plate_fill: blur`` (runs before the option are filled too)."""
    return None if cfg.get("plate_fill", "white") == "blur" else WHITE


def save_jpeg(
    out: Image.Image, dst: Path, source: Image.Image, quality: str
) -> None:
    """Save ``out`` like ``source`` ("keep") or at a fixed quality."""
    if quality != "keep":
        out.save(dst, quality=int(quality))
        return
    out.save(
        dst,
        qtables=source.quantization,  # type: ignore[attr-defined]
        subsampling=JpegImagePlugin.get_sampling(source),
    )
