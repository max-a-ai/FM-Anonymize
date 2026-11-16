"""Square crops with context: cut, upscale to the model, paste back.

The generator works on a square crop around each head (``context`` x
the head box's long side), upscaled to the model resolution; its result
is downscaled and pasted back. Crops are padded with a constant grey,
never reflected, so nothing outside the image is invented.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from blur_data.anonymize.blur import U8
from blur_data.generate.routing import Box, long_side
from blur_data.utils.masks import BoolMask, dilate

F32 = npt.NDArray[np.float32]
#: a crop: ``(x0, y0, side)`` in image pixels
Window = tuple[int, int, int]
#: padding grey of crops that reach over the image border
FILL = 127
#: VAE downsampling factor: the generator's mask is grown by one latent
#: cell so the masked area has no hard edge in latent space
LATENT_PX = 8


def square_window(
    box: Box, shape: tuple[int, int], context: float, min_context: float
) -> Window:
    """Square crop ``(x0, y0, side)`` around ``box`` with context.

    The side is ``context`` x the box's long side, but at most the
    image's short side (and never below ``min_context`` x the long
    side). A window that fits is shifted into the image; a larger one
    stays centred and is padded by :func:`crop_square`.
    """
    h, w = shape
    long = long_side(box)
    side = min(context * long, max(min_context * long, min(h, w)))
    s = max(1, int(np.ceil(side)))
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return _place(cx, s, w), _place(cy, s, h), s


def _place(centre: float, side: int, n: int) -> int:
    start = int(round(centre - side / 2))
    if side <= n:
        return min(max(start, 0), n - side)
    return start


def crop_square(
    a: npt.NDArray[Any], x0: int, y0: int, side: int, fill: Any
) -> npt.NDArray[Any]:
    """``side`` x ``side`` window of ``a``; outside the image: ``fill``."""
    h, w = a.shape[:2]
    out = np.full((side, side, *a.shape[2:]), fill, dtype=a.dtype)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, w), min(y0 + side, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = a[sy0:sy1, sx0:sx1]
    return out


def paste_square(dst: U8, patch: U8, alpha: F32, x0: int, y0: int) -> None:
    """Blend ``patch`` into ``dst`` (in place) where it is in the image."""
    h, w = dst.shape[:2]
    side = patch.shape[0]
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, w), min(y0 + side, h)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    win = (slice(sy0 - y0, sy1 - y0), slice(sx0 - x0, sx1 - x0))
    a = alpha[win][..., None]
    old = dst[sy0:sy1, sx0:sx1].astype(np.float32)
    mixed = a * patch[win].astype(np.float32) + (1.0 - a) * old
    dst[sy0:sy1, sx0:sx1] = np.clip(np.round(mixed), 0, 255).astype(np.uint8)


def resize(a: U8, side: int) -> U8:
    """Square resize: area filter when shrinking, cubic when growing."""
    interp = cv2.INTER_AREA if a.shape[0] > side else cv2.INTER_CUBIC
    out = cv2.resize(a, (side, side), interpolation=interp)
    return np.asarray(out, dtype=np.uint8)


def model_input(crop: U8, mask: BoolMask, res: int) -> tuple[U8, BoolMask]:
    """The crop and its generation mask at the model resolution.

    The crop is the SDEdit start (FM-Anonymize's blur), so it is passed
    as it is; the mask is resized conservatively (any cover counts) and
    grown by one latent cell.
    """
    big = resize(crop.copy(), res)
    m = cv2.resize(
        mask.astype(np.float32), (res, res), interpolation=cv2.INTER_LINEAR
    )
    return big, dilate(m > 0, LATENT_PX)
