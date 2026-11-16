"""Which regions are generated, their masks, the blurred scene, seeds.

A region of FM-Anonymize is *generated* when its kind is in
``generate_kinds`` (heads and occupants) and its box is neither below
``min_generate_px`` nor above ``max_generate_frac`` of the image's
short side; every other region (plates, windows, the SAM 3 face net,
box-top fallbacks, outliers) keeps FM-Anonymize's blur.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from blur_data.anonymize.blur import U8, blur_mask
from blur_data.anonymize.blur import apply_regions as blur_regions
from blur_data.anonymize.regions import Region
from blur_data.generate.config import GenConfig
from blur_data.utils.masks import BoolMask, dilate, mask_box

Box = tuple[int, int, int, int]


def long_side(box: Box) -> int:
    """Longer side of an ``(x0, y0, x1, y1)`` box."""
    return max(box[2] - box[0], box[3] - box[1])


def blur_reason(
    region: Region, cfg: GenConfig, shape: tuple[int, int]
) -> str | None:
    """Why a region of an ``(h, w)`` image is blurred; None: generated.

    ``"kind"`` (not a generated kind), ``"small"`` (below
    ``min_generate_px``) or ``"large"`` (above ``max_generate_frac`` of
    the image's short side), checked in this order.
    """
    if region.kind not in cfg.generate_kinds:
        return "kind"
    side = long_side(region.box)
    if side < cfg.min_generate_px:
        return "small"
    if side > cfg.max_generate_frac * min(shape):
        return "large"
    return None


def is_generated(
    region: Region, cfg: GenConfig, shape: tuple[int, int]
) -> bool:
    """Whether a region of an ``(h, w)`` image gets a synthetic head."""
    return blur_reason(region, cfg, shape) is None


def grow_mask(mask: BoolMask, cfg: GenConfig) -> BoolMask:
    """The region mask dilated by ``dilate_frac`` of its long side: the
    whole head plus a margin is regenerated, no original pixel of it
    survives."""
    box = mask_box(mask)
    if box is None:
        return mask
    radius = max(cfg.dilate_min_px, round(cfg.dilate_frac * long_side(box)))
    return dilate(mask, radius)


def plan(
    img: U8, regions: Sequence[Region], cfg: GenConfig
) -> tuple[list[Region], dict[int, BoolMask]]:
    """The blurred regions, and the grown masks of the generated ones
    by their index in ``regions`` (edit order)."""
    shape = (int(img.shape[0]), int(img.shape[1]))
    blurred, grown = [], {}
    for i, r in enumerate(regions):
        if is_generated(r, cfg, shape):
            grown[i] = grow_mask(r.mask(), cfg)
        else:
            blurred.append(r)
    return blurred, grown


def blurred_scene(img: U8, blurred: Sequence[Region], cfg: GenConfig) -> U8:
    """The image with the blurred regions blurred (the context the
    generated heads are blended into)."""
    out = img
    for r in blurred:
        out = blur_mask(out, r.mask(), cfg.blur_scale)
    return out.copy()


def fm_blur(img: U8, regions: Sequence[Region], cfg: GenConfig) -> U8:
    """``img`` as FM-Anonymize's blur renders ``regions``: the SDEdit
    start of every generated head."""
    return blur_regions(img, regions, scale=cfg.blur_scale).copy()


def region_seed(key: str, index: int) -> int:
    """Seed of region ``index`` of image ``key`` (a head without a
    track): derived on the fly, never stored."""
    digest = hashlib.sha256(f"{key}#{index}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & (2**63 - 1)


def track_seed(track: str) -> int:
    """Seed of a track (``sequence/first-frame#region``), never stored:
    every head of the track gets the same synthetic identity."""
    digest = hashlib.sha256(f"track:{track}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & (2**63 - 1)
