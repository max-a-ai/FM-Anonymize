"""One image: generate its heads, blur everything else.

:func:`apply_regions` is the whole per-image step of FM-Anonymize-Gen;
the generator (:class:`blur_data.models.face_generator.FaceGenerator`)
is passed in, so this module stays free of model code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from blur_data.anonymize.blur import U8, blur_mask
from blur_data.anonymize.regions import Region
from blur_data.generate.config import GenConfig
from blur_data.generate.crops import (
    FILL,
    Window,
    crop_square,
    model_input,
    paste_square,
    resize,
    square_window,
)
from blur_data.generate.harmonize import harmonize
from blur_data.generate.pose_guide import Guide
from blur_data.generate.routing import (
    blurred_scene,
    fm_blur,
    plan,
    region_seed,
)
from blur_data.utils.masks import BoolMask, mask_box


class Generator(Protocol):
    """Edit ``mask`` of a square uint8 RGB image, seeded.

    ``guide`` steers the head pose (None: no usable body fit); ``seed``
    also picks the synthetic identity, so one seed per track keeps one
    face.
    """

    def __call__(
        self, image: U8, mask: BoolMask, seed: int, guide: Guide | None = None
    ) -> U8: ...


class GuideFn(Protocol):
    """The guide of region ``index`` in crop ``window``; None: unguided."""

    def __call__(
        self, index: int, window: Window, res: int
    ) -> Guide | None: ...


def generate_head(
    out: U8,
    scene: U8,
    start: U8,
    mask: BoolMask,
    generator: Generator,
    seed: int,
    cfg: GenConfig,
    guide: Callable[[Window], Guide | None] | None = None,
    qtables: Any | None = None,
) -> None:
    """Generate one head (grown ``mask``) into ``out`` (in place).

    ``start`` is the image as FM-Anonymize blurs it (the SDEdit start);
    ``scene`` the image with only the blurred regions blurred, which
    the result is blended into. ``guide`` renders the head's ControlNet
    inputs for a crop; ``qtables``: the source's JPEG tables for the
    JPEG matching. ``scene`` and ``start`` get the result too, so a
    later head in the same crop sees it.
    """
    box = mask_box(mask)
    if box is None:
        return
    shape = (int(mask.shape[0]), int(mask.shape[1]))
    x0, y0, side = square_window(box, shape, cfg.context, cfg.min_context)
    crop = crop_square(scene, x0, y0, side, FILL)
    src = crop_square(start, x0, y0, side, FILL)
    m = crop_square(mask, x0, y0, side, False)
    big, gen_mask = model_input(src, m, cfg.resolution)
    g = guide((x0, y0, side)) if guide is not None else None
    result = resize(generator(big, gen_mask, seed, g), side)
    result, changed = harmonize(
        result, crop, m, cfg.blend_levels, seed, qtables
    )
    alpha = changed.astype(np.float32)
    for dst in (out, scene, start):
        paste_square(dst, result, alpha, x0, y0)


def apply_regions(
    img: U8,
    regions: Sequence[Region],
    generator: Generator,
    key: str,
    cfg: GenConfig,
    guides: GuideFn | None = None,
    seeds: Mapping[int, int] | None = None,
    qtables: Any | None = None,
) -> U8:
    """Generate the heads of ``img``, blur its other regions.

    ``regions`` in edit order (``ImageRegions.ordered()``). A head's
    seed is its track's (``seeds``, by index in ``regions``) or else
    derived from ``key`` and its index; ``guides`` supplies the pose
    guide per head.
    """
    blurred, grown = plan(img, regions, cfg)
    scene = blurred_scene(img, blurred, cfg)
    start = fm_blur(img, regions, cfg)
    out = img.copy()
    for i, m in grown.items():
        seed = (seeds or {}).get(i, region_seed(key, i))
        g = _bind(guides, i, cfg.resolution)
        generate_head(out, scene, start, m, generator, seed, cfg, g, qtables)
    for r in blurred:
        out = blur_mask(out, r.mask(), cfg.blur_scale)
    return out


def _bind(
    guides: GuideFn | None, index: int, res: int
) -> Callable[[Window], Guide | None] | None:
    if guides is None:
        return None
    return lambda window: guides(index, window, res)
