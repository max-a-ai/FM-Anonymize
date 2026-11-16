"""Privacy metrics for blur masks against face / plate ground truth.

Detection AP does not fit anonymisation: what matters is whether the
identifying pixels end up blurred, whatever shape the blur has (a mesh
hull, an ellipse, a window mask). So every method is scored on the blur
mask it actually applies:

* **protected**: a ground-truth box is protected when at least ``tau``
  of its pixels are blurred; *privacy recall* = protected / all.
* **over-blur**: share of the image blurred outside every ground-truth
  box (boxes grown by ``grow`` to forgive tight annotations).
* **false regions**: blurred regions with less than ``min_hit`` of their
  area on a (grown) ground-truth box.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from blur_data.utils.masks import BoolMask

#: object size bins on sqrt(box area), pixels
SIZE_BINS = ((0, 16), (16, 32), (32, 64), (64, 1e9))


@dataclass(frozen=True)
class GtBox:
    """One annotated face or plate, pixel corners (x1, y1 exclusive)."""

    cls: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def size(self) -> float:
        """sqrt of the box area, pixels."""
        return float(
            np.sqrt(max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0))
        )

    def window(
        self, shape: tuple[int, int], grow: float = 0.0
    ) -> tuple[int, int, int, int]:
        """Integer window clipped to an ``(h, w)`` image."""
        h, w = shape
        gx = grow * (self.x1 - self.x0)
        gy = grow * (self.y1 - self.y0)
        x0 = int(np.clip(np.floor(self.x0 - gx), 0, w))
        y0 = int(np.clip(np.floor(self.y0 - gy), 0, h))
        x1 = int(np.clip(np.ceil(self.x1 + gx), 0, w))
        y1 = int(np.clip(np.ceil(self.y1 + gy), 0, h))
        return x0, y0, x1, y1


def box_coverage(gt: GtBox, blur: BoolMask) -> float:
    """Fraction of the ground-truth box that is blurred (0 if empty)."""
    x0, y0, x1, y1 = gt.window((int(blur.shape[0]), int(blur.shape[1])))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(blur[y0:y1, x0:x1].mean())


def gt_mask(
    gts: Sequence[GtBox], shape: tuple[int, int], grow: float
) -> BoolMask:
    """Union of the (grown) ground-truth boxes."""
    out = np.zeros(shape, dtype=bool)
    for g in gts:
        x0, y0, x1, y1 = g.window(shape, grow)
        out[y0:y1, x0:x1] = True
    return out


@dataclass
class ImageScore:
    """Scores of one method on one image."""

    coverages: list[tuple[str, float, float]]  # (cls, size, coverage)
    over_blur: float  # share of the image
    false_regions: int
    regions: int


def score_image(
    gts: Sequence[GtBox],
    blur: BoolMask,
    region_masks: Sequence[BoolMask],
    grow: float = 0.25,
    min_hit: float = 0.1,
) -> ImageScore:
    """Score one image: ``blur`` is the union of ``region_masks``."""
    shape = (int(blur.shape[0]), int(blur.shape[1]))
    covered = gt_mask(gts, shape, grow)
    false = 0
    for m in region_masks:
        n = int(m.sum())
        if n and float(np.logical_and(m, covered).sum()) / n < min_hit:
            false += 1
    return ImageScore(
        coverages=[(g.cls, g.size, box_coverage(g, blur)) for g in gts],
        over_blur=float(np.logical_and(blur, ~covered).mean()),
        false_regions=false,
        regions=len(region_masks),
    )


def recall(
    scores: Sequence[ImageScore],
    cls: str,
    tau: float,
    size: tuple[float, float] | None = None,
) -> tuple[float, int]:
    """Privacy recall of ``cls`` at threshold ``tau`` (and its count)."""
    cov = [
        c
        for s in scores
        for k, sz, c in s.coverages
        if k == cls and (size is None or size[0] <= sz < size[1])
    ]
    if not cov:
        return float("nan"), 0
    return float(np.mean(np.asarray(cov) >= tau)), len(cov)
