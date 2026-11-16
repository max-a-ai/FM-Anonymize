"""Precision / recall and confusion matrices for blur regions vs GT.

Two levels, per class (face, plate):

* **object level** -- a GT box is a TP when >= ``tau`` of it is blurred
  by the method (any region kind), otherwise an FN. A predicted region
  of the class is an FP when < ``min_hit`` of its area lies on a
  (grown) GT box of that class. Object detection has no countable true
  negatives, so TN is not defined here.
  precision = matched regions / regions, recall = TP / (TP + FN).
  A GT box is also **detected** when it is protected or a region of the
  class overlaps it with box IoU >= ``det_iou``; this second recall is
  fair to methods that blur a tight face inside a loose head-sized box.
* **frame level** -- a frame is positive when it contains >= 1 GT box of
  the class and predicted positive when the method blurred >= 1 region
  of the class; this gives a full 2x2 matrix incl. TN.

Region kinds count as faces: head, head_fallback, face, occupant; as
plates: plate. Windows are context blur and are counted separately.

**Neutral** boxes (faces below the size limit, :func:`split_small_faces`)
are neither ground truth nor background: a region that lies on one but
on no GT box of its class is neither matched nor an FP (``neutral``), so
blurring a real small face is no error for any method.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from blur_data.anonymize.regions import Region
from blur_data.metrics.privacy import GtBox, box_coverage, gt_mask

FACE_KINDS = frozenset({"head", "head_fallback", "face", "occupant"})
PLATE_KINDS = frozenset({"plate"})


def kinds_of(cls: str) -> frozenset[str]:
    """Region kinds that predict ``cls``."""
    return FACE_KINDS if cls == "face" else PLATE_KINDS


@dataclass
class Counts:
    """Object-level and frame-level counts of one class."""

    tp: int = 0  # protected GT objects
    fn: int = 0  # unprotected GT objects
    fp: int = 0  # regions not on any GT box of the class
    matched: int = 0  # regions on a GT box of the class
    detected: int = 0  # GT objects protected or hit with IoU >= det_iou
    neutral: int = 0  # regions on a neutral (small) face only
    frame: dict[str, int] = field(
        default_factory=lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    )
    #: (object size in px, protected?, detected?) of every GT object
    objects: list[tuple[float, bool, bool]] = field(default_factory=list)

    @property
    def recall(self) -> float:
        n = self.tp + self.fn
        return self.tp / n if n else float("nan")

    @property
    def detected_recall(self) -> float:
        n = self.tp + self.fn
        return self.detected / n if n else float("nan")

    @property
    def precision(self) -> float:
        n = self.matched + self.fp
        return self.matched / n if n else float("nan")


def box_iou(gt: GtBox, box: tuple[int, int, int, int]) -> float:
    """IoU of a GT box and a region box (both x1, y1 exclusive)."""
    iw = min(gt.x1, box[2]) - max(gt.x0, box[0])
    ih = min(gt.y1, box[3]) - max(gt.y0, box[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    a = (gt.x1 - gt.x0) * (gt.y1 - gt.y0)
    b = (box[2] - box[0]) * (box[3] - box[1])
    return float(inter / (a + b - inter))


def split_small_faces(
    boxes: Sequence[GtBox], min_face_px: float
) -> tuple[list[GtBox], list[GtBox]]:
    """(scored, neutral): faces whose box side (sqrt of the area) is
    below ``min_face_px`` become neutral; plates are always scored."""
    scored = [g for g in boxes if g.cls != "face" or g.size >= min_face_px]
    small = [g for g in boxes if g.cls == "face" and g.size < min_face_px]
    return scored, small


def add_frame(
    counts: Counts,
    cls: str,
    gts: Sequence[GtBox],
    regions: Sequence[Region],
    tau: float = 0.8,
    grow: float = 0.25,
    min_hit: float = 0.1,
    det_iou: float = 0.3,
    neutral: Sequence[GtBox] = (),
) -> None:
    """Accumulate one frame into ``counts``; regions on ``neutral`` boxes
    of the class only are neither matched nor FP."""
    gt_c = [g for g in gts if g.cls == cls]
    pred = [r for r in regions if r.kind in kinds_of(cls)]
    masks = [r.mask() for r in regions]
    if masks:
        blur = np.logical_or.reduce(masks)
        shape = (int(blur.shape[0]), int(blur.shape[1]))
    else:
        shape = (0, 0)
    for g in gt_c:
        cov = box_coverage(g, blur) if masks else 0.0
        det = cov >= tau or any(box_iou(g, r.box) >= det_iou for r in pred)
        counts.objects.append((g.size, cov >= tau, det))
        if cov >= tau:
            counts.tp += 1
        else:
            counts.fn += 1
        counts.detected += det
    if pred:
        on_gt = gt_mask(gt_c, shape, grow)
        on_neutral = gt_mask([g for g in neutral if g.cls == cls], shape, grow)
        for r in pred:
            m = r.mask()
            area = max(1, int(m.sum()))
            if float(np.logical_and(m, on_gt).sum()) / area >= min_hit:
                counts.matched += 1
            elif float(np.logical_and(m, on_neutral).sum()) / area >= min_hit:
                counts.neutral += 1
            else:
                counts.fp += 1
    key = {
        (True, True): "tp",
        (False, True): "fp",
        (True, False): "fn",
        (False, False): "tn",
    }[(bool(gt_c), bool(pred))]
    counts.frame[key] += 1
