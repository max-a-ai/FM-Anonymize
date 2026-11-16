"""Re-identification scores of an anonymised face set.

A probe set (anonymised crops) is matched against a gallery (one crop
per probe, same order) by cosine similarity of recognizer embeddings:

* **rank-1**: a probe's most similar gallery crop is its own; chance
  is 1 / N.
* **TAR @ FAR**: the share of genuine pairs above the threshold that
  lets through ``far`` of the impostor pairs (all off-diagonal pairs);
  chance is ``far`` itself.

Faces are also scored per size bin (probes of the bin against the full
gallery). For the parrot attacker the gallery is a second photo of the
same person, found by following its region through the next frames.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]
Box = tuple[float, float, float, float]

#: face size bins on sqrt(GT box area), px; the attack starts at 20 px
FACE_BINS = ((20.0, 32.0), (32.0, 64.0), (64.0, float("inf")))


def bin_name(lo: float, hi: float) -> str:
    """``"20-32"`` or ``">=64"``."""
    return f">={lo:g}" if hi == float("inf") else f"{lo:g}-{hi:g}"


def rank1(sim: F32, rows: npt.NDArray[np.bool_] | None = None) -> float:
    """Share of probes (``rows``) whose best gallery match is their own."""
    hit = sim.argmax(axis=1) == np.arange(len(sim))
    if rows is not None:
        hit = hit[rows]
    return float(hit.mean()) if hit.size else float("nan")


def far_threshold(sim: F32, far: float) -> float:
    """Similarity above which ``far`` of the impostor pairs lie."""
    impostor = sim[~np.eye(len(sim), dtype=bool)]
    return float(np.quantile(impostor, 1.0 - far))


def tar(
    sim: F32, thr: float, rows: npt.NDArray[np.bool_] | None = None
) -> float:
    """Share of genuine pairs (``rows``) above ``thr``."""
    hit = np.diag(sim) > thr
    if rows is not None:
        hit = hit[rows]
    return float(hit.mean()) if hit.size else float("nan")


def scores(sim: F32, sizes: Sequence[float], far: float) -> dict[str, object]:
    """Rank-1, TAR @ FAR and own similarity, overall and per size bin."""
    n = len(sim)
    thr = far_threshold(sim, far) if n > 1 else float("inf")
    size = np.asarray(sizes, dtype=np.float64)
    bins: dict[str, object] = {}
    for lo, hi in FACE_BINS:
        rows = (size >= lo) & (size < hi)
        bins[bin_name(lo, hi)] = {
            "n": int(rows.sum()),
            "rank1": rank1(sim, rows),
            "tar": tar(sim, thr, rows),
        }
    return {
        "rank1": rank1(sim),
        "own_sim_median": float(np.median(np.diag(sim))),
        "tar": tar(sim, thr),
        "far_threshold": thr,
        "bins": bins,
    }


def box_iou(a: Box, b: Box) -> float:
    """IoU of two ``(x0, y0, x1, y1)`` boxes."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area - inter) if area > inter else 0.0


def track(
    start: Box, frames: Sequence[Sequence[Box]], min_iou: float
) -> Box | None:
    """Follow ``start`` through ``frames`` by best IoU; None if lost.

    Each step must have exactly one box above ``min_iou`` with the
    current box, or the best one must beat the runner-up by 2x, so a
    crowd never swaps identities silently.
    """
    cur = start
    for boxes in frames:
        ious = sorted(((box_iou(cur, b), b) for b in boxes), reverse=True)
        if not ious or ious[0][0] < min_iou:
            return None
        if len(ious) > 1 and ious[1][0] * 2 > ious[0][0]:
            return None
        cur = ious[0][1]
    return cur


def moved(window: Box, before: Box, after: Box) -> Box:
    """``window`` moved and scaled like ``before`` -> ``after``."""
    s = max(after[2] - after[0], after[3] - after[1]) / max(
        1e-6, max(before[2] - before[0], before[3] - before[1])
    )
    bcx, bcy = (before[0] + before[2]) / 2, (before[1] + before[3]) / 2
    acx, acy = (after[0] + after[2]) / 2, (after[1] + after[3]) / 2
    wcx, wcy = (window[0] + window[2]) / 2, (window[1] + window[3]) / 2
    hw, hh = s * (window[2] - window[0]) / 2, s * (window[3] - window[1]) / 2
    cx, cy = acx + s * (wcx - bcx), acy + s * (wcy - bcy)
    return cx - hw, cy - hh, cx + hw, cy + hh
