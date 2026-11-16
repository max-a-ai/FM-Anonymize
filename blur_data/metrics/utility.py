"""Downstream utility of anonymised images (review item r31e).

A pretrained person + keypoint detector runs on the original and on
each anonymised version. Its confident output on the original is the
pseudo-ground truth (as in BLANKET, arXiv 2512.15542), so a version
scores 100 % when anonymisation changes nothing the detector relies
on. Scored with COCO AP (pycocotools): person boxes (IoU), all 17
keypoints (OKS) and the five head keypoints (nose, eyes, ears; OKS over
those only), overall and per person size (sqrt of the box area), for
all persons and for the persons whose head was anonymised.

The OKS scale is the pseudo-GT box area (COCO uses the segment area,
about half of it), so keypoint APs are more lenient than COCO's; this
is the same for every version.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

#: COCO keypoint OKS sigmas (pycocotools), in keypoint order
COCO_SIGMAS = (
    np.array(
        [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62]
        + [1.07, 1.07, 0.87, 0.87, 0.89, 0.89]
    )
    / 10.0
)
#: nose, left/right eye, left/right ear
HEAD = (0, 5)
#: person size bins on sqrt(box area), px
SIZE_BINS = (
    ("all", 0.0, 1e5),
    ("<64", 0.0, 64.0),
    ("64-128", 64.0, 128.0),
    (">=128", 128.0, 1e5),
)


@dataclass
class Persons:
    """Detections of one image: boxes (N, 4) xyxy, scores (N,),
    keypoints (N, 17, 3) as x, y, keypoint score."""

    boxes: npt.NDArray[np.float32]
    scores: npt.NDArray[np.float32]
    keypoints: npt.NDArray[np.float32]

    def to_json(self) -> dict[str, Any]:
        return {
            "boxes": self.boxes.round(2).tolist(),
            "scores": self.scores.round(4).tolist(),
            "keypoints": self.keypoints.round(2).tolist(),
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Persons:
        return cls(
            np.asarray(raw["boxes"], np.float32).reshape(-1, 4),
            np.asarray(raw["scores"], np.float32),
            np.asarray(raw["keypoints"], np.float32).reshape(-1, 17, 3),
        )


def _xywh(box: npt.NDArray[np.float32]) -> list[float]:
    x0, y0, x1, y1 = (float(v) for v in box)
    return [x0, y0, x1 - x0, y1 - y0]


def gt_annotations(
    preds: dict[int, Persons],
    score_thr: float,
    kp_thr: float,
    kps: tuple[int, int] = (0, 17),
) -> list[dict[str, Any]]:
    """Pseudo-GT: persons >= ``score_thr``, keypoints > ``kp_thr`` visible."""
    out: list[dict[str, Any]] = []
    for img_id, p in preds.items():
        for box, score, kp in zip(p.boxes, p.scores, p.keypoints, strict=True):
            if score < score_thr:
                continue
            k = kp[kps[0] : kps[1]]
            vis = k[:, 2] > kp_thr
            flat = np.stack([k[:, 0], k[:, 1], 2.0 * vis], 1)
            xywh = _xywh(box)
            out.append(
                {
                    "id": len(out) + 1,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": xywh,
                    "area": xywh[2] * xywh[3],
                    "iscrowd": 0,
                    "keypoints": flat.ravel().tolist(),
                    "num_keypoints": int(vis.sum()),
                }
            )
    return out


def subset(
    gt: list[dict[str, Any]], centres: dict[int, list[tuple[float, float]]]
) -> list[dict[str, Any]]:
    """Keep persons with a region centre in the top half of their box.

    The others become crowd regions, which COCOeval ignores (a
    detection on them is neither a hit nor a false positive).
    """
    out = []
    for g in gt:
        x, y, w, h = g["bbox"]
        hit = any(
            x <= cx <= x + w and y <= cy <= y + h / 2
            for cx, cy in centres.get(g["image_id"], [])
        )
        out.append(g | {"iscrowd": 0 if hit else 1})
    return out


def dt_annotations(
    preds: dict[int, Persons], kps: tuple[int, int] | None
) -> list[dict[str, Any]]:
    """Detections as COCO results; ``kps`` None for box results."""
    out: list[dict[str, Any]] = []
    for img_id, p in preds.items():
        for box, score, kp in zip(p.boxes, p.scores, p.keypoints, strict=True):
            d: dict[str, Any] = {
                "image_id": img_id,
                "category_id": 1,
                "bbox": _xywh(box),
                "score": float(score),
            }
            if kps is not None:
                k = kp[kps[0] : kps[1]]
                flat = np.stack([k[:, 0], k[:, 1], np.ones(len(k))], 1)
                d["keypoints"] = flat.ravel().tolist()
            out.append(d)
    return out


def coco_ap(
    gt: list[dict[str, Any]],
    dt: list[dict[str, Any]],
    image_ids: list[int],
    kps: tuple[int, int] | None,
) -> dict[str, dict[str, float]]:
    """AP, AP50 and pseudo-GT count per size bin (box or keypoint)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    names = [b[0] for b in SIZE_BINS]
    res: dict[str, dict[str, float]] = {"AP": {}, "AP50": {}, "n": {}}
    keep = [
        g
        for g in gt
        if not g["iscrowd"] and (kps is None or g["num_keypoints"] > 0)
    ]
    for name, lo, hi in SIZE_BINS:
        n = sum(lo**2 <= g["area"] < hi**2 for g in keep)
        res["n"][name] = float(n)
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = {
            "images": [{"id": i} for i in image_ids],
            "annotations": gt,
            "categories": [{"id": 1, "name": "person"}],
        }
        coco.createIndex()
        if not dt or not gt:
            return res
        ev = COCOeval(
            coco, coco.loadRes(dt), "bbox" if kps is None else "keypoints"
        )
        ev.params.areaRng = [[lo**2, hi**2] for _, lo, hi in SIZE_BINS]
        ev.params.areaRngLbl = names
        if kps is not None:  # pycocotools: 20 for keypoints, too few
            ev.params.kpt_oks_sigmas = COCO_SIGMAS[kps[0] : kps[1]]
            ev.params.maxDets = [100]
        ev.evaluate()
        ev.accumulate()
    prec = ev.eval["precision"][:, :, 0, :, -1]  # T x R x A
    for a, name in enumerate(names):
        p, p50 = prec[:, :, a], prec[0, :, a]
        res["AP"][name] = float(p[p > -1].mean()) if (p > -1).any() else -1
        res["AP50"][name] = (
            float(p50[p50 > -1].mean()) if (p50 > -1).any() else -1
        )
    return res


def _iou(a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]) -> float:
    iw = max(0.0, float(min(a[2], b[2]) - max(a[0], b[0])))
    ih = max(0.0, float(min(a[3], b[3]) - max(a[1], b[1])))
    inter = iw * ih
    area = float((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area - inter) if area > inter else 0.0


def extra_lost(
    orig: dict[int, Persons],
    version: dict[int, Persons],
    score: float = 0.5,
    match_score: float = 0.3,
    iou: float = 0.3,
) -> dict[str, int]:
    """Persons (>= ``score``) a version adds or loses against the original.

    ``extra``: a version's person with no original person (>=
    ``match_score``) at IoU >= ``iou``; ``lost``: the reverse.
    """
    extra = lost = 0
    for i, o in orig.items():
        v = version[i]
        ob, vb = (
            o.boxes[o.scores >= match_score],
            v.boxes[v.scores >= match_score],
        )
        extra += sum(
            all(_iou(b, c) < iou for c in ob)
            for b in v.boxes[v.scores >= score]
        )
        lost += sum(
            all(_iou(b, c) < iou for c in vb)
            for b in o.boxes[o.scores >= score]
        )
    return {"extra": int(extra), "lost": int(lost)}
