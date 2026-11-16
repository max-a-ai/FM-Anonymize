"""Why a run misses ground-truth faces: one row per unprotected face.

Each missed face (>= ``min_face_px`` by its box) is matched to the
smallest SAM 3 person box containing its centre, and that person's
pipeline decision becomes the reason: ``too_small`` (the 20 px rule
judged the face unidentifiable), ``away`` (the mesh says the head turns
away), ``fallback`` / ``facing`` (a region exists but covers < tau of
the box), or ``no person`` (SAM 3 found no person there).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from blur_data.anonymize.regions import ImageRegions, regions_path
from blur_data.metrics.privacy import GtBox, box_coverage


@dataclass(frozen=True)
class Miss:
    """One unprotected ground-truth face."""

    key: str
    gt: GtBox
    coverage: float  # blurred share of the GT box
    reason: str
    person: dict[str, Any] | None  # the matched SAM 3 person, if any


def _area(box: list[float]) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def find_misses(
    run_dir: Path,
    gt: Iterable[tuple[str, list[GtBox]]],
    min_face_px: float = 20.0,
    tau: float = 0.8,
) -> list[Miss]:
    """Every face of ``gt`` (key, boxes) that the run leaves exposed."""
    out = []
    for key, boxes in gt:
        faces = [g for g in boxes if g.cls == "face" and g.size >= min_face_px]
        if not faces:
            continue
        ir = ImageRegions.load(regions_path(run_dir, key))
        masks = [r.mask() for r in ir.regions]
        blur = np.logical_or.reduce(masks) if masks else None
        persons = [d for d in ir.detections if d["kind"] == "person"]
        for g in faces:
            cov = box_coverage(g, blur) if blur is not None else 0.0
            if cov >= tau:
                continue
            cx, cy = (g.x0 + g.x1) / 2, (g.y0 + g.y1) / 2
            hits = [
                p
                for p in persons
                if p["box"][0] <= cx <= p["box"][2]
                and p["box"][1] <= cy <= p["box"][3]
            ]
            person = min(hits, key=lambda p: _area(p["box"])) if hits else None
            reason = "no person" if person is None else person["decision"]
            out.append(Miss(key, g, cov, reason, person))
    return out
