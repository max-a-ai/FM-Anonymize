"""Image -> regions to blur.

Route 1, plates: SAM3 ``vehicle_prompts`` -> vehicles; ``plate_prompt``
on the full frame/tiles (kept only inside a vehicle) and again on a
zoomed crop of every vehicle wide enough to carry a readable plate.

Route 2, heads: SAM3 ``person_prompts`` -> person boxes -> SAM 3D Body
with the real intrinsics -> per person: if the head faces the camera
(angle < ``max_angle_deg``), blur the hull of the projected head
vertices (those skinned to the head joints, not neck or shoulders).
The top of the person box is blurred instead when the person is
shorter than ``small_person_px`` (orientation unreliable), when the fit
failed, or when the head hull lies mostly outside the person box
(implausible fit).

Route 3, occupants: the zoomed vehicle crops of route 1 are also asked
for ``person``/``face``; hits inside a vehicle >= ``occupant_min_width``
px wide that are not a person of route 2 (drivers, passengers behind
the glass) get their head blurred: the mesh head of a ``person`` hit
(whatever its orientation), the top of its box if the fit fails, or
the mask of a ``face`` hit.

Route 4, windows: the windows (``window_prompts``) of every vehicle at
least ``window_min_height_px`` tall in which no occupant was found are
blurred, because an occupant behind strong reflections is recognisable
but not detectable.

Route 5, faces (safety net): SAM3 ``face_prompt`` detections that are
not already covered by a head or occupant region are blurred as well.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from blur_data.anonymize import head as headgeo
from blur_data.anonymize.config import AnonConfig
from blur_data.anonymize.regions import FACE_KINDS, Region
from blur_data.models.body import BodyFit, BodyFitter
from blur_data.models.detection import Box, Detection, box_inside, dedupe
from blur_data.models.segmenter import Sam3Segmenter
from blur_data.utils.masks import (
    BoolMask,
    coverage,
    dilate,
    fill_box,
    mask_box,
)

F64 = headgeo.F64


@dataclass
class PersonResult:
    """What happened to one person detection."""

    det: Detection
    fit: BodyFit | None
    angle_deg: float | None
    decision: str  # facing, away, fallback, head_off_image, too_small


@dataclass
class AnonResult:
    """Regions of one image plus the intermediate detections."""

    regions: list[Region]
    vehicles: list[Detection] = field(default_factory=list)
    plates: list[Detection] = field(default_factory=list)
    persons: list[PersonResult] = field(default_factory=list)
    faces: list[Detection] = field(default_factory=list)
    occupants: list[Detection] = field(default_factory=list)
    windows: list[Detection] = field(default_factory=list)

    def detections(self) -> list[dict[str, Any]]:
        """Vehicles and persons as plain dicts (for the region file)."""
        out: list[dict[str, Any]] = [
            {
                "kind": "vehicle",
                "prompt": v.prompt,
                "score": v.score,
                "box": list(v.box),
            }
            for v in self.vehicles
        ]
        out += [
            {
                "kind": "person",
                "prompt": p.det.prompt,
                "score": p.det.score,
                "box": list(p.det.box),
                "decision": p.decision,
                "angle_deg": p.angle_deg,
            }
            for p in self.persons
        ]
        return out

    def stats(self) -> dict[str, Any]:
        """Counts for the region file and the run summary."""
        decisions = [p.decision for p in self.persons]
        kinds = [r.kind for r in self.regions]
        return {
            "vehicles": len(self.vehicles),
            "plates": kinds.count("plate"),
            "persons": len(self.persons),
            **{
                f"persons_{d}": decisions.count(d)
                for d in (
                    "facing",
                    "away",
                    "fallback",
                    "head_off_image",
                    "too_small",
                )
            },
            "faces_detected": len(self.faces),
            "faces_extra": kinds.count("face"),
            "occupants": kinds.count("occupant"),
            "windows": kinds.count("window"),
            "regions": len(self.regions),
        }


def grow(mask: BoolMask, frac: float, min_px: int) -> BoolMask:
    """Dilate by ``frac`` x the shorter side of the mask's box."""
    box = mask_box(mask)
    if box is None:
        return mask
    short = min(box[2] - box[0], box[3] - box[1])
    return dilate(mask, max(min_px, int(round(frac * short))))


def plate_mask(mask: BoolMask, cfg: AnonConfig) -> BoolMask:
    """Blur mask of a plate: dilated, and filled to its box if set."""
    m = grow(mask, cfg.plate_dilate_frac, cfg.plate_dilate_min_px)
    return fill_box(m) if cfg.plate_fill_box else m


def int_box(mask: BoolMask) -> tuple[int, int, int, int]:
    """Box of a non-empty mask (0-box for an empty one)."""
    return mask_box(mask) or (0, 0, 0, 0)


class Anonymizer:
    """Holds both models; :meth:`process` handles one image."""

    def __init__(
        self,
        cfg: AnonConfig,
        sam3_checkpoint: Path,
        body_checkpoint_dir: Path,
        device: str = "cuda",
    ) -> None:
        self.cfg = cfg
        self.segmenter = Sam3Segmenter(sam3_checkpoint, device)
        self.body = BodyFitter(body_checkpoint_dir, device)

    def process(
        self, image: Image.Image, k: F64, channel: str = ""
    ) -> AnonResult:
        """Detect everything in ``image`` (RGB) with intrinsics ``k``.

        ``channel`` selects the ego-vehicle zones (``ego_vehicle_zones``):
        a "vehicle" centred there is part of the ego car (side mirror)
        and is ignored with everything found inside it.
        """
        cfg = self.cfg
        thr = {p: cfg.vehicle_threshold for p in cfg.vehicle_prompts}
        thr.update({p: cfg.person_threshold for p in cfg.person_prompts})
        thr[cfg.plate_prompt] = cfg.plate_threshold
        if cfg.face_safety_net:
            thr[cfg.face_prompt] = cfg.face_threshold
        found = self.segmenter.detect(image, thr, cfg.tiles, cfg.dedupe_iou)

        def union(prompts: list[str]) -> list[Detection]:
            dets = [d for p in prompts for d in found[p]]
            return dedupe(dets, cfg.dedupe_iou)

        zones = cfg.ego_vehicle_zones.get(channel, [])
        vehicles = [
            v
            for v in union(cfg.vehicle_prompts)
            if not any(box_inside(v.box, _box(z), 0.0) for z in zones)
        ]
        person_dets = union(cfg.person_prompts)
        plates, occupants, windows = self._vehicle_crops(
            image, vehicles, found[cfg.plate_prompt]
        )
        persons = self._persons(image, k, person_dets)
        regions = [
            Region.from_mask(
                "plate",
                d.score,
                m := plate_mask(d.mask, cfg),
                int_box(m),
                {"source": d.source},
            )
            for d in plates
        ]
        heads = [r for _, r in persons if r is not None]
        occ = self._occupants(image, k, occupants, heads)
        win = [
            Region.from_mask(
                "window",
                d.score,
                m := grow(d.mask, cfg.window_dilate_frac, cfg.head_min_px),
                int_box(m),
                {"source": d.prompt},
            )
            for d in windows
        ]
        faces = found.get(cfg.face_prompt, []) if cfg.face_safety_net else []
        regions += heads + occ + win + self._faces(faces, heads + occ)
        regions = [
            r.grown(cfg.face_region_scale) if r.kind in FACE_KINDS else r
            for r in regions
        ]
        return AnonResult(
            regions=sorted(regions, key=lambda r: -r.score),
            vehicles=vehicles,
            plates=plates,
            persons=[p for p, _ in persons],
            faces=faces,
            occupants=occupants,
            windows=windows,
        )

    def _vehicle_crops(
        self,
        image: Image.Image,
        vehicles: list[Detection],
        full_plates: list[Detection],
    ) -> tuple[list[Detection], list[Detection], list[Detection]]:
        """Plates, occupants and windows from zoomed vehicle crops.

        Plates: full-frame plates inside a vehicle plus the crop plates.
        Occupants: ``person``/``face`` hits on that crop and on a crop
        of the exact vehicle box (``occupant_crop_pad``) of a vehicle at
        least ``occupant_min_width`` wide whose mask lies >=
        ``occupant_inside_frac`` inside the vehicle's own mask (seen
        through the glass); a pedestrian behind a car sticks out of it.
        Windows: ``window_prompts`` hits inside the mask of a vehicle at
        least ``window_min_height_px`` tall (close enough that someone
        behind the glass could be recognised even when undetectable)
        in which no occupant was found.
        """
        cfg = self.cfg
        plates = [
            d
            for d in full_plates
            if any(
                box_inside(d.box, v.box, cfg.plate_in_vehicle_pad)
                for v in vehicles
            )
        ]
        thr = {cfg.plate_prompt: cfg.plate_threshold}
        occ_thr = {p: cfg.occupant_threshold for p in cfg.person_prompts}
        occ_thr[cfg.face_prompt] = cfg.occupant_face_threshold
        win_thr = {p: cfg.window_threshold for p in cfg.window_prompts}
        occupants: list[Detection] = []
        windows: list[Detection] = []
        for v in vehicles:
            n_occ, v_windows = len(occupants), []
            width = v.box[2] - v.box[0]
            if width < cfg.plate_crop_min_width:
                continue
            look_inside = cfg.occupants and width >= cfg.occupant_min_width
            close = (
                cfg.window_blur
                and v.box[3] - v.box[1] >= cfg.window_min_height_px
            )
            prompts = dict(thr)
            if look_inside:
                prompts.update(occ_thr)
            if close:
                prompts.update(win_thr)
            body = dilate(v.mask, 5)
            hits = self.segmenter.detect_in_box(
                image, v.box, prompts, cfg.plate_crop_pad
            )
            if look_inside:  # SAM3 is crop-sensitive: also the exact box
                hits += self.segmenter.detect_in_box(
                    image, v.box, occ_thr, cfg.occupant_crop_pad
                )
            for d in hits:
                if d.prompt == cfg.plate_prompt:
                    plates.append(d)
                elif coverage(d.mask, body) < cfg.occupant_inside_frac:
                    continue
                elif d.prompt in win_thr:
                    v_windows.append(d)
                else:
                    occupants.append(d)
            if len(occupants) == n_occ:  # nobody found: blur the glass
                windows += v_windows
        return (
            dedupe([p for p in plates if p.mask.any()], cfg.dedupe_iou),
            dedupe([o for o in occupants if o.mask.any()], cfg.dedupe_iou),
            dedupe([w for w in windows if w.mask.any()], cfg.dedupe_iou),
        )

    def _persons(
        self, image: Image.Image, k: F64, dets: list[Detection]
    ) -> list[tuple[PersonResult, Region | None]]:
        """Every person with its head region (None if not blurred)."""
        boxes = [d.box for d in dets]
        try:
            fits: list[BodyFit | None] = list(
                self.body.fit(np.asarray(image), boxes, k)
            )
        except RuntimeError:
            fits = [None] * len(dets)
        return [
            self._one_person(det, fit, k, image.size)
            for det, fit in zip(dets, fits, strict=True)
        ]

    def _one_person(
        self,
        det: Detection,
        fit: BodyFit | None,
        k: F64,
        size: tuple[int, int],
    ) -> tuple[PersonResult, Region | None]:
        cfg = self.cfg
        w, h = size
        box_face = cfg.face_per_person_height * (det.box[3] - det.box[1])
        small = det.box[3] - det.box[1] < cfg.small_person_px
        if fit is None or small:
            if box_face < cfg.min_face_px:  # not identifiable
                return PersonResult(det, fit, None, "too_small"), None
            return self._fallback(det, fit, "small" if small else "no_fit")
        pose = headgeo.head_pose(fit.keypoints)
        depth = float(pose.center[2])
        face_px = float(k[0, 0]) * cfg.face_width_m / max(depth, 1e-3)
        if face_px < cfg.min_face_px:  # not identifiable
            return PersonResult(det, fit, pose.angle_deg, "too_small"), None
        meta = {
            "angle_deg": pose.angle_deg,
            "person_box": list(det.box),
            "face_px": face_px,
        }
        hull = self._head_hull(fit, k, (h, w))
        if hull is not None and not _plausible(hull, det.box):
            if box_face < cfg.min_face_px:
                return PersonResult(det, fit, None, "too_small"), None
            return self._fallback(det, fit, "implausible_fit")
        if pose.angle_deg >= cfg.max_angle_deg:
            return PersonResult(det, fit, pose.angle_deg, "away"), None
        if hull is None:
            res = PersonResult(det, fit, pose.angle_deg, "head_off_image")
            return res, None
        region = Region.from_mask("head", det.score, hull, int_box(hull), meta)
        return PersonResult(det, fit, pose.angle_deg, "facing"), region

    def _head_hull(
        self, fit: BodyFit, k: F64, shape: tuple[int, int]
    ) -> BoolMask | None:
        """Hull of the projected head vertices of a mesh."""
        verts = fit.vertices[self.body.head_ids]
        return headgeo.hull_mask(
            headgeo.project(verts, k),
            shape,
            self.cfg.head_margin,
            self.cfg.head_min_px,
        )

    def _occupants(
        self,
        image: Image.Image,
        k: F64,
        occupants: list[Detection],
        heads: list[Region],
    ) -> list[Region]:
        """Head regions of the occupants not already covered by a head.

        A ``person`` hit gets its mesh head (any orientation: inside a
        vehicle the body pose says little about the face), or the top of
        its box when the fit fails or is implausible; a ``face`` hit its
        own mask, unless a person's head already covers it.
        """
        cfg = self.cfg
        w, h = image.size
        people = [
            d
            for d in occupants
            if d.prompt != cfg.face_prompt
            and _occupant_face(d) >= cfg.min_face_px
        ]
        try:
            fits: list[BodyFit | None] = list(
                self.body.fit(np.asarray(image), [d.box for d in people], k)
            )
        except RuntimeError:
            fits = [None] * len(people)
        out: list[Region] = []
        for d, fit in zip(people, fits, strict=True):
            hull = None if fit is None else self._head_hull(fit, k, (h, w))
            if hull is None or not _plausible(hull, d.box):
                hull = self._fallback_mask(d.box, (h, w))
            if _covered(hull, heads + out, cfg.face_covered_frac):
                continue
            out.append(
                Region.from_mask(
                    "occupant",
                    d.score,
                    hull,
                    int_box(hull),
                    {"source": d.prompt, "mesh": fit is not None},
                )
            )
        for d in occupants:
            if d.prompt != cfg.face_prompt or _side(d.box) < cfg.min_face_px:
                continue
            if _covered(d.mask, heads + out, cfg.face_covered_frac):
                continue
            m = grow(d.mask, cfg.face_dilate_frac, cfg.head_min_px)
            out.append(
                Region.from_mask(
                    "occupant", d.score, m, int_box(m), {"source": d.prompt}
                )
            )
        return out

    def _fallback(
        self, det: Detection, fit: BodyFit | None, reason: str
    ) -> tuple[PersonResult, Region]:
        """Blur the top of the person box (no trustworthy head mesh)."""
        w, h = det.mask.shape[1], det.mask.shape[0]
        mask = self._fallback_mask(det.box, (h, w))
        region = Region.from_mask(
            "head_fallback",
            det.score,
            mask,
            int_box(mask),
            {"reason": reason, "person_box": list(det.box)},
        )
        return PersonResult(det, fit, None, "fallback"), region

    def _fallback_mask(self, box: Box, shape: tuple[int, int]) -> BoolMask:
        """Top ``fallback_top_frac`` of the person box, full width."""
        h, w = shape
        x0, y0, x1, y1 = box
        top = y0 + self.cfg.fallback_top_frac * (y1 - y0)
        mask = np.zeros((h, w), dtype=bool)
        mask[
            max(0, int(y0)) : min(h, int(np.ceil(top))),
            max(0, int(x0)) : min(w, int(np.ceil(x1))),
        ] = True
        return grow(mask, 0.1, self.cfg.head_min_px)

    def _faces(
        self, faces: list[Detection], heads: list[Region]
    ) -> list[Region]:
        """Face detections not already inside a head region."""
        cfg = self.cfg
        covered: BoolMask | None = None
        for r in heads:
            m = r.mask()
            covered = m if covered is None else covered | m
        out = []
        for d in faces:
            if (
                covered is not None
                and coverage(d.mask, covered) >= cfg.face_covered_frac
            ):
                continue
            if _side(d.box) < cfg.min_face_px:  # not identifiable
                continue
            m = grow(d.mask, cfg.face_dilate_frac, cfg.head_min_px)
            out.append(
                Region.from_mask(
                    "face", d.score, m, int_box(m), {"source": d.source}
                )
            )
        return out


def _side(box: Box) -> float:
    """sqrt of a box area (side of the equal-area square), px."""
    return float(
        np.sqrt(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))
    )


def _occupant_face(d: Detection) -> float:
    """Face size of an occupant hit: the box for a face, else a share
    of the person box width (head ~ 0.35 x shoulder width; behind a
    window mostly the upper body is visible)."""
    if d.prompt == "face":
        return _side(d.box)
    return 0.35 * (d.box[2] - d.box[0])


def _box(z: list[int]) -> Box:
    """A config zone ``[x0, y0, x1, y1]`` as a float box."""
    return float(z[0]), float(z[1]), float(z[2]), float(z[3])


def _covered(mask: BoolMask, regions: list[Region], frac: float) -> bool:
    """Is ``mask`` at least ``frac`` inside the union of ``regions``?"""
    if not regions:
        return False
    union = np.zeros_like(mask)
    for r in regions:
        union |= r.mask()
    return coverage(mask, union) >= frac


def _plausible(hull: BoolMask, box: Box, pad: float = 0.15) -> bool:
    """At least half of the head hull lies in the (grown) person box."""
    h, w = hull.shape
    bw, bh = box[2] - box[0], box[3] - box[1]
    x0 = max(0, int(box[0] - pad * bw))
    y0 = max(0, int(box[1] - pad * bh))
    x1 = min(w, int(np.ceil(box[2] + pad * bw)))
    y1 = min(h, int(np.ceil(box[3] + pad * bh)))
    inside = int(hull[y0:y1, x0:x1].sum())
    return inside >= 0.5 * int(hull.sum())
