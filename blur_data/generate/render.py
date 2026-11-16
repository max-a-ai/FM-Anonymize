"""Render a run of FM-Anonymize-Gen: frames, tracks, generated heads.

For one *segment* (consecutive frames of a sequence, see
:func:`blur_data.generate.tracks.segments`) the renderer loads every
frame's regions and image, fits SAM 3D Body to its generated heads,
links the heads into tracks, gives each track of at least two heads one
seed, and writes every output frame of the segment
(:func:`blur_data.generate.inpaint.apply_regions`).
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from blur_data.anonymize.blur import U8
from blur_data.anonymize.head import F64
from blur_data.anonymize.regions import ImageRegions, Region, regions_path
from blur_data.data import pp4av
from blur_data.generate.config import GenConfig
from blur_data.generate.inpaint import apply_regions
from blur_data.generate.pose_guide import FitGuides, HeadFit, HeadGuider
from blur_data.generate.routing import is_generated, plan, track_seed
from blur_data.generate.tracks import NO_HEAD, Box, head_tracks
from blur_data.utils.images import save_like
from blur_data.utils.masks import BoolMask, mask_box

#: file extension of the rendered images per dataset (as the sources)
EXT = {"v2b": ".jpg", "pp4av": ".png"}


@dataclass
class Frame:
    """One frame of a segment: its regions, image, generated heads (grown
    masks by region index) and their body fits."""

    key: str
    regs: ImageRegions
    ordered: list[Region]
    rgb: U8
    grown: dict[int, BoolMask]
    k: F64
    fits: dict[int, HeadFit | str]

    def head_boxes(self) -> list[Box]:
        """Boxes of the generated heads, in region-index order."""
        out = []
        for idx in sorted(self.grown):
            b = mask_box(self.grown[idx])
            assert b is not None
            out.append((float(b[0]), float(b[1]), float(b[2]), float(b[3])))
        return out

    def person_boxes(self) -> list[Box]:
        """The frame's SAM 3 person detections."""
        return [
            (float(d["box"][0]), float(d["box"][1]), float(d["box"][2]),
             float(d["box"][3]))
            for d in self.regs.detections
            if d.get("kind") == "person"
        ]  # fmt: skip

    def head_person_boxes(self) -> list[Box | None]:
        """Per generated head (region-index order): its person box."""
        out: list[Box | None] = []
        for idx in sorted(self.grown):
            r = self.ordered[idx]
            pb = r.meta.get("person_box") if r.kind == "head" else None
            out.append(None if pb is None else (pb[0], pb[1], pb[2], pb[3]))
        return out


class Renderer:
    """The models of a run and its per-segment steps."""

    def __init__(
        self,
        cfg: GenConfig,
        dataset: str,
        regions_dir: Path,
        data: Path,
        checkpoints: Path,
        track_scale: float = 1.0,
    ) -> None:
        from blur_data.models.face_generator import FaceGenerator

        self.generator = FaceGenerator(cfg, checkpoints / "arcface")
        self.guider = HeadGuider(checkpoints / "sam-3d-body")
        self.cfg, self.dataset, self.data = cfg, dataset, data
        self.regions_dir, self.track_scale = regions_dir, track_scale
        self.k_v2b: dict[str, F64] = {}
        if dataset == "v2b":
            from blur_data.data.v2b import list_camera_images

            ims = list_camera_images(data / "v2b")
            self.k_v2b = {im.key: im.k_matrix() for im in ims}

    def frame(self, key: str, fit: bool = True) -> Frame:
        """Load one frame and (``fit``) fit SAM 3D Body to its generated
        heads; context frames only serve the track linking."""
        regs = ImageRegions.load(regions_path(self.regions_dir, key))
        ordered = regs.ordered()
        with Image.open(self.data / self.dataset / regs.rel_path) as im:
            rgb = np.asarray(im.convert("RGB"))
        _, grown = plan(rgb, ordered, self.cfg)
        if self.dataset == "v2b":
            k = self.k_v2b[key]
        else:
            h, w = rgb.shape[:2]
            k = pp4av.Pp4avImage(regs.rel_path, w, h).k_matrix()
        fits = {}
        if fit and grown:
            fits = self.guider.fits(rgb, ordered, k, self.cfg)
        return Frame(key, regs, ordered, rgb, grown, k, fits)

    def track_seeds(
        self, seq: str, frames: list[Frame], wanted: set[int]
    ) -> dict[int, dict[int, int]]:
        """Frame index -> region index -> seed of its track, for the
        tracks of at least two heads that reach a wanted frame."""
        tracks = head_tracks(
            [f.head_boxes() for f in frames],
            [f.person_boxes() for f in frames],
            [f.head_person_boxes() for f in frames],
            self.cfg.link_on,
            self.cfg.track_iou,
            self.track_scale,
        )
        seeds: dict[int, dict[int, int]] = {}
        for tr in tracks:
            if not wanted & {fi for fi, _ in tr}:
                continue
            f0 = frames[tr[0][0]]
            first = sorted(f0.grown)[tr[0][1]]
            tid = f"{seq}/{f0.key.rsplit('/', 1)[1]}#{first}"
            heads = [(fi, ri) for fi, ri in tr if ri != NO_HEAD]
            if len(heads) < 2:
                continue
            seed = track_seed(tid)
            for fi, ri in heads:
                seeds.setdefault(fi, {})[sorted(frames[fi].grown)[ri]] = seed
        return seeds

    def segment(
        self, seq: str, keys: list[str], outputs: dict[str, list[Path]]
    ) -> list[dict[str, Any]]:
        """Render the output frames of one segment; summary rows."""
        frames = [self.frame(k, k in outputs) for k in keys]
        wanted = {i for i, f in enumerate(frames) if f.key in outputs}
        seeds = self.track_seeds(seq, frames, wanted)
        return [
            self.compose(frames[i], outputs[frames[i].key], seeds.get(i))
            for i in sorted(wanted)
        ]

    def compose(
        self, f: Frame, dsts: list[Path], seeds: dict[int, int] | None
    ) -> dict[str, Any]:
        """Generate the heads of one frame, blur the rest, save it to
        every path of ``dsts`` (a frame can belong to two runs)."""
        t0 = time.time()
        ok = {i: x for i, x in f.fits.items() if isinstance(x, HeadFit)}
        guides = FitGuides(self.guider.template, ok, f.k)
        with Image.open(self.data / self.dataset / f.regs.rel_path) as src:
            jpeg = src.format == "JPEG"
            qt = getattr(src, "quantization", None) if jpeg else None
            out = apply_regions(
                f.rgb.copy(), f.ordered, self.generator, f.key, self.cfg,
                guides, seeds, qt,
            )  # fmt: skip
            for dst in dsts:
                save_like(Image.fromarray(out), dst, src, "keep")
        unguided = Counter(
            f"unguided_{why}"
            for why in f.fits.values()
            if isinstance(why, str)
        )
        return {
            "key": f.key,
            "generated": len(f.grown),
            "tracked": len(seeds or {}),
            **unguided,
            "blurred": len(f.ordered) - len(f.grown),
            "seconds": round(time.time() - t0, 2),
        }


def has_generated_head(regions_dir: Path, key: str, cfg: GenConfig) -> bool:
    """Whether any region of ``key`` gets a synthetic head."""
    regs = ImageRegions.load(regions_path(regions_dir, key)).regions
    return any(
        is_generated(r, cfg, (r.rle["size"][0], r.rle["size"][1]))
        for r in regs
    )
