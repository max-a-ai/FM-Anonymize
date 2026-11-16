"""A region to blur ("one edit") and its per-image JSON file.

Every image gets one JSON file listing its regions, ordered by the
order in which edits are applied (most confident first). Rendering the
blurred dataset and the edit-budget notebook both read these files, so
detection (GPU, slow) and blurring (CPU, fast) are decoupled.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from blur_data.utils.masks import (
    BoolMask,
    decode_rle,
    encode_rle,
    mask_box,
    scale_mask,
)

#: Region kinds, one per anonymisation route.
KINDS = (
    "plate",  # licence plate inside a vehicle (SAM3)
    "head",  # projected head of a camera-facing SAM 3D Body mesh
    "head_fallback",  # top of a person box when the body fit failed
    "face",  # SAM3 "face" safety net not covered by a head region
    "occupant",  # person/face inside a vehicle (driver, passenger)
    "window",  # window of a close vehicle (hidden occupants)
)
#: Kinds that cover a face (grown by ``AnonConfig.face_region_scale``).
FACE_KINDS = ("head", "head_fallback", "face", "occupant")


@dataclass
class Region:
    """One blurred area of one image.

    ``rle`` holds the final (already dilated) mask that gets blurred.
    ``meta`` carries route-specific numbers, e.g. the facing angle of
    a head or the vehicle box a plate was found in.
    """

    kind: str
    score: float
    rle: dict[str, Any]
    box: tuple[int, int, int, int]
    meta: dict[str, Any] = field(default_factory=dict)

    def mask(self) -> BoolMask:
        """Decode the blur mask."""
        return decode_rle(self.rle)

    @classmethod
    def from_mask(
        cls,
        kind: str,
        score: float,
        mask: BoolMask,
        box: tuple[int, int, int, int],
        meta: dict[str, Any] | None = None,
    ) -> Region:
        """Build a region from a bool mask."""
        if kind not in KINDS:
            raise ValueError(f"unknown region kind {kind!r}")
        return cls(kind, float(score), encode_rle(mask), box, meta or {})

    def grown(self, factor: float) -> Region:
        """This region with its mask scaled by ``factor`` about its
        centre (``scale_mask``); itself if ``factor`` <= 1."""
        if factor <= 1.0:
            return self
        m = scale_mask(self.mask(), factor)
        box = mask_box(m) or self.box
        meta = {**self.meta, "grown": factor}
        return Region(self.kind, self.score, encode_rle(m), box, meta)


@dataclass
class ImageRegions:
    """All regions of one image plus what was seen but not blurred."""

    key: str
    rel_path: str
    regions: list[Region]
    stats: dict[str, Any] = field(default_factory=dict)
    #: every detection, blurred or not: ``{"kind", "prompt", "score",
    #: "box", ...}`` (vehicles, persons with their facing decision)
    detections: list[dict[str, Any]] = field(default_factory=list)

    def ordered(self) -> list[Region]:
        """Regions in edit order: highest score first."""
        return sorted(self.regions, key=lambda r: -r.score)

    def save(self, path: Path) -> None:
        """Write as JSON (parent directories are created)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path: Path) -> ImageRegions:
        """Read a file written by :meth:`save`."""
        raw = json.loads(path.read_text())
        regions = [
            Region(
                kind=r["kind"],
                score=r["score"],
                rle=r["rle"],
                box=tuple(r["box"]),
                meta=r["meta"],
            )
            for r in raw["regions"]
        ]
        return cls(
            raw["key"],
            raw["rel_path"],
            regions,
            raw["stats"],
            raw.get("detections", []),
        )


def regions_path(run_dir: Path, key: str) -> Path:
    """Where the regions of image ``key`` live inside a run."""
    return run_dir / "regions" / f"{key}.json"
