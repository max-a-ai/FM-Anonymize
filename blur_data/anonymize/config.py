"""Anonymisation settings, read from ``configs/*.yaml``."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

#: Everything that may carry a licence plate (incl. the small insurance
#: plates of mopeds and e-scooters) or a company/fleet number plate.
PLATE_CARRIERS = (
    "car",
    "van",
    "truck",
    "bus",
    "trailer",
    "tram",
    "motorcycle",
    "moped",
    "scooter",
    "e-scooter",
    "cargo bike",
    "delivery bike",
    "bicycle trailer",
    "bike trailer",
    "bike rack",
    "caravan",
    "tractor",
)


@dataclass
class AnonConfig:
    """Every knob of the pipeline; the yaml may set any subset."""

    # vehicles -> licence plates
    vehicle_prompts: list[str] = field(
        default_factory=lambda: list(PLATE_CARRIERS)
    )
    vehicle_threshold: float = 0.3
    plate_prompt: str = "license plate"
    plate_threshold: float = 0.3
    plate_crop_min_width: int = 60  # px; narrower vehicles: full pass only
    plate_crop_pad: float = 0.1
    plate_in_vehicle_pad: float = 0.1
    plate_dilate_frac: float = 0.25  # of the plate's shorter side
    plate_dilate_min_px: int = 3
    # blur the box of the dilated plate mask instead of the mask (off:
    # the plate is blurred in its own shape; a steep plate's axis-aligned
    # GT box then is not fully covered, see notebook 09)
    plate_fill_box: bool = False
    # people -> SAM 3D Body -> head
    person_prompts: list[str] = field(default_factory=lambda: ["person"])
    person_threshold: float = 0.3
    max_angle_deg: float = 130.0  # facing if below; 90 = exact profile
    small_person_px: int = 60  # shorter boxes: always blur box top
    head_margin: float = 0.05  # of the head hull's larger side
    head_min_px: int = 3
    fallback_top_frac: float = 0.25
    # occupants: person/face inside vehicles, from the vehicle crops
    occupants: bool = True
    occupant_min_width: int = 150  # px; vehicle width to look inside
    occupant_threshold: float = 0.3
    occupant_face_threshold: float = 0.4
    occupant_inside_frac: float = 0.7  # of the hit inside the vehicle mask
    occupant_crop_pad: float = 0.0  # second, tight crop (crop-sensitive)
    # windows of close vehicles in which no occupant was found
    # (occupants hidden behind reflections)
    window_blur: bool = True
    window_prompts: list[str] = field(
        default_factory=lambda: ["car window", "windshield"]
    )
    window_threshold: float = 0.3
    window_min_height_px: int = 250  # vehicle box height, ~< 10 m away
    window_dilate_frac: float = 0.03
    # ego car parts per channel (x0, y0, x1, y1): vehicles centred there
    # are the ego car's own side mirror and are ignored
    ego_vehicle_zones: dict[str, list[list[int]]] = field(default_factory=dict)
    # size policy (a precision setting): faces whose estimated side (px)
    # is below this are not blurred. 8 px: on our GT (every face scored)
    # recall 97.1 % at precision 69.3 %, against 87.4 / 80.9 at the
    # earlier 16 px (= 0.8 x 20 px, Merino et al., FG 2025); 0 = blur all.
    min_face_px: float = 8.0
    # every face region (FACE_KINDS) scaled by this about its centre
    # last; 1.36 = 1 / 0.735 matches benchmarks that box the whole head
    face_region_scale: float = 1.0
    face_width_m: float = 0.16  # physical face width (IEC 62676-4)
    face_per_person_height: float = 0.13  # face ~ 1/7.5 of body height
    # SAM3 "face" safety net
    face_safety_net: bool = True
    face_prompt: str = "face"
    face_threshold: float = 0.4
    face_dilate_frac: float = 0.25
    face_covered_frac: float = 0.8  # skip a face this much inside a head
    # shared
    tiles: bool = True
    dedupe_iou: float = 0.5
    blur_scale: float = 0.3
    plate_fill: str = "white"  # "white" (cannot be de-blurred) | "blur"
    jpeg_quality: str = "keep"  # PIL: reuse the source quantisation

    @classmethod
    def from_yaml(cls, path: Path) -> AnonConfig:
        """Load a yaml file; unknown keys are an error."""
        raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        """Plain dict, for ``config.json`` of a run."""
        return asdict(self)
