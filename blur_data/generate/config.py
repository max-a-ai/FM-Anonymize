"""Every setting of FM-Anonymize-Gen; ``configs/generate.yaml`` sets them.

The defaults are the paper's values (Sec. V, "v7a, full head"). Model
references are ``"repo@revision"``: every Hugging Face download is
pinned.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

NEGATIVE_PROMPT = (
    "cartoon, painting, drawing, illustration, deformed, distorted, "
    "disfigured, bad anatomy, extra eyes, mask, text, watermark, lowres"
)
#: how heads are linked into tracks (one synthetic identity per track):
#: on their PERSON boxes (stable while the head region blinks with the
#: facing decision) or on the head regions themselves
LINK_ON = ("person", "head")


@dataclass
class GenConfig:
    """Settings of the generative head replacement."""

    # diffusion model: SD 1.5 inpainting, fp16, DPM-Solver++ 2M
    model: str = "stable-diffusion-v1-5/stable-diffusion-inpainting"
    revision: str = "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb"
    resolution: int = 512  # side of the square model crop
    steps: int = 30
    scheduler: str = "dpmpp"  # "dpmpp" (DPM-Solver++ 2M) or "default"
    guidance_scale: float = 5.0
    prompt: str = "a photo of a person, realistic"
    negative_prompt: str = NEGATIVE_PROMPT

    # which regions are generated; every other region is blurred
    generate_kinds: list[str] = field(
        default_factory=lambda: ["head", "occupant"]
    )
    min_generate_px: int = 24  # smaller regions (box long side): blurred
    # regions whose box long side exceeds this x the image's short side
    # are blurred: an outlier mask would regenerate most of the frame
    max_generate_frac: float = 0.25

    # SDEdit start: the head as FM-Anonymize blurs it
    strength: float = 0.8
    blur_scale: float = 0.3  # FM-Anonymize's blur (sigma / box size)

    # pose guide: two ControlNets on a SAM 3D Body fit
    controlnet_depth: str = (
        "lllyasviel/control_v11f1p_sd15_depth"
        "@539f99181d33db39cf1af2e517cd8056785f0a87"
    )
    controlnet_pose: str = (
        "lllyasviel/control_v11p_sd15_openpose"
        "@9ae9f970358db89e211b87c915f9535c6686d5ba"
    )
    depth_scale: float = 0.35
    pose_scale: float = 0.6

    # identity: IP-Adapter-FaceID plus-v2 on the ArcFace embedding of a
    # synthetic reference face painted from the track's seed
    faceid_adapter: str = (
        "h94/IP-Adapter-FaceID@43907e6f44d079bf1a9102d9a6e56aef7a219bae"
    )
    faceid_weight: str = "ip-adapter-faceid-plusv2_sd15.bin"
    # CLIP image encoder of the plus-v2 adapter (pinned here: diffusers
    # would load it from "main" and fail offline)
    image_encoder: str = (
        "h94/IP-Adapter@018e402774aeeddd60609b4ecdb7e298259dc729"
    )
    identity_scale: float = 0.7
    reference_prompt: str = (
        "a color photo, close-up portrait of a person, face centered, "
        "looking at the camera, neutral expression, realistic"
    )
    reference_steps: int = 25

    # tracks: heads of consecutive frames linked by box IoU
    link_on: str = "person"
    track_iou: float = 0.3

    # crop geometry
    context: float = 2.5  # crop side / head box long side
    min_context: float = 1.25  # lower bound when the image is too small
    dilate_frac: float = 0.1  # head mask grown by this x its long side
    dilate_min_px: int = 2

    # composite: Laplacian levels of the blend band
    blend_levels: int = 3

    def __post_init__(self) -> None:
        if self.link_on not in LINK_ON:
            raise ValueError(f"link_on {self.link_on!r} not in {LINK_ON}")
        if not 0.0 < self.strength <= 1.0:
            raise ValueError(f"strength {self.strength} not in (0, 1]")

    @classmethod
    def from_yaml(cls, path: Path) -> GenConfig:
        """Load a yaml file; unknown keys are an error."""
        raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        """Plain dict, for a run's ``config.json``."""
        return asdict(self)


def split_ref(ref: str) -> tuple[str, str | None]:
    """``"repo@revision"`` -> ``(repo, revision)``."""
    repo, _, rev = ref.partition("@")
    return repo, rev or None
