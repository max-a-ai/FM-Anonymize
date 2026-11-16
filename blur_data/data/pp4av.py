"""PP4AV benchmark (Trinh et al., WACV 2023): faces + plates in driving.

Layout: ``<root>/<city>/<name>.png`` with ``<name>.txt`` YOLO labels
(``cls cx cy w h`` normalised; class 0 = face, 1 = plate). The images
carry no intrinsics; SAM 3D Body's own default is used (focal length =
image diagonal, principal point at the centre), so the mesh projection
matches what the model assumed. The fisheye (WoodScape) subset is left
out: the head projection is pinhole only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from blur_data.metrics.privacy import GtBox

CLASSES = {0: "face", 1: "plate"}
FISHEYE = "fisheye"


@dataclass(frozen=True)
class Pp4avImage:
    """One PP4AV frame; ``key`` and ``rel_path`` mirror ``CameraImage``."""

    rel_path: str
    width: int
    height: int

    @property
    def channel(self) -> str:
        """The city folder (plays the role of a camera channel)."""
        return Path(self.rel_path).parent.name

    @property
    def key(self) -> str:
        """``<city>/<stem>``."""
        return str(Path(self.rel_path).with_suffix(""))

    def k_matrix(self) -> npt.NDArray[np.float64]:
        """SAM 3D Body's default pinhole intrinsics for this size."""
        f = float(np.hypot(self.width, self.height))
        return np.array(
            [
                [f, 0.0, self.width / 2.0],
                [0.0, f, self.height / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )


def list_images(root: Path, fisheye: bool = False) -> list[Pp4avImage]:
    """All labelled pinhole images (``fisheye`` adds WoodScape)."""
    out = []
    for img in sorted(root.glob("*/*.png")):
        if img.parent.name == FISHEYE and not fisheye:
            continue
        if not img.with_suffix(".txt").exists():
            continue
        with Image.open(img) as im:
            w, h = im.size
        out.append(Pp4avImage(str(img.relative_to(root)), w, h))
    return out


def read_gt(root: Path, image: Pp4avImage) -> list[GtBox]:
    """The image's YOLO labels as pixel boxes.

    Labels whose centre lies outside the image are dropped: two face
    labels of ``netherlands_day/frame_000035`` hold pixel instead of
    normalised coordinates and would count as misses for every method.
    """
    boxes = []
    txt = (root / image.rel_path).with_suffix(".txt")
    for line in txt.read_text().split("\n"):
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            continue
        w, h = image.width, image.height
        boxes.append(
            GtBox(
                CLASSES[cls],
                (cx - bw / 2) * w,
                (cy - bh / 2) * h,
                (cx + bw / 2) * w,
                (cy + bh / 2) * h,
            )
        )
    return boxes
