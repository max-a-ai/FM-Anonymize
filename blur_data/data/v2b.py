"""Camera images of a v2b export (nuScenes ``v1.0-mini`` layout).

Only the metadata needed for anonymisation is read: the image file, its
camera channel and the pinhole intrinsics of that camera. The devkit is
not needed for this, so the JSON tables are read directly.

The root is either one scene (``<root>/v1.0-mini``) or a folder of
scenes (``<root>/<scene>/v1.0-mini``, e.g. ``v2b/scene_a``); all scenes
are listed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

VERSION = "v1.0-mini"


@dataclass(frozen=True)
class CameraImage:
    """One camera frame of the export.

    ``rel_path`` is relative to the dataset root
    (``[<scene>/]sensor_data/...``), so the same object addresses the
    original and the blurred copy.
    """

    rel_path: str
    scene: str  # "" when the root itself is the scene
    channel: str
    timestamp: float
    width: int
    height: int
    intrinsics: tuple[tuple[float, ...], ...]

    @property
    def stem(self) -> str:
        """File name without extension (the nanosecond timestamp)."""
        return Path(self.rel_path).stem

    @property
    def key(self) -> str:
        """Stable id ``[<scene>/]<channel>/<stem>`` for region files."""
        prefix = f"{self.scene}/" if self.scene else ""
        return f"{prefix}{self.channel}/{self.stem}"

    def k_matrix(self) -> npt.NDArray[np.float64]:
        """3x3 intrinsics as a float64 array."""
        return np.asarray(self.intrinsics, dtype=np.float64)


def _table(root: Path, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = json.loads(
        (root / VERSION / f"{name}.json").read_text()
    )
    return rows


def scene_dirs(root: Path) -> list[Path]:
    """The scene folders under ``root`` (``root`` itself if flat)."""
    if (root / VERSION).is_dir():
        return [root]
    scenes = sorted(d for d in root.iterdir() if (d / VERSION).is_dir())
    if not scenes:
        raise FileNotFoundError(f"no {VERSION} under {root}")
    return scenes


def list_camera_images(root: Path) -> list[CameraImage]:
    """All jpg frames of all scenes, sorted by scene, channel, time."""
    images: list[CameraImage] = []
    for scene_dir in scene_dirs(root):
        scene = "" if scene_dir == root else scene_dir.name
        images += _scene_images(scene_dir, scene)
    images.sort(key=lambda im: (im.scene, im.channel, im.timestamp))
    return images


def _scene_images(scene_dir: Path, scene: str) -> list[CameraImage]:
    sensors = {s["token"]: s for s in _table(scene_dir, "sensor")}
    calib = {c["token"]: c for c in _table(scene_dir, "calibrated_sensor")}
    prefix = f"{scene}/" if scene else ""
    images: list[CameraImage] = []
    for sd in _table(scene_dir, "sample_data"):
        if sd["fileformat"] != "jpg":
            continue
        cs = calib[sd["calibrated_sensor_token"]]
        sensor = sensors[cs["sensor_token"]]
        k = tuple(
            tuple(float(v) for v in row) for row in cs["camera_intrinsic"]
        )
        images.append(
            CameraImage(
                rel_path=prefix + sd["filename"],
                scene=scene,
                channel=sensor["channel"],
                timestamp=float(sd["timestamp"]),
                width=int(sd["width"]),
                height=int(sd["height"]),
                intrinsics=k,
            )
        )
    return images
