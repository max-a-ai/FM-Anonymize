"""Find the images of a folder and describe them before a run.

A folder in nuScenes layout (``v1.0-*`` tables, e.g. the v2b export)
yields its camera frames with their real intrinsics and channel, so
the result matches ``blur-data-anonymize.py``; any other folder is
crawled for image files, which get SAM 3D Body's default pinhole
intrinsics (as PP4AV does).
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from blur_data.anonymize.regions import regions_path
from blur_data.data.v2b import VERSION, list_camera_images

EXTENSIONS = (".jpg", ".jpeg", ".png")
#: images whose header is read for the size statistics
SIZE_SAMPLE = 300


@dataclass(frozen=True)
class SourceImage:
    """One image to anonymize, relative to the source folder."""

    key: str  # rel_path without suffix: names the region file
    rel_path: str
    channel: str  # nuScenes channel ("" elsewhere): ego-vehicle zones
    intrinsics: tuple[tuple[float, ...], ...] | None  # None: default K

    def k_matrix(self, width: int, height: int) -> npt.NDArray[np.float64]:
        """The camera matrix: the real one or the default pinhole."""
        if self.intrinsics is not None:
            return np.array(self.intrinsics, dtype=np.float64)
        f = float(np.hypot(width, height))
        return np.array(
            [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]]
        )


def is_nuscenes(root: Path) -> bool:
    """``root`` is a nuScenes scene or a folder of scenes."""
    if (root / VERSION).is_dir():
        return True
    return any((d / VERSION).is_dir() for d in subdirs(root))


def subdirs(root: Path) -> list[Path]:
    """Visible sub-folders of ``root``, sorted (empty if unreadable)."""
    try:
        return sorted(
            d
            for d in root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
    except OSError:
        return []


def list_source(root: Path) -> list[SourceImage]:
    """Every image of ``root`` in a stable order."""
    if is_nuscenes(root):
        return [
            SourceImage(im.key, im.rel_path, im.channel, im.intrinsics)
            for im in list_camera_images(root)
        ]
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if name.lower().endswith(EXTENSIONS) and not name.startswith("."):
                rel = (Path(dirpath) / name).relative_to(root).as_posix()
                out.append(
                    SourceImage(str(Path(rel).with_suffix("")), rel, "", None)
                )
    return out


def analyse(
    root: Path, images: list[SourceImage], dst: Path | None
) -> dict[str, Any]:
    """Counts, folders, sizes and what ``dst`` already holds."""
    folders: Counter[str] = Counter()
    for im in images:
        folders[folder_of(im.rel_path)] += 1
    step = max(1, len(images) // SIZE_SAMPLE)
    sizes: Counter[str] = Counter()
    nbytes = 0
    for im in images[::step]:
        path = root / im.rel_path
        nbytes += path.stat().st_size
        with Image.open(path) as pil:
            sizes[f"{pil.width}x{pil.height}"] += 1
    sampled = len(images[::step])
    done = (
        sum(regions_path(dst, im.key).exists() for im in images)
        if dst is not None and dst.is_dir()
        else 0
    )
    return {
        "root": str(root),
        "layout": "nuScenes" if is_nuscenes(root) else "image folder",
        "images": len(images),
        "extensions": dict(
            Counter(Path(im.rel_path).suffix.lower() for im in images)
        ),
        "folders": folders.most_common(40),
        "sizes": sizes.most_common(),
        "bytes_est": round(nbytes / sampled * len(images)) if sampled else 0,
        "channels": sorted({im.channel for im in images if im.channel}),
        "done": done,
    }


def folder_of(rel_path: str) -> str:
    """The folder an image lies in (``.`` for the root): one camera of a
    nuScenes scene, one city of PP4AV."""
    return Path(rel_path).parent.as_posix()


def check_paths(src: Path, dst: Path) -> str | None:
    """Why ``dst`` may not take the output of ``src`` (None if fine)."""
    if not src.is_dir():
        return f"source {src} is not a folder"
    s, d = src.resolve(), dst.resolve()
    if d == s or s in d.parents:
        return "the output folder must lie outside the source folder"
    if d in s.parents:
        return "the output folder must not contain the source folder"
    return None
