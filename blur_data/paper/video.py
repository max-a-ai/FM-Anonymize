"""Videos: raw | anonymised side by side, and the blurred-only teaser.

The teaser draws an orange box around every blurred region. Boxes come
per frame, so they are tracked to keep them steady: greedy IoU matching
to the previous frame's tracks, exponential smoothing of the corners,
a track is drawn only after ``min_hits`` consecutive frames and kept for
``max_miss`` frames after it was last seen.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

U8 = npt.NDArray[np.uint8]
F64 = npt.NDArray[np.float64]
ORANGE = (255, 140, 0)  # RGB


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Box IoU of ``(x0, y0, x1, y1)``."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (
        (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    )
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    """A smoothed box over time."""

    box: F64
    hits: int = 1
    miss: int = 0


@dataclass
class BoxSmoother:
    """Turns per-frame boxes into steady, non-flickering boxes."""

    alpha: float = 0.5  # weight of the new observation
    min_iou: float = 0.2
    min_hits: int = 2
    max_miss: int = 2
    tracks: list[Track] = field(default_factory=list)

    def update(self, boxes: Sequence[Sequence[float]]) -> list[F64]:
        """Feed one frame's boxes; return the boxes to draw."""
        free = list(range(len(boxes)))
        for t in self.tracks:
            best, best_iou = -1, self.min_iou
            for i in free:
                v = iou(t.box.tolist(), boxes[i])
                if v > best_iou:
                    best, best_iou = i, v
            if best >= 0:
                new = np.asarray(boxes[best], dtype=np.float64)
                t.box = self.alpha * new + (1 - self.alpha) * t.box
                t.hits += 1
                t.miss = 0
                free.remove(best)
            else:
                t.miss += 1
        self.tracks = [t for t in self.tracks if t.miss <= self.max_miss]
        self.tracks += [
            Track(np.asarray(boxes[i], dtype=np.float64)) for i in free
        ]
        return [t.box for t in self.tracks if t.hits >= self.min_hits]


def draw_boxes(img: U8, boxes: Sequence[F64], width: int) -> U8:
    """Orange rectangles (RGB image)."""
    out = img.copy()
    for b in boxes:
        p0 = (int(b[0]), int(b[1]))
        p1 = (int(b[2]), int(b[3]))
        cv2.rectangle(out, p0, p1, ORANGE, width)
    return out


def label(img: U8, text: str, scale: float = 1.0) -> U8:
    """Text with a dark backing box in the top-left corner."""
    out = img.copy()
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(out, (0, 0), (tw + 16, th + base + 14), (20, 20, 20), -1)
    cv2.putText(
        out,
        text,
        (8, th + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        2,
    )
    return out


def write_mp4(
    frames: Iterator[U8],
    path: Path,
    fps: int,
    size: tuple[int, int],
    crf: int = 20,
) -> None:
    """Pipe RGB frames of ``size`` (w, h) into ffmpeg (H.264, yuv420p)."""
    w, h = size
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    assert proc.stdin is not None
    for f in frames:
        if f.shape[:2] != (h, w):
            raise ValueError(f"frame {f.shape} != {h}x{w}")
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")
