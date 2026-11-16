"""ArcFace embeddings of square face crops (the re-identification attack).

InsightFace ``buffalo_l`` ``w600k_r50.onnx`` on CPU; crops are resized to
112 px unaligned, the same for every version of a face.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from blur_data.metrics.privacy import GtBox

U8 = npt.NDArray[np.uint8]


def square(g: GtBox, shape: tuple[int, int], f: float) -> tuple[int, ...]:
    """Square window of ``f`` x the box's longer side, clipped."""
    h, w = shape
    side = f * max(g.x1 - g.x0, g.y1 - g.y0)
    cx, cy = (g.x0 + g.x1) / 2, (g.y0 + g.y1) / 2
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    return x0, y0, int(min(w, cx + side / 2)), int(min(h, cy + side / 2))


class ArcFace:
    def __init__(self, path: Path) -> None:
        import onnxruntime as ort

        self.sess = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self.name = self.sess.get_inputs()[0].name

    def embed(self, crops: list[U8]) -> npt.NDArray[np.float32]:
        x = np.stack(
            [
                cv2.resize(c, (112, 112), interpolation=cv2.INTER_LINEAR)
                for c in crops
            ]
        ).astype(np.float32)
        x = ((x - 127.5) / 127.5).transpose(0, 3, 1, 2)
        e = np.asarray(self.sess.run(None, {self.name: x})[0], np.float32)
        out: npt.NDArray[np.float32] = e / np.linalg.norm(
            e, axis=1, keepdims=True
        )
        return out
