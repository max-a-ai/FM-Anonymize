"""ArcFace identity embedding of a SYNTHETIC reference face.

IP-Adapter-FaceID is conditioned on an InsightFace ``buffalo_l``
embedding of an aligned face. Here it is only ever computed on a
track's synthetic reference face (painted by SD 1.5 from the track
seed), never on an original. The detector ``det_10g.onnx`` (SCRFD) is
decoded here (no InsightFace package): best face, its 5 landmarks, a
similarity warp to the ArcFace 112-px template (InsightFace's
``norm_crop``), then ``w600k_r50.onnx``.

Licence: the ``buffalo_l`` models are for non-commercial research only
(InsightFace), and so is IP-Adapter-FaceID. ``w600k_r50`` is also the
recognizer of the re-identification attack
(:mod:`blur_data.metrics.arcface`), so that attack may judge the
generated faces too favourably.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from blur_data.metrics.arcface import ArcFace

U8 = npt.NDArray[np.uint8]
F32 = npt.NDArray[np.float32]

#: ArcFace 112x112 template: eyes, nose, mouth corners
TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    np.float32,
)
DET_SIZE = 640
STRIDES = (8, 16, 32)


def decode(
    outs: list[F32], scale: float, thr: float = 0.5
) -> tuple[float, F32] | None:
    """Best SCRFD detection: (score, 5x2 landmarks in image px)."""
    best: tuple[float, F32] | None = None
    for i, s in enumerate(STRIDES):
        scores, kps = outs[i][:, 0], outs[i + 6] * s
        n = DET_SIZE // s
        ys, xs = np.mgrid[:n, :n]
        centres = np.stack([xs, ys], -1).reshape(-1, 2) * s
        centres = np.repeat(centres, 2, axis=0).astype(np.float32)
        j = int(np.argmax(scores))
        if scores[j] >= thr and (best is None or scores[j] > best[0]):
            pts = centres[j] + kps[j].reshape(5, 2)
            best = (float(scores[j]), (pts / scale).astype(np.float32))
    return best


def align(rgb: U8, kps: F32, size: int) -> U8:
    """Warp the face to the ArcFace template scaled to ``size`` px."""
    m, _ = cv2.estimateAffinePartial2D(kps, TEMPLATE * (size / 112.0))
    return np.asarray(
        cv2.warpAffine(rgb, m, (size, size), borderValue=(0, 0, 0)), np.uint8
    )


class FaceEmbedder:
    """Normed ``w600k_r50`` embedding and aligned crops of one face."""

    def __init__(self, folder: Path) -> None:
        import onnxruntime as ort

        self.det = ort.InferenceSession(
            str(folder / "det_10g.onnx"), providers=["CPUExecutionProvider"]
        )
        self.rec = ArcFace(folder / "w600k_r50.onnx")
        self.missed = 0  # faces not found: centre crop used instead

    def landmarks(self, rgb: U8, fill: float = 0.5) -> F32 | None:
        """5 landmarks of the best face; the image is placed at ``fill``
        of the detector input (a portrait face filling the frame is
        too large for SCRFD's anchors)."""
        h, w = rgb.shape[:2]
        scale = fill * DET_SIZE / max(h, w)
        small = cv2.resize(rgb, (round(w * scale), round(h * scale)))
        canvas = np.zeros((DET_SIZE, DET_SIZE, 3), np.uint8)
        oy, ox = (
            (DET_SIZE - small.shape[0]) // 2,
            (DET_SIZE - small.shape[1]) // 2,
        )
        canvas[oy : oy + small.shape[0], ox : ox + small.shape[1]] = small
        x = (canvas.astype(np.float32) - 127.5) / 128.0
        x = x.transpose(2, 0, 1)[None]
        name = self.det.get_inputs()[0].name
        outs = [
            np.asarray(o, np.float32) for o in self.det.run(None, {name: x})
        ]
        found = decode(outs, 1.0)
        if found is None:
            return None
        pts: F32 = (
            (found[1] - np.array([ox, oy], np.float32)) / scale
        ).astype(np.float32)
        return pts

    def __call__(self, rgb: U8) -> tuple[F32, U8, bool]:
        """(normed 512-d embedding, aligned 224-px crop for CLIP, whether
        a face was found; if not, a centre crop stands in)."""
        kps = None
        for fill in (0.5, 0.35, 0.7, 0.25):
            kps = self.landmarks(rgb, fill)
            if kps is not None:
                break
        if kps is None:
            self.missed += 1
            h, w = rgb.shape[:2]
            s = min(h, w)
            c = rgb[(h - s) // 2 : (h + s) // 2, (w - s) // 2 : (w + s) // 2]
            face112 = np.asarray(cv2.resize(c, (112, 112)), np.uint8)
            face224 = np.asarray(cv2.resize(c, (224, 224)), np.uint8)
        else:
            face112, face224 = align(rgb, kps, 112), align(rgb, kps, 224)
        e = self.rec.embed([face112])[0]
        return e.astype(np.float32), face224, kps is not None
