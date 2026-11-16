"""Baseline anonymisers, wrapped to emit the same region files as ours.

Each baseline is scored on the blur its own tool applies by default:
deface and EgoBlur (Gen1 and Gen2) blur an ellipse inscribed in each
(scaled) detection box, so a box becomes that ellipse mask here;
UrbanAnonymizer masks the detection box grown by 20 % per side, so a
box becomes that rectangle; the Autoware rosbag2 anonymizer blurs the
SAM 2.1 mask of each box inside that (integer) box.
"""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from blur_data.anonymize.regions import Region
from blur_data.utils.masks import BoolMask, decode_rle, mask_box

U8 = npt.NDArray[np.uint8]


def _hw(img: U8) -> tuple[int, int]:
    return int(img.shape[0]), int(img.shape[1])


def ellipse_mask(
    box: Sequence[float], shape: tuple[int, int], scale: float
) -> BoolMask:
    """Filled ellipse inscribed in ``box`` grown by ``scale`` about its
    centre (deface ``mask_scale`` / EgoBlur ``scale_factor_detections``)."""
    h, w = shape
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ax = max(1, int(round(scale * (x1 - x0) / 2)))
    ay = max(1, int(round(scale * (y1 - y0) / 2)))
    canvas = np.zeros((h, w), np.uint8)
    center = (int(round(cx)), int(round(cy)))
    cv2.ellipse(canvas, center, (ax, ay), 0, 0, 360, (1,), -1)
    out: BoolMask = canvas.astype(bool)
    return out


def to_regions(
    kind: str,
    dets: Sequence[tuple[Sequence[float], float]],
    shape: tuple[int, int],
    scale: float,
) -> list[Region]:
    """Detections ``(box, score)`` as ellipse-blur regions."""
    out = []
    for box, score in dets:
        m = ellipse_mask(box, shape, scale)
        b = mask_box(m)
        if b is not None:
            meta = {"box": [float(v) for v in box[:4]]}
            out.append(Region.from_mask(kind, float(score), m, b, meta))
    return out


def padded_box(
    box: Sequence[float], shape: tuple[int, int], padding: float
) -> tuple[int, int, int, int] | None:
    """``box`` grown by ``padding`` of its width/height on each side and
    clipped to the image, rounded outwards as UrbanAnonymizer's
    ``apply_masks``; None if nothing is left."""
    h, w = shape
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    dx, dy = (x1 - x0) * padding, (y1 - y0) * padding
    out = (
        max(0, math.floor(x0 - dx)),
        max(0, math.floor(y0 - dy)),
        min(w, math.ceil(x1 + dx)),
        min(h, math.ceil(y1 + dy)),
    )
    return out if out[2] > out[0] and out[3] > out[1] else None


def to_box_regions(
    kind: str,
    dets: Sequence[tuple[Sequence[float], float]],
    shape: tuple[int, int],
    padding: float,
) -> list[Region]:
    """Detections ``(box, score)`` as padded-rectangle regions."""
    out = []
    for box, score in dets:
        b = padded_box(box, shape, padding)
        if b is None:
            continue
        m = np.zeros(shape, bool)
        m[b[1] : b[3], b[0] : b[2]] = True
        meta = {"box": [float(v) for v in box[:4]]}
        out.append(Region.from_mask(kind, float(score), m, b, meta))
    return out


def box_sliced_mask(mask: BoolMask, box: Sequence[float]) -> BoolMask:
    """The part of ``mask`` the Autoware anonymizer blurs: its
    ``blur_detections`` slices ``[int(y0):int(y1), int(x0):int(x1)]``
    (Python slicing, so a start below -1 wraps) and blurs the mask
    pixels inside that slice."""
    x0, y0, x1, y1 = (int(v) for v in box[:4])
    out = np.zeros_like(mask, dtype=bool)
    out[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return out


class Deface:
    """deface 1.5 (CenterFace, faces only) with its CLI defaults."""

    name = "deface"

    def __init__(
        self, threshold: float = 0.2, mask_scale: float = 1.3
    ) -> None:
        self.models: dict[tuple[int, int], Any] = {}
        self.threshold = threshold
        self.mask_scale = mask_scale

    def _model(self, shape: tuple[int, int]) -> Any:
        """One CenterFace per image size: a CenterFace keeps the input
        shape of its first image, so a later image of another size is
        decoded wrongly (thousands of boxes along the border)."""
        if shape not in self.models:
            from deface.centerface import CenterFace

            self.models[shape] = CenterFace(in_shape=None, backend="auto")
        return self.models[shape]

    def __call__(self, rgb: U8) -> list[Region]:
        dets, _ = self._model(_hw(rgb))(rgb, threshold=self.threshold)
        pairs = [(d[:4], float(d[4])) for d in dets]
        return to_regions("face", pairs, _hw(rgb), self.mask_scale)


class EgoBlur:
    """EgoBlur gen1 (faces + plates, TorchScript) with its CLI defaults."""

    name = "egoblur"

    def __init__(
        self,
        weights: Path,
        threshold: float = 0.9,
        nms_iou: float = 0.3,
        scale: float = 1.0,
        device: str = "cuda",
    ) -> None:
        import torch
        import torchvision  # noqa: F401  registers torchvision::nms for jit

        self.face = (
            torch.jit.load(  # type: ignore[no-untyped-call]
                str(weights / "ego_blur_face.jit"), map_location="cpu"
            )
            .to(device)
            .eval()
        )  # as in the official demo
        self.plate = (
            torch.jit.load(  # type: ignore[no-untyped-call]
                str(weights / "ego_blur_lp.jit"), map_location="cpu"
            )
            .to(device)
            .eval()
        )  # as in the official demo
        self.threshold, self.nms_iou, self.scale = threshold, nms_iou, scale
        self.device = device

    def _detect(
        self, model: object, bgr: U8
    ) -> list[tuple[list[float], float]]:
        import torch
        import torchvision

        tensor = torch.from_numpy(np.transpose(bgr, (2, 0, 1))).to(self.device)
        with torch.no_grad():
            boxes, _, scores, _ = model(tensor)  # type: ignore[operator]
        keep = torchvision.ops.nms(boxes, scores, self.nms_iou)
        b = boxes[keep].cpu().numpy()
        s = scores[keep].cpu().numpy()
        return [
            (list(map(float, bb)), float(ss))
            for bb, ss in zip(b, s, strict=True)
            if ss > self.threshold
        ]

    def __call__(self, rgb: U8) -> list[Region]:
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        shape = _hw(rgb)
        return to_regions(
            "face", self._detect(self.face, bgr), shape, self.scale
        ) + to_regions(
            "plate", self._detect(self.plate, bgr), shape, self.scale
        )


class EgoBlur2:
    """EgoBlur Gen2 (faces + plates, TorchScript, ``egoblur`` package)
    with its ``camera-rgb`` defaults: score thresholds 0.67416 (face) /
    0.74475 (plate), NMS IoU 0.5, short edge resized to 1200 px."""

    name = "egoblur2"

    def __init__(
        self,
        weights: Path,
        face_threshold: float = 0.67416,
        plate_threshold: float = 0.74475,
        nms_iou: float = 0.5,
        scale: float = 1.15,
        device: str = "cuda",
    ) -> None:
        from gen2.script.predictor import ClassID, EgoblurDetector

        resize = {"min_size_test": 1200, "max_size_test": 1200}
        self.detectors = {
            kind: EgoblurDetector(
                model_path=str(weights / f"ego_blur_{name}_gen2.jit"),
                device=device,
                detection_class=cid,
                score_threshold=thr,
                nms_iou_threshold=nms_iou,
                resize_aug=resize,
            )
            for kind, name, cid, thr in (
                ("face", "face", ClassID.FACE, face_threshold),
                ("plate", "lp", ClassID.LICENSE_PLATE, plate_threshold),
            )
        }
        self.scale, self.device = scale, device

    def __call__(self, rgb: U8) -> list[Region]:
        import torch
        from gen2.script.detectron2.export.torchscript_patch import (
            patch_instances,
        )
        from gen2.script.predictor import PATCH_INSTANCES_FIELDS

        bgr = np.ascontiguousarray(rgb[..., ::-1].transpose(2, 0, 1))
        tensor = torch.from_numpy(bgr).to(self.device)[None]
        shape = _hw(rgb)
        out: list[Region] = []
        with patch_instances(fields=PATCH_INSTANCES_FIELDS):
            for kind, det in self.detectors.items():
                batch, orig_hw, in_hw = det.pre_process(tensor)
                boxes, scores = det._post_process(
                    det.inference(batch),
                    model_input_hw_list=in_hw,
                    target_img_hw_list=orig_hw,
                )
                pairs = [
                    (list(map(float, b)), float(s))
                    for b, s in zip(boxes[0], scores[0], strict=True)
                ]
                out += to_regions(kind, pairs, shape, self.scale)
        return out


class UrbanAnonymizer:
    """UrbanAnonymizer D-FINE-X @1408 (faces + plates) with its
    recommended thresholds 0.1 / 0.2 and 20 % mask padding.

    The detector runs in its own venv (``urban_runner.py``); this class
    talks to it over a pipe, one image path per call.
    """

    name = "urban"

    def __init__(
        self,
        repo: Path,
        python: Path | None = None,
        face_score: float = 0.1,
        plate_score: float = 0.2,
        padding: float = 0.2,
    ) -> None:
        runner = Path(__file__).with_name("urban_runner.py")
        cmd = [str(python or repo / ".venv" / "bin" / "python"), "-I"]
        cmd += [str(runner), "--repo", str(repo)]
        cmd += ["--face-score", str(face_score)]
        cmd += ["--plate-score", str(plate_score)]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        self.padding = padding
        if not self._read().get("ready"):
            raise RuntimeError("urban runner did not start")

    def _read(self) -> dict[str, Any]:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("urban runner exited")
        out: dict[str, Any] = json.loads(line)
        return out

    def run(
        self, path: Path, shape: tuple[int, int]
    ) -> tuple[list[Region], float]:
        """Regions for the image at ``path`` and the detector seconds."""
        assert self.proc.stdin is not None
        self.proc.stdin.write(f"{path}\n")
        self.proc.stdin.flush()
        res = self._read()
        out: list[Region] = []
        for kind, cls in (("face", 0), ("plate", 1)):
            pairs = [(d[2:6], float(d[1])) for d in res["dets"] if d[0] == cls]
            out += to_box_regions(kind, pairs, shape, self.padding)
        return out, float(res["seconds"])

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        self.proc.wait()


class AutowareAnonymizer:
    """autoware_rosbag2_anonymizer in its unified mode (GroundingDINO
    SwinB + OpenCLIP + bbox validation + YOLO11x, SAM 2.1 masks) with
    the defaults of ``config/anonymize_with_unified_model.yaml``.

    The tool runs in its own venv (``autoware_runner.py``); this class
    talks to it over a pipe, one image path per call.
    """

    name = "autoware"
    kinds = {"license plate": "plate", "human face": "face"}

    def __init__(self, repo: Path, python: Path | None = None) -> None:
        runner = Path(__file__).with_name("autoware_runner.py")
        cmd = [str(python or repo / ".venv" / "bin" / "python"), "-I"]
        cmd += [str(runner), "--repo", str(repo)]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        ready = self._read()
        if not ready.get("ready"):
            raise RuntimeError("autoware runner did not start")
        self.classes: list[str] = ready["classes"]

    def _read(self) -> dict[str, Any]:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("autoware runner exited")
        out: dict[str, Any] = json.loads(line)
        return out

    def run(self, path: Path) -> tuple[list[Region], float]:
        """Regions for the image at ``path`` and the tool's seconds."""
        assert self.proc.stdin is not None
        self.proc.stdin.write(f"{path}\n")
        self.proc.stdin.flush()
        res = self._read()
        out: list[Region] = []
        for cls, score, *box, rle in res["dets"]:
            m = box_sliced_mask(decode_rle(rle), box)
            b = mask_box(m)
            if b is not None:
                kind = self.kinds[self.classes[cls]]
                meta = {"box": [float(v) for v in box]}
                out.append(Region.from_mask(kind, float(score), m, b, meta))
        return out, float(res["seconds"])

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        self.proc.wait()
