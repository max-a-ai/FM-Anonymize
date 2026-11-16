"""autoware_rosbag2_anonymizer unified-mode server, run in its own venv.

    third_party/autoware_rosbag2_anonymizer/.venv/bin/python -I \
        blur_data/baselines/autoware_runner.py \
        --repo third_party/autoware_rosbag2_anonymizer

The tool only reads ROS 2 bags, so this imports its own model classes
and does per image exactly what ``tools/anonymize_with_unified_model.py``
does per image message, with ``config/anonymize_with_unified_model.yaml``
and the repo's ``validation.json``:

1. ``UnifiedLanguageModel(image)``: GroundingDINO SwinB on all prompts,
   NMS, OpenCLIP / bbox-inside validation of the "license plate" and
   "human face" boxes, YOLO11x anonymizer merged in, NMS again;
2. ``SAM2(image, detections)`` (``blur.region: mask``).

The image is the BGR array ``cv_bridge`` would hand the tool (it is
passed on to every model as the tool does). The package ``__init__``
imports the ROS bag reader (``rosbag2_py``), so the package is
registered without running it; nothing else is changed.

Reads one image path per stdin line and answers with one JSON line
``{"dets": [[cls, score, x0, y0, x1, y1, rle], ...], "seconds": s}``:
``cls`` indexes the tool's detection classes (0 = license plate,
1 = human face), ``rle`` is the COCO RLE of the SAM 2.1 mask (the tool
blurs that mask inside the integer box). ``seconds`` covers the two
steps above, not the image decode or the model load.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path
from typing import Any


def config(weights: Path) -> dict[str, Any]:
    """``config/anonymize_with_unified_model.yaml`` with the weights
    paths pointed at ``weights/`` (the ROS bag keys are unused)."""
    return {
        "grounding_dino": {
            "config_path": str(weights / "GroundingDINO_SwinB.cfg.py"),
            "checkpoint_path": str(
                weights / "groundingdino_swinb_cogcoor.pth"
            ),
            "box_threshold": 0.1,
            "text_threshold": 0.1,
            "nms_threshold": 0.1,
        },
        "segment_anything_2": {
            "model_cfg": "configs/sam2.1/sam2.1_hiera_s.yaml",
            "checkpoint_path": str(weights / "sam2.1_hiera_small.pt"),
        },
        "openclip": {
            "run": True,
            "model_name": "ViT-B-32",
            "pretrained_model": "laion2b_s34b_b79k",
            "score_threshold": 0.7,
        },
        "yolo": {
            "model": str(weights / "yolo11x_anonymizer.pt"),
            "config_path": str(weights / "yolo_config.yaml"),
            "confidence": 0.15,
        },
        "bbox_validation": {"iou_threshold": 0.9},
        "blur": {"region": "mask", "kernel_size": 31, "sigma_x": 19},
    }


def register_package(repo: Path) -> None:
    """Make ``autoware_rosbag2_anonymizer.*`` importable without its
    ``__init__`` (which needs ROS)."""
    pkg = types.ModuleType("autoware_rosbag2_anonymizer")
    pkg.__path__ = [str(repo / "autoware_rosbag2_anonymizer")]
    sys.modules["autoware_rosbag2_anonymizer"] = pkg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=None)
    args = ap.parse_args()
    out = sys.stdout
    sys.stdout = sys.stderr  # keep library prints off the protocol
    repo = args.repo.resolve()
    register_package(repo)

    import cv2
    import numpy as np
    import torch
    from autoware_rosbag2_anonymizer.model.sam2 import SAM2
    from autoware_rosbag2_anonymizer.model.unified_language_model import (
        UnifiedLanguageModel,
    )
    from pycocotools import mask as mask_util

    device = "cuda" if torch.cuda.is_available() else "cpu"  # as main.py
    cfg = config((args.weights or repo / "weights").resolve())
    json_data = json.loads((repo / "validation.json").read_text())
    sam2 = SAM2(
        cfg["segment_anything_2"]["model_cfg"],
        cfg["segment_anything_2"]["checkpoint_path"],
        device,
    )
    ulm = UnifiedLanguageModel(cfg, json_data, device)
    if not ulm.is_yolo_model_exist:
        raise FileNotFoundError("YOLO weights or yolo_config.yaml missing")
    out.write(
        json.dumps({"ready": True, "classes": ulm.detection_classes}) + "\n"
    )
    out.flush()
    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        image = cv2.imread(path)  # BGR, as cv_bridge gives the tool
        t0 = time.time()
        dets = ulm(image)
        dets = sam2(image=image, detections=dets)
        if device == "cuda":
            torch.cuda.synchronize()
        secs = time.time() - t0
        h, w = image.shape[:2]
        masks = dets.mask if dets.mask is not None else np.zeros((0, h, w))
        res = []
        for xyxy, conf, cid, m in zip(
            dets.xyxy, dets.confidence, dets.class_id, masks, strict=True
        ):
            rle = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
            rle_j = {
                "size": list(rle["size"]),
                "counts": rle["counts"].decode(),
            }
            res.append([int(cid), float(conf), *map(float, xyxy), rle_j])
        out.write(json.dumps({"dets": res, "seconds": secs}) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
