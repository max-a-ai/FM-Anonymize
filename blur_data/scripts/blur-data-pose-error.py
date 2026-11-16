"""Does the anonymised head keep the person's head pose?

    uv run --group inpaint python blur_data/scripts/blur-data-pose-error.py \
        --dataset v2b --regions-run anonymize-002 \
        --keys outputs/gt/annotations.reviewed.json \
        --version blur=blur-gt --version gen=gen-gt \
        --out outputs/eval/pose-error-gt.json

SAM 3D Body is fitted on the original and on each version
(``outputs/<run>/images``) with the same person box (``meta.person_box``
of every head FM-Anonymize-Gen generates, ``--config``) and the frame's
K (PP4AV: SAM 3D Body's default). Per head and version, against the fit
on the original: the facing-angle difference and the head rotation
difference (geodesic angle between the head frames: up = neck -> ears,
forward = ears -> nose). Median and 90th percentile in degrees. Numbers
only; no image is shown or written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from blur_data.anonymize.head import F64, head_pose
from blur_data.anonymize.regions import ImageRegions, regions_path
from blur_data.data import pp4av
from blur_data.generate.config import GenConfig
from blur_data.generate.pose_guide import head_rotation, rotation_deg
from blur_data.generate.routing import is_generated

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs"
DATA = REPO / "resources" / "data"
CKPT = REPO / "resources" / "pretrained-checkpoints"
EXT = {"v2b": ".jpg", "pp4av": ".png"}


def heads(regs: ImageRegions, cfg: GenConfig) -> list[Any]:
    """Person boxes of the heads FM-Anonymize-Gen generates."""
    out = []
    for r in regs.regions:
        shape = (r.rle["size"][0], r.rle["size"][1])
        pb = r.meta.get("person_box")
        if r.kind == "head" and pb is not None and is_generated(r, cfg, shape):
            out.append((pb[0], pb[1], pb[2], pb[3]))
    return out


def poses(body: Any, path: Path, boxes: list[Any], k: F64) -> list[Any]:
    """(facing angle, head frame) per box; None where the fit failed."""
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    try:
        fits = body.fit(rgb, boxes, k)
    except (RuntimeError, ValueError, IndexError):
        return [None] * len(boxes)
    out = []
    for f in fits:
        p = head_pose(f.keypoints)
        out.append((p.angle_deg, head_rotation(p.up, p.forward)))
    return out


def summary(v: list[float]) -> dict[str, float]:
    a = np.asarray(v)
    if not len(a):
        return {"n": 0}
    return {
        "n": len(a),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
    }


def k_matrix(dataset: str, regs: ImageRegions, kv2b: dict[str, F64]) -> F64:
    if dataset == "v2b":
        return kv2b[regs.key]
    h, w = regs.regions[0].rle["size"]
    return pp4av.Pp4avImage(regs.rel_path, int(w), int(h)).k_matrix()


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(EXT), required=True)
    ap.add_argument("--regions-run", required=True)
    ap.add_argument("--keys", type=Path, default=None, help="json list/obj")
    ap.add_argument("--version", action="append", required=True)
    ap.add_argument(
        "--config", type=Path, default=REPO / "configs" / "generate.yaml"
    )
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args()


def main() -> None:
    from blur_data.data.v2b import list_camera_images
    from blur_data.models.body import BodyFitter

    args = parse()
    runs = dict(s.split("=", 1) for s in args.version)
    cfg = GenConfig.from_yaml(args.config)
    src = OUT / args.regions_run
    if args.keys:
        keys = [str(k) for k in json.loads(args.keys.read_text())]
    else:
        root = src / "regions"
        keys = sorted(
            str(p.relative_to(root).with_suffix(""))
            for p in root.rglob("*.json")
        )
    kv2b: dict[str, F64] = {}
    if args.dataset == "v2b":
        kv2b = {
            im.key: im.k_matrix() for im in list_camera_images(DATA / "v2b")
        }
    body = BodyFitter(CKPT / "sam-3d-body")
    facing: dict[str, list[float]] = {k: [] for k in runs}
    rot: dict[str, list[float]] = {k: [] for k in runs}
    failed = {k: 0 for k in ["original", *runs]}
    n = 0
    for key in tqdm(keys, desc="pose"):
        regs = ImageRegions.load(regions_path(src, key))
        boxes = heads(regs, cfg)
        if not boxes:
            continue
        k = k_matrix(args.dataset, regs, kv2b)
        ref = poses(body, DATA / args.dataset / regs.rel_path, boxes, k)
        n += len(boxes)
        failed["original"] += sum(p is None for p in ref)
        ext = EXT[args.dataset]
        for name, run in runs.items():
            got = poses(body, OUT / run / "images" / f"{key}{ext}", boxes, k)
            for a, b in zip(ref, got, strict=True):
                if a is None or b is None:
                    failed[name] += b is None
                    continue
                facing[name].append(abs(a[0] - b[0]))
                rot[name].append(rotation_deg(a[1], b[1]))
    res = {
        "dataset": args.dataset,
        "versions": runs,
        "heads": n,
        "failed_fits": failed,
        "facing_deg": {k: summary(v) for k, v in facing.items()},
        "rotation_deg": {k: summary(v) for k, v in rot.items()},
    }
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
