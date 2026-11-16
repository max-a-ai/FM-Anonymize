"""Does a person / keypoint detector still work on anonymised images?

    uv run python blur_data/scripts/blur-data-utility.py --dataset v2b \
        --version reencode=reencode-gt --version blur=blur-gt \
        --version gen=gen-gt --out outputs/eval/utility-gt.json

torchvision ``keypointrcnn_resnet50_fpn`` (COCO) runs on the original
image and on each version (``outputs/<run>/images/<key>.<ext>``, as
written by ``blur-data-generate.py``). Its persons with score >=
``--score-thr`` on the original are the pseudo-ground truth, their
keypoints with score > ``--kp-thr`` the visible ones. Every version
(the original too, as the ceiling) is scored with box AP, keypoint AP
and head-keypoint AP, overall and per person size, for all persons and
for those whose head FM-Anonymize-Gen generates (``anonymised``; its
regions per ``--config``, the others ignored)
(:mod:`blur_data.metrics.utility`), and the persons a version adds or
loses against the original (``extra_lost``). Images present in every version
are used. The detections are cached in ``outputs/<out stem>/preds.json``
(per image path), so a re-scoring with other thresholds needs no GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from tqdm import tqdm

from blur_data.anonymize.regions import ImageRegions, regions_path
from blur_data.generate.config import GenConfig
from blur_data.generate.routing import is_generated
from blur_data.metrics.utility import (
    HEAD,
    Persons,
    coco_ap,
    dt_annotations,
    extra_lost,
    gt_annotations,
    subset,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs"
DATA = REPO / "resources" / "data"
EXT = {"v2b": ".jpg", "pp4av": ".png"}


class Detector:
    """torchvision Keypoint R-CNN (ResNet-50 FPN, COCO weights)."""

    def __init__(self, device: str = "cuda") -> None:
        from torchvision.models.detection import (
            KeypointRCNN_ResNet50_FPN_Weights,
            keypointrcnn_resnet50_fpn,
        )

        weights = KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
        self.model = keypointrcnn_resnet50_fpn(weights=weights)
        self.model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, rgb: npt.NDArray[np.uint8]) -> Persons:
        x = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1)
        out = self.model([x.float() / 255.0])[0]
        kp = out["keypoints"].clone()
        kp[..., 2] = out["keypoints_scores"]
        return Persons(
            out["boxes"].cpu().numpy().astype(np.float32),
            out["scores"].cpu().numpy().astype(np.float32),
            kp.cpu().numpy().astype(np.float32),
        )


def load_rgb(path: Path) -> npt.NDArray[np.uint8]:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def image_paths(
    dataset: str, runs: dict[str, str], keys: list[str] | None
) -> dict[str, dict[str, Path]]:
    """Key -> version -> image path, for keys present in every run."""
    ext = EXT[dataset]
    found = [
        {
            str(p.relative_to(OUT / r / "images").with_suffix(""))
            for p in (OUT / r / "images").rglob(f"*{ext}")
        }
        for r in runs.values()
    ]
    common = sorted(set.intersection(*found))
    if keys is not None:
        common = [k for k in keys if k in set(common)]
    first = json.loads(
        (OUT / next(iter(runs.values())) / "config.json").read_text()
    )
    src = OUT / first["regions_run"]
    out = {}
    for key in common:
        rel = (
            f"{key}{ext}"
            if dataset == "pp4av"
            else ImageRegions.load(regions_path(src, key)).rel_path
        )
        out[key] = {"original": DATA / dataset / rel} | {
            k: OUT / r / "images" / f"{key}{ext}" for k, r in runs.items()
        }
    return out


def detect_all(
    paths: dict[str, dict[str, Path]], cache: Path
) -> dict[str, dict[int, Persons]]:
    """Version -> image id -> detections; cached per image path."""
    raw: dict[str, Any] = {}
    if cache.exists():
        raw = json.loads(cache.read_text())
    cache.parent.mkdir(parents=True, exist_ok=True)
    det: Detector | None = None
    for i, key in enumerate(tqdm(paths, desc="detect")):
        for path in paths[key].values():
            if str(path) not in raw:
                det = det or Detector()
                raw[str(path)] = det(load_rgb(path)).to_json()
        if det is not None and (i % 200 == 199 or i == len(paths) - 1):
            cache.write_text(json.dumps(raw))
    versions = next(iter(paths.values()))
    return {
        v: {
            i: Persons.from_json(raw[str(paths[k][v])])
            for i, k in enumerate(paths)
        }
        for v in versions
    }


def head_centres(
    paths: dict[str, dict[str, Path]], regions_run: str, cfg: GenConfig
) -> dict[int, list[tuple[float, float]]]:
    """Image id -> centres of the regions FM-Anonymize-Gen generates."""
    out = {}
    for i, key in enumerate(paths):
        regs = ImageRegions.load(regions_path(OUT / regions_run, key))
        out[i] = [
            ((r.box[0] + r.box[2]) / 2, (r.box[1] + r.box[3]) / 2)
            for r in regs.regions
            if is_generated(r, cfg, (r.rle["size"][0], r.rle["size"][1]))
        ]
    return out


def score(
    preds: dict[str, dict[int, Persons]],
    score_thr: float,
    kp_thr: float,
    centres: dict[int, list[tuple[float, float]]],
) -> dict[str, Any]:
    """Box, keypoint and head-keypoint AP of every version.

    ``all`` persons and ``anonymised`` ones (a generated region in the
    top half of the box; the others ignored).
    """
    ids = list(preds["original"])
    gts = {
        "all": gt_annotations(preds["original"], score_thr, kp_thr),
        "head": gt_annotations(preds["original"], score_thr, kp_thr, HEAD),
    }
    res: dict[str, Any] = {}
    for part in ("all", "anonymised"):
        gt_all, gt_head = gts["all"], gts["head"]
        if part == "anonymised":
            gt_all, gt_head = subset(gt_all, centres), subset(gt_head, centres)
        for v, p in preds.items():
            res.setdefault(part, {})[v] = {
                "box": coco_ap(gt_all, dt_annotations(p, None), ids, None),
                "keypoints": coco_ap(
                    gt_all, dt_annotations(p, (0, 17)), ids, (0, 17)
                ),
                "head": coco_ap(gt_head, dt_annotations(p, HEAD), ids, HEAD),
            }
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(EXT), required=True)
    ap.add_argument("--version", action="append", required=True)
    ap.add_argument("--keys", type=Path, default=None, help="json list")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument("--kp-thr", type=float, default=2.0)
    ap.add_argument(
        "--config", type=Path, default=REPO / "configs" / "generate.yaml"
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    runs = dict(s.split("=", 1) for s in args.version)
    keys = json.loads(args.keys.read_text()) if args.keys else None
    paths = image_paths(args.dataset, runs, keys)
    paths = dict(list(paths.items())[: args.limit])
    cache = OUT / args.out.stem / "preds.json"
    preds = detect_all(paths, cache)
    first = json.loads(
        (OUT / next(iter(runs.values())) / "config.json").read_text()
    )
    cfg = GenConfig.from_yaml(args.config)
    centres = head_centres(paths, first["regions_run"], cfg)
    res = {
        "dataset": args.dataset,
        "versions": runs,
        "n_images": len(paths),
        "score_thr": args.score_thr,
        "kp_thr": args.kp_thr,
        "detector": "torchvision keypointrcnn_resnet50_fpn COCO_V1",
        "scores": score(preds, args.score_thr, args.kp_thr, centres),
        "extra_lost": {
            v: extra_lost(preds["original"], p) for v, p in preds.items()
        },
    }
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["scores"], indent=1))


if __name__ == "__main__":
    main()
