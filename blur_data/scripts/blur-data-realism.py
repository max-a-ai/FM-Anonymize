"""Seams and realism of the anonymised heads, numbers only.

    uv run python blur_data/scripts/blur-data-realism.py --dataset pp4av \
        --regions-run pp4av-ours --keys outputs/gen-keys/pp4av-subset.json \
        --version blur=blur-pp4av --version gen=gen-pp4av \
        --out outputs/eval/realism-pp4av.json

For every head FM-Anonymize-Gen generates (``--config``):
the seam at the boundary of what each version changed, on a crop of
2x the head (:func:`blur_data.metrics.realism.seam`), and the head crop
(1.3x the box, 299 px) for KID against the original head crops
(Inception-v3, ImageNet weights from torchvision). Images present in
every version are used. No image is shown or written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from blur_data.anonymize.regions import ImageRegions, regions_path
from blur_data.generate.config import GenConfig
from blur_data.generate.crops import crop_square, square_window
from blur_data.generate.routing import is_generated
from blur_data.metrics.realism import U8, kid, seam

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs"
DATA = REPO / "resources" / "data"
EXT = {"v2b": ".jpg", "pp4av": ".png"}


class Inception:
    """Inception-v3 pool features (2048) of 299-px crops."""

    def __init__(self, device: str = "cuda") -> None:
        from torchvision.models import Inception_V3_Weights, inception_v3

        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        net.fc = torch.nn.Identity()
        self.net, self.device = net.eval().to(device), device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device)

    @torch.no_grad()
    def __call__(self, crops: list[U8]) -> np.ndarray:  # type: ignore[type-arg]
        out = []
        for i in range(0, len(crops), 32):
            x = torch.from_numpy(np.stack(crops[i : i + 32])).to(self.device)
            x = (x.float() / 255 - self.mean) / self.std
            out.append(self.net(x.permute(0, 3, 1, 2)).cpu().numpy())
        feats: np.ndarray = np.concatenate(out).astype(np.float64)  # type: ignore[type-arg]
        return feats


def head_crop(img: U8, x0: int, y0: int, side: int) -> U8:
    c = cv2.resize(crop_square(img, x0, y0, side, 0), (299, 299))
    return np.asarray(c, np.uint8)


def load(path: Path) -> U8:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def heads(regs: ImageRegions, cfg: GenConfig) -> list[Any]:
    """Boxes of the regions FM-Anonymize-Gen generates."""
    out = []
    for r in regs.regions:
        if is_generated(r, cfg, (r.rle["size"][0], r.rle["size"][1])):
            out.append(r.box)
    return out


def collect(args: argparse.Namespace, runs: dict[str, str]) -> dict[str, Any]:
    """Seams per version and head crops per version (incl. original)."""
    cfg = GenConfig.from_yaml(args.config)
    src, ext = OUT / args.regions_run, EXT[args.dataset]
    keys = [str(k) for k in json.loads(args.keys.read_text())]
    keys = [
        k
        for k in keys
        if all(
            (OUT / r / "images" / f"{k}{ext}").exists() for r in runs.values()
        )
    ]
    seams: dict[str, list[tuple[float, float]]] = {k: [] for k in runs}
    crops: dict[str, list[U8]] = {"original": [], **{k: [] for k in runs}}
    for key in tqdm(keys, desc="heads"):
        regs = ImageRegions.load(regions_path(src, key))
        boxes = heads(regs, cfg)
        if not boxes:
            continue
        orig = load(DATA / args.dataset / regs.rel_path)
        anon = {
            k: load(OUT / r / "images" / f"{key}{ext}")
            for k, r in runs.items()
        }
        shape = (int(orig.shape[0]), int(orig.shape[1]))
        for box in boxes:
            x0, y0, side = square_window(tuple(box), shape, 2.0, 1.25)
            oc = crop_square(orig, x0, y0, side, 0)
            hx, hy, hs = square_window(tuple(box), shape, 1.3, 1.0)
            crops["original"].append(head_crop(orig, hx, hy, hs))
            for k, img in anon.items():
                s = seam(oc, crop_square(img, x0, y0, side, 0))
                if s is not None:
                    seams[k].append(s)
                crops[k].append(head_crop(img, hx, hy, hs))
    return {"keys": len(keys), "seams": seams, "crops": crops}


def summary(v: list[float]) -> dict[str, float]:
    a = np.asarray(v)
    if not len(a):
        return {"n": 0}
    return {
        "n": len(a),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(EXT), required=True)
    ap.add_argument("--regions-run", required=True)
    ap.add_argument("--keys", type=Path, required=True)
    ap.add_argument("--version", action="append", required=True)
    ap.add_argument(
        "--config", type=Path, default=REPO / "configs" / "generate.yaml"
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    runs = dict(s.split("=", 1) for s in args.version)
    got = collect(args, runs)
    feats = Inception()
    ref = feats(got["crops"]["original"])
    res: dict[str, Any] = {
        "dataset": args.dataset,
        "versions": runs,
        "images": got["keys"],
        "heads": len(ref),
    }
    for k in runs:
        s = got["seams"][k]
        res[k] = {
            "seam_grad": summary([a for a, _ in s]),
            "seam_colour": summary([b for _, b in s]),
            "kid_x1000": 1000 * kid(feats(got["crops"][k]), ref),
        }
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
