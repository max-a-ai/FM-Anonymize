"""Score every method on a ground truth and write the paper's numbers.

    uv run python blur_data/scripts/blur-data-paper-eval.py \
        --gt own  --runs ours=anonymize-003 deface=deface-v2b \
        --out outputs/eval/own.json
    uv run python blur_data/scripts/blur-data-paper-eval.py \
        --gt pp4av --runs ours=pp4av-ours deface=deface-pp4av \
        --out outputs/eval/pp4av.json

``own`` reads ``outputs/gt/annotations.json`` (the annotation tool's
output); ``pp4av`` reads the benchmark labels of the images every run
has processed. Per method: privacy recall of faces and plates at
``--tau`` (overall and per size bin), the recall curve over tau,
over-blur, false regions per image and seconds per image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from blur_data.anonymize.regions import ImageRegions, regions_path
from blur_data.data import pp4av
from blur_data.metrics.privacy import (
    SIZE_BINS,
    GtBox,
    ImageScore,
    recall,
    score_image,
)

REPO = Path(__file__).resolve().parents[2]
TAUS = [round(t, 2) for t in np.linspace(0.1, 1.0, 19)]


def own_gt(path: Path) -> dict[str, list[GtBox]]:
    """The annotation tool's file -> key -> boxes."""
    raw: dict[str, Any] = json.loads(path.read_text())
    return {
        k: [
            GtBox(b["cls"], b["x0"], b["y0"], b["x1"], b["y1"])
            for b in v["boxes"]
        ]
        for k, v in raw.items()
    }


def pp4av_gt(root: Path) -> dict[str, list[GtBox]]:
    """PP4AV pinhole images -> key -> boxes."""
    return {im.key: pp4av.read_gt(root, im) for im in pp4av.list_images(root)}


def score_run(run: str, gt: dict[str, list[GtBox]]) -> list[ImageScore]:
    """Scores of one run on every ground-truth image."""
    out = []
    for key, boxes in gt.items():
        path = regions_path(REPO / "outputs" / run, key)
        if not path.exists():
            raise SystemExit(f"{run}: no regions for {key}")
        regs = ImageRegions.load(path).regions
        masks = [r.mask() for r in regs]
        if masks:
            blur = np.logical_or.reduce(masks)
        else:
            h, w = _shape(path)
            blur = np.zeros((h, w), dtype=bool)
        out.append(score_image(boxes, blur, masks))
    return out


def _shape(path: Path) -> tuple[int, int]:
    """Image shape of a region file without regions (from its rel_path)."""
    from PIL import Image

    ir = ImageRegions.load(path)
    for root in ("v2b", "pp4av"):
        img = REPO / "resources" / "data" / root / ir.rel_path
        if img.exists():
            with Image.open(img) as im:
                return im.size[1], im.size[0]
    raise SystemExit(f"image of {path} not found")


def seconds(run: str) -> float:
    """Mean seconds per image from the run's summary.jsonl."""
    rows = [
        json.loads(line)
        for line in (REPO / "outputs" / run / "summary.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    return float(np.mean([r["seconds"] for r in rows]))


def summarise(scores: list[ImageScore], tau: float) -> dict[str, Any]:
    """Everything the paper reports for one method."""
    res: dict[str, Any] = {}
    for cls in ("face", "plate"):
        r, n = recall(scores, cls, tau)
        res[cls] = {
            "recall": r,
            "n": n,
            "by_size": [
                {
                    "bin": list(b),
                    "recall": recall(scores, cls, tau, b)[0],
                    "n": recall(scores, cls, tau, b)[1],
                }
                for b in SIZE_BINS
            ],
            "curve": [recall(scores, cls, t)[0] for t in TAUS],
        }
    res["over_blur_pct"] = 100 * float(np.mean([s.over_blur for s in scores]))
    res["false_regions"] = float(np.mean([s.false_regions for s in scores]))
    res["regions"] = float(np.mean([s.regions for s in scores]))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", choices=["own", "pp4av"], required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="name=run")
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    gt = (
        own_gt(REPO / "outputs" / "gt" / "annotations.json")
        if args.gt == "own"
        else pp4av_gt(REPO / "resources" / "data" / "pp4av")
    )
    result: dict[str, Any] = {
        "gt": args.gt,
        "tau": args.tau,
        "taus": TAUS,
        "images": len(gt),
        "methods": {},
    }
    for spec in args.runs:
        name, run = spec.split("=", 1)
        res = summarise(score_run(run, gt), args.tau)
        res["seconds"] = seconds(run)
        res["run"] = run
        result["methods"][name] = res
        print(
            f"{name:12s} face {res['face']['recall']:.3f}"
            f" (n={res['face']['n']})"
            f"  plate {res['plate']['recall']:.3f} (n={res['plate']['n']})"
            f"  over-blur {res['over_blur_pct']:.2f}%"
            f"  false/img {res['false_regions']:.2f}  {res['seconds']:.2f}s"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
