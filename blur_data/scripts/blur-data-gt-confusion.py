"""Precision, recall and confusion matrices on the annotated GT frames.

    uv run python blur_data/scripts/blur-data-gt-confusion.py \
        --runs ours=anonymize-003 deface=deface-v2b

Definitions: ``blur_data/metrics/confusion.py``. Writes
``outputs/eval/confusion.json`` and prints the tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blur_data.anonymize.regions import ImageRegions, regions_path
from blur_data.data import pp4av
from blur_data.metrics.confusion import Counts, add_frame, split_small_faces
from blur_data.metrics.privacy import SIZE_BINS, GtBox

REPO = Path(__file__).resolve().parents[2]


def by_size(
    objects: list[tuple[float, bool, bool]],
) -> list[dict[str, Any]]:
    """Privacy and detected recall per object-size bin (``SIZE_BINS``)."""
    out = []
    for lo, hi in SIZE_BINS:
        hits = [(ok, det) for size, ok, det in objects if lo <= size < hi]
        n = len(hits)
        out.append(
            {
                "bin": [lo, hi],
                "n": n,
                "recall": sum(o for o, _ in hits) / n if n else float("nan"),
                "detected_recall": (
                    sum(d for _, d in hits) / n if n else float("nan")
                ),
            }
        )
    return out


def face_box(g: GtBox, scale: float) -> GtBox:
    """Shrink a face box about its centre by ``scale`` (side length)."""
    if g.cls != "face" or scale == 1.0:
        return g
    cx, cy = (g.x0 + g.x1) / 2, (g.y0 + g.y1) / 2
    hw, hh = scale * (g.x1 - g.x0) / 2, scale * (g.y1 - g.y0) / 2
    return GtBox(g.cls, cx - hw, cy - hh, cx + hw, cy + hh)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True, help="name=run")
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument(
        "--min-face-px",
        type=float,
        default=20.0,
        help="GT faces below this sqrt(area) are not identifiable",
    )
    ap.add_argument(
        "--face-box-scale",
        type=float,
        default=1.0,
        help="shrink GT face boxes to the face first (PP4AV draws head"
        " boxes; 0.735 = median CenterFace face / PP4AV box)",
    )
    ap.add_argument(
        "--gt",
        type=Path,
        default=REPO / "outputs/gt/annotations.reviewed.json",
        help="annotation file, or 'pp4av' for the benchmark labels",
    )
    ap.add_argument(
        "--out", type=Path, default=REPO / "outputs/eval/confusion.json"
    )
    args = ap.parse_args()
    if str(args.gt) == "pp4av":
        root = REPO / "resources" / "data" / "pp4av"
        all_gt = {
            im.key: pp4av.read_gt(root, im) for im in pp4av.list_images(root)
        }
    else:
        raw = json.loads(args.gt.read_text())
        all_gt = {
            k: [
                GtBox(b["cls"], b["x0"], b["y0"], b["x1"], b["y1"])
                for b in v["boxes"]
            ]
            for k, v in raw.items()
        }
    # faces below the size limit are neutral, not dropped (confusion.py)
    split = {
        k: split_small_faces(
            [face_box(b, args.face_box_scale) for b in v], args.min_face_px
        )
        for k, v in all_gt.items()
    }
    gt = {k: s[0] for k, s in split.items()}
    neutral = {k: s[1] for k, s in split.items()}
    result = {}
    for spec in args.runs:
        name, run = spec.split("=", 1)
        per = {"face": Counts(), "plate": Counts()}
        secs = [
            json.loads(line)["seconds"]
            for line in (REPO / "outputs" / run / "summary.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]
        for key, boxes in gt.items():
            regs = ImageRegions.load(
                regions_path(REPO / "outputs" / run, key)
            ).regions
            for cls, c in per.items():
                add_frame(c, cls, boxes, regs, args.tau, neutral=neutral[key])
        result[name] = {
            cls: {
                "precision": c.precision,
                "recall": c.recall,
                "detected_recall": c.detected_recall,
                "detected": c.detected,
                "tp": c.tp,
                "fp": c.fp,
                "neutral_regions": c.neutral,
                "fn": c.fn,
                "matched_regions": c.matched,
                "frame": c.frame,
                "by_size": by_size(c.objects),
            }
            for cls, c in per.items()
        }
        result[name]["seconds"] = sum(secs) / max(1, len(secs))
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "tau": args.tau,
                "min_face_px": args.min_face_px,
                "face_box_scale": args.face_box_scale,
                "frames": len(gt),
                **result,
            },
            indent=1,
        )
    )
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
