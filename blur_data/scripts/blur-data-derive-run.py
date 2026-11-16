"""Derive a run by dropping region kinds from an existing run.

    uv run python blur_data/scripts/blur-data-derive-run.py \
        --src anonymize-004 --dst abl-no-facenet --drop face

Exact for steps that only *add* regions at the end of the pipeline
(the SAM 3 face safety net), so no GPU re-run is needed for them.

``--grow-faces 1.36`` scales every face region (``FACE_KINDS``) about
its centre, as ``face_region_scale`` does as the pipeline's last step.

``--rescale face=1.15`` instead redraws a baseline's ellipses from the
stored detection boxes with another scale (EgoBlur's
``scale_factor_detections``), again without re-running the detector.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from blur_data.anonymize.regions import FACE_KINDS, ImageRegions
from blur_data.baselines import to_regions

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--drop", nargs="*", default=[], help="region kinds")
    ap.add_argument(
        "--rescale", nargs="*", default=[], help="kind=scale, ellipses"
    )
    ap.add_argument("--grow-faces", type=float, default=1.0)
    args = ap.parse_args()
    scales = {k: float(v) for k, v in (a.split("=") for a in args.rescale)}
    src = REPO / "outputs" / args.src
    dst = REPO / "outputs" / args.dst
    files = sorted((src / "regions").rglob("*.json"))
    for f in files:
        ir = ImageRegions.load(f)
        ir.regions = [r for r in ir.regions if r.kind not in args.drop]
        ir.regions = [
            r.grown(args.grow_faces) if r.kind in FACE_KINDS else r
            for r in ir.regions
        ]
        if scales:
            kept = []
            for r in ir.regions:
                if r.kind not in scales:
                    kept.append(r)
                    continue
                h, w = r.rle["size"]
                kept += to_regions(
                    r.kind,
                    [(r.meta["box"], r.score)],
                    (int(h), int(w)),
                    scales[r.kind],
                )
            ir.regions = kept
        ir.stats = {**ir.stats, "derived_from": args.src}
        ir.save(dst / "regions" / f.relative_to(src / "regions"))
    shutil.copy2(src / "summary.jsonl", dst / "summary.jsonl")
    (dst / "config.json").write_text(
        json.dumps(
            {
                "derived_from": args.src,
                "dropped": args.drop,
                "rescaled": scales,
                "grow_faces": args.grow_faces,
            },
            indent=2,
        )
    )
    print(f"{len(files)} region files -> {dst}")


if __name__ == "__main__":
    main()
