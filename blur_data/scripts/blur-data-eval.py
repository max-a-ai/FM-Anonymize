"""Summarise a finished anonymisation run.

    uv run python blur_data/scripts/blur-data-eval.py --run anonymize-000

Prints region counts per kind and per camera from the region files and
writes them to ``outputs/<run>/eval.json``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from blur_data.anonymize.regions import ImageRegions

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="outputs/<run>")
    args = ap.parse_args()

    run_dir = REPO / "outputs" / args.run
    files = sorted((run_dir / "regions").rglob("*.json"))
    kinds: Counter[str] = Counter()
    per_cam: dict[str, Counter[str]] = {}
    stats: Counter[str] = Counter()
    max_edits = 0
    for f in files:
        ir = ImageRegions.load(f)
        cam = ir.key.split("/")[-2]  # [<scene>/]<channel>/<stem>
        per_cam.setdefault(cam, Counter()).update(r.kind for r in ir.regions)
        kinds.update(r.kind for r in ir.regions)
        stats.update({k: v for k, v in ir.stats.items() if k != "seconds"})
        max_edits = max(max_edits, len(ir.regions))
    out = {
        "frames": len(files),
        "regions_by_kind": dict(kinds),
        "regions_by_camera": {c: dict(v) for c, v in sorted(per_cam.items())},
        "detection_totals": dict(stats),
        "max_val_edits_per_image": max_edits,
    }
    (run_dir / "eval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
