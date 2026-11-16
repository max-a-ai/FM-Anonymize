"""UrbanAnonymizer detection server, run in its own venv.

    third_party/urbananonymizer/.venv/bin/python -I \
        blur_data/baselines/urban_runner.py --repo third_party/urbananonymizer

UrbanAnonymizer pins torch 2.5.1 and vendors D-FINE, so it runs in a
separate interpreter (stdlib + its own requirements, no ``blur_data``
imports). It reads one image path per stdin line and answers with one
JSON line ``{"dets": [[cls, score, x0, y0, x1, y1], ...], "seconds":
s}``: the raw detector boxes (``cls`` 0 = face, 1 = plate) before the
tool's mask padding, kept at its per-class thresholds, as its
``scripts/anonymize.py`` does for images (full frame, no tiles).
``seconds`` covers preprocessing, inference and postprocessing; the
image decode and the model load are excluded, as for the other
baselines.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any


def load_detector(repo: Path, size: int) -> Any:
    """The tool's own ``Detector`` with its recommended checkpoint."""
    spec = importlib.util.spec_from_file_location(
        "urban_anonymize", repo / "scripts" / "anonymize.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"no anonymize.py in {repo}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ckpt = repo / "dfine_x_1408"
    return mod.Detector(
        str(ckpt / "config.yml"), str(ckpt / "model.pth"), size=size
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--size", type=int, default=1408)
    ap.add_argument("--face-score", type=float, default=0.1)
    ap.add_argument("--plate-score", type=float, default=0.2)
    args = ap.parse_args()
    out = sys.stdout
    sys.stdout = sys.stderr  # keep library prints off the protocol
    from PIL import Image

    det = load_detector(args.repo.resolve(), args.size)
    thr = {0: args.face_score, 1: args.plate_score}
    out.write(json.dumps({"ready": True}) + "\n")
    out.flush()
    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        with Image.open(path) as im:
            image = im.convert("RGB")
        t0 = time.time()
        raw = det.predict(image, score_threshold=min(thr.values()))
        secs = time.time() - t0
        dets = [
            [d["category_id"], d["score"], *d["box"]]
            for d in raw
            if d["score"] >= thr.get(d["category_id"], 1.0)
        ]
        out.write(json.dumps({"dets": dets, "seconds": secs}) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
