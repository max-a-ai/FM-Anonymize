"""Run a baseline anonymiser and write region files like ours.

    uv run python blur_data/scripts/blur-data-baseline.py \
        --method deface --dataset v2b --run deface-v2b
    uv run python blur_data/scripts/blur-data-baseline.py \
        --method egoblur --dataset pp4av --run egoblur-pp4av
    uv run python blur_data/scripts/blur-data-baseline.py \
        --method urban --dataset v2b --keys outputs/gt/keys.json \
        --run urban-gt

Same output layout as ``blur-data-anonymize.py`` (``regions/``,
``summary.jsonl`` with seconds per image, ``DONE``), so rendering and
scoring treat every method alike. EgoBlur needs its two ``.jit`` files
in ``--egoblur-weights`` (default ``third_party/egoblur/models``, a
local clone of the lab mirror; untracked). EgoBlur Gen2 needs
``ego_blur_{face,lp}_gen2.jit`` in ``--egoblur2-weights`` (default
``third_party/egoblur/models_gen2``; download from
projectaria.com/tools/egoblur after accepting the licence) and is
scored with its 1.15 detection scale. UrbanAnonymizer runs in its own
venv in ``--urban-repo`` (default ``third_party/urbananonymizer``, a
clone with ``dfine_x_1408/`` downloaded; untracked). The Autoware
rosbag2 anonymizer (unified mode) runs in its own venv in
``--autoware-repo`` (default ``third_party/autoware_rosbag2_anonymizer``,
a clone with its weights in ``weights/``; untracked).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from blur_data.anonymize.regions import ImageRegions, regions_path
from blur_data.baselines import (
    AutowareAnonymizer,
    Deface,
    EgoBlur,
    EgoBlur2,
    UrbanAnonymizer,
)
from blur_data.data import pp4av
from blur_data.data.v2b import CameraImage, list_camera_images

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "resources"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--method",
        choices=["deface", "egoblur", "egoblur2", "urban", "autoware"],
        required=True,
    )
    ap.add_argument("--dataset", choices=["v2b", "pp4av"], default="v2b")
    ap.add_argument("--run", required=True, help="outputs/<run>")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="score threshold (default: the tool's own; deface, EgoBlur"
        " Gen1 and UrbanAnonymizer)",
    )
    ap.add_argument(
        "--egoblur-weights",
        type=Path,
        default=REPO / "third_party" / "egoblur" / "models",
    )
    ap.add_argument(
        "--egoblur2-weights",
        type=Path,
        default=REPO / "third_party" / "egoblur" / "models_gen2",
    )
    ap.add_argument(
        "--urban-repo",
        type=Path,
        default=REPO / "third_party" / "urbananonymizer",
    )
    ap.add_argument(
        "--autoware-repo",
        type=Path,
        default=REPO / "third_party" / "autoware_rosbag2_anonymizer",
    )
    ap.add_argument("--keys", type=Path, default=None, help="json key list")
    args = ap.parse_args()

    data = args.data or RES / "data" / args.dataset
    lister = (
        pp4av.list_images if args.dataset == "pp4av" else list_camera_images
    )
    images: list[pp4av.Pp4avImage | CameraImage] = list(lister(data))
    if args.keys:
        wanted = set(json.loads(args.keys.read_text()))
        images = [im for im in images if im.key in wanted]
    run_dir = REPO / "outputs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "method": args.method,
                "dataset": args.dataset,
                "threshold": args.threshold,
            },
            indent=2,
        )
    )
    todo = [im for im in images if not regions_path(run_dir, im.key).exists()]
    print(f"{len(images)} images, {len(todo)} to do -> {run_dir}")
    model: Deface | EgoBlur | EgoBlur2 | UrbanAnonymizer | AutowareAnonymizer
    if args.method == "deface":
        model = Deface() if args.threshold is None else Deface(args.threshold)
    elif args.method == "egoblur":
        model = (
            EgoBlur(args.egoblur_weights)
            if args.threshold is None
            else EgoBlur(args.egoblur_weights, threshold=args.threshold)
        )
    elif args.method == "egoblur2":
        model = EgoBlur2(args.egoblur2_weights)
    elif args.method == "autoware":
        model = AutowareAnonymizer(args.autoware_repo)
    elif args.threshold is None:
        model = UrbanAnonymizer(args.urban_repo)
    else:  # one threshold for both classes (threshold sweeps)
        model = UrbanAnonymizer(
            args.urban_repo,
            face_score=args.threshold,
            plate_score=args.threshold,
        )
    with (run_dir / "summary.jsonl").open("a") as log:
        for im in tqdm(todo, desc=args.method):
            path = data / im.rel_path
            if isinstance(model, UrbanAnonymizer):
                with Image.open(path) as pil:
                    shape = (pil.height, pil.width)
                regions, secs = model.run(path, shape)
            elif isinstance(model, AutowareAnonymizer):
                regions, secs = model.run(path)
            else:
                rgb = np.asarray(Image.open(path).convert("RGB"))
                t0 = time.time()
                regions = model(rgb)
                secs = time.time() - t0
            stats = {"regions": len(regions), "seconds": round(secs, 3)}
            ImageRegions(im.key, im.rel_path, regions, stats).save(
                regions_path(run_dir, im.key)
            )
            log.write(json.dumps({"key": im.key, **stats}) + "\n")
            log.flush()  # a killed run keeps every finished line
    if isinstance(model, UrbanAnonymizer | AutowareAnonymizer):
        model.close()
    (run_dir / "DONE").touch()


if __name__ == "__main__":
    main()
