"""Detect every region to blur and write one JSON per image.

    uv run python blur_data/scripts/blur-data-anonymize.py \
        --config configs/anonymize.yaml --run anonymize-003
    uv run python blur_data/scripts/blur-data-anonymize.py \
        --config configs/anonymize.yaml --run pp4av-ours --dataset pp4av

Writes ``outputs/<run>/config.json``, ``regions/<channel>/<stem>.json``,
``summary.jsonl`` (one stats row per image) and ``DONE``. Images that
already have a region file are skipped, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from blur_data.anonymize.config import AnonConfig
from blur_data.anonymize.pipeline import Anonymizer
from blur_data.anonymize.regions import ImageRegions, regions_path
from blur_data.data import pp4av
from blur_data.data.v2b import CameraImage, list_camera_images

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "resources"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run", required=True, help="outputs/<run>")
    ap.add_argument("--dataset", choices=["v2b", "pp4av"], default="v2b")
    ap.add_argument("--data", type=Path, default=None, help="dataset root")
    ap.add_argument("--limit", type=int, default=None, help="first N only")
    ap.add_argument("--keys", type=Path, default=None, help="json key list")
    args = ap.parse_args()

    cfg = AnonConfig.from_yaml(args.config)
    run_dir = REPO / "outputs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({"config": str(args.config), **cfg.to_dict()}, indent=2)
    )
    data = args.data or RES / "data" / args.dataset
    lister = (
        pp4av.list_images if args.dataset == "pp4av" else list_camera_images
    )
    every: list[pp4av.Pp4avImage | CameraImage] = list(lister(data))
    images = every[: args.limit]
    if args.keys:
        wanted = set(json.loads(args.keys.read_text()))
        images = [im for im in images if im.key in wanted]
    todo = [im for im in images if not regions_path(run_dir, im.key).exists()]
    print(f"{len(images)} images, {len(todo)} to do -> {run_dir}")
    if not todo:
        (run_dir / "DONE").touch()
        return

    ckpt = RES / "pretrained-checkpoints"
    anon = Anonymizer(cfg, ckpt / "sam3" / "sam3.pt", ckpt / "sam-3d-body")
    with (run_dir / "summary.jsonl").open("a") as log:
        for im in tqdm(todo, desc="anonymize"):
            t0 = time.time()
            image = Image.open(data / im.rel_path).convert("RGB")
            res = anon.process(image, im.k_matrix(), im.channel)
            stats = {**res.stats(), "seconds": round(time.time() - t0, 2)}
            ImageRegions(
                im.key, im.rel_path, res.regions, stats, res.detections()
            ).save(regions_path(run_dir, im.key))
            log.write(json.dumps({"key": im.key, **stats}) + "\n")
            log.flush()
    if len(images) == len(every):
        (run_dir / "DONE").touch()


if __name__ == "__main__":
    main()
