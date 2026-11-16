"""Blur the detected regions into the copy of the dataset.

    uv run python blur_data/scripts/blur-data-render.py --run anonymize-000

Always reads the ORIGINAL image (``resources/data/v2b``) and overwrites
the same relative path in the copy (``resources/data/v2b-blurred``), so
re-running is idempotent. An image without regions is copied back
byte-identical. JPEGs are written with ``quality="keep"`` (the source's
quantisation tables and subsampling).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from blur_data.anonymize.blur import apply_regions, plate_fill_of, save_jpeg
from blur_data.anonymize.regions import ImageRegions, regions_path
from blur_data.data.v2b import list_camera_images

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "resources" / "data"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="outputs/<run>")
    ap.add_argument("--src", type=Path, default=DATA / "v2b")
    ap.add_argument("--dst", type=Path, default=DATA / "v2b-blurred")
    args = ap.parse_args()

    run_dir = REPO / "outputs" / args.run
    cfg = json.loads((run_dir / "config.json").read_text())
    if args.src.resolve() == args.dst.resolve():
        raise SystemExit("refusing to write into the original dataset")
    missing, blurred, edits = 0, 0, 0
    for im in tqdm(list_camera_images(args.src), desc="render"):
        rpath = regions_path(run_dir, im.key)
        if not rpath.exists():
            missing += 1
            continue
        regions = ImageRegions.load(rpath).ordered()
        src, dst = args.src / im.rel_path, args.dst / im.rel_path
        if not regions:
            shutil.copy2(src, dst)
            continue
        with Image.open(src) as image:
            arr = np.asarray(image.convert("RGB"))
            out = apply_regions(
                arr,
                regions,
                scale=cfg["blur_scale"],
                plate_fill=plate_fill_of(cfg),
            )
            save_jpeg(Image.fromarray(out), dst, image, cfg["jpeg_quality"])
        blurred += 1
        edits += len(regions)
    print(f"blurred {blurred} images, {edits} regions; {missing} missing")
    if missing:
        raise SystemExit(f"{missing} images have no region file")


if __name__ == "__main__":
    main()
