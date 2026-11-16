"""Is the synthetic identity stable over a person track?

    uv run --group attack python blur_data/scripts/blur-data-temporal.py \
        --dataset v2b --tracks outputs/gen-keys/tracks-v2b.json \
        --version gen=gen-seq --out outputs/eval/gen-temporal.json

For every track of ``--tracks`` (``{"tracks": [[key, ...]], "boxes":
[[box, ...]]}``, consecutive 10 Hz frames of one camera, the head box
per frame) ArcFace embeds a square crop (1.3 x the head box) of the
original and of each version, and the cosine between consecutive
frames is reported (median and mean per track and over all pairs).
Higher = a more stable identity; the original is the reference (the
same real person). ``flicker``: the mean absolute change (0-255) of the
head crop, resized to 64 px, between consecutive frames; the
original's value is the motion floor. ``--dataset pp4av`` reads PNGs
of PP4AV tracks. The versions are ``blur-data-generate.py --seq`` runs
of the same tracks. Numbers only; no image is shown or written.
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from blur_data.metrics.arcface import U8, ArcFace, square
from blur_data.metrics.privacy import GtBox

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs"
DATA = REPO / "resources" / "data" / "v2b"
ARCFACE = REPO / "resources" / "pretrained-checkpoints" / "arcface"


@functools.cache
def v2b_paths() -> dict[str, str]:
    """Key -> image path (relative to the data root) of our recordings."""
    from blur_data.data.v2b import list_camera_images

    return {im.key: im.rel_path for im in list_camera_images(DATA)}


def original(key: str, dataset: str) -> Path:
    """The original image of ``key``."""
    if dataset == "pp4av":
        return DATA.parent / "pp4av" / f"{key}.png"
    return DATA / v2b_paths()[key]


def crops(
    keys: list[str],
    boxes: list[list[float]],
    runs: dict[str, str],
    dataset: str = "v2b",
) -> dict[str, list[U8]]:
    """Version -> head crops of one track, frame order."""
    out: dict[str, list[U8]] = {"original": [], **{k: [] for k in runs}}
    ext = ".png" if dataset == "pp4av" else ".jpg"
    for key, b in zip(keys, boxes, strict=True):
        paths = {"original": original(key, dataset)} | {
            k: OUT / r / "images" / f"{key}{ext}" for k, r in runs.items()
        }
        for name, path in paths.items():
            with Image.open(path) as im:
                rgb = np.asarray(im.convert("RGB"))
            g = GtBox("face", *b)
            x0, y0, x1, y1 = square(g, (rgb.shape[0], rgb.shape[1]), 1.3)
            out[name].append(np.asarray(rgb[y0:y1, x0:x1], np.uint8))
    return out


def consecutive(arc: ArcFace, frames: list[U8]) -> list[float]:
    e = arc.embed(frames)
    return [float(v) for v in (e[:-1] * e[1:]).sum(axis=1)]


def flicker(frames: list[U8], size: int = 64) -> list[float]:
    """Mean absolute change (0-255) of the head crop, resized to
    ``size``, between consecutive frames (motion is in the original's
    value too)."""
    small = [
        cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA).astype(
            np.float32
        )
        for f in frames
    ]
    return [
        float(np.abs(a - b).mean())
        for a, b in zip(small[:-1], small[1:], strict=True)
    ]


def stats(v: list[float]) -> dict[str, float]:
    a = np.asarray(v)
    return {
        "n": len(a),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracks", type=Path, required=True)
    ap.add_argument("--version", action="append", required=True)
    ap.add_argument("--dataset", choices=["v2b", "pp4av"], default="v2b")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    runs = dict(s.split("=", 1) for s in args.version)
    raw = json.loads(args.tracks.read_text())
    arc = ArcFace(ARCFACE / "w600k_r50.onnx")
    res: dict[str, Any] = {"versions": runs, "tracks": []}
    pooled: dict[str, list[float]] = {}
    flick: dict[str, list[float]] = {}
    for keys, boxes in zip(raw["tracks"], raw["boxes"], strict=True):
        per = {}
        for name, frames in crops(keys, boxes, runs, args.dataset).items():
            cos = consecutive(arc, frames)
            per[name] = stats(cos) | {"flicker": stats(flicker(frames))}
            pooled.setdefault(name, []).extend(cos)
            flick.setdefault(name, []).extend(flicker(frames))
        res["tracks"].append({"first": keys[0], "frames": len(keys)} | per)
    res["all"] = {k: stats(v) for k, v in pooled.items()}
    res["flicker"] = {k: stats(v) for k, v in flick.items()}
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["all"], indent=1))


if __name__ == "__main__":
    main()
