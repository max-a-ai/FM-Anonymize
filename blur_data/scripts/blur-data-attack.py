"""Can a recognizer still read what each method blurred? (review r31b)

    uv run --group attack python blur_data/scripts/blur-data-attack.py
    uv run --group attack python blur_data/scripts/blur-data-attack.py \
        --rendered blur=blur-gt --rendered gen=gen-gt \
        --parrot --parrot-gallery blur=blur-gt-next \
        --parrot-gallery gen=gen-gt-next --out outputs/eval/gen-attack.json

Faces: ArcFace (InsightFace ``buffalo_l`` ``w600k_r50.onnx`` in
``resources/pretrained-checkpoints/arcface/``) embeds a square crop
(1.3 x the annotated box, resized to 112 px, unaligned for every
version alike) of each annotated face (>= 20 px) of our data: the
original, a lightly blurred control (Gaussian sigma = 4 % of the crop
side) and each method's output. Rank-1 identification: the blurred
crop's most similar original crop is its own. Chance is 1 / N; the
control shows the attack works on these small faces. Also reported:
TAR at 0.1 % FAR (chance 0.1 %) and both per face-size bin
(:mod:`blur_data.metrics.reid`).

Versions are region runs blurred on the fly (``RUNS``) and, with
``--rendered NAME=RUN``, images rendered by ``blur-data-generate.py``
(``outputs/<RUN>/images/<key>.jpg``).

``--parrot``: a second photo of the same person, ``--parrot-delta``
frames later (same camera), found by following the person box that
holds the face through the frames of ``anonymize-007`` (IoU tracker).
Each version's probes are matched against that photo clear (``next``)
and anonymised by the same method (``parrot``: the attacker
anonymises the gallery too; generation draws new seeds there unless
the two photos share a track); ``original`` vs the clear next photo is
the protocol's ceiling. A rendered version's next-frame images come
from ``--parrot-gallery NAME=RUN``; ``--write-next-keys`` writes the
keys those runs need (``blur-data-generate.py --next-keys``).

``--dataset pp4av``: the same face attack on PP4AV's labelled faces
(>= 20 px, boxes as labelled), FM-Anonymize's ``pp4av-ours2`` regions
plus ``--rendered`` PNG runs; no plates, no parrot; ``--keys``
restricts it to a subset (e.g. the stratified 125-image subset).

Plates: EasyOCR reads each annotated plate (crop upscaled to >= 64 px
height) on the original and on each method's output. A plate counts
as readable if the original yields >= 4 alphanumerics, and as leaked
if the blurred reading recovers >= half of them in order (longest
common subsequence).

Writes ``--out`` (default ``outputs/eval/attack.json``).
"""

from __future__ import annotations

import argparse
import functools
import json
import re
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image
from tqdm import tqdm

from blur_data.anonymize.blur import WHITE, apply_regions
from blur_data.anonymize.regions import ImageRegions, regions_path
from blur_data.metrics import reid
from blur_data.metrics.arcface import ArcFace, square
from blur_data.metrics.privacy import GtBox

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs"
DATA = REPO / "resources" / "data" / "v2b"
ARCFACE = REPO / "resources" / "pretrained-checkpoints" / "arcface"
U8 = npt.NDArray[np.uint8]
RUNS = {
    "ours": "anonymize-007",
    "urban": "urban-v2b",
    "egoblur05": "egoblur05s115-v2b",
    "deface": "deface-v2b",
    "autoware": "autoware-gt",
}
#: PP4AV (faces only): FM-Anonymize's regions, blurred on the fly
PP4AV_RUNS = {"ours": "pp4av-ours2"}
FAR = 1e-3
TRACK_IOU = 0.3


def lcs(a: str, b: str) -> int:
    """Length of the longest common subsequence."""
    row = [0] * (len(b) + 1)
    for ca in a:
        prev = 0
        for j, cb in enumerate(b):
            cur = row[j + 1]
            row[j + 1] = prev + 1 if ca == cb else max(row[j + 1], row[j])
            prev = cur
    return row[-1]


def alnum(s: str) -> str:
    return re.sub(r"[^0-9A-Z]", "", s.upper())


def read_plate(reader: Any, img: U8, box: tuple[int, ...]) -> str:
    """OCR of a plate crop, upscaled to >= 64 px height."""
    x0, y0, x1, y1 = box
    f = max(1.0, 64 / max(1, y1 - y0))
    c = cv2.resize(
        img[y0:y1, x0:x1], None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC
    )
    return alnum("".join(reader.readtext(c, detail=0)))


def load_rgb(path: Path) -> U8:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def control(crop: U8) -> U8:
    """The lightly blurred control: sigma = 4 % of the crop side."""
    s = 0.04 * max(crop.shape[:2])
    return np.asarray(cv2.GaussianBlur(crop, (0, 0), s), np.uint8)


def versions(
    rgb: U8,
    key: str,
    runs: dict[str, str],
    rendered: dict[str, str],
    ext: str = ".jpg",
) -> dict[str, U8]:
    """Each method's output of image ``key``; missing region runs left out."""
    out = {}
    for k, r in runs.items():
        path = regions_path(OUT / r, key)
        if path.exists():
            out[k] = apply_regions(
                rgb,
                ImageRegions.load(path).ordered(),
                plate_fill=WHITE if k == "ours" else None,
            )
    for k, r in rendered.items():
        out[k] = load_rgb(OUT / r / "images" / f"{key}{ext}")
    return out


def gt_boxes(v: dict[str, Any]) -> list[GtBox]:
    boxes = [
        GtBox(b["cls"], b["x0"], b["y0"], b["x1"], b["y1"]) for b in v["boxes"]
    ]
    return [g for g in boxes if g.cls == "plate" or g.size >= 20]


#: one image to attack: key, original image, its GT boxes
Item = tuple[str, Path, list[GtBox]]


def v2b_items(raw: dict[str, Any]) -> list[Item]:
    """Our GT frames with a face (>= 20 px) or plate."""
    out = []
    for key, v in raw.items():
        boxes = gt_boxes(v)
        if boxes:
            path = regions_path(OUT / RUNS["ours"], key)
            out.append((key, DATA / ImageRegions.load(path).rel_path, boxes))
    return out


def pp4av_items(keys: set[str] | None = None) -> list[Item]:
    """PP4AV images (pinhole) with a labelled face >= 20 px; no plates.

    ``keys``: only these images (a subset).
    """
    from blur_data.data import pp4av

    root = REPO / "resources" / "data" / "pp4av"
    out = []
    for im in tqdm(pp4av.list_images(root), desc="pp4av"):
        if keys is not None and im.key not in keys:
            continue
        boxes = [
            g
            for g in pp4av.read_gt(root, im)
            if g.cls == "face" and g.size >= 20
        ]
        if boxes:
            out.append((im.key, root / im.rel_path, boxes))
    return out


def collect(
    items: list[Item],
    runs: dict[str, str],
    rendered: dict[str, str],
    reader: Any,
    ext: str = ".jpg",
) -> tuple[dict[str, list[U8]], list[float], list[dict[str, str]]]:
    """Face crops per version, face sizes, plate readings per version."""
    names = ["original", "control", *runs, *rendered]
    faces: dict[str, list[U8]] = {k: [] for k in names}
    sizes: list[float] = []
    plates: list[dict[str, str]] = []
    for key, path, boxes in tqdm(items, desc="crops"):
        rgb = load_rgb(path)
        out = versions(rgb, key, runs, rendered, ext)
        for g in boxes:
            if g.cls == "face":
                x0, y0, x1, y1 = square(g, (rgb.shape[0], rgb.shape[1]), 1.3)
                crop = rgb[y0:y1, x0:x1]
                faces["original"].append(crop)
                faces["control"].append(control(crop))
                for k, img in out.items():
                    faces[k].append(np.asarray(img[y0:y1, x0:x1], np.uint8))
                sizes.append(g.size)
            elif reader is not None:
                box = (int(g.x0), int(g.y0), int(g.x1) + 1, int(g.y1) + 1)
                plates.append(
                    {"original": read_plate(reader, rgb, box)}
                    | {
                        k: read_plate(reader, img, box)
                        for k, img in out.items()
                    }
                )
    return faces, sizes, plates


def face_results(
    arc: ArcFace, faces: dict[str, list[U8]], sizes: list[float]
) -> dict[str, Any]:
    """Rank-1 / TAR of every version against the original crops."""
    gallery = arc.embed(faces["original"])
    n = len(gallery)
    res: dict[str, Any] = {"n": n, "chance": 1 / max(1, n), "far": FAR}
    for k, crops in faces.items():
        if k == "original" or len(crops) != n:
            continue
        res[k] = reid.scores(arc.embed(crops) @ gallery.T, sizes, FAR)
    return res


def plate_results(plates: list[dict[str, str]]) -> dict[str, Any]:
    readable = [p for p in plates if len(p["original"]) >= 4]
    res: dict[str, Any] = {"n": len(plates), "readable": len(readable)}
    names = [k for k in plates[0] if k != "original"] if plates else []
    for k in names:
        res[k] = sum(
            lcs(p["original"], p[k]) >= len(p["original"]) / 2
            for p in readable
        )
    return res


class Pair(NamedTuple):
    """A GT face and the same person ``delta`` frames later."""

    key: str
    face: GtBox
    next_key: str
    next_face: GtBox
    person: reid.Box  # the person box holding the face, frame t
    next_person: reid.Box  # the tracked person box, frame t + delta


def neighbours(delta: int) -> dict[str, list[str]]:
    """Key -> the keys of the next ``delta`` frames of its camera."""
    from blur_data.data.v2b import list_camera_images

    ims = list_camera_images(DATA)
    out: dict[str, list[str]] = {}
    for i, im in enumerate(ims):
        nxt = ims[i + 1 : i + 1 + delta]
        same = [
            n.key
            for n in nxt
            if (n.scene, n.channel) == (im.scene, im.channel)
        ]
        if len(same) == delta:
            out[im.key] = same
    return out


@functools.cache
def persons(key: str) -> list[reid.Box]:
    """Person boxes of image ``key`` (FM-Anonymize's detections)."""
    path = regions_path(OUT / RUNS["ours"], key)
    dets = ImageRegions.load(path).detections if path.exists() else []
    return [
        (
            float(d["box"][0]),
            float(d["box"][1]),
            float(d["box"][2]),
            float(d["box"][3]),
        )
        for d in dets
        if d.get("kind") == "person"
    ]


def holder(g: GtBox, boxes: list[reid.Box]) -> reid.Box | None:
    """Smallest person box holding >= 80 % of the face box."""
    area = max(1e-6, (g.x1 - g.x0) * (g.y1 - g.y0))
    fits = []
    for b in boxes:
        iw = max(0.0, min(g.x1, b[2]) - max(g.x0, b[0]))
        ih = max(0.0, min(g.y1, b[3]) - max(g.y0, b[1]))
        if iw * ih / area >= 0.8:
            fits.append(((b[2] - b[0]) * (b[3] - b[1]), b))
    return min(fits)[1] if fits else None


def pairs(raw: dict[str, Any], delta: int) -> list[Pair]:
    """Every GT face (>= 20 px) whose person can be followed ``delta``."""
    nbr = neighbours(delta)
    out = []
    for key, v in tqdm(raw.items(), desc="track"):
        if key not in nbr:
            continue
        for g in gt_boxes(v):
            start = holder(g, persons(key)) if g.cls == "face" else None
            if start is None:
                continue
            frames = [persons(k) for k in nbr[key]]
            end = reid.track(start, frames, TRACK_IOU)
            if end is None:
                continue
            x0, y0, x1, y1 = reid.moved((g.x0, g.y0, g.x1, g.y1), start, end)
            nxt = GtBox("face", x0, y0, x1, y1)
            out.append(Pair(key, g, nbr[key][-1], nxt, start, end))
    return out


def parrot_crops(
    ps: list[Pair], rendered: dict[str, str], gallery: dict[str, str]
) -> tuple[dict[str, list[U8]], dict[str, list[U8]]]:
    """Probe crops (frame t) and gallery crops (t + delta) per version."""
    probe: dict[str, list[U8]] = {}
    nxt: dict[str, list[U8]] = {}
    for p in tqdm(ps, desc="parrot crops"):
        for crops, key, g, ren in (
            (probe, p.key, p.face, rendered),
            (nxt, p.next_key, p.next_face, gallery),
        ):
            rel = ImageRegions.load(
                regions_path(OUT / RUNS["ours"], key)
            ).rel_path
            rgb = load_rgb(DATA / rel)
            x0, y0, x1, y1 = square(g, (rgb.shape[0], rgb.shape[1]), 1.3)
            crop = rgb[y0:y1, x0:x1]
            imgs = {"original": rgb} | versions(rgb, key, RUNS, ren)
            for k, img in imgs.items():
                crops.setdefault(k, []).append(img[y0:y1, x0:x1])
            crops.setdefault("control", []).append(control(crop))
    return probe, nxt


def parrot_results(
    arc: ArcFace,
    ps: list[Pair],
    probe: dict[str, list[U8]],
    nxt: dict[str, list[U8]],
) -> dict[str, Any]:
    """Each version vs the clear next photo and vs its own anonymised one."""
    n = len(ps)
    sizes = [p.face.size for p in ps]
    clear = arc.embed(nxt["original"])
    res: dict[str, Any] = {"n": n, "chance": 1 / max(1, n), "far": FAR}
    res["original"] = reid.scores(
        arc.embed(probe["original"]) @ clear.T, sizes, FAR
    )
    for k, crops in probe.items():
        if k == "original" or len(crops) != n:
            continue
        e = arc.embed(crops)
        res[k] = {"next": reid.scores(e @ clear.T, sizes, FAR)}
        if len(nxt.get(k, [])) == n:
            res[k]["parrot"] = reid.scores(e @ arc.embed(nxt[k]).T, sizes, FAR)
    return res


def write_next_keys(args: argparse.Namespace, raw: dict[str, Any]) -> None:
    """``--write-next-keys``: the frames of the second photos."""
    ps = pairs(raw, args.parrot_delta)
    keys = sorted({p.next_key for p in ps})
    args.write_next_keys.parent.mkdir(parents=True, exist_ok=True)
    args.write_next_keys.write_text(json.dumps(keys, indent=1))
    print(f"{len(ps)} faces, {len(keys)} frames -> {args.write_next_keys}")


def named(specs: list[str]) -> dict[str, str]:
    """``["a=run-a", ...]`` -> ``{"a": "run-a"}``."""
    return dict(s.split("=", 1) for s in specs)


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["v2b", "pp4av"], default="v2b")
    ap.add_argument("--rendered", action="append", default=[])
    ap.add_argument("--parrot", action="store_true")
    ap.add_argument("--parrot-delta", type=int, default=5)
    ap.add_argument("--parrot-gallery", action="append", default=[])
    ap.add_argument("--write-next-keys", type=Path, default=None)
    ap.add_argument("--no-plates", action="store_true")
    ap.add_argument("--keys", type=Path, default=None, help="PP4AV subset")
    ap.add_argument("--out", type=Path, default=OUT / "eval" / "attack.json")
    return ap.parse_args()


def main() -> None:
    args = parse()
    raw = json.loads((OUT / "gt" / "annotations.reviewed.json").read_text())
    rendered = named(args.rendered)
    if args.write_next_keys:
        write_next_keys(args, raw)
        return
    pp = args.dataset == "pp4av"
    if pp and args.parrot:
        raise SystemExit("--parrot needs our data (person tracks)")
    reader = None
    if not args.no_plates and not pp:
        import easyocr

        reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    subset = set(json.loads(args.keys.read_text())) if args.keys else None
    items = pp4av_items(subset) if pp else v2b_items(raw)
    runs = PP4AV_RUNS if pp else RUNS
    ext = ".png" if pp else ".jpg"
    faces, sizes, plates = collect(items, runs, rendered, reader, ext)
    arc = ArcFace(ARCFACE / "w600k_r50.onnx")
    res: dict[str, Any] = {"faces": face_results(arc, faces, sizes)}
    if reader is not None:
        res["plates"] = plate_results(plates)
    if args.parrot:
        ps = pairs(raw, args.parrot_delta)
        probe, nxt = parrot_crops(ps, rendered, named(args.parrot_gallery))
        res["parrot"] = {
            "delta": args.parrot_delta,
            **parrot_results(arc, ps, probe, nxt),
        }
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
