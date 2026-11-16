"""Replace FM-Anonymize's blurred heads by synthetic ones (FM-Anonymize-Gen).

Our recordings (the 300 annotated frames; heads of 4 frames before and
8 after are rendered too, so tracks reach over them; the second photos
of the re-identification test go to a second run):

    uv run --group inpaint python blur_data/scripts/blur-data-generate.py \
        --regions-run anonymize-002 --dataset v2b --run gen-gt \
        --keys outputs/gt/annotations.reviewed.json --window 4 8 \
        --next-keys outputs/gen-keys/next5.json --next-run gen-gt-next

PP4AV (single frames of 5-frame steps: head tracks, boxes grown 2x for
the linking):

    uv run --group inpaint python blur_data/scripts/blur-data-generate.py \
        --regions-run pp4av-ours --dataset pp4av --run gen-pp4av \
        --config configs/generate-pp4av.yaml --track-scale 2.0

Reads the ORIGINAL image of every key (``resources/data/<dataset>``)
and its FM-Anonymize regions (``outputs/<regions-run>``); writes
``outputs/<run>/images/<key>.<ext>`` (the source's format; JPEGs keep
its quantisation tables), ``config.json``, ``summary.jsonl`` (per
output frame: heads generated, heads with a track seed, heads without
a pose guide by reason, regions blurred, seconds) and ``DONE`` once
every output frame is written. Finished segments are skipped, so an
interrupted run resumes; a run is never resumed with another config.

``--mode blur`` writes the same keys with FM-Anonymize's blur only,
``--mode reencode`` writes them untouched (the JPEG re-encoding floor
of a comparison); neither needs a GPU.

``--keys`` / ``--next-keys``: a JSON list, or a JSON object (its keys,
e.g. the GT annotations); ``--seq``: ``{"tracks": [[key, ...], ...]}``
(every key of every track is an output frame). Default: every region
file of the regions run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from blur_data.anonymize import blur
from blur_data.anonymize.regions import ImageRegions, regions_path
from blur_data.generate.config import GenConfig
from blur_data.generate.render import EXT, Renderer, has_generated_head
from blur_data.generate.tracks import segments, sequences
from blur_data.utils.images import save_like

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs"
DATA = REPO / "resources" / "data"
CKPT = REPO / "resources" / "pretrained-checkpoints"


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regions-run", required=True, help="outputs/<run>")
    ap.add_argument("--dataset", choices=sorted(EXT), required=True)
    ap.add_argument("--run", required=True, help="outputs/<new run>")
    ap.add_argument(
        "--config", type=Path, default=REPO / "configs" / "generate.yaml"
    )
    ap.add_argument(
        "--mode", choices=["generate", "blur", "reencode"], default="generate"
    )
    ap.add_argument("--keys", type=Path, default=None, help="output keys")
    ap.add_argument("--seq", type=Path, default=None, help="track keys")
    ap.add_argument(
        "--window",
        type=int,
        nargs=2,
        default=(0, 0),
        metavar=("BEFORE", "AFTER"),
        help="context frames around each output frame (tracks)",
    )
    ap.add_argument("--next-keys", type=Path, default=None)
    ap.add_argument("--next-run", default=None)
    ap.add_argument(
        "--track-scale",
        type=float,
        default=1.0,
        help="link boxes grown by this factor (2 for PP4AV)",
    )
    ap.add_argument("--limit", type=int, default=None, help="first N keys")
    return ap.parse_args()


def read_keys(path: Path) -> list[str]:
    """A JSON list of keys, or the keys of a JSON object."""
    return [str(k) for k in json.loads(path.read_text())]


def output_paths(args: argparse.Namespace) -> dict[str, list[Path]]:
    """Output key -> the image path(s) to write; a key that is also a
    ``--next-keys`` key is rendered once and written to both runs."""
    ext = EXT[args.dataset]
    keys: list[str] = []
    if args.seq:
        tracks = json.loads(args.seq.read_text())["tracks"]
        keys = [str(k) for t in tracks for k in t]
    if args.keys:
        keys += read_keys(args.keys)
    if not args.seq and not args.keys:
        root = OUT / args.regions_run / "regions"
        keys = sorted(
            str(p.relative_to(root).with_suffix(""))
            for p in root.rglob("*.json")
        )
    keys = keys[: args.limit]
    out = {k: [OUT / args.run / "images" / f"{k}{ext}"] for k in keys}
    if args.next_keys and args.next_run:
        for k in read_keys(args.next_keys)[: args.limit]:
            path = OUT / args.next_run / "images" / f"{k}{ext}"
            out.setdefault(k, []).append(path)
    return out


def rel(path: Path) -> str:
    """``path`` relative to the repo when inside it."""
    p = path.resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def write_config(args: argparse.Namespace, cfg: GenConfig) -> None:
    """``config.json`` of the run(s); an existing run must match it."""
    record: dict[str, Any] = {
        "mode": args.mode,
        "regions_run": args.regions_run,
        "dataset": args.dataset,
    }
    if args.mode == "generate":
        record |= {
            "config": rel(args.config),
            "window": list(args.window),
            "track_scale": args.track_scale,
            "generate": cfg.to_dict(),
        }
    else:
        record["blur_scale"] = cfg.blur_scale
    for run in filter(None, (args.run, args.next_run)):
        path = OUT / run / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            old = json.loads(path.read_text())
            if old != record:
                raise SystemExit(f"{path} exists with another config")
            continue
        path.write_text(json.dumps(record, indent=2))


def generate(
    args: argparse.Namespace, cfg: GenConfig, outputs: dict[str, list[Path]]
) -> None:
    """Render the output frames segment by segment."""
    src = OUT / args.regions_run
    primary = [
        k for k, ps in outputs.items() if ps[0].is_relative_to(OUT / args.run)
    ]
    active = {k for k in primary if has_generated_head(src, k, cfg)}
    before, after = args.window
    segs = segments(
        sequences(args.dataset, DATA / args.dataset, src),
        list(outputs),
        before,
        after,
        active,
    )
    todo = [
        (seq, keys)
        for seq, keys in segs
        if any(not p.exists() for k in keys for p in outputs.get(k, []))
    ]
    n = f"{len(outputs)} output frames, {len(segs)} segments"
    print(f"{n}, {len(todo)} to do")
    if not todo:
        return
    r = Renderer(cfg, args.dataset, src, DATA, CKPT, args.track_scale)
    with (OUT / args.run / "summary.jsonl").open("a") as log:
        for seq, keys in tqdm(todo, desc="segments"):
            t0 = time.time()
            mine = {k: outputs[k] for k in keys if k in outputs}
            for row in r.segment(seq, keys, mine):
                row["segment_seconds"] = round(time.time() - t0, 2)
                log.write(json.dumps(row) + "\n")
            log.flush()


def plain(
    args: argparse.Namespace, cfg: GenConfig, outputs: dict[str, list[Path]]
) -> None:
    """``--mode blur`` / ``reencode``: the comparison renders."""
    src = OUT / args.regions_run
    todo = {
        k: ps for k, ps in outputs.items() if not all(p.exists() for p in ps)
    }
    with (OUT / args.run / "summary.jsonl").open("a") as log:
        for key, paths in tqdm(todo.items(), desc=args.mode):
            regs = ImageRegions.load(regions_path(src, key))
            ordered = regs.ordered()
            source = DATA / args.dataset / regs.rel_path
            for dst in paths:
                dst.parent.mkdir(parents=True, exist_ok=True)
            if not ordered:  # nothing to anonymize: the file as it is
                for dst in paths:
                    shutil.copyfile(source, dst)
            else:
                with Image.open(source) as image:
                    arr = np.asarray(image.convert("RGB"))
                    if args.mode == "blur":
                        arr = blur.apply_regions(
                            arr, ordered, scale=cfg.blur_scale
                        )
                    for dst in paths:
                        save_like(Image.fromarray(arr), dst, image, "keep")
            log.write(json.dumps({"key": key, "regions": len(ordered)}) + "\n")


def main() -> None:
    args = parse()
    cfg = GenConfig.from_yaml(args.config)
    write_config(args, cfg)
    outputs = output_paths(args)
    if args.mode == "generate":
        generate(args, cfg, outputs)
    else:
        plain(args, cfg, outputs)
    for run in filter(None, (args.run, args.next_run)):
        paths = [p for ps in outputs.values() for p in ps]
        mine = [p for p in paths if p.is_relative_to(OUT / run)]
        if all(p.exists() for p in mine):
            (OUT / run / "DONE").touch()


if __name__ == "__main__":
    main()
