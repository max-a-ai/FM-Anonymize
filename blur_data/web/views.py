"""What the web app shows of a run: images, overlays, crops, statistics.

A run is any folder with ``regions/`` (one written by the app, or an
existing ``outputs/<run>``), read together with the source folder its
``rel_path`` entries are relative to. The blurred image is read from
``anonymized/`` when the app rendered it, otherwise blurred on the fly.
"""

from __future__ import annotations

import io
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image

from blur_data.anonymize.blur import apply_regions
from blur_data.anonymize.regions import KINDS, ImageRegions, regions_path
from blur_data.utils.viz import PALETTE, crop_box
from blur_data.web.crawl import folder_of
from blur_data.web.job import ANON_DIR

U8 = npt.NDArray[np.uint8]
#: persons the pipeline saw but did not blur, drawn dashed
UNBLURRED = ("away", "too_small", "head_off_image")


@dataclass
class Entry:
    """One image of the run index (no masks: those stay on disk)."""

    key: str
    rel_path: str
    kinds: dict[str, int]
    regions: int
    seconds: float | None
    decisions: dict[str, int]
    boxes: list[tuple[str, float, float, float]]  # kind, score, w, h


class RunIndex:
    """The region files of a run, re-read only when they change."""

    def __init__(self) -> None:
        self._cache: dict[Path, tuple[float, Entry]] = {}

    def entries(self, run: Path) -> list[Entry]:
        """Every image with a region file, in path order."""
        out = []
        for path in sorted((run / "regions").rglob("*.json")):
            mtime = path.stat().st_mtime
            hit = self._cache.get(path)
            if hit is None or hit[0] != mtime:
                hit = (mtime, _entry(path))
                self._cache[path] = hit
            out.append(hit[1])
        return out


def _entry(path: Path) -> Entry:
    raw = json.loads(path.read_text())
    regs = raw["regions"]
    persons = [d for d in raw.get("detections", []) if d["kind"] == "person"]
    return Entry(
        key=raw["key"],
        rel_path=raw["rel_path"],
        kinds=dict(Counter(r["kind"] for r in regs)),
        regions=len(regs),
        seconds=raw.get("stats", {}).get("seconds"),
        decisions=dict(Counter(p.get("decision", "?") for p in persons)),
        boxes=[
            (
                r["kind"],
                float(r["score"]),
                float(r["box"][2] - r["box"][0]),
                float(r["box"][3] - r["box"][1]),
            )
            for r in regs
        ],
    )


def statistics(entries: list[Entry]) -> dict[str, Any]:
    """Totals and histograms over a run for the statistics tab."""
    kinds: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    folders: Counter[str] = Counter()
    sizes: dict[str, list[float]] = {}
    scores: dict[str, list[float]] = {}
    for e in entries:
        kinds.update(e.kinds)
        decisions.update(e.decisions)
        folders[folder_of(e.rel_path)] += e.regions
        for kind, score, w, h in e.boxes:
            sizes.setdefault(kind, []).append(max(w, h))
            scores.setdefault(kind, []).append(score)
    secs = [e.seconds for e in entries if e.seconds is not None]
    size_edges = [0, 10, 20, 40, 80, 160, 320, 10_000]
    score_edges = [round(0.1 * i, 1) for i in range(11)]
    return {
        "images": len(entries),
        "without_regions": sum(e.regions == 0 for e in entries),
        "regions": sum(e.regions for e in entries),
        "kinds": {k: kinds.get(k, 0) for k in KINDS},
        "decisions": dict(decisions),
        "folders": folders.most_common(30),
        "size_edges": size_edges,
        "sizes": {k: _hist(v, size_edges) for k, v in sizes.items()},
        "score_edges": score_edges,
        "scores": {k: _hist(v, score_edges) for k, v in scores.items()},
        "seconds": {
            "mean": round(float(np.mean(secs)), 2) if secs else None,
            "p95": round(float(np.percentile(secs, 95)), 2) if secs else None,
            "total_h": round(sum(secs) / 3600, 2),
        },
    }


def _hist(values: list[float], edges: Sequence[float]) -> list[int]:
    counts, _ = np.histogram(values, bins=edges)
    return [int(c) for c in counts]


def load_regions(run: Path, key: str) -> ImageRegions:
    """The region file of ``key``; refuses keys that leave ``regions/``."""
    path = regions_path(run, key)
    if (run / "regions").resolve() not in path.resolve().parents:
        raise LookupError(f"bad key {key!r}")
    return ImageRegions.load(path)


def load(src: Path, run: Path, key: str) -> tuple[ImageRegions, U8]:
    """The region file of ``key`` and its original image (RGB); refuses
    a ``rel_path`` that points outside ``src``."""
    regs = load_regions(run, key)
    path = src / regs.rel_path
    if src.resolve() not in path.resolve().parents:
        raise LookupError(f"{regs.rel_path} lies outside {src}")
    with Image.open(path) as pil:
        rgb: U8 = np.asarray(pil.convert("RGB"))
    return regs, rgb


def anonymized(
    run: Path, regs: ImageRegions, rgb: U8, kinds: set[str] | None
) -> U8:
    """The blurred image: rendered copy if there, else blurred now."""
    path = run / ANON_DIR / regs.rel_path
    if kinds is None and path.is_file():
        with Image.open(path) as pil:
            out: U8 = np.asarray(pil.convert("RGB"))
        return out
    scale = _blur_scale(run)
    chosen = [r for r in regs.ordered() if kinds is None or r.kind in kinds]
    return apply_regions(rgb, chosen, scale=scale)


def _blur_scale(run: Path) -> float:
    try:
        cfg = json.loads((run / "config.json").read_text())
        return float(cfg.get("blur_scale", 0.3))
    except (OSError, ValueError):
        return 0.3


def overlay(
    base: U8,
    regs: ImageRegions,
    kinds: set[str] | None,
    masks: bool = True,
    boxes: bool = True,
    labels: bool = True,
    detections: bool = False,
) -> U8:
    """Region masks tinted by kind, boxes, labels and (optionally) the
    persons and vehicles that were detected but not blurred."""
    out = base.copy()
    t = max(2, round(out.shape[1] / 900))
    for r in regs.ordered():
        if kinds is not None and r.kind not in kinds:
            continue
        color = PALETTE[r.kind]
        if masks:
            m = r.mask()
            out[m] = (0.55 * out[m] + 0.45 * np.asarray(color)).astype(
                np.uint8
            )
            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(out, contours, -1, color, t)
        if boxes:
            x0, y0, x1, y1 = r.box
            cv2.rectangle(out, (x0, y0), (x1, y1), color, t)
        if labels:
            angle = r.meta.get("angle_deg")
            text = f"{r.kind} {r.score:.2f}" + (
                f" {angle:.0f}deg" if angle is not None else ""
            )
            _label(out, text, r.box[0], r.box[1], color, t)
    if detections:
        for d in regs.detections:
            x0, y0, x1, y1 = (int(v) for v in d["box"])
            if d["kind"] == "vehicle":
                cv2.rectangle(out, (x0, y0), (x1, y1), PALETTE["vehicle"], 1)
            elif d.get("decision") in UNBLURRED:
                _dashed(out, (x0, y0, x1, y1), PALETTE["away"], t)
                angle = d.get("angle_deg")
                text = d["decision"] + (
                    f" {angle:.0f}deg" if angle is not None else ""
                )
                _label(out, text, x0, y1 + 18 * t, PALETTE["away"], t)
    return out


def _label(
    img: U8, text: str, x: int, y: int, color: tuple[int, ...], t: int
) -> None:
    scale = 0.45 * t
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    y = max(h + 4, y - 4)
    cv2.rectangle(img, (x, y - h - 4), (x + w + 4, y + 2), color, -1)
    cv2.putText(
        img,
        text,
        (x + 2, y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        max(1, t // 2),
        cv2.LINE_AA,
    )


def _dashed(
    img: U8, box: tuple[int, int, int, int], color: tuple[int, ...], t: int
) -> None:
    x0, y0, x1, y1 = box
    dash = 8 * t
    for a, b, fixed, horizontal in (
        (x0, x1, y0, True),
        (x0, x1, y1, True),
        (y0, y1, x0, False),
        (y0, y1, x1, False),
    ):
        for s in range(a, b, 2 * dash):
            e = min(s + dash, b)
            p, q = (
                ((s, fixed), (e, fixed))
                if horizontal
                else ((fixed, s), (fixed, e))
            )
            cv2.line(img, p, q, color, t)


def mask_view(rgb: U8, regs: ImageRegions, kinds: set[str] | None) -> U8:
    """Everything outside the blur masks darkened: what gets touched."""
    keep = np.zeros(rgb.shape[:2], dtype=bool)
    for r in regs.regions:
        if kinds is None or r.kind in kinds:
            keep |= r.mask()
    out = (rgb * 0.25).astype(np.uint8)
    out[keep] = rgb[keep]
    return out


def region_crop(img: U8, box: tuple[int, ...], size: int = 220) -> U8:
    """A square crop around ``box`` (2.5 x its side), scaled to ``size``."""
    x0, y0, x1, y1 = box[:4]
    pad = 0.75
    a0, b0, a1, b1 = crop_box(img.shape, (x0, y0, x1, y1), pad, True)
    crop = img[b0:b1, a0:a1]
    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    # nearest: keeps small plates honest, no smoothing
    return np.asarray(
        cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST),
        dtype=np.uint8,
    )


def encode(img: U8, width: int | None = None, quality: int = 88) -> bytes:
    """JPEG bytes, optionally scaled down to ``width``."""
    if width and img.shape[1] > width:
        h = round(img.shape[0] * width / img.shape[1])
        img = np.asarray(
            cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA),
            dtype=np.uint8,
        )
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


#: methods and their colours in the compare view (RGB)
METHODS = {
    "FM-Blur": (0x00, 0xE8, 0x9D),
    "EgoBlur": (0x2A, 0x78, 0xD6),
    "deface": (0xEB, 0x68, 0x34),
    "UrbanAnonymizer": (0xE8, 0x7B, 0xA4),
    "Autoware": (0xED, 0xA1, 0x00),
    "other": (0x88, 0x92, 0xA8),
}
_BASELINES = {
    "egoblur": "EgoBlur",
    "egoblur2": "EgoBlur",
    "deface": "deface",
    "urban": "UrbanAnonymizer",
    "autoware": "Autoware",
}


def method_of(run: Path) -> str:
    """The method behind a run: ``method`` of a baseline's config.json,
    FM-Blur for a run made from a pipeline config, else by name."""
    try:
        cfg = json.loads((run / "config.json").read_text())
    except (OSError, ValueError):
        cfg = {}
    if cfg.get("method") in _BASELINES:
        return _BASELINES[cfg["method"]]
    if "config" in cfg or "vehicle_prompts" in cfg:
        return "FM-Blur"
    name = run.name.lower()
    for prefix, method in (
        ("egoblur", "EgoBlur"),
        ("deface", "deface"),
        ("urban", "UrbanAnonymizer"),
        ("autoware", "Autoware"),
        ("anonymize", "FM-Blur"),
        ("abl", "FM-Blur"),
        ("pp4av-", "FM-Blur"),
    ):
        if name.startswith(prefix):
            return method
    return "other"


def compare_overlay(
    rgb: U8,
    runs: Sequence[tuple[ImageRegions, tuple[int, int, int]]],
    masks: bool = False,
) -> U8:
    """The original with every run's blur boxes in that run's colour
    (and, with ``masks``, the outline of each blur mask)."""
    out = rgb.copy()
    t = max(3, round(out.shape[1] / 600))  # still visible scaled down
    for i, (regs, color) in enumerate(runs):
        grow = 2 * t * i  # nested boxes: equal boxes stay visible
        for r in regs.regions:
            if masks:
                contours, _ = cv2.findContours(
                    r.mask().astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(out, contours, -1, color, max(1, t // 2))
            x0, y0, x1, y1 = r.box
            cv2.rectangle(
                out, (x0 - grow, y0 - grow), (x1 + grow, y1 + grow), color, t
            )
    return out
