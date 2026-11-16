"""One anonymization run in a background thread, for the web app.

Writes the same run layout as ``blur-data-anonymize.py`` into the output
folder (``config.json``, ``regions/<key>.json``, ``summary.jsonl``,
``DONE``) plus the blurred copy under ``anonymized/`` (same relative
paths as the source; with ``mirror`` every other file of the source is
copied too, so ``anonymized/`` is a complete dataset). Images that
already have a region file are skipped, so a stopped run resumes.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import traceback
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from blur_data.anonymize.blur import apply_regions, save_jpeg
from blur_data.anonymize.config import AnonConfig
from blur_data.anonymize.regions import ImageRegions, Region, regions_path
from blur_data.web.crawl import SourceImage, list_source

if TYPE_CHECKING:
    from blur_data.anonymize.pipeline import Anonymizer

ANON_DIR = "anonymized"


@dataclass
class Options:
    """What the start form sends."""

    src: Path
    dst: Path
    config: Path
    limit: int | None = None  # first N images only
    every: int = 1  # every n-th image only
    render: bool = True  # write the blurred copy, not only regions
    mirror: bool = False  # copy the non-image files of the source too
    kinds: tuple[str, ...] | None = None  # region kinds to blur (all)

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Options:
        """Parse the form; empty fields keep their default."""
        kinds = raw.get("kinds")
        return cls(
            src=Path(raw["src"]).expanduser().resolve(),
            dst=Path(raw["dst"]).expanduser().resolve(),
            config=Path(raw["config"]),
            limit=int(raw["limit"]) if raw.get("limit") else None,
            every=max(1, int(raw.get("every") or 1)),
            render=bool(raw.get("render", True)),
            mirror=bool(raw.get("mirror", False)),
            kinds=tuple(kinds) if kinds else None,
        )


@dataclass
class Job:
    """State of the current run; :meth:`status` is what the UI polls."""

    opts: Options
    phase: str = "starting"  # loading | running | stopped | done | error
    total: int = 0
    done: int = 0
    skipped: int = 0
    started: float = field(default_factory=time.time)
    seconds: deque[float] = field(default_factory=lambda: deque(maxlen=20))
    kinds: Counter[str] = field(default_factory=Counter)
    last: str | None = None
    errors: list[str] = field(default_factory=list)
    log: deque[str] = field(default_factory=lambda: deque(maxlen=200))
    stop: threading.Event = field(default_factory=threading.Event)

    def status(self) -> dict[str, Any]:
        """Progress, speed, ETA and the region counts so far."""
        per = sum(self.seconds) / len(self.seconds) if self.seconds else None
        left = self.total - self.done - self.skipped
        return {
            "phase": self.phase,
            "src": str(self.opts.src),
            "dst": str(self.opts.dst),
            "total": self.total,
            "done": self.done,
            "skipped": self.skipped,
            "elapsed": round(time.time() - self.started),
            "per_image": round(per, 2) if per else None,
            "eta": round(per * left) if per else None,
            "kinds": dict(self.kinds),
            "last": self.last,
            "errors": self.errors[-20:],
            "log": list(self.log)[-30:],
        }

    def say(self, line: str) -> None:
        """Append a time-stamped line to the run log."""
        self.log.append(f"{time.strftime('%H:%M:%S')} {line}")


class Runner:
    """Holds the models across runs; starts at most one job at a time."""

    def __init__(self, checkpoints: Path) -> None:
        self.checkpoints = checkpoints
        self.job: Job | None = None
        self._model: tuple[Path, Anonymizer] | None = None
        self._lock = threading.Lock()

    def start(self, opts: Options) -> Job:
        """Start a run, unless one is already going."""
        with self._lock:
            if self.job and self.job.phase in (
                "starting",
                "loading",
                "running",
            ):
                raise RuntimeError("a run is already going")
            self.job = Job(opts)
            threading.Thread(
                target=self._run, args=(self.job,), daemon=True
            ).start()
            return self.job

    def _anonymizer(self, job: Job, cfg: AnonConfig) -> Anonymizer:
        """The models for ``cfg``, loaded once per config file."""
        if self._model and self._model[0] == job.opts.config:
            self._model[1].cfg = cfg
            return self._model[1]
        from blur_data.anonymize.pipeline import Anonymizer  # torch: lazy

        job.phase = "loading"
        job.say("loading SAM 3 and SAM 3D Body")
        self._model = None  # free the previous models first
        anon = Anonymizer(
            cfg,
            self.checkpoints / "sam3" / "sam3.pt",
            self.checkpoints / "sam-3d-body",
        )
        self._model = (job.opts.config, anon)
        return anon

    def _run(self, job: Job) -> None:
        try:
            self._loop(job)
        except Exception as exc:  # shown in the UI, the server keeps going
            job.phase = "error"
            job.errors.append(f"{type(exc).__name__}: {exc}")
            job.say(traceback.format_exc(limit=3))

    def _loop(self, job: Job) -> None:
        o = job.opts
        cfg = AnonConfig.from_yaml(o.config)
        every = list_source(o.src)
        images = every[: o.limit][:: o.every]
        job.total = len(images)
        o.dst.mkdir(parents=True, exist_ok=True)
        (o.dst / "config.json").write_text(
            json.dumps(
                {"config": str(o.config), "src": str(o.src), **cfg.to_dict()},
                indent=2,
            )
        )
        if o.mirror:
            job.say("copying the non-image files")
            mirror_other(
                o.src, o.dst / ANON_DIR, {im.rel_path for im in every}
            )
        todo = []
        for im in images:
            if regions_path(o.dst, im.key).exists():
                job.skipped += 1
            else:
                todo.append(im)
        job.say(f"{job.total} images, {len(todo)} to do")
        anon = self._anonymizer(job, cfg) if todo else None
        job.phase = "running"
        with (o.dst / "summary.jsonl").open("a") as log:
            for im in todo:
                if job.stop.is_set():
                    job.phase = "stopped"
                    job.say("stopped")
                    return
                assert anon is not None
                try:
                    stats = process_one(anon, cfg, im, o)
                except Exception as exc:  # one bad file does not end the run
                    job.errors.append(f"{im.rel_path}: {exc}")
                    continue
                job.seconds.append(stats["seconds"])
                job.kinds.update(stats.get("kinds", {}))
                job.done += 1
                job.last = im.key
                log.write(json.dumps({"key": im.key, **stats}) + "\n")
                log.flush()
        if len(images) == len(every):
            (o.dst / "DONE").touch()
        job.phase = "done"
        job.say(f"done: {job.done} new, {job.skipped} skipped")


def process_one(
    anon: Anonymizer, cfg: AnonConfig, im: SourceImage, o: Options
) -> dict[str, Any]:
    """Detect, save the region file and (if asked) the blurred image."""
    t0 = time.time()
    src = o.src / im.rel_path
    with Image.open(src) as pil:
        rgb = pil.convert("RGB")
        res = anon.process(rgb, im.k_matrix(pil.width, pil.height), im.channel)
        stats: dict[str, Any] = {
            **res.stats(),
            "seconds": round(time.time() - t0, 2),
            "kinds": dict(Counter(r.kind for r in res.regions)),
        }
        ImageRegions(
            im.key, im.rel_path, res.regions, stats, res.detections()
        ).save(regions_path(o.dst, im.key))
        if o.render:
            render_one(
                pil,
                src,
                res.regions,
                o.dst / ANON_DIR / im.rel_path,
                cfg,
                o.kinds,
            )
    return stats


def render_one(
    pil: Image.Image,
    src: Path,
    regions: list[Region],
    dst: Path,
    cfg: AnonConfig,
    kinds: tuple[str, ...] | None,
) -> None:
    """Blur ``regions`` (of ``kinds``) into ``pil`` (read from ``src``)
    and save it at ``dst``; without regions ``src`` is copied as is."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    chosen = [r for r in regions if kinds is None or r.kind in kinds]
    chosen.sort(key=lambda r: -r.score)
    if not chosen:
        shutil.copy2(src, dst)  # byte-identical
        return
    out = Image.fromarray(
        apply_regions(
            np.asarray(pil.convert("RGB")), chosen, scale=cfg.blur_scale
        )
    )
    # write next to the target, then rename: the live view never reads
    # a half-written image
    tmp = dst.with_name(f".{dst.stem}.part{dst.suffix}")
    if dst.suffix.lower() in (".jpg", ".jpeg"):
        save_jpeg(out, tmp, pil, cfg.jpeg_quality)
    else:
        out.save(tmp)
    os.replace(tmp, dst)


def mirror_other(src: Path, dst: Path, images: set[str]) -> None:
    """Copy every file of ``src`` that is not one of ``images``."""
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src).as_posix()
        if path.is_file() and rel not in images:
            target = dst / rel
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
