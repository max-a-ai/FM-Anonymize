"""FM-Blur web app: anonymize a folder and re-check the result.

    uv run --group app python -m blur_data.web   -> http://127.0.0.1:5010

Pick the image folder, read the analysis, pick the output folder and
start; the run shows progress and the latest frame live. The review
tabs (gallery, viewer, statistics) also open any existing run folder
with ``regions/`` (e.g. ``outputs/anonymize-007``) next to its source.
Listens on 127.0.0.1 only: the pages show original, unblurred frames.
PORT=... overrides the port.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, render_template, request
from flask.typing import ResponseReturnValue

from blur_data.anonymize.regions import KINDS
from blur_data.utils.viz import PALETTE
from blur_data.web import views
from blur_data.web.crawl import (
    EXTENSIONS,
    analyse,
    check_paths,
    list_source,
    subdirs,
)
from blur_data.web.job import Options, Runner

REPO = Path(__file__).resolve().parents[2]
CHECKPOINTS = REPO / "resources" / "pretrained-checkpoints"
PORT = 5010
#: output folders made with the app (they may lie outside outputs/)
RECENT = REPO / "outputs" / "web-projects.json"
MAX_COMPARE = 3


class Session:
    """The source folder and run folder the review tabs look at."""

    def __init__(self) -> None:
        self.src: Path | None = None
        self.run: Path | None = None
        self.compare: list[Path] = []  # runs side by side (same source)

    def need(self) -> tuple[Path, Path]:
        """Both folders, or a 400 for the UI."""
        if self.src is None or self.run is None:
            raise LookupError("open a run first (Start tab)")
        return self.src, self.run


def create_app() -> Flask:
    """The app with every route."""
    app = Flask(__name__)
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    runner = Runner(CHECKPOINTS)
    session = Session()
    index = views.RunIndex()

    @app.errorhandler(LookupError)
    def _missing(exc: LookupError) -> ResponseReturnValue:
        return jsonify({"ok": False, "error": str(exc)}), 400

    @app.route("/")
    def page() -> ResponseReturnValue:
        colors = {k: "#{:02x}{:02x}{:02x}".format(*PALETTE[k]) for k in KINDS}
        methods = {m: _hex(c) for m, c in views.METHODS.items()}
        return render_template(
            "index.html", kinds=KINDS, colors=colors, methods=methods
        )

    @app.get("/api/browse")
    def browse() -> ResponseReturnValue:
        path = Path(request.args.get("path") or Path.home()).expanduser()
        if not path.is_dir():
            return jsonify({"ok": False, "error": f"{path} is not a folder"})
        dirs = []
        for d in subdirs(path)[:400]:
            dirs.append({"name": d.name, "run": (d / "regions").is_dir()})
        try:
            here = sum(
                1
                for f in path.iterdir()
                if f.suffix.lower() in EXTENSIONS and f.is_file()
            )
        except OSError:
            here = 0
        return jsonify(
            {
                "ok": True,
                "path": str(path),
                "parent": str(path.parent),
                "dirs": dirs,
                "images_here": here,
                "is_run": (path / "regions").is_dir(),
            }
        )

    @app.get("/api/shortcuts")
    def shortcuts() -> ResponseReturnValue:
        data = REPO / "resources" / "data"
        sources = (
            [str(p.resolve()) for p in subdirs(data)] if data.is_dir() else []
        )
        runs = [
            str(p)
            for p in subdirs(REPO / "outputs")
            if (p / "regions").is_dir()
        ]
        # the main config first: the dropdown must not default to an ablation
        yamls = (REPO / "configs").rglob("*.yaml")
        configs = sorted(
            (str(p.relative_to(REPO)) for p in yamls),
            key=lambda c: (c != "configs/anonymize.yaml", c),
        )
        return jsonify({"sources": sources, "runs": runs, "configs": configs})

    @app.get("/api/projects")
    def projects() -> ResponseReturnValue:
        out = []
        for run_dir in project_dirs():
            files = list((run_dir / "regions").rglob("*.json"))
            src = guess_source(run_dir, files[:1])
            out.append(
                {
                    "name": run_dir.name,
                    "path": str(run_dir),
                    "images": len(files),
                    "src": str(src) if src else None,
                    "done": (run_dir / "DONE").exists(),
                    "method": views.method_of(run_dir),
                    "mtime": max(
                        (f.stat().st_mtime for f in files[-5:]), default=0
                    ),
                }
            )
        out.sort(key=lambda p: -float(p["mtime"]))  # type: ignore[arg-type]
        return jsonify(out)

    @app.post("/api/scan")
    def scan() -> ResponseReturnValue:
        body = request.get_json(force=True)
        src = Path(body["src"]).expanduser().resolve()
        dst = Path(body["dst"]).expanduser() if body.get("dst") else None
        if not src.is_dir():
            return jsonify({"ok": False, "error": f"{src} is not a folder"})
        images = list_source(src)
        if not images:
            return jsonify({"ok": False, "error": "no images found"})
        out: dict[str, Any] = {"ok": True, **analyse(src, images, dst)}
        if dst is not None:
            out["dst_error"] = check_paths(src, dst)
        return jsonify(out)

    @app.post("/api/run")
    def run() -> ResponseReturnValue:
        opts = Options.from_json(request.get_json(force=True))
        opts.config = (REPO / opts.config).resolve()
        if REPO / "configs" not in opts.config.parents:
            return jsonify(
                {"ok": False, "error": "config must be in configs/"}
            )
        err = check_paths(opts.src, opts.dst)
        if err:
            return jsonify({"ok": False, "error": err})
        try:
            runner.start(opts)
        except RuntimeError as exc:
            return jsonify({"ok": False, "error": str(exc)})
        session.src, session.run = opts.src, opts.dst
        remember(opts.dst)
        return jsonify({"ok": True})

    @app.get("/api/status")
    def status() -> ResponseReturnValue:
        job = runner.job
        return jsonify(job.status() if job else {"phase": "idle"})

    @app.post("/api/stop")
    def stop() -> ResponseReturnValue:
        if runner.job:
            runner.job.stop.set()
        return jsonify({"ok": True})

    @app.post("/api/open")
    def open_run() -> ResponseReturnValue:
        body = request.get_json(force=True)
        run_dir = Path(body["run"]).expanduser().resolve()
        if not (run_dir / "regions").is_dir():
            return jsonify({"ok": False, "error": f"no regions/ in {run_dir}"})
        first = next((run_dir / "regions").rglob("*.json"), None)
        src = (
            Path(body["src"]).expanduser().resolve()
            if body.get("src")
            else guess_source(run_dir, [first] if first else [])
        )
        if src is None:
            return jsonify(
                {"ok": False, "error": "source folder not found, give it"}
            )
        session.src, session.run = src, run_dir
        return jsonify({"ok": True, "src": str(session.src)})

    @app.get("/api/session")
    def get_session() -> ResponseReturnValue:
        return jsonify(
            {
                "src": str(session.src) if session.src else None,
                "run": str(session.run) if session.run else None,
            }
        )

    @app.get("/api/images")
    def images() -> ResponseReturnValue:
        _, run_dir = session.need()
        entries = index.entries(run_dir)
        kind = request.args.get("kind") or ""
        decision = request.args.get("decision") or ""
        text = (request.args.get("q") or "").lower()
        if kind == "none":
            entries = [e for e in entries if e.regions == 0]
        elif kind:
            entries = [e for e in entries if e.kinds.get(kind)]
        if decision:
            entries = [e for e in entries if e.decisions.get(decision)]
        if text:
            entries = [e for e in entries if text in e.rel_path.lower()]
        sort = request.args.get("sort") or "path"
        if sort == "regions":
            entries.sort(key=lambda e: -e.regions)
        elif sort == "seconds":
            entries.sort(key=lambda e: -(e.seconds or 0))
        elif sort == "smallest":
            entries.sort(
                key=lambda e: min(
                    (max(w, h) for _, _, w, h in e.boxes), default=1e9
                )
            )
        return jsonify(
            [
                {
                    "key": e.key,
                    "rel_path": e.rel_path,
                    "regions": e.regions,
                    "kinds": e.kinds,
                    "decisions": e.decisions,
                    "seconds": e.seconds,
                }
                for e in entries
            ]
        )

    @app.get("/api/stats")
    def stats() -> ResponseReturnValue:
        _, run_dir = session.need()
        return jsonify(views.statistics(index.entries(run_dir)))

    @app.get("/api/regions")
    def regions() -> ResponseReturnValue:
        _, run_dir = session.need()
        key = request.args["key"]
        regs = views.load_regions(run_dir, key)
        return jsonify(
            {
                "key": key,
                "rel_path": regs.rel_path,
                "stats": regs.stats,
                "regions": [
                    {
                        "i": i,
                        "kind": r.kind,
                        "score": r.score,
                        "box": r.box,
                        "meta": r.meta,
                    }
                    for i, r in enumerate(regs.ordered())
                ],
                "detections": regs.detections,
            }
        )

    @app.get("/img/<view>")
    def image(view: str) -> ResponseReturnValue:
        src, run_dir = session.need()
        key = request.args["key"]
        regs, rgb = views.load(src, run_dir, key)
        kinds = _kinds()

        def flag(name: str, default: str) -> bool:
            return request.args.get(name, default) == "1"

        if view == "original":
            img = rgb
        elif view == "anonymized":
            img = views.anonymized(run_dir, regs, rgb, kinds)
        elif view == "overlay":
            base = (
                views.anonymized(run_dir, regs, rgb, kinds)
                if request.args.get("base") == "anon"
                else rgb
            )
            img = views.overlay(
                base,
                regs,
                kinds,
                masks=flag("masks", "1"),
                boxes=flag("boxes", "1"),
                labels=flag("labels", "1"),
                detections=flag("det", "0"),
            )
        elif view == "mask":
            img = views.mask_view(rgb, regs, kinds)
        elif view == "crop":
            i = int(request.args["i"])
            box = regs.ordered()[i].box
            base = (
                views.anonymized(run_dir, regs, rgb, None)
                if request.args.get("which") == "anon"
                else rgb
            )
            img = views.region_crop(base, box)
        else:
            return jsonify({"ok": False, "error": f"unknown view {view}"}), 404
        width = request.args.get("w", type=int)
        return Response(views.encode(img, width), mimetype="image/jpeg")

    @app.post("/api/compare")
    def compare() -> ResponseReturnValue:
        runs = [
            Path(p).expanduser().resolve()
            for p in request.get_json(force=True)["runs"]
        ]
        if not 1 <= len(runs) <= MAX_COMPARE:
            return jsonify({"ok": False, "error": "pick 1 to 3 runs"})
        sources = set()
        for run_dir in runs:
            first = next((run_dir / "regions").rglob("*.json"), None)
            sources.add(guess_source(run_dir, [first] if first else []))
        if None in sources or len(sources) != 1:
            return jsonify(
                {"ok": False, "error": "these runs are not on the same images"}
            )
        session.src = sources.pop()
        session.compare = runs
        return jsonify({"ok": True, "src": str(session.src), "runs": _runs()})

    def _runs() -> list[dict[str, Any]]:
        return [
            {
                "name": run_dir.name,
                "path": str(run_dir),
                "method": views.method_of(run_dir),
                "color": _hex(_rgb_shaded(i)),
            }
            for i, run_dir in enumerate(session.compare)
        ]

    @app.get("/api/compare/images")
    def compare_images() -> ResponseReturnValue:
        if not session.compare:
            raise LookupError("pick runs to compare first")
        kind = request.args.get("kind") or ""
        per_run = [
            {e.key: e for e in index.entries(r)} for r in session.compare
        ]
        keys = sorted(set.intersection(*(set(d) for d in per_run)))
        rows: list[dict[str, Any]] = []
        for key in keys:
            entries = [d[key] for d in per_run]
            counts = [
                e.kinds.get(kind, 0) if kind else e.regions for e in entries
            ]
            rows.append(
                {"key": key, "rel_path": entries[0].rel_path, "counts": counts}
            )
        sort = request.args.get("sort") or "diff"
        if sort == "diff":
            rows.sort(key=lambda r: -(max(r["counts"]) - min(r["counts"])))
        elif sort == "sum":
            rows.sort(key=lambda r: -sum(r["counts"]))
        if request.args.get("only") == "diff":
            rows = [r for r in rows if len(set(r["counts"])) > 1]
        return jsonify({"runs": _runs(), "images": rows, "common": len(keys)})

    @app.get("/img/compare/<which>")
    def compare_image(which: str) -> ResponseReturnValue:
        if not session.compare or session.src is None:
            raise LookupError("pick runs to compare first")
        key = request.args["key"]
        width = request.args.get("w", type=int)
        if which == "raw":
            on = {
                int(i)
                for i in (request.args.get("on") or "0,1,2").split(",")
                if i
            }
            _, rgb = views.load(session.src, session.compare[0], key)
            loaded = [
                (views.load_regions(run_dir, key), _rgb_shaded(i))
                for i, run_dir in enumerate(session.compare)
                if i in on
            ]
            img = views.compare_overlay(
                rgb, loaded, masks=request.args.get("masks") == "1"
            )
        else:
            i = int(which)
            regs, rgb = views.load(session.src, session.compare[i], key)
            img = views.anonymized(session.compare[i], regs, rgb, None)
        return Response(views.encode(img, width), mimetype="image/jpeg")

    def _rgb_shaded(i: int) -> tuple[int, int, int]:
        """The run's method colour; a second run of the same method is
        lighter, so the runs stay apart."""
        method = views.method_of(session.compare[i])
        n = sum(views.method_of(r) == method for r in session.compare[:i])
        return _shade(views.METHODS[method], n)

    def _kinds() -> set[str] | None:
        text = request.args.get("kinds")
        return set(text.split(",")) if text else None

    return app


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _shade(rgb: tuple[int, int, int], n: int) -> tuple[int, int, int]:
    """``rgb`` mixed 40 % (n = 1) or 80 % (n = 2) towards white."""
    f = min(0.8, 0.4 * n)
    r, g, b = (round(c + (255 - c) * f) for c in rgb)
    return (r, g, b)


def project_dirs() -> list[Path]:
    """Runs under ``outputs/`` plus every output folder the app made."""
    runs = [p for p in subdirs(REPO / "outputs") if (p / "regions").is_dir()]
    made = [Path(p) for p in _recent()]
    seen: set[Path] = set()
    out = []
    for p in made + runs:
        if p not in seen and (p / "regions").is_dir():
            seen.add(p)
            out.append(p)
    return out


def remember(run_dir: Path) -> None:
    """Add an output folder of the app to the project list."""
    paths = [str(run_dir)] + [p for p in _recent() if p != str(run_dir)]
    RECENT.parent.mkdir(parents=True, exist_ok=True)
    RECENT.write_text(json.dumps(paths[:50], indent=1))


def _recent() -> list[str]:
    try:
        return [str(p) for p in json.loads(RECENT.read_text())]
    except (OSError, ValueError):
        return []


def guess_source(run_dir: Path, region_files: list[Path]) -> Path | None:
    """The folder a run's ``rel_path`` entries resolve in: ``src`` of its
    config.json, else the first folder under ``resources/data`` that
    holds the run's first image."""
    candidates = []
    if (text := _config_src(run_dir)) is not None:
        candidates.append(Path(text))
    data = REPO / "resources" / "data"
    candidates += [p.resolve() for p in subdirs(data)] if data.is_dir() else []
    if not region_files:
        return candidates[0] if candidates else None
    try:
        rel = json.loads(region_files[0].read_text())["rel_path"]
    except (OSError, ValueError, KeyError):
        return None
    return next((c for c in candidates if (c / rel).is_file()), None)


def _config_src(run_dir: Path) -> str | None:
    try:
        cfg = json.loads((run_dir / "config.json").read_text())
    except (OSError, ValueError):
        return None
    value = cfg.get("src")
    return str(value) if value else None


def main() -> None:
    """Serve on 127.0.0.1 (threaded: the run thread keeps going)."""
    port = int(os.environ.get("PORT", PORT))
    print(f"FM-Blur app on http://127.0.0.1:{port}", flush=True)
    create_app().run(host="127.0.0.1", port=port, threaded=True)
