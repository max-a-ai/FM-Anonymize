# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "flask>=3.1",
# ]
# ///
"""FM-Anonymize start page: what a run needs and how to start one.

    uv run app.py                 -> http://127.0.0.1:5050

Needs no GPU, no checkpoints and no ``uv sync``: it checks this checkout
for each setup step (submodules, Python environment, the gated SAM 3
and SAM 3D Body checkpoints, PP4AV) and shows the command for every
missing one, followed by the commands of a run. It shows no images.
``PORT=...`` overrides the port; ``--no-browser`` does not open a
browser. Listens on 127.0.0.1 only.
"""

from __future__ import annotations

import os
import sys
import threading
import webbrowser
from pathlib import Path

from flask import Flask, render_template
from flask.typing import ResponseReturnValue

REPO = Path(__file__).resolve().parent
TEMPLATES = REPO / "templates"
PORT = 5050


def checks() -> list[dict[str, object]]:
    """What a run of your own needs, each with the command that fixes
    it (shown on the setup page)."""
    ckpt = REPO / "resources" / "pretrained-checkpoints"
    data = REPO / "resources" / "data"

    def row(name: str, ok: bool, fix: str) -> dict[str, object]:
        return {"name": name, "ok": ok, "fix": fix}

    return [
        row(
            "Submodules SAM 3 and SAM 3D Body",
            (REPO / "third_party" / "sam3" / "pyproject.toml").exists()
            and any((REPO / "third_party" / "sam-3d-body").iterdir()),
            "git submodule update --init",
        ),
        row(
            "Python environment (torch, CUDA 12.6)",
            (REPO / ".venv").exists(),
            "uv sync",
        ),
        row(
            "SAM 3 checkpoint",
            (ckpt / "sam3" / "sam3.pt").exists(),
            "uv run hf download facebook/sam3 --local-dir checkpoints/sam3"
            "\npython3 blur_data/scripts/dm_link.py",
        ),
        row(
            "SAM 3D Body checkpoint (with the MHR model)",
            (ckpt / "sam-3d-body" / "model.ckpt").exists()
            and (ckpt / "sam-3d-body" / "assets" / "mhr_model.pt").exists(),
            "uv run hf download facebook/sam-3d-body-dinov3"
            " --local-dir checkpoints/sam-3d-body"
            "\npython3 blur_data/scripts/dm_link.py",
        ),
        row(
            "PP4AV images and labels",
            (data / "pp4av").exists(),
            "uv run hf download khaclinh/pp4av --repo-type dataset"
            ' --include "data/*.zip" --local-dir datasets/pp4av-zip'
            "\nunzip -q datasets/pp4av-zip/data/images.zip"
            " -d datasets/pp4av/data"
            "\nunzip -q datasets/pp4av-zip/data/annotations.zip"
            " -d datasets/pp4av/data"
            "\npython3 blur_data/scripts/dm_link.py",
        ),
    ]


def create_app() -> Flask:
    """The start page."""
    app = Flask(__name__, template_folder=str(TEMPLATES))

    @app.get("/")
    def setup() -> ResponseReturnValue:
        return render_template("setup.html", checks=checks())

    return app


def main() -> None:
    port = int(os.environ.get("PORT", PORT))
    url = f"http://127.0.0.1:{port}"
    print(f"FM-Anonymize on {url}  (Ctrl+C stops it)")
    if "--no-browser" not in sys.argv:
        threading.Timer(1.0, webbrowser.open, (url,)).start()
    create_app().run(host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
