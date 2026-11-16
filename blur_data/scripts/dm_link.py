#!/usr/bin/env python3
"""Materialise ``resources/`` and ``outputs/`` for the current machine.

Reads ``config-global.json`` (machine-independent) and creates the
per-machine symlink tree that the rest of the repo assumes:

``resources/data/<name>``
    -> ``<nas_root>/<dataset.nas_path>``, or
    -> ``<local_root>/<dataset.local_path>`` for data that lives on this
    machine's disk rather than on the NAS (absolute paths are kept)
``resources/data/generated/``
    derived data (shards, tokens, extracted frames) -- a real directory
``resources/pretrained-checkpoints/<name>``
    -> ``<nas_root>/<checkpoint.nas_path>``
``outputs/``, ``outputs/logs/``
    real directories

Both trees are gitignored by design: they are machine-dependent, and
this script is how a fresh clone becomes runnable.

    python3 blur_data/scripts/dm_link.py            # build the tree
    python3 blur_data/scripts/dm_link.py --check    # report, change nothing
    python3 blur_data/scripts/dm_link.py --smoke    # check the smoke target
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "config-global.json"


def load_config() -> dict[str, Any]:
    """Read ``config-global.json``, dropping ``_``-prefixed keys."""
    if not CONFIG.exists():
        raise SystemExit(f"missing {CONFIG}")
    raw: dict[str, Any] = json.loads(CONFIG.read_text())
    return raw


def public(mapping: dict[str, Any]) -> dict[str, Any]:
    """Drop documentation keys (those starting with ``_``)."""
    return {k: v for k, v in mapping.items() if not k.startswith("_")}


def nas_root(cfg: dict[str, Any]) -> Path:
    """NAS root for this host, falling back to the default entry."""
    machines = public(cfg.get("machines", {}))
    host = os.environ.get("BLUR_DATA_MACHINE") or socket.gethostname()
    entry = machines.get(host) or machines.get("default") or {}
    return Path(entry.get("nas_root", cfg.get("nas_root", "/mnt/nas")))


def local_root(cfg: dict[str, Any]) -> Path | None:
    """Local (non-NAS) data root for this host, if the machine has one."""
    machines = public(cfg.get("machines", {}))
    host = os.environ.get("BLUR_DATA_MACHINE") or socket.gethostname()
    entry = machines.get(host) or machines.get("default") or {}
    value = entry.get("local_root")
    if not value:
        return None
    root = Path(value).expanduser()
    return root if root.is_absolute() else REPO / root


def source(entry: dict[str, Any], nas: Path, local: Path | None) -> Path:
    """Resolve one registry entry to its path on this host."""
    if "local_path" in entry:
        base = local if local is not None else Path("/nonexistent")
        return base / str(entry["local_path"])
    return nas / str(entry["nas_path"])


def link(src: Path, dst: Path, check: bool) -> str:
    """Create ``dst`` -> ``src``; return a one-word status."""
    if dst.is_symlink():
        if dst.resolve() == src.resolve():
            return "ok" if src.exists() else "dangling"
        if check:
            return "wrong-target"
        if not src.exists():
            # A hand-made link to data this host *can* see (e.g. a
            # local scratch copy) must survive a re-link whose own source
            # is absent here; replacing it with nothing helps nobody.
            return "kept-other-target"
        dst.unlink()
    elif dst.exists():
        return "blocked"

    if check:
        return "missing" if src.exists() else "absent-on-nas"
    if not src.exists():
        return "absent-on-nas"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src)
    return "created"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check", action="store_true", help="report only, change nothing"
    )
    ap.add_argument(
        "--smoke", action="store_true", help="verify the smoke target exists"
    )
    args = ap.parse_args()

    cfg = load_config()
    root = nas_root(cfg)
    local = local_root(cfg)
    host = os.environ.get("BLUR_DATA_MACHINE") or socket.gethostname()
    print(f"host={host}  nas_root={root}  local_root={local}")
    if not root.exists():
        print(f"  WARNING: nas_root does not exist: {root}")

    data_dir = REPO / "resources" / "data"
    ckpt_dir = REPO / "resources" / "pretrained-checkpoints"
    if not args.check:
        for d in (
            data_dir / "generated",
            ckpt_dir,
            REPO / "outputs" / "logs",
        ):
            d.mkdir(parents=True, exist_ok=True)

    problems = 0
    for name, entry in public(cfg.get("datasets", {})).items():
        src = source(entry, root, local)
        status = link(src, data_dir / name, args.check)
        if status not in {"ok", "created"}:
            problems += 1
        print(f"  data/{name:<18} {status}")

    for name, entry in public(cfg.get("checkpoints", {})).items():
        src = source(entry, root, local)
        status = link(src, ckpt_dir / name, args.check)
        if status not in {"ok", "created"}:
            problems += 1
        print(f"  ckpt/{name:<18} {status}")

    if args.smoke:
        smoke = cfg.get("smoke", {})
        base = data_dir / smoke.get("dataset", "")
        print(f"\nsmoke: {base}")
        for rel in smoke.get("expect_files", []):
            found = (base / rel).exists()
            problems += 0 if found else 1
            print(f"  {rel:<30} {'ok' if found else 'MISSING'}")

    print(f"\n{problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
