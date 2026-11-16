"""Write rendered images like their source, atomically."""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, JpegImagePlugin


def save_like(
    out: Image.Image, dst: Path, source: Image.Image, quality: str = "keep"
) -> None:
    """Save ``out`` in ``dst``'s format, JPEGs like ``source``.

    ``quality="keep"`` reuses the source's quantisation tables and
    subsampling (PIL's own ``"keep"`` needs the source object itself).
    The file is written next to ``dst`` and renamed, so an interrupted
    run never leaves a truncated image that looks finished.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp")
    if dst.suffix.lower() in (".jpg", ".jpeg"):
        if quality == "keep" and source.format == "JPEG":
            out.save(
                tmp,
                format="JPEG",
                qtables=source.quantization,  # type: ignore[attr-defined]
                subsampling=JpegImagePlugin.get_sampling(source),
            )
        else:
            q = 95 if quality == "keep" else int(quality)
            out.save(tmp, format="JPEG", quality=q)
    else:
        fmt = Image.registered_extensions()[dst.suffix.lower()]
        out.save(tmp, format=fmt)
    os.replace(tmp, dst)
