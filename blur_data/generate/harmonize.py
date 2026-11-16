"""Blend a generated head into the scene and match its look.

The generated crop is composited with a Laplacian-pyramid blend in a
thin band around the mask, so low frequencies (light, shadow) meet the
scene smoothly while the generated detail stays sharp. The region is
then made as blurry and as noisy as a ring of its surroundings and, for
JPEG sources, compressed once with the source's quantisation tables, so
it does not stand out as sharper or cleaner than the camera image.
"""

from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image

from blur_data.utils.masks import BoolMask, dilate

U8 = npt.NDArray[np.uint8]
F32 = npt.NDArray[np.float32]


def _pyramid(img: F32, levels: int) -> list[F32]:
    out = [img]
    for _ in range(levels):
        out.append(np.asarray(cv2.pyrDown(out[-1]), np.float32))
    return out


def _laplacian(img: F32, levels: int) -> list[F32]:
    g = _pyramid(img, levels)
    out = []
    for i in range(levels):
        up = cv2.pyrUp(g[i + 1], dstsize=(g[i].shape[1], g[i].shape[0]))
        out.append(g[i] - np.asarray(up, np.float32))
    out.append(g[-1])
    return out


def multiband(src: U8, dst: U8, mask: BoolMask, levels: int) -> U8:
    """Laplacian-pyramid blend of ``src`` into ``dst`` where ``mask``,
    in a thin band: deeper inside the mask than half the coarsest
    level's footprint, ``src`` is taken as it is (the coarse levels would
    otherwise mix ``dst``'s low frequencies into a small head)."""
    a, b = src.astype(np.float32), dst.astype(np.float32)
    m = mask.astype(np.float32)[..., None].repeat(3, axis=2)
    la, lb, gm = (
        _laplacian(a, levels),
        _laplacian(b, levels),
        _pyramid(m, levels),
    )
    out = la[-1] * gm[-1] + lb[-1] * (1 - gm[-1])
    for i in range(levels - 1, -1, -1):
        size = (la[i].shape[1], la[i].shape[0])
        out = np.asarray(cv2.pyrUp(out, dstsize=size), np.float32)
        out += la[i] * gm[i] + lb[i] * (1 - gm[i])
    res: U8 = np.clip(np.round(out), 0, 255).astype(np.uint8)
    inside = ~dilate(~mask, 2 ** (levels - 1))  # mask eroded by half a band
    res[inside] = src[inside]
    return res


def sharpness(img: U8, where: BoolMask) -> float:
    """Variance of the Laplacian of the grey image inside ``where``."""
    if not where.any():
        return 0.0
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(cv2.Laplacian(grey, cv2.CV_32F)[where].var())


def noise_level(img: U8, where: BoolMask) -> float:
    """Robust noise std: MAD of the residual to a 3x3 median."""
    if not where.any():
        return 0.0
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    res = grey.astype(np.float32) - cv2.medianBlur(grey, 3).astype(np.float32)
    r = res[where]
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def match_blur(img: U8, region: BoolMask, ring: BoolMask) -> U8:
    """Blur ``region`` (Gaussian, sigma from a short list) until it is
    no sharper than ``ring``."""
    target = sharpness(img, ring)
    best = img
    for sigma in (0.6, 0.9, 1.3, 1.8, 2.5):
        if sharpness(best, region) <= 1.2 * target:
            break
        blurred = np.asarray(cv2.GaussianBlur(img, (0, 0), sigma), np.uint8)
        best = img.copy()
        best[region] = blurred[region]
    return best


def match_noise(img: U8, region: BoolMask, ring: BoolMask, seed: int) -> U8:
    """Add Gaussian noise to ``region`` up to the noise level of ``ring``."""
    have, want = noise_level(img, region), noise_level(img, ring)
    if want <= have:
        return img
    std = float(np.sqrt(want**2 - have**2))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, std, img.shape).astype(np.float32)
    out = img.astype(np.float32)
    out[region] += noise[region]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def match_jpeg(img: U8, region: BoolMask, qtables: Any | None) -> U8:
    """Compress ``region`` once with the source's tables (JPEG sources)."""
    if qtables is None:
        return img
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", qtables=qtables)
    dec = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    out = img.copy()
    out[region] = dec[region]
    return out


def harmonize(
    gen: U8,
    orig: U8,
    mask: BoolMask,
    levels: int,
    seed: int,
    qtables: Any | None = None,
) -> tuple[U8, BoolMask]:
    """Composite ``gen`` into ``orig`` (same-size crops) and match it.

    Returns the crop and the area it changed (mask grown by the blend
    band), to paste with a hard alpha: outside it the crop is ``orig``.
    """
    band = 2**levels
    out = multiband(gen, orig, mask, levels)
    region = dilate(mask, max(1, band // 4))
    ring = dilate(mask, band) & ~region
    out = match_blur(out, region, ring)
    out = match_noise(out, region, ring, seed)
    out = match_jpeg(out, region, qtables)
    changed = dilate(mask, band)
    final = orig.copy()
    final[changed] = out[changed]
    return final, changed
