"""Seams and realism of anonymised heads (v6).

* **Seam**: where a version changed the image (``|anon - orig| > 10``,
  closed), the band across that boundary should not carry extra edges
  or a colour step. Per head: the mean gradient magnitude on the band
  minus the original's at the same pixels (``grad``), and the colour
  difference between the band's inner and outer half minus the
  original's (``colour``, RGB L2). 0 = no seam; blur and grey-fill
  inpainting have positive values.
* **Realism**: KID (kernel inception distance, unbiased MMD^2 with the
  cubic polynomial kernel) between Inception-v3 pool features of the
  anonymised head crops and of the original head crops of the same
  dataset. Lower = closer to real heads; reported x 1000.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from blur_data.utils.masks import BoolMask, dilate

U8 = npt.NDArray[np.uint8]
F64 = npt.NDArray[np.float64]


def changed_area(orig: U8, anon: U8, thr: int = 10) -> BoolMask:
    """Pixels a version changed, closed and cleaned (3 px)."""
    diff = np.abs(orig.astype(np.int16) - anon.astype(np.int16)).max(axis=2)
    m = (diff > thr).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    out: BoolMask = m.astype(bool)
    return out


def _grad(img: U8) -> npt.NDArray[np.float32]:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    out: npt.NDArray[np.float32] = np.sqrt(gx**2 + gy**2)
    return out


def seam(orig: U8, anon: U8, width: int = 2) -> tuple[float, float] | None:
    """(gradient excess, colour step excess) at the changed area's
    boundary; None if nothing changed."""
    area = changed_area(orig, anon)
    if area.sum() < 20:
        return None
    inner = area & dilate(~area, width)  # inside, near the border
    outer = ~area & dilate(area, width)
    band = inner | outer
    if not inner.any() or not outer.any():
        return None
    ga, go = _grad(anon), _grad(orig)
    grad = float(ga[band].mean() - go[band].mean())

    def step(img: U8) -> float:
        a, b = img[inner].astype(np.float64), img[outer].astype(np.float64)
        return float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))

    return grad, step(anon) - step(orig)


def kid(x: F64, y: F64) -> float:
    """Unbiased KID (MMD^2, kernel ``(x.y / d + 1)^3``) of two sets."""
    d = x.shape[1]
    kxx = (x @ x.T / d + 1) ** 3
    kyy = (y @ y.T / d + 1) ** 3
    kxy = (x @ y.T / d + 1) ** 3
    m, n = len(x), len(y)
    sxx = (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
    syy = (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
    return float(sxx + syy - 2 * kxy.mean())
