"""Realism metrics of anonymised heads: seam strength and KID."""

import numpy as np

from blur_data.metrics.realism import kid, seam


def test_seam_and_kid() -> None:
    orig = np.full((64, 64, 3), 100, np.uint8)
    hard = orig.copy()
    hard[20:44, 20:44] = 180  # a pasted block: a hard seam
    soft = orig.copy()
    yy, xx = np.mgrid[:64, :64]
    r = np.hypot(yy - 32, xx - 32)
    soft[:] = (100 + 80 * np.clip((16 - r) / 8, 0, 1))[..., None].astype(
        np.uint8
    )
    h, s = seam(orig, hard), seam(orig, soft)
    assert h is not None and s is not None
    assert h[0] > s[0] > 0 and h[1] > s[1]
    assert seam(orig, orig) is None
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(200, 8)), rng.normal(size=(200, 8))
    assert abs(kid(a, b)) < 0.05 < kid(a, b + 1.0)
