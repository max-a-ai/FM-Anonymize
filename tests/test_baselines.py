"""Detection box -> region conversions of the baselines."""

import numpy as np

from blur_data.baselines import (
    box_sliced_mask,
    padded_box,
    to_box_regions,
    to_regions,
)


def test_padded_box_grows_and_clips() -> None:
    # 20 % of 10 x 20 px per side, rounded outwards
    assert padded_box([10, 10, 20, 30], (100, 100), 0.2) == (8, 6, 22, 34)
    # clipped at the image border
    assert padded_box([0, 0, 10, 10], (8, 9), 0.2) == (0, 0, 9, 8)
    # fully outside after clipping
    assert padded_box([120, 5, 130, 9], (50, 100), 0.2) is None


def test_to_box_regions_is_the_padded_rectangle() -> None:
    (r,) = to_box_regions("plate", [([10, 10, 20, 30], 0.7)], (50, 40), 0.2)
    m = r.mask()
    assert r.kind == "plate" and r.score == 0.7
    assert r.box == (8, 6, 22, 34)
    assert m.sum() == 14 * 28 and m[6:34, 8:22].all()
    assert r.meta["box"] == [10.0, 10.0, 20.0, 30.0]


def test_to_regions_ellipse_scales_with_box() -> None:
    (a,) = to_regions("face", [([20, 20, 40, 40], 0.9)], (60, 60), 1.0)
    (b,) = to_regions("face", [([20, 20, 40, 40], 0.9)], (60, 60), 1.15)
    assert b.mask().sum() > a.mask().sum()
    assert np.array_equal(a.mask() & b.mask(), a.mask())


def test_box_sliced_mask_keeps_mask_inside_int_box() -> None:
    m = np.ones((10, 10), bool)
    out = box_sliced_mask(m, [2.7, 1.2, 6.9, 4.99])
    # int() truncates: rows 1..3, cols 2..5
    assert out.sum() == 3 * 4 and out[1:4, 2:6].all()
    # a start below -1 wraps as in the tool's slicing: nothing is blurred
    assert not box_sliced_mask(m, [-3.5, 0, 5, 5]).any()
