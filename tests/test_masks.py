"""RLE codec and mask helpers."""

import numpy as np

from blur_data.utils.masks import (
    coverage,
    decode_rle,
    dilate,
    encode_rle,
    fill_box,
    mask_box,
    mask_iou,
)


def test_rle_roundtrip() -> None:
    rng = np.random.default_rng(0)
    mask = rng.random((37, 53)) > 0.7
    assert np.array_equal(decode_rle(encode_rle(mask)), mask)


def test_mask_box_and_empty() -> None:
    mask = np.zeros((10, 10), bool)
    assert mask_box(mask) is None
    mask[2:5, 3:8] = True
    assert mask_box(mask) == (3, 2, 8, 5)


def test_dilate_grows_and_keeps() -> None:
    mask = np.zeros((21, 21), bool)
    mask[10, 10] = True
    out = dilate(mask, 3)
    assert out[10, 10] and out[10, 13] and not out[10, 14]
    assert np.array_equal(dilate(mask, 0), mask)


def test_iou_and_coverage() -> None:
    a = np.zeros((4, 4), bool)
    b = np.zeros((4, 4), bool)
    a[:2] = True
    b[:1] = True
    assert mask_iou(a, b) == 0.5
    assert coverage(b, a) == 1.0
    assert coverage(a, b) == 0.5


def test_fill_box_covers_a_sheared_plate() -> None:
    m = np.zeros((10, 12), bool)
    for y in range(2, 8):  # parallelogram, shifted one px per row
        m[y, y : y + 4] = True
    f = fill_box(m)
    assert f[m].all()
    assert f.sum() == 6 * 9  # rows 2..7, columns 2..10
    assert not fill_box(np.zeros((3, 3), bool)).any()


def test_scale_mask_grows_about_centre() -> None:
    from blur_data.utils.masks import mask_box, scale_mask

    m = np.zeros((100, 100), dtype=bool)
    m[40:60, 30:50] = True
    big = scale_mask(m, 1.5)
    assert big[m].all()
    assert mask_box(big) == (25, 35, 55, 65)
    assert np.array_equal(scale_mask(m, 1.0), m)
