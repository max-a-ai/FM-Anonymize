"""Blur compositing and edit limits."""

from pathlib import Path

import numpy as np

from blur_data.anonymize.blur import (
    U8,
    WHITE,
    apply_regions,
    blur_mask,
    plate_fill_of,
)
from blur_data.anonymize.regions import ImageRegions, Region


def _img() -> U8:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)


def _region(x0: int, score: float) -> Region:
    m = np.zeros((60, 80), bool)
    m[10:30, x0 : x0 + 20] = True
    return Region.from_mask("plate", score, m, (x0, 10, x0 + 20, 30))


def test_blur_changes_inside_only_near() -> None:
    img = _img()
    r = _region(5, 0.9)
    out = blur_mask(img, r.mask(), 0.3)
    inside = out[10:30, 5:25].astype(float) - img[10:30, 5:25]
    assert np.abs(inside).mean() > 20
    assert np.array_equal(out[50:, 60:], img[50:, 60:])


def test_limit_counts_edits() -> None:
    img = _img()
    regions = [_region(5, 0.9), _region(50, 0.5)]
    assert np.array_equal(apply_regions(img, regions, limit=0), img)
    one = apply_regions(img, regions, limit=1)
    assert np.array_equal(one[:, 45:], img[:, 45:])
    assert not np.array_equal(apply_regions(img, regions), one)


def test_region_file_roundtrip(tmp_path: Path) -> None:
    regs = ImageRegions("CAM/1", "x.jpg", [_region(50, 0.5), _region(5, 0.9)])
    path = tmp_path / "a" / "b.json"
    regs.save(path)
    back = ImageRegions.load(path)
    assert [r.score for r in back.ordered()] == [0.9, 0.5]
    assert np.array_equal(back.regions[0].mask(), regs.regions[0].mask())


def test_review_sheet_layout() -> None:
    from blur_data.anonymize.review import GAP_PX, HEADER_PX, review_sheet

    img = _img()
    regions = [_region(5, 0.9), _region(50, 0.5)]
    blurred = apply_regions(img, regions)
    sheet = review_sheet(img, blurred, regions, "CAM/1")
    h, w = img.shape[:2]
    assert sheet.shape == (HEADER_PX + h, 2 * w + GAP_PX, 3)
    assert np.array_equal(sheet[HEADER_PX:, w + GAP_PX :], blurred)
    left = sheet[HEADER_PX:, :w]
    assert np.array_equal(left[45:, 30:45], img[45:, 30:45])  # unmarked
    assert not np.array_equal(left[10:30, 5:25], img[10:30, 5:25])


def test_plate_filled_white_others_blurred() -> None:
    img = _img()
    plate = _region(5, 0.9)
    out = apply_regions(img, [plate], plate_fill=WHITE)
    assert (out[10:30, 5:25] == 255).all()
    assert np.array_equal(out[:, 30:], img[:, 30:])
    m = plate.mask()
    face = Region.from_mask("face", 0.9, m, (5, 10, 25, 30))
    blurred = apply_regions(img, [face], plate_fill=WHITE)
    assert not (blurred[10:30, 5:25] == 255).all()


def test_plate_fill_of_run_config() -> None:
    assert plate_fill_of({}) == WHITE
    assert plate_fill_of({"plate_fill": "blur"}) is None
