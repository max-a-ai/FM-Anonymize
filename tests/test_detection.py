"""Tiling, placement and duplicate suppression."""

import numpy as np

from blur_data.models.detection import (
    Detection,
    box_inside,
    dedupe,
    place,
    square_tiles,
    touches_inner_edge,
)


def test_square_tiles_cover_frame() -> None:
    tiles = square_tiles(2200, 1200)
    assert tiles == [(0, 0, 1200, 1200), (1000, 0, 2200, 1200)]
    assert square_tiles(1000, 1200) == []


def test_inner_edge_only() -> None:
    win = (1000, 0, 2200, 1200)
    frame = (2200, 1200)
    assert touches_inner_edge((1001, 50, 1100, 90), win, frame)
    assert not touches_inner_edge((1100, 0, 2200, 90), win, frame)


def test_place() -> None:
    m = np.ones((2, 3), bool)
    full = place(m, (4, 1, 7, 3), (10, 5))
    assert full.sum() == 6 and full[1:3, 4:7].all()


def _det(score: float, x0: int, x1: int) -> Detection:
    mask = np.zeros((10, 20), bool)
    mask[2:8, x0:x1] = True
    return Detection("p", score, (x0, 2, x1, 8), mask)


def test_dedupe_iou_containment_and_occlusion() -> None:
    big = _det(0.9, 0, 10)
    dup = _det(0.5, 1, 10)  # 100 % inside big -> dropped
    other = _det(0.7, 12, 18)  # disjoint -> kept
    kept = dedupe([dup, other, big], iou_thr=0.5)
    assert [d.score for d in kept] == [0.9, 0.7]


def test_box_inside() -> None:
    assert box_inside((5, 5, 7, 7), (0, 0, 10, 10), 0.0)
    assert not box_inside((11, 5, 13, 7), (0, 0, 10, 10), 0.0)
    assert box_inside((11, 5, 12, 7), (0, 0, 10, 10), 0.2)


def test_plausible_head_hull() -> None:
    from blur_data.anonymize.pipeline import _plausible

    hull = np.zeros((100, 100), bool)
    hull[10:20, 40:50] = True
    assert _plausible(hull, (35.0, 12.0, 60.0, 90.0))
    assert not _plausible(hull, (70.0, 40.0, 90.0, 95.0))


def test_box_smoother_is_steady() -> None:
    from blur_data.paper.video import BoxSmoother

    sm = BoxSmoother()
    assert sm.update([[0, 0, 10, 10]]) == []  # needs two hits
    out = sm.update([[2, 0, 12, 10]])
    assert len(out) == 1 and abs(out[0][0] - 1.0) < 1e-9  # smoothed
    assert len(sm.update([])) == 1  # one missed frame: still drawn
    sm.update([])
    sm.update([])
    assert sm.update([]) == []  # gone after max_miss
