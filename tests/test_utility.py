"""Pseudo-GT utility scoring (COCO AP via pycocotools)."""

import numpy as np

from blur_data.metrics.utility import (
    HEAD,
    Persons,
    coco_ap,
    dt_annotations,
    extra_lost,
    gt_annotations,
    subset,
)


def _persons(shift: float = 0.0) -> dict[int, Persons]:
    boxes = np.array([[10, 10, 110, 310], [200, 20, 240, 100]], np.float32)
    kp = np.zeros((2, 17, 3), np.float32)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        kp[i, :, 0] = np.linspace(x0, x1, 17)
        kp[i, :, 1] = np.linspace(y0, y1, 17)
        kp[i, :, 2] = 5.0
    kp[1, 5:, 2] = -1.0  # small person: only the head is visible
    kp[:, :, 0] += shift
    return {0: Persons(boxes, np.array([0.9, 0.8], np.float32), kp)}


def test_identical_scores_one_and_counts() -> None:
    gt = gt_annotations(_persons(), 0.5, 2.0)
    assert [g["num_keypoints"] for g in gt] == [17, 5]
    res = coco_ap(gt, dt_annotations(_persons(), None), [0], None)
    assert res["AP"]["all"] == 1.0 and res["n"]["<64"] == 1.0
    kp = coco_ap(gt, dt_annotations(_persons(), (0, 17)), [0], (0, 17))
    assert kp["AP"]["all"] == 1.0


def test_moved_head_keypoints_lower_head_ap() -> None:
    gt = gt_annotations(_persons(), 0.5, 2.0, HEAD)
    moved = _persons(shift=6.0)
    head = coco_ap(gt, dt_annotations(moved, HEAD), [0], HEAD)
    same = coco_ap(gt, dt_annotations(_persons(), HEAD), [0], HEAD)
    assert head["AP"]["all"] < same["AP"]["all"] == 1.0


def test_score_threshold_and_empty() -> None:
    assert len(gt_annotations(_persons(), 0.85, 2.0)) == 1
    res = coco_ap([], [], [0], None)
    assert res["AP"] == {} and res["n"]["all"] == 0.0


def test_subset_ignores_untouched_persons() -> None:
    gt = gt_annotations(_persons(), 0.5, 2.0)
    part = subset(gt, {0: [(60.0, 40.0)]})  # head of the first person
    assert [g["iscrowd"] for g in part] == [0, 1]
    moved = _persons(shift=40.0)  # second person's detection now misses
    res = coco_ap(part, dt_annotations(moved, None), [0], None)
    assert res["n"]["all"] == 1.0


def test_extra_and_lost_persons() -> None:
    orig = _persons()
    moved = _persons(shift=0.0)
    moved[0].boxes[1] += 200.0  # the small person moves away
    assert extra_lost(orig, orig) == {"extra": 0, "lost": 0}
    assert extra_lost(orig, moved) == {"extra": 1, "lost": 1}
