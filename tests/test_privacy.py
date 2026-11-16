"""Privacy metrics and the PP4AV reader."""

from pathlib import Path

import numpy as np
from PIL import Image

from blur_data.anonymize.regions import Region
from blur_data.data.pp4av import list_images, read_gt
from blur_data.metrics.confusion import Counts, add_frame, box_iou
from blur_data.metrics.privacy import (
    GtBox,
    box_coverage,
    recall,
    score_image,
)


def test_coverage_and_recall() -> None:
    blur = np.zeros((100, 100), bool)
    blur[10:30, 10:30] = True
    face = GtBox("face", 10, 10, 30, 30)  # fully blurred
    plate = GtBox("plate", 20, 20, 40, 40)  # a quarter blurred
    assert box_coverage(face, blur) == 1.0
    assert box_coverage(plate, blur) == 0.25
    s = score_image([face, plate], blur, [blur])
    assert recall([s], "face", 0.8) == (1.0, 1)
    assert recall([s], "plate", 0.8) == (0.0, 1)
    assert recall([s], "plate", 0.2) == (1.0, 1)
    assert np.isnan(recall([s], "face", 0.8, (100, 200))[0])


def test_over_blur_and_false_regions() -> None:
    gt = [GtBox("face", 0, 0, 10, 10)]
    hit = np.zeros((100, 100), bool)
    hit[0:10, 0:10] = True
    miss = np.zeros((100, 100), bool)
    miss[50:60, 50:60] = True
    s = score_image(gt, hit | miss, [hit, miss], grow=0.0)
    assert s.false_regions == 1 and s.regions == 2
    assert abs(s.over_blur - 0.01) < 1e-9


def test_pp4av_reader(tmp_path: Path) -> None:
    city = tmp_path / "paris"
    city.mkdir()
    Image.new("RGB", (200, 100)).save(city / "f1.png")
    (city / "f1.txt").write_text(
        "0 0.5 0.5 0.1 0.2\n1 0.25 0.5 0.1 0.1\n0 2048.0 619.0 0.1 0.1\n"
    )
    fish = tmp_path / "fisheye"
    fish.mkdir()
    Image.new("RGB", (50, 50)).save(fish / "x.png")
    (fish / "x.txt").write_text("")
    ims = list_images(tmp_path)
    assert [im.key for im in ims] == ["paris/f1"]
    gt = read_gt(tmp_path, ims[0])
    assert gt[0].cls == "face"
    assert np.allclose(
        [gt[0].x0, gt[0].y0, gt[0].x1, gt[0].y1], [90, 40, 110, 60]
    )
    assert gt[1].cls == "plate" and len(gt) == 2  # pixel-coord label
    k = ims[0].k_matrix()
    assert k[0, 2] == 100.0 and np.isclose(k[0, 0], np.hypot(200, 100))


def test_detected_recall_counts_tight_blur() -> None:
    """A tight blur inside a loose GT box is detected, not protected."""
    gt = [GtBox("face", 10, 10, 30, 30)]
    m = np.zeros((40, 40), dtype=bool)
    m[14:26, 14:26] = True  # 36 % of the box, IoU 0.36
    c = Counts()
    add_frame(
        c, "face", gt, [Region.from_mask("face", 1.0, m, (14, 14, 26, 26))]
    )
    assert (c.tp, c.fn, c.detected) == (0, 1, 1)
    assert c.detected_recall == 1.0
    assert box_iou(gt[0], (40, 40, 50, 50)) == 0.0
