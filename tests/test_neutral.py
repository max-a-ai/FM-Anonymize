import numpy as np

from blur_data.anonymize.regions import Region
from blur_data.metrics.confusion import Counts, add_frame, split_small_faces
from blur_data.metrics.privacy import GtBox


def _region(kind: str, box: tuple[int, int, int, int]) -> Region:
    m = np.zeros((100, 100), bool)
    m[box[1] : box[3], box[0] : box[2]] = True
    return Region.from_mask(kind, 1.0, m, box)


def test_split_small_faces_keeps_plates() -> None:
    boxes = [
        GtBox("face", 0, 0, 10, 10),
        GtBox("face", 0, 0, 30, 30),
        GtBox("plate", 0, 0, 5, 5),
    ]
    scored, neutral = split_small_faces(boxes, 20)
    assert [g.size for g in scored] == [30, 5]
    assert [g.size for g in neutral] == [10]


def test_blur_on_small_face_is_neutral_not_fp() -> None:
    big, small = GtBox("face", 50, 50, 80, 80), GtBox("face", 5, 5, 15, 15)
    regions = [
        _region("face", (50, 50, 80, 80)),  # the scored face: matched
        _region("face", (5, 5, 15, 15)),  # the small face: neutral
        _region("face", (85, 5, 95, 15)),  # nothing there: FP
    ]
    c = Counts()
    add_frame(c, "face", [big], regions, neutral=[small])
    assert (c.tp, c.fn, c.matched, c.neutral, c.fp) == (1, 0, 1, 1, 1)
    plain = Counts()
    add_frame(plain, "face", [big], regions)
    assert (plain.neutral, plain.fp) == (0, 2)
