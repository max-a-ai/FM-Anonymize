"""Re-identification scores and the parrot tracker."""

import numpy as np

from blur_data.metrics import reid


def _sim() -> reid.F32:
    rng = np.random.default_rng(0)
    sim = rng.uniform(-0.2, 0.2, (50, 50)).astype(np.float32)
    sim[np.arange(25), np.arange(25)] = 0.9  # first half re-identified
    return sim


def test_rank1_and_tar() -> None:
    sim = _sim()
    assert reid.rank1(sim) == 0.5
    thr = reid.far_threshold(sim, 1e-3)
    assert 0.15 < thr <= 0.2
    assert reid.tar(sim, thr) == 0.5
    rows = np.zeros(50, bool)
    rows[:10] = True
    assert reid.rank1(sim, rows) == 1.0 and reid.tar(sim, thr, rows) == 1.0


def test_scores_bins() -> None:
    sizes = [25.0] * 25 + [40.0] * 20 + [80.0] * 5
    res = reid.scores(_sim(), sizes, 1e-3)
    bins = res["bins"]
    assert isinstance(bins, dict)
    assert bins["20-32"] == {"n": 25, "rank1": 1.0, "tar": 1.0}
    assert bins[">=64"]["n"] == 5 and bins[">=64"]["rank1"] == 0.0
    assert res["rank1"] == 0.5


def test_track_follows_and_refuses_ambiguity() -> None:
    start = (0.0, 0.0, 10.0, 20.0)
    frames = [[(1.0, 0.0, 11.0, 20.0), (50.0, 0.0, 60.0, 20.0)]] * 3
    assert reid.track(start, frames, 0.3) == (1.0, 0.0, 11.0, 20.0)
    assert reid.track(start, [[(30.0, 0.0, 40.0, 20.0)]], 0.3) is None
    twin = [[(1.0, 0.0, 11.0, 20.0), (0.0, 1.0, 10.0, 21.0)]]
    assert reid.track(start, twin, 0.3) is None


def test_moved_window() -> None:
    before = (0.0, 0.0, 10.0, 20.0)
    after = (100.0, 0.0, 120.0, 40.0)  # shifted, 2x larger
    face = (3.0, 0.0, 7.0, 4.0)
    assert reid.moved(face, before, after) == (106.0, 0.0, 114.0, 8.0)
