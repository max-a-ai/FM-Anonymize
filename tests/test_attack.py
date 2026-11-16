import importlib.util
from pathlib import Path

from blur_data.metrics.privacy import GtBox

SCRIPT = Path(__file__).parents[1] / "blur_data/scripts/blur-data-attack.py"
spec = importlib.util.spec_from_file_location("attack", SCRIPT)
assert spec is not None and spec.loader is not None
attack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(attack)


def test_lcs_and_alnum() -> None:
    assert attack.alnum("m-ab 123") == "MAB123"
    assert attack.lcs("MAB123", "XAB1Y3") == 4
    assert attack.lcs("ABC", "") == 0


def test_square_window_is_clipped() -> None:
    g = GtBox("face", 0, 0, 10, 20)
    assert attack.square(g, (100, 100), 1.3) == (0, 0, 18, 23)
