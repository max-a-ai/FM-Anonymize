"""The shipped config loads and matches the dataclass."""

from pathlib import Path

import pytest

from blur_data.anonymize.config import AnonConfig
from blur_data.anonymize.regions import KINDS
from blur_data.utils.viz import PALETTE

CFG = Path(__file__).resolve().parents[1] / "configs" / "anonymize.yaml"


def test_yaml_loads_every_key() -> None:
    cfg = AnonConfig.from_yaml(CFG)
    assert cfg.occupants and cfg.occupant_crop_pad == 0.0
    assert set(cfg.to_dict()) == set(AnonConfig().to_dict())


def test_unknown_key_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("not_a_key: 1\n")
    with pytest.raises(ValueError):
        AnonConfig.from_yaml(bad)


def test_every_kind_has_a_colour() -> None:
    assert set(KINDS) <= set(PALETTE)


def test_ego_zones_cover_the_side_mirrors() -> None:
    cfg = AnonConfig.from_yaml(CFG)
    assert set(cfg.ego_vehicle_zones) == {
        "CAMERA_LEFT_FRONT",
        "CAMERA_RIGHT_FRONT",
    }
    for zones in cfg.ego_vehicle_zones.values():
        for x0, y0, x1, y1 in zones:
            assert 0 <= x0 < x1 <= 2200 and 0 <= y0 < y1 <= 1200
