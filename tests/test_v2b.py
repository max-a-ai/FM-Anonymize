"""Listing of the real export (skipped when it is not linked)."""

import json
from pathlib import Path

import pytest

from blur_data.data.v2b import list_camera_images

ROOT = Path(__file__).resolve().parents[1] / "resources" / "data" / "v2b"


@pytest.mark.skipif(not ROOT.exists(), reason="v2b not linked")
def test_lists_all_cameras() -> None:
    images = list_camera_images(ROOT)
    scenes = {im.scene for im in images}
    assert len({im.channel for im in images}) == 8
    for scene in scenes:  # 8 cameras, one frame each per sample
        n = sum(im.scene == scene for im in images)
        assert n % 8 == 0 and n >= 8 * 100
    first = images[0]
    assert (ROOT / first.rel_path).exists()
    assert first.key.endswith(f"{first.channel}/{first.stem}")
    assert first.k_matrix().shape == (3, 3)
    assert first.width == 2200 and first.height == 1200


def _write_scene(scene: Path) -> None:
    v = scene / "v1.0-mini"
    v.mkdir(parents=True)
    k = [[1000.0, 0, 500], [0, 1000.0, 400], [0, 0, 1]]
    tables = {
        "sensor": [{"token": "s", "channel": "CAM_A"}],
        "calibrated_sensor": [
            {"token": "c", "sensor_token": "s", "camera_intrinsic": k}
        ],
        "sample_data": [
            {
                "fileformat": "jpg",
                "calibrated_sensor_token": "c",
                "filename": "sensor_data/CAM_A/1.jpg",
                "timestamp": 1.0,
                "width": 4,
                "height": 3,
            },
            {"fileformat": "pcd.bin", "filename": "x.bin"},
        ],
    }
    for name, rows in tables.items():
        (v / f"{name}.json").write_text(json.dumps(rows))


def test_flat_and_nested_layouts(tmp_path: Path) -> None:
    _write_scene(tmp_path / "flat")
    flat = list_camera_images(tmp_path / "flat")
    assert [im.key for im in flat] == ["CAM_A/1"]
    assert flat[0].rel_path == "sensor_data/CAM_A/1.jpg"
    _write_scene(tmp_path / "multi" / "b")
    _write_scene(tmp_path / "multi" / "a")
    nested = list_camera_images(tmp_path / "multi")
    assert [im.key for im in nested] == ["a/CAM_A/1", "b/CAM_A/1"]
    assert nested[0].rel_path == "a/sensor_data/CAM_A/1.jpg"
