"""Web app: crawling, path checks, run index and views (no GPU)."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blur_data.anonymize.regions import ImageRegions, Region, regions_path
from blur_data.web import views
from blur_data.web.crawl import analyse, check_paths, folder_of, list_source
from blur_data.web.job import Options, render_one


def _image(path: Path, size: tuple[int, int] = (64, 48)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (90, 120, 150)).save(path)


def _run(tmp_path: Path) -> tuple[Path, Path, str]:
    """A source with one image and a run with one plate region on it."""
    src, run = tmp_path / "src", tmp_path / "run"
    _image(src / "cam" / "f1.png")
    mask = np.zeros((48, 64), dtype=bool)
    mask[10:20, 20:40] = True
    region = Region.from_mask("plate", 0.9, mask, (20, 10, 40, 20))
    det = {"kind": "person", "score": 0.8, "box": [1, 1, 9, 30],
           "decision": "away", "angle_deg": 150.0}  # fmt: skip
    ImageRegions(
        "cam/f1", "cam/f1.png", [region], {"seconds": 1.5}, [det]
    ).save(regions_path(run, "cam/f1"))
    return src, run, "cam/f1"


def test_list_source_skips_hidden_and_other_files(tmp_path: Path) -> None:
    _image(tmp_path / "b" / "x.jpg")
    _image(tmp_path / "a" / "y.png")
    _image(tmp_path / ".cache" / "z.png")
    (tmp_path / "a" / "y.txt").write_text("0 0.5 0.5 0.1 0.1")
    images = list_source(tmp_path)
    assert [im.rel_path for im in images] == ["a/y.png", "b/x.jpg"]
    assert images[0].key == "a/y"
    k = images[0].k_matrix(64, 48)
    assert k[0, 2] == 32 and k[1, 2] == 24 and k[0, 0] == pytest.approx(80)


def test_analyse_counts_folders_sizes_and_done(tmp_path: Path) -> None:
    src, run, _ = _run(tmp_path)
    _image(src / "cam" / "f2.png")
    a = analyse(src, list_source(src), run)
    assert a["images"] == 2 and a["done"] == 1
    assert a["folders"] == [("cam", 2)]
    assert a["sizes"] == [("64x48", 2)]
    assert folder_of("x.png") == "."


def test_check_paths_keeps_output_out_of_source(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    assert check_paths(src, tmp_path / "out") is None
    assert check_paths(src, src) is not None
    assert check_paths(src, src / "out") is not None
    assert check_paths(src, tmp_path) is not None
    assert check_paths(tmp_path / "missing", tmp_path / "out") is not None


def test_options_from_form() -> None:
    o = Options.from_json(
        {
            "src": "/a",
            "dst": "/b",
            "config": "c.yaml",
            "limit": "",
            "every": "0",
            "kinds": None,
        }  # fmt: skip
    )
    assert o.limit is None and o.every == 1 and o.kinds is None and o.render


def test_index_statistics_and_views(tmp_path: Path) -> None:
    src, run, key = _run(tmp_path)
    entries = views.RunIndex().entries(run)
    assert [e.key for e in entries] == [key]
    st = views.statistics(entries)
    assert st["kinds"]["plate"] == 1 and st["decisions"] == {"away": 1}
    assert st["seconds"]["mean"] == 1.5
    regs, rgb = views.load(src, run, key)
    blurred = views.anonymized(run, regs, rgb, None)
    assert blurred.shape == rgb.shape
    over = views.overlay(rgb, regs, None, detections=True)
    assert (over != rgb).any()
    touched = views.mask_view(rgb, regs, None)
    assert (touched[15, 30] == rgb[15, 30]).all()
    assert (touched[40, 5] < rgb[40, 5]).all()
    assert views.region_crop(rgb, regs.regions[0].box).shape == (220, 220, 3)
    assert views.encode(rgb, width=32)[:2] == b"\xff\xd8"


def test_load_refuses_paths_outside(tmp_path: Path) -> None:
    src, run, _ = _run(tmp_path)
    with pytest.raises(LookupError):
        views.load_regions(run, "../../etc/passwd")
    ImageRegions("evil", "../outside.png", []).save(regions_path(run, "evil"))
    with pytest.raises(LookupError):
        views.load(src, run, "evil")


def test_render_one_copies_unchanged_and_blurs(tmp_path: Path) -> None:
    from blur_data.anonymize.config import AnonConfig

    src, run, key = _run(tmp_path)
    regs = ImageRegions.load(regions_path(run, key))
    path = src / regs.rel_path
    cfg = AnonConfig()
    with Image.open(path) as pil:
        render_one(pil, path, regs.regions, tmp_path / "o" / "a.png", cfg, ())
        render_one(
            pil, path, regs.regions, tmp_path / "o" / "b.png", cfg, None
        )
    assert (tmp_path / "o" / "a.png").read_bytes() == path.read_bytes()
    assert (tmp_path / "o" / "b.png").exists()
    assert not list((tmp_path / "o").glob(".*part*"))


def test_guess_source_from_config(tmp_path: Path) -> None:
    import json

    pytest.importorskip("flask")
    from blur_data.web.app import guess_source

    src, run, key = _run(tmp_path)
    (run / "config.json").write_text(json.dumps({"src": str(src)}))
    files = [regions_path(run, key)]
    assert guess_source(run, files) == src
    (run / "config.json").write_text(json.dumps({"src": str(tmp_path)}))
    assert guess_source(run, files) is None  # its image is not there


def test_method_of_reads_config_then_name(tmp_path: Path) -> None:
    import json

    run = tmp_path / "x"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"method": "urban"}))
    assert views.method_of(run) == "UrbanAnonymizer"
    (run / "config.json").write_text(json.dumps({"config": "a.yaml"}))
    assert views.method_of(run) == "FM-Blur"
    assert views.method_of(tmp_path / "deface-v2b") == "deface"
    assert views.method_of(tmp_path / "something") == "other"


def test_compare_overlay_draws_each_run(tmp_path: Path) -> None:
    src, run, key = _run(tmp_path)
    regs, rgb = views.load(src, run, key)
    green, blue = (0, 255, 0), (0, 0, 255)
    out = views.compare_overlay(rgb, [(regs, green), (regs, blue)], masks=True)
    pixels = {tuple(int(v) for v in p) for p in out.reshape(-1, 3)}
    assert green in pixels and blue in pixels
