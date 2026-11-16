"""FM-Anonymize-Gen without a GPU: routing, crops, the per-image step,
pose guide, compositing, tracks, config, FaceID helpers, CLI keys."""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from blur_data.anonymize import blur
from blur_data.anonymize.blur import U8
from blur_data.anonymize.head import F64
from blur_data.anonymize.regions import Region
from blur_data.generate.config import GenConfig, split_ref
from blur_data.generate.crops import (
    FILL,
    crop_square,
    model_input,
    square_window,
)
from blur_data.generate.harmonize import (
    harmonize,
    match_blur,
    match_noise,
    multiband,
    noise_level,
    sharpness,
)
from blur_data.generate.inpaint import apply_regions
from blur_data.generate.pose_guide import (
    FitGuides,
    Guide,
    HeadFit,
    TemplateHead,
    depth_image,
    head_faces,
    openpose_image,
    openpose_points,
    placed,
    plausible,
    rasterize_depth,
    rotation_deg,
    similarity,
    to_crop,
)
from blur_data.generate.routing import (
    blur_reason,
    grow_mask,
    is_generated,
    region_seed,
    track_seed,
)
from blur_data.generate.tracks import (
    NO_HEAD,
    grown,
    head_tracks,
    link,
    person_tracks,
    segments,
)
from blur_data.utils.masks import BoolMask, dilate, mask_box

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "configs" / "generate.yaml"
SCRIPT = REPO / "blur_data" / "scripts" / "blur-data-generate.py"
K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])


def _cfg() -> GenConfig:
    return GenConfig(resolution=64)


def _region(
    kind: str, x0: int, y0: int, side: int, shape: tuple[int, int] = (120, 160)
) -> Region:
    m = np.zeros(shape, bool)
    m[y0 : y0 + side, x0 : x0 + side] = True
    box = (x0, y0, min(x0 + side, shape[1]), min(y0 + side, shape[0]))
    return Region.from_mask(kind, 0.9, m, box)


class Recorder:
    """Mock generator: records its inputs, returns a constant or noise."""

    def __init__(self, value: int | None = None) -> None:
        self.value = value
        self.calls: list[tuple[U8, BoolMask, int]] = []
        self.guides: list[Guide | None] = []

    def __call__(
        self, image: U8, mask: BoolMask, seed: int, guide: Guide | None = None
    ) -> U8:
        self.calls.append((image.copy(), mask.copy(), seed))
        self.guides.append(guide)
        if self.value is not None:
            return np.full_like(image, self.value)
        rng = np.random.default_rng(seed)
        return rng.integers(0, 256, image.shape, dtype=np.uint8)


# routing ----------------------------------------------------------------


def test_routing_and_size_gate() -> None:
    cfg, shape = _cfg(), (120, 160)
    assert is_generated(_region("head", 0, 0, 30), cfg, shape)
    assert is_generated(_region("occupant", 0, 0, 30), cfg, shape)
    for kind in ("face", "head_fallback", "plate", "window"):
        r = _region(kind, 0, 0, 30)
        assert blur_reason(r, cfg, shape) == "kind"
    assert blur_reason(_region("head", 0, 0, 23), cfg, shape) == "small"
    assert is_generated(_region("head", 0, 0, 24), cfg, shape)


def test_size_cap_blurs_outliers() -> None:
    cfg, shape = _cfg(), (120, 160)  # cap: 0.25 x 120 = 30 px
    assert is_generated(_region("head", 0, 0, 30), cfg, shape)
    assert blur_reason(_region("head", 0, 0, 31), cfg, shape) == "large"
    big = (1200, 2200)  # our frames: 300 px
    assert is_generated(_region("head", 0, 0, 300, big), cfg, big)
    assert not is_generated(_region("occupant", 0, 0, 301, big), cfg, big)


def test_seeds_are_deterministic_and_distinct() -> None:
    assert region_seed("a/b", 0) == region_seed("a/b", 0)
    seeds = {region_seed(k, i) for k in ("a/b", "a/c") for i in range(3)}
    assert len(seeds) == 6
    assert track_seed("a/b/1") == track_seed("a/b/1")
    assert track_seed("a/b/1") != track_seed("a/b/2")
    assert track_seed("a/b/1") != region_seed("a/b/1", 0)


# config -------------------------------------------------------------------


def test_shipped_configs_load_every_key(tmp_path: Path) -> None:
    cfg = GenConfig.from_yaml(CFG)
    assert set(cfg.to_dict()) == set(GenConfig().to_dict())
    assert cfg.generate_kinds == ["head", "occupant"]
    assert cfg.link_on == "person" and cfg.strength == 0.8
    pp = GenConfig.from_yaml(REPO / "configs" / "generate-pp4av.yaml")
    diff = {k for k, v in cfg.to_dict().items() if pp.to_dict()[k] != v}
    assert diff == {"link_on"} and pp.link_on == "head"
    assert split_ref(cfg.image_encoder)[1]  # every model is pinned
    bad = tmp_path / "bad.yaml"
    bad.write_text("not_a_key: 1\n")
    with pytest.raises(ValueError):
        GenConfig.from_yaml(bad)
    with pytest.raises(ValueError):
        GenConfig(link_on="frame")
    with pytest.raises(ValueError):
        GenConfig(strength=0.0)


# crops ----------------------------------------------------------------------


def test_square_window_centred_shifted_padded() -> None:
    win = square_window((40, 40, 60, 60), (100, 100), 2.0, 1.25)
    assert win == (30, 30, 40)
    x0, y0, side = square_window((0, 0, 20, 10), (100, 100), 2.0, 1.25)
    assert (x0, y0, side) == (0, 0, 40)  # shifted into the image
    x0, y0, side = square_window((5, 5, 95, 45), (50, 100), 2.0, 1.25)
    assert side == 113 and y0 < 0  # larger than the image: centred
    assert x0 <= 5 and x0 + side >= 95


def test_crop_pads_with_fill_never_reflects() -> None:
    img = np.full((10, 10, 3), 200, np.uint8)
    crop = crop_square(img, -4, -4, 8, FILL)
    assert crop.shape == (8, 8, 3)
    assert (crop[:4] == FILL).all() and (crop[:, :4] == FILL).all()
    assert (crop[4:, 4:] == 200).all()


# one image ------------------------------------------------------------------


def test_blurred_routes_match_fm_anonymize() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    regions = [_region("plate", 10, 10, 30), _region("head", 90, 60, 20)]
    gen = Recorder(0)
    out = apply_regions(img, regions, gen, "k", _cfg())
    assert not gen.calls  # 20 px head: below the gate
    assert np.array_equal(out, blur.apply_regions(img, regions))


def test_generator_starts_from_fm_anonymizes_blur() -> None:
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    regions = [_region("head", 40, 40, 30), _region("plate", 100, 80, 30)]
    cfg = _cfg()
    gen = Recorder(30)
    out = apply_regions(img, regions, gen, "k", cfg)
    ((image, mask, _),) = gen.calls
    # the generator sees FM-Anonymize's blur of every region
    m = grow_mask(regions[0].mask(), cfg)
    box = mask_box(m)
    assert box is not None
    x0, y0, side = square_window(box, (120, 160), cfg.context, cfg.min_context)
    want = crop_square(blur.apply_regions(img, regions), x0, y0, side, FILL)
    cm = crop_square(m, x0, y0, side, False)
    big, gen_mask = model_input(want, cm, 64)
    assert np.array_equal(image, big) and np.array_equal(mask, gen_mask)
    orig, _ = model_input(crop_square(img, x0, y0, side, FILL), cm, 64)
    assert not np.array_equal(image[mask], orig[mask])  # no original head
    # outside the head and the plate the output stays the original
    far = ~dilate(regions[0].mask() | regions[1].mask(), 12)
    assert np.array_equal(out[far], img[far])


def test_generated_head_is_pasted_inside_its_band() -> None:
    img = np.zeros((120, 160, 3), np.uint8)
    r = _region("head", 70, 50, 30)
    out = apply_regions(img, [r], Recorder(255), "k", _cfg())
    inner = ~dilate(~r.mask(), 8)
    assert (out[inner] == 255).all()  # the head region is replaced
    assert (out[~dilate(r.mask(), 3 + 8 + 1)] == 0).all()  # grown + band


def test_region_and_track_seeds_reach_the_generator() -> None:
    img = np.zeros((120, 160, 3), np.uint8)
    regions = [_region("head", 20, 20, 30), _region("head", 90, 60, 30)]
    g1, g2 = Recorder(), Recorder()
    apply_regions(img, regions, g1, "x/1", _cfg())
    assert [s for *_, s in g1.calls] == [region_seed("x/1", i) for i in (0, 1)]
    apply_regions(img, regions, g2, "x/1", _cfg(), seeds={1: 7})
    assert [s for *_, s in g2.calls] == [region_seed("x/1", 0), 7]


# pose guide -----------------------------------------------------------------


def _rot(deg: float) -> F64:
    a = np.radians(deg)
    return np.array(
        [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    )


def _template() -> TemplateHead:
    """A unit square (two triangles) as the 'head'."""
    v = np.array(
        [[-0.1, -0.1, 0], [0.1, -0.1, 0], [0.1, 0.1, 0], [-0.1, 0.1, 0]]
    )
    f = np.array([[0, 1, 2], [0, 2, 3]])
    return TemplateHead(np.arange(4), v, f)


def _fit(t: TemplateHead, z: float = 2.0) -> HeadFit:
    eye = np.eye(3)
    return HeadFit(1.0, eye, np.array([0, 0, z]), {}, 0.0, eye[1], eye[2])


def test_similarity_recovers_a_placement() -> None:
    rng = np.random.default_rng(0)
    src = rng.normal(size=(50, 3))
    r, t = _rot(30), np.array([0.5, -0.2, 3.0])
    s, r2, t2 = similarity(src, 1.7 * (r @ src.T).T + t)
    assert abs(s - 1.7) < 1e-9 and np.allclose(r2, r) and np.allclose(t2, t)
    assert (
        rotation_deg(r, r2) < 1e-6
        and abs(rotation_deg(np.eye(3), r) - 30) < 1e-6
    )


def test_head_faces_keep_only_head_triangles() -> None:
    faces = np.array([[0, 1, 2], [2, 3, 4], [1, 2, 3]])
    out = head_faces(faces, np.array([1, 2, 3]))
    assert out.tolist() == [[0, 1, 2]]


def test_depth_raster_lands_in_the_crop() -> None:
    t = _template()
    z = rasterize_depth(placed(t, _fit(t)), t.faces, K, (270, 190, 100), 100)
    hit = np.isfinite(z)
    assert hit.any() and np.allclose(z[hit], 2.0)
    ys, xs = np.nonzero(hit)
    assert abs(xs.mean() - 50) < 2 and abs(ys.mean() - 50) < 2
    img = depth_image(z)
    assert img.shape == (100, 100, 3) and img[~hit].max() == 0
    assert img[hit].min() >= 64  # the head stays off black


def test_openpose_geometry_and_back_view() -> None:
    kp3d = np.zeros((70, 3))
    kp3d[:, 2] = 2.0
    kp3d[3], kp3d[4] = [-0.08, 0, 2.0], [0.08, 0, 2.0]  # ears
    kp3d[0] = [0, 0, 1.9]  # nose towards the camera
    kp2d = np.tile([320.0, 240.0], (70, 1))
    assert 0 in openpose_points(kp3d, kp2d)
    kp3d[0] = [0, 0, 2.1]  # nose away: a back view draws no face
    assert 0 not in openpose_points(kp3d, kp2d)
    canvas = openpose_image({0: (320.0, 240.0)}, (270, 190, 100), 100)
    assert canvas[50, 50].max() > 0 and canvas[0, 0].max() == 0


def test_plausible_rejects_off_box_and_behind_camera() -> None:
    t = _template()
    box = (290.0, 210.0, 350.0, 270.0)
    assert plausible(t, _fit(t), K, box, box)
    assert not plausible(t, _fit(t), K, (0.0, 0.0, 50.0, 50.0), box)
    assert not plausible(t, _fit(t, -2.0), K, box, box)


def test_crop_mapping() -> None:
    uv = np.array([[110.0, 60.0]])
    assert to_crop(uv, (100, 50, 20), 200).tolist() == [[100.0, 100.0]]


def _head(x0: int) -> Region:
    m = np.zeros((480, 640), bool)
    m[200:240, x0 : x0 + 40] = True
    return Region.from_mask("head", 0.9, m, (x0, 200, x0 + 40, 240))


def test_guides_and_unguided_heads() -> None:
    t = _template()
    fit = HeadFit(
        0.2, np.eye(3), np.array([0, 0, 2.0]), {0: (320.0, 220.0)}, 0.0,
        np.eye(3)[1], np.eye(3)[2],
    )  # fmt: skip
    guides = FitGuides(t, {0: fit}, K)  # region 1 has no fit
    rec = Recorder(90)
    img = np.zeros((480, 640, 3), np.uint8)
    apply_regions(img, [_head(300), _head(450)], rec, "s/C/1", _cfg(), guides)
    g0, g1 = rec.guides
    assert g0 is not None and g0.depth.shape == (64, 64, 3) and g1 is None
    assert g0.depth.max() > 0


# compositing ----------------------------------------------------------------


def _img(seed: int = 0) -> npt.NDArray[np.uint8]:
    rng = np.random.default_rng(seed)
    return rng.integers(60, 200, (64, 64, 3), dtype=np.uint8)


def test_multiband_keeps_dst_outside_and_src_inside() -> None:
    src, dst = np.full((64, 64, 3), 200, np.uint8), _img()
    m = np.zeros((64, 64), bool)
    m[20:44, 20:44] = True
    out = multiband(src, dst, m, 3)
    assert np.abs(out[:4, :4].astype(int) - dst[:4, :4]).max() <= 2
    assert np.abs(out[30:34, 30:34].astype(int) - 200).max() <= 8


def test_noise_and_blur_matching() -> None:
    rng = np.random.default_rng(1)
    flat = np.full((64, 64, 3), 128, np.uint8)
    noise = rng.normal(0, 8, (64, 64, 3))
    noisy = np.clip(128 + noise, 0, 255).astype(np.uint8)
    img = noisy.copy()
    region = np.zeros((64, 64), bool)
    region[16:48, 16:48] = True
    img[region] = flat[region]  # a clean patch in a noisy scene
    ring = ~region
    out = match_noise(img, region, ring, seed=0)
    assert noise_level(out, region) > 0.6 * noise_level(out, ring)
    sharp = _img(2)
    soft = np.asarray(cv2.GaussianBlur(sharp, (0, 0), 2.0), np.uint8)
    mix = soft.copy()
    mix[region] = sharp[region]  # a sharp patch in a soft scene
    out = match_blur(mix, region, ring)
    assert sharpness(out, region) < sharpness(mix, region)


def test_harmonize_changes_only_near_the_mask() -> None:
    gen, orig = np.full((64, 64, 3), 220, np.uint8), _img(3)
    m = np.zeros((64, 64), bool)
    m[24:40, 24:40] = True
    out, changed = harmonize(gen, orig, m, 2, seed=0)
    assert np.array_equal(out[~changed], orig[~changed])
    assert changed[m].all() and not changed[:8, :8].any()


# tracks ----------------------------------------------------------------------


def test_link_follows_and_splits() -> None:
    a = [(0.0, 0.0, 10.0, 10.0), (50.0, 0.0, 60.0, 10.0)]
    b = [(52.0, 1.0, 62.0, 11.0), (1.0, 0.0, 11.0, 10.0)]
    c = [(200.0, 0.0, 210.0, 10.0)]  # nothing continues: new track
    tracks = link([a, b, c], 0.3)
    assert sorted(tracks) == [[(0, 0), (1, 1)], [(0, 1), (1, 0)], [(2, 0)]]


def test_same_frame_duplicate_does_not_split_a_track() -> None:
    one = [(0.0, 0.0, 10.0, 10.0)]
    two = [(1.0, 0.0, 11.0, 10.0), (1.0, 0.0, 11.0, 10.5)]  # head twice
    tracks = link([one, two, one, one], 0.3)
    assert [(0, 0), (1, 0), (2, 0), (3, 0)] in tracks
    assert all((1, 1) not in t for t in tracks)


def test_link_scale_bridges_fast_motion() -> None:
    a, b = [(0.0, 0.0, 10.0, 10.0)], [(8.0, 0.0, 18.0, 10.0)]
    assert len(link([a, b], 0.3)) == 2  # IoU 0.11
    assert link([a, b], 0.3, 2.0) == [[(0, 0), (1, 0)]]
    assert grown((0.0, 0.0, 10.0, 10.0), 2.0) == (-5.0, -5.0, 15.0, 15.0)


def test_person_tracks_keep_frames_without_a_head() -> None:
    p = (0.0, 0.0, 10.0, 30.0)
    persons = [[p], [(1.0, 0.0, 11.0, 30.0)], [(2.0, 0.0, 12.0, 30.0)]]
    heads: list[list[tuple[float, float, float, float] | None]] = [
        [p],  # the head of the person in frame 0
        [None],  # an occupant, no person box
        [(2.0, 0.0, 12.0, 30.0)],
    ]
    tracks = person_tracks(persons, heads, 0.3)
    assert tracks == [[(0, 0), (1, NO_HEAD), (2, 0)]]
    only = person_tracks(persons, [[p], [None], [None]], 0.3)
    assert only == [[(0, 0)]]  # trailing frames without a head dropped


def test_head_tracks_link_occupants_on_their_own_boxes() -> None:
    p = (0.0, 0.0, 10.0, 30.0)
    occ = (100.0, 0.0, 110.0, 10.0)
    heads = [[(0.0, 0.0, 10.0, 8.0), occ], [(1.0, 0.0, 11.0, 8.0), occ]]
    persons = [[p], [(1.0, 0.0, 11.0, 30.0)]]
    head_persons: list[list[tuple[float, float, float, float] | None]] = [
        [p, None],
        [(1.0, 0.0, 11.0, 30.0), None],
    ]
    by_person = head_tracks(heads, persons, head_persons, "person", 0.3)
    assert sorted(by_person) == [[(0, 0), (1, 0)], [(0, 1), (1, 1)]]
    by_head = head_tracks(heads, persons, head_persons, "head", 0.3)
    assert sorted(by_head) == sorted(by_person)


def test_segments_merge_windows() -> None:
    seqs = {"s/C": [f"s/C/{i}" for i in range(20)]}
    segs = segments(seqs, ["s/C/5", "s/C/8", "s/C/18"], 2, 3)
    assert [ks for _, ks in segs] == [
        [f"s/C/{i}" for i in range(3, 12)],
        [f"s/C/{i}" for i in range(16, 20)],
    ]
    only = segments(seqs, ["s/C/5", "s/C/18"], 2, 3, {"s/C/5"})
    assert [ks for _, ks in only] == [
        [f"s/C/{i}" for i in range(3, 9)],
        ["s/C/18"],
    ]


# models (no weights loaded) ----------------------------------------------


def test_scrfd_decode_and_align() -> None:
    from blur_data.models.faceid import DET_SIZE, TEMPLATE, align, decode

    outs = []
    for width in (1, 4, 10):  # scores, boxes, landmarks per stride
        for s in (8, 16, 32):
            n = (DET_SIZE // s) ** 2 * 2
            outs.append(np.zeros((n, width), np.float32))
    n32 = DET_SIZE // 32
    j = 2 * (3 * n32 + 5)  # stride 32, cell (x=5, y=3), first anchor
    outs[2][j, 0] = 0.9
    outs[8][j] = np.tile([1.0, 2.0], 5)  # offsets in stride units
    found = decode(outs, 2.0)
    assert found is not None and abs(found[0] - 0.9) < 1e-6
    assert np.allclose(found[1][0], [(5 * 32 + 32) / 2, (3 * 32 + 64) / 2])
    img = np.zeros((112, 112, 3), np.uint8)
    img[50:54, 36:42] = 255  # a blob at the left eye of the template
    out = align(img, TEMPLATE.copy(), 112)
    assert out[51, 38].max() == 255  # identity warp keeps it in place


def test_image_encoder_is_loaded_at_the_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformers = pytest.importorskip("transformers")
    from blur_data.models import face_generator

    seen: dict[str, Any] = {}

    def fake(repo: str, **kw: Any) -> str:
        seen.update(kw, repo=repo)
        return "encoder"

    monkeypatch.setattr(
        transformers.CLIPVisionModelWithProjection, "from_pretrained", fake
    )
    cfg = GenConfig()
    assert face_generator.load_image_encoder(cfg) == "encoder"
    repo, rev = split_ref(cfg.image_encoder)
    assert seen["repo"] == repo and seen["revision"] == rev and rev
    assert seen["subfolder"] == "models/image_encoder"


# CLI ------------------------------------------------------------------------


def _script() -> Any:
    spec = importlib.util.spec_from_file_location("gen_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gen_cli"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_output_paths_of_keys_tracks_and_second_photos(tmp_path: Path) -> None:
    mod = _script()
    keys, nxt, seq = (tmp_path / n for n in ("k.json", "n.json", "s.json"))
    keys.write_text('{"s/C/1": {}, "s/C/2": {}}')  # an annotation file
    nxt.write_text('["s/C/2", "s/C/7"]')
    seq.write_text('{"tracks": [["s/C/3", "s/C/4"]]}')
    args = mod.argparse.Namespace(
        dataset="v2b", run="gen", seq=seq, keys=keys, next_keys=nxt,
        next_run="gen-next", limit=None, regions_run="r",
    )  # fmt: skip
    out = mod.output_paths(args)
    assert list(out) == ["s/C/3", "s/C/4", "s/C/1", "s/C/2", "s/C/7"]

    def runs(key: str) -> list[str]:
        return [p.relative_to(mod.OUT).parts[0] for p in out[key]]

    assert runs("s/C/2") == ["gen", "gen-next"] and runs("s/C/7") == [
        "gen-next"
    ]
    assert out["s/C/1"][0].name == "1.jpg"
