"""Pose guide of a generated head: template head depth + OpenPose map.

Region files keep no mesh, so every generated ``head`` is fitted again
with SAM 3D Body on its stored ``meta.person_box`` and the frame's K
(PP4AV: SAM 3D Body's default K). Two ControlNet inputs are drawn from
the fit; the person's own head geometry is never rendered (CIAGAN:
landmarks alone identify 30.5 %):

* **depth**: a template head, the MHR rest mesh (mean shape, neutral
  expression), placed at the fitted head's pose by a similarity
  (Umeyama: rotation, translation, uniform scale) of its head vertices
  onto the fitted ones (MHR topology is shared, so the vertices
  correspond one to one), rasterised with K into the crop (a small
  numpy z-buffer);
* **pose**: the sparse OpenPose map (nose, eyes, ears, neck, shoulders)
  of the person's own keypoints; face points turned away are hidden.

A head without a usable fit (no person box, a failed or implausible
fit) is generated without the pose guide. Everything is in the OpenCV
camera frame (x right, y down, z forward, metres) and in the crop of
:func:`blur_data.generate.crops.square_window`, at the model
resolution.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from blur_data.anonymize.head import F64, head_pose
from blur_data.anonymize.regions import Region
from blur_data.generate.config import GenConfig
from blur_data.generate.routing import is_generated

U8 = npt.NDArray[np.uint8]
F32 = npt.NDArray[np.float32]
Window = tuple[int, int, int]  # x0, y0, side

#: OpenPose-18 index -> MHR70 keypoint (only face and upper body)
OPENPOSE_FROM_MHR = {0: 0, 1: 69, 2: 6, 5: 5, 14: 2, 15: 1, 16: 4, 17: 3}
#: OpenPose limbs (pairs of OpenPose-18 indices) with their colour index
LIMBS: tuple[tuple[int, int, int], ...] = (
    (1, 2, 0), (1, 5, 1), (1, 0, 12), (0, 14, 13), (14, 16, 14),
    (0, 15, 15), (15, 17, 16),
)  # fmt: skip
#: the 18 OpenPose colours (RGB), as in controlnet_aux
COLOURS = (
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170), (255, 0, 85),
)  # fmt: skip
#: face keypoints hidden when they point this far away from the camera
HIDDEN_COS = -0.3


@dataclass(frozen=True)
class Guide:
    """ControlNet inputs of one head at the model resolution (RGB)."""

    depth: U8
    pose: U8


@dataclass(frozen=True)
class TemplateHead:
    """The MHR rest head: vertex ids, vertices (m) and local faces."""

    ids: npt.NDArray[np.int64]
    verts: F64
    faces: npt.NDArray[np.int64]


def head_faces(
    faces: npt.NDArray[np.int64], ids: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """Triangles with all corners in ``ids``, re-indexed into ``ids``."""
    local = np.full(int(faces.max()) + 1, -1, np.int64)
    local[ids] = np.arange(len(ids))
    tri = local[faces]
    out: npt.NDArray[np.int64] = tri[(tri >= 0).all(axis=1)]
    return out


def load_template(mhr_path: Path) -> TemplateHead:
    """Template head from SAM 3D Body's MHR model (rest pose, mean shape)."""
    import torch

    from blur_data.models.body import head_vertex_ids

    mhr = torch.jit.load(  # type: ignore[no-untyped-call]
        str(mhr_path), map_location="cpu"
    )
    mesh = mhr.character_torch.mesh
    verts = mesh.rest_vertices.numpy().astype(np.float64) / 100.0  # cm
    faces = mesh.faces.numpy().astype(np.int64)
    ids = head_vertex_ids(mhr_path)
    return TemplateHead(ids, verts[ids], head_faces(faces, ids))


def similarity(src: F64, dst: F64) -> tuple[float, F64, F64]:
    """Umeyama: ``s, r, t`` minimising ``|s r src + t - dst|``."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(b.T @ a / len(src))
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt))
    r = u @ d @ vt
    var = float((a**2).sum() / len(src))
    s = float(np.trace(np.diag(sig) @ d) / var)
    return s, r, mu_d - s * r @ mu_s


def to_crop(uv: F64, window: Window, res: int) -> F64:
    """Image pixels -> pixels of the crop resized to ``res``."""
    x0, y0, side = window
    out: F64 = (uv - np.array([x0, y0], np.float64)) * (res / side)
    return out


def rasterize_depth(
    verts: F64, faces: npt.NDArray[np.int64], k: F64, window: Window, res: int
) -> F32:
    """Z-buffer depth (m) of a mesh in the crop; inf where empty."""
    zbuf = np.full((res, res), np.inf, np.float32)
    front = verts[:, 2] > 0.05
    uvw = (k @ verts.T).T
    uv = to_crop(uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-6), window, res)
    for tri in faces:
        if not front[tri].all():
            continue
        _raster_triangle(zbuf, uv[tri], verts[tri, 2])
    return zbuf


def _raster_triangle(zbuf: F32, p: F64, z: F64) -> None:
    res = zbuf.shape[0]
    x0, y0 = np.floor(p.min(axis=0)).astype(int)
    x1, y1 = np.ceil(p.max(axis=0)).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, res - 1), min(y1, res - 1)
    if x1 < x0 or y1 < y0:
        return
    den = (p[1, 1] - p[2, 1]) * (p[0, 0] - p[2, 0]) + (p[2, 0] - p[1, 0]) * (
        p[0, 1] - p[2, 1]
    )
    if abs(den) < 1e-12:
        return
    xs, ys = np.meshgrid(
        np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5
    )
    w0 = (
        (p[1, 1] - p[2, 1]) * (xs - p[2, 0])
        + (p[2, 0] - p[1, 0]) * (ys - p[2, 1])
    ) / den
    w1 = (
        (p[2, 1] - p[0, 1]) * (xs - p[2, 0])
        + (p[0, 0] - p[2, 0]) * (ys - p[2, 1])
    ) / den
    w2 = 1.0 - w0 - w1
    inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    if not inside.any():
        return
    depth = (w0 * z[0] + w1 * z[1] + w2 * z[2]).astype(np.float32)
    win = zbuf[y0 : y1 + 1, x0 : x1 + 1]
    better = inside & (depth < win)
    win[better] = depth[better]


def depth_image(zbuf: F32, span: float = 0.6) -> U8:
    """Depth-ControlNet input: near = bright, the head's front 255,
    ``span`` metres behind it 0, empty pixels black (RGB)."""
    out = np.zeros(zbuf.shape, np.float32)
    hit = np.isfinite(zbuf)
    if hit.any():
        near = float(zbuf[hit].min())
        out[hit] = np.clip(1.0 - (zbuf[hit] - near) / span, 0.0, 1.0)
        out[hit] = 0.25 + 0.75 * out[hit]  # keep the head off black
    gray = np.round(out * 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def openpose_points(kp3d: F64, kp2d: F64) -> dict[int, tuple[float, float]]:
    """Visible OpenPose-18 points (image px) from the 70 MHR keypoints.

    Nose, eyes and ears pointing away from the camera (seen from the
    middle of the ears) are hidden, so a back view draws no face.
    """
    ears = 0.5 * (kp3d[3] + kp3d[4])
    to_cam = -ears / max(1e-9, float(np.linalg.norm(ears)))
    out = {}
    for op, mhr in OPENPOSE_FROM_MHR.items():
        if op in (0, 14, 15, 16, 17):
            v = kp3d[mhr] - ears
            n = float(np.linalg.norm(v))
            if n > 1e-9 and float(np.dot(v / n, to_cam)) < HIDDEN_COS:
                continue
        out[op] = (float(kp2d[mhr, 0]), float(kp2d[mhr, 1]))
    return out


def openpose_image(
    points: dict[int, tuple[float, float]], window: Window, res: int
) -> U8:
    """OpenPose-format map of the points in the crop (black canvas)."""
    canvas = np.zeros((res, res, 3), np.uint8)
    crop = {
        i: to_crop(np.array([p], np.float64), window, res)[0]
        for i, p in points.items()
    }
    stick = max(2, round(4 * res / 512))
    limbs = canvas.copy()
    for a, b, c in LIMBS:
        if a in crop and b in crop:
            pa, pb = crop[a], crop[b]
            mid = (pa + pb) / 2
            length = float(np.linalg.norm(pa - pb)) / 2
            angle = float(np.degrees(np.arctan2(pb[1] - pa[1], pb[0] - pa[0])))
            poly = cv2.ellipse2Poly(
                (int(mid[0]), int(mid[1])), (int(length), stick),
                int(angle), 0, 360, 1,
            )  # fmt: skip
            pts = np.asarray(poly, np.int32)
            cv2.fillConvexPoly(limbs, pts, COLOURS[c])
    canvas = (0.4 * canvas + 0.6 * limbs).astype(np.uint8)
    for i, p in crop.items():
        cv2.circle(canvas, (int(p[0]), int(p[1])), stick, COLOURS[i], -1)
    return canvas


@dataclass(frozen=True)
class HeadFit:
    """What is kept of one body fit: the template head's placement, the
    visible OpenPose points and the head's pose (never the mesh)."""

    scale: float
    rot: F64
    trans: F64
    points: dict[int, tuple[float, float]]
    angle_deg: float
    up: F64
    forward: F64


def head_fit(
    template: TemplateHead, verts: F64, kp3d: F64, kp2d: F64
) -> HeadFit:
    """Place the template on a fit's head vertices; keep its pose."""
    s, r, t = similarity(template.verts, verts[template.ids])
    pose = head_pose(kp3d)
    return HeadFit(
        s,
        r,
        t,
        openpose_points(kp3d, kp2d),
        pose.angle_deg,
        pose.up,
        pose.forward,
    )


def placed(template: TemplateHead, fit: HeadFit) -> F64:
    """Template head vertices at the fitted pose (camera frame)."""
    out: F64 = fit.scale * (fit.rot @ template.verts.T).T + fit.trans
    return out


def guide_maps(
    template: TemplateHead, fit: HeadFit, k: F64, window: Window, res: int
) -> tuple[U8, U8]:
    """Depth and OpenPose ControlNet inputs of one head in a crop."""
    zbuf = rasterize_depth(
        placed(template, fit), template.faces, k, window, res
    )
    return depth_image(zbuf), openpose_image(fit.points, window, res)


def head_rotation(up: F64, forward: F64) -> F64:
    """Head frame (columns right, up, forward) from its two axes."""
    right = np.cross(up, forward)
    out: F64 = np.stack([right, up, forward], axis=1)
    return out


def rotation_deg(a: F64, b: F64) -> float:
    """Geodesic angle between two rotation matrices, degrees."""
    cos = (np.trace(a.T @ b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _inside(uv: F64, box: tuple[float, ...], pad: float) -> float:
    """Share of the points inside ``box`` grown by ``pad`` per side."""
    bw, bh = box[2] - box[0], box[3] - box[1]
    x0, y0 = box[0] - pad * bw, box[1] - pad * bh
    x1, y1 = box[2] + pad * bw, box[3] + pad * bh
    ok = (uv[:, 0] >= x0) & (uv[:, 0] <= x1)
    ok &= (uv[:, 1] >= y0) & (uv[:, 1] <= y1)
    return float(ok.mean()) if len(uv) else 0.0


def plausible(
    template: TemplateHead,
    fit: HeadFit,
    k: F64,
    person_box: tuple[float, ...],
    region_box: tuple[float, ...],
) -> bool:
    """The placed template is in front of the camera and at least half
    of it projects into the (grown) person box and head region."""
    verts = placed(template, fit)
    if (verts[:, 2] <= 0.05).any():
        return False
    uvw = (k @ verts.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    return (
        _inside(uv, person_box, 0.15) >= 0.5
        and _inside(uv, region_box, 0.5) >= 0.5
    )


class FitGuides:
    """The guides of one image's fits, by region index; a region
    without a usable fit gets none."""

    def __init__(
        self, template: TemplateHead, fits: dict[int, HeadFit], k: F64
    ) -> None:
        self.template, self.fits, self.k = template, fits, k

    def __call__(self, index: int, window: Window, res: int) -> Guide | None:
        fit = self.fits.get(index)
        if fit is None:
            return None
        depth, pose = guide_maps(self.template, fit, self.k, window, res)
        return Guide(depth, pose)


#: why a generated head has no pose guide
NO_BOX, FIT_FAILED, IMPLAUSIBLE = "no_person_box", "fit_failed", "implausible"


class HeadGuider:
    """Fits the generated heads of an image with SAM 3D Body (GPU)."""

    def __init__(self, checkpoint_dir: Path, device: str = "cuda") -> None:
        from blur_data.models.body import BodyFitter

        self.body = BodyFitter(checkpoint_dir, device)
        self.template: TemplateHead = load_template(
            checkpoint_dir / "assets" / "mhr_model.pt"
        )

    def fits(
        self,
        img: U8,
        regions: Sequence[Region],
        k: F64,
        cfg: GenConfig,
    ) -> dict[int, HeadFit | str]:
        """Per generated region (index in ``regions``): a fit, or the
        reason why there is none (:data:`NO_BOX`, :data:`FIT_FAILED`,
        :data:`IMPLAUSIBLE`)."""
        shape = (int(img.shape[0]), int(img.shape[1]))
        out: dict[int, HeadFit | str] = {}
        todo: list[tuple[int, tuple[float, float, float, float]]] = []
        for i, r in enumerate(regions):
            if not is_generated(r, cfg, shape):
                continue
            pb = r.meta.get("person_box")
            if r.kind != "head" or pb is None:
                out[i] = NO_BOX
            else:
                todo.append((i, (pb[0], pb[1], pb[2], pb[3])))
        if not todo:
            return out
        try:
            fits = self.body.fit(img, [b for _, b in todo], k)
        except (RuntimeError, ValueError, IndexError):
            return out | {i: FIT_FAILED for i, _ in todo}
        for (i, box), f in zip(todo, fits, strict=True):
            hf = head_fit(
                self.template, f.vertices, f.keypoints, f.keypoints_2d
            )
            ok = plausible(self.template, hf, k, box, regions[i].box)
            out[i] = hf if ok else IMPLAUSIBLE
        return out
