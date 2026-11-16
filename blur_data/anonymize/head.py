"""Head geometry of a SAM 3D Body mesh in the camera frame.

All inputs are in the OpenCV camera frame (x right, y down, z forward,
metres): ``pred_vertices + pred_cam_t`` and
``pred_keypoints_3d + pred_cam_t`` of a SAM 3D Body output. Keypoint
indices follow ``sam_3d_body/metadata/mhr70.py``.

*Facing*: the head's forward axis is the nose seen from the midpoint of
the ears, made orthogonal to the neck->ears axis. The head faces the
camera when the angle between that axis and the ray head->camera is
below ``max_angle_deg``; 90 deg is an exact profile, so a threshold
above 90 also blurs profiles and three-quarter rear views.

*Head region*: the mesh vertices skinned to the head joints
(``blur_data.models.body.head_vertex_ids``: skull, face, jaw, eyes; not
neck or shoulders), projected with K; the blurred area is their convex
hull, dilated by a small margin.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from blur_data.utils.masks import BoolMask, dilate

F64 = npt.NDArray[np.float64]

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
NECK = 69


@dataclass(frozen=True)
class HeadPose:
    """Head centre, axes and the facing angle to the camera."""

    center: F64
    up: F64
    forward: F64
    angle_deg: float  # 0 = looks straight into the camera, 180 = away


def _unit(v: F64) -> F64:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v * 0.0


def head_pose(kp_cam: F64) -> HeadPose:
    """Facing angle from the 70 camera-frame keypoints."""
    face = kp_cam[[NOSE, L_EYE, R_EYE, L_EAR, R_EAR]]
    center = face.mean(axis=0)
    ears_mid = 0.5 * (kp_cam[L_EAR] + kp_cam[R_EAR])
    # ears, not the face centroid: the face sits in front of the neck,
    # which would tip "up" forward and "forward" down by ~15 deg
    up = _unit(ears_mid - kp_cam[NECK])
    fwd = kp_cam[NOSE] - ears_mid
    fwd = _unit(fwd - np.dot(fwd, up) * up)
    to_cam = _unit(-center)
    cos = float(np.clip(np.dot(fwd, to_cam), -1.0, 1.0))
    return HeadPose(center, up, fwd, float(np.degrees(np.arccos(cos))))


def project(points: F64, k: F64, min_z: float = 0.05) -> F64:
    """Pinhole projection of the points in front of the camera."""
    pts = points[points[:, 2] > min_z]
    uv = (k @ pts.T).T
    out: F64 = uv[:, :2] / uv[:, 2:3]
    return out


def hull_mask(
    uv: F64, shape: tuple[int, int], margin: float, min_px: int
) -> BoolMask | None:
    """Filled, dilated convex hull of 2D points; None if off-image.

    The dilation radius is ``margin`` times the larger hull side, at
    least ``min_px`` pixels.
    """
    h, w = shape
    if len(uv) < 3:
        return None
    lim = 4.0 * max(h, w)  # keep cv2 happy with far-off points
    pts = np.clip(np.round(uv), -lim, lim).astype(np.int32)
    hull = cv2.convexHull(pts)
    canvas = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(canvas, hull, (1,))
    mask = canvas.astype(bool)
    if not mask.any():
        return None
    span = pts.max(axis=0) - pts.min(axis=0)
    radius = max(min_px, int(round(margin * float(span.max()))))
    return dilate(mask, radius)
