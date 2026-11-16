"""Facing angle and head projection on a synthetic skeleton."""

import numpy as np

from blur_data.anonymize import head

K = np.array([[1000.0, 0, 500], [0, 1000.0, 400], [0, 0, 1]])


def skeleton(yaw_deg: float, z: float = 5.0) -> head.F64:
    """70 keypoints; the face looks at the camera for yaw 0.

    Camera frame: y down, so "up" is -y. The nose points to -z (to the
    camera) and the head is rotated about the vertical axis by yaw.
    """
    kp = np.zeros((70, 3))
    yaw = np.radians(yaw_deg)
    fwd = np.array([np.sin(yaw), 0.0, -np.cos(yaw)])
    side = np.array([np.cos(yaw), 0.0, np.sin(yaw)])
    c = np.array([0.0, 0.0, z])  # on the optical axis
    kp[head.NOSE] = c + 0.1 * fwd
    kp[head.L_EYE] = c + 0.08 * fwd + 0.03 * side - [0, 0.03, 0]
    kp[head.R_EYE] = c + 0.08 * fwd - 0.03 * side - [0, 0.03, 0]
    kp[head.L_EAR] = c + 0.07 * side
    kp[head.R_EAR] = c - 0.07 * side
    kp[head.NECK] = c + [0, 0.2, 0]
    return kp


def test_facing_angles() -> None:
    assert head.head_pose(skeleton(0)).angle_deg < 5
    assert 85 < head.head_pose(skeleton(90)).angle_deg < 95
    assert head.head_pose(skeleton(180)).angle_deg > 170


def test_head_hull() -> None:
    kp = skeleton(0)
    pose = head.head_pose(kp)
    rng = np.random.default_rng(0)
    hv = pose.center + 0.1 * rng.normal(size=(500, 3))
    mask = head.hull_mask(head.project(hv, K), (800, 1000), 0.05, 3)
    assert mask is not None
    u, v = head.project(pose.center[None], K)[0]
    assert mask[int(v), int(u)]
    ys, _ = np.nonzero(mask)
    neck_v = head.project(kp[head.NECK][None] + [0, 0.3, 0], K)[0, 1]
    assert ys.max() < neck_v


def test_hull_off_image() -> None:
    uv = np.array([[-500.0, -500], [-400, -500], [-450, -400]])
    assert head.hull_mask(uv, (100, 100), 0.1, 1) is None
