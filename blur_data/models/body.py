"""SAM 3D Body: one MHR mesh per person box, in the camera frame.

``third_party/sam-3d-body`` ships no package metadata, so its root is
put on ``sys.path`` here (its dependencies are in ``pyproject.toml``).
Boxes come from SAM3 and the intrinsics from the v2b calibration, so
neither the repo's detector nor its FOV estimator is used.
"""

from __future__ import annotations

import contextlib
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

REPO = Path(__file__).resolve().parents[2]
SAM3D_BODY_ROOT = REPO / "third_party" / "sam-3d-body"

F64 = npt.NDArray[np.float64]

#: MHR joints whose skinned vertices form the head (skull, face, jaw,
#: eyes, mouth); neck and shoulders are left out on purpose
HEAD_JOINTS = (
    "c_head",
    "c_head_null",
    "c_jaw",
    "c_jaw_null",
    "c_teeth",
    "c_tongue0",
    "c_tongue1",
    "c_tongue2",
    "c_tongue3",
    "c_tongue4",
    "l_eye",
    "l_eye_null",
    "r_eye",
    "r_eye_null",
)


def head_vertex_ids(mhr_path: Path) -> npt.NDArray[np.int64]:
    """Mesh vertices whose strongest skinning joint is a head joint."""
    mhr = torch.jit.load(  # type: ignore[no-untyped-call]
        str(mhr_path), map_location="cpu"
    )
    character = mhr.character_torch
    names = list(character.skeleton.joint_names)
    lbs = character.linear_blend_skinning
    verts = lbs.vert_indices_flattened.numpy()
    joints = lbs.skin_indices_flattened.numpy()
    weights = lbs.skin_weights_flattened.numpy()
    w = np.zeros((int(verts.max()) + 1, len(names)), np.float32)
    np.add.at(w, (verts, joints), weights)
    head = [names.index(j) for j in HEAD_JOINTS if j in names]
    ids: npt.NDArray[np.int64] = np.nonzero(np.isin(w.argmax(axis=1), head))[
        0
    ].astype(np.int64)
    return ids


@dataclass
class BodyFit:
    """Mesh and keypoints of one person, camera frame, metres."""

    box: tuple[float, float, float, float]
    vertices: F64  # (18439, 3), pred_vertices + pred_cam_t
    keypoints: F64  # (70, 3), pred_keypoints_3d + pred_cam_t
    keypoints_2d: F64  # (70, 2), the model's own projection


class BodyFitter:
    """Batched SAM 3D Body inference for given person boxes."""

    def __init__(self, checkpoint_dir: Path, device: str = "cuda") -> None:
        if str(SAM3D_BODY_ROOT) not in sys.path:
            sys.path.insert(0, str(SAM3D_BODY_ROOT))
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

        with contextlib.redirect_stdout(io.StringIO()):
            model, cfg = load_sam_3d_body(
                str(checkpoint_dir / "model.ckpt"),
                device=device,
                mhr_path=str(checkpoint_dir / "assets" / "mhr_model.pt"),
            )
            self.estimator = SAM3DBodyEstimator(
                sam_3d_body_model=model, model_cfg=cfg
            )
        self.faces: npt.NDArray[np.int64] = np.asarray(self.estimator.faces)
        self.head_ids = head_vertex_ids(
            checkpoint_dir / "assets" / "mhr_model.pt"
        )

    def fit(
        self,
        img_rgb: npt.NDArray[np.uint8],
        boxes: list[tuple[float, float, float, float]],
        k: F64,
    ) -> list[BodyFit]:
        """One fit per box, same order as ``boxes``."""
        if not boxes:
            return []
        cam = torch.as_tensor(k, dtype=torch.float32)[None]
        with contextlib.redirect_stdout(io.StringIO()):
            outs = self.estimator.process_one_image(
                img_rgb,
                bboxes=np.asarray(boxes, dtype=np.float32),
                cam_int=cam,
                inference_type="body",
            )
        if len(outs) != len(boxes):
            raise RuntimeError(
                f"SAM 3D Body returned {len(outs)} fits for {len(boxes)} boxes"
            )
        fits = []
        for box, o in zip(boxes, outs, strict=True):
            t = np.asarray(o["pred_cam_t"], dtype=np.float64)
            fits.append(
                BodyFit(
                    box=box,
                    vertices=np.asarray(o["pred_vertices"], np.float64) + t,
                    keypoints=np.asarray(o["pred_keypoints_3d"], np.float64)
                    + t,
                    keypoints_2d=np.asarray(
                        o["pred_keypoints_2d"], np.float64
                    ),
                )
            )
        return fits
