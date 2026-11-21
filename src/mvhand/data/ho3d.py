"""HO3D v3 five-camera reader that returns calibrated multi-view groups in the common input format."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import re
from typing import Mapping, Sequence

import cv2
import numpy as np
import torch
from torch import Tensor

from mvhand.geometry.mesh_joints import (
    POEM_PUBLISHED_MANO_21_REORDER,
    poem_published_mesh_joints_21,
)


PHYSICAL_TO_SEQUENCE_SUFFIX = (1, 0, 4, 3, 2)
# Form2Frame's production geometry contract is deliberately reference-slot
# invariant: runtime slot zero is the 3D target/reference camera.  The source
# sequence suffix order remains publisher metadata and must not leak into the
# runtime slot order.
RUNTIME_SLOT_TO_PHYSICAL = (0, 1, 4, 3, 2)
REFERENCE_PHYSICAL_CAMERA = 0
REFERENCE_RUNTIME_SLOT = RUNTIME_SLOT_TO_PHYSICAL.index(REFERENCE_PHYSICAL_CAMERA)
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0]).astype(np.float64)
DEFAULT_RUNTIME_VIEW_MASK = (True, True, True, True, True, False, False, False)


@dataclass(frozen=True)
class HO3DMVGroup:
    key: str
    rgb_crops: Tensor
    normalized_crops: Tensor
    raw_to_crop: Tensor
    crop_to_raw: Tensor
    intrinsics: Tensor
    camera_to_world: Tensor
    gt_joint_visibility: Tensor
    runtime_view_mask: Tensor
    camera_ids: tuple[str, ...]
    raw_size: Tensor
    label: Mapping[str, object]
    source_provenance: tuple[Mapping[str, object], ...]

    def to_form2frame_batch(self, *, device: torch.device | str = "cpu"):
        from mvhand.interfaces import Form2FrameBatch
        return Form2FrameBatch(
            crops=self.rgb_crops.unsqueeze(0).to(device),
            intrinsics=self.intrinsics.unsqueeze(0).to(device),
            camera_to_world=self.camera_to_world.unsqueeze(0).to(device),
            crop_to_raw=self.crop_to_raw.unsqueeze(0).to(device),
            runtime_view_mask=self.runtime_view_mask.unsqueeze(0).to(device),
        )


def load_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_localizer_ledger(path: Path) -> dict[tuple[str, int], dict]:
    rows = load_jsonl(path)
    result = {(row["key"], int(row["physical_camera_id"])): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate localizer ledger identity")
    return result


def parse_intrinsics(path: Path) -> np.ndarray:
    text = Path(path).read_text()
    fields = {key: float(value) for key, value in re.findall(r"(fx|fy|ppx|ppy):\s*([-+0-9.eE]+)", text)}
    if set(fields) != {"fx", "fy", "ppx", "ppy"}:
        raise ValueError(f"invalid HO3D intrinsic serialization: {path}")
    return np.array([[fields["fx"], 0.0, fields["ppx"]],
                     [0.0, fields["fy"], fields["ppy"]],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def load_reference_camera_transforms(calibration: Path) -> np.ndarray:
    order = tuple(int(value) for value in np.loadtxt(calibration / "cam_orders.txt").reshape(-1))
    if order != PHYSICAL_TO_SEQUENCE_SUFFIX:
        raise ValueError(f"unexpected cam_orders: {order}")
    transforms = np.stack([np.loadtxt(calibration / f"trans_{index}.txt") for index in range(5)])
    if transforms.shape != (5, 4, 4) or not np.isfinite(transforms).all():
        raise ValueError("invalid HO3D extrinsics")
    reference = np.linalg.inv(transforms[0])
    result = np.stack([reference @ transform for transform in transforms])
    if not np.allclose(result[0], np.eye(4), atol=1e-10, rtol=0):
        raise ValueError("physical camera 0 is not the reference")
    return result


def opengl_points_to_reference(points: np.ndarray, reference_from_camera: np.ndarray) -> np.ndarray:
    """Convert local HO3D OpenGL points to positive-depth OpenCV, then to reference.

    HO3D calibration matrices already act on OpenCV camera coordinates and
    therefore must not themselves be conjugated by the OpenGL basis change.
    """
    points = np.asarray(points, dtype=np.float64)
    local_cv = points @ OPENGL_TO_OPENCV.T
    homogeneous = np.concatenate([local_cv, np.ones((*local_cv.shape[:-1], 1))], axis=-1)
    return (homogeneous @ np.asarray(reference_from_camera).T)[..., :3]


def project_opencv(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if np.any(points[..., 2] <= 0):
        raise ValueError("canonical OpenCV projection requires positive depth")
    projected = points @ np.asarray(intrinsics).T
    return projected[..., :2] / projected[..., 2:3]


def apply_affine(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate([points, np.ones((*points.shape[:-1], 1))], axis=-1)
    return homogeneous @ np.asarray(affine, dtype=np.float64).T


def reconstruct_mano_vertices(metadata: Mapping[str, object], mano_layer) -> np.ndarray:
    pose = torch.as_tensor(np.asarray(metadata["handPose"]), dtype=torch.float32).reshape(1, 48)
    betas = torch.as_tensor(np.asarray(metadata["handBeta"]), dtype=torch.float32).reshape(1, 10)
    translation = torch.as_tensor(np.asarray(metadata["handTrans"]), dtype=torch.float32).reshape(1, 3)
    with torch.no_grad():
        output = mano_layer(
            global_orient=pose[:, :3], hand_pose=pose[:, 3:], betas=betas, transl=translation,
        )
    vertices = output.vertices[0].detach().cpu().numpy()
    if vertices.shape != (778, 3) or not np.isfinite(vertices).all():
        raise ValueError("HO3D MANO reconstruction failed")
    return vertices


def _crop(image: np.ndarray, raw_to_crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    patch = cv2.warpAffine(
        image, np.asarray(raw_to_crop, dtype=np.float32), (256, 256),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    )
    rgb = patch[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
    normalized = (rgb - (255.0 * np.array([0.485, 0.456, 0.406], dtype=np.float32))[:, None, None])
    normalized /= (255.0 * np.array([0.229, 0.224, 0.225], dtype=np.float32))[:, None, None]
    return rgb / 255.0, normalized


def decode_ho3d_group(
    row: Mapping[str, object], *, rgb_root: Path, metadata_root: Path,
    localizer: Mapping[tuple[str, int], Mapping[str, object]], mano_layer,
    mano_base_regressor: Tensor,
) -> HO3DMVGroup:
    if REFERENCE_RUNTIME_SLOT != 0:
        raise RuntimeError("Form2Frame requires the reference camera in runtime slot zero")
    views = list(row["views"])
    physical_ids = tuple(int(view["physical_camera_id"]) for view in views)
    suffixes = tuple(int(view["sequence_suffix"]) for view in views)
    if physical_ids != tuple(range(5)) or suffixes != PHYSICAL_TO_SEQUENCE_SUFFIX:
        raise ValueError("HO3D physical camera order mismatch")
    calibration = Path(metadata_root) / str(row["calibration_prefix"])
    reference_from_camera = load_reference_camera_transforms(calibration)

    rgb_crops = np.zeros((8, 3, 256, 256), dtype=np.float32)
    normalized = np.zeros_like(rgb_crops)
    raw_to_crop = np.zeros((8, 2, 3), dtype=np.float32)
    crop_to_raw = np.zeros((8, 2, 3), dtype=np.float32)
    raw_to_crop[5:, :2, :2] = np.eye(2, dtype=np.float32)
    crop_to_raw[5:, :2, :2] = np.eye(2, dtype=np.float32)
    intrinsics = np.tile(np.eye(3, dtype=np.float32), (8, 1, 1))
    camera_to_world = np.tile(np.eye(4, dtype=np.float32), (8, 1, 1))
    runtime_mask = np.zeros(8, dtype=bool)
    raw_size = np.zeros((8, 2), dtype=np.int64); raw_size[:5] = (640, 480)
    vertices_reference = []
    joints_reference = []
    joints_2d = np.zeros((8, 21, 2), dtype=np.float32)
    provenance = []

    for physical_id, view in enumerate(views):
        runtime_slot = RUNTIME_SLOT_TO_PHYSICAL.index(physical_id)
        metadata_path = Path(metadata_root) / str(view["metadata_member"])
        with metadata_path.open("rb") as stream:
            metadata = pickle.load(stream)
        K = np.asarray(metadata["camMat"], dtype=np.float64)
        suffix = int(view["sequence_suffix"])
        # Calibration files are indexed by physical-camera position; cam_orders
        # maps that position to the camera-specific sequence suffix.
        K_file = parse_intrinsics(calibration / f"cam_{physical_id}_intrinsics.txt")
        if not np.allclose(K, K_file, atol=1e-3, rtol=0):
            raise ValueError("metadata/calibration intrinsic mismatch")
        intrinsics[runtime_slot] = K
        camera_to_world[runtime_slot] = reference_from_camera[physical_id]
        native_joints = np.asarray(metadata["handJoints3D"], dtype=np.float64)
        local_cv_joints = native_joints @ OPENGL_TO_OPENCV.T
        projected_raw = project_opencv(local_cv_joints, K)
        joints_reference.append(opengl_points_to_reference(native_joints, reference_from_camera[physical_id]))
        vertices_gl = reconstruct_mano_vertices(metadata, mano_layer)
        vertices_reference.append(opengl_points_to_reference(vertices_gl, reference_from_camera[physical_id]))

        localization = localizer[(str(row["key"]), physical_id)]
        localized = localization["localization_status"] == "LOCALIZED"
        runtime_mask[runtime_slot] = localized
        if localized:
            affine = np.asarray(localization["raw_to_crop"], dtype=np.float32)
            inverse = np.asarray(localization["crop_to_raw"], dtype=np.float32)
            image = cv2.imread(str(Path(rgb_root) / str(view["rgb_member"])), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"failed to decode {view['rgb_member']}")
            rgb_crops[runtime_slot], normalized[runtime_slot] = _crop(image, affine)
            raw_to_crop[runtime_slot], crop_to_raw[runtime_slot] = affine, inverse
            joints_2d[runtime_slot] = apply_affine(projected_raw, affine)
        provenance.append({
            "physical_camera_id": physical_id,
            "runtime_slot": runtime_slot,
            "sequence_suffix": suffix,
            "rgb_member": view["rgb_member"],
            "metadata_member": view["metadata_member"],
            "localization_status": localization["localization_status"],
        })

    provenance.sort(key=lambda item: int(item["runtime_slot"]))
    if not runtime_mask[REFERENCE_RUNTIME_SLOT] or int(runtime_mask[:5].sum()) < 2:
        raise ValueError("DETECTOR_INVALID_GROUP")
    if not np.allclose(camera_to_world[0], np.eye(4), atol=1e-7, rtol=0):
        raise ValueError("HO3D runtime slot zero is not physical camera zero/reference")
    vertices_stack = np.stack(vertices_reference)
    joints_stack = np.stack(joints_reference)
    gt_vertices = vertices_stack.mean(axis=0).astype(np.float32)
    native_dataset_joints = joints_stack.mean(axis=0).astype(np.float32)
    # HO3D stores the augmented MANO sequence (16 base joints followed by five
    # tips). Form2Frame observations use the published POEM/OpenPose ordering.
    dataset_joints = native_dataset_joints[list(POEM_PUBLISHED_MANO_21_REORDER)]
    regressor = mano_base_regressor.detach().cpu().to(torch.float32)
    gt_mesh_joints = poem_published_mesh_joints_21(torch.from_numpy(gt_vertices), regressor).numpy()
    # HO3D exposes no verified per-joint visibility. All-false is an explicit
    # absence sentinel required by the legacy collator; visibility-dependent
    # losses are disabled by the frozen HO3D label contract.
    gt_joint_visibility = np.zeros((8, 21), dtype=bool)
    label = {
        "joints_2d": joints_2d[:, list(POEM_PUBLISHED_MANO_21_REORDER)],
        "joints_2d_space": "crop",
        "reference_physical_camera": REFERENCE_PHYSICAL_CAMERA,
        "reference_runtime_slot": REFERENCE_RUNTIME_SLOT,
        "joints_3d": dataset_joints,
        "verts_3d": gt_vertices,
        "dataset_joints_3d": dataset_joints,
        "native_ho3d_joints_3d": native_dataset_joints,
        "gt_mesh_joints_3d": gt_mesh_joints,
        "per_view_vertices_reference": vertices_stack.astype(np.float32),
        "per_view_dataset_joints_reference": joints_stack.astype(np.float32),
        "visibility_supervision_available": False,
        "headline_joint_endpoint": "matched POEM mesh-derived joints",
    }
    return HO3DMVGroup(
        str(row["key"]), torch.from_numpy(rgb_crops), torch.from_numpy(normalized),
        torch.from_numpy(raw_to_crop), torch.from_numpy(crop_to_raw),
        torch.from_numpy(intrinsics), torch.from_numpy(camera_to_world),
        torch.from_numpy(gt_joint_visibility), torch.from_numpy(runtime_mask),
        tuple(
            [f"physical_camera_{i}" for i in RUNTIME_SLOT_TO_PHYSICAL]
            + ["unused_5", "unused_6", "unused_7"]
        ),
        torch.from_numpy(raw_size), label, tuple(provenance),
    )
