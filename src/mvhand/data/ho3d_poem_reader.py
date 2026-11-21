"""Reader for the fixed-revision POEM HO3D multi-view WebDataset shards.

This is deliberately distinct from the official HO3D-v3 directory adapter.
The WDS labels contain raw 640x480 RGB, five camera intrinsics/extrinsics,
synchronized sequence/frame identities, and metric mesh/joint supervision.
Detector crops still come exclusively from the frozen localizer ledger.
"""

from __future__ import annotations

import io
from pathlib import Path
import pickle
import re
from typing import Mapping

import cv2
import numpy as np
import torch
from torch import Tensor

from mvhand.data.ho3d import (
    HO3DMVGroup,
    PHYSICAL_TO_SEQUENCE_SUFFIX,
    REFERENCE_PHYSICAL_CAMERA,
    REFERENCE_RUNTIME_SLOT,
    RUNTIME_SLOT_TO_PHYSICAL,
    _crop,
    apply_affine,
)
from mvhand.geometry.mesh_joints import poem_published_mesh_joints_21


ADAPTER_KIND = "JUBSTEVEN_POEM_V2_HO3D_MV_WDS"
ADAPTER_VERSION = "d6f16d9c28652e2916624256734a89159f5ccf39-v3-reference-slot0-crop1x"
SOURCE_REVISION = "d6f16d9c28652e2916624256734a89159f5ccf39"
SUPERVISION_TARGET_POLICY = "REFERENCE_PHYSICAL_CAMERA_0_LABEL_V1"
CROSS_VIEW_ALIGNMENT_TOLERANCE_M = 5e-5
EXPECTED_LABEL_FIELDS = {
    "sample_idx", "cam_extr", "cam_serial", "idx", "bbox_center", "bbox_scale",
    "cam_intr", "joints_2d", "joints_3d", "verts_3d", "joints_vis",
    "mano_pose", "mano_shape", "image_path", "raw_size",
}
IMAGE_PATH = re.compile(
    r"^data/HO3D_v3/(?:train|evaluation)/([^/]+)/rgb/([0-9]+)\.jpg$"
)


class _RestrictedNumpyUnpickler(pickle.Unpickler):
    """Decode the publisher's numpy-only payload without arbitrary globals."""

    def find_class(self, module: str, name: str):
        allowed = {
            ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
            ("numpy._core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
            ("numpy.core.multiarray", "scalar"): np.core.multiarray.scalar,
            ("numpy._core.multiarray", "scalar"): np.core.multiarray.scalar,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
        }
        try:
            return allowed[(module, name)]
        except KeyError as error:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}") from error


def decode_label_bytes(payload: bytes) -> dict:
    value = _RestrictedNumpyUnpickler(io.BytesIO(payload)).load()
    if not isinstance(value, dict):
        raise ValueError("HO3D WDS label must be a mapping")
    return value


def _array_list(label: Mapping[str, object], name: str, shape: tuple[int, ...]) -> list[np.ndarray]:
    values = label[name]
    if not isinstance(values, (list, tuple)) or len(values) != 5:
        raise ValueError(f"{name} must contain exactly five views")
    result = [np.asarray(value) for value in values]
    if any(value.shape != shape for value in result):
        raise ValueError(f"{name} has an invalid per-view shape")
    if any(value.dtype.kind not in "fiu" or not np.isfinite(value).all() for value in result):
        raise ValueError(f"{name} contains invalid values")
    return result


def validate_wds_label(label: Mapping[str, object]) -> dict:
    missing = EXPECTED_LABEL_FIELDS - set(label)
    if missing:
        raise ValueError(f"HO3D WDS label misses {sorted(missing)}")
    serials = tuple(str(value) for value in label["cam_serial"])
    paths = tuple(str(value) for value in label["image_path"])
    if len(serials) != 5 or len(paths) != 5 or len(set(serials)) != 5:
        raise ValueError("HO3D WDS camera identity must contain five unique views")
    parsed = []
    for serial, path in zip(serials, paths):
        match = IMAGE_PATH.fullmatch(path)
        if match is None or match.group(1) != serial:
            raise ValueError("HO3D WDS image path/camera identity mismatch")
        if not serial[-1:].isdigit():
            raise ValueError("HO3D WDS sequence suffix is not numeric")
        parsed.append((serial[:-1], int(serial[-1]), match.group(2)))
    bases = {value[0] for value in parsed}
    frames = {value[2] for value in parsed}
    suffixes = {value[1] for value in parsed}
    if len(bases) != 1 or len(frames) != 1 or suffixes != set(PHYSICAL_TO_SEQUENCE_SUFFIX):
        raise ValueError("HO3D WDS views are not a synchronized five-camera group")

    extrinsics = _array_list(label, "cam_extr", (4, 4))
    intrinsics = _array_list(label, "cam_intr", (3, 3))
    joints_2d = _array_list(label, "joints_2d", (21, 2))
    joints_3d = _array_list(label, "joints_3d", (21, 3))
    vertices = _array_list(label, "verts_3d", (778, 3))
    _array_list(label, "joints_vis", (21,))
    _array_list(label, "mano_pose", (48,))
    _array_list(label, "mano_shape", (10,))
    raw_size = _array_list(label, "raw_size", (2,))
    if any(tuple(int(item) for item in value) != (640, 480) for value in raw_size):
        raise ValueError("HO3D WDS raw image size changed")
    if any(not np.allclose(value[3], (0, 0, 0, 1), atol=1e-6, rtol=0) for value in extrinsics):
        raise ValueError("HO3D WDS camera extrinsic is not homogeneous")
    if any(abs(np.linalg.det(value[:3, :3]) - 1.0) > 1e-4 for value in extrinsics):
        raise ValueError("HO3D WDS camera rotation is invalid")
    reprojection_max_px = 0.0
    for K, xyz, uv in zip(intrinsics, joints_3d, joints_2d):
        if np.any(xyz[:, 2] <= 0):
            raise ValueError("HO3D WDS joints must have positive OpenCV depth")
        projected = xyz @ K.T
        projected = projected[:, :2] / projected[:, 2:]
        # Publisher arrays are independently serialized as IEEE float32.  A
        # full 9,087-group audit bounds the resulting reprojection residual at
        # 0.008082 pixels; 0.01 px remains a strict fail-closed margin.
        reprojection_max_px = max(reprojection_max_px, float(np.abs(projected - uv).max()))
        if not np.allclose(projected, uv, atol=1e-2, rtol=0):
            raise ValueError("HO3D WDS 2D/3D/intrinsic projection mismatch")

    suffix_to_index = {suffix: index for index, (_, suffix, _) in enumerate(parsed)}
    physical_to_index = tuple(suffix_to_index[suffix] for suffix in PHYSICAL_TO_SEQUENCE_SUFFIX)
    reference_index = physical_to_index[REFERENCE_PHYSICAL_CAMERA]
    reference_from_world = np.linalg.inv(extrinsics[reference_index])
    reference_from_camera = [reference_from_world @ value for value in extrinsics]
    aligned = []
    for value, transform in zip(vertices, reference_from_camera):
        homogeneous = np.concatenate([value.astype(np.float64), np.ones((778, 1))], axis=1)
        aligned.append((homogeneous @ transform.T)[:, :3])
    aligned_array = np.stack(aligned)
    residual = float(np.sqrt(np.mean(np.sum((aligned_array - aligned_array.mean(axis=0)) ** 2, axis=-1))))
    return {
        "sequence_id": next(iter(bases)),
        "frame_id": next(iter(frames)),
        "serials": serials,
        "physical_to_wds_index": physical_to_index,
        "mesh_alignment_rms_m": residual,
        "cross_view_mesh_consistent": residual <= CROSS_VIEW_ALIGNMENT_TOLERANCE_M,
        "supervision_target_policy": SUPERVISION_TARGET_POLICY,
        "reprojection_max_px": reprojection_max_px,
    }


def read_wds_member(wds_root: Path, row: Mapping[str, object], member: Mapping[str, object]) -> bytes:
    shard = Path(wds_root) / str(row["wds_shard"])
    if not shard.is_file():
        raise FileNotFoundError(shard)
    offset, size = int(member["offset"]), int(member["size"])
    if offset < 0 or size <= 0 or offset + size > shard.stat().st_size:
        raise ValueError("invalid WDS member offset/size")
    with shard.open("rb") as stream:
        stream.seek(offset)
        payload = stream.read(size)
    if len(payload) != size:
        raise OSError("short WDS member read")
    return payload


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points.astype(np.float64), np.ones((*points.shape[:-1], 1))], axis=-1)
    return (homogeneous @ transform.T)[..., :3]


def decode_ho3d_wds_group(
    row: Mapping[str, object], *, wds_root: Path,
    localizer: Mapping[tuple[str, int], Mapping[str, object]],
    mano_base_regressor: Tensor,
) -> HO3DMVGroup:
    if REFERENCE_RUNTIME_SLOT != 0:
        raise RuntimeError("Form2Frame requires the reference camera in runtime slot zero")
    if row.get("adapter_kind") != ADAPTER_KIND or row.get("adapter_version") != ADAPTER_VERSION:
        raise ValueError("unregistered HO3D WDS adapter identity")
    label = decode_label_bytes(read_wds_member(wds_root, row, row["label_member"]))
    identity = validate_wds_label(label)
    if identity["sequence_id"] != str(row["sequence_id"]) or identity["frame_id"] != str(row["frame_id"]):
        raise ValueError("HO3D WDS row/label identity mismatch")
    physical_to_index = tuple(int(value) for value in identity["physical_to_wds_index"])
    extrinsics = [np.asarray(value, dtype=np.float64) for value in label["cam_extr"]]
    reference_index = physical_to_index[REFERENCE_PHYSICAL_CAMERA]
    reference_from_world = np.linalg.inv(extrinsics[reference_index])
    reference_from_camera = [reference_from_world @ value for value in extrinsics]

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
    joints_2d = np.zeros((8, 21, 2), dtype=np.float32)
    vertices_reference, joints_reference, provenance = [], [], []
    image_members = {int(value["wds_image_index"]): value for value in row["views"]}

    for physical_id, wds_index in enumerate(physical_to_index):
        runtime_slot = RUNTIME_SLOT_TO_PHYSICAL.index(physical_id)
        K = np.asarray(label["cam_intr"][wds_index], dtype=np.float64)
        transform = reference_from_camera[wds_index]
        intrinsics[runtime_slot] = K
        camera_to_world[runtime_slot] = transform
        vertices_reference.append(_transform_points(np.asarray(label["verts_3d"][wds_index]), transform))
        joints_reference.append(_transform_points(np.asarray(label["joints_3d"][wds_index]), transform))

        localization = localizer[(str(row["key"]), physical_id)]
        localized = localization["localization_status"] == "LOCALIZED"
        runtime_mask[runtime_slot] = localized
        if localized:
            affine = np.asarray(localization["raw_to_crop"], dtype=np.float32)
            inverse = np.asarray(localization["crop_to_raw"], dtype=np.float32)
            encoded = read_wds_member(wds_root, row, image_members[wds_index])
            image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape != (480, 640, 3):
                raise ValueError("failed to decode raw 640x480 HO3D WDS RGB")
            rgb_crops[runtime_slot], normalized[runtime_slot] = _crop(image, affine)
            raw_to_crop[runtime_slot], crop_to_raw[runtime_slot] = affine, inverse
            joints_2d[runtime_slot] = apply_affine(
                np.asarray(label["joints_2d"][wds_index], dtype=np.float64), affine
            )
        provenance.append({
            "physical_camera_id": physical_id,
            "runtime_slot": runtime_slot,
            "sequence_suffix": PHYSICAL_TO_SEQUENCE_SUFFIX[physical_id],
            "wds_shard": row["wds_shard"],
            "wds_member": image_members[wds_index]["wds_member"],
            "localization_status": localization["localization_status"],
            "adapter_kind": ADAPTER_KIND,
            "adapter_version": ADAPTER_VERSION,
        })

    provenance.sort(key=lambda item: int(item["runtime_slot"]))
    if not runtime_mask[REFERENCE_RUNTIME_SLOT] or int(runtime_mask[:5].sum()) < 2:
        raise ValueError("DETECTOR_INVALID_GROUP")
    if not np.allclose(camera_to_world[0], np.eye(4), atol=1e-7, rtol=0):
        raise ValueError("HO3D runtime slot zero is not physical camera zero/reference")
    vertices_stack = np.stack(vertices_reference)
    joints_stack = np.stack(joints_reference)
    # A late portion of publisher sequence SB1 contains individually valid
    # camera labels that are not synchronized across the five views.  Averaging
    # those labels creates a target that does not exist in any source record.
    # The task's world frame is the frozen physical-camera-0 frame, so bind
    # supervision to that camera's own label and retain every transformed view
    # below strictly as a data-quality diagnostic.
    gt_vertices = vertices_stack[REFERENCE_PHYSICAL_CAMERA].astype(np.float32)
    dataset_joints = joints_stack[REFERENCE_PHYSICAL_CAMERA].astype(np.float32)
    regressor = mano_base_regressor.detach().cpu().to(torch.float32)
    gt_mesh_joints = poem_published_mesh_joints_21(torch.from_numpy(gt_vertices), regressor).numpy()
    gt_joint_visibility = np.zeros((8, 21), dtype=bool)
    label_output = {
        "joints_2d": joints_2d,
        "joints_2d_space": "crop",
        "reference_physical_camera": REFERENCE_PHYSICAL_CAMERA,
        "reference_runtime_slot": REFERENCE_RUNTIME_SLOT,
        "joints_3d": dataset_joints,
        "verts_3d": gt_vertices,
        "dataset_joints_3d": dataset_joints,
        "native_ho3d_joints_3d": dataset_joints,
        "gt_mesh_joints_3d": gt_mesh_joints,
        "per_view_vertices_reference": vertices_stack.astype(np.float32),
        "per_view_dataset_joints_reference": joints_stack.astype(np.float32),
        "cross_view_mesh_alignment_rms_m": identity["mesh_alignment_rms_m"],
        "cross_view_mesh_consistent": identity["cross_view_mesh_consistent"],
        "supervision_target_policy": SUPERVISION_TARGET_POLICY,
        "visibility_supervision_available": False,
        "headline_joint_endpoint": "matched POEM mesh-derived joints",
        "adapter_kind": ADAPTER_KIND,
        "adapter_version": ADAPTER_VERSION,
    }
    return HO3DMVGroup(
        str(row["key"]), torch.from_numpy(rgb_crops), torch.from_numpy(normalized),
        torch.from_numpy(raw_to_crop), torch.from_numpy(crop_to_raw),
        torch.from_numpy(intrinsics), torch.from_numpy(camera_to_world),
        torch.from_numpy(gt_joint_visibility), torch.from_numpy(runtime_mask),
        tuple([f"physical_camera_{i}" for i in RUNTIME_SLOT_TO_PHYSICAL] + ["unused_5", "unused_6", "unused_7"]),
        torch.from_numpy(raw_size), label_output, tuple(provenance),
    )
