"""HanCo eight-camera adapter for the canonical Form2Frame interface."""
from __future__ import annotations

from dataclasses import dataclass
import io
import json
from pathlib import Path
from typing import Mapping, Sequence
import zipfile

import cv2
import numpy as np
import torch
from torch import Tensor

from mvhand.geometry.mesh import poem_published_mesh_joints_21


CAMERAS = tuple(f"cam{index}" for index in range(8))
IMAGE_SIZE = 224
CROP_SIZE = 256
CROP_EXPANSION = 1.5


@dataclass(frozen=True)
class HanCoMVGroup:
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

    def to_form2frame_batch(self, *, device="cpu"):
        from mvhand.models.interfaces import Form2FrameBatch
        return Form2FrameBatch(
            crops=self.rgb_crops.unsqueeze(0).to(device),
            intrinsics=self.intrinsics.unsqueeze(0).to(device),
            camera_to_world=self.camera_to_world.unsqueeze(0).to(device),
            crop_to_raw=self.crop_to_raw.unsqueeze(0).to(device),
            runtime_view_mask=self.runtime_view_mask.unsqueeze(0).to(device),
        )


class HanCoArchives:
    """Read official HanCo ZIPs without extracting hundreds of thousands of files."""

    def __init__(self, *, rgb: Path, xyz: Path, shape: Path, calib_meta: Path):
        self.rgb = zipfile.ZipFile(rgb)
        self.xyz = zipfile.ZipFile(xyz)
        self.shape = zipfile.ZipFile(shape)
        self.calib = zipfile.ZipFile(calib_meta)
        self._suffix_cache: dict[tuple[int, str], str] = {}

    def close(self) -> None:
        for archive in (self.rgb, self.xyz, self.shape, self.calib):
            archive.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def read(self, archive: zipfile.ZipFile, candidates: Sequence[str]) -> bytes:
        for candidate in candidates:
            try:
                return archive.read(candidate)
            except KeyError:
                pass
        # Accommodate a single release-directory prefix, but reject ambiguous
        # suffix matches rather than guessing an identity.
        for candidate in candidates:
            cache_key = (id(archive), candidate)
            name = self._suffix_cache.get(cache_key)
            if name is None:
                matches = [value for value in archive.namelist() if value.endswith("/" + candidate)]
                if len(matches) != 1:
                    continue
                name = matches[0]
                self._suffix_cache[cache_key] = name
            return archive.read(name)
        raise KeyError(f"HanCo member not found: {tuple(candidates)}")


class HanCoManoMesh:
    """Reconstruct the released world-space MANO fit using the pinned asset."""

    def __init__(self, mano_right: Path):
        import smplx
        self.layer = smplx.MANO(
            model_path=str(mano_right), is_rhand=True, use_pca=False,
            flat_hand_mean=False,
        )

    def __call__(self, payload: Mapping[str, object]) -> np.ndarray:
        poses = torch.as_tensor(np.asarray(payload["poses"]), dtype=torch.float32)
        shapes = torch.as_tensor(np.asarray(payload["shapes"]), dtype=torch.float32)
        translation = torch.as_tensor(np.asarray(payload["global_t"]), dtype=torch.float32).reshape(-1, 3)
        if poses.shape != (1, 48) or shapes.shape != (1, 10) or translation.shape != (1, 3):
            raise ValueError("HanCo MANO world-fit parameter shape mismatch")
        with torch.no_grad():
            output = self.layer(
                global_orient=poses[:, :3], hand_pose=poses[:, 3:],
                betas=shapes, transl=translation,
            )
        vertices = output.vertices[0].detach().cpu().numpy().astype(np.float32)
        if vertices.shape != (778, 3) or not np.isfinite(vertices).all():
            raise ValueError("HanCo MANO reconstruction failed")
        return vertices


def project_world(points: np.ndarray, world_to_camera: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    camera = (np.asarray(world_to_camera, dtype=np.float64) @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    pixels_h = (np.asarray(intrinsics, dtype=np.float64) @ camera.T).T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    valid = np.isfinite(pixels).all(axis=-1) & np.isfinite(camera).all(axis=-1) & (camera[:, 2] > 0)
    return pixels.astype(np.float32), valid


def gt_joint_crop_affine(points: np.ndarray, *, expansion: float = CROP_EXPANSION, output_size: int = CROP_SIZE) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (21, 2) or not np.isfinite(points).all():
        raise ValueError("HanCo crop requires 21 finite projected joints")
    minimum, maximum = points.min(0), points.max(0)
    center = (minimum + maximum) / 2.0
    side = float(np.max(maximum - minimum) * expansion)
    if not np.isfinite(side) or side <= 1e-6:
        raise ValueError("degenerate HanCo GT crop")
    scale = float(output_size) / side
    raw_to_crop = np.asarray([
        [scale, 0.0, output_size / 2.0 - scale * center[0]],
        [0.0, scale, output_size / 2.0 - scale * center[1]],
    ], dtype=np.float32)
    homogeneous = np.c_[points, np.ones(len(points))]
    return raw_to_crop, (homogeneous @ raw_to_crop.T).astype(np.float32)


def _decode_rgb(value: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError("HanCo RGB decode or dimensions failed")
    return image


def _crop(image: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    patch = cv2.warpAffine(
        image, affine, (CROP_SIZE, CROP_SIZE), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    rgb = patch[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    mean = np.asarray([.485, .456, .406], np.float32)[:, None, None]
    std = np.asarray([.229, .224, .225], np.float32)[:, None, None]
    return rgb, (rgb - mean) / std


def decode_hanco_group(
    row: Mapping[str, object], *, archives: HanCoArchives,
    mano_mesh: HanCoManoMesh, mano_base_regressor: Tensor,
) -> HanCoMVGroup:
    sequence = str(row["sequence_id"]); frame = str(row["frame_id"]); key = str(row["key"])
    if key != f"{sequence}/{frame}":
        raise ValueError("HanCo row identity mismatch")
    xyz = np.asarray(json.loads(archives.read(
        archives.xyz, (f"xyz/{sequence}/{frame}.json",)
    )), dtype=np.float64)
    calibration = json.loads(archives.read(
        archives.calib, (f"calib/{sequence}/{frame}.json",)
    ))
    shape_payload = json.loads(archives.read(
        archives.shape,
        (f"shape/{sequence}/{frame}.json", f"shape/{frame}.json"),
    ))
    if xyz.shape != (21, 3):
        raise ValueError("HanCo world joints shape mismatch")
    intrinsics = np.asarray(calibration["K"], dtype=np.float64)
    world_to_camera = np.asarray(calibration["M"], dtype=np.float64)
    if intrinsics.shape != (8, 3, 3) or world_to_camera.shape != (8, 4, 4):
        raise ValueError("HanCo calibration shape mismatch")
    if not all(np.isfinite(value).all() for value in (xyz, intrinsics, world_to_camera)):
        raise ValueError("nonfinite HanCo annotation")

    vertices_world = mano_mesh(shape_payload).astype(np.float64)
    reference_from_world = world_to_camera[0]
    joints_reference = (reference_from_world @ np.c_[xyz, np.ones(21)].T).T[:, :3].astype(np.float32)
    vertices_reference = (reference_from_world @ np.c_[vertices_world, np.ones(778)].T).T[:, :3].astype(np.float32)

    rgb = np.zeros((8, 3, CROP_SIZE, CROP_SIZE), np.float32)
    normalized = np.zeros_like(rgb)
    raw_to_crop = np.zeros((8, 2, 3), np.float32)
    crop_to_raw = np.zeros_like(raw_to_crop)
    crop_joints = np.zeros((8, 21, 2), np.float32)
    annotation_valid = np.zeros((8, 21), bool)
    provenance = []
    for camera in range(8):
        raw_joints, valid = project_world(xyz, world_to_camera[camera], intrinsics[camera])
        if not valid.all():
            raise ValueError(f"HanCo annotated joint has non-positive depth in cam{camera}")
        affine, crop_points = gt_joint_crop_affine(raw_joints)
        homogeneous = np.eye(3, dtype=np.float64); homogeneous[:2] = affine
        inverse = np.linalg.inv(homogeneous)[:2].astype(np.float32)
        member = f"rgb/{sequence}/cam{camera}/{frame}.jpg"
        image = _decode_rgb(archives.read(archives.rgb, (member,)))
        rgb[camera], normalized[camera] = _crop(image, affine)
        raw_to_crop[camera] = affine; crop_to_raw[camera] = inverse
        crop_joints[camera] = crop_points; annotation_valid[camera] = valid
        provenance.append({
            "physical_camera_id": camera, "physical_camera_name": f"cam{camera}",
            "rgb_member": member, "crop_policy": "gt_joint_square_1.5",
        })
    camera_to_world = np.linalg.inv(world_to_camera).astype(np.float32)
    regressor = mano_base_regressor.detach().cpu().float()
    mesh_joints = poem_published_mesh_joints_21(torch.from_numpy(vertices_reference), regressor).numpy()
    label = {
        "joints_2d": crop_joints,
        "joints_2d_space": "crop",
        "joint_2d_annotation_valid": annotation_valid,
        "joints_3d": joints_reference,
        "dataset_joints_3d": joints_reference,
        "native_hanco_joints_3d": joints_reference,
        "verts_3d": vertices_reference,
        "gt_mesh_joints_3d": mesh_joints,
        "reference_camera_id": 0,
        "source_coordinate_space": "world_metres",
        "target_coordinate_space": "camera0_metres",
        "crop_policy": "gt_joint_square_1.5",
        "visibility_supervision_available": False,
        "headline_joint_endpoint": "mesh-derived joints; dataset joints retained for xgeo",
    }
    tensors = (rgb, normalized, raw_to_crop, crop_to_raw, intrinsics, camera_to_world,
               crop_joints, joints_reference, vertices_reference)
    if not all(np.isfinite(value).all() for value in tensors):
        raise ValueError("nonfinite HanCo decoded group")
    return HanCoMVGroup(
        key, torch.from_numpy(rgb), torch.from_numpy(normalized),
        torch.from_numpy(raw_to_crop), torch.from_numpy(crop_to_raw),
        torch.from_numpy(intrinsics.astype(np.float32)), torch.from_numpy(camera_to_world),
        torch.zeros((8, 21), dtype=torch.bool), torch.ones(8, dtype=torch.bool),
        CAMERAS, torch.full((8, 2), IMAGE_SIZE, dtype=torch.int64), label,
        tuple(provenance),
    )
