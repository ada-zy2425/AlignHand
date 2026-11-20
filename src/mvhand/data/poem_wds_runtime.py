from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import torch
from torch import Tensor
import webdataset as wds

from mvhand.data.safe_numpy_pickle import restricted_numpy_loads


@dataclass(frozen=True)
class PoemWDSGroup:
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

    def to_form2frame_batch(self, *, device: torch.device | str = "cpu"):
        from mvhand.models.interfaces import Form2FrameBatch
        return Form2FrameBatch(
            crops=self.rgb_crops.unsqueeze(0).to(device),
            intrinsics=self.intrinsics.unsqueeze(0).to(device),
            camera_to_world=self.camera_to_world.unsqueeze(0).to(device),
            crop_to_raw=self.crop_to_raw.unsqueeze(0).to(device),
            runtime_view_mask=self.runtime_view_mask.unsqueeze(0).to(device),
        )

    def permute_views(self, order: Sequence[int]) -> "PoemWDSGroup":
        order = tuple(order)
        if sorted(order) != list(range(8)):
            raise ValueError("view permutation must be a bijection over eight views")
        index = torch.tensor(order, dtype=torch.long)
        label = {
            key: (np.asarray(value)[list(order)].copy() if np.asarray(value).ndim and np.asarray(value).shape[0] == 8 else value)
            for key, value in self.label.items()
        }
        return PoemWDSGroup(
            self.key, self.rgb_crops[index], self.normalized_crops[index],
            self.raw_to_crop[index], self.crop_to_raw[index], self.intrinsics[index],
            self.camera_to_world[index], self.gt_joint_visibility[index],
            self.runtime_view_mask[index],
            tuple(self.camera_ids[i] for i in order), self.raw_size[index], label,
        )


def decode_canonical_sample(
    sample: Mapping[str, object], hamer_source: Path, *, qualified_key: str | None = None,
) -> PoemWDSGroup:
    """Decode one exact indexed or sequential POEM-v2 sample."""
    expected = {f"image_{i}.jpg" for i in range(8)} | {"label.pyd"}
    if not expected.issubset(sample):
        raise ValueError("WDS record does not contain eight images and label.pyd")
    label_payload = sample["label.pyd"]
    if not isinstance(label_payload, bytes):
        raise TypeError("label.pyd must be publisher bytes")
    label = restricted_numpy_loads(label_payload)
    if not isinstance(label, Mapping):
        raise TypeError("trusted label payload must decode to a mapping")
    # Publisher coordinates are raw-image pixels.  Make that interface explicit
    # so the collator performs exactly one raw-to-crop transform.
    label = dict(label)
    label["joints_2d_space"] = "raw"
    center = np.asarray(label["bbox_center"], dtype=np.float32)
    scale = np.asarray(label["bbox_scale"], dtype=np.float32)
    if center.shape != (8, 2) or scale.shape != (8,):
        raise ValueError("bbox center/scale must be eight-view arrays")
    from mvhand.models.hamer_runtime import install_renderer_free_hamer_namespace
    install_renderer_free_hamer_namespace(hamer_source)
    from hamer.datasets.utils import generate_image_patch_cv2
    mean = 255.0 * np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = 255.0 * np.array([0.229, 0.224, 0.225], dtype=np.float32)
    crops=[]
    rgb_crops=[]
    raw_to_crop=[]
    crop_to_raw=[]
    for index in range(8):
        image = cv2.imdecode(np.frombuffer(sample[f"image_{index}.jpg"], dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"image_{index} could not be decoded")
        patch, transform = generate_image_patch_cv2(
            image, float(center[index,0]), float(center[index,1]), float(scale[index]),
            float(scale[index]), 256, 256, False, 1.0, 0, border_mode=cv2.BORDER_CONSTANT,
        )
        rgb = patch[:,:,::-1].transpose(2,0,1).astype(np.float32)
        rgb_crops.append(rgb / 255.0)
        crops.append((rgb - mean[:,None,None]) / std[:,None,None])
        raw_to_crop.append(np.asarray(transform, dtype=np.float32))
        crop_to_raw.append(cv2.invertAffineTransform(transform).astype(np.float32))
    tensor=torch.from_numpy(np.stack(crops))
    if tensor.shape != (8,3,256,256) or not torch.isfinite(tensor).all().item():
        raise ValueError("canonical crop tensor contract failed")
    raw_to_crop_tensor = torch.from_numpy(np.stack(raw_to_crop))
    crop_to_raw_tensor = torch.from_numpy(np.stack(crop_to_raw))
    identity = torch.eye(3).expand(8, -1, -1).clone()
    raw_h = identity.clone(); raw_h[:, :2] = raw_to_crop_tensor
    crop_h = identity.clone(); crop_h[:, :2] = crop_to_raw_tensor
    # OpenCV returns the inverse in float64, but the maintained batch contract
    # stores float32 affines. Large crop translations can therefore accumulate
    # roughly one float32 ULP (~1.22e-4) in the homogeneous product.
    affine_atol = 2e-4
    if not (
        torch.allclose(raw_h @ crop_h, identity, atol=affine_atol, rtol=0)
        and torch.allclose(crop_h @ raw_h, identity, atol=affine_atol, rtol=0)
    ):
        raise ValueError("raw/crop affine transforms are not inverses")
    intrinsics = torch.as_tensor(np.asarray(label["cam_intr"]), dtype=torch.float32)
    camera_to_world = torch.as_tensor(np.asarray(label["cam_extr"]), dtype=torch.float32)
    gt_joint_visibility = torch.as_tensor(np.asarray(label["joints_vis"]) > 0, dtype=torch.bool)
    # Runtime availability describes inputs actually supplied to inference. GT
    # annotation visibility is supervision metadata and must never enter the
    # image-conditioned geometry path.
    runtime_view_mask = torch.ones(8, dtype=torch.bool)
    raw_size = torch.as_tensor(np.asarray(label["raw_size"]), dtype=torch.int64)
    camera_ids = tuple(str(value) for value in np.asarray(label["cam_serial"]))
    from mvhand.geometry.raw_pixel_dlt import build_reference_to_camera
    reference = build_reference_to_camera(camera_to_world.unsqueeze(0))[0, 0]
    if not torch.allclose(reference, torch.eye(4), atol=2e-5, rtol=0):
        raise ValueError("stored view 0 is not the reference camera")
    return PoemWDSGroup(
        qualified_key or str(sample["__key__"]), torch.from_numpy(np.stack(rgb_crops)), tensor,
        raw_to_crop_tensor, crop_to_raw_tensor, intrinsics, camera_to_world,
        gt_joint_visibility, runtime_view_mask, camera_ids, raw_size, label,
    )


def read_first_canonical_group(shard: Path, hamer_source: Path) -> PoemWDSGroup:
    """Read exactly one trusted canonical POEM-v2 WDS record."""
    shard = Path(shard).resolve()
    if not shard.is_file():
        raise FileNotFoundError(shard)
    sample = next(iter(wds.WebDataset([str(shard)], shardshuffle=False)))
    return decode_canonical_sample(sample, hamer_source)
