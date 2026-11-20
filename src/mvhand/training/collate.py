from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np, torch
from mvhand.data.reference_targets import reference_target
from mvhand.models.interfaces import Form2FrameBatch

@dataclass
class TrainingBatch:
    keys: tuple[str,...]; model: Form2FrameBatch; raw_to_crop: torch.Tensor
    joints_2d_crop: torch.Tensor; joints_3d: torch.Tensor; vertices_3d: torch.Tensor
    gt_joint_visibility: torch.Tensor; joint_2d_supervision_mask: torch.Tensor

    @property
    def joints_2d(self) -> torch.Tensor:
        """Backward-compatible alias; canonical training coordinates are crop pixels."""
        return self.joints_2d_crop


def _crop_joint_supervision(groups: Sequence, raw_to_crop: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor]:
    points = torch.as_tensor(
        np.stack([g.label["joints_2d"] for g in groups]), dtype=torch.float32, device=device,
    )
    spaces = tuple(g.label.get("joints_2d_space") for g in groups)
    if any(space is None for space in spaces):
        raise ValueError("joints_2d_space must be declared by every dataset adapter")
    if any(space not in {"raw", "crop"} for space in spaces):
        raise ValueError("joints_2d_space must be 'raw' or 'crop'")
    for index, space in enumerate(spaces):
        if space == "raw":
            homogeneous = torch.cat((points[index], torch.ones_like(points[index, ..., :1])), dim=-1)
            points[index] = torch.einsum("vrc,vjc->vjr", raw_to_crop[index], homogeneous)

    masks = []
    for group in groups:
        annotation_valid = group.label.get("joint_2d_annotation_valid")
        if annotation_valid is not None:
            mask = torch.as_tensor(annotation_valid, dtype=torch.bool, device=device)
        elif bool(group.label.get("visibility_supervision_available", True)):
            mask = group.gt_joint_visibility.to(device=device, dtype=torch.bool)
        else:
            # Annotation coordinates remain valid for occluded joints.  In the
            # absence of an occlusion annotation, supervise every joint only in
            # views whose image crop is actually supplied to the model.
            mask = group.runtime_view_mask[:, None].expand(-1, points.shape[2]).to(device)
        mask = mask & group.runtime_view_mask[:, None].to(device)
        masks.append(mask)
    supervision = torch.stack(masks)
    if supervision.shape != points.shape[:-1]:
        raise ValueError("2D supervision mask must have shape [B,V,J]")
    return points, supervision


def _reference_targets(groups: Sequence, name: str, count: int, device) -> torch.Tensor:
    values=[]
    for group in groups:
        original=np.asarray(group.label[name],dtype=np.float32)
        reference=reference_target(group.label,name,count).astype(np.float32,copy=False)
        value=np.repeat(reference[None],8,axis=0) if original.shape==(count,3) else original
        values.append(value)
    return torch.as_tensor(np.stack(values),dtype=torch.float32,device=device)

def collate_groups(groups: Sequence, device="cpu"):
    if not groups or len({g.key for g in groups})!=len(groups): raise ValueError("groups must be nonempty and unique")
    if any(g.rgb_crops.shape!=(8,3,256,256) for g in groups): raise ValueError("group crop contract mismatch")
    raw_to_crop=torch.stack([g.raw_to_crop for g in groups]).to(device)
    model=Form2FrameBatch(
      crops=torch.stack([g.rgb_crops for g in groups]).to(device),
      intrinsics=torch.stack([g.intrinsics for g in groups]).to(device),
      camera_to_world=torch.stack([g.camera_to_world for g in groups]).to(device),
      crop_to_raw=torch.stack([g.crop_to_raw for g in groups]).to(device),
      runtime_view_mask=torch.stack([g.runtime_view_mask for g in groups]).to(device),
      raw_to_crop=raw_to_crop)
    joints_2d_crop,supervision=_crop_joint_supervision(groups,raw_to_crop,device)
    return TrainingBatch(tuple(g.key for g in groups),model,raw_to_crop,
      joints_2d_crop,
      _reference_targets(groups,"joints_3d",21,device),
      _reference_targets(groups,"verts_3d",778,device),
      torch.stack([g.gt_joint_visibility for g in groups]).to(device),supervision)
