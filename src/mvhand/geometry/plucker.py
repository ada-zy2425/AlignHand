"""Reference-frame Plucker features used by Stage I."""
from __future__ import annotations
import torch
from torch import Tensor
from torch.nn import functional as F


def reference_plucker_features(
    observations_raw: Tensor,
    intrinsics: Tensor,
    reference_to_camera: Tensor,
) -> tuple[Tensor, Tensor]:
    inverse_k = torch.linalg.inv(intrinsics)
    pixel_h = torch.cat((observations_raw, torch.ones_like(observations_raw[..., :1])), dim=-1)
    camera_rays = torch.einsum("bvrc,bvjc->bvjr", inverse_k, pixel_h)
    rotation = reference_to_camera[..., :3, :3]
    translation = reference_to_camera[..., :3, 3]
    centers = -(rotation.transpose(-1, -2) @ translation[..., None]).squeeze(-1)
    directions = torch.einsum("bvcr,bvjc->bvjr", rotation, camera_rays)
    directions = F.normalize(directions, dim=-1, eps=1e-8)
    moments = torch.cross(centers[:, :, None].expand_as(directions), directions, dim=-1)
    return directions, moments
