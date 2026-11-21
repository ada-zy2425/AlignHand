"""Camera utilities: homogeneous coordinates, rigid transforms, projection and pixel rays."""
from __future__ import annotations

import torch
from torch import Tensor


def _check_last_dims(tensor: Tensor, dims: tuple[int, ...], name: str) -> None:
    if tensor.shape[-len(dims) :] != dims:
        raise ValueError(f"{name} must end in shape {dims}; got {tuple(tensor.shape)}")


def homogenize(points: Tensor) -> Tensor:
    _check_last_dims(points, (3,), "points")
    ones = torch.ones_like(points[..., :1])
    return torch.cat([points, ones], dim=-1)


def dehomogenize(points_h: Tensor, eps: float = 1e-8) -> Tensor:
    if points_h.shape[-1] < 2:
        raise ValueError("Homogeneous points must have at least 2 coordinates")
    denominator = points_h[..., -1:]
    safe = torch.where(
        denominator.abs() < eps,
        denominator.sign().where(denominator != 0, torch.ones_like(denominator)) * eps,
        denominator,
    )
    return points_h[..., :-1] / safe


def invert_rigid_transform(transform: Tensor) -> Tensor:
    """Invert an SE(3) transform with convention X_dst = T_src_to_dst X_src."""
    _check_last_dims(transform, (4, 4), "transform")
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    rotation_t = rotation.transpose(-1, -2)
    inverse_translation = -(rotation_t @ translation.unsqueeze(-1)).squeeze(-1)
    out = torch.zeros_like(transform)
    out[..., :3, :3] = rotation_t
    out[..., :3, 3] = inverse_translation
    out[..., 3, 3] = 1.0
    return out


def apply_transform(transform: Tensor, points: Tensor) -> Tensor:
    """Apply transform to 3D points, supporting broadcastable leading dimensions."""
    _check_last_dims(transform, (4, 4), "transform")
    _check_last_dims(points, (3,), "points")
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    return torch.matmul(rotation, points.unsqueeze(-1)).squeeze(-1) + translation


def compose_projection_matrix(intrinsics: Tensor, t_ref_to_cam: Tensor) -> Tensor:
    """Return P = K [R|t], mapping reference-camera 3D to image pixels."""
    _check_last_dims(intrinsics, (3, 3), "intrinsics")
    _check_last_dims(t_ref_to_cam, (4, 4), "t_ref_to_cam")
    return intrinsics @ t_ref_to_cam[..., :3, :]


def project_points(
    points_ref: Tensor,
    intrinsics: Tensor,
    t_ref_to_cam: Tensor,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """
    Project points expressed in the reference camera to all target cameras.

    Common shapes:
      points_ref:       [B, J, 3]
      intrinsics:       [B, V, 3, 3]
      t_ref_to_cam:     [B, V, 4, 4]
    Returns:
      uv:               [B, V, J, 2]
      depth:            [B, V, J]
    """
    _check_last_dims(points_ref, (3,), "points_ref")
    _check_last_dims(intrinsics, (3, 3), "intrinsics")
    _check_last_dims(t_ref_to_cam, (4, 4), "t_ref_to_cam")

    if points_ref.ndim == intrinsics.ndim - 1:
        points = points_ref.unsqueeze(-3)
    else:
        points = points_ref
    rotation = t_ref_to_cam[..., :3, :3]
    translation = t_ref_to_cam[..., :3, 3]
    points_cam = torch.einsum("...vab,...vjb->...vja", rotation, points) + translation.unsqueeze(-2)
    pixels_h = torch.einsum("...vab,...vjb->...vja", intrinsics, points_cam)
    depth = points_cam[..., 2]
    safe_depth = torch.where(
        depth.abs() < eps,
        depth.sign().where(depth != 0, torch.ones_like(depth)) * eps,
        depth,
    )
    uv = pixels_h[..., :2] / safe_depth.unsqueeze(-1)
    return uv, depth


def pixels_to_normalized_rays(uv: Tensor, intrinsics: Tensor, eps: float = 1e-8) -> Tensor:
    """Convert pixel coordinates to unit-length camera-frame ray directions."""
    if uv.shape[-1] != 2:
        raise ValueError(f"uv must end in 2; got {tuple(uv.shape)}")
    _check_last_dims(intrinsics, (3, 3), "intrinsics")
    ones = torch.ones_like(uv[..., :1])
    uv_h = torch.cat([uv, ones], dim=-1)
    inv_k = torch.linalg.inv(intrinsics)
    rays = torch.einsum("...vab,...vjb->...vja", inv_k, uv_h)
    return rays / rays.norm(dim=-1, keepdim=True).clamp_min(eps)
