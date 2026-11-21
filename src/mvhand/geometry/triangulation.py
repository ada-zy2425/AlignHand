"""Support-weighted DLT triangulation in raw pixel coordinates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class ProjectiveTriangulationResult:
    points_3d: Tensor
    valid: Tensor
    weights: Tensor
    condition_ratio: Tensor
    singular_values: Tensor


def build_reference_to_camera(camera_to_world: Tensor) -> Tensor:
    """Build ``T_reference_to_camera[v] = inv(E[v]) @ E[0]``."""
    if camera_to_world.ndim != 4 or camera_to_world.shape[-2:] != (4, 4):
        raise ValueError("camera_to_world must have shape [B,V,4,4]")
    if not camera_to_world.is_floating_point() or not torch.isfinite(camera_to_world).all().item():
        raise ValueError("camera_to_world must be finite and floating")
    b, v = camera_to_world.shape[:2]
    inverse = torch.linalg.inv(camera_to_world.reshape(b * v, 4, 4)).reshape(b, v, 4, 4)
    return inverse @ camera_to_world[:, :1]


def build_projection_matrices(intrinsics: Tensor, reference_to_camera: Tensor) -> Tensor:
    if intrinsics.shape[:2] != reference_to_camera.shape[:2] or intrinsics.shape[-2:] != (3, 3) or reference_to_camera.shape[-2:] != (4, 4):
        raise ValueError("expected intrinsics [B,V,3,3] and transforms [B,V,4,4]")
    if not torch.isfinite(intrinsics).all().item() or not torch.isfinite(reference_to_camera).all().item():
        raise ValueError("camera tensors must be finite")
    return intrinsics @ reference_to_camera[..., :3, :]


def normalize_relative_weights(logits: Tensor, visibility: Tensor) -> Tensor:
    if logits.ndim != 3 or visibility.shape != logits.shape:
        raise ValueError("logits and visibility must have shape [B,V,J]")
    if visibility.dtype != torch.bool:
        raise TypeError("visibility must be Boolean")
    if not logits.is_floating_point() or not torch.isfinite(logits).all().item():
        raise ValueError("logits must be finite and floating")
    if logits.device != visibility.device:
        raise ValueError("logits and visibility must share a device")
    selected = visibility.sum(dim=1, keepdim=True)
    if torch.any(selected == 0):
        raise ValueError("each joint must have at least one visible view")
    masked = logits.masked_fill(~visibility, float("-inf"))
    return torch.softmax(masked, dim=1)


def triangulate_projective_dlt(
    observations_raw: Tensor,
    projection_matrices: Tensor,
    weights: Optional[Tensor] = None,
    visibility: Optional[Tensor] = None,
    *,
    min_views: int = 2,
    eps: float = 1e-12,
) -> ProjectiveTriangulationResult:
    """Raw-pixel ``K[R|t]`` DLT with ``sqrt(weight)`` row scaling.

    Inputs are ``[B,V,J,2]``, ``[B,V,3,4]``, and optional ``[B,V,J]``.
    The output is in the view-zero reference-camera frame. No pruning,
    geometric rejection, or robust fallback is performed.
    """
    if observations_raw.ndim != 4 or observations_raw.shape[-1] != 2:
        raise ValueError("observations_raw must have shape [B,V,J,2]")
    b, v, j = observations_raw.shape[:3]
    if projection_matrices.shape != (b, v, 3, 4):
        raise ValueError("projection_matrices must have shape [B,V,3,4]")
    if v < min_views:
        raise ValueError("available view dimension is below min_views")
    for name, value in (("observations_raw", observations_raw), ("projection_matrices", projection_matrices)):
        if not value.is_floating_point() or not torch.isfinite(value).all().item():
            raise ValueError(f"{name} must be finite and floating")
    if observations_raw.device != projection_matrices.device or observations_raw.dtype != projection_matrices.dtype:
        raise ValueError("observations and projection matrices must match dtype/device")
    if visibility is None:
        visibility = torch.ones((b, v, j), dtype=torch.bool, device=observations_raw.device)
    if visibility.shape != (b, v, j) or visibility.dtype != torch.bool or visibility.device != observations_raw.device:
        raise ValueError("visibility must be Boolean [B,V,J] on the input device")
    if weights is None:
        weights = visibility.to(observations_raw.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1)
    if weights.shape != (b, v, j) or weights.dtype != observations_raw.dtype or weights.device != observations_raw.device:
        raise ValueError("weights must be [B,V,J] with matching dtype/device")
    if not torch.isfinite(weights).all().item() or torch.any(weights < 0).item():
        raise ValueError("weights must be finite and non-negative")
    effective = torch.where(visibility, weights, torch.zeros_like(weights))
    p1 = projection_matrices[:, :, 0, :].unsqueeze(2)
    p2 = projection_matrices[:, :, 1, :].unsqueeze(2)
    p3 = projection_matrices[:, :, 2, :].unsqueeze(2)
    row_u = observations_raw[..., 0, None] * p3 - p1
    row_v = observations_raw[..., 1, None] * p3 - p2
    matrix = torch.stack((row_u, row_v), dim=-2).permute(0, 2, 1, 3, 4)
    # Preserve exact zero-weight rows while defining their backward derivative
    # as zero.  A direct sqrt(0) has an infinite Jacobian, so a zero upstream
    # gradient can become NaN (0 * inf), particularly after BF16 softmax
    # underflow.  clamp_min is inactive for positive weights and has zero
    # derivative for exact zeros; the explicit mask preserves the forward DLT.
    positive = effective > 0
    safe_effective = effective.clamp_min(torch.finfo(effective.dtype).tiny)
    stable_sqrt = safe_effective.sqrt() * positive.to(effective.dtype)
    row_weights = stable_sqrt.permute(0, 2, 1).repeat_interleave(2, dim=-1)[..., None]
    matrix = matrix.reshape(b, j, 2 * v, 4) * row_weights
    # The weighted null-space solve is conditioning-sensitive: legitimate
    # positive view weights can span nearly the full FP32 dynamic range.  Keep
    # the exact design matrix and row weighting above, but solve and normalize
    # the homogeneous coordinates uniformly in FP64 before converting outputs
    # back to the invocation dtype.
    _, singular, vh = torch.linalg.svd(matrix.to(torch.float64), full_matrices=False)
    homogeneous = vh[..., -1, :]
    scale = homogeneous[..., 3:4]
    safe_scale = torch.where(scale.abs() >= eps, scale, torch.ones_like(scale))
    points = (homogeneous[..., :3] / safe_scale).to(observations_raw.dtype)
    active = (effective > 0).sum(dim=1)
    valid = (active >= min_views) & (scale[..., 0].abs() >= eps) & torch.isfinite(points).all(dim=-1) & torch.isfinite(singular).all(dim=-1)
    points = torch.where(valid[..., None], points, torch.zeros_like(points))
    ratio = singular[..., -1] / singular[..., -2].clamp_min(eps)
    return ProjectiveTriangulationResult(points, valid, effective, ratio.to(observations_raw.dtype), singular.to(observations_raw.dtype))
