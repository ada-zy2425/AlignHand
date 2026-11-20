from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class MeshRecoveryResult:
    triangulated_joints_3d: Tensor
    aligned_prior_joints_3d: Tensor
    aligned_prior_vertices_3d: Tensor
    positioned_prior_joints_3d: Tensor
    final_joints_3d: Tensor
    final_vertices_3d: Tensor
    alignment_valid: Tensor
    alignment_residual_m: Tensor
    fusion_view_weights: Tensor
    valid: Tensor


def _weighted_similarity(source: Tensor, target: Tensor, weights: Tensor, eps: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    total = weights.sum(dim=-1, keepdim=True)
    valid = (weights > 0).sum(dim=-1) >= 3
    normalized = weights / total.clamp_min(eps)
    source_center = (source * normalized[..., None]).sum(dim=-2)
    target_center = (target * normalized[..., None]).sum(dim=-2)
    source_zero = source - source_center[..., None, :]
    target_zero = target - target_center[..., None, :]
    covariance = torch.einsum("bvji,bvj,bvjk->bvik", source_zero, normalized, target_zero)
    u, singular, vh = torch.linalg.svd(covariance)
    correction = torch.ones_like(singular)
    correction[..., -1] = torch.where(torch.linalg.det(vh.transpose(-1, -2) @ u.transpose(-1, -2)) < 0, -1.0, 1.0)
    rotation = vh.transpose(-1, -2) @ torch.diag_embed(correction) @ u.transpose(-1, -2)
    # The forward Procrustes rotation is well-defined at repeated singular
    # values while the SVD derivative is not. Preserve the exact rotation and
    # use the fixed-rotation envelope gradient for the remaining quantities.
    rotation = rotation.detach()
    source_energy = (normalized * source_zero.square().sum(dim=-1)).sum(dim=-1)
    target_energy = (normalized * target_zero.square().sum(dim=-1)).sum(dim=-1)
    rotated_source_zero = torch.einsum("bvik,bvjk->bvji", rotation, source_zero)
    aligned_covariance = (
        normalized * (rotated_source_zero * target_zero).sum(dim=-1)
    ).sum(dim=-1)
    scale = aligned_covariance / source_energy.clamp_min(eps)
    translation = target_center - scale[..., None] * torch.einsum("bvik,bvk->bvi", rotation, source_center)
    valid = (
        valid
        & (total[..., 0] > eps)
        & torch.isfinite(source_energy)
        & (source_energy > eps)
        & torch.isfinite(target_energy)
        & (target_energy > eps)
        & torch.isfinite(rotation).all(dim=(-2, -1))
        & torch.isfinite(scale)
        & (scale > eps)
        & torch.isfinite(translation).all(dim=-1)
    )
    return rotation, scale, translation, valid


def _apply_similarity(rotation: Tensor, scale: Tensor, translation: Tensor, points: Tensor) -> Tensor:
    return scale[..., None, None] * torch.einsum("bvik,bvpk->bvpi", rotation, points) + translation[..., None, :]


def recover_positioned_mesh(
    triangulated_joints_3d: Tensor,
    prior_joints_3d: Tensor,
    prior_vertices_3d: Tensor,
    joint_view_weights: Tensor,
    triangulation_valid: Tensor,
    *,
    geometry_blend: float,
    iterations: int = 3,
    huber_delta_m: float = 0.012,
    eps: float = 1e-8,
) -> MeshRecoveryResult:
    """Align each local prior to reference-frame joints, then fuse views.

    Inputs are ``[B,J,3]``, ``[B,V,J,3]``, ``[B,V,M,3]`` and
    ``[B,V,J]``. Similarity alignment includes rotation, scale and translation.
    """
    if triangulated_joints_3d.ndim != 3 or triangulated_joints_3d.shape[-1] != 3:
        raise ValueError("triangulated joints must have shape [B,J,3]")
    b, j = triangulated_joints_3d.shape[:2]
    if prior_joints_3d.ndim != 4 or prior_joints_3d.shape[0] != b or prior_joints_3d.shape[2:] != (j, 3):
        raise ValueError("prior joints must have shape [B,V,J,3]")
    v = prior_joints_3d.shape[1]
    if prior_vertices_3d.ndim != 4 or prior_vertices_3d.shape[:2] != (b, v) or prior_vertices_3d.shape[-1] != 3:
        raise ValueError("prior vertices must have shape [B,V,M,3]")
    if joint_view_weights.shape != (b, v, j):
        raise ValueError("joint view weights must have shape [B,V,J]")
    if triangulation_valid.shape != (b, j) or triangulation_valid.dtype != torch.bool:
        raise ValueError("triangulation_valid must be Boolean [B,J]")
    if not 0 <= geometry_blend <= 1:
        raise ValueError("geometry_blend must be in [0,1]")
    tensors = (triangulated_joints_3d, prior_joints_3d, prior_vertices_3d, joint_view_weights)
    if any(not value.is_floating_point() or not torch.isfinite(value).all().item() for value in tensors):
        raise ValueError("mesh recovery inputs must be finite floating tensors")
    if any(value.device != triangulated_joints_3d.device or value.dtype != triangulated_joints_3d.dtype for value in tensors[1:]):
        raise ValueError("mesh recovery tensors must match dtype and device")
    if triangulation_valid.device != triangulated_joints_3d.device:
        raise ValueError("triangulation_valid must share the input device")
    target = triangulated_joints_3d[:, None].expand(-1, v, -1, -1)
    robust_weights = joint_view_weights * triangulation_valid[:, None].to(joint_view_weights.dtype)
    rotation = scale = translation = valid = None
    aligned_joints = prior_joints_3d
    for _ in range(iterations):
        rotation, scale, translation, valid = _weighted_similarity(prior_joints_3d, target, robust_weights, eps)
        aligned_joints = _apply_similarity(rotation, scale, translation, prior_joints_3d)
        distance = torch.linalg.vector_norm(aligned_joints - target, dim=-1)
        huber = torch.where(distance <= huber_delta_m, torch.ones_like(distance), huber_delta_m / distance.clamp_min(eps))
        robust_weights = joint_view_weights * triangulation_valid[:, None].to(joint_view_weights.dtype) * huber
    assert rotation is not None and scale is not None and translation is not None and valid is not None
    aligned_vertices = _apply_similarity(rotation, scale, translation, prior_vertices_3d)
    positioned_finite = torch.isfinite(aligned_joints).all(dim=(-2, -1)) & torch.isfinite(aligned_vertices).all(dim=(-2, -1))
    valid = valid & positioned_finite
    residual = torch.linalg.vector_norm(aligned_joints - target, dim=-1).mean(dim=-1)
    reliability = joint_view_weights.mean(dim=-1) / (residual + 1e-3)
    reliability = torch.where(valid, reliability, torch.zeros_like(reliability))
    total = reliability.sum(dim=1, keepdim=True)
    uniform_valid = valid.to(reliability.dtype)
    fallback = uniform_valid / uniform_valid.sum(dim=1, keepdim=True).clamp_min(1)
    fusion_weights = torch.where(total > eps, reliability / total.clamp_min(eps), fallback)
    positioned_joints = torch.einsum("bv,bvjc->bjc", fusion_weights, aligned_joints)
    final_vertices = torch.einsum("bv,bvmc->bmc", fusion_weights, aligned_vertices)
    final_joints = geometry_blend * triangulated_joints_3d + (1 - geometry_blend) * positioned_joints
    final_valid = triangulation_valid & valid.any(dim=1, keepdim=True)
    invalid = torch.full((), float("nan"), dtype=final_joints.dtype, device=final_joints.device)
    positioned_joints = torch.where(final_valid[..., None], positioned_joints, invalid)
    final_joints = torch.where(final_valid[..., None], final_joints, invalid)
    mesh_valid = final_valid.all(dim=-1)
    final_vertices = torch.where(mesh_valid[:, None, None], final_vertices, invalid)
    return MeshRecoveryResult(
        triangulated_joints_3d,
        aligned_joints,
        aligned_vertices,
        positioned_joints,
        final_joints,
        final_vertices,
        valid,
        residual,
        fusion_weights,
        final_valid,
    )
