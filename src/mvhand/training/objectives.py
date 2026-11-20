from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from mvhand.geometry.camera import dehomogenize
from mvhand.models.form2frame import raw_to_normalized_crop


ENTROPY_FLOOR_NATS = 1.2


def masked_mean(values: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(eps)


def crop_supported_supervision_mask(
    joints_2d_crop: Tensor, supervision_mask: Tensor, *, patch_size: int = 256,
) -> Tensor:
    """Keep finite 2D annotations that are actually supported by the crop."""
    if joints_2d_crop.shape != (*supervision_mask.shape, 2):
        raise ValueError("crop joints and supervision mask shapes disagree")
    if supervision_mask.dtype != torch.bool:
        raise TypeError("supervision_mask must be Boolean")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    supported = torch.isfinite(joints_2d_crop).all(dim=-1)
    supported = supported & (joints_2d_crop >= 0).all(dim=-1)
    supported = supported & (joints_2d_crop < patch_size).all(dim=-1)
    return supervision_mask & supported


def form2frame_training_loss(
    prediction: Any,
    *,
    gt_joints_3d: Tensor,
    raw_to_crop: Tensor,
    gt_joints_2d_raw: Tensor | None = None,
    gt_joints_2d_crop: Tensor | None = None,
    visibility: Tensor | None = None,
    supervision_mask: Tensor | None = None,
    view_availability_mask: Tensor | None = None,
    entropy_target_fraction: float = 1.0,
    patch_size: int = 256,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Recovered Form2Frame objective in metres and normalized crop space."""
    x_metric = prediction.weighted.points_3d
    if gt_joints_3d.shape != x_metric.shape:
        raise ValueError("reference-camera GT joints must match X_metric")
    if visibility is not None and supervision_mask is not None:
        raise ValueError("provide visibility or supervision_mask, not both")
    mask = supervision_mask if supervision_mask is not None else visibility
    if mask is None:
        raise ValueError("2D supervision mask is required")
    if mask.shape != prediction.relative_view_weights.shape:
        raise ValueError("2D supervision mask must match [B,V,J]")
    if supervision_mask is not None and mask.dtype != torch.bool:
        raise TypeError("supervision_mask must be Boolean")
    mask = mask.bool()
    if raw_to_crop.shape != (*mask.shape[:2], 2, 3):
        raise ValueError("raw_to_crop must have shape [B,V,2,3]")
    if (gt_joints_2d_raw is None) == (gt_joints_2d_crop is None):
        raise ValueError("provide exactly one raw-space or crop-space 2D target")
    if gt_joints_2d_crop is None:
        if gt_joints_2d_raw.shape != (*mask.shape, 2):
            raise ValueError("gt_joints_2d_raw must have shape [B,V,J,2]")
        target_h = torch.cat(
            (gt_joints_2d_raw, torch.ones_like(gt_joints_2d_raw[..., :1])), dim=-1
        )
        target_crop = torch.einsum("bvrc,bvjc->bvjr", raw_to_crop, target_h)
    else:
        if gt_joints_2d_crop.shape != (*mask.shape, 2):
            raise ValueError("gt_joints_2d_crop must have shape [B,V,J,2]")
        target_crop = gt_joints_2d_crop
    gt_normalized = target_crop / float(patch_size) - 0.5
    if hasattr(prediction, "corrected_joints_2d_raw"):
        corrected = raw_to_normalized_crop(
            prediction.corrected_joints_2d_raw, raw_to_crop, patch_size
        )
        if not torch.allclose(
            corrected, prediction.corrected_joints_2d_normalized,
            rtol=0.0, atol=2 * torch.finfo(corrected.dtype).eps,
        ):
            raise ValueError("prediction corrected normalized coordinate contract failed")
        correction_equivalent = corrected - prediction.prior.joints_2d_normalized
    else:
        corrected = (
            prediction.prior.joints_2d_normalized
            + prediction.correction_normalized
        )
        correction_equivalent = prediction.correction_normalized

    def masked_smooth(actual: Tensor, expected: Tensor) -> Tensor:
        value = F.smooth_l1_loss(actual, expected, beta=0.02, reduction="none").mean(-1)
        return masked_mean(value, mask)

    smooth_3d = F.smooth_l1_loss(x_metric, gt_joints_3d, beta=0.01)
    euclidean_3d = torch.linalg.vector_norm(x_metric - gt_joints_3d, dim=-1).mean()
    observation_2d = masked_smooth(corrected, gt_normalized)
    homogeneous = torch.cat((x_metric, torch.ones_like(x_metric[..., :1])), dim=-1)
    projected = torch.einsum("bvrc,bjc->bvjr", prediction.projection_matrices, homogeneous)
    projected_raw = dehomogenize(projected, eps=1e-12)
    projected_h = torch.cat((projected_raw, torch.ones_like(projected_raw[..., :1])), dim=-1)
    projected_crop = torch.einsum("bvrc,bvjc->bvjr", raw_to_crop, projected_h)
    reprojection = masked_smooth(projected_crop / float(patch_size) - 0.5, corrected)
    weights = prediction.relative_view_weights
    if view_availability_mask is None:
        view_availability_mask = torch.ones(
            weights.shape[:2], dtype=torch.bool, device=weights.device,
        )
    if view_availability_mask.shape != mask.shape[:2]:
        raise ValueError("view_availability_mask must have shape [B,V]")
    if view_availability_mask.dtype != torch.bool:
        raise TypeError("view_availability_mask must be Boolean")
    if view_availability_mask.device != weights.device:
        raise ValueError("view_availability_mask must share the prediction device")
    if torch.any(view_availability_mask.sum(dim=1) == 0):
        raise ValueError("each sample must have at least one runtime-available view")
    if not 0.0 < entropy_target_fraction <= 1.0:
        raise ValueError("entropy_target_fraction must be in (0, 1]")
    available = view_availability_mask[:, :, None].expand_as(weights)
    if torch.any(weights.masked_select(~available) != 0):
        raise ValueError("unavailable views must have exactly zero relative weight")
    safe_weights = weights.clamp_min(torch.finfo(weights.dtype).tiny)
    entropy = -(torch.where(available, weights * safe_weights.log(), 0.0)).sum(dim=1)
    available_count = available.sum(dim=1)
    entropy_target = torch.minimum(
        entropy.new_full(entropy.shape, ENTROPY_FLOOR_NATS),
        entropy_target_fraction * available_count.to(entropy.dtype).clamp_min(1).log(),
    )
    entropy_active = available_count > 1
    confidence = masked_mean(torch.relu(entropy_target - entropy).square(), entropy_active)
    correction = correction_equivalent.square().mean()
    components = {
        "loss_3d_smooth": smooth_3d,
        "loss_3d_euclidean": euclidean_3d,
        "loss_observation_2d": observation_2d,
        "loss_multiview": reprojection,
        "loss_confidence": confidence,
        "entropy_target_mean": masked_mean(entropy_target, entropy_active),
        "entropy_active_elements": entropy_active.sum(),
        "runtime_available_views_mean": available_count.to(entropy.dtype).mean(),
        "loss_correction": correction,
        "supervised_2d_elements": mask.sum(),
    }
    total = (
        smooth_3d + 0.25 * euclidean_3d + 0.5 * observation_2d
        + 0.25 * reprojection + 0.01 * confidence + 0.01 * correction
    )
    return total, components
