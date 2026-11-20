"""V2 Stage I: camera geometry without device identities or a view-count limit."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from mvhand.models.form2frame import Form2FrameModel


def relative_pose_encoding(reference_to_camera: Tensor, available: Tensor) -> Tensor:
    """Camera-to-reference rotation columns (6D) and scale-normalized centers.

    The distance mean includes the reference camera (distance zero) and excludes
    unavailable padding. Calibration defines the frame even if its image is missing.
    """
    if reference_to_camera.shape != (*available.shape, 4, 4):
        raise ValueError("pose/mask shapes disagree")
    if available.dtype != torch.bool or not available.any(1).all():
        raise ValueError("need a Boolean mask with at least one available camera")
    rotation = reference_to_camera[..., :3, :3].transpose(-1, -2)
    center = -(rotation @ reference_to_camera[..., :3, 3:4]).squeeze(-1)
    distance = torch.linalg.vector_norm(center, dim=-1)
    scale = torch.where(available, distance, 0).sum(1) / available.sum(1)
    scale = scale.clamp_min(torch.finfo(center.dtype).eps)
    rotation6 = rotation[..., :2].transpose(-1, -2).flatten(-2)
    pose = torch.cat((rotation6, center / scale[:, None, None]), -1)
    return torch.where(available[..., None], pose, 0)


def masked_plucker_bias(directions: Tensor, moments: Tensor, known: Tensor) -> Tensor:
    """The existing reciprocal-ray bias, normalized using available pairs only."""
    d, m = directions.transpose(1, 2), moments.transpose(1, 2)
    reciprocal = ((d[:, :, :, None] * m[:, :, None]).sum(-1)
                  + (d[:, :, None] * m[:, :, :, None]).sum(-1)).abs()
    valid = known.transpose(1, 2)
    pairs = valid[..., :, None] & valid[..., None, :]
    samples = reciprocal.detach().masked_fill(~pairs, float("nan")).flatten(-2)
    scale = torch.nanmedian(samples, dim=-1).values.nan_to_num(1e-5).clamp_min(1e-5)
    bias = -(reciprocal / scale[..., None, None]).clamp(0, 12)
    return torch.where(pairs, bias, 0)


class GeometryCrossViewEvidence(nn.Module):
    def __init__(self, feature_dim: int, joints: int = 21, dim: int = 128,
                 heads: int = 4, dropout: float = 0.0, *, variant: str = "M1"):
        super().__init__()
        if variant not in ("M1", "M2", "M3") or dim % heads:
            raise ValueError("invalid v2 variant or attention dimensions")
        self.variant, self.joints, self.heads = variant, joints, heads
        self.head_dim = dim // heads
        self.visual = nn.Linear(feature_dim, dim)
        self.geometry = nn.Linear(2 + 6 + 9, dim)
        self.joint_embedding = nn.Embedding(joints, dim)
        if variant == "M3":
            self.missing_camera = nn.Parameter(torch.zeros(15))
        else:
            self.register_parameter("missing_camera", None)
        self.norm = nn.LayerNorm(dim)
        self.query, self.key, self.value = (nn.Linear(dim, dim) for _ in range(3))
        self.output = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(),
                                          nn.Dropout(dropout), nn.Linear(2 * dim, dim))

    def forward(self, visual: Tensor, points_norm: Tensor, directions: Tensor,
                moments: Tensor, visibility: Tensor, reference_to_camera: Tensor,
                *, camera_information_mask: Tensor | None = None) -> Tensor:
        b, v, j, _ = visual.shape
        if j != self.joints or visibility.shape != (b, v, j) or visibility.dtype != torch.bool:
            raise ValueError("cross-view evidence shape/mask contract failed")
        if not visibility.any(1).all():
            raise ValueError("each joint needs at least one available input view")
        available = visibility.any(-1)
        pose = relative_pose_encoding(reference_to_camera, available)
        camera = torch.cat((directions, moments, pose[:, :, None].expand(-1, -1, j, -1)), -1)
        if camera_information_mask is None:
            known = available.clone()
            if self.training and self.variant == "M3":
                known &= torch.rand((b, v), device=visual.device) >= 0.1
        else:
            if camera_information_mask.shape != (b, v) or camera_information_mask.dtype != torch.bool:
                raise ValueError("camera information mask must be Boolean [B,V]")
            known = available & camera_information_mask
        if self.variant != "M3" and torch.any(available & ~known):
            raise ValueError("only M3 accepts missing camera descriptors")
        if self.missing_camera is not None:
            camera = torch.where(known[..., None, None], camera, self.missing_camera)
        geometry = torch.cat((points_norm, camera), -1)
        # Frozen-prior grid sampling and NPZ loads can have different strides.
        # Choose one layout before GEMM so cached and RGB v2 inference use the
        # same TF32 arithmetic. This does not alter frozen L0 compatibility.
        token = (self.visual(visual.contiguous()) + self.geometry(geometry)).transpose(1, 2)
        token = token + self.joint_embedding(torch.arange(j, device=visual.device))[None, :, None]
        z = self.norm(token)
        shape = (b, j, v, self.heads, self.head_dim)
        query, key, value = (layer(z).reshape(shape) for layer in (self.query, self.key, self.value))
        scores = torch.einsum("bjvhd,bjuhd->bjhvu", query, key) * self.head_dim ** -0.5
        # A hidden camera must not leak back through the pairwise ray bias.
        bias = masked_plucker_bias(directions, moments, visibility & known[..., None])
        scores = scores + bias[:, :, None]
        scores = scores.masked_fill(~visibility.transpose(1, 2)[:, :, None, None], float("-inf"))
        attention = torch.softmax(scores, -1)
        fused = torch.einsum("bjhvu,bjuhd->bjvhd", attention, value).reshape(b, j, v, -1)
        fused = token + self.dropout(self.output(fused))
        fused = fused + self.dropout(self.feed_forward(self.feed_forward_norm(fused)))
        return fused * visibility.transpose(1, 2)[..., None]


class V2Form2FrameModel(Form2FrameModel):
    def __init__(self, prior, *, feature_dim: int, variant: str, joints: int = 21,
                 evidence_dim: int = 128, attention_heads: int = 4, dropout: float = 0.0,
                 **kwargs):
        evidence = GeometryCrossViewEvidence(feature_dim, joints, evidence_dim,
                                              attention_heads, dropout, variant=variant)
        super().__init__(prior, feature_dim=feature_dim, joints=joints, max_views=0,
                         evidence_dim=evidence_dim, attention_heads=attention_heads,
                         dropout=dropout, evidence=evidence, **kwargs)
        self.variant = variant
        if isinstance(self.prior, nn.Module):
            self.prior.eval().requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if isinstance(self.prior, nn.Module):
            self.prior.eval()
        return self

    def _fuse_evidence(self, prior, directions, moments, solver_mask, reference_to_camera):
        return self.evidence(prior.visual_features, prior.joints_2d_normalized,
                             directions, moments, solver_mask, reference_to_camera)
