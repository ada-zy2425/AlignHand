from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mvhand.geometry.plucker import reference_plucker_features
from mvhand.geometry.camera import dehomogenize
from mvhand.geometry.raw_pixel_dlt import (
    ProjectiveTriangulationResult,
    build_projection_matrices,
    build_reference_to_camera,
    normalize_relative_weights,
    triangulate_projective_dlt,
)
from mvhand.models.interfaces import (
    Form2FrameBatch,
    HandPrior,
    HandPriorOutput,
    validate_inference_batch,
    validate_prior_output,
)


def apply_crop_to_raw(affine: Tensor, crop_points: Tensor) -> Tensor:
    if affine.shape != (*crop_points.shape[:2], 2, 3):
        raise ValueError("crop_to_raw must match [B,V,2,3]")
    homogeneous = torch.cat((crop_points, torch.ones_like(crop_points[..., :1])), dim=-1)
    return torch.einsum("bvrc,bvjc->bvjr", affine, homogeneous)


def normalized_crop_to_raw(points: Tensor, crop_to_raw: Tensor, patch_size: int) -> Tensor:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    return apply_crop_to_raw(crop_to_raw, (points + 0.5) * float(patch_size))


def invert_crop_to_raw(crop_to_raw: Tensor) -> Tensor:
    if crop_to_raw.ndim != 4 or crop_to_raw.shape[-2:] != (2, 3):
        raise ValueError("crop_to_raw must have shape [B,V,2,3]")
    bottom = torch.zeros((*crop_to_raw.shape[:2], 1, 3), dtype=crop_to_raw.dtype, device=crop_to_raw.device)
    bottom[..., 0, 2] = 1
    homogeneous = torch.cat((crop_to_raw, bottom), dim=-2)
    return torch.linalg.inv(homogeneous)[..., :2, :]


def raw_to_normalized_crop(points: Tensor, raw_to_crop: Tensor, patch_size: int) -> Tensor:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    return apply_crop_to_raw(raw_to_crop, points) / float(patch_size) - 0.5




def plucker_pair_bias(directions: Tensor, moments: Tensor) -> Tensor:
    d = directions.permute(0, 2, 1, 3)
    m = moments.permute(0, 2, 1, 3)
    reciprocal = torch.abs(
        (d[:, :, :, None] * m[:, :, None]).sum(-1)
        + (d[:, :, None] * m[:, :, :, None]).sum(-1)
    )
    scale = torch.median(reciprocal.detach().flatten(-2), dim=-1).values.clamp_min(1e-5)
    return -(reciprocal / scale[..., None, None]).clamp(0, 12)


class CrossViewEvidence(nn.Module):
    def __init__(self, feature_dim: int, joints: int, max_views: int, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("evidence dimension must be divisible by heads")
        self.joints = joints
        self.max_views = max_views
        self.heads = heads
        self.head_dim = dim // heads
        self.visual = nn.Linear(feature_dim, dim)
        self.geometry = nn.Linear(8, dim)
        self.joint_embedding = nn.Embedding(joints, dim)
        self.view_embedding = nn.Embedding(max_views, dim)
        # Runtime camera sorting must retain the camera identity learned by each
        # checkpoint slot. This adapter metadata never changes checkpoint weights.
        self.register_buffer("runtime_slot_to_training_slot", torch.arange(max_views), persistent=False)
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
        )

    def bind_camera_order(self, runtime_camera_ids, training_camera_ids) -> None:
        runtime = tuple(runtime_camera_ids)
        training = tuple(training_camera_ids)
        if (len(runtime) != self.max_views or len(training) != self.max_views
                or len(set(runtime)) != self.max_views or set(runtime) != set(training)):
            raise ValueError("runtime and parent training camera identities must be the same unique complete set")
        mapping = torch.tensor([training.index(camera) for camera in runtime],
                               device=self.runtime_slot_to_training_slot.device)
        self.runtime_slot_to_training_slot.copy_(mapping)

    def forward(self, visual: Tensor, points_norm: Tensor, directions: Tensor, moments: Tensor, visibility: Tensor) -> Tensor:
        b, v, j, _ = visual.shape
        if j != self.joints or v > self.max_views or visibility.shape != (b, v, j):
            raise ValueError("cross-view evidence shape contract failed")
        geometry = torch.cat((points_norm, directions, moments), dim=-1)
        token = self.visual(visual).permute(0, 2, 1, 3) + self.geometry(geometry).permute(0, 2, 1, 3)
        token = token + self.joint_embedding(torch.arange(j, device=token.device))[None, :, None]
        token = token + self.view_embedding(self.runtime_slot_to_training_slot[:v])[None, None]
        z = self.norm(token)
        shape = (b, j, v, self.heads, self.head_dim)
        query = self.query(z).reshape(shape)
        key = self.key(z).reshape(shape)
        value = self.value(z).reshape(shape)
        scores = torch.einsum("bjvhd,bjuhd->bjhvu", query, key) * self.head_dim ** -0.5
        scores = scores + plucker_pair_bias(directions, moments)[:, :, None]
        key_valid = visibility.permute(0, 2, 1)[:, :, None, None]
        scores = scores.masked_fill(~key_valid, float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        fused = torch.einsum("bjhvu,bjuhd->bjvhd", attention, value).reshape(b, j, v, -1)
        fused = token + self.dropout(self.output(fused))
        fused = fused + self.dropout(self.feed_forward(self.feed_forward_norm(fused)))
        return fused * visibility.permute(0, 2, 1)[..., None]


@dataclass
class Form2FramePrediction:
    prior: HandPriorOutput
    reference_to_camera: Tensor
    projection_matrices: Tensor
    corrected_joints_2d_raw: Tensor
    corrected_joints_2d_normalized: Tensor
    correction_normalized: Tensor
    relative_view_logits: Tensor
    relative_view_weights: Tensor
    uniform: ProjectiveTriangulationResult
    weighted: ProjectiveTriangulationResult
    fused_evidence: Tensor


class Form2FrameModel(nn.Module):
    """Canonical neural core with an injected hand prior and projective geometry."""

    def __init__(
        self,
        prior: HandPrior,
        *,
        feature_dim: int,
        joints: int,
        max_views: int,
        evidence_dim: int,
        attention_heads: int,
        dropout: float,
        correction_limit: float,
        weight_correction_scale: float,
        residual_scale_px: float,
        residual_temperature: float,
        patch_size: int,
        parameterization: str = "PIXEL",
        evidence: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if correction_limit <= 0 or residual_scale_px <= 0 or residual_temperature <= 0:
            raise ValueError("correction and residual scales must be positive")
        self.prior = prior
        self.patch_size = patch_size
        self.correction_limit = float(correction_limit)
        self.weight_correction_scale = float(weight_correction_scale)
        self.residual_denominator = float(residual_scale_px * residual_temperature)
        # Evaluation-only single-factor intervention.  The default is the
        # executed model.  Alternate modes retain the same observations and
        # geometry while removing only learned weight logits or all
        # non-uniform view weighting.
        self.view_weight_mode = "full"
        if parameterization != "PIXEL":
            raise ValueError("only normalized-crop pixel correction is supported")
        self.evidence = (CrossViewEvidence(feature_dim, joints, max_views, evidence_dim, attention_heads, dropout)
                         if evidence is None else evidence)
        self.correction_head = nn.Sequential(nn.LayerNorm(evidence_dim), nn.Linear(evidence_dim, evidence_dim), nn.GELU(), nn.Linear(evidence_dim, 2))
        self.weight_head = nn.Sequential(nn.LayerNorm(evidence_dim), nn.Linear(evidence_dim, max(1, evidence_dim // 2)), nn.GELU(), nn.Linear(max(1, evidence_dim // 2), 1))
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)
        nn.init.zeros_(self.weight_head[-1].weight)
        nn.init.zeros_(self.weight_head[-1].bias)

    def forward(self, normalized_crops: Tensor, batch: Form2FrameBatch) -> Form2FramePrediction:
        validate_inference_batch(batch)
        if normalized_crops.shape != batch.crops.shape or normalized_crops.device != batch.crops.device:
            raise ValueError("normalized crops must match the batch crop tensor")
        prior = self.prior(normalized_crops)
        return self.forward_from_prior(prior, batch)

    def _fuse_evidence(self, prior, directions, moments, solver_mask, reference_to_camera):
        return self.evidence(prior.visual_features, prior.joints_2d_normalized,
                             directions, moments, solver_mask)

    def forward_from_prior(self, prior: HandPriorOutput, batch: Form2FrameBatch) -> Form2FramePrediction:
        """Run the unchanged observation/geometry chain from a frozen prior cache."""
        validate_inference_batch(batch)
        validate_prior_output(prior, batch)
        reference_to_camera = build_reference_to_camera(batch.camera_to_world)
        coarse_raw = normalized_crop_to_raw(prior.joints_2d_normalized, batch.crop_to_raw, self.patch_size)
        directions, moments = reference_plucker_features(coarse_raw, batch.intrinsics, reference_to_camera)
        solver_mask = batch.runtime_view_mask[:, :, None].expand(-1, -1, prior.joints_2d_normalized.shape[2])
        fused = self._fuse_evidence(prior, directions, moments, solver_mask, reference_to_camera)
        head_tanh = torch.tanh(self.correction_head(fused)).permute(0, 2, 1, 3)
        desired_correction = self.correction_limit * head_tanh
        corrected_norm_direct = prior.joints_2d_normalized + desired_correction
        corrected_raw = normalized_crop_to_raw(corrected_norm_direct, batch.crop_to_raw, self.patch_size)
        raw_to_crop = batch.raw_to_crop if batch.raw_to_crop is not None else invert_crop_to_raw(batch.crop_to_raw)
        corrected_norm = raw_to_normalized_crop(corrected_raw, raw_to_crop, self.patch_size)
        projection = build_projection_matrices(batch.intrinsics, reference_to_camera)
        # Softmax is promoted to FP32 by CUDA autocast. DLT requires one explicit
        # numerical dtype, so bring normalized weights back to the geometry dtype.
        uniform_weights = normalize_relative_weights(
            torch.zeros_like(solver_mask, dtype=corrected_raw.dtype),
            solver_mask,
        ).to(corrected_raw.dtype)
        uniform = triangulate_projective_dlt(corrected_raw, projection, uniform_weights, solver_mask)
        homogeneous = torch.cat((uniform.points_3d, torch.ones_like(uniform.points_3d[..., :1])), dim=-1)
        projected = torch.einsum("bvrc,bjc->bvjr", projection, homogeneous)
        projected = dehomogenize(projected, eps=1e-12)
        residual = torch.linalg.vector_norm(projected - corrected_raw, dim=-1)
        learned = self.weight_head(fused).squeeze(-1).permute(0, 2, 1)
        mode = self.view_weight_mode
        if mode == "full":
            logits = -residual / self.residual_denominator + self.weight_correction_scale * learned
            weights = normalize_relative_weights(logits, solver_mask).to(corrected_raw.dtype)
        elif mode == "geometry_only":
            logits = -residual / self.residual_denominator
            weights = normalize_relative_weights(logits, solver_mask).to(corrected_raw.dtype)
        elif mode == "uniform_valid":
            logits = torch.zeros_like(residual)
            weights = uniform_weights
        else:
            raise ValueError(f"unsupported view_weight_mode: {mode}")
        weighted = triangulate_projective_dlt(corrected_raw, projection, weights, solver_mask)
        return Form2FramePrediction(
            prior=prior,
            reference_to_camera=reference_to_camera,
            projection_matrices=projection,
            corrected_joints_2d_raw=corrected_raw,
            corrected_joints_2d_normalized=corrected_norm,
            correction_normalized=desired_correction,
            relative_view_logits=logits,
            relative_view_weights=weights,
            uniform=uniform,
            weighted=weighted,
            fused_evidence=fused,
        )
