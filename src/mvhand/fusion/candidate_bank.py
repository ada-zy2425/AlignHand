"""Single-prior candidate packing for the unchanged regional scorer and fusion."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor

from mvhand.fusion.candidates import (
    REGIONS,
    build_candidate_geometry,
    fit_standardize,
    metric_joints,
    native_visual_region_descriptor,
    project_points,
)
from mvhand.fusion.regional_fusion import fuse_anchor_candidates
from mvhand.observation.model import invert_crop_to_raw


@dataclass(frozen=True)
class SinglePriorBank:
    anchor: Tensor
    candidates: Tensor
    valid: Tensor
    q_init: Tensor
    origins: Tensor
    source_tags: Tensor


def build_single_prior_bank(anchor: Tensor, placed: Tensor, valid: Tensor) -> SinglePriorBank:
    b, views = valid.shape
    if placed.shape != (b, views, 778, 3) or anchor.shape != (b, 778, 3):
        raise ValueError("single-prior mesh shapes disagree")
    if valid.dtype != torch.bool or not valid.any(1).all():
        raise ValueError("no placed prior: this group must be recorded as an output failure")
    if not torch.isfinite(anchor).all() or not torch.isfinite(placed[valid]).all():
        raise ValueError("a valid candidate must have finite vertices")
    clean = torch.where(valid[..., None, None], placed, anchor[:, None])
    candidates = torch.cat((anchor[:, None], clean), 1)
    candidate_valid = torch.cat((torch.ones_like(valid[:, :1]), valid), 1)
    q = anchor.new_zeros((b, REGIONS, 1 + views), dtype=torch.float64)
    q[..., 0] = 0.90
    q[..., 1:] = 0.10 * valid.double()[:, None] / valid.sum(1)[:, None, None]
    origins = torch.arange(-1, views, device=anchor.device)
    # Keep the old scorer's input width; the unused secondary tag is always zero.
    source = anchor.new_zeros((1 + views, 3), dtype=torch.float64)
    source[0, 0] = 1
    source[1:, 1] = 1
    return SinglePriorBank(anchor, candidates, candidate_valid, q, origins, source)


def raw_to_feature_grid(raw: Tensor, raw_to_crop: Tensor, *, feature_crop_bounds) -> Tensor:
    """Map raw pixels to the actual prior backbone input rectangle in crop pixels."""
    left, top, width, height = map(float, feature_crop_bounds)
    if width <= 0 or height <= 0:
        raise ValueError("feature footprint must have positive width and height")
    homogeneous = torch.cat((raw, torch.ones_like(raw[..., :1])), -1)
    crop = torch.einsum('bvrd,bcvjd->bcvjr', raw_to_crop.double(), homogeneous.double())
    offset = crop.new_tensor((left, top))
    size = crop.new_tensor((width, height))
    return 2 * (crop - offset) / size - 1


def assemble_single_prior_inputs(prediction, recovery, batch, *, basis, regressor,
                                 feature_crop_bounds, candidate_bank=None,
                                 candidate_placement_residual=None,candidate_placement_alpha=None):
    """Construct geometric/image-evidence sampling inputs; accepts no targets."""
    anchor = recovery.final_vertices_3d.double()
    available = batch.runtime_view_mask.bool()
    placed_valid = recovery.alignment_valid.bool() & available
    if not recovery.valid.all():
        raise ValueError("invalid triangulation/placement: record an output failure")
    bank = (build_single_prior_bank(anchor, recovery.aligned_prior_vertices_3d.double(), placed_valid)
            if candidate_bank is None else candidate_bank)
    if not torch.equal(bank.anchor,anchor):raise ValueError('candidate additions must preserve the same anchor')
    if candidate_bank is not None and any(x is None or x.shape!=bank.valid.shape
            for x in (candidate_placement_residual,candidate_placement_alpha)):
        raise ValueError('each added candidate needs matched placement diagnostics')
    residual = recovery.alignment_residual_m.double()
    alpha = recovery.fusion_view_weights.double()
    geometry, geometry_valid, raw, projection_valid, diagnostics = build_candidate_geometry(
        candidates=bank.candidates, l88=anchor, r83=anchor,
        placement_residual=(torch.cat((torch.zeros_like(residual[:, :1]), residual), 1)
                            if candidate_bank is None else candidate_placement_residual),
        placement_alpha=(torch.cat((torch.zeros_like(alpha[:, :1]), alpha), 1)
                         if candidate_bank is None else candidate_placement_alpha),
        candidate_valid=bank.valid, origins=bank.origins,
        xgeo=prediction.weighted.points_3d.double(),
        observations_raw=prediction.corrected_joints_2d_raw.double(),
        projection_matrices=prediction.projection_matrices.double(),
        joint_view_weights=prediction.relative_view_weights.double(),
        runtime_view_mask=available, reference_to_camera=prediction.reference_to_camera.double(),
        basis=basis, regressor=regressor,
    )
    affine = batch.raw_to_crop if batch.raw_to_crop is not None else invert_crop_to_raw(batch.crop_to_raw)
    grid = raw_to_feature_grid(raw, affine, feature_crop_bounds=feature_crop_bounds)
    projection_valid &= torch.isfinite(grid).all(-1) & (grid.abs() <= 1).all(-1)
    anchor_raw, anchor_depth = project_points(
        prediction.projection_matrices, metric_joints(anchor[:, None], regressor))
    anchor_grid = raw_to_feature_grid(anchor_raw, affine, feature_crop_bounds=feature_crop_bounds)
    anchor_valid = (anchor_depth[:, 0] > 0) & available[..., None]
    anchor_valid &= torch.isfinite(anchor_grid[:, 0]).all(-1) & (anchor_grid[:, 0].abs() <= 1).all(-1)
    return dict(bank=bank, geometry_raw=geometry, geometry_valid=geometry_valid,
                candidate_raw=raw, candidate_grid=grid, projection_valid=projection_valid,
                anchor_raw=anchor_raw[:, 0], anchor_grid=anchor_grid[:, 0],
                anchor_valid=anchor_valid, view_valid=available), diagnostics


def forward_single_prior(head, basis, inputs, feature_maps, *, mean, scale):
    bank = inputs['bank']
    geometry = fit_standardize(inputs['geometry_raw'], inputs['geometry_valid'], mean, scale)
    visual = native_visual_region_descriptor(
        head, feature_maps, inputs['candidate_grid'], inputs['candidate_raw'],
        inputs['projection_valid'], inputs['anchor_grid'], inputs['anchor_raw'],
        inputs['anchor_valid'], inputs['view_valid'])
    return fuse_anchor_candidates(
        head=head, anchor=bank.anchor, candidates=bank.candidates,
        geometry=geometry, basis=basis, q_init=bank.q_init, candidate_valid=bank.valid,
        family='P', secondary_valid=torch.zeros_like(inputs['view_valid']),
        visual_enabled=True, visual_descriptor=visual, enforce_secondary_fallback=False,
        candidate_source_tags=bank.source_tags,
    )
