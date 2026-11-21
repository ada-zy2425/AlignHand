#!/usr/bin/env python3
"""Seventeen-slot complementary-candidate fusion and evidence features."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mvhand.geometry.triangulation import triangulate_projective_dlt
from mvhand.geometry.mesh_joints import poem_published_mesh_joints_21_ieee_fp32
from mvhand.fusion.regions import REGION_JOINTS, REGION_NAMES
from mvhand.fusion.feature_stats import safe_standardize
from mvhand.pipeline.precision import READER_VERSION, canonical_feature_maps


VIEWS = 8
CANDIDATES = 17
REGIONS = 6
JOINTS = 21
VERTICES = 778
SLOTS = 6
RAW_VISUAL_DIM = 1280
VISUAL_DIM = 16
SUMMARY_NAMES = (
    "joint_weight_mean", "joint_weight_std", "joint_weight_min", "joint_weight_max",
    "candidate_minus_xgeo_x_10mm", "candidate_minus_xgeo_y_10mm", "candidate_minus_xgeo_z_10mm",
    "candidate_minus_xgeo_norm_mean_10mm", "candidate_minus_xgeo_norm_std_10mm", "candidate_minus_xgeo_norm_max_10mm",
    "candidate_minus_r83_x_10mm", "candidate_minus_r83_y_10mm", "candidate_minus_r83_z_10mm",
    "candidate_minus_r83_norm_mean_10mm", "candidate_minus_r83_norm_std_10mm", "candidate_minus_r83_norm_max_10mm",
    "candidate_minus_l88_norm_mean_10mm", "candidate_minus_l88_norm_std_10mm", "candidate_minus_l88_norm_max_10mm",
    "placement_residual_10mm", "placement_alpha", "log_placement_alpha",
    "camera_ray_pair_cosine_mean", "camera_ray_pair_cosine_min", "camera_distance_100mm", "candidate_valid",
)
SEMANTIC_FIELDS = (
    "candidate_to_xgeo_mm",
    "candidate_to_r83_joint_mm",
    "candidate_to_l88_joint_mm",
    "candidate_to_leave_one_equation_mm",
    "cross_reprojection_mean_raw_px",
    "cross_reprojection_max_raw_px",
    "origin_reprojection_raw_px",
    "local_bone_delta_to_l88_mm",
    "ray_pair_cosine_mean",
    "ray_pair_cosine_min",
    "condition_lambda_min_trace",
    "geometry_valid",
    "joint_present",
)
GEOMETRY_NAMES = SUMMARY_NAMES + tuple(
    f"slot{slot}_{field}" for slot in range(SLOTS) for field in SEMANTIC_FIELDS
)
GEOMETRY_DIM = len(GEOMETRY_NAMES)
VISUAL_SLOT_DIM = (VISUAL_DIM * 3 + 2) * 2
VISUAL_REGION_DIM = SLOTS * VISUAL_SLOT_DIM
SOURCE_DIM = 3
PARENT_JOINT = torch.tensor((0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19), dtype=torch.long)


def _joint_pool(device, dtype) -> Tensor:
    value = torch.zeros((REGIONS, JOINTS), device=device, dtype=dtype)
    for region, joints in enumerate(REGION_JOINTS):
        value[region, list(joints)] = 1.0 / len(joints)
    return value


def _vertex_pool(basis: Tensor) -> Tensor:
    return (basis / basis.sum(0, keepdim=True).clamp_min(1e-12)).transpose(0, 1)


def metric_joints(vertices: Tensor, regressor: Tensor) -> Tensor:
    shape = vertices.shape[:-2]
    flat = vertices.reshape(-1, VERTICES, 3)
    value = poem_published_mesh_joints_21_ieee_fp32(flat.float(), regressor.float()).double()
    return value.reshape(*shape, JOINTS, 3)


def project_points(projection: Tensor, points: Tensor) -> tuple[Tensor, Tensor]:
    """Projection [B,8,3,4], points [B,C,J,3] -> raw [B,C,8,J,2]."""
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), -1)
    image = torch.einsum("bvrd,bcjd->bcvjr", projection.double(), homogeneous.double())
    depth = image[..., 2]
    return image[..., :2] / depth[..., None].clamp_min(1e-12), depth


def camera_centers(reference_to_camera: Tensor) -> Tensor:
    rotation = reference_to_camera[..., :3, :3].double()
    translation = reference_to_camera[..., :3, 3].double()
    return -torch.einsum("bvji,bvj->bvi", rotation, translation)


def _region_mean(pool: Tensor, values: Tensor) -> Tensor:
    return torch.einsum("rj,bcvj...->bcrv...", pool, values)


def _region_vertices(pool: Tensor, values: Tensor) -> Tensor:
    return torch.einsum("rm,bcm...->bcr...", pool, values)


def build_candidate_geometry(
    *,
    candidates: Tensor,
    l88: Tensor,
    r83: Tensor,
    placement_residual: Tensor,
    placement_alpha: Tensor,
    candidate_valid: Tensor,
    origins: Tensor,
    xgeo: Tensor,
    observations_raw: Tensor,
    projection_matrices: Tensor,
    joint_view_weights: Tensor,
    runtime_view_mask: Tensor,
    reference_to_camera: Tensor,
    basis: Tensor,
    regressor: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, dict]:
    """Return raw [B,6,C,104], validity, projections and depth validity."""
    b, c = candidates.shape[:2]
    if runtime_view_mask.ndim != 2:
        raise ValueError("runtime_view_mask must have shape [B,V]")
    views = runtime_view_mask.shape[1]
    if c < 1 or candidates.shape[-2:] != (VERTICES, 3):
        raise ValueError("candidate mesh contract")
    if runtime_view_mask.shape != (b, views) or origins.shape != (c,):
        raise ValueError(f"runtime_view_mask must be [B,{views}]")
    if torch.any(origins < -1) or torch.any(origins >= views):
        raise ValueError("candidate origin outside the physical input views")
    runtime_view_mask = runtime_view_mask.bool()
    # Padded slots are not data. Clear arbitrary bytes before any camera/ray
    # algebra so NaN * 0 cannot leak through a later mask.
    observations_raw = torch.where(
        runtime_view_mask[:, :, None, None],
        torch.nan_to_num(observations_raw),
        torch.zeros_like(observations_raw),
    )
    projection_matrices = torch.where(
        runtime_view_mask[:, :, None, None],
        torch.nan_to_num(projection_matrices),
        torch.zeros_like(projection_matrices),
    )
    reference_identity = torch.eye(
        4, device=reference_to_camera.device, dtype=reference_to_camera.dtype
    )[None, None]
    reference_to_camera = torch.where(
        runtime_view_mask[:, :, None, None],
        torch.nan_to_num(reference_to_camera),
        reference_identity,
    )
    joint_view_weights = torch.where(
        runtime_view_mask[:, :, None],
        torch.nan_to_num(joint_view_weights),
        torch.zeros_like(joint_view_weights),
    )
    joints = metric_joints(candidates, regressor)
    l88_joints = metric_joints(l88[:, None], regressor)[:, 0]
    r83_joints = metric_joints(r83[:, None], regressor)[:, 0]
    joint_pool = _joint_pool(candidates.device, torch.float64)
    vertex_pool = _vertex_pool(basis.double())
    valid = candidate_valid.bool()
    origin_valid = origins >= 0
    gather_origin = origins.clamp_min(0)

    origin_weights = joint_view_weights.double()[:, gather_origin].permute(0, 1, 2)
    origin_weights = torch.where(origin_valid[None, :, None], origin_weights, torch.zeros_like(origin_weights))
    weight_mean = torch.einsum("rj,bcj->bcr", joint_pool, origin_weights)
    weight_square = torch.einsum("rj,bcj->bcr", joint_pool, origin_weights.square())
    weight_std = (weight_square - weight_mean.square()).clamp_min(0).sqrt()
    weight_min = torch.stack([origin_weights[..., list(region)].amin(-1) for region in REGION_JOINTS], -1)
    weight_max = torch.stack([origin_weights[..., list(region)].amax(-1) for region in REGION_JOINTS], -1)

    xdelta = (joints - xgeo.double()[:, None]) / 0.01
    xmean = torch.einsum("rj,bcjk->bcrk", joint_pool, xdelta)
    xnorm = torch.linalg.vector_norm(xdelta, dim=-1)
    xnorm_mean = torch.einsum("rj,bcj->bcr", joint_pool, xnorm)
    xnorm_std = (torch.einsum("rj,bcj->bcr", joint_pool, xnorm.square()) - xnorm_mean.square()).clamp_min(0).sqrt()
    xnorm_max = torch.stack([xnorm[..., list(region)].amax(-1) for region in REGION_JOINTS], -1)

    mesh_delta = (candidates.double() - r83.double()[:, None]) / 0.01
    mesh_mean = _region_vertices(vertex_pool, mesh_delta)
    mesh_norm = torch.linalg.vector_norm(mesh_delta, dim=-1)
    mesh_norm_mean = _region_vertices(vertex_pool, mesh_norm)
    mesh_norm_std = (_region_vertices(vertex_pool, mesh_norm.square()) - mesh_norm_mean.square()).clamp_min(0).sqrt()
    mesh_norm_max = torch.stack([mesh_norm[..., basis[:, region] > 0].amax(-1) for region in range(REGIONS)], -1)

    l88_norm = torch.linalg.vector_norm((candidates.double() - l88.double()[:, None]) / 0.01, dim=-1)
    l88_mean = _region_vertices(vertex_pool, l88_norm)
    l88_std = (_region_vertices(vertex_pool, l88_norm.square()) - l88_mean.square()).clamp_min(0).sqrt()
    l88_max = torch.stack([l88_norm[..., basis[:, region] > 0].amax(-1) for region in range(REGIONS)], -1)

    centers = camera_centers(reference_to_camera)
    centroid = _region_vertices(vertex_pool, candidates.double())
    rays = centroid[:, :, :, None] - centers[:, None, None]
    distance = torch.linalg.vector_norm(rays, dim=-1).clamp_min(1e-12)
    direction = rays / distance[..., None]
    origin_direction = torch.stack(
        [direction[:, candidate, :, int(gather_origin[candidate])] for candidate in range(c)],
        dim=1,
    )
    all_cos = torch.einsum("bcrd,bcrvd->bcrv", origin_direction, direction)
    peer = torch.ones((c, views), dtype=torch.bool, device=candidates.device)
    peer[torch.arange(c, device=candidates.device), gather_origin] = False
    peer &= origin_valid[:, None]
    # The eight-slot layout is retained for HanCo identity, but only physical
    # runtime views contribute to peer geometry on variable-view datasets.
    runtime_peer = peer[None] & runtime_view_mask[:, None]
    clean_cos = torch.nan_to_num(all_cos)
    ray_mean = torch.where(
        runtime_peer[:, :, None], clean_cos, torch.zeros_like(clean_cos)
    ).sum(-1) / runtime_peer.sum(-1).clamp_min(1)[:, :, None]
    ray_min = clean_cos.masked_fill(~runtime_peer[:, :, None], float("inf")).amin(-1)
    ray_min = torch.where(origin_valid[None, :, None], ray_min, torch.zeros_like(ray_min))
    origin_distance = torch.stack(
        [distance[:, candidate, :, int(gather_origin[candidate])] for candidate in range(c)],
        dim=1,
    ) / 0.1
    origin_distance = torch.where(origin_valid[None, :, None], origin_distance, torch.zeros_like(origin_distance))

    summary = torch.cat(
        (
            torch.stack((weight_mean, weight_std, weight_min, weight_max), -1),
            xmean,
            torch.stack((xnorm_mean, xnorm_std, xnorm_max), -1),
            mesh_mean,
            torch.stack((mesh_norm_mean, mesh_norm_std, mesh_norm_max), -1),
            torch.stack((l88_mean, l88_std, l88_max), -1),
            torch.stack(
                (
                    placement_residual.double()[:, :, None].expand(-1, -1, REGIONS) / 0.01,
                    placement_alpha.double()[:, :, None].expand(-1, -1, REGIONS),
                    placement_alpha.double().clamp_min(1e-12)[:, :, None].log().expand(-1, -1, REGIONS),
                ),
                -1,
            ),
            torch.stack((ray_mean, ray_min, origin_distance), -1),
            valid.double()[:, :, None, None].expand(-1, -1, REGIONS, 1),
        ),
        -1,
    ).permute(0, 2, 1, 3)
    if summary.shape[-1] != len(SUMMARY_NAMES):
        raise AssertionError(summary.shape)

    loo_points = []
    loo_valid = []
    for leave in range(views):
        visible = runtime_view_mask[:, :, None].expand(-1, -1, JOINTS).clone()
        visible[:, leave] = False
        weights = torch.where(visible, joint_view_weights.double(), torch.zeros_like(joint_view_weights.double()))
        result = triangulate_projective_dlt(
            observations_raw.double(), projection_matrices.double(), weights, visible, min_views=2, eps=1e-12
        )
        loo_points.append(result.points_3d.double())
        loo_valid.append(result.valid)
    loo = torch.stack(loo_points, 1)
    loo_ok = torch.stack(loo_valid, 1)
    candidate_raw, depth = project_points(projection_matrices, joints)
    projected_valid = torch.isfinite(candidate_raw).all(-1) & (depth > 0) & runtime_view_mask[:, None, :, None]
    residual = torch.linalg.vector_norm(candidate_raw - observations_raw.double()[:, None], dim=-1)
    peer_mask = runtime_view_mask[:, None, :, None].expand(-1, c, -1, JOINTS).clone()
    peer_mask[:, torch.arange(c, device=candidates.device), gather_origin] = False
    peer_mask &= origin_valid[None, :, None, None]
    peer_mask &= projected_valid
    peer_count = peer_mask.sum(2)
    cross_mean = torch.where(peer_mask, residual, torch.zeros_like(residual)).sum(2) / peer_count.clamp_min(1)
    cross_max = residual.masked_fill(~peer_mask, float("-inf")).amax(2)
    cross_max = torch.where(peer_count > 0, cross_max, torch.zeros_like(cross_max))
    origin_reprojection = torch.stack(
        [residual[:, candidate, int(gather_origin[candidate])] for candidate in range(c)],
        dim=1,
    )
    origin_reprojection = torch.where(origin_valid[None, :, None], origin_reprojection, torch.zeros_like(origin_reprojection))
    selected_loo = loo.index_select(1, gather_origin)
    selected_loo_valid = loo_ok.index_select(1, gather_origin)

    joint_ray = xgeo.double()[:, None] - centers[:, :, None]
    joint_ray = joint_ray / torch.linalg.vector_norm(joint_ray, dim=-1, keepdim=True).clamp_min(1e-12)
    origin_joint_ray = joint_ray[:, gather_origin]
    ray_cos = torch.einsum("bcjd,bvjd->bcvj", origin_joint_ray, joint_ray)
    joint_peer = runtime_peer[..., None]
    clean_ray_cos = torch.nan_to_num(ray_cos)
    ray_joint_mean = torch.where(
        joint_peer, clean_ray_cos, torch.zeros_like(clean_ray_cos)
    ).sum(2) / joint_peer.sum(2).clamp_min(1)
    ray_joint_min = clean_ray_cos.masked_fill(~joint_peer, float("inf")).amin(2)
    ray_joint_min = torch.where(origin_valid[None, :, None], ray_joint_min, torch.zeros_like(ray_joint_min))
    wp = joint_view_weights.double()[:, None].expand(-1, c, -1, -1).clone()
    wp[:, torch.arange(c, device=candidates.device), gather_origin] = 0
    wp *= runtime_view_mask[:, None, :, None]
    wp = wp / wp.sum(2, keepdim=True).clamp_min(1e-12)
    eye = torch.eye(3, device=candidates.device, dtype=torch.float64)
    outer = joint_ray[..., :, None] * joint_ray[..., None, :]
    h = torch.einsum("bcvj,bvjkl->bcjkl", wp, eye[None, None, None] - outer)
    eigen = torch.linalg.eigvalsh(h)
    condition = eigen[..., 0] / eigen.sum(-1).clamp_min(1e-12)
    condition = torch.where(origin_valid[None, :, None], condition, torch.zeros_like(condition))

    parent = PARENT_JOINT.to(candidates.device)
    candidate_bone = torch.linalg.vector_norm(joints - joints.index_select(2, parent), dim=-1)
    l88_bone = torch.linalg.vector_norm(l88_joints - l88_joints.index_select(1, parent), dim=-1)
    bone_delta = (candidate_bone - l88_bone[:, None]).abs() * 1000
    origin_depth_ok = torch.stack(
        [projected_valid[:, candidate, int(gather_origin[candidate])] for candidate in range(c)],
        dim=1,
    )
    semantic_valid = (
        valid[:, :, None]
        & origin_valid[None, :, None]
        & selected_loo_valid
        & origin_depth_ok
        & (peer_count > 0)
    )
    raw_fields = torch.stack(
        (
            torch.linalg.vector_norm(joints - xgeo.double()[:, None], dim=-1) * 1000,
            torch.linalg.vector_norm(joints - r83_joints[:, None], dim=-1) * 1000,
            torch.linalg.vector_norm(joints - l88_joints[:, None], dim=-1) * 1000,
            torch.linalg.vector_norm(joints - selected_loo, dim=-1) * 1000,
            cross_mean,
            cross_max,
            origin_reprojection,
            bone_delta,
            ray_joint_mean,
            ray_joint_min,
            condition,
            semantic_valid.double(),
            torch.ones_like(condition),
        ),
        -1,
    )
    semantic = torch.zeros((b, REGIONS, c, SLOTS * len(SEMANTIC_FIELDS)), device=candidates.device, dtype=torch.float64)
    semantic_mask = torch.zeros_like(semantic, dtype=torch.bool)
    for region, region_joints in enumerate(REGION_JOINTS):
        for slot in range(SLOTS):
            offset = slot * len(SEMANTIC_FIELDS)
            if slot < len(region_joints):
                joint = region_joints[slot]
                semantic[:, region, :, offset : offset + len(SEMANTIC_FIELDS)] = raw_fields[:, :, joint]
                semantic_mask[:, region, :, offset : offset + 11] = semantic_valid[:, :, joint, None]
                semantic_mask[:, region, :, offset + 11 : offset + 13] = True
            else:
                semantic_mask[:, region, :, offset + 11 : offset + 13] = True
                semantic[:, region, :, offset + 12] = 0
    summary_mask = torch.ones_like(summary, dtype=torch.bool)
    summary_mask[..., 0:4] &= origin_valid[None, None, :, None]
    summary_mask[..., 19:25] &= origin_valid[None, None, :, None]
    geometry = torch.cat((summary, semantic), -1)
    geometry_valid = torch.cat((summary_mask, semantic_mask), -1)
    finite = torch.isfinite(geometry)
    geometry_valid &= finite
    geometry = torch.where(finite, geometry, torch.zeros_like(geometry))
    diagnostics = {
        "geometry_dimensions": GEOMETRY_DIM,
        "semantic_valid_fraction": float(semantic_valid.float().mean().cpu()),
        "candidate_projection_valid_fraction": float(projected_valid.float().mean().cpu()),
    }
    return geometry, geometry_valid, candidate_raw, projected_valid, diagnostics


def raw_to_hamer_grid(raw: Tensor, raw_to_crop: Tensor) -> Tensor:
    """Map [B,C,8,J,2] raw pixels into HaMeR 12x16 grid coordinates."""
    homogeneous = torch.cat((raw, torch.ones_like(raw[..., :1])), -1)
    crop = torch.einsum("bvrd,bcvjd->bcvjr", raw_to_crop.double(), homogeneous.double())
    grid = torch.empty_like(crop)
    grid[..., 0] = 2.0 * (crop[..., 0] - 32.0) / 192.0 - 1.0
    grid[..., 1] = 2.0 * crop[..., 1] / 256.0 - 1.0
    return grid


def fit_standardize(geometry: Tensor, validity: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
    filled = torch.where(validity, geometry, mean)
    return safe_standardize(filled, mean, scale)


class CandidateEvidenceHead(nn.Module):
    def __init__(self, feature_dim: int = RAW_VISUAL_DIM) -> None:
        super().__init__()
        local = GEOMETRY_DIM + VISUAL_REGION_DIM + REGIONS + SOURCE_DIM + 2
        score = 32 * 3 + REGIONS + SOURCE_DIM + 2
        self.visual_projection = nn.Linear(feature_dim, VISUAL_DIM)
        self.phi = nn.Sequential(nn.LayerNorm(local), nn.Linear(local, 64), nn.SiLU(), nn.Linear(64, 32))
        self.score = nn.Sequential(nn.LayerNorm(score), nn.Linear(score, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)
        # Registered contract: visual projection in FP32, scoring and mesh
        # fusion in FP64.
        self.phi.double()
        self.score.double()


def source_tags(device, dtype) -> Tensor:
    result = torch.zeros((CANDIDATES, SOURCE_DIM), device=device, dtype=dtype)
    result[0, 0] = 1
    result[1:9, 1] = 1
    result[9:, 2] = 1
    return result




def _visual_region_descriptor(
    head: CandidateEvidenceHead,
    feature_map: Tensor,
    candidate_grid: Tensor,
    candidate_raw: Tensor,
    projection_valid: Tensor,
    l88_grid: Tensor,
    l88_raw: Tensor,
    l88_valid: Tensor,
    runtime_view_mask: Tensor | None = None,
    *,
    canonical_reader: bool,
    strict_valid_finite: bool,
    native_feature_maps: bool = False,
) -> Tensor:
    b, views, feature_channels, height, width = feature_map.shape
    candidates = candidate_grid.shape[1]
    if runtime_view_mask is None:
        runtime_view_mask = torch.ones((b, views), device=feature_map.device, dtype=torch.bool)
    else:
        runtime_view_mask = runtime_view_mask.to(device=feature_map.device, dtype=torch.bool)
        if runtime_view_mask.shape != (b, views):
            raise ValueError(f"runtime_view_mask must be [B,{views}], got {tuple(runtime_view_mask.shape)}")
    # Invalid slots may contain arbitrary bytes. Valid physical features and
    # explicitly valid projections are formal inputs and may not be silently
    # repaired when the canonical target reader is active.
    view_mask = runtime_view_mask[:, :, None, None, None]
    projection_valid = projection_valid.bool() & runtime_view_mask[:, None, :, None]
    l88_valid = l88_valid.bool() & runtime_view_mask[:, :, None]
    if strict_valid_finite:
        if not torch.isfinite(feature_map).masked_select(view_mask.expand_as(feature_map)).all():
            raise ValueError("non-finite feature in a valid physical view")
        candidate_required = projection_valid[..., None].expand_as(candidate_grid)
        if not torch.isfinite(candidate_grid).masked_select(candidate_required).all():
            raise ValueError("non-finite candidate grid marked projection-valid")
        if not torch.isfinite(candidate_raw).masked_select(candidate_required).all():
            raise ValueError("non-finite candidate raw coordinate marked projection-valid")
        reference_required = l88_valid[..., None].expand_as(l88_grid)
        if not torch.isfinite(l88_grid).masked_select(reference_required).all():
            raise ValueError("non-finite reference grid marked valid")
        if not torch.isfinite(l88_raw).masked_select(reference_required).all():
            raise ValueError("non-finite reference raw coordinate marked valid")
    feature_map = torch.where(
        view_mask, torch.nan_to_num(feature_map), torch.zeros_like(feature_map)
    ).contiguous()
    candidate_grid = torch.nan_to_num(candidate_grid)
    candidate_raw = torch.nan_to_num(candidate_raw)
    l88_grid = torch.nan_to_num(l88_grid)
    l88_raw = torch.nan_to_num(l88_raw)
    if native_feature_maps:
        if feature_channels != head.visual_projection.in_features:
            raise ValueError("prior feature channel count does not match regional head")
        maps = feature_map.float().reshape(b * views, feature_channels, height, width)
    elif canonical_reader:
        maps = canonical_feature_maps(feature_map).float().reshape(
            b * views, RAW_VISUAL_DIM, 16, 12
        )
    else:
        maps = feature_map.float().reshape(b * views, RAW_VISUAL_DIM, 12, 16)
    # A shared linear projection commutes with bilinear interpolation.  Project
    # the 12x16 map once so every candidate does not materialize 1280 channels.
    projected_maps = head.visual_projection(maps.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    grid = candidate_grid.permute(0, 2, 1, 3, 4).reshape(b * views, candidates * JOINTS, 1, 2).float()
    sampled = F.grid_sample(projected_maps, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    candidate_projected = sampled.squeeze(-1).transpose(1, 2).reshape(b, views, candidates, JOINTS, VISUAL_DIM).permute(0, 2, 1, 3, 4)
    lgrid = l88_grid.reshape(b * views, JOINTS, 1, 2).float()
    reference = F.grid_sample(projected_maps, lgrid, mode="bilinear", padding_mode="zeros", align_corners=False)
    reference_projected = reference.squeeze(-1).transpose(1, 2).reshape(b, views, JOINTS, VISUAL_DIM)
    reference_projected = reference_projected[:, None].expand(-1, candidates, -1, -1, -1)
    candidate_in_bounds = torch.isfinite(candidate_grid).all(-1) & (candidate_grid.abs() <= 1).all(-1)
    reference_in_bounds = torch.isfinite(l88_grid).all(-1) & (l88_grid.abs() <= 1).all(-1)
    both = projection_valid & candidate_in_bounds & l88_valid[:, None] & reference_in_bounds[:, None]
    distance = torch.linalg.vector_norm(candidate_raw - l88_raw[:, None], dim=-1, keepdim=True).float() / 256.0
    per_view = torch.cat(
        (
            candidate_projected,
            reference_projected,
            candidate_projected - reference_projected,
            distance,
            both[..., None].float(),
        ),
        -1,
    )
    mask = both[..., None]
    mean = torch.where(mask, per_view, torch.zeros_like(per_view)).sum(2) / mask.sum(2).clamp_min(1)
    maximum = per_view.masked_fill(~mask, float("-inf")).amax(2)
    maximum = torch.where(mask.any(2), maximum, torch.zeros_like(maximum))
    pooled = torch.cat((mean, maximum), -1)
    output = torch.zeros((b, REGIONS, candidates, VISUAL_REGION_DIM), device=feature_map.device, dtype=pooled.dtype)
    for region, joints in enumerate(REGION_JOINTS):
        slots = torch.zeros((b, candidates, SLOTS, VISUAL_SLOT_DIM), device=feature_map.device, dtype=pooled.dtype)
        slots[:, :, : len(joints)] = pooled[:, :, list(joints)]
        output[:, region] = slots.flatten(-2)
    return output




def masked_canonical_visual_region_descriptor(
    head: CandidateEvidenceHead,
    feature_map: Tensor,
    candidate_grid: Tensor,
    candidate_raw: Tensor,
    projection_valid: Tensor,
    l88_grid: Tensor,
    l88_raw: Tensor,
    l88_valid: Tensor,
    runtime_view_mask: Tensor | None = None,
) -> Tensor:
    """Masked target reader using HanCo R's canonical row-major reshape."""
    return _visual_region_descriptor(
        head, feature_map, candidate_grid, candidate_raw, projection_valid,
        l88_grid, l88_raw, l88_valid, runtime_view_mask,
        canonical_reader=True, strict_valid_finite=True,
    )


# Backward-compatible legacy entry. New transfer code must import the explicit
# masked canonical function above and bind READER_VERSION in its checkpoint.
visual_region_descriptor = masked_canonical_visual_region_descriptor


def native_visual_region_descriptor(head, feature_map, candidate_grid, candidate_raw,
                                    projection_valid, anchor_grid, anchor_raw,
                                    anchor_valid, runtime_view_mask):
    """Same paired evidence pooling on native [B,V,C,H,W] prior features."""
    return _visual_region_descriptor(
        head, feature_map, candidate_grid, candidate_raw, projection_valid,
        anchor_grid, anchor_raw, anchor_valid, runtime_view_mask,
        canonical_reader=False, strict_valid_finite=True, native_feature_maps=True,
    )
