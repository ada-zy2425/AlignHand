"""Shared R minibatches, anchor-centered forward pass and endpoint primitives."""
from __future__ import annotations
import hashlib
import numpy as np
import torch
from mvhand.fusion.candidates import (
    GEOMETRY_DIM,
    CandidateEvidenceHead,
    fit_standardize,
    masked_canonical_visual_region_descriptor,
)
from mvhand.fusion.regional_fusion import (
    TransferCache,
    candidate_validity,
    initial_probabilities,
    fuse_anchor_candidates,
)
from mvhand.pipeline.precision import require_ieee_mesh_joint_reader
mesh_joints=require_ieee_mesh_joint_reader()
def state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        # Keep the historical NumPy dtype spelling (for example ``float32``)
        # without asking NumPy to format a dtype.  Some otherwise valid
        # Torch/NumPy combinations recurse in ``dtype.__str__``; Torch uses
        # the same scalar names with only a ``torch.`` prefix.
        dtype_name = str(value.dtype).removeprefix("torch.")
        digest.update(dtype_name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()

def make_batch(
    cache: TransferCache,
    keys: list[str],
    device: str,
    family: str,
    visual: bool,
    mean: torch.Tensor,
    scale: torch.Tensor,
    *,
    include_target: bool = True,
) -> dict[str, torch.Tensor | None]:
    parent = cache.tensor("parent_vertices", keys, device, torch.float64)
    anchor = cache.tensor("anchor", keys, device, torch.float64)
    h = cache.tensor("h_vertices", keys, device, torch.float64)
    w = cache.tensor("w_vertices", keys, device, torch.float64)
    target = cache.tensor("target", keys, device, torch.float64) if include_target else None
    view_valid = cache.tensor("view_valid", keys, device).bool()
    w_valid_actual = cache.tensor("w_valid", keys, device).bool() & view_valid
    geometry = cache.tensor("geometry_hw", keys, device, torch.float64)
    geometry_valid = cache.tensor("geometry_valid_hw", keys, device).bool()
    if geometry.shape[-1] != GEOMETRY_DIM:
        raise ValueError(f"geometry dimension {geometry.shape[-1]} != {GEOMETRY_DIM}")
    if family == "HH":
        candidates = torch.cat((anchor[:, None], h, h), 1)
        geometry = torch.cat((geometry[:, :, :9], geometry[:, :, 1:9]), 2)
        geometry_valid = torch.cat(
            (geometry_valid[:, :, :9], geometry_valid[:, :, 1:9]), 2
        )
        secondary_valid = view_valid
    else:
        candidates = torch.cat((anchor[:, None], h, w), 1)
        secondary_valid = w_valid_actual
    geometry = fit_standardize(geometry, geometry_valid, mean, scale)
    candidate_valid = candidate_validity(
        view_valid, secondary_valid, family=family
    )
    if cache.manifest['identity']['parent_policy'] != 'unified_anchor_v1':
        raise ValueError('Only the unified anchor policy is supported')
    q_init, candidate_valid = initial_probabilities(view_valid, secondary_valid, family=family)
    result: dict[str, torch.Tensor | None] = {
        "parent": parent,
        "anchor": anchor,
        "target": target,
        "candidates": candidates,
        "geometry": geometry,
        "view_valid": view_valid,
        "secondary_valid": secondary_valid,
        "w_valid_actual": w_valid_actual,
        "q_init": q_init,
        "candidate_valid": candidate_valid,
        "feature_map": None,
        "candidate_grid": None,
        "candidate_raw": None,
        "projection_valid": None,
        "reference_grid": None,
        "reference_raw": None,
        "reference_valid": None,
    }
    if visual:
        grid = cache.tensor("candidate_grid_hw", keys, device, torch.float64)
        raw = cache.tensor("candidate_raw_hw", keys, device, torch.float64)
        projected = cache.tensor(
            "candidate_projection_valid_hw", keys, device
        ).bool()
        if family == "HH":
            grid = torch.cat((grid[:, :9], grid[:, 1:9]), 1)
            raw = torch.cat((raw[:, :9], raw[:, 1:9]), 1)
            projected = torch.cat((projected[:, :9], projected[:, 1:9]), 1)
        result.update({
            "feature_map": cache.tensor("feature_maps", keys, device),
            "candidate_grid": grid,
            "candidate_raw": raw,
            "projection_valid": projected,
            "reference_grid": cache.tensor(
                "reference_grid", keys, device, torch.float64
            ),
            "reference_raw": cache.tensor(
                "reference_raw", keys, device, torch.float64
            ),
            "reference_valid": cache.tensor(
                "reference_projection_valid", keys, device
            ).bool(),
        })
    return result

def forward(
    head: CandidateEvidenceHead,
    basis: torch.Tensor,
    batch: dict[str, torch.Tensor | None],
    family: str,
    visual: bool,
    *,
    enforce_secondary_fallback: bool = True,
    output_mode: str = "anchor_centered",
):
    descriptor = None
    if visual:
        descriptor = masked_canonical_visual_region_descriptor(
            head,
            batch["feature_map"],
            batch["candidate_grid"],
            batch["candidate_raw"],
            batch["projection_valid"],
            batch["reference_grid"],
            batch["reference_raw"],
            batch["reference_valid"],
            batch["view_valid"],
        )
    return fuse_anchor_candidates(
        head=head,
        anchor=batch["anchor"],
        candidates=batch["candidates"],
        geometry=batch["geometry"],
        basis=basis,
        q_init=batch["q_init"],
        candidate_valid=batch["candidate_valid"],
        family=family,
        secondary_valid=batch["secondary_valid"],
        visual_enabled=visual,
        visual_descriptor=descriptor,
        enforce_secondary_fallback=enforce_secondary_fallback,
        output_mode=output_mode,
    )

def errors(
    vertices: torch.Tensor,
    target: torch.Tensor,
    regressor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    vertex = torch.linalg.vector_norm(vertices - target, dim=-1).mean(-1) * 1000
    joints = mesh_joints(vertices.float(), regressor.float()).double()
    target_joints = mesh_joints(target.float(), regressor.float()).double()
    joint = torch.linalg.vector_norm(joints - target_joints, dim=-1).mean(-1) * 1000
    return vertex, joint
