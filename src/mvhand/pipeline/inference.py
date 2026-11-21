"""Full AlignHand forward pass from frozen-prior outputs to the final metric mesh."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from mvhand.contracts import Form2FrameOutput
from mvhand.geometry.metric_alignment import MeshRecoveryResult, recover_positioned_mesh
from mvhand.geometry.mesh_joints import poem_published_mesh_joints_21, validate_mano_base_regressor
from mvhand.observation.model import Form2FrameModel, Form2FramePrediction
from mvhand.prior.hamer_adapter import normalize_rgb_crops
from mvhand.interfaces import Form2FrameBatch


@dataclass
class InferenceResult:
    output: Form2FrameOutput
    prediction: Form2FramePrediction
    mesh_recovery: MeshRecoveryResult


class Form2FrameInference:
    """Asset-free orchestrator around injected prior, neural core and geometry."""

    def __init__(self, model: Form2FrameModel, *, geometry_blend: float, mano_base_regressor) -> None:
        if not 0 <= geometry_blend <= 1:
            raise ValueError("geometry_blend must be in [0,1]")
        self.model = model
        self.geometry_blend = float(geometry_blend)
        validate_mano_base_regressor(mano_base_regressor)
        self.mano_base_regressor = mano_base_regressor

    def __call__(self, batch: Form2FrameBatch) -> InferenceResult:
        normalized = normalize_rgb_crops(batch.crops)
        prediction = self.model(normalized, batch)
        geometry = prediction.weighted.points_3d
        recovery_dtype = (
            torch.float32 if geometry.dtype in (torch.float16, torch.bfloat16)
            else geometry.dtype
        )
        geometry = geometry.to(recovery_dtype)
        # CUDA autocast may leave MANO outputs in FP32 while projective geometry
        # is BF16. Mesh recovery is a single-dtype numerical boundary.
        with torch.autocast(device_type=geometry.device.type, enabled=False):
            recovery = recover_positioned_mesh(
                geometry,
                prediction.prior.local_joints_3d.to(geometry.dtype),
                prediction.prior.local_vertices_3d.to(geometry.dtype),
                prediction.relative_view_weights.to(geometry.dtype),
                prediction.weighted.valid,
                geometry_blend=self.geometry_blend,
            )
        regressor = self.mano_base_regressor.to(
            device=recovery.final_vertices_3d.device,
            dtype=recovery.final_vertices_3d.dtype,
        )
        safe_vertices = torch.nan_to_num(recovery.final_vertices_3d)
        final_mesh_joints = poem_published_mesh_joints_21(safe_vertices, regressor)
        mesh_valid = recovery.valid.all(dim=-1)
        final_mesh_joints = torch.where(
            mesh_valid[:, None, None], final_mesh_joints,
            torch.full((), float("nan"), dtype=final_mesh_joints.dtype, device=final_mesh_joints.device),
        )
        output = Form2FrameOutput(
            final_vertices_3d=recovery.final_vertices_3d,
            final_mesh_joints_3d=final_mesh_joints,
            xgeo_joints_3d=recovery.triangulated_joints_3d,
            positioned_prior_joints_3d=recovery.positioned_prior_joints_3d,
            mesh_joints_mano_3d=final_mesh_joints,
            mesh_joints_dataset_3d=None,
            direct_joints_3d=recovery.final_joints_3d,
            calibrated_scaffold_3d=recovery.triangulated_joints_3d,
            relative_view_weights=prediction.relative_view_weights,
            metadata={
                "coordinate_frame": "camera_reference",
                "correction_coordinate_system": "normalized_crop_residual_to_raw_pixels",
                "triangulation": "raw_pixel_projective_dlt",
                "valid": recovery.valid,
            },
        )
        return InferenceResult(output, prediction, recovery)
