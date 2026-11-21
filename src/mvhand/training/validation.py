"""Runtime assets, sequential readers and the validation executor used during training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import tarfile
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from mvhand.data.poem_reader import PoemWDSGroup, decode_canonical_sample
from mvhand.geometry.camera import dehomogenize
from mvhand.pipeline.inference import Form2FrameInference, InferenceResult
from mvhand.geometry.mesh_joints import poem_published_mesh_joints_21
from mvhand.observation.model import Form2FrameModel
from mvhand.prior.hamer import (
    CanonicalHamerProvider,
    Form2FrameRuntimeAssets,
    RuntimeAsset,
    inject_canonical_hamer_lora,
    load_canonical_hamer,
)
from mvhand.prior.numpy_compat import install_numpy_chumpy_compat
from mvhand.training.collate import collate_groups
from mvhand.training.index import IndexedWDS


@dataclass(frozen=True)
class CanonicalRuntimePaths:
    dataset_root: Path
    index: Path
    hamer_source: Path
    hamer_checkpoint: Path
    hamer_config: Path
    mano_right: Path
    mano_mean_params: Path
    dataset_contract: Path
    dataset_contract_sha256: str = (
        "4a74744392a8c3bc844c758bb23e5a11cdece0e238e93d9054d97bbc814a0739"
    )


def canonical_runtime_assets(paths: CanonicalRuntimePaths) -> Form2FrameRuntimeAssets:
    """Bind every canonical runtime path to its declared asset identity."""
    return Form2FrameRuntimeAssets(
        paths.hamer_source,
        RuntimeAsset(paths.hamer_checkpoint, "e5cc06f294d88a92dee24e603480aab04de532b49f0e08200804ee7d90e16f53"),
        RuntimeAsset(paths.hamer_config, "0e5eeb82752e47dfd01db8e13ccc4c5eba9bf83f53da8285523b8d3e87247aa3"),
        RuntimeAsset(paths.mano_right, "45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767"),
        RuntimeAsset(paths.mano_mean_params, "efc0ec58e4a5cef78f3abfb4e8f91623b8950be9eff8b8e0dbb0d036ebc63988"),
        RuntimeAsset(paths.dataset_contract, paths.dataset_contract_sha256),
    )


def read_sequential_sample(dataset_root: Path, qualified_key: str) -> dict[str, bytes | str]:
    """Trusted TAR-order lookup used for bounded parity checks, not random training."""
    shard, key = qualified_key.split("::", 1)
    sample: dict[str, bytes | str] = {"__key__": key}
    with tarfile.open(Path(dataset_root) / shard, "r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = Path(member.name).name
            member_key, suffix = name.split(".", 1)
            if member_key != key:
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"unreadable TAR member: {member.name}")
            sample[suffix] = stream.read()
            if len(sample) == 10:
                break
    if len(sample) != 10:
        raise KeyError(qualified_key)
    return sample


def read_sequential_group(dataset_root: Path, qualified_key: str, hamer_source: Path) -> PoemWDSGroup:
    return decode_canonical_sample(
        read_sequential_sample(dataset_root, qualified_key), hamer_source,
        qualified_key=qualified_key,
    )


def load_mano_base_regressor(path: Path) -> torch.Tensor:
    # MANO's legacy pickle imports chumpy, whose last release still imports
    # NumPy aliases removed in NumPy 1.24. Restore only those compatibility
    # names at the licensed-asset boundary before unpickling.
    install_numpy_chumpy_compat()
    with Path(path).open("rb") as stream:
        data = pickle.load(stream, encoding="latin1")
    value = data["J_regressor"]
    if hasattr(value, "toarray"):
        value = value.toarray()
    tensor = torch.as_tensor(np.asarray(value), dtype=torch.float32)
    if tensor.shape != (16, 778) or not torch.isfinite(tensor).all():
        raise ValueError("MANO base regressor contract failed")
    return tensor


def official_mesh_joint_endpoints(
    predicted_vertices: torch.Tensor,
    ground_truth_vertices: torch.Tensor,
    mano_base_regressor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct both official P310B joint endpoints with one POEM helper."""
    regressor = mano_base_regressor.to(
        device=predicted_vertices.device,
        dtype=predicted_vertices.dtype,
    )
    if ground_truth_vertices.device != predicted_vertices.device:
        raise ValueError("predicted and GT vertices must share a device")
    ground_truth_vertices = ground_truth_vertices.to(dtype=predicted_vertices.dtype)
    return (
        poem_published_mesh_joints_21(predicted_vertices, regressor),
        poem_published_mesh_joints_21(ground_truth_vertices, regressor),
    )


def build_canonical_inference(
    paths: CanonicalRuntimePaths,
    *,
    device: torch.device | str = "cuda",
    validate_assets: bool = True,
    parameterization: str = "PIXEL",
) -> Form2FrameInference:
    # HaMeR constructs smplx.MANO while loading its Lightning checkpoint;
    # install the legacy chumpy aliases before that import/pickle boundary.
    install_numpy_chumpy_compat()
    assets = canonical_runtime_assets(paths)
    if not validate_assets:
        raise ValueError("canonical production construction requires asset validation")
    hamer, _ = load_canonical_hamer(assets)
    inject_canonical_hamer_lora(hamer)
    provider = CanonicalHamerProvider(hamer)
    model = Form2FrameModel(
        provider, feature_dim=1280, joints=21, max_views=8, evidence_dim=128,
        attention_heads=4, dropout=0.0, correction_limit=0.05,
        weight_correction_scale=0.25, residual_scale_px=10.0,
        residual_temperature=0.5, patch_size=256,
        parameterization=parameterization,
    ).to(device)
    return Form2FrameInference(
        model, geometry_blend=0.5,
        mano_base_regressor=load_mano_base_regressor(paths.mano_right).to(device),
    )


class CanonicalValidationExecutor:
    """Production validation executor consumed by ``CanonicalTrainer.validate``."""

    def __init__(
        self, inference: Form2FrameInference, reader: IndexedWDS,
        hamer_source: Path, *, device: torch.device | str,
    ) -> None:
        self.inference = inference
        self.reader = reader
        self.hamer_source = Path(hamer_source)
        self.device = torch.device(device)

    def __call__(self, keys: Sequence[str]) -> Iterable[Mapping[str, object]]:
        for key in keys:
            group = self.reader.get_group("val", key, self.hamer_source)
            batch = collate_groups((group,), device=self.device)
            result = self.inference(batch.model)
            prediction, output = result.prediction, result.output
            gt_dataset_joints = batch.joints_3d[:, 0]
            gt_vertices = batch.vertices_3d[:, 0]
            pred_mesh_joints, gt_mesh_joints = official_mesh_joint_endpoints(
                output.final_vertices_3d,
                gt_vertices,
                self.inference.mano_base_regressor,
            )
            if not torch.allclose(
                pred_mesh_joints,
                output.final_mesh_joints_3d,
                rtol=0.0,
                atol=1e-7,
                equal_nan=True,
            ):
                raise ValueError("inference and metric mesh-joint endpoints disagree")
            homogeneous = torch.cat((prediction.weighted.points_3d, torch.ones_like(prediction.weighted.points_3d[..., :1])), dim=-1)
            projected = torch.einsum("bvrc,bjc->bvjr", prediction.projection_matrices, homogeneous)
            projected = dehomogenize(projected, eps=1e-12)
            reprojection = torch.linalg.vector_norm(projected - prediction.corrected_joints_2d_raw, dim=-1)
            valid = bool(result.mesh_recovery.valid.all().item())
            yield {
                "xgeo_joints_3d": output.xgeo_joints_3d.detach().cpu(),
                "final_mesh_joints_3d": pred_mesh_joints.detach().cpu(),
                "final_vertices_3d": output.final_vertices_3d.detach().cpu(),
                "gt_mesh_joints_3d": gt_mesh_joints.detach().cpu(),
                "gt_dataset_joints_3d": gt_dataset_joints.detach().cpu(),
                "gt_vertices_3d": gt_vertices.detach().cpu(),
                "correction": prediction.correction_normalized.detach().cpu(),
                "reprojection_residual": reprojection.detach().cpu(),
                "weights": prediction.relative_view_weights.detach().cpu().permute(0, 2, 1),
                "weight_mask": batch.model.runtime_view_mask[:, None, :].expand(
                    -1, prediction.relative_view_weights.shape[2], -1
                ).detach().cpu(),
                "valid": valid,
            }
