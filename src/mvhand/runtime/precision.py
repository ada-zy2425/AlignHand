#!/usr/bin/env python3
"""Frozen numerical and feature-reader contract for R transfer execution."""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import torch
from torch import Tensor


READER_VERSION = "R_CANONICAL_RESHAPE_V1"
MESH_JOINT_READER = "POEM_PUBLISHED_MESH_JOINTS_21_IEEE_FP32"
RUNTIME_CONTRACT_VERSION = "R_TRANSFER_RUNTIME_V2"
VIEWS = 8
RAW_VISUAL_DIM = 1280
STORAGE_HW = (12, 16)
LOGICAL_HW = (16, 12)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_feature_maps(feature_map: Tensor) -> Tensor:
    """Restore row-major storage [12,16] as the traced logical [16,12]."""
    if feature_map.ndim != 5 or tuple(feature_map.shape[1:3]) != (
        VIEWS,
        RAW_VISUAL_DIM,
    ):
        raise ValueError(f"feature-map cache contract: {tuple(feature_map.shape)}")
    if tuple(feature_map.shape[-2:]) != STORAGE_HW:
        raise ValueError(
            f"reader {READER_VERSION} requires storage {STORAGE_HW}, "
            f"got {tuple(feature_map.shape[-2:])}"
        )
    if not feature_map.is_contiguous():
        raise ValueError("canonical feature-map cache must be row-major contiguous")
    return feature_map.reshape(
        feature_map.shape[0], VIEWS, RAW_VISUAL_DIM, *LOGICAL_HW
    )


def configure_transfer_runtime() -> dict:
    """Apply the same frozen numerical settings in train/eval/canary processes."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    return {
        "version": RUNTIME_CONTRACT_VERSION,
        "reader": READER_VERSION,
        "visual_projection": "FP32",
        "scoring_and_fusion": "FP64",
        "cuda_matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch_version": torch.__version__,
    }


def require_ieee_mesh_joint_reader():
    """Return the registered metric reader or fail without a legacy fallback."""
    try:
        from mvhand.geometry.mesh import poem_published_mesh_joints_21_ieee_fp32
    except ImportError as error:
        raise RuntimeError(
            "formal R transfer requires "
            "mvhand.geometry.mesh.poem_published_mesh_joints_21_ieee_fp32; "
            "use the qualified frozen runtime source"
        ) from error
    return poem_published_mesh_joints_21_ieee_fp32


def runtime_source_identity() -> dict:
    reader = require_ieee_mesh_joint_reader()
    source = Path(inspect.getsourcefile(reader) or "").resolve()
    if not source.is_file():
        raise RuntimeError("cannot resolve registered mesh reader source")
    return {
        "mesh_reader": MESH_JOINT_READER,
        "mesh_reader_source": str(source),
        "mesh_reader_source_sha256": sha256(source),
    }
