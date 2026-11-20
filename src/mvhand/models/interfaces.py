from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


@dataclass
class Form2FrameBatch:
    """Tensor batch for canonical inference.

    Shapes are crops ``[B,V,3,H,W]``, intrinsics ``[B,V,3,3]``,
    camera-to-world transforms ``[B,V,4,4]``, crop-to-raw affines
    ``[B,V,2,3]``, and Boolean runtime view availability ``[B,V]``. Runtime
    availability is inference-visible input state, never GT joint visibility.
    """

    crops: Tensor
    intrinsics: Tensor
    camera_to_world: Tensor
    crop_to_raw: Tensor
    runtime_view_mask: Tensor
    raw_to_crop: Tensor | None = None


@dataclass
class HandPriorOutput:
    """Only the dependency-injected hand-prior values consumed by Form2Frame.

    ``joints_2d_normalized`` is ``[B,V,J,2]`` in approximately ``[-.5,.5]``;
    visual features are ``[B,V,J,C]``; local joints and vertices are
    ``[B,V,J,3]`` and ``[B,V,M,3]`` floating tensors.
    """

    joints_2d_normalized: Tensor
    visual_features: Tensor
    local_joints_3d: Tensor
    local_vertices_3d: Tensor


class HandPrior(Protocol):
    def __call__(self, normalized_crops: Tensor) -> HandPriorOutput:
        """Run a prior on ImageNet-normalized ``[B,V,3,H,W]`` crops."""


def validate_inference_batch(batch: Form2FrameBatch) -> None:
    if batch.crops.ndim != 5 or batch.crops.shape[2] != 3:
        raise ValueError("crops must have shape [B,V,3,H,W]")
    b, v = batch.crops.shape[:2]
    expected = {
        "intrinsics": (b, v, 3, 3),
        "camera_to_world": (b, v, 4, 4),
        "crop_to_raw": (b, v, 2, 3),
    }
    for name, shape in expected.items():
        value = getattr(batch, name)
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not value.is_floating_point() or not torch.isfinite(value).all().item():
            raise ValueError(f"{name} must be a finite floating tensor")
    if not batch.crops.is_floating_point() or not torch.isfinite(batch.crops).all().item():
        raise ValueError("crops must be a finite floating tensor")
    if batch.runtime_view_mask.shape != (b, v):
        raise ValueError("runtime_view_mask must have shape [B,V]")
    if batch.runtime_view_mask.dtype != torch.bool:
        raise TypeError("runtime_view_mask must be Boolean")
    if torch.any(batch.runtime_view_mask.sum(dim=1) < 2):
        raise ValueError("geometry requires at least two runtime-available views")
    if batch.raw_to_crop is not None:
        if batch.raw_to_crop.shape != (b, v, 2, 3):
            raise ValueError("raw_to_crop must have shape [B,V,2,3]")
        if not batch.raw_to_crop.is_floating_point() or not torch.isfinite(batch.raw_to_crop).all().item():
            raise ValueError("raw_to_crop must be finite and floating")
    tensors = (batch.intrinsics, batch.camera_to_world, batch.crop_to_raw, batch.runtime_view_mask)
    if batch.raw_to_crop is not None:
        tensors = (*tensors, batch.raw_to_crop)
    if any(value.device != batch.crops.device for value in tensors):
        raise ValueError("all inference tensors must share one device")


def validate_prior_output(output: HandPriorOutput, batch: Form2FrameBatch) -> None:
    b, v = batch.crops.shape[:2]
    if output.joints_2d_normalized.ndim != 4 or output.joints_2d_normalized.shape[:2] != (b, v) or output.joints_2d_normalized.shape[-1] != 2:
        raise ValueError("prior joints_2d_normalized must have shape [B,V,J,2]")
    j = output.joints_2d_normalized.shape[2]
    shapes = {
        "visual_features": ((b, v, j), 4),
        "local_joints_3d": ((b, v, j), 4),
        "local_vertices_3d": ((b, v), 4),
    }
    for name, (prefix, ndim) in shapes.items():
        value = getattr(output, name)
        if value.ndim != ndim or value.shape[: len(prefix)] != prefix or value.shape[-1] != 3 and name != "visual_features":
            raise ValueError(f"invalid prior {name} shape")
    for value in (output.joints_2d_normalized, output.visual_features, output.local_joints_3d, output.local_vertices_3d):
        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise TypeError("prior outputs must be floating tensors")
        if value.device != batch.crops.device or not torch.isfinite(value).all().item():
            raise ValueError("prior outputs must be finite and share the batch device")
