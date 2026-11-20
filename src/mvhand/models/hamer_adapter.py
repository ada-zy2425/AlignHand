from __future__ import annotations

from typing import Callable, Mapping

import torch
from torch import Tensor, nn

from mvhand.models.interfaces import HandPriorOutput


def normalize_rgb_crops(crops: Tensor) -> Tensor:
    """Normalize RGB ``[B,V,3,H,W]`` crops using the active prior contract."""
    if crops.ndim != 5 or crops.shape[2] != 3 or not crops.is_floating_point():
        raise ValueError("crops must be a floating tensor with shape [B,V,3,H,W]")
    if not torch.isfinite(crops).all().item():
        raise ValueError("crops must contain only finite values")
    mean = crops.new_tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    std = crops.new_tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    return (crops - mean) / std


class HamerPriorAdapter:
    """Adapter around an injected HaMeR-like model; imports or assets are external."""

    def __init__(
        self,
        model: nn.Module,
        feature_provider: Callable[[Mapping[str, Tensor], Tensor], Tensor],
    ) -> None:
        self.model = model
        self.feature_provider = feature_provider

    def __call__(self, normalized_crops: Tensor) -> HandPriorOutput:
        b, v = normalized_crops.shape[:2]
        flat = normalized_crops.reshape(b * v, *normalized_crops.shape[2:])
        raw = self.model({"img": flat})
        if not isinstance(raw, Mapping):
            raise TypeError("injected prior must return a mapping")
        required = ("pred_keypoints_2d", "pred_keypoints_3d", "pred_vertices")
        missing = [name for name in required if not isinstance(raw.get(name), Tensor)]
        if missing:
            raise ValueError(f"injected prior is missing tensor outputs: {missing}")
        joints_2d = raw["pred_keypoints_2d"]
        visual = self.feature_provider(raw, joints_2d)
        return HandPriorOutput(
            joints_2d_normalized=joints_2d.reshape(b, v, *joints_2d.shape[1:]),
            visual_features=visual.reshape(b, v, *visual.shape[1:]),
            local_joints_3d=raw["pred_keypoints_3d"].reshape(b, v, *raw["pred_keypoints_3d"].shape[1:]),
            local_vertices_3d=raw["pred_vertices"].reshape(b, v, *raw["pred_vertices"].shape[1:]),
        )
