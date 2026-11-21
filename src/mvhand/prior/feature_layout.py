#!/usr/bin/env python3
"""Capture-only conversion from HaMeR logical layout to frozen cache storage."""
from __future__ import annotations

import torch
from torch import Tensor


VIEWS = 8
CHANNELS = 1280
LOGICAL_HW = (16, 12)
STORAGE_HW = (12, 16)


def canonical_storage_feature_maps(feature_map: Tensor) -> Tensor:
    """Serialize a traced logical [16,12] map into row-major [12,16] storage."""
    if feature_map.ndim != 4 or tuple(feature_map.shape[1:]) != (
        CHANNELS,
        *LOGICAL_HW,
    ):
        raise ValueError(
            "backbone logical feature-map contract requires "
            f"(N,{CHANNELS},{LOGICAL_HW[0]},{LOGICAL_HW[1]}), "
            f"got {tuple(feature_map.shape)}"
        )
    if not feature_map.is_contiguous():
        raise ValueError("backbone logical feature map must be row-major contiguous")
    if feature_map.shape[0] % VIEWS:
        raise ValueError("backbone logical feature-map batch is not view-aligned")
    return feature_map.reshape(feature_map.shape[0], CHANNELS, *STORAGE_HW)
