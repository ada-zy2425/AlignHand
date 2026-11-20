"""Shared reference-view target contract for canonical multi-view labels."""
from __future__ import annotations

from typing import Mapping

import numpy as np


def reference_target(label: Mapping[str, object], name: str, count: int) -> np.ndarray:
    value = np.asarray(label[name])
    if value.shape == (8, count, 3):
        value = value[0]
    elif value.shape != (count, 3):
        raise ValueError(
            f"{name} must be shared [{count},3] or per-view [8,{count},3]"
        )
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise ValueError(f"{name} target must be finite numeric values")
    return value
