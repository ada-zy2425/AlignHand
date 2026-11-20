"""Restricted loader for publisher-provided NumPy label pickles.

POEM-v2 stores each label as a pickle containing plain Python containers,
NumPy arrays, dtypes, and NumPy scalars.  General pickle deserialization can
import and invoke arbitrary callables, so the canonical reader admits only
the four NumPy constructors used by the frozen publisher payload.
"""
from __future__ import annotations

import io
import pickle
from typing import Any

import numpy as np


_ALLOWED_GLOBALS = {
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
    ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
    ("numpy.core.multiarray", "scalar"): np.core.multiarray.scalar,
    ("numpy._core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
    ("numpy._core.multiarray", "scalar"): np.core.multiarray.scalar,
}


class _RestrictedNumpyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        try:
            return _ALLOWED_GLOBALS[(module, name)]
        except KeyError as error:
            raise pickle.UnpicklingError(
                f"forbidden pickle global: {module}.{name}"
            ) from error


def restricted_numpy_loads(payload: bytes) -> Any:
    """Deserialize a NumPy-only pickle and reject trailing payload bytes."""
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("label pickle must be nonempty bytes")
    stream = io.BytesIO(payload)
    value = _RestrictedNumpyUnpickler(stream).load()
    if stream.tell() != len(payload):
        raise pickle.UnpicklingError("trailing bytes after label pickle")
    return value
