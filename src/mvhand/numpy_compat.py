"""Narrow compatibility boundary for legacy chumpy/MANO pickles."""
from __future__ import annotations

import numpy as np


def install_numpy_chumpy_compat() -> tuple[str, ...]:
    """Restore aliases removed by NumPy 1.24 before importing chumpy.

    The aliases reproduce their historical Python/NumPy scalar targets and do
    not alter arrays, dtypes, model parameters, or numerical kernels.
    """
    installed = []
    for name, value in (
        ("bool", np.bool_), ("int", int), ("float", float),
        ("complex", complex), ("object", np.object_), ("unicode", str),
        ("str", str),
    ):
        if name not in np.__dict__:
            setattr(np, name, value)
            installed.append(name)
    return tuple(installed)
