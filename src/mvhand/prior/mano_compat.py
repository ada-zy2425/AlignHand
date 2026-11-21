"""Import the MANO pickle dependency (chumpy) with modern NumPy/Python."""
from contextlib import contextmanager
import inspect

import numpy as np


@contextmanager
def legacy_mano_imports():
    """Restore removed spelling aliases only while loading the legacy pickle.

    No numerical dtype, model parameter, or dependency file is changed. The
    aliases match the builtins used by NumPy before their removal in 1.24.
    """
    aliases = dict(bool=bool, int=int, float=float, complex=complex,
                   object=object, str=str, unicode=str)
    added = [name for name in aliases if name not in np.__dict__]
    for name in added:
        setattr(np, name, aliases[name])
    added_inspect = not hasattr(inspect, 'getargspec')
    if added_inspect:
        inspect.getargspec = inspect.getfullargspec
    try:
        yield
    finally:
        for name in added:
            delattr(np, name)
        if added_inspect:
            delattr(inspect, 'getargspec')
