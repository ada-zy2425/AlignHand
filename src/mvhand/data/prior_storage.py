"""Lossy feature storage only; all predicted geometry stays in FP32."""
import numpy as np

from mvhand.data.v2_access import validate_input_field_names


def prior_storage_arrays(values, *, feature_dtype='float32'):
    if feature_dtype not in ('float16','float32'):
        raise ValueError('unsupported feature storage precision')
    validate_input_field_names(values)
    arrays={}
    for name,value in values.items():
        array=value.detach().cpu().numpy()
        dtype=feature_dtype if name in ('feature_maps','visual_features') else 'float32'
        stored=array.astype(dtype)
        if not np.isfinite(stored).all():
            raise ValueError('nonfinite prior data after storage conversion')
        arrays[name]=stored
    return arrays
