"""Independent CPU endpoint implementation used to cross-check training."""
import numpy as np

def numpy_metrics(vertices, target, regressor):
    """Independent CPU readout: no project metric/evaluator calls."""
    def joints(v):
        v = v.astype(np.float32)
        base = np.sum(regressor.astype(np.float32)[:, :, None] * v[None],
                      axis=1, dtype=np.float32)
        tips = v[[744, 320, 443, 555, 672]]
        return np.concatenate((base, tips))[[0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20]].astype(np.float64)
    v = np.linalg.norm(vertices.astype(np.float64)-target.astype(np.float64),axis=-1).mean()*1000
    j = np.linalg.norm(joints(vertices)-joints(target),axis=-1).mean()*1000
    return float(v), float(j)
