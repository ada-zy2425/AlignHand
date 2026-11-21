"""Evaluation metrics. Inputs in metres, every reported number in mm.

MPVPE / MPJPE (absolute and Procrustes-aligned), AUC, F-scores at 5 / 15 mm,
and sequence-level bootstrap confidence intervals. Joints use the 21-joint
MANO layout (16 regressed joints + 5 fingertip vertices).
"""
import os
import hashlib
from pathlib import Path

import numpy as np

STAT_SEED = 20261028
BOOT_B = 1000
TH = np.linspace(0.0, 50.0, 100)           # mm, includes 0 and 50

_REG_PATH = Path(os.environ.get('ALIGNHAND_DATA', 'data') + '/handoff/models/assets/'
                 'mano_base_joint_regressor_16x778.npy')


def load_q21():
    import torch
    from mvhand.geometry.mesh_joints import build_poem_published_21_regressor
    base = torch.as_tensor(np.load(_REG_PATH), dtype=torch.float64)
    q21 = build_poem_published_21_regressor(base).numpy().astype(np.float64)
    sha = dict(
        base_regressor=hashlib.sha256(_REG_PATH.read_bytes()).hexdigest(),
        q21_array=hashlib.sha256(np.ascontiguousarray(q21).tobytes())
        .hexdigest(),
        metrics2_source=hashlib.sha256(Path(__file__).read_bytes())
        .hexdigest())
    return q21, sha


def _mm(a, b):
    return float(np.linalg.norm(a - b, axis=-1).mean() * 1000.0)


def _sim3(src, dst):
    """Umeyama with positive scale, no reflection: maps src -> dst."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    u, s, vt = np.linalg.svd(cov)
    sgn = np.sign(np.linalg.det(u @ vt)) or 1.0
    d = np.diag([1.0, 1.0, sgn])
    rot = u @ d @ vt
    var = (sc ** 2).sum() / len(src)
    scale = float(np.trace(np.diag(s) @ d) / var)
    return lambda p: scale * (rot @ (p - mu_s).T).T + mu_d


def _auc(dist_mm):
    pck = (dist_mm[None, :] <= TH[:, None]).mean(1)
    return float(np.trapz(pck, TH) / 50.0)


def _nn_d2(x_mm, y_mm):
    # bidirectional NN distances on vertex point sets (778x778 per group)
    d2 = ((x_mm[:, None, :] - y_mm[None, :, :]) ** 2).sum(-1)
    return d2.min(1), d2.min(0)


def _fscore(nn, th):
    p = float((nn[0] <= th * th).mean())
    r = float((nn[1] <= th * th).mean())
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def group_metrics(X, Y, q21, light=False):
    """X, Y: (778,3) float64 metres -> dict of scalars (mm / unitless).
    light=True computes only LIGHT_KEYS (no AUC / F-score)."""
    jx, jy = q21 @ X, q21 @ Y
    out = dict(abs_v=_mm(X, Y), abs_j21=_mm(jx, jy),
               rr_wrist_v=_mm(X - jx[0], Y - jy[0]),
               rr_wrist_j=_mm(jx - jx[0], jy - jy[0]),
               rr_centroid_v=_mm(X - X.mean(0), Y - Y.mean(0)),
               rr_centroid_j=_mm(jx - X.mean(0), jy - Y.mean(0)))
    tv = _sim3(X, Y)
    xa = tv(X)
    out['pa_v'] = _mm(xa, Y)
    tj = _sim3(jx, jy)
    out['pa_j'] = _mm(tj(jx), jy)
    if light:
        return out
    dv = np.linalg.norm(X - Y, axis=-1) * 1000.0
    dj = np.linalg.norm(jx - jy, axis=-1) * 1000.0
    out['auc_v_raw'] = _auc(dv)
    out['auc_j_raw'] = _auc(dj)
    xm, ym = X * 1000.0, Y * 1000.0
    nn = _nn_d2(xm, ym)
    out['f5_raw'] = _fscore(nn, 5.0)
    out['f15_raw'] = _fscore(nn, 15.0)
    dva = np.linalg.norm(xa - Y, axis=-1) * 1000.0
    out['auc_v_pa'] = _auc(dva)
    nna = _nn_d2(xa * 1000.0, ym)
    out['f5_pa'] = _fscore(nna, 5.0)
    out['f15_pa'] = _fscore(nna, 15.0)
    return out


LIGHT_KEYS = ['abs_v', 'abs_j21', 'rr_wrist_v', 'rr_wrist_j',
              'rr_centroid_v', 'rr_centroid_j', 'pa_v', 'pa_j']
METRIC_KEYS = LIGHT_KEYS + ['auc_v_raw', 'auc_j_raw', 'f5_raw', 'f15_raw',
                            'auc_v_pa', 'f5_pa', 'f15_pa']


class SeqBoot:
    """Sequence-clustered bootstrap with a FIXED draw set shared by every
    statistic in one report (paired deltas therefore use the same draws)."""

    def __init__(self, seqs):
        self.seqs = np.asarray(seqs)
        self.useq = sorted(set(seqs))
        self.sidx = {s: np.where(self.seqs == s)[0] for s in self.useq}
        rng = np.random.default_rng(STAT_SEED)
        self.draws = [rng.choice(self.useq, size=len(self.useq), replace=True)
                      for _ in range(BOOT_B)]

    def ci90(self, x):
        ms = [float(np.mean(np.concatenate([x[self.sidx[s]] for s in d])))
              for d in self.draws]
        return [float(np.percentile(ms, 5)), float(np.percentile(ms, 95))]


def aggregate(x, seqs, sidx=None, useq=None):
    useq = useq or sorted(set(seqs))
    sidx = sidx or {s: np.where(np.asarray(seqs) == s)[0] for s in useq}
    return dict(group_mean=float(x.mean()),
                seq_mean=float(np.mean([x[sidx[s]].mean() for s in useq])),
                p95=float(np.quantile(x, 0.95)))
