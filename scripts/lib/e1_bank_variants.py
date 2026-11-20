"""Stage 6 E1 (spec D1): variant input banks for the ablation rows that
change the Stage I -> triangulation interface.

Rows (what is swapped before triangulation; everything downstream --
recover_positioned_mesh, assemble_single_prior_inputs, export -- is the
unchanged s_common path):
  raw_uniform : corrected 2D := raw prior 2D (delta-u removed) AND view
                weights := uniform over valid views (joint removal row).
  no_du       : corrected 2D := raw prior 2D (delta-u = 0); the learned
                Stage I support weights are KEPT unchanged.
  uniform_w   : delta-u KEPT; weights := uniform over valid views, via the
                model's registered evaluation intervention
                view_weight_mode='uniform_valid' (form2frame.py).

Implementation: s_common._init is wrapped (a3_export_3view_bundle /
u2_chain.patched_export pattern) so each export worker's FrozenForward
stage.forward_from_prior returns a prediction with the row's (2D, weights)
and BOTH triangulations recomputed via the same triangulate_projective_dlt
call signature used inside forward_from_prior. Group indices replicate the
registered bundles (fit8192 meta indices; devfull meta indices).

Outputs: $ALIGNHAND_DATA/results/e1/<row>/bundles/<ds>/{fit,dev}
usage: e1_bank_variants.py --row raw_uniform|no_du|uniform_w
       --dataset dexycb|ho3d|hanco_a3 [--split fit|dev|both]
       [--workers 16] [--limit N]   (limit N -> smoke export to *_smokeN)
"""
import os
import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
sys.path.insert(0, os.environ.get('ALIGNHAND_SRC', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src')))

import s_common  # noqa: E402
from mvhand.geometry.raw_pixel_dlt import (  # noqa: E402
    normalize_relative_weights, triangulate_projective_dlt)
from mvhand.models.form2frame import (  # noqa: E402
    invert_crop_to_raw, normalized_crop_to_raw, raw_to_normalized_crop)

SRC = Path(os.environ.get('ALIGNHAND_DATA', 'data') + '/results/s1/bundles')
E1 = Path(os.environ.get('ALIGNHAND_DATA', 'data') + '/results/e1')
ROWS = ('raw_uniform', 'no_du', 'uniform_w')
SRC_NAME = {'fit': 'fit8192', 'dev': 'devfull'}
ROW = None  # set in main() before the Pool forks; workers inherit it.


def variant_prediction(pred, prior, batch, row, patch_size):
    """Swap (2D, weights) per row and recompute both triangulations."""
    joints = prior.joints_2d_normalized.shape[2]
    solver = batch.runtime_view_mask[:, :, None].expand(-1, -1, joints)
    corrected_raw = normalized_crop_to_raw(
        prior.joints_2d_normalized, batch.crop_to_raw, patch_size)
    raw_to_crop = (batch.raw_to_crop if batch.raw_to_crop is not None
                   else invert_crop_to_raw(batch.crop_to_raw))
    corrected_norm = raw_to_normalized_crop(corrected_raw, raw_to_crop,
                                            patch_size)
    uniform_weights = normalize_relative_weights(
        torch.zeros_like(solver, dtype=corrected_raw.dtype), solver,
    ).to(corrected_raw.dtype)
    uniform = triangulate_projective_dlt(
        corrected_raw, pred.projection_matrices, uniform_weights, solver)
    if row == 'raw_uniform':
        weights = uniform_weights
        logits = torch.zeros_like(pred.relative_view_logits)
        weighted = uniform
    elif row == 'no_du':
        # Keep the ORIGINAL learned support weights (computed by the full
        # forward, delta-u applied there); only the triangulated 2D changes.
        weights = pred.relative_view_weights
        logits = pred.relative_view_logits
        weighted = triangulate_projective_dlt(
            corrected_raw, pred.projection_matrices, weights, solver)
    else:
        raise ValueError(row)
    return dataclasses.replace(
        pred,
        corrected_joints_2d_raw=corrected_raw,
        corrected_joints_2d_normalized=corrected_norm,
        correction_normalized=torch.zeros_like(pred.correction_normalized),
        relative_view_logits=logits,
        relative_view_weights=weights,
        uniform=uniform,
        weighted=weighted,
    )


def install(fwd, row):
    if row == 'uniform_w':
        # Registered single-factor intervention inside the model itself:
        # same observations (delta-u kept), uniform weights + uniform DLT.
        fwd.stage.view_weight_mode = 'uniform_valid'
        return
    stage = fwd.stage
    orig = stage.forward_from_prior
    patch_size = stage.patch_size

    def patched(prior, batch):
        return variant_prediction(orig(prior, batch), prior, batch, row,
                                  patch_size)

    stage.forward_from_prior = patched


_orig_init = s_common._init


def _init_variant(ds, split):
    _orig_init(ds, split)
    install(s_common._CTX['fwd'], ROW)


s_common._init = _init_variant


def smoke_checks(out, ds, tag, limit):
    """Positions 0..limit-1 match the registered bundle (same index list)."""
    ref = SRC / ds / SRC_NAME[tag]
    checks = {}
    cm = np.asarray(np.load(out / 'correction_mag.npy', mmap_mode='r'))
    checks['correction_mag_max_px'] = float(np.abs(cm).max())
    sw = np.asarray(np.load(out / 'support_weights.npy', mmap_mode='r'))
    vv = np.asarray(np.load(out / 'view_valid.npy', mmap_mode='r'))
    w = np.where(vv[:, :, None], sw, np.nan)
    spread = np.nanmax(w, 1) - np.nanmin(w, 1)
    checks['support_weight_spread_max'] = float(np.nanmax(spread))
    sw_ref = np.asarray(np.load(ref / 'support_weights.npy',
                                mmap_mode='r')[:limit])
    checks['support_weights_vs_registered_maxdiff'] = float(
        np.abs(sw - sw_ref).max())
    an = np.asarray(np.load(out / 'anchor.npy', mmap_mode='r'))
    an_ref = np.asarray(np.load(ref / 'anchor.npy', mmap_mode='r')[:limit])
    checks['anchor_vs_registered_mean_mm'] = float(
        np.linalg.norm(an - an_ref, axis=-1).mean() * 1000)
    x0 = np.asarray(np.load(out / 'x0.npy', mmap_mode='r'))
    checks['x0_finite'] = bool(np.isfinite(x0).all())
    print('SMOKE_CHECKS', ds, tag, json.dumps(checks), flush=True)


def main():
    global ROW
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--row', required=True, choices=ROWS)
    ap.add_argument('--dataset', required=True,
                    choices=('dexycb', 'ho3d', 'hanco_a3'))
    ap.add_argument('--split', default='both', choices=('fit', 'dev', 'both'))
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    ds = a.dataset
    if ds == 'hanco_a3':
        import a3_export_bundles  # noqa: F401 (prepared/cache patches)
    ROW = a.row
    tags = ('fit', 'dev') if a.split == 'both' else (a.split,)
    for tag in tags:
        split = 'fit' if tag == 'fit' else 'development'
        meta = json.loads((SRC / ds / SRC_NAME[tag] / 'meta.json').read_text())
        idx = [int(i) for i in meta['indices']]
        if a.limit:
            idx = idx[:a.limit]
        name = tag + (f'_smoke{a.limit}' if a.limit else '')
        out = E1 / a.row / 'bundles' / ds / name
        if (out / 'meta.json').exists():
            print('EXPORTED_ALREADY', a.row, ds, name, flush=True)
            continue
        print('exporting', a.row, ds, name, len(idx), 'groups', flush=True)
        s_common.export_bundle(ds, split, idx, out, workers=a.workers)
        m = json.loads((out / 'meta.json').read_text())
        m['e1_row'] = a.row
        m['e1_intervention'] = dict(
            raw_uniform='2D := raw prior 2D; weights := uniform over valid '
                        'views; both DLTs recomputed',
            no_du='2D := raw prior 2D; learned support weights kept; '
                  'weighted DLT recomputed',
            uniform_w="view_weight_mode='uniform_valid' (delta-u kept, "
                      'uniform weights + uniform DLT)')[a.row]
        m['source_bundle'] = str(SRC / ds / SRC_NAME[tag])
        (out / 'meta.json').write_text(json.dumps(m))
        if a.limit:
            smoke_checks(out, ds, tag, a.limit)
    print('E1_BANK_VARIANTS_DONE', a.row, ds, flush=True)


if __name__ == '__main__':
    main()
