"""A3 3-view evaluation on the CLEAN retrained stack (frozen protocol).

Protocol: $ALIGNHAND_DATA/protocol/hanco_a3_3view.json (frozen pre-eval).
Development rule: primary = camera_ids[0]; 2 auxiliaries per draw seed
s in {20261021,20261022,20261023}; main result = mean of the three draws.
Final (1200+) is NOT touched here.

Per draw and group: one stage1 forward with runtime_view_mask restricted to
the 3 selected cameras (true 3-view fusion, not a post-hoc column mask),
then on the masked outputs:
    shippedR   the handoff R (leak-contaminated; comparison only)
    cleanR x3  the a3clean retrained heads (seeds 20261023/24/25)
    cleanRef x3  the a3clean Refs on their own R's x0
GT scores only (dev). Groups whose 3-view placement fails are excluded from
the means and counted (all arms affected identically).
Outputs $ALIGNHAND_DATA/results/a3clean/A3_3VIEW_{dev256,devfull}.json.
"""
import os
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import s_common
from s_common import F2F, PREPARED, guard
import torch

HANDOFF = Path(os.environ.get('ALIGNHAND_DATA', 'data') + '/handoff')
PROTO = json.loads(Path(os.environ.get('ALIGNHAND_DATA', 'data') + '/protocol/hanco_a3_3view.json').read_text())
CLEAN = Path(os.environ.get('ALIGNHAND_DATA', 'data') + '/results/a3clean')
SEEDS = (20261023, 20261024, 20261025)
DRAWS = ('20261021', '20261022', '20261023')

PREPARED[('hanco_a3', 'fit')] = F2F / 'prepared/v24_hanco_a3_fit_v1'
PREPARED[('hanco_a3', 'development')] = F2F / 'prepared/v24_hanco_a3_development_v1'
s_common.PARENTS['hanco_a3'] = s_common.PARENTS['hanco']
link = HANDOFF / 'models/single_prior_R/hanco_a3'
if not link.exists() and (HANDOFF / 'models/single_prior_R/hanco').exists():
    link.symlink_to(HANDOFF / 'models/single_prior_R/hanco')

_orig_cache_files = s_common.cache_files


def cache_files(ds, split):
    if ds != 'hanco_a3':
        return _orig_cache_files(ds, split)
    root = guard(F2F / 'caches/hanco')
    tag = 'dev' if split == 'development' else 'fit'
    dirs = sorted(root.glob(f'v24_a3_{tag}_hamer_*_v1'),
                  key=lambda d: int(d.name.split('_')[4]))
    files = []
    for d in dirs:
        index = json.loads((d / 'CACHE_INDEX.json').read_text())
        for i, entry in enumerate(index):
            if entry['index'] != len(files):
                raise ValueError(f'cache index out of order in {d}')
            files.append(d / entry['file'])
    prepared = guard(PREPARED[(ds, split)])
    rows = [json.loads(s) for s in (prepared / 'ROWS.jsonl').read_text().splitlines()]
    if len(rows) != len(files) or any(r['split'] != split for r in rows):
        raise ValueError('cache/prepared mismatch')
    return files, rows, prepared


s_common.cache_files = cache_files

_W = {}


def _init():
    from mvhand.regional.candidates import CandidateEvidenceHead
    from mvhand.regional.refinement import MultiViewRefinement
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
    from ref23_train import build_operators
    torch.set_num_threads(1)
    fwd = s_common.FrozenForward('hanco_a3', 'development')
    stats = torch.load(CLEAN / 'CLEAN_GEOMETRY_STATS.pt', map_location='cpu',
                       weights_only=False)
    heads, refs = {}, {}
    ops = build_operators()
    for s in SEEDS:
        h = CandidateEvidenceHead()
        h.load_state_dict(torch.load(CLEAN / f'seed{s}/R_CHECKPOINT_EPOCH_010.pt',
                                     map_location='cpu', weights_only=False)['state_dict'])
        heads[s] = h.eval().requires_grad_(False)
        m = MultiViewRefinement(ops, displacement_bound_m=.005,
                                include_correction_magnitude=True)
        m.load_state_dict(torch.load(CLEAN / f'seed{s}/REF_CHECKPOINT_EPOCH_010.pt',
                                     map_location='cpu', weights_only=False)['state_dict'])
        refs[s] = m.eval().requires_grad_(False)
    reg = torch.from_numpy(np.load(
        HANDOFF / 'models/assets/mano_base_joint_regressor_16x778.npy')).double()
    gt = np.load(Path(fwd.prepared) / 'PRIVATE_SCORE_TARGETS.npz',
                 allow_pickle=False)['vertices_reference_m'].astype(np.float64)
    _W.update(fwd=fwd, heads=heads, refs=refs, reg=reg, gt=gt,
              stats=(stats['mean'], stats['scale']),
              rvm0=np.array(fwd.inputs['runtime_view_mask']))


def out_to_batch(out):
    from mvhand.regional.single_prior import SinglePriorBank
    a = lambda k: torch.as_tensor(np.ascontiguousarray(out[k]))
    count = out['candidates'].shape[1]
    views = out['view_valid'].shape[1]
    source = torch.zeros(count, 3, dtype=torch.float64)
    source[0, 0] = 1
    source[1:, 1] = 1
    origins = torch.arange(-1, views)
    bank = SinglePriorBank(a('anchor').double(), a('candidates').double(),
                           a('valid').bool(), a('q_init').double(),
                           origins, source)
    inputs = {'bank': bank,
              'geometry_raw': a('geometry_raw').double(),
              'geometry_valid': a('geometry_valid').bool(),
              'candidate_raw': a('candidate_raw').double(),
              'candidate_grid': a('candidate_grid').double(),
              'projection_valid': a('projection_valid').bool(),
              'anchor_raw': a('anchor_raw').double(),
              'anchor_grid': a('anchor_grid').double(),
              'anchor_valid': a('anchor_valid').bool(),
              'view_valid': a('view_valid').bool()}
    fm = a('fm')
    extras = {k: a(k) for k in
              ('support_weights', 'corrected_raw', 'projection',
               'reference_to_camera', 'raw_to_crop', 'intrinsics',
               'scaffold', 'correction_mag', 'placed', 'placed_valid', 'x0')}
    extras['fm'] = fm
    extras['view_valid'] = inputs['view_valid']
    return inputs, fm, extras


def _work(item):
    """item = (ids, draw). Returns rows of per-group metrics or failures."""
    from mvhand.regional.single_prior import forward_single_prior
    from ref23_train import ref_inputs
    from mvhand.regional.training import errors
    ids, draw = item
    fwd = _W['fwd']
    rvm0 = _W['rvm0']
    mean, scale = _W['stats']
    rows_out = []
    for i in ids:
        row = fwd.rows[i]
        key = row['key']
        prot = PROTO['development'].get(key)
        if prot is None:
            rows_out.append((i, None, 'no_protocol'))
            continue
        cams3 = prot['draws'][draw]
        cam_ids = row['camera_ids']
        mask = np.array([c in cams3 for c in cam_ids], bool)
        if mask.sum() != 3:
            rows_out.append((i, None, 'bad_mask'))
            continue
        patched = rvm0[i].copy()
        orig = rvm0[i] & True
        fwd.inputs['runtime_view_mask'] = rvm0.copy()
        fwd.inputs['runtime_view_mask'][i] = orig & mask
        try:
            out = fwd.forward([i])
        finally:
            fwd.inputs['runtime_view_mask'] = rvm0
        del patched
        if out is None:
            rows_out.append((i, None, 'placement_fail'))
            continue
        inputs, fm, extras = out_to_batch(out)
        gt = torch.as_tensor(_W['gt'][[i]])
        reg = _W['reg']
        ent = {}
        with torch.no_grad():
            sv, _ = errors(extras['x0'].double(), gt, reg)
            ent['shippedR'] = float(sv[0])
            for s in SEEDS:
                res = forward_single_prior(_W['heads'][s], fwd.basis, inputs, fm,
                                           mean=mean, scale=scale)
                cv, _ = errors(res.vertices, gt, reg)
                ent[f'cleanR_{s}'] = float(cv[0])
                ex2 = dict(extras)
                ex2['x0'] = res.vertices.detach()
                rres = _W['refs'][s](**ref_inputs(ex2, 'cpu'))
                rv, _ = errors(rres.vertices, gt, reg)
                ent[f'cleanRef_{s}'] = float(rv[0])
        rows_out.append((i, ent, 'ok'))
    return draw, rows_out


def run(tag, limit):
    files, rows, prepared = cache_files('hanco_a3', 'development')
    n = len(rows)
    idx = list(range(n))
    if limit:
        idx = sorted(np.random.default_rng(20261026).choice(
            n, size=min(limit, n), replace=False).tolist())
    items = []
    for draw in DRAWS:
        for start in range(0, len(idx), 4):
            items.append((idx[start:start + 4], draw))
    acc = {d: {} for d in DRAWS}
    fails = {d: {} for d in DRAWS}
    t0 = time.time()
    with Pool(40, initializer=_init) as pool:
        for k, (draw, rows_out) in enumerate(pool.imap_unordered(_work, items)):
            for i, ent, status in rows_out:
                if ent is None:
                    fails[draw][i] = status
                else:
                    acc[draw][i] = ent
            if (k + 1) % 50 == 0:
                done = sum(len(v) for v in acc.values())
                print(json.dumps(dict(tag=tag, chunks=k + 1, of=len(items),
                                      scored=done,
                                      sec_per_group=round((time.time() - t0) /
                                                          max(done, 1), 2))),
                      flush=True)
    arms = ['shippedR'] + [f'cleanR_{s}' for s in SEEDS] + \
           [f'cleanRef_{s}' for s in SEEDS]
    rep = dict(tag=tag, n_groups=len(idx), draws={}, fails={
        d: len(v) for d, v in fails.items()})
    for d in DRAWS:
        vals = acc[d]
        rep['draws'][d] = {a: float(np.mean([e[a] * 1.0 for e in vals.values()]))
                           for a in arms} | dict(n=len(vals))
    rep['mean_of_draws'] = {a: float(np.mean([rep['draws'][d][a] for d in DRAWS]))
                            for a in arms}
    rep['mean_of_draws']['cleanR_seedmean'] = float(np.mean(
        [rep['mean_of_draws'][f'cleanR_{s}'] for s in SEEDS]))
    rep['mean_of_draws']['cleanRef_seedmean'] = float(np.mean(
        [rep['mean_of_draws'][f'cleanRef_{s}'] for s in SEEDS]))
    out = CLEAN / f'A3_3VIEW_{tag}.json'
    out.write_text(json.dumps(rep, indent=1))
    print(f'A3_3VIEW_{tag}_DONE', json.dumps(rep['mean_of_draws']), flush=True)


if __name__ == '__main__':
    if not (CLEAN / 'A3_3VIEW_dev256.json').exists():
        run('dev256', 256)
    run('devfull', None)
    print('A3_3VIEW_ALL_DONE', flush=True)
