"""Stage 6 E4: direct source->target transfer cells (spec B3/D3).

One cell per run: the SOURCE dataset's U1 stack (source Stage I M1, source R,
source Ref, source-fitted geometry stats) evaluated on the TARGET Dev caches
with the target's true cameras / native view mask / registered crop. No
target-fitted statistics, no re-tuning, no checkpoint swap.

Source stacks (seed 20261023, the spec's initial E4 seed):
  hanco_a3 : frozen hanco M1 Stage I + a3clean R/REF_CHECKPOINT_EPOCH_010
             (seed20261023) + CLEAN_GEOMETRY_STATS (TrainCore-fitted)
  dexycb / ho3d : their M1 Stage I + shipped single_prior_R seed20261023 +
             u15a REF_SINGLE (per-dataset Ref, gamma=1 identity)
Arms: HG (prior+geometry, source-free), A, X_R, Full. Full metrics v2.
Output: results/paper/transfer/<source>__to__<target>/
usage: e4_transfer_eval.py --source ds --target ds [--limit N]
"""
import os
import argparse, csv, hashlib, json, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
sys.path.insert(0, os.environ.get('ALIGNHAND_SRC', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src')))
import metrics2 as M2  # noqa: E402

CLEAN = Path(os.environ.get('ALIGNHAND_DATA', 'data') + '/results/a3clean')
U15A = os.environ.get('ALIGNHAND_DATA', 'data') + '/results/s4/u15a_{ds}/REF_SINGLE.pt'
SEED = 20261023
V_OF = {'hanco_a3': 8, 'dexycb': 8, 'ho3d': 5}

_W = {}


def _source_parts(source):
    """(stage_parent_file, r_ckpt, stats(mean,scale), ref_ckpt, gamma)."""
    import torch
    import s_common as S
    if source == 'hanco_a3':
        var, dig, parent = S.PARENTS['hanco']
        rck = CLEAN / f'seed{SEED}' / 'R_CHECKPOINT_EPOCH_010.pt'
        stats = torch.load(CLEAN / 'CLEAN_GEOMETRY_STATS.pt',
                           map_location='cpu', weights_only=False)
        ref = CLEAN / f'seed{SEED}' / 'REF_CHECKPOINT_EPOCH_010.pt'
        gamma = None               # a3clean Ref outputs the endpoint itself
    else:
        var, dig, parent = S.PARENTS[source]
        root = (S.HANDOFF / 'models/single_prior_R' / source / var
                / f'seed{SEED}')
        rck = root / 'CHECKPOINT_EPOCH_010.pt'
        stats = torch.load(root / 'GEOMETRY_STATS.pt', map_location='cpu',
                           weights_only=False)
        ref = Path(U15A.format(ds=source))
        gamma = 1.0                # u15a registered identity
    mean = stats['mean'] if isinstance(stats, dict) and 'mean' in stats \
        else stats[0]
    scale = stats['scale'] if isinstance(stats, dict) and 'scale' in stats \
        else stats[1]
    return parent, rck, (mean, scale), ref, gamma


def _init(source, target):
    import torch
    torch.set_num_threads(1)
    if 'hanco_a3' in (source, target):
        import a3_3view_eval  # noqa: F401  (installs hanco_a3 patches)
    import s_common as S
    import ref23_train as R
    from mvhand.regional.refinement import MultiViewRefinement
    from mvhand.regional.candidates import CandidateEvidenceHead
    fwd = S.FrozenForward(target, 'development')
    parent, rck_path, (mean, scale), ref_path, gamma = _source_parts(source)
    if source != target:
        ck = torch.load(parent, map_location='cpu', weights_only=False)
        fwd.stage.load_state_dict(ck['trainable_state'], strict=True)
        fwd.stage.eval().requires_grad_(False)
        rck = torch.load(rck_path, map_location='cpu', weights_only=False)
        fwd.head = CandidateEvidenceHead()
        fwd.head.load_state_dict(rck.get('state_dict', rck), strict=True)
        fwd.head.eval().requires_grad_(False)
        fwd.mean, fwd.scale = mean, scale
    ops = R.build_operators()
    ref = MultiViewRefinement(ops, displacement_bound_m=.005,
                              include_correction_magnitude=True)
    rc = torch.load(ref_path, map_location='cpu', weights_only=False)
    ref.load_state_dict(rc.get('state_dict', rc))
    _W.update(fwd=fwd, ref=ref.eval(), gamma=gamma, R=R,
              q21=M2.load_q21()[0],
              gt=np.load(fwd.prepared / 'PRIVATE_SCORE_TARGETS.npz',
                         allow_pickle=False)['vertices_reference_m'])


def _hg(fwd, i):
    import cv2, torch
    from mvhand.runtime.unified import uniform_geometry
    from mvhand.models.interfaces import Form2FrameBatch
    vals = {}
    with np.load(fwd.files[i], allow_pickle=False) as z:
        for n in ('joints_2d_normalized', 'local_joints_3d',
                  'local_vertices_3d'):
            vals[n] = torch.as_tensor(z[n][0][None]).float()
    affine = fwd.inputs['raw_to_crop_s25'][[i]]
    inverse = np.stack([[cv2.invertAffineTransform(a) for a in row]
                        for row in affine])
    t = torch.as_tensor
    mask = fwd.inputs['runtime_view_mask'][i]
    batch = Form2FrameBatch(
        torch.zeros((1, mask.shape[0], 3, 1, 1)),
        t(fwd.inputs['intrinsics'][[i]]),
        t(fwd.inputs['camera_to_world'][[i]]),
        t(inverse), t(mask[None]), t(affine))
    v, ok = uniform_geometry(
        joints_2d_norm=vals['joints_2d_normalized'],
        local_joints=vals['local_joints_3d'],
        local_vertices=vals['local_vertices_3d'],
        batch=batch, geometry_blend=.5)
    return None if not bool(ok[0]) else np.asarray(v[0].numpy(), np.float64)


def _work(ids):
    import torch
    fwd, R, q21 = _W['fwd'], _W['R'], _W['q21']
    rows_out = []
    for i in ids:
        out = fwd.forward([i])
        if out is None:
            rows_out.append((i, None, 'placement_fail'))
            continue
        Y = np.asarray(_W['gt'][i], np.float64)
        ent, status = {}, 'ok'
        hg = _hg(fwd, i)
        ent['HG'] = (M2.group_metrics(hg, Y, q21) if hg is not None
                     else None)
        if hg is None:
            status = 'ok_hg_fail'
        ent['A'] = M2.group_metrics(
            np.asarray(out['anchor'][0], np.float64), Y, q21)
        t = torch.as_tensor
        x0 = t(out['x0']).double()
        ent['X_R'] = M2.group_metrics(np.asarray(x0[0].numpy()), Y, q21)
        extras = {kk: t(out[kk]) for kk in
                  ('support_weights', 'reference_to_camera', 'raw_to_crop',
                   'intrinsics', 'scaffold', 'correction_mag', 'placed')}
        extras['x0'] = x0
        extras['fm'] = t(out['fm'])
        extras['view_valid'] = t(out['view_valid']).bool()
        with torch.no_grad():
            res = _W['ref'](**R.ref_inputs(extras, 'cpu'))
        if _W['gamma'] is None:
            xf = res.vertices.double()[0]
        else:
            xf = (x0 + _W['gamma'] * (res.vertices.double() - x0))[0]
        ent['Full'] = M2.group_metrics(np.asarray(xf.numpy()), Y, q21)
        rows_out.append((i, ent, status))
    return rows_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, choices=list(V_OF))
    ap.add_argument('--target', required=True, choices=list(V_OF))
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    if 'hanco_a3' in (a.source, a.target):
        import a3_3view_eval  # noqa: F401
    import s_common as S
    files, rows, prepared = S.cache_files(a.target, 'development')
    n = len(rows) if not a.limit else a.limit
    rowmeta = [json.loads(l) for l in
               (Path(prepared) / 'ROWS.jsonl').read_text().splitlines()]
    outd = Path(os.environ.get('ALIGNHAND_DATA', 'data') + '/repo/results/paper/transfer/'
                f'{a.source}__to__{a.target}')
    outd.mkdir(parents=True, exist_ok=True)
    acc, fails = {}, {}
    items = [list(range(s_, min(s_ + 4, n))) for s_ in range(0, n, 4)]
    t0 = time.time()
    with Pool(38, initializer=_init, initargs=(a.source, a.target)) as pool:
        for ci, rows_out in enumerate(pool.imap_unordered(_work, items)):
            for i, ent, status in rows_out:
                if ent is None:
                    fails[i] = status
                else:
                    acc[i] = ent
                    if status != 'ok':
                        fails[i] = status
            if (ci + 1) % 150 == 0:
                print(json.dumps(dict(chunks=ci + 1, of=len(items),
                                      sec=round(time.time() - t0, 1))),
                      flush=True)
    arms = ['HG', 'A', 'X_R', 'Full']
    pop = sorted(i for i, e in acc.items()
                 if all(e.get(arm) is not None for arm in arms))
    seqs = [str(rowmeta[i]['sequence_id']) for i in pop]
    boot = M2.SeqBoot(seqs)
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    long_rows = []
    per = {arm: {m: np.array([acc[i][arm][m] for i in pop])
                 for m in M2.METRIC_KEYS} for arm in arms}
    for arm in arms:
        for m in M2.METRIC_KEYS:
            x = per[arm][m]
            agg = M2.aggregate(x, seqs, boot.sidx, boot.useq)
            ci90 = boot.ci90(x)
            long_rows.append(dict(
                dataset=a.target, experiment='E4',
                row=f'{a.source}__to__{a.target}', method=arm,
                source_datasets=a.source, model_role='DEV_ROLE',
                stage1_seed='frozen_M1_source', r_seed=str(SEED),
                ref_seed=str(SEED), replication_scope='single',
                view_k='native', view_seed='', view_manifest_sha='',
                split='dev', metric=m,
                alignment='pa' if 'pa' in m else 'raw',
                aggregation='group_mean', value=round(agg['group_mean'], 6),
                unit=('ratio' if 'auc' in m or m.startswith('f') else 'mm'),
                n_expected=len(rows), n_scored=len(pop),
                coverage=round(len(pop) / len(rows), 4),
                ci_low=round(ci90[0], 6), ci_high=round(ci90[1], 6),
                run_id=f'e4_{a.source}_{a.target}',
                ckpt_sha='see_STATUS', eval_key=f'metrics2@{code_sha}',
                input_provenance_key=f'e4_source_stack_{a.source}',
                code_sha=code_sha, status='SUCCEEDED', reason=''))
    with open(outd / 'runs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0]))
        w.writeheader()
        w.writerows(long_rows)
    np.savez_compressed(outd / 'PERGROUP.npz', pop=np.array(pop),
                        arms=np.array(arms),
                        metrics=np.array(M2.METRIC_KEYS),
                        M=np.stack([[per[arm][m] for m in M2.METRIC_KEYS]
                                    for arm in arms]).astype(np.float32),
                        seqs=np.array(seqs))
    (outd / 'STATUS.json').write_text(json.dumps(dict(
        written_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        source=a.source, target=a.target, n_dev=len(rows),
        fixed_population=len(pop),
        fails={str(i): s for i, s in fails.items()},
        source_stack=('a3clean s23 R/Ref + CLEAN_GEOMETRY_STATS + hanco M1'
                      if a.source == 'hanco_a3' else
                      f'{a.source} M1 + shipped R s23 + u15a Ref (gamma 1)'),
        note='source-fitted stats only; target cameras/views/crop are '
             'legitimate inputs; no target-fitted statistics',
        smoke=bool(a.limit)), indent=1))
    hl = {arm: round(float(per[arm]['abs_v'].mean()), 4) for arm in arms}
    print('E4_CELL_DONE', a.source, '->', a.target, json.dumps(hl),
          flush=True)


if __name__ == '__main__':
    main()
