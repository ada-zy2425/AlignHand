"""STAGE6_FINAL_V2_DEXYCB: the one DexYCB official-identifier S0-TEST batch
(FINAL_PLAN_2.json). Endpoints exported GT-free, ONE scoring read.

Stacks (per plan rows): U1_DEX (default dexycb FrozenForward: M1 + shipped R
s23 + its stats) with gate3 Refs s25/26/27 (gamma .5) and u15a Ref (gamma 1);
U2_SHARED (E3 stack); U2F_MERGED (u2merged lock). Arms: HG, A/X_R per stack,
Full per Ref. Output: results/paper/final_dexycb/.
usage: test_dexycb.py [--limit N]
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

PREP = Path(os.environ.get('ALIGNHAND_DATA', 'data') + '/prepared/v21_dexycb_tst_s0_v1')
CACHE = Path(os.environ.get('ALIGNHAND_DATA', 'data') + '/caches/dexycb/v21_s0_dexycb_tst_hamer_s25_v1')
U2R = Path(os.environ.get('ALIGNHAND_DATA', 'data') + '/results/u2')
U2F = Path(os.environ.get('ALIGNHAND_DATA', 'data') + '/results/u2merged')
OUT = Path(os.environ.get('ALIGNHAND_DATA', 'data') + '/repo/results/paper/final_dexycb')
GATE3 = {s: os.environ.get('ALIGNHAND_DATA', 'data') + f'/results/ref23/gate3{"" if s == 25 else f"_s{s}"}'
            f'/gate3/CHECKPOINT_EPOCH_010.pt' for s in (25, 26, 27)}
U15A = os.environ.get('ALIGNHAND_DATA', 'data') + '/results/s4/u15a_dexycb/REF_SINGLE.pt'
SEED = 20261023
# Internal comparison stacks (per-dataset U1 with its Refs, merged U2F) are
# not released; they are evaluated only when present. The released shared
# model is the 'u2' stack, whose arms are prefixed u2_ in the outputs.
HAVE_U1 = Path(U15A).exists() and all(Path(p).exists() for p in GATE3.values())
HAVE_U2F = (U2F / 'U2F_MODEL_LOCK.json').exists()
_W = {}


def _patch():
    import s_common
    from s_common import F2F, PREPARED, guard
    PREPARED[('dexycb', 'tst')] = F2F / 'prepared/v21_dexycb_tst_s0_v1'
    orig = s_common.cache_files

    def cache_files(ds, split):
        if ds != 'dexycb' or split != 'tst':
            return orig(ds, split)
        d = guard(F2F / 'caches/dexycb/v21_s0_dexycb_tst_hamer_s25_v1')
        index = json.loads((d / 'CACHE_INDEX.json').read_text())
        files = [d / 'prior_inputs' / e['file'] for e in index]
        prepared = guard(PREPARED[(ds, split)])
        rows = [json.loads(s) for s in
                (prepared / 'ROWS.jsonl').read_text().splitlines()]
        assert len(rows) == len(files)
        return files, rows, prepared

    s_common.cache_files = cache_files
    return s_common


def _build_ref(path):
    import torch
    from ref23_train import build_operators
    from mvhand.regional.refinement import MultiViewRefinement
    m = MultiViewRefinement(build_operators(), displacement_bound_m=.005,
                            include_correction_magnitude=True)
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m.load_state_dict(ck.get('state_dict', ck))
    return m.eval().requires_grad_(False)


def _init():
    import torch
    torch.set_num_threads(1)
    S = _patch()
    from mvhand.regional.candidates import CandidateEvidenceHead

    def alt_stack(stage_ck_path, r_ck_path, stats):
        fwd = S.FrozenForward('dexycb', 'tst')
        ck = torch.load(stage_ck_path, map_location='cpu',
                        weights_only=False)
        fwd.stage.load_state_dict(ck['trainable_state'], strict=True)
        fwd.stage.eval().requires_grad_(False)
        rck = torch.load(r_ck_path, map_location='cpu', weights_only=False)
        fwd.head = CandidateEvidenceHead()
        fwd.head.load_state_dict(rck.get('state_dict', rck), strict=True)
        fwd.head.eval().requires_grad_(False)
        st = torch.load(stats, map_location='cpu', weights_only=False)
        fwd.mean, fwd.scale = st['mean'], st['scale']
        return fwd

    if HAVE_U1:
        _W['u1'] = dict(fwd=S.FrozenForward('dexycb', 'tst'),
                        refs={f'Full_s{s}': (_build_ref(p), 0.5)
                              for s, p in GATE3.items()}
                        | {'Full_u15a': (_build_ref(U15A), 1.0)})
    _W['u2'] = dict(fwd=alt_stack(
        U2R / 'stage1_s20261023/CHECKPOINT_UPDATE_16800.pt',
        U2R / f'r_s{SEED}/R_FINAL.pt', U2R / 'U2_GEOMETRY_STATS.pt'),
        refs={'Full': (_build_ref(U2R / f'ref_s{SEED}/REF_FINAL.pt'),
                       None)})
    if HAVE_U2F:
        lock = json.loads((U2F / 'U2F_MODEL_LOCK.json').read_text())
        _W['u2f'] = dict(fwd=alt_stack(
            lock['stage1'], U2F / 'r/R_FINAL.pt',
            U2F / 'U2F_GEOMETRY_STATS.pt'),
            refs={'Full': (_build_ref(U2F / 'ref/REF_FINAL.pt'), None)})


def _endpoints(stack, i):
    import torch
    from ref23_train import ref_inputs
    fwd = stack['fwd']
    out = fwd.forward([i])
    if out is None:
        return None
    t = torch.as_tensor
    x0 = t(out['x0']).double()
    extras = {k: t(out[k]) for k in
              ('support_weights', 'reference_to_camera', 'raw_to_crop',
               'intrinsics', 'scaffold', 'correction_mag', 'placed')}
    extras['x0'] = x0
    extras['fm'] = t(out['fm'])
    extras['view_valid'] = t(out['view_valid']).bool()
    ent = dict(A=np.asarray(out['anchor'][0], np.float32),
               X_R=np.asarray(x0[0].numpy(), np.float32))
    with torch.no_grad():
        for name, (ref, gamma) in stack['refs'].items():
            res = ref(**ref_inputs(extras, 'cpu'))
            if gamma is None:
                xf = res.vertices.double()[0]
            else:
                xf = (x0 + gamma * (res.vertices.double() - x0))[0]
            ent[name] = np.asarray(xf.numpy(), np.float32)
    return ent


def _hg(i):
    import cv2, torch
    from mvhand.runtime.unified import uniform_geometry
    from mvhand.models.interfaces import Form2FrameBatch
    # prior outputs and crops only; identical for every stack
    fwd = _W['u1' if 'u1' in _W else 'u2']['fwd']
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
    batch = Form2FrameBatch(torch.zeros((1, mask.shape[0], 3, 1, 1)),
                            t(fwd.inputs['intrinsics'][[i]]),
                            t(fwd.inputs['camera_to_world'][[i]]),
                            t(inverse), t(mask[None]), t(affine))
    v, ok = uniform_geometry(joints_2d_norm=vals['joints_2d_normalized'],
                             local_joints=vals['local_joints_3d'],
                             local_vertices=vals['local_vertices_3d'],
                             batch=batch, geometry_blend=.5)
    return None if not bool(ok[0]) else np.asarray(v[0].numpy(), np.float32)


def _work(ids):
    rows_out = []
    for i in ids:
        ent = {}
        bad = None
        for name in [n for n in ('u1', 'u2', 'u2f') if n in _W]:
            eps = _endpoints(_W[name], i)
            if eps is None:
                bad = f'{name}_placement_fail'
                break
            pref = '' if name == 'u1' else name + '_'
            for arm, X in eps.items():
                ent[pref + arm] = X
        if bad:
            rows_out.append((i, None, bad))
            continue
        hg = _hg(i)
        if hg is not None:
            ent['HG'] = hg
        rows_out.append((i, ent, 'ok' if hg is not None else 'ok_hg_fail'))
    return rows_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in
            (PREP / 'ROWS.jsonl').read_text().splitlines()]
    n = len(rows) if not a.limit else a.limit
    OUT.mkdir(parents=True, exist_ok=True)
    pred_f = OUT / ('PRED_FINAL.npz' if not a.limit else
                    f'PRED_FINAL_SMOKE{n}.npz')
    if not pred_f.exists():
        items = [list(range(s_, min(s_ + 4, n))) for s_ in range(0, n, 4)]
        acc, fails = {}, {}
        t0 = time.time()
        with Pool(30, initializer=_init) as pool:
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
        pop = sorted(i for i, e in acc.items() if 'HG' in e)
        arms = sorted(acc[pop[0]])
        np.savez_compressed(
            pred_f, pop=np.array(pop),
            keys=np.array([f"{rows[i]['sequence_id'].replace('/', '|')}/"
                           f"{rows[i]['key']}" for i in pop]),
            arms=np.array(arms), unit=np.array('metres_f32'),
            fails=np.array(json.dumps({str(k): v
                                       for k, v in fails.items()})),
            **{f'V_{arm}': np.stack([acc[i][arm] for i in pop])
               for arm in arms})
        print('FINAL2_ENDPOINTS_EXPORTED', len(pop), flush=True)
    z = np.load(pred_f, allow_pickle=False)
    pop = z['pop']
    keys = [str(k) for k in z['keys']]
    arms = [str(x) for x in z['arms']]
    seqs = [k.split('/')[0] for k in keys]
    gt = np.load(PREP / 'PRIVATE_SCORE_TARGETS.npz',
                 allow_pickle=False)['vertices_reference_m']
    q21, shas = M2.load_q21()
    boot = M2.SeqBoot(seqs)
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    long_rows, summ = [], []
    for arm in arms:
        V = z[f'V_{arm}']
        per = {m: np.zeros(len(pop)) for m in M2.METRIC_KEYS}
        for gi in range(len(pop)):
            row = M2.group_metrics(V[gi].astype(np.float64),
                                   gt[pop[gi]].astype(np.float64), q21)
            for m in M2.METRIC_KEYS:
                per[m][gi] = row[m]
        for m in M2.METRIC_KEYS:
            agg = M2.aggregate(per[m], seqs, boot.sidx, boot.useq)
            ci90 = boot.ci90(per[m])
            long_rows.append(dict(
                dataset='dexycb', experiment='FINAL2', row='s0_test',
                method=arm, source_datasets='dexycb',
                model_role=('DEV_ROLE' if arm.startswith('u2') and not
                            arm.startswith('u2f') else
                            'FINAL_MERGED_U2' if arm.startswith('u2f')
                            else 'registered_U1'),
                stage1_seed=str(SEED), r_seed=str(SEED),
                ref_seed=str(SEED), replication_scope='registered',
                view_k='native', view_seed='', view_manifest_sha='',
                split='S0_TEST_8view', metric=m,
                alignment='pa' if 'pa' in m else 'raw',
                aggregation='group_mean', value=round(agg['group_mean'], 6),
                unit=('ratio' if 'auc' in m or m.startswith('f') else 'mm'),
                n_expected=len(rows), n_scored=len(pop),
                coverage=round(len(pop) / len(rows), 4),
                ci_low=round(ci90[0], 6), ci_high=round(ci90[1], 6),
                run_id='STAGE6_FINAL_V2_DEXYCB', ckpt_sha='see_plan',
                eval_key=f'metrics2@{code_sha}',
                input_provenance_key='v21_dexycb_tst_s0_v1',
                code_sha=code_sha, status='SUCCEEDED', reason=''))
            if m == 'abs_v':
                summ.append((arm, round(agg['group_mean'], 4),
                             round(ci90[0], 4), round(ci90[1], 4)))
    with open(OUT / 'runs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0]))
        w.writeheader()
        w.writerows(long_rows)
    lines = ['| arm | abs_v mm | ci90 |', '|---|---|---|'] + \
            [f'| {a_} | {v} | [{lo}, {hi}] |' for a_, v, lo, hi in summ]
    (OUT / 'table.md').write_text(
        '# STAGE6_FINAL_V2 - DexYCB official-identifier S0 TEST '
        '(8-view registered protocol, metrics v2)\n\n'
        + '\n'.join(lines) + '\n')
    (OUT / 'STATUS.json').write_text(json.dumps(dict(
        written_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        batch='STAGE6_FINAL_V2_DEXYCB', n_expected=len(rows),
        n_scored=len(pop), smoke=bool(a.limit), reader_shas=shas),
        indent=1))
    (OUT / 'ACCESS_LEDGER.json').write_text(json.dumps(dict(
        final_gt_scoring_reads=1, batch='STAGE6_FINAL_V2_DEXYCB',
        at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        build_reads='see dexycb_test_build/ACCESS_LEDGER.json'), indent=1))
    print('FINAL2_BATCH_DONE', json.dumps(
        {a_: v for a_, v, _, _ in summ}), flush=True)


if __name__ == '__main__':
    main()
