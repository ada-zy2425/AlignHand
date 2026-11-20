"""Round 3 H.3/H.4: UST-Hand-protocol scoring of the fixed HanCo models.

Population: prepared/v24_hanco_a3_ust20198_v1 (all 20,198 authors' frames,
8 views, s2.5 annotation-assisted crops). Models (frozen, inference only):
  HG      frozen HaMeR prior + uniform geometry (no trained parts)
  refit   HanCo FINAL refit stack (results/refit0to1199, MODEL_LOCK
          verified before any forward)      -> refit_A / refit_Full
  u2      locked shared stack (J1)          -> u2_A / u2_Full
Endpoints are exported GT-free FIRST; ONE scoring pass per model computes
every requested metric: UST root-aligned MPVPE/MPJPE (their regressor and
root convention) plus absolute / wrist-relative / PA in our convention
(metrics v2), on the full population AND, from the same outputs, the
legacy 1,685-group subset by key join.

UST metric: native MANO 16-joint base regressor + fingertip one-hot rows
at vertices 745/317/445/556/673, reorder
[0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20]; root = joint 9
(middle MCP, = native joint 4); prediction and GT each translated by
their own root; cam0 frame (the population's reference frame).

usage: test_hanco.py selftest            # synthetic + Development checks
       test_hanco.py run [--limit N] [--workers N]
"""
import os
import argparse, hashlib, json, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
sys.path.insert(0, os.environ.get('ALIGNHAND_SRC', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src')))
import metrics2 as M2  # noqa: E402

F2F = Path(os.environ.get('ALIGNHAND_DATA', 'data'))
PREP = F2F / 'prepared/v24_hanco_a3_ust20198_v1'
LEGACY = F2F / 'prepared/v24_hanco_a3_f1516_v1'
REFIT = F2F / 'results/refit0to1199'
# The HanCo-only refit stack is an internal comparison and is not released;
# without its MODEL_LOCK only HG and the shared model (u2) are evaluated.
HAVE_REFIT = (REFIT / 'MODEL_LOCK.json').exists()
U2R = F2F / 'results/u2'
OUTD = F2F / 'results/sr_followup/h_ust'
RES = F2F / 'srwt/followup/results'
LEDGER = F2F / 'srwt/followup/final_access.json'
SEED = 20261023
_W = {}

REG16 = np.load(F2F / 'handoff/models/assets/'
                'mano_base_joint_regressor_16x778.npy')
TIPS = [745, 317, 445, 556, 673]          # thumb,index,middle,ring,little
PERM = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18,
        10, 11, 12, 19, 7, 8, 9, 20]
ROOT_UST = 9                              # middle MCP in UST order


def ust_joints(V):
    j16 = REG16 @ V
    return np.concatenate([j16, V[TIPS]])[PERM]


def ust_metrics(Vp, Vg):
    jp, jg = ust_joints(Vp), ust_joints(Vg)
    rp, rg = jp[ROOT_UST], jg[ROOT_UST]
    mpvpe = float(np.linalg.norm((Vp - rp) - (Vg - rg),
                                 axis=-1).mean() * 1000)
    mpjpe = float(np.linalg.norm((jp - rp) - (jg - rg),
                                 axis=-1).mean() * 1000)
    return mpvpe, mpjpe


def selftest():
    rng = np.random.default_rng(0)
    out = {}
    out['perm_is_permutation'] = sorted(PERM) == list(range(21))
    out['root_is_native_joint_4_middle_mcp'] = PERM[ROOT_UST] == 4
    out['tip_rows_one_hot'] = True       # tips taken directly as vertices
    out['tip_positions_in_order'] = [PERM.index(16 + t) for t in range(5)]
    V = rng.normal(size=(778, 3))
    out['gt_vs_gt_zero'] = ust_metrics(V, V) == (0.0, 0.0)
    t = np.array([0.03, -0.02, 0.05])
    a, b = ust_metrics(V + t, V)
    out['translation_residual_mm'] = (a, b)
    out['translation_cancels'] = a < 1e-6 and b < 1e-6
    th = 0.3
    R = np.array([[np.cos(th), -np.sin(th), 0],
                  [np.sin(th), np.cos(th), 0], [0, 0, 1.]])
    a, _ = ust_metrics(V @ R.T, V)
    out['rotation_penalized'] = a > 1.0
    a, _ = ust_metrics(V * 1.1, V)
    out['scale_penalized'] = a > 1.0
    # Development pipeline check: GT as prediction -> 0; a real Dev
    # endpoint -> finite plausible mm (no test GT touched)
    dev = F2F / 'prepared/v24_hanco_a3_development_v1'
    gt = np.load(dev / 'PRIVATE_SCORE_TARGETS.npz',
                 allow_pickle=False)['vertices_reference_m']
    out['dev_gt_vs_gt'] = ust_metrics(gt[0].astype(np.float64),
                                      gt[0].astype(np.float64))
    e0 = np.load(F2F / 'repo/results/paper/e0/hanco_a3/PRED_DEV_NATIVE.npz',
                 allow_pickle=False)
    arm = [c for c in e0['arms'] if 'Ref' in str(c) or 'full' in str(c).lower()]
    V0 = e0['V_' + str(arm[0])][0].astype(np.float64)
    i0 = int(e0['pop'][0])
    mv, mj = ust_metrics(V0, gt[i0].astype(np.float64))
    out['dev_endpoint_ust_mpvpe_mm'] = round(mv, 3)
    out['dev_endpoint_plausible'] = 0.5 < mv < 30
    (RES / 'h3_scorer_selftest.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    assert all(v is True or not isinstance(v, bool) for v in out.values())
    print('H3_SELFTEST_PASS')


def cache_files_sr3():
    dirs = sorted((F2F / 'caches/hanco').glob('sr3_ust_hamer_*_v1'),
                  key=lambda d: int(d.name.split('_')[3]))
    files = []
    for d in dirs:
        for f in sorted(d.glob('group_*.npz')):
            if int(f.stem.split('_')[1]) != len(files):
                raise ValueError(f'shard order broken at {f}')
            files.append(f)
    return files


def _patch():
    import s_common
    from s_common import PREPARED, guard
    import a3_3view_eval  # noqa: F401  (installs hanco_a3 patches)
    PREPARED[('hanco_a3', 'development')] = PREP
    files = cache_files_sr3()
    rows = [json.loads(s) for s in
            (PREP / 'ROWS.jsonl').read_text().splitlines()]
    assert len(files) == len(rows), (len(files), len(rows))
    orig = s_common.cache_files

    def cache_files(ds, split):
        if ds == 'hanco_a3':
            return files, rows, guard(PREPARED[(ds, 'development')])
        return orig(ds, split)

    s_common.cache_files = cache_files
    return s_common


def _verify_locks():
    import torch
    if not HAVE_REFIT:
        return None
    lock = json.loads((REFIT / 'MODEL_LOCK.json').read_text())

    def sha(p):
        ck = torch.load(p, map_location='cpu', weights_only=False)
        sd = ck.get('trainable_state', ck.get('state_dict', ck))
        return hashlib.sha256(b''.join(
            np.ascontiguousarray(v.numpy()).tobytes()
            for _, v in sorted(sd.items()))).hexdigest()
    for key, pk in (('stage1_state_sha256', 'stage1'),
                    ('r_state_sha256', 'r'), ('ref_state_sha256', 'ref')):
        if sha(lock[pk]) != lock[key]:
            raise ValueError(f'MODEL_LOCK mismatch: {pk}')
    return lock


def _init():
    import torch
    torch.set_num_threads(1)
    S = _patch()
    from mvhand.regional.candidates import CandidateEvidenceHead
    from ref23_train import build_operators
    from mvhand.regional.refinement import MultiViewRefinement
    lock = (json.loads((REFIT / 'MODEL_LOCK.json').read_text())
            if HAVE_REFIT else None)

    def build(stage_ck, r_ck, mean, scale):
        fwd = S.FrozenForward('hanco_a3', 'development')
        ck = torch.load(stage_ck, map_location='cpu', weights_only=False)
        fwd.stage.load_state_dict(ck['trainable_state'], strict=True)
        fwd.stage.eval().requires_grad_(False)
        rck = torch.load(r_ck, map_location='cpu', weights_only=False)
        fwd.head = CandidateEvidenceHead()
        fwd.head.load_state_dict(rck.get('state_dict', rck), strict=True)
        fwd.head.eval().requires_grad_(False)
        fwd.mean, fwd.scale = mean, scale
        return fwd

    def ref_of(path):
        m = MultiViewRefinement(build_operators(),
                                displacement_bound_m=.005,
                                include_correction_magnitude=True)
        ck = torch.load(path, map_location='cpu', weights_only=False)
        m.load_state_dict(ck.get('state_dict', ck))
        return m.eval().requires_grad_(False)

    stu = torch.load(U2R / 'U2_GEOMETRY_STATS.pt', map_location='cpu',
                     weights_only=False)
    if HAVE_REFIT:
        st = torch.load(REFIT / 'REFIT_GEOMETRY_STATS.pt', map_location='cpu',
                        weights_only=False)
        _W['refit'] = dict(fwd=build(lock['stage1'], lock['r'],
                                     st['mean'], st['scale']),
                           ref=ref_of(lock['ref']))
    _W['u2'] = dict(fwd=build(
        U2R / 'stage1_s20261023/CHECKPOINT_UPDATE_16800.pt',
        U2R / f'r_s{SEED}/R_FINAL.pt', stu['mean'], stu['scale']),
        ref=ref_of(U2R / f'ref_s{SEED}/REF_FINAL.pt'))


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
    with torch.no_grad():
        res = stack['ref'](**ref_inputs(extras, 'cpu'))
    return dict(A=np.asarray(out['anchor'][0], np.float32),
                Full=np.asarray(res.vertices.double()[0].numpy(),
                                np.float32))


def _hg(i):
    import cv2, torch
    from mvhand.runtime.unified import uniform_geometry
    from mvhand.models.interfaces import Form2FrameBatch
    # prior outputs and crops only; identical for every stack
    fwd = _W['refit' if HAVE_REFIT else 'u2']['fwd']
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
    return None if not bool(ok[0]) else np.asarray(v[0].numpy(),
                                                   np.float32)


def _work(ids):
    rows_out = []
    for i in ids:
        ent, bad = {}, None
        for name in (('refit', 'u2') if HAVE_REFIT else ('u2',)):
            eps = _endpoints(_W[name], i)
            if eps is None:
                bad = f'{name}_placement_fail'
                break
            ent[f'{name}_A'] = eps['A']
            ent[f'{name}_Full'] = eps['Full']
        if bad:
            rows_out.append((i, None, bad))
            continue
        hg = _hg(i)
        if hg is None:
            rows_out.append((i, None, 'hg_fail'))
            continue
        ent['HG'] = hg
        rows_out.append((i, ent, 'ok'))
    return rows_out


def run(a):
    import csv
    _verify_locks()
    rows = [json.loads(s) for s in
            (PREP / 'ROWS.jsonl').read_text().splitlines()]
    n = len(rows) if not a.limit else a.limit
    OUTD.mkdir(parents=True, exist_ok=True)
    pred_f = OUTD / ('PRED_UST.npz' if not a.limit
                     else f'PRED_UST_SMOKE{n}.npz')
    arms = (['HG', 'refit_A', 'refit_Full', 'u2_A', 'u2_Full'] if HAVE_REFIT
            else ['HG', 'u2_A', 'u2_Full'])
    if not pred_f.exists():
        items = [list(range(s, min(s + 4, n))) for s in range(0, n, 4)]
        acc, fails = {}, {}
        t0 = time.time()
        with Pool(a.workers, initializer=_init) as pool:
            for ci, rws in enumerate(pool.imap_unordered(_work, items)):
                for i, ent, st in rws:
                    if ent is None:
                        fails[i] = st
                    else:
                        acc[i] = ent
                if (ci + 1) % 200 == 0:
                    print(json.dumps(dict(chunks=ci + 1, of=len(items),
                                          sec=round(time.time() - t0))),
                          flush=True)
        pop = sorted(acc)
        np.savez_compressed(
            pred_f, pop=np.array(pop),
            keys=np.array([rows[i]['key'] for i in pop]),
            arms=np.array(arms), unit=np.array('metres_f32'),
            fails=np.array(json.dumps({str(k): v
                                       for k, v in fails.items()})),
            **{f'V_{m}': np.stack([acc[i][m] for i in pop])
               for m in arms})
        print('H_ENDPOINTS_EXPORTED', len(pop), 'fails', len(fails),
              flush=True)
    if a.limit:
        print('SMOKE_DONE_NO_GT_READ')
        return
    z = np.load(pred_f, allow_pickle=False)
    pop = z['pop']
    keys = [str(k) for k in z['keys']]
    gt = np.load(PREP / 'PRIVATE_SCORE_TARGETS.npz',
                 allow_pickle=False)['vertices_reference_m']
    q21, _ = M2.load_q21()
    have_legacy = (LEGACY / 'ROWS.jsonl').exists()
    legacy_keys = ({json.loads(s)['key'] for s in
                    (LEGACY / 'ROWS.jsonl').read_text().splitlines()}
                   if have_legacy else set())
    in_legacy = np.array([k in legacy_keys for k in keys])
    METRICS = ['ust_mpvpe_root', 'ust_mpjpe_root', 'abs_v', 'abs_j21',
               'rr_wrist_v', 'rr_wrist_j', 'pa_v', 'pa_j']
    out_rows = []
    for arm in arms:
        V = z[f'V_{arm}']
        per = {m: np.zeros(len(pop)) for m in METRICS}
        for gi in range(len(pop)):
            Vp = V[gi].astype(np.float64)
            Vg = gt[pop[gi]].astype(np.float64)
            mv, mj = ust_metrics(Vp, Vg)
            per['ust_mpvpe_root'][gi] = mv
            per['ust_mpjpe_root'][gi] = mj
            lm = M2.group_metrics(Vp, Vg, q21, light=True)
            for m in METRICS[2:]:
                per[m][gi] = lm[m]
        pops = [('ust20198', np.ones(len(pop), bool))]
        if have_legacy:
            pops.append(('legacy1685_subset', in_legacy))
        for popname, sel in pops:
            for m in METRICS:
                out_rows.append(dict(
                    population=popname, arm=arm, metric=m,
                    joint_convention=('UST/HaMuCo 21 (their regressor)'
                                      if m.startswith('ust') else
                                      'POEM-21 (metrics v2)'),
                    alignment=('root translation only (middle MCP)'
                               if m.startswith('ust') else
                               'per metrics v2'),
                    value=round(float(per[m][sel].mean()), 4), unit='mm',
                    n_expected=(20198 if popname == 'ust20198' else 1685),
                    n_scored=int(sel.sum())))
        print('scored', arm, flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    with open(RES / 'h_hanco_usthand_protocol.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    led = json.loads(LEDGER.read_text()) if LEDGER.exists() else []
    models = [('HG (frozen HaMeR + uniform geometry, no trained parts)',
               'prior e5cc06f2 (capture REPORT)')]
    if HAVE_REFIT:
        models.append(('HanCo FINAL refit stack (frozen)',
                       'results/refit0to1199 MODEL_LOCK (state shas verified)'))
    models.append(('shared U2 stack (frozen, J1 lock)',
                   'results/u2 s20261023 (J1 file shas)'))
    for model, ck in models:
        led.append(dict(
            model=model, checkpoint=ck,
            population='HanCo UST-protocol test (20,198 frames, seqs '
                       '1200-1517, 8 views); legacy 1,685-group subset '
                       'derived from the same outputs',
            date=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            purpose='Round 3 Part H: one scoring pass per fixed model '
                    'with UST root-aligned + our-convention metrics'))
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(led, indent=1))
    summ = {r['arm'] + '|' + r['metric']: r['value'] for r in out_rows
            if r['population'] == 'ust20198'
            and r['metric'] in ('ust_mpvpe_root', 'ust_mpjpe_root')}
    print('SR3_H_SCORED', json.dumps(summ))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('selftest', 'run'))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()
    if a.mode == 'selftest':
        selftest()
    else:
        run(a)
