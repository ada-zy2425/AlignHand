"""Round 3 J3: Development analyses with the locked shared model (U2).

Inference only, Development only, locked U2 stack everywhere:
Stage I  results/u2/stage1_s20261023/CHECKPOINT_UPDATE_16800.pt
R        results/u2/r_s20261023/R_FINAL.pt      (+ U2_GEOMETRY_STATS.pt)
Ref      results/u2/ref_s20261023/REF_FINAL.pt  (gamma = 1: Ref output)

Tasks (same group lists as the registered per-dataset analyses; common
valid denominator per paired comparison; failures recorded, never
dropped silently):
  stagewise      HG / A / fusion-only (shared R out, X_R) / Full
  interventions  anchor A under learned / zero-correction (no_du) /
                 uniform support (uniform_w) / both (raw_uniform),
                 via the registered e1_bank_variants interventions
  views          HG / A / Full at k=3..V-1 x 3 registered nested draws
                 + native (dexycb 3-8, ho3d 3-5)
  occlusion      Full under learned vs uniform_valid support on the
                 registered clean_rerun and occhalf prior caches
                 (same frames, occluded cameras, masks, seeds)
usage: dev_analysis.py --task stagewise|interventions|views|occlusion
                 --dataset dexycb|ho3d|hanco_a3 [--limit N] [--workers N]
"""
import os
import argparse, csv, json, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
sys.path.insert(0, os.environ.get('ALIGNHAND_SRC', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src')))
import metrics2 as M2  # noqa: E402

F2F = Path(os.environ.get('ALIGNHAND_DATA', 'data'))
U2 = F2F / 'results/u2'
RES = F2F / 'srwt/followup/results'
NODE = F2F / 'results/sr_followup/j3'
VSEEDS = (20261021, 20261022, 20261023)
V_OF = {'hanco_a3': 8, 'dexycb': 8, 'ho3d': 5}
OCC = F2F / 'results/e8_occlusion'
MKEYS = ['abs_v', 'abs_j21', 'rr_wrist_v', 'rr_wrist_j', 'pa_v', 'pa_j']
_W = {}


def u2_forward(ds):
    import torch
    import s_common as S
    from mvhand.regional.candidates import CandidateEvidenceHead
    if ds == 'hanco_a3':
        import a3_3view_eval  # noqa: F401 (installs hanco_a3 patches)
    fwd = S.FrozenForward(ds, 'development')
    ck = torch.load(U2 / 'stage1_s20261023/CHECKPOINT_UPDATE_16800.pt',
                    map_location='cpu', weights_only=False)
    fwd.stage.load_state_dict(ck['trainable_state'], strict=True)
    fwd.stage.eval().requires_grad_(False)
    rck = torch.load(U2 / 'r_s20261023/R_FINAL.pt', map_location='cpu',
                     weights_only=False)
    fwd.head = CandidateEvidenceHead()
    fwd.head.load_state_dict(rck.get('state_dict', rck), strict=True)
    fwd.head.eval().requires_grad_(False)
    st = torch.load(U2 / 'U2_GEOMETRY_STATS.pt', map_location='cpu',
                    weights_only=False)
    fwd.mean, fwd.scale = st['mean'], st['scale']
    return fwd


def u2_ref():
    import torch
    from ref23_train import build_operators
    from mvhand.regional.refinement import MultiViewRefinement
    m = MultiViewRefinement(build_operators(), displacement_bound_m=.005,
                            include_correction_magnitude=True)
    ck = torch.load(U2 / 'ref_s20261023/REF_FINAL.pt', map_location='cpu',
                    weights_only=False)
    m.load_state_dict(ck.get('state_dict', ck))
    return m.eval().requires_grad_(False)


def full_from_out(out, refm):
    import torch
    from ref23_train import ref_inputs
    t = torch.as_tensor
    x0 = t(out['x0']).double()
    extras = {k: t(out[k]) for k in
              ('support_weights', 'reference_to_camera', 'raw_to_crop',
               'intrinsics', 'scaffold', 'correction_mag', 'placed')}
    extras['x0'] = x0
    extras['fm'] = t(out['fm'])
    extras['view_valid'] = t(out['view_valid']).bool()
    with torch.no_grad():
        res = refm(**ref_inputs(extras, 'cpu'))
    return np.asarray(x0[0].numpy()), np.asarray(
        res.vertices.double()[0].numpy())


def hg_of(fwd, i, mask):
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
    batch = Form2FrameBatch(torch.zeros((1, mask.shape[0], 3, 1, 1)),
                            t(fwd.inputs['intrinsics'][[i]]),
                            t(fwd.inputs['camera_to_world'][[i]]),
                            t(inverse), t(mask[None]), t(affine))
    v, ok = uniform_geometry(joints_2d_norm=vals['joints_2d_normalized'],
                             local_joints=vals['local_joints_3d'],
                             local_vertices=vals['local_vertices_3d'],
                             batch=batch, geometry_blend=.5)
    return np.asarray(v[0].numpy(), np.float64) if bool(ok[0]) else None


def _init(ds, task, row=None):
    import torch
    torch.set_num_threads(1)
    _W['ds'] = ds
    _W['task'] = task
    _W['q21'] = M2.load_q21()[0]
    fwd = u2_forward(ds)
    if task == 'interventions' and row and row != 'learned':
        import e1_bank_variants as E1
        E1.install(fwd, row)
    _W['fwd'] = fwd
    if task in ('stagewise', 'views', 'occlusion'):
        _W['ref'] = u2_ref()
    _W['gt'] = np.load(Path(fwd.prepared) / 'PRIVATE_SCORE_TARGETS.npz',
                       allow_pickle=False)['vertices_reference_m']
    _W['rvm0'] = np.array(fwd.inputs['runtime_view_mask'])
    if task == 'views':
        _W['perms'] = {int(k[1:]): v for k, v in np.load(
            F2F / f'repo/results/paper/views/{ds}/VIEW_MANIFESTS.npz')
            .items()}
    if task == 'occlusion':
        _W['conds'] = {c: OCC / ds / c for c in ('clean_rerun', 'occhalf')}
        _W['files0'] = list(fwd.files)
        fwd.stage._orig_mode = fwd.stage.view_weight_mode


def m_of(X, Y):
    return M2.group_metrics(np.asarray(X, np.float64),
                            np.asarray(Y, np.float64), _W['q21'],
                            light=True)


def work_stagewise(ids):
    fwd, gt = _W['fwd'], _W['gt']
    out_rows = []
    for i in ids:
        out = fwd.forward([i])
        if out is None:
            out_rows.append((i, None, 'placement_fail'))
            continue
        Y = gt[i]
        x0, xf = full_from_out(out, _W['ref'])
        ent = dict(A=m_of(out['anchor'][0], Y), X_R=m_of(x0, Y),
                   Full=m_of(xf, Y))
        hg = hg_of(fwd, i, _W['rvm0'][i])
        ent['HG'] = m_of(hg, Y) if hg is not None else None
        ent['_anchor'] = np.asarray(out['anchor'][0], np.float32)
        ent['_full'] = xf.astype(np.float32)
        out_rows.append((i, ent, 'ok' if hg is not None else 'ok_hg_fail'))
    return out_rows


def work_interventions(ids):
    fwd, gt = _W['fwd'], _W['gt']
    out_rows = []
    for i in ids:
        out = fwd.forward([i])
        if out is None:
            out_rows.append((i, None, 'placement_fail'))
            continue
        out_rows.append((i, dict(A=m_of(out['anchor'][0], gt[i])), 'ok'))
    return out_rows


def work_views(item):
    ids, k, seedname = item
    fwd, gt, rvm0 = _W['fwd'], _W['gt'], _W['rvm0']
    V = V_OF[_W['ds']]
    out_rows = []
    for i in ids:
        if seedname == 'native':
            mask = rvm0[i].copy()
        else:
            sel = np.zeros(V, bool)
            sel[0] = True
            sel[_W['perms'][int(seedname)][i][:k - 1]] = True
            mask = rvm0[i] & sel
        if mask.sum() != k:
            out_rows.append((i, None, 'bad_mask'))
            continue
        fwd.inputs['runtime_view_mask'] = rvm0.copy()
        fwd.inputs['runtime_view_mask'][i] = mask
        try:
            out = fwd.forward([i])
        finally:
            fwd.inputs['runtime_view_mask'] = rvm0
        if out is None:
            out_rows.append((i, None, 'placement_fail'))
            continue
        Y = gt[i]
        _, xf = full_from_out(out, _W['ref'])
        hg = hg_of(fwd, i, mask)
        ent = dict(A=m_of(out['anchor'][0], Y), Full=m_of(xf, Y),
                   HG=m_of(hg, Y) if hg is not None else None)
        out_rows.append((i, ent, 'ok' if hg is not None else 'ok_hg_fail'))
    return k, seedname, out_rows


def work_occlusion(item):
    import torch
    ids, cond, mode = item
    fwd, gt = _W['fwd'], _W['gt']
    fwd.stage.view_weight_mode = ('uniform_valid' if mode == 'uniform'
                                  else fwd.stage._orig_mode)
    out_rows = []
    for i in ids:
        f = _W['conds'][cond] / f'group_{i:05d}.npz'
        if not f.exists():
            out_rows.append((i, None, 'no_cache'))
            continue
        fwd.files[i] = f
        out = fwd.forward([i])
        fwd.files[i] = _W['files0'][i]
        if out is None:
            out_rows.append((i, None, 'placement_fail'))
            continue
        _, xf = full_from_out(out, _W['ref'])
        out_rows.append((i, dict(Full=m_of(xf, gt[i])), 'ok'))
    return cond, mode, out_rows


def aggregate(per, label, ds, n_expected, extra):
    rows = []
    common = None
    for arm, vals in per.items():
        ok = set(i for i, m in vals.items() if m is not None)
        common = ok if common is None else (common & ok)
    for arm, vals in per.items():
        for mk in MKEYS:
            x = [vals[i][mk] for i in sorted(common)]
            rows.append(dict(dataset=ds, analysis=label, arm=arm,
                             metric=mk, value=round(float(np.mean(x)), 4),
                             unit='mm', n_expected=n_expected,
                             n_paired=len(common), **extra))
    return rows, common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True,
                    choices=('stagewise', 'interventions', 'views',
                             'occlusion'))
    ap.add_argument('--dataset', required=True, choices=list(V_OF))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()
    ds, task = a.dataset, a.task
    import s_common as S
    if ds == 'hanco_a3':
        import a3_export_bundles  # noqa: F401
    files, rows_j, prepared = S.cache_files(ds, 'development')
    n = len(rows_j) if not a.limit else a.limit
    ids = list(range(n))
    chunks = [ids[s:s + 4] for s in range(0, n, 4)]
    NODE.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    out_csv = RES / f'j_shared_dev_{task}_{ds}.csv'
    diag = dict(dataset=ds, task=task, stack='U2_SHARED s20261023 gamma=1',
                n_expected=n)

    if task == 'stagewise':
        per = {k: {} for k in ('HG', 'A', 'X_R', 'Full')}
        keep = {}
        with Pool(a.workers, initializer=_init,
                  initargs=(ds, task)) as pool:
            for ci, rws in enumerate(pool.imap_unordered(work_stagewise,
                                                         chunks)):
                for i, ent, st in rws:
                    if ent is None:
                        for k in per:
                            per[k][i] = None
                        continue
                    keep[i] = (ent.pop('_anchor'), ent.pop('_full'))
                    for k in per:
                        per[k][i] = ent.get(k)
                if ci % 100 == 0:
                    print(json.dumps(dict(c=ci, of=len(chunks),
                                          s=round(time.time() - t0))),
                          flush=True)
        rows, common = aggregate(per, 'stagewise', ds, n, dict())
        # cross-check vs the registered u2_chain Dev export
        reg = F2F / f'repo/results/paper/multidataset/{ds}/PRED_DEV_NATIVE.npz'
        if reg.exists() and not a.limit:
            try:
                z = np.load(reg, allow_pickle=False)
                pop = list(z['pop'])
                arms = [str(c) for c in z['arms']]
                diag['registered_export_arms'] = arms
                arm_a = [c for c in arms if c.lower() in ('a', 'anchor')]
                if arm_a:
                    idx = [pi for pi, p in enumerate(pop) if p in keep]
                    da = float(np.mean([np.linalg.norm(
                        z['V_' + arm_a[0]][pi].astype(np.float64)
                        - keep[pop[pi]][0], axis=-1).mean() for pi in
                        idx[:50]]) * 1000)
                    diag['anchor_vs_registered_export_mm'] = round(da, 4)
            except Exception as e:                      # noqa: BLE001
                diag['anchor_crosscheck_error'] = str(e)
        np.savez_compressed(
            NODE / f'stagewise_{ds}_pergroup.npz',
            ids=np.array(sorted(common)),
            **{f'{k}_{mk}': np.array([per[k][i][mk] for i in
                                      sorted(common)])
               for k in per for mk in MKEYS})
    elif task == 'interventions':
        per = {}
        for row in ('learned', 'uniform_w', 'no_du', 'raw_uniform'):
            vals = {}
            with Pool(a.workers, initializer=_init,
                      initargs=(ds, task, row)) as pool:
                for rws in pool.imap_unordered(work_interventions, chunks):
                    for i, ent, st in rws:
                        vals[i] = ent['A'] if ent else None
            per[row] = vals
            print('intervention row done', row,
                  round(time.time() - t0), flush=True)
        rows, common = aggregate(per, 'interventions_at_A', ds, n,
                                 dict(note='A endpoint; rows via '
                                           'registered e1_bank_variants '
                                           'interventions'))
    elif task == 'views':
        V = V_OF[ds]
        cells = [(k, str(s)) for k in range(3, V) for s in VSEEDS] + \
                [(V, 'native')]
        items = [(c, k, s) for (k, s) in cells for c in chunks]
        acc = {(k, s): {arm: {} for arm in ('HG', 'A', 'Full')}
               for (k, s) in cells}
        with Pool(a.workers, initializer=_init,
                  initargs=(ds, task)) as pool:
            for ci, (k, s, rws) in enumerate(
                    pool.imap_unordered(work_views, items)):
                for i, ent, st in rws:
                    for arm in ('HG', 'A', 'Full'):
                        acc[(k, s)][arm][i] = (ent or {}).get(arm)
                if ci % 200 == 0:
                    print(json.dumps(dict(c=ci, of=len(items),
                                          s=round(time.time() - t0))),
                          flush=True)
        rows = []
        pg = {}
        for (k, s), per in acc.items():
            r, common = aggregate(per, 'views', ds, n,
                                  dict(view_k=k, view_seed=s))
            rows += r
            for arm in per:
                pg[f'k{k}_{s}_{arm}'] = np.array(
                    [per[arm][i]['abs_v'] for i in sorted(common)])
            pg[f'k{k}_{s}_ids'] = np.array(sorted(common))
        np.savez_compressed(NODE / f'views_{ds}_pergroup.npz', **pg)
    else:  # occlusion
        items = []
        for cond in ('clean_rerun', 'occhalf'):
            for mode in ('learned', 'uniform'):
                items += [(c, cond, mode) for c in chunks]
        acc = {(cond, mode): {}
               for cond in ('clean_rerun', 'occhalf')
               for mode in ('learned', 'uniform')}
        with Pool(a.workers, initializer=_init,
                  initargs=(ds, task)) as pool:
            for ci, (cond, mode, rws) in enumerate(
                    pool.imap_unordered(work_occlusion, items)):
                for i, ent, st in rws:
                    acc[(cond, mode)][i] = (ent or {}).get('Full')
                if ci % 200 == 0:
                    print(json.dumps(dict(c=ci, of=len(items),
                                          s=round(time.time() - t0))),
                          flush=True)
        per = {f'{m}|{c}': acc[(c, m)] for (c, m) in acc}
        rows, common = aggregate(per, 'occlusion_full', ds, n, dict(
            note='registered clean_rerun/occhalf prior caches, same '
                 'frames, occluded cameras, masks, seeds; Full endpoint'))
        # signed deltas
        for mode in ('learned', 'uniform'):
            for mk in MKEYS:
                cl = np.mean([acc[('clean_rerun', mode)][i][mk]
                              for i in sorted(common)])
                oc = np.mean([acc[('occhalf', mode)][i][mk]
                              for i in sorted(common)])
                rows.append(dict(dataset=ds, analysis='occlusion_full',
                                 arm=f'{mode}|delta_occ_minus_clean',
                                 metric=mk,
                                 value=round(float(oc - cl), 4), unit='mm',
                                 n_expected=n, n_paired=len(common),
                                 note='signed occluded-minus-clean'))
        for mk in MKEYS:
            dl = (np.mean([acc[('occhalf', 'learned')][i][mk]
                           for i in sorted(common)])
                  - np.mean([acc[('clean_rerun', 'learned')][i][mk]
                             for i in sorted(common)]))
            du = (np.mean([acc[('occhalf', 'uniform')][i][mk]
                           for i in sorted(common)])
                  - np.mean([acc[('clean_rerun', 'uniform')][i][mk]
                             for i in sorted(common)]))
            rows.append(dict(dataset=ds, analysis='occlusion_full',
                             arm='degradation_diff_learned_minus_uniform',
                             metric=mk, value=round(float(dl - du), 4),
                             unit='mm', n_expected=n,
                             n_paired=len(common),
                             note='difference in degradation between '
                                  'support modes'))

    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows
                                                 for k in r}))
        w.writeheader()
        w.writerows(rows)
    (NODE / f'{task}_{ds}_diag.json').write_text(json.dumps(diag, indent=1))
    print('SR3_J3_TASK_DONE', task, ds, out_csv, round(time.time() - t0),
          flush=True)


if __name__ == '__main__':
    main()
