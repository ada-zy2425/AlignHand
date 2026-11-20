"""Stage 6 E8 occlusion robustness (spec D7) - machinery + smoke.

Controlled, PRE-REGISTERED black-square occlusion applied at the consistent
image entry: the registered s25 256x256 RGB crop that the frozen HaMeR prior
consumes.  The frozen prior is RE-RUN on the occluded crops (no clean feature
cache is ever reused under occlusion - F3), then the unchanged downstream
chain (Stage I + placement + R + Ref) produces A / X_R / Full; HG comes from
the same occluded prior outputs.

Pre-registered grid (independent of any method error):
  selection seed 20261028: groups = default_rng(SEED).permutation(N)[:G]
  box: side 96 px, top-left x0 ~ U{32..128}, y0 ~ U{0..160} (inside the
       (32,0,192,256) feature-visible crop area), drawn per (group, view)
       from default_rng([SEED, pop_index, view_slot])
  occluded views per condition, drawn from the group's valid views with
       default_rng([SEED, pop_index, 7]):
       occ1 = 1 view; occhalf = floor(V_valid/2) views (superset of occ1's
       draw order); clean_rerun = none (prior re-run sanity gate).
Fixed-checkpoint intervention: the SAME checkpoints are also evaluated with
stage.view_weight_mode='uniform_valid' (weights-only switch, no retraining).

usage: make_occlusion_caches.py --dataset dexycb [--groups 8] [--device cuda:0]
Writes caches to $ALIGNHAND_DATA/results/e8_occlusion/<ds>/ and results to
$ALIGNHAND_DATA/repo/results/paper/robustness/.
"""
import os
import argparse, csv, json, sys, time
from dataclasses import fields as dc_fields
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
sys.path.insert(0, os.environ.get('ALIGNHAND_SRC', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src')))

SEED = 20261028
BOX = 96
CONFIGS = {'dexycb': os.environ.get('ALIGNHAND_DATA', 'data') + '/caches/queue/'
                     'v21_s0_dexycb_hamer_s25_v1.config.json'}
GATE3_S25 = (os.environ.get('ALIGNHAND_DATA', 'data') + '/results/ref23/gate3/gate3/'
             'CHECKPOINT_EPOCH_010.pt')
WORK = Path(os.environ.get('ALIGNHAND_DATA', 'data') + '/results/e8_occlusion')
OUTD = Path(os.environ.get('ALIGNHAND_DATA', 'data') + '/repo/results/paper/robustness')
REGION_NAMES = ['region0', 'region1', 'region2', 'region3', 'region4',
                'region5']


def make_spec(ds, n_pop, mask_all, groups_n):
    rng = np.random.default_rng(SEED)
    groups = [int(g) for g in rng.permutation(n_pop)[:groups_n]]
    spec = dict(seed=SEED, dataset=ds, box_side=BOX,
                box_x0_range=[32, 128], box_y0_range=[0, 160],
                entry='registered s25 256x256 RGB crop consumed by the '
                      'frozen HaMeR prior (occluded BEFORE the prior; prior '
                      'and all downstream re-run)',
                conditions=['clean_rerun', 'occ1', 'occhalf'],
                groups=groups, per_group={})
    for g in groups:
        valid = np.flatnonzero(mask_all[g]).tolist()
        order = list(np.random.default_rng([SEED, g, 7])
                     .permutation(valid))
        boxes = {}
        for v in valid:
            r = np.random.default_rng([SEED, g, int(v)])
            boxes[int(v)] = [int(r.integers(32, 129)),
                             int(r.integers(0, 161))]
        spec['per_group'][str(g)] = dict(
            valid_views=valid,
            occ1=[int(order[0])],
            occhalf=[int(x) for x in order[:max(1, len(valid) // 2)]],
            box_x0y0_per_view=boxes)
    return spec


def capture_group(prior, reader, inputs, rows, g, occ_views, boxes, outfile,
                  device, montage=None):
    """Replicates mvhand.cli.v21_s0_baseline crop->prior capture, with the
    pre-registered black squares applied to the crops of occ_views."""
    import cv2, torch
    from mvhand.data.v2_access import validate_input_field_names
    from mvhand.models.hamer_adapter import normalize_rgb_crops
    from mvhand.models.interfaces import Form2FrameBatch
    images = reader.read(rows[g])
    affine = inputs['raw_to_crop_s25'][g]
    rgb = []
    for v, ((im, _), a) in enumerate(zip(images, affine)):
        warped = cv2.warpAffine(im[:, :, ::-1], a, (256, 256),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
        if v in occ_views:
            x0, y0 = boxes[str(v)] if str(v) in boxes else boxes[v]
            warped = warped.copy()
            warped[y0:y0 + BOX, x0:x0 + BOX] = 0
        rgb.append(warped)
    if montage is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(rgb), figsize=(2.2 * len(rgb), 2.6))
        axes = np.atleast_1d(axes)
        for v, (ax, w) in enumerate(zip(axes, rgb)):
            ax.imshow(w)
            ax.set_title(f'v{v}' + (' OCC' if v in occ_views else ''),
                         fontsize=8)
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(montage, dpi=90)
        plt.close(fig)
    stack = np.stack([w.transpose(2, 0, 1).astype(np.float32) / 255
                      for w in rgb])
    tensor = lambda x: torch.as_tensor(x, device=device)[None]
    batch = Form2FrameBatch(
        tensor(stack), tensor(inputs['intrinsics'][g]),
        tensor(inputs['camera_to_world'][g]),
        tensor(np.stack([cv2.invertAffineTransform(a) for a in affine])),
        tensor(inputs['runtime_view_mask'][g]), tensor(affine))
    with torch.inference_mode():
        cap = prior.capture(normalize_rgb_crops(batch.crops),
                            batch.runtime_view_mask)
    p = cap.prior
    values = {f.name: getattr(p, f.name) for f in dc_fields(p)}
    values.update(feature_maps=cap.feature_maps,
                  camera_translation=cap.camera_translation,
                  focal_length=cap.focal_length,
                  projection_error_crop_px=cap.projection_error_crop_px)
    validate_input_field_names(values)
    np.savez(outfile, **{k: v.cpu().numpy() for k, v in values.items()})


def downstream(fwd, refm, gamma, g, q21, basis6):
    """FrozenForward + Ref on whatever fwd.files[g] points to."""
    import torch
    import ref23_train as R
    import e4_transfer_eval as E4
    import metrics2 as M2
    out = fwd.forward([g])
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
        res = refm(**R.ref_inputs(extras, 'cpu'))
    xf = (x0 + gamma * (res.vertices.double() - x0))[0].numpy()
    hg = E4._hg(fwd, g)
    w = np.asarray(out['support_weights'][0])          # (V,21)
    return dict(A=np.asarray(out['anchor'][0], np.float64),
                X_R=np.asarray(x0[0].numpy()), Full=np.asarray(xf),
                HG=hg, weights=w.mean(1),
                view_valid=np.asarray(out['view_valid'][0], bool))


def region_errors(X, Y, basis6):
    ev = np.linalg.norm(X - Y, axis=-1) * 1000.0       # (778,) mm
    out = {}
    for r in range(basis6.shape[1]):
        m = basis6[:, r] > 0
        out[REGION_NAMES[r]] = float((ev * basis6[:, r])[m].sum()
                                     / basis6[:, r][m].sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=list(CONFIGS), default='dexycb')
    ap.add_argument('--groups', type=int, default=8)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    ds = a.dataset
    import torch
    torch.set_num_threads(4)
    import s_common as S
    import ref23_train as R
    import metrics2 as M2
    from mvhand.cli.v2_capture_hanco_prior import load_prior
    from mvhand.data.v21_prepared_rgb import PreparedDevelopmentRGB
    from mvhand.models.frozen_prior import FrozenOfficialPrior
    from mvhand.regional.refinement import MultiViewRefinement
    from mvhand.runtime.mano_compat import legacy_mano_imports

    t_all = time.time()
    config = json.loads(Path(CONFIGS[ds]).read_text())
    fwd = S.FrozenForward(ds, 'development')
    orig_files = list(fwd.files)
    mask_all = np.asarray(fwd.inputs['runtime_view_mask'], bool)
    spec = make_spec(ds, len(fwd.rows), mask_all, a.groups)
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / 'OCCLUSION_SPEC.json').write_text(json.dumps(spec, indent=1))

    gt_all = np.load(fwd.prepared / 'PRIVATE_SCORE_TARGETS.npz',
                     allow_pickle=False)['vertices_reference_m']
    q21, _ = M2.load_q21()
    basis6 = np.load(os.environ.get('ALIGNHAND_DATA', 'data') + '/handoff/models/assets/'
                     'region_mask_b_778x6.npy')
    ops = R.build_operators()
    refm = MultiViewRefinement(ops, displacement_bound_m=.005,
                               include_correction_magnitude=True)
    have_internal = Path(GATE3_S25).exists()
    if have_internal:
        rc = torch.load(GATE3_S25, map_location='cpu', weights_only=False)
        refm.load_state_dict(rc.get('state_dict', rc))
    refm.eval().requires_grad_(False)
    gamma = 0.5

    print(json.dumps({'stage': 'loading_prior'}), flush=True)
    with legacy_mano_imports():
        prior = FrozenOfficialPrior(load_prior(config, a.device),
                                    name=config['prior_name'])

    conds = ('clean_rerun', 'occ1', 'occhalf')
    cachedirs = {}
    montdir = OUTD / f'{ds}_crops'
    montdir.mkdir(exist_ok=True)
    with PreparedDevelopmentRGB(config) as reader:
        for cond in conds:
            d = WORK / ds / cond
            d.mkdir(parents=True, exist_ok=True)
            cachedirs[cond] = d
            for g in spec['groups']:
                pg = spec['per_group'][str(g)]
                occ = [] if cond == 'clean_rerun' else pg[cond]
                f = d / f'group_{g:05d}.npz'
                if not f.exists():
                    capture_group(
                        prior, reader, fwd.inputs, fwd.rows, g, occ,
                        pg['box_x0y0_per_view'], f, a.device,
                        montage=(montdir / f'{cond}_pop{g}.png'
                                 if cond != 'clean_rerun' else None))
            print(json.dumps({'captured': cond}), flush=True)

    if not have_internal:
        # The occluded prior caches are all the shared model needs; score them
        # with `dev_analysis.py --task occlusion`. The per-dataset scoring below
        # uses an internal stack that is not released.
        print(json.dumps({'caches': str(WORK / ds), 'scoring': 'dev_analysis.py --task occlusion'}))
        return

    results, csv_rows = {}, []
    x0_sanity = []
    pred = np.load(Path(os.environ.get('ALIGNHAND_DATA', 'data') + '/repo/results/paper/e0') / ds
                   / 'PRED_DEV_NATIVE.npz')
    v_x0 = pred['V_x0']
    for mode in ('full', 'uniform_valid'):
        fwd.stage.view_weight_mode = mode
        for cond in ('cached_clean',) + conds:
            if mode == 'uniform_valid' and cond in ('cached_clean', 'occ1'):
                continue          # intervention runs on clean_rerun+occhalf
            key = f'{cond}@{mode}'
            results[key] = {}
            for g in spec['groups']:
                fwd.files = list(orig_files)
                if cond != 'cached_clean':
                    fwd.files[g] = cachedirs[cond] / f'group_{g:05d}.npz'
                r = downstream(fwd, refm, gamma, g, q21, basis6)
                pg = spec['per_group'][str(g)]
                ent = dict(placement_ok=r is not None)
                if r is not None:
                    Y = gt_all[g].astype(np.float64)
                    occ = ([] if cond in ('cached_clean', 'clean_rerun')
                           else pg[cond])
                    nocc = [v for v in pg['valid_views'] if v not in occ]
                    for arm in ('HG', 'A', 'X_R', 'Full'):
                        if r[arm] is None:
                            ent[arm] = None
                            continue
                        met = M2.group_metrics(r[arm], Y, q21, light=True)
                        ent[arm] = {k: round(met[k], 4) for k in
                                    ('abs_v', 'abs_j21', 'pa_v', 'pa_j')}
                        for mname, mv in ent[arm].items():
                            csv_rows.append(dict(
                                dataset=ds, condition=cond, weight_mode=mode,
                                pop=g, arm=arm, metric=mname, value=mv,
                                unit='mm'))
                    ent['region_err_Full_mm'] = region_errors(r['Full'], Y,
                                                              basis6)
                    ent['mean_weight_occluded'] = (
                        float(r['weights'][occ].mean()) if occ else None)
                    ent['mean_weight_unoccluded'] = (
                        float(r['weights'][nocc].mean()) if nocc else None)
                    ent['weights_per_view'] = [round(float(x), 5)
                                               for x in r['weights']]
                    if cond == 'clean_rerun' and mode == 'full':
                        x0_sanity.append(float(np.abs(
                            r['X_R'] - v_x0[g].astype(np.float64)).max()))
                results[key][str(g)] = ent
            print(json.dumps({'scored': key}), flush=True)
    fwd.stage.view_weight_mode = 'full'

    with (OUTD / 'SMOKE_runs.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)
    summary = {}
    for key, ent in results.items():
        ok = [e for e in ent.values() if e['placement_ok']]
        summary[key] = dict(
            coverage=f'{len(ok)}/{len(ent)}',
            mean_abs_v={arm: (round(float(np.mean(
                [e[arm]['abs_v'] for e in ok if e.get(arm)])), 3)
                if any(e.get(arm) for e in ok) else None)
                for arm in ('HG', 'A', 'X_R', 'Full')},
            mean_weight_occluded=(round(float(np.mean(
                [e['mean_weight_occluded'] for e in ok
                 if e['mean_weight_occluded'] is not None])), 5)
                if any(e['mean_weight_occluded'] is not None
                       for e in ok) else None),
            mean_weight_unoccluded=round(float(np.mean(
                [e['mean_weight_unoccluded'] for e in ok
                 if e['mean_weight_unoccluded'] is not None])), 5))
    (OUTD / 'SMOKE_RESULTS.json').write_text(json.dumps(
        dict(summary=summary, per_group=results), indent=1))
    (OUTD / 'STATUS.json').write_text(json.dumps(dict(
        written_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        dataset=ds, status='SMOKE_8_GROUPS',
        machinery='prior re-run on occluded registered crops + unchanged '
                  'downstream; fixed-checkpoint uniform_valid intervention '
                  '(weights-only, no retraining)',
        groups=spec['groups'],
        clean_rerun_x0_vs_cached_max_abs_m=(
            dict(max=float(np.max(x0_sanity)), per_group=x0_sanity)
            if x0_sanity else 'NA'),
        f3_note='no clean feature cache is reused under occlusion: every '
                'condition (incl. clean_rerun) re-runs the frozen prior on '
                'its own crops',
        gt_note='RESULTS compare to Dev GT locally; crop montages contain '
                'DexYCB RGB only (no GT), kept local anyway',
        full_run_command=(
            'cd $ALIGNHAND_DATA && source venv/bin/activate && '
            'export PYTHONPATH=$ALIGNHAND_DATA/repo/src:$ALIGNHAND_DATA/'
            'p1code:$ALIGNHAND_DATA/stage6 && CUDA_VISIBLE_DEVICES=<free> '
            'python $ALIGNHAND_DATA/stage6/make_occlusion_caches.py --dataset dexycb '
            '--groups 1412'),
        sec=round(time.time() - t_all, 1)), indent=1))
    print('E8_OCC_DONE', ds, json.dumps(
        {k: v['mean_abs_v'] for k, v in summary.items()}), flush=True)


if __name__ == '__main__':
    main()
