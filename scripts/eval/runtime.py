"""Part C: end-to-end runtime on one B200 (supervisor review).

Rows produced per dataset (dexycb = 8 views, ho3d = 5 views; Development
groups, batch = 1 group as deployed, fp32 inputs, 20 warmup + 200 timed
iterations with cuda synchronisation):
  prior_forward            frozen HaMeR forward, all views of one group
                           batched (dedicated loop, crops pre-normalised)
  model_gpu components     cached-prior downstream chain, repeated in this
                           session (comparability with the existing
                           efficiency/timing.csv rows)
  e2e_gpu components       crops in -> mesh out, NO prior-output caches:
                           preprocess_warp (cv2 affine crop of the raw
                           frames, CPU) + preprocess_normalize_h2d +
                           prior_forward + downstream_chain + postprocess_d2h
                           and the summed e2e_total
Hand detection is outside the pipeline (annotation-assisted crop affines
from the prepared population); raw frames are pre-decoded in memory, JPEG
decode is not included.

usage: CUDA_VISIBLE_DEVICES=<g> python runtime.py --dataset dexycb|ho3d
                      [--iters 200] [--warmup 20] [--smoke]
"""
import os
import argparse, csv, json, sys, tarfile, time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
sys.path.insert(0, os.environ.get('ALIGNHAND_SRC', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src')))

import e8_efficiency as E8  # noqa: E402  (reuses the registered harness)

F2F = Path(os.environ.get('ALIGNHAND_DATA', 'data'))
OUT = F2F / 'srwt/followup/results'
HAMER_CFG = {
    'dexycb': F2F / 'caches/queue/v21_s0_dexycb_hamer_s25_v1.config.json',
    'ho3d': F2F / 'caches/queue/v21_s0_ho3d_hamer_s25_v1.config.json',
}
WDS_ROOT = {'ho3d': F2F / 'handoff/data/ho3d_mv',
            'dexycb': F2F / 'handoff/data/dexycb_mv'}


def stats_row(ds, mode, batch, views, comp, xs, caches, note=''):
    a = np.asarray(xs, float)
    return dict(dataset=ds, mode=mode, batch=batch, views=views,
                component=comp, mean_ms=float(a.mean()),
                std_ms=float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                median_ms=float(np.median(a)),
                p95_ms=float(np.percentile(a, 95)), n=len(a),
                per_group_ms=round(float(a.mean()) / batch, 4),
                caches_used=caches, note=note)


def load_raw_images(ds, prep, row, mask):
    imgs = []
    if ds == 'ho3d':
        shard = WDS_ROOT[ds] / row['shard']
        with shard.open('rb') as f:
            for v, member in enumerate(row['image_members']):
                if not mask[v]:
                    imgs.append(np.zeros((480, 640, 3), np.uint8))
                    continue
                f.seek(int(member['offset']))
                payload = f.read(int(member['size']))
                im = cv2.imdecode(np.frombuffer(payload, np.uint8),
                                  cv2.IMREAD_COLOR)
                assert im is not None
                imgs.append(im)
    else:
        # Round-3 Part I: one crop source - the production reader
        # (applies row['image_order'] and the camera binding), instead of
        # raw tar-member indexing.  Byte-identical for the registered
        # populations (i_pairing_diagnostic.json), enforced regardless.
        from dev_reader import dexycb_dev_reader
        reader = dexycb_dev_reader()
        for v, (im, _sha) in enumerate(reader.read(row)):
            imgs.append(im if mask[v]
                        else np.zeros((480, 640, 3), np.uint8))
    return imgs


def u2_build(ds, device):
    """Locked shared U2 chain (Round-3 J1 lock): U2 Stage I + shared R +
    U2_GEOMETRY_STATS + shared Ref, gamma = 1 (Ref output = endpoint).
    Returns (fwd, ref, gamma=None, counts, identity)."""
    import hashlib, torch
    from mvhand.regional.candidates import CandidateEvidenceHead
    from mvhand.regional.refinement import MultiViewRefinement
    import ref23_train as R
    import s_common as S
    U2 = F2F / 'results/u2'
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
    ref = MultiViewRefinement(R.build_operators(),
                              displacement_bound_m=.005,
                              include_correction_magnitude=True)
    refck = torch.load(U2 / 'ref_s20261023/REF_FINAL.pt',
                       map_location='cpu', weights_only=False)
    ref.load_state_dict(refck.get('state_dict', refck))
    ref.eval().requires_grad_(False)
    counts = dict(
        stage1_params=sum(p.numel() for p in fwd.stage.parameters()),
        r_head_params=sum(p.numel() for p in fwd.head.parameters()),
        ref_params=sum(p.numel() for p in ref.parameters()))
    s1sha = hashlib.sha256(b''.join(
        v.cpu().numpy().tobytes() for _, v in
        sorted(ck['trainable_state'].items()))).hexdigest()
    identity = dict(
        stage1='results/u2/stage1_s20261023/CHECKPOINT_UPDATE_16800.pt',
        stage1_trainable_state_sha256=s1sha,
        r='results/u2/r_s20261023/R_FINAL.pt',
        ref='results/u2/ref_s20261023/REF_FINAL.pt',
        stats='results/u2/U2_GEOMETRY_STATS.pt', gamma='1 (Ref endpoint)')
    return fwd, ref, None, counts, identity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=('dexycb', 'ho3d'))
    ap.add_argument('--stack', default='u2',
                    choices=('per_dataset', 'u2'))
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        a.iters, a.warmup = 5, 2
    ds = a.dataset
    import torch
    torch.set_num_threads(4)
    from mvhand.cli.v2_capture_hanco_prior import load_prior
    from mvhand.models.frozen_prior import FrozenOfficialPrior
    from mvhand.models.hamer_adapter import normalize_rgb_crops
    from mvhand.models.interfaces import Form2FrameBatch, HandPriorOutput
    from mvhand.runtime.mano_compat import legacy_mano_imports
    import s_common as S

    dev = a.device
    sync = torch.cuda.synchronize
    if a.stack == 'u2':
        fwd, ref, gamma, counts, identity = u2_build(ds, dev)
    else:
        fwd, ref, gamma, counts = E8.build(ds, dev)
        identity = dict(stack='per_dataset (registered e8 build: gate3 '
                              's25 Ref gamma 0.5 / a3clean)')

    gid = None
    for i in range(50):
        if fwd.forward([i]) is not None:
            gid = i
            break
    assert gid is not None, 'no forward-able group in the first 50'
    run = E8.chain_fn(fwd, ref, gamma, dev)   # moves modules to the GPU

    prep = S.PREPARED[(ds, 'development')]
    rows_j = [json.loads(l) for l in
              (Path(prep) / 'ROWS.jsonl').read_text().splitlines()]
    row = rows_j[gid]
    mask_np = np.asarray(fwd.inputs['runtime_view_mask'][gid], bool)
    views = int(mask_np.sum())
    affine = fwd.inputs['raw_to_crop_s25'][gid]
    raw_images = load_raw_images(ds, prep, row, mask_np)

    config = json.loads(HAMER_CFG[ds].read_text())
    with legacy_mano_imports():
        prior_model = FrozenOfficialPrior(load_prior(config, dev),
                                          name=config['prior_name'])

    t = lambda x: torch.as_tensor(x, device=dev)
    mask_t = t(mask_np[None])
    batch_geo = Form2FrameBatch(
        torch.zeros((1, len(mask_np), 3, 1, 1), device=dev),
        t(fwd.inputs['intrinsics'][[gid]]),
        t(fwd.inputs['camera_to_world'][[gid]]),
        t(np.stack([cv2.invertAffineTransform(x) for x in affine])[None]),
        mask_t, t(affine[None]))

    def warp():
        return [cv2.warpAffine(im[:, :, ::-1], aff, (256, 256),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT)
                .transpose(2, 0, 1).astype(np.float32) / 255
                for im, aff in zip(raw_images, affine)]

    rows = []
    PRIOR_FIELDS = ('joints_2d_normalized', 'visual_features',
                    'local_joints_3d', 'local_vertices_3d')
    with torch.inference_mode():
        # ---- dedicated prior forward loop (crops pre-normalised) ----
        crops_fixed = t(np.stack(warp())[None])
        norm_fixed = normalize_rgb_crops(crops_fixed)
        for _ in range(a.warmup):
            prior_model.capture(norm_fixed, mask_t)
        sync()
        xs = []
        for _ in range(a.iters):
            t0 = time.perf_counter()
            prior_model.capture(norm_fixed, mask_t)
            sync()
            xs.append((time.perf_counter() - t0) * 1000)
        rows.append(stats_row(ds, 'prior_gpu', 1, views, 'prior_forward',
                              xs, 'none',
                              'frozen HaMeR forward, all valid views of one '
                              'group in one batch, fp32'))
        print('prior_forward done', flush=True)

        # ---- cached-path chain components (existing measurement repeated)
        base = E8.prepare_group(fwd, gid, dev)
        for _ in range(a.warmup):
            run(*base)
        timers = {}
        for _ in range(a.iters):
            run(*base, timers)
        for comp, xs in timers.items():
            rows.append(stats_row(ds, 'model_gpu', 1, views, comp, xs,
                                  'prior outputs cached',
                                  'downstream chain only, same session'))
        print('cached chain done', flush=True)

        # ---- end-to-end, no caches ----
        def one_pass(timers=None):
            marks = [time.perf_counter()]

            def mark(gpu=True):
                if gpu:
                    sync()
                marks.append(time.perf_counter())
            rgb = warp()
            mark(gpu=False)
            crops = t(np.stack(rgb)[None])
            normalized = normalize_rgb_crops(crops)
            mark()
            capture = prior_model.capture(normalized, mask_t)
            mark()
            prior = HandPriorOutput(
                **{f: getattr(capture.prior, f).float()
                   for f in PRIOR_FIELDS})
            fmt = capture.feature_maps.half()
            xf = run(prior, batch_geo, fmt)
            mark()
            verts = xf.float().cpu().numpy()
            mark(gpu=False)
            assert np.isfinite(verts).all()
            if timers is not None:
                deltas = np.diff(marks) * 1000.0
                names = ('preprocess_warp', 'preprocess_normalize_h2d',
                         'prior_forward', 'downstream_chain',
                         'postprocess_d2h')
                for name, d in zip(names, deltas):
                    timers.setdefault(name, []).append(float(d))
                timers.setdefault('e2e_total', []).append(
                    float(sum(deltas)))

        for _ in range(a.warmup):
            one_pass()
        timers = {}
        for _ in range(a.iters):
            one_pass(timers)
        for comp, xs in timers.items():
            rows.append(stats_row(
                ds, 'e2e_gpu', 1, views, comp, xs, 'none',
                'crops in -> mesh out, no prior-output caches; crop warp '
                'from pre-decoded raw frames included, JPEG decode and '
                'hand detection excluded'))
        print('e2e done', flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    tag = '_smoke' if a.smoke else ''
    fcsv = OUT / (f'c_timing_{ds}{tag}.csv' if a.stack == 'per_dataset'
                  else f'j4_timing_{ds}_u2{tag}.csv')
    with fcsv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    hp = Path(os.environ.get('ALIGNHAND_DATA', 'data') + '/paperwt/results/efficiency/HAMER_PARAMS.json')
    meta = dict(dataset=ds, stack=a.stack, identity=identity,
                group_index=gid, views=views,
                batch=1, iters=a.iters, warmup=a.warmup,
                device=torch.cuda.get_device_name(0),
                gamma=('None (a3clean Ref endpoint)' if gamma is None
                       else gamma),
                counts=counts,
                hamer_frozen_params=json.loads(hp.read_text())
                ['total_params'] if hp.exists() else None,
                crop_pipeline='raw frame -> 256x256 s25 affine crop -> '
                              'prior-contract normalisation (ImageNet '
                              'mean/std); prior consumes the HaMeR '
                              '256x192 field of view via its adapter',
                precision='fp32 inputs as deployed (R-head bank cast to '
                          'float64 kept; feature maps float16 as in the '
                          'deployed cache reader)')
    (OUT / f'c_timing_{ds}{tag}_meta.json').write_text(
        json.dumps(meta, indent=1))
    print('SR_C_DONE', ds, fcsv, flush=True)


if __name__ == '__main__':
    main()
