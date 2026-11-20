"""Stage 6 E8 efficiency harness (spec D7).

Measures, per dataset at native views, on ONE fixed GPU slot:
  (a) MODEL: the downstream prior->mesh chain (the frozen HaMeR prior forward
      is cached and NOT timed here): Stage I forward_from_prior +
      recover_positioned_mesh + assemble_single_prior_inputs + R head
      (forward_single_prior) + Ref head (MultiViewRefinement) + endpoint
      combine.  Inputs fp32 as deployed (the deployed path casts the R-head
      bank inputs to float64 internally; that cast is kept, not hidden).
      batch 1 and batch 8, warmup 20 + torch.cuda.synchronize, 200 timed
      iterations, per-component wall deltas with syncs between components.
  (b) E2E as deployed: FrozenForward.forward (CPU, 1 torch thread) per group
      including cache np.load I/O + batch assembly, plus the Ref forward,
      over the first 30 forward-able Dev groups.
Also reports trainable/frozen parameter counts (Stage I / R / Ref; the frozen
HaMeR prior via --hamer-params once, from the checkpoint state_dict) and
prior calls per group (= valid native views, from runtime_view_mask).

usage:
  e8_efficiency.py --dataset dexycb|ho3d|hanco_a3 [--device cuda:0]
                   [--iters 200] [--warmup 20] [--e2e-groups 30] [--smoke]
  e8_efficiency.py --hamer-params      # one-off HaMeR frozen param count
Writes $ALIGNHAND_DATA/repo/results/paper/efficiency/{timing.csv,...}.
"""
import argparse, csv, json, os, resource, subprocess, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
sys.path.insert(0, os.environ.get('ALIGNHAND_SRC', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src')))

OUTD = Path(os.environ.get('ALIGNHAND_DATA', 'data') + '/repo/results/paper/efficiency')
GATE3_S25 = (os.environ.get('ALIGNHAND_DATA', 'data') + '/results/ref23/gate3/gate3/'
             'CHECKPOINT_EPOCH_010.pt')
A3_REF = (os.environ.get('ALIGNHAND_DATA', 'data') + '/results/a3clean/seed20261023/'
          'REF_CHECKPOINT_EPOCH_010.pt')
HAMER_CKPT = os.environ.get('ALIGNHAND_DATA', 'data') + '/handoff/models/priors/hamer.ckpt'
R_FIELDS = ('geometry_raw', 'candidate_raw', 'candidate_grid',
            'anchor_raw', 'anchor_grid')


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def gpu_snapshot():
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,'
             'utilization.gpu', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=30).stdout.strip()
        return out.splitlines()
    except Exception as e:                                  # noqa: BLE001
        return [f'nvidia-smi failed: {e}']


def hamer_params():
    import torch
    t0 = time.time()
    ck = torch.load(HAMER_CKPT, map_location='cpu', weights_only=False)
    sd = ck.get('state_dict', ck)
    groups = {}
    total = 0
    for k, v in sd.items():
        if not hasattr(v, 'numel'):
            continue
        total += v.numel()
        groups[k.split('.')[0]] = groups.get(k.split('.')[0], 0) + v.numel()
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / 'HAMER_PARAMS.json').write_text(json.dumps(dict(
        checkpoint=HAMER_CKPT, total_params=int(total),
        by_top_level={k: int(v) for k, v in sorted(groups.items())},
        note='frozen prior; state_dict numel from the shipped checkpoint '
             '(includes any non-inference heads stored in the ckpt)',
        loaded_sec=round(time.time() - t0, 1)), indent=1))
    print('HAMER_PARAMS', total, flush=True)


def build(ds, device):
    """Returns (fwd, ref, gamma, stage/head/ref param counts)."""
    import torch
    if ds == 'hanco_a3':
        import a3_3view_eval  # noqa: F401  (installs hanco_a3 patches)
    import s_common as S
    import ref23_train as R
    from mvhand.regional.refinement import MultiViewRefinement
    fwd = S.FrozenForward(ds, 'development')
    ops = R.build_operators()
    ref = MultiViewRefinement(ops, displacement_bound_m=.005,
                              include_correction_magnitude=True)
    if ds == 'hanco_a3':
        rc = torch.load(A3_REF, map_location='cpu', weights_only=False)
        gamma = None               # a3clean Ref outputs the endpoint itself
    else:
        rc = torch.load(GATE3_S25, map_location='cpu', weights_only=False)
        gamma = 0.5                # GATE4 headline identity
    ref.load_state_dict(rc.get('state_dict', rc))
    ref.eval().requires_grad_(False)
    counts = dict(
        stage1_params=sum(p.numel() for p in fwd.stage.parameters()),
        r_head_params=sum(p.numel() for p in fwd.head.parameters()),
        ref_params=sum(p.numel() for p in ref.parameters()))
    return fwd, ref, gamma, counts


def prepare_group(fwd, i, device):
    """One group's cached prior outputs + calibrated batch, on device, fp32.
    Mirrors s_common.FrozenForward.forward input assembly exactly."""
    import cv2, torch
    from mvhand.models.interfaces import Form2FrameBatch, HandPriorOutput
    names = ('joints_2d_normalized', 'visual_features', 'local_joints_3d',
             'local_vertices_3d')
    with np.load(fwd.files[i], allow_pickle=False) as z:
        vals = {n: z[n][0] for n in names}
        fm = z['feature_maps'][0].astype(np.float16)
    affine = fwd.inputs['raw_to_crop_s25'][[i]]
    inverse = np.stack([[cv2.invertAffineTransform(a) for a in row]
                        for row in affine])
    t = lambda x: torch.as_tensor(x, device=device)
    mask = t(fwd.inputs['runtime_view_mask'][[i]])
    batch = Form2FrameBatch(
        torch.zeros((*mask.shape, 3, 1, 1), device=device),
        t(fwd.inputs['intrinsics'][[i]]),
        t(fwd.inputs['camera_to_world'][[i]]),
        t(inverse), mask, t(affine))
    prior = HandPriorOutput(**{n: t(v[None]).float() for n, v in vals.items()})
    fmt = t(fm[None])
    return prior, batch, fmt


def tile(prior, batch, fmt, b):
    import torch
    from mvhand.models.interfaces import Form2FrameBatch, HandPriorOutput
    if b == 1:
        return prior, batch, fmt
    rep = lambda x: torch.cat([x] * b) if x is not None else None
    prior = HandPriorOutput(**{f: rep(getattr(prior, f)) for f in
                               ('joints_2d_normalized', 'visual_features',
                                'local_joints_3d', 'local_vertices_3d')})
    batch = Form2FrameBatch(*[rep(getattr(batch, f)) for f in
                              ('crops', 'intrinsics', 'camera_to_world',
                               'crop_to_raw', 'runtime_view_mask',
                               'raw_to_crop')])
    return prior, batch, rep(fmt)


def chain_fn(fwd, ref, gamma, device):
    import torch
    from mvhand.geometry.mesh_recovery import recover_positioned_mesh
    from mvhand.regional.single_prior import (assemble_single_prior_inputs,
                                              forward_single_prior)
    basis = fwd.basis.to(device)
    reg = fwd.reg.to(device)
    mean = fwd.mean.to(device) if hasattr(fwd.mean, 'to') else fwd.mean
    scale = fwd.scale.to(device) if hasattr(fwd.scale, 'to') else fwd.scale
    stage = fwd.stage.to(device)
    head = fwd.head.to(device)
    refm = ref.to(device)
    sync = (torch.cuda.synchronize if device.startswith('cuda')
            else (lambda: None))

    def run(prior, batch, fmt, timers=None):
        marks = [time.perf_counter()]

        def mark():
            sync()
            marks.append(time.perf_counter())
        with torch.inference_mode():
            pred = stage.forward_from_prior(prior, batch)
            mark()
            rec = recover_positioned_mesh(
                pred.weighted.points_3d, prior.local_joints_3d,
                prior.local_vertices_3d, pred.relative_view_weights,
                pred.weighted.valid, geometry_blend=.5)
            if not rec.valid.all():
                raise RuntimeError('placement failed for the timing group')
            mark()
            inputs, _ = assemble_single_prior_inputs(
                pred, rec, batch, basis=basis, regressor=reg,
                feature_crop_bounds=(32, 0, 192, 256))
            for name in R_FIELDS:
                inputs[name] = inputs[name].float().double()
            mark()
            result = forward_single_prior(head, basis, inputs, fmt,
                                          mean=mean, scale=scale)
            mark()
            x0 = result.vertices.double()
            correction = torch.linalg.vector_norm(
                pred.corrected_joints_2d_normalized
                - prior.joints_2d_normalized, dim=-1) * 256
            res = refm(vertices=x0, feature_maps=fmt,
                       intrinsics=batch.intrinsics.float(),
                       reference_to_camera=pred.reference_to_camera.float(),
                       raw_to_crop=batch.raw_to_crop.float(),
                       view_mask=inputs['view_valid'].bool(),
                       support_weights=pred.relative_view_weights.float(),
                       scaffold_joints=pred.weighted.points_3d.float(),
                       placed_prior_vertices=(
                           rec.aligned_prior_vertices_3d.float()),
                       correction_magnitude=correction.float())
            xf = (res.vertices.double() if gamma is None
                  else x0 + gamma * (res.vertices.double() - x0))
            mark()
        if timers is not None:
            deltas = np.diff(marks) * 1000.0
            for name, d in zip(('stage1_forward', 'recover_mesh',
                                'assemble_inputs', 'r_head', 'ref_head'),
                               deltas):
                timers.setdefault(name, []).append(float(d))
            timers.setdefault('total_chain', []).append(
                float(sum(deltas)))
        return xf
    return run


def stats(xs):
    a = np.asarray(xs)
    return dict(mean_ms=float(a.mean()), median_ms=float(np.median(a)),
                p95_ms=float(np.percentile(a, 95)), n=len(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=('dexycb', 'ho3d', 'hanco_a3'))
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--e2e-groups', type=int, default=30)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--hamer-params', action='store_true')
    a = ap.parse_args()
    if a.hamer_params:
        return hamer_params()
    import torch
    torch.set_num_threads(1)
    if a.smoke:
        a.iters, a.warmup, a.e2e_groups = 5, 2, 3
    ds = a.dataset
    OUTD.mkdir(parents=True, exist_ok=True)
    snap_before = gpu_snapshot()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')
    t_start = time.time()
    fwd, ref, gamma, counts = build(ds, a.device)
    mask_all = np.asarray(fwd.inputs['runtime_view_mask'], bool)
    prior_calls = mask_all.sum(1)

    # ---- fixed timing group: first Dev group whose CPU forward succeeds
    gid = None
    for i in range(50):
        if fwd.forward([i]) is not None:
            gid = i
            break
    if gid is None:
        raise RuntimeError('no forward-able group in the first 50')

    rows = []
    gpu_ok, gpu_err = True, ''
    mem = {}
    try:
        run = chain_fn(fwd, ref, gamma, a.device)
        base = prepare_group(fwd, gid, a.device)
        for b in (1, 8):
            prior, batch, fmt = tile(*base, b)
            if a.device.startswith('cuda'):
                torch.cuda.reset_peak_memory_stats(a.device)
            for _ in range(a.warmup):
                run(prior, batch, fmt)
            timers = {}
            for _ in range(a.iters):
                run(prior, batch, fmt, timers)
            if a.device.startswith('cuda'):
                mem[f'batch{b}_peak_alloc_mb'] = round(
                    torch.cuda.max_memory_allocated(a.device) / 2**20, 1)
                mem[f'batch{b}_peak_reserved_mb'] = round(
                    torch.cuda.max_memory_reserved(a.device) / 2**20, 1)
            for comp, xs in timers.items():
                rows.append(dict(dataset=ds, mode='model_gpu', batch=b,
                                 component=comp, **stats(xs),
                                 per_group_ms=round(stats(xs)['mean_ms'] / b,
                                                    4)))
    except Exception as e:                                  # noqa: BLE001
        gpu_ok, gpu_err = False, f'{type(e).__name__}: {e}'

    # ---- (b) end-to-end as deployed: CPU FrozenForward + Ref per group
    fwd.stage.to('cpu')            # chain_fn may have moved these to cuda
    fwd.head.to('cpu')
    e2e, e2e_ids = [], []
    i = 0
    import ref23_train as R
    refm = ref.to('cpu')
    while len(e2e_ids) < a.e2e_groups and i < len(fwd.files):
        t0 = time.perf_counter()
        out = fwd.forward([i])            # includes cache np.load + assembly
        if out is None:
            i += 1
            continue
        t_fwd = time.perf_counter()
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
        _ = (res.vertices.double() if gamma is None
             else x0 + gamma * (res.vertices.double() - x0))
        t1 = time.perf_counter()
        e2e.append(dict(total=(t1 - t0) * 1000,
                        frozen_forward=(t_fwd - t0) * 1000,
                        ref=(t1 - t_fwd) * 1000))
        e2e_ids.append(i)
        i += 1
    for comp in ('total', 'frozen_forward', 'ref'):
        xs = [r[comp] for r in e2e]
        rows.append(dict(dataset=ds, mode='e2e_cpu_deployed', batch=1,
                         component=comp, **stats(xs),
                         per_group_ms=round(stats(xs)['mean_ms'], 4)))

    tag = '_smoke' if a.smoke else ''
    fcsv = OUTD / f'timing_{ds}{tag}.csv'
    with fcsv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    hamer = {}
    hp = OUTD / 'HAMER_PARAMS.json'
    if hp.exists():
        hamer = json.loads(hp.read_text())
    status = dict(
        written_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        dataset=ds, smoke=a.smoke, device=a.device,
        cuda_visible_devices=visible,
        gpu_name=(torch.cuda.get_device_name(0)
                  if torch.cuda.is_available() else 'none'),
        timing_group_index=int(gid),
        views_native=int(mask_all.shape[1]),
        prior_calls_per_group=dict(
            mean=float(prior_calls.mean()), min=int(prior_calls.min()),
            max=int(prior_calls.max()),
            note='= valid views in runtime_view_mask; the frozen HaMeR prior '
                 'runs once per valid view'),
        dtype='fp32 inputs as deployed; R-head bank inputs cast to float64 '
              'inside the deployed path (kept); Ref casts to float32 '
              'internally; feature maps fp16',
        warmup=a.warmup, iters=a.iters,
        gpu_model_timing_ok=gpu_ok, gpu_error=gpu_err,
        e2e_groups=e2e_ids, e2e_note='FrozenForward.forward wall time (CPU, '
        '1 torch thread, includes per-group cache np.load + assembly) + Ref '
        'forward; the frozen HaMeR prior itself is cached and not re-run',
        params=dict(counts, frozen_hamer_prior=hamer.get(
            'total_params', 'run --hamer-params'),
            trainable_breakdown='stage1 (CrossViewEvidence+heads) / R head '
            '(CandidateEvidenceHead) / Ref (MultiViewRefinement); all frozen '
            'at eval, trainable during their training phases'),
        ref_identity=('a3clean seed20261023 Ref endpoint' if gamma is None
                      else 'gate3 s25 Ref at gamma 0.5 (GATE4 identity)'),
        weights_note='timing is weight-agnostic: hanco_a3 uses the shipped '
                     'hanco R head weights (architecture identical to the '
                     'a3clean heads used for accuracy numbers)',
        peak_gpu_memory=mem,
        peak_cpu_rss_mb=round(rss_mb(), 1),
        nvidia_smi_before=snap_before, nvidia_smi_after=gpu_snapshot(),
        contention=('OFFICIAL_STATIC_SLOT' if gpu_ok else 'NO_GPU'),
        total_sec=round(time.time() - t_start, 1))
    (OUTD / f'STATUS_{ds}{tag}.json').write_text(json.dumps(status, indent=1))
    print('E8_EFF_DONE', ds, json.dumps(
        {r['component']: round(r['mean_ms'], 2) for r in rows
         if r['mode'] == 'model_gpu' and r['batch'] == 1} |
        {'e2e_total': round(rows[-3]['mean_ms'], 1)}), flush=True)


if __name__ == '__main__':
    main()
