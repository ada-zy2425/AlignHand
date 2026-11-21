"""Train regional fusion (R) and pixel-aligned refinement (Ref) on top of Stage I.

Steps (--step, default all), each resumable:
  export     run the trained Stage I and save per-group inputs for R and Ref
  stats      geometry normalisation statistics over the pooled training splits
  train-r    train regional fusion R
  train-ref  train pixel-aligned refinement Ref
  eval       evaluate the stages on the development splits

usage: train_fusion_refinement.py [--step STEP] [--seed S] [--workers N]
"""
import os
import argparse, json, sys, time
import hashlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))

U2ROOT = Path(os.environ.get('ALIGNHAND_DATA', 'data') + '/results/u2')
SRC_BUNDLES = Path(os.environ.get('ALIGNHAND_DATA', 'data') + '/results/s1/bundles')
DSS = ('hanco_a3', 'dexycb', 'ho3d')   # overridden by --datasets
FIT_SRC = {'hanco_a3': 'fit8192', 'dexycb': 'fit8192', 'ho3d': 'fit8192'}
R_UPDATES, R_BATCH = 3194, 64
REF_UPDATES, REF_MICRO, REF_EFF = 3840, 8, 64


STAGE1_BUDGET = {'hanco_a3': 8000, 'dexycb': 8000, 'ho3d': 800}
MAP_S1 = {'hanco_a3': 'hanco', 'dexycb': 'dexycb', 'ho3d': 'ho3d'}


def subset_tag(dss):
    return '' if len(dss) == 3 else '_' + '-'.join(sorted(MAP_S1[d]
                                                          for d in dss))


def stage_ckpt(seed, dss):
    total = sum(STAGE1_BUDGET[d] for d in dss)
    return (U2ROOT / f'stage1_s{seed}{subset_tag(dss)}'
            / f'CHECKPOINT_UPDATE_{total:05d}.pt')


ACTIVE = list(DSS)


def patched_export(ds, split, indices, out_dir, seed, workers):
    """export_bundle with the U2 shared Stage I swapped into every worker."""
    import pipeline
    if ds == 'hanco_a3':
        import hanco_bundles  # noqa: F401 (prepared/cache patches)
    orig_init = pipeline._init
    ckpt_path = str(stage_ckpt(seed, ACTIVE))

    def init_u2(ds_, split_):
        orig_init(ds_, split_)
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        fwd = pipeline._CTX['fwd']
        fwd.stage.load_state_dict(ck['trainable_state'], strict=True)
        fwd.stage.eval().requires_grad_(False)

    pipeline._init = init_u2
    try:
        pipeline.export_bundle(ds, split, indices, out_dir, workers=workers)
    finally:
        pipeline._init = orig_init


def step_export(seed, workers):
    import pipeline
    todo = {ds: ['fit', 'dev'] for ds in ACTIVE}
    for ds in EVAL_DSS:            # held-out LOO targets: dev only (eval)
        todo.setdefault(ds, []).append('dev')
    for ds, tags in todo.items():
        if ds == 'hanco_a3':
            import hanco_bundles  # noqa: F401
        for tag, (src_split, src_name) in {
                t: (('fit', FIT_SRC[ds]) if t == 'fit' else
                    ('development', 'devfull'))
                for t in dict.fromkeys(tags)}.items():
            out = U2ROOT / ('bundles' + subset_tag(ACTIVE)) / ds / tag
            if (out / 'meta.json').exists():
                print('EXPORTED_ALREADY', ds, tag, flush=True)
                continue
            meta = json.loads((SRC_BUNDLES / ds / src_name / 'meta.json')
                              .read_text())
            idx = [int(i) for i in meta['indices']]
            print('exporting', ds, tag, len(idx), flush=True)
            patched_export(ds, src_split, idx, out, seed, workers)
            m = json.loads((out / 'meta.json').read_text())
            m['u2_stage1'] = str(stage_ckpt(seed, ACTIVE))
            m['x0_note'] = ('bundle x0 = shipped R on U2 banks; unused '
                            'downstream - shared R recomputes live')
            (out / 'meta.json').write_text(json.dumps(m, indent=1))
    print('U2_EXPORT_DONE', flush=True)


def union_stats():
    from pipeline import Bundle
    f = U2ROOT / f'U2_GEOMETRY_STATS{subset_tag(ACTIVE)}.pt'
    if f.exists():
        s = torch.load(f, map_location='cpu', weights_only=False)
        return s['mean'], s['scale']
    total = squares = count = None
    for ds in ACTIVE:
        fit = Bundle(U2ROOT / ('bundles' + subset_tag(ACTIVE)) / ds / 'fit')
        g, v = fit.arrays['geometry_raw'], fit.arrays['geometry_valid']
        rows = fit.usable
        if total is None:
            dim = g.shape[-1]
            total = np.zeros(dim)
            squares = np.zeros(dim)
            count = np.zeros(dim, np.int64)
        for start in range(0, len(rows), 256):
            sel = rows[start:start + 256]
            x = np.asarray(g[sel], np.float64)
            m = np.asarray(v[sel], bool)
            if m.shape != x.shape:
                m = np.broadcast_to(
                    m[..., None] if m.ndim == x.ndim - 1 else m, x.shape)
            if not np.isfinite(x[m]).all():
                raise ValueError(f'nonfinite valid geometry in {ds}')
            x = np.where(m, x, 0)
            axes = tuple(range(x.ndim - 1))
            total += x.sum(axis=axes)
            squares += (x * x).sum(axis=axes)
            count += m.sum(axis=axes)
    mean = total / count
    scale = np.sqrt(np.maximum(0, squares / count - mean ** 2)).clip(1e-8)
    s = dict(schema='u2-union-fit-stats.v1', split='fit',
             datasets=list(ACTIVE), mean=torch.from_numpy(mean),
             scale=torch.from_numpy(scale), counts=torch.from_numpy(count))
    tmp = f.with_suffix('.tmp')
    torch.save(s, tmp)
    tmp.rename(f)
    print('U2_STATS_DONE', flush=True)
    return s['mean'], s['scale']


class DsSampler:
    """ds-equal -> sequence-equal -> row; n rows per draw from ONE ds."""

    def __init__(self, bundles, seed):
        self.rng = np.random.default_rng(seed)
        self.ds = sorted(bundles)
        self.seqmap = {}
        for ds, b in bundles.items():
            rowmeta = [json.loads(l) for l in
                       (Path(b.meta['prepared']) / 'ROWS.jsonl')
                       .read_text().splitlines()]
            idx = b.meta['indices']
            m = {}
            for pos in b.usable:
                s = str(rowmeta[idx[pos]]['sequence_id'])
                m.setdefault(s, []).append(int(pos))
            self.seqmap[ds] = {k: np.array(v) for k, v in m.items()}

    def next(self, n):
        ds = self.ds[int(self.rng.integers(len(self.ds)))]
        seqs = sorted(self.seqmap[ds])
        out = []
        for _ in range(n):
            s = seqs[int(self.rng.integers(len(seqs)))]
            rows = self.seqmap[ds][s]
            out.append(int(rows[int(self.rng.integers(len(rows)))]))
        return ds, out


def step_train_r(seed, device):
    from pipeline import Bundle
    from fusion_refinement_training import assets, r_forward
    from mvhand.fusion.candidates import CandidateEvidenceHead
    from mvhand.fusion.training import errors
    out = U2ROOT / f'r_s{seed}{subset_tag(ACTIVE)}'
    if (out / 'R_FINAL.pt').exists():
        print('R_ALREADY', flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    bundles = {ds: Bundle(U2ROOT / ('bundles' + subset_tag(ACTIVE)) / ds
                          / 'fit') for ds in ACTIVE}
    gts = {ds: torch.as_tensor(b.targets(), device=device)
           for ds, b in bundles.items()}
    mean, scale = union_stats()
    basis, reg = assets(device)
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31))
    head = CandidateEvidenceHead().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
    sampler = DsSampler(bundles, seed)
    t0 = time.time()
    presented = {ds: 0 for ds in ACTIVE}
    r_updates = round(10 * sum(len(b.usable) for b in bundles.values()) / 64)
    for update in range(1, r_updates + 1):
        ds, sel = sampler.next(R_BATCH)
        presented[ds] += len(sel)
        head.train()
        opt.zero_grad(set_to_none=True)
        res, _, _, _ = r_forward(head, basis, bundles[ds], sel, device,
                                 mean, scale)
        v, j = errors(res.vertices, gts[ds][sel], reg)
        loss = (v + .5 * j).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite U2-R loss')
        loss.backward()
        opt.step()
        if update == 1 or update % 100 == 0:
            print(json.dumps(dict(phase='u2r', update=update,
                                  loss=float(loss),
                                  sec=round(time.time() - t0, 1))),
                  flush=True)
        if update % 800 == 0 or update == r_updates:
            torch.save(dict(state_dict=head.state_dict(), update=update,
                            seed=seed, presented=dict(presented)),
                       out / (f'R_CHECKPOINT_U{update:05d}.pt'))
    torch.save(dict(state_dict=head.state_dict(), update=r_updates,
                    seed=seed, presented=dict(presented),
                    recipe='clean-retrain R, union ds-equal, 3194 updates'),
               out / 'R_FINAL.pt')
    print('U2_R_DONE', json.dumps(presented), flush=True)


def _load_shared_r(seed, device):
    from mvhand.fusion.candidates import CandidateEvidenceHead
    ck = torch.load(U2ROOT / f'r_s{seed}{subset_tag(ACTIVE)}' / 'R_FINAL.pt',
                    map_location='cpu', weights_only=False)
    head = CandidateEvidenceHead()
    head.load_state_dict(ck['state_dict'], strict=True)
    return head.to(device).eval().requires_grad_(False)


def step_train_ref(seed, device):
    from pipeline import Bundle
    from pathlib import Path as P
    from fusion_refinement_training import assets, clean_extras
    from refinement import (build_operators, ref_inputs, load_supervision,
                             supervision_batch)
    from mvhand.refinement.model import MultiViewRefinement
    from mvhand.prior.mano_compat import legacy_mano_imports
    from mvhand.training.validation import load_mano_base_regressor
    from mvhand.refinement.supervision import refinement_loss
    out = U2ROOT / f'ref_s{seed}{subset_tag(ACTIVE)}'
    if (out / 'REF_FINAL.pt').exists():
        print('REF_ALREADY', flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    bundles = {ds: Bundle(U2ROOT / ('bundles' + subset_tag(ACTIVE)) / ds
                          / 'fit') for ds in ACTIVE}
    sup = {ds: load_supervision(b) for ds, b in bundles.items()}
    mean, scale = union_stats()
    basis, _ = assets(device)
    head = _load_shared_r(seed, device)
    torch.manual_seed(seed + 1)
    np.random.seed((seed + 1) % (2 ** 31))
    model = MultiViewRefinement(build_operators(), displacement_bound_m=.005,
                                include_correction_magnitude=True).to(device)
    with legacy_mano_imports():
        reg = load_mano_base_regressor(
            P(os.environ.get('ALIGNHAND_DATA', 'data') + '/MANO_RIGHT.pkl')).to(device).float()
    model.eval()
    with torch.no_grad():
        b0 = bundles[sorted(bundles)[0]]
        ex = clean_extras(head, basis, b0, list(b0.usable[:2]), device,
                          mean, scale)
        diff = float((model(**ref_inputs(ex, device)).vertices
                      - ex['x0'].double()).abs().max())
    if diff != 0:
        raise ValueError(f'U2 Ref zero init broken: {diff}')
    print(json.dumps(dict(gate1_zero_identity=diff)), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sampler = DsSampler(bundles, seed + 1)
    t0 = time.time()
    presented = {ds: 0 for ds in ACTIVE}
    ref_updates = 1280 * len(ACTIVE)
    for update in range(1, ref_updates + 1):
        ds, sel = sampler.next(REF_EFF)
        presented[ds] += len(sel)
        gt, j2d = sup[ds]
        model.train()
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for start in range(0, len(sel), REF_MICRO):
            part = sel[start:start + REF_MICRO]
            extras = clean_extras(head, basis, bundles[ds], part, device,
                                  mean, scale)
            result = model(**ref_inputs(extras, device))
            targets = supervision_batch(gt, j2d, part,
                                        extras['view_valid'].shape[1],
                                        device)
            losses, _ = refinement_loss(
                result, targets, regressor=reg,
                projection_matrices=extras['projection'].double(),
                raw_to_crop=extras['raw_to_crop'].double(),
                supervision_mask=targets['mask'],
                view_mask=extras['view_valid'], laplacian=model.laplacian)
            loss = losses.mean() * len(part) / len(sel)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite U2 Ref loss')
            loss.backward()
            total += float(loss.detach())
        opt.step()
        if update == 1 or update % 100 == 0:
            print(json.dumps(dict(phase='u2ref', update=update, loss=total,
                                  sec=round(time.time() - t0, 1))),
                  flush=True)
        if update % 960 == 0 or update == ref_updates:
            torch.save(dict(state_dict=model.state_dict(), update=update,
                            seed=seed, presented=dict(presented)),
                       out / f'REF_CHECKPOINT_U{update:05d}.pt')
    torch.save(dict(state_dict=model.state_dict(), update=ref_updates,
                    seed=seed, presented=dict(presented),
                    recipe='gate3 constants, union ds-equal, x0 live from '
                           'shared R'),
               out / 'REF_FINAL.pt')
    print('U2_REF_DONE', json.dumps(presented), flush=True)


def step_eval(seed, device):
    from pipeline import Bundle
    from fusion_refinement_training import assets, clean_extras
    from refinement import ref_inputs
    from mvhand.refinement.model import MultiViewRefinement
    from refinement import build_operators
    mean, scale = union_stats()
    basis, _ = assets(device)
    head = _load_shared_r(seed, device)
    model = MultiViewRefinement(build_operators(), displacement_bound_m=.005,
                                include_correction_magnitude=True).to(device)
    ck = torch.load(U2ROOT / f'ref_s{seed}{subset_tag(ACTIVE)}'
                    / 'REF_FINAL.pt',
                    map_location='cpu', weights_only=False)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    import hashlib
    sd_sha = hashlib.sha256(b''.join(
        v.cpu().numpy().tobytes() for _, v in
        sorted(model.state_dict().items()))).hexdigest()[:16]
    for ds in EVAL_DSS:
        b = Bundle(U2ROOT / ('bundles' + subset_tag(ACTIVE)) / ds / 'dev')
        rows = list(b.usable)
        acc = {k: [] for k in ('anchor', 'u2R', 'u2Full')}
        with torch.no_grad():
            for start in range(0, len(rows), 32):
                sel = rows[start:start + 32]
                extras = clean_extras(head, basis, b, sel, device, mean,
                                      scale)
                acc['anchor'].append(
                    np.asarray(b.arrays['anchor'][sel], np.float32))
                acc['u2R'].append(
                    extras['x0'].float().cpu().numpy())
                res = model(**ref_inputs(extras, device))
                acc['u2Full'].append(
                    res.vertices.float().cpu().numpy())
        rowmeta = [json.loads(l) for l in
                   (Path(b.meta['prepared']) / 'ROWS.jsonl')
                   .read_text().splitlines()]
        idx = b.meta['indices']
        keys = [f"{rowmeta[idx[i]]['sequence_id'].replace('/', '|')}/"
                f"{rowmeta[idx[i]]['key']}" for i in rows]
        outd = Path(OUT_TMPL.format(ds=ds))
        outd.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            outd / 'PRED_DEV_NATIVE.npz',
            pop=np.array([idx[i] for i in rows]), keys=np.array(keys),
            arms=np.array(list(acc)), unit=np.array('metres_f32'),
            **{f'V_{k}': np.concatenate(v).astype(np.float32)
               for k, v in acc.items()})
        (outd / 'EXPORT_REPORT.json').write_text(json.dumps(dict(
            node='U2 three-dataset eval endpoints (one shared state_dict)',
            dataset=ds, n=len(rows), ref_state_sha16=sd_sha,
            arms=dict(anchor='U2 shared Stage I fused',
                      u2R='shared R (union stats)',
                      u2Full='shared Ref endpoint')), indent=1))
        print('U2_EVAL_EXPORTED', ds, len(rows), flush=True)
    print('U2_EVAL_DONE', sd_sha, flush=True)


EVAL_DSS = list(DSS)
OUT_TMPL = os.environ.get('ALIGNHAND_DATA', 'data') + '/repo/results/paper/multidataset/{ds}'


def main():
    global ACTIVE, EVAL_DSS, OUT_TMPL
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='hanco_a3,dexycb,ho3d')
    ap.add_argument('--eval-datasets', default='')
    ap.add_argument('--out-tmpl', default='')
    ap.add_argument('--step', default='all',
                    choices=('all', 'export', 'stats', 'train-r',
                             'train-ref', 'eval'))
    ap.add_argument('--seed', type=int, default=20261023)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    ACTIVE = a.datasets.split(',')
    EVAL_DSS = (a.eval_datasets.split(',') if a.eval_datasets else
                list(ACTIVE))
    if a.out_tmpl:
        OUT_TMPL = a.out_tmpl
    if not stage_ckpt(a.seed, ACTIVE).exists():
        raise FileNotFoundError(f'U2 stage1 endpoint missing: '
                                f'{stage_ckpt(a.seed, ACTIVE)}')
    steps = ([a.step] if a.step != 'all' else
             ['export', 'stats', 'train-r', 'train-ref', 'eval'])
    for s in steps:
        if s == 'export':
            step_export(a.seed, a.workers)
        elif s == 'stats':
            union_stats()
        elif s == 'train-r':
            step_train_r(a.seed, a.device)
        elif s == 'train-ref':
            step_train_ref(a.seed, a.device)
        elif s == 'eval':
            step_eval(a.seed, a.device)
    print('U2_CHAIN_DONE', flush=True)


if __name__ == '__main__':
    main()
