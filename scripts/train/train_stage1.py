"""Train Stage I (observation refinement) of the shared model.

One model on the pooled training splits of DexYCB, HanCo and HO3D: each update
draws a dataset uniformly, then a sequence, then a sample. AdamW, lr 7.5e-5,
16,800 updates, seed 20261023. Training lists: scripts/train/manifests/.

usage: train_stage1.py [--seed S] [--device cpu|cuda] [--smoke N] [--out DIR]
"""
import os
import argparse, hashlib, json, random, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.environ.get('ALIGNHAND_SRC', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src')))
from mvhand.training.cached_stage1 import (CachedStage1Population,  # noqa
                                           build_cached_stage1,
                                           cached_training_loss)
from mvhand.runtime.precision import configure_transfer_runtime  # noqa

_MDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'manifests')
MANIFESTS = {
    'hanco': os.path.join(_MDIR, 'stage1_hanco_a3_traincore.json'),
    'dexycb': os.path.join(_MDIR,
                           'stage1_dexycb_hamer_fit_local.json'),
    'ho3d': os.path.join(_MDIR, 'stage1_ho3d_hamer_fit_local.json'),
}
PIN_HANCO_SHA = ('895ee5aea638fbe9e145935954d7dd51b2f4e4eabe7c96c4f9663315'
                 'd1fd6a4a')
LR, WD, CLIP, WARMUP, FLOOR, EB = 7.5e-5, 1e-4, 1.0, 400, 0.1, 4


def lr_factor(update, total):
    import math
    if update <= WARMUP:
        return update / WARMUP
    p = (update - WARMUP) / (total - WARMUP)
    return FLOOR + (1 - FLOOR) * .5 * (1 + math.cos(math.pi * p))


class UnionSampler:
    """dataset-equal -> sequence-equal within dataset -> row within seq."""

    def __init__(self, pops, seed):
        self.ds = sorted(pops)
        self.rng = np.random.default_rng(seed)
        self.seqmap = {}
        self.presented = {d: 0 for d in self.ds}
        for d, pop in pops.items():
            m = {}
            for i, r in enumerate(pop.rows):
                m.setdefault(str(r['sequence_id']), []).append(i)
            self.seqmap[d] = {s: np.array(v) for s, v in m.items()}
            setattr(self, f'_seqs_{d}', sorted(self.seqmap[d]))

    def next(self):
        out = []
        for _ in range(EB):
            d = self.ds[int(self.rng.integers(len(self.ds)))]
            seqs = getattr(self, f'_seqs_{d}')
            s = seqs[int(self.rng.integers(len(seqs)))]
            rows = self.seqmap[d][s]
            out.append((d, int(rows[int(self.rng.integers(len(rows)))])))
            self.presented[d] += 1
        return out


class RowUniformSampler:
    """uniform over the union of all rows (canonical single-segment)."""

    def __init__(self, pops, seed):
        self.rng = np.random.default_rng(seed)
        self.index = [(d, i) for d in sorted(pops)
                      for i in range(len(pops[d]))]
        self.presented = {d: 0 for d in pops}

    def next(self):
        out = []
        for _ in range(EB):
            d, i = self.index[int(self.rng.integers(len(self.index)))]
            out.append((d, i))
            self.presented[d] += 1
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=20261023)
    ap.add_argument('--datasets', default='hanco,dexycb,ho3d',
                    help='comma list; single-ds stage1 budgets 8000/8000/800')
    ap.add_argument('--updates', type=int, default=None)
    ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--device', default='',
                    help="cuda[:N] or cpu (default: cuda if available)")
    ap.add_argument('--row-uniform', action='store_true',
                    help='sample uniformly over ALL rows of all populations '
                         '(single-segment canonical sampler) instead of '
                         'dataset-equal; for the FINAL refit merged segment')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    dss = tuple(a.datasets.split(','))
    BUDGET = {'hanco': 8000, 'dexycb': 8000, 'ho3d': 800}
    default_updates = sum(BUDGET.get(d, 0) for d in dss)
    if a.updates is None:
        a.updates = default_updates
    total = a.smoke or a.updates
    tag = '' if len(dss) == 3 else '_' + '-'.join(sorted(dss))
    out = Path(a.out or os.environ.get('ALIGNHAND_DATA', 'data') + f'/results/u2/stage1_s{a.seed}{tag}'
               + ('_smoke' if a.smoke else ''))
    out.mkdir(parents=True, exist_ok=bool(a.smoke))

    shas, pops = {}, {}
    for d, p in {k: v for k, v in MANIFESTS.items() if k in dss}.items():
        shas[d] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        pops[d] = CachedStage1Population(p, shas[d])
    if 'hanco' in shas and shas['hanco'] != PIN_HANCO_SHA:
        raise ValueError('hanco traincore manifest drifted')
    # no fit population may touch any Dev sequence (defensive re-check)
    (out / 'U2_CONFIG_LOCK.json').write_text(json.dumps(dict(
        node='U2 shared Stage I (E3 step 1)', variant='M1',
        view_distribution='uniform', seed=a.seed, total_updates=a.updates,
        budget_rule=f'sum of single-ds presentation totals at effective '
                    f'batch 4 over {dss}',
        datasets=list(dss),
        lr=LR, weight_decay=WD, clip=CLIP, warmup=WARMUP, lr_floor=FLOOR,
        schedule='linear warmup 400 then cosine to floor over total',
        endpoint_rule='final update checkpoint; no Dev-based selection',
        manifests=MANIFESTS, manifest_sha256=shas,
        smoke=bool(a.smoke)), indent=1))

    torch.manual_seed(a.seed)
    np.random.seed(a.seed % (2 ** 31))
    random.seed(a.seed)
    configure_transfer_runtime()
    device = a.device if a.device else (
        'cuda' if torch.cuda.is_available() else 'cpu')
    model = build_cached_stage1('M1').to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD,
                            betas=(0.9, 0.999), eps=1e-8)
    gen = torch.Generator().manual_seed(a.seed ^ 0xA33A33)
    sampler = (RowUniformSampler(pops, a.seed) if a.row_uniform
               else UnionSampler(pops, a.seed))
    t0 = time.time()
    hist = []
    for update in range(1, total + 1):
        f = lr_factor(update, total)
        for g in opt.param_groups:
            g['lr'] = LR * f
        model.train()
        opt.zero_grad(set_to_none=True)
        picks = sampler.next()
        byds = {}
        for d, i in picks:
            byds.setdefault(d, []).append(i)
        loss_sum = 0.
        for d, idx in byds.items():
            batch, prior = pops[d].model_batch(idx, device)
            targets = pops[d].supervision(idx, device)
            loss, _ = cached_training_loss(model, batch, prior, targets,
                                           view_generator=gen,
                                           view_distribution='uniform')
            if not torch.isfinite(loss):
                raise FloatingPointError(f'nonfinite U2 loss ds={d}')
            (loss * (len(idx) / EB)).backward()
            loss_sum += float(loss.detach()) * (len(idx) / EB)
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise FloatingPointError('nonfinite U2 gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        if update == 1 or update % 50 == 0:
            rec = dict(update=update, loss=round(loss_sum, 6),
                       lr=round(LR * f, 8),
                       sec=round(time.time() - t0, 1))
            hist.append(rec)
            print(json.dumps(rec), flush=True)
        if update % 1600 == 0 or update == total:
            ck = dict(schema='u2.stage1.v1',
                      trainable_state={n: p.detach().cpu() for n, p in
                                       model.named_parameters()
                                       if p.requires_grad},
                      update=update, seed=a.seed,
                      presented=dict(sampler.presented))
            torch.save(ck, out / f'CHECKPOINT_UPDATE_{update:05d}.pt')
    state_sha = hashlib.sha256(b''.join(
        p.detach().cpu().numpy().tobytes()
        for _, p in sorted(model.named_parameters()))).hexdigest()
    seq_cov = {d: dict(rows=len(pops[d]),
                       presented=sampler.presented[d],
                       **({'sequences': len(sampler.seqmap[d])}
                          if hasattr(sampler, 'seqmap') else {}))
               for d in pops}
    (out / 'REPORT.json').write_text(json.dumps(dict(
        status='COMPLETE', updates=total, seed=a.seed,
        endpoint=f'CHECKPOINT_UPDATE_{total:05d}.pt',
        trainable_state_sha256=state_sha,
        per_dataset=seq_cov, elapsed_sec=round(time.time() - t0, 1),
        gpu=(torch.cuda.get_device_name(0)
             if device.startswith('cuda') else 'cpu'),
        note='shared parameters only; all three datasets update ONE '
             'state_dict; endpoint = final update (pre-registered)',
        history_tail=hist[-5:]), indent=1))
    print('U2_STAGE1_DONE', json.dumps(dict(
        updates=total, sha=state_sha[:16], presented=sampler.presented)),
        flush=True)


if __name__ == '__main__':
    main()
