"""Training loops for regional fusion (R) and pixel-aligned refinement (Ref),
called by scripts/train/train_fusion_refinement.py.
"""
import os
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from pipeline import Bundle, guard, HANDOFF  # noqa: E402
from mvhand.fusion.candidates import CandidateEvidenceHead  # noqa: E402
from mvhand.fusion.candidate_bank import forward_single_prior  # noqa: E402
from mvhand.fusion.training import errors, state_hash  # noqa: E402
from refinement import (build_operators, ref_inputs, batch_extras,  # noqa: E402
                         load_supervision, supervision_batch)
from mvhand.refinement.model import MultiViewRefinement  # noqa: E402
from mvhand.prior.mano_compat import legacy_mano_imports  # noqa: E402
from mvhand.training.validation import load_mano_base_regressor  # noqa: E402
from mvhand.refinement.supervision import refinement_loss  # noqa: E402

A3 = Path(os.environ.get("ALIGNHAND_DATA", "data") + "/results/s1/bundles/hanco_a3")
OUTROOT = guard(Path(os.environ.get("ALIGNHAND_DATA", "data") + "/results/a3clean"))


def clean_stats(fit):
    f = OUTROOT / "CLEAN_GEOMETRY_STATS.pt"
    if f.exists():
        s = torch.load(f, map_location="cpu", weights_only=False)
        return s["mean"], s["scale"]
    g = fit.arrays["geometry_raw"]
    v = fit.arrays["geometry_valid"]
    rows = fit.usable
    dim = g.shape[-1]
    total = np.zeros(dim); squares = np.zeros(dim); count = np.zeros(dim, np.int64)
    for start in range(0, len(rows), 256):
        sel = rows[start:start + 256]
        x = np.asarray(g[sel], np.float64)
        m = np.asarray(v[sel], bool)
        if m.shape != x.shape:
            m = np.broadcast_to(m[..., None] if m.ndim == x.ndim - 1 else m, x.shape)
        if not np.isfinite(x[m]).all():
            raise ValueError("nonfinite valid geometry in clean fit bundle")
        x = np.where(m, x, 0)
        axes = tuple(range(x.ndim - 1))
        total += x.sum(axis=axes); squares += (x * x).sum(axis=axes)
        count += m.sum(axis=axes)
    if (count == 0).any():
        raise ValueError("unsupported feature dimension")
    mean = total / count
    scale = np.sqrt(np.maximum(0, squares / count - mean ** 2)).clip(1e-8)
    s = dict(schema="a3clean-fit-stats.v1", split="fit",
             fit_groups=int(len(rows)), mean=torch.from_numpy(mean),
             scale=torch.from_numpy(scale), counts=torch.from_numpy(count))
    tmp = f.with_suffix(".tmp")
    torch.save(s, tmp); tmp.rename(f)
    return s["mean"], s["scale"]


def assets(device):
    basis = torch.from_numpy(
        np.load(HANDOFF / "models/assets/region_mask_b_778x6.npy")).double().to(device)
    reg = torch.from_numpy(np.load(
        HANDOFF / "models/assets/mano_base_joint_regressor_16x778.npy")).double().to(device)
    return basis, reg


def r_forward(head, basis, bundle, sel, device, mean, scale):
    inputs, fm, extras = bundle.batch(sel, device)
    res = forward_single_prior(head, basis, inputs, fm,
                               mean=mean.to(device), scale=scale.to(device))
    return res, extras, inputs, fm


def eval_r(head, basis, reg, bundle, device, mean, scale, batch=64):
    gt_all = torch.as_tensor(bundle.targets(), device=device)
    rows = list(bundle.usable)
    sums = dict(v=0.0, j=0.0, bv=0.0, bj=0.0)
    per_key = {}
    head.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            sel = rows[start:start + batch]
            res, extras, _, _ = r_forward(head, basis, bundle, sel, device, mean, scale)
            gt = gt_all[sel]
            v, j = errors(res.vertices, gt, reg)
            bv, bj = errors(extras["x0"].double(), gt, reg)
            sums["v"] += float(v.sum()); sums["j"] += float(j.sum())
            sums["bv"] += float(bv.sum()); sums["bj"] += float(bj.sum())
            for o, r_ in enumerate(sel):
                per_key[bundle.meta["keys"][r_]] = dict(
                    clean_mpvpe_mm=float(v[o]), shipped_mpvpe_mm=float(bv[o]))
    n = len(rows)
    return dict(groups=n, clean_mpvpe_mm=sums["v"] / n,
                clean_mesh_mpjpe_mm=sums["j"] / n,
                shipped_mpvpe_mm=sums["bv"] / n,
                shipped_mesh_mpjpe_mm=sums["bj"] / n), per_key


def train_r(args, out, fit, dev256, device, mean, scale):
    basis, reg = assets(device)
    torch.manual_seed(args.seed); np.random.seed(args.seed % (2 ** 31))
    head = CandidateEvidenceHead().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
    gt_all = torch.as_tensor(fit.targets(), device=device)
    rows = np.asarray(fit.usable)
    rng = np.random.default_rng(args.seed)
    orders = [rng.permutation(rows).tolist() for _ in range(args.epochs)]
    started = time.monotonic(); updates = 0
    for epoch, order in enumerate(orders, 1):
        head.train()
        for start in range(0, len(order), 64):
            sel = order[start:start + 64]
            opt.zero_grad(set_to_none=True)
            res, _, _, _ = r_forward(head, basis, fit, sel, device, mean, scale)
            v, j = errors(res.vertices, gt_all[sel], reg)
            loss = (v + .5 * j).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite R loss")
            loss.backward()
            if not all(torch.isfinite(q.grad).all() for q in head.parameters()
                       if q.grad is not None):
                raise FloatingPointError("nonfinite R gradients")
            opt.step(); updates += 1
            if updates == 1 or updates % 50 == 0:
                print(json.dumps(dict(phase="r", epoch=epoch, updates=updates,
                                      loss=float(loss),
                                      elapsed_s=time.monotonic() - started)),
                      flush=True)
        torch.save(dict(state_dict=head.state_dict(), epoch=epoch,
                        identity=dict(dataset="hanco_a3", variant="M1",
                                      seed=args.seed, recipe="v2_train_regional-mirror",
                                      stats="clean-fit-bundle",
                                      train_bundle=str(fit.path)),
                        state_hash=state_hash(head)),
                   out / f"R_CHECKPOINT_EPOCH_{epoch:03d}.pt")
    metrics, per_key = eval_r(head, basis, reg, dev256, device, mean, scale)
    (out / "R_DEV256.json").write_text(json.dumps(dict(
        seed=args.seed, metrics=metrics, updates=updates), indent=1))
    (out / "R_DEV256_PERKEY.json").write_text(json.dumps(per_key))
    print("R_PHASE_DONE", json.dumps(metrics), flush=True)
    return head


def load_r(out, device):
    ck = torch.load(out / "R_CHECKPOINT_EPOCH_010.pt", map_location="cpu",
                    weights_only=False)
    head = CandidateEvidenceHead()
    head.load_state_dict(ck["state_dict"], strict=True)
    return head.to(device).eval().requires_grad_(False)


def clean_extras(head, basis, bundle, sel, device, mean, scale):
    with torch.no_grad():
        res, extras, inputs, fm = r_forward(head, basis, bundle, sel, device,
                                            mean, scale)
    extras["fm"] = fm
    extras["view_valid"] = inputs["view_valid"]
    extras["x0"] = res.vertices.detach()
    return extras


def eval_ref(model, head, basis, reg, bundle, device, mean, scale, batch=32):
    gt_all = torch.as_tensor(bundle.targets(), device=device)
    rows = list(bundle.usable)
    sums = dict(v=0.0, bv=0.0, sv=0.0)
    per_key = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            sel = rows[start:start + batch]
            extras = clean_extras(head, basis, bundle, sel, device, mean, scale)
            shipped_x0 = bundle.batch(sel, device)[2]["x0"].double()
            result = model(**ref_inputs(extras, device))
            gt = gt_all[sel]
            v, _ = errors(result.vertices, gt, reg)
            bv, _ = errors(extras["x0"].double(), gt, reg)
            sv, _ = errors(shipped_x0, gt, reg)
            sums["v"] += float(v.sum()); sums["bv"] += float(bv.sum())
            sums["sv"] += float(sv.sum())
            for o, r_ in enumerate(sel):
                per_key[bundle.meta["keys"][r_]] = dict(
                    ref_mpvpe_mm=float(v[o]), cleanR_mpvpe_mm=float(bv[o]),
                    shippedR_mpvpe_mm=float(sv[o]))
    n = len(rows)
    return dict(groups=n, ref_mpvpe_mm=sums["v"] / n,
                cleanR_mpvpe_mm=sums["bv"] / n,
                shippedR_mpvpe_mm=sums["sv"] / n), per_key


def train_ref(args, out, fit, device, mean, scale):
    basis, reg_d = assets(device)
    head = load_r(out, device)
    torch.manual_seed(args.seed + 1); np.random.seed((args.seed + 1) % (2 ** 31))
    operators = build_operators()
    model = MultiViewRefinement(operators, displacement_bound_m=.005,
                                include_correction_magnitude=True).to(device)
    with legacy_mano_imports():
        reg = load_mano_base_regressor(
            Path(os.environ.get("ALIGNHAND_DATA", "data") + "/MANO_RIGHT.pkl")).to(device).float()
    gt, j2d = load_supervision(fit)
    # gate 1: zero-init Ref returns the clean-R mesh exactly
    model.eval()
    with torch.no_grad():
        extras = clean_extras(head, basis, fit, list(fit.usable[:2]), device,
                              mean, scale)
        diff = float((model(**ref_inputs(extras, device)).vertices
                      - extras["x0"].double()).abs().max())
    if diff != 0:
        raise ValueError(f"Ref zero init broken: {diff}")
    print(json.dumps(dict(gate1_zero_identity=diff)), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed + 1)
    rows = fit.usable
    rounds = len(rows) // 64
    started = time.monotonic(); updates = 0
    queue = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for _ in range(rounds):
            while len(queue) < 64:
                queue.extend(rng.permutation(rows).tolist())
            sel = queue[:64]; del queue[:64]
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for start in range(0, len(sel), 8):
                part = sel[start:start + 8]
                extras = clean_extras(head, basis, fit, part, device, mean, scale)
                result = model(**ref_inputs(extras, device))
                targets = supervision_batch(gt, j2d, part,
                                            extras["view_valid"].shape[1], device)
                losses, _ = refinement_loss(
                    result, targets, regressor=reg,
                    projection_matrices=extras["projection"].double(),
                    raw_to_crop=extras["raw_to_crop"].double(),
                    supervision_mask=targets["mask"],
                    view_mask=extras["view_valid"], laplacian=model.laplacian)
                loss = losses.mean() * len(part) / len(sel)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite Ref loss")
                loss.backward()
                total += float(loss.detach())
            opt.step(); updates += 1
            if updates == 1 or updates % 50 == 0:
                print(json.dumps(dict(phase="ref", epoch=epoch, update=updates,
                                      loss=total,
                                      elapsed_s=time.monotonic() - started)),
                      flush=True)
        torch.save(dict(state_dict=model.state_dict(), epoch=epoch,
                        seed=args.seed, r_checkpoint="R_CHECKPOINT_EPOCH_010.pt"),
                   out / f"REF_CHECKPOINT_EPOCH_{epoch:03d}.pt")
    for tag in ("dev256", "devfull"):
        bundle = Bundle(A3 / tag)
        metrics, per_key = eval_ref(model, head, basis, reg_d, bundle, device,
                                    mean, scale)
        (out / f"REF_{tag.upper()}.json").write_text(json.dumps(dict(
            seed=args.seed, metrics=metrics), indent=1))
        (out / f"REF_{tag.upper()}_PERKEY.json").write_text(json.dumps(per_key))
        print(f"REF_EVAL_{tag}", json.dumps(metrics), flush=True)
    print("REF_PHASE_DONE", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--phase", choices=["r", "ref", "both"], default="both")
    p.add_argument("--epochs", type=int, default=10)
    args = p.parse_args()
    out = OUTROOT / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    fit = Bundle(A3 / "fit8192")
    dev256 = Bundle(A3 / "dev256")
    mean, scale = clean_stats(fit)
    if args.phase in ("r", "both") and not (out / "R_DEV256.json").exists():
        train_r(args, out, fit, dev256, args.device, mean, scale)
    if args.phase in ("ref", "both"):
        train_ref(args, out, fit, args.device, mean, scale)
    print("A3CLEAN_DONE", flush=True)


if __name__ == "__main__":
    main()
