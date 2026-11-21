"""Pixel-aligned refinement (Ref): MANO coarse operators, input construction
and training. The released model uses its output directly (gamma = 1); the
residual is bounded to +-5 mm per coordinate.
"""
import os
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from pipeline import Bundle, guard  # noqa: E402
from mvhand.geometry.mesh_sampling import coarse_mesh_operators  # noqa: E402
from mvhand.regional.refinement import MultiViewRefinement  # noqa: E402
from mvhand.regional.training import errors, state_hash  # noqa: E402
from mvhand.runtime.mano_compat import legacy_mano_imports  # noqa: E402
from mvhand.training.real import load_mano_base_regressor  # noqa: E402
from mvhand.training.refinement import refinement_loss  # noqa: E402

BUNDLES = Path(os.environ.get("ALIGNHAND_DATA", "data") + "/results/s1/bundles")
DATASETS = ("dexycb", "ho3d", "hanco")


def build_operators():
    with legacy_mano_imports(), open(os.environ.get("ALIGNHAND_DATA", "data") + "/MANO_RIGHT.pkl", "rb") as f:
        template = pickle.load(f, encoding="latin1")
    ops = coarse_mesh_operators(
        np.asarray(template["v_template"], dtype=np.float64), template["f"], 195)
    _shipped_p = os.environ.get("ALIGNHAND_DATA", "data") + "/handoff/models/assets/ref_upsample_195_to_778.npy"
    if os.path.exists(_shipped_p):
        shipped = np.load(_shipped_p)
        if np.abs(ops["upsample"] - shipped).max() != 0:
            raise ValueError("rebuilt operators differ from the handoff asset")
    else:
        print('refinement: cached upsample asset not found; '
              'operators rebuilt from MANO without the equality '
              'check', flush=True)
    return ops


def ref_inputs(extras, device):
    return dict(
        vertices=extras["x0"].double(),
        feature_maps=extras["fm"],
        intrinsics=extras["intrinsics"],
        reference_to_camera=extras["reference_to_camera"],
        raw_to_crop=extras["raw_to_crop"],
        view_mask=extras["view_valid"],
        support_weights=extras["support_weights"],
        scaffold_joints=extras["scaffold"],
        placed_prior_vertices=extras["placed"],
        correction_magnitude=extras["correction_mag"],
    )


def batch_extras(bundle, sel, device):
    inputs, fm, extras = bundle.batch(sel, device)
    extras["fm"] = fm
    extras["view_valid"] = inputs["view_valid"]
    return extras


def load_supervision(bundle):
    with np.load(Path(bundle.meta["prepared"]) / "PRIVATE_SCORE_TARGETS.npz",
                 allow_pickle=False) as z:
        gt = z["vertices_reference_m"][bundle.meta["indices"]].astype(np.float64)
        j2d = z["joints_2d_raw"][bundle.meta["indices"]].astype(np.float64)
    return gt, j2d


def supervision_batch(gt, j2d, sel, views, device):
    truth = torch.as_tensor(gt[sel], device=device)
    raw = torch.as_tensor(j2d[sel], device=device)
    mask = torch.ones(raw.shape[:-1], dtype=torch.bool, device=device)
    return {"vertices_reference_m": truth, "joints_2d_raw": raw, "mask": mask}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["gate12", "gate3"], required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--device", default="cuda:7")
    p.add_argument("--seed", type=int, default=20261025)
    p.add_argument("--out", type=Path,
                   default=Path(os.environ.get("ALIGNHAND_DATA", "data") + "/results/ref23"))
    args = p.parse_args()
    started = time.monotonic()
    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(8)
    out = guard(args.out / args.mode)
    out.mkdir(parents=True, exist_ok=True)
    operators = build_operators()
    model = MultiViewRefinement(operators, displacement_bound_m=.005,
                                include_correction_magnitude=True).to(device)
    with legacy_mano_imports():
        reg = load_mano_base_regressor(
            Path(os.environ.get("ALIGNHAND_DATA", "data") + "/MANO_RIGHT.pkl")).to(device).float()
    train_bundles = {ds: Bundle(BUNDLES / ds / "fit8192") for ds in DATASETS}
    dev_bundles = {ds: Bundle(BUNDLES / ds / "dev256") for ds in DATASETS}
    supervision = {ds: load_supervision(b) for ds, b in train_bundles.items()}

    # Gate 1: zero-initialized Ref must return the base mesh exactly.
    model.eval()
    zero_identity = {}
    with torch.no_grad():
        for ds, bundle in train_bundles.items():
            extras = batch_extras(bundle, list(bundle.usable[:2]), device)
            result = model(**ref_inputs(extras, device))
            diff = float((result.vertices - extras["x0"].double()).abs().max())
            zero_identity[ds] = diff
            if diff != 0:
                raise ValueError(f"Ref zero init broken on {ds}: {diff}")
    print(json.dumps({"gate1_zero_identity": zero_identity}), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    log = []
    updates = 0

    def update(ds, sel):
        nonlocal updates
        bundle = train_bundles[ds]
        gt, j2d = supervision[ds]
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for start in range(0, len(sel), 8):
            part = sel[start:start + 8]
            extras = batch_extras(bundle, part, device)
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
        opt.step()
        updates += 1
        return total

    if args.mode == "gate12":
        ds = "dexycb"
        rows = list(train_bundles[ds].usable[:32])
        gt, _ = supervision[ds]
        with torch.no_grad():
            extras = batch_extras(train_bundles[ds], rows, device)
            base_v, _ = errors(extras["x0"].double(),
                               torch.as_tensor(gt[rows], device=device), reg)
        model.train()
        for step in range(200):
            value = update(ds, rows)
            if step % 25 == 0:
                log.append({"step": step, "loss": value})
                print(json.dumps(log[-1]), flush=True)
        model.eval()
        with torch.no_grad():
            extras = batch_extras(train_bundles[ds], rows, device)
            result = model(**ref_inputs(extras, device))
            v, _ = errors(result.vertices,
                          torch.as_tensor(gt[rows], device=device), reg)
        report = {"mode": "gate12", "zero_identity": zero_identity,
                  "train_groups": 32,
                  "baseline_train_mpvpe_mm": float(base_v.mean()),
                  "after_200_updates_mpvpe_mm": float(v.mean()),
                  "pass": bool(float(v.mean()) < float(base_v.mean()) - 0.2),
                  "wall_hours": (time.monotonic() - started) / 3600}
    else:
        rounds = max(len(b.usable) for b in train_bundles.values()) // 64
        queues = {ds: [] for ds in DATASETS}
        for epoch in range(1, args.epochs + 1):
            model.train()
            for _ in range(rounds):
                for ds in rng.permutation(DATASETS):
                    rows = train_bundles[ds].usable
                    while len(queues[ds]) < 64:
                        queues[ds].extend(rng.permutation(rows).tolist())
                    sel = queues[ds][:64]
                    del queues[ds][:64]
                    value = update(ds, sel)
                    if updates == 1 or updates % 50 == 0:
                        log.append({"epoch": epoch, "update": updates,
                                    "dataset": str(ds), "loss": value,
                                    "elapsed_s": time.monotonic() - started})
                        print(json.dumps(log[-1]), flush=True)
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "updates": updates, "bound_m": .005,
                        "state_hash": state_hash(model)},
                       out / f"CHECKPOINT_EPOCH_{epoch:03d}.pt")
        model.eval()
        metrics = {}
        for ds, bundle in dev_bundles.items():
            gt = bundle.targets()
            rows = list(bundle.usable)
            sums = {"v": 0.0, "bv": 0.0, "j": 0.0, "bj": 0.0}
            with torch.no_grad():
                for start in range(0, len(rows), 32):
                    sel = rows[start:start + 32]
                    extras = batch_extras(bundle, sel, device)
                    result = model(**ref_inputs(extras, device))
                    truth = torch.as_tensor(gt[sel], device=device)
                    v, j = errors(result.vertices, truth, reg)
                    bv, bj = errors(extras["x0"].double(), truth, reg)
                    sums["v"] += float(v.sum()); sums["j"] += float(j.sum())
                    sums["bv"] += float(bv.sum()); sums["bj"] += float(bj.sum())
            n = len(rows)
            metrics[ds] = {"groups": n, "mpvpe_mm": sums["v"] / n,
                           "baseline_mpvpe_mm": sums["bv"] / n,
                           "mesh_mpjpe_mm": sums["j"] / n,
                           "baseline_mesh_mpjpe_mm": sums["bj"] / n,
                           "delta_mm": (sums["v"] - sums["bv"]) / n}
        report = {"mode": "gate3", "zero_identity": zero_identity,
                  "bound_m": .005, "updates": updates, "epochs": args.epochs,
                  "metrics_dev256": metrics, "train_log": log[-5:],
                  "deviations": ["bound .005 instead of v24's .008 (branch spec)",
                                 "all-true 2D supervision mask (asset not shipped)"],
                  "wall_hours": (time.monotonic() - started) / 3600}
    report["Final_accesses"] = 0
    (out / "REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
