"""Shared inference pipeline used by every script.

Loads cached frozen-HaMeR outputs for a multi-view group and runs Stage I
(observation refinement), support-weighted triangulation, metric alignment and
regional fusion (FrozenForward). Also holds the data-root paths and helpers.
"""
import hashlib
import json
import os
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, os.environ.get('ALIGNHAND_SRC', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src')))
from mvhand.geometry.mesh_recovery import recover_positioned_mesh  # noqa: E402
from mvhand.models.interfaces import Form2FrameBatch, HandPriorOutput  # noqa: E402
from mvhand.regional.candidates import CandidateEvidenceHead  # noqa: E402
from mvhand.regional.single_prior import (  # noqa: E402
    SinglePriorBank, assemble_single_prior_inputs, forward_single_prior)
from mvhand.training.cached_stage1 import build_cached_stage1  # noqa: E402

F2F = Path(os.environ.get("ALIGNHAND_DATA", "data"))
HANDOFF = F2F / "handoff"
SEED = 20261023
PARENTS = {
    "dexycb": ("M1", "a374f08fae73",
               HANDOFF / "exchange/parents/dexycb/hamer/M1/CHECKPOINT_UPDATE_08000.pt"),
    "hanco": ("M1", "2c4163dd3c1d",
              HANDOFF / "exchange/parents/hanco/hamer/M1/CHECKPOINT_UPDATE_08000.pt"),
    "ho3d": ("M1", "e0a73b0eff5f",
             HANDOFF / "exchange/parents/ho3d/hamer/M1/CHECKPOINT_UPDATE_00800.pt"),
}
PREPARED = {
    ("dexycb", "fit"): F2F / "prepared/v21_dexycb_fit_v2",
    ("ho3d", "fit"): F2F / "prepared/v21_ho3d_fit_v1",
    ("hanco", "fit"): F2F / "prepared/v21_hanco_fit_v1",
    ("dexycb", "development"): F2F / "prepared/v21_dexycb_dev_s0_v1",
    ("ho3d", "development"): F2F / "prepared/v21_ho3d_dev_s0_v1",
    ("hanco", "development"): F2F / "prepared/v21_hanco_dev_s0_v1",
}
FIELDS64 = ("anchor", "candidates", "q_init", "alpha0")
FIELDS32 = ("geometry_raw", "candidate_grid", "candidate_raw", "anchor_grid",
            "anchor_raw", "support_weights", "corrected_raw", "projection",
            "reference_to_camera", "raw_to_crop", "intrinsics", "scaffold",
            "correction_mag", "placed", "x0")
FIELDSB = ("geometry_valid", "valid", "projection_valid", "anchor_valid",
           "view_valid", "placed_valid")


def guard(path):
    if "final" in str(path).lower():
        raise PermissionError(f"path mentions a Final population: {path}")
    return Path(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cache_files(ds, split):
    root = guard(F2F / "caches" / ds)
    if split == "development":
        dirs = [root / f"v21_s0_{ds}_hamer_s25_v1"]
        sub = "prior_inputs"
    else:
        dirs = sorted(root.glob(f"v21_fit_{ds}_hamer_*_v1"),
                      key=lambda d: int(d.name.split("_")[4]))
        sub = None
    files = []
    for d in dirs:
        index = json.loads((d / "CACHE_INDEX.json").read_text())
        for i, entry in enumerate(index):
            if entry["index"] not in (i, len(files)):
                raise ValueError(f"cache index out of order in {d}")
            files.append((d / sub / entry["file"]) if sub else (d / entry["file"]))
    prepared = guard(PREPARED[(ds, split)])
    rows = [json.loads(s) for s in (prepared / "ROWS.jsonl").read_text().splitlines()]
    if len(rows) != len(files) or any(r["split"] != split for r in rows):
        raise ValueError("cache/prepared mismatch")
    return files, rows, prepared


class FrozenForward:
    """Frozen M1 stage1 + R(seed20261023); superset of the V4 forward."""

    def __init__(self, ds, split):
        torch.set_num_threads(1)
        variant, digest, parentfile = PARENTS[ds]
        # Released-model use: the per-dataset parent Stage I / R / stats are
        # not distributed. When absent, the modules are only constructed here
        # and every caller loads the shared (U2) weights over them with
        # strict=True before any forward, so results are unchanged.
        self.stage = build_cached_stage1(variant)
        if Path(parentfile).exists():
            ck = torch.load(parentfile, map_location="cpu", weights_only=False)
            self.stage.load_state_dict(ck["trainable_state"], strict=True)
        self.stage.eval().requires_grad_(False)
        self.basis = torch.from_numpy(
            np.load(HANDOFF / "models/assets/region_mask_b_778x6.npy")).double()
        self.reg = torch.from_numpy(np.load(
            HANDOFF / "models/assets/mano_base_joint_regressor_16x778.npy")).double()
        root = HANDOFF / "models/single_prior_R" / ds / variant / f"seed{SEED}"
        self.head = CandidateEvidenceHead()
        self.mean = self.scale = None
        if (root / "CHECKPOINT_EPOCH_010.pt").exists():
            rck = torch.load(root / "CHECKPOINT_EPOCH_010.pt", map_location="cpu",
                             weights_only=False)
            self.head.load_state_dict(rck["state_dict"], strict=True)
            stats = torch.load(root / "GEOMETRY_STATS.pt", map_location="cpu",
                               weights_only=False)
            if stats["split"] != "fit":
                raise ValueError("geometry stats must come from the Fit split")
            self.mean, self.scale = stats["mean"], stats["scale"]
        self.head.eval().requires_grad_(False)
        self.files, self.rows, self.prepared = cache_files(ds, split)
        self.inputs = dict(np.load(self.prepared / "INPUTS.npz", allow_pickle=False))
        self.ds, self.split = ds, split

    def forward(self, ids):
        """Returns dict of numpy arrays per group (len(ids) first axis), or
        None entries for groups whose placement fails."""
        values = {f: [] for f in ("joints_2d_normalized", "visual_features",
                                  "local_joints_3d", "local_vertices_3d")}
        fmaps = []
        for i in ids:
            with np.load(self.files[i], allow_pickle=False) as z:
                for name in values:
                    values[name].append(z[name][0])
                fmaps.append(z["feature_maps"][0].astype(np.float16))
        tensor = lambda x: torch.as_tensor(x)
        affine = self.inputs["raw_to_crop_s25"][ids]
        inverse = np.stack([[cv2.invertAffineTransform(a) for a in row]
                            for row in affine])
        mask = tensor(self.inputs["runtime_view_mask"][ids])
        batch = Form2FrameBatch(torch.zeros((*mask.shape, 3, 1, 1)),
                                tensor(self.inputs["intrinsics"][ids]),
                                tensor(self.inputs["camera_to_world"][ids]),
                                tensor(inverse), mask, tensor(affine))
        prior = HandPriorOutput(**{n: tensor(np.stack(v)).float()
                                   for n, v in values.items()})
        with torch.inference_mode():
            pred = self.stage.forward_from_prior(prior, batch)
            rec = recover_positioned_mesh(
                pred.weighted.points_3d, prior.local_joints_3d,
                prior.local_vertices_3d, pred.relative_view_weights,
                pred.weighted.valid, geometry_blend=.5)
            if not rec.valid.all():
                if len(ids) == 1:
                    return None
                parts = [self.forward([i]) for i in ids]
                ok = [p for p in parts if p is not None]
                if not ok:
                    return None
                merged = {k: np.concatenate([p[k] for p in ok]) for k in ok[0]}
                merged["ids"] = np.concatenate(
                    [p["ids"] for p in ok]) if "ids" in ok[0] else None
                return merged
            inputs, _ = assemble_single_prior_inputs(
                pred, rec, batch, basis=self.basis, regressor=self.reg,
                feature_crop_bounds=(32, 0, 192, 256))
            for name in ("geometry_raw", "candidate_raw", "candidate_grid",
                         "anchor_raw", "anchor_grid"):
                inputs[name] = inputs[name].float().double()
            fm = torch.from_numpy(np.stack(fmaps))
            result = forward_single_prior(self.head, self.basis, inputs, fm,
                                          mean=self.mean, scale=self.scale)
            bank = inputs["bank"]
            correction = torch.linalg.vector_norm(
                pred.corrected_joints_2d_normalized - prior.joints_2d_normalized,
                dim=-1) * 256
            out = {
                "ids": np.asarray(ids),
                "fm": np.stack(fmaps),
                "anchor": bank.anchor.numpy(),
                "candidates": bank.candidates.numpy(),
                "q_init": bank.q_init.numpy(),
                "alpha0": result.q_new.numpy(),
                "valid": bank.valid.numpy(),
                "geometry_raw": inputs["geometry_raw"].float().numpy(),
                "geometry_valid": inputs["geometry_valid"].numpy(),
                "candidate_grid": inputs["candidate_grid"].float().numpy(),
                "candidate_raw": inputs["candidate_raw"].float().numpy(),
                "projection_valid": inputs["projection_valid"].numpy(),
                "anchor_grid": inputs["anchor_grid"].float().numpy(),
                "anchor_raw": inputs["anchor_raw"].float().numpy(),
                "anchor_valid": inputs["anchor_valid"].numpy(),
                "view_valid": inputs["view_valid"].numpy(),
                "support_weights": pred.relative_view_weights.float().numpy(),
                "corrected_raw": pred.corrected_joints_2d_raw.float().numpy(),
                "projection": pred.projection_matrices.float().numpy(),
                "reference_to_camera": pred.reference_to_camera.float().numpy(),
                "raw_to_crop": batch.raw_to_crop.float().numpy(),
                "intrinsics": batch.intrinsics.float().numpy(),
                "scaffold": pred.weighted.points_3d.float().numpy(),
                "correction_mag": correction.float().numpy(),
                "placed": rec.aligned_prior_vertices_3d.float().numpy(),
                "placed_valid": (rec.alignment_valid.bool()
                                 & batch.runtime_view_mask.bool()).numpy(),
                "x0": result.vertices.float().numpy(),
            }
        return out


_CTX = {}


def _init(ds, split):
    _CTX["fwd"] = FrozenForward(ds, split)


def _work(chunk):
    out = _CTX["fwd"].forward(list(chunk))
    return out


def seq_stratified(rows, count, salt="s-line-v1"):
    """Deterministic sequence-balanced round-robin selection of row indices."""
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["sequence_id"], []).append(i)
    def order_key(i, seq):
        return hashlib.sha256(f"{salt}|{seq}|{rows[i]['key']}".encode()).hexdigest()
    for seq in groups:
        groups[seq].sort(key=lambda i: order_key(i, seq))
    sequences = sorted(groups, key=lambda s: hashlib.sha256(
        f"{salt}|{s}".encode()).hexdigest())
    selected = []
    position = 0
    while len(selected) < min(count, len(rows)):
        for seq in sequences:
            if position < len(groups[seq]):
                selected.append(groups[seq][position])
                if len(selected) >= min(count, len(rows)):
                    break
        position += 1
    return sorted(selected)


def export_bundle(ds, split, indices, out_dir, workers=48, chunk=8):
    out = guard(Path(out_dir))
    out.mkdir(parents=True, exist_ok=True)
    fwd = FrozenForward(ds, split)
    n = len(indices)
    shapes = None
    maps = {}
    written = np.zeros(n, bool)
    chunks = [indices[i:i + chunk] for i in range(0, n, chunk)]
    position = {int(i): j for j, i in enumerate(indices)}

    def allocate(sample):
        nonlocal shapes
        shapes = {}
        for k, v in sample.items():
            if k == "ids":
                continue
            dtype = (np.float16 if k == "fm" else np.float64 if k in FIELDS64
                     else np.float32 if k in FIELDS32 else bool)
            maps[k] = np.lib.format.open_memmap(
                out / f"{k}.npy", mode="w+", dtype=dtype,
                shape=(n, *v.shape[1:]))

    def store(res):
        if res is None:
            return
        for j, gid in enumerate(res["ids"]):
            p = position[int(gid)]
            for k, m in maps.items():
                m[p] = res[k][j]
            written[p] = True

    with Pool(workers, initializer=_init, initargs=(ds, split)) as pool:
        for res in pool.imap_unordered(_work, chunks):
            if res is not None and shapes is None:
                allocate(res)
            store(res)
    for m in maps.values():
        m.flush()
    meta = {"dataset": ds, "split": split, "seed": SEED,
            "indices": [int(i) for i in indices],
            "keys": [fwd.rows[i]["key"] for i in indices],
            "sequence_ids": [fwd.rows[i]["sequence_id"] for i in indices],
            "success": [bool(x) for x in written],
            "prepared": str(fwd.prepared)}
    (out / "meta.json").write_text(json.dumps(meta) + "\n")
    print(json.dumps({"bundle": str(out), "groups": n,
                      "failures": int((~written).sum())}), flush=True)
    return out


class Bundle:
    def __init__(self, path):
        self.path = guard(Path(path))
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.arrays = {}
        for f in self.path.glob("*.npy"):
            self.arrays[f.stem] = np.load(f, mmap_mode="r")
        self.n = len(self.meta["indices"])
        ok = np.asarray(self.meta["success"], bool)
        self.usable = np.flatnonzero(ok)

    def targets(self):
        """GT vertices for scoring only, aligned to bundle order (meters->mm kept in m)."""
        with np.load(Path(self.meta["prepared"]) / "PRIVATE_SCORE_TARGETS.npz",
                     allow_pickle=False) as z:
            gt = z["vertices_reference_m"][self.meta["indices"]].astype(np.float64)
        return gt

    def batch(self, rows, device):
        """Rows are positions within the bundle. Returns (inputs, fm, extras)."""
        rows = list(map(int, rows))
        a = lambda k: torch.as_tensor(np.ascontiguousarray(self.arrays[k][rows]),
                                      device=device)
        views = self.arrays["view_valid"].shape[1]
        count = self.arrays["candidates"].shape[1]
        source = torch.zeros(count, 3, device=device, dtype=torch.float64)
        source[0, 0] = 1
        source[1:, 1] = 1
        origins = torch.arange(-1, views, device=device)
        bank = SinglePriorBank(a("anchor").double(), a("candidates").double(),
                               a("valid").bool(), a("q_init").double(),
                               origins, source)
        inputs = {"bank": bank,
                  "geometry_raw": a("geometry_raw").double(),
                  "geometry_valid": a("geometry_valid").bool(),
                  "candidate_raw": a("candidate_raw").double(),
                  "candidate_grid": a("candidate_grid").double(),
                  "projection_valid": a("projection_valid").bool(),
                  "anchor_raw": a("anchor_raw").double(),
                  "anchor_grid": a("anchor_grid").double(),
                  "anchor_valid": a("anchor_valid").bool(),
                  "view_valid": a("view_valid").bool()}
        fm = a("fm")
        extras = {k: a(k) for k in
                  ("alpha0", "support_weights", "corrected_raw", "projection",
                   "reference_to_camera", "raw_to_crop", "intrinsics",
                   "scaffold", "correction_mag", "placed", "placed_valid", "x0")}
        return inputs, fm, extras
