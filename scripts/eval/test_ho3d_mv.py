"""STAGE6_FINAL_V4_HO3D_POEM (FINAL_PLAN_4.json): the one POEM HO3D-MV
terminal-test confirmation batch (frozen v6 2706-group benchmark), scored
against the rebuilt POEM PRIVATE_SCORE_TARGETS, identity V6_HO3D_COMPLETION.
Endpoints exported target-free; ONE targets read (binding row + scoring).

usage: test_ho3d_mv.py [--limit N]   # --limit: endpoints only, NO targets read
"""
import os
import argparse, csv, hashlib, json, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
sys.path.insert(0, os.environ.get('ALIGNHAND_SRC', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src')))
import metrics2 as M2  # noqa: E402

PREP = Path(os.environ.get("ALIGNHAND_DATA", "data") + "/prepared/v7_ho3d_poemtest2706_v1")
CACHE = Path(os.environ.get("ALIGNHAND_DATA", "data") + "/caches/ho3d/v7_poemtest_ho3d_hamer_00000_02706_v1")
U2R = Path(os.environ.get("ALIGNHAND_DATA", "data") + "/results/u2")
U2F = Path(os.environ.get("ALIGNHAND_DATA", "data") + "/results/u2merged")
OUT = Path(os.environ.get("ALIGNHAND_DATA", "data") + "/repo/results/paper/final_ho3d/poem2706")
REL = Path(os.environ.get("ALIGNHAND_DATA", "data") + "/data_ho3d_eval/extracted/ho3d_l0_predictions_v2")
GATE3 = {s: os.environ.get("ALIGNHAND_DATA", "data") + "/results/ref23/gate3{}/gate3/CHECKPOINT_EPOCH_010.pt"
            .format("" if s == 25 else "_s{}".format(s)) for s in (25, 26, 27)}
U15A = os.environ.get("ALIGNHAND_DATA", "data") + "/results/s4/u15a_ho3d/REF_SINGLE.pt"
SEED = 20261023
# Internal comparison stacks (per-dataset U1 with its Refs, merged U2F) are
# not released; they are evaluated only when present. The released shared
# model is the 'u2' stack, whose arms are prefixed u2_ in the outputs.
HAVE_U1 = Path(U15A).exists() and all(Path(p).exists() for p in GATE3.values())
HAVE_U2F = (U2F / 'U2F_MODEL_LOCK.json').exists()
HIST = {"hg": ("HG", 6.752676987278957), "wg": ("WG", 6.753957167878485),
        "parent": ("w/o R", 5.637478587592988),
        "form2frame_h": ("Form2Frame-H", 5.496897226028216),
        "form2frame": ("Form2Frame", 5.578674903313537)}
_W = {}


def _patch():
    import s_common
    from s_common import F2F, PREPARED, guard
    PREPARED[("ho3d", "poemtest")] = PREP
    orig = s_common.cache_files

    def cache_files(ds, split):
        if ds != "ho3d" or split != "poemtest":
            return orig(ds, split)
        d = guard(CACHE)
        index = json.loads((d / "CACHE_INDEX.json").read_text())
        files = [d / e["file"] for e in index]
        prepared = guard(PREPARED[(ds, split)])
        rows = [json.loads(s) for s in
                (prepared / "ROWS.jsonl").read_text().splitlines()]
        assert len(rows) == len(files)
        return files, rows, prepared

    s_common.cache_files = cache_files
    return s_common


def _build_ref(path):
    import torch
    from ref23_train import build_operators
    from mvhand.regional.refinement import MultiViewRefinement
    m = MultiViewRefinement(build_operators(), displacement_bound_m=.005,
                            include_correction_magnitude=True)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck.get("state_dict", ck))
    return m.eval().requires_grad_(False)


def _init():
    import torch
    torch.set_num_threads(1)
    S = _patch()
    from mvhand.regional.candidates import CandidateEvidenceHead

    def alt_stack(stage_ck_path, r_ck_path, stats):
        fwd = S.FrozenForward("ho3d", "poemtest")
        ck = torch.load(stage_ck_path, map_location="cpu", weights_only=False)
        fwd.stage.load_state_dict(ck["trainable_state"], strict=True)
        fwd.stage.eval().requires_grad_(False)
        rck = torch.load(r_ck_path, map_location="cpu", weights_only=False)
        fwd.head = CandidateEvidenceHead()
        fwd.head.load_state_dict(rck.get("state_dict", rck), strict=True)
        fwd.head.eval().requires_grad_(False)
        st = torch.load(stats, map_location="cpu", weights_only=False)
        fwd.mean, fwd.scale = st["mean"], st["scale"]
        return fwd

    if HAVE_U1:
        refs_u1 = {"Full_s{}".format(s): (_build_ref(p), 0.5)
                   for s, p in GATE3.items()}
        refs_u1["Full_u15a"] = (_build_ref(U15A), 1.0)
        _W["u1"] = dict(fwd=S.FrozenForward("ho3d", "poemtest"), refs=refs_u1)
    _W["u2"] = dict(fwd=alt_stack(
        U2R / "stage1_s20261023/CHECKPOINT_UPDATE_16800.pt",
        U2R / "r_s{}/R_FINAL.pt".format(SEED), U2R / "U2_GEOMETRY_STATS.pt"),
        refs={"Full": (_build_ref(U2R / "ref_s{}/REF_FINAL.pt".format(SEED)),
                       None)})
    if HAVE_U2F:
        lock = json.loads((U2F / "U2F_MODEL_LOCK.json").read_text())
        _W["u2f"] = dict(fwd=alt_stack(
            lock["stage1"], U2F / "r/R_FINAL.pt", U2F / "U2F_GEOMETRY_STATS.pt"),
            refs={"Full": (_build_ref(U2F / "ref/REF_FINAL.pt"), None)})


def _endpoints(stack, i):
    import torch
    from ref23_train import ref_inputs
    fwd = stack["fwd"]
    out = fwd.forward([i])
    if out is None:
        return None
    t = torch.as_tensor
    x0 = t(out["x0"]).double()
    extras = {k: t(out[k]) for k in
              ("support_weights", "reference_to_camera", "raw_to_crop",
               "intrinsics", "scaffold", "correction_mag", "placed")}
    extras["x0"] = x0
    extras["fm"] = t(out["fm"])
    extras["view_valid"] = t(out["view_valid"]).bool()
    ent = dict(A=np.asarray(out["anchor"][0], np.float32),
               X_R=np.asarray(x0[0].numpy(), np.float32))
    with torch.no_grad():
        for name, (ref, gamma) in stack["refs"].items():
            res = ref(**ref_inputs(extras, "cpu"))
            if gamma is None:
                xf = res.vertices.double()[0]
            else:
                xf = (x0 + gamma * (res.vertices.double() - x0))[0]
            ent[name] = np.asarray(xf.numpy(), np.float32)
    return ent


def _hg(i):
    import cv2, torch
    from mvhand.runtime.unified import uniform_geometry
    from mvhand.models.interfaces import Form2FrameBatch
    # prior outputs and crops only; identical for every stack
    fwd = _W["u1" if "u1" in _W else "u2"]["fwd"]
    vals = {}
    with np.load(fwd.files[i], allow_pickle=False) as z:
        for n in ("joints_2d_normalized", "local_joints_3d",
                  "local_vertices_3d"):
            vals[n] = torch.as_tensor(z[n][0][None]).float()
    affine = fwd.inputs["raw_to_crop_s25"][[i]]
    inverse = np.stack([[cv2.invertAffineTransform(a) for a in row]
                        for row in affine])
    t = torch.as_tensor
    mask = fwd.inputs["runtime_view_mask"][i]
    batch = Form2FrameBatch(torch.zeros((1, mask.shape[0], 3, 1, 1)),
                            t(fwd.inputs["intrinsics"][[i]]),
                            t(fwd.inputs["camera_to_world"][[i]]),
                            t(inverse), t(mask[None]), t(affine))
    v, ok = uniform_geometry(joints_2d_norm=vals["joints_2d_normalized"],
                             local_joints=vals["local_joints_3d"],
                             local_vertices=vals["local_vertices_3d"],
                             batch=batch, geometry_blend=.5)
    return None if not bool(ok[0]) else np.asarray(v[0].numpy(), np.float32)


def _work(ids):
    rows_out = []
    for i in ids:
        ent = {}
        bad = None
        for name in [n for n in ("u1", "u2", "u2f") if n in _W]:
            eps = _endpoints(_W[name], i)
            if eps is None:
                bad = name + "_placement_fail"
                break
            pref = "" if name == "u1" else name + "_"
            for arm, X in eps.items():
                ent[pref + arm] = X
        if bad:
            rows_out.append((i, None, bad))
            continue
        hg = _hg(i)
        if hg is not None:
            ent["HG"] = hg
        rows_out.append((i, ent, "ok" if hg is not None else "ok_hg_fail"))
    return rows_out


def binding_row(rows, gt, q21):
    per = {name: [] for name, _ in HIST.values()}
    for i, r in enumerate(rows):
        f = REL / "records" / (hashlib.sha256(r["key"].encode()).hexdigest()
                               + ".npz")
        with np.load(f, allow_pickle=False) as z:
            for src, (name, _) in HIST.items():
                per[name].append(M2.group_metrics(
                    z[src].astype(np.float64), gt[i].astype(np.float64),
                    q21)["abs_v"])
    out = {}
    for src, (name, hist) in HIST.items():
        m = float(np.mean(per[name]))
        out[name] = dict(abs_v_mean_mm=round(m, 4),
                         historical_mm=round(hist, 4),
                         delta_mm=round(m - hist, 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in (PREP / "ROWS.jsonl").read_text().splitlines()]
    predictable = [i for i, r in enumerate(rows) if r["predictable"]]
    n = len(predictable) if not a.limit else a.limit
    todo = predictable[:n]
    OUT.mkdir(parents=True, exist_ok=True)
    pred_f = OUT / ("PRED_FINAL.npz" if not a.limit
                    else "PRED_FINAL_SMOKE{}.npz".format(n))
    if not pred_f.exists():
        items = [todo[s_:s_ + 4] for s_ in range(0, len(todo), 4)]
        acc, fails = {}, {}
        for i, r in enumerate(rows):
            if not r["predictable"]:
                fails[i] = "DEGENERATE_KEYPOINT_BOX_REFERENCE_VIEW"
        t0 = time.time()
        with Pool(30, initializer=_init) as pool:
            for ci, rows_out in enumerate(pool.imap_unordered(_work, items)):
                for i, ent, status in rows_out:
                    if ent is None:
                        fails[i] = status
                    else:
                        acc[i] = ent
                        if status != "ok":
                            fails[i] = status
                if (ci + 1) % 100 == 0:
                    print(json.dumps(dict(chunks=ci + 1, of=len(items),
                                          sec=round(time.time() - t0, 1))),
                          flush=True)
        pop = sorted(i for i, e in acc.items() if "HG" in e)
        arms = sorted(acc[pop[0]])
        np.savez_compressed(
            pred_f, pop=np.array(pop),
            keys=np.array([rows[i]["key"] for i in pop]),
            arms=np.array(arms), unit=np.array("metres_f32"),
            fails=np.array(json.dumps({str(k): v for k, v in fails.items()})),
            **{"V_" + arm: np.stack([acc[i][arm] for i in pop])
               for arm in arms})
        print("FINAL4_ENDPOINTS_EXPORTED", len(pop), flush=True)
        print("PRED_SHA256",
              hashlib.sha256(pred_f.read_bytes()).hexdigest(), flush=True)
    if a.limit:
        print("SMOKE_DONE_NO_TARGETS_READ", flush=True)
        return
    z = np.load(pred_f, allow_pickle=False)
    pop = [int(p) for p in z["pop"]]
    arms = [str(x) for x in z["arms"]]
    seqs = [rows[i]["sequence_id"] for i in pop]
    q21, shas = M2.load_q21()
    # ONE targets read: binding row + scoring
    gt_all = np.load(PREP / "PRIVATE_SCORE_TARGETS.npz",
                     allow_pickle=False)["vertices_reference_m"].astype(np.float64)
    if (REL / "records").is_dir():
        binding = binding_row(rows, gt_all, q21)
        (OUT / "BINDING_ROW.json").write_text(json.dumps(
            dict(binding_row=binding, n=len(rows),
                 source="frozen v2 release ho3d-l0-predictions-v2 vs rebuilt POEM targets"),
            indent=1))
        print("BINDING", json.dumps(
            {k: v["delta_mm"] for k, v in binding.items()}), flush=True)
    else:
        binding = None
        print("BINDING skipped: legacy ho3d-l0-predictions-v2 record "
              "store not present (internal historical cross-check only; "
              "no effect on HG/u2 scoring)", flush=True)
    boot = M2.SeqBoot(seqs)
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    long_rows, summ = [], []
    for arm in arms:
        V = z["V_" + arm]
        per = {m: np.zeros(len(pop)) for m in M2.METRIC_KEYS}
        for gi in range(len(pop)):
            row = M2.group_metrics(V[gi].astype(np.float64), gt_all[pop[gi]],
                                   q21)
            for m in M2.METRIC_KEYS:
                per[m][gi] = row[m]
        for m in M2.METRIC_KEYS:
            agg = M2.aggregate(per[m], seqs, boot.sidx, boot.useq)
            ci90 = boot.ci90(per[m])
            long_rows.append(dict(
                dataset="ho3d_mv_poemtest", experiment="FINAL4",
                row="poem_terminal_test", method=arm, source_datasets="ho3d",
                model_role=("DEV_ROLE" if arm.startswith("u2_") else
                            "FINAL_MERGED_U2" if arm.startswith("u2f")
                            else "registered_U1"),
                stage1_seed=str(SEED), r_seed=str(SEED), ref_seed=str(SEED),
                replication_scope="registered", view_k="native", view_seed="",
                view_manifest_sha="", split="POEM_TEST_5view_2706",
                metric=m, alignment="pa" if "pa" in m else "raw",
                aggregation="group_mean", value=round(agg["group_mean"], 6),
                unit=("ratio" if "auc" in m or m.startswith("f") else "mm"),
                n_expected=len(rows), n_scored=len(pop),
                coverage=round(len(pop) / len(rows), 4),
                ci_low=round(ci90[0], 6), ci_high=round(ci90[1], 6),
                run_id="STAGE6_FINAL_V4_HO3D_POEM", ckpt_sha="see_plan",
                eval_key="metrics2@" + code_sha,
                input_provenance_key="v7_ho3d_poemtest2706_v1",
                code_sha=code_sha, status="SUCCEEDED", reason=""))
            if m == "abs_v":
                med = float(np.median(per[m]))
                p95 = float(np.percentile(per[m], 95))
                summ.append((arm, round(agg["group_mean"], 4), round(med, 4),
                             round(p95, 4), round(ci90[0], 4),
                             round(ci90[1], 4)))
    with open(OUT / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0]))
        w.writeheader(); w.writerows(long_rows)
    lines = ["| arm | abs_v mean mm | median | p95 | ci90 |",
             "|---|---|---|---|---|"] + \
            ["| {} | {} | {} | {} | [{}, {}] |".format(*s) for s in summ]
    (OUT / "table.md").write_text(
        "# STAGE6_FINAL_V4 - POEM HO3D-MV terminal test "
        "(frozen 2706-group benchmark, POEM labels, metrics v2)\n\n"
        + "\n".join(lines) + "\n")
    cov = {s: dict(constructed=sum(1 for r in rows if r["sequence_id"] == s),
                   predictable=sum(1 for r in rows
                                   if r["sequence_id"] == s and r["predictable"]),
                   scored=sum(1 for q in seqs if q == s))
           for s in sorted(set(r["sequence_id"] for r in rows))}
    (OUT / "STATUS.json").write_text(json.dumps(dict(
        written_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        batch="STAGE6_FINAL_V4_HO3D_POEM", identity="V6_HO3D_COMPLETION",
        n_expected=len(rows), n_scored=len(pop),
        coverage=round(len(pop) / len(rows), 4), by_sequence=cov,
        fails=json.loads(str(z["fails"])), smoke=False,
        binding_row_deltas=({k: v["delta_mm"] for k, v in binding.items()}
                            if binding else "skipped (store absent)"),
        reader_shas=shas), indent=1))
    (OUT / "ACCESS_LEDGER.json").write_text(json.dumps(dict(
        poem_targets_reads={"single_session": 1,
                            "contents": "binding row + scoring in one read"},
        batch="STAGE6_FINAL_V4_HO3D_POEM",
        at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        targets_file=str(PREP / "PRIVATE_SCORE_TARGETS.npz")), indent=1))
    print("FINAL4_BATCH_DONE", json.dumps({s[0]: s[1] for s in summ}),
          flush=True)


if __name__ == "__main__":
    main()
