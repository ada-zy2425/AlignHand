# Public check — summary (2026-11-20)

- Base: `main` @ `fd2d8bcb47ea84132dd1b950dfb92e4fdbe2c341`. Final
  tested code SHA on `public-check`: `94cc0f5875c7…` (6 minimal fixes,
  each regression-tested; this file is the only later commit).
- Environment: fresh clone, isolated Python 3.10.19 venv, exact
  `configs/requirements-lock.txt` pins (torch 2.7.1+cu128, numpy
  1.26.4). Install notes: chumpy 0.70 needs `--no-build-isolation`;
  the lock set must be installed with `--no-deps` (pyrender 0.1.45
  declares PyOpenGL 3.1.0 while the lock pins 3.1.9); non-headless
  opencv needs system `libgl1`/`libglib2.0-0`. All evaluation ran on
  CPU.
- Commands passed: `pytest -o addopts="" -q` (107 passed, 0 skipped);
  `dev_analysis.py --dataset {dexycb,ho3d}`; the four test scripts
  (`--limit 8` smokes, then full); `demo/infer.py` (CPU);
  `train_stage1.py --smoke 20 --device cpu`;
  `train_fusion_refinement.py --help`; `bash -n run_all.sh`;
  `checkpoints/download.py` (5/5 verified from a hand-downloaded dir).
- Dev stagewise: all 8 MPVPE cells within **0.0004 mm** of the paper
  table (gate 0.002), denominators 1,412 / 3,904, group IDs identical
  to the registered run; an equivalence rerun with the full internal
  inputs present gives per-group arrays identical to the public-root
  run (max diff 0.0).
- Test sets (one-time full evaluation with the release weights): all
  16 MPVPE/MPJPE cells across DexYCB-MV, HanCo (UST root-translation),
  HO3D-Eval-MV and HO3D-MV reproduce the paper values within
  **0.0005 mm**; group counts exact (4,951 / 20,198 / 3,178 / 2,706
  constructed with the 4 recorded degenerate exclusions); arms exactly
  {HG, u2_A, u2_X_R, u2_Full} ({HG, u2_A, u2_Full} for HanCo); weights
  strict-loaded.
- Demo: implementation-equivalent to the original release demo to
  0.000000 mm; vs the historical GPU-cache endpoint 0.098 mm mean /
  0.228 mm max (expected capture difference, report-only).
- Release: draft `v1.0` at the tested SHA with the 5 MANIFEST files
  (name/size/sha256 verified after upload). Anonymous
  `checkpoints/download.py` currently 404s (repo private, release
  draft): PUBLIC_DOWNLOAD_PENDING.
- Known limitations: R/Ref training verified at entry level only;
  occlusion/view-curve/timing/intervention extras are out of scope of
  the public repo; the legacy HO3D binding-row cross-check is skipped
  without the internal store.

CORE_EXECUTION: PASS
DEV_STAGEWISE: PASS
TEST_TABLE1: PASS
PUBLIC_DISTRIBUTION: PUBLIC_DOWNLOAD_PENDING (draft release on private repo; flips on publish)
