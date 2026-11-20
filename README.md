# AlignHand: Multi-View Hand Reconstruction from a Frozen Monocular Prior

[**Project page**](https://ada-zy2425.github.io/AlignHand/) ·
[**Paper**](docs/static/AlignHand.pdf) ·
[**Checkpoints**](https://github.com/ada-zy2425/AlignHand/releases/tag/v1.0)

*From independent monocular predictions to one metric hand.*

AlignHand aligns and enhances meshes from a frozen monocular estimator (HaMeR)
using calibrated multi-view observations:

1. **Observation refinement**: joint-level tokens from every view attend to each
   other under a geometry-aware bias, predicting bounded 2D joint corrections and
   per-joint view support.
2. **Metric alignment**: support-weighted triangulation gives a shared metric
   skeleton; each per-view mesh is placed on it by a similarity fit, and a
   fit-aware anchor is selected.
3. **Surface enhancement**: anchor-centred regional fusion re-weights the aligned
   candidates for the palm and each finger, then pixel-aligned refinement adds
   bounded residuals (at most 5 mm per coordinate).

Frozen monocular backbone · 0.98M trainable integration parameters · one shared
integration model, trained on the pooled training splits of DexYCB, HanCo and HO3D.

## Results

MPVPE / MPJPE in mm of the shared model (lower is better).

| Dataset | Views | HG | AlignHand | MPVPE reduction |
|---|---|---|---|---|
| DexYCB-MV | 8 | 5.997 / 5.892 | **4.193 / 4.090** | 30.1% |
| HanCo | 8 | 7.166 / 6.643 | **5.083 / 4.587** | 29.1% |
| HO3D-Eval-MV | 5 | 11.119 / 10.875 | **10.233 / 9.882** | 8.0% |

HG (HaMeR + geometry) uses the same frozen HaMeR predictions, crops, calibration
and similarity fitting. Each dataset follows its own evaluation protocol; HanCo
uses root-translation alignment. Comparisons with published multi-view methods
are in Table 1 of the paper.

## Repository layout

```
src/mvhand/     implementation: models, geometry, regional fusion, refinement, training, evaluation;
                src/mvhand/cli/ holds the prior-capture and cache tools
scripts/
  demo/         run AlignHand on one multi-view group
  train/        train the shared model
  eval/         every result in the paper (table below)
  lib/          helpers shared by the scripts above
checkpoints/    MANIFEST.json (sha256) + download.py
configs/        HaMeR config example, training configuration, recorded environment
splits/         group identifiers of every split (IDs only)
docs/           project page
```

## Installation

```bash
git clone https://github.com/ada-zy2425/AlignHand.git && cd AlignHand
pip install -e ".[data]"
```

The frozen prior needs the official [HaMeR](https://github.com/geopavlakos/hamer)
code and checkpoint, and [MANO](https://mano.is.tue.mpg.de/) (`MANO_RIGHT.pkl`).
[DATA.md](DATA.md) lists every external asset, where to get it and its expected sha256.

## Checkpoints

The shared model is 7.9 MB in total (five files, sha256 in
[`checkpoints/MANIFEST.json`](checkpoints/MANIFEST.json)):

```bash
python checkpoints/download.py --root data   # downloads the v1.0 release and verifies sha256
```

## Demo

```bash
python scripts/demo/make_mano_assets.py --mano MANO_RIGHT.pkl --out data/handoff/models/assets
python scripts/demo/infer.py \
  --spec group.json --hamer-config configs/hamer_prior.example.json \
  --stage1 data/results/u2/stage1_s20261023/CHECKPOINT_UPDATE_16800.pt \
  --r data/results/u2/r_s20261023/R_FINAL.pt --ref data/results/u2/ref_s20261023/REF_FINAL.pt \
  --stats data/results/u2/U2_GEOMETRY_STATS.pt \
  --region-basis data/handoff/models/assets/region_mask_b_778x6.npy \
  --mano MANO_RIGHT.pkl --out outputs/ --device cuda
```

`group.json` describes the synchronized views of one hand: `images` (V paths),
`intrinsics` (V×3×3), `camera_to_world` (V×4×4) and `boxes` (V × `[cx, cy, side]`
in pixels; the 2.5× crop expansion is applied by the script). The demo writes the
final mesh (`.obj`) and 21 joints (`.json`) in the reference camera frame. Fill in
the sha256 placeholders of the HaMeR config example with `sha256sum` of your files.

## Reproducing the paper

All scripts read data and write results under `$ALIGNHAND_DATA` (see [DATA.md](DATA.md)).

| Paper | Script |
|---|---|
| Table 1, DexYCB-MV | `scripts/eval/test_dexycb.py` |
| Table 1, HanCo | `scripts/eval/test_hanco.py run` |
| Table 1, HO3D-Eval-MV | `scripts/eval/test_ho3d_eval_mv.py` |
| Table 1, HO3D-MV | `scripts/eval/test_ho3d_mv.py` |
| Table 2, stage-wise and interventions | `scripts/eval/dev_analysis.py --task stagewise` / `--task interventions` |
| Fig. 5 and Table S9, number of views | `scripts/eval/dev_analysis.py --task views` |
| Occlusion | `scripts/eval/make_occlusion_caches.py`, then `dev_analysis.py --task occlusion` |
| Runtime | `scripts/eval/runtime.py --dataset dexycb` (or `ho3d`) |
| Training | `scripts/train/run_all.sh` |

`dev_analysis.py` takes `--dataset dexycb` or `--dataset ho3d`. With the released
checkpoints, the development stage-wise MPVPE (mm) is reproduced to within 0.0004 mm:

| Stage | DexYCB dev | HO3D dev |
|---|---|---|
| HG | 5.567 | 4.161 |
| Aligned anchor | 3.724 | 3.532 |
| + Regional fusion | 3.545 | 3.449 |
| + Pixel-aligned refinement | 3.409 | 3.562 |

The test scripts also evaluate two internal comparison models when their files
are present; with only the released checkpoints they evaluate HG and the shared
model, reported under the `u2_` prefix.

## Citation

```bibtex
@misc{yang2026alignhand,
  title  = {AlignHand: Multi-View Hand Reconstruction from a Frozen Monocular Prior},
  author = {Yang, Zhilu and Yang, Kun and Wang, Shiyan and Xu, Anqi and Huang, Xinran
            and Jiao, Yizheng and Yang, Huan and Potamias, Rolandos Alexandros},
  note   = {arXiv preprint},
  year   = {2026}
}
```

## License and acknowledgements

Code is released under the MIT License. HaMeR, MANO, DexYCB, HanCo and HO3D are
subject to their own licences. We thank the authors of
[HaMeR](https://github.com/geopavlakos/hamer) and of the datasets.
