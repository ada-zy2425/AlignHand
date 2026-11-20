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
src/mvhand/            implementation (models, geometry, regional fusion, refinement, training, evaluation)
checkpoints/           MANIFEST.json (sha256) + download.py
reproduce/
  scripts/train/       pooled Stage I training and the R / Ref chain
  scripts/eval/        test-set evaluators, development stage-wise / views / occlusion, timing
  scripts/demo/        single-group inference demo, MANO asset generator
  scripts/common/      shared helpers and the scorer
  configs/             as-used configuration locks of the shared model
  env/                 recorded software environment
  splits/              group identifiers of every split (IDs only)
docs/                  project page (GitHub Pages)
```

## Installation

```bash
git clone https://github.com/ada-zy2425/AlignHand.git && cd AlignHand
pip install -e ".[data]"
```

The frozen prior needs the official [HaMeR](https://github.com/geopavlakos/hamer)
code and checkpoint, and [MANO](https://mano.is.tue.mpg.de/) (`MANO_RIGHT.pkl`).
See [DATA.md](DATA.md) for all external assets and their expected sha256.

## Checkpoints

The shared model is 7.9 MB in total (five files, listed with sha256 in
[`checkpoints/MANIFEST.json`](checkpoints/MANIFEST.json)):

```bash
python checkpoints/download.py --root data   # downloads from the v1.0 release and verifies sha256
```

## Demo: one multi-view group

```bash
python reproduce/scripts/demo/gen_mano_assets.py --mano MANO_RIGHT.pkl --out data/handoff/models/assets
python reproduce/scripts/demo/infer_one_group.py \
  --spec group.json --hamer-config reproduce/configs/hamer_prior.example.json \
  --stage1 data/results/u2/stage1_s20261023/CHECKPOINT_UPDATE_16800.pt \
  --r data/results/u2/r_s20261023/R_FINAL.pt --ref data/results/u2/ref_s20261023/REF_FINAL.pt \
  --stats data/results/u2/U2_GEOMETRY_STATS.pt \
  --region-basis data/handoff/models/assets/region_mask_b_778x6.npy \
  --mano MANO_RIGHT.pkl --out outputs/ --device cuda
```

`group.json` holds the synchronized views of one hand:
`images` (V paths), `intrinsics` (V×3×3), `camera_to_world` (V×4×4) and `boxes`
(V × `[cx, cy, side]` in pixels; the 2.5× crop expansion is applied by the script).
The demo writes the final mesh (`.obj`) and 21 joints (`.json`) in the reference
camera frame. Fill in the sha256 placeholders of the HaMeR config example with
`sha256sum` of your files.

## Reproducing the paper

With the datasets prepared as described in [DATA.md](DATA.md) under
`$ALIGNHAND_DATA`, the development stage-wise table is regenerated with

```bash
cd reproduce/scripts/eval
ALIGNHAND_DATA=/path/to/data python sr3_j3.py --task stagewise --dataset dexycb   # or ho3d
```

A fresh-clone rerun with the released checkpoints matches the paper to within
0.0004 mm in every cell (development MPVPE, mm):

| Stage | DexYCB dev | HO3D dev |
|---|---|---|
| HG | 5.567 | 4.161 |
| Aligned anchor | 3.724 | 3.532 |
| + Regional fusion | 3.545 | 3.449 |
| + Pixel-aligned refinement | 3.409 | 3.562 |

Test-set evaluators (`f_final_eval_dex.py`, `f_final_eval.py` for HanCo,
`f_final_eval_ho3d.py`, `f_final_eval_poem.py`), the view-subset, occlusion and
timing scripts, and the training drivers are in `reproduce/scripts/`;
[`reproduce/RELEASE_NOTES.md`](reproduce/RELEASE_NOTES.md) records how this bundle
was assembled and verified.

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
