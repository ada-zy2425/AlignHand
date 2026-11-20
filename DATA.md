# Data and third-party assets

None of the datasets, the HaMeR weights or the MANO model can be redistributed.
Obtain them from the official sources below and accept their licences.

| Asset | Used for | Official source |
|---|---|---|
| DexYCB | training, development, DexYCB-MV test | https://dex-ycb.github.io/ |
| HanCo | training, development, HanCo test (UST-Hand protocol) | https://lmb.informatik.uni-freiburg.de/projects/contrastive_hands/ |
| HO3D v3 | training, development, HO3D-Eval-MV / HO3D-MV tests | https://github.com/shreyashampali/ho3d |
| HaMeR (code + `hamer.ckpt`) | frozen monocular prior | https://github.com/geopavlakos/hamer |
| MANO (`MANO_RIGHT.pkl`) | mesh topology, joint regressor | https://mano.is.tue.mpg.de/ |

Expected identities (verify with `sha256sum`):

- `hamer.ckpt`: `e5cc06f294d88a92dee24e603480aab04de532b49f0e08200804ee7d90e16f53`
- `MANO_RIGHT.pkl`: `45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767`

The software environment used for all paper numbers is recorded in
[`configs/environment.json`](configs/environment.json) and [`configs/requirements-lock.txt`](configs/requirements-lock.txt) (Python 3.10, PyTorch 2.7.1, CUDA 12.8).

## Splits

[`splits/`](splits/) lists every training, development and
test group by identifier only (group key, sequence id, frame id); see its
README for counts and the HO3D-MV exclusions. No annotations, coordinates or
errors are included.

## MANO-derived arrays

The 16x778 base joint regressor and the 195-vertex coarse operators are rebuilt
from your own `MANO_RIGHT.pkl`:

```bash
python scripts/demo/make_mano_assets.py --mano MANO_RIGHT.pkl \
    --out $ALIGNHAND_DATA/handoff/models/assets
```

## Data root used by the reproduction scripts

All scripts read from `$ALIGNHAND_DATA` (default `data/`). The relevant parts:

```
$ALIGNHAND_DATA/
  MANO_RIGHT.pkl
  handoff/models/assets/      region basis + MANO-derived arrays
  handoff/models/priors/      hamer.ckpt
  prepared/                   per-dataset prepared populations (images, calibration, boxes)
  caches/{dexycb,hanco,ho3d}/ cached frozen-HaMeR outputs per group
  results/u2/                 shared-model checkpoints (checkpoints/download.py)
```

The prepared populations and HaMeR caches are generated from the raw datasets
with the tools in `src/mvhand/cli/` (capture and cache finalisation) and the
population manifests in `scripts/train/manifests/`. Because they
contain dataset content, they are not distributed.
