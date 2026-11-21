#!/usr/bin/env bash
# Train the shared AlignHand model (U2) end to end on the pooled training
# splits of DexYCB, HanCo and HO3D.
#   1. Stage I observation refinement (16,800 updates, seed 20261023)
#   2. bundles -> pooled geometry statistics -> regional fusion R -> refinement Ref
# Requires the prepared populations and HaMeR caches described in the README.
set -euo pipefail
: "${ALIGNHAND_DATA:?set ALIGNHAND_DATA to your data root}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEED="${SEED:-20261023}"

python -u "$HERE/train_stage1.py" --seed "$SEED"
python -u "$HERE/train_fusion_refinement.py" --step all --seed "$SEED" --workers "${WORKERS:-16}"
