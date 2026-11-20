"""Generate the MANO-derived fixed arrays used by AlignHand U2 from a
user-provided, licensed MANO_RIGHT.pkl (none of these are distributed):
  - mano_base_joint_regressor_16x778.npy  (MANO J_regressor, 16x778)
  - ref_upsample_195_to_778.npy           (coarse 195-vertex operators)
usage: make_mano_assets.py --mano MANO_RIGHT.pkl --out assets_dir
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get(
    'ALIGNHAND_SRC', os.path.join(os.path.dirname(os.path.abspath(
        __file__)), '..', '..', 'src')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mano', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    from mvhand.geometry.mesh_sampling import coarse_mesh_operators
    from mvhand.runtime.mano_compat import legacy_mano_imports
    from mvhand.training.real import load_mano_base_regressor
    a.out.mkdir(parents=True, exist_ok=True)
    reg = load_mano_base_regressor(a.mano).numpy().astype(np.float32)
    np.save(a.out / 'mano_base_joint_regressor_16x778.npy', reg)
    with legacy_mano_imports(), a.mano.open('rb') as f:
        template = pickle.load(f, encoding='latin1')
    ops = coarse_mesh_operators(
        np.asarray(template['v_template'], dtype=np.float64),
        template['f'], 195)
    np.save(a.out / 'ref_upsample_195_to_778.npy', ops['upsample'])
    print('wrote', sorted(p.name for p in a.out.iterdir()))


if __name__ == '__main__':
    main()
