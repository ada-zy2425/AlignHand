"""AlignHand U2 minimal inference demo: one multi-view group in, final
mesh (.obj) and 21 joints (.json) out. Reads NO ground truth.

Inputs
  --spec     JSON: {"images": [V image paths], "intrinsics": [V,3,3],
             "camera_to_world": [V,4,4], "boxes": [V,[cx,cy,side]] hand
             boxes in raw pixels (side = unexpanded box side; the script
             applies the registered 2.5x expansion to a 256x256 crop),
             "view_mask": [V] optional bools}
  --hamer-config  queue-style prior config JSON (prior_source,
             prior_checkpoint(+sha256), hamer_assets) for the frozen
             HaMeR release
  --stage1 / --r / --ref / --stats   the four U2_SHARED files from the
             private HF repo yy-megadepth/alignhand-u2
  --region-basis  region_mask_b_778x6.npy (HF repo)
  --mano     licensed MANO_RIGHT.pkl (joints regressor + faces; never
             distributed)
usage: infer.py --spec g.json --hamer-config cfg.json
       --stage1 s.pt --r R_FINAL.pt --ref REF_FINAL.pt
       --stats U2_GEOMETRY_STATS.pt --region-basis b.npy
       --mano MANO_RIGHT.pkl --out out_dir [--device cpu]
"""
import argparse
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'lib'))
sys.path.insert(0, os.environ.get(
    'ALIGNHAND_SRC', os.path.join(_HERE, '..', '..', 'src')))


def crop_affine(box, size=256, expansion=2.5):
    cx, cy, side = box
    s = size / (side * expansion)
    return np.array([[s, 0, size / 2 - s * cx],
                     [0, s, size / 2 - s * cy]], np.float64)


def main():
    ap = argparse.ArgumentParser()
    for name in ('spec', 'hamer-config', 'stage1', 'r', 'ref', 'stats',
                 'region-basis', 'mano', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    import cv2
    import torch
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    import s_common as S
    from mvhand.cli.v2_capture_hanco_prior import load_prior
    from mvhand.geometry.mesh import build_poem_published_21_regressor
    from mvhand.geometry.mesh_sampling import coarse_mesh_operators
    from mvhand.models.frozen_prior import FrozenOfficialPrior
    from mvhand.models.hamer_adapter import normalize_rgb_crops
    from mvhand.models.interfaces import Form2FrameBatch
    from mvhand.regional.candidates import CandidateEvidenceHead
    from mvhand.regional.refinement import MultiViewRefinement
    from mvhand.runtime.mano_compat import legacy_mano_imports
    from mvhand.training.cached_stage1 import build_cached_stage1
    from mvhand.training.real import load_mano_base_regressor
    from dataclasses import fields as dc_fields

    spec = json.loads(a.spec.read_text())
    V = len(spec['images'])
    K = np.asarray(spec['intrinsics'], np.float32)
    c2w = np.asarray(spec['camera_to_world'], np.float32)
    mask = np.asarray(spec.get('view_mask', [True] * V), bool)
    affine = np.stack([crop_affine(b) for b in spec['boxes']]) \
        .astype(np.float32)

    # 1) frozen HaMeR capture on the registered 2.5x 256 crops (no GT)
    cfg = json.loads(Path(a.hamer_config).read_text())
    prior = FrozenOfficialPrior(load_prior(cfg, a.device),
                                name=cfg['prior_name'])
    crops = []
    for p, af in zip(spec['images'], affine):
        im = cv2.imread(str(Path(a.spec).parent / p), cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(p)
        w = cv2.warpAffine(im[:, :, ::-1], af, (256, 256),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
        crops.append(w.transpose(2, 0, 1).astype(np.float32) / 255)
    t = lambda x: torch.as_tensor(x, device=a.device)[None]
    inverse = np.stack([cv2.invertAffineTransform(af) for af in affine])
    batch = Form2FrameBatch(t(np.stack(crops)), t(K), t(c2w), t(inverse),
                            t(mask), t(affine))
    with torch.inference_mode():
        cap = prior.capture(normalize_rgb_crops(batch.crops),
                            batch.runtime_view_mask)
    p0 = cap.prior
    values = {f.name: getattr(p0, f.name) for f in dc_fields(p0)}
    values['feature_maps'] = cap.feature_maps

    # 2) U2 forward (Stage I + shared R) via the frozen evaluation path
    fwd = object.__new__(S.FrozenForward)
    ck = torch.load(a.stage1, map_location='cpu', weights_only=False)
    fwd.stage = build_cached_stage1('M1')
    fwd.stage.load_state_dict(ck['trainable_state'], strict=True)
    fwd.stage.eval().requires_grad_(False)
    fwd.basis = torch.from_numpy(np.load(a.region_basis)).double()
    fwd.reg = load_mano_base_regressor(a.mano).double()
    rck = torch.load(a.r, map_location='cpu', weights_only=False)
    fwd.head = CandidateEvidenceHead()
    fwd.head.load_state_dict(rck.get('state_dict', rck), strict=True)
    fwd.head.eval().requires_grad_(False)
    st = torch.load(a.stats, map_location='cpu', weights_only=False)
    fwd.mean, fwd.scale = st['mean'], st['scale']
    fwd.inputs = dict(intrinsics=K[None], camera_to_world=c2w[None],
                      raw_to_crop_s25=affine[None],
                      runtime_view_mask=mask[None])
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / 'group_00000.npz'
        np.savez(f, **{k: np.asarray(v.cpu().numpy())
                       for k, v in values.items()})
        fwd.files = [f]
        fwd.rows = [dict(key='demo')]
        fwd.ds, fwd.split = 'demo', 'demo'
        out = S.FrozenForward.forward(fwd, [0])
    if out is None:
        raise SystemExit('placement failed for this group')

    # 3) shared pixel-aligned refinement (gamma = 1 endpoint)
    with legacy_mano_imports(), a.mano.open('rb') as fh:
        template = pickle.load(fh, encoding='latin1')
    ops = coarse_mesh_operators(
        np.asarray(template['v_template'], np.float64), template['f'],
        195)
    refm = MultiViewRefinement(ops, displacement_bound_m=.005,
                               include_correction_magnitude=True)
    rck = torch.load(a.ref, map_location='cpu', weights_only=False)
    refm.load_state_dict(rck.get('state_dict', rck))
    refm.eval().requires_grad_(False)
    from ref23_train import ref_inputs
    tt = torch.as_tensor
    extras = {k: tt(out[k]) for k in
              ('support_weights', 'reference_to_camera', 'raw_to_crop',
               'intrinsics', 'scaffold', 'correction_mag', 'placed')}
    extras['x0'] = tt(out['x0']).double()
    extras['fm'] = tt(out['fm'])
    extras['view_valid'] = tt(out['view_valid']).bool()
    with torch.no_grad():
        res = refm(**ref_inputs(extras, 'cpu'))
    mesh = np.asarray(res.vertices.double()[0].numpy())

    # 4) outputs: .obj mesh + 21 joints (reference-camera metres)
    a.out.mkdir(parents=True, exist_ok=True)
    faces = np.asarray(template['f'], np.int64)
    with (a.out / 'mesh.obj').open('w') as fh:
        for v in mesh:
            fh.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        for tri in faces + 1:
            fh.write(f'f {tri[0]} {tri[1]} {tri[2]}\n')
    base = load_mano_base_regressor(a.mano).to(torch.float64)
    q21 = build_poem_published_21_regressor(base).numpy() \
        .astype(np.float64)
    joints = (q21 @ mesh)
    (a.out / 'joints21.json').write_text(json.dumps(dict(
        frame='OpenCV reference-camera coordinates, metres',
        joints_21x3=[[round(float(x), 6) for x in j] for j in joints]),
        indent=1))
    print('DEMO_DONE', a.out / 'mesh.obj', a.out / 'joints21.json')


if __name__ == '__main__':
    main()
