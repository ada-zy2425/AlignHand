"""Construct fixed Ref sampling operators from the licensed MANO template."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
from mvhand.geometry.mesh_sampling import coarse_mesh_operators
from mvhand.runtime.mano_compat import legacy_mano_imports
from mvhand.training.cached_stage1 import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mano-right',type=Path,required=True);p.add_argument('--mano-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--vertices',type=int,choices=(195,389),default=195)
    a=p.parse_args()
    if sha256(a.mano_right)!=a.mano_sha256:raise ValueError('template asset identity mismatch')
    with legacy_mano_imports(),a.mano_right.open('rb') as f:template=pickle.load(f,encoding='latin1')
    vertices=np.asarray(template['v_template'],dtype=np.float64)
    operators=coarse_mesh_operators(vertices,template['f'],a.vertices)
    a.output.mkdir(parents=True,exist_ok=False)
    path=a.output/'MESH_OPERATORS.npz';np.savez(path,**operators)
    report={'status':'COMPLETE','source':'licensed MANO model template only; no dataset frames',
        'template_sha256':a.mano_sha256,'coarse_vertices':a.vertices,'full_vertices':len(vertices),
        'downsampling':'deterministic farthest-point selection, vertex 0 first, index tie-break',
        'upsampling':'nonnegative normalized inverse-distance interpolation from three coarse neighbours; exact at coarse vertices',
        'laplacian':'fixed row-normalized template mesh adjacency',
        'asset_sha256':sha256(path),'contains_template_coordinates':False,
        'GT_coordinate_reads':0,'GPU_hours':0,'Final_accesses':0}
    (a.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
