"""Capture one registered Fit shard from immutable offline crop inputs."""
import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from mvhand.cli.v2_capture_hanco_prior import digest,load_prior
from mvhand.data.v2_access import authorize_population,validate_input_field_names
from mvhand.data.v21_prepared_rgb import PreparedDevelopmentRGB
from mvhand.data.prior_storage import prior_storage_arrays
from mvhand.models.frozen_prior import FrozenOfficialPrior
from mvhand.models.hamer_adapter import normalize_rgb_crops
from mvhand.models.interfaces import Form2FrameBatch
from mvhand.runtime.mano_compat import legacy_mano_imports
from mvhand.runtime.precision import configure_transfer_runtime


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    run(json.loads(a.config.read_text()))


def run(config, *, hanco_a3=False):
    authorize_population(config['split'],phase=1)
    expected={'dexycb':20329,'ho3d':4056,'hanco':73371}
    if hanco_a3:
        if config['dataset']!='hanco' or config['split'] not in ('fit','development'):
            raise PermissionError('A3 capture only accepts HanCo Fit/Development')
        expected={'hanco':75198 if config['split']=='fit' else 7084}
    elif config['split']!='fit' or config['dataset'] not in expected:
        raise PermissionError('only preregistered Fit cache preparation is permitted')
    prepared=Path(config['prepared']);report=json.loads((prepared/'REPORT.json').read_text())
    if (digest(prepared/'REPORT.json')!=config['prepared_report_sha256'] or report['status']!='PASS'
            or report['groups']!=expected[config['dataset']] or report['split']!=config['split']):
        raise ValueError('Fit preparation identity mismatch')
    for file,key in [('INPUTS.npz','inputs_sha256'),('ROWS.jsonl','rows_sha256')]:
        if digest(prepared/file)!=report[key]:raise ValueError('Fit input identity mismatch')
    selection=Path(config['S0_scale_selection'])
    if digest(selection)!=config['S0_scale_selection_sha256']:
        raise ValueError('S0 scale selection identity mismatch')
    selected=json.loads(selection.read_text())
    if selected['status']!='COMPLETE' or selected['completed_arms']!=18:
        raise ValueError('all eighteen S0 arms must finish before Fit capture')
    expansion=float(selected['selected_expansion'][config['prior_name']])
    if expansion!=float(config['expansion']) or expansion not in (1.5,2.,2.5):
        raise ValueError('Fit crop differs from the selected per-prior expansion')
    inputs=dict(np.load(prepared/'INPUTS.npz',allow_pickle=False))
    if set(inputs)!={'intrinsics','camera_to_world','runtime_view_mask','raw_to_crop_s15','raw_to_crop_s20','raw_to_crop_s25'}:
        raise ValueError('unexpected inference-input fields')
    rows=[json.loads(s) for s in (prepared/'ROWS.jsonl').read_text().splitlines()]
    if len(rows)!=expected[config['dataset']] or any(r['split']!=config['split'] for r in rows):
        raise ValueError('Fit shard population mismatch')
    if hanco_a3:
        from mvhand.data.a3_capture import selected_indices
        indices=selected_indices(config,report,rows)
        start,end=min(indices),max(indices)+1
    else:
        start,end=int(config['start']),int(config['end'])
        if not 0<=start<end<=len(rows) or end-start>4096:raise ValueError('Fit shard population mismatch')
        indices=list(range(start,end))
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(20261021)
    runtime=configure_transfer_runtime();started=time.monotonic()
    print(json.dumps({'stage':'loading_prior','prior':config['prior_name'],'start':start,'end':end}),flush=True)
    with legacy_mano_imports():model=FrozenOfficialPrior(load_prior(config,'cuda'),name=config['prior_name'])
    affine_all=inputs[f'raw_to_crop_s{int(expansion*10)}'];records=[];projection_sum=0.
    with PreparedDevelopmentRGB(config) as reader,torch.inference_mode():
        for position,index in enumerate(indices):
            row=rows[index];images=reader.read(row);affine=affine_all[index]
            rgb=[cv2.warpAffine(im[:,:,::-1],aff,(256,256),flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT).transpose(2,0,1).astype(np.float32)/255 for (im,_),aff in zip(images,affine)]
            tensor=lambda x:torch.as_tensor(x,device='cuda')[None]
            batch=Form2FrameBatch(tensor(np.stack(rgb)),tensor(inputs['intrinsics'][index]),
                tensor(inputs['camera_to_world'][index]),tensor(np.stack([cv2.invertAffineTransform(aff) for aff in affine])),
                tensor(inputs['runtime_view_mask'][index]),tensor(affine))
            capture=model.capture(normalize_rgb_crops(batch.crops),batch.runtime_view_mask)
            values={f.name:getattr(capture.prior,f.name) for f in fields(capture.prior)}
            values.update(feature_maps=capture.feature_maps,camera_translation=capture.camera_translation,
                focal_length=capture.focal_length,projection_error_crop_px=capture.projection_error_crop_px)
            if config.get('capture_predicted_mano',False):
                if capture.predicted_mano is None:
                    raise ValueError('official prior did not expose its predicted MANO parameters')
                values.update({'predicted_mano_'+k:v for k,v in capture.predicted_mano.items()})
            validate_input_field_names(values)
            file=out/f'group_{index:05d}.npz'
            np.savez(file,**prior_storage_arrays(values,feature_dtype=config.get('feature_storage_dtype','float32')))
            projection_sum+=float(capture.projection_error_crop_px.mean())
            records.append(dict(index=index,key=row['key'],file=file.name,sha256=digest(file),image_sha256=[sha for _,sha in images]))
            if position%50==0:print(json.dumps({'completed':position+1,'total':len(indices)}),flush=True)
    (out/'CACHE_INDEX.json').write_text(json.dumps(records)+'\n');elapsed=time.monotonic()-started
    result=dict(status='QUALIFICATION_PASS' if hanco_a3 and config.get('qualification') else 'COMPLETE',
        dataset=config['dataset'],split=config['split'],prior=config['prior_name'],
        feature_storage_dtype=config.get('feature_storage_dtype','float32'),geometry_storage_dtype='float32',
        predicted_mano_cached=config.get('capture_predicted_mano',False),
        expansion=expansion,start=start,end=end,groups=len(indices),source_commit=config['source_commit'],
        prepared_report_sha256=config['prepared_report_sha256'],S0_scale_selection_sha256=config['S0_scale_selection_sha256'],
        prior_checkpoint_sha256=config['prior_checkpoint_sha256'],cache_index_sha256=digest(out/'CACHE_INDEX.json'),
        mean_projection_error_px=projection_sum/len(indices),runtime=runtime,elapsed_seconds=elapsed,GPU_hours=elapsed/3600,
        GT_coordinate_reads=0,annotation_use='previously exported offline crop affines only; score targets not opened',
        Final_accesses=0,pid=os.getpid(),GPU_UUID=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if hanco_a3:result.update(population='hanco_a3',selector_sha256=config['selector_sha256'])
    (out/'REPORT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
