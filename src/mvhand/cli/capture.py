#!/usr/bin/env python3
"""Capture preregistered nonterminal RGB into immutable, resumable chunks."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from mvhand.runtime.adapters import _dex_groups, load_assignments, load_wilor
from mvhand.io import atomic_json, sha256, tensor_sha256, module_source_hashes
from mvhand.evaluation.metrics import numpy_metrics
from mvhand.runtime.precision import configure_transfer_runtime
from mvhand.runtime.unified import build_unified_route, qualified_official_baselines, capture_values, storage_array, canonicalize_group, iter_unified_groups


def main():
    p=argparse.ArgumentParser()
    for arg in ('config','output','preregistration','dispatch','restore-receipt'):
        p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--shard-count',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    started=time.monotonic()
    if torch.cuda.device_count()!=1: raise ValueError('one UUID assigned GPU required')
    config=json.loads(a.config.read_text())
    if config['dataset'] not in ('dexycb','ho3d','hanco') or config['route'] not in ('Z','L0'): raise ValueError('route not yet qualified')
    if not 0<=a.shard_index<a.shard_count: raise ValueError('shard')
    if json.loads(a.restore_receipt.read_text())['status']!='PASS': raise ValueError('asset restore not verified')
    assets=config['assets']
    for name in ('hamer_checkpoint','hamer_config','mano_right','mano_mean_params','region_basis','dataset_contract','wilor_checkpoint'):
        if sha256(Path(assets[name]))!=config['asset_sha256'][name]: raise ValueError('asset '+name)
    assignments=load_assignments(Path(config['split_assignments']))
    counts={s:sum(r['split']==s for r in assignments.values()) for s in ('fit','screen','development')}
    expected=config.get('population_counts',{'fit':20329,'screen':5058,'development':1412})
    if counts!=expected or len(assignments)!=sum(expected.values()): raise ValueError(counts)
    if sha256(Path(config['split_assignments']))!=config['split_assignments_sha256']:raise ValueError('split assignment identity')
    keys=sorted(assignments)[a.shard_index::a.shard_count]
    if a.limit: keys=keys[:a.limit]
    if not keys: raise ValueError('empty capture')
    runtime=configure_transfer_runtime()
    torch.manual_seed(20261021)
    np.random.seed(20261021)
    official_cache=qualified_official_baselines(config)
    identity={'route':config['route'],'dataset':config['dataset'],'population_counts':expected,'parent_policy':'unified_anchor_v1','output_mode':'anchor_centered',
        'config_sha256':sha256(a.config),'assignment_sha256':sha256(Path(config['split_assignments'])),
        'preregistration_sha256':sha256(a.preregistration),'dispatch_sha256':sha256(a.dispatch),
        'restore_receipt_sha256':sha256(a.restore_receipt),'runtime':runtime,'asset_sha256':config['asset_sha256'],
        'source_commit':config['source_commit'],'source_sha256':module_source_hashes('mvhand.cli.capture','mvhand.runtime.unified'),
        'view_weight_mode':'geometry_only' if config['route']=='Z' else 'full','reference_physical_camera':0,'camera_order':config['camera_serials'],
        'keys_sha256':hashlib.sha256('\n'.join(keys).encode()).hexdigest(),'terminal_accesses':0}
    if official_cache is not None:
        identity['official_baseline_manifest_sha256']=sha256(official_cache.manifest_path)
        identity['parent_binding_config_sha256']=config['parent_binding_config_sha256']
    a.output.mkdir(parents=True,exist_ok=True)
    ip=a.output/'IDENTITY.json'
    if ip.exists() and json.loads(ip.read_text())!=identity: raise ValueError('resume identity mismatch')
    atomic_json(ip,identity)
    inference=build_unified_route(config,'cuda:0')
    if hasattr(inference,'unified_parent_binding'):atomic_json(a.output/'PARENT_BINDING.json',inference.unified_parent_binding)
    wilor,_=load_wilor(Path(assets['wilor_source']),Path(assets['wilor_checkpoint']),'cuda:0')
    basis=torch.from_numpy(np.load(assets['region_basis'])['basis']).to('cuda:0',torch.float64)
    reg=inference.mano_base_regressor.to('cuda:0',torch.float32)
    np.save(a.output/'regressor.npy',reg.cpu().numpy())
    completed=0
    for start in range(0,len(keys),64):
        selected=keys[start:start+64]
        folder=a.output/f'chunk_{start//64:05d}'
        receipt=folder/'COMPLETE.json'
        if receipt.exists():
            done=json.loads(receipt.read_text())
            if [r['key'] for r in done['rows']]!=selected: raise ValueError('chunk UID mismatch')
            for name,digest in done['arrays'].items():
                if sha256(folder/name)!=digest: raise ValueError('corrupt resumed chunk')
            completed+=len(selected)
            continue
        folder.mkdir(exist_ok=True)
        maps={}; rows=[]
        with torch.inference_mode():
            for i,group in enumerate(iter_unified_groups(config,assignments,selected)):
                if group.key!=selected[i]: raise ValueError('reader UID mismatch')
                group=canonicalize_group(group,config['camera_serials'])
                values,valid,_=capture_values(inference,wilor,group,basis,reg,'cuda:0',official_cache=official_cache)
                if not valid or not torch.isfinite(values['target']).all():
                    atomic_json(folder/'FAILURE.json',{'key':group.key,'valid':valid,'omitted':False})
                    raise ValueError('unsupported invalid case; retain UID and stop qualification')
                if not torch.equal(values['parent_vertices'],values['anchor']): raise ValueError('B != A')
                row={**assignments[group.key],'key':group.key,'ordinal':i,'target_uid':tensor_sha256(values['target'][0]),'valid':valid,'target_reexpression':group.label['canonical_target_diagnostics'],'metrics':{}}
                for baseline,name in (('B','parent_vertices'),('HG','hg_vertices'),('WG','wg_vertices')):
                    v,j=numpy_metrics(values[name][0].cpu().numpy(),values['target'][0].cpu().numpy(),reg.cpu().numpy())
                    if not np.isfinite([v,j]).all(): raise ValueError('baseline invalid '+baseline)
                    row['metrics'][baseline]={'mpvpe_mm':v,'mesh_mpjpe_mm':j}
                for name,value in values.items():
                    array=storage_array(name,value)
                    if name not in maps:
                        maps[name]=np.lib.format.open_memmap(folder/(name+'.npy'),mode='w+',dtype=array.dtype,shape=(len(selected),*array.shape[1:]))
                    maps[name][i]=array[0]
                rows.append(row)
                elapsed=time.monotonic()-started
                if elapsed>float(json.loads(a.dispatch.read_text())['capture_estimate_gpu_hours'])*2*3600/a.shard_count:
                    raise RuntimeError('capture GPU cost guard; author decision required')
                atomic_json(a.output/'PROGRESS.json',{'done':completed+i+1,'total':len(keys),'elapsed_seconds':elapsed,'key':group.key,
                    'pid':os.getpid(),'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'metrics':row['metrics']})
                print(json.dumps({'done':completed+i+1,'total':len(keys),'key':group.key,'metrics':row['metrics']}),flush=True)
        for array in maps.values(): array.flush()
        del maps
        hashes={p.name:sha256(p) for p in folder.glob('*.npy')}
        atomic_json(receipt,{'identity_sha256':sha256(ip),'rows':rows,'arrays':hashes})
        completed+=len(rows)
    atomic_json(a.output/'COMPLETION.json',{'status':'COMPLETE','groups':completed,'elapsed_seconds':time.monotonic()-started,
        'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'pid':os.getpid(),'terminal_accesses':0})


if __name__=='__main__': main()
