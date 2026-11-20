#!/usr/bin/env python3
"""Qualify complete unified captures and fit geometry statistics on Fit only."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from mvhand.io import atomic_json, atomic_torch, sha256
from mvhand.runtime.unified import UnifiedCache


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--shards',type=int,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--prior-capture-gpu-seconds',type=float,default=0)
    a=p.parse_args();rows=[];shards=[];ident=None;gpu_seconds=a.prior_capture_gpu_seconds
    for index in range(a.shards):
        folder=a.root/f'shard_{index:03d}'
        complete=json.loads((folder/'COMPLETION.json').read_text())
        if complete['status']!='COMPLETE':raise ValueError('capture incomplete')
        gpu_seconds+=complete['elapsed_seconds']
        current=json.loads((folder/'IDENTITY.json').read_text());current.pop('keys_sha256')
        if ident is None:ident=current
        if current!=ident:raise ValueError('mixed cache identities')
        chunk_groups=0
        for cp in sorted(folder.glob('chunk_*/COMPLETE.json')):
            ck=json.loads(cp.read_text())
            if ck['identity_sha256']!=sha256(folder/'IDENTITY.json'):raise ValueError('chunk identity')
            for name,digest in ck['arrays'].items():
                if sha256(cp.parent/name)!=digest:raise ValueError('corrupt array '+str(cp.parent/name))
            sid=len(shards)
            shards.append({'path':str(cp.parent.relative_to(a.root)),'arrays':ck['arrays'],'receipt_sha256':sha256(cp)})
            rows.extend({**r,'shard':sid} for r in ck['rows']);chunk_groups+=len(ck['rows'])
        if chunk_groups!=complete['groups']:raise ValueError('missing chunk')
    if len({r['key'] for r in rows})!=len(rows):raise ValueError('duplicate UID')
    expected=ident.get('population_counts',{'fit':20329,'screen':5058,'development':1412})
    if not a.smoke and len(rows)!=sum(expected.values()):raise ValueError('population mismatch')
    counts={s:sum(r['split']==s for r in rows) for s in ('fit','screen','development')}
    if not a.smoke and counts!=expected:raise ValueError(counts)
    manifest={'schema':'form2frame.unified-route-cache.v1','dataset':ident['dataset'],'identity':ident,'shards':shards,'rows':sorted(rows,key=lambda r:r['key'])}
    atomic_json(a.root/'manifest.json',manifest)
    cache=UnifiedCache(a.root/'manifest.json')
    total=np.zeros(104);squares=np.zeros(104);count=np.zeros(104,dtype=np.int64)
    fit=cache.keys('fit')
    for start in range(0,len(fit),128):
        keys=fit[start:start+128];x=cache.tensor('geometry_hw',keys,'cpu',torch.float64).numpy()
        valid=cache.tensor('geometry_valid_hw',keys,'cpu').numpy().astype(bool)
        if valid.shape!=x.shape:valid=np.broadcast_to(valid,x.shape)
        if not np.isfinite(x[valid]).all():raise ValueError('nonfinite valid geometry')
        x=np.where(valid,x,0);axes=tuple(range(x.ndim-1))
        total+=x.sum(axis=axes);squares+=(x*x).sum(axis=axes);count+=valid.sum(axis=axes)
    if (count==0).any():raise ValueError('unsupported feature dimension')
    mean=total/count;scale=np.sqrt(np.maximum(0,squares/count-mean**2)).clip(1e-8)
    atomic_torch(a.root/'GEOMETRY_STATS.pt',{'schema':'form2frame.unified-fit-stats.v1','split':'fit','fit_groups':len(fit),
        'cache_manifest_sha256':sha256(a.root/'manifest.json'),'mean':torch.from_numpy(mean),'scale':torch.from_numpy(scale),'counts':torch.from_numpy(count)})
    baseline={}
    dev=[r for r in rows if r['split']=='development']
    if a.smoke and not dev: dev=[r for r in rows if r['split']=='fit']
    for arm in ('HG','WG','B'):
        baseline[arm]={}
        for metric in ('mpvpe_mm','mesh_mpjpe_mm'):
            values=np.array([r['metrics'][arm][metric] for r in dev]);seqs=sorted({r['sequence_id'] for r in dev})
            baseline[arm][metric]={'group_mean':float(values.mean()),'sequence_equal':float(np.mean([np.mean([r['metrics'][arm][metric] for r in dev if r['sequence_id']==s]) for s in seqs]))}
    atomic_json(a.root/'QUALIFICATION.json',{'status':'PASS','formal_population':not a.smoke,'population':counts,'fit_statistics_groups':len(fit),
        'development_baselines':baseline,'capture_gpu_hours':gpu_seconds/3600,'terminal_accesses':0,
        'manifest_sha256':sha256(a.root/'manifest.json'),'geometry_stats_sha256':sha256(a.root/'GEOMETRY_STATS.pt')})
    print(json.dumps(baseline),flush=True)

if __name__=='__main__':main()
