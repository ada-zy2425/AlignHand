"""Unified route runtime: official zero-update initialization and common capture.

No legacy regional parent is loaded. Shared geometric readers keep their frozen
numerical conventions; new R outputs are explicitly anchor-centered.
"""
from pathlib import Path
import json
import numpy as np
import torch
from mvhand.runtime.adapters import _batch, dense_tensor
from mvhand.regional.candidates import build_candidate_geometry, metric_joints, project_points, raw_to_hamer_grid
from mvhand.io import sha256
from mvhand.geometry.raw_pixel_dlt import build_projection_matrices, build_reference_to_camera, normalize_relative_weights, triangulate_projective_dlt
from mvhand.geometry.mesh import poem_published_mesh_joints_21_ieee_fp32
from mvhand.geometry.mesh_recovery import recover_positioned_mesh
from mvhand.models.form2frame import normalized_crop_to_raw
from mvhand.models.hamer_adapter import normalize_rgb_crops
from mvhand.models.interfaces import Form2FrameBatch
from mvhand.training.real import CanonicalRuntimePaths, build_canonical_inference
from mvhand.regional.fusion import TransferCache

def build_zero_route(config: dict, device: str):
    assets = config["assets"]
    paths = CanonicalRuntimePaths(
        dataset_root=Path(config["dataset_root"]),
        index=Path(config.get("dexycb_index", config.get("ho3d_cache_root", "."))),
        hamer_source=Path(assets["hamer_source"]),
        hamer_checkpoint=Path(assets["hamer_checkpoint"]),
        hamer_config=Path(assets["hamer_config"]),
        mano_right=Path(assets["mano_right"]),
        mano_mean_params=Path(assets["mano_mean_params"]),
        dataset_contract=Path(assets["dataset_contract"]),
        dataset_contract_sha256=str(assets["dataset_contract_sha256"]),
    )
    inference = build_canonical_inference(paths, device=device)
    inference.model.eval().requires_grad_(False)
    contaminated = []
    for name, value in inference.model.named_parameters():
        if name.endswith(("attn.qkv.q_b", "attn.qkv.v_b")):
            if bool(torch.count_nonzero(value.detach()).cpu()):
                contaminated.append(name)
    if contaminated:
        raise RuntimeError({"nonzero_official_hamer_lora_increment": contaminated})
    for name in ('correction_head', 'weight_head'):
        if any(torch.count_nonzero(p).item() for p in getattr(inference.model,name)[-1].parameters()):
            raise RuntimeError('zero route has nonzero observation head')
    inference.model.view_weight_mode = 'geometry_only'
    return inference


def build_unified_route(config: dict, device: str):
    """Z freezes all upstream components; L0 adds the frozen learned Stage I."""
    if config['route'] not in ('Z', 'L0'):
        raise ValueError('Supported routes are Z and L0')
    inference = build_zero_route(config, device)
    if config['route'] == 'L0':
        from mvhand.stage1_assets import load_stage1
        path = Path(config['assets']['stage1_checkpoint'])
        if sha256(path) != config['asset_sha256']['stage1_checkpoint']:
            raise ValueError('Stage I asset SHA mismatch')
        inference.stage1_identity = load_stage1(inference.model, path,
            dataset=config['dataset'], prior_sha256=config['asset_sha256']['hamer_checkpoint'],
            parent_sha256=config['parent_checkpoint_sha256'], camera_order=config['camera_serials'])
    return inference


def qualified_official_baselines(config):
    if config['route'] not in ('Z', 'L0'):
        raise ValueError('Supported routes are Z and L0; both keep the official backbone frozen')
    return None

def uniform_geometry(
    *, joints_2d_norm: torch.Tensor, local_joints: torch.Tensor,
    local_vertices: torch.Tensor, batch: Form2FrameBatch, geometry_blend: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = batch.runtime_view_mask.bool()
    solver = mask[:, :, None].expand(-1, -1, joints_2d_norm.shape[2])
    raw = normalized_crop_to_raw(joints_2d_norm, batch.crop_to_raw, 256)
    reference_to_camera = build_reference_to_camera(batch.camera_to_world)
    projection = build_projection_matrices(batch.intrinsics, reference_to_camera)
    weights = normalize_relative_weights(
        torch.zeros_like(solver, dtype=raw.dtype), solver,
    ).to(raw.dtype)
    triangulated = triangulate_projective_dlt(raw, projection, weights, solver)
    recovery = recover_positioned_mesh(
        triangulated.points_3d.float(), local_joints.float(),
        local_vertices.float(), weights.float(), triangulated.valid,
        geometry_blend=geometry_blend,
    )
    valid_views = recovery.alignment_valid & mask
    count = valid_views.sum(1, keepdim=True)
    uniform = valid_views.float() / count.clamp_min(1).float()
    vertices = torch.einsum(
        "bv,bvmc->bmc", uniform, recovery.aligned_prior_vertices_3d,
    )
    valid = triangulated.valid.all(-1) & (count[:, 0] > 0)
    vertices = torch.where(
        valid[:, None, None], vertices,
        torch.full((), float("nan"), dtype=vertices.dtype, device=vertices.device),
    )
    return vertices.double(), valid

def capture_input_values(inference, wilor, model_batch, basis, regressor, *,
                         target=None, official_cache=None, group_key=None,
                         compute_baselines=True):
    """Shared image-to-candidate path; inference inputs contain no GT labels.

    Training/evaluation may attach a target after assembling their input batch.
    Targets are only returned for scoring or checked against an existing cache;
    they do not enter any model, geometry or confidence computation.
    """
    device = model_batch.crops.device
    from mvhand.models.form2frame import invert_crop_to_raw
    raw_to_crop = (model_batch.raw_to_crop if model_batch.raw_to_crop is not None
                   else invert_crop_to_raw(model_batch.crop_to_raw)).double()
    captured = []
    hook = inference.model.prior.model.backbone.register_forward_hook(
        lambda _m, _i, value: captured.append(value))
    try:
        result = inference(model_batch)
    finally:
        hook.remove()
    if len(captured) != 1:
        raise RuntimeError("expected exactly one HaMeR dense map")
    feature = dense_tensor(captured[0]).reshape(1,8,1280,12,16)
    parent = result.output.final_vertices_3d.double()
    pred, rec = result.prediction, result.mesh_recovery
    valid = model_batch.runtime_view_mask.bool()
    hv = rec.alignment_valid.bool() & valid
    h = torch.where(hv[...,None,None], rec.aligned_prior_vertices_3d.double(), parent[:,None])
    if wilor is not None:
        wo = wilor({"img": normalize_rgb_crops(model_batch.crops).float().reshape(-1,3,256,256)})
        wl = wo["pred_vertices"].reshape(1,8,778,3).float()
        wj = poem_published_mesh_joints_21_ieee_fp32(wl, regressor.float())
        wr = recover_positioned_mesh(pred.weighted.points_3d.float(), wj, wl,
            pred.relative_view_weights.float(), pred.weighted.valid, geometry_blend=inference.geometry_blend)
        wv = wr.alignment_valid.bool() & valid
        w = torch.where(wv[...,None,None], wr.aligned_prior_vertices_3d.double(), h)
    else:
        # HH-GV duplicates the HaMeR candidates and does not execute WiLoR.
        wr, wv, w = rec, hv, h
    anchor = parent
    candidates = torch.cat((anchor[:,None],h,w),1)
    hr, ha = rec.alignment_residual_m.double(), rec.fusion_view_weights.double()
    pr = torch.cat((torch.zeros_like(hr[:,:1]),hr,torch.where(wv,wr.alignment_residual_m.double(),hr)),1)
    pa = torch.cat((torch.zeros_like(ha[:,:1]),ha,torch.where(wv,wr.fusion_view_weights.double(),ha)),1)
    cv = torch.cat((torch.ones_like(valid[:,:1]),hv,wv),1)
    geometry, gv, raw, pv, diagnostics = build_candidate_geometry(
        candidates=candidates,l88=parent,r83=anchor,placement_residual=pr,
        placement_alpha=pa,candidate_valid=cv,origins=torch.tensor([-1,*range(8),*range(8)],device=device),
        xgeo=pred.weighted.points_3d.double(),observations_raw=pred.corrected_joints_2d_raw.double(),
        projection_matrices=pred.projection_matrices.double(),joint_view_weights=pred.relative_view_weights.double(),
        runtime_view_mask=valid,reference_to_camera=pred.reference_to_camera.double(),basis=basis,regressor=regressor.double())
    grid = raw_to_hamer_grid(raw,raw_to_crop)
    pv &= torch.isfinite(grid).all(-1) & (grid.abs()<=1).all(-1)
    rr,rd = project_points(pred.projection_matrices.double(),metric_joints(parent[:,None],regressor.double()))
    rr = rr[:,0]
    rg = raw_to_hamer_grid(rr[:,None],raw_to_crop)[:,0]
    rv = torch.isfinite(rr).all(-1)&(rd[:,0]>0)&valid[:,:,None]&torch.isfinite(rg).all(-1)&(rg.abs()<=1).all(-1)
    values = dict(parent_vertices=parent,anchor=anchor,h_vertices=h,w_vertices=w,
        view_valid=valid,w_valid=wv,geometry_hw=geometry,geometry_valid_hw=gv,feature_maps=feature,
        candidate_grid_hw=grid,candidate_raw_hw=raw,candidate_projection_valid_hw=pv,
        reference_grid=rg,reference_raw=rr,reference_projection_valid=rv)
    if target is not None:
        values['target'] = target
    if not compute_baselines:
        values.update(support_weights=pred.relative_view_weights, observations_raw=pred.corrected_joints_2d_raw)
        return values, bool(rec.valid.all()), diagnostics
    if wilor is None:
        raise ValueError('Computing WG requires the official WiLoR model')
    if official_cache is None:
        hg, hg_valid = uniform_geometry(joints_2d_norm=pred.prior.joints_2d_normalized.float(),
            local_joints=pred.prior.local_joints_3d.float(), local_vertices=pred.prior.local_vertices_3d.float(),
            batch=model_batch, geometry_blend=inference.geometry_blend)
        wg, wg_valid = uniform_geometry(joints_2d_norm=wo['pred_keypoints_2d'].reshape(1,8,21,2).float(),
            local_joints=wj, local_vertices=wl, batch=model_batch, geometry_blend=inference.geometry_blend)
    else:
        if target is None or not torch.equal(target,official_cache.tensor('target',[group_key],device)):
            raise ValueError('official baseline target changed')
        hg,hg_valid,wg,wg_valid=(official_cache.tensor(n,[group_key],device) for n in ('hg_vertices','hg_valid','wg_vertices','wg_valid'))
    values.update(hg_vertices=hg, wg_vertices=wg, hg_valid=hg_valid, wg_valid=wg_valid,
        support_weights=pred.relative_view_weights, observations_raw=pred.corrected_joints_2d_raw)
    return values, bool(rec.valid.all()), diagnostics


def capture_values(inference, wilor, group, basis, regressor, device, official_cache=None):
    model_batch, _raw_to_crop, target = _batch(group, device)
    return capture_input_values(inference, wilor, model_batch, basis, regressor,
                                target=target, official_cache=official_cache, group_key=group.key)


class UnifiedCache(TransferCache):
    """Explicit new schema; cannot relabel a historical transfer manifest."""
    def __init__(self, path):
        self.manifest_path=Path(path).resolve()
        self.manifest=json.loads(self.manifest_path.read_text())
        if self.manifest.get('schema')!='form2frame.unified-route-cache.v1':
            raise ValueError('unified cache schema required')
        ident=self.manifest['identity']
        if ident['parent_policy']!='unified_anchor_v1' or ident['output_mode']!='anchor_centered':
            raise ValueError('unified route identity')
        self.rows={r['key']:r for r in self.manifest['rows']}
        if len(self.rows)!=len(self.manifest['rows']): raise ValueError('duplicate UID')
        if any(r['split'] not in ('fit','screen','development') for r in self.rows.values()):
            raise ValueError('WP3 terminal access prohibited')
        self.directories=[(self.manifest_path.parent/s['path']).resolve() for s in self.manifest['shards']]
        self.arrays={}


def storage_array(name,value):
    a=value.detach().cpu().numpy()
    if name=='feature_maps': return a.astype(np.float16)
    if name in {'geometry_hw','candidate_grid_hw','candidate_raw_hw','reference_grid','reference_raw'}:
        return a.astype(np.float32)
    return a


def canonicalize_group(group, ordered_camera_ids):
    """Keep the original population's reference target under linked camera sort.

    Publisher per-camera MANO targets differ slightly from rigidly transformed
    copies. Transform the registered source-reference target itself in FP64;
    record the alternate publisher target discrepancy instead of changing GT.
    """
    from dataclasses import replace
    if set(group.camera_ids)!=set(ordered_camera_ids): raise ValueError('physical cameras differ')
    order=[group.camera_ids.index(c) for c in ordered_camera_ids]
    transform=torch.linalg.inv(group.camera_to_world[order[0]].double()) @ group.camera_to_world[0].double()
    inverse=torch.linalg.inv(transform).numpy()
    if hasattr(group,'permute_views'):
        result=group.permute_views(order)
    else:
        index=torch.tensor(order,dtype=torch.long)
        names=('rgb_crops','normalized_crops','raw_to_crop','crop_to_raw','intrinsics','camera_to_world','gt_joint_visibility','runtime_view_mask','raw_size')
        changes={name:getattr(group,name)[index] for name in names}
        changes['camera_ids']=tuple(group.camera_ids[i] for i in order)
        changes['label']={k:(np.asarray(v)[order].copy() if np.asarray(v).ndim and np.asarray(v).shape[0]==8 else v) for k,v in group.label.items()}
        if hasattr(group,'source_provenance'):
            inverse_order={old:new for new,old in enumerate(order)}
            provenance=[dict(r) for r in group.source_provenance]
            for r in provenance:
                if 'runtime_slot' in r:r['runtime_slot']=inverse_order[int(r['runtime_slot'])]
            changes['source_provenance']=tuple(sorted(provenance,key=lambda r:int(r.get('runtime_slot',r.get('physical_camera_id',0)))))
        result=replace(group,**changes)
    label=dict(result.label);diagnostics={}
    for name,count in (('verts_3d',778),('joints_3d',21)):
        original=np.asarray(group.label[name])
        if original.shape not in ((8,count,3),(count,3)): raise ValueError('unsupported target shape')
        first=original[0] if original.ndim==3 else original
        expected=first @ transform[:3,:3].numpy().T + transform[:3,3].numpy()
        roundtrip=expected @ inverse[:3,:3].T + inverse[:3,3]
        if not np.isfinite(roundtrip).all() or np.max(np.abs(roundtrip-first))>1e-6:
            raise ValueError('target reference transform fails algebraic roundtrip')
        if original.ndim==3:
            d=np.asarray(label[name])[0]-expected
            diagnostics[name]={'alternate_publisher_max_m':float(np.abs(d).max()),
                'alternate_publisher_mean_norm_mm':float(np.linalg.norm(d,axis=-1).mean()*1000)}
        label[name]=expected
    label['canonical_target_source']='original population source-reference target, FP64 rigid reexpression'
    label['canonical_target_diagnostics']=diagnostics
    label['canonical_reference_transform']=transform.numpy()
    return replace(result,label=label)


def iter_unified_groups(config, assignments, keys):
    """Only data adapters differ; every route consumes the same canonical group."""
    from mvhand.runtime.adapters import _dex_groups, _ho3d_groups, load_mano_base_regressor
    dataset=config['dataset']
    if dataset=='dexycb':yield from _dex_groups(config,assignments,keys)
    elif dataset=='ho3d':yield from _ho3d_groups(config,keys)
    elif dataset=='hanco':
        from mvhand.data.hanco_runtime import HanCoArchives, HanCoManoMesh, decode_hanco_group
        rows={}
        for name in ('train_manifest','validation_manifest'):
            p=Path(config[name])
            if sha256(p)!=config[name+'_sha256']:raise ValueError('HanCo population manifest identity')
            for line in p.read_text().splitlines():
                row=json.loads(line)
                if row['key'] in rows:raise ValueError('duplicate HanCo population UID')
                rows[row['key']]=row
        mano=HanCoManoMesh(Path(config['assets']['mano_right']))
        reg=load_mano_base_regressor(Path(config['assets']['mano_right']))
        paths={name:Path(path) for name,path in config['hanco_archives'].items()}
        with HanCoArchives(**paths) as archives:
            for key in keys:
                if key not in rows:raise ValueError('HanCo source missing '+key)
                yield decode_hanco_group(rows[key],archives=archives,mano_mesh=mano,mano_base_regressor=reg)
    else:raise ValueError('unknown dataset adapter')
