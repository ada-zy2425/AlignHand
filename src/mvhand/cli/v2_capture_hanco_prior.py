"""Nonterminal frozen-prior capture, gated by the 50-group qualification."""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from mvhand.data.v2_access import authorize_population, validate_input_field_names
from mvhand.data.v2_crops import write_frozen_crop_ledger
from mvhand.data.v2_hanco_input import HanCoRGBInputReader
from mvhand.models.frozen_prior import FrozenOfficialPrior
from mvhand.models.hamer_adapter import normalize_rgb_crops
from mvhand.runtime.precision import configure_transfer_runtime
from mvhand.runtime.mano_compat import legacy_mano_imports


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_prior(config, device):
    if digest(config['prior_checkpoint']) != config['prior_checkpoint_sha256']:
        raise ValueError('official prior checkpoint identity mismatch')
    if config['prior_name'] == 'WiLoR':
        from mvhand.runtime.adapters import load_wilor
        return load_wilor(Path(config['prior_source']), Path(config['prior_checkpoint']), device)[0]
    if config['prior_name'] != 'HaMeR':
        raise ValueError('unsupported official prior')
    from mvhand.models.hamer_runtime import Form2FrameRuntimeAssets, RuntimeAsset, load_canonical_hamer
    assets = config['hamer_assets']
    kwargs = {name: RuntimeAsset(Path(value['path']), value['sha256']) for name, value in assets.items()}
    bundle = Form2FrameRuntimeAssets(hamer_source=Path(config['prior_source']), **kwargs)
    model, _ = load_canonical_hamer(bundle)
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    authorize_population(config['split'], phase=1)
    if config['split'] != 'development':
        raise PermissionError('qualification is limited to the frozen Development panel')
    if digest(config['population_manifest']) != config['population_manifest_sha256']:
        raise ValueError('population manifest identity mismatch')
    rows = [json.loads(line) for line in Path(config['population_manifest']).read_text().splitlines() if line.strip()]
    if len(rows) != 810 or any(r['split'] not in ('validation', 'development') for r in rows):
        raise PermissionError('qualification manifest is not the 810-group Development population')
    limit = int(config['groups'])
    if limit not in (50, 810):
        raise ValueError('only the registered 50-group panel or complete Development is allowed')
    if limit == 810:
        qualifications = []
        for item in config['qualification_reports']:
            if digest(item['path']) != item['sha256']:
                raise ValueError('qualification report identity mismatch')
            report = json.loads(Path(item['path']).read_text())
            if (report['status'] != 'PASS' or report['groups'] != 50 or report['views'] != 400
                    or report['projection_mean_crop_px'] > 0.5 or report['Final_accesses'] != 0):
                raise ValueError('50-group qualification did not pass')
            qualifications.append(report)
        if ({r['prior'] for r in qualifications} != {'HaMeR', 'WiLoR'}
                or len({r['crop_ledger_sha256'] for r in qualifications}) != 1):
            raise ValueError('both priors must qualify with identical frozen crops')
    selected = rows[:limit]
    if config['crop_policy'] != 'verified_hanco_small_full_image_v2':
        raise ValueError('qualification crop policy was not frozen')
    output = Path(config['output']); output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    runtime = configure_transfer_runtime()
    started = time.monotonic()
    print(json.dumps({'stage': 'loading_frozen_prior', 'prior': config['prior_name']}), flush=True)
    with legacy_mano_imports():
        model = FrozenOfficialPrior(load_prior(config, args.device), name=config['prior_name'])
    print(json.dumps({'stage': 'prior_loaded', 'prior': config['prior_name']}), flush=True)
    group_results = []; crop_records = []
    with HanCoRGBInputReader(rgb_archive=Path(config['rgb_archive']),
            calibration_archive=Path(config['calibration_archive']), split='development') as reader:
        with torch.inference_mode():
            for index, row in enumerate(selected):
                group = reader.read(row['key'], verified_small_centered_release=True)
                batch = replace(group.batch, **{f.name: (None if getattr(group.batch, f.name) is None else
                    getattr(group.batch, f.name).to(args.device)) for f in fields(group.batch)})
                capture = model.capture(normalize_rgb_crops(batch.crops), batch.runtime_view_mask)
                tensors = {f.name: getattr(capture.prior, f.name) for f in fields(capture.prior)}
                tensors.update(feature_maps=capture.feature_maps, camera_translation=capture.camera_translation,
                    focal_length=capture.focal_length, projection_error_crop_px=capture.projection_error_crop_px,
                    intrinsics=batch.intrinsics, camera_to_world=batch.camera_to_world,
                    raw_to_crop=batch.raw_to_crop, crop_to_raw=batch.crop_to_raw,
                    runtime_view_mask=batch.runtime_view_mask)
                validate_input_field_names(tensors)
                file = output / f'group_{index:05d}.npz'
                np.savez(file, **{k: v.detach().cpu().numpy() for k, v in tensors.items()})
                errors = capture.projection_error_crop_px[capture.available]
                record = dict(index=index, key=row['key'], views=int(capture.available.sum()),
                    projection_mean_crop_px=float(errors.mean()), projection_max_crop_px=float(errors.max()),
                    cache_file=file.name, cache_sha256=digest(file))
                group_results.append(record); crop_records.extend(group.crop_records)
                print(json.dumps({k: v for k, v in record.items() if k not in ('key', 'cache_sha256')}), flush=True)
    crop_sha = write_frozen_crop_ledger(output/'CROP_LEDGER.json', crop_records,
        policy_identity={'id': config['crop_policy'], 'GT_used': False, 'cached_once': True})
    elapsed = time.monotonic() - started
    result = dict(status='PASS', prior=config['prior_name'], population='HanCo Development',
        groups=len(group_results), views=sum(r['views'] for r in group_results),
        source_commit=config['source_commit'], crop_ledger_sha256=crop_sha,
        prior_checkpoint_sha256=config['prior_checkpoint_sha256'], records=group_results,
        feature_layout='native_B_V_C_16_12', feature_crop_bounds=[32, 0, 192, 256],
        projection_mean_crop_px=float(np.mean([r['projection_mean_crop_px'] for r in group_results])),
        runtime=runtime, elapsed_seconds=elapsed, GPU_hours=elapsed/3600 if args.device.startswith('cuda') else 0,
        GT_reads=0, Final_accesses=0, pid=os.getpid(), GPU_UUID=os.environ.get('CUDA_VISIBLE_DEVICES'))
    (output/'REPORT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}), flush=True)


if __name__ == '__main__':
    main()
