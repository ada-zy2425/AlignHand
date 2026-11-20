"""Stage I training from frozen prior outputs, with separate supervision."""
from dataclasses import fields
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from mvhand.data.v2_access import authorize_population
from mvhand.models.interfaces import Form2FrameBatch, HandPriorOutput
from mvhand.models.v2_evidence import V2Form2FrameModel
from mvhand.training.objectives import crop_supported_supervision_mask, form2frame_training_loss
from mvhand.training.view_augmentation import (
    sample_training_views, select_batch_views, select_prior_views, select_view_tensor,
)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def build_cached_stage1(variant, feature_dim=1280):
    # The official backbone is already frozen in the immutable prior cache.
    return V2Form2FrameModel(None, feature_dim=feature_dim, variant=variant,
        evidence_dim=128, attention_heads=4, dropout=0.0, correction_limit=0.05,
        weight_correction_scale=0.25, residual_scale_px=10.,
        residual_temperature=0.5, patch_size=256)


class CachedStage1Population:
    """No dense-map/RGB read is needed to train the small observation heads.

    ``model_batch`` returns only calibrated inputs and frozen prior predictions.
    Targets are exposed through a separate method used by the loss/evaluator.
    """
    def __init__(self, manifest, expected_sha256):
        self.path = Path(manifest)
        if sha256(self.path) != expected_sha256:
            raise ValueError('Stage I population manifest SHA mismatch')
        spec = self.spec = json.loads(self.path.read_text())
        if spec['schema'] != 'form2frame.v2.stage1-cached-population.v1':
            raise ValueError('unsupported Stage I population schema')
        authorize_population(spec['split'], phase=2)
        if spec['split'] not in ('fit', 'development') or spec['status'] != 'COMPLETE':
            raise PermissionError('only complete Fit/Development caches are allowed')
        prepared = Path(spec['prepared'])
        report = json.loads((prepared/'REPORT.json').read_text())
        if (sha256(prepared/'REPORT.json') != spec['prepared_report_sha256']
                or report['dataset'] != spec['dataset'] or report['split'] != spec['split']):
            raise ValueError('prepared population identity mismatch')
        for file, key in [('INPUTS.npz', 'inputs_sha256'), ('ROWS.jsonl', 'rows_sha256')]:
            if sha256(prepared/file) != report[key]:
                raise ValueError('prepared input identity mismatch')
        self.inputs = dict(np.load(prepared/'INPUTS.npz', allow_pickle=False))
        if set(self.inputs) != {'intrinsics', 'camera_to_world', 'runtime_view_mask',
                'raw_to_crop_s15', 'raw_to_crop_s20', 'raw_to_crop_s25'}:
            raise ValueError('unexpected model-input cache fields')
        self.rows = [json.loads(line) for line in (prepared/'ROWS.jsonl').read_text().splitlines()]
        self.entries = spec['rows']
        if (len(self.rows) != len(self.entries) or len(self.rows) != report['groups']
                or any(row['split'] != spec['split'] for row in self.rows)):
            raise ValueError('population count/split mismatch')
        if any(entry['index'] != i or entry['key'] != self.rows[i]['key']
               for i, entry in enumerate(self.entries)):
            raise ValueError('prior cache and prepared rows are not aligned')
        self.affines = self.inputs[f"raw_to_crop_s{int(spec['expansion'] * 10)}"]
        self.inverse = np.stack([[cv2.invertAffineTransform(a) for a in row] for row in self.affines])
        self.verified = set()
        self._targets = None

    def __len__(self):
        return len(self.rows)

    def model_batch(self, indices, device):
        indices = list(map(int, indices))
        values = {field.name: [] for field in fields(HandPriorOutput)}
        for index in indices:
            entry = self.entries[index]
            file = Path(entry['file'])
            if index not in self.verified:
                if sha256(file) != entry['sha256']:
                    raise ValueError('frozen prior cache SHA mismatch')
                self.verified.add(index)
            with np.load(file, allow_pickle=False) as source:
                for name in values:
                    values[name].append(source[name][0])
        tensor = lambda x: torch.as_tensor(x, device=device)
        mask = tensor(self.inputs['runtime_view_mask'][indices])
        # forward_from_prior consumes no image pixels. A tiny finite placeholder
        # carries batch/view shape only; the frozen visual features are the input.
        batch = Form2FrameBatch(torch.zeros((*mask.shape, 3, 1, 1), device=device),
            tensor(self.inputs['intrinsics'][indices]), tensor(self.inputs['camera_to_world'][indices]),
            tensor(self.inverse[indices]), mask, tensor(self.affines[indices]))
        prior = HandPriorOutput(**{name: tensor(np.stack(value)).float() for name, value in values.items()})
        return batch, prior

    def supervision(self, indices, device):
        if self._targets is None:
            prepared = Path(self.spec['prepared'])
            report = json.loads((prepared/'REPORT.json').read_text())
            target_file = prepared/'PRIVATE_SCORE_TARGETS.npz'
            if sha256(target_file) != report['targets_private_sha256']:
                raise ValueError('separate supervision identity mismatch')
            directory = Path(self.spec['supervision'])
            if sha256(directory/'REPORT.json') != self.spec['supervision_report_sha256']:
                raise ValueError('Stage I supervision manifest SHA mismatch')
            manifest = json.loads((directory/'REPORT.json').read_text())
            if manifest['rows_sha256'] != report['rows_sha256']:
                raise ValueError('supervision belongs to a different population order')
            for name, expected in manifest['files'].items():
                if sha256(directory/name) != expected:
                    raise ValueError('Stage I supervision SHA mismatch')
            with np.load(target_file, allow_pickle=False) as source:
                self._targets = {name: source[name] for name in ('vertices_reference_m', 'joints_2d_raw')}
            self._targets['joints_reference_m'] = np.load(directory/'dataset_joints_reference_m.npy', allow_pickle=False)
            self._targets['mask'] = np.load(directory/'joint_2d_supervision_mask.npy', allow_pickle=False)
        indices = list(map(int, indices))
        return {name: torch.as_tensor(value[indices], device=device) for name, value in self._targets.items()}


def cached_training_loss(model, batch, prior, targets, *, view_generator,
                         view_distribution='uniform'):
    """Apply linked M2 augmentation, then the unchanged Stage I objective."""
    points, mask = targets['joints_2d_raw'], targets['mask']
    if model.variant in ('M2', 'M3'):
        selection = sample_training_views(batch.runtime_view_mask, generator=view_generator,
                                           distribution=view_distribution)
        batch, prior = select_batch_views(batch, selection), select_prior_views(prior, selection)
        points, mask = select_view_tensor(points, selection), select_view_tensor(mask, selection)
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), -1)
    crop = torch.einsum('bvrc,bvjc->bvjr', batch.raw_to_crop, homogeneous)
    mask = crop_supported_supervision_mask(crop, mask & batch.runtime_view_mask[..., None])
    prediction = model.forward_from_prior(prior, batch)
    loss, parts = form2frame_training_loss(prediction,
        gt_joints_3d=targets['joints_reference_m'].float(), raw_to_crop=batch.raw_to_crop,
        gt_joints_2d_crop=crop, supervision_mask=mask,
        view_availability_mask=batch.runtime_view_mask)
    return loss, parts
