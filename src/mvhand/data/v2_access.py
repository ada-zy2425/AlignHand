"""Fail before opening any image/cache when a population is not authorized."""
from __future__ import annotations

from collections.abc import Mapping


NONFINAL_SPLITS = frozenset(('fit', 'screen', 'development'))
FINAL_SPLITS = frozenset(('final', 'terminal', 'test', 'confirmation'))


def authorize_population(split: str, *, phase: int, final_registration: Mapping | None = None,
                          task_id: str | None = None) -> None:
    if split in NONFINAL_SPLITS:
        return
    if split not in FINAL_SPLITS:
        raise PermissionError('unknown population label; refusing data access')
    if phase != 7 or final_registration is None:
        raise PermissionError('Final images, detection and caches are sealed until phase 7 freeze')
    if (final_registration.get('status') != 'FROZEN'
            or final_registration.get('final_window_ordinal') != 3
            or task_id not in final_registration.get('task_ids', ())):
        raise PermissionError('Final task is not explicitly registered for the third window')
    for key in ('commit', 'config_sha256', 'checkpoint_manifest_sha256'):
        value = final_registration.get(key, '')
        length = 40 if key == 'commit' else 64
        if len(value) != length or any(x not in '0123456789abcdef' for x in value):
            raise PermissionError('Final registration has no frozen source/config/checkpoint identity')


def validate_input_field_names(fields) -> None:
    allowed = {'key', 'dataset', 'split', 'camera_ids', 'image_sha256', 'crop_ledger_sha256',
               'prior_name', 'prior_checkpoint_sha256', 'feature_maps', 'joints_2d_normalized',
               'visual_features', 'local_joints_3d', 'local_vertices_3d',
               'camera_translation', 'focal_length', 'intrinsics', 'camera_to_world',
               'raw_to_crop', 'crop_to_raw', 'runtime_view_mask', 'feature_crop_bounds',
               'projection_error_crop_px', 'predicted_mano_global_orient',
               'predicted_mano_hand_pose', 'predicted_mano_betas'}
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError('unregistered fields in inference-only cache; targets must be stored separately')
