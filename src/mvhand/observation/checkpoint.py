"""Load the small learned Stage I asset on top of the frozen official prior."""
from pathlib import Path
import torch


MODULES = ('evidence', 'correction_head', 'weight_head')


def load_stage1(model, path, *, dataset, prior_sha256, parent_sha256, camera_order):
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    identity = payload['identity']
    if (identity.get('schema') != 'form2frame.stage1-weights.v1'
            or identity.get('route') != 'L0' or identity.get('dataset') != dataset
            or identity.get('official_hamer_checkpoint_sha256') != prior_sha256
            or identity.get('source_parent_checkpoint_sha256') != parent_sha256):
        raise ValueError('Stage I dataset, frozen prior or parent identity mismatch')
    states = payload['modules']
    if set(states) != set(MODULES):
        raise ValueError('Stage I asset must contain exactly evidence, correction and support heads')
    # All state dictionaries are validated before changing any module.
    for name in MODULES:
        expected = getattr(model, name).state_dict()
        if set(expected) != set(states[name]):
            raise ValueError('Incomplete or unexpected Stage I state: ' + name)
        for key, value in states[name].items():
            if value.shape != expected[key].shape or not torch.isfinite(value).all():
                raise ValueError('Invalid Stage I tensor: ' + name + '.' + key)
    model.evidence.bind_camera_order(camera_order, identity['training_camera_order'])
    for name in MODULES:
        getattr(model, name).load_state_dict(states[name], strict=True)
    model.view_weight_mode = 'full'
    model.eval().requires_grad_(False)
    return identity
