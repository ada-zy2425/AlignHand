"""Common native-feature cache interface for official HaMeR and WiLoR."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mvhand.interfaces import HandPriorOutput


@dataclass(frozen=True)
class FrozenPriorCapture:
    prior: HandPriorOutput
    feature_maps: Tensor
    camera_translation: Tensor
    focal_length: Tensor
    projection_error_crop_px: Tensor
    available: Tensor
    predicted_mano: dict[str, Tensor] | None = None


def sample_stage1_features(feature_maps: Tensor, joints_2d_normalized: Tensor) -> Tensor:
    """Retain the released L0 Stage I sampling convention for both priors."""
    b, v, channels, height, width = feature_maps.shape
    grid = (2.0 * joints_2d_normalized).reshape(b * v, -1, 1, 2)
    values = F.grid_sample(feature_maps.reshape(b * v, channels, height, width),
                           grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return values.squeeze(-1).transpose(1, 2).reshape(b, v, -1, channels)


def verify_projected_joints(joints_local: Tensor, camera_translation: Tensor,
                            focal_length: Tensor, cached_2d: Tensor, *, patch_size=256) -> Tensor:
    """Independent perspective projection of the official MANO joint output."""
    camera = joints_local + camera_translation[..., None, :]
    if torch.any(camera[..., 2] <= 0):
        raise ValueError('official prior produced non-positive projected depth')
    normalized = camera[..., :2] / camera[..., 2:] * focal_length[..., None, :] / patch_size
    return torch.linalg.vector_norm((normalized - cached_2d) * patch_size, dim=-1)


class FrozenOfficialPrior(nn.Module):
    def __init__(self, model: nn.Module, *, name: str):
        super().__init__()
        if name not in ('HaMeR', 'WiLoR'):
            raise ValueError('v2 uses exactly one official prior')
        self.name = name
        self.model = model.eval().requires_grad_(False)
        self.feature_crop_bounds = (32, 0, 192, 256)

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, normalized_crops: Tensor) -> HandPriorOutput:
        return self.capture(normalized_crops).prior

    @torch.no_grad()
    def capture(self, normalized_crops: Tensor, available: Tensor | None = None) -> FrozenPriorCapture:
        if normalized_crops.ndim != 5 or normalized_crops.shape[-3:] != (3, 256, 256):
            raise ValueError('official prior expects [B,V,3,256,256] normalized RGB')
        b, views = normalized_crops.shape[:2]
        if available is None:
            available = torch.ones((b, views), dtype=torch.bool, device=normalized_crops.device)
        if available.shape != (b, views) or available.dtype != torch.bool:
            raise ValueError('availability must be Boolean [B,V]')
        if not available.any():
            raise ValueError('no RGB available; record output failure without executing the prior')
        mask = available.flatten()
        images = normalized_crops.reshape(-1, 3, 256, 256)[mask]
        dense = []
        def capture_map(_module, _inputs, output):
            if isinstance(output, Tensor):
                value = output
            elif isinstance(output, (tuple, list)):
                values = [x for x in output if isinstance(x, Tensor) and x.ndim == 4]
                if len(values) != 1:
                    raise ValueError('official backbone must expose one native dense map')
                value = values[0]
            else:
                raise TypeError('unsupported official backbone output')
            if value.ndim != 4 or value.shape[-2:] != (16, 12):
                raise ValueError('native prior feature footprint must be [C,16,12]')
            dense.append(value)
        hook = self.model.backbone.register_forward_hook(capture_map)
        try:
            raw = self.model({'img': images})
        finally:
            hook.remove()
        if len(dense) != 1:
            raise ValueError('one backbone execution is required per prior capture')
        required = ('pred_keypoints_2d', 'pred_keypoints_3d', 'pred_vertices',
                    'pred_cam_t', 'focal_length')
        if any(not isinstance(raw.get(key), Tensor) for key in required):
            raise ValueError('official prior output is incomplete')
        error = verify_projected_joints(raw['pred_keypoints_3d'], raw['pred_cam_t'],
                                        raw['focal_length'], raw['pred_keypoints_2d'])
        if not torch.isfinite(error).all() or error.mean().item() > 0.5:
            raise ValueError('prior MANO-to-2D projection disagreement exceeds 0.5 px')
        def scatter(value):
            if not torch.isfinite(value).all():
                raise ValueError('nonfinite official prior output')
            out = value.new_zeros((b * views, *value.shape[1:]))
            out[mask] = value
            return out.reshape(b, views, *value.shape[1:])
        maps = scatter(dense[0])
        joints_2d = scatter(raw['pred_keypoints_2d'])
        prior = HandPriorOutput(joints_2d, sample_stage1_features(maps, joints_2d),
                                 scatter(raw['pred_keypoints_3d']), scatter(raw['pred_vertices']))
        mano = raw.get('pred_mano_params')
        predicted_mano = None
        if mano is not None:
            expected = {'global_orient': (1,3,3), 'hand_pose': (15,3,3), 'betas': (10,)}
            if set(mano) != set(expected):
                raise ValueError('unregistered official predicted MANO fields')
            predicted_mano = {name: scatter(mano[name].reshape(len(images),*shape))
                              for name,shape in expected.items()}
        return FrozenPriorCapture(prior, maps, scatter(raw['pred_cam_t']),
                                   scatter(raw['focal_length']), scatter(error), available, predicted_mano)
