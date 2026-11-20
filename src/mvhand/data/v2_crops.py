"""Frozen RGB-only localization decisions, shared by priors and experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class CropDecision:
    available: bool
    source: str
    raw_to_crop: tuple
    confidence: float | None

    def to_dict(self):
        return asdict(self)


def frozen_detector_affine(bbox_xyxy, *, expansion: float = 2.0) -> np.ndarray:
    """The frozen HO3D crop rule: 2x box, 192:256 aspect padding, square crop.

    This reproduces expand_to_aspect_ratio(..., [192, 256]).max() before the
    256-square affine; it does not use dataset annotation boxes or coordinates.
    """
    box = np.asarray(bbox_xyxy, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all() or expansion <= 0:
        raise ValueError("invalid RGB detector box")
    wh = box[2:] - box[:2]
    if np.any(wh <= 0):
        raise ValueError("detector box has no area")
    center = (box[:2] + box[2:]) / 2
    side = float(max(wh[1], wh[0] / (192.0 / 256.0)) * expansion)
    scale = 256.0 / side
    return np.asarray([[scale, 0, 128 - scale * center[0]],
                       [0, scale, 128 - scale * center[1]]], dtype=np.float64)


def decide_crop(*, dataset: str, image_size: tuple[int, int], detections: Iterable[dict],
                confidence_threshold: float, hanco_centered_release_verified: bool = False,
                expansion: float = 2.0) -> CropDecision:
    """Input is image dimensions and frozen RGB detections, never annotations.

    Detection rows contain exactly box, score and optional right-hand status.
    An unavailable view carries identity calibration/zero RGB downstream, not a
    guessed crop. The caller runs the detector only when the full-image rule
    does not apply, and commits the decision once to the immutable crop ledger.
    """
    if dataset not in ('dexycb', 'ho3d', 'hanco'):
        raise ValueError("unknown v2 dataset")
    width, height = image_size
    if min(width, height) <= 0 or not 0 <= confidence_threshold <= 1:
        raise ValueError("invalid image dimensions or frozen detector threshold")
    if dataset == 'hanco' and hanco_centered_release_verified and max(width, height) <= 320:
        affine = ((256.0 / width, 0.0, 0.0), (0.0, 256.0 / height, 0.0))
        return CropDecision(True, 'released_small_hanco_full_image', affine, None)
    accepted = []
    for row in detections:
        if set(row) - {'box', 'score', 'right_hand'}:
            raise ValueError("detector row contains unsupported annotation/metadata fields")
        score = float(row['score'])
        if np.isfinite(score) and score >= confidence_threshold and row.get('right_hand', True):
            box = np.asarray(row['box'], dtype=np.float64)
            if box.shape == (4,) and np.isfinite(box).all() and np.all(box[2:] > box[:2]):
                accepted.append((score, tuple(box.tolist())))
    if not accepted:
        return CropDecision(False, 'detector_no_accepted_hand', ((1., 0., 0.), (0., 1., 0.)), None)
    # Deterministic score ranking and box-coordinate tie break, without GT matching.
    score, box = sorted(accepted, key=lambda x: (-x[0], x[1]))[0]
    affine = frozen_detector_affine(box, expansion=expansion)
    return CropDecision(True, 'frozen_rgb_detector', tuple(map(tuple, affine.tolist())), score)


def crop_rgb(image_rgb: np.ndarray, decision: CropDecision) -> np.ndarray:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
        raise ValueError("expected RGB uint8 image")
    if not decision.available:
        return np.zeros((256, 256, 3), dtype=np.uint8)
    return cv2.warpAffine(image_rgb, np.asarray(decision.raw_to_crop, dtype=np.float64),
                          (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def write_frozen_crop_ledger(path: Path, records: list[dict], *, policy_identity: dict) -> str:
    """Write once; callers must create a new version rather than alter a ledger."""
    keys = [(r['dataset'], r['split'], r['key'], r['camera_id']) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate physical-view crop decision")
    payload = json.dumps({'schema': 'form2frame.v2.crops.v1',
                          'policy': policy_identity, 'records': records},
                         sort_keys=True, allow_nan=False, separators=(',', ':')) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(payload)
    return hashlib.sha256(payload.encode()).hexdigest()
