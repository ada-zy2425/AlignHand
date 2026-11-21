"""HanCo RGB/calibration-only reader. No GT archive can be supplied."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import torch

from mvhand.data.access_control import authorize_population
from mvhand.data.crops import CropDecision, crop_rgb, decide_crop
from mvhand.interfaces import Form2FrameBatch


class InputZip:
    def __init__(self, path: Path):
        self.archive = zipfile.ZipFile(path)
        self.prefix = None

    def read(self, member: str) -> bytes:
        if self.prefix is None:
            try:
                self.archive.getinfo(member)
                self.prefix = ''
            except KeyError:
                hits = [n for n in self.archive.namelist() if n.endswith('/' + member)]
                if len(hits) != 1:
                    raise ValueError('image/calibration member identity is not unique')
                self.prefix = hits[0][:-len(member)]
        return self.archive.read(self.prefix + member)

    def close(self):
        self.archive.close()


@dataclass(frozen=True)
class RGBInputGroup:
    key: str
    batch: Form2FrameBatch
    crop_records: tuple[dict, ...]


class HanCoRGBInputReader:
    def __init__(self, *, rgb_archive: Path, calibration_archive: Path,
                 split: str, phase: int = 1):
        authorize_population(split, phase=phase)
        self.split = split
        self.rgb = InputZip(rgb_archive)
        self.calibration = InputZip(calibration_archive)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.rgb.close(); self.calibration.close()

    def read(self, key: str, *, frozen_crops: dict[int, dict] | None = None,
             verified_small_centered_release: bool = False) -> RGBInputGroup:
        parts = key.split('/')
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ValueError('HanCo key must contain only sequence/frame identifiers')
        sequence, frame = parts
        raw_calibration = json.loads(self.calibration.read(f'calib/{sequence}/{frame}.json'))
        intrinsics = np.asarray(raw_calibration['K'], dtype=np.float32)
        extrinsics = np.asarray(raw_calibration['M'], dtype=np.float64)
        if intrinsics.shape != (8, 3, 3) or extrinsics.shape != (8, 4, 4):
            raise ValueError('HanCo calibration contract failed')
        crops, affines, inverse, available, records = [], [], [], [], []
        for camera in range(8):
            data = self.rgb.read(f'rgb/{sequence}/cam{camera}/{frame}.jpg')
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('RGB decoding failed')
            height, width = image.shape[:2]
            digest = hashlib.sha256(data).hexdigest()
            if frozen_crops is None:
                if not verified_small_centered_release or max(width, height) > 320:
                    raise ValueError('detector crop must be prepared and frozen before prior capture')
                decision = decide_crop(dataset='hanco', image_size=(width, height), detections=[],
                    confidence_threshold=0.3, hanco_centered_release_verified=True)
            else:
                frozen = frozen_crops[camera]
                if frozen['image_sha256'] != digest:
                    raise ValueError('RGB changed after crop freeze')
                decision = CropDecision(**frozen['decision'])
            rgb = crop_rgb(image[:, :, ::-1], decision)
            affine = np.asarray(decision.raw_to_crop, dtype=np.float64)
            crops.append(rgb.transpose(2, 0, 1).astype(np.float32) / 255)
            affines.append(affine.astype(np.float32))
            inverse.append(cv2.invertAffineTransform(affine).astype(np.float32))
            available.append(decision.available)
            records.append(dict(dataset='hanco', split=self.split, key=key, camera_id=camera,
                                image_sha256=digest, image_size=[width, height], decision=decision.to_dict()))
        batch = Form2FrameBatch(torch.from_numpy(np.stack(crops))[None],
            torch.from_numpy(intrinsics)[None], torch.from_numpy(np.linalg.inv(extrinsics).astype(np.float32))[None],
            torch.from_numpy(np.stack(inverse))[None], torch.tensor([available], dtype=torch.bool),
            torch.from_numpy(np.stack(affines))[None])
        return RGBInputGroup(key, batch, tuple(records))
