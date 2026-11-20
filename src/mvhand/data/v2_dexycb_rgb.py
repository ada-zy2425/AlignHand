"""Indexed DexYCB RGB access without opening annotation payloads."""
import hashlib
from pathlib import Path
import sqlite3

import cv2
import numpy as np

from mvhand.data.v2_access import authorize_population


class DexYCBRGBReader:
    def __init__(self, *, index: Path, dataset_root: Path, dataset_sha256: str, split: str):
        authorize_population(split, phase=1)
        self.split = split
        self.root = Path(dataset_root).resolve()
        self.db = sqlite3.connect(f'file:{Path(index).resolve()}?mode=ro', uri=True)
        identity = self.db.execute("SELECT v FROM meta WHERE k='dataset_sha256'").fetchone()
        if identity != (dataset_sha256,):
            self.db.close()
            raise ValueError('RGB index dataset identity mismatch')

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.db.close()

    def read(self, key: str, source_split: str):
        expected = 'val' if self.split == 'development' else 'train'
        if source_split != expected:
            raise PermissionError('RGB source split differs from authorized population')
        shard, sample = key.split('::', 1)
        if Path(shard).name != shard or not shard.startswith(f'DexYCB_mv_{expected}-'):
            raise PermissionError('RGB shard identity differs from authorized population')
        images = []
        with (self.root/shard).open('rb') as stream:
            for camera in range(8):
                row = self.db.execute(
                    'SELECT shard,off,size FROM members WHERE split=? AND key=? AND suffix=?',
                    (expected, sample, f'image_{camera}.jpg')).fetchone()
                if row is None or row[0] != shard:
                    raise ValueError('missing or mismatched physical image slot')
                stream.seek(row[1]); payload = stream.read(row[2])
                if len(payload) != row[2]:
                    raise ValueError('incomplete RGB payload')
                image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError('RGB decoding failed')
                images.append((image, hashlib.sha256(payload).hexdigest()))
        return images
