"""Read RGB for offline-crop S0 inputs; this reader cannot open score targets."""
import hashlib
from pathlib import Path

import cv2
import numpy as np

from mvhand.data.v2_dexycb_rgb import DexYCBRGBReader
from mvhand.data.v2_hanco_input import InputZip
from mvhand.data.v2_access import authorize_population


class PreparedDevelopmentRGB:
    def __init__(self, config):
        self.dataset = config['dataset']
        self.split = config.get('split','development')
        authorize_population(self.split,phase=1)
        self.root = Path(config.get('dataset_root','.'))
        self.reader = None
        if self.dataset == 'dexycb':
            self.reader = DexYCBRGBReader(index=Path(config['index']),dataset_root=self.root,
                dataset_sha256=config['dataset_sha256'],split=self.split)
        elif self.dataset == 'hanco':
            self.reader = InputZip(Path(config['rgb_archive']))
        elif self.dataset != 'ho3d':
            raise ValueError('unregistered Development RGB adapter')

    def __enter__(self):
        return self

    def __exit__(self,*_):
        if isinstance(self.reader,DexYCBRGBReader):
            self.reader.db.close()
        elif self.reader is not None:
            self.reader.close()

    def read(self,row):
        if row['split'] != self.split:
            raise PermissionError('RGB row differs from the authorized nonterminal population')
        if self.dataset == 'dexycb':
            images = self.reader.read(row['key'],row['source_split'])
            order = row.get('image_order',list(range(8)))
            if sorted(order) != list(range(8)):
                raise ValueError('invalid linked physical camera permutation')
            return [images[i] for i in order]
        payloads = []
        if self.dataset == 'hanco':
            for member in row['rgb_members']:
                if not member.startswith('rgb/') or '..' in Path(member).parts:
                    raise ValueError('unregistered RGB member')
                payloads.append(self.reader.read(member))
        else:
            shard = row['shard']
            if (Path(shard).name != shard or not shard.startswith('HO3D_mv_train-')
                    or tuple(row['camera_ids']) != (0,1,4,3,2)):
                raise PermissionError('HO3D Development shard or camera binding mismatch')
            with (self.root/shard).open('rb') as stream:
                for member in row['image_members']:
                    if member['offset'] < 0 or member['size'] <= 0:
                        raise ValueError('invalid image member')
                    stream.seek(member['offset']);payload=stream.read(member['size'])
                    if len(payload)!=member['size']:
                        raise ValueError('short RGB read')
                    payloads.append(payload)
        images=[]
        for payload in payloads:
            image=cv2.imdecode(np.frombuffer(payload,np.uint8),cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('RGB decoding failed')
            images.append((image,hashlib.sha256(payload).hexdigest()))
        if len(images) != (5 if self.dataset=='ho3d' else 8):
            raise ValueError('physical camera count mismatch')
        return images
