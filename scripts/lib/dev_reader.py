"""DexYCB development RGB reader used by the runtime benchmark.

Wraps the production reader (`mvhand.data.v21_prepared_rgb.PreparedDevelopmentRGB`),
which applies each row's image order and camera binding, so the timed crops
come from exactly the frames the prior caches were built from.
"""
import json
import os
import sys
from pathlib import Path

DATA = Path(os.environ.get('ALIGNHAND_DATA', 'data'))
sys.path.insert(0, os.environ.get('ALIGNHAND_SRC', os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src')))


def dexycb_dev_reader():
    from mvhand.data.v21_prepared_rgb import PreparedDevelopmentRGB
    cfg = json.loads((DATA / 'caches/queue/'
                      'v21_s0_dexycb_hamer_s25_v1.config.json').read_text())
    return PreparedDevelopmentRGB(dict(
        dataset='dexycb', split='development',
        dataset_root=cfg['dataset_root'], index=cfg['index'],
        dataset_sha256=cfg['dataset_sha256']))
