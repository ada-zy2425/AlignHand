"""Export HanCo-A3 (v24) bundles from the fresh A3 HaMeR captures.

Reuses s_common's validated FrozenForward/export machinery via a pseudo
dataset 'hanco_a3': prepared -> v24 populations, caches -> v24_a3_* capture
shards (fit-CLI layout), parent/R models -> the hanco ones (symlinked).
Exports: devfull (7084), dev256, fit8192.
"""
import os
import json, sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import s_common
from s_common import F2F, PREPARED, export_bundle, seq_stratified, guard

HANDOFF = Path(os.environ.get('ALIGNHAND_DATA', 'data') + '/handoff')

PREPARED[('hanco_a3', 'fit')] = F2F / 'prepared/v24_hanco_a3_fit_v1'
PREPARED[('hanco_a3', 'development')] = F2F / 'prepared/v24_hanco_a3_development_v1'
s_common.PARENTS['hanco_a3'] = s_common.PARENTS['hanco']

link = HANDOFF / 'models/single_prior_R/hanco_a3'
if not link.exists() and (HANDOFF / 'models/single_prior_R/hanco').exists():
    link.symlink_to(HANDOFF / 'models/single_prior_R/hanco')

_orig_cache_files = s_common.cache_files

def cache_files(ds, split):
    if ds != 'hanco_a3':
        return _orig_cache_files(ds, split)
    root = guard(F2F / 'caches/hanco')
    tag = 'dev' if split == 'development' else 'fit'
    dirs = sorted(root.glob(f'v24_a3_{tag}_hamer_*_v1'),
                  key=lambda d: int(d.name.split('_')[4]))
    files = []
    for d in dirs:
        index = json.loads((d / 'CACHE_INDEX.json').read_text())
        for i, entry in enumerate(index):
            if entry['index'] != len(files):
                raise ValueError(f'cache index out of order in {d}')
            files.append(d / entry['file'])
    prepared = guard(PREPARED[(ds, split)])
    rows = [json.loads(s) for s in (prepared / 'ROWS.jsonl').read_text().splitlines()]
    if len(rows) != len(files) or any(r['split'] != split for r in rows):
        raise ValueError('cache/prepared mismatch')
    return files, rows, prepared

s_common.cache_files = cache_files

def main():
    root = Path(os.environ.get('ALIGNHAND_DATA', 'data') + '/results/s1/bundles')
    for split, count, name in (('development', None, 'devfull'),
                               ('development', 256, 'dev256'),
                               ('fit', 8192, 'fit8192')):
        out = root / 'hanco_a3' / name
        if (out / 'meta.json').exists():
            print('skip', out, flush=True)
            continue
        fwd = s_common.FrozenForward('hanco_a3', split)
        n = len(fwd.rows)
        idx = list(range(n)) if count is None else seq_stratified(fwd.rows, min(count, n))
        print('exporting', name, len(idx), 'groups', flush=True)
        export_bundle('hanco_a3', split, idx, out, workers=48)
        print('A3_BUNDLE_DONE', name, flush=True)
    print('A3_BUNDLES_ALL_DONE', flush=True)

if __name__ == '__main__':
    main()
