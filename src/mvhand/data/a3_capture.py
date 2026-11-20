"""Strictly bind additional A3 frozen-prior captures to nonterminal inputs."""
import json
from pathlib import Path
from mvhand.training.cached_stage1 import sha256


def selected_indices(config, report, rows):
    if report.get('population')!='hanco_a3' or config['dataset']!='hanco':
        raise PermissionError('not the registered A3 population')
    split=config['split']
    if split not in ('fit','development'):raise PermissionError('A3 Final capture is not enabled')
    def valid(key):
        sequence=int(key.split('/')[0])
        return 0<=sequence<1100 if split=='fit' else 1100<=sequence<1200
    if any(not valid(r['key']) for r in rows):raise PermissionError('A3 split leakage')
    selector=Path(config['selector'])
    if sha256(selector)!=config['selector_sha256']:raise ValueError('A3 selector changed')
    selected=json.loads(selector.read_text())
    if not isinstance(selected,list) or not 1<=len(selected)<=4096:raise ValueError('invalid A3 shard size')
    indices=[int(r['index']) for r in selected]
    if len(set(indices))!=len(indices) or indices!=sorted(indices):raise ValueError('duplicate or reordered A3 indices')
    for i,item in zip(indices,selected):
        if i<0 or i>=len(rows) or rows[i]['key']!=item['key']:raise ValueError('A3 selector binding mismatch')
    if config.get('qualification') and len(indices)!=4:raise ValueError('A3 qualification uses four discarded captures')
    return indices
