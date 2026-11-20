"""Construct the hanco a3 DEV (1100-1199) hamer stage1 manifest by mirroring
the TrainCore manifest schema with the v24_a3_dev cache files/shas.
FINAL-refit merged-segment input; dev 1100-1199 is a legal TRAINING segment
for FINAL_REFIT_ROLE only (spec A2). split field says 'development'
honestly; the trainer treats it as a training population for the refit.
"""
import hashlib, json, os

tc = json.load(open(os.environ.get('ALIGNHAND_DATA', 'data') + '/ghpush/results/us/s4v46/'
                    'stage1_hanco_a3_traincore.json'))
cdirs = [os.environ.get('ALIGNHAND_DATA', 'data') + '/caches/hanco/v24_a3_dev_hamer_00000_04096_v1',
         os.environ.get('ALIGNHAND_DATA', 'data') + '/caches/hanco/v24_a3_dev_hamer_04096_07084_v1']
ent = {}
for cd in cdirs:
    for e in json.load(open(os.path.join(cd, 'CACHE_INDEX.json'))):
        ent[e['key']] = (os.path.join(cd, e['file']), e['sha256'])
rows_meta = [json.loads(l) for l in open(
    os.environ.get('ALIGNHAND_DATA', 'data') + '/prepared/v24_hanco_a3_development_v1/ROWS.jsonl')]
rows = []
for i, r in enumerate(rows_meta):
    f, sha = ent[r['key']]
    assert os.path.exists(f)
    rows.append(dict(index=i, key=r['key'], file=f, sha256=sha))
m = dict(schema=tc['schema'], status='COMPLETE', dataset='hanco',
         prior='HaMeR', split='development',
         note='FINAL_REFIT merged-segment component: dev 1100-1199 used as '
              'TRAINING data for FINAL_REFIT_ROLE only (spec A2)',
         expansion=tc.get('expansion'),
         prepared=os.environ.get('ALIGNHAND_DATA', 'data') + '/prepared/v24_hanco_a3_development_v1',
         prepared_report_sha256=hashlib.sha256(open(
             os.environ.get('ALIGNHAND_DATA', 'data') + '/prepared/v24_hanco_a3_development_v1/'
             'REPORT.json', 'rb').read()).hexdigest(),
         supervision=os.environ.get('ALIGNHAND_DATA', 'data') + '/handoff/protocol/targets_training/'
                     'current/v24_hanco_a3_development_v1',
         supervision_report_sha256='', rows=rows)
out = os.environ.get('ALIGNHAND_DATA', 'data') + '/stage6/manifests/stage1_hanco_a3_dev_for_refit.json'
body = json.dumps(m)
open(out, 'w').write(body)
print('rows', len(rows), 'sha',
      hashlib.sha256(body.encode()).hexdigest()[:20])
