"""ETKF arm of Table 4 (raydrop) on the fine mesh, k=5 filter seeds.

The WaveSplat arm is already in ws_runs.jsonl; this fills the two ETKF columns so the raydrop
table can be reassembled on the regenerated data. ETKF is cheap (~4 s/episode, numpy/CPU), so
it runs alongside the GPU headline without contending for the GPU. Note the ETKF numbers are
the as-published default config; the raydrop table is about relative sensitivity, not absolute
level, so this matches the paper's own framing for that table.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))

from run_p1_comparison import run_etkf

DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
OUT = os.path.join(HERE, 'etkf_raydrop_runs.jsonl')

PAIRS = [('BF2', '1comp'), ('BF3', '1comp'), ('BF3', '2comp'), ('BF3', '5comp'),
         ('BF4', '1comp'), ('BF4', '5comp'), ('BF4', '15comp'), ('BF4', '30comp'),
         ('BF5', '15comp'), ('BF5', '30comp')]
SEEDS = [42, 1, 2, 3, 4]

done = set()
if os.path.exists(OUT):
    with open(OUT) as f:
        for line in f:
            try:
                done.add(json.loads(line)['job_id'])
            except (json.JSONDecodeError, KeyError):
                pass

with open(OUT, 'a') as fh:
    for bf, c in PAIRS:
        for cond in ('uniform', 'slope'):
            ep = f'episode_{bf}_{c}_seed0_{cond}.h5'
            p = os.path.join(DATA, ep)
            if not os.path.exists(p):
                continue
            for s in SEEDS:
                jid = f'{ep}|{s}'
                if jid in done:
                    continue
                np.random.seed(s)
                try:
                    r = run_etkf(p, 'medium')
                    rec = {'job_id': jid, 'episode': ep, 'cond': cond, 'seed': s,
                           'avg_nRMSE': float(r['avg_nRMSE'])}
                except Exception as e:
                    rec = {'job_id': jid, 'episode': ep, 'cond': cond, 'seed': s,
                           'error': repr(e)}
                fh.write(json.dumps(rec) + '\n')
                fh.flush()
                print(f'{ep:<40s} s{s} {rec.get("avg_nRMSE", rec.get("error"))}', flush=True)
print('done')
