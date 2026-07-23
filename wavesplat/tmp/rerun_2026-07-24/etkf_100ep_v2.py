"""100-ep tuned ETKF, v2 — clears etkf_lib._CACHE each episode so 100 x ~90MB finemesh files
don't accumulate in memory. Per-episode progress. Paper's tuned config, seed 42."""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'tmp'))

import etkf_lib
from etkf_lib import run_etkf

DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
OUT = os.path.join(HERE, 'etkf_100ep_tuned_runs.jsonl')
CFG = dict(nx=40, ny=30, n_ensemble=100, mult_inflation=1.00,
           add_inflation=0.005, obs_noise=0.05, max_obs=40)

done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try:
            done.add(json.loads(line)['job_id'])
        except (json.JSONDecodeError, KeyError):
            pass

eps = sorted(f for f in os.listdir(DATA) if f.endswith('.h5'))
with open(OUT, 'a') as fh:
    for i, ep in enumerate(eps):
        jid = f'{ep}|42'
        if jid in done:
            continue
        t0 = time.time()
        try:
            r = run_etkf(os.path.join(DATA, ep), seed=42, diagnostics=False, **CFG)
            rec = {'job_id': jid, 'episode': ep, 'seed': 42, 'nRMSE': float(r['avg_nRMSE'])}
        except Exception as e:
            rec = {'job_id': jid, 'episode': ep, 'seed': 42, 'error': repr(e)}
        fh.write(json.dumps(rec) + '\n')
        fh.flush()
        etkf_lib._CACHE.clear()  # prevent 100x90MB accumulation
        print(f'[{i+1}/{len(eps)}] {ep:<42s} {rec.get("nRMSE", rec.get("error"))} '
              f'({time.time()-t0:.1f}s)', flush=True)
print('done')
