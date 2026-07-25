"""ETKF raydrop arm (Table 4) on the fine mesh, RETUNED hold-out-clean config.

Task 2026-07-25: the existing etkf_raydrop_runs.jsonl (and the paper's current Table 4
ETKF column) used the AS-PUBLISHED default config (40x30, Ne=50, ai=0.005, obs_noise=0.05).
The 2026-07-24 ETKF retune (tmp/ETKF_FINEMESH_2026-07-24.md sec 4.3, etkf_tune_stageC json)
produced a hold-out-clean config tuned on seed-1 episodes and reported on seed-0. This runner
re-measures the raydrop arm with that retuned config so Table 4 can use the same honestly-tuned
ETKF used elsewhere in the refreshed paper.

Retuned config: grid 160x120, Ne=50, mult_inflation=1.0, add_inflation=0.0,
                obs_noise=0.20, max_obs=20.

Protocol matches the WS arm (tmp/rerun_2026-07-24/ws_runs.jsonl): seed-0 episodes,
median over 5 method seeds {42,1,2,3,4}. rho recorded to document the zero-predictor caveat
(sec 4.4/5 of the ETKF doc: this config is nRMSE-degenerate, rho ~ 0.015).
CPU/numpy only. No GPU.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'tmp'))

import etkf_lib
from etkf_lib import run_etkf

DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
OUT = os.path.join(HERE, 'etkf_raydrop_retuned_runs.jsonl')

CFG = dict(nx=160, ny=120, n_ensemble=50, mult_inflation=1.0,
           add_inflation=0.0, obs_noise=0.20, max_obs=20)

PAIRS = [('BF2', '1comp'), ('BF3', '1comp'), ('BF3', '2comp'), ('BF3', '5comp'),
         ('BF4', '1comp'), ('BF4', '5comp'), ('BF4', '15comp'), ('BF4', '30comp'),
         ('BF5', '15comp'), ('BF5', '30comp')]
SEEDS = [42, 1, 2, 3, 4]

done = set()
if os.path.exists(OUT):
    for line in open(OUT):
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
                print(f'MISSING {ep}', flush=True)
                continue
            for s in SEEDS:
                jid = f'{ep}|{s}'
                if jid in done:
                    continue
                t0 = time.time()
                try:
                    r = run_etkf(p, seed=s, diagnostics=True, **CFG)
                    rec = {'job_id': jid, 'episode': ep, 'cond': cond, 'seed': s,
                           'avg_nRMSE': float(r['avg_nRMSE']),
                           'avg_rho': float(r['avg_rho']),
                           'cfg': 'retuned_holdout'}
                except Exception as e:
                    rec = {'job_id': jid, 'episode': ep, 'cond': cond, 'seed': s,
                           'error': repr(e)}
                fh.write(json.dumps(rec) + '\n')
                fh.flush()
                print(f'{ep:<42s} s{s} nRMSE={rec.get("avg_nRMSE", rec.get("error"))} '
                      f'rho={rec.get("avg_rho", "-")} ({time.time()-t0:.1f}s)', flush=True)
            etkf_lib._CACHE.clear()  # release the ~90MB finemesh episode
print('done', flush=True)
