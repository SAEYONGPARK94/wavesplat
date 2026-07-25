"""SECONDARY reference: ETKF raydrop arm with the PAPER-TUNED config (tab:etkf_sweep winner),
which is the 'tuned ETKF' the rest of the refreshed paper uses (tab:main, headline).

Not the task's primary deliverable (that is the retuned hold-out config, etkf_raydrop_retuned.py),
but measured for context because the config-consistency of Table 4 vs the rest of the paper is a
live question: the retuned hold-out config is an nRMSE-degenerate zero predictor, whereas the
paper headlines the paper-tuned config. This lets the coordinator see the raydrop Delta% under
the config the paper actually cites elsewhere.

Paper-tuned config: grid 40x30, Ne=100, mult_inflation=1.0, add_inflation=0.005,
                    obs_noise=0.05, max_obs=40.
Same protocol: seed-0 episodes, median over 5 method seeds {42,1,2,3,4}, rho recorded.
CPU/numpy only.
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
OUT = os.path.join(HERE, 'etkf_raydrop_papertuned_runs.jsonl')

CFG = dict(nx=40, ny=30, n_ensemble=100, mult_inflation=1.0,
           add_inflation=0.005, obs_noise=0.05, max_obs=40)

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
                           'cfg': 'paper_tuned'}
                except Exception as e:
                    rec = {'job_id': jid, 'episode': ep, 'cond': cond, 'seed': s,
                           'error': repr(e)}
                fh.write(json.dumps(rec) + '\n')
                fh.flush()
                print(f'{ep:<42s} s{s} nRMSE={rec.get("avg_nRMSE", rec.get("error"))} '
                      f'rho={rec.get("avg_rho", "-")} ({time.time()-t0:.1f}s)', flush=True)
            etkf_lib._CACHE.clear()
print('done', flush=True)
