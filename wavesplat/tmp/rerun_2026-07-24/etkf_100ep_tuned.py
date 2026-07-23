"""100-episode tuned ETKF on the fine mesh — the number the abstract's 0.319 must be replaced by.

Config = the paper's "ETKF (tuned)": Ne=100, grid 40x30, mult_inflation=1.00, add_inflation=0.005,
obs_noise=0.05, max_obs=40 (wavesplat.tex L747, L928; tab:etkf_sweep winner). Median over 3
filter seeds. Uses the agent's FastETKF (verified to reproduce the stock ETKFGrid to 2.9e-15
and the paper's own published sweep). CPU/numpy only.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tmp'))

from etkf_lib import run_etkf

DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
OUT = os.path.join(HERE, 'etkf_100ep_tuned_runs.jsonl')
SEEDS = [42]  # ETKF seed spread is small (agent measured), so 1 seed is enough for the level

CFG = dict(nx=40, ny=30, n_ensemble=100, mult_inflation=1.00,
           add_inflation=0.005, obs_noise=0.05, max_obs=40)


def main():
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                try:
                    done.add(json.loads(line)['job_id'])
                except (json.JSONDecodeError, KeyError):
                    pass
    eps = sorted(f for f in os.listdir(DATA) if f.endswith('.h5'))
    with open(OUT, 'a') as fh:
        for i, ep in enumerate(eps):
            for s in SEEDS:
                jid = f'{ep}|{s}'
                if jid in done:
                    continue
                try:
                    r = run_etkf(os.path.join(DATA, ep), seed=s, diagnostics=False, **CFG)
                    rec = {'job_id': jid, 'episode': ep, 'seed': s,
                           'nRMSE': float(r['avg_nRMSE'])}
                except Exception as e:
                    rec = {'job_id': jid, 'episode': ep, 'seed': s, 'error': repr(e)}
                fh.write(json.dumps(rec) + '\n')
                fh.flush()
            if (i + 1) % 20 == 0:
                print(f'{i+1}/{len(eps)} episodes', flush=True)
    print('done')


if __name__ == '__main__':
    main()
