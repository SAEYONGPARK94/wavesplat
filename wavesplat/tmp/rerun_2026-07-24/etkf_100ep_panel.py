"""ETKF tuned config, full metric panel (nRMSE/VR/rho/SS) on all 100 fine-mesh episodes.

The fast nRMSE-only run showed 0.332; this adds the panel metrics the five-metric table needs,
because the agent measured ETKF's VR to *change* on the fine mesh (it can no longer be carried
from the old numbers). Uses etkf_lib's FastETKF for the assimilation, then the repo's own
compute_all_metrics on the interpolated field. Clears the episode cache each step. seed 42.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tmp'))

import etkf_lib
from etkf_lib import get_episode, FastETKF, _grid_to_gt
from wavesplat.evaluation.metrics import compute_all_metrics

DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
OUT = os.path.join(HERE, 'etkf_100ep_panel_runs.jsonl')
KEYS = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score']
CFG = dict(n_ensemble=100, mult_inflation=1.00, add_inflation=0.005,
           obs_noise=0.05, max_obs=40)


def run_panel(path, seed=42):
    ep = get_episode(path)
    np.random.seed(seed)
    nx, ny = 40, 30
    gx = np.linspace(ep['gt_grid_x'][0], ep['gt_grid_x'][-1], nx)
    gy = np.linspace(ep['gt_grid_y'][0], ep['gt_grid_y'][-1], ny)
    etkf = FastETKF(gx, gy, depth=ep['depth'], **CFG)
    fr = []
    for i, (x, y, z) in enumerate(ep['obs']):
        etkf.update(x, y, z)
        pred = _grid_to_gt(etkf.predict_grid(), gx, gy, ep['gt_grid_x'], ep['gt_grid_y'])
        m = compute_all_metrics(pred, ep['gt'][i], ep['gt_grid_x'], ep['gt_grid_y'], ep['Hs'])
        fr.append(m)
    return {k: float(np.mean([f[k] for f in fr if np.isfinite(f.get(k, np.nan))]))
            for k in KEYS}


def main():
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                done.add(json.loads(line)['episode'])
            except (json.JSONDecodeError, KeyError):
                pass
    eps = sorted(f for f in os.listdir(DATA) if f.endswith('.h5'))
    with open(OUT, 'a') as fh:
        for i, ep in enumerate(eps):
            if ep in done:
                continue
            t0 = time.time()
            try:
                rec = {'episode': ep, **run_panel(os.path.join(DATA, ep))}
            except Exception as e:
                rec = {'episode': ep, 'error': repr(e)}
            fh.write(json.dumps(rec) + '\n')
            fh.flush()
            etkf_lib._CACHE.clear()
            print(f'[{i+1}/{len(eps)}] {ep:<42s} '
                  f'{rec.get("nRMSE", rec.get("error"))} ({time.time()-t0:.1f}s)', flush=True)
    print('done')


if __name__ == '__main__':
    main()
