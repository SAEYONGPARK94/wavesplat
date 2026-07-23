"""ETKF median-of-k seed spread at the AS-PUBLISHED config, both meshes, 12 episodes.
Answers: is the coarse->fine per-episode swing inside the ETKF's own ensemble-seed spread?
k=7 filter seeds.  CPU/numpy only."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, json
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

SCEN = [('BF2_1comp','uniform'),('BF3_1comp','uniform'),('BF3_2comp','uniform'),
        ('BF3_2comp','slope'),('BF3_5comp','uniform'),('BF3_5comp','slope'),
        ('BF4_1comp','uniform'),('BF4_5comp','uniform'),('BF4_15comp','uniform'),
        ('BF4_30comp','uniform'),('BF5_15comp','uniform'),('BF5_30comp','uniform')]
REPORT = [f'episode_{s}_seed0_{r}.h5' for s, r in SCEN]
SEEDS = [0, 1, 2, 3, 4, 5, 6]
CFG = dict(nx=40, ny=30, n_ensemble=50, mult_inflation=1.01,
           add_inflation=0.005, obs_noise=0.05, max_obs=40)


def _job(a):
    from etkf_lib import run_etkf
    ds, ep, seed = a
    r = run_etkf(os.path.join(ROOT, ds, ep), seed=seed, diagnostics=True, **CFG)
    return dict(ds=ds, ep=ep, seed=seed, nRMSE=r['avg_nRMSE'], rho=r['avg_rho'],
                alpha=r['avg_alpha'], nRMSE_alpha=r['avg_nRMSE_alpha'])


if __name__ == '__main__':
    jobs = [(ds, ep, s) for ds in ('data', 'data_finemesh_2026-07-23')
            for ep in REPORT for s in SEEDS]
    with Pool(10) as p:
        res = p.map(_job, jobs, chunksize=1)
    json.dump(res, open(os.path.join(ROOT, 'tmp', 'etkf_seedspread_2026-07-24.json'), 'w'),
              indent=1)
    print('done')
