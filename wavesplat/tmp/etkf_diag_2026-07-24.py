"""
etkf_diag_2026-07-24.py — Stage 1 diagnosis of the ETKF coarse->fine regression.

Pre-registration: tmp/ETKF_FINEMESH_PREREG_2026-07-24.md  (written first).

Runs, paired on the SAME episodes in data/ and data_finemesh_2026-07-23/:
  D1  oracle-grid representation floor vs analysis-grid resolution
  D2  shrinkage diagnostic (rho, alpha*, nRMSE at alpha*) at the shipped config
  D3  analysis-grid resolution sweep

CPU/numpy only, one BLAS thread per worker.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import sys
import json
import itertools
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OLD = os.path.join(ROOT, 'data')
NEW = os.path.join(ROOT, 'data_finemesh_2026-07-23')

EPISODES = [
    'episode_BF2_1comp_seed0_uniform.h5',
    'episode_BF3_1comp_seed0_uniform.h5',
    'episode_BF3_2comp_seed0_uniform.h5',
    'episode_BF3_5comp_seed0_uniform.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]

GRIDS = [(20, 15), (40, 30), (60, 45), (80, 60), (120, 90), (160, 120)]
SEEDS = [0, 1, 2]


def _floor_job(a):
    from etkf_lib import oracle_grid_floor
    ep, nx, ny = a
    return (ep, nx, ny, oracle_grid_floor(os.path.join(NEW, ep), nx, ny))


def _run_job(a):
    from etkf_lib import run_etkf
    ds, ep, nx, ny, seed = a
    d = os.path.join(ROOT, ds)
    r = run_etkf(os.path.join(d, ep), nx=nx, ny=ny, seed=seed, diagnostics=True)
    return dict(ds=ds, ep=ep, nx=nx, ny=ny, seed=seed,
                nRMSE=r['avg_nRMSE'], rho=r['avg_rho'], alpha=r['avg_alpha'],
                nRMSE_alpha=r['avg_nRMSE_alpha'], ms=r['avg_ms'])


def main():
    out = {}

    # ---- D1 representation floor (GT identical in both datasets)
    jobs = [(ep, nx, ny) for ep in EPISODES for nx, ny in GRIDS]
    with Pool(14) as p:
        res = p.map(_floor_job, jobs)
    floors = {}
    for ep, nx, ny, v in res:
        floors.setdefault(ep, {})[f'{nx}x{ny}'] = v
    out['D1_oracle_grid_floor'] = floors
    print('D1 done', flush=True)

    # ---- D2/D3 grid sweep, both datasets, 3 seeds
    jobs = [(ds, ep, nx, ny, s)
            for ds in ('data', 'data_finemesh_2026-07-23')
            for ep in EPISODES for nx, ny in GRIDS for s in SEEDS]
    with Pool(14) as p:
        res = p.map(_run_job, jobs, chunksize=1)
    out['D3_sweep'] = res
    print('D3 done', flush=True)

    with open(os.path.join(ROOT, 'tmp', 'etkf_diag_2026-07-24.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('written')


if __name__ == '__main__':
    main()
