"""Timing probe for the 2026-07-24 full re-run: how long does one episode cost?"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.run import run_estimation

DATA = os.path.join(os.path.dirname(__file__), '..', 'data_finemesh_2026-07-23')

CASES = [
    ('episode_BF3_2comp_seed0_uniform.h5', 'sparse'),
    ('episode_BF4_15comp_seed0_uniform.h5', 'crowded'),
    ('episode_BF5_30comp_seed0_uniform.h5', 'worst'),
]

P1 = {'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
      'adaptive_n': True, 'tracking_window': 3, 'seed': 42}

for name, tag in CASES:
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        print(f'MISSING {name}')
        continue
    t0 = time.time()
    r = run_estimation(p, dict(P1))
    dt = time.time() - t0
    print(f'{tag:8s} {name:45s} wall={dt:7.1f}s  nRMSE={r["avg_nRMSE"]:.4f}  '
          f'ms/f={r["avg_inference_ms"]:.1f}', flush=True)
