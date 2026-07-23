"""ETKF timing probe (2026-07-24 full re-run planning)."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from run_p1_comparison import run_etkf
from wavesplat.data_loader import EpisodeDataLoader

DATA = os.path.join(os.path.dirname(__file__), '..', 'data_finemesh_2026-07-23')

for name in ['episode_BF3_2comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']:
    p = os.path.join(DATA, name)
    loader = EpisodeDataLoader(p)
    n_frames = sum(1 for _ in loader)
    t0 = time.time()
    r = run_etkf(p, 'medium')
    dt = time.time() - t0
    print(f'{name:45s} frames={n_frames:4d} wall={dt:7.1f}s '
          f'nRMSE={r["avg_nRMSE"]:.4f} ms/f={r["avg_inference_ms"]:.1f}', flush=True)
