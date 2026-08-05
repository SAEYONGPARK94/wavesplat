"""Smoke: cost + sanity of the RELAX init arms before committing to the full grid."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
from wavesplat.run import run_estimation

DATA = os.path.join(ROOT, 'data_finemesh_2026-07-23')
BASE = {'use_dispersion': True, 'use_confidence': True, 'adaptive_n': True,
        'tracking_window': 3, 'seed': 42}

for ep in ['BF3_1comp_seed0_uniform', 'BF4_30comp_seed0_uniform']:
    for im in ['v2_konly', 'relax_konly', 'relax']:
        p = os.path.join(DATA, f'episode_{ep}.h5')
        t0 = time.time()
        try:
            r = run_estimation(p, {**BASE, 'init_method': im}, verbose=False)
            print(f"{ep:28s} {im:12s} nRMSE={r['avg_nRMSE']:.4f} "
                  f"init={r['init_nrmse']:.4f} N={r.get('n_gaussians')} "
                  f"{r['avg_inference_ms']:.1f}ms/f wall={time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"{ep:28s} {im:12s} ERROR {type(e).__name__}: {e} wall={time.time()-t0:.1f}s", flush=True)
