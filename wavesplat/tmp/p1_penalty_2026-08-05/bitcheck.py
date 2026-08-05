"""Bit-identity + thread-sensitivity probe. Targets = tmp/rerun_2026-08-03/progress.jsonl."""
import json, os, sys, time
sys.path.insert(0, '/home/vexit/wavesplat')
from wavesplat.run import run_estimation

DATA = '/home/vexit/wavesplat/wavesplat/data_finemesh_2026-07-23'
CFG = {'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
       'adaptive_n': True, 'tracking_window': 3}
TARGETS = [('BF3_1comp_seed0_uniform', 42, 0.008163), ('BF3_1comp_seed0_uniform', 123, 0.008731),
           ('BF4_5comp_seed0_uniform', 42, 0.031043), ('BF4_15comp_seed0_uniform', 42, 0.271975)]
for ep, seed, tgt in TARGETS:
    t0 = time.time()
    r = run_estimation(f'{DATA}/episode_{ep}.h5', {**CFG, 'seed': seed}, verbose=False)
    got = float(r['avg_nRMSE'])
    print(f"{ep:<26} seed={seed:<4} got={got:.6f} target={tgt:.6f} "
          f"d={got-tgt:+.2e} {'OK' if abs(got-tgt) < 5e-7 else 'MISMATCH'} "
          f"threads={os.environ.get('OMP_NUM_THREADS','default')} ({time.time()-t0:.0f}s)", flush=True)
