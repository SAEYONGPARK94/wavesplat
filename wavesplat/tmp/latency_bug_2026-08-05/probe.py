"""Cheap probe: rerun the exact config that logged avg_inference_ms=-0.972
(BF4_15comp_seed0_uniform / OURS / seed 42) and dump the per-frame elapsed_ms
distribution.  Also record a monotonic-clock reference for the whole tracking loop.
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
from wavesplat.run import run_estimation

EP = os.path.join(ROOT, 'data_finemesh_2026-07-23',
                  'episode_BF4_15comp_seed0_uniform.h5')
CFG = {'use_dispersion': True, 'use_confidence': True, 'adaptive_n': True,
       'tracking_window': 3, 'init_method': 'v2_konly', 'seed': int(sys.argv[1]) if len(sys.argv) > 1 else 42}

m0 = time.monotonic(); w0 = time.time()
r = run_estimation(EP, CFG, verbose=False)
m1 = time.monotonic(); w1 = time.time()

e = np.array([f['elapsed_ms'] for f in r['frame_results']], dtype=float)
order = np.argsort(e)
print(f"cfg seed={CFG['seed']}  avg_inference_ms={r['avg_inference_ms']:.4f}  n_frames={len(e)}")
print(f"wall(time.time)={w1-w0:.3f}s  monotonic={m1-m0:.3f}s  diff={(w1-w0)-(m1-m0):+.6f}s")
print(f"per-frame ms: min={e.min():.3f} p1={np.percentile(e,1):.3f} med={np.median(e):.3f} "
      f"max={e.max():.3f} sum={e.sum():.1f}ms  n_negative={(e<0).sum()}")
print("5 smallest frames (idx, ms):", [(int(i), round(float(e[i]), 3)) for i in order[:5]])
print("5 largest  frames (idx, ms):", [(int(i), round(float(e[i]), 3)) for i in order[-5:]])
# what would the mean be with the outliers dropped?
print(f"median-based avg (robust) = {np.median(e):.3f} ; trimmed10% mean = "
      f"{np.mean(np.sort(e)[len(e)//20: len(e)-len(e)//20]):.3f}")
json.dump({'seed': CFG['seed'], 'avg_inference_ms': float(r['avg_inference_ms']),
           'per_frame_ms': e.tolist(), 'wall_s': w1-w0, 'monotonic_s': m1-m0},
          open(os.path.join(HERE, f"probe_seed{CFG['seed']}.json"), 'w'))
