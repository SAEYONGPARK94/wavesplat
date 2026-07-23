"""
estimator_variance.py — how large is WaveSplat's run-to-run variance on a FIXED episode?

Motivation (2026-07-23 mesh pilot, tmp/MESH_PILOT_RESULT_2026-07-23.md): the two arms of
`BF3_5comp_seed0` carry *identical wave components* (same generator seed; they differ only in
the raydrop model) yet moved +85.4% and -1.9% under the same mesh refinement. If the
estimator's own run-to-run spread is of that order, then every per-episode delta in this
project -- including the paper's per-episode table and today's mesh deltas -- is a point
estimate with no error bar, and small deltas are not interpretable.

This measures it directly: same episode file, same config, only the estimator's `seed`
varies (initialization / optimizer stochasticity). Nothing about the data changes.

Pre-registered read (before the numbers):
  - If sd(nRMSE)/mean is small (say <3%) relative to the deltas being cited (mesh effect was
    ~4% on the mean, per-episode up to 85%), then per-episode numbers are stable and the
    outlier needs a different explanation (e.g. a genuine bimodal convergence basin).
  - If sd is comparable to the cited deltas, then the project's per-episode numbers require
    error bars and the small-delta conclusions must be re-stated or withdrawn.
  - Watch specifically for BIMODALITY (a few runs landing in a bad basin) rather than a
    smooth spread -- that is what an adaptive-N / non-convex fit would produce, and the mean
    alone would hide it.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from wavesplat.run import run_estimation

RESULT = 'tmp/estimator_variance_results.json'
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7]
EPISODES = [
    'data/episode_BF3_2comp_seed0_uniform.h5',    # easy, low-comp
    'data/episode_BF3_5comp_seed0_uniform.h5',    # the +85.4% outlier arm
    'data/episode_BF3_5comp_seed0_slope.h5',      # its twin: identical components, -1.9%
    'data/episode_BF4_15comp_seed0_uniform.h5',   # crowded
]

res = json.load(open(RESULT)) if os.path.exists(RESULT) else {}

for ep in EPISODES:
    key = os.path.basename(ep)
    res.setdefault(key, {})
    for s in SEEDS:
        if str(s) in res[key]:
            continue
        t0 = time.time()
        v = float(run_estimation(ep, config={
            'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
            'adaptive_n': True, 'tracking_window': 3, 'seed': s})['avg_nRMSE'])
        res[key][str(s)] = v
        json.dump(res, open(RESULT, 'w'), indent=2)
        print(f'  {key:<42} seed={s:<3} nRMSE={v:.4f}  [{time.time()-t0:.0f}s]', flush=True)
    a = np.array([res[key][str(s)] for s in SEEDS])
    print(f'{key:<42} mean {a.mean():.4f}  sd {a.std(ddof=1):.4f}  '
          f'cv {100*a.std(ddof=1)/a.mean():.1f}%  min {a.min():.4f}  max {a.max():.4f}  '
          f'range/mean {100*(a.max()-a.min())/a.mean():.1f}%', flush=True)
    print()
