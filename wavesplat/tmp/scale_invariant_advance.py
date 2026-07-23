"""
scale_invariant_advance.py — fix the ROOT cause, not the randomness that samples it.

tmp/VARIANCE_TRACE_fable_2026-07-23.md located the divergence precisely: `should_advance`
compares the RAW training loss against absolute thresholds [0.1, 0.05, None]. Those constants
are scale-dependent, so on a low-Hs sea the loss already sits near/below 0.05 and the N=8
stage advances after ~1 gradient step (seed 1) instead of ~4 (seed 42). Whatever residual
snapshot happens to exist at that moment then decides what the N=8->15 residual beamforming
finds -- and therefore the whole run.

My earlier fix (deterministic phase, tmp/grow_phase_fix.py) removed the *randomness that
samples* this arbitrary timing, giving CV 53% -> 0.0%. But it leaves the defect: the stage
still advances after one step, just always the same way. Determinism != correctness.

This arm changes the criterion itself: normalise the loss by the observation variance, so the
SAME thresholds [0.1, 0.05] now mean "fraction of signal variance still unexplained". Random
phase is left ON, so this measures the threshold fix ALONE (clean attribution).

Pre-registered decision rule (before the numbers):
  - CV drops substantially AND mean nRMSE improves or holds -> ROOT FIX: the scale-dependent
    threshold was the real defect; the phase determinism becomes optional polish.
  - CV drops but mean worsens -> TRADE-OFF; the early advance was accidentally beneficial.
    Report the exchange rate; do not adopt.
  - CV unchanged -> the threshold is not sufficient on its own; the phase draw dominates after
    all, and the trace's root-cause claim needs qualifying.
  Secondary: does the crowded episode (BF4 15comp, high Hs) change at all? If the defect is
  scale-dependence, a high-Hs episode should be affected far less.
"""



import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import wavesplat.optimizer.initializer as INIT
import wavesplat.optimizer.coarse_to_fine as CTF
from wavesplat.run import run_estimation

RESULT = 'tmp/scale_invariant_advance_results.json'
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7]
FLAGS = [True, False]   # True = historical raw-loss threshold (control); False = scale-invariant
EPISODES = [
    ('data/episode_BF3_2comp_seed0_uniform.h5', 'the CV=53% bimodal case'),
    ('data/episode_BF4_15comp_seed0_uniform.h5', 'crowded control'),
]

res = json.load(open(RESULT)) if os.path.exists(RESULT) else {}

for ep, note in EPISODES:
    key = os.path.basename(ep)
    res.setdefault(key, {})
    for pw in FLAGS:
        pk = str(pw)
        res[key].setdefault(pk, {})
        CTF.SCALE_INVARIANT_ADVANCE = (not pw)
        for s in SEEDS:
            if str(s) in res[key][pk]:
                continue
            t0 = time.time()
            out = run_estimation(ep, config={
                'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
                'adaptive_n': True, 'tracking_window': 3, 'seed': s})
            res[key][pk][str(s)] = float(out['avg_nRMSE'])
            json.dump(res, open(RESULT, 'w'), indent=2)
            print(f'  {key:<38} scale_inv={not pw!s:<5} seed={s:<3} nRMSE={out["avg_nRMSE"]:.4f}'
                  f'  [{time.time()-t0:.0f}s]', flush=True)
        a = np.array([res[key][pk][str(s)] for s in SEEDS])
        print(f'{key:<38} scale_inv={not pw!s:<5} mean {a.mean():.4f}  sd {a.std(ddof=1):.4f}  '
              f'CV {100*a.std(ddof=1)/a.mean():5.1f}%  min {a.min():.4f}  max {a.max():.4f}',
              flush=True)
    print(f'  ({note})\n', flush=True)
CTF.SCALE_INVARIANT_ADVANCE = False
