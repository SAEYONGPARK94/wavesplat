"""
grow_phase_fix.py — remove BOTH random-phase sites at once. Is the estimator then deterministic?

tmp/VARIANCE_MECHANISM_HUNT_2026-07-23.md killed three hypotheses. Removing the initializer's
random phase alone (KONLY_RANDOM_WPHI=False) cut CV only 53.0 -> 47.8% -- the bad basin moved
seed rather than disappearing, which is exactly what you would expect if a SECOND random-phase
site remains. That site is the adaptive-N growth path: coarse_to_fine.py draws
`init_phi = torch.rand(n_new) * 2*pi` for every newly grown component.

This arm sets BOTH to deterministic (initializer keeps v2's least-squares (w, phi); grown
components start at phase 0).

Pre-registered decision rule (before the numbers):
  - CV collapses toward ~0 on BOTH episodes -> CAUSE FOUND: run-to-run variance is entirely
    random phase initialisation, at two sites, and it is removable.
  - CV drops a lot but stays >10% -> PARTIAL: a third stochastic input remains; name the
    next suspect rather than claiming the fix.
  - CV barely moves -> REFUTED; the growth path is not it either, and the phase-freeze result
    that motivated this whole line needs re-examination.
  Separately (secondary, not the gate): report whether mean nRMSE worsens. Determinism that
  costs accuracy is a trade-off to disclose, not a free win.
"""


import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import wavesplat.optimizer.initializer as INIT
import wavesplat.optimizer.coarse_to_fine as CTF
from wavesplat.run import run_estimation

RESULT = 'tmp/grow_phase_fix_results.json'
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7]
FLAGS = [True, False]   # True = historical random w/phi (control); False = v2's fitted w/phi
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
        CTF.GROW_RANDOM_PHI = pw
        INIT.KONLY_RANDOM_WPHI = pw
        for s in SEEDS:
            if str(s) in res[key][pk]:
                continue
            t0 = time.time()
            out = run_estimation(ep, config={
                'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
                'adaptive_n': True, 'tracking_window': 3, 'seed': s})
            res[key][pk][str(s)] = float(out['avg_nRMSE'])
            json.dump(res, open(RESULT, 'w'), indent=2)
            print(f'  {key:<38} rand_phase={pw!s:<5} seed={s:<3} nRMSE={out["avg_nRMSE"]:.4f}'
                  f'  [{time.time()-t0:.0f}s]', flush=True)
        a = np.array([res[key][pk][str(s)] for s in SEEDS])
        print(f'{key:<38} rand_phase={pw!s:<5} mean {a.mean():.4f}  sd {a.std(ddof=1):.4f}  '
              f'CV {100*a.std(ddof=1)/a.mean():5.1f}%  min {a.min():.4f}  max {a.max():.4f}',
              flush=True)
    print(f'  ({note})\n', flush=True)
CTF.GROW_RANDOM_PHI = True
INIT.KONLY_RANDOM_WPHI = True
