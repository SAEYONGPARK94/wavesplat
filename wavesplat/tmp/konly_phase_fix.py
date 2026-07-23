"""
konly_phase_fix.py — the actual source of the bimodal variance, and its fix.

Found by reading `initialize_from_frames` (initializer.py, the 'v2_konly' branch): v2
computes (w, phi) for every detected component by least-squares linear projection onto the
data, and the 'v2_konly' path -- the project's documented default -- THROWS THEM AWAY and
substitutes w = 0.01 (uniform) and phi ~ Uniform(0, 2*pi). The comment says this is
"so optimizer explores the amplitude/phase landscape freely".

That random phase draw is the only stochastic input to the estimator, which matches every
measurement so far: fable's phase-freeze collapsed CV 53% -> 0.0%
(tmp/VARIANCE_VERIFY_fable_2026-07-23.md); prune_threshold is not the lever
(tmp/prune_threshold_fix.py: identical min/max across a 20x sweep); and padding amplitude is
not either (tmp/pad_w_fix.py: byte-identical results at PAD_W 0.01 vs 0.0, i.e. the padding
branch never even executes for these episodes).

Test: keep v2's fitted (w, phi) instead of re-drawing (INIT.KONLY_RANDOM_WPHI = False).

Pre-registered decision rule (before the numbers):
  - CV on BF3 2comp collapses AND mean nRMSE does not worsen on either episode
    -> FIX FOUND: a deterministic estimator, and the "explore freely" rationale is wrong.
  - CV collapses but mean worsens -> TRADE-OFF: the random restart was buying exploration;
    report the exchange rate and reconsider median-of-k instead.
  - CV does not collapse -> REFUTED; something else is stochastic and I have mis-read the
    code path. Report plainly.
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import wavesplat.optimizer.initializer as INIT
from wavesplat.run import run_estimation

RESULT = 'tmp/konly_phase_fix_results.json'
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
            print(f'  {key:<38} rand_wphi={pw!s:<5} seed={s:<3} nRMSE={out["avg_nRMSE"]:.4f}'
                  f'  [{time.time()-t0:.0f}s]', flush=True)
        a = np.array([res[key][pk][str(s)] for s in SEEDS])
        print(f'{key:<38} rand_wphi={pw!s:<5} mean {a.mean():.4f}  sd {a.std(ddof=1):.4f}  '
              f'CV {100*a.std(ddof=1)/a.mean():5.1f}%  min {a.min():.4f}  max {a.max():.4f}',
              flush=True)
    print(f'  ({note})\n', flush=True)
INIT.KONLY_RANDOM_WPHI = True
