"""
prune_threshold_fix.py — can the bimodal basin be closed at its identified cause?

Mechanism (fable's independent verification, tmp/VARIANCE_VERIFY_fable_2026-07-23.md):
the fork is NOT initialization (`init_nrmse` ~0.047 either way). It is the one-shot
`_adaptive_prune_grow` decision (sequential_update.py:395-456), which discards the
lowest-energy components down to a `1 - prune_threshold` energy fraction, evaluated only
against the short init window. On BF3 2comp an unlucky initial phase draw lets the weaker
true component fall inside the discarded energy tail, collapsing N=2 -> N=1 permanently.
Its residual-rollback guard (fires only if RMS residual worsens by >30%) does not catch it,
and independently we measured that the bad basin actually fits the observations *better*
(tmp/RESTART_SELECTION_RESULT_2026-07-23.md) -- so no observation-based guard can.

fable also showed `adaptive_n=False` is NOT the fix: CV only 53.0 -> 43.5% while mean nRMSE
gets 32% worse. So the target is the *threshold*, not the mechanism's existence.

Pre-registered decision rule (before the numbers):
  Sweep `prune_threshold` (default 0.10) down. Success = CV on BF3 2comp drops substantially
  (say <15%) AND mean nRMSE does not worsen on the crowded episode BF4 15comp (where an
  over-permissive threshold should cause overfitting, per fable's adaptive_n=False result).
  If CV drops but the crowded episode degrades, report it as a TRADE-OFF, not a fix, and
  give the exchange rate. If CV does not drop, the energy threshold is not the lever and the
  fix must target the rollback guard or the prune's evaluation window instead.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from wavesplat.run import run_estimation

RESULT = 'tmp/prune_threshold_fix_results.json'
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7]
THRESHOLDS = [0.10, 0.02, 0.005]
EPISODES = [
    ('data/episode_BF3_2comp_seed0_uniform.h5', 'the CV=53% bimodal case'),
    ('data/episode_BF4_15comp_seed0_uniform.h5', 'crowded control - watch for overfitting'),
]

res = json.load(open(RESULT)) if os.path.exists(RESULT) else {}

for ep, note in EPISODES:
    key = os.path.basename(ep)
    res.setdefault(key, {})
    for th in THRESHOLDS:
        tk = str(th)
        res[key].setdefault(tk, {})
        for s in SEEDS:
            if str(s) in res[key][tk]:
                continue
            t0 = time.time()
            out = run_estimation(ep, config={
                'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
                'adaptive_n': True, 'tracking_window': 3, 'seed': s,
                'prune_threshold': th})
            res[key][tk][str(s)] = float(out['avg_nRMSE'])
            json.dump(res, open(RESULT, 'w'), indent=2)
            print(f'  {key:<38} thr={th:<6} seed={s:<3} nRMSE={out["avg_nRMSE"]:.4f}'
                  f'  [{time.time()-t0:.0f}s]', flush=True)
        a = np.array([res[key][tk][str(s)] for s in SEEDS])
        print(f'{key:<38} thr={th:<6} mean {a.mean():.4f}  sd {a.std(ddof=1):.4f}  '
              f'CV {100*a.std(ddof=1)/a.mean():5.1f}%  min {a.min():.4f}  max {a.max():.4f}',
              flush=True)
    print(f'  ({note})\n', flush=True)
