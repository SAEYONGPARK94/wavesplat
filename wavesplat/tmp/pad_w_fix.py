"""
pad_w_fix.py — is the bimodal variance caused by ACTIVE random padding components?

Chain of evidence so far:
  1. tmp/ESTIMATOR_VARIANCE_RESULT_2026-07-23.md — bimodal run-to-run variance, CV 53% on
     BF3 2comp with only the estimator seed varying.
  2. tmp/VARIANCE_VERIFY_fable_2026-07-23.md (independent) — CONFIRMED; freezing the initial
     phase draw collapses CV to 0.0%, so the phase draw drives 100% of it. Attributed the
     fork to the one-shot energy prune in `_adaptive_prune_grow`.
  3. tmp/prune_threshold_fix.py — REFUTES the energy-threshold half of that attribution:
     sweeping prune_threshold 0.10 -> 0.02 -> 0.005 leaves CV at 53.0 / 57.4 / 57.2% with
     *identical* min and max (0.0524 / 0.1682). The energy cut is not the lever, so the
     prune is a symptom, not the cause.

New hypothesis, from reading `initialize_from_frames_v2` (initializer.py, padding block):
when the beamformer finds fewer peaks than `n_gaussians`, the model is padded with
components at **random wavenumbers** (`rng.randn`, std 1.0 -- wavelengths down to a few
metres), **random phase**, and a **non-zero amplitude of 0.01**. Those are active spurious
components. Which of them captures energy during optimisation depends on the random draw,
and the energy prune then decides who survives -- consistent with both (2) and (3).

Test: set the padding amplitude to exactly 0 (`initializer.PAD_W = 0.0`). The components
still exist and can still grow if the residual demands it (zero-init warm start), but they
cannot steer the fit from the start.

Pre-registered decision rule (before the numbers):
  - CV on BF3 2comp drops substantially (<15%) AND BF4 15comp does not degrade
    -> FIX FOUND; the defect is active random padding, and it is a one-line change.
  - CV drops but the crowded episode degrades -> TRADE-OFF; report the exchange rate.
  - CV does not drop -> hypothesis REFUTED; padding is not the cause either, and the next
    target is the initial phase of the *detected* components (from `_linear_projection`).
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import wavesplat.optimizer.initializer as INIT
from wavesplat.run import run_estimation

RESULT = 'tmp/pad_w_fix_results.json'
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7]
PAD_WS = [0.01, 0.0]          # 0.01 = current default (control), 0.0 = inert padding
EPISODES = [
    ('data/episode_BF3_2comp_seed0_uniform.h5', 'the CV=53% bimodal case'),
    ('data/episode_BF4_15comp_seed0_uniform.h5', 'crowded control'),
]

res = json.load(open(RESULT)) if os.path.exists(RESULT) else {}

for ep, note in EPISODES:
    key = os.path.basename(ep)
    res.setdefault(key, {})
    for pw in PAD_WS:
        pk = str(pw)
        res[key].setdefault(pk, {})
        INIT.PAD_W = pw
        for s in SEEDS:
            if str(s) in res[key][pk]:
                continue
            t0 = time.time()
            out = run_estimation(ep, config={
                'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
                'adaptive_n': True, 'tracking_window': 3, 'seed': s})
            res[key][pk][str(s)] = float(out['avg_nRMSE'])
            json.dump(res, open(RESULT, 'w'), indent=2)
            print(f'  {key:<38} PAD_W={pw:<5} seed={s:<3} nRMSE={out["avg_nRMSE"]:.4f}'
                  f'  [{time.time()-t0:.0f}s]', flush=True)
        a = np.array([res[key][pk][str(s)] for s in SEEDS])
        print(f'{key:<38} PAD_W={pw:<5} mean {a.mean():.4f}  sd {a.std(ddof=1):.4f}  '
              f'CV {100*a.std(ddof=1)/a.mean():5.1f}%  min {a.min():.4f}  max {a.max():.4f}',
              flush=True)
    print(f'  ({note})\n', flush=True)
INIT.PAD_W = 0.01
