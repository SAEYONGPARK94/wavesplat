"""
restart_selection.py — does GT-FREE restart selection actually pick the good basin?

Follows tmp/ESTIMATOR_VARIANCE_RESULT_2026-07-23.md, which measured bimodal run-to-run
variance (CV up to 53% on a fixed episode, 3 of 4 episodes splitting into a good and a bad
convergence basin). The proposed fix is multi-restart selecting on the **observation-fit
loss** — the only quality signal available without ground truth, hence the only one a
deployed system could use.

The whole fix hinges on one empirical question: **is obs_loss actually correlated with
nRMSE across restarts of the same episode?** If the bad basin fits the observations just as
well as the good one, selection is impossible and the honest response is to report error
bars instead.

Pre-registered decision rule (stated before the numbers):
  - Spearman rho(obs_loss, nRMSE) across seeds, per episode. Selection is USEFUL if the
    argmin-obs_loss run's nRMSE beats the *median* run's nRMSE in most episodes, and beats
    the seed-42 default too.
  - Report best-of-k under the GT-free rule vs (i) median-of-k, (ii) the current single
    seed=42 number, (iii) the ORACLE best-of-k (selecting on true nRMSE — an upper bound
    that is NOT deployable and must never be cited as a result).
  - If rho is near zero or the wrong sign: selection FAILS -> the fix is error bars /
    median-of-k reporting, not best-of-k. Say so plainly; that is a legitimate outcome.

Reuses the seeds already measured so the comparison is apples-to-apples.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import spearmanr
from wavesplat.run import run_estimation

RESULT = 'tmp/restart_selection_results.json'
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7]
EPISODES = [
    'data/episode_BF3_2comp_seed0_uniform.h5',
    'data/episode_BF3_5comp_seed0_uniform.h5',
    'data/episode_BF3_5comp_seed0_slope.h5',
    'data/episode_BF4_15comp_seed0_uniform.h5',
]

res = json.load(open(RESULT)) if os.path.exists(RESULT) else {}

for ep in EPISODES:
    key = os.path.basename(ep)
    res.setdefault(key, {})
    for s in SEEDS:
        if str(s) in res[key]:
            continue
        t0 = time.time()
        out = run_estimation(ep, config={
            'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
            'adaptive_n': True, 'tracking_window': 3, 'seed': s})
        res[key][str(s)] = {'nrmse': float(out['avg_nRMSE']),
                            'obs_loss': float(out['avg_obs_loss'])}
        json.dump(res, open(RESULT, 'w'), indent=2)
        print(f"  {key:<42} seed={s:<3} nRMSE={out['avg_nRMSE']:.4f} "
              f"obs_loss={out['avg_obs_loss']:.6g}  [{time.time()-t0:.0f}s]", flush=True)

    nr = np.array([res[key][str(s)]['nrmse'] for s in SEEDS])
    ol = np.array([res[key][str(s)]['obs_loss'] for s in SEEDS])
    rho, p = spearmanr(ol, nr)
    sel = nr[int(np.argmin(ol))]          # GT-free selection (deployable)
    print(f"{key:<42} rho(obs_loss,nRMSE)={rho:+.3f} (p={p:.3f})   "
          f"selected={sel:.4f}  median={np.median(nr):.4f}  seed42={nr[0]:.4f}  "
          f"oracle_best={nr.min():.4f}  worst={nr.max():.4f}", flush=True)
    print(flush=True)

# pooled read
allr = []
for key, d in res.items():
    nr = np.array([d[str(s)]['nrmse'] for s in SEEDS])
    ol = np.array([d[str(s)]['obs_loss'] for s in SEEDS])
    allr.append((key, spearmanr(ol, nr)[0], nr[int(np.argmin(ol))], float(np.median(nr)),
                 nr[0], nr.min()))
if allr:
    print('=== POOLED ===')
    sel = np.array([a[2] for a in allr]); med = np.array([a[3] for a in allr])
    s42 = np.array([a[4] for a in allr]); orc = np.array([a[5] for a in allr])
    print(f'  mean over {len(allr)} episodes: selected {sel.mean():.4f}  median {med.mean():.4f}  '
          f'seed42 {s42.mean():.4f}  oracle {orc.mean():.4f}')
    print(f'  selection beats median in {int((sel<med).sum())}/{len(allr)}, '
          f'beats seed42 in {int((sel<s42).sum())}/{len(allr)}')
    print(f'  rho per episode: {[f"{a[1]:+.2f}" for a in allr]}')
