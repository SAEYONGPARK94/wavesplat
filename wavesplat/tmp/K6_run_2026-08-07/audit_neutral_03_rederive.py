"""NEUTRAL independent re-derivation of every K6 headline (PREREG sec.4 / sec.5).

Written from the PREREG text only; does not import or read any scorer.
Aggregation, frozen in PREREG sec.4:
   per (cell, episode): median over the 3 seeds
   per pair: Delta = median_seed(cell) - median_seed(reference)      [SIGNED]
   headline: median over episodes of Delta
   test: paired two-sided EXACT sign test, TIES (Delta == 0) DROPPED FROM n
Degradation direction: nRMSE / init_nrmse / absVR / m / decorr / shrink -> larger worse
                       Pearson_rho -> smaller worse (sign flipped so that "win" = better)
Null rule sec.4: episodes where the CONTROL cell AUTO fails the flat-sea null are
excluded from PRIMARY; the un-excluded result is reported alongside.
"""
import json
import math
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl'))]
NULL = {r['episode']: r for r in json.load(open(os.path.join(HERE, 'nulls_k6.json')))}

EPS = sorted({r['episode'] for r in R})
SEEDS = sorted({r['seed'] for r in R})
CELLS = ['AUTO', 'AUTO-DUP', 'K6-KLR03', 'K6-KLR01', 'K6-KLR00',
         'K6-NSCHED', 'ORACLE-K', 'ORACLE']

# metric -> (+1 if larger is worse, -1 if smaller is worse)
METRICS = {'nRMSE': +1, 'init_nrmse': +1, 'Pearson_rho': -1, 'absVR_framewise': +1,
           'm_mean': +1, 'decorr_mean': +1, 'shrink_mean': +1}

V = defaultdict(dict)     # V[(cell, metric)][episode] = median over seeds
raw = defaultdict(dict)
for r in R:
    raw[(r['cell'], r['episode'])][r['seed']] = r
for c in CELLS:
    for m in METRICS:
        for e in EPS:
            V[(c, m)][e] = float(np.median([raw[(c, e)][s][m] for s in SEEDS]))


def binom_sf(k, n, p=0.5):
    """P(X >= k) for Binomial(n, p)."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def sign_test(wins, losses):
    """Exact two-sided sign test on n = wins + losses (ties already dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    return min(1.0, 2.0 * binom_sf(k, n))


def contrast(cell, ref, metric, episodes):
    sgn = METRICS[metric]
    d = {e: sgn * (V[(cell, metric)][e] - V[(ref, metric)][e]) for e in episodes}
    vals = np.array([d[e] for e in episodes])
    wins = int((vals < 0).sum())      # negative = cell better
    loss = int((vals > 0).sum())
    ties = int((vals == 0).sum())
    return dict(median=float(np.median(vals)), mean=float(vals.mean()),
                wins=wins, losses=loss, ties=ties, n=wins + loss,
                p=sign_test(wins, loss), per_ep=d,
                std=float(vals.std(ddof=1)) if len(vals) > 1 else float('nan'))


# ---------------------------------------------------------------- null rule
print('=' * 78)
print('SEC.4 NULL RULE -- flat-sea null, control cell AUTO')
print('=' * 78)
print(f'{"episode":36s} {"AUTO nRMSE":>11s} {"null":>9s} {"margin":>9s}  keep?')
keep = []
for e in EPS:
    a = V[('AUTO', 'nRMSE')][e]
    z = NULL[e]['zero_null_tracking_nRMSE_mean']
    ok = a < z
    keep.append(e) if ok else None
    print(f'{e[8:-3]:36s} {a:11.5f} {z:9.5f} {a-z:+9.5f}  {"KEEP" if ok else "** DROP **"}')
print(f'\nkept {len(keep)}/16 episodes;  dropped: '
      f'{[e[8:-3] for e in EPS if e not in keep] or "none"}')

# same check on init_nrmse (panel only)
bad_init = [e for e in EPS
            if V[('AUTO', 'init_nrmse')][e] >= NULL[e]['zero_null_init_nrmse']]
print(f'(panel) AUTO init_nrmse fails its own null on: '
      f'{[e[8:-3] for e in bad_init] or "none"}')
# and how every cell does vs the null
print('\nnull-beat count per cell (nRMSE, out of 16):')
for c in CELLS:
    n_ok = sum(V[(c, 'nRMSE')][e] < NULL[e]['zero_null_tracking_nRMSE_mean'] for e in EPS)
    med = np.median([V[(c, 'nRMSE')][e] for e in EPS])
    print(f'  {c:10s} beats null {n_ok:2d}/16   median nRMSE = {med:.5f}')

BLOCK = keep


def table(title, pairs, metric='nRMSE', episodes=None):
    episodes = episodes or BLOCK
    print('\n' + '=' * 78)
    print(f'{title}   [metric={metric}, n_ep={len(episodes)}]')
    print('=' * 78)
    print(f'{"contrast":28s} {"medianD":>9s} {"meanD":>9s} {"W-L-T":>10s} '
          f'{"n":>3s} {"p(2-sided)":>11s}')
    out = {}
    for cell, ref in pairs:
        r = contrast(cell, ref, metric, episodes)
        out[(cell, ref)] = r
        print(f'{cell+" - "+ref:28s} {r["median"]:+9.5f} {r["mean"]:+9.5f} '
              f'{r["wins"]:3d}-{r["losses"]:2d}-{r["ties"]:2d} {r["n"]:3d} '
              f'{r["p"]:11.5f}')
    return out


VS_AUTO = [(c, 'AUTO') for c in
           ['AUTO-DUP', 'K6-KLR03', 'K6-KLR01', 'K6-KLR00',
            'K6-NSCHED', 'ORACLE-K', 'ORACLE']]
CORE = [('ORACLE-K', 'K6-NSCHED'), ('ORACLE', 'ORACLE-K'),
        ('K6-KLR03', 'AUTO'), ('K6-KLR01', 'AUTO'), ('K6-KLR00', 'AUTO')]

A = table('A. ALL CELLS vs AUTO  (null-excluded block)', VS_AUTO)
B = table('B. ALL CELLS vs AUTO  (NO null exclusion, all 16)', VS_AUTO, episodes=EPS)
C = table('C. CORE CONTRASTS (each exactly ONE knob)', CORE)
C16 = table('C16. CORE CONTRASTS, no null exclusion', CORE, episodes=EPS)

for m in ['init_nrmse', 'Pearson_rho', 'absVR_framewise', 'm_mean',
          'decorr_mean', 'shrink_mean']:
    table(f'PANEL {m}  (sign flipped so negative = better)',
          VS_AUTO + [('ORACLE-K', 'K6-NSCHED'), ('ORACLE', 'ORACLE-K')], metric=m)

# ------------------------------------------------- sec.5 PRIMARY verdict
print('\n' + '=' * 78)
print('SEC.5 PRIMARY:  K6-KLR01 vs AUTO, nRMSE, null-excluded crowded-16')
print('=' * 78)
r = C[('K6-KLR01', 'AUTO')]
n = len(BLOCK)
need = {16: 13, 15: 12, 14: 12, 13: 11, 12: 10, 11: 10, 10: 9}.get(r['n'], None)
print(f'  median Delta = {r["median"]:+.6f}   (PASS needs < 0)')
print(f'  wins         = {r["wins"]}/{r["n"]}  (ties dropped: {r["ties"]})'
      f'   PASS needs >= 13/16')
print(f'  exact 2-sided sign-test p = {r["p"]:.5f}')
print(f'  pre-frozen threshold for n={r["n"]}: {need} wins')
verdict = 'PASS' if (r['median'] < 0 and r['wins'] >= 13 and r['n'] == 16) else 'FAIL'
print(f'  ==> {verdict}')
print('  per-episode Delta:')
for e in BLOCK:
    print(f'    {e[8:-3]:36s} {r["per_ep"][e]:+.6f}')

# ------------------------------------------------- config folds (n=4)
print('\n' + '=' * 78)
print('CONFIG FOLDS (n=4)  -- "significant" is FORBIDDEN here (floor p = 0.125)')
print('=' * 78)
FOLDS = defaultdict(list)
for e in EPS:
    FOLDS[e.split('_seed')[0]].append(e)
for cell, ref in CORE + [('AUTO-DUP', 'AUTO')]:
    line = []
    for fk in sorted(FOLDS):
        ee = [e for e in FOLDS[fk] if e in BLOCK]
        if not ee:
            line.append(f'{fk[8:]}: -')
            continue
        rr = contrast(cell, ref, 'nRMSE', ee)
        line.append(f'{fk[8:]}: {rr["median"]:+.4f}({rr["wins"]}/{rr["n"]})')
    fold_meds = []
    for fk in sorted(FOLDS):
        ee = [e for e in FOLDS[fk] if e in BLOCK]
        if ee:
            fold_meds.append(contrast(cell, ref, 'nRMSE', ee)['median'])
    fm = np.array(fold_meds)
    print(f'{cell+" - "+ref:26s} ' + '  '.join(line))
    print(f'{"":26s} fold-median of medians = {np.median(fm):+.5f}   '
          f'folds better = {(fm<0).sum()}/{len(fm)}  (floor p=0.125)')

# ------------------------------------------------- budget ladder (medians)
print('\n' + '=' * 78)
print('BUDGET LADDER -- block medians of nRMSE (null-excluded block)')
print('=' * 78)
for c in CELLS:
    med = np.median([V[(c, 'nRMSE')][e] for e in BLOCK])
    med_i = np.median([V[(c, 'init_nrmse')][e] for e in BLOCK])
    print(f'  {c:10s} nRMSE {med:.5f}   init_nrmse {med_i:.5f}')

json.dump({'block': BLOCK,
           'primary': {k: v for k, v in r.items() if k != 'per_ep'},
           'A_vs_auto': {f'{a}-{b}': {k: v for k, v in x.items() if k != 'per_ep'}
                         for (a, b), x in A.items()},
           'C_core': {f'{a}-{b}': {k: v for k, v in x.items() if k != 'per_ep'}
                      for (a, b), x in C.items()}},
          open(os.path.join(HERE, 'audit_neutral_rederived.json'), 'w'), indent=1)
