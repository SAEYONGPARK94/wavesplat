#!/usr/bin/env python3
"""R2/P3 task A: is the BS-1 30-episode subset a biased draw?

The 30 are NOT a random stratified sample: bs1_shipping_default.py:33 fixes
SUFF = [seed0_uniform, seed0_slope, seed1_uniform] for every cell.  So the
same two wave seeds (0 and 1) carry all 10 cells, and seed 0 is counted twice
(its uniform and slope raydrop masks are the SAME wave field).

Everything below uses only files that already exist:
  results/100ep_ws_letkf.json                      (published WS = v2_konly, LETKF)
  tmp/BLANKSLATE_2026-08-12/letkf_framematch.json  (frame-matched LETKF + zero)
"""
import os, sys, json, math, itertools
import numpy as np
from scipy import stats

ROOT = '/home/vexit/wavesplat/wavesplat'
FM = json.load(open(os.path.join(ROOT, 'tmp', 'BLANKSLATE_2026-08-12',
                                 'letkf_framematch.json')))['per_episode']
CELLS = ['BF2_1comp', 'BF3_1comp', 'BF3_2comp', 'BF3_5comp', 'BF4_1comp',
         'BF4_5comp', 'BF4_15comp', 'BF4_30comp', 'BF5_15comp', 'BF5_30comp']
SUFF = ['seed0_uniform', 'seed0_slope', 'seed1_uniform']
IN30 = {f'{c}_{s}' for c in CELLS for s in SUFF}


def sign_test(w, n):
    return stats.binomtest(w, n, 0.5).pvalue


print('=' * 78)
print('A1. Is the 30-subset a biased draw?  (published v2_konly arm, all 100)')
print('=' * 78)
rows = []
for k, v in FM.items():
    rows.append(dict(k=k, ncomp=v['ncomp'], ws=v['ws'],
                     letkf=v['letkf_match'], zero=v['zero_match'],
                     seed=int(k.split('seed')[1].split('_')[0]),
                     rd=k.split('_')[-1], sel=k in IN30))
assert len(rows) == 100 and sum(r['sel'] for r in rows) == 30

for name, sub in [('in-30', [r for r in rows if r['sel']]),
                  ('out-70', [r for r in rows if not r['sel']]),
                  ('all-100', rows)]:
    w = sum(1 for r in sub if r['ws'] < r['letkf'])
    n = len(sub)
    rel = np.median([(r['ws'] - r['letkf']) / r['letkf'] for r in sub])
    print(f'  {name:8s} n={n:3d}  WS wins {w:3d}/{n:3d}  p={sign_test(w,n):.3f}  '
          f'med_rel={100*rel:+7.1f}%  medWS={np.median([r["ws"] for r in sub]):.4f} '
          f'medLETKF={np.median([r["letkf"] for r in sub]):.4f}')

# difference of the two win rates
a = [1 if r['ws'] < r['letkf'] else 0 for r in rows if r['sel']]
b = [1 if r['ws'] < r['letkf'] else 0 for r in rows if not r['sel']]
tab = [[sum(a), len(a) - sum(a)], [sum(b), len(b) - sum(b)]]
print(f'  Fisher in30 vs out70 win-rate: {stats.fisher_exact(tab)[1]:.3f}  (table {tab})')
# effect size on the paired relative difference
ra = [(r['ws'] - r['letkf']) / r['letkf'] for r in rows if r['sel']]
rb = [(r['ws'] - r['letkf']) / r['letkf'] for r in rows if not r['sel']]
print(f'  Mann-Whitney on rel diff: p={stats.mannwhitneyu(ra, rb)[1]:.3f}')

print()
print('A2. Is seed 0/1 special?  (published arm, per wave seed)')
for s in range(5):
    sub = [r for r in rows if r['seed'] == s]
    w = sum(1 for r in sub if r['ws'] < r['letkf'])
    rel = np.median([(r['ws'] - r['letkf']) / r['letkf'] for r in sub])
    print(f'  seed{s}: n={len(sub)}  WS wins {w:2d}  med_rel={100*rel:+7.1f}%  '
          f'medWS={np.median([r["ws"] for r in sub]):.4f}')
kw = stats.kruskal(*[[(r['ws'] - r['letkf']) / r['letkf']
                      for r in rows if r['seed'] == s] for s in range(5)])
print(f'  Kruskal-Wallis across 5 wave seeds: H={kw[0]:.2f} p={kw[1]:.3f}')

print()
print('A3. LETKF difficulty by seed (is seed0/1 an EASY draw for the baseline?)')
for s in range(5):
    sub = [r for r in rows if r['seed'] == s]
    print(f'  seed{s}: medLETKF={np.median([r["letkf"] for r in sub]):.4f}  '
          f'medZERO={np.median([r["zero"] for r in sub]):.4f}  '
          f'medLETKF/ZERO={np.median([r["letkf"]/r["zero"] for r in sub]):.3f}')
kw = stats.kruskal(*[[r['letkf'] / r['zero'] for r in rows if r['seed'] == s]
                     for s in range(5)])
print(f'  Kruskal-Wallis LETKF/ZERO across seeds: H={kw[0]:.2f} p={kw[1]:.3f}')
print('  in-30  medLETKF/ZERO = %.3f' % np.median([r['letkf'] / r['zero'] for r in rows if r['sel']]))
print('  out-70 medLETKF/ZERO = %.3f' % np.median([r['letkf'] / r['zero'] for r in rows if not r['sel']]))

print()
print('A4. Paired-seed dependency inside the 30: seed0_uniform & seed0_slope')
print('    share the SAME wave field (only the raydrop mask differs).')
same = []
for c in CELLS:
    u, s = FM.get(f'{c}_seed0_uniform'), FM.get(f'{c}_seed0_slope')
    if u and s:
        same.append(((u['ws'] - u['letkf_match']) / u['letkf_match'],
                     (s['ws'] - s['letkf_match']) / s['letkf_match']))
x, y = zip(*same)
print(f'  Spearman(uniform, slope) of rel-diff over 10 cells: '
      f'rho={stats.spearmanr(x, y)[0]:.3f} p={stats.spearmanr(x, y)[1]:.3f}')
print(f'  sign agreement: {sum(1 for a_,b_ in same if (a_<0)==(b_<0))}/10')
print('  => effective independent wave draws in the 30 = 10 cells x 2 seeds = 20,')
print('     NOT 30.  Any sign test on n=30 overstates n by ~1.5x.')
