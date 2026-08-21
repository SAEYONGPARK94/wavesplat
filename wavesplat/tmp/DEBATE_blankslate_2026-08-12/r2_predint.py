#!/usr/bin/env python3
"""R2/P3 task A2: per-cell PREDICTION intervals for the other 70 episodes.

Written and frozen BEFORE the 70-episode run finished (see r2_auto_rest70.log
timestamps).  Method: for each of the 10 cells we observe s_c wins out of 3
(shipping `auto` avg_nRMSE < frame-matched LETKF).  Put a Jeffreys Beta(1/2,1/2)
prior on the cell win probability, get the Beta-Binomial posterior predictive
for the remaining 7 in that cell, and convolve the 10 cell distributions to
get the predictive distribution of total wins in the held-out 70.
"""
import os, sys, json
import numpy as np
from scipy import stats
from scipy.special import betaln

ROOT = '/home/vexit/wavesplat/wavesplat'
HERE = os.path.join(ROOT, 'tmp', 'DEBATE_blankslate_2026-08-12')
FM = json.load(open(os.path.join(ROOT, 'tmp', 'BLANKSLATE_2026-08-12',
                                 'letkf_framematch.json')))['per_episode']
SD = json.load(open(os.path.join(ROOT, 'tmp', 'BLANKSLATE_2026-08-12',
                                 'shipping_default.json')))
CELLS = ['BF2_1comp', 'BF3_1comp', 'BF3_2comp', 'BF3_5comp', 'BF4_1comp',
         'BF4_5comp', 'BF4_15comp', 'BF4_30comp', 'BF5_15comp', 'BF5_30comp']
SUFF = ['seed0_uniform', 'seed0_slope', 'seed1_uniform']
A = B = 0.5   # Jeffreys


from math import comb


def bb_pmf(k, n, a, b):
    return float(np.exp(np.log(comb(n, k)) +
                        betaln(k + a, n - k + b) - betaln(a, b)))


rows = {}
for c in CELLS:
    for s in SUFF:
        k = f'{c}_{s}'
        v = SD.get(f'{k}|shipping_default')
        if v and 'avg_nRMSE' in v:
            rows[k] = (v['avg_nRMSE'], FM[k]['letkf_match'], FM[k]['zero_match'])

print('30-episode observed (shipping auto vs frame-matched LETKF):')
tot = 0
dist = np.array([1.0])
per_cell = []
for c in CELLS:
    ks = [f'{c}_{s}' for s in SUFF if f'{c}_{s}' in rows]
    s_c = sum(1 for k in ks if rows[k][0] < rows[k][1])
    n_c = len(ks)
    tot += s_c
    rem = 10 - n_c
    a, b = A + s_c, B + (n_c - s_c)
    pmf = np.array([bb_pmf(k, rem, a, b) for k in range(rem + 1)])
    pmf /= pmf.sum()
    lo = int(np.searchsorted(np.cumsum(pmf), 0.05))
    hi = int(np.searchsorted(np.cumsum(pmf), 0.95))
    mean = float((np.arange(rem + 1) * pmf).sum())
    per_cell.append((c, s_c, n_c, rem, mean, lo, hi))
    print(f'  {c:<12s} obs {s_c}/{n_c}   predicted wins in remaining {rem}: '
          f'mean {mean:.2f}  90% PI [{lo}, {hi}]')
    dist = np.convolve(dist, pmf)

cum = np.cumsum(dist)
lo = int(np.searchsorted(cum, 0.05)); hi = int(np.searchsorted(cum, 0.95))
lo50 = int(np.searchsorted(cum, 0.25)); hi50 = int(np.searchsorted(cum, 0.75))
mean = float((np.arange(len(dist)) * dist).sum())
print(f'\n30-ep observed total: {tot}/30 wins')
print(f'PREDICTED wins in the held-out 70: mean {mean:.1f}  '
      f'50% PI [{lo50}, {hi50}]  90% PI [{lo}, {hi}]')
print(f'P(held-out 70 win rate > 0.5)  = {float(dist[36:].sum()):.4f}')
print(f'P(held-out 70 wins <= 35, i.e. no better than a coin) = '
      f'{float(dist[:36].sum()):.4f}')
# what 100-episode sign test would look like
full = np.zeros(101)
for k, p in enumerate(dist):
    full[tot + k] += p
cf = np.cumsum(full)
print(f'PREDICTED 100-episode win count: mean {float((np.arange(101)*full).sum()):.1f} '
      f'90% PI [{int(np.searchsorted(cf,0.05))}, {int(np.searchsorted(cf,0.95))}]')
print(f'P(100-ep sign test significant at two-sided .05, i.e. >=61 wins) = '
      f'{float(full[61:].sum()):.4f}')
json.dump(dict(per_cell=per_cell, obs30=tot,
               pred70_mean=mean, pred70_pi90=[lo, hi], pred70_pi50=[lo50, hi50]),
          open(os.path.join(HERE, 'r2_predint.json'), 'w'), indent=1)
