"""NEUTRAL: is H3 (ORACLE - ORACLE-K < 0) really a REPRODUCTION?
Re-derives the prior estimate from tmp/rx2_init_2026-08-05/results.jsonl myself with
the K6 sec.4 convention, and puts it next to the K6 estimate.
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    R = [json.loads(l) for l in open(path)]
    seeds = sorted({r['seed'] for r in R})
    eps = sorted({r['episode'] for r in R})
    raw = {(r['cell'], r['episode'], r['seed']): r for r in R}
    return R, seeds, eps, raw


def sf(k, n):
    return sum(math.comb(n, i) * 0.5 ** n for i in range(k, n + 1))


def contrast(raw, seeds, eps, c, ref, m='nRMSE'):
    d = np.array([np.median([raw[(c, e, s)][m] for s in seeds])
                  - np.median([raw[(ref, e, s)][m] for s in seeds]) for e in eps])
    w, l = int((d < 0).sum()), int((d > 0).sum())
    p = 1.0 if w + l == 0 else min(1.0, 2 * sf(max(w, l), w + l))
    return np.median(d), w, w + l, p, d.std(ddof=1)


P = os.path.join(HERE, '..', 'rx2_init_2026-08-05', 'results.jsonl')
Ro, so, eo, ro = load(P)
Rk, sk, ek, rk = load(os.path.join(HERE, 'results.jsonl'))
NULLk = {r['episode']: r for r in json.load(open(os.path.join(HERE, 'nulls_k6.json')))}
BLOCK = [e for e in ek
         if np.median([rk[('AUTO', e, s)]['nRMSE'] for s in sk])
         < NULLk[e]['zero_null_tracking_nRMSE_mean']]

print('=' * 88)
print('H3 -- "ORACLE - ORACLE-K is negative again on an independent sample"')
print('=' * 88)
print(f'prior batch tmp/rx2_init_2026-08-05: seeds {so}, {len(eo)} episodes '
      f'(crowded-4, finemesh 2026-07-23)')
for e in eo:
    print(f'   {e}')
for m in ('nRMSE', 'init_nrmse'):
    a = contrast(ro, so, eo, 'ORACLE', 'ORACLE-K', m)
    b = contrast(rk, sk, BLOCK, 'ORACLE', 'ORACLE-K', m)
    print(f'\n  {m}')
    print(f'    prior (crowded-4,   n=4 ):  median {a[0]:+.5f}  {a[1]}/{a[2]}  '
          f'p={a[3]:.4f} (FLOOR 0.125)  sd(D)={a[4]:.5f}')
    print(f'    K6    (indep-14,    n=14):  median {b[0]:+.5f}  {b[1]}/{b[2]}  '
          f'p={b[3]:.5f}                 sd(D)={b[4]:.5f}')

print('\n  ALSO in the prior batch, for the ladder the memory quotes:')
for c in ['X-BASE', 'AUTO', 'ORACLE-K', 'ORACLE']:
    v = np.median([np.median([ro[(c, e, s)]['nRMSE'] for s in so]) for e in eo])
    print(f'    {c:9s} block median nRMSE {v:.5f}')
print('  (these are MEDIANS OF MEDIANS -- the frozen sec.4 statistic is the median of')
print('   PAIRED deltas, which is NOT the difference of these numbers.  Example: on K6')
a = np.median([np.median([rk[("ORACLE-K", e, s)]["nRMSE"] for s in sk]) for e in BLOCK])
b = np.median([np.median([rk[("AUTO", e, s)]["nRMSE"] for s in sk]) for e in BLOCK])
c = contrast(rk, sk, BLOCK, 'ORACLE-K', 'AUTO')
print(f'   difference of block medians = {a-b:+.5f} but median paired delta = '
      f'{c[0]:+.5f} -- OPPOSITE SIGN.)')
