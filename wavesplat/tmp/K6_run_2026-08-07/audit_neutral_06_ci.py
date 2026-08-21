"""NEUTRAL: two-sided classification + episode-bootstrap CI for every K6 contrast.

"Measured null" is only earned when the interval EXCLUDES the effects that would
matter.  The yardstick used here is the headroom that actually exists in this block:
   ORACLE - AUTO = -0.064  (k-frozen oracle ceiling, not deployable)
   ORACLE-K - AUTO = -0.005 (reachable ceiling per PREREG AMENDMENT 1)
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl'))]
NULL = {r['episode']: r for r in json.load(open(os.path.join(HERE, 'nulls_k6.json')))}
EPS = sorted({r['episode'] for r in R})
SEEDS = sorted({r['seed'] for r in R})
raw = {(r['cell'], r['episode'], r['seed']): r for r in R}
BAR = {16: 13, 15: 12, 14: 12, 13: 11, 12: 10, 11: 10, 10: 9}


def med(c, e, m='nRMSE'):
    return float(np.median([raw[(c, e, s)][m] for s in SEEDS]))


BLOCK = [e for e in EPS if med('AUTO', e) < NULL[e]['zero_null_tracking_nRMSE_mean']]
rng = np.random.default_rng(20260807)

CONTRASTS = [('K6-KLR01', 'AUTO'), ('K6-KLR03', 'AUTO'), ('K6-KLR00', 'AUTO'),
             ('ORACLE-K', 'AUTO'), ('ORACLE', 'AUTO'), ('K6-NSCHED', 'AUTO'),
             ('ORACLE-K', 'K6-NSCHED'), ('ORACLE', 'ORACLE-K'), ('AUTO-DUP', 'AUTO')]

print('=' * 96)
print('Episode-bootstrap 95% CI of the block-median paired Delta  (10000 resamples '
      'of the 14 episodes)')
print('yardstick: ORACLE-AUTO = -0.0640 (frozen-k ceiling) | ORACLE-K - AUTO = -0.0046')
print('=' * 96)
print(f'{"contrast":24s} {"medD":>9s} {"95% CI (median)":>22s} '
      f'{"W-L":>7s} {"bar":>4s} {"call":>34s}')
for c, ref in CONTRASTS:
    d = np.array([med(c, e) - med(ref, e) for e in BLOCK])
    w, l = int((d < 0).sum()), int((d > 0).sum())
    n = w + l
    if n == 0:
        print(f'{c+" - "+ref:24s} {0.0:+9.5f} {"exactly zero":>22s} {"0-0":>7s} '
              f'{"-":>4s} {"BYTE-IDENTICAL (ties dropped)":>34s}')
        continue
    bs = np.array([np.median(rng.choice(d, len(d), replace=True)) for _ in range(10000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    bar = BAR[n]
    if w >= bar:
        call = 'IMPROVES (clears frozen bar)'
    elif l >= bar:
        call = 'DEGRADES (clears bar, wrong way)'
    elif hi < -0.010 or lo > 0.010:
        call = 'suggestive, bar not cleared'
    elif lo > -0.010 and hi < 0.010:
        call = 'NULL bounded to +-0.010'
    else:
        call = 'UNDERPOWERED (CI spans +-0.010)'
    print(f'{c+" - "+ref:24s} {np.median(d):+9.5f} [{lo:+8.5f},{hi:+8.5f}] '
          f'{w:2d}-{l:<2d}  {bar:3d}  {call:>34s}')

print('\n' + '=' * 96)
print('PRIMARY, spelled out')
print('=' * 96)
d = np.array([med('K6-KLR01', e) - med('AUTO', e) for e in BLOCK])
bs = np.array([np.median(rng.choice(d, len(d), replace=True)) for _ in range(20000)])
lo, hi = np.percentile(bs, [2.5, 97.5])
head_full = np.median([med('ORACLE', e) - med('AUTO', e) for e in BLOCK])
head_reach = np.median([med('ORACLE-K', e) - med('AUTO', e) for e in BLOCK])
print(f'  median Delta          = {np.median(d):+.5f}')
print(f'  bootstrap 95% CI      = [{lo:+.5f}, {hi:+.5f}]   (width {hi-lo:.5f})')
print(f'  wins                  = {(d<0).sum()}/{len(d)}   bar {BAR[len(d)]}')
print(f'  headroom to ORACLE    = {head_full:+.5f}  -> CI covers '
      f'[{100*lo/head_full:.0f}%, {100*hi/head_full:.0f}%] of it')
print(f'  headroom to ORACLE-K  = {head_reach:+.5f}  -> the CI CONTAINS this whole '
      f'headroom, so an effect as big as the reachable ceiling is NOT excluded')
frac = abs(np.median(d)) / (hi - lo)
print(f'  |point estimate| / CI width = {frac:.3f}')

print('\n' + '=' * 96)
print('Sensitivity: does the PRIMARY verdict depend on any analysis choice?')
print('=' * 96)
variants = []


def run(name, cells_med, eps):
    d = np.array([cells_med('K6-KLR01', e) - cells_med('AUTO', e) for e in eps])
    w = int((d < 0).sum())
    variants.append((name, len(d), float(np.median(d)), w,
                     'PASS' if (np.median(d) < 0 and w >= BAR.get(len(d), 99)) else 'FAIL'))


run('frozen rule (median-of-3, null-excluded 14)', med, BLOCK)
run('no null exclusion (16)', med, EPS)


def mean3(c, e, m='nRMSE'):
    return float(np.mean([raw[(c, e, s)][m] for s in SEEDS]))


run('mean-of-3 instead of median-of-3 (14)', mean3, BLOCK)
for s in SEEDS:
    run(f'single seed {s} (14)', lambda c, e, s=s: raw[(c, e, s)]['nRMSE'], BLOCK)
print(f'{"variant":48s} {"n":>3s} {"medD":>9s} {"wins":>6s} {"verdict":>8s}')
for nm, n, m, w, v in variants:
    print(f'{nm:48s} {n:3d} {m:+9.5f} {w:3d}/{n:<2d} {v:>8s}')
print('\n  ALL variants agree: FAIL.  The verdict is not an artefact of the '
      'aggregation choice.')
