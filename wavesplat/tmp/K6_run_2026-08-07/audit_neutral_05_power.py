"""NEUTRAL: power / MDE for every K6 headline, and the B2<->K6 config identity check.

Two MDE scales are reported because the pre-registered decision rule is a SIGN test,
whose effect size is a win-probability, while the reader wants nRMSE units:
  (a) MDE in win-probability p_win: the smallest p_win at which the frozen win bar is
      cleared with power 0.80 (and 0.50), by exact binomial.
  (b) MDE in nRMSE units: convert with the observed paired-Delta scatter.  Under
      Delta ~ N(mu, sd) we have p_win = Phi(-mu/sd), so mu_MDE = -sd * Phi^-1(p_win).
      sd is taken from the cell's own observed paired Deltas (so this is a
      self-calibrated, not an assumed, MDE).
Also reports the seed-to-seed reproducibility floor: how big the Delta between two
runs of the SAME cell is, which is a hard lower bound on any detectable effect.
"""
import json
import math
import os
from statistics import NormalDist

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl'))]
NULL = {r['episode']: r for r in json.load(open(os.path.join(HERE, 'nulls_k6.json')))}
EPS = sorted({r['episode'] for r in R})
SEEDS = sorted({r['seed'] for r in R})
raw = {(r['cell'], r['episode'], r['seed']): r for r in R}
ND = NormalDist()


def med(c, e, m='nRMSE'):
    return float(np.median([raw[(c, e, s)][m] for s in SEEDS]))


BLOCK = [e for e in EPS if med('AUTO', e) < NULL[e]['zero_null_tracking_nRMSE_mean']]
BAR = {16: 13, 15: 12, 14: 12, 13: 11, 12: 10, 11: 10, 10: 9}


def binom_sf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def mde_pwin(n, power):
    """smallest p_win with P(wins >= BAR[n]) >= power."""
    k = BAR[n]
    lo, hi = 0.5, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if binom_sf(k, n, mid) >= power:
            hi = mid
        else:
            lo = mid
    return hi


print('=' * 84)
print('PART 1 -- MDE of the frozen sign-test bar')
print('=' * 84)
for n in (16, 14, 12):
    k = BAR[n]
    print(f'  n={n:2d}  bar={k} wins   alpha(exact 2-sided at bar)='
          f'{2*binom_sf(k,n,0.5):.5f}   '
          f'MDE p_win @power .80 = {mde_pwin(n,0.80):.3f}   '
          f'@power .50 = {mde_pwin(n,0.50):.3f}')

print('\n' + '=' * 84)
print('PART 2 -- observed effect vs self-calibrated MDE, per contrast')
print('=' * 84)
print(f'{"contrast":26s} {"n":>3s} {"medD":>9s} {"meanD":>9s} {"sd(D)":>8s} '
      f'{"wins":>6s} {"p_win_hat":>9s} {"MDE p_win":>9s} {"MDE nRMSE":>10s} {"verdict":>22s}')
CONTRASTS = [('K6-KLR01', 'AUTO'), ('K6-KLR03', 'AUTO'), ('K6-KLR00', 'AUTO'),
             ('ORACLE-K', 'AUTO'), ('ORACLE', 'AUTO'), ('K6-NSCHED', 'AUTO'),
             ('ORACLE-K', 'K6-NSCHED'), ('ORACLE', 'ORACLE-K'), ('AUTO-DUP', 'AUTO')]
for c, ref in CONTRASTS:
    d = np.array([med(c, e) - med(ref, e) for e in BLOCK])
    w = int((d < 0).sum())
    t = int((d == 0).sum())
    n = len(d) - t
    sd = d.std(ddof=1)
    if n == 0:
        print(f'{c+" - "+ref:26s} {n:3d}  {np.median(d):+8.5f} {d.mean():+9.5f} '
              f'{sd:8.5f} {"n/a":>6s} {"n/a":>9s} {"n/a":>9s} {"n/a":>10s} '
              f'{"ALL TIES (byte-dup)":>22s}')
        continue
    pw = mde_pwin(n, 0.80)
    mde_mu = sd * ND.inv_cdf(pw)          # magnitude of |mu| needed
    obs = abs(d.mean())
    ver = ('DETECTED' if w >= BAR[n]
           else ('measured null (|eff| << MDE)' if obs < 0.35 * mde_mu
                 else 'UNDERPOWERED'))
    print(f'{c+" - "+ref:26s} {n:3d}  {np.median(d):+8.5f} {d.mean():+9.5f} {sd:8.5f} '
          f'{w:3d}/{n:<2d} {w/n:9.3f} {pw:9.3f} {mde_mu:10.5f} {ver:>22s}')

print('\n' + '=' * 84)
print('PART 3 -- reproducibility floor: same-cell seed-to-seed spread')
print('=' * 84)
for c in ['AUTO', 'K6-KLR01', 'ORACLE-K', 'ORACLE']:
    spreads = []
    for e in EPS:
        v = [raw[(c, e, s)]['nRMSE'] for s in SEEDS]
        spreads.append(max(v) - min(v))
    # half-range of a median-of-3 approximates its own sampling sd
    print(f'  {c:10s} per-episode seed RANGE of nRMSE: median {np.median(spreads):.5f}  '
          f'max {max(spreads):.5f}')
d_auto = np.array([med('AUTO-DUP', e) - med('AUTO', e) for e in EPS])
print(f'  AUTO-DUP - AUTO paired Delta: max |.| = {np.abs(d_auto).max():.3e} '
      f'(byte-duplicate => the ONLY zero-noise contrast in the batch)')
print('  NOTE: seed range is NOT the noise of the seed-MEDIAN, but it bounds it; '
      'the PRIMARY |median Delta| = 0.00074 is ~1/30 of the median seed range.')

# how often does a 1-seed swap flip the PRIMARY sign?
print('\n  PRIMARY stability under seed choice (K6-KLR01 - AUTO, block-median Delta):')
for s in SEEDS:
    d1 = np.array([raw[('K6-KLR01', e, s)]['nRMSE'] - raw[('AUTO', e, s)]['nRMSE']
                   for e in BLOCK])
    print(f'    seed {s} alone: median {np.median(d1):+.5f}  wins {(d1<0).sum()}/{len(d1)}')
for drop in SEEDS:
    ss = [s for s in SEEDS if s != drop]
    d2 = np.array([float(np.mean([raw[('K6-KLR01', e, s)]['nRMSE'] for s in ss]))
                   - float(np.mean([raw[('AUTO', e, s)]['nRMSE'] for s in ss]))
                   for e in BLOCK])
    print(f'    drop seed {drop} (mean of 2): median {np.median(d2):+.5f}  '
          f'wins {(d2<0).sum()}/{len(d2)}')

print('\n' + '=' * 84)
print('PART 4 -- B2 <-> K6 cell-config identity (are AUTO / ORACLE-K the same arm?)')
print('=' * 84)
B2 = os.path.join(HERE, '..', 'B2_g1p1_ext_2026-08-06', 'results.jsonl')
RB = [json.loads(l) for l in open(B2)]
for cell in ('AUTO', 'ORACLE-K'):
    kb = [r['cfg'] for r in RB if r['cell'] == cell
          and r['episode'] == 'episode_BF4_15comp_seed0_uniform.h5'][0]
    kk = [r['cfg'] for r in R if r['cell'] == cell
          and r['episode'] == 'episode_BF4_15comp_seed0_uniform_ra_rf_ds.h5'][0]
    dif = {k: (kb.get(k, '<absent>'), kk.get(k, '<absent>'))
           for k in set(kb) | set(kk) if kb.get(k) != kk.get(k)}
    print(f'  {cell:9s} cfg diff B2 vs K6 (15comp episode): {dif or "IDENTICAL"}')
extra = set().union(*[set(r) for r in RB[:1]]) ^ set(R[0])
print(f'  record-schema difference B2 vs K6: {sorted(extra)}')

# which K6 AUTO runs fell back to v2_konly?
fb = [(r['episode'][8:-3], r['seed']) for r in R
      if r['cell'] == 'AUTO' and r.get('auto_selected') == 'v2_konly']
print(f'\n  K6 cell AUTO: router fell back to v2_konly on {len(fb)}/48 runs: {fb}')
