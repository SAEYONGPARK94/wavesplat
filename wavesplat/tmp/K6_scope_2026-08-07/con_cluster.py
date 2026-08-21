"""CON (6) k-position -- attack 2: does the SAMPLE support "ORACLE-K opens the ceiling"?

Reads ONLY:
  tmp/B2_g1p1_ext_2026-08-06/results.jsonl   (already-run, not re-run)
  data_finemesh_2026-07-23/*.h5              (generator params + truth field, read-only)

No GPU, no estimator calls.

Re-derives, under the SAME frozen conventions as B2 PREREG (median-of-3 seeds ->
per-episode value; paired delta vs X-BASE; two-sided sign test), the ORACLE-K
verdict at three observation-unit definitions:
  (a) episode        n=12   <- what was reported ("11/12, p=0.0063")
  (b) truth-surface  n=10   <- bit-identical gt_wave_field collapsed
  (c) phase vector   n=5    <- the axis the tracker actually moves
"""
import hashlib
import json
import os
from collections import defaultdict
from math import comb

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
RES = os.path.join(REPO, 'tmp', 'B2_g1p1_ext_2026-08-06', 'results.jsonl')

CROWDED4 = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
            'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
SPARSE8 = ['episode_BF2_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed0_uniform.h5',
           'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_slope.h5',
           'episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
           'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5']
P1_12 = SPARSE8 + CROWDED4


def md5(v):
    return hashlib.md5(np.ascontiguousarray(v, dtype='<f8').tobytes()).hexdigest()[:8]


def sign_p(k, n):
    """two-sided exact sign test, k wins out of n (ties already excluded)."""
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    tail = sum(comb(n, i) for i in range(lo + 1))
    return min(1.0, 2.0 * tail / 2 ** n)


# ---------------------------------------------------------------- load results
rows = []
for line in open(RES):
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    if 'error' not in r or r.get('error') is None:
        rows.append(r)

by = defaultdict(list)
for r in rows:
    if 'nRMSE' in r and np.isfinite(r.get('nRMSE', np.nan)):
        by[(r['cell'], r['episode'])].append(r)


def med(cell, ep, key='nRMSE'):
    v = [r[key] for r in by[(cell, ep)] if np.isfinite(r.get(key, np.nan))]
    return float(np.median(v)) if v else float('nan')


def seed_range(cell, ep, key='nRMSE'):
    v = [r[key] for r in by[(cell, ep)] if np.isfinite(r.get(key, np.nan))]
    return (max(v) - min(v)) if len(v) > 1 else float('nan')


# ---------------------------------------------------------------- truth keys
field_key, phase_key, ngt = {}, {}, {}
for ep in P1_12:
    with h5py.File(os.path.join(DATA, ep), 'r') as f:
        wc = np.array(f['config/wave_components'])
        fld = np.array(f['frames/gt_wave_field'])
    field_key[ep] = md5(fld)
    phase_key[ep] = md5(wc[:, 4])
    ngt[ep] = wc.shape[0]

print('=' * 104)
print('CON-6 attack 2 -- observation units behind "ORACLE-K 11/12, p=0.0063"')
print('=' * 104)
print(f"{'episode':38s} {'N_gt':>4s} {'fieldkey':>9s} {'phasekey':>9s} "
      f"{'X-BASE':>8s} {'ORACLE-K':>9s} {'delta':>9s} {'XB seedRange':>13s} {'null':>7s}")
NULL = {}
for ep in P1_12:
    xb, ok = med('X-BASE', ep), med('ORACLE-K', ep)
    print(f'{ep[8:-3]:38s} {ngt[ep]:4d} {field_key[ep]:>9s} {phase_key[ep]:>9s} '
          f'{xb:8.4f} {ok:9.4f} {ok - xb:+9.4f} {seed_range("X-BASE", ep):13.4f}')

# ---------------------------------------------------------------- cluster tests
def cluster_test(name, eps, keyfun, cell='ORACLE-K', ctrl='X-BASE'):
    groups = defaultdict(list)
    for ep in eps:
        groups[keyfun(ep)].append(ep)
    deltas = []
    detail = []
    for k, members in sorted(groups.items()):
        d = [med(cell, e) - med(ctrl, e) for e in members]
        dm = float(np.median(d))
        deltas.append(dm)
        detail.append((k, [e[8:-3] for e in members], dm))
    n = len(deltas)
    wins = sum(1 for d in deltas if d < 0)
    ties = sum(1 for d in deltas if d == 0)
    p = sign_p(wins, n - ties)
    print(f'\n  {name}: clusters n={n}  wins={wins}/{n - ties}  '
          f'median cluster delta={np.median(deltas):+.4f}  two-sided p={p:.4f}   '
          f'(floor p = {sign_p(n - ties, n - ties):.4f})')
    for k, members, dm in detail:
        flag = 'WIN ' if dm < 0 else 'LOSS'
        print(f'      [{k}] {flag} {dm:+.4f}   ' + ', '.join(members))
    return n, wins, p


print('\n' + '=' * 104)
print('ORACLE-K vs X-BASE, ALL 12 -- collapsing to real observation units')
print('=' * 104)
cluster_test('(a) episode      ', P1_12, lambda e: e)
cluster_test('(b) truth surface', P1_12, lambda e: field_key[e])
cluster_test('(c) phase vector ', P1_12, lambda e: phase_key[e])

print('\n' + '=' * 104)
print('Same collapse for G1P1 (the "-0.0701 / 4/4 / 73.6%" headline lives on CROWDED4)')
print('=' * 104)
cluster_test('(a) episode      ', CROWDED4, lambda e: e, cell='G1P1')
cluster_test('(c) phase vector ', CROWDED4, lambda e: phase_key[e], cell='G1P1')

# ---------------------------------------------------------------- confound size
print('\n' + '=' * 104)
print('ATTACK 1 -- ORACLE-K is a BUNDLE. How large is the non-k part on SPARSE8?')
print('  X-BASE  : init v2_konly, n_gaussians=[3,8,15], thresholds [.1,.05,None], iters [200,200,500]')
print('  ORACLE-K: init oracle  , n_gaussians=[N_gt] , thresholds [None]        , iters [900]')
print('  ==> 3 factors change at once: (k placement) + (component count N) + (stage schedule).')
print('=' * 104)
print(f"{'episode':38s} {'N_gt':>4s} {'XB final N':>11s} {'OK final N':>11s} "
      f"{'XB detect':>10s} {'OK detect':>10s} {'delta nRMSE':>12s}")
for ep in P1_12:
    nxb = np.median([r['N'] for r in by[('X-BASE', ep)]])
    nok = np.median([r['N'] for r in by[('ORACLE-K', ep)]])
    dxb = np.median([r['spec']['detection_rate'] for r in by[('X-BASE', ep)] if 'spec' in r])
    dok = np.median([r['spec']['detection_rate'] for r in by[('ORACLE-K', ep)] if 'spec' in r])
    print(f'{ep[8:-3]:38s} {ngt[ep]:4d} {nxb:11.1f} {nok:11.1f} {dxb:10.3f} {dok:10.3f} '
          f'{med("ORACLE-K", ep) - med("X-BASE", ep):+12.4f}')

print('\n  s1s2 measured the matched-schedule control S2-C1 (v2_konly + N=N_gt + [900])')
print('  ONLY on CROWDED4: 0.2818 vs shipped ladder 0.2667 -> the non-k bundle is HARMFUL there,')
print('  so on CROWDED4 the k factor is if anything understated. On SPARSE8 no such control exists.')

# ---------------------------------------------------------------- magnitude vs null
print('\n' + '=' * 104)
print('ATTACK 2b -- is the SPARSE8 ORACLE-K win of practical size, or noise-scale?')
print('=' * 104)
print(f"{'episode':38s} {'|delta|':>9s} {'XB seedRange':>13s} {'ratio':>7s} {'XB nRMSE':>9s}")
for ep in SPARSE8:
    d = abs(med('ORACLE-K', ep) - med('X-BASE', ep))
    sr = seed_range('X-BASE', ep)
    print(f'{ep[8:-3]:38s} {d:9.4f} {sr:13.4f} {d / sr if sr else float("nan"):7.2f} '
          f'{med("X-BASE", ep):9.4f}')
d8 = [abs(med('ORACLE-K', e) - med('X-BASE', e)) for e in SPARSE8]
s8 = [seed_range('X-BASE', e) for e in SPARSE8]
print(f"  median |delta| SPARSE8 = {np.median(d8):.4f} | median X-BASE seed range = "
      f"{np.median(s8):.4f} | ratio = {np.median(d8) / np.median(s8):.2f}")
d4 = [abs(med('ORACLE-K', e) - med('X-BASE', e)) for e in CROWDED4]
print(f"  median |delta| CROWDED4 = {np.median(d4):.4f}  "
      f"({np.median(d4) / np.median(d8):.1f}x the SPARSE8 effect)")
