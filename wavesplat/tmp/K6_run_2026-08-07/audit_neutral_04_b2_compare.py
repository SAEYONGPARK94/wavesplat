"""NEUTRAL: is the K6 batch comparable to the B2 batch?
Re-derives ORACLE-K - AUTO from the B2 raw jsonl with the SAME sec.4 convention
used on K6, on every sub-block, then lists every dimension on which the two
batches differ.  Read-only on tmp/B2_g1p1_ext_2026-08-06/.
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
B2 = os.path.join(HERE, '..', 'B2_g1p1_ext_2026-08-06')

R = [json.loads(l) for l in open(os.path.join(B2, 'results.jsonl'))]
NULL = {r['episode']: r for r in json.load(open(os.path.join(B2, 'nulls12.json')))}
EPS = sorted({r['episode'] for r in R})
SEEDS = sorted({r['seed'] for r in R})
CROWDED4 = [e for e in EPS if ('15comp' in e or '30comp' in e)]
SPARSE8 = [e for e in EPS if e not in CROWDED4]

raw = {}
for r in R:
    raw[(r['cell'], r['episode'], r['seed'])] = r


def med(c, e, m='nRMSE'):
    return float(np.median([raw[(c, e, s)][m] for s in SEEDS]))


def binom_sf(k, n):
    return sum(math.comb(n, i) * 0.5 ** n for i in range(k, n + 1))


def sign_test(w, l):
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * binom_sf(max(w, l), n))


def contrast(c, ref, eps, m='nRMSE'):
    d = np.array([med(c, e, m) - med(ref, e, m) for e in eps])
    w, l, t = int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum())
    return np.median(d), d.mean(), w, l, t, sign_test(w, l), d.std(ddof=1)


print('=' * 78)
print('B2 batch re-derived by ME, sec.4 convention (seed-median -> paired D -> median)')
print('=' * 78)
print(f'seeds {SEEDS}   episodes {len(EPS)} (crowded {len(CROWDED4)} / sparse {len(SPARSE8)})')

keep = [e for e in EPS if med('AUTO', e) < NULL[e]['zero_null_tracking_nRMSE_mean']]
print(f'AUTO beats flat-sea null on {len(keep)}/{len(EPS)}: dropped '
      f'{[e[8:-3] for e in EPS if e not in keep] or "none"}')

for name, block in [('ALL 12', EPS), ('crowded-4', CROWDED4), ('sparse-8', SPARSE8),
                    ('ALL 12, null-excl', keep),
                    ('crowded-4, null-excl', [e for e in CROWDED4 if e in keep])]:
    m, mu, w, l, t, p, sd = contrast('ORACLE-K', 'AUTO', block)
    print(f'  ORACLE-K - AUTO  {name:22s} n={len(block):2d}  median {m:+.5f}  '
          f'mean {mu:+.5f}  {w}/{w+l}  p={p:.4f}  sd(D)={sd:.5f}')

print('\nB2 block medians of nRMSE:')
for c in ['AUTO', 'ORACLE-K', 'X-BASE', 'G1', 'G1P1']:
    print(f'  {c:9s} all12 {np.median([med(c,e) for e in EPS]):.5f}   '
          f'crowded4 {np.median([med(c,e) for e in CROWDED4]):.5f}   '
          f'sparse8 {np.median([med(c,e) for e in SPARSE8]):.5f}')

# ---------- K6 side, same numbers, for the comparison table --------------------
K = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl'))]
KN = {r['episode']: r for r in json.load(open(os.path.join(HERE, 'nulls_k6.json')))}
KEPS = sorted({r['episode'] for r in K})
KSEEDS = sorted({r['seed'] for r in K})
kraw = {(r['cell'], r['episode'], r['seed']): r for r in K}


def kmed(c, e, m='nRMSE'):
    return float(np.median([kraw[(c, e, s)][m] for s in KSEEDS]))


kkeep = [e for e in KEPS if kmed('AUTO', e) < KN[e]['zero_null_tracking_nRMSE_mean']]
d = np.array([kmed('ORACLE-K', e) - kmed('AUTO', e) for e in kkeep])
print(f'\nK6  ORACLE-K - AUTO  crowded-16 null-excl n={len(kkeep)}  '
      f'median {np.median(d):+.5f}  mean {d.mean():+.5f}  '
      f'{int((d<0).sum())}/{len(d)}  sd(D)={d.std(ddof=1):.5f}')

print('\n' + '=' * 78)
print('DIFFERENCES BETWEEN THE TWO BATCHES (why a direct number-to-number '
      'comparison is not licensed)')
print('=' * 78)
b2c = [e for e in CROWDED4]
rows = [
    ('episode generator', 'data_finemesh_2026-07-23 (fixed A/omega/phi per config)',
     'data_indep2_2026-08-07 (random_amplitudes+frequencies+decorrelate_stream)'),
    ('block', f'P1_12 = 8 sparse + 4 crowded', 'crowded-16 only'),
    ('independent draws (narrowest axis)', '4', '16'),
    ('run seeds', str(SEEDS), str(KSEEDS)),
    ('episodes kept after null rule', f'{len(keep)}/12', f'{len(kkeep)}/16'),
]
for a, b, c in rows:
    print(f'  {a:36s} B2: {b}\n  {"":36s} K6: {c}')

# difficulty match: are the two crowded blocks the same difficulty?
print('\nDifficulty check -- flat-sea null level (how hard the episode is):')
print(f'  B2 crowded-4 null median  = '
      f'{np.median([NULL[e]["zero_null_tracking_nRMSE_mean"] for e in b2c]):.5f}')
print(f'  K6 crowded-16 null median = '
      f'{np.median([KN[e]["zero_null_tracking_nRMSE_mean"] for e in KEPS]):.5f}')
print(f'  B2 crowded-4 AUTO median  = {np.median([med("AUTO",e) for e in b2c]):.5f}')
print(f'  K6 crowded-16 AUTO median = {np.median([kmed("AUTO",e) for e in KEPS]):.5f}')
print(f'  B2 crowded-4 ORACLE-K med = {np.median([med("ORACLE-K",e) for e in b2c]):.5f}')
print(f'  K6 crowded-16 ORACLE-K md = {np.median([kmed("ORACLE-K",e) for e in KEPS]):.5f}')

# router composition, both batches
from collections import Counter
print('\nRouter composition of cell AUTO:')
print('  B2:', Counter(r.get('auto_selected') for r in R if r['cell'] == 'AUTO'))
print('  K6:', Counter(r.get('auto_selected') for r in K if r['cell'] == 'AUTO'))
print('  B2 crowded-only:',
      Counter(r.get('auto_selected') for r in R
              if r['cell'] == 'AUTO' and r['episode'] in CROWDED4))
