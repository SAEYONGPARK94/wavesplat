"""NEUTRAL (Rx-6) part 4: does C(S) reproduce on the other prereg sec.4 metrics,
and how concentrated is C(S1)?  Read-only."""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = {}
eps, seeds = set(), set()
for line in open(os.path.join(HERE, 'results.jsonl')):
    r = json.loads(line)
    R[(r['cell'], r['episode'], r['seed'])] = r
    eps.add(r['episode']); seeds.add(r['seed'])
EPS, SEEDS = sorted(eps), sorted(seeds)
NL = {r['episode']: r for r in json.load(open(os.path.join(HERE, 'nulls_rx6.json')))}
REQ = {32: 22, 31: 22, 30: 21, 29: 21, 28: 20, 27: 20, 26: 19}


def med(cell, e, key):
    return float(np.median([R[(cell, e, s)][key] for s in SEEDS]))


KEEP = [e for e in EPS if med('AUTO', e, 'nRMSE') < NL[e]['zero_null_tracking_nRMSE_mean']]
req = REQ[len(KEEP)]


def sgn(d):
    pos = sum(1 for x in d if x > 0); neg = sum(1 for x in d if x < 0)
    n = pos + neg
    k = max(pos, neg)
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n) if n else 1.0
    return pos, neg, p


# prereg sec.4 degradation directions: bigger = worse for all of these except Pearson_rho
METRICS = [('nRMSE', +1), ('Pearson_rho', -1), ('absVR_framewise', +1),
           ('m_mean', +1), ('decorr_mean', +1), ('shrink_mean', +1)]
print('=' * 92)
print(f'C(S) on all prereg sec.4 metrics, oriented so + = AMPLITUDE channel worse. '
      f'n={len(KEEP)}, req={req}')
print('=' * 92)
print(f'{"metric":18s}' + ''.join(f'{"S%d" % i:>22s}' for i in range(1, 6)))
for key, sign in METRICS:
    row = f'{key:18s}'
    for i in range(1, 6):
        d = [sign * (med(f'W-S{i}', e, key) - med(f'P-S{i}', e, key)) for e in KEEP]
        C = float(np.median(d))
        pos, neg, p = sgn(d)
        mark = '*' if max(pos, neg) >= req else ' '
        row += f'{C:+.5f} {pos:2d}/{neg:<2d}{mark}'.rjust(22)
    print(row)
print('  (* = clears the frozen required-win count; sign convention already flipped for rho)')

print()
print('=' * 92)
print('concentration of C(S1): jackknife + trimmed medians')
print('=' * 92)
d1 = [med(f'W-S1', e, 'nRMSE') - med(f'P-S1', e, 'nRMSE') for e in KEEP]
d1 = np.array(d1)
jk = [float(np.median(np.delete(d1, k))) for k in range(len(d1))]
print(f'  C(S1)={np.median(d1):+.5f}  jackknife range=[{min(jk):+.5f},{max(jk):+.5f}]  '
      f'mean(d)={d1.mean():+.5f}  sd={d1.std(ddof=1):.5f}')
srt = np.sort(d1)
print(f'  10%-trimmed mean={float(np.mean(srt[3:-3])):+.5f}   '
      f'25th/75th pct = {np.percentile(d1,25):+.5f}/{np.percentile(d1,75):+.5f}')
# leave-one-block-out
for bf in (4, 5):
    for nc in (15, 30):
        sub = [e for e in KEEP if f'BF{bf}_{nc}comp' not in e]
        dd = [med('W-S1', e, 'nRMSE') - med('P-S1', e, 'nRMSE') for e in sub]
        pos, neg, p = sgn(dd)
        # NOTE: the frozen required-win table only covers n=26..32, so for these
        # leave-one-fold-out subsets (n=23,24) I judge on the exact p directly.
        print(f'  drop BF{bf}/{nc}c -> n={len(dd)} C={np.median(dd):+.5f} {pos}/{neg} '
              f'p={p:.4f} {"p<=0.05" if p <= 0.05 else "p>0.05"}')

print()
print('=' * 92)
print('sanity: is C(S1) just "W-S1 is over-driven"?  compare each cell to AUTO and to floor')
print('=' * 92)
for c in ('W-S1', 'P-S1'):
    for ref in ('AUTO', 'AUTO-NOTRACK'):
        d = [med(c, e, 'nRMSE') - med(ref, e, 'nRMSE') for e in KEEP]
        pos, neg, p = sgn(d)
        print(f'  {c} - {ref:12s} med={np.median(d):+.5f}  {pos}/{neg}  p={p:.4f}')
print('  D_active: W-S1 = 1.0133e-01 m vs P-S1 = 1.1600e-01 m -> the PHASE cell moved the')
print('  field 14% FURTHER, yet the AMPLITUDE cell scored worse: C(S1) is against-drive.')
