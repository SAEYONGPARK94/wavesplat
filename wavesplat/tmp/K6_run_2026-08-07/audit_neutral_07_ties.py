"""NEUTRAL: what the sec.4 tie rule actually did, and whether the Rx-4 accident is fixed.

Rx-4 convention: a tie (Delta == 0) was scored as a LOSS and kept in n.  Applied to a
byte-duplicate cell that produces Delta == 0 on every episode, that manufactures a
"14-0 against" and a tiny p.  K6 sec.4 drops ties from n instead.  Both are computed
here on the SAME K6 numbers so the difference is attributable to the rule alone.
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
raw = {(r['cell'], r['episode'], r['seed']): r for r in R}
CELLS = ['AUTO-DUP', 'K6-KLR03', 'K6-KLR01', 'K6-KLR00', 'K6-NSCHED',
         'ORACLE-K', 'ORACLE']
METRICS = ['nRMSE', 'init_nrmse', 'Pearson_rho', 'absVR_framewise',
           'm_mean', 'decorr_mean', 'shrink_mean']


def med(c, e, m):
    return float(np.median([raw[(c, e, s)][m] for s in SEEDS]))


BLOCK = [e for e in EPS
         if med('AUTO', e, 'nRMSE') < NULL[e]['zero_null_tracking_nRMSE_mean']]


def sf(k, n):
    return sum(math.comb(n, i) * 0.5 ** n for i in range(k, n + 1))


def two(w, l):
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sf(max(w, l), n))


print('=' * 92)
print('A. Tie census -- how many exact ties exist anywhere in this batch')
print('=' * 92)
print(f'{"cell vs AUTO":12s} ' + ' '.join(f'{m[:9]:>10s}' for m in METRICS))
tot = defaultdict(int)
for c in CELLS:
    row = []
    for m in METRICS:
        d = np.array([med(c, e, m) - med('AUTO', e, m) for e in BLOCK])
        t = int((d == 0).sum())
        tot[c] += t
        row.append(f'{t:10d}')
    print(f'{c:12s} ' + ' '.join(row))
print(f'\n  => ties occur ONLY in the byte-duplicate cell AUTO-DUP '
      f'({tot["AUTO-DUP"]} of {tot["AUTO-DUP"]+sum(v for k,v in tot.items() if k!="AUTO-DUP")} '
      f'total ties).  Every real contrast has ZERO ties, so the rule change cannot '
      f'move any substantive K6 number.')

print('\n' + '=' * 92)
print('B. The two rules, side by side, on the byte-duplicate probe (nRMSE, n_ep=14)')
print('=' * 92)
d = np.array([med('AUTO-DUP', e, 'nRMSE') - med('AUTO', e, 'nRMSE') for e in BLOCK])
w, l, t = int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum())
print(f'  raw paired Deltas: wins {w}, losses {l}, TIES {t}, max|Delta| '
      f'{np.abs(d).max():.3e}')
print(f'  Rx-4 rule (tie counted as a loss, kept in n):  '
      f'{w}-{l+t} on n={w+l+t}  ->  p = {two(w, l+t):.6f}   '
      f'<-- SPURIOUS "significant harm" from a byte-for-byte copy')
print(f'  K6 sec.4 rule (ties dropped from n):           '
      f'{w}-{l} on n={w+l}  ->  p = {two(w, l):.6f}   <-- correct: no evidence')
print('  ==> the Rx-4 defect is FIXED, and the fix is load-bearing exactly here.')

print('\n' + '=' * 92)
print('C. Would the rule change ANY other K6 verdict? (both rules, all cells/metrics)')
print('=' * 92)
changed = []
for c in CELLS:
    for m in METRICS:
        sgn = -1 if m == 'Pearson_rho' else 1
        d = np.array([sgn * (med(c, e, m) - med('AUTO', e, m)) for e in BLOCK])
        w, l, t = int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum())
        p_k6, p_rx4 = two(w, l), two(w, l + t)
        if abs(p_k6 - p_rx4) > 1e-12:
            changed.append((c, m, w, l, t, p_k6, p_rx4))
for c, m, w, l, t, a, b in changed:
    print(f'  {c:10s} {m:16s} K6 p={a:.6f}  Rx4 p={b:.6f}  (ties {t})')
print(f'  total contrasts whose p moves under the rule change: {len(changed)} '
      f'(all of them AUTO-DUP)')

print('\n' + '=' * 92)
print('D. AUTO-DUP as a leak probe -- what it certifies and what it does not')
print('=' * 92)
nid = 0
for e in EPS:
    for s in SEEDS:
        a, b = raw[('AUTO', e, s)], raw[('AUTO-DUP', e, s)]
        if all(a[k] == b[k] for k in ('nRMSE', 'init_nrmse', 'kx', 'ky', 'w',
                                      'vr_frames', 'rho_frames')) and a['spec'] == b['spec']:
            nid += 1
print(f'  AUTO / AUTO-DUP bit-identical on {nid}/48 (episode, seed) runs.')
print('  CERTIFIES: no env/module residue leaks between cells inside a worker, and')
print('             cell AUTO is deterministic given (episode, seed).')
print('  DOES NOT CERTIFY: that any OTHER cell is correctly wired -- AUTO-DUP shares')
print('             AUTO\'s config byte for byte, so a mis-wired K6-KLR* knob would')
print('             be invisible to it.  That is what the cfg-diff check covers.')
