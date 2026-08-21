"""NEUTRAL (Rx-6) part 2: why the D ratio differs, and how load-bearing the
match window is for the sec.5 verdict.  Read-only."""
import json
import math
import os
from collections import defaultdict

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


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def dvecs(cell, e, s):
    nt = R[('AUTO-NOTRACK', e, s)]
    g = R[(cell, e, s)]
    w0 = np.array(nt['w'], float)
    dw = np.array(g['w'], float) - w0
    dp = w0 * wrap(np.array(g['phi'], float) - np.array(nt['phi'], float))
    return dw, dp


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


print('=' * 78)
print('A. D ratio under FIVE aggregations (prereg sec.4 says: seed-median -> block-median;')
print('   prereg sec.4.2 also FORBIDS "ratio of aggregates", which the match check violates)')
print('=' * 78)
rows = []
for i in range(1, 6):
    per_ep_w, per_ep_p, pool_w, pool_p = [], [], [], []
    per_ep_ratio = []
    for e in EPS:
        aw = [rms(dvecs(f'W-S{i}', e, s)[0]) for s in SEEDS]
        ap = [rms(dvecs(f'P-S{i}', e, s)[1]) for s in SEEDS]
        per_ep_w.append(np.median(aw)); per_ep_p.append(np.median(ap))
        per_ep_ratio.append(np.median(ap) / np.median(aw))
        for s in SEEDS:
            pool_w.append(dvecs(f'W-S{i}', e, s)[0]); pool_p.append(dvecs(f'P-S{i}', e, s)[1])
    a1 = np.median(per_ep_p) / np.median(per_ep_w)          # prereg-literal (mine)
    a2 = np.mean(per_ep_p) / np.mean(per_ep_w)              # mean-of-blocks
    a3 = rms(np.concatenate(pool_p)) / rms(np.concatenate(pool_w))   # atom-pooled rms
    a4 = float(np.median(per_ep_ratio))                     # median of per-episode ratios
    a5 = float(np.exp(np.median(np.log(per_ep_ratio))))     # geometric (== a4 for medians)
    rows.append((i, a1, a2, a3, a4, np.mean((np.array(per_ep_ratio) >= .5)
                                            & (np.array(per_ep_ratio) <= 2.))))
    print(f'  S{i}: blockmed-ratio={a1:.4f}  mean-ratio={a2:.4f}  atompooled-rms={a3:.4f} '
          f' median-of-per-ep-ratio={a4:.4f}   frac-ep-in[0.5,2]={rows[-1][5]:.3f}')
print('\n  -> the PRIMARY reported 0.963/0.999/1.437/2.259/3.085; none of these five')
print('     aggregations reproduces S1/S2 (I get 1.06-1.17 where PRIMARY got 0.96-1.00).')
print('     S4/S5 are >2 under EVERY aggregation; S1/S2/S3 are <2 under EVERY aggregation.')
print('     => the MATCH/FAIL PARTITION is aggregation-robust even though the ratio VALUE is not.')

print()
print('=' * 78)
print('B. is the [0.5, 2.0] window in the frozen PREREG?')
print('=' * 78)
txt = open(os.path.join(HERE, 'PREREG.md')).read()
for probe in ['0.5', '2.0', '[0.5', 'window', '범위', '큰게 어긋']:
    print(f'  grep {probe!r:16s} in PREREG.md: {probe in txt}')
print('  sec.1.1 verbatim criterion = "정합 쌍의 D 가 크게 어긋나면" '
      '(= "if D is greatly mismatched") -- NO NUMERIC THRESHOLD IS FROZEN.')

print()
print('=' * 78)
print('C. verdict sensitivity to the (unfrozen) match window upper edge')
print('=' * 78)


def medn(cell, e):
    return float(np.median([R[(cell, e, s)]['nRMSE'] for s in SEEDS]))


KEEP = [e for e in EPS if medn('AUTO', e) < NL[e]['zero_null_tracking_nRMSE_mean']]
req = REQ[len(KEEP)]
C, PASS = {}, {}
for i in range(1, 6):
    d = [medn(f'W-S{i}', e) - medn(f'P-S{i}', e) for e in KEEP]
    pos = sum(1 for x in d if x > 0); neg = sum(1 for x in d if x < 0)
    C[i] = float(np.median(d)); PASS[i] = max(pos, neg) >= req
ratio = {r[0]: r[1] for r in rows}
print('  C(S) = ' + '  '.join(
    'S%d:%+.5f%s' % (i, C[i], '*PASS' if PASS[i] else '') for i in range(1, 6)))
for hi in (1.5, 2.0, 2.4, 2.5, 3.0, 3.5, 99.0):
    M = [i for i in range(1, 6) if 0.5 <= ratio[i] <= hi]
    p = [i for i in M if PASS[i]]
    sg = set(np.sign(C[i]) for i in p)
    if len(p) >= 3 and len(sg) == 1:
        v = 'CHANNEL-REAL'
    elif len(p) <= 1:
        v = 'NO-DIFFERENCE'
    elif len(sg) > 1:
        v = 'CROSSOVER'
    else:
        v = 'UNLABELLED (2 same-sign)'
    print(f'  window [0.5,{hi:5.2f}] -> matched {M}  passing {p}  => {v}')
print('  ** the verdict flips NO-DIFFERENCE -> CROSSOVER once the upper edge passes '
      f'{ratio[4]:.2f} (S4). **')

print()
print('=' * 78)
print('D. does the realised-drive mismatch EXPLAIN the sign of C(S)?')
print('=' * 78)
print(f'  {"S":3s} {"D_w(W)":>11s} {"D_phi(P)":>11s} {"ratio":>7s} {"who drove harder":>17s} '
      f'{"C(S)":>10s} {"who was worse":>14s} {"consistent?":>12s}')
for i in range(1, 6):
    dw = np.median([np.median([rms(dvecs(f'W-S{i}', e, s)[0]) for s in SEEDS]) for e in EPS])
    dp = np.median([np.median([rms(dvecs(f'P-S{i}', e, s)[1]) for s in SEEDS]) for e in EPS])
    harder = 'phase' if dp > dw else 'amplitude'
    worse = 'amplitude' if C[i] > 0 else 'phase'
    print(f'  S{i}  {dw:11.4e} {dp:11.4e} {dp/dw:7.3f} {harder:>17s} {C[i]:+10.5f} '
          f'{worse:>14s} {"CONFOUNDED" if harder==worse else "against-drive":>12s}')

print()
print('=' * 78)
print('E. how much of the sec.5 CHANNEL-REAL arm was reachable?')
print('=' * 78)
print('  CHANNEL-REAL needs >=3 same-sign S over threshold.  Excluding the D-failed S')
print('  leaves only 3 candidate S, so CHANNEL-REAL required UNANIMITY of S1,S2,S3.')
for i in (1, 2, 3):
    dW = [medn(f'W-S{i}', e) - medn('AUTO', e) for e in KEEP]
    dP = [medn(f'P-S{i}', e) - medn('AUTO', e) for e in KEEP]
    pW = sum(1 for x in dW if x > 0); pP = sum(1 for x in dP if x > 0)
    print(f'   S{i}: W-S{i} vs AUTO {pW}/{len(KEEP)-pW}   P-S{i} vs AUTO {pP}/{len(KEEP)-pP}'
          f'   |C|={abs(C[i]):.5f}  MDE_t={2.8016*np.std([a-b for a,b in zip(dW,dP)],ddof=1)/math.sqrt(len(KEEP)):.5f}')
print('  S2 and S3 have BOTH cells indistinguishable from AUTO (see part 1), so their C is')
print('  structurally compressed -> the CHANNEL-REAL arm was effectively unreachable.')
