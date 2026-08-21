"""NEUTRAL: how much weight can the placement contrast ORACLE-K - K6-NSCHED carry?

Two things the PREREG did not pre-specify:
  (a) the null rule is written for the CONTROL cell AUTO only.  K6-NSCHED is the
      REFERENCE of the secondary placement contrast and it is itself below the
      flat-sea null on most episodes.  Sensitivity is computed here.
  (b) the placement x (N+schedule) design is a 2x2 but only ONE ROW was run:
      (oracle, N_gt+900) and (blind, N_gt+900).  The other row -- (blind, ladder)
      = X-BASE and (oracle, ladder) -- was not run on this dataset, so the
      placement effect is measured at exactly one schedule level and the
      interaction is unidentified.  This script shows what IS and IS NOT available.
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


def med(c, e, m='nRMSE'):
    return float(np.median([raw[(c, e, s)][m] for s in SEEDS]))


def sf(k, n):
    return sum(math.comb(n, i) * 0.5 ** n for i in range(k, n + 1))


def st(w, l):
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sf(max(w, l), n))


def rep(tag, c, ref, eps, m='nRMSE'):
    d = np.array([med(c, e, m) - med(ref, e, m) for e in eps])
    w, l = int((d < 0).sum()), int((d > 0).sum())
    print(f'  {tag:44s} n={len(eps):2d}  median {np.median(d):+.5f}  '
          f'{w}/{w+l}  p={st(w,l):.5f}')


AUTO_OK = [e for e in EPS if med('AUTO', e) < NULL[e]['zero_null_tracking_nRMSE_mean']]
NS_OK = [e for e in EPS if med('K6-NSCHED', e) < NULL[e]['zero_null_tracking_nRMSE_mean']]
BOTH = [e for e in AUTO_OK if e in NS_OK]

print('=' * 90)
print('A. The reference arm of the placement contrast is itself a below-null arm')
print('=' * 90)
for c in ['AUTO', 'K6-NSCHED', 'ORACLE-K', 'ORACLE']:
    ok = [e for e in EPS if med(c, e) < NULL[e]['zero_null_tracking_nRMSE_mean']]
    print(f'  {c:10s} beats flat-sea null on {len(ok):2d}/16   '
          f'median nRMSE {np.median([med(c,e) for e in EPS]):.5f}   '
          f'flat-sea null median {np.median([NULL[e]["zero_null_tracking_nRMSE_mean"] for e in EPS]):.5f}')
print(f'\n  K6-NSCHED is WORSE THAN PREDICTING A FLAT SEA on {16-len(NS_OK)}/16 episodes.')

print('\n' + '=' * 90)
print('B. Placement contrast under three episode blocks')
print('=' * 90)
rep('ORACLE-K - K6-NSCHED, all 16', 'ORACLE-K', 'K6-NSCHED', EPS)
rep('ORACLE-K - K6-NSCHED, AUTO-null block (14)', 'ORACLE-K', 'K6-NSCHED', AUTO_OK)
rep('ORACLE-K - K6-NSCHED, K6-NSCHED itself beats null',
    'ORACLE-K', 'K6-NSCHED', NS_OK)
rep('ORACLE-K - K6-NSCHED, both arms beat null', 'ORACLE-K', 'K6-NSCHED', BOTH)
print(f'  (K6-NSCHED beats the null on: {[e[8:-14] for e in NS_OK]})')

print('\n' + '=' * 90)
print('C. The 2x2 that would license "k placement is worth X" -- one row is missing')
print('=' * 90)
print(f'{"":26s} {"schedule = N_gt + single 900":>30s} {"schedule = 3-stage ladder":>28s}')
ok = np.median([med('ORACLE-K', e) for e in AUTO_OK])
ns = np.median([med('K6-NSCHED', e) for e in AUTO_OK])
au = np.median([med('AUTO', e) for e in AUTO_OK])
print(f'{"placement = oracle (kx,ky)":26s} {ok:30.5f} {"NOT RUN":>28s}')
print(f'{"placement = blind v2_konly":26s} {ns:30.5f} {"NOT RUN (X-BASE absent)":>28s}')
print(f'{"placement = auto router":26s} {"NOT RUN":>30s} {au:28.5f}')
print(f'\n  measured placement effect AT THE 900-STEP SCHEDULE ONLY: {ok-ns:+.5f}')
print(f'  the same oracle placement measured against the SHIPPING arm : '
      f'{ok-au:+.5f}  (8/14, p=0.79)')
print('  => the placement factor is NOT transportable across schedules IN THIS DATA:')
print('     it is the largest effect in the batch at one schedule level and')
print('     indistinguishable from zero at the shipping one.  The interaction is')
print('     unidentified because the (blind, ladder) cell was not run here.')

print('\n' + '=' * 90)
print('D. Pre-registered falsifiable predictions D1 / D2 / D3, scored literally')
print('=' * 90)
for c in ['K6-KLR03', 'K6-KLR01', 'K6-KLR00']:
    d = np.array([med(c, e) - med('AUTO', e) for e in AUTO_OK])
    w = int((d < 0).sum())
    print(f'  {c:9s} median {np.median(d):+.5f}  wins {w}/14  '
          f'bar 12  -> {"clears" if w >= 12 else "does NOT clear"}')
print('  D1 ("at least one K6-KLR* beats AUTO => H1 alive; all lose => H1 rejected"):')
print('     under the sec.5 operative definition of "beats" (median<0 AND win bar),')
print('     0 of 3 beat AUTO  ->  H1 REJECTED.')
print('     under a bare median<0 reading, KLR03 and KLR01 have negative medians')
print('     (-0.0018 / -0.0007) but neither clears any win bar and both CIs span 0.')
print('     ** The PREREG does not define "이긴다" in D1; this is a text ambiguity, but')
print('        it is NOT outcome-relevant: no reading yields a PASS. **')
d = np.array([med('ORACLE-K', e) - med('K6-NSCHED', e) for e in AUTO_OK])
print(f'\n  D2 ("if ORACLE-K - K6-NSCHED ~ 0 then k placement was not the prize"):')
print(f'     measured {np.median(d):+.5f}, 14/14 -> the antecedent is FALSE.')
print('     D2 does not fire.  It is NOT licence to say "placement IS the prize" at')
print('     the shipping operating point -- see part C.')
d0 = np.array([med('K6-KLR00', e) - med('AUTO', e) for e in AUTO_OK])
print(f'\n  D3 ("if KLR00 also harms in crowded, the middle values are the reason to")')
print(f'     KLR00 median {np.median(d0):+.5f}, {int((d0<0).sum())}/14 '
      f'-> harmful direction, but 3/14 does not clear the 12-win bar (p=0.057).')
print('     The pre-registered "all three harmful => axis dead" branch does NOT fire')
print('     (KLR03 and KLR01 are not harmful); the "middle values exist for a reason"')
print('     branch is only weakly supported and never reaches the bar.')
