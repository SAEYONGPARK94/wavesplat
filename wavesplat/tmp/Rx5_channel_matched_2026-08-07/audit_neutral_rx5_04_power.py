"""NEUTRAL audit 04 -- power / MDE, and the 'no effect vs underpowered' call.
Also tests the degeneracy PREREG sec.7 warned about: has W-MATCH collapsed onto
AUTO-NOTRACK?  If so, PRIMARY-A is not a channel contrast."""
import json, os, statistics as st
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
recs = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl')) if l.strip()]
EPS = sorted({r['episode'] for r in recs})
CELLS = sorted({r['cell'] for r in recs})

def seedmed(c, e):
    return st.median([r['nRMSE'] for r in recs if r['cell'] == c and r['episode'] == e])
M = {c: {e: seedmed(c, e) for e in EPS} for c in CELLS}

def signtest(d):
    pos = sum(1 for x in d if x > 0); neg = sum(1 for x in d if x < 0)
    n = pos + neg
    if n == 0: return n, neg, pos, 1.0
    k = max(pos, neg)
    return n, neg, pos, min(1.0, 2.0 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)

def binom_ge(n, k, p):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))

THRESH = {16: 13, 15: 12, 14: 12, 13: 11, 12: 10}

def dfree_ci(vals, alpha=0.05):
    """Distribution-free CI for the median from order statistics (matches sign test)."""
    n = len(vals); s = sorted(vals)
    best = None
    for k in range(0, n // 2):
        cov = 2 * sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
        if cov <= alpha:
            best = k
        else:
            break
    if best is None: return (s[0], s[-1], 'none')
    return (s[best], s[n - 1 - best], f'order stats {best+1}..{n-best}')

def headline(a, b, label, block=EPS):
    d = [M[a][e] - M[b][e] for e in block]
    n, neg, pos, p = signtest(d)
    win = max(pos, neg); thr = THRESH.get(n, None)
    med = st.median(d)
    lo, hi, how = dfree_ci(d)
    print(f'\n  {label}')
    print(f'    median paired delta = {med:+.5f}   95% distribution-free CI '
          f'[{lo:+.5f}, {hi:+.5f}]  ({how})')
    print(f'    wins {win}/{n}   threshold {thr}   exact two-sided p = {p:.5f}   '
          f'{"MEETS" if thr and win>=thr else "short by " + str(thr-win) + " win(s)" if thr else ""}')
    # shift-MDE: smallest |shift| in nRMSE that would move the block to the threshold
    if thr and win < thr:
        sgn = -1 if neg > pos else 1
        wrong = sorted([x for x in d if (x > 0 if sgn < 0 else x < 0)], key=abs)
        need = thr - win
        mde = abs(wrong[need - 1]) if len(wrong) >= need else float('nan')
        print(f'    shift-MDE: a uniform shift of {mde:.5f} nRMSE ({100*mde/abs(med):.1f}% of '
              f'the observed |median|) would flip {need} episode(s) and reach the threshold')
        print(f'      -> the {need} blocking episode(s) have |delta| = '
              f'{[round(abs(x),5) for x in wrong[:need+1]]}')
    if thr:
        # probability-MDE: per-episode win prob needed for 80% power at this n/threshold
        p80 = next((x / 1000 for x in range(500, 1001)
                    if binom_ge(n, thr, x / 1000) >= 0.80), None)
        obs_p = win / n
        print(f'    power: observed win-rate {obs_p:.3f}; '
              f'per-episode win-prob needed for 80% power at n={n},thr={thr} is '
              f'{p80:.3f}; achieved power at the observed rate = '
              f'{binom_ge(n, thr, obs_p):.3f}')
        print(f'    n needed for 80% power at the observed win-rate {obs_p:.3f}: ', end='')
        for nn in range(16, 201):
            th = None
            for k in range(nn // 2 + 1, nn + 1):
                if 2 * sum(comb(nn, i) for i in range(k, nn + 1)) / 2 ** nn <= 0.05:
                    th = k; break
            if th and binom_ge(nn, th, obs_p) >= 0.80:
                print(f'n={nn} (needs {th} wins)'); break
        else:
            print('>200')
    return d

print('=' * 84)
print('PRIMARY-A and its neighbours')
print('=' * 84)
headline('W-MATCH', 'PHI-5e3', 'PRIMARY-A: W-MATCH vs PHI-5e3 (>0 = W-MATCH worse)')

print('\n' + '=' * 84)
print('DEGENERACY PROBE (PREREG sec.7): has the matched cell collapsed to tracking-off?')
print('=' * 84)
headline('W-MATCH', 'AUTO-NOTRACK', 'W-MATCH vs AUTO-NOTRACK (>0 = W-MATCH worse)')
headline('W-5e6', 'AUTO-NOTRACK', 'W-5e6   vs AUTO-NOTRACK')
headline('PHI-5e3', 'AUTO-NOTRACK', 'PHI-5e3 vs AUTO-NOTRACK')
headline('PHI-5e3', 'AUTO', 'PHI-5e3 vs AUTO  (does phase-only reproduce shipped?)')
headline('W-5e3', 'AUTO', 'W-5e3   vs AUTO')
headline('AUTO', 'AUTO-NOTRACK', 'AUTO vs AUTO-NOTRACK (total tracking damage D)')

print('\n' + '=' * 84)
print('CHANNEL ADDITIVITY (all relative to AUTO-NOTRACK, nRMSE units)')
print('=' * 84)
base = 'AUTO-NOTRACK'
for c in ['PHI-5e3', 'W-5e3', 'AUTO', 'W-MATCH', 'W-5e4', 'W-5e6']:
    d = [M[c][e] - M[base][e] for e in EPS]
    n, neg, pos, p = signtest(d)
    print(f'  {c:13s} median = {st.median(d):+.5f}   wins {max(pos,neg)}/{n}  p={p:.4f}')
dphi = st.median([M['PHI-5e3'][e] - M[base][e] for e in EPS])
dw = st.median([M['W-5e3'][e] - M[base][e] for e in EPS])
dboth = st.median([M['AUTO'][e] - M[base][e] for e in EPS])
print(f'\n  additive prediction for w_phi = {dphi:+.5f} + {dw:+.5f} = {dphi+dw:+.5f}')
print(f'  observed w_phi (=AUTO)          = {dboth:+.5f}')
print(f'  -> STRONGLY SUB-ADDITIVE: joint damage is {dboth/(dphi+dw)*100:.0f}% of the sum,'
      f' and even {"below" if dboth < dw else "above"} the w-only arm alone.')

print('\n' + '=' * 84)
print('PRIMARY-B power (frozen denominator 160, ratio form)')
print('=' * 84)
DAM = {e: M['AUTO'][e] - M['AUTO-NOTRACK'][e] for e in EPS}
for E in (2, 5, 10, 20, 40, 80):
    d = [((M['AUTO'][e] - M[f'E-{E}'][e]) / DAM[e]) - (1 - E / 160) for e in EPS]
    n, neg, pos, p = signtest(d)
    win = max(pos, neg); thr = THRESH[n]
    sgn = -1 if neg > pos else 1
    wrong = sorted([x for x in d if (x > 0 if sgn < 0 else x < 0)], key=abs)
    need = max(0, thr - win)
    print(f'  E-{E:<3d} wins {win}/{n} (thr {thr}, short {need})  p={p:.5f}  '
          f'median(R_obs-R_pred) = {st.median(d):+.4f}   '
          f'blocking |delta| = {[round(abs(x),4) for x in wrong[:max(need,1)]]}')
