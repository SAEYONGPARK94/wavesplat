#!/usr/bin/env python3
"""S17 analyser -- implements PREREG.md + AMENDMENT_1/2/3 EXACTLY.  No rule is chosen here;
every threshold, direction, tie-break and kill condition was frozen and committed before the
run (81c33b0 / c65203e / 8cad725) and thin19.json was frozen before any S17 row was read
(020f7be).  Verdict priority is AMENDMENT_1 A-2 (a total order): the FIRST rule that fires
is the only one printed."""
import collections
import json
import os
import statistics as st
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
S17 = os.path.join(HERE, 'results.jsonl')
S16 = os.path.join(ROOT, 'tmp', 'S16_BASELINE_2026-08-14', 'results.jsonl')
S13 = os.path.join(ROOT, 'tmp', 'S13_oos_2026-08-12', 'results.jsonl')
S13B = os.path.join(ROOT, 'tmp', 'S13b_seed_robust_2026-08-13', 'results_s13b.jsonl')

# (S, A, L) coordinates -- PREREG section 2.2.  WHITE50 is the anchor, NOT in the pool (1.1).
COORD = {'WHITER50':   ('W', 0.1,    50.0),
         'WHITER25':   ('W', 0.1,    25.0),
         'WHITER12':   ('W', 0.1,    12.5),
         'WHITER6':    ('W', 0.1,     6.25),
         'WHITERHS50': ('W', 'Hs/4', 50.0),
         'SPEC50':     ('J', 'Hs/4', 50.0),
         'SPEC25':     ('J', 'Hs/4', 25.0),
         'SPEC100':    ('J', 'Hs/4', 100.0)}
POOL = tuple(COORD)
S0 = 'WHITER50'
CHAIN = {                                   # PREREG 2.2, enumerated BEFORE the run
    'WHITER50':   [],
    'WHITER25':   ['WHITER50', 'WHITER25'],
    'WHITER12':   ['WHITER50', 'WHITER25', 'WHITER12'],
    'WHITER6':    ['WHITER50', 'WHITER25', 'WHITER12', 'WHITER6'],
    'WHITERHS50': ['WHITER50', 'WHITERHS50'],
    'SPEC50':     ['WHITER50', 'WHITERHS50', 'SPEC50'],
    'SPEC25':     ['WHITER50', 'WHITERHS50', 'SPEC50', 'SPEC25'],
    'SPEC100':    ['WHITER50', 'WHITERHS50', 'SPEC50', 'SPEC100'],
}
THR = 41                                     # PREREG 1: >=41/64, one-sided alpha 0.016383
HOLM = (43, 43, 42, 41)                      # PREREG 3.1 two-sided Holm-4, family alpha 0.05
SHELLS = ['obs:0_5', 'obs:5_10', 'obs:10_20', 'obs:20_30', 'obs:30_50',
          'obs:50_75', 'obs:75_100', 'obs:100_150', 'obs:150_inf']


def load(p):
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except ValueError:
                    pass


def med(v):
    v = [x for x in v if x is not None and x == x]
    return st.median(v) if v else float('nan')


def p_one(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2.0 ** n


def p_two(w, n):
    return min(1.0, 2.0 * min(p_one(w, n), p_one(n - w, n)))


def main():
    # ---- assemble -------------------------------------------------------------------
    raw = collections.defaultdict(lambda: collections.defaultdict(list))
    rawrows = collections.defaultdict(list)
    zero = collections.defaultdict(list)
    shells = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    zshell = collections.defaultdict(lambda: collections.defaultdict(list))
    ncomp, frac = {}, {}
    for src in (S16, S17):
        for r in load(src):
            if 'error' in r or r.get('ncomp') not in (15, 30):
                continue
            e, a = r['episode'], r['arm']
            raw[e][a].append(r['avg_nRMSE'])
            rawrows[a].append(r)
            zero[e].append(r['zero_nRMSE'])
            ncomp[e] = r['ncomp']
            for k in SHELLS:
                # baseline rows: keys carry the `obs:` prefix.  SHIP rows: bare keys.
                v = (r.get('shell_rmse') or {}).get(k)
                if v is not None:
                    shells[e][a][k].append(v)
                zv = (r.get('zero_shell') or {}).get(k)
                if zv is not None:
                    zshell[e][k].append(zv)
            for kk, vv in (r.get('shell_frac') or {}).items():
                frac.setdefault(kk if kk.startswith('obs:') else 'obs:' + kk, []).append(vv)
    ship = collections.defaultdict(dict)
    for r in load(S13):
        if r.get('arm') == 'SHIP' and 'error' not in r:
            ship[r['episode']][42] = r
    for r in load(S13B):
        if 'error' in r or not str(r.get('arm', '')).startswith('SHIP'):
            continue
        s = r.get('est_seed') or (r.get('effective_config') or {}).get('seed')
        if s in (43, 44):
            ship[r['episode']][int(s)] = r

    eps = sorted(e for e in raw if ncomp.get(e) in (15, 30)
                 and set(raw[e]) >= set(POOL) and len(ship.get(e, {})) == 3)
    n = len(eps)
    A = {a: {e: st.median(raw[e][a]) for e in eps} for a in POOL}
    Z = {e: st.median(zero[e]) for e in eps}
    S = {e: st.median([ship[e][s]['avg_nRMSE'] for s in (42, 43, 44)]) for e in eps}
    Ash = {a: {e: {k: med(shells[e][a][k]) for k in SHELLS} for e in eps} for a in POOL}
    Ssh = {e: {k: med([(ship[e][s].get('shell_rmse') or {}).get(k.split(':', 1)[1])
                       for s in (42, 43, 44)]) for k in SHELLS} for e in eps}

    print('=' * 78)
    print('S17 ANALYSIS  n=%d crowded episodes, pool=%d arms, med-of-3 both sides' % (n, len(POOL)))
    print('=' * 78)

    W = {a: sum(S[e] < A[a][e] for e in eps) for a in POOL}
    rel = {a: st.median([(S[e] - A[a][e]) / A[a][e] for e in eps]) for a in POOL}
    mean = {a: st.mean([A[a][e] for e in eps]) for a in POOL}

    # ---- eligibility (PREREG 2.1 as amended: R-1a' / R-1b') ------------------------
    print('\n-- eligibility (AMENDMENT_1 A-4 R-1a\' / A-5 R-1b\') ' + '-' * 24)
    print('%-11s %8s %8s %10s %9s %9s  %s' %
          ('arm', 'mean', 'W_a', 'zeroLoss%', 'a<ZERO', 'skill%', 'eligible'))
    elig = []
    for a in POOL:
        wz = sum(A[a][e] < Z[e] for e in eps)
        loss = sum(1 for r in rawrows[a] if r['avg_nRMSE'] >= r['zero_nRMSE']) / max(1, len(rawrows[a]))
        skill = st.median([(A[a][e] - Z[e]) / Z[e] for e in eps])
        ok = (wz >= THR) and (loss <= 0.10) and (skill <= -0.05)
        if ok:
            elig.append(a)
        print('%-11s %8.4f %5d/%d %9.1f%% %6d/%d %8.1f%%  %s'
              % (a, mean[a], W[a], n, 100 * loss, wz, n, 100 * skill, 'YES' if ok else 'no'))

    # ---- BEST (PREREG 1.1) ---------------------------------------------------------
    best = None
    if elig:
        best = sorted(elig, key=lambda a: (W[a], -rel[a], COORD[a][2], a))[0]
    print('\nP_elig = %s' % (', '.join(elig) if elig else '(empty)'))
    print('BEST   = %s   (argmin W_a; ties: larger med(rel), then smaller loc, then name)' % best)

    # ---- envelope co-PRIMARY (AMENDMENT_1 A-7) ------------------------------------
    env = {e: min(A[a][e] for a in elig) for e in eps} if elig else None
    W_env = sum(S[e] < env[e] for e in eps) if elig else None

    # ---- verdict chain, in the frozen priority order (A-2) -------------------------
    verdict, why = None, []
    w_iut = sum(S[e] < Z[e] for e in eps)
    if w_iut < THR:
        verdict = 'FAIL (IUT: SHIP does not beat the zero field)'
    elif n < 56:
        verdict = 'UNDECIDED (n=%d < 56)' % n
    elif not elig:
        verdict = 'UNDECIDED (no eligible baseline)'
    else:
        ch = CHAIN[best]
        missing = [x for x in ch if x not in A]
        if missing:
            verdict = 'UNDECIDED (chain unmeasurable: %s)' % missing
        else:
            harmful = []
            for x, y in zip(ch, ch[1:]):
                w_bad = sum(A[y][e] > A[x][e] for e in eps)
                if w_bad >= THR:
                    harmful.append('%s->%s %d/%d' % (x, y, w_bad, n))
            w_s0 = sum(A[best][e] < A[S0][e] for e in eps) if best != S0 else None
            if best == S0:
                verdict = 'N/A-PASS (empty chain: no repair strengthened the baseline)'
                why.append('PREREG 2.2 asterisk: the word "repaired/fair baseline" is FORBIDDEN.')
            elif harmful or (w_s0 is not None and w_s0 < THR):
                verdict = 'UNDECIDED (chain not strengthening)'
                why.append('harmful steps: %s' % (harmful or 'none'))
                why.append('BEST<s0: %s/%d (threshold %d)' % (w_s0, n, THR))
            elif COORD[best][2] == 100.0:
                verdict = 'UNDECIDED (grid boundary: loc at the upper edge)'

    print('\n-- verdict chain (AMENDMENT_1 A-2 total order) ' + '-' * 31)
    print('  D-6 IUT  SHIP<ZERO      : %d/%d  %s' % (w_iut, n, 'PASS' if w_iut >= THR else 'FAIL'))
    if best:
        print('  R-2 chain              : %s' % ' -> '.join(CHAIN[best]) or '(empty)')
        for x, y in zip(CHAIN[best], CHAIN[best][1:]):
            wb = sum(A[y][e] > A[x][e] for e in eps)
            print('      step %-11s -> %-11s weakens in %d/%d  %s'
                  % (x, y, wb, n, 'HARMFUL' if wb >= THR else 'ok'))
        if best != S0:
            print('  R-2(iii) BEST<s0       : %d/%d'
                  % (sum(A[best][e] < A[S0][e] for e in eps), n))
    if verdict:
        print('\n  ==> EARLY VERDICT: %s' % verdict)
        for x in why:
            print('      %s' % x)

    # ---- PRIMARY -------------------------------------------------------------------
    print('\n-- PRIMARY ' + '-' * 66)
    if best:
        w = W[best]
        print('  P-1  SHIP < BEST(%s) : %d/%d  med(rel)=%+.1f%%  p=%.3e  %s'
              % (best, w, n, 100 * rel[best], p_one(w, n), 'PASS' if w >= THR else 'FAIL'))
        print('  P-1b SHIP < ENVELOPE       : %d/%d  p=%.3e  %s'
              % (W_env, n, p_one(W_env, n), 'PASS' if W_env >= THR else 'FAIL'))
        both = (w >= THR) and (W_env >= THR)
        one = (w >= THR) != (W_env >= THR)
        print('  combined (A-7)             : %s'
              % ('PASS' if both else ('PARTIAL' if one else 'FAIL')))
        if verdict:
            print('  ** superseded by the earlier rule above: %s **' % verdict)

    # ---- FACTOR family (PREREG 3.1, two-sided Holm-4) ------------------------------
    print('\n-- FACTOR family (two-sided Holm-4, family alpha 0.05, reject %s) %s'
          % (str(HOLM), '-' * 5))
    F = [('F-L1 loc 50->25   ', 'WHITER25', 'WHITER50'),
         ('F-L2 loc 25->12.5 ', 'WHITER12', 'WHITER25'),
         ('F-L3 loc 12.5->6.25', 'WHITER6', 'WHITER12'),
         ('F-A  amp 0.1->Hs/4', 'WHITERHS50', 'WHITER50')]
    res = []
    for lab, y, x in F:
        w = sum(A[y][e] < A[x][e] for e in eps)
        res.append((p_two(w, n), lab, y, x, w))
    res.sort()
    for i, (p, lab, y, x, w) in enumerate(res):
        rj = HOLM[i]
        sig = (w >= rj) or (w <= n - rj)
        d = st.median([(A[y][e] - A[x][e]) / A[x][e] for e in eps])
        print('  %s  %d/%d better  med(rel)=%+6.2f%%  p=%.3e  Holm step %d (reject %d/%d)  %s'
              % (lab, w, n, 100 * d, p, i + 1, rj, n - rj, 'SIG' if sig else 'ns'))

    # ---- SHELL family --------------------------------------------------------------
    if best:
        print('\n-- SHELL family: SHIP < BEST(%s) per obs: shell (one-sided, >=%d/%d) %s'
              % (best, THR, n, '-' * 4))
        for k in SHELLS:
            w = sum(Ssh[e][k] < Ash[best][e][k] for e in eps
                    if Ssh[e][k] == Ssh[e][k] and Ash[best][e][k] == Ash[best][e][k])
            m = sum(1 for e in eps if Ssh[e][k] == Ssh[e][k])
            print('  %-13s %3d/%-3d  %s' % (k, w, m, 'PASS' if w >= THR else 'ns'))

    # ---- S-T (AMENDMENT_1 A-8), list frozen at 020f7be -----------------------------
    print('\n-- S-T thin-margin test (list frozen BEFORE any S17 row was read) ' + '-' * 12)
    doc = json.load(open(os.path.join(HERE, 'thin19.json')))
    thin = [d['episode'] for d in doc['episodes']]
    old = {d['episode']: d['env_old'] for d in doc['episodes']}
    new_arms = [a for a in elig if a in ('WHITER25', 'WHITER12', 'WHITER6', 'WHITERHS50')]
    got = sum(1 for e in thin if new_arms and min(A[a][e] for a in new_arms) < old[e])
    print('  new arms strictly beat every S16 pool arm in %d/%d thin episodes (threshold >=14)  %s'
          % (got, len(thin), 'PASS' if got >= 14 else 'FAIL'))

    # ---- descriptive D family ------------------------------------------------------
    print('\n-- D-2 A_win = area fraction of shells where the arm beats ZERO (descriptive) --')
    area = {k: st.mean(v) for k, v in frac.items() if v}
    tot = sum(area.get(k, 0.0) for k in SHELLS)
    for a in POOL:
        aw = 0.0
        for k in SHELLS:
            w = sum(1 for e in eps
                    if Ash[a][e][k] == Ash[a][e][k] and med(zshell[e][k]) == med(zshell[e][k])
                    and Ash[a][e][k] < med(zshell[e][k]))
            if w >= THR:
                aw += area.get(k, 0.0)
        print('  %-11s A_win=%.4f' % (a, aw / tot if tot else float('nan')))

    print('\n-- D-1 loc ladder (descriptive) ' + '-' * 45)
    for a in sorted(POOL, key=lambda a: (COORD[a][0], COORD[a][1] == 'Hs/4', -COORD[a][2])):
        print('  %-11s S=%s A=%-5s loc=%6.2f   mean=%.4f  W=%d/%d'
              % (a, COORD[a][0], COORD[a][1], COORD[a][2], mean[a], W[a], n))

    json.dump(dict(n=n, W=W, mean=mean, rel=rel, elig=elig, best=best, W_env=W_env,
                   verdict=verdict, iut=w_iut, thin_hits=got, thin_n=len(thin)),
              open(os.path.join(HERE, 'verdict_s17.json'), 'w'), indent=1, default=str)
    print('\n-> verdict_s17.json')


if __name__ == '__main__':
    main()
