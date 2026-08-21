"""NEUTRAL audit 03 -- independent re-derivation of PRIMARY-A and PRIMARY-B.

Written from the PREREG text only; does not import or reuse any run_rx5.py helper.
Aggregation, verbatim PREREG sec.4:
    seed-median  ->  per-episode SIGNED paired delta  ->  median of the deltas
Test: exact two-sided sign test, ties dropped from n.
NO block-median-of-differences anywhere.
"""
import json, os, statistics as st
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
NULLS = os.path.join(HERE, '..', 'K6_run_2026-08-07', 'nulls_k6.json')

recs = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl')) if l.strip()]
EPS = sorted({r['episode'] for r in recs})
CELLS = sorted({r['cell'] for r in recs})
SEEDS = sorted({r['seed'] for r in recs})

# ---- seed-median table: M[cell][episode][metric] --------------------------------
def seedmed(cell, ep, metric):
    v = [r[metric] for r in recs if r['cell'] == cell and r['episode'] == ep]
    assert len(v) == len(SEEDS), (cell, ep, len(v))
    return st.median(v)

M = {c: {e: seedmed(c, e, 'nRMSE') for e in EPS} for c in CELLS}
NFR = {e: len([r for r in recs if r['episode'] == e][0]['vr_frames']) for e in EPS}

# ---- exact two-sided sign test ---------------------------------------------------
def signtest(deltas, tol=0.0):
    """deltas: list of signed paired differences. Returns (n, wins_neg, wins_pos, ties, p)."""
    pos = sum(1 for d in deltas if d > tol)
    neg = sum(1 for d in deltas if d < -tol)
    ties = len(deltas) - pos - neg
    n = pos + neg
    if n == 0:
        return n, neg, pos, ties, 1.0
    k = max(pos, neg)
    p = 2.0 * sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return n, neg, pos, ties, min(1.0, p)

THRESH = {16: 13, 15: 12, 14: 12, 13: 11, 12: 10}   # PREREG sec.4.4, frozen

# ---- PREREG sec.4.5 null rule ----------------------------------------------------
nulls = {d['episode']: d for d in json.load(open(NULLS))}
print('=== sec.4 NULL RULE (control cell = AUTO, metric nRMSE) ===')
print(f"{'episode':46s} {'AUTO':>8} {'null':>8} {'margin':>9}  beats?")
keep = []
for e in EPS:
    a, nl = M['AUTO'][e], nulls[e]['zero_null_tracking_nRMSE_mean']
    ok = a < nl
    keep.append(e) if ok else None
    print(f'  {e[8:-14]:44s} {a:8.4f} {nl:8.4f} {nl-a:+9.4f}  {"yes" if ok else "NO -> EXCLUDE"}')
excl = [e for e in EPS if e not in keep]
print(f'\n  kept = {len(keep)} / {len(EPS)}   excluded = {len(excl)}  {[x[8:-14] for x in excl]}')
if not excl:
    print('  ** ZERO episodes excluded ** (PREREG sec.4.5 directionality note is moot)')

# ================================================================================
# PRIMARY-A
# ================================================================================
print('\n' + '=' * 82)
print('=== sec.5 PRIMARY-A :  median[ d(W-MATCH - AUTO) - d(PHI-5e3 - AUTO) ] ===')
print('    (AUTO cancels algebraically -> per-episode paired diff W-MATCH minus PHI-5e3)')
print('=' * 82)

def primary_a(block, label):
    d = []
    for e in block:
        dm = M['W-MATCH'][e] - M['AUTO'][e]
        dp = M['PHI-5e3'][e] - M['AUTO'][e]
        d.append(dm - dp)
    med = st.median(d)
    n, neg, pos, ties, p = signtest(d)
    thr = THRESH.get(n, '?')
    win = max(pos, neg)
    print(f'\n  [{label}]  n_episodes={len(block)}')
    print(f'    median of paired deltas = {med:+.5f}   (sign: '
          f'{"W-MATCH worse" if med > 0 else "PHI-5e3 worse"})')
    print(f'    sign test: pos(W-MATCH worse)={pos}  neg(PHI worse)={neg}  ties={ties}  n={n}')
    print(f'    dominant-direction wins = {win} / {n}   threshold = {thr}   '
          f'exact two-sided p = {p:.5f}')
    return d, med, n, pos, neg, ties, p, thr, win

dA, medA, nA, posA, negA, tiesA, pA, thrA, winA = primary_a(keep, 'PRIMARY (null-filtered)')
if excl:
    primary_a(EPS, 'un-filtered (sec.4.5 mandatory companion)')
else:
    print('    (un-filtered block is identical -- no exclusions)')

print('\n  per-episode detail:')
print(f"  {'episode':40s} {'W-MATCH':>9} {'PHI-5e3':>9} {'AUTO':>9} {'delta':>10}")
for e, d in zip(keep, dA):
    print(f'  {e[8:-14]:40s} {M["W-MATCH"][e]:9.4f} {M["PHI-5e3"][e]:9.4f} '
          f'{M["AUTO"][e]:9.4f} {d:+10.5f}')

# verdict
if winA >= thrA and medA > 0 and posA > negA:
    VA = 'CHANNEL-REAL'
elif winA >= thrA and medA < 0 and negA > posA:
    VA = 'INVERTED'
elif winA >= thrA:
    VA = 'AMBIGUOUS (win-count met but median sign disagrees with win direction)'
else:
    VA = 'NO-DIFFERENCE'
print(f'\n  >>> PRIMARY-A VERDICT = {VA}')

# ---- dose-response context (descriptive only, PREREG sec.2 note) -----------------
print('\n  --- amplitude dose-response (W-* vs W-5e3, one knob) descriptive ---')
print(f"  {'cell':10s} {'rel.step':>10} {'medianDelta vs AUTO':>21} {'wins/n vs AUTO':>16}")
ws = json.load(open(os.path.join(HERE, 'wscale.json')))
STEPS = {'W-5e3': 0.005, 'W-5e4': 5e-4, 'W-MATCH': ws['RX5_LR_W_MATCHED'], 'W-5e6': 5e-6}
for c in ['W-5e3', 'W-5e4', 'W-MATCH', 'W-5e6', 'PHI-5e3', 'AUTO-NOTRACK']:
    d = [M[c][e] - M['AUTO'][e] for e in keep]
    n, neg, pos, ties, p = signtest(d)
    rs = f'{STEPS[c]/ws["median_abs_w"]*100:.4f}%' if c in STEPS else '-'
    print(f'  {c:10s} {rs:>10} {st.median(d):+21.5f} {f"{max(pos,neg)}/{n}":>16}  p={p:.4f}'
          f'  ({"worse" if st.median(d)>0 else "better"} than AUTO)')

# ================================================================================
# PRIMARY-B
# ================================================================================
print('\n' + '=' * 82)
print('=== sec.5 PRIMARY-B :  E-sweep monotonicity vs null R(E) = 1 - E/160 ===')
print('=' * 82)
ES = [2, 5, 10, 20, 40, 80]
CELL = {E: f'E-{E}' for E in ES}

print('\n  denominator check (PREREG sec.1 asserts 160 for every episode):')
den = {e: NFR[e] for e in EPS}
print(f'    160-frame episodes = {sum(1 for e in EPS if den[e]==160)}, '
      f'120-frame episodes = {sum(1 for e in EPS if den[e]==120)}')

# total tracking damage per episode (denominator of the ratio)
DAM = {e: M['AUTO'][e] - M['AUTO-NOTRACK'][e] for e in EPS}
print('\n  --- total tracking damage  D = AUTO - AUTO-NOTRACK  (>0 = tracking HURTS) ---')
print(f"  {'episode':40s} {'AUTO':>8} {'NOTRACK':>9} {'D':>10}")
for e in EPS:
    print(f'  {e[8:-14]:40s} {M["AUTO"][e]:8.4f} {M["AUTO-NOTRACK"][e]:9.4f} {DAM[e]:+10.5f}')
n, neg, pos, ties, p = signtest([DAM[e] for e in keep])
print(f'  median D = {st.median([DAM[e] for e in keep]):+.5f}   '
      f'tracking-hurts wins {pos}/{n}  p={p:.5f}')
tiny = [e for e in EPS if abs(DAM[e]) < 0.005]
neg_d = [e for e in EPS if DAM[e] < 0]
print(f'  |D| < 0.005 (ratio unstable): {len(tiny)} episodes   D < 0 (sign flip): {len(neg_d)}')

def report_B(block, denom_mode, label):
    print(f'\n  ---------- {label} ----------')
    print(f"  {'E':>4} {'R_pred':>8} {'R_obs(med)':>11} {'n':>3} {'wins':>6} {'thr':>4} "
          f"{'p':>9}   verdict-direction")
    out = {}
    for E in ES:
        rows = []
        for e in block:
            D = DAM[e]
            if abs(D) < 1e-12:
                continue
            Robs = (M['AUTO'][e] - M[CELL[E]][e]) / D
            dd = 160 if denom_mode == 'declared' else den[e]
            Rpred = 1 - E / dd
            rows.append((e, Robs, Rpred, Robs - Rpred))
        dl = [r[3] for r in rows]
        n, neg, pos, ties, p = signtest(dl)
        thr = THRESH.get(n, '?')
        win = max(pos, neg)
        med_pred = st.median([r[2] for r in rows])
        direc = ('FRONT-LOADED(R below pred)' if st.median(dl) < 0
                 else 'BACK-LOADED(R above pred)')
        flag = 'DISTINGUISHABLE' if (isinstance(thr, int) and win >= thr) else 'not distinguishable'
        print(f'  {E:4d} {med_pred*100:7.1f}% {st.median([r[1] for r in rows])*100:10.1f}% '
              f'{n:3d} {f"{win}":>6} {str(thr):>4} {p:9.5f}   {direc}  [{flag}]')
        out[E] = (n, win, thr, p, st.median(dl))
    return out

B_declared = report_B(keep, 'declared', 'RATIO form, PRE-DECLARED denominator 160 (frozen rule)')
B_actual = report_B(keep, 'actual', 'RATIO form, ACTUAL per-episode denominator (160/120)')

# ---- difference form (ratio-free), same hypothesis --------------------------------
def report_B_diff(block, denom_mode, label):
    print(f'\n  ---------- {label} ----------')
    print(f"  {'E':>4} {'med(obs-pred) [nRMSE]':>22} {'n':>3} {'wins':>6} {'thr':>4} {'p':>9}   direction")
    for E in ES:
        dl = []
        for e in block:
            D = DAM[e]
            dd = 160 if denom_mode == 'declared' else den[e]
            obs = M['AUTO'][e] - M[CELL[E]][e]          # observed recovery, nRMSE units
            pred = (1 - E / dd) * D                     # null-model recovery, nRMSE units
            dl.append(obs - pred)
        n, neg, pos, ties, p = signtest(dl)
        thr = THRESH.get(n, '?')
        win = max(pos, neg)
        direc = 'FRONT-LOADED' if st.median(dl) < 0 else 'BACK-LOADED'
        flag = 'DISTINGUISHABLE' if (isinstance(thr, int) and win >= thr) else 'not distinguishable'
        print(f'  {E:4d} {st.median(dl):+22.5f} {n:3d} {f"{win}":>6} {str(thr):>4} {p:9.5f}   '
              f'{direc}  [{flag}]')

report_B_diff(keep, 'declared', 'DIFFERENCE form (ratio-free), denominator 160')
report_B_diff(keep, 'actual', 'DIFFERENCE form (ratio-free), actual denominator')

# ---- each E cell vs AUTO, descriptive (PREREG sec.5 last bullet) -----------------
print('\n  --- each E cell vs AUTO (descriptive, PREREG sec.5) ---')
print(f"  {'cell':8s} {'medianDelta':>13} {'wins/n':>9} {'thr':>4} {'p':>9}")
for E in ES:
    d = [M[CELL[E]][e] - M['AUTO'][e] for e in keep]
    n, neg, pos, ties, p = signtest(d)
    print(f'  {CELL[E]:8s} {st.median(d):+13.5f} {f"{max(pos,neg)}/{n}":>9} '
          f'{str(THRESH.get(n,"?")):>4} {p:9.5f}')
