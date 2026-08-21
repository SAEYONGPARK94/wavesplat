"""NEUTRAL (Rx-6) independent re-derivation.  Written from scratch against PREREG.md.
Reads only results.jsonl / nulls_rx6.json.  Writes nothing except stdout.
"""
import json
import math
import os
from collections import defaultdict
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results.jsonl')
NULLS = os.path.join(HERE, 'nulls_rx6.json')

CELLS13 = ['AUTO', 'AUTO-DUP', 'AUTO-NOTRACK'] + [f'W-S{i}' for i in range(1, 6)] \
          + [f'P-S{i}' for i in range(1, 6)]
SGRID = [5.0e-3, 5.0e-4, 2.205e-4, 5.0e-5, 5.0e-6]
MEDW = 0.044095315039157870
# PREREG sec.4.3 frozen required-win table
REQ = {32: 22, 31: 22, 30: 21, 29: 21, 28: 20, 27: 20, 26: 19}

# ---------------------------------------------------------------- load
R = {}          # (cell, ep, seed) -> record
eps, seeds = set(), set()
for line in open(RES):
    r = json.loads(line)
    key = (r['cell'], r['episode'], r['seed'])
    assert key not in R, f'duplicate {key}'
    R[key] = r
    eps.add(r['episode'])
    seeds.add(r['seed'])
EPS = sorted(eps)
SEEDS = sorted(seeds)
print(f'loaded {len(R)} runs | {len(EPS)} episodes | seeds {SEEDS} | '
      f'{len(set(k[0] for k in R))} cells')


def hr(t):
    print('\n' + '=' * 78 + f'\n{t}\n' + '=' * 78)


# ================================================================ sec.6 INTEGRITY
hr('sec.6 INTEGRITY (NEUTRAL re-run)')

# --- 6.1 grid completeness (the gate32 file itself is re-run separately) --------
missing = [(c, e, s) for c in CELLS13 for e in EPS for s in SEEDS if (c, e, s) not in R]
print(f'[6.0] grid 13x32x3 complete: {"PASS" if not missing else "FAIL " + str(missing[:5])}'
      f'  (expected 1248, got {len(R)})')

# --- 6.2 worker leak: AUTO-DUP == AUTO bit-exact --------------------------------
FLOATK = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score', 'Directional_R2',
          'Hs_error', 'absVR_framewise', 'decorr_mean', 'shrink_mean', 'm_mean',
          'avg_RMSE_unobserved', 'init_nrmse', 'init_rmse']
LISTK = ['kx', 'ky', 'w', 'phi', 'vr_frames', 'rho_frames']
leak = []
for e in EPS:
    for s in SEEDS:
        a, d = R[('AUTO', e, s)], R[('AUTO-DUP', e, s)]
        for k in FLOATK:
            if a[k] != d[k]:
                leak.append((e, s, k, a[k], d[k]))
        for k in LISTK:
            if a[k] != d[k]:
                leak.append((e, s, k, 'listdiff', ''))
        for k in ('N', 'auto_selected', 'ensemble_k', 'auto_n_selected'):
            if a[k] != d[k]:
                leak.append((e, s, k, a[k], d[k]))
print(f'[6.2] worker leak (AUTO-DUP == AUTO bit-exact over {len(EPS)*len(SEEDS)} pairs, '
      f'{len(FLOATK)+len(LISTK)+4} fields): {"PASS" if not leak else "FAIL"}')
for x in leak[:5]:
    print('      ', x)

# --- 6.3 Phase-1 invariance across all 13 cells ---------------------------------
p1bad = []
for e in EPS:
    for s in SEEDS:
        ref = R[('AUTO', e, s)]
        for c in CELLS13:
            g = R[(c, e, s)]
            for k in ('init_nrmse', 'init_rmse', 'auto_selected', 'N', 'ensemble_k',
                      'auto_n_selected'):
                if g[k] != ref[k]:
                    p1bad.append((c, e, s, k, ref[k], g[k]))
print(f'[6.3] Phase-1 invariance (init_nrmse/init_rmse/auto_selected/N/ensemble_k/'
      f'auto_n_selected identical over 13 cells x 96): '
      f'{"PASS" if not p1bad else "FAIL"}  ({len(p1bad)} violations)')
for x in p1bad[:5]:
    print('      ', x)

# --- 6.4 per-cell knobs from the cfg records ------------------------------------
CFGK = ['init_method', 'use_dispersion', 'use_confidence', 'adaptive_n', 'tracking_window',
        'prune_threshold', 'max_n', 'analytic_ensemble_k', 'tracking_mode', 'tracking_lr',
        'n_gaussians', 'loss_thresholds', 'max_iters_per_stage']


def cfgof(c):
    seen = set()
    for e in EPS:
        for s in SEEDS:
            cf = R[(c, e, s)]['cfg']
            seen.add(json.dumps({k: cf.get(k) for k in CFGK}, sort_keys=True))
    assert len(seen) == 1, f'{c} cfg varies across runs: {len(seen)}'
    return json.loads(seen.pop())


CFG = {c: cfgof(c) for c in CELLS13}
auto = CFG['AUTO']
print('[6.4] per-cell knobs (cfg diff vs AUTO; cfg constant within cell: PASS)')
knob_ok = True
for c in CELLS13:
    d = {k: (auto.get(k), CFG[c].get(k)) for k in CFGK if auto.get(k) != CFG[c].get(k)}
    exp = 0 if c == 'AUTO-DUP' else (1 if c == 'AUTO-NOTRACK' else (0 if c == 'AUTO' else 2))
    ok = len(d) == exp
    knob_ok &= ok
    print(f'      {c:14s} nknob={len(d)} (exp {exp}) {"OK" if ok else "MISMATCH"}  {d}')
# grid identity: S_w == S_phi
print('[6.4b] matched grid S_w vs S_phi*median|w| :')
grid_ok = True
for i, S in enumerate(SGRID, 1):
    sw = CFG[f'W-S{i}']['tracking_lr'] * 1.0
    sp = CFG[f'P-S{i}']['tracking_lr'] * MEDW
    ok = abs(sw / sp - 1) < 1e-12
    grid_ok &= ok
    print(f'      S{i}: S_w={sw:.6e}  S_phi={sp:.6e}  ratio={sw/sp:.12f} '
          f'{"OK" if ok else "MISMATCH"}')
print(f'      => sec.6.4 {"PASS" if knob_ok and grid_ok else "FAIL"}')

# --- 6.4c wiring cross-check on the OUTPUT params (not just cfg) ----------------
# w_only must leave phi bit-identical to NOTRACK; phi_only must leave w bit-identical.
wire = []
for e in EPS:
    for s in SEEDS:
        nt = R[('AUTO-NOTRACK', e, s)]
        for i in range(1, 6):
            if R[(f'W-S{i}', e, s)]['phi'] != nt['phi']:
                wire.append((f'W-S{i}', e, s, 'phi moved'))
            if R[(f'P-S{i}', e, s)]['w'] != nt['w']:
                wire.append((f'P-S{i}', e, s, 'w moved'))
        for c in CELLS13:
            if R[(c, e, s)]['kx'] != nt['kx'] or R[(c, e, s)]['ky'] != nt['ky']:
                wire.append((c, e, s, 'k moved'))
print(f'[6.4c] output-level wiring (W-* freeze phi, P-* freeze w, all cells freeze k): '
      f'{"PASS" if not wire else "FAIL"}  ({len(wire)} violations)')
for x in wire[:6]:
    print('      ', x)

# ================================================================ sec.1.1  D
hr('sec.1.1  D VERIFICATION  (atom alignment checked FIRST)')

# --- atom order/count consistency across cells ----------------------------------
align_bad = []
for e in EPS:
    for s in SEEDS:
        nt = R[('AUTO-NOTRACK', e, s)]
        n0 = len(nt['w'])
        for c in CELLS13:
            g = R[(c, e, s)]
            if len(g['w']) != n0 or len(g['phi']) != n0 or len(g['kx']) != n0:
                align_bad.append((c, e, s, 'len', n0, len(g['w'])))
            elif g['kx'] != nt['kx'] or g['ky'] != nt['ky']:
                align_bad.append((c, e, s, 'korder'))
print(f'ATOM ALIGNMENT: identical count AND bit-identical (kx,ky) vector in the same '
      f'order across all 13 cells, all 96 (ep,seed): '
      f'{"PASS" if not align_bad else "FAIL"}  ({len(align_bad)} violations)')
ns = sorted(set(len(R[('AUTO', e, s)]['w']) for e in EPS for s in SEEDS))
print(f'  atom counts observed across runs: N in {ns}  '
      f'(varies BETWEEN (ep,seed), constant WITHIN -> Delta is well-defined)')


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def D_of(cell, e, s):
    nt = R[('AUTO-NOTRACK', e, s)]
    g = R[(cell, e, s)]
    w0 = np.array(nt['w'], float)
    dw = np.array(g['w'], float) - w0
    dphi = wrap(np.array(g['phi'], float) - np.array(nt['phi'], float))
    return float(np.sqrt(np.mean(dw ** 2))), float(np.sqrt(np.mean((w0 * dphi) ** 2)))


def blockmed(vals_by_ep):
    return float(np.median([np.median(v) for v in vals_by_ep]))


Dtab = {}
for c in CELLS13:
    dw = [[D_of(c, e, s)[0] for s in SEEDS] for e in EPS]
    dp = [[D_of(c, e, s)[1] for s in SEEDS] for e in EPS]
    Dtab[c] = (blockmed(dw), blockmed(dp))
print('\n  cell            D_w [m]      D_phi [m]    D_active [m]')
for c in CELLS13:
    dw, dp = Dtab[c]
    act = dw if c.startswith('W') else (dp if c.startswith('P') else max(dw, dp))
    print(f'  {c:14s} {dw:.6e} {dp:.6e}  {act:.6e}')

print('\n  MATCH CHECK  (D_phi(P-Si) / D_w(W-Si), pass window [0.5, 2.0]):')
match = {}
for i in range(1, 6):
    dw = Dtab[f'W-S{i}'][0]
    dp = Dtab[f'P-S{i}'][1]
    r = dp / dw
    match[i] = (0.5 <= r <= 2.0)
    print(f'   S{i}  D_w(W-S{i})={dw:.6e}  D_phi(P-S{i})={dp:.6e}  ratio={r:.4f}  '
          f'{"MATCH-OK" if match[i] else "MATCH-FAIL"}')

# per-episode ratio distribution (robustness of the block-median ratio)
print('\n  per-episode ratio distribution (median of 3 seeds each), sanity only:')
for i in range(1, 6):
    rs = []
    for e in EPS:
        a = np.median([D_of(f'W-S{i}', e, s)[0] for s in SEEDS])
        b = np.median([D_of(f'P-S{i}', e, s)[1] for s in SEEDS])
        rs.append(b / a)
    rs = np.array(rs)
    print(f'   S{i}  median={np.median(rs):.4f}  IQR=[{np.percentile(rs,25):.4f},'
          f'{np.percentile(rs,75):.4f}]  min={rs.min():.4f} max={rs.max():.4f}  '
          f'frac in [0.5,2]={np.mean((rs>=0.5)&(rs<=2.0)):.3f}')

# ================================================================ sec.4  NULLS
hr('sec.4  NULL RULE')
NL = {r['episode']: r for r in json.load(open(NULLS))}
missing_null = [e for e in EPS if e not in NL]
print(f'nulls present for all {len(EPS)} episodes: '
      f'{"PASS" if not missing_null else "FAIL " + str(missing_null)}')


def med_nrmse(cell, e):
    return float(np.median([R[(cell, e, s)]['nRMSE'] for s in SEEDS]))


excl = []
print('\n  AUTO vs null (seed-median nRMSE < zero_null_tracking_nRMSE_mean):')
for e in EPS:
    a = med_nrmse('AUTO', e)
    nl = NL[e]['zero_null_tracking_nRMSE_mean']
    if a >= nl:
        excl.append(e)
        print(f'   EXCLUDE {e[8:-3]:34s} AUTO={a:.4f}  null={nl:.4f}  m={a/nl:.3f}')
print(f'  excluded: {len(excl)} -> n = {len(EPS)-len(excl)}   '
      f'required wins = {REQ[len(EPS)-len(excl)]}')

# per-cell null pass table
print('\n  per-cell null pass count (of 32 episodes) and m>1 count:')
for c in CELLS13:
    ms = [med_nrmse(c, e) / NL[e]['zero_null_tracking_nRMSE_mean'] for e in EPS]
    print(f'   {c:14s} pass {sum(1 for m in ms if m < 1):2d}/32   m>1: '
          f'{sum(1 for m in ms if m >= 1):2d}   median m = {np.median(ms):.4f}')

KEEP = [e for e in EPS if e not in excl]

# ================================================================ stats helpers
def sign_test(d, alt='two-sided'):
    """exact two-sided sign test on paired differences; ties dropped."""
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    tie = len(d) - pos - neg
    n = pos + neg
    if n == 0:
        return pos, neg, tie, 1.0
    k = max(pos, neg)
    p = 0.0
    for j in range(k, n + 1):
        p += math.comb(n, j)
    p = min(1.0, 2 * p / 2 ** n)
    return pos, neg, tie, p


def mde_t(d, alpha=0.05, power=0.80):
    """paired two-sided MDE at 80% power, normal approx on the observed sd."""
    n = len(d)
    sd = float(np.std(d, ddof=1))
    return 1.959963985 * sd / math.sqrt(n) + 0.8416212336 * sd / math.sqrt(n), sd


def mde_sign(d):
    """smallest constant shift |delta| making the two-sided sign test reach p<=0.05."""
    n = len(d)
    req = REQ.get(n)
    if req is None:
        return float('nan')
    a = np.sort(np.array(d, float))
    # shift up so that >=req values are >0  => need a[n-req] + delta > 0
    up = max(0.0, -a[n - req])
    dn = max(0.0, a[req - 1])
    return min(up, dn)


def paired(cellA, cellB, epset):
    return [med_nrmse(cellA, e) - med_nrmse(cellB, e) for e in epset]


# ================================================================ sec.5 PRIMARY
hr('sec.5  PRIMARY  C(S) = median[ (W-S - AUTO) - (P-S - AUTO) ]  (+ = amplitude worse)')
print('  note: AUTO cancels exactly pointwise -> C(S) = median[W-S - P-S] (paired per episode)\n')


def report_C(epset, label):
    n = len(epset)
    req = REQ.get(n, None)
    print(f'--- {label}: n={n}, required wins = {req}')
    print(f'  {"S":3s} {"C(S)":>11s} {"pos":>4s} {"neg":>4s} {"tie":>4s} {"p_exact":>9s} '
          f'{"thr?":>6s} {"MDE_t":>10s} {"obs/MDE":>8s} {"match":>10s}')
    out = {}
    for i in range(1, 6):
        d = paired(f'W-S{i}', f'P-S{i}', epset)
        C = float(np.median(d))
        pos, neg, tie, p = sign_test(d)
        wins = max(pos, neg)
        thr = (req is not None and wins >= req)
        m, sd = mde_t(d)
        out[i] = dict(C=C, pos=pos, neg=neg, tie=tie, p=p, thr=thr, mde=m, sd=sd,
                      mde_sign=mde_sign(d))
        print(f'  S{i}  {C:+11.5f} {pos:4d} {neg:4d} {tie:4d} {p:9.4f} '
              f'{"PASS" if thr else "  - ":>6s} {m:10.5f} {abs(C)/m:8.2f} '
              f'{"OK" if match[i] else "FAIL":>10s}')
    return out


C_keep = report_C(KEEP, 'PRIMARY (null-excluded)')
C_all = report_C(EPS, 'SENSITIVITY (no exclusion)')

print('\n  exclusion directionality (the dropped episode(s) alone):')
for e in excl:
    for i in range(1, 6):
        d = med_nrmse(f'W-S{i}', e) - med_nrmse(f'P-S{i}', e)
        print(f'   {e[8:-3]:34s} S{i}  W-P = {d:+.5f}  '
              f'(sign {"+" if d>0 else "-"} vs C(S)_keep {C_keep[i]["C"]:+.5f} -> '
              f'{"same" if (d>0)==(C_keep[i]["C"]>0) else "opposite"})')

# ================================================================ degeneracy
hr('sec.5 SECONDARY  DEGENERACY  (each cell vs AUTO and vs AUTO-NOTRACK)')
req = REQ[len(KEEP)]
print(f'  n={len(KEEP)}, required wins={req}.  "DEGEN" = cannot be distinguished.')
print(f'  {"cell":14s} {"vs AUTO d":>11s} {"pos/neg":>9s} {"p":>8s} {"dist?":>6s} | '
      f'{"vs NOTRACK d":>13s} {"pos/neg":>9s} {"p":>8s} {"dist?":>6s}')
DEG = {}
for c in [f'W-S{i}' for i in range(1, 6)] + [f'P-S{i}' for i in range(1, 6)] + ['AUTO-NOTRACK']:
    row = []
    for ref in ('AUTO', 'AUTO-NOTRACK'):
        if c == ref:
            row.append((0.0, 0, 0, 1.0, False))
            continue
        d = paired(c, ref, KEEP)
        pos, neg, tie, p = sign_test(d)
        row.append((float(np.median(d)), pos, neg, p, max(pos, neg) >= req))
    DEG[c] = dict(vsAUTO=row[0], vsNT=row[1])
    a, b = row
    print(f'  {c:14s} {a[0]:+11.5f} {a[1]:4d}/{a[2]:<4d} {a[3]:8.4f} '
          f'{"DIST" if a[4] else "DEGEN":>6s} | {b[0]:+13.5f} {b[1]:4d}/{b[2]:<4d} '
          f'{b[3]:8.4f} {"DIST" if b[4] else "DEGEN":>6s}')

print('\n  informativeness per S (a cell is uninformative if it is degenerate with BOTH '
      'ends, or if BOTH cells of the pair are degenerate with the same end):')
for i in range(1, 6):
    W, P = DEG[f'W-S{i}'], DEG[f'P-S{i}']
    wd = ('AUTO' if not W['vsAUTO'][4] else '') + ('|NOTRACK' if not W['vsNT'][4] else '')
    pd = ('AUTO' if not P['vsAUTO'][4] else '') + ('|NOTRACK' if not P['vsNT'][4] else '')
    both_dead = (not W['vsAUTO'][4] and not P['vsAUTO'][4]) or \
                (not W['vsNT'][4] and not P['vsNT'][4])
    print(f'   S{i}: W-S{i} degenerate-with[{wd or "none"}]  P-S{i} degenerate-with'
          f'[{pd or "none"}]  -> {"NO CHANNEL INFO" if both_dead else "informative"}')

# ================================================================ headroom / power
hr('POWER')
d = paired('AUTO-NOTRACK', 'AUTO', KEEP)
pos, neg, tie, p = sign_test(d)
head = float(np.median(d))
m, sd = mde_t(d)
print(f'  tracker headroom  AUTO-NOTRACK - AUTO = {head:+.5f}  '
      f'({pos} pos / {neg} neg / {tie} tie, exact two-sided p = {p:.5f}, n={len(KEEP)})')
print(f'    -> tracker "damage budget" |headroom| = {abs(head):.5f}')
print(f'\n  {"S":3s} {"|C(S)|":>10s} {"MDE_t(80%)":>11s} {"MDE_sign":>10s} '
      f'{"obs/MDE_t":>10s} {"MDE_t/headroom":>15s}')
for i in range(1, 6):
    o = C_keep[i]
    print(f'  S{i}  {abs(o["C"]):10.5f} {o["mde"]:11.5f} {o["mde_sign"]:10.5f} '
          f'{abs(o["C"])/o["mde"]:10.2f} {o["mde"]/abs(head):15.3f}')

# ================================================================ config fold
hr('CONFIG FOLD (n=8 blocks) -- reported, NO significance / NO effect size cited')
for i in range(1, 6):
    print(f'  S{i}:', end='')
    for bf in (4, 5):
        for nc in (15, 30):
            sub = [e for e in KEEP if f'BF{bf}_{nc}comp' in e]
            dd = paired(f'W-S{i}', f'P-S{i}', sub)
            pos = sum(1 for x in dd if x > 0)
            print(f'  BF{bf}/{nc}c: med={np.median(dd):+.5f} {pos}/{len(dd)}', end='')
    print()

# ================================================================ curve shape
hr('SECONDARY  curve shape: nRMSE(cell)-nRMSE(NOTRACK) vs realised field displacement D')
print(f'  {"cell":14s} {"D_active":>12s} {"dNRMSE vs NOTRACK":>19s} {"pos/neg":>9s}')
for c in [f'W-S{i}' for i in range(1, 6)] + [f'P-S{i}' for i in range(1, 6)] + ['AUTO']:
    dw, dp = Dtab[c]
    act = dw if c.startswith('W') else (dp if c.startswith('P') else float('nan'))
    dd = paired(c, 'AUTO-NOTRACK', KEEP)
    pos, neg, tie, _ = sign_test(dd)
    print(f'  {c:14s} {act:12.6e} {np.median(dd):19.5f} {pos:4d}/{neg:<4d}')

# ================================================================ sec.7 ill-posed
hr('sec.7  ILL-POSED CLAUSE')
n_match = sum(1 for i in range(1, 6) if match[i])
matched_S = [i for i in range(1, 6) if match[i]]
alldegen = all(
    ((not DEG[f'W-S{i}']['vsAUTO'][4] and not DEG[f'P-S{i}']['vsAUTO'][4]) or
     (not DEG[f'W-S{i}']['vsNT'][4] and not DEG[f'P-S{i}']['vsNT'][4]))
    for i in matched_S) if matched_S else True
print(f'  matched pairs (D ratio in [0.5,2]): {n_match}  -> {matched_S}')
print(f'  condition A "matched pairs < 2": {n_match < 2}')
print(f'  condition B "every matched pair degenerate": {alldegen}')
print(f'  => ILL-POSED CLAUSE FIRES: {(n_match < 2) or alldegen}')

hr('sec.5 VERDICT ARITHMETIC')
passing = [i for i in range(1, 6) if C_keep[i]['thr']]
passing_matched = [i for i in passing if match[i]]
signs = set(np.sign(C_keep[i]['C']) for i in passing_matched)
print(f'  S passing threshold (any match status): {passing}')
print(f'  S passing threshold AND D-matched:      {passing_matched}')
print(f'  signs among matched+passing: {signs}')
if len(passing_matched) >= 3 and len(signs) == 1:
    v = 'CHANNEL-REAL'
elif len(passing_matched) <= 1:
    v = 'NO-DIFFERENCE'
elif len(signs) > 1:
    v = 'CROSSOVER'
else:
    v = 'INDETERMINATE (2 passing, same sign -- prereg gives no label)'
print(f'  => PRIMARY VERDICT (integrity permitting): {v}')
