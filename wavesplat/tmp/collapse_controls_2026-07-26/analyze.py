"""Scoring of the frozen escape grid. PREREG: tmp/collapse_controls_2026-07-26/PREREG.md.
No parameter is chosen here; every cell was fixed before escapes.json existed."""
import os, sys, json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
D = os.path.join(ROOT, 'tmp/collapse_controls_2026-07-26')
esc = json.load(open(os.path.join(D, 'escapes.json')))
hv = json.load(open(os.path.join(ROOT, 'tmp/heldout_real_2026-07-26/harvest_heldout.json')))
hd = json.load(open(os.path.join(ROOT, 'tmp/unified_2026-07-26/hand_amp.json')))

BASE = {(r['record'], r['win']): r['hand_now'] for r in hv['rows'] if r.get('hand_now') is not None}
for r in hd['rows']:
    BASE[(r['record'], r['win'])] = r['hand_now']
LEARNED = {(r['record'], r['win']): r['learned_now'] for r in hd['rows']}
AMP = {(r['record'], r['win']): r['amp_ratio_grid'] for r in hv['rows']}
AMP.update({(r['record'], r['win']): r['amp_ratio_grid'] for r in hd['rows']})

rows = esc['rows']
print('=' * 78)
print('0. BIT-IDENTITY CONTROL (baseline recheck vs stored harvest value)')
mx = 0.0
for r in rows:
    b = BASE[(r['record'], r['win'])]
    mx = max(mx, abs(r['baseline_recheck'] - b))
print(f'  max |recheck - stored| over 18 windows = {mx:.3e}   ->  '
      f'{"PASS" if mx < 1e-9 else "FAIL — RUN INVALID"}')
assert mx < 1e-9, 'baseline mismatch: run invalid per PREREG'

# ---- assemble readings -----------------------------------------------------
DEPLOY = ['E1-median', 'E2-peak2', 'E2-half', 'E2-double', 'E3-half', 'E3-double']
ORACLE = ['E1-best', 'E2-bestvar', 'E3-bestvar', 'ANY-bestvar']
READ = {}
for r in rows:
    k = (r['record'], r['win'])
    e1 = [v for v in r['E1'].values() if v is not None]
    e2 = {'E2-peak2': r['E2']['peak2'], 'E2-half': r['E2']['kp_x0.5'],
          'E2-double': r['E2']['kp_x2.0']}
    e3 = {'E3-half': r['E3']['sig_x0.5'], 'E3-double': r['E3']['sig_x2.0']}
    d = {'E1-median': float(np.median(e1)), 'E1-best': float(np.min(e1)), **e2, **e3}
    ok2 = [v for v in e2.values() if v is not None]
    d['E2-bestvar'] = float(np.min(ok2)) if ok2 else None
    d['E3-bestvar'] = float(np.min(list(e3.values())))
    alls = [v for v in list(d.values()) if v is not None]
    d['ANY-bestvar'] = float(np.min(alls))
    READ[k] = d

keys = [(r['record'], r['win']) for r in rows]
hkeys = [k for k in keys if (k[0], k[1]) not in [('AA14', 0), ('AA15a', 3), ('LJ18g', 3)]]
dkeys = [k for k in keys if k not in hkeys]


def counts(name, ks):
    a = b = 0; na = 0
    for k in ks:
        v = READ[k][name]
        if v is None:
            na += 1; continue
        if BASE[k] - v >= 0.5:
            a += 1
        if v < 1.0:
            b += 1
    return a, b, na


print()
print('=' * 78)
print('1. FROZEN-CELL COUNTS over all 18 collapse windows')
print('   (a) LARGE REDUCTION  base - escape >= 0.5     (b) BELOW-LINE  escape < 1.0')
print()
print('  DEPLOYABLE (GT-free, single config / median)      (a)/18   (b)/18')
for nm in DEPLOY:
    a, b, na = counts(nm, keys)
    print(f'    {nm:<14s} {a:>3d}/18   {b:>3d}/18' + (f'   [{na} n/a]' if na else ''))
print()
print('  ORACLE-FLAVORED (winner picked with ground truth — NOT eligible for any bar)')
for nm in ORACLE:
    a, b, na = counts(nm, keys)
    print(f'    {nm:<14s} {a:>3d}/18   {b:>3d}/18' + (f'   [{na} n/a]' if na else ''))

ca = max(counts(n, keys)[0] for n in DEPLOY)
cb = max(counts(n, keys)[1] for n in DEPLOY)
arg_a = [n for n in DEPLOY if counts(n, keys)[0] == ca]
arg_b = [n for n in DEPLOY if counts(n, keys)[1] == cb]
print()
print(f'  C_best(a) = {ca}/18  ({", ".join(arg_a)})')
print(f'  C_best(b) = {cb}/18  ({", ".join(arg_b)})')
print(f'  P2 bar for the learned arm (>= 1.5 x C_best): '
      f'(a) needs >= {int(np.ceil(1.5*ca))}/18 ; (b) needs >= {int(np.ceil(1.5*cb))}/18')

print()
print('  split by arena:')
for nm in DEPLOY + ORACLE:
    ah, bh, _ = counts(nm, hkeys); ad, bd, _ = counts(nm, dkeys)
    print(f'    {nm:<14s} held-out15 (a){ah:>3d} (b){bh:>3d}   design3 (a){ad} (b){bd}')

print()
print('=' * 78)
print('2. DESIGN-3 SPOT CHECK vs the KNOWN learned repairs (n=3, underpowered)')
print('   learned: AA14_w0 2.309->1.071 | AA15a_w3 1.120->0.791 | LJ18g_w3 2.925->0.961')
la = sum(1 for k in dkeys if BASE[k] - LEARNED[k] >= 0.5)
lb = sum(1 for k in dkeys if LEARNED[k] < 1.0)
print(f'   learned counts on design-3: (a) {la}/3   (b) {lb}/3')
print()
hdr = f'   {"window":<12s}{"base":>7s}{"learned":>9s}' + ''.join(f'{n:>12s}' for n in DEPLOY)
print(hdr)
for k in dkeys:
    print(f'   {k[0]+"_w"+str(k[1]):<12s}{BASE[k]:>7.3f}{LEARNED[k]:>9.3f}'
          + ''.join(('        n/a' if READ[k][n] is None else f'{READ[k][n]:>12.3f}') for n in DEPLOY))
print()
for nm in DEPLOY:
    a, b, _ = counts(nm, dkeys)
    tag = []
    if a >= la: tag.append(f'MATCHES/BEATS learned on (a) [{a} vs {la}]')
    if b >= lb: tag.append(f'MATCHES/BEATS learned on (b) [{b} vs {lb}]')
    print(f'   {nm:<14s} (a){a}/3 (b){b}/3   ' + ('; '.join(tag) if tag else ''))

print()
print('=' * 78)
print('3. PER-WINDOW TABLE (all 18)')
print(f'   {"window":<12s}{"arena":<9s}{"base":>7s}{"amp":>7s}'
      + ''.join(f'{n:>11s}' for n in DEPLOY) + f'{"E1-best":>10s}{"ANY-best":>10s}')
for k in keys:
    r = READ[k]
    print(f'   {k[0]+"_w"+str(k[1]):<12s}'
          f'{("design" if k in dkeys else "heldout"):<9s}{BASE[k]:>7.3f}{AMP[k]:>7.3f}'
          + ''.join(('       n/a' if r[n] is None else f'{r[n]:>11.3f}') for n in DEPLOY)
          + f'{r["E1-best"]:>10.3f}{r["ANY-bestvar"]:>10.3f}')

# ---- E1 dispersion diagnostic ---------------------------------------------
print()
print('=' * 78)
print('4. E1 DIAGNOSTIC — how much run-to-run freedom does the val split actually give?')
nid = 0; sds = []
for r in rows:
    vals = list(r['E1'].values()) + [r['baseline_recheck']]
    sds.append(float(np.std(list(r['E1'].values()))))
    nid += sum(1 for v in r['E1'].values() if abs(v - r['baseline_recheck']) < 1e-12)
print(f'   E1 variants numerically IDENTICAL to the baseline: {nid}/90 runs')
print(f'   per-window sd of the 5 E1 readings: median {np.median(sds):.4f}, '
      f'min {min(sds):.4f}, max {max(sds):.4f}')

# ---- EXPLORATORY -----------------------------------------------------------
print()
print('=' * 78)
print('5. EXPLORATORY (PREREG s4) — two-sided amplitude rule on all 78 real windows')
allrows = [dict(record=r['record'], win=r['win'], hand=r['hand_now'], amp=r['amp_ratio_grid'],
                arena='heldout') for r in hv['rows'] if r.get('hand_now') is not None]
allrows += [dict(record=r['record'], win=r['win'], hand=r['hand_now'], amp=r['amp_ratio_grid'],
                 arena='design') for r in hd['rows']]
print(f'   n = {len(allrows)} windows')
col = [r for r in allrows if r['hand'] >= 1.0]
nor = [r for r in allrows if r['hand'] < 1.0]
ca_, na_ = np.array([r['amp'] for r in col]), np.array([r['amp'] for r in nor])
print(f'   collapses {len(col)}  amp range [{ca_.min():.3f}, {ca_.max():.3f}] median {np.median(ca_):.3f}')
print(f'   normals   {len(nor)}  amp range [{na_.min():.3f}, {na_.max():.3f}] median {np.median(na_):.3f}')
lo_col = ca_[ca_ < 1.5]; lo_nor = na_[na_ < 1.5]
print(f'   suppressed-mode collapses (amp < 1.5): n={len(lo_col)}  '
      f'range [{lo_col.min():.3f}, {lo_col.max():.3f}]')
print(f'   normals in the same band:              n={len(lo_nor)}  '
      f'range [{lo_nor.min():.3f}, {lo_nor.max():.3f}]')
print('   -> the low-amp collapse range is ENTIRELY CONTAINED in the normal range: '
      f'{bool(lo_col.min() >= lo_nor.min() and lo_col.max() <= lo_nor.max())}')
# best attainable two-sided rule, FITTED ON THIS DATA (not deployable)
cands = np.unique(np.concatenate([ca_, na_]))
best = None
for lo in np.concatenate([[-1.0], cands]):
    for hi in np.concatenate([cands, [1e9]]):
        if hi <= lo:
            continue
        fire = lambda a: (a < lo) | (a > hi)
        tp = int(fire(ca_).sum()); fp = int(fire(na_).sum())
        yj = tp / len(ca_) - fp / len(nor)
        if best is None or yj > best[0]:
            best = (yj, lo, hi, tp, fp)
yj, lo, hi, tp, fp = best
print(f'   BEST two-sided rule FITTED ON THESE SAME 78 WINDOWS (in-sample ceiling, NOT deployable):')
print(f'     fire if amp < {lo:.3f} or amp > {hi:.3f}  ->  sens {tp}/{len(col)} = {tp/len(col):.2f}, '
      f'false fires {fp}/{len(nor)} = {fp/len(nor):.2f}, Youden J = {yj:.3f}')
print(f'   frozen one-sided tau=1.5 on the same data: sens {int((ca_>1.5).sum())}/{len(col)}, '
      f'false fires {int((na_>1.5).sum())}/{len(nor)}')
# low-side only
bestlo = None
for lo2 in np.concatenate([[-1.0], cands]):
    tp = int((ca_ < lo2).sum()); fp = int((na_ < lo2).sum())
    yj2 = tp / len(ca_) - fp / len(nor)
    if bestlo is None or yj2 > bestlo[0]:
        bestlo = (yj2, lo2, tp, fp)
print(f'   BEST LOW-SIDE-ONLY rule (in-sample): amp < {bestlo[1]:.3f} -> '
      f'sens {bestlo[2]}/{len(col)}, false fires {bestlo[3]}/{len(nor)}, J = {bestlo[0]:.3f}')
# rank statistic
from itertools import product
auc_lo = np.mean([1.0 if a < b else (0.5 if a == b else 0.0) for a, b in product(ca_, na_)])
print(f'   P(amp_collapse < amp_normal) = {auc_lo:.3f}  (0.5 = no low-side information)')

json.dump(dict(readings={f'{k[0]}_w{k[1]}': READ[k] for k in keys},
               base={f'{k[0]}_w{k[1]}': BASE[k] for k in keys},
               c_best_a=ca, c_best_b=cb, argmax_a=arg_a, argmax_b=arg_b,
               counts_all18={n: dict(zip(('a', 'b', 'na'), counts(n, keys)))
                             for n in DEPLOY + ORACLE},
               counts_heldout15={n: dict(zip(('a', 'b', 'na'), counts(n, hkeys)))
                                 for n in DEPLOY + ORACLE},
               counts_design3={n: dict(zip(('a', 'b', 'na'), counts(n, dkeys)))
                               for n in DEPLOY + ORACLE},
               learned_design3=dict(a=la, b=lb),
               exploratory=dict(n=len(allrows), n_collapse=len(col), n_normal=len(nor),
                                best_twosided=dict(lo=float(lo), hi=float(hi), tp=int(tp),
                                                   fp=int(fp), youden=float(yj)),
                                best_lowside=dict(thr=float(bestlo[1]), tp=int(bestlo[2]),
                                                  fp=int(bestlo[3]), youden=float(bestlo[0])),
                                p_amp_collapse_lt_normal=float(auc_lo))),
          open(os.path.join(D, 'results.json'), 'w'), indent=1)
print(f"\nwrote {os.path.join(D, 'results.json')}")

# ---- POST-HOC ADDENDUM (NOT pre-registered; labelled as such) ---------------
np_path = os.path.join(D, 'normals_posthoc.json')
if os.path.exists(np_path):
    nph = json.load(open(np_path))['rows']
    d = np.array([r['base'] - r['e2_double'] for r in nph])
    newcol = [r for r in nph if r['e2_double'] >= 1.0]
    print()
    print('=' * 78)
    print('6. POST-HOC (NOT PRE-REGISTERED) — what does the winning escape cost elsewhere?')
    print(f'   kp_hint x2 on the {len(nph)} NORMAL windows:')
    print(f'     improved {int((d>0).sum())}/{len(d)}, worsened {int((d<0).sum())}/{len(d)}, '
          f'median delta {np.median(d):+.4f}, mean {d.mean():+.4f}')
    print(f'     NEW collapses created (normal -> >=1.0): {len(newcol)}  '
          f'({", ".join(r["record"]+"_w"+str(r["win"]) for r in newcol)})')
    rep = sum(1 for k in keys if READ[k]['E2-double'] < 1.0)
    print(f'   NET as a blanket default over all 78 windows: '
          f'{18} collapses -> {18-rep} surviving + {len(newcol)} new = {18-rep+len(newcol)}')
    print('   => kp_hint x2 is NOT a better default; it is a genuine trade, i.e. a real escape.')
    print()
    print('   OPERATIONAL SUBSET (post-hoc but deployment-defined): the 5 windows the frozen')
    print('   tau=1.5 interlock actually FIRES on — the only windows where a fallback runs.')
    fired = [k for k in keys if AMP[k] > 1.5]
    print(f'   {"window":<12s}{"base":>8s}{"amp":>8s}{"E2-double":>11s}{"learned":>10s}')
    fa = fb = 0
    for k in fired:
        lv = LEARNED.get(k)
        e = READ[k]['E2-double']
        fa += (BASE[k] - e >= 0.5); fb += (e < 1.0)
        print(f'   {k[0]+"_w"+str(k[1]):<12s}{BASE[k]:>8.3f}{AMP[k]:>8.3f}{e:>11.3f}'
              + (f'{lv:>10.3f}' if lv is not None else f'{"PENDING":>10s}'))
    print(f'   E2-double on the fired subset: (a) {fa}/{len(fired)}   (b) {fb}/{len(fired)}')
    la2 = sum(1 for k in fired if LEARNED.get(k) is not None and BASE[k] - LEARNED[k] >= 0.5)
    lb2 = sum(1 for k in fired if LEARNED.get(k) is not None and LEARNED[k] < 1.0)
    nk = sum(1 for k in fired if LEARNED.get(k) is not None)
    print(f'   learned on the fired subset:   (a) {la2}/{nk} known, (b) {lb2}/{nk} known '
          f'({len(fired)-nk} PENDING-GPU)')
    res = json.load(open(os.path.join(D, 'results.json')))
    res['posthoc'] = dict(normals_n=len(nph), improved=int((d > 0).sum()),
                          worsened=int((d < 0).sum()), median_delta=float(np.median(d)),
                          new_collapses=[f'{r["record"]}_w{r["win"]}' for r in newcol],
                          net_collapses_before=18, net_collapses_after=18 - rep + len(newcol),
                          fired_subset=[f'{k[0]}_w{k[1]}' for k in fired],
                          fired_e2double_a=int(fa), fired_e2double_b=int(fb),
                          fired_learned_known=int(nk))
    json.dump(res, open(os.path.join(D, 'results.json'), 'w'), indent=1)
