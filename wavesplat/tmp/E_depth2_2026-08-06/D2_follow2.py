"""Depth-2 follow-ups #2: 2x2 null cells, held-out raydrop stratification, wp_crowded null,
sparse over-proposal check.  Same pre-specified conventions.  Run from /home/vexit/wavesplat/wavesplat.
"""
import json, os
import numpy as np
import h5py
from scipy.stats import binomtest, spearmanr

NULLS = json.load(open('tmp/E_depth2_2026-08-06/D2_nulls.json'))
CROWDED = [f'episode_BF{bf}_{nc}comp_seed{s}_{arm}.h5'
           for bf in (4, 5) for nc in (15, 30) for s in range(5)
           for arm in ('uniform', 'slope')]
SPARSE = ['episode_BF2_1comp_seed0_uniform.h5', 'episode_BF2_1comp_seed0_slope.h5',
          'episode_BF3_1comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_uniform.h5',
          'episode_BF3_2comp_seed0_slope.h5', 'episode_BF3_5comp_seed0_uniform.h5',
          'episode_BF3_5comp_seed1_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
          'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5',
          'episode_BF4_5comp_seed0_slope.h5', 'episode_BF4_5comp_seed1_uniform.h5']


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def lanes(rows, arm=None, field='avg_nRMSE'):
    d = {}
    for r in rows:
        if arm is not None and r.get('arm') != arm:
            continue
        v = r.get(field)
        if v is None:
            continue
        d.setdefault(r['episode'], {})[(r['est_seed'], r['diff_seed'])] = float(v)
    return d


def med(lm):
    return {e: float(np.median(list(v.values()))) for e, v in lm.items()}


def st(d):
    d = [x for x in d if x != 0]
    if not d:
        return (0, 0, float('nan'))
    neg = sum(1 for x in d if x < 0)
    return (neg, len(d), binomtest(neg, len(d), 0.5).pvalue)


jul = load('tmp/wp_integration_2026-07-25/arm_runs.jsonl')
v2, lr = med(lanes(jul, 'v2_konly')), med(lanes(jul, 'learned'))
N = {e: NULLS['data_finemesh_2026-07-23/' + e]['null_avg_nRMSE'] for e in CROWDED + SPARSE}

print("=" * 96)
print("G1 — 2x2 CELLS (ncomp x raydrop): which cell actually beats the flat sea?  n=10, floor p=0.00195")
print("=" * 96)
print(f"  {'cell':18s} {'v2':>7s} {'learned':>8s} {'null':>7s} | {'v2 beats null':>14s} "
      f"| {'lrn beats null':>15s} | {'GAIN':>8s} wins")
for nc in ('15comp', '30comp'):
    for rd in ('uniform', 'slope'):
        eps = [e for e in CROWDED if nc in e and e.endswith(rd + '.h5')]
        a = st([v2[e] - N[e] for e in eps])
        b = st([lr[e] - N[e] for e in eps])
        g = st([lr[e] - v2[e] for e in eps])
        print(f"  {nc}/{rd:8s} {np.median([v2[e] for e in eps]):7.4f} "
              f"{np.median([lr[e] for e in eps]):8.4f} {np.median([N[e] for e in eps]):7.4f} "
              f"| {a[0]}/{a[1]} p={a[2]:<8.3g} | {b[0]}/{b[1]} p={b[2]:<8.3g} "
              f"| {np.median([lr[e]-v2[e] for e in eps]):+8.4f} {g[0]}/{g[1]} p={g[2]:.3g}")
print()

print("=" * 96)
print("G2 — HELD-OUT 40 (auto vs v2): raydrop stratification of the routed-18 result")
print("=" * 96)
ha = load('tmp/heldout40_2026-07-26/arm_runs_auto_live.jsonl')
hv = load('tmp/heldout40_2026-07-26/arm_runs_v2_control.jsonl')
aM, vM = med(lanes(ha)), med(lanes(hv))
route = {}
for r in ha:
    route.setdefault(r['episode'], []).append(r.get('auto_selected'))
dom = {e: max(set(v), key=v.count) for e, v in route.items()}
eps_ho = sorted(set(aM) & set(vM))
NH = {e: NULLS['data_heldout40_2026-07-26/' + e]['null_avg_nRMSE'] for e in eps_ho}
for rd in ('uniform', 'slope'):
    for lab, sel in (('ALL', [e for e in eps_ho if e.endswith(rd + '.h5')]),
                     ('route=learned', [e for e in eps_ho if e.endswith(rd + '.h5')
                                        and dom[e] == 'learned'])):
        if not sel:
            continue
        d = [aM[e] - vM[e] for e in sel]
        w, n, p = st(d)
        print(f"  raydrop={rd:8s} {lab:14s} n={len(sel):2d} medΔ={np.median(d):+.4f} "
              f"wins {w}/{n} p={p:.3g} floor={2/2**max(n,1):.3g}")
print("  crowded-only (ncomp>=15) within held-out 40:")
cr = [e for e in eps_ho if int(e.split('comp')[0].split('_')[-1]) >= 15]
d = [aM[e] - vM[e] for e in cr]
w, n, p = st(d)
print(f"    n={len(cr)} medΔ={np.median(d):+.4f} wins {w}/{n} p={p:.3g}")
for nm, m in (('auto', aM), ('v2_konly', vM)):
    dd = [m[e] - NH[e] for e in cr]
    w, n, p = st(dd)
    print(f"    {nm:9s} med={np.median([m[e] for e in cr]):.4f} nullMed={np.median([NH[e] for e in cr]):.4f} "
          f"med(arm-null)={np.median(dd):+.4f} beats-null {w}/{n} p={p:.3g}")
print()

print("=" * 96)
print("G3 — wp_crowded (-45%): null control for BOTH quantities + raydrop stratum")
print("=" * 96)
ws = load('tmp/wp_crowded_2026-07-25/ws_runs.jsonl')
cond = load('tmp/wp_crowded_2026-07-25/cond_runs.jsonl')
W, C = {}, {}
for r in ws:
    if r.get('avg_nRMSE') is None:
        continue
    W.setdefault(r['episode'], []).append(min(float(r['init_nrmse']), float(r['avg_nRMSE'])))
for r in cond:
    if r.get('cond_polish_nrmse_Hs') is None:
        continue
    C.setdefault(r['episode'], []).append(float(r['cond_polish_nrmse_Hs']))
Wm = {e: float(np.median(v)) for e, v in W.items()}
Cm = {e: float(np.median(v)) for e, v in C.items()}
eps = [e for e in CROWDED if e in Wm and e in Cm]
print(f"  crowded episodes with both arms: {len(eps)}")
for nm, m in (('ws min(init,avg)', Wm), ('cond nowcast', Cm)):
    d = [m[e] - N[e] for e in eps]
    w, n, p = st(d)
    print(f"  {nm:18s} med={np.median([m[e] for e in eps]):.4f} nullMed={np.median([N[e] for e in eps]):.4f} "
          f"med(arm-null)={np.median(d):+.4f} beats-null {w}/{n} p={p:.3g}")
d = [Cm[e] - Wm[e] for e in eps]
w, n, p = st(d)
print(f"  GAIN(cond-ws) med={np.median(d):+.4f} rel={100*np.median(d)/np.median([Wm[e] for e in eps]):+.1f}% "
      f"wins {w}/{n} p={p:.3g}")
for rd in ('uniform', 'slope'):
    sel = [e for e in eps if e.endswith(rd + '.h5')]
    d = [Cm[e] - Wm[e] for e in sel]
    w, n, p = st(d)
    dn = [Cm[e] - N[e] for e in sel]
    wn, nn, pn = st(dn)
    print(f"    raydrop={rd:8s} n={len(sel)} GAIN med={np.median(d):+.4f} wins {w}/{n} p={p:.3g} "
          f"| cond beats null {wn}/{nn} p={pn:.3g}")
print()

print("=" * 96)
print("G4 — SPARSE: is the learned penalty an over-proposal (n_selected saturation) effect?")
print("=" * 96)
ns = {}
for r in load('tmp/wp_integration_2026-07-25/atoms_cache.jsonl'):
    ns.setdefault(r['episode'], []).append(r['n_selected'])
nsm = {e: float(np.median(v)) for e, v in ns.items()}
rows = []
for e in SPARSE:
    ncomp = int(e.split('comp')[0].split('_')[-1])
    rows.append((e, ncomp, nsm.get(e, np.nan), v2[e], lr[e], lr[e] - v2[e]))
print(f"  {'episode':44s} {'ncomp':>5s} {'n_sel':>6s} {'v2':>7s} {'learned':>8s} {'Δ':>8s}")
for e, nc, n_, a, b, d in sorted(rows, key=lambda r: -r[5]):
    print(f"  {e:44s} {nc:5d} {n_:6.1f} {a:7.4f} {b:8.4f} {d:+8.4f}")
x = np.array([r[2] for r in rows])
y = np.array([r[5] for r in rows])
rho, p = spearmanr(x, y)
print(f"  Spearman(n_selected, Δ) = {rho:+.3f} p={p:.4f}   over-proposal ratio med "
      f"= {np.median([r[2]/max(r[1],1) for r in rows]):.1f}x true ncomp")
print(f"  crowded over-proposal ratio med = "
      f"{np.median([nsm[e]/int(e.split('comp')[0].split('_')[-1]) for e in CROWDED if e in nsm]):.2f}x")
