"""Depth-2 follow-ups: null-by-stratum, count-vs-placement decomposition, confirm2 null,
projected sd for the proposed ckpt-swap contrast.  Same pre-specified conventions as D2_main.py.
Run from /home/vexit/wavesplat/wavesplat.
"""
import json, os
import numpy as np
import h5py
from scipy.stats import binomtest, wilcoxon, spearmanr, theilslopes

NULLS = json.load(open('tmp/E_depth2_2026-08-06/D2_nulls.json'))
CROWDED = [f'episode_BF{bf}_{nc}comp_seed{s}_{arm}.h5'
           for bf in (4, 5) for nc in (15, 30) for s in range(5)
           for arm in ('uniform', 'slope')]


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
v2n, lrn = med(lanes(jul, 'v2_konly', 'final_n')), med(lanes(jul, 'learned', 'final_n'))
v2i, lri = med(lanes(jul, 'v2_konly', 'init_nrmse')), med(lanes(jul, 'learned', 'init_nrmse'))
N = {e: NULLS['data_finemesh_2026-07-23/' + e]['null_avg_nRMSE'] for e in CROWDED}

print("=" * 96)
print("F1 — NULL by raydrop stratum (is the baseline broken only in one stratum?)")
print("=" * 96)
for rd in ('uniform', 'slope'):
    eps = [e for e in CROWDED if e.endswith(rd + '.h5')]
    for nm, m in (('v2_konly', v2), ('learned', lr)):
        d = [m[e] - N[e] for e in eps]
        w, n, p = st(d)
        print(f"  {rd:8s} {nm:9s} armMed={np.median([m[e] for e in eps]):.4f} "
              f"nullMed={np.median([N[e] for e in eps]):.4f} "
              f"med(arm-null)={np.median(d):+.4f} beats-null {w}/{n} p={p:.3g}")
    d = [lr[e] - v2[e] for e in eps]
    w, n, p = st(d)
    print(f"  {rd:8s} GAIN med={np.median(d):+.4f} wins {w}/{n} p={p:.3g}")
print()

print("=" * 96)
print("F2 — NULL by ncomp stratum")
print("=" * 96)
for nc in ('15comp', '30comp'):
    eps = [e for e in CROWDED if nc in e]
    for nm, m in (('v2_konly', v2), ('learned', lr)):
        d = [m[e] - N[e] for e in eps]
        w, n, p = st(d)
        print(f"  {nc} {nm:9s} armMed={np.median([m[e] for e in eps]):.4f} "
              f"med(arm-null)={np.median(d):+.4f} beats-null {w}/{n} p={p:.3g}")
print()

print("=" * 96)
print("F3 — COUNT vs PLACEMENT: does the gain survive where learned uses no more components?")
print("=" * 96)
dn = {e: lrn[e] - v2n[e] for e in CROWDED}
for lab, sel in (('Δfinal_n <= 0', [e for e in CROWDED if dn[e] <= 0]),
                 ('Δfinal_n  > 0', [e for e in CROWDED if dn[e] > 0]),
                 ('Δfinal_n == 0', [e for e in CROWDED if dn[e] == 0])):
    if not sel:
        continue
    d = [lr[e] - v2[e] for e in sel]
    w, n, p = st(d)
    print(f"  {lab:14s} n={len(sel):2d} medΔ={np.median(d):+.4f} "
          f"rel={100*np.median(d)/np.median([v2[e] for e in sel]):+.1f}% wins {w}/{n} p={p:.3g} "
          f"floor_p={2/2**len(sel):.3g}")
sl = theilslopes([lr[e] - v2[e] for e in CROWDED], [dn[e] for e in CROWDED])
print(f"  Theil-Sen slope of Δavg on Δfinal_n = {sl[0]:+.5f} nRMSE per component "
      f"(CI {sl[2]:+.5f}..{sl[3]:+.5f})")
print(f"  => projected effect of the new ckpt's +4..+8 components: "
      f"{sl[0]*4:+.4f} .. {sl[0]*8:+.4f}  [ARGUED, not measured]")
print()

print("=" * 96)
print("F4 — INIT vs TRACK: where does the gain live, and does the tracker give it back?")
print("=" * 96)
di = np.array([lri[e] - v2i[e] for e in CROWDED])
da = np.array([lr[e] - v2[e] for e in CROWDED])
print(f"  med Δinit={np.median(di):+.4f}  med Δavg={np.median(da):+.4f}  "
      f"med(Δavg-Δinit)={np.median(da-di):+.4f} (positive = tracker erodes the init gain)")
w, n, p = st(list(da - di))
print(f"  tracker erodes in {n-w}/{n} episodes (sign p={p:.3g})")
for nm, m, mi in (('v2_konly', v2, v2i), ('learned', lr, lri)):
    print(f"  {nm:9s}: init med={np.median([mi[e] for e in CROWDED]):.4f} -> "
          f"track med={np.median([m[e] for e in CROWDED]):.4f} "
          f"(tracker changes {np.median([m[e]-mi[e] for e in CROWDED]):+.4f})")
print()

print("=" * 96)
print("F5 — CONFIRM2 (new ckpt, fresh 40): null control + raydrop stratum")
print("=" * 96)
c2dir = 'data_confirm2_2026-07-27'
if os.path.isdir(c2dir):
    eps2 = sorted(e for e in os.listdir(c2dir) if e.endswith('.h5'))
    nl2 = {}
    for e in eps2:
        with h5py.File(os.path.join(c2dir, e), 'r') as h:
            Hs = float(h['gt_params'].attrs['Hs_true'])
            Tp = float(h['gt_params'].attrs['Tp_true'])
            ts = h['frames/timestamps'][:]
            idx = np.where(ts > 2 * Tp)[0]
            g = h['frames/gt_wave_field']
            nl2[e] = float(np.mean([np.sqrt(np.mean(g[i] ** 2)) / max(Hs, 1e-6) for i in idx]))
    json.dump(nl2, open('tmp/E_depth2_2026-08-06/D2_nulls_confirm2.json', 'w'), indent=1)
    lrn2 = load('tmp/confirm2_2026-07-27/arm_runs_confirm2_v1.jsonl')
    v2m3 = load('tmp/confirm2_2026-07-27/arm_runs_v2m3.jsonl')
    at = {}
    for r in load('tmp/confirm2_2026-07-27/atoms_confirm2_v1.jsonl'):
        at[(r['episode'], r.get('diff_seed', r.get('seed')))] = r['n_selected']
    L = lanes(lrn2)
    V = lanes(v2m3)
    auto = {}
    for e in L:
        vals = []
        for k in L[e]:
            ns = at.get((e, k[1]))
            vals.append(L[e][k] if (ns is not None and ns >= 8) else V.get(e, {}).get(k, np.nan))
        auto[e] = float(np.nanmedian(vals))
    Vm = med(V)
    crowd = [e for e in auto if int(e.split('comp')[0].split('_')[-1]) >= 15]
    print(f"  confirm2 episodes={len(auto)} crowded={len(crowd)} "
          f"nullMed={np.median([nl2[e] for e in crowd]):.4f}")
    for nm, m in (('auto(assembled)', auto), ('v2_m3', Vm)):
        d = [m[e] - nl2[e] for e in crowd]
        w, n, p = st(d)
        print(f"  crowded {nm:16s} med={np.median([m[e] for e in crowd]):.4f} "
              f"med(arm-null)={np.median(d):+.4f} beats-null {w}/{n} p={p:.3g}")
    d = [auto[e] - Vm[e] for e in crowd]
    w, n, p = st(d)
    print(f"  crowded GAIN med={np.median(d):+.4f} wins {w}/{n} p={p:.3g} floor={2/2**len(crowd):.3g}")
    for rd in ('uniform', 'slope'):
        sel = [e for e in crowd if e.endswith(rd + '.h5')]
        if sel:
            d = [auto[e] - Vm[e] for e in sel]
            w, n, p = st(d)
            print(f"    raydrop={rd:8s} n={len(sel)} medΔ={np.median(d):+.4f} wins {w}/{n} "
                  f"p={p:.3g} floor={2/2**len(sel):.3g}")
else:
    print("  data_confirm2_2026-07-27 not found")
print()

print("=" * 96)
print("F6 — SD components for the PROPOSED ckpt-swap contrast (power realism)")
print("=" * 96)
Lc = lanes(jul, 'learned')
lane_sd = np.median([np.std(list(Lc[e].values()), ddof=1) for e in CROWDED])
paired_sd = np.std([lr[e] - v2[e] for e in CROWDED], ddof=1)
print(f"  within-episode lane sd (learned, crowded) med = {lane_sd:.4f}")
print(f"  paired episode-level Δ sd (learned - v2)      = {paired_sd:.4f}")
print(f"  MDE80 if ckpt swap adds ONLY lane noise  (n=40): {2.8016*lane_sd*np.sqrt(2/5)/np.sqrt(40):.4f}")
print(f"  MDE80 if ckpt swap behaves like an init-method change (n=40): "
      f"{2.8016*paired_sd/np.sqrt(40):.4f}")
