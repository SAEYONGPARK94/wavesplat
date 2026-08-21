#!/usr/bin/env python3
"""CON audit — independent re-derivation of Rx-5 / E-sweep headlines.
Written from scratch (no import of run_rx5 scorer). Read-only on all inputs.
"""
import json, os, itertools, math
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results.jsonl')
NULLS = '/home/vexit/wavesplat/wavesplat/tmp/K6_run_2026-08-07/nulls_k6.json'

# ---------------- load ----------------
recs = []
KEEP = ('cell','episode','seed','nRMSE','init_nrmse','Pearson_rho','absVR_framewise',
        'm_mean','decorr_mean','shrink_mean','N','auto_selected','auto_n_selected',
        'ensemble_k','final_n','init_rmse','elapsed_ms','Variance_Ratio','Skill_Score',
        'Directional_R2','Hs_error','avg_RMSE_unobserved','wall_s')
for line in open(RES):
    r = json.loads(line)
    d = {k: r.get(k) for k in KEEP}
    d['spec'] = r.get('spec')
    d['cfg'] = r.get('cfg')
    d['_w'] = r.get('w')
    d['_kx'] = r.get('kx'); d['_ky'] = r.get('ky')
    d['_vr'] = r.get('vr_frames'); d['_rho'] = r.get('rho_frames')
    recs.append(d)

cells = sorted({r['cell'] for r in recs})
eps   = sorted({r['episode'] for r in recs})
seeds = sorted({r['seed'] for r in recs})
print(f"loaded {len(recs)} runs | {len(cells)} cells | {len(eps)} eps | seeds {seeds}")

by = {(r['cell'], r['episode'], r['seed']): r for r in recs}
assert len(by) == len(recs), "duplicate keys"

def med(xs):
    xs = sorted(xs); n = len(xs)
    return xs[n//2] if n % 2 else 0.5*(xs[n//2-1]+xs[n//2])

# seed-median per (cell, ep) for a metric
def block(cell, metric):
    out = {}
    for e in eps:
        vals = [by[(cell,e,s)][metric] for s in seeds if (cell,e,s) in by]
        vals = [v for v in vals if v is not None]
        if len(vals) == len(seeds):
            out[e] = med(vals)
    return out

# exact two-sided sign test
def signtest(deltas, direction='less'):
    """deltas: list of (a-b). ties dropped. returns n, n_neg, n_pos, p_two_sided."""
    nz = [d for d in deltas if d != 0.0]
    n = len(nz)
    npos = sum(1 for d in nz if d > 0); nneg = n - npos
    k = min(npos, nneg)
    if n == 0: return 0,0,0,1.0
    p = 0.0
    for i in range(0, k+1):
        p += math.comb(n, i)
    p = min(1.0, 2*p/(2**n))
    return n, nneg, npos, p

def paired(cellA, cellB, metric='nRMSE', epsub=None):
    A = block(cellA, metric); B = block(cellB, metric)
    use = epsub if epsub is not None else eps
    ds = [(e, A[e]-B[e]) for e in use if e in A and e in B]
    vals = [d for _,d in ds]
    n, nneg, npos, p = signtest(vals)
    return dict(cellA=cellA, cellB=cellB, metric=metric, n=len(vals),
                medD=med(vals) if vals else float('nan'),
                n_eff=n, wins_neg=nneg, wins_pos=npos, p=p, per_ep=dict(ds))

# ---------------- null gate ----------------
nulls = {d['episode']: d for d in json.load(open(NULLS))}
autoB = block('AUTO','nRMSE')
excluded = []
kept = []
for e in eps:
    nl = nulls[e]['zero_null_tracking_nRMSE_mean']
    if autoB[e] < nl: kept.append(e)
    else: excluded.append(e)
print(f"\nNULL GATE (AUTO vs zero-null): kept {len(kept)} / excluded {len(excluded)}")
for e in excluded:
    print(f"   EXCLUDED {e}  AUTO={autoB[e]:.4f} null={nulls[e]['zero_null_tracking_nRMSE_mean']:.4f}")

# ---------------- levels ----------------
print("\n=== BLOCK LEVELS (median over ep of seed-median nRMSE) — descriptive only ===")
lv = {}
for c in cells:
    b = block(c,'nRMSE')
    lv[c] = med([b[e] for e in kept])
    print(f"  {c:14s} kept-block-median={lv[c]:.4f}   all16={med(list(b.values())):.4f}")

json.dump(dict(levels=lv, kept=kept, excluded=excluded),
          open(os.path.join(HERE,'audit_con_levels.json'),'w'), indent=1)

# ---------------- ATTACK 1: is lr->0 just interpolation to OFF? ----------------
print("\n" + "="*78)
print("ATTACK 1 — dose-response = interpolation to AUTO-NOTRACK?")
print("="*78)
LRS = [('W-5e3',5e-3),('W-5e4',5e-4),('W-MATCH',3.508993677838184e-05),('W-5e6',5e-6),
       ('AUTO-NOTRACK',0.0)]
print(f"{'cell':14s} {'lr':>12s} {'relstep%':>9s} {'kept-med':>9s} {'medD vs NOTRACK':>16s} {'n-/n+':>8s} {'p':>8s}")
for c,lr in LRS:
    r = paired(c,'AUTO-NOTRACK','nRMSE',kept)
    rel = 100*lr/0.044095
    print(f"{c:14s} {lr:12.3e} {rel:9.3f} {lv[c]:9.4f} {r['medD']:16.5f} {r['wins_neg']}/{r['wins_pos']:<4} {r['p']:8.4f}")

print("\n-- monotonicity of block level vs lr --")
seq = [(lr, lv[c], c) for c,lr in LRS]
print("   ", [(f"{c}:{v:.4f}") for lr,v,c in seq])

print("\n-- W-MATCH vs AUTO-NOTRACK, all metrics --")
for m in ['nRMSE','Pearson_rho','absVR_framewise','m_mean','decorr_mean','shrink_mean']:
    r = paired('W-MATCH','AUTO-NOTRACK',m,kept)
    print(f"   {m:18s} medD={r['medD']:+.5f}  n-/n+={r['wins_neg']}/{r['wins_pos']}  p={r['p']:.4f}")

print("\n-- W-5e6 vs AUTO-NOTRACK (should be nearly identical if pure interpolation) --")
for m in ['nRMSE','Pearson_rho','m_mean']:
    r = paired('W-5e6','AUTO-NOTRACK',m,kept)
    print(f"   {m:18s} medD={r['medD']:+.5f}  n-/n+={r['wins_neg']}/{r['wins_pos']}  p={r['p']:.4f}")

print("\n-- Per-episode |W-MATCH - NOTRACK| vs |W-5e3 - NOTRACK| (kept) --")
wm = block('W-MATCH','nRMSE'); nt = block('AUTO-NOTRACK','nRMSE'); w3 = block('W-5e3','nRMSE')
au = block('AUTO','nRMSE')
import statistics
d_wm = [abs(wm[e]-nt[e]) for e in kept]
d_w3 = [abs(w3[e]-nt[e]) for e in kept]
d_au = [abs(au[e]-nt[e]) for e in kept]
print(f"   median |W-MATCH - NOTRACK| = {med(d_wm):.5f}")
print(f"   median |W-5e3   - NOTRACK| = {med(d_w3):.5f}")
print(f"   median |AUTO    - NOTRACK| = {med(d_au):.5f}")
print(f"   ratio  W-MATCH/AUTO distance to floor = {med(d_wm)/med(d_au):.3f}")

# ---------------- ATTACK 2: PRIMARY-A ----------------
print("\n" + "="*78)
print("ATTACK 2 — PRIMARY-A: channel contrast or on/off contrast?")
print("="*78)
def primaryA(epsub, label):
    A = block('W-MATCH','nRMSE'); P = block('PHI-5e3','nRMSE'); AU = block('AUTO','nRMSE')
    ds = [ (A[e]-AU[e]) - (P[e]-AU[e]) for e in epsub ]
    # identical to A-P
    ds2 = [ A[e]-P[e] for e in epsub ]
    assert max(abs(a-b) for a,b in zip(ds,ds2)) < 1e-12
    n,nneg,npos,p = signtest(ds)
    print(f"  {label:26s} medD={med(ds):+.5f}  neg(W-MATCH better)={nneg} pos={npos} n={n} p={p:.4f}")
    return ds
dsk = primaryA(kept,'PRIMARY-A (null-gated)')
dsa = primaryA(eps,'PRIMARY-A (all 16)')

print("\n  Decomposition: is W-MATCH == floor, PHI-5e3 == real tracking?")
for c in ['W-MATCH','PHI-5e3','W-5e6','W-5e3','AUTO']:
    r = paired(c,'AUTO-NOTRACK','nRMSE',kept)
    print(f"    {c:14s} vs NOTRACK: medD={r['medD']:+.5f} n-/n+={r['wins_neg']}/{r['wins_pos']} p={r['p']:.4f}")

print("\n  ** The on/off surrogate test **")
print("  If PRIMARY-A is really (off vs on), then substituting AUTO-NOTRACK for W-MATCH")
print("  should reproduce it:")
r = paired('AUTO-NOTRACK','PHI-5e3','nRMSE',kept)
print(f"    AUTO-NOTRACK - PHI-5e3 : medD={r['medD']:+.5f} n-/n+={r['wins_neg']}/{r['wins_pos']} p={r['p']:.4f}")
r2 = paired('W-MATCH','PHI-5e3','nRMSE',kept)
print(f"    W-MATCH      - PHI-5e3 : medD={r2['medD']:+.5f} n-/n+={r2['wins_neg']}/{r2['wins_pos']} p={r2['p']:.4f}")
print(f"    -> difference of the two point estimates: {r2['medD']-r['medD']:+.5f}")
# correlation of per-ep deltas
import statistics as st
a = [r['per_ep'][e] for e in kept]; b = [r2['per_ep'][e] for e in kept]
def pear(x,y):
    mx,my=sum(x)/len(x),sum(y)/len(y)
    num=sum((xi-mx)*(yi-my) for xi,yi in zip(x,y))
    den=math.sqrt(sum((xi-mx)**2 for xi in x)*sum((yi-my)**2 for yi in y))
    return num/den if den else float('nan')
print(f"    per-episode Pearson r(NOTRACK-PHI , WMATCH-PHI) = {pear(a,b):.4f}")

print("\n  ** Channel contrast at MATCHED relative step but nonzero drive **")
print("  W-5e4 (1.13% rel) is 14x the phase rel-step; W-5e3 is 142x.")
for c in ['W-5e3','W-5e4','W-MATCH','W-5e6']:
    r = paired(c,'PHI-5e3','nRMSE',kept)
    print(f"    {c:8s} - PHI-5e3: medD={r['medD']:+.5f} n-/n+={r['wins_neg']}/{r['wins_pos']} p={r['p']:.4f}")

# ---------------- ATTACK 3: E-sweep ratio ----------------
print("\n" + "="*78)
print("ATTACK 3 — E-sweep R(E) ratio artifact")
print("="*78)
ECELLS = [('E-2',2),('E-5',5),('E-10',10),('E-20',20),('E-40',40),('E-80',80)]
AU = block('AUTO','nRMSE'); NT = block('AUTO-NOTRACK','nRMSE')
print("  denominator = (AUTO - NOTRACK) per episode  [damage full tracking does]")
den = {e: AU[e]-NT[e] for e in eps}
print(f"  denominator stats (kept): min={min(den[e] for e in kept):+.5f} med={med([den[e] for e in kept]):+.5f} max={max(den[e] for e in kept):+.5f}")
neg_den = [e for e in kept if den[e] <= 0]
print(f"  episodes where AUTO is BETTER than floor (den<=0): {len(neg_den)} -> ratio sign flips")
for e in kept:
    print(f"    {e[:46]:48s} AUTO={AU[e]:.4f} NT={NT[e]:.4f} den={den[e]:+.5f}")

print("\n  R(E) = 1 - (AUTO - E_cell)/(AUTO - NOTRACK) ... i.e. fraction of damage NOT yet recovered")
print("  (block-ratio) vs (per-episode-ratio median) vs (paired sign test vs uniform null)")
Ecache = {}
for c,E in ECELLS:
    B = block(c,'nRMSE'); Ecache[c]=B
    # block ratio using block medians
    br = (med([B[e] for e in kept]) - med([NT[e] for e in kept])) / (med([AU[e] for e in kept]) - med([NT[e] for e in kept]))
    per = []
    for e in kept:
        if abs(den[e]) > 1e-12:
            per.append((B[e]-NT[e])/den[e])
    pred = 1.0 - E/160.0
    dd = [ (B[e]-NT[e])/den[e] - pred for e in kept if abs(den[e])>1e-12]
    n,nneg,npos,p = signtest(dd)
    r_auto = paired(c,'AUTO','nRMSE',kept)
    print(f"  {c:6s} E={E:3d} pred={100*pred:5.1f}%  blockR={100*br:7.1f}%  perEpMedR={100*med(per):7.1f}%"
          f"  [min {100*min(per):8.1f}, max {100*max(per):8.1f}]  vs-null n-/n+={nneg}/{npos} p={p:.3f}"
          f"  | vs AUTO medD={r_auto['medD']:+.5f} {r_auto['wins_neg']}/{r_auto['wins_pos']} p={r_auto['p']:.3f}")

print("\n  --- per-episode R(E=10) sorted by |denominator| (small denom -> blowup?) ---")
B10 = Ecache['E-10']
rows = []
for e in kept:
    if abs(den[e])>1e-12:
        rows.append((abs(den[e]), (B10[e]-NT[e])/den[e], e, den[e]))
rows.sort()
for ad, R, e, d in rows:
    print(f"    |den|={ad:.5f} den={d:+.5f}  R={100*R:9.1f}%   {e[:44]}")

print("\n  --- robustness: R(E) computed as trimmed mean / median vs 'block ratio' ---")
for c,E in ECELLS:
    B = Ecache[c]
    per = sorted((B[e]-NT[e])/den[e] for e in kept if abs(den[e])>1e-12)
    k = 2
    trim = per[k:-k] if len(per) > 2*k else per
    print(f"    {c:6s} med={100*med(per):7.1f}%  trimmed{k}mean={100*sum(trim)/len(trim):7.1f}%  mean={100*sum(per)/len(per):9.1f}%")

# ---------------- ATTACK 5: Phase-1 invariance ----------------
print("\n" + "="*78)
print("ATTACK 5 — Phase-1 invariance across all 14 cells")
print("="*78)
bad = defaultdict(list)
FIELDS = ['init_nrmse','init_rmse','auto_selected','auto_n_selected','N','ensemble_k']
for e in eps:
    for s in seeds:
        ref = by[('AUTO',e,s)]
        for c in cells:
            r = by[(c,e,s)]
            for f in FIELDS:
                a,b = ref[f], r[f]
                if isinstance(a,float) and isinstance(b,float):
                    same = (a==b)
                else:
                    same = (a==b)
                if not same:
                    bad[(c,f)].append((e,s,a,b))
if not bad:
    print("  PASS: all 14 cells identical to AUTO on", FIELDS, "for all 48 (ep,seed)")
else:
    for (c,f),v in sorted(bad.items()):
        print(f"  FAIL {c:14s} {f:16s} {len(v)} mismatches, e.g. {v[0]}")

# AUTO-DUP bit-identity
print("\n  AUTO-DUP bit-identity vs AUTO (all scalar fields):")
SCAL = ['nRMSE','Pearson_rho','init_nrmse','absVR_framewise','m_mean','decorr_mean','shrink_mean',
        'Variance_Ratio','Skill_Score','Directional_R2','Hs_error','avg_RMSE_unobserved','N','auto_n_selected','ensemble_k']
mism = 0; tot = 0
for e in eps:
    for s in seeds:
        a = by[('AUTO',e,s)]; b = by[('AUTO-DUP',e,s)]
        for f in SCAL:
            tot += 1
            if a[f] != b[f]: mism += 1
        if a['_w'] != b['_w'] or a['_kx'] != b['_kx']: mism += 1
        tot += 1
print(f"    {tot-mism}/{tot} fields bit-identical, mismatches={mism}")

# ---------------- ATTACK 6: seed variation vs headline ----------------
print("\n" + "="*78)
print("ATTACK 6 — seed variation vs headline size")
print("="*78)
for c in ['AUTO','W-MATCH','PHI-5e3','AUTO-NOTRACK','E-10']:
    spreads = []
    for e in eps:
        v = [by[(c,e,s)]['nRMSE'] for s in seeds]
        spreads.append(max(v)-min(v))
    print(f"  {c:14s} per-ep seed range: med={med(spreads):.5f} max={max(spreads):.5f}")

print("\n  single-seed block levels (no median-of-3):")
for c in ['AUTO','W-MATCH','PHI-5e3','AUTO-NOTRACK']:
    row = []
    for s in seeds:
        v = med([by[(c,e,s)]['nRMSE'] for e in kept])
        row.append(f"{s}:{v:.4f}")
    print(f"    {c:14s} " + "  ".join(row) + f"   med3-block={lv[c]:.4f}")

print("\n  PRIMARY-A recomputed per single seed (kept episodes):")
for s in seeds:
    ds = [by[('W-MATCH',e,s)]['nRMSE'] - by[('PHI-5e3',e,s)]['nRMSE'] for e in kept]
    n,nneg,npos,p = signtest(ds)
    print(f"    seed {s}: medD={med(ds):+.5f} n-/n+={nneg}/{npos} p={p:.4f}")

print("\n  leave-one-seed-out (median-of-2):")
for drop in seeds:
    ss = [x for x in seeds if x != drop]
    dsw = []
    for e in kept:
        a = med([by[('W-MATCH',e,s)]['nRMSE'] for s in ss])
        b = med([by[('PHI-5e3',e,s)]['nRMSE'] for s in ss])
        dsw.append(a-b)
    n,nneg,npos,p = signtest(dsw)
    print(f"    drop {drop}: PRIMARY-A medD={med(dsw):+.5f} {nneg}/{npos} p={p:.4f}")

json.dump({'note':'see stdout'}, open(os.path.join(HERE,'audit_con_done.json'),'w'))
print("\nDONE")
