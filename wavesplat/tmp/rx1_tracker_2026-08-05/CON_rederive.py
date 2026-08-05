"""CON re-derivation of the Rx-1 premise numbers from raw s1s2 results.jsonl.
Pre-specified per PREREG §1: median-of-3 per (arm, episode), paired episode deltas,
median of paired deltas, two-sided sign test, |VR-1| transform."""
import json, itertools, math
import numpy as np
from collections import defaultdict

P = 'tmp/s1s2_2026-08-05/results.jsonl'
rows = [json.loads(l) for l in open(P)]

METRICS = ['nRMSE','init_nrmse','Pearson_rho','Skill_Score','Hs_error',
           'avg_RMSE_unobserved','Variance_Ratio','N']
SPEC = ['detection_rate','k_mag_rel_error_mean','amp_rel_error_mean',
        'direction_error_deg_mean','n_est']

def val(r, m):
    if m in r: return float(r[m])
    s = r.get('spec') or {}
    return float(s[m]) if m in s else float('nan')

cell = defaultdict(list)   # (arm, ep) -> list of rows
for r in rows:
    cell[(r['arm'], r['episode'])].append(r)

arms = sorted(set(r['arm'] for r in rows))
eps  = sorted(set(r['episode'] for r in rows))

def med(arm, ep, m):
    rs = cell[(arm,ep)]
    return float(np.median([val(r,m) for r in rs]))

def signtest(k, n):
    # two-sided exact binomial p, H0 p=0.5
    from math import comb
    kk = min(k, n-k)
    p = sum(comb(n,i) for i in range(0, kk+1)) / 2**n * 2
    return min(1.0, p)

print('='*100)
print('SECTION 1 — cell counts and seed ledger')
print('='*100)
for a in arms:
    ns = [len(cell[(a,e)]) for e in eps]
    seeds = sorted(set(r['seed'] for e in eps for r in cell[(a,e)]))
    print(f'{a:16s} n_per_ep={ns} seeds={seeds}')

print()
print('='*100)
print('SECTION 2 — arm medians (median over 3 seeds, then MEAN over the 4 episodes) '
      'vs (median over episodes)')
print('='*100)
hdr = f"{'arm':16s}" + ''.join(f'{m[:11]:>13s}' for m in ['nRMSE','init_nrmse','Pearson_rho','Variance_Ratio','Hs_error','N'])
print(hdr)
for a in arms:
    v = [np.mean([med(a,e,m) for e in eps]) for m in ['nRMSE','init_nrmse','Pearson_rho','Variance_Ratio','Hs_error','N']]
    print(f'{a:16s}' + ''.join(f'{x:13.4f}' for x in v))

print()
print('='*100)
print('SECTION 3 — IS S2-T1f BIT-IDENTICAL TO S2-T4?  (config + per-run values)')
print('='*100)
for e in eps:
    for s in sorted(set(r['seed'] for r in cell[('S2-T1f',e)])):
        a1 = [r for r in cell[('S2-T1f',e)] if r['seed']==s][0]
        a2 = [r for r in cell[('S2-T4',e)] if r['seed']==s][0]
        same = all(abs(val(a1,m)-val(a2,m))<1e-12 for m in METRICS)
        print(f'  {e[:28]:30s} seed={s:5d}  all-metrics-identical={same}   '
              f'nRMSE {val(a1,"nRMSE"):.6f} vs {val(a2,"nRMSE"):.6f}')
        if s == sorted(set(r["seed"] for r in cell[("S2-T1f",e)]))[0]:
            c1, c2 = a1.get('cfg',{}), a2.get('cfg',{})
            d = {k:(c1.get(k),c2.get(k)) for k in set(c1)|set(c2) if c1.get(k)!=c2.get(k)}
            print(f'      cfg diff T1f vs T4: {d}')

print()
print('='*100)
print('SECTION 4 — TRACKER CONTRAST AT TWO PLACEMENTS (paired, median-of-3)')
print('='*100)
def contrast(a_on, a_off, label):
    print(f'\n--- {label}:  {a_off} (tracker OFF) minus {a_on} (tracker ON) ---')
    for m in ['nRMSE','Pearson_rho','Variance_Ratio','Hs_error','avg_RMSE_unobserved','init_nrmse','N']:
        d = []
        for e in eps:
            d.append(med(a_off,e,m) - med(a_on,e,m))
        d = np.array(d)
        k = int((d>0).sum()); n=len(d)
        print(f'  {m:22s} medΔ={np.median(d):+8.4f}  per-ep={[round(x,4) for x in d]}  '
              f'k/n={k}/{n} p={signtest(k,n):.4f}')
    # |VR-1|
    d = np.array([abs(med(a_off,e,'Variance_Ratio')-1)-abs(med(a_on,e,'Variance_Ratio')-1) for e in eps])
    k=int((d>0).sum())
    print(f'  {"|VR-1|":22s} medΔ={np.median(d):+8.4f}  per-ep={[round(x,4) for x in d]}  k/n={k}/4 p={signtest(k,4):.4f}')

contrast('S2-T1f','X-T1f-notrack','ORACLE placement')
contrast('S1-A0','X-A0-notrack','DEPLOYED placement (shipped v2_konly)')

print()
print('='*100)
print('SECTION 5 — the "61%" and the 2x2, re-derived under 3 aggregation rules')
print('='*100)
for rule, agg in [('mean-over-eps of per-ep medians', np.mean),
                  ('median-over-eps of per-ep medians', np.median)]:
    on_o  = agg([med('S2-T1f',e,'nRMSE') for e in eps])
    off_o = agg([med('X-T1f-notrack',e,'nRMSE') for e in eps])
    on_d  = agg([med('S1-A0',e,'nRMSE') for e in eps])
    off_d = agg([med('X-A0-notrack',e,'nRMSE') for e in eps])
    print(f'\n[{rule}]')
    print(f'   oracle   ON={on_o:.4f} OFF={off_o:.4f}  tracker term=+{on_o-off_o:.4f}  '
          f'share of ON error = {(on_o-off_o)/on_o*100:.1f}%')
    print(f'   deployed ON={on_d:.4f} OFF={off_d:.4f}  tracker term={on_d-off_d:+.4f}  '
          f'share of ON error = {(on_d-off_d)/on_d*100:.1f}%')
    print(f'   ratio |oracle term| / |deployed term| = {abs(on_o-off_o)/abs(on_d-off_d):.2f}x')
# PREREG's own primary statistic: median of PAIRED deltas
d_o = np.array([med('S2-T1f',e,'nRMSE')-med('X-T1f-notrack',e,'nRMSE') for e in eps])
d_d = np.array([med('S1-A0',e,'nRMSE')-med('X-A0-notrack',e,'nRMSE') for e in eps])
print(f'\n[PREREG PRIMARY = median of paired deltas]')
print(f'   oracle   medΔ(ON-OFF)={np.median(d_o):+.4f}  worse-with-tracker {int((d_o>0).sum())}/4  p={signtest(int((d_o>0).sum()),4):.4f}')
print(f'   deployed medΔ(ON-OFF)={np.median(d_d):+.4f}  worse-with-tracker {int((d_d>0).sum())}/4  p={signtest(int((d_d>0).sum()),4):.4f}')
print(f'   per-ep oracle  : {[round(x,4) for x in d_o]}')
print(f'   per-ep deployed: {[round(x,4) for x in d_d]}')

print()
print('='*100)
print('SECTION 6 — MODEL ORDER CONFOUND in the 2x2')
print('='*100)
for a in ['S1-A0','X-A0-notrack','S2-T1f','X-T1f-notrack','S2-C1','S2-T1']:
    ns = [med(a,e,'N') for e in eps]
    print(f'  {a:16s} N per ep = {ns}  mean={np.mean(ns):.1f}')

print()
print('='*100)
print('SECTION 7 — seed noise floor (within-cell spread) vs the effects claimed')
print('='*100)
for a in ['S1-A0','X-A0-notrack','S2-T1f','X-T1f-notrack']:
    sp=[]
    for e in eps:
        v=[val(r,'nRMSE') for r in cell[(a,e)]]
        sp.append(max(v)-min(v))
    print(f'  {a:16s} within-cell nRMSE range per ep = {[round(x,4) for x in sp]}  median={np.median(sp):.4f}')
