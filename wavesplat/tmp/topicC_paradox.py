#!/usr/bin/env python3
"""Why do rho and |VR-1| improve while nRMSE does not, in the deployment arm?"""
import json, statistics as st
import numpy as np
from math import comb

rows = [json.loads(l) for l in open('tmp/rx1_tracker_2026-08-05/results2.jsonl') if l.strip()]
rows = [r for r in rows if 'error' not in r]
EPS = sorted({r['episode'] for r in rows})

def sign_p(k, n):
    lo = min(k, n-k); return min(1.0, 2*sum(comb(n,i) for i in range(lo+1))/2**n)

def per_run(d):
    v = np.array(d['vr_frames'], float); r = np.array(d['rho_frames'], float)
    m = (1-r**2) + (np.sqrt(np.clip(v,0,None))-r)**2
    return dict(rho=r.mean(), absVR1=np.abs(v-1).mean(), m=m.mean(),
                sqrtm=np.sqrt(np.clip(m,0,None)).mean(), decorr=(1-r**2).mean(),
                shrink=((np.sqrt(np.clip(v,0,None))-r)**2).mean(),
                vr=v.mean(), nRMSE=d['nRMSE'])

agg = {}
for d in rows:
    agg.setdefault((d['cell'], d['episode']), []).append(per_run(d))

def med(cell, key):
    return st.median([st.median([x[key] for x in agg[(cell,e)]]) for e in EPS])

def pd_(cell, base, key):
    dl = [st.median([x[key] for x in agg[(cell,e)]]) - st.median([x[key] for x in agg[(base,e)]]) for e in EPS]
    k = sum(1 for x in dl if x < 0)
    return st.median(dl), k, len(dl), sign_p(k, len(dl))

print('CELL LEVELS (seed-median then ep-median)')
print(f"{'cell':11s} {'rho':>7s} {'|VR-1|':>8s} {'VR':>7s} {'m':>8s} {'sqrt(m)':>8s} {'decorr':>8s} {'shrink':>8s} {'nRMSE':>8s}")
for c in ['X-BASE','X-NOTRACK','K4','D-BASE','D-NOTRACK','D-K12','D-K6']:
    print(f"{c:11s} {med(c,'rho'):7.4f} {med(c,'absVR1'):8.4f} {med(c,'vr'):7.4f} {med(c,'m'):8.4f} "
          f"{med(c,'sqrtm'):8.4f} {med(c,'decorr'):8.4f} {med(c,'shrink'):8.4f} {med(c,'nRMSE'):8.4f}")

print('\nPAIRED DELTAS vs D-BASE (deployment arm) -- prereg convention, n=4, floor p=0.125')
print(f"{'cell':11s} " + ' '.join(f"{k:>18s}" for k in ['rho','absVR1','m','sqrtm','nRMSE']))
for c in ['D-NOTRACK','D-K12','D-K6']:
    cells = []
    for k in ['rho','absVR1','m','sqrtm','nRMSE']:
        v, kk, n, p = pd_(c, 'D-BASE', k)
        cells.append(f"{v:+9.4f}[{kk}/{n},{p:.2f}]")
    print(f"{c:11s} " + ' '.join(f"{s:>18s}" for s in cells))

print('\n*** JENSEN GAP TEST: mean(m) vs mean(sqrt(m)) ***')
print('  m is a SQUARED error; nRMSE is a root. Averaging m over frames overweights bad frames.')
for c in ['D-BASE','D-NOTRACK']:
    ms = [x['m'] for e in EPS for x in agg[(c,e)]]
    ss = [x['sqrtm'] for e in EPS for x in agg[(c,e)]]
    print(f"  {c:10s} mean(m)={st.mean(ms):.4f} -> sqrt = {st.mean(ms)**0.5:.4f} ;  mean(sqrt(m))={st.mean(ss):.4f}"
          f"  Jensen gap={st.mean(ms)**0.5-st.mean(ss):+.4f}")
d1, d2 = pd_('D-NOTRACK','D-BASE','m'), pd_('D-NOTRACK','D-BASE','sqrtm')
print(f"  DELTA m      = {d1[0]:+.4f} ({d1[1]}/{d1[2]}, p={d1[3]:.3f})")
print(f"  DELTA sqrt(m)= {d2[0]:+.4f} ({d2[1]}/{d2[2]}, p={d2[3]:.3f})   <-- this is the nRMSE-compatible one")

print('\n*** WHICH FRAMES CARRY THE m IMPROVEMENT? (tail-frame check) ***')
def traj(c, key):
    out = []
    for d in rows:
        if d['cell'] != c: continue
        v = np.array(d['vr_frames'], float); r = np.array(d['rho_frames'], float)
        m = (1-r**2)+(np.sqrt(np.clip(v,0,None))-r)**2
        out.append(m if key=='m' else np.sqrt(np.clip(m,0,None)))
    L = min(len(t) for t in out)
    return np.stack([t[:L] for t in out]).mean(0)
a, b = traj('D-BASE','m'), traj('D-NOTRACK','m')
sa, sb = traj('D-BASE','sqrtm'), traj('D-NOTRACK','sqrtm')
L = min(len(a), len(b))
q = L//4
for i in range(4):
    s, e = i*q, (i+1)*q
    print(f"  frames {s+1:3d}-{e:3d}: D(m)={b[s:e].mean()-a[s:e].mean():+.4f}   D(sqrt m)={sb[s:e].mean()-sa[s:e].mean():+.4f}"
          f"   levels m: BASE={a[s:e].mean():.3f} NOTRACK={b[s:e].mean():.3f}")
print(f"  worst 5%% of D-BASE frames carry m={np.sort(a)[-max(1,L//20):].mean():.3f} vs overall {a.mean():.3f}")
