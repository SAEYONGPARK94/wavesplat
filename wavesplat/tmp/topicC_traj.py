#!/usr/bin/env python3
"""Does the placement gain decay because of the TRACKER, or because of TIME-propagation?
Uses stored per-frame rho/v; no new runs. m_f = (1-rho^2)+(sqrt(v)-rho)^2, sqrt(m_f) ~ nRMSE_f
up to a per-episode-per-frame sigma_gt(f)/Hs scale which is COMMON to all cells -> cell
comparisons at fixed (episode, frame) are exact ratios.
"""
import json, statistics as st
import numpy as np

def load(fn):
    out = []
    for l in open(fn):
        l = l.strip()
        if not l: continue
        d = json.loads(l)
        if 'error' in d: continue
        out.append(d)
    return out

def mtraj(d):
    v = np.array(d['vr_frames'], float); r = np.array(d['rho_frames'], float)
    return np.sqrt(np.clip((1 - r**2) + (np.sqrt(np.clip(v, 0, None)) - r)**2, 0, None))

def cellmean(rows, cell, nf=None):
    """mean over (episode,seed) of sqrt(m_f) per frame index"""
    ts = [mtraj(d) for d in rows if d['cell'] == cell]
    L = min(len(t) for t in ts) if nf is None else nf
    A = np.stack([t[:L] for t in ts])
    return A.mean(0), A.shape

print('='*88)
print('A) rx1: same operating point (v2_konly), tracker ON vs OFF -- per-frame trajectory')
print('='*88)
r1 = load('tmp/rx1_tracker_2026-08-05/results2.jsonl')
for c in ['D-BASE','D-NOTRACK','X-BASE','X-NOTRACK','K4']:
    t, shp = cellmean(r1, c)
    idx = [0, 4, 9, 19, 39, 79, len(t)-1]
    idx = [i for i in idx if i < len(t)]
    print(f"  {c:10s} n_runs={shp[0]} n_frames={shp[1]}  sqrt(m) at frames " +
          ' '.join(f"f{i+1}={t[i]:.3f}" for i in idx) + f"  | mean={t.mean():.4f}")

print('\n  --> tracker-attributable growth (BASE minus NOTRACK), per frame:')
for pref, base, nt in [('deploy(v2_konly)','D-BASE','D-NOTRACK'), ('oracle','X-BASE','X-NOTRACK')]:
    a, _ = cellmean(r1, base); b, _ = cellmean(r1, nt)
    L = min(len(a), len(b)); d = a[:L] - b[:L]
    print(f"  {pref:18s} f1={d[0]:+.4f} f5={d[4]:+.4f} f10={d[9]:+.4f} f20={d[19]:+.4f} "
          f"f40={d[39]:+.4f} last={d[L-1]:+.4f}  mean={d.mean():+.4f}")

print('\n  --> does the FROZEN model degrade over time at all? (NOTRACK trajectory slope)')
for c in ['D-NOTRACK','X-NOTRACK']:
    t, _ = cellmean(r1, c)
    print(f"  {c:10s} f1={t[0]:.4f} -> f10={t[9]:.4f} -> f40={t[39]:.4f} -> last={t[-1]:.4f}"
          f"   (growth f1->last = {t[-1]-t[0]:+.4f}, {100*(t[-1]/t[0]-1):+.1f}%)")

print('\n' + '='*88)
print('B) rx2: how the PLACEMENT gain decays across frames (all arms keep tracker ON)')
print('='*88)
r2 = load('tmp/rx2_init_2026-08-05/results.jsonl')
base2, _ = cellmean(r2, 'X-BASE')
for c in ['P3','G1','G1P1','AUTO','ORACLE-K','ORACLE']:
    t, shp = cellmean(r2, c)
    L = min(len(t), len(base2)); d = t[:L] - base2[:L]
    g0 = d[0]
    print(f"  {c:9s} gain f1={d[0]:+.4f}  f5={d[4]:+.4f}  f10={d[9]:+.4f}  f20={d[19]:+.4f}  "
          f"f40={d[39]:+.4f}  last={d[L-1]:+.4f}  mean={d.mean():+.4f}  "
          f"retention(mean/f1)={100*d.mean()/g0:5.1f}%")

print('\n  --> absolute levels: does X-BASE itself get WORSE over frames (common denominator)?')
for c in ['X-BASE','G1P1','AUTO','ORACLE']:
    t, _ = cellmean(r2, c)
    print(f"  {c:9s} f1={t[0]:.4f} f10={t[9]:.4f} f40={t[39]:.4f} last={t[-1]:.4f}  "
          f"growth={100*(t[-1]/t[0]-1):+.1f}%")

print('\n' + '='*88)
print('C) CROSS-CHECK: is rx1 D-BASE the same operating point as rx2 X-BASE?')
print('='*88)
def med_of_ep(rows, cell, metric):
    EPS = sorted({d['episode'] for d in rows})
    per = {}
    for d in rows:
        if d['cell'] == cell: per.setdefault(d['episode'], []).append(d[metric])
    return st.median([st.median(per[e]) for e in EPS])
print(f"  rx1 D-BASE : seeds={sorted({d['seed'] for d in r1 if d['cell']=='D-BASE'})} "
      f"nRMSE={med_of_ep(r1,'D-BASE','nRMSE'):.4f} init={med_of_ep(r1,'D-BASE','init_nrmse'):.4f} "
      f"N={med_of_ep(r1,'D-BASE','N'):.1f}")
print(f"  rx2 X-BASE : seeds={sorted({d['seed'] for d in r2 if d['cell']=='X-BASE'})} "
      f"nRMSE={med_of_ep(r2,'X-BASE','nRMSE'):.4f} init={med_of_ep(r2,'X-BASE','init_nrmse'):.4f} "
      f"N={med_of_ep(r2,'X-BASE','N'):.1f}")
print('  cfg identical? rx1 D-BASE vs rx2 X-BASE:')
c1 = [d['cfg'] for d in r1 if d['cell']=='D-BASE'][0]
c2 = [d['cfg'] for d in r2 if d['cell']=='X-BASE'][0]
allk = sorted(set(c1)|set(c2))
diffs = [(k, c1.get(k,'<absent>'), c2.get(k,'<absent>')) for k in allk if c1.get(k,'<absent>') != c2.get(k,'<absent>')]
print('    diffs:', diffs if diffs else 'NONE (identical cfg dicts)')
