#!/usr/bin/env python3
"""CON audit part 2: did the NEW dataset change the difficulty?
Compares data_indep2_2026-08-07 (ra+rf+ds) against data_finemesh_2026-07-23 (the
CROWDED16 set every previous crowded number was measured on) on:
  * true-component nearest-neighbour |k| spacing distribution  (crowding)
  * amplitude spectrum shape (flat vs peaked): Gini / participation ratio / top1 share
  * Hs of the truth field, per-episode field variance
  * AUTO's ABSOLUTE nRMSE, old set vs new set, and its margin over the flat-sea null
Read-only.  Nothing is written outside this directory."""
import json, os, math, glob
from collections import defaultdict
import numpy as np
import h5py

REPO = '/home/vexit/wavesplat/wavesplat'
NEW = os.path.join(REPO, 'data_indep2_2026-08-07')
OLD = os.path.join(REPO, 'data_finemesh_2026-07-23')

CROWDED16 = [f'episode_BF{bf}_{nc}comp_seed{es}_uniform.h5'
             for bf in (4, 5) for nc in (15, 30) for es in (0, 1, 2, 3)]

def med(v):
    v = sorted(v); n = len(v)
    return v[n//2] if n % 2 else 0.5*(v[n//2-1]+v[n//2])

def comps(path):
    with h5py.File(path, 'r') as f:
        c = np.asarray(f['config/wave_components'])
        gt = np.asarray(f['frames/gt_wave_field']) if 'frames/gt_wave_field' in f else None
    return c, gt

def describe(path):
    c, gt = comps(path)
    A, kx, ky = c[:, 0], c[:, 2], c[:, 3]
    K = np.stack([kx, ky], 1)
    D = np.linalg.norm(K[:, None, :] - K[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    nn = D.min(1)                                  # nearest-neighbour spacing in k-plane
    P = A**2
    P = P / P.sum()
    gini = float(np.abs(np.subtract.outer(P, P)).sum() / (2*len(P)**2*P.mean()))
    part = float(1.0/np.sum(P**2))                 # participation ratio (effective #comps)
    return dict(N=len(A),
                nn_min=float(nn.min()), nn_med=float(np.median(nn)),
                nn_mean=float(nn.mean()),
                A_max_share=float(P.max()),
                gini=gini, part_ratio=part, part_frac=part/len(A),
                kmag_med=float(np.linalg.norm(K, axis=1).mean()),
                field_std=float(np.std(gt)) if gt is not None else float('nan'))

print("="*100)
print("1. TRUE-COMPONENT geometry / spectrum shape:  OLD (finemesh 07-23) vs NEW (indep2)")
print("="*100)
hdr = f"{'episode':28s} {'set':4s} {'N':>3s} {'nn_min':>8s} {'nn_med':>8s} {'top1%':>7s} {'gini':>6s} {'PR/N':>6s} {'fieldSD':>8s}"
print(hdr)
rows = defaultdict(dict)
for e in CROWDED16:
    o = describe(os.path.join(OLD, e))
    n = describe(os.path.join(NEW, e.replace('.h5', '_ra_rf_ds.h5')))
    rows[e]['old'], rows[e]['new'] = o, n
    for tag, d in (('old', o), ('new', n)):
        print(f"{e[8:36]:28s} {tag:4s} {d['N']:3d} {d['nn_min']:8.4f} {d['nn_med']:8.4f}"
              f" {100*d['A_max_share']:7.2f} {d['gini']:6.3f} {d['part_frac']:6.3f} {d['field_std']:8.4f}")

print("\n-- BLOCK MEDIANS --")
for key in ['nn_min','nn_med','nn_mean','A_max_share','gini','part_frac','kmag_med','field_std']:
    o = med([rows[e]['old'][key] for e in CROWDED16])
    n = med([rows[e]['new'][key] for e in CROWDED16])
    wins = sum(1 for e in CROWDED16 if rows[e]['new'][key] < rows[e]['old'][key])
    print(f"  {key:14s} old={o:9.4f}  new={n:9.4f}  ratio={n/o if o else float('nan'):6.3f}"
          f"   new<old in {wins}/16")

# pooled nearest-neighbour spacing over all true components in the block
def pooled_nn(root, names):
    out = []
    for e in names:
        c, _ = comps(os.path.join(root, e))
        K = c[:, 2:4]
        D = np.linalg.norm(K[:, None, :]-K[None, :, :], axis=-1); np.fill_diagonal(D, np.inf)
        out += D.min(1).tolist()
    return np.array(out)
po = pooled_nn(OLD, CROWDED16)
pn = pooled_nn(NEW, [e.replace('.h5','_ra_rf_ds.h5') for e in CROWDED16])
print(f"\n-- POOLED nearest-neighbour |dk| over all true components ({len(po)} old / {len(pn)} new)")
for q in (5, 10, 25, 50, 75, 90):
    print(f"   p{q:02d}   old={np.percentile(po,q):.4f}   new={np.percentile(pn,q):.4f}")
print(f"   frac below min_dk=0.15 :  old={np.mean(po<0.15):.3f}   new={np.mean(pn<0.15):.3f}")
print(f"   frac below dedup 0.03  :  old={np.mean(po<0.03):.3f}   new={np.mean(pn<0.03):.3f}")

# ----------------------------------------------------------------------------
print("\n" + "="*100)
print("2. AUTO's ABSOLUTE performance: new independent set vs the historical crowded16")
print("="*100)
def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]
K6 = load(os.path.join(REPO, 'tmp/K6_run_2026-08-07/results.jsonl'))
nulls_new = {n['episode']: n for n in json.load(open(os.path.join(REPO,'tmp/K6_run_2026-08-07/nulls_k6.json')))}

hist = {}
for p in glob.glob(os.path.join(REPO, 'tmp/*/results.jsonl')):
    for r in load(p):
        if r.get('cell') == 'AUTO' and r.get('episode') in CROWDED16:
            hist.setdefault(r['episode'], {}).setdefault(r['seed'], r['nRMSE'])
print(f"historical AUTO runs found on CROWDED16: {sum(len(v) for v in hist.values())}"
      f" runs over {len(hist)} episodes, seeds={sorted({s for v in hist.values() for s in v})}")

newA = defaultdict(list)
for r in K6:
    if r['cell'] == 'AUTO':
        newA[r['episode']].append(r['nRMSE'])

print(f"\n{'episode':30s} {'OLD AUTO':>10s} {'NEW AUTO':>10s} {'NEW null':>10s} {'NEW margin':>11s}")
oldv, newv, marg = [], [], []
for e in CROWDED16:
    ne = e.replace('.h5', '_ra_rf_ds.h5')
    o = med(list(hist.get(e, {}).values())) if hist.get(e) else float('nan')
    n = med(newA[ne])
    z = nulls_new[ne]['zero_null_tracking_nRMSE_mean']
    oldv.append(o); newv.append(n); marg.append(n - z)
    print(f"{e[8:38]:30s} {o:10.4f} {n:10.4f} {z:10.4f} {n-z:+11.4f}")
print(f"{'MEDIAN':30s} {med([x for x in oldv if x==x]):10.4f} {med(newv):10.4f}"
      f" {med([nulls_new[e.replace('.h5','_ra_rf_ds.h5')]['zero_null_tracking_nRMSE_mean'] for e in CROWDED16]):10.4f}"
      f" {med(marg):+11.4f}")

# how many of the OLD episodes did AUTO beat its own old null?  need old nulls
oldnullp = None
for cand in glob.glob(os.path.join(REPO, 'tmp/*/nulls*.json')):
    d = json.load(open(cand))
    if isinstance(d, list) and any(x.get('episode') in CROWDED16 for x in d):
        oldnullp = cand; break
if oldnullp:
    on = {x['episode']: x for x in json.load(open(oldnullp)) if x['episode'] in CROWDED16}
    print(f"\nold nulls from {os.path.relpath(oldnullp, REPO)} ({len(on)} eps)")
    w = [e for e in CROWDED16 if e in on and med(list(hist[e].values())) <
         on[e]['zero_null_tracking_nRMSE_mean']]
    print(f"  OLD: AUTO beats old flat null on {len(w)}/{len(on)}")
    om = [med(list(hist[e].values())) - on[e]['zero_null_tracking_nRMSE_mean']
          for e in CROWDED16 if e in on]
    print(f"  OLD median margin over null = {med(om):+.4f}"
          f"   |   NEW median margin = {med(marg):+.4f}")
