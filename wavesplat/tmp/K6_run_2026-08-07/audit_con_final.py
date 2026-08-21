#!/usr/bin/env python3
"""CON audit part 5: clean OLD-vs-NEW AUTO comparison (Rx-4 batch only, same
median-of-3 protocol), step-budget check, and what the ORACLE cell is actually
handed (parameter accounting)."""
import json, os, math, glob
from collections import defaultdict
import numpy as np, h5py

REPO='/home/vexit/wavesplat/wavesplat'
def med(v):
    v=sorted(v); n=len(v); return v[n//2] if n%2 else .5*(v[n//2-1]+v[n//2])
CROWDED16=[f'episode_BF{bf}_{nc}comp_seed{es}_uniform.h5' for bf in (4,5) for nc in (15,30) for es in (0,1,2,3)]

print("="*92); print("(k) CLEAN old-vs-new AUTO: Rx-4 batch (seeds 1524-1526, median-of-3, same cell)"); print("="*92)
rx4=[json.loads(l) for l in open(os.path.join(REPO,'tmp/Rx4_tracker_init_2026-08-06/results.jsonl')) if l.strip()]
k6=[json.loads(l) for l in open(os.path.join(REPO,'tmp/K6_run_2026-08-07/results.jsonl')) if l.strip()]
on=json.load(open(os.path.join(REPO,'tmp/Rx4_tracker_init_2026-08-06/nulls_full.json')))
onm={x['episode']:x for x in on}
nn={x['episode']:x for x in json.load(open(os.path.join(REPO,'tmp/K6_run_2026-08-07/nulls_k6.json')))}
O=defaultdict(list); N=defaultdict(list)
sd_o=set(); sd_n=set()
for r in rx4:
    if r.get('cell')=='AUTO' and r['episode'] in CROWDED16: O[r['episode']].append(r['nRMSE']); sd_o.add(r['seed'])
for r in k6:
    if r['cell']=='AUTO': N[r['episode']].append(r['nRMSE']); sd_n.add(r['seed'])
print(f"  old seeds={sorted(sd_o)}  new seeds={sorted(sd_n)}   eps old={len(O)} new={len(N)}")
om=[med(O[e]) for e in CROWDED16]; nm=[med(N[e.replace('.h5','_ra_rf_ds.h5')]) for e in CROWDED16]
omar=[med(O[e])-onm[e]['zero_null_tracking_nRMSE_mean'] for e in CROWDED16]
nmar=[med(N[e.replace('.h5','_ra_rf_ds.h5')])-nn[e.replace('.h5','_ra_rf_ds.h5')]['zero_null_tracking_nRMSE_mean'] for e in CROWDED16]
print(f"  AUTO median nRMSE        old={med(om):.4f}   new={med(nm):.4f}   ({'NEW EASIER' if med(nm)<med(om) else 'NEW HARDER'})")
print(f"  flat-sea null median     old={med([onm[e]['zero_null_tracking_nRMSE_mean'] for e in CROWDED16]):.4f}"
      f"   new={med([nn[e.replace('.h5','_ra_rf_ds.h5')]['zero_null_tracking_nRMSE_mean'] for e in CROWDED16]):.4f}")
print(f"  AUTO margin over null    old={med(omar):+.4f}   new={med(nmar):+.4f}"
      f"   -> headroom {'GREW' if med(nmar)<med(omar) else 'SHRANK'} by {abs(med(nmar)-med(omar)):.4f}")
print(f"  AUTO beats null          old={sum(1 for x in omar if x<0)}/16   new={sum(1 for x in nmar if x<0)}/16")
print(f"  AUTO across-episode IQR  old={np.percentile(om,75)-np.percentile(om,25):.4f}"
      f"   new={np.percentile(nm,75)-np.percentile(nm,25):.4f}")

print("\n"+"="*92); print("(l) STEP BUDGET: is K6-NSCHED given FEWER gradient steps than AUTO?"); print("="*92)
print("  AUTO      : max_iters_per_stage=[200,200,500] = 900 MAX, loss_thresholds=[0.1,0.05,None]")
print("              -> stages 1 and 2 EARLY-STOP, so AUTO uses <= 900 steps.")
print("  K6-NSCHED : max_iters_per_stage=[900]         = 900, loss_thresholds=[None]")
print("              -> no early stop, so K6-NSCHED uses EXACTLY 900 steps.")
print("  => K6-NSCHED gets AT LEAST as many gradient steps as AUTO.  The 'fewer steps'")
print("     version of the under-convergence attack is REFUTED by the config itself.")
print("  What differs is the CURRICULUM: AUTO fits 3 atoms, then 8, then 15;")
print("  K6-NSCHED releases all N_gt (15 or 30) atoms at step 0 with blind placement.")

print("\n"+"="*92); print("(m) WHAT THE ORACLE CELL IS ACTUALLY HANDED (per-component DOF accounting)"); print("="*92)
print("  spectral_gaussian.py:63-64  omega_free is a Parameter ONLY when use_dispersion=False.")
print("  spectral_gaussian.py:99-101 use_dispersion=True -> omega = dispersion_omega(|k|, depth)")
print("  dispersion.py:33            omega = sqrt(g*|k|*tanh(|k|*d))   -- a PURE FUNCTION of |k|.")
print("  BASE_CFG has use_dispersion=True for every K6 cell.")
print("  => handing exact (kx,ky) hands exact omega TOO.")
print()
print("  per-component DOF = (A, kx, ky, phi) with omega := f(|k|).  ORACLE receives:")
print("    N   (cardinality)        : EXACT  (n_gaussians=[N_gt], read from config/wave_components)")
print("    kx  : EXACT (initialize_oracle:616-628, from ORACLE_COMPONENTS)  AND FROZEN (init_k_lr_scale=0.0)")
print("    ky  : EXACT, FROZEN                (tracking_mode='w_phi' freezes kx/ky in Phase 2 too)")
print("    omega: EXACT, FROZEN               (derived from the frozen |k|)")
print("    A   : NOT given (w = 0.01 uniform, initializer.py:635)")
print("    phi : NOT given (torch.rand*2pi,   initializer.py:636)")
print("    ordering: atoms are emitted sorted by TRUE amplitude descending"
      " (initialize_oracle:623 argsort(A)[::-1])")
print("  => 3 of 4 per-component DOF plus the cardinality are ground truth and never move.")
print("     ORACLE is fitting only 2N numbers (A, phi) against a field whose k-support is exact.")

# how much of the field is determined by k alone?  compare ORACLE init_nrmse to AUTO's
k6M=defaultdict(dict); B=defaultdict(list)
for r in k6: B[(r['cell'],r['episode'])].append(r)
for (c,e),rs in B.items():
    k6M[c][e]={'nRMSE':med([x['nRMSE'] for x in rs]),'init':med([x['init_nrmse'] for x in rs]),
               'det':med([x['spec'].get('detection_rate',float('nan')) for x in rs]),
               'dir':med([x['spec'].get('direction_error_deg_mean',float('nan')) for x in rs]),
               'kmag':med([x['spec'].get('k_mag_rel_error_mean',float('nan')) for x in rs])}
eps=sorted(k6M['AUTO'])
print("\n  -- how far does the optimiser drag PERFECT k?  (ORACLE-K, k trainable) --")
for c in ['ORACLE','ORACLE-K','AUTO']:
    print(f"     {c:9s} final det={med([k6M[c][e]['det'] for e in eps]):.3f}"
          f"  dir_err={med([k6M[c][e]['dir'] for e in eps]):6.2f} deg"
          f"  |k| rel err={med([k6M[c][e]['kmag'] for e in eps]):.3f}")

print("\n"+"="*92); print("(n) IS 'freeze perfect k helps' EMPIRICALLY SURPRISING?  transfer test"); print("="*92)
def sgn(ds):
    d=[x for x in ds if x!=0]; n=len(d); w=sum(1 for x in d if x<0); k=max(w,n-w)
    return n,w,min(1.,2*sum(math.comb(n,i) for i in range(k,n+1))/2**n),med(ds)
KEEP=[e for e in eps if k6M['AUTO'][e]['nRMSE']<nn[e]['zero_null_tracking_nRMSE_mean']]
for a,b,lab in [('ORACLE','ORACLE-K','freeze k when k is EXACT   (oracle arm)'),
                ('K6-KLR00','AUTO','freeze k when k is ESTIMATED (shipped arm)'),
                ('K6-KLR01','AUTO','k lr x0.1 on shipped arm'),
                ('K6-KLR03','AUTO','k lr x0.3 on shipped arm')]:
    n,w,p,m=sgn([k6M[a][e]['nRMSE']-k6M[b][e]['nRMSE'] for e in KEEP])
    print(f"  {lab:42s} med={m:+.4f}  {w}/{n}  p={p:.4g}")
print("  => the k-freeze lever REVERSES SIGN when the placement is not ground truth.")
