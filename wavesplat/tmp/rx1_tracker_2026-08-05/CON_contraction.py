"""CON probe B — the tracker as a 1-parameter contraction, and what that implies
for the deployed operating point.

Fit  tracked_nRMSE = a + b * init_nrmse  over every tracker-ON cell we have, then
   harm(init) = tracked(init) - notrack(init),  notrack(init) ~= init (measured).
The 61% headline is then a point evaluation of that line at an init no deployable
configuration has ever reached."""
import json, os
import numpy as np
from collections import defaultdict

S12 = 'tmp/s1s2_2026-08-05/results.jsonl'
E1  = 'tmp/e1_grid_2026-08-05/results.jsonl'
rows = [json.loads(l) for l in open(S12)]
e1rows = [json.loads(l) for l in open(E1)] if os.path.exists(E1) else []

def med_cells(rs, keys=('arm','episode')):
    c = defaultdict(list)
    for r in rs:
        if 'error' in r: continue
        c[tuple(r[k] for k in keys)].append(r)
    out = {}
    for k, v in c.items():
        out[k] = {m: float(np.median([x[m] for x in v]))
                  for m in ('nRMSE','init_nrmse','Variance_Ratio','Pearson_rho','Hs_error')
                  if m in v[0]}
        out[k]['n'] = len(v)
    return out

s12 = med_cells(rows)
TRACK_ON = ['S1-A0','S1-A1','S1-A2','S1-A3','S2-C1','S2-T1','S2-T1f','S2-T2','S2-T4']
NOTRACK  = {'S2-T1f':'X-T1f-notrack','S1-A0':'X-A0-notrack'}

print('='*96)
print('B1 — contraction fit  tracked = a + b*init   (tracker-ON cells only)')
print('='*96)
X = np.array([v['init_nrmse'] for k,v in s12.items() if k[0] in TRACK_ON])
Y = np.array([v['nRMSE']      for k,v in s12.items() if k[0] in TRACK_ON])
b, a = np.polyfit(X, Y, 1)
r = np.corrcoef(X, Y)[0,1]
print(f'  n cells = {len(X)} (9 tracker-ON arms x 4 crowded episodes, median-of-3)')
print(f'  tracked = {a:+.4f} {b:+.4f} * init      r = {r:+.4f}   R^2 = {r**2:.3f}')
print(f'  contraction factor b = {b:.4f}  (<1 => the tracker pulls every init toward a fixed point)')
print(f'  fixed point init* = a/(1-b) = {a/(1-b):.4f}')

print()
print('  harm(init) = tracked(init) - init = a - (1-b)*init   [notrack ~= init, verified below]')
for lab, init in [('ORACLE (S2-T1f)', 0.0629), ('deployed shipped (S1-A0)', 0.2612),
                  ('best deployable init measured (S1-A3)', 0.2438),
                  ('hypothetical -20% init', 0.2612*0.8),
                  ('hypothetical -40% init', 0.2612*0.6),
                  ('hypothetical -60% init', 0.2612*0.4)]:
    print(f'    init={init:.4f}  {lab:38s} predicted tracker harm = {a-(1-b)*init:+.4f}')
print(f'  break-even init (harm = 0) = {a/(1-b):.4f}')

print()
print('='*96)
print('B2 — is notrack really == init?  (the denominator of the 61%)')
print('='*96)
for on, off in NOTRACK.items():
    for k in sorted(s12):
        if k[0] != off: continue
        ep = k[1]
        print(f'  {off:16s} {ep[:26]:28s} notrack_nRMSE={s12[(off,ep)]["nRMSE"]:.4f} '
              f'init_nrmse={s12[(off,ep)]["init_nrmse"]:.4f} '
              f'delta={s12[(off,ep)]["nRMSE"]-s12[(off,ep)]["init_nrmse"]:+.4f}')

print()
print('='*96)
print('B3 — HOW FAR IS ORACLE PLACEMENT FROM ANY DEPLOYABLE CONFIGURATION?')
print('='*96)
dep = [a_ for a_ in TRACK_ON if a_.startswith('S1') or a_ == 'S2-C1']
best_dep = min((np.mean([s12[(a_,e)]['init_nrmse'] for e in sorted(set(k[1] for k in s12))]), a_)
               for a_ in dep)
orc = np.mean([s12[('S2-T1f',e)]['init_nrmse'] for e in sorted(set(k[1] for k in s12))])
print(f'  best DEPLOYABLE init_nrmse over all measured arms: {best_dep[0]:.4f}  ({best_dep[1]})')
print(f'  ORACLE init_nrmse                                : {orc:.4f}')
print(f'  gap = {best_dep[0]-orc:.4f}  = oracle is {best_dep[0]/orc:.2f}x better than the best deployable init')
if e1rows:
    e1c = med_cells(e1rows, keys=('cell','episode'))
    cells = sorted(set(k[0] for k in e1c))
    eps4 = ['episode_BF4_15comp_seed0_uniform.h5','episode_BF4_30comp_seed0_uniform.h5',
            'episode_BF5_15comp_seed0_uniform.h5','episode_BF5_30comp_seed0_uniform.h5']
    print(f'\n  E1 grid cells (placement interventions), same 4 crowded episodes:')
    for c in cells:
        vs = [e1c[(c,e)]['init_nrmse'] for e in eps4 if (c,e) in e1c]
        vt = [e1c[(c,e)]['nRMSE'] for e in eps4 if (c,e) in e1c]
        if vs:
            print(f'    cell {str(c):26s} init={np.mean(vs):.4f}  tracked={np.mean(vt):.4f}  '
                  f'closes {(best_dep[0]-np.mean(vs))/(best_dep[0]-orc)*100:+6.1f}% of the gap to oracle')

print()
print('='*96)
print('B4 — SHRINKAGE vs DECORRELATION: what an amplitude fix can possibly buy')
print('='*96)
print('  nRMSE^2 = 1 - 2*rho*sqrt(v) + v  with v = VR.  Optimal rescale alpha* = rho*sqrt(v)/v')
print('  -> best achievable nRMSE with amplitude ALONE fixed = sqrt(1-rho^2)')
eps4 = sorted(set(k[1] for k in s12))
for arm in ['S2-T1f','X-T1f-notrack','S1-A0','X-A0-notrack']:
    rho = np.mean([s12[(arm,e)]['Pearson_rho'] for e in eps4])
    vr  = np.mean([s12[(arm,e)]['Variance_Ratio'] for e in eps4])
    nr  = np.mean([s12[(arm,e)]['nRMSE'] for e in eps4])
    floor = np.sqrt(max(0.0,1-rho**2))
    print(f'  {arm:16s} rho={rho:.4f} VR={vr:.4f} nRMSE={nr:.4f} '
          f'| perfect-rescale floor sqrt(1-rho^2)={floor:.4f} (in Hs units x std_gt/Hs)')
# same in the arm-consistent normalisation: report ratio to notrack counterpart
print('\n  ratio of decorrelation floors (oracle):  '
      f'{np.sqrt(1-np.mean([s12[("S2-T1f",e)]["Pearson_rho"] for e in eps4])**2):.4f} (ON) vs '
      f'{np.sqrt(1-np.mean([s12[("X-T1f-notrack",e)]["Pearson_rho"] for e in eps4])**2):.4f} (OFF)')
print('  => the tracker moves rho, not only scale. An amplitude-only knob cannot cross the ON floor.')
