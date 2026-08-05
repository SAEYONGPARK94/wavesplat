"""PRO agent -- summarise the single-cell diagnostic (PRO_diag_shrink.jsonl)."""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, 'PRO_diag_shrink.jsonl'))]
by = {}
for r in rows:
    by.setdefault(r['arm'], {})[r['seed']] = r

ORDER = ['D6-notrack', 'D0-default', 'D1p2-track_only_l1', 'D2p2-track_only',
         'D1-l1_0', 'D2-l1_smooth_0', 'D3-noconf', 'D4-steps1', 'D8-lr10x_less',
         'D5-window12', 'D7-noclamp',
         'E6-A0-notrack', 'E0-A0-default', 'E1p2-track_only_l1', 'E2p2-track_only',
         'E1-A0-l1_0', 'E2-A0-l1smooth_0']
WHAT = {
    'D6-notrack': 'tracker frozen (control, = X-T1f-notrack)',
    'D0-default': 'SHIPPED tracker (control, = S2-T1f)',
    'D1p2-track_only_l1': 'PHASE-2 ONLY: lambda_l1=0',
    'D2p2-track_only': 'PHASE-2 ONLY: lambda_l1=0, lambda_smooth=0',
    'D1-l1_0': 'both phases: lambda_l1=0',
    'D2-l1_smooth_0': 'both phases: lambda_l1=0, lambda_smooth=0',
    'D3-noconf': 'use_confidence=False',
    'D4-steps1': 'tracking_steps 10 -> 1',
    'D8-lr10x_less': 'tracking_lr 0.005 -> 0.0005',
    'D5-window12': 'tracking_window 3 -> 12',
    'D7-noclamp': 'clamp_steepness disabled',
    'E6-A0-notrack': 'tracker frozen (control, = X-A0-notrack)',
    'E0-A0-default': 'SHIPPED tracker (control, = S1-A0)',
    'E1p2-track_only_l1': 'PHASE-2 ONLY: lambda_l1=0',
    'E2p2-track_only': 'PHASE-2 ONLY: lambda_l1=0, lambda_smooth=0',
    'E1-A0-l1_0': 'both phases: lambda_l1=0',
    'E2-A0-l1smooth_0': 'both phases: lambda_l1=0, lambda_smooth=0',
}

print('=' * 108)
print('SINGLE-CELL DIAGNOSTIC  episode BF4_15comp (crowded), seeds as listed, medians')
print('  D* = ORACLE placement (Phase 1 = S2-T1f)   E* = BEAMFORM placement (Phase 1 = S1-A0)')
print('=' * 108)
hdr = f"{'arm':<20}{'n':>2}{'init':>8}{'nRMSE':>9}{'VR':>7}{'rho':>7}{'sum|w|':>8}{'unobs':>7}{'Hs_e':>7}  what"
print(hdr)
for grp in ('D', 'E'):
    for a in [x for x in ORDER if x.startswith(grp)]:
        d = by.get(a)
        if not d:
            continue
        s = sorted(d)
        f = lambda k: float(np.median([d[q][k] for q in s]))
        print(f"{a:<20}{len(s):>2}{f('init_nrmse'):8.4f}{f('nRMSE'):9.4f}{f('VR'):7.3f}"
              f"{f('rho'):7.3f}{f('sum_w_ratio'):8.3f}{f('RMSE_unobs'):7.3f}"
              f"{f('Hs_error'):7.3f}  {WHAT[a]}")
    print('-' * 108)

print()
print('PAIRED per-seed (identical Phase 1 by construction for the p2 arms):')
for base, fix in [('D0-default', 'D2p2-track_only'), ('D0-default', 'D1p2-track_only_l1'),
                  ('E0-A0-default', 'E2p2-track_only'), ('E0-A0-default', 'E1p2-track_only_l1')]:
    s = sorted(set(by[base]) & set(by[fix]))
    dn = [by[fix][q]['nRMSE'] - by[base][q]['nRMSE'] for q in s]
    dr = [by[fix][q]['rho'] - by[base][q]['rho'] for q in s]
    same_init = all(abs(by[fix][q]['init_nrmse'] - by[base][q]['init_nrmse']) < 1e-9 for q in s)
    print(f'  {fix:<20} minus {base:<16} n={len(s)}  d_nRMSE med {np.median(dn):+.4f} '
          f'({sum(x<0 for x in dn)}/{len(s)} better)  d_rho med {np.median(dr):+.4f}  '
          f'Phase-1 identical: {same_init}')

print()
print('COLLAPSE SPEED (arm D0-default, seed 42): how fast the shipped tracker undoes the init')
tr = by['D0-default'][42]['traj']
print(f"{'frame':>6}{'t[s]':>7}{'nRMSE':>9}{'VR':>7}{'rho':>7}{'sum|w|':>8}"
      f"{'lam1*sum|w|':>12}{'data term':>11}{'penalty share':>14}")
for i in [0, 1, 2, 5, 10, 20, 40, 80, 159]:
    p = tr[i]
    l1 = 0.001 * p['sum_w']
    data = p['obs_loss'] - l1
    print(f"{i:6d}{p['t']:7.1f}{p['nRMSE']:9.4f}{p['VR']:7.3f}{p['rho']:7.3f}"
          f"{p['sum_w']:8.3f}{l1:12.6f}{data:11.6f}{l1/p['obs_loss']*100:13.1f}%")
print('  (`obs_loss` returned by update() is the FULL objective = data + L1 + smooth;')
print('   the smoothness term is not separated here, so "data term" is an upper bound.)')

print()
print('CLAMP / WINDOW falsifications:')
for a in ['D7-noclamp', 'D5-window12']:
    r = by[a][42]; b = by['D0-default'][42]
    print(f"  {a:<14} nRMSE {r['nRMSE']:.6f} vs default {b['nRMSE']:.6f}  "
          f"delta {r['nRMSE']-b['nRMSE']:+.6f}   clamp fired on "
          f"{r['clip']['frames_with_clip']}/{r['n_frames']} frames (default: "
          f"{b['clip']['frames_with_clip']}/{b['n_frames']})")
