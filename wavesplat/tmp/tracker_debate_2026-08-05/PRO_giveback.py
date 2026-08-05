"""PRO agent -- "the better the initialisation, the more the tracker destroys".

Reads tmp/s1s2_2026-08-05/results.jsonl only (no new wave runs) and regresses the
per-cell GIVE-BACK  (tracked nRMSE - init nRMSE)  on the init quality itself.
Unit = (arm, episode) median-of-3, matching the S1/S2 pre-registration.
The two tracking_lr=0 arms are excluded from the fit and shown as the control.
"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 's1s2_2026-08-05', 'results.jsonl')
EPS = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
       'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
SH = {e: e.split('_')[1] + '_' + e.split('_')[2] for e in EPS}
NOTRACK = {'X-A0-notrack', 'X-T1f-notrack'}

rec = {}
for line in open(SRC):
    r = json.loads(line)
    if 'error' in r:
        continue
    rec.setdefault((r['arm'], r['episode']), []).append(r)

pts, ctrl = [], []
for (arm, ep), rs in sorted(rec.items()):
    i = float(np.median([x['init_nrmse'] for x in rs]))
    t = float(np.median([x['nRMSE'] for x in rs]))
    (ctrl if arm in NOTRACK else pts).append((arm, SH[ep], i, t, t - i))

print('=' * 88)
print('TRACKER GIVE-BACK vs INITIALISATION QUALITY   (unit = arm x episode, median-of-3)')
print('  give-back = mean tracked nRMSE  -  nRMSE at the last init frame')
print('=' * 88)
print(f"{'arm':<16}{'episode':<12}{'init':>9}{'tracked':>9}{'give-back':>11}{'x init':>8}")
for a, e, i, t, d in sorted(pts, key=lambda z: z[2]):
    print(f'{a:<16}{e:<12}{i:9.4f}{t:9.4f}{d:+11.4f}{t/i:8.2f}')
print('-' * 88)
print('CONTROL: tracker frozen (tracking_lr = 0)')
for a, e, i, t, d in sorted(ctrl, key=lambda z: z[2]):
    print(f'{a:<16}{e:<12}{i:9.4f}{t:9.4f}{d:+11.4f}{t/i:8.2f}')

x = np.array([p[2] for p in pts]); y = np.array([p[4] for p in pts])
xc = np.array([p[2] for p in ctrl]); yc = np.array([p[4] for p in ctrl])
b, a0 = np.polyfit(x, y, 1)
r = np.corrcoef(x, y)[0, 1]
print()
print(f'TRACKED arms  n={len(x)}   give-back = {a0:+.4f} {b:+.4f} * init      '
      f'Pearson r = {r:+.3f}')
print(f'    -> extrapolated give-back at a PERFECT init (init=0): {a0:+.4f} nRMSE')
print(f'    -> give-back is zero at init = {-a0/b:.4f}  (above that the tracker helps)')
bc = np.polyfit(xc, yc, 1)
print(f'FROZEN arms   n={len(xc)}   give-back = {bc[1]:+.4f} {bc[0]:+.4f} * init      '
      f'Pearson r = {np.corrcoef(xc, yc)[0,1]:+.3f}')
print()
print('Split by init quality (tracked arms only):')
for lo, hi in [(0.0, 0.10), (0.10, 0.20), (0.20, 1.0)]:
    s = [p for p in pts if lo <= p[2] < hi]
    if s:
        print(f'  init in [{lo:.2f},{hi:.2f}) : n={len(s):2d}  '
              f'median give-back {np.median([q[4] for q in s]):+.4f}   '
              f'median ratio {np.median([q[3]/q[2] for q in s]):.2f}x   '
              f'hurt {sum(1 for q in s if q[4] > 0)}/{len(s)}')

# The bound the paper states for this same quantity
print()
print('paper/wavesplat.tex L2547-2551 states, for 24 arm-episode pairs of the re-measured')
print('ablation/model-order experiments: tracking ends above its own init on 20/24, margin')
print('+0.001 to +0.022 nRMSE, median +0.003.  Cells here that exceed +0.022:')
for a, e, i, t, d in sorted(pts, key=lambda z: -z[4]):
    if d > 0.022:
        print(f'   {a:<16}{e:<12} give-back {d:+.4f}   ({d/0.022:.1f}x the stated bound)')
