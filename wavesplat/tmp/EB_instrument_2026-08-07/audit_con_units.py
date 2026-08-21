"""CON audit part 3: unit-consistent radius table, DoF / variance-explained, Rx-7 overlap."""
import json
import math
import os
import statistics as st
import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = '/home/vexit/wavesplat/wavesplat'
D4 = os.path.join(REPO, 'data_indep4_2026-08-07')
L = ['AUTO-NOTRACK', 'AUTO-LR1000', 'AUTO-LR100', 'AUTO-LR10', 'AUTO']


def sgn_p(k, n):
    lo = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n) if n else 1.0


def build(f):
    rows = [json.loads(l) for l in open(os.path.join(HERE, f)) if l.strip()]
    I = {}
    for r in rows:
        if 'error' not in r:
            I.setdefault((r['cell'], r['episode']), []).append(r)
    return rows, I


rows1, I1 = build('results_lane1.jsonl')
EPS1 = sorted({r['episode'] for r in rows1})
NULL = {n['episode']: n['zero_null_tracking_nRMSE_mean'] for n in
        json.load(open(os.path.join(HERE, '..', 'Rx7_tracker_deploy_2026-08-07',
                                    'nulls_rx7.json')))}


def med(I, c, e, k):
    v = [x[k] for x in I.get((c, e), []) if x.get(k) is not None and x[k] == x[k]]
    return st.median(v) if v else None


def medr(I, c, e, rk):
    v = [x['rmse_by_radius'][rk] for x in I.get((c, e), [])]
    return st.median(v)


KEPT = [e for e in EPS1 if med(I1, 'AUTO', e, 'nRMSE') < NULL[e]]

HS = {}
for e in EPS1:
    with h5py.File(os.path.join(D4, e), 'r') as f:
        gt = f['frames/gt_wave_field'][:]
        HS[e] = 4.0 * float(np.std(gt))
        z = f['frames/lidar_points'][:, :, 2]
        m = f['frames/lidar_valid_mask'][:]
        HS[e + '|zrms'] = float(np.sqrt(np.mean(z[m] ** 2)))
        HS[e + '|npts'] = float(np.median(m.sum(1)))

print('=' * 78)
print('1. UNIT-CONSISTENT RADIUS TABLE  (analysis.txt mixes metres and nRMSE)')
print('   grid = x[10,250] y[-100,100] m ; max min-dist to a return ~ 220-231 m')
print('   geometric mask coverage (measured, 2 episodes x 3 frames):')
print('     r=5 -> 2-3.5% | r=10 -> 3.5-6.7% | r=20 -> 8-13% | r=30 -> 14-22% |'
      ' r=50 -> 26-39% | r=75 -> 39-56% | r=100 -> 51-69% | r=150 -> 73-93%')
print(f'   {"r[m]":>6s} {"medD_obs[m]":>12s} {"win":>7s} {"p":>9s}   '
      f'{"gridcov":>8s}')
for rk in ('5', '10', '20', '30', '50'):
    d = [medr(I1, 'AUTO', e, rk) - medr(I1, 'AUTO-NOTRACK', e, rk) for e in KEPT]
    d = [x for x in d if x == x]
    w = sum(1 for x in d if x < 0)
    cov = {'5': '~3%', '10': '~5%', '20': '~11%', '30': '~18%', '50': '~31%'}[rk]
    print(f'   {rk:>6s} {st.median(d):+12.5f} {w:3d}/{len(d):<3d} '
          f'{sgn_p(w,len(d)):9.5f}   {cov:>8s}')
dm = [(med(I1, 'AUTO', e, 'nRMSE') - med(I1, 'AUTO-NOTRACK', e, 'nRMSE')) * HS[e]
      for e in KEPT]
w = sum(1 for x in dm if x < 0)
print(f'   {"~230":>6s} {st.median(dm):+12.5f} {w:3d}/{len(dm):<3d} '
      f'{sgn_p(w,len(dm)):9.5f}   {"100%":>8s}   <- full grid, CONVERTED TO METRES')
print('   => the profile never leaves the inner ~31% of the grid.  The sign change')
print('      lies inside the ~69% of grid AREA that carries no measurement.')

print('\n2. DoF / VARIANCE EXPLAINED  (is obs_resid_rms an interpolation residual?)')
print(f'   median LiDAR returns per frame = {st.median([HS[e+"|npts"] for e in EPS1]):.0f}'
      f'  (tracking_window=3 => ~{3*st.median([HS[e+"|npts"] for e in EPS1]):.0f} obs in the fitted window)')
zr = st.median([HS[e + '|zrms'] for e in EPS1])
print(f'   RMS(z_lidar) over all valid returns, block median = {zr:.4f} m'
      f'   (= obs_resid_rms of a ZERO field)')
print(f'   {"cell":14s} {"N med":>6s} {"2N (w,phi)":>11s} {"obs/param":>10s} '
      f'{"resid[m]":>9s} {"1-resid^2/z^2":>14s} {"nRMSE":>7s}')
for c in L[::-1]:
    Nm = st.median([med(I1, c, e, 'N') for e in KEPT])
    rr = st.median([med(I1, c, e, 'obs_resid_rms') for e in KEPT])
    nr = st.median([med(I1, c, e, 'nRMSE') for e in KEPT])
    print(f'   {c:14s} {Nm:6.0f} {2*Nm:11.0f} {3*49/(2*Nm):10.1f} {rr:9.4f} '
          f'{100*(1-(rr/zr)**2):13.1f}% {nr:7.4f}')
print('   => the arm that explains ~98% of the observations in-sample is the arm with')
print('      the WORST field; the arm that explains ~37% has the best field.')

print('\n3. IS LANE 1 NEW DATA?  overlap with Rx-7')
r7 = [json.loads(l) for l in open(os.path.join(HERE, '..',
      'Rx7_tracker_deploy_2026-08-07', 'results.jsonl')) if l.strip()]
e7 = {x['episode'] for x in r7}
s7 = sorted({x['seed'] for x in r7})
print(f'   Rx-7 episodes n={len(e7)} seeds={s7}')
print(f'   E-B lane1 episodes n={len(EPS1)} seeds=[1539,1540,1541]')
print(f'   EPISODE overlap = {len(e7 & set(EPS1))}/16 ; SEED overlap = 0')
I7 = {}
for x in r7:
    if 'error' not in x:
        I7.setdefault((x['cell'], x['episode']), []).append(x)
kept7 = [e for e in sorted(e7) if med(I7, 'AUTO', e, 'nRMSE') < NULL[e]]
d7 = [med(I7, 'AUTO', e, 'nRMSE') - med(I7, 'AUTO-NOTRACK', e, 'nRMSE') for e in kept7]
w7 = sum(1 for x in d7 if x < 0)
print(f'   Rx-7 nRMSE D(AUTO-NOTRACK) on ITS kept block n={len(kept7)}: '
      f'med={st.median(d7):+.5f} improve {w7}/{len(d7)}')
print('   => lane 1 re-scores the SAME 16 episodes with 3 new estimator seeds.')
print('      The nRMSE limb of PRIMARY-1 is a seed replication, not a new block.')
print('      Only avg_obs_loss / obs_resid_rms / rmse_by_radius are new measurements.')

print('\n4. WHAT AN OBJECTIVE-ONLY SIGN TEST IS WORTH')
print('   avg_obs_loss fell in 48/48 runs and in 192/192 adjacent lr-rung comparisons.')
print('   The sign test assumes P(down)=0.5 under H0.  Measured exception rate = 0/192,')
print('   so the realised P(down) is indistinguishable from 1.  p=1.2e-04 on that limb')
print('   is a statement about a quantity with no observed variance in sign.')
