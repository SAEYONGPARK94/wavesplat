"""CON audit — run-level consistency of the observed/unobserved split with nRMSE,
plus quantification of how close the 'observed region' is to the tracker's training set.
Read-only.
"""
import glob, json, os, sys
import statistics as st
import numpy as np
import h5py

R7 = '/home/vexit/wavesplat/wavesplat/tmp/Rx7_tracker_deploy_2026-08-07'
DATA = '/home/vexit/wavesplat/wavesplat/data_indep4_2026-08-07'
rows = [json.loads(l) for l in open(f'{R7}/results.jsonl') if l.strip()]
rows = [r for r in rows if 'error' not in r]
NULLJ = json.load(open(f'{R7}/nulls_rx7.json'))
HS = {n['episode']: n['Hs_true'] for n in NULLJ}
NULL = {n['episode']: n['zero_null_tracking_nRMSE_mean'] for n in NULLJ}
IDX = {}
for r in rows:
    IDX.setdefault((r['cell'], r['episode']), []).append(r)
EPS = sorted({r['episode'] for r in rows})


def med(c, e, k):
    v = [x[k] for x in IDX[(c, e)] if x.get(k) is not None and x[k] == x[k]]
    return st.median(v)


KEPT = [e for e in EPS if med('AUTO', e, 'nRMSE') < NULL[e]]

print('=' * 78)
print('A. observed-region fraction f per episode (tracking frames only, min_dist=10)')
print('=' * 78)
# tracking frames = last part of the episode (tp_factor=2.0). Use the tail 160 frames
# as an approximation and also report the whole record; f barely moves either way.
F = {}
for e in EPS:
    with h5py.File(os.path.join(DATA, e), 'r') as f:
        gx = f['frames/gt_grid_x'][:]; gy = f['frames/gt_grid_y'][:]
        pts = f['frames/lidar_points'][:]; msk = f['frames/lidar_valid_mask'][:]
    Xg, Yg = np.meshgrid(gx, gy)
    cnt = []
    for t in range(pts.shape[0] - 160, pts.shape[0], 4):
        m = msk[t]
        if not m.sum():
            cnt.append(0); continue
        ox, oy = pts[t, m, 0], pts[t, m, 1]
        d = np.sqrt((Xg[..., None] - ox[None, None, :]) ** 2 +
                    (Yg[..., None] - oy[None, None, :]) ** 2).min(axis=2)
        cnt.append(int((d < 10.0).sum()))
    F[e] = float(np.mean(cnt)) / Xg.size
    print(f'  {e[8:-25]:24s} f={F[e]:.4f}  ({np.mean(cnt):.0f}/{Xg.size} grid pts)')

print('\n' + '=' * 78)
print('B. does the split reconstruct nRMSE at run level? (PREREG 5.6 "verify in the actual runs")')
print('=' * 78)
print(f'  {"cell":14s} {"median implied/actual avg-RMSE ratio":>40s}')
for c in ['AUTO', 'AUTO-LR10', 'AUTO-LR100', 'AUTO-LR1000', 'AUTO-NOTRACK']:
    rr = []
    for r in rows:
        if r['cell'] != c:
            continue
        e = r['episode']; f = F[e]
        implied = np.sqrt(f * r['avg_RMSE_observed'] ** 2 + (1 - f) * r['avg_RMSE_unobserved'] ** 2)
        actual = r['nRMSE'] * HS[e]
        rr.append(implied / actual)
    print(f'  {c:14s} {st.median(rr):40.4f}   (min {min(rr):.3f} max {max(rr):.3f})')
print('  NOTE: exact equality is not expected (frame-mean of RMSE vs RMSE of pooled squares,')
print('        and f varies frame to frame) -- this is a magnitude check, not an identity.')

print('\n  Implied nRMSE ratio AUTO/NOTRACK from the split, vs the measured one:')
ra, rn = [], []
for e in KEPT:
    f = F[e]
    a = np.sqrt(f * med('AUTO', e, 'avg_RMSE_observed') ** 2 +
                (1 - f) * med('AUTO', e, 'avg_RMSE_unobserved') ** 2)
    n = np.sqrt(f * med('AUTO-NOTRACK', e, 'avg_RMSE_observed') ** 2 +
                (1 - f) * med('AUTO-NOTRACK', e, 'avg_RMSE_unobserved') ** 2)
    ra.append(a / n)
    rn.append(med('AUTO', e, 'nRMSE') / med('AUTO-NOTRACK', e, 'nRMSE'))
print(f'   from split: median ratio {st.median(ra):.4f}   measured nRMSE: {st.median(rn):.4f}')

print('\n' + '=' * 78)
print('C. TAUTOLOGY quantification: how close is the "observed region" to the training set?')
print('=' * 78)
e = EPS[0]
with h5py.File(os.path.join(DATA, e), 'r') as f:
    gx = f['frames/gt_grid_x'][:]; gy = f['frames/gt_grid_y'][:]
    pts = f['frames/lidar_points'][:]; msk = f['frames/lidar_valid_mask'][:]
    gt = f['frames/gt_wave_field'][:]
    ts = f['frames/timestamps'][:]
Xg, Yg = np.meshgrid(gx, gy)
t = 200
m = msk[t]; ox, oy = pts[t, m, 0], pts[t, m, 1]
d = np.sqrt((Xg[..., None] - ox[None, None, :]) ** 2 + (Yg[..., None] - oy[None, None, :]) ** 2).min(axis=2)
sel = d < 10.0
print(f'  n returns this frame          = {m.sum()}')
print(f'  n grid points in observed mask= {sel.sum()}  -> {sel.sum()/m.sum():.1f} grid pts per return')
print(f'  grid spacing dx={gx[1]-gx[0]:.1f} m, dy={gy[1]-gy[0]:.1f} m -> 10 m = '
      f'{10/(gx[1]-gx[0]):.1f} cells in x, {10/(gy[1]-gy[0]):.1f} cells in y')
print(f'  distance percentiles inside the observed mask: '
      f'p50={np.percentile(d[sel],50):.2f} m  p90={np.percentile(d[sel],90):.2f} m')
for Tp, lbl in ((7.0, 'BF4'), (9.0, 'BF5')):
    lam = 9.81 * Tp ** 2 / (2 * np.pi)
    print(f'  {lbl}: Tp={Tp}s -> deep-water peak wavelength {lam:.0f} m; '
          f'min_dist=10 m = lambda/{lam/10:.0f}')
print('  => within the "observed" mask the true surface is essentially the measured')
print('     surface interpolated; the tracker was fit to those points at THIS SAME t.')

print('\n  timing check (is the evaluation in-sample?):')
print('    run.py:198  update_result = updater.update(frame)')
print('    run.py:201  pred = updater.predict_field(grid_x, grid_y, frame.t)')
print('    run.py:206  compute_all_metrics(..., obs_x=frame.x_obs, obs_y=frame.y_obs)')
print('    -> same frame, same instant, same points: RMSE_observed is a POST-FIT')
print('       training residual, evaluated on the grid nodes nearest the fit points.')

print('\n  measurement noise floor (what a perfect fit to the data would leave):')
sys.path.insert(0, '/home/vexit/wavesplat/wavesplat')
try:
    from wavesim.config import LiDARConfig
    lc = LiDARConfig()
    for k in dir(lc):
        if 'nois' in k or 'std' in k or 'sigma' in k or 'accur' in k:
            print(f'    LiDARConfig.{k} = {getattr(lc,k)}')
except Exception as ex:
    print('    (could not import LiDARConfig:', ex, ')')

print('\n' + '=' * 78)
print('D. is observed/unobserved just a NEAR/FAR split in disguise?')
print('=' * 78)
xs = []
for t in range(pts.shape[0] - 160, pts.shape[0], 8):
    mm = msk[t]
    if mm.sum():
        xs.append(pts[t, mm, 0])
xs = np.concatenate(xs)
print(f'  returns x-range over tracking frames: [{xs.min():.1f}, {xs.max():.1f}] m; '
      f'grid x-range [{gx[0]:.1f}, {gx[-1]:.1f}] m')
print(f'  => the "observed" mask can only ever live in x < {xs.max()+10:.0f} m, i.e. the near '
      f'{100*(xs.max()+10-gx[0])/(gx[-1]-gx[0]):.0f}% of the x-extent.')
print(f'  metrics.py already ships rmse_spatial_split(split_distance=80 m) = RMSE_near/RMSE_far.')
rn_, rf_ = [], []
for r in rows:
    if r['cell'] == 'AUTO':
        rn_.append(r.get('RMSE_near')); rf_.append(r.get('RMSE_far'))
print(f'  but RMSE_near/RMSE_far were NOT dumped to results.jsonl '
      f'(present={rn_[0] is not None}) -> the near/far confound cannot be separated from raw data.')
