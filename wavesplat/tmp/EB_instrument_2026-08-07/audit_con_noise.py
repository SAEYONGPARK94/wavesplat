"""CON audit part 4: what is the vertical observation-noise floor?
If obs_resid_rms sits BELOW it, the 'GT-free residual' is fitting sensor noise."""
import os
import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

REPO = '/home/vexit/wavesplat/wavesplat'
D4 = os.path.join(REPO, 'data_indep4_2026-08-07')
EPS = sorted(e for e in os.listdir(D4) if e.endswith('.h5'))

print('vertical observation-noise floor  RMS(z_lidar - GT(x_lidar,y_lidar))')
print(f'{"episode":42s} {"sigma_z[m]":>10s} {"RMS(z)[m]":>10s} {"bilin_err_est":>13s}')
allv = []
for e in EPS:
    with h5py.File(os.path.join(D4, e), 'r') as f:
        gx = f['frames/gt_grid_x'][:]
        gy = f['frames/gt_grid_y'][:]
        gt = f['frames/gt_wave_field'][:]          # (T, ny, nx)
        pts = f['frames/lidar_points'][:]          # (T, K, 3)
        msk = f['frames/lidar_valid_mask'][:]
    res, zz = [], []
    for t in range(0, gt.shape[0], 7):
        m = msk[t]
        if m.sum() == 0:
            continue
        p = pts[t][m]
        itp = RegularGridInterpolator((gy, gx), gt[t], bounds_error=False,
                                      fill_value=np.nan)
        g = itp(np.stack([p[:, 1], p[:, 0]], 1))
        ok = np.isfinite(g)
        res.append(p[ok, 2] - g[ok])
        zz.append(p[ok, 2])
    r = np.concatenate(res)
    z = np.concatenate(zz)
    # crude bilinear-interpolation error scale: (dx^2/8)*|d2z/dx2| ~ using field curvature
    d2 = np.diff(gt[::20], 2, axis=2)
    bil = float((gx[1] - gx[0]) ** 2 / 8 * np.sqrt(np.mean((d2 / (gx[1] - gx[0]) ** 2) ** 2)))
    allv.append((float(np.sqrt(np.mean(r ** 2))), float(np.sqrt(np.mean(z ** 2)))))
    print(f'{e[8:-3]:42s} {allv[-1][0]:10.4f} {allv[-1][1]:10.4f} {bil:13.4f}')
a = np.array(allv)
print(f'\nblock median sigma_z = {np.median(a[:,0]):.4f} m   '
      f'RMS(z) = {np.median(a[:,1]):.4f} m')
print('\ncompare with obs_resid_rms block medians (lane1, 14 kept episodes):')
print('  AUTO 0.0364 | LR10 0.0524 | LR100 0.1551 | LR1000 0.2303 | NOTRACK 0.2397')
print('  (sigma_z above INCLUDES bilinear interpolation error, so it is an UPPER bound')
print('   on the true sensor noise floor.)')

print('\n=== DISCREPANCY-PRINCIPLE CHECK (the stated consequence of SURROGATE-VALID) ===')
sig = float(np.median(a[:, 0]))
lvl = {'AUTO': 0.03640, 'AUTO-LR10': 0.05243, 'AUTO-LR100': 0.15507,
       'AUTO-LR1000': 0.23025, 'AUTO-NOTRACK': 0.23966}
nr = {'AUTO': 0.1976, 'AUTO-LR10': 0.1895, 'AUTO-LR100': 0.1732,
      'AUTO-LR1000': 0.1743, 'AUTO-NOTRACK': 0.1748}
print(f'  sigma_z (UPPER bound, includes bilinear interp error) = {sig:.4f} m')
for c in ('AUTO-NOTRACK', 'AUTO-LR1000', 'AUTO-LR100', 'AUTO-LR10', 'AUTO'):
    print(f'  {c:14s} obs_resid={lvl[c]:.4f} = {lvl[c]/sig:5.1f} sigma   nRMSE={nr[c]:.4f}')
best = min(nr, key=nr.get)
print(f'  nRMSE-optimal rung = {best} at {lvl[best]/sig:.1f} sigma.')
print('  A discrepancy-principle rule ("iterate until the residual reaches the noise')
print('  floor") would run PAST every rung on this ladder and land beyond AUTO, i.e. at')
print('  the WORST field.  Picking the right rung needs the constant 13.8*sigma, which is')
print('  only knowable from GT.  So SURROGATE-VALID does not deliver a GT-free stopping rule.')
