"""CON audit — geometry of the observed/unobserved partition.

How many grid points does `rmse_observed(min_dist=10)` actually average over?
Read-only: opens the episode HDF5s, no estimator, no GPU.
"""
import glob, os
import numpy as np
import h5py

D = '/home/vexit/wavesplat/wavesplat/data_indep4_2026-08-07'
files = sorted(glob.glob(os.path.join(D, '*.h5')))

print(f'{"episode":26s} {"grid":>9s} {"dx":>5s} {"dy":>5s} '
      f'{"npts/frame":>11s} {"obs@5":>16s} {"obs@10":>16s} {"obs@20":>16s}')
allrows = []
for p in files:
    with h5py.File(p, 'r') as f:
        gx = f['frames/gt_grid_x'][:]
        gy = f['frames/gt_grid_y'][:]
        pts = f['frames/lidar_points'][:]
        msk = f['frames/lidar_valid_mask'][:]
    Xg, Yg = np.meshgrid(gx, gy)
    NT = Xg.size
    stats = {5.0: [], 10.0: [], 20.0: []}
    npts = []
    # tracking frames are the tail of the episode; sample every 5th frame for speed
    for t in range(0, pts.shape[0], 5):
        m = msk[t]
        ox, oy = pts[t, m, 0], pts[t, m, 1]
        npts.append(int(m.sum()))
        if len(ox) == 0:
            for k in stats:
                stats[k].append(0)
            continue
        d = np.sqrt((Xg[..., None] - ox[None, None, :]) ** 2 +
                    (Yg[..., None] - oy[None, None, :]) ** 2).min(axis=2)
        for k in stats:
            stats[k].append(int((d < k).sum()))
    row = (os.path.basename(p)[8:-25], f'{len(gy)}x{len(gx)}',
           float(gx[1] - gx[0]), float(gy[1] - gy[0]),
           np.median(npts),
           np.median(stats[5.0]), np.median(stats[10.0]), np.median(stats[20.0]), NT)
    allrows.append((row, stats, npts, NT))
    print(f'{row[0]:26s} {row[1]:>9s} {row[2]:5.1f} {row[3]:5.1f} {row[4]:11.0f} '
          f'{row[5]:7.0f} ({100*row[5]/NT:4.1f}%) {row[6]:7.0f} ({100*row[6]/NT:4.1f}%) '
          f'{row[7]:7.0f} ({100*row[7]/NT:4.1f}%)')

print('\n--- distribution of the min_dist=10 observed-point count, all frames all episodes ---')
allc = np.concatenate([np.array(s[10.0]) for _, s, _, _ in allrows])
NT = allrows[0][3]
print(f'  grid total = {NT}')
for q in (0, 1, 5, 25, 50, 75, 95, 100):
    print(f'   p{q:3d} = {np.percentile(allc, q):7.1f}  ({100*np.percentile(allc,q)/NT:5.2f}% of grid)')
print(f'  frames with ZERO observed points: {(allc==0).sum()}/{allc.size}')
print(f'  frames with <30 observed points: {(allc<30).sum()}/{allc.size}')

print('\n--- how much of the TOTAL squared error can the observed region carry? ---')
fo = np.median(allc) / NT
print(f'  median observed fraction f = {fo:.4f}')
print(f'  RMSE_total^2 = f*RMSE_obs^2 + (1-f)*RMSE_unobs^2')
print('  check with the Rx-7 block medians (AUTO vs NOTRACK):')
for lbl, ro, ru in (('AUTO', 0.08599, 0.29443), ('AUTO-NOTRACK', 0.24308, 0.27910)):
    tot = np.sqrt(fo * ro ** 2 + (1 - fo) * ru ** 2)
    print(f'   {lbl:14s} obs={ro:.5f} unobs={ru:.5f} -> implied total RMSE={tot:.5f}')

print('\n--- geometry: where are the returns relative to the grid? ---')
p = files[0]
with h5py.File(p, 'r') as f:
    gx = f['frames/gt_grid_x'][:]; gy = f['frames/gt_grid_y'][:]
    pts = f['frames/lidar_points'][:]; msk = f['frames/lidar_valid_mask'][:]
print(f'  grid x [{gx[0]:.1f},{gx[-1]:.1f}] y [{gy[0]:.1f},{gy[-1]:.1f}]')
for t in (0, 150, 300):
    m = msk[t]
    if m.sum():
        print(f'  frame {t}: n={m.sum():3d} x[{pts[t,m,0].min():7.1f},{pts[t,m,0].max():7.1f}] '
              f'y[{pts[t,m,1].min():7.1f},{pts[t,m,1].max():7.1f}]')
    else:
        print(f'  frame {t}: n=0')
# fraction of returns that fall INSIDE the gt grid at all
ins = tot = 0
for t in range(pts.shape[0]):
    m = msk[t]
    if not m.sum():
        continue
    x, y = pts[t, m, 0], pts[t, m, 1]
    ins += int(((x >= gx[0]) & (x <= gx[-1]) & (y >= gy[0]) & (y <= gy[-1])).sum())
    tot += int(m.sum())
print(f'  returns inside the GT grid box: {ins}/{tot} = {100*ins/max(tot,1):.1f}%')
