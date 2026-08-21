"""CON audit part 2: grid geometry / mask coverage / DoF ratio.  CPU only, read only."""
import os
import sys
import h5py
import numpy as np

REPO = '/home/vexit/wavesplat/wavesplat'
sys.path.insert(0, REPO)
D = os.path.join(REPO, 'data_indep4_2026-08-07')
EPS = ['episode_BF4_15comp_seed8_uniform_ra_rf_ds.h5',
       'episode_BF5_30comp_seed10_uniform_ra_rf_ds.h5']

for ep in EPS:
    p = os.path.join(D, ep)
    with h5py.File(p, 'r') as f:
        def walk(g, pre=''):
            out = []
            for k in g:
                v = g[k]
                if isinstance(v, h5py.Group):
                    out += walk(v, pre + k + '/')
                else:
                    out.append((pre + k, v.shape, str(v.dtype)))
            return out
        if ep == EPS[0]:
            print('=== h5 layout ===')
            for n, s, d in walk(f)[:40]:
                print('  ', n, s, d)
            print('   attrs:', dict(f.attrs))
    break

from wavesplat.data_loader import EpisodeDataLoader  # noqa: E402

print('\n=== grid extent, LiDAR footprint, mask coverage, DoF ===')
for ep in EPS:
    ld = EpisodeDataLoader(os.path.join(D, ep))
    gx = np.asarray(ld.gt_grid_x.cpu() if hasattr(ld.gt_grid_x, 'cpu') else ld.gt_grid_x)
    gy = np.asarray(ld.gt_grid_y.cpu() if hasattr(ld.gt_grid_y, 'cpu') else ld.gt_grid_y)
    Xg, Yg = np.meshgrid(gx, gy)
    frames = ld.get_tracking_frames(tp_factor=2.0)
    print(f'\n {ep}')
    print(f'   Hs_true={ld.Hs_true:.3f} Tp={ld.Tp_true:.2f}  grid x[{gx[0]:.1f},{gx[-1]:.1f}] '
          f'({len(gx)} pts, d={gx[1]-gx[0]:.2f}m)  y[{gy[0]:.1f},{gy[-1]:.1f}] ({len(gy)})')
    print(f'   tracking frames={len(frames)}')
    npts = [len(fr.x_obs) for fr in frames]
    print(f'   n_pts/frame: min={min(npts)} median={int(np.median(npts))} max={max(npts)}')
    zr = []
    for fr in frames[:20]:
        z = fr.z_obs.cpu().numpy()
        zr.append(float(np.sqrt(np.mean(z ** 2))))
    print(f'   RMS(z_obs) over first 20 frames = {np.mean(zr):.4f} m  '
          f'(= the residual a ZERO field would score)')
    for idx in (0, len(frames) // 2, len(frames) - 1):
        fr = frames[idx]
        ox = fr.x_obs.cpu().numpy()
        oy = fr.y_obs.cpu().numpy()
        md = np.full(Xg.shape, np.inf)
        for j in range(len(ox)):
            md = np.minimum(md, np.sqrt((Xg - ox[j]) ** 2 + (Yg - oy[j]) ** 2))
        cov = {r: float((md < r).mean()) for r in (5, 10, 20, 30, 50, 75, 100, 150)}
        print(f'   frame{idx:4d} t={fr.t:6.1f} n={len(ox):5d} '
              f'x[{ox.min():.0f},{ox.max():.0f}] y[{oy.min():.0f},{oy.max():.0f}] '
              f'maxmindist={md.max():.1f}m  coverage: '
              + ' '.join(f'r{r}={100*c:.1f}%' for r, c in cov.items()))
    # DoF: components x 4 params vs observations in the 3-frame tracking window
    print(f'   DoF ratio: window={3} frames -> ~{3*int(np.median(npts))} obs vs '
          f'N*3 free params (N in 3..30 => 9..90); ratio '
          f'{3*int(np.median(npts))/90:.0f}x..{3*int(np.median(npts))/9:.0f}x')
