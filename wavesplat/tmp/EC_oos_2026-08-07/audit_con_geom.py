"""CON audit (E-C) part 2 -- geometry, mask coverage, and the temporal-leakage scale.

Answers, from the raw episodes (CPU, read-only, no estimator run):
  (a) grid coverage fraction of the min_dist mask at every audited radius
  (b) frame dt and how much the TRUE field moves in one dt at the observation points
      -> the entire dynamic range available to an "out-of-sample" test at 1-frame lead
  (c) how much the observation POINTS move in one dt (spatial overlap = spatial leakage)
"""
import os
import sys

import numpy as np

REPO = '/home/vexit/wavesplat/wavesplat'
sys.path.insert(0, REPO)
from wavesplat.data_loader import EpisodeDataLoader  # noqa: E402

DIRS = [os.path.join(REPO, d) for d in ('data_indep2_2026-08-07',
                                        'data_indep3_2026-08-07',
                                        'data_indep4_2026-08-07')]
EPS = ['episode_BF4_15comp_seed0_uniform_ra_rf_ds.h5',
       'episode_BF4_30comp_seed3_uniform_ra_rf_ds.h5',
       'episode_BF5_15comp_seed8_uniform_ra_rf_ds.h5',
       'episode_BF5_30comp_seed11_uniform_ra_rf_ds.h5']
RADS = (5, 10, 20, 30, 50, 75, 100, 150)


def resolve(ep):
    for d in DIRS:
        p = os.path.join(d, ep)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(ep)


def npy(t):
    return t.cpu().numpy() if hasattr(t, 'cpu') else np.asarray(t)


print('=== (a) grid, footprint, MASK COVERAGE per radius ===')
cov_all = {r: [] for r in RADS}
for ep in EPS:
    ld = EpisodeDataLoader(resolve(ep))
    gx, gy = npy(ld.gt_grid_x), npy(ld.gt_grid_y)
    Xg, Yg = np.meshgrid(gx, gy)
    fr = ld.get_tracking_frames(tp_factor=2.0)
    print(f'\n {ep}')
    print(f'   Hs={ld.Hs_true:.3f}m Tp={ld.Tp_true:.2f}s  grid x[{gx[0]:.0f},{gx[-1]:.0f}] '
          f'n={len(gx)} d={gx[1]-gx[0]:.2f}m | y[{gy[0]:.0f},{gy[-1]:.0f}] n={len(gy)}  '
          f'diag={np.hypot(gx[-1]-gx[0], gy[-1]-gy[0]):.0f}m')
    print(f'   tracking frames={len(fr)}  t=[{fr[0].t:.2f},{fr[-1].t:.2f}]s  '
          f'dt={fr[1].t-fr[0].t:.4f}s  n_pts/frame med={int(np.median([len(f.x_obs) for f in fr]))}')
    idxs = [0, len(fr) // 2, len(fr) - 1]
    for i in idxs:
        ox, oy = npy(fr[i].x_obs), npy(fr[i].y_obs)
        md = np.full(Xg.shape, np.inf)
        for j in range(len(ox)):
            md = np.minimum(md, np.hypot(Xg - ox[j], Yg - oy[j]))
        c = {r: float((md < r).mean()) for r in RADS}
        for r in RADS:
            cov_all[r].append(c[r])
        print(f'   frame {i:3d} t={fr[i].t:6.2f}s  cover%: ' +
              ' '.join(f'r{r}={100*c[r]:5.1f}' for r in RADS))
        print(f'              obs x[{ox.min():.0f},{ox.max():.0f}] y[{oy.min():.0f},{oy.max():.0f}]')
print('\n  MEAN coverage over sampled frames/episodes:')
for r in RADS:
    print(f'    r={r:4d} m -> {100*np.mean(cov_all[r]):5.1f}% of grid inside the '
          f'"observed" mask  (complement {100*(1-np.mean(cov_all[r])):5.1f}%)')

print('\n=== (b) how much does the TRUE field move in one frame? ===')
print('  (upper bound on what any 1-frame-ahead "out-of-sample" test can detect)')
for ep in EPS:
    ld = EpisodeDataLoader(resolve(ep))
    fr = ld.get_tracking_frames(tp_factor=2.0)
    gx, gy = npy(ld.gt_grid_x), npy(ld.gt_grid_y)
    dt = fr[1].t - fr[0].t
    dfs, dfs2, dfs5, rms = [], [], [], []
    for i in range(1, min(60, len(fr))):
        a = npy(fr[i - 1].gt_field)
        b = npy(fr[i].gt_field)
        dfs.append(float(np.sqrt(np.mean((b - a) ** 2))))
        rms.append(float(np.sqrt(np.mean(b ** 2))))
    for i in range(10, min(60, len(fr))):
        a10 = npy(fr[i - 10].gt_field)
        b = npy(fr[i].gt_field)
        dfs2.append(float(np.sqrt(np.mean((b - a10) ** 2))))
    for i in range(50, min(120, len(fr))):
        a50 = npy(fr[i - 50].gt_field)
        b = npy(fr[i].gt_field)
        dfs5.append(float(np.sqrt(np.mean((b - a50) ** 2))))
    print(f'  {ep[8:-20]:22s} dt={dt:.3f}s  rms(field)={np.mean(rms):.4f}m  '
          f'rms(Δ1frame)={np.mean(dfs):.5f}m ({100*np.mean(dfs)/np.mean(rms):.1f}% of field) | '
          f'Δ10fr={np.mean(dfs2):.4f}m | Δ50fr={np.mean(dfs5) if dfs5 else float("nan"):.4f}m')

print('\n=== (c) do the OBSERVATION POINTS move between consecutive frames? ===')
print('  (if frame i lands on the same coordinates as frame i-1, the "out-of-sample"')
print('   residual is evaluated where the model was just fitted = spatial in-sample)')
for ep in EPS[:2]:
    ld = EpisodeDataLoader(resolve(ep))
    fr = ld.get_tracking_frames(tp_factor=2.0)
    for i in (1, len(fr) // 2):
        a = np.stack([npy(fr[i - 1].x_obs), npy(fr[i - 1].y_obs)], 1)
        b = np.stack([npy(fr[i].x_obs), npy(fr[i].y_obs)], 1)
        d = np.sqrt(((b[:, None, :] - a[None, :, :]) ** 2).sum(-1)) if len(a) * len(b) < 4e6 \
            else None
        if d is None:
            print('   (skip, too many points)')
            continue
        nn = d.min(1)
        print(f'  {ep[8:-20]:22s} frame {i:3d}: |obs_i|={len(b)} |obs_(i-1)|={len(a)}  '
              f'nearest-neighbour distance to previous frame: '
              f'med={np.median(nn):.3f}m p90={np.percentile(nn,90):.3f}m  '
              f'frac<0.5m={100*float((nn<0.5).mean()):.1f}%  frac<2m={100*float((nn<2).mean()):.1f}%')
        # how far does the frame-i cloud sit from the frame-(i-1) cloud in bulk?
        print(f'      centroid shift = {np.hypot(*(b.mean(0)-a.mean(0))):.3f} m')
