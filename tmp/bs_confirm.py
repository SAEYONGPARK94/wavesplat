"""
Confirm BS positive across frames x seeds, with FAIR (in-hull) scoring for the naive comparison.
"""
import os, sys, numpy as np
sys.path.insert(0, '/home/vexit/wavesplat/wavesplat')
os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0, 'scripts')
from scripts.realdata_analysis import load_surface_nc
from d18_relax_estimator import relax_step
from d17_recursive_estimator import refine_params
from scipy.interpolate import griddata
from scipy.spatial import Delaunay

F = 'data/real_bergamasco/BS_2011/Surfaces_20111004_113800_short.nc'
DEPTH = 30.0
Z, gx, gy, t, meta = load_surface_nc(F, tmax_frames=400)
T, Ny, Nx = Z.shape; dt = float(np.median(np.diff(t))); Xg, Yg = np.meshgrid(gx, gy)
Tp = 2.4; n_back = min(int(4*Tp/dt), T-2)
frames = np.linspace(n_back+2, T-2, 5).astype(int)

def obs_window(i_star, seed, nper=40):
    rng = np.random.default_rng(seed); Ps, zs = [], []
    for i in range(i_star-n_back, i_star+1):
        idx = rng.choice(Nx*Ny, nper, replace=False)
        Ps.append(np.column_stack([Xg.ravel()[idx], Yg.ravel()[idx], np.full(nper, i*dt)])); zs.append(Z[i].ravel()[idx])
    return Ps, np.vstack(Ps).astype(np.float32), np.concatenate(zs).astype(np.float32)

def relax(P, z, target, klim, mc, pr, it):
    kx, ky = np.zeros(0), np.zeros(0); fld=None
    for _ in range(3):
        kx, ky = relax_step(kx, ky, P, z, DEPTH, max_comp=mc, prune_frac=pr, klim=klim)
        kx, ky, fld = refine_params(kx, ky, P, z, DEPTH, target, iters=it)
    return fld

res = {k: {'nr': [], 'cr': []} for k in ['default', 'fixed', 'naive']}
for i_star in frames:
    gt = Z[i_star]; Hs = 4*np.nanstd(gt)
    target = {'gx': gx, 'gy': gy, 't': float(i_star*dt), 'gt': np.nan_to_num(gt), 'Hs': Hs}
    for seed in range(3):
        Ps, P, z = obs_window(i_star, seed)
        # fair region: inside convex hull of last-frame obs (where naive interpolates, not extrapolates)
        hull = Delaunay(Ps[-1][:, :2]); inside = hull.find_simplex(np.column_stack([Xg.ravel(), Yg.ravel()])) >= 0
        m = inside.reshape(gt.shape) & ~np.isnan(gt)
        def sc(f): return float(np.sqrt(np.mean((f[m]-gt[m])**2))/Hs), float(np.corrcoef(f[m], gt[m])[0,1])
        fd = relax(P, z, target, 0.6, 32, 0.05, 200); a = sc(fd); res['default']['nr'].append(a[0]); res['default']['cr'].append(a[1])
        ff = relax(P, z, target, 1.5, 8, 0.12, 400); b = sc(ff); res['fixed']['nr'].append(b[0]); res['fixed']['cr'].append(b[1])
        nv = griddata(Ps[-1][:, :2], z[-40:], (Xg, Yg), method='linear')
        c = sc(np.nan_to_num(nv, nan=np.nanmean(z))); res['naive']['nr'].append(c[0]); res['naive']['cr'].append(c[1])

print(f"BS_2011 confirm — {len(frames)} frames x 3 seeds, FAIR in-hull scoring")
for k in ['default', 'fixed', 'naive']:
    nr = np.array(res[k]['nr']); cr = np.array(res[k]['cr'])
    print(f"  {k:8s}  nRMSE {nr.mean():.3f}+-{nr.std():.3f}   corr {cr.mean():.3f}+-{cr.std():.3f}")
d = np.array(res['fixed']['nr']); n = np.array(res['naive']['nr'])
print(f"  fixed beats naive in {int((d<n).sum())}/{len(d)} cases; mean margin {(n-d).mean():+.3f}")
