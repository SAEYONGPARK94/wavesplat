"""
B) BS_2011 — quantify the two things naive/interp CANNOT do, with the fixed estimator:
   (1) FORECAST  : fit dispersion comps from past window -> propagate. vs persistence(frozen).
   (2) EXTRAPOLATION: reconstruct OUTSIDE the observation hull. vs naive(collapses to mean there).
Multi-frame x seed. Fixed estimator = klim=1.5 + model-order control.
"""
import os, sys, numpy as np
sys.path.insert(0, '/home/vexit/wavesplat/wavesplat')
os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0, 'scripts')
from scripts.realdata_analysis import load_surface_nc
from d18_relax_estimator import relax_step
from d17_recursive_estimator import refine_params
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from scipy.interpolate import griddata
from scipy.spatial import Delaunay

F = 'data/real_bergamasco/BS_2011/Surfaces_20111004_113800_short.nc'; DEPTH = 30.0
Z, gx, gy, t, meta = load_surface_nc(F, tmax_frames=400)
T, Ny, Nx = Z.shape; dt = float(np.median(np.diff(t))); Xg, Yg = np.meshgrid(gx, gy)
Tp = 2.4; n_back = min(int(4*Tp/dt), T-2)
frames = np.linspace(n_back+2, T-2-int(8/dt), 4).astype(int)   # leave room for +8s forecast
LEADS = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]

def fixed_components(P, z, target):
    kx, ky = np.zeros(0), np.zeros(0)
    for _ in range(3):
        kx, ky = relax_step(kx, ky, P, z, DEPTH, max_comp=8, prune_frac=0.12, klim=1.5)
        kx, ky, _ = refine_params(kx, ky, P, z, DEPTH, target, iters=400)
    obs = np.column_stack([P[:,0],P[:,1],P[:,2],z]).astype(np.float32)
    A, phi = fit_amplitudes_phases(kx, ky, obs, DEPTH, lambda_reg=0.5)
    return kx, ky, A, phi

fc = {dl: [] for dl in LEADS}; pers = {dl: [] for dl in LEADS}
extr_ws, extr_nv = [], []
for i_star in frames:
    for seed in range(2):
        rng = np.random.default_rng(seed)
        Ps, zs = [], []
        for i in range(i_star-n_back, i_star+1):
            idx = rng.choice(Nx*Ny, 40, replace=False)
            Ps.append(np.column_stack([Xg.ravel()[idx], Yg.ravel()[idx], np.full(40, i*dt)])); zs.append(Z[i].ravel()[idx])
        P = np.vstack(Ps).astype(np.float32); z = np.concatenate(zs).astype(np.float32)
        gt_now = Z[i_star]; Hs = 4*np.nanstd(gt_now)
        target = {'gx': gx, 'gy': gy, 't': float(i_star*dt), 'gt': np.nan_to_num(gt_now), 'Hs': Hs}
        kx, ky, A, phi = fixed_components(P, z, target)
        # (1) FORECAST vs persistence
        for dl in LEADS:
            j = i_star + int(round(dl/dt))
            if j >= T: continue
            gt = Z[j]; v = ~np.isnan(gt)
            pred = render_wave_field(kx, ky, A, phi, DEPTH, gx, gy, float(j*dt))
            fc[dl].append(float(np.sqrt(np.mean((pred[v]-gt[v])**2))/Hs))
            pers[dl].append(float(np.sqrt(np.mean((gt_now[v]-gt[v])**2))/Hs))
        # (2) EXTRAPOLATION: score OUTSIDE obs hull (physics vs naive)
        hull = Delaunay(Ps[-1][:, :2])
        outside = (hull.find_simplex(np.column_stack([Xg.ravel(), Yg.ravel()])) < 0).reshape(gt_now.shape) & ~np.isnan(gt_now)
        if outside.sum() > 50:
            ws = render_wave_field(kx, ky, A, phi, DEPTH, gx, gy, float(i_star*dt))
            nv = griddata(Ps[-1][:, :2], z[-40:], (Xg, Yg), method='linear')
            nv = np.nan_to_num(nv, nan=np.nanmean(z[-40:]))
            extr_ws.append(float(np.sqrt(np.mean((ws[outside]-gt_now[outside])**2))/Hs))
            extr_nv.append(float(np.sqrt(np.mean((nv[outside]-gt_now[outside])**2))/Hs))

print("=== B) BS_2011 forecast vs persistence (fixed estimator, 4 frames x 2 seeds) ===")
print(f"{'lead':>5} {'physics':>9} {'persist':>9} {'skill':>7}  {'win%':>5}")
for dl in LEADS:
    if not fc[dl]: continue
    p = np.array(fc[dl]); q = np.array(pers[dl]); sk = 1-(p.mean()/q.mean())**2
    print(f"{dl:5.1f} {p.mean():9.3f} {q.mean():9.3f} {sk:7.2f}  {100*np.mean(p<q):4.0f}%")
print(f"\n=== EXTRAPOLATION (outside obs hull) — physics vs naive ===")
w=np.array(extr_ws); n=np.array(extr_nv)
print(f"  physics nRMSE {w.mean():.3f}+-{w.std():.3f} | naive {n.mean():.3f}+-{n.std():.3f} | physics wins {100*np.mean(w<n):.0f}%  margin {(n-w).mean():+.3f}")
