"""
BS_2011 (short-crested wind-sea) diagnostic — completes the (directionality x aperture) picture
and tests methods-review #1 (klim too low) + #2 (model order) on real short waves.
"""
import os, sys, numpy as np
sys.path.insert(0, '/home/vexit/wavesplat/wavesplat')
os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0, 'scripts')
from scripts.realdata_analysis import load_surface_nc, directional_spread
from d18_relax_estimator import relax_step
from d17_recursive_estimator import refine_params
from scipy.interpolate import griddata

F = 'data/real_bergamasco/BS_2011/Surfaces_20111004_113800_short.nc'
DEPTH = 30.0
Z, gx, gy, t, meta = load_surface_nc(F, tmax_frames=400)
T, Ny, Nx = Z.shape; dt = float(np.median(np.diff(t)))
dx, dy = meta['dx'], meta['dy']; Xg, Yg = np.meshgrid(gx, gy); rng = np.random.default_rng(0)

# directional spread averaged over several frames (resolve 20 vs 109 discrepancy)
sprs = [directional_spread(Z[k], dx, dy)[0] for k in range(0, T, T//8)]
kp = directional_spread(Z[0], dx, dy)[3]
print(f"BS_2011: patch {gx.max()-gx.min():.1f}x{gy.max()-gy.min():.1f}m  Hs={4*np.nanstd(Z[0]):.2f}m  "
      f"dir_spread={np.mean(sprs):.0f}+-{np.std(sprs):.0f} deg  kp={kp:.2f} (spectrum~{3*kp:.2f})")

Tp = 2.4; n_back = min(int(4*Tp/dt), T-2); i_star = T-2
gt = Z[i_star]; Hs = 4*np.nanstd(gt)
Ps, zs = [], []
for i in range(i_star-n_back, i_star+1):
    idx = rng.choice(Nx*Ny, 40, replace=False)
    Ps.append(np.column_stack([Xg.ravel()[idx], Yg.ravel()[idx], np.full(40, i*dt)])); zs.append(Z[i].ravel()[idx])
P = np.vstack(Ps).astype(np.float32); z = np.concatenate(zs).astype(np.float32)
target = {'gx': gx, 'gy': gy, 't': float(i_star*dt), 'gt': np.nan_to_num(gt), 'Hs': Hs}

def score(f):
    v = ~np.isnan(gt); return float(np.sqrt(np.mean((f[v]-gt[v])**2))/Hs), float(np.corrcoef(f[v], gt[v])[0,1])

def run(klim, max_comp, prune, iters, label):
    kx, ky = np.zeros(0), np.zeros(0); fld=None
    for _ in range(3):
        kx, ky = relax_step(kx, ky, P, z, DEPTH, max_comp=max_comp, prune_frac=prune, klim=klim)
        kx, ky, fld = refine_params(kx, ky, P, z, DEPTH, target, iters=iters)
    nr, cr = score(fld); print(f"  {label:34s} N={len(kx):2d}  nRMSE {nr:.3f}  corr {cr:.3f}"); return nr,cr

print(f"  --- nowcast ({len(z)} obs, {n_back+1} frames) ---")
run(0.6, 32, 0.05, 200, "default RELAX (klim=0.6)")
run(1.5, 32, 0.05, 200, "klim-fix (klim=1.5)")
run(1.5, 8, 0.12, 400, "klim-fix + model-order control")
# dense-GT ceiling with klim-fix
vy, vx = np.where(~np.isnan(gt)); sel = rng.choice(len(vx), 2500, replace=False)
Pd = np.column_stack([gx[vx[sel]], gy[vy[sel]], np.full(len(sel), i_star*dt)]).astype(np.float32)
zd = gt[vy[sel], vx[sel]].astype(np.float32)
kxd, kyd = relax_step(np.zeros(0), np.zeros(0), Pd, zd, DEPTH, max_comp=16, prune_frac=0.05, klim=1.5)
kxd, kyd, fld_d = refine_params(kxd, kyd, Pd, zd, DEPTH, target, iters=600)
print(f"  {'dense-GT ceiling (klim=1.5)':34s} N={len(kxd):2d}  nRMSE {score(fld_d)[0]:.3f}  corr {score(fld_d)[1]:.3f}")
# naive
naive = griddata(Ps[-1][:,:2], zs[-1], (Xg,Yg), method='linear'); naive=np.where(np.isnan(naive), np.nanmean(zs[-1]), naive)
print(f"  {'naive interp':34s}      nRMSE {score(naive)[0]:.3f}  corr {score(naive)[1]:.3f}")
