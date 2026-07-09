"""
La Jument: separate 'tuning gap' from 'info/aperture wall'.
- default RELAX (N~20)  vs  low-order RELAX (few components, aggressive prune)
- dense-GT dispersion ceiling (best dispersion fit to ALL gt points)
If low-order/dense-fit approach naive(0.16) -> 0.53 was overfitting(tuning);
if all stay high -> genuine aperture/info wall.
"""
import os, sys, numpy as np
sys.path.insert(0, '/home/vexit/wavesplat/wavesplat')
os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0, 'scripts')
from scripts.realdata_analysis import load_surface_nc
from d18_relax_estimator import relax_step, ls_fit
from d17_recursive_estimator import refine_params

F = 'data/real_bergamasco/LJ_2018/Surfaces_20180103_093938_Grid1_short.nc'
DEPTH = 40.0
Z, gx, gy, t, meta = load_surface_nc(F, tmax_frames=650)
T, Ny, Nx = Z.shape; dt = float(np.median(np.diff(t)))
i_star = T - 2; gt = Z[i_star]; Hs = 4*np.nanstd(gt)
Xg, Yg = np.meshgrid(gx, gy)
rng = np.random.default_rng(0)
Tp = 12.8; n_back = min(int(4*Tp/dt), T-2)
# sparse obs over window
Ps, zs = [], []
for i in range(i_star-n_back, i_star+1):
    fr = Z[i]; idx = rng.choice(Nx*Ny, 30, replace=False)
    Ps.append(np.column_stack([Xg.ravel()[idx], Yg.ravel()[idx], np.full(30, i*dt)]))
    zs.append(fr.ravel()[idx])
P = np.vstack(Ps).astype(np.float32); z = np.concatenate(zs).astype(np.float32)
target = {'gx': gx, 'gy': gy, 't': float(i_star*dt), 'gt': np.nan_to_num(gt), 'Hs': Hs}

def score(fld):
    v = ~np.isnan(gt)
    return float(np.sqrt(np.mean((fld[v]-gt[v])**2))/Hs), float(np.corrcoef(fld[v], gt[v])[0,1])

# (1) low-order RELAX: aggressive prune, few comps
kx, ky = relax_step(np.zeros(0), np.zeros(0), P, z, DEPTH, max_comp=4, prune_frac=0.30)
kx, ky, fld_lo = refine_params(kx, ky, P, z, DEPTH, target, iters=600)
nr_lo, cr_lo = score(fld_lo)
print(f"[low-order RELAX]  N={len(kx)}  nRMSE {nr_lo:.3f}  corr {cr_lo:.3f}")

# (2) single dominant component only
kx1, ky1 = relax_step(np.zeros(0), np.zeros(0), P, z, DEPTH, max_comp=1, prune_frac=0.0)
kx1, ky1, fld1 = refine_params(kx1, ky1, P, z, DEPTH, target, iters=800)
nr1, cr1 = score(fld1)
print(f"[single comp]      N={len(kx1)}  nRMSE {nr1:.3f}  corr {cr1:.3f}")

# (3) dense-GT dispersion ceiling: fit dispersion comps to MANY dense GT points
vy, vx = np.where(~np.isnan(gt))
sel = rng.choice(len(vx), 2500, replace=False)
Pd = np.column_stack([gx[vx[sel]], gy[vy[sel]], np.full(len(sel), i_star*dt)]).astype(np.float32)
zd = gt[vy[sel], vx[sel]].astype(np.float32)
kxd, kyd = relax_step(np.zeros(0), np.zeros(0), Pd, zd, DEPTH, max_comp=12, prune_frac=0.05)
kxd, kyd, fld_d = refine_params(kxd, kyd, Pd, zd, DEPTH, target, iters=600)
nr_d, cr_d = score(fld_d)
print(f"[dense-GT ceiling] N={len(kxd)}  nRMSE {nr_d:.3f}  corr {cr_d:.3f}  (best dispersion fit to dense GT)")

print(f"\nreference: default RELAX 0.530/0.22 ; naive 0.160/0.77")
print("interpret: if dense-ceiling & low-order approach naive -> 0.53 was overfitting(tuning);")
print("           if all high -> genuine aperture wall (field not identifiable from 40m/256m patch).")
