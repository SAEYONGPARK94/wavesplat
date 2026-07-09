"""
La Jument FORECAST test: physics(dispersion) vs persistence vs naive-frozen.
nowcast에선 naive가 이기지만(smooth field), naive는 FORECAST를 못 함(시간모델 없음).
물리모델(dispersion 전파)이 실측 swell을 예보에서 persistence를 이기는가?
"""
import os, sys, numpy as np
sys.path.insert(0, '/home/vexit/wavesplat/wavesplat')
os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0, 'scripts')
from scripts.realdata_analysis import load_surface_nc
from d18_relax_estimator import relax_step
from d17_recursive_estimator import refine_params
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from scipy.interpolate import griddata

F = 'data/real_bergamasco/LJ_2018/Surfaces_20180103_093938_Grid1_short.nc'
DEPTH = 40.0
Z, gx, gy, t, meta = load_surface_nc(F, tmax_frames=900)
T, Ny, Nx = Z.shape; dt = float(np.median(np.diff(t)))
Xg, Yg = np.meshgrid(gx, gy); rng = np.random.default_rng(0)
Tp = 12.8; n_back = int(4*Tp/dt)
i_star = n_back + 5                          # leave room to forecast forward
gt_now = Z[i_star]; Hs = 4*np.nanstd(gt_now)

# past-window sparse obs (only <= i_star) -> fit low-order dispersion components
Ps, zs = [], []
for i in range(i_star-n_back, i_star+1):
    idx = rng.choice(Nx*Ny, 30, replace=False)
    Ps.append(np.column_stack([Xg.ravel()[idx], Yg.ravel()[idx], np.full(30, i*dt)]))
    zs.append(Z[i].ravel()[idx])
P = np.vstack(Ps).astype(np.float32); z = np.concatenate(zs).astype(np.float32)
target = {'gx': gx, 'gy': gy, 't': float(i_star*dt), 'gt': np.nan_to_num(gt_now), 'Hs': Hs}
kx, ky = relax_step(np.zeros(0), np.zeros(0), P, z, DEPTH, max_comp=4, prune_frac=0.30)
kx, ky, _ = refine_params(kx, ky, P, z, DEPTH, target, iters=600)
obs = np.column_stack([P[:,0],P[:,1],P[:,2],z]).astype(np.float32)
A, phi = fit_amplitudes_phases(kx, ky, obs, DEPTH, lambda_reg=0.5)
# naive-frozen nowcast field (last frame interp) for a naive 'forecast' = frozen
naive_now = griddata(Ps[-1][:,:2], zs[-1], (Xg,Yg), method='linear')
naive_now = np.where(np.isnan(naive_now), np.nanmean(zs[-1]), naive_now)

def nr(f, g):
    v = ~np.isnan(g); return float(np.sqrt(np.mean((f[v]-g[v])**2))/Hs)

print(f"La Jument FORECAST (N_comp={len(kx)}, Tp={Tp}s, Hs={Hs:.2f}m)")
print(f"{'lead':>5} {'physics':>8} {'persist':>8} {'naive-froz':>10}  {'skill_vs_pers':>13}")
for dl in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
    j = i_star + int(round(dl/dt))
    if j >= T: break
    gt = Z[j]
    pred = render_wave_field(kx, ky, A, phi, DEPTH, gx, gy, float(j*dt))
    nr_phys = nr(pred, gt); nr_pers = nr(gt_now, gt); nr_naive = nr(naive_now, gt)
    skill = 1 - (nr_phys/nr_pers)**2 if nr_pers>0 else 0
    print(f"{dl:5.1f} {nr_phys:8.3f} {nr_pers:8.3f} {nr_naive:10.3f}  {skill:13.2f}")
print("skill>0 => physics forecast beats persistence (naive/interp는 froze만 가능 = persist와 동급).")
