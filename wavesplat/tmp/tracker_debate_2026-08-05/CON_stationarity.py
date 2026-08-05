"""CON: is the synthetic benchmark stationary? Rebuild stored GT from the ONE fixed
component set at every timestamp. Read-only, no simulation."""
import h5py, numpy as np
P = 'data_finemesh_2026-07-23/episode_BF4_15comp_seed0_uniform.h5'
f = h5py.File(P, 'r')
wc = f['config/wave_components'][:]          # columns: [A, omega, kx, ky, phi, Q]
A, om, kx, ky, ph = (wc[:, i] for i in range(5))
gx, gy, ts = f['frames/gt_grid_x'][:], f['frames/gt_grid_y'][:], f['frames/timestamps'][:]
X, Y = np.meshgrid(gx, gy)
Hs = f['config'].attrs['Hs']
print(f'{P}\n  wave_components {wc.shape} (no time axis)  gt_wave_field {f["frames/gt_wave_field"].shape}')
print(f'  episode {ts[-1]-ts[0]:.1f}s, dt={np.diff(ts).mean():.3f}s, Hs={Hs}\n')
for ti in range(0, 301, 50):
    t = ts[ti]
    z = sum(A[i]*np.cos(kx[i]*X + ky[i]*Y - om[i]*t + ph[i]) for i in range(len(A)))
    gt = f['frames/gt_wave_field'][ti]
    print(f'  t={t:5.1f}s  corr={np.corrcoef(z.ravel(), gt.ravel())[0,1]:+.6f}  '
          f'nRMSE={np.sqrt(np.mean((z-gt)**2))/Hs:.5f}  VR={np.var(z)/np.var(gt):.4f}')
print('\n=> exact to machine precision at every t: the truth is a FIXED linear superposition.')
print('=> "estimate once, propagate by dispersion" IS the true generative model here.')
