"""H6 -- end-to-end check: v2 fixedT EPISODE FILES -> H5 recipe -> rho.

Everything so far ran on script-side scenes. This closes the loop on the
actual benchmark artifact: load episode_*_fixedT.h5 (pipeline-generated,
301 frames x 8000 pts, T recorded in attrs), apply the H5 recipe
(range-stratified 2000 obs from a 12-frame/20 s window + misspecification
noise floor: no_em, pexp 1, smult 10), score against the GT stored in the
file itself (no phase-convention risk).

Scored per episode:
  ann   elevation rho on the stored GT grid, r in [15, 75] m, at window end
  fan   same, full grid r in [15, 200] m
  enc   rho of eta(x0,y0,t) over the 5 s AFTER the window (t_end..t_end+5),
        at the grid point nearest the sensor. CAVEAT: pipeline episodes use
        direction_deg = 0 (waves travel +x, AWAY from the sensor) -- the
        anti-causal configuration for encounter (07-05). A --direction flag
        for head-sea v2 episodes is a logged spec item; enc here validates
        machinery, not operational skill.
"""
import glob
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_gpu_mmse import mmse_gpu, build_bank, design
from d3_g0b_selection_aware import DEV, DT, BAND

rng = np.random.default_rng(11)
rows = []
for path in sorted(glob.glob('data/episode_*_fixedT.h5')):
    with h5py.File(path, 'r') as f:
        comp = f['config/wave_components'][:]          # [A, om, kx, ky, phi, Q]
        ts = f['frames/timestamps'][:]
        pts = f['frames/lidar_points']
        gx = f['frames/gt_grid_x'][:]
        gy = f['frames/gt_grid_y'][:]
        gz = f['frames/gt_wave_field']

        kp = float(np.hypot(comp[:, 2], comp[:, 3])[np.argmax(comp[:, 0])])
        obs_i = list(range(50, 249, 18))               # 12 frames, ~20 s
        t_end = ts[obs_i[-1]]
        P, zv = [], []
        for i in obs_i:
            p = pts[i]
            P.append(np.column_stack([p[:, 0], p[:, 1],
                                      np.full(len(p), ts[i])]))
            zv.append(p[:, 2])
        P, zv = np.vstack(P), np.concatenate(zv)
        rr = np.hypot(P[:, 0], P[:, 1])

        # recipe: range-stratified 2000
        edges = np.linspace(rr.min(), rr.max() + 1e-6, 13)
        b = np.clip(np.digitize(rr, edges) - 1, 0, 11)
        take = []
        for j in range(12):
            idx = np.where(b == j)[0]
            k = min(len(idx), 167)
            if k:
                take.append(rng.choice(idx, k, replace=False))
        sel = np.concatenate(take)
        Ps, zs = P[sel], zv[sel]

        t32 = torch.tensor(Ps, device=DEV, dtype=torch.float32)
        z32 = torch.tensor(zs, device=DEV, dtype=torch.float32)
        _, mdl = mmse_gpu(t32, z32, 50.0, t32[:64], 0.05, kp, seed=0,
                          return_model=True, no_em=True, pexp=1.0, smult=10.0)
        gam, s2 = mdl['gam'].to(DT), float(mdl['s2'])
        kxb, kyb, om = build_bank(kp, BAND, kp / 10.0, 50.0, DEV,
                                  torch.float32)
        kxb, kyb, om = kxb.to(DT), kyb.to(DT), om.to(DT)
        Phi = design(torch.tensor(Ps, device=DEV, dtype=DT), kxb, kyb, om)
        A0 = Phi.T @ Phi / s2 + torch.diag(1.0 / gam)
        a = torch.linalg.solve(A0, Phi.T @ torch.tensor(
            zs, device=DEV, dtype=DT) / s2)

        X, Y = np.meshgrid(gx, gy)
        R = np.hypot(X, Y)
        iend = obs_i[-1]
        gt_end = gz[iend]

        def rho_region(rmin, rmax):
            m = (R >= rmin) & (R <= rmax)
            Q = np.column_stack([X[m], Y[m], np.full(m.sum(), t_end)])
            C = design(torch.tensor(Q, device=DEV, dtype=DT), kxb, kyb, om)
            fp = (C @ a).cpu().numpy()
            return float(np.corrcoef(fp, gt_end[m])[0, 1])

        ann = rho_region(15.0, 75.0)
        fan = rho_region(15.0, float(R.max()))

        iy, ix = np.unravel_index(np.argmin(R), R.shape)
        fut = [i for i in range(iend + 1, len(ts)) if ts[i] <= t_end + 5.0]
        Qe = np.column_stack([np.full(len(fut), X[iy, ix]),
                              np.full(len(fut), Y[iy, ix]),
                              ts[fut]])
        Ce = design(torch.tensor(Qe, device=DEV, dtype=DT), kxb, kyb, om)
        fe = (Ce @ a).cpu().numpy()
        ge = np.array([gz[i][iy, ix] for i in fut])
        enc = float(np.corrcoef(fe, ge)[0, 1])

        rows.append({'file': os.path.basename(path), 'n_win': len(sel),
                     'ann': ann, 'fan': fan, 'enc': enc,
                     'r0': float(R[iy, ix])})
        print(f"{os.path.basename(path):40s} ann {ann:+.2f} fan {fan:+.2f} "
              f"enc(dir0, r0={R[iy,ix]:.0f}m) {enc:+.2f}", flush=True)

import json
json.dump(rows, open('results/direction3/h6_v2_e2e.json', 'w'), indent=2)
print("wrote results/direction3/h6_v2_e2e.json")
