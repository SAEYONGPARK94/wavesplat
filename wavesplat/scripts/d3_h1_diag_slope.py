"""H1 diagnostic: WHY is the naive in-band slope skill only rho ~ 0.33?

One sea, no EP. Prints, for the naive posterior on the informed annulus:
  - std(fs) / std(gs_ib)      amplitude miscalibration of the slope estimate
  - rho(fs, gs_ib), rho(fs, gs_tot)
  - posterior slope sd (from A0) / prior slope sd / true in-band slope sd
  - the same three for ELEVATION as the healthy reference
  - rho by k-band thirds of the in-band slope (which scales carry the failure)
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

import d3_g0b_selection_aware as g0b
from d3_g0b_selection_aware import DEV, DT, BAND
from d3_gpu_mmse import mmse_gpu, build_bank, design
from d3_s2_mask_information import grad_design
from d3_door1b_field_bias import eta_dr, R_MIN, NOISE, DEPTH, FOV_H
from d3_h1_mask_destination import sea_dir, eta_xt, var_of

g0b.sea = sea_dir(180.0)
rng = np.random.default_rng(21)
seed = 0
S, kp, amps, missP, miss_r, Ts = g0b.gen_scene(seed, 3.33, rng)
P, zval = amps[:, :3], amps[:, 3] + rng.normal(0, NOISE, len(amps))
ret_r = amps[:, 4]
r90 = float(np.quantile(ret_r, 0.90))

nq = 400
rq = np.sqrt(rng.uniform((R_MIN / r90) ** 2, 1.0, nq)) * r90
aq = rng.uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), nq)
Qi = np.column_stack([rq * np.cos(aq), rq * np.sin(aq), np.zeros(nq)])
gt_i = eta_xt(Qi[:, 0], Qi[:, 1], Qi[:, 2], S)
gs_i = eta_dr(Qi[:, 0], Qi[:, 1], 0.0, S)
kt = np.hypot(S[0], S[1])
ib = (kt >= BAND[0] * kp) & (kt <= BAND[1] * kp)
S_ib = tuple(c[ib] for c in S)
gs_ib = eta_dr(Qi[:, 0], Qi[:, 1], 0.0, S_ib)

tP = torch.tensor(P, device=DEV, dtype=torch.float32)
tz = torch.tensor(zval, device=DEV, dtype=torch.float32)
tQ = torch.tensor(Qi, device=DEV, dtype=torch.float32)
_, mdl = mmse_gpu(tP, tz, DEPTH, tQ, NOISE, kp, seed=seed, return_model=True)
gam, s2 = mdl['gam'].to(DT), float(mdl['s2'])

kxb, kyb, om = build_bank(kp, BAND, kp / 10.0, DEPTH, DEV, torch.float32)
kxb, kyb, om = kxb.to(DT), kyb.to(DT), om.to(DT)
Phi = design(torch.tensor(P, device=DEV, dtype=DT), kxb, kyb, om)
Ci = design(torch.tensor(Qi, device=DEV, dtype=DT), kxb, kyb, om)
Cs = grad_design(torch.tensor(Qi, device=DEV, dtype=DT), kxb, kyb, om)
tz64 = torch.tensor(zval, device=DEV, dtype=DT)

A0 = Phi.T @ Phi / s2 + torch.diag(1.0 / gam)
a = torch.linalg.solve(A0, Phi.T @ tz64 / s2)

fi = (Ci @ a).cpu().numpy()
fs = (Cs @ a).cpu().numpy()
v_eta = var_of(A0, Ci).cpu().numpy()
v_slp = var_of(A0, Cs).cpu().numpy()
pr_eta = float(torch.sqrt(((Ci ** 2) * gam[None]).sum(1).mean()))
pr_slp = float(torch.sqrt(((Cs ** 2) * gam[None]).sum(1).mean()))


def rho(p, g):
    return float(np.corrcoef(p, g)[0, 1])


print(f"kp {kp:.4f}  r90 {r90:.1f}  n_obs {len(P)}")
print(f"ELEV : std(fi)/std(gt) {np.std(fi)/np.std(gt_i):.3f}  "
      f"rho {rho(fi, gt_i):+.3f}  post_sd/true_sd "
      f"{np.sqrt(v_eta.mean())/np.std(gt_i):.3f}  "
      f"prior_sd/true_sd {pr_eta/np.std(gt_i):.3f}")
print(f"SLOPE: std(fs)/std(gs_ib) {np.std(fs)/np.std(gs_ib):.3f}  "
      f"rho_ib {rho(fs, gs_ib):+.3f}  rho_tot {rho(fs, gs_i):+.3f}  "
      f"post_sd/true_ib_sd {np.sqrt(v_slp.mean())/np.std(gs_ib):.3f}  "
      f"prior_sd/true_ib_sd {pr_slp/np.std(gs_ib):.3f}")

# oracle shrink floors
b_or = np.dot(fs, gs_ib) / np.dot(fs, fs)
print(f"SLOPE oracle-shrunk nRMSE vs ib: "
      f"{np.sqrt(np.mean((b_or*fs - gs_ib)**2))/np.std(gs_ib):.3f} "
      f"(= sqrt(1-rho^2) {np.sqrt(1-rho(fs,gs_ib)**2):.3f}), beta_or {b_or:.3f}")

# which scales fail: split the in-band GT slope into k-thirds
edges = np.quantile(kt[ib], [0, 1/3, 2/3, 1.0])
for j in range(3):
    m = ib & (kt >= edges[j]) & (kt <= edges[j+1])
    gsj = eta_dr(Qi[:, 0], Qi[:, 1], 0.0, tuple(c[m] for c in S))
    print(f"  k-third {j} [{edges[j]/kp:.2f},{edges[j+1]/kp:.2f}]kp "
          f"share {np.var(gsj)/np.var(gs_ib):.2f}  rho(fs, gs_j) "
          f"{rho(fs, gsj):+.3f}  rho(gt_slice, fs after removing others) ")

# and the elevation-mode sanity: slope of the *elevation* GT in-band vs fs
# via finite difference of fi along r-hat (bank-free check of grad_design)
h = 0.5
Qp = Qi.copy(); Qp[:, 0] += h * np.cos(aq); Qp[:, 1] += h * np.sin(aq)
Qm = Qi.copy(); Qm[:, 0] -= h * np.cos(aq); Qm[:, 1] -= h * np.sin(aq)
Cp = design(torch.tensor(Qp, device=DEV, dtype=DT), kxb, kyb, om)
Cm = design(torch.tensor(Qm, device=DEV, dtype=DT), kxb, kyb, om)
fd = ((Cp @ a - Cm @ a) / (2 * h)).cpu().numpy()
print(f"grad_design vs finite-diff of posterior: rho {rho(fs, fd):+.5f} "
      f"std ratio {np.std(fs)/np.std(fd):.3f}")
gs_fd = (eta_xt(Qp[:, 0], Qp[:, 1], Qp[:, 2], S_ib)
         - eta_xt(Qm[:, 0], Qm[:, 1], Qm[:, 2], S_ib)) / (2 * h)
print(f"gs_ib GT vs finite-diff GT:            rho {rho(gs_ib, gs_fd):+.5f}")
