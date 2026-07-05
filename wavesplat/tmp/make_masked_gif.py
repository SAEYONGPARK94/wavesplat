"""Confidence-masked, amplitude-calibrated forecast GIF (sea4, track100).
Panels: GT | calibrated net-MMSE shown only where posterior std is low
(alpha-fade elsewhere) | pointwise z=|err|/sigma_post. Amplitude
calibration per pixel: eta_cal = eta_hat * sqrt(v_prior/(v_prior-v_post))
capped at 2 (undo conditional-mean shrinkage for DISPLAY; phase untouched).
(repo tmp tool)"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from d18_relax_estimator import beamform_top, omega
from d3_sa_fim_identifiability import omega_of, render_modes
from d3_sa_nomp import beamform_power, split_masks
from d3_prior_net_eval import get_net, DEVICE
from d3_prior_net_train import NGRID, KREL
from d3_sa_achievability_gap import (dispersion_lam, sample_obs, hashabs,
                                     N_OBS, NOISE_FRAC, TP, HS, DEPTH)

SI = 4
lam = dispersion_lam(TP, DEPTH)
cp = lam / TP
cg = 0.5 * cp
T = 10 * TP
fp = 0.18 * lam
r_fp = fp / 2
side = 2.0 * lam
NG = 72
gx = np.linspace(0, side, NG)
gy = np.linspace(0, side, NG)
Xg, Yg = np.meshgrid(gx, gy)
t_star = T
net = get_net()

c = jonswap_components(Hs=HS, Tp=TP, n_components=200, depth=DEPTH,
                       seed=4000 + SI)
th = (np.array([x.kx for x in c]), np.array([x.ky for x in c]),
      np.array([x.A for x in c]), np.array([x.phi for x in c]))
om_t = omega_of(th[0], th[1], DEPTH)
rng = np.random.default_rng(9000 + 17 * SI + hashabs('track100'))
P = sample_obs('moving', cp, lam, cp, T, gx, gy, r_fp, rng)
psi = (th[0][None] * P[:, 0:1] + th[1][None] * P[:, 1:2]
       - om_t[None] * P[:, 2:3] + th[3][None])
sigma = NOISE_FRAC * HS
z = (th[2][None] * np.cos(psi)).sum(1) + rng.normal(0, sigma, N_OBS)

# --- net-gamma MMSE with posterior variance (mirrors net_gamma_mmse) ---
n = len(z)
rngv = np.random.default_rng(SI)
vi = rngv.choice(n, max(8, n // 4), replace=False)
vmask = np.zeros(n, bool)
vmask[vi] = True
Ptr, ztr = P[~vmask], z[~vmask]
Pv, zv = P[vmask], z[vmask]
bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32), DEPTH,
                      klim=2.0, n=120)
kp = max(np.hypot(bx, by), 1e-3)
bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32), DEPTH,
                      klim=3.5 * kp, n=120)
kp = max(np.hypot(bx, by), 1e-3)
m1, m2 = split_masks(P, mode='legs')
_, _, p1 = beamform_power(P[m1], z[m1], DEPTH, KREL * kp, NGRID)
_, _, p2 = beamform_power(P[m2], z[m2], DEPTH, KREL * kp, NGRID)
p1 = (p1 / max(p1.max(), 1e-12)).reshape(NGRID, NGRID)
p2 = (p2 / max(p2.max(), 1e-12)).reshape(NGRID, NGRID)
pm = np.minimum(p1, p2)
X = np.stack([p1, p2, pm,
              np.log10(p1 + 1e-6) / 6 + 1, np.log10(p2 + 1e-6) / 6 + 1,
              np.log10(pm + 1e-6) / 6 + 1]).astype(np.float32)
with torch.no_grad():
    Ylog = net(torch.tensor(X[None], device=DEVICE))[0].cpu().numpy()
ax_ = np.linspace(-KREL * kp, KREL * kp, NGRID)
KX, KY = np.meshgrid(ax_, ax_)
kk = np.hypot(KX, KY)
sel = (kk > 0.35 * kp) & (kk < KREL * kp)
kxb, kyb = KX[sel].astype(float), KY[sel].astype(float)
om_b = omega(np.hypot(kxb, kyb) + 1e-12, DEPTH)
g = np.maximum(10.0 ** Ylog[sel], 1e-8)
g = g / g.sum()
gam = np.concatenate([g, g]) * np.var(z)


def design(Q):
    ps = kxb[None] * Q[:, 0:1] + kyb[None] * Q[:, 1:2] \
        - om_b[None] * Q[:, 2:3]
    return np.concatenate([np.cos(ps), np.sin(ps)], axis=1)


Phi_tr, Phi_v = design(Ptr), design(Pv)
best = None
for smult in (1.0, 3.0, 10.0):
    s2 = (smult * sigma) ** 2
    K = (Phi_tr * gam[None]) @ Phi_tr.T + s2 * np.eye(len(ztr))
    fv = (Phi_v * gam[None]) @ (Phi_tr.T @ np.linalg.solve(K, ztr))
    vr = float(np.sqrt(np.mean((fv - zv) ** 2)))
    if best is None or vr < best[0]:
        best = (vr, smult)
s2 = (best[1] * sigma) ** 2
Phi = design(P)
K = (Phi * gam[None]) @ Phi.T + s2 * np.eye(n)
Ki = np.linalg.inv(K)
w_pred = Phi.T @ (Ki @ z)

dts = np.arange(0, 33, 2.0)
os.makedirs('results/gifs', exist_ok=True)
out = 'results/gifs/masked_forecast_sea4.gif'
frames = []
for dt in dts:
    te = t_star + dt
    Q = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, te)])
    Dq = design(Q)
    pred = (Dq * gam[None]) @ w_pred
    v_pri = (Dq ** 2 * gam[None]).sum(1)
    A = (Dq * gam[None]) @ Phi.T
    v_red = np.einsum('ij,jk,ik->i', A, Ki, A)
    v_post = np.maximum(v_pri - v_red, 1e-12)
    frac_known = np.clip(v_red / v_pri, 0, 1)          # explained variance
    cal = np.clip(np.sqrt(v_pri / np.maximum(v_red, 1e-12)), 1.0, 2.0)
    pred_cal = pred * cal
    gt_f = render_modes(*th, DEPTH, Xg.ravel(), Yg.ravel(), te)
    zsc = np.abs(pred - gt_f) / np.sqrt(v_post + s2)
    G_ = gt_f.reshape(NG, NG)
    Pc = pred_cal.reshape(NG, NG)
    Fk = frac_known.reshape(NG, NG)
    Zs = zsc.reshape(NG, NG)
    fig, axs = plt.subplots(1, 3, figsize=(11.2, 3.9))
    vmax = HS * 0.6
    axs[0].imshow(G_, origin='lower', extent=[0, side, 0, side],
                  vmin=-vmax, vmax=vmax, cmap='RdBu_r')
    axs[0].set_title('Ground truth', fontsize=10)
    # confidence-masked calibrated forecast: alpha = frac_known
    rgba = plt.cm.RdBu_r((np.clip(Pc, -vmax, vmax) + vmax) / (2 * vmax))
    rgba[..., 3] = np.clip(Fk / 0.5, 0, 1)
    axs[1].imshow(np.ones_like(G_), origin='lower',
                  extent=[0, side, 0, side], cmap='gray', vmin=0, vmax=2)
    axs[1].imshow(rgba, origin='lower', extent=[0, side, 0, side])
    axs[1].set_title('forecast (amp-calibrated,\nshown where known)',
                     fontsize=9)
    im = axs[2].imshow(Fk, origin='lower', extent=[0, side, 0, side],
                       vmin=0, vmax=1, cmap='viridis')
    axs[2].set_title('known fraction (posterior)', fontsize=10)
    xf = np.clip(cg * dt, 0, side)
    for ax in axs:
        ax.plot(P[:, 0], P[:, 1], '.', ms=0.7, color='0.45', alpha=0.35)
        ax.set_xticks([])
        ax.set_yticks([])
    axs[2].axvline(min(side - 1, xf), color='w', lw=1.2, ls='--')
    fig.suptitle(f"confidence-masked forecast | lead dt = {dt:.0f} s "
                 f"(c_g dt = {cg * dt:.0f} m)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(),
                        dtype=np.uint8).reshape(h, w, 4)[..., :3]
    frames.append(Image.fromarray(img))
    plt.close(fig)
frames[0].save(out, save_all=True, append_images=frames[1:],
               duration=350, loop=0)
print(f"saved {out}")
