"""Predictable-zone advection GIF: |error| map of the net-MMSE forecast
vs lead time dt (0..32 s), sea 4 track100. The low-error region should
slide downstream at c_g — the visual proof that 'now-wrong far zone'
and 'later-right far zone' are different wave groups. (repo tmp tool)"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from d3_sa_fim_identifiability import omega_of, render_modes
from d3_prior_net_eval import get_net, net_gamma_mmse
from d3_sa_achievability_gap import (dispersion_lam, sample_obs, hashabs,
                                     N_OBS, NOISE_FRAC, TP, HS, DEPTH, G)

SI = 4
lam = dispersion_lam(TP, DEPTH)
cp = lam / TP
cg = 0.5 * cp                      # deep-water group velocity at peak
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

dts = np.arange(0, 33, 2.0)
Qbig = np.vstack([np.column_stack([Xg.ravel(), Yg.ravel(),
                                   np.full(Xg.size, t_star + dt)])
                  for dt in dts])
print("rendering net-MMSE at all lead times...", flush=True)
fn = net_gamma_mmse(net, P, z, DEPTH, Qbig, sigma, seed=SI,
                    use_em=False).reshape(len(dts), NG, NG)

os.makedirs('results/gifs', exist_ok=True)
out = 'results/gifs/advect_error_sea4.gif'
frames = []
sm_k = np.ones((5, 5)) / 25.0
from scipy.ndimage import uniform_filter
for i, dt in enumerate(dts):
    gt_f = render_modes(*th, DEPTH, Xg.ravel(), Yg.ravel(),
                        t_star + dt).reshape(NG, NG)
    err = uniform_filter(np.abs(fn[i] - gt_f), size=7) / (HS / 4)
    fig, axs = plt.subplots(1, 3, figsize=(11.2, 3.9))
    for ax, F, ttl, kw in (
            (axs[0], gt_f, 'Ground truth', dict(vmin=-HS * .6, vmax=HS * .6,
                                                cmap='RdBu_r')),
            (axs[1], fn[i], 'net-MMSE forecast', dict(vmin=-HS * .6,
                                                      vmax=HS * .6,
                                                      cmap='RdBu_r')),
            (axs[2], err, '|error| / (Hs/4)  (smoothed)',
             dict(vmin=0, vmax=1.6, cmap='magma_r'))):
        ax.imshow(F, origin='lower', extent=[0, side, 0, side], **kw)
        ax.plot(P[:, 0], P[:, 1], '.', ms=0.8, color='0.5', alpha=0.4)
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    # advection front marker: x = corridor + cg*dt
    xf = np.clip(cg * dt, 0, side)
    axs[2].axvline(min(side - 1, xf), color='cyan', lw=1.5, ls='--')
    axs[2].text(0.02, 0.965, f"c_g x dt = {cg * dt:.0f} m",
                transform=axs[2].transAxes, color='cyan', fontsize=9,
                va='top')
    fig.suptitle(f"lead time dt = {dt:.0f} s   (waves propagate ->, "
                 f"c_g = {cg:.1f} m/s)", fontsize=11)
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
