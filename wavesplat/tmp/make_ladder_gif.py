"""Estimator-ladder comparison GIF: GT | RELAX | hand-MMSE | net-MMSE on
one synthetic harness sea, track100 (U = c_p ping-pong), animated over
~2 Tp around t_star. Obs track drawn on every panel; corr (common-unobs,
at t_star) in each title. Output: results/gifs/ladder_track100_seaX.gif
(repo tmp tool, 2026-07-05 사용자 요청)"""
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
from d18_relax_estimator import omega
from d3_sa_fim_identifiability import omega_of, render_modes
from estimator_improved import iterative_relax_cv
from d3_sa_nomp import mmse_bank_estimate
from d3_prior_net_eval import get_net, net_gamma_mmse
from d3_sa_achievability_gap import (dispersion_lam, sample_obs,
                                     masks_common_unobs, hashabs, corr,
                                     N_OBS, NOISE_FRAC, TP, HS, DEPTH)

SI = int(sys.argv[1]) if len(sys.argv) > 1 else 4
COND = 'track100'
lam = dispersion_lam(TP, DEPTH)
cp = lam / TP
T = 10 * TP
fp = 0.18 * lam
r_fp = fp / 2
side = 2.0 * lam
NG = 72
gx = np.linspace(0, side, NG)
gy = np.linspace(0, side, NG)
Xg, Yg = np.meshgrid(gx, gy)
mu = masks_common_unobs(np.linspace(0, side, 96), np.linspace(0, side, 96),
                        r_fp, 0.6 * fp)
t_star = T

c = jonswap_components(Hs=HS, Tp=TP, n_components=200, depth=DEPTH,
                       seed=4000 + SI)
th = (np.array([x.kx for x in c]), np.array([x.ky for x in c]),
      np.array([x.A for x in c]), np.array([x.phi for x in c]))
om_t = omega_of(th[0], th[1], DEPTH)
rng = np.random.default_rng(9000 + 17 * SI + hashabs(COND))
P = sample_obs('moving', cp, lam, cp, T, gx, gy, r_fp, rng)
psi = (th[0][None] * P[:, 0:1] + th[1][None] * P[:, 1:2]
       - om_t[None] * P[:, 2:3] + th[3][None])
sigma = NOISE_FRAC * HS
z = (th[2][None] * np.cos(psi)).sum(1) + rng.normal(0, sigma, N_OBS)

# fit once; each estimator must render at arbitrary t
print("fitting estimators...", flush=True)
gt_grid96x = np.linspace(0, side, 96)
Xg96, Yg96 = np.meshgrid(gt_grid96x, gt_grid96x)
target = {'gx': gt_grid96x, 'gy': gt_grid96x, 't': t_star,
          'gt': render_modes(*th, DEPTH, Xg96.ravel(), Yg96.ravel(),
                             t_star).reshape(96, 96),
          'Hs': HS}

times = t_star + np.linspace(-TP, TP, 24, endpoint=False)


def q_at(t):
    return np.column_stack([Xg.ravel(), Yg.ravel(),
                            np.full(Xg.size, t)])


# RELAX: get refined comps by re-running refine to recover k/coefs is
# internal; simplest faithful render = call iterative_relax_cv per frame
# with target t (fast enough at 24 frames? no — instead approximate: fit
# once on a fine time grid via its returned field only at t_star).
# For animation fairness we re-fit RELAX ONCE and re-render via its
# internal comps using refine_params return values.
from d18_relax_estimator import beamform_top, ls_fit
from d17_recursive_estimator import refine_params
from estimator_improved import relax_cv, estimate_klim

klim = estimate_klim(P.astype(np.float32), z.astype(np.float32), DEPTH)
kxr, kyr = relax_cv(P.astype(np.float32), z.astype(np.float32), DEPTH,
                    klim, seed=SI)
for _ in range(2):
    kxr, kyr, _ = refine_params(kxr, kyr, P.astype(np.float32),
                                z.astype(np.float32), DEPTH, target,
                                iters=400)
coef_r, _, _ = ls_fit(kxr, kyr, P, z, DEPTH, lam=1e-2)


def relax_at(t):
    kk = np.hypot(kxr, kyr) + 1e-12
    omr = omega(kk, DEPTH)
    Q = q_at(t)
    ps = kxr[None] * Q[:, 0:1] + kyr[None] * Q[:, 1:2] - omr[None] * t
    N = len(kxr)
    return (np.cos(ps) @ coef_r[:N] + np.sin(ps) @ coef_r[N:]).reshape(
        NG, NG)


print("hand-MMSE / net-MMSE renders...", flush=True)
hand_frames = {}
net_frames = {}
net = get_net()
# render all frames in ONE call each by stacking query times
Qbig = np.vstack([q_at(t) for t in times])
fh = mmse_bank_estimate(P, z, DEPTH, Qbig, sigma0=sigma, seed=SI)
fn = net_gamma_mmse(net, P, z, DEPTH, Qbig, sigma, seed=SI, use_em=False)
fh = fh.reshape(len(times), NG, NG)
fn = fn.reshape(len(times), NG, NG)

# corr at t_star on common-unobs px (96-grid mask -> recompute on NG grid)
muNG = masks_common_unobs(gx, gy, r_fp, 0.6 * fp)
gt_star = render_modes(*th, DEPTH, Xg[muNG], Yg[muNG], t_star)
i_star = len(times) // 2
cors = {}
cors['RELAX'] = corr(relax_at(t_star)[muNG], gt_star)
cors['hand-MMSE'] = corr(
    mmse_bank_estimate(P, z, DEPTH, np.column_stack(
        [Xg[muNG], Yg[muNG], np.full(muNG.sum(), t_star)]),
        sigma0=sigma, seed=SI), gt_star)
cors['net-MMSE'] = corr(
    net_gamma_mmse(net, P, z, DEPTH, np.column_stack(
        [Xg[muNG], Yg[muNG], np.full(muNG.sum(), t_star)]),
        sigma, seed=SI, use_em=False), gt_star)
print("corrs:", {k: round(v, 3) for k, v in cors.items()}, flush=True)

os.makedirs('results/gifs', exist_ok=True)
out = f'results/gifs/ladder_{COND}_sea{SI}.gif'
vmax = 1.2 * HS / 2
frames = []
for i, t in enumerate(times):
    gt_f = render_modes(*th, DEPTH, Xg.ravel(), Yg.ravel(), t).reshape(
        NG, NG)
    panels = [('Ground truth', gt_f, ''),
              ('RELAX', relax_at(t), f" corr {cors['RELAX']:+.2f}"),
              ('hand-MMSE', fh[i], f" corr {cors['hand-MMSE']:+.2f}"),
              ('net-MMSE', fn[i], f" corr {cors['net-MMSE']:+.2f}")]
    fig, axs = plt.subplots(1, 4, figsize=(13.2, 3.6))
    sel_t = np.abs(P[:, 2] - t) < 2.5
    for ax, (name, F, tag) in zip(axs, panels):
        ax.imshow(F, origin='lower', extent=[0, side, 0, side],
                  vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        ax.plot(P[:, 0], P[:, 1], '.', ms=0.8, color='0.4', alpha=0.35)
        if sel_t.any():
            ax.plot(P[sel_t, 0], P[sel_t, 1], '.', ms=2.5, color='k')
        ax.set_title(name + tag, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"sea {SI} | {COND} (U=c_p ping-pong) | t = "
                 f"{t - t_star:+.1f} s vs t*  | obs n={N_OBS}, "
                 f"patch 2$\\lambda$", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(),
                        dtype=np.uint8).reshape(h, w, 4)[..., :3]
    frames.append(Image.fromarray(img))
    plt.close(fig)
frames[0].save(out, save_all=True, append_images=frames[1:],
               duration=120, loop=0)
print(f"saved {out} ({len(frames)} frames)")
