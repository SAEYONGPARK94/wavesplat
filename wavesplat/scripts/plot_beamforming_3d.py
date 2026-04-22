"""
plot_beamforming_3d.py
Dispersion-aware beamforming power map visualization.

Produces:
  1. beamforming_powermap_3d.png    -- raw P(kx,ky) 3D surface (4 episodes)
  2. beamforming_3stage.png         -- 3-column: Raw | BF Init | Coarse-to-fine Opt
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wavesim.data_writer import load_episode

# ── constants ─────────────────────────────────────────────
G = 9.81
DEPTH         = 50.0
N_INIT_FRAMES = 10
KXY_RANGE     = 4.0    # ±4.0 rad/m  (matches initializer.py)
N_COARSE      = 81     # grid points per axis
MIN_DK        = 0.15   # minimum peak separation [rad/m]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
os.makedirs(OUT_DIR, exist_ok=True)

EPISODES = [
    os.path.join(DATA_DIR, "episode_BF2_1comp_seed0_uniform.h5"),
    os.path.join(DATA_DIR, "episode_BF3_5comp_seed0_uniform.h5"),
    os.path.join(DATA_DIR, "episode_BF4_5comp_seed0_uniform.h5"),
    os.path.join(DATA_DIR, "episode_BF4_15comp_seed0_slope.h5"),
]
TITLES = [
    "BF2 · 1 comp",
    "BF3 · 5 comps",
    "BF4 · 5 comps",
    "BF4 · 15 comps",
]

# ── beamforming helpers ────────────────────────────────────

def beamform_cartesian(x, y, z, t, depth, krange=KXY_RANGE, n=N_COARSE):
    kx_ax = np.linspace(-krange, krange, n)
    ky_ax = np.linspace(-krange, krange, n)
    z_c   = z - z.mean()
    power = np.zeros((n, n))
    for iy, ky in enumerate(ky_ax):
        kx_arr = kx_ax[:, None]
        k_mag  = np.sqrt(kx_arr**2 + ky**2 + 1e-12)
        omega  = np.sqrt(G * k_mag * np.tanh(k_mag * depth))
        phase  = kx_arr * x[None, :] + ky * y[None, :] - omega * t[None, :]
        cos_c  = np.mean(z_c[None, :] * np.cos(phase), axis=1)
        sin_c  = np.mean(z_c[None, :] * np.sin(phase), axis=1)
        power[iy, :] = cos_c**2 + sin_c**2
    return power, kx_ax, ky_ax


def linear_projection(x, y, z, t, kx, ky, depth):
    k_mag = np.sqrt(kx**2 + ky**2 + 1e-12)
    omega = np.sqrt(G * k_mag * np.tanh(k_mag * depth))
    ph = kx * x + ky * y - omega * t
    c, s = np.cos(ph), np.sin(ph)
    cc, ss, cs = c @ c, s @ s, c @ s
    cz, sz = c @ z, s @ z
    det = cc * ss - cs**2
    if abs(det) < 1e-12:
        return 0.01, 0.0
    a = (ss * cz - cs * sz) / det
    b = (cc * sz - cs * cz) / det
    return float(max(np.sqrt(a**2 + b**2), 1e-4)), float(np.arctan2(-b, a))


def greedy_extract(x, y, z, t, depth, n_comp,
                   krange=KXY_RANGE, n=N_COARSE, min_dk=MIN_DK):
    power, kx_ax, ky_ax = beamform_cartesian(x, y, z, t, depth, krange, n)
    dkx = kx_ax[1] - kx_ax[0]
    dky = ky_ax[1] - ky_ax[0]
    cx = np.argmin(np.abs(kx_ax))
    cy = np.argmin(np.abs(ky_ax))
    power[max(0, cy-1):cy+2, max(0, cx-1):cx+2] = 0.0
    z_res = z.copy()
    kx_list, ky_list = [], []
    for _ in range(n_comp):
        flat = np.argmax(power.ravel())
        iy, ix = np.unravel_index(flat, power.shape)
        if power[iy, ix] < 1e-14:
            break
        kx0, ky0 = kx_ax[ix], ky_ax[iy]
        delta = 0.15
        kxf = np.linspace(kx0 - delta, kx0 + delta, 21)
        kyf = np.linspace(ky0 - delta, ky0 + delta, 21)
        best_p, best_kx, best_ky = -1, kx0, ky0
        z_c = z_res - z_res.mean()
        for kyv in kyf:
            kx_a = kxf[:, None]
            km   = np.sqrt(kx_a**2 + kyv**2 + 1e-12)
            om   = np.sqrt(G * km * np.tanh(km * depth))
            ph   = kx_a * x[None, :] + kyv * y[None, :] - om * t[None, :]
            cc   = np.mean(z_c[None, :] * np.cos(ph), axis=1)
            ss   = np.mean(z_c[None, :] * np.sin(ph), axis=1)
            pw   = cc**2 + ss**2
            mi   = np.argmax(pw)
            if pw[mi] > best_p:
                best_p, best_kx, best_ky = pw[mi], kxf[mi], kyv
        amp, phi = linear_projection(x, y, z_res, t, best_kx, best_ky, depth)
        kx_list.append(best_kx)
        ky_list.append(best_ky)
        km   = np.sqrt(best_kx**2 + best_ky**2 + 1e-12)
        om   = np.sqrt(G * km * np.tanh(km * depth))
        z_res -= amp * np.cos(best_kx * x + best_ky * y - om * t + phi)
        sx = int(min_dk / dkx) + 1
        sy = int(min_dk / dky) + 1
        power[max(0, iy-sy):iy+sy+1, max(0, ix-sx):ix+sx+1] = 0.0
    return np.array(kx_list), np.array(ky_list)


# ── run optimization via SequentialUpdater ─────────────────

def run_coarse_to_fine(loader, cfg):
    from wavesplat.optimizer.sequential_update import SequentialUpdater
    updater = SequentialUpdater(cfg)
    init_frames = loader.get_init_window(tp_factor=2.0)
    updater.initialize(
        init_frames,
        x_range=(float(loader.gt_grid_x[0]), float(loader.gt_grid_x[-1])),
        y_range=(float(loader.gt_grid_y[0]), float(loader.gt_grid_y[-1])),
        verbose=False,
    )
    params = updater.model.get_spectrum_params()
    nrmse_after = None
    # quick eval on last init frame
    try:
        import torch
        from wavesplat.evaluation.metrics import nrmse as nrmse_fn
        last = init_frames[-1]
        pred = updater.predict_field(loader.gt_grid_x, loader.gt_grid_y, last.t)
        if last.gt_field is not None:
            nrmse_after = float(nrmse_fn(
                pred.cpu().numpy(), last.gt_field.cpu().numpy(), loader.Hs_true))
    except Exception:
        pass
    return params, nrmse_after


# ── plot helper ────────────────────────────────────────────

def plot_kspace(ax, power, kx_ax, ky_ax, title,
                gt_kx=None, gt_ky=None,
                ext_kx=None, ext_ky=None,
                show_legend=True):
    extent = [kx_ax[0], kx_ax[-1], ky_ax[0], ky_ax[-1]]
    P_db = 10 * np.log10(power + 1e-12)
    P_db -= P_db.min()
    ax.set_facecolor('#0f1117')
    im = ax.imshow(P_db, origin='lower', aspect='equal',
                   extent=extent, cmap='plasma', interpolation='bilinear',
                   alpha=0.85 if ext_kx is not None else 1.0)

    handles = []
    if gt_kx is not None:
        for i, (kx, ky) in enumerate(zip(gt_kx, gt_ky)):
            for sign in [1, -1]:
                h = ax.plot(sign * kx, sign * ky, '+',
                            color='#00e676', ms=12, mew=2.5, zorder=5)
        handles.append(plt.Line2D([0],[0], marker='+', color='#00e676',
                                   ms=10, mew=2.5, ls='none', label='GT'))

    if ext_kx is not None:
        for kx, ky in zip(ext_kx, ext_ky):
            ax.plot(kx, ky, 'o', color='#ffeb3b', ms=9, mew=2,
                    markerfacecolor='none', zorder=6)
            ax.plot(-kx, -ky, 'o', color='#ffeb3b', ms=9, mew=2,
                    markerfacecolor='none', zorder=6)
        handles.append(plt.Line2D([0],[0], marker='o', color='#ffeb3b',
                                   ms=8, mew=2, ls='none',
                                   markerfacecolor='none', label='Extracted'))

    ax.axhline(0, color='#444', lw=0.5, ls='--')
    ax.axvline(0, color='#444', lw=0.5, ls='--')
    ax.set_title(title, color='white', fontsize=10, fontweight='bold', pad=6)
    ax.set_xlabel('kx [rad/m]', color='#9e9e9e', fontsize=8)
    ax.set_ylabel('ky [rad/m]', color='#9e9e9e', fontsize=8)
    ax.tick_params(colors='#666', labelsize=7)
    if show_legend and handles:
        ax.legend(handles=handles, loc='upper right', fontsize=7,
                  facecolor='#1a1a2e', labelcolor='white', edgecolor='#444')
    return im


# ══════════════════════════════════════════════════════════
# Load data + run beamforming + run optimization
# ══════════════════════════════════════════════════════════

from wavesplat.data_loader import EpisodeDataLoader

OPT_CFG = {
    'n_gaussians': [3, 8, 15], 'depth': 50.0, 'lr': 0.01,
    'lambda_l1': 0.001, 'lambda_smooth': 0.0001,
    'tracking_steps': 10, 'tracking_mode': 'w_phi', 'tracking_lr': 0.005,
    'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
    'adaptive_n': False, 'device': 'cpu', 'seed': 42,
}

episode_data = []
for ep_path, title in zip(EPISODES, TITLES):
    print(f"\n{'='*50}")
    print(f"Loading: {os.path.basename(ep_path)}")

    loader = EpisodeDataLoader(ep_path)
    wc    = loader.wave_components
    depth = loader.depth
    Hs    = loader.Hs_true

    # Collect init-window observations
    init_frames = loader.get_init_window(tp_factor=2.0)
    all_x, all_y, all_z, all_t = [], [], [], []
    for f in init_frames:
        all_x.append(f.x_obs.numpy()); all_y.append(f.y_obs.numpy())
        all_z.append(f.z_obs.numpy())
        all_t.append(np.full(len(f.x_obs), f.t))
    x = np.concatenate(all_x); y = np.concatenate(all_y)
    z = np.concatenate(all_z); t = np.concatenate(all_t)
    print(f"  {len(x)} obs in {len(init_frames)} init frames, depth={depth}m")

    # Beamforming power map
    print("  Cartesian beamforming...", flush=True)
    power, kx_ax, ky_ax = beamform_cartesian(x, y, z, t, depth)

    # Greedy init
    n_gt = wc.shape[0]
    print(f"  Greedy extraction ({n_gt} comps)...", flush=True)
    bf_kx, bf_ky = greedy_extract(x, y, z, t, depth, n_gt)

    # Coarse-to-fine optimization
    print("  Coarse-to-fine optimization...", flush=True)
    t0 = time.time()
    opt_params, opt_nrmse = run_coarse_to_fine(loader, OPT_CFG)
    elapsed = time.time() - t0
    opt_kx = opt_params['kx']
    opt_ky = opt_params['ky']
    opt_w  = opt_params['w']
    print(f"  Done in {elapsed:.1f}s  |  N_opt={len(opt_kx)}"
          + (f"  nRMSE_init={opt_nrmse:.3f}" if opt_nrmse else ""))

    # GT matching stats  — format [A, omega, kx, ky, phi, Q]
    print(f"  GT kx={np.round(wc[:,2],3)}")
    for i in range(n_gt):
        dists = np.sqrt((opt_kx - wc[i,2])**2 + (opt_ky - wc[i,3])**2)
        best  = np.argmin(dists)
        print(f"    GT[{i}]=({wc[i,2]:.3f},{wc[i,3]:.3f})  "
              f"OPT=({opt_kx[best]:.3f},{opt_ky[best]:.3f})  "
              f"err={dists[best]:.3f}")

    episode_data.append(dict(
        title=title, Hs=Hs, wc=wc, depth=depth,
        power=power, kx_ax=kx_ax, ky_ax=ky_ax,
        bf_kx=bf_kx, bf_ky=bf_ky,
        opt_kx=opt_kx, opt_ky=opt_ky, opt_w=opt_w,
        opt_nrmse=opt_nrmse, n_obs=len(x),
    ))

# ══════════════════════════════════════════════════════════
# Figure 1: Raw 3D power maps
# ══════════════════════════════════════════════════════════
print("\nFigure 1: 3D power maps...")
fig1 = plt.figure(figsize=(22, 18))
fig1.patch.set_facecolor('#0f1117')

for idx, ep in enumerate(episode_data):
    KX, KY = np.meshgrid(ep['kx_ax'], ep['ky_ax'])
    P_db = 10 * np.log10(ep['power'] + 1e-12)
    P_db -= P_db.min()

    ax = fig1.add_subplot(2, 2, idx+1, projection='3d')
    ax.set_facecolor('#0f1117')
    surf = ax.plot_surface(KX, KY, P_db, cmap='plasma', alpha=0.85,
                           linewidth=0, antialiased=True, rcount=81, ccount=81)
    p_max = P_db.max()
    wc = ep['wc']
    for ci in range(wc.shape[0]):
        kx_gt, ky_gt = float(wc[ci, 2]), float(wc[ci, 3])  # [A,omega,kx,ky,phi,Q]
        for sign in [1, -1]:
            ax.plot([sign*kx_gt]*2, [sign*ky_gt]*2, [0, p_max*0.95],
                    color='#00e676', lw=2.5, alpha=0.9)
            ax.scatter([sign*kx_gt], [sign*ky_gt], [p_max*0.95],
                       color='#00e676', s=60, zorder=5)
    ax.set_xlabel('kx [rad/m]', color='#9e9e9e', fontsize=9, labelpad=8)
    ax.set_ylabel('ky [rad/m]', color='#9e9e9e', fontsize=9, labelpad=8)
    ax.set_zlabel('Power [dB]', color='#9e9e9e', fontsize=9, labelpad=8)
    ax.set_title(f"{ep['title']}  $H_s$={ep['Hs']:.2f}m  ({ep['n_obs']} pts)",
                 color='white', fontsize=12, fontweight='bold', pad=12)
    ax.tick_params(colors='#666', labelsize=7)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#333')
    ax.grid(True, color='#333', alpha=0.5)
    ax.view_init(elev=30, azim=-55)
    cb = fig1.colorbar(surf, ax=ax, shrink=0.5, pad=0.08, aspect=15)
    cb.ax.tick_params(colors='#9e9e9e', labelsize=7)
    cb.set_label('Power [dB]', color='#9e9e9e', fontsize=8)

fig1.suptitle("Dispersion-aware Beamforming  $P(k_x, k_y)$  [Raw 3D]\n"
              "green = GT wave components",
              color='white', fontsize=14, fontweight='bold', y=0.98)
out1 = os.path.join(OUT_DIR, "beamforming_powermap_3d.png")
fig1.savefig(out1, dpi=130, bbox_inches='tight',
             facecolor='#0f1117', edgecolor='none')
plt.close(fig1)
print(f"  Saved: {out1}")

# ══════════════════════════════════════════════════════════
# Figure 2: 3-stage comparison  (4 rows × 3 cols)
#   Col 1: Raw power map + GT
#   Col 2: BF greedy init  + GT
#   Col 3: Coarse-to-fine opt + GT
# ══════════════════════════════════════════════════════════
print("Figure 2: 3-stage comparison...")
fig2, axes = plt.subplots(4, 3, figsize=(18, 24))
fig2.patch.set_facecolor('#0f1117')
fig2.subplots_adjust(hspace=0.38, wspace=0.28)

col_titles = [
    "① Raw $P(k_x,k_y)$\n(No processing)",
    "② Beamforming Init\n(Greedy extraction)",
    "③ After Coarse-to-fine Opt\n(Adam optimizer)",
]

for col, ct in enumerate(col_titles):
    axes[0, col].set_title(ct, color='#aaaaff', fontsize=11,
                           fontweight='bold', pad=4)

for row, ep in enumerate(episode_data):
    power  = ep['power']
    kx_ax  = ep['kx_ax']
    ky_ax  = ep['ky_ax']
    wc     = ep['wc']
    gt_kx  = wc[:, 2]  # [A, omega, kx, ky, phi, Q]
    gt_ky  = wc[:, 3]
    bf_kx  = ep['bf_kx']
    bf_ky  = ep['bf_ky']
    opt_kx = ep['opt_kx']
    opt_ky = ep['opt_ky']
    opt_w  = ep['opt_w']
    Hs     = ep['Hs']
    title  = ep['title']
    n_gt   = wc.shape[0]

    row_label = f"{title}  |  $H_s$={Hs:.2f}m"
    if ep['opt_nrmse'] is not None:
        row_label += f"  |  nRMSE(init)={ep['opt_nrmse']:.3f}"

    # Col 0: Raw power map + GT only
    im0 = plot_kspace(axes[row, 0], power, kx_ax, ky_ax,
                      row_label,
                      gt_kx=gt_kx, gt_ky=gt_ky)

    # Col 1: Raw power map + GT + BF extracted peaks
    im1 = plot_kspace(axes[row, 1], power, kx_ax, ky_ax,
                      f"BF greedy: {len(bf_kx)}/{n_gt} extracted",
                      gt_kx=gt_kx, gt_ky=gt_ky,
                      ext_kx=bf_kx, ext_ky=bf_ky)

    # Col 2: Raw power map + GT + Optimized peaks
    # Scale marker size by amplitude
    w_abs = np.abs(opt_w)
    w_norm = w_abs / (w_abs.max() + 1e-8)

    im2 = plot_kspace(axes[row, 2], power, kx_ax, ky_ax,
                      f"After C2F opt: N={len(opt_kx)} components",
                      gt_kx=gt_kx, gt_ky=gt_ky,
                      ext_kx=opt_kx, ext_ky=opt_ky)

    # Draw GT-to-nearest-opt pairing lines on col 2
    for i in range(n_gt):
        if len(opt_kx) == 0:
            break
        dists = np.sqrt((opt_kx - gt_kx[i])**2 + (opt_ky - gt_ky[i])**2)
        best  = np.argmin(dists)
        axes[row, 2].plot([gt_kx[i], opt_kx[best]],
                          [gt_ky[i], opt_ky[best]],
                          '--', color='white', lw=0.7, alpha=0.4, zorder=3)
        err = dists[best]
        mx = (gt_kx[i] + opt_kx[best]) / 2
        my = (gt_ky[i] + opt_ky[best]) / 2
        axes[row, 2].text(mx, my, f"{err:.2f}",
                          color='white', fontsize=6, ha='center', alpha=0.7)

    # Colorbars on rightmost column
    cb = fig2.colorbar(im2, ax=axes[row, 2], shrink=0.85, pad=0.02)
    cb.ax.tick_params(colors='#9e9e9e', labelsize=6)
    cb.set_label('Power [dB]', color='#9e9e9e', fontsize=7)

# Column header row
fig2.suptitle(
    "Beamforming Power Map: 3-Stage Comparison\n"
    "Green + = GT  |  Yellow o = Extracted/Optimized  |  "
    "Dashed = GT-to-nearest pairing (error in rad/m)",
    color='white', fontsize=12, fontweight='bold', y=0.995)

out2 = os.path.join(OUT_DIR, "beamforming_3stage.png")
fig2.savefig(out2, dpi=130, bbox_inches='tight',
             facecolor='#0f1117', edgecolor='none')
plt.close(fig2)
print(f"  Saved: {out2}")
print("\nDone.")
