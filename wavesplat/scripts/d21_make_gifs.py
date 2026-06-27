"""
Render reconstruction-vs-GT GIFs for visual inspection of the deployable estimator.

Estimate components from the fused window [t*-T, t*] (observations only), then animate
GT | reconstruction | error over a nowcast→short-forecast sequence. The components
propagate by dispersion, so frames after t* are genuine forecasts.

Run: python3 scripts/d21_make_gifs.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from wavesplat.evaluation.metrics import pearson_correlation
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames
from d17_recursive_estimator import refine_params
from d18_relax_estimator import relax_step, MCAP

G = 9.81
OUT = 'results/direction3/gifs'
os.makedirs(OUT, exist_ok=True)


def estimate(tag, i_star, T):
    L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
    fstar = L[i_star]
    target = {'gx': L.gt_grid_x.numpy(), 'gy': L.gt_grid_y.numpy(),
              't': float(fstar.t), 'gt': fstar.gt_field.numpy(), 'Hs': Hs}
    P, z = gather_frames(L, i_star, int(round(T / 0.1)))
    kx, ky = relax_step(np.zeros(0), np.zeros(0), P, z, depth)
    kx, ky, _ = refine_params(kx, ky, P, z, depth, target)
    obs = np.column_stack([P[:, 0], P[:, 1], P[:, 2], z]).astype(np.float32)
    if len(obs) > MCAP:
        obs = obs[np.random.default_rng(1).choice(len(obs), MCAP, replace=False)]
    A, phi = fit_amplitudes_phases(kx, ky, obs, depth, lambda_reg=0.5)
    return L, depth, Hs, kx, ky, A, phi


def make_gif(tag, i_star, T, label, n_fwd=25):
    L, depth, Hs, kx, ky, A, phi = estimate(tag, i_star, T)
    n_fwd = min(n_fwd, len(L) - 1 - i_star)
    gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
    ext = [gx.min(), gx.max(), gy.min(), gy.max()]
    vm = float(2.0 * np.std(L[i_star].gt_field.numpy()))
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    ims = []
    titles = ['Ground truth', f'Reconstruction ({label})', 'Error']
    cmaps = ['RdBu_r', 'RdBu_r', 'inferno']
    vlims = [(-vm, vm), (-vm, vm), (0, vm)]
    for ax, ti, cm, vl in zip(axes, titles, cmaps, vlims):
        im = ax.imshow(np.zeros((len(gy), len(gx))), origin='lower', extent=ext,
                       aspect='auto', cmap=cm, vmin=vl[0], vmax=vl[1])
        ax.set_title(ti, fontsize=11); ax.set_xlabel('x [m]'); ims.append(im)
    axes[0].set_ylabel('y [m]')

    def upd(f):
        i = i_star + f; fr = L[i]; gt = fr.gt_field.numpy()
        rec = render_wave_field(kx, ky, A, phi, depth, gx, gy, float(fr.t))
        err = np.abs(rec - gt)
        ims[0].set_data(gt); ims[1].set_data(rec); ims[2].set_data(err)
        nr = np.sqrt(np.mean((rec - gt)**2)) / Hs
        pe = pearson_correlation(rec, gt)
        dt = (i - i_star) * 0.1
        kind = 'nowcast' if dt <= 0.01 else f'forecast +{dt:.1f}s'
        fig.suptitle(f'{tag}  T_obs={T:.0f}s   {kind}   nRMSE={nr:.3f}  Pearson={pe:.2f}',
                     fontsize=12)
        return ims

    anim = FuncAnimation(fig, upd, frames=n_fwd, interval=120, blit=False)
    path = f'{OUT}/{tag}_T{T:.0f}.gif'
    anim.save(path, writer=PillowWriter(fps=8)); plt.close(fig)
    print(f'saved {path}')


def main():
    cases = [
        ('BF4_5c_s0', 235, 15.0, 'good'),
        ('BF4_15c_s0', 235, 15.0, 'good'),
        ('BF5_30c_s1', 270, 15.0, 'marginal'),
        ('BF5_30c_s1', 270, 25.0, 'fixed: longer window'),
    ]
    for tag, i_star, T, label in cases:
        make_gif(tag, i_star, T, label)


if __name__ == '__main__':
    main()
