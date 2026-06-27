"""
Reconstruction GIFs with the final iterative (3-pass) estimator — complex seas.
GT | reconstruction | error, nowcast→forecast. Period-scaled window, observations only.
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
from d28_forecast_complex import estimate_components, K_TP

OUT = 'results/direction3/gifs'
os.makedirs(OUT, exist_ok=True)


def make_gif(tag, i_star):
    L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
    gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
    n_back = int(round(K_TP * L.Tp_true / 0.1))
    fstar = L[i_star]
    target = {'gx': gx, 'gy': gy, 't': float(fstar.t), 'gt': fstar.gt_field.numpy(), 'Hs': Hs}
    P, z = gather_frames(L, i_star, n_back)
    kx, ky, A, phi = estimate_components(P, z, depth, target, n_outer=3)
    ext = [gx.min(), gx.max(), gy.min(), gy.max()]
    vm = float(2.0 * np.std(fstar.gt_field.numpy()))
    n_fwd = min(30, len(L) - 1 - i_star)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    ims = []
    for ax, ti, cm, vl in zip(axes, ['Ground truth', 'Reconstruction (iterative)', 'Error'],
                              ['RdBu_r', 'RdBu_r', 'inferno'], [(-vm, vm), (-vm, vm), (0, vm)]):
        im = ax.imshow(np.zeros((len(gy), len(gx))), origin='lower', extent=ext, aspect='auto',
                       cmap=cm, vmin=vl[0], vmax=vl[1])
        ax.set_title(ti, fontsize=11); ax.set_xlabel('x [m]'); ims.append(im)
    axes[0].set_ylabel('y [m]')

    def upd(f):
        i = i_star + f; fr = L[i]; gt = fr.gt_field.numpy()
        rec = render_wave_field(kx, ky, A, phi, depth, gx, gy, float(fr.t))
        ims[0].set_data(gt); ims[1].set_data(rec); ims[2].set_data(np.abs(rec - gt))
        nr = np.sqrt(np.mean((rec - gt)**2)) / Hs; pe = pearson_correlation(rec, gt)
        dt = (i - i_star) * 0.1
        kind = 'nowcast' if dt <= 0.01 else f'forecast +{dt:.1f}s'
        fig.suptitle(f'{tag}  {kind}   nRMSE={nr:.3f}  Pearson={pe:.2f}', fontsize=12)
        return ims

    anim = FuncAnimation(fig, upd, frames=n_fwd, interval=120, blit=False)
    path = f'{OUT}/{tag}_iterative.gif'
    anim.save(path, writer=PillowWriter(fps=8)); plt.close(fig)
    print(f'saved {path}', flush=True)


def main():
    for tag, i_star in [('BF4_15c_s0', 200), ('BF4_30c_s0', 200), ('BF5_15c_s4', 200)]:
        make_gif(tag, i_star)


if __name__ == '__main__':
    main()
