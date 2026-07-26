"""Animated field comparison for the UNIFIED estimator (2026-07-26 state of the repo).

Runs the real pipeline twice per episode — pure analytic (v2_konly) and the deployed
single-config path (init_method='auto', guard on) — with the fixed tracker
(TRACK_PHASE_PREDICT=False, arm protocol), and renders GT | v2 | auto side by side.

Crowded episode uses the Stage-1 cached diffusion atoms (diff_seed=0) so the auto arm
is bit-identical to the pre-registered lane; routing itself is live.

Run from wavesplat/ package root:  python3 tmp/figs_program_2026-07-26/make_gif.py
"""
import os, json
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import wavesplat.optimizer.sequential_update as su
su.TRACK_PHASE_PREDICT = False
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.sequential_update import SequentialUpdater
from wavesplat.run import DEFAULT_CONFIG, set_seed

FINE = 'data_finemesh_2026-07-23/'
S1_CACHE = 'tmp/wp_integration_2026-07-25/atoms_cache.jsonl'
X_CACHE = 'tmp/unified_2026-07-26/atoms_cache_extra.jsonl'
OUT = 'results/gifs'
INK, SEC, SURF = '#0b0b0b', '#52514e', '#fcfcfb'


def load_atoms(ep):
    for src in (S1_CACHE, X_CACHE):
        for l in open(src):
            d = json.loads(l)
            if d['episode'] == ep and d.get('diff_seed') == 0 and d.get('kx') is not None:
                return dict(kx=d['kx'], ky=d['ky'], energy=d['energy'])
    return None


def run_arm(ep, init_method, atoms=None):
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(adaptive_n=True, seed=42, device='cpu', init_method=init_method,
               learned_diffusion_seed=0)
    if atoms is not None:
        cfg['learned_atoms'] = dict(atoms)
    set_seed(42)
    loader = EpisodeDataLoader(FINE + ep, device='cpu')
    cfg['depth'] = loader.depth
    init_frames = loader.get_init_window(tp_factor=2.0)
    upd = SequentialUpdater(cfg)
    upd.initialize(init_frames,
                   x_range=(float(loader.gt_grid_x[0]), float(loader.gt_grid_x[-1])),
                   y_range=(float(loader.gt_grid_y[0]), float(loader.gt_grid_y[-1])))
    frames = loader.get_tracking_frames(tp_factor=2.0)
    preds, nr = [], []
    for fr in frames:
        upd.update(fr)
        p = upd.predict_field(loader.gt_grid_x, loader.gt_grid_y, fr.t).cpu().numpy()
        preds.append(p)
        gt = fr.gt_field.cpu().numpy()
        nr.append(float(np.sqrt(np.mean((p - gt) ** 2)) / loader.Hs_true))
    return dict(loader=loader, frames=frames, preds=preds, nr=nr,
                sel=getattr(upd, '_auto_selected', None),
                guard=getattr(upd, '_auto_guard_fired', None))


def make_gif(ep, label, stem, use_atoms=True):
    atoms = load_atoms(ep) if use_atoms else None
    v2 = run_arm(ep, 'v2_konly')
    au = run_arm(ep, 'auto', atoms=atoms)
    L = v2['loader']
    gx, gy = L.gt_grid_x.cpu().numpy(), L.gt_grid_y.cpu().numpy()
    ext = [gx.min(), gx.max(), gy.min(), gy.max()]
    n = len(v2['frames'])
    step = max(1, n // 80)
    idx = list(range(0, n, step))
    gt0 = v2['frames'][0].gt_field.cpu().numpy()
    vm = float(2.2 * np.std(gt0))
    route = au['sel'] or 'auto'
    gtag = ' (guard veto→v2)' if au['guard'] else ''
    print(f'{ep}: auto routed -> {route}{gtag}  '
          f'avg nRMSE v2={np.mean(v2["nr"]):.4f} auto={np.mean(au["nr"]):.4f}')

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.15))
    fig.patch.set_facecolor(SURF)
    titles = ['Ground truth (+ LiDAR hits)', 'pure analytic  (v2_konly)',
              f"init_method='auto'  → routed: {route}{gtag}"]
    ims = []
    for ax, t in zip(axes, titles):
        im = ax.imshow(np.zeros((len(gy), len(gx))), origin='lower', extent=ext,
                       cmap='RdBu_r', vmin=-vm, vmax=vm, aspect='auto')
        ax.set_title(t, fontsize=10.5, color=INK)
        ax.set_xlabel('x [m]', fontsize=9, color=SEC)
        ax.tick_params(labelsize=8, colors=SEC)
        ims.append(im)
    axes[0].set_ylabel('y [m]', fontsize=9, color=SEC)
    sc = axes[0].scatter([], [], s=1.5, c='#0b0b0b', alpha=0.55)
    cb = fig.colorbar(ims[0], ax=axes, fraction=0.02, pad=0.015)
    cb.set_label('η [m]', fontsize=9, color=SEC)
    cb.ax.tick_params(labelsize=8, colors=SEC)

    def upd_fn(j):
        i = idx[j]
        fr = v2['frames'][i]
        gt = fr.gt_field.cpu().numpy()
        ims[0].set_data(gt)
        ims[1].set_data(v2['preds'][i])
        ims[2].set_data(au['preds'][i])
        if len(fr.x_obs) > 0:
            sc.set_offsets(np.column_stack([fr.x_obs.cpu().numpy(),
                                            fr.y_obs.cpu().numpy()]))
        fig.suptitle(f'{label}   t={fr.t:.1f}s    nRMSE  v2={v2["nr"][i]:.3f}   '
                     f'auto={au["nr"][i]:.3f}    '
                     f'(episode avg: v2 {np.mean(v2["nr"]):.3f} / '
                     f'auto {np.mean(au["nr"]):.3f})',
                     fontsize=11.5, color=INK)
        return ims

    anim = FuncAnimation(fig, upd_fn, frames=len(idx), interval=120, blit=False)
    path = f'{OUT}/{stem}.gif'
    anim.save(path, writer=PillowWriter(fps=8), savefig_kwargs={'facecolor': SURF})
    plt.close(fig)
    print('saved', path, f'{os.path.getsize(path)/1e6:.1f} MB', flush=True)


if __name__ == '__main__':
    make_gif('episode_BF4_15comp_seed0_uniform.h5',
             'CROWDED sea (BF4, 15 components)', 'unified_auto_crowded_BF4_15comp')
    make_gif('episode_BF2_1comp_seed0_uniform.h5',
             'SPARSE sea (BF2, 1 component)', 'unified_auto_sparse_BF2_1comp')
