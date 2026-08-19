#!/usr/bin/env python3
"""GT vs our estimate vs today's remedy -- the same episode, the same frames, one colour scale.

Panel 1 is the truth.  Panels 2-3 are what the estimator actually reconstructs on the
scoring grid: the shipping arm (`L-SHIP`, 500-step joint fit) and the W5 replacement arm
(`L-VP-ONLY`, joint fit off + 400 VARPRO steps).  Panel 4 is the signed error of the
remedy, on a scale tight enough to see what is left.

Everything is re-run here rather than read from a results file, because results.jsonl
stores scores, not fields.
"""
import os
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                               # noqa: E402
from matplotlib import font_manager                           # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

for _f in ('/mnt/c/Windows/Fonts/malgun.ttf',
           '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'):
    if os.path.exists(_f):
        font_manager.fontManager.addfont(_f)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=_f).get_name()
        break
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
W4 = os.path.join(ROOT, 'tmp', 'W4_SOLVER_2026-08-19')
W5 = os.path.join(ROOT, 'tmp', 'W5_WHY_2026-08-19')
for p in (ROOT, W4, W5):
    sys.path.insert(0, p)
os.chdir(ROOT)

EP = 'data_s8_crowded_2026-08-10/episode_BF4_15comp_seed32_uniform_ra_rf_ds.h5'
SEED = 1595
NF = int(sys.argv[1]) if len(sys.argv) > 1 else 48


def fields(arm):
    """Run one arm and capture the fields the scorer actually used.

    `run_estimation` returns a summary, not the updater, so instead of re-deriving the
    reconstruction (which would risk showing something the scorer never saw) we wrap
    `SequentialUpdater.predict_field` and record every call run.py:277 makes.
    """
    import arms_config_w5 as A5
    import varpro
    from wavesplat.optimizer.sequential_update import SequentialUpdater
    from wavesplat.run import run_estimation
    A5.register()
    cfg = A5.arm_config(arm, est_seed=SEED)
    spec = A5.ALL_SPECS[arm]

    grabbed = []
    orig = SequentialUpdater.predict_field

    def cap(self, gx, gy, t, *a, **k):
        out = orig(self, gx, gy, t, *a, **k)
        arr = out.detach().cpu().numpy() if hasattr(out, 'detach') else np.asarray(out)
        grabbed.append((float(t), arr, np.asarray(gx), np.asarray(gy)))
        return out

    SequentialUpdater.predict_field = cap
    if spec['vp']:
        varpro.install(steps=spec['vp'], weighting=spec['vp_weighting'])
    try:
        run_estimation(EP, config=cfg, verbose=False)
    finally:
        SequentialUpdater.predict_field = orig
        if spec['vp']:
            varpro.uninstall()

    # keep only the tracking-loop calls (they are the scored ones: one per frame, in order)
    grabbed = [g for g in grabbed if g[1].ndim == 2]
    step = max(1, len(grabbed) // NF)
    sel = grabbed[::step][:NF]
    print('   %s: captured %d predictions, using %d' % (arm, len(grabbed), len(sel)))
    return (np.array([g[0] for g in sel]), np.array([g[1] for g in sel]),
            sel[0][2], sel[0][3])


def main():
    print('running L-SHIP ...', flush=True)
    t, f_ship, gx, gy = fields('L-SHIP')
    print('running L-VP-ONLY ...', flush=True)
    _, f_vp, _, _ = fields('L-VP-ONLY')

    d = h5py.File(EP, 'r')
    tt = d['frames/timestamps'][:]
    sel = [int(np.argmin(np.abs(tt - x))) for x in t]
    gt = d['frames/gt_wave_field'][sel]
    obs = d['frames/lidar_points'][sel]
    val = d['frames/lidar_valid_mask'][sel]
    n = len(t)

    lim, elim = 1.15, .55
    fig = plt.figure(figsize=(16.6, 4.9), facecolor='#0e1315')
    gs = fig.add_gridspec(1, 4, left=.04, right=.905, top=.755, bottom=.13, wspace=.16)
    axes = [fig.add_subplot(g) for g in gs]
    for ax in axes:
        ax.set_facecolor('#0e1315')
        for s in ax.spines.values():
            s.set_color('#3a4548')
        ax.tick_params(colors='#8b9799', labelsize=8)
        ax.set_aspect('equal')
        ax.set_xlabel('x [m]', color='#8b9799', fontsize=9)
    axes[0].set_ylabel('y [m]', color='#8b9799', fontsize=9)

    ims = [axes[0].pcolormesh(gx, gy, gt[0], cmap='RdBu_r', vmin=-lim, vmax=lim, shading='auto'),
           axes[1].pcolormesh(gx, gy, f_ship[0], cmap='RdBu_r', vmin=-lim, vmax=lim, shading='auto'),
           axes[2].pcolormesh(gx, gy, f_vp[0], cmap='RdBu_r', vmin=-lim, vmax=lim, shading='auto'),
           axes[3].pcolormesh(gx, gy, f_vp[0] - gt[0], cmap='PuOr_r', vmin=-elim, vmax=elim,
                              shading='auto')]
    sc = axes[0].scatter([], [], s=6, c='#ffd166', edgecolors='none', zorder=3)

    titles = ['진실 (GT)', '배포 추정  ·  결합 하강 500스텝',
              '오늘의 처방  ·  VARPRO 400스텝', '처방의 오차  (추정 − 진실)']
    for ax, ti in zip(axes, titles):
        ax.set_title(ti, color='#e8edea', fontsize=10.5, pad=8)

    cb1 = fig.colorbar(ims[0], ax=axes[:3], fraction=.014, pad=.01)
    cb1.set_label('η [m]', color='#c8d1d0', fontsize=9)
    cb2 = fig.colorbar(ims[3], ax=[axes[3]], fraction=.05, pad=.03)
    cb2.set_label('오차 [m]', color='#c8d1d0', fontsize=9)
    for cb in (cb1, cb2):
        cb.ax.tick_params(colors='#8b9799', labelsize=8)
        cb.outline.set_edgecolor('#3a4548')

    fig.text(.04, .925, '같은 에피소드, 같은 프레임, 같은 색 범위', color='#e8edea', fontsize=14)
    sub = fig.text(.04, .865, '', color='#9fb0ae', fontsize=9.5)
    clock = fig.text(.905, .865, '', color='#5fbdb4', fontsize=10, ha='right')

    def nr(p, g):
        return float(np.sqrt(np.mean((p - g) ** 2)) / (4 * np.std(g)))

    def upd(i):
        ims[0].set_array(gt[i].ravel())
        ims[1].set_array(f_ship[i].ravel())
        ims[2].set_array(f_vp[i].ravel())
        ims[3].set_array((f_vp[i] - gt[i]).ravel())
        m = val[i]
        p = obs[i]
        sc.set_offsets(np.c_[p[m, 0], p[m, 1]] if m.any() else np.empty((0, 2)))
        sub.set_text('배포 nRMSE %.3f      처방 nRMSE %.3f      노란 점 = LiDAR 관측 %d 개 '
                     '(전체 격자의 %.1f%%)'
                     % (nr(f_ship[i], gt[i]), nr(f_vp[i], gt[i]), int(m.sum()),
                        100 * m.sum() / gt[i].size))
        clock.set_text('t = %5.1f s' % t[i])
        return ims + [sc, sub, clock]

    ani = FuncAnimation(fig, upd, frames=n, blit=False, interval=160)
    out = os.path.join(HERE, 'gt_vs_estimate.gif')
    ani.save(out, writer=PillowWriter(fps=6), savefig_kwargs=dict(facecolor='#0e1315'))
    print('wrote %s  (%d frames, %.1f MB)' % (out, n, os.path.getsize(out) / 1e6))


if __name__ == '__main__':
    main()
