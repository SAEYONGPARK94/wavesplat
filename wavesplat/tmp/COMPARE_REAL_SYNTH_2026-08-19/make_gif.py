#!/usr/bin/env python3
"""Side-by-side animation: the synthetic universe we optimise against vs the real sea.

Why this comparison and not a prettier one: the repository measured (2026-08-13) that the
synthetic ground truth is reconstructible from the estimator's own model equation to
max|delta| = 0.000e+00 -- the corpus IS a sum of sinusoids, exactly the family the
estimator fits.  The Bergamasco stereo surface is not.  So the honest framing of this
figure is not "our sim looks like the sea"; it is "here is the gap the numbers live in".

Both panels share one physical metre scale and one symmetric colour scale, so the
aperture difference and the texture difference are both visible without reading axes.

    python3 make_gif.py [n_frames]
"""
import glob
import os
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                               # noqa: E402
from matplotlib import font_manager                           # noqa: E402

# Korean labels render as tofu on the default DejaVu stack; WSL exposes the Windows fonts.
for _f in ('/mnt/c/Windows/Fonts/malgun.ttf',
           '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'):
    if os.path.exists(_f):
        font_manager.fontManager.addfont(_f)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=_f).get_name()
        break
plt.rcParams['axes.unicode_minus'] = False
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
os.chdir(ROOT)

REAL = 'data/real_bergamasco/AA_2014/Surfaces_20140327_091000.nc'
SYNTH = 'data_s8_crowded_2026-08-10/episode_BF4_15comp_seed32_uniform_ra_rf_ds.h5'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60


def _dec(ds, name):
    v = ds[name]
    a = v.attrs
    arr = v[:].astype(float)
    fv = a.get('_FillValue', [None])[0]
    if fv is not None:
        arr[arr == fv] = np.nan
    return arr * a.get('scale_factor', [1.0])[0] + a.get('add_offset', [0.0])[0]


def load_real(n):
    d = h5py.File(REAL, 'r')
    xg, yg = _dec(d, '_nc4_non_coord_X'), _dec(d, 'Y')
    sf = d['Z'].attrs['scale_factor'][0]
    # 12 Hz native; step to ~2 Hz so the animation shows real wave motion, not jitter
    idx = np.arange(n) * 6
    z = d['Z'][idx].astype(float)
    z[z == -32767] = np.nan
    return xg, yg, z * sf, float(np.diff(d['time'][:2])[0]) * 6


def load_synth(n):
    d = h5py.File(SYNTH, 'r')
    gx, gy = d['frames/gt_grid_x'][:], d['frames/gt_grid_y'][:]
    t = d['frames/timestamps'][:]
    step = max(1, len(t) // n)
    idx = np.arange(n) * step
    idx = idx[idx < len(t)]
    z = d['frames/gt_wave_field'][idx]
    obs = d['frames/lidar_points'][idx]
    val = d['frames/lidar_valid_mask'][idx]
    ncomp = d['config/wave_components'].shape[0]
    return gx, gy, z, obs, val, float(t[1] - t[0]) * step, ncomp


def main():
    rx, ry, rz, rdt = load_real(N)
    gx, gy, sz, obs, val, sdt, ncomp = load_synth(N)
    n = min(len(rz), len(sz))

    lim = 1.15
    cmap = 'RdBu_r'
    # Three panels, because the two stories need different framings and both are true:
    #   1-2 share one metre scale  -> the APERTURE gap (the real footprint is a corner of
    #       what the synthetic corpus assumes; B-3 already found no real record big enough)
    #   3 is the same real data at its own extent -> the TEXTURE gap, which the shared
    #       scale physically cannot show.
    fig = plt.figure(figsize=(15.4, 5.6), facecolor='#0e1315')
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, .78],
                          left=.045, right=.965, top=.775, bottom=.115, wspace=.24)
    axes = [fig.add_subplot(g) for g in gs]
    for ax in axes:
        ax.set_facecolor('#0e1315')
        for sp in ax.spines.values():
            sp.set_color('#3a4548')
        ax.tick_params(colors='#8b9799', labelsize=8)
        ax.set_aspect('equal')

    a0, a1, a2 = axes
    im0 = a0.pcolormesh(gx, gy, sz[0], cmap=cmap, vmin=-lim, vmax=lim, shading='auto')
    sc0 = a0.scatter([], [], s=6, c='#ffd166', edgecolors='none', zorder=3)
    im1 = a1.pcolormesh(rx, ry, rz[0], cmap=cmap, vmin=-lim, vmax=lim, shading='auto')
    im2 = a2.pcolormesh(rx, ry, rz[0], cmap=cmap, vmin=-lim, vmax=lim, shading='auto')

    for ax in (a0, a1):
        ax.set_xlim(gx.min(), gx.max())
        ax.set_ylim(gy.min(), gy.max())
    a2.set_xlim(np.nanmin(rx), np.nanmax(rx))
    a2.set_ylim(np.nanmin(ry), np.nanmax(ry))

    a0.set_title('합성 코퍼스 · BF4 · %d성분\n240 × 200 m 채점 격자' % ncomp,
                 color='#e8edea', fontsize=11, pad=9)
    a1.set_title('실측 스테레오 · Bergamasco AA_2014\n같은 축척 — 개구가 이만큼이다',
                 color='#e8edea', fontsize=11, pad=9)
    a2.set_title('같은 실측 자료, 자기 범위로 확대\n86 × 70 m — 질감을 보라',
                 color='#e8edea', fontsize=11, pad=9)
    for ax in axes:
        ax.set_xlabel('x [m]', color='#8b9799', fontsize=9)
    a0.set_ylabel('y [m]', color='#8b9799', fontsize=9)

    cb = fig.colorbar(im0, ax=axes, fraction=.016, pad=.008)
    cb.set_label('표면 고도 η [m]', color='#c8d1d0', fontsize=9)
    cb.ax.tick_params(colors='#8b9799', labelsize=8)
    cb.outline.set_edgecolor('#3a4548')

    fig.text(.045, .935, '우리가 최적화하는 우주  vs  실제 바다',
             color='#e8edea', fontsize=15)
    fig.text(.045, .875,
             '합성 GT 는 추정기의 모델식 Σ A·cos(k·x - wt + φ) 으로 max|Δ| = 0 으로 재구성된다 '
             '(2026-08-13 실측). 실측 면은 그렇지 않다.',
             color='#9fb0ae', fontsize=9.5)
    clock = fig.text(.965, .875, '', color='#5fbdb4', fontsize=10, ha='right')

    def upd(i):
        im0.set_array(sz[i].ravel())
        im1.set_array(rz[i].ravel())
        im2.set_array(rz[i].ravel())
        p, m = obs[i], val[i]
        sc0.set_offsets(np.c_[p[m, 0], p[m, 1]] if m.any() else np.empty((0, 2)))
        clock.set_text('t = %5.1f s      LiDAR 유효 관측 %d 점' % (i * sdt, int(m.sum())))
        return im0, im1, im2, sc0, clock

    ani = FuncAnimation(fig, upd, frames=n, blit=False, interval=140)
    out = os.path.join(HERE, 'synth_vs_real.gif')
    ani.save(out, writer=PillowWriter(fps=7), savefig_kwargs=dict(facecolor='#0e1315'))
    print('wrote %s  (%d frames, synth dt=%.2fs real dt=%.2fs)' % (out, n, sdt, rdt))
    print('  size: %.1f MB' % (os.path.getsize(out) / 1e6))


if __name__ == '__main__':
    main()
