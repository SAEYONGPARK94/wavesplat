#!/usr/bin/env python3
"""EXPLORATORY -- the shipping estimator, driven on REAL stereo waves.

⛔ No PREREG.  No verdict.  This produces a picture and two descriptive numbers, nothing
more.  It exists because every accuracy claim in this repository was measured on a
synthetic corpus whose ground truth is reconstructible from the estimator's own model
equation to max|delta| = 0 -- so "what does the estimator do when the truth is NOT in its
model class" had never been shown, only argued.

Record: YS17 (Bergamasco stereo, 241x221 grid, 240 x 220 m, dt = 0.1 s).  A fixed
forward-looking sensor at the patch edge samples a Falcon-like fan (FOV +-60 deg, range
20-100 m) of the real surface each frame; z gets N(0, 2 cm) range noise.  Those points are
the only thing the estimator sees.  It is then asked to reconstruct the WHOLE patch,
including the 80%+ it never observed -- the same task the synthetic scoring poses.

⚠ The learned initialiser is out of distribution here (trained on the synthetic corpus).
That is not a bug in this script; it is the finding the picture is meant to show honestly.
"""
import os
import sys

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                               # noqa: E402
from matplotlib import font_manager                           # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

for _f in ('/mnt/c/Windows/Fonts/malgun.ttf',):
    if os.path.exists(_f):
        font_manager.fontManager.addfont(_f)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=_f).get_name()
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (ROOT, os.path.join(ROOT, 'tmp', 'W4_SOLVER_2026-08-19'),
          os.path.join(ROOT, 'tmp', 'W5_WHY_2026-08-19')):
    sys.path.insert(0, p)
os.chdir(ROOT)

REC = 'data/real_bergamasco/YS_2017/Surfaces_20170513_050000.nc'
DEPTH, DT = 50.0, 0.1
N_INIT, N_TRACK, STRIDE = 30, 40, 5      # frames: init window, tracked frames, decimation
FAN_R, FAN_DEG, M_CAP, SIG = (20.0, 100.0), 60.0, 64, 0.02


def load(n_frames):
    d = h5py.File(REC, 'r')

    def dec(name):
        v = d[name]
        a = v.attrs
        arr = v[:].astype(float)
        fv = a.get('_FillValue', [None])[0]
        if fv is not None:
            arr[arr == fv] = np.nan
        return arr * a.get('scale_factor', [1.0])[0] + a.get('add_offset', [0.0])[0]

    X, Y = dec('_nc4_non_coord_X'), dec('Y')
    sf = d['Z'].attrs['scale_factor'][0]
    idx = np.arange(n_frames) * STRIDE
    Z = d['Z'][idx].astype(float)
    Z[Z == -32767] = np.nan
    return X, Y, Z * sf, idx * DT


def to_native(X, Y, oy):
    """Stereo frame -> the estimator's frame: sensor at origin, water ahead in +x."""
    return (Y - oy), X                      # x' = downrange, y' = lateral


def fan_sample(X, Y, Z, rng, ox, oy):
    """Falcon-like forward fan, sampled from the real surface. Returns (x, y, z)."""
    dx, dy = X - ox, Y - oy
    r = np.hypot(dx, dy)
    az = np.degrees(np.arctan2(dx, dy))          # +y is the look direction
    ok = (r >= FAN_R[0]) & (r <= FAN_R[1]) & (np.abs(az) <= FAN_DEG) & np.isfinite(Z)
    ii = np.flatnonzero(ok.ravel())
    if len(ii) > M_CAP:
        ii = rng.choice(ii, M_CAP, replace=False)
    return (X.ravel()[ii], Y.ravel()[ii],
            Z.ravel()[ii] + rng.normal(0, SIG, len(ii)))


def run(arm, X, Y, Z, ts, ox, oy):
    import arms_config_w5 as A5
    import varpro
    from wavesplat.data_loader import FrameData
    from wavesplat.optimizer.sequential_update import make_updater
    A5.register()
    cfg = A5.arm_config(arm, est_seed=1595)
    spec = A5.ALL_SPECS[arm]
    rng = np.random.default_rng(0)

    def mkframe(i):
        x, y, z = fan_sample(X, Y, Z[i], rng, ox, oy)
        T = lambda a: torch.as_tensor(a, dtype=torch.float32)              # noqa: E731
        xn, yn = (y - oy), x                 # same rotation as the scoring grid
        return FrameData(t=float(ts[i]), x_obs=T(xn), y_obs=T(yn), z_obs=T(z),
                         confidence=T(np.ones_like(z)),
                         incidence_angles=T(np.zeros_like(z)))

    cfg = dict(cfg, depth=DEPTH)
    upd = make_updater(cfg)
    if spec['vp']:
        varpro.install(steps=spec['vp'], weighting=spec['vp_weighting'])
    try:
        Xn, Yn = to_native(X, Y, oy)
        upd.initialize([mkframe(i) for i in range(N_INIT)],
                       x_range=(float(np.nanmin(Xn)), float(np.nanmax(Xn))),
                       y_range=(float(np.nanmin(Yn)), float(np.nanmax(Yn))))
        # ⛔ BUG FIXED 2026-08-19 (audit R3 K-1): the first version sliced the wrong axis.
        # In this file `Y` varies along axis1 and `X` along axis0, so `Xn[:, 0]` and
        # `Yn[0]` each held ONE unique value -- every one of the 53,261 query points was
        # evaluated at the same coordinate, which sits outside the fan.  The resulting flat
        # panels were an artifact of this line, not a property of the estimator.
        gx = torch.as_tensor(Xn[0, :], dtype=torch.float32)   # downrange axis (221 pts)
        gy = torch.as_tensor(Yn[:, 0], dtype=torch.float32)   # lateral axis   (241 pts)
        out = []
        for i in range(N_INIT, N_INIT + N_TRACK):
            upd.update(mkframe(i))
            f = upd.predict_field(gx, gy, float(ts[i]))
            f = f.detach().cpu().numpy() if hasattr(f, 'detach') else np.asarray(f)
            out.append(f)                    # already (lateral, downrange) = stereo shape
        return np.array(out)
    finally:
        if spec['vp']:
            varpro.uninstall()


def main():
    X, Y, Z, ts = load(N_INIT + N_TRACK)
    ox, oy = 0.0, float(np.nanmin(Y)) - 5.0        # sensor just outside the near edge
    print('YS17: grid %s  |  sensor at (%.0f, %.0f)' % (str(Z[0].shape), ox, oy), flush=True)
    print('running L-SHIP ...', flush=True)
    fs = run('L-SHIP', X, Y, Z, ts, ox, oy)
    print('running L-VP-ONLY ...', flush=True)
    fv = run('L-VP-ONLY', X, Y, Z, ts, ox, oy)
    gt = Z[N_INIT:N_INIT + N_TRACK]
    tt = ts[N_INIT:N_INIT + N_TRACK]
    m = np.isfinite(gt)

    def nr(p):
        return np.array([np.sqrt(np.nanmean((p[i][m[i]] - gt[i][m[i]]) ** 2))
                         / (4 * np.nanstd(gt[i][m[i]])) for i in range(len(gt))])
    ns, nv = nr(fs), nr(fv)
    zz = np.array([np.sqrt(np.nanmean(gt[i][m[i]] ** 2)) / (4 * np.nanstd(gt[i][m[i]]))
                   for i in range(len(gt))])
    print('  nRMSE  배포 %.3f | VARPRO %.3f | 제로필드 %.3f  (중앙)'
          % (np.median(ns), np.median(nv), np.median(zz)))

    lim = 1.3
    fig = plt.figure(figsize=(16.4, 5.0), facecolor='#0e1315')
    gs = fig.add_gridspec(1, 4, left=.042, right=.9, top=.74, bottom=.13, wspace=.16)
    ax = [fig.add_subplot(g) for g in gs]
    for a in ax:
        a.set_facecolor('#0e1315')
        for s in a.spines.values():
            s.set_color('#3a4548')
        a.tick_params(colors='#8b9799', labelsize=8)
        a.set_aspect('equal')
        a.set_xlabel('x [m]', color='#8b9799', fontsize=9)
    ax[0].set_ylabel('y [m]', color='#8b9799', fontsize=9)

    ims = [ax[0].pcolormesh(X, Y, gt[0], cmap='RdBu_r', vmin=-lim, vmax=lim, shading='auto'),
           ax[1].pcolormesh(X, Y, fs[0], cmap='RdBu_r', vmin=-lim, vmax=lim, shading='auto'),
           ax[2].pcolormesh(X, Y, fv[0], cmap='RdBu_r', vmin=-lim, vmax=lim, shading='auto'),
           ax[3].pcolormesh(X, Y, fv[0] - gt[0], cmap='PuOr_r', vmin=-lim, vmax=lim,
                            shading='auto')]
    sc = ax[0].scatter([], [], s=7, c='#ffd166', edgecolors='none', zorder=3)
    for a, ti in zip(ax, ['실측 스테레오 (YS17) — 진실', '배포 추정',
                          '오늘의 처방 (VARPRO)', '처방의 오차']):
        a.set_title(ti, color='#e8edea', fontsize=10.5, pad=8)

    cb = fig.colorbar(ims[0], ax=ax, fraction=.014, pad=.01)
    cb.set_label('η [m]', color='#c8d1d0', fontsize=9)
    cb.ax.tick_params(colors='#8b9799', labelsize=8)
    cb.outline.set_edgecolor('#3a4548')

    fig.text(.042, .915, '실측 파도에서 — 참해가 우리 모델식 안에 없을 때',
             color='#e8edea', fontsize=14)
    fig.text(.042, .855,
             '⛔ 탐색적: 사전등록 없음, 판정 없음. 학습 초기화기는 여기서 분포 밖(OOD)이다.',
             color='#e08a8c', fontsize=9.5)
    sub = fig.text(.042, .800, '', color='#9fb0ae', fontsize=9.5)
    clock = fig.text(.9, .855, '', color='#5fbdb4', fontsize=10, ha='right')

    rng = np.random.default_rng(0)

    def upd(i):
        ims[0].set_array(gt[i].ravel())
        ims[1].set_array(fs[i].ravel())
        ims[2].set_array(fv[i].ravel())
        ims[3].set_array((fv[i] - gt[i]).ravel())
        x, y, _ = fan_sample(X, Y, gt[i], rng, ox, oy)
        sc.set_offsets(np.c_[x, y])
        sub.set_text('nRMSE   배포 %.3f      처방 %.3f      제로필드(무기술 기준) %.3f'
                     % (ns[i], nv[i], zz[i]))
        clock.set_text('t = %5.1f s' % tt[i])
        return ims + [sc, sub, clock]

    ani = FuncAnimation(fig, upd, frames=len(gt), blit=False, interval=170)
    out = os.path.join(HERE, 'real_gt_vs_estimate.gif')
    ani.save(out, writer=PillowWriter(fps=6), savefig_kwargs=dict(facecolor='#0e1315'))
    print('wrote %s (%.1f MB)' % (out, os.path.getsize(out) / 1e6))


if __name__ == '__main__':
    main()
