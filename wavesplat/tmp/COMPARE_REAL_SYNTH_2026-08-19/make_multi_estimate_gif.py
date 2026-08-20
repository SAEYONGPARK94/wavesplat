#!/usr/bin/env python3
"""GT vs estimate vs error, on four cases at once.

The four rows are chosen so the figure shows what today's C batch measured:

  1  synthetic crowded, sensor STATIONARY      -- the geometry every past batch used
  2  synthetic crowded, sensor MOVING 11.3 m/s -- the DECLARED operating point
  3  the same moving episode with the learned checkpoint removed (v2_konly)
  4  real stereo waves (Acqua Alta tower), through the deployed real-domain route

Rows 1-2 are the encounter cost (C: 0/16 episodes, error ~1.9x).  Row 3 is why the label
GEOMETRY-SURVIVES is not the whole story (C S-1: v2_konly loses to the zero field at the
operating point).  Row 4 is what the instrument actually sees.

Fields are captured by wrapping SequentialUpdater.predict_field, so what is drawn is what
the scorer used -- not a re-derivation that might differ.
"""
import os
import sys

import h5py
import matplotlib
import numpy as np

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
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.chdir(ROOT)
NF = 24
SEED = 1614

CASES = [
    ('정지 센서 (지금까지 전 배치)', 'auto',
     'tmp/C_ENCOUNTER_2026-08-20/data/A_static_following'),
    ('이동 11.3 m/s (선언된 작동점)', 'auto',
     'tmp/C_ENCOUNTER_2026-08-20/data/C_moving_following'),
    ('이동 + 학습 제거 (v2_konly)', 'v2_konly',
     'tmp/C_ENCOUNTER_2026-08-20/data/C_moving_following'),
]


def synth_case(folder, init_method):
    from wavesplat.optimizer.sequential_update import SequentialUpdater
    from wavesplat.run import DEFAULT_CONFIG, run_estimation
    path = os.path.join(ROOT, folder,
                        sorted(f for f in os.listdir(os.path.join(ROOT, folder))
                               if f.endswith('.h5'))[0])
    grabbed = []
    orig = SequentialUpdater.predict_field

    def spy(self, gx, gy, t, *a, **k):
        out = orig(self, gx, gy, t, *a, **k)
        grabbed.append(np.asarray(out.detach().cpu() if hasattr(out, 'detach') else out))
        return out

    SequentialUpdater.predict_field = spy
    try:
        cfg = dict(DEFAULT_CONFIG)
        cfg['init_method'] = init_method
        cfg['seed'] = SEED
        summary = run_estimation(h5_path=path, config=cfg)
    finally:
        SequentialUpdater.predict_field = orig

    with h5py.File(path, 'r') as d:
        gx = d['frames/gt_grid_x'][:]
        gy = d['frames/gt_grid_y'][:]
        gt_all = d['frames/gt_wave_field'][:]
    est = np.stack(grabbed[-len(grabbed):])
    gt = gt_all[-len(est):]
    n = min(NF, len(est))
    idx = np.linspace(0, len(est) - 1, n).astype(int)
    x1 = gx[0, :] if gx.ndim == 2 else gx
    y1 = gy[:, 0] if gy.ndim == 2 else gy
    return dict(x=x1, y=y1, gt=gt[idx], est=est[idx],
                nrmse=summary.get('avg_nRMSE'))


def real_case():
    """The DEPLOYED real route, not the research arm.

    real_estimate.run() takes an arms_config_w5 arm, and every one of those pins
    init_method='learned', which BYPASSES the real-domain gate (sequential_update.py:
    434-470) and runs the synthetic gradient path on real input.  Numbers from that arm
    are on the citation-ban list.  So this rebuilds the same loop with DEFAULT_CONFIG,
    where init_method='auto' lets the gate fire and route to bank_mmse -- the estimator
    behind every real-sea number this project has published.
    """
    import torch
    import real_estimate as RE
    from wavesplat.data_loader import FrameData
    from wavesplat.optimizer.sequential_update import make_updater
    from wavesplat.run import DEFAULT_CONFIG

    X, Y, Z, ts = RE.load(RE.N_INIT + RE.N_TRACK)
    ox, oy = 0.0, float(np.nanmin(Y)) - 5.0     # sensor just outside the near edge
    rng = np.random.default_rng(0)

    def mkframe(i):
        x, y, z = RE.fan_sample(X, Y, Z[i], rng, ox, oy)
        T = lambda a: torch.as_tensor(a, dtype=torch.float32)          # noqa: E731
        return FrameData(t=float(ts[i]), x_obs=T(y - oy), y_obs=T(x), z_obs=T(z),
                         confidence=T(np.ones_like(z)),
                         incidence_angles=T(np.zeros_like(z)))

    upd = make_updater(dict(DEFAULT_CONFIG, depth=RE.DEPTH, seed=SEED))
    Xn, Yn = RE.to_native(X, Y, oy)
    info = upd.initialize([mkframe(i) for i in range(RE.N_INIT)],
                          x_range=(float(np.nanmin(Xn)), float(np.nanmax(Xn))),
                          y_range=(float(np.nanmin(Yn)), float(np.nanmax(Yn))))
    print('   real route =', (info or {}).get('method', (info or {}).get('init_method')),
          '| n_iters', (info or {}).get('n_iters'), flush=True)
    gx = torch.as_tensor(Xn[0, :], dtype=torch.float32)
    gy = torch.as_tensor(Yn[:, 0], dtype=torch.float32)
    fv = []
    for i in range(RE.N_INIT, RE.N_INIT + RE.N_TRACK):
        upd.update(mkframe(i))
        f = upd.predict_field(gx, gy, float(ts[i]))
        fv.append(f.detach().cpu().numpy() if hasattr(f, 'detach') else np.asarray(f))
    fv = np.array(fv)
    gt = Z[RE.N_INIT:RE.N_INIT + RE.N_TRACK]
    n = min(NF, len(fv), len(gt))
    idx = np.linspace(0, min(len(fv), len(gt)) - 1, n).astype(int)
    gt, fv = np.asarray(gt)[idx], np.asarray(fv)[idx]
    m = np.isfinite(gt)
    nr = float(np.mean([np.sqrt(np.nanmean((fv[i][m[i]] - gt[i][m[i]]) ** 2))
                        / (4 * np.nanstd(gt[i][m[i]])) for i in range(len(gt))]))
    return dict(x=X, y=Y, gt=gt, est=fv, nrmse=nr, mesh=True)


def main():
    rows = []
    for label, init, folder in CASES:
        print('running', label, flush=True)
        r = synth_case(folder, init)
        r['label'] = label
        rows.append(r)
    try:
        print('running 실측', flush=True)
        r = real_case(); r['label'] = '실측 스테레오 (Acqua Alta, 배포 경로)'
        rows.append(r)
    except Exception as exc:                                  # noqa: BLE001
        print('  real row skipped:', exc)

    nfr = min(len(r['gt']) for r in rows)
    fig, axes = plt.subplots(len(rows), 3, figsize=(11.4, 3.05 * len(rows)))
    fig.patch.set_facecolor('#0d1117')
    axes = np.atleast_2d(axes)
    ims = []
    for j, r in enumerate(rows):
        lim = float(np.nanpercentile(np.abs(r['gt'][:nfr]), 99)) or 1.0
        elim = lim
        for i, (arr, cmap, vl, ttl) in enumerate((
                (r['gt'], 'RdBu_r', lim, '참값 (GT)'),
                (r['est'], 'RdBu_r', lim, '우리 추정'),
                (r['est'] - r['gt'], 'PuOr_r', elim, '오차 (추정 - 참값)'))):
            ax = axes[j, i]
            if r.get('mesh'):
                im = ax.pcolormesh(r['x'], r['y'], arr[0], cmap=cmap, vmin=-vl,
                                   vmax=vl, shading='auto')
            else:
                im = ax.imshow(arr[0], origin='lower', cmap=cmap, vmin=-vl, vmax=vl,
                               extent=[r['x'].min(), r['x'].max(),
                                       r['y'].min(), r['y'].max()],
                               aspect='equal', interpolation='nearest')
            ims.append((im, arr, bool(r.get('mesh'))))
            ax.set_facecolor('#0d1117')
            ax.tick_params(colors='#8b949e', labelsize=6)
            if j == 0:
                ax.set_title(ttl, color='#e6edf3', fontsize=10, pad=6)
            if i == 0:
                ax.set_ylabel('%s\nnRMSE %.3f' % (r['label'], r['nrmse']),
                              color='#79c0ff', fontsize=8.2)
            for s in ax.spines.values():
                s.set_color('#30363d')
    fig.suptitle('같은 추정기, 네 가지 조건 — 낮을수록 좋다',
                 color='#e6edf3', fontsize=12.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])

    def upd(i):
        for im, arr, mesh in ims:
            im.set_array(arr[i].ravel() if mesh else arr[i])
        return [x[0] for x in ims]

    out = os.path.join(HERE, 'multi_gt_vs_estimate.gif')
    FuncAnimation(fig, upd, frames=nfr, blit=False).save(
        out, writer=PillowWriter(fps=5), dpi=62,
        savefig_kwargs={'facecolor': '#0d1117'})
    for r in rows:
        print('  %-32s nRMSE %.4f' % (r['label'], r['nrmse']))
    print('wrote', out, '%.1f MB' % (os.path.getsize(out) / 1e6))


if __name__ == '__main__':
    main()
