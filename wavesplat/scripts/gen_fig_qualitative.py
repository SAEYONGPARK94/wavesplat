#!/usr/bin/env python3
"""`paper/figures/fig_qualitative.pdf` --- **정성 예시** 한 에피소드.

⛔이 스크립트가 존재하는 이유(2026-09-02, 4차 적대적 감사 B8):

> *"재구성 예시 0 · 오차-거리 0 · 스펙트럼 0. 심사 가능한 증거 표면이 표 둘뿐이다."*

★★**이 그림은 주장이 아니다.** 에피소드 **하나**, 프레임 **하나**, 우리 팔 **하나**다.
비교자 팔을 나란히 그리지 않는다 --- 그렇게 그리면 그것은 **팔간 비교**가 되고,
이 저장소의 규율상 팔간 비교는 사전등록된 배치에서만 나온다.
정량 주장은 전부 Table~\\ref{tab:earns} / `fig_results.pdf` 에 있다.

★재현성: 에피소드 경로 · 프레임 · 배포 구성 스냅숏 · git HEAD 를 **그림 옆 json 에** 남긴다
(`paper/figures/fig_qualitative.provenance.json`). 그림만 있고 출처가 없으면 그것은
이 저장소가 반복해 대가를 치른 「추적 불가 산출물」이다.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from wavesplat.run import DEFAULT_CONFIG                     # noqa: E402
from wavesplat.data_loader import EpisodeDataLoader          # noqa: E402
from wavesplat.optimizer.sequential_update import make_updater  # noqa: E402

EPISODE = 'data_s8_crowded_2026-08-10/episode_BF4_15comp_seed32_uniform_ra_rf_ds.h5'
FRAME_IDX = 39          # 추적 시작 후 40번째 프레임 (t = 18.0 s)
TORCH_SEED = 42

import torch                                                  # noqa: E402
import h5py                                                   # noqa: E402


def main():
    torch.manual_seed(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    cfg = dict(DEFAULT_CONFIG)
    loader = EpisodeDataLoader(EPISODE)
    cfg['depth'] = float(loader.depth)
    cfg['seed'] = TORCH_SEED

    t0 = time.time()
    upd = make_updater(cfg)
    rec = upd.initialize(loader.get_init_window(tp_factor=2.0))
    init_s = time.time() - t0
    trk = loader.get_tracking_frames(tp_factor=2.0)
    for f in trk[:FRAME_IDX + 1]:
        upd.update(f)
    fr = trk[FRAME_IDX]

    gx = loader.gt_grid_x.cpu().numpy()
    gy = loader.gt_grid_y.cpu().numpy()
    gt = fr.gt_field.cpu().numpy()
    XX, YY = np.meshgrid(gx, gy)
    model = getattr(upd, '_primary', upd).model
    with torch.no_grad():
        est = model(torch.tensor(XX.ravel(), dtype=torch.float32),
                    torch.tensor(YY.ravel(), dtype=torch.float32),
                    torch.tensor(float(fr.t))).cpu().numpy().reshape(gt.shape)
    err = est - gt

    xo, yo = fr.x_obs.cpu().numpy(), fr.y_obs.cpu().numpy()
    d = np.sqrt((XX[..., None] - xo) ** 2 + (YY[..., None] - yo) ** 2).min(axis=-1)

    with h5py.File(EPISODE, 'r') as h:
        comp = h['config/wave_components'][:]          # [A, omega, kx, ky, phi, Q]
    tA, tkx, tky = comp[:, 0], comp[:, 2], comp[:, 3]
    ekx = model.kx.detach().cpu().numpy()
    eky = model.ky.detach().cpu().numpy()
    ew = np.abs(model.w.detach().cpu().numpy())

    lim = float(np.nanmax(np.abs(gt)))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    fig, ax = plt.subplots(2, 3, figsize=(7.6, 4.4))

    for a, F, ttl in ((ax[0, 0], gt, 'true field'),
                      (ax[0, 1], est, 'estimate')):
        im = a.pcolormesh(XX, YY, F, cmap='RdBu_r', norm=norm, shading='auto')
        a.set_title(ttl, fontsize=8)
        a.set_aspect('equal')
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
    ax[0, 0].plot(xo, yo, '.', ms=1.6, color='k', alpha=0.85)
    ax[0, 0].set_title(f"true field + this frame's {len(xo)} observations",
                       fontsize=8)

    elim = float(np.nanmax(np.abs(err)))
    im = ax[0, 2].pcolormesh(XX, YY, err, cmap='PuOr',
                             norm=TwoSlopeNorm(vmin=-elim, vcenter=0.0, vmax=elim),
                             shading='auto')
    ax[0, 2].plot(xo, yo, '.', ms=1.6, color='k', alpha=0.6)
    ax[0, 2].set_title('estimate $-$ truth  (colour scale '
                       f'$\\times${lim / elim:.1f} finer)', fontsize=8)
    ax[0, 2].set_aspect('equal')
    fig.colorbar(im, ax=ax[0, 2], fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
    for a in ax[0]:
        a.tick_params(labelsize=6)
        a.set_xlabel('x [m]', fontsize=7)
    ax[0, 0].set_ylabel('y [m]', fontsize=7)

    # ── 오차 대 최근접 관측까지의 거리
    edges = np.array([0, 5, 10, 20, 40, 80, 160, d.max() + 1e-6])
    mid, rms, cnt = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() < 20:
            continue
        mid.append(0.5 * (lo + min(hi, d.max())))
        rms.append(float(np.sqrt((err[m] ** 2).mean())))
        cnt.append(int(m.sum()))
    ax[1, 0].plot(mid, rms, 'o-', ms=4, lw=1.2, color='#0b5394')
    ax[1, 0].axvline(5.0, color='#8a8a8a', ls=':', lw=0.9)
    ax[1, 0].annotate('5 m band', xy=(5.4, max(rms)), fontsize=6,
                      color='#8a8a8a', va='top')
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_xlabel('distance to nearest observation [m]', fontsize=7)
    ax[1, 0].set_ylabel('RMS error [m]', fontsize=7)
    ax[1, 0].tick_params(labelsize=6)
    ax[1, 0].set_title('error against observation density', fontsize=8)
    for sp in ('top', 'right'):
        ax[1, 0].spines[sp].set_visible(False)

    # ── 파수 평면
    a = ax[1, 1]
    # ⚠[2026-09-02] 5차 감사: 참 성분(빈 원)이 추정 원자(채운 원)에 **가려** 캡션의
    #   'most are recovered' 를 그림에서 확인할 수 없었다. ⟹ 참 성분을 **위로** 올린다.
    a.scatter(ekx, eky, s=8 + 700 * ew, color='#0b5394', alpha=0.45, zorder=2)
    a.scatter(tkx, tky, s=8 + 700 * tA, facecolor='none', edgecolor='#a33', lw=1.4,
              zorder=4)
    a.scatter([], [], s=30, facecolor='none', edgecolor='#a33', lw=1.0,
              label='true components')
    a.scatter([], [], s=30, color='#0b5394', alpha=0.75, label='estimated atoms')
    a.axhline(0, color='#ddd', lw=0.6); a.axvline(0, color='#ddd', lw=0.6)
    a.set_xlabel('$k_x$ [rad/m]', fontsize=7); a.set_ylabel('$k_y$ [rad/m]', fontsize=7)
    a.tick_params(labelsize=6)
    a.legend(fontsize=6, frameon=False, loc='upper left')
    a.set_title('wavenumber plane (marker area $\\propto$ amplitude)', fontsize=8)

    # ── 반경 스펙트럼
    a = ax[1, 2]
    tk = np.sqrt(tkx ** 2 + tky ** 2); ek = np.sqrt(ekx ** 2 + eky ** 2)
    m1, s1, _ = a.stem(tk, tA, linefmt='-', markerfmt='o', basefmt=' ', label='true')
    m2, s2, _ = a.stem(ek, ew, linefmt='-', markerfmt='s', basefmt=' ', label='estimated')
    # ⚠matplotlib 의 stem 은 색을 인자로 안 받는다 --- 반환 아티스트에 직접 칠한다.
    #   (초판은 두 계열이 **같은 파란색**으로 나와 범례가 그림과 어긋났다.)
    for art, col in ((m1, '#a33'), (s1, '#a33'), (m2, '#0b5394'), (s2, '#0b5394')):
        art.set_color(col)
    m1.set_markersize(4); m2.set_markersize(4)
    m1.set_markerfacecolor('none')
    a.set_xlabel('$|k|$ [rad/m]', fontsize=7); a.set_ylabel('amplitude [m]', fontsize=7)
    a.tick_params(labelsize=6)
    a.legend(fontsize=6, frameon=False)
    a.set_title('radial spectrum', fontsize=8)
    for sp in ('top', 'right'):
        a.spines[sp].set_visible(False)

    fig.tight_layout(pad=0.5)
    out = os.path.join(ROOT, 'paper', 'figures', 'fig_qualitative.pdf')
    fig.savefig(out, bbox_inches='tight', pad_inches=0.02)

    head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                          text=True, cwd=ROOT).stdout.strip()
    prov = {
        'episode': EPISODE, 'frame_index': FRAME_IDX, 'frame_t': float(fr.t),
        'torch_seed': TORCH_SEED, 'git_head': head,
        'route': getattr(getattr(upd, '_primary', upd), '_auto_selected', None),
        'final_n': rec.get('final_n'), 'init_seconds': round(init_s, 2),
        'n_obs_this_frame': int(len(xo)),
        'rmse_this_frame': float(np.sqrt((err ** 2).mean())),
        'config': {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                   for k, v in sorted(cfg.items())},
        'note': ('ILLUSTRATIVE ONLY --- one episode, one frame, one arm. '
                 'No comparative claim. Quantitative claims live in the frozen '
                 'verdict files behind Table tab:earns.'),
    }
    with open(os.path.join(ROOT, 'paper', 'figures',
                           'fig_qualitative.provenance.json'), 'w') as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)
    print('wrote', out)
    print('  route', prov['route'], '| final_n', prov['final_n'],
          '| init %.1fs' % init_s, '| frame RMSE %.4f m' % prov['rmse_this_frame'])


if __name__ == '__main__':
    main()
