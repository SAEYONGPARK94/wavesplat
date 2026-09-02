#!/usr/bin/env python3
"""⚠**이름 규약**(학위논문 §용어와 같다): **WavePrior** = 학습된 스펙트럼 사전
(원자 집합을 한 번에 제안하는 모형, 이 연구의 **주제**) · **WaveSplat** = 그 원자를 받아
파랑장을 유지하는 격자 비의존 파이프라인(**방법론의 한 구성요소**).
⟹ 패널 제목은 *"estimate: WavePrior atoms, tracked"* 다 --- 추정장은 사전 **혼자**의
산물이 아니고(VARPRO·추적이 함께 한다), 그렇다고 사전을 지우면 팔이 작동하지 않는다
(`B-22`: 해석적 제안으로 갈면 참 제로필드를 48 중 15 로만 넘는다).

`paper/figures/wave_estimate_3d.gif` --- 같은 자료의 **3-D 표면** 렌더링.

`gen_gif_estimate.py` 와 **같은 에피소드·같은 시드·같은 구성**이고 그림만 다르다.
★2-D 색지도가 못 보이는 것을 보인다: **파고 자체의 기복**과, 관측점이 표면 위 어디에
붙어 있는지.

★★**주장이 아니다** --- 에피소드 하나, 우리 팔 하나. 팔간 비교를 그리지 않는다.
출처는 `paper/figures/wave_estimate_3d.provenance.json`.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib import cm                                     # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch                                                  # noqa: E402
from wavesplat.run import DEFAULT_CONFIG                      # noqa: E402
from wavesplat.data_loader import EpisodeDataLoader           # noqa: E402
from wavesplat.optimizer.sequential_update import make_updater  # noqa: E402

EPISODE = 'data_s8_crowded_2026-08-10/episode_BF4_15comp_seed32_uniform_ra_rf_ds.h5'
TORCH_SEED = 42
STRIDE = 3
FPS = 8
SUB = 1      # ⚠2 로 두면 표면이 톱니처럼 보인다(격자가 60x80 뿐이다)
OUT = os.path.join(ROOT, 'paper', 'figures', 'wave_estimate_3d.gif')


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=ROOT, text=True).strip()[:12]
    except Exception:                                        # noqa: BLE001
        return None


def main():
    torch.manual_seed(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    cfg = dict(DEFAULT_CONFIG)
    loader = EpisodeDataLoader(EPISODE)
    cfg['depth'] = float(loader.depth)
    cfg['seed'] = TORCH_SEED
    t0 = time.time()
    upd = make_updater(cfg)
    upd.initialize(loader.get_init_window(tp_factor=2.0))
    init_s = time.time() - t0
    trk = loader.get_tracking_frames(tp_factor=2.0)
    prim = getattr(upd, '_primary', upd)

    gx, gy = loader.gt_grid_x.cpu().numpy(), loader.gt_grid_y.cpu().numpy()
    XX, YY = np.meshgrid(gx, gy)
    xf = torch.tensor(XX.ravel(), dtype=torch.float32)
    yf = torch.tensor(YY.ravel(), dtype=torch.float32)

    snaps = []
    for i, f in enumerate(trk):
        upd.update(f)
        if i % STRIDE:
            continue
        with torch.no_grad():
            est = prim.model(xf, yf, torch.tensor(float(f.t))).cpu().numpy()
        gt = f.gt_field.cpu().numpy()
        est = est.reshape(gt.shape)
        snaps.append(dict(t=float(f.t), gt=gt, est=est,
                          xo=f.x_obs.cpu().numpy(), yo=f.y_obs.cpu().numpy(),
                          zo=f.z_obs.cpu().numpy(),
                          rmse=float(np.sqrt(((est - gt) ** 2).mean()))))
    lim = max(float(np.abs(s['gt']).max()) for s in snaps)
    Xs, Ys = XX[::SUB, ::SUB], YY[::SUB, ::SUB]

    fig = plt.figure(figsize=(9.4, 4.0))
    axes = [fig.add_subplot(1, 2, k, projection='3d') for k in (1, 2)]
    txt = fig.text(0.5, 0.955, '', ha='center', fontsize=10)
    sub = fig.text(0.5, 0.03, '', ha='center', fontsize=7.4, color='#555')

    def draw(k):
        s = snaps[k]
        for a, F, ttl in ((axes[0], s['gt'], 'true wave field'),
                          (axes[1], s['est'], 'estimate: WavePrior atoms, tracked')):
            a.clear()
            a.plot_surface(Xs, Ys, F[::SUB, ::SUB], cmap=cm.RdBu_r,
                           vmin=-lim, vmax=lim, linewidth=0, antialiased=True,
                           rstride=1, cstride=1, alpha=.98, shade=True)
            a.set_zlim(-lim * 1.15, lim * 1.15)
            a.set_xlabel('x [m]', fontsize=7.5, labelpad=-4)
            a.set_ylabel('y [m]', fontsize=7.5, labelpad=-4)
            a.set_zlabel('elevation [m]', fontsize=7.5, labelpad=-6)
            a.tick_params(labelsize=6, pad=-2)
            a.set_title(ttl, fontsize=9.5, pad=-2)
            a.view_init(elev=30, azim=-124)
            a.set_box_aspect((2.4, 1.9, 0.85))
        # ★관측점은 **참 표면 위에만** 찍는다 --- 추정 쪽에 찍으면 "추정이 관측을 통과한다"
        #   로 오독될 수 있고, 그 주장은 이 그림이 지탱하지 않는다.
        axes[0].scatter(s['xo'], s['yo'], s['zo'], s=5, c='k', depthshade=False)
        txt.set_text(f"t = {s['t']:5.1f} s      frame RMSE = {s['rmse']:.3f} m      "
                     f"{len(s['xo'])} LiDAR returns")
        sub.set_text('black dots: every observation this frame (they sit near the '
                     'sensor). WavePrior places the atoms once; the field is then '
                     'evaluated across the whole domain')
        return axes

    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    ani = FuncAnimation(fig, draw, frames=len(snaps), blit=False)
    ani.save(OUT, writer=PillowWriter(fps=FPS), dpi=100)

    json.dump(dict(episode=EPISODE, torch_seed=TORCH_SEED, git_head=git_head(),
                   n_frames_animated=len(snaps), stride=STRIDE, fps=FPS,
                   surface_subsample=SUB,
                   t_first=snaps[0]['t'], t_last=snaps[-1]['t'],
                   init_seconds=round(init_s, 2),
                   final_n=int(prim.model.n_gaussians),
                   route=getattr(prim, '_auto_selected', None) or prim.init_method,
                   rmse_median=float(np.median([s['rmse'] for s in snaps])),
                   note=('ILLUSTRATIVE ONLY --- one episode, one arm. Same run as '
                         'wave_estimate.gif, rendered as a surface.')),
              open(os.path.join(ROOT, 'paper', 'figures',
                                'wave_estimate_3d.provenance.json'), 'w'),
              indent=2, ensure_ascii=False)
    print('wrote', OUT, f'· {len(snaps)} frames · N={int(prim.model.n_gaussians)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
