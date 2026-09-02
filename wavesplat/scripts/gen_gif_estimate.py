#!/usr/bin/env python3
"""⚠**이름 규약**(학위논문 §용어와 같다): **WavePrior** = 학습된 스펙트럼 사전
(원자 집합을 한 번에 제안하는 모형, 이 연구의 **주제**) · **WaveSplat** = 그 원자를 받아
파랑장을 유지하는 격자 비의존 파이프라인(**방법론의 한 구성요소**).
⟹ 패널 제목은 *"estimate: WavePrior atoms, tracked"* 다 --- 추정장은 사전 **혼자**의
산물이 아니고(VARPRO·추적이 함께 한다), 그렇다고 사전을 지우면 팔이 작동하지 않는다
(`B-22`: 해석적 제안으로 갈면 참 제로필드를 48 중 15 로만 넘는다).

`paper/figures/wave_estimate.gif` --- **파고와 추정을 시간축으로** 보이는 애니메이션.

★★**이 애니메이션은 주장이 아니다.** 에피소드 **하나**, 우리 팔 **하나**다.
비교자 팔을 나란히 그리지 않는다 --- 그렇게 그리면 **팔간 비교**가 되고, 이 저장소의
규율상 팔간 비교는 **사전등록된 배치에서만** 나온다(정량은 `tab:earns`/`fig_results`).
`gen_fig_qualitative.py` 의 규약을 그대로 따른다.

★그런데 정지 그림이 못 보이는 것을 하나 보인다: **관측은 센서 근처 한 줌인데 추정은
도메인 전체를 시간에 따라 전개한다.** 그것이 「격자없는 스펙트럼 표현」이 사는 이유다.

★재현성: 에피소드 · 프레임 · 배포 구성 스냅숏 · git HEAD · 프레임별 RMSE 를
`paper/figures/wave_estimate.provenance.json` 에 남긴다.
⛔출처 없는 산출물은 이 저장소가 반복해 대가를 치른 부류다.
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
from matplotlib.colors import TwoSlopeNorm                    # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch                                                  # noqa: E402
from wavesplat.run import DEFAULT_CONFIG                      # noqa: E402
from wavesplat.data_loader import EpisodeDataLoader           # noqa: E402
from wavesplat.optimizer.sequential_update import make_updater  # noqa: E402

# ⚠`fig_qualitative` 와 **같은 에피소드**를 쓴다 --- 독자가 정지 그림과 애니메이션을
#   맞춰 볼 수 있고, 에피소드 선택이 결과를 고른 것이 아님이 드러난다.
EPISODE = 'data_s8_crowded_2026-08-10/episode_BF4_15comp_seed32_uniform_ra_rf_ds.h5'
TORCH_SEED = 42
STRIDE = 2          # 추적 프레임 2개마다 한 장 (10 Hz -> 5 fps 상당)
FPS = 8
OUT = os.path.join(ROOT, 'paper', 'figures', 'wave_estimate.gif')


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

    # ── 추적을 돌면서 매 프레임의 (참, 추정, 관측) 을 모은다 ────────────────
    snaps = []
    for i, f in enumerate(trk):
        upd.update(f)                       # ⛔추적은 **전 프레임** 돈다(건너뛰지 않는다)
        if i % STRIDE:
            continue
        with torch.no_grad():
            est = prim.model(xf, yf, torch.tensor(float(f.t))).cpu().numpy()
        gt = f.gt_field.cpu().numpy()
        est = est.reshape(gt.shape)
        snaps.append(dict(t=float(f.t), gt=gt, est=est,
                          xo=f.x_obs.cpu().numpy(), yo=f.y_obs.cpu().numpy(),
                          rmse=float(np.sqrt(((est - gt) ** 2).mean()))))
    assert snaps, '⛔추적 프레임이 없다'

    lim = max(float(np.abs(s['gt']).max()) for s in snaps)
    elim = max(float(np.abs(s['est'] - s['gt']).max()) for s in snaps)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    enorm = TwoSlopeNorm(vmin=-elim, vcenter=0.0, vmax=elim)

    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.1))
    ims, dots = [], []
    for a, (F, ttl, nm, cm) in zip(ax, (
            (snaps[0]['gt'], 'true wave field', norm, 'RdBu_r'),
            (snaps[0]['est'], 'estimate: WavePrior atoms, tracked', norm, 'RdBu_r'),
            (snaps[0]['est'] - snaps[0]['gt'], 'estimate $-$ truth', enorm, 'PuOr'))):
        im = a.pcolormesh(XX, YY, F, cmap=cm, norm=nm, shading='auto')
        ims.append(im)
        a.set_title(ttl, fontsize=9)
        a.set_aspect('equal')
        a.set_xlabel('x [m]', fontsize=8)
        a.tick_params(labelsize=7)
        cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6)
        cb.set_label('elevation [m]', fontsize=6.5)
        d, = a.plot([], [], '.', ms=2.2, color='k', alpha=.85)
        dots.append(d)
    ax[0].set_ylabel('y [m]', fontsize=8)
    # ★오차 패널의 컬러바는 **훨씬 촘촘하다** --- 그 배수를 적어 둔다(오독 방지).
    fig.axes[-1].set_ylabel(f'error [m]  (scale $\\times${lim / elim:.1f} finer)',
                            fontsize=6.5)
    txt = fig.text(0.5, 0.965, '', ha='center', fontsize=9.5)
    sub = fig.text(0.5, 0.02, '', ha='center', fontsize=7.2, color='#555')

    def draw(k):
        s = snaps[k]
        ims[0].set_array(s['gt'].ravel())
        ims[1].set_array(s['est'].ravel())
        ims[2].set_array((s['est'] - s['gt']).ravel())
        for d in dots:
            d.set_data(s['xo'], s['yo'])
        txt.set_text(f"t = {s['t']:5.1f} s      "
                     f"frame RMSE = {s['rmse']:.3f} m      "
                     f"{len(s['xo'])} LiDAR returns this frame")
        sub.set_text('black dots: every observation this frame. WavePrior (a learned '
                     'set-valued prior) places the atoms once at initialisation; the '
                     'field is then evaluated everywhere from those spectral parameters')
        return ims + dots + [txt, sub]

    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    ani = FuncAnimation(fig, draw, frames=len(snaps), blit=False)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ani.save(OUT, writer=PillowWriter(fps=FPS), dpi=110)

    prov = dict(
        episode=EPISODE, torch_seed=TORCH_SEED, git_head=git_head(),
        n_frames_animated=len(snaps), stride=STRIDE, fps=FPS,
        t_first=snaps[0]['t'], t_last=snaps[-1]['t'],
        init_seconds=round(init_s, 2),
        final_n=int(prim.model.n_gaussians),
        route=getattr(prim, '_auto_selected', None) or prim.init_method,
        rmse_median=float(np.median([s['rmse'] for s in snaps])),
        rmse_min=min(s['rmse'] for s in snaps),
        rmse_max=max(s['rmse'] for s in snaps),
        obs_per_frame_median=float(np.median([len(s['xo']) for s in snaps])),
        colour_scale_ratio=round(float(lim / elim), 1),
        config={k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                for k, v in sorted(cfg.items())},
        note=('ILLUSTRATIVE ONLY --- one episode, one arm. No comparative claim. '
              'Quantitative claims live in the frozen verdict files behind tab:earns.'))
    json.dump(prov, open(os.path.join(ROOT, 'paper', 'figures',
                                      'wave_estimate.provenance.json'), 'w'),
              indent=2, ensure_ascii=False)
    print('wrote', OUT)
    print(f"  {len(snaps)} frames · t {snaps[0]['t']:.1f}–{snaps[-1]['t']:.1f} s · "
          f"route {prov['route']} · N={prov['final_n']}")
    print(f"  frame RMSE  median {prov['rmse_median']:.4f} m  "
          f"(min {prov['rmse_min']:.4f} · max {prov['rmse_max']:.4f})")
    print(f"  obs/frame median {prov['obs_per_frame_median']:.0f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
