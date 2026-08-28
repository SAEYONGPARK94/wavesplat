"""합성(시뮬레이션) 추정 결과 GIF — **배포 팔**, HOS M=3 비선형 참값, 혼잡 층.

## 자료·팔·채점 (전부 캡션에 박는다)
- 코퍼스 `tmp/HOSGATE_2026-08-13/data_hos/` = `B-11`/`B-12` 가 쓴 **HOS M=3 비선형 참값**.
  ⟹ ⛔*"선형 GT"* 가 아니다. 이 코퍼스는 「합성 GT 가 추정기 모델식과 100% 동일」 단서를 제거한다.
- 에피소드 선택 = **결과에서 고르지 않는다**: `B-12` 동결 결과(`results_b12.jsonl`) 48 혼잡
  에피소드의 배포 팔 median-of-3 `avg_nRMSE` 의 **중앙값에 가장 가까운 에피소드**
  (동점은 이름 오름차순). ⟹ `BF5_30comp_seed1638` (0.13419 vs 48개 중앙 0.13460).
  ⚠**최고도 최악도 아니다** — 48개 중 24위(48개 범위 0.0593–0.2412).
- 추정기 시드 = **median seed**(42/43/44 중 avg_nRMSE 중앙 = 44). 저장소 관례.
- ⚠**오늘 코드는 그 동결 행을 재현하지 않는다**: 같은 (에피소드, 시드)가 오늘 `0.1593`,
  `B-12` 동결 행은 `0.13419`. 08-23 이후 패키지가 움직였고(그 사이 커밋 13개, HEAD 의
  「전역 RNG 결합」 조사 포함) 선택은 **동결 수로** 했으므로 오늘 결과를 보고 고른 것이 아니다.
  ⟹ ★**이 GIF 는 `B-12` 의 런이 아니라 오늘의 배포 경로다.**
- 팔 = `wavesplat.run.DEFAULT_CONFIG` 그대로 (2026-08-22 배포값:
  `varpro_steps=400`, `learned_init_iters=0`, `auto_switch_thresh=0`) · `device='cpu'`.
- 채점 = **`nRMSE = RMSE / Hs_true`** (합성 규약; ⛔실측의 σ 규약과 직접 비교 금지)
  기준선 = **참 제로필드**, 즉 같은 채점 함수에 `Z=0` 을 통과시킨 값 —
  상수로 선언하지 않는다(2026-08-23 `AMENDMENT_ZEROREF.md` 의 처방).

## ⛔이 GIF 로 주장할 수 없는 것
- 외부 기준선(LETKF/ETKF) 이 이 그림에 **없다**. 화면의 어떤 것도 *"경쟁자보다 낫다"* 가 아니다.
- 한 에피소드 · 한 시드다. 크기(%)를 인용하지 마라 — 인용 가능한 크기는 `B-12` 의 집계다.
"""
import json
import os
import re
import statistics as st
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = ''          # ⛔GPU 금지 (M1 생성이 점유 중)
os.environ.setdefault('OMP_NUM_THREADS', '4')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib import font_manager                              # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))     # .../wavesplat/wavesplat
sys.path.insert(0, ROOT)
os.chdir(ROOT)

for _f in ('/home/vexit/.local/share/fonts/malgun.ttf',):
    if os.path.exists(_f):
        font_manager.fontManager.addfont(_f)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=_f).get_name()
plt.rcParams['axes.unicode_minus'] = False

HOS = os.path.join(ROOT, 'tmp', 'HOSGATE_2026-08-13', 'data_hos')
B12 = os.path.join(ROOT, 'tmp', 'B12_HOS_EXTEND_2026-08-23', 'results_b12.jsonl')
OUT = os.path.join(HERE, 'synthetic_crowded_hos.gif')
INFO = os.path.join(HERE, 'synthetic_crowded_hos.json')
N_GIF_FRAMES, FPS, DPI = 60, 8, 78


def pick_episode():
    """⛔결과에서 고르지 않는다 — 동결 배치의 중앙 성능 에피소드를 규칙으로 집는다."""
    per = {}
    for line in open(B12):
        r = json.loads(line)
        if 'error' in r or r['arm'] != 'SHIP-NOW':
            continue
        per.setdefault(r['episode'], {})[r['est_seed']] = r['avg_nRMSE']
    med = {k: st.median(v.values()) for k, v in per.items()}
    m = st.median(med.values())
    ep = sorted(med, key=lambda k: (abs(med[k] - m), k))[0]
    seeds = per[ep]
    med_seed = sorted(seeds, key=lambda s: seeds[s])[len(seeds) // 2]
    return ep, med_seed, med[ep], m, seeds


def run(path, seed):
    from wavesplat.run import DEFAULT_CONFIG, set_seed
    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.optimizer.sequential_update import make_updater
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(seed=seed, device='cpu')
    set_seed(seed)
    loader = EpisodeDataLoader(path, device='cpu')
    cfg['depth'] = loader.depth
    init_frames = loader.get_init_window(tp_factor=2.0)
    upd = make_updater(cfg)
    upd.initialize(init_frames,
                   x_range=(float(loader.gt_grid_x[0]), float(loader.gt_grid_x[-1])),
                   y_range=(float(loader.gt_grid_y[0]), float(loader.gt_grid_y[-1])),
                   verbose=False)
    gx, gy = loader.gt_grid_x, loader.gt_grid_y
    out = []
    for fr in loader.get_tracking_frames(tp_factor=2.0):
        upd.update(fr)
        pred = upd.predict_field(gx, gy, fr.t).cpu().numpy()
        gt = fr.gt_field.cpu().numpy()
        out.append(dict(t=fr.t, pred=pred, gt=gt,
                        ox=fr.x_obs.cpu().numpy(), oy=fr.y_obs.cpu().numpy()))
    return loader, cfg, out


def main():
    ep, seed, ep_med, corpus_med, seeds = pick_episode()
    path = os.path.join(HOS, f'episode_{ep}.h5')
    print(f'episode {ep}  seed {seed}  (ep median {ep_med:.5f} / corpus median '
          f'{corpus_med:.5f})', flush=True)
    loader, cfg, fr = run(path, seed)
    Hs = loader.Hs_true
    n = np.array([np.sqrt(np.mean((f['pred'] - f['gt']) ** 2)) / Hs for f in fr])
    z = np.array([np.sqrt(np.mean(f['gt'] ** 2)) / Hs for f in fr])       # ★참 제로필드
    ts = np.array([f['t'] for f in fr])
    print(f'  frames {len(fr)}  avg_nRMSE {n.mean():.4f}  zero {z.mean():.4f}  '
          f'ratio {n.mean() / z.mean():.3f}', flush=True)

    gx = loader.gt_grid_x.cpu().numpy()
    gy = loader.gt_grid_y.cpu().numpy()
    ext = [gx[0], gx[-1], gy[0], gy[-1]]
    lim = float(np.percentile(np.abs([f['gt'] for f in fr]), 99.5))
    elim = lim * 0.6
    idx = np.unique(np.linspace(0, len(fr) - 1, min(N_GIF_FRAMES, len(fr))).astype(int))

    fig = plt.figure(figsize=(13.6, 4.35), facecolor='white')
    gs = fig.add_gridspec(1, 4, left=.037, right=.985, top=.72, bottom=.175,
                          wspace=.30, width_ratios=[1, 1, 1, 1.25])
    ax = [fig.add_subplot(g) for g in gs]
    kw = dict(extent=ext, origin='lower', aspect='equal', interpolation='bilinear')
    im0 = ax[0].imshow(fr[0]['gt'], cmap='RdBu_r', vmin=-lim, vmax=lim, **kw)
    im1 = ax[1].imshow(fr[0]['pred'], cmap='RdBu_r', vmin=-lim, vmax=lim, **kw)
    im2 = ax[2].imshow(fr[0]['pred'] - fr[0]['gt'], cmap='PuOr_r',
                       vmin=-elim, vmax=elim, **kw)
    sc = ax[0].scatter(fr[0]['ox'], fr[0]['oy'], s=3.0, c='#111111', alpha=.75, lw=0)
    for a, t in zip(ax[:3], ['참값 GT (HOS M=3, 비선형)', '추정 (배포 팔)',
                             '오차 = 추정 - 참값']):
        a.set_title(t, fontsize=9.5, pad=4)
        a.set_xlabel('x [m]', fontsize=8)
        a.tick_params(labelsize=7)
    ax[0].set_ylabel('y [m]', fontsize=8)
    fig.colorbar(im0, ax=ax[1], fraction=.036, pad=.02).ax.tick_params(labelsize=6.5)
    fig.colorbar(im2, ax=ax[2], fraction=.036, pad=.02).ax.tick_params(labelsize=6.5)

    ax[3].plot(ts, z, color='#999999', lw=1.4, label='참 제로필드 (Z=0 을 같은 채점기에)')
    ax[3].plot(ts, n, color='#c1440e', lw=1.7, label='배포 팔 WaveSplat')
    mk, = ax[3].plot([ts[0]], [n[0]], 'o', color='#c1440e', ms=5.5)
    ax[3].set_ylim(0, max(z.max(), n.max()) * 1.18)
    ax[3].set_xlabel('추적 시각 t [s]', fontsize=8)
    ax[3].set_ylabel('nRMSE  (RMSE / Hs)', fontsize=8)
    ax[3].set_title('채점: 합성 규약 = RMSE / Hs_true', fontsize=9.5, pad=4)
    ax[3].tick_params(labelsize=7)
    ax[3].legend(fontsize=6.8, loc='upper right', framealpha=.9)
    ax[3].grid(alpha=.25, lw=.5)

    head = fig.text(.037, .955, '', fontsize=11.5, ha='left', va='top')
    sub = fig.text(.037, .885, '', fontsize=7.6, ha='left', va='top', color='#333333')
    foot = fig.text(.037, .010, '', fontsize=6.6, ha='left', va='bottom', color='#666666')
    m_ep = re.match(r'(BF\d)_(\d+)comp_seed(\d+)_', ep)
    bf, ncomp, gseed = m_ep.group(1), int(m_ep.group(2)), int(m_ep.group(3))
    head.set_text('합성 (시뮬레이션) - 혼잡 · 비선형 HOS 참값에서 배포 팔은 참값을 따라간다')
    sub.set_text(
        f'자료: {ep}.h5  (HOSGATE / B-11·B-12 코퍼스.  참값 = HOS M=3 비선형 - 선형 GT 가 아니다. '
        f'생성 시드 {gseed} = 학습 코퍼스 밖)   |   {bf}, 성분 {ncomp}개(= 혼잡 층), '
        f'Hs={Hs:.2f} m, Tp={loader.Tp_true:.1f} s, depth={loader.depth:.0f} m\n'
        f'팔: 배포 기본값 wavesplat.run.DEFAULT_CONFIG (2026-08-22 배포: varpro_steps=400, '
        f'learned_init_iters=0, auto_switch_thresh=0), 추정 시드 {seed} (= B-12 3시드 중 median), CPU\n'
        f'채점: nRMSE = RMSE / Hs_true (합성 규약).  기준선 = 참 제로필드 - 상수가 아니라 '
        f'같은 채점기에 Z=0 을 통과시킨 값.   구간 평균 {n.mean():.3f} vs 제로필드 {z.mean():.3f}')
    foot.set_text(
        '선택 사유: B-12 동결 결과 48개 혼잡 에피소드 중 배포 팔 median-of-3 avg_nRMSE 가 '
        '48개의 중앙값에 가장 가까운 에피소드 - 잘 나온 것을 고르지 않았다 (48개 범위 0.059-0.241).\n'
        '주장할 수 없는 것: 이 그림에 외부 기준선(LETKF/ETKF)은 없다 - "경쟁자보다 낫다" 가 아니다.  '
        '한 에피소드·한 시드이므로 % 크기를 인용하지 마라 (인용 가능한 크기는 B-12 의 48건 집계다).  '
        f'오늘 코드의 이 런은 {n.mean():.4f} 로 B-12 동결 행({ep_med:.4f})과 다르다 - 08-23 이후 패키지가 움직였다.')

    def upd_f(k):
        i = idx[k]
        im0.set_data(fr[i]['gt'])
        im1.set_data(fr[i]['pred'])
        im2.set_data(fr[i]['pred'] - fr[i]['gt'])
        sc.set_offsets(np.column_stack([fr[i]['ox'], fr[i]['oy']]))
        mk.set_data([ts[i]], [n[i]])
        ax[1].set_title(f'추정 (배포 팔)   nRMSE={n[i]:.3f}', fontsize=9.5, pad=4)
        ax[0].set_title(f'참값 GT (HOS M=3, 비선형)   t={ts[i]:.1f} s\n'
                        f'점 = LiDAR 관측 {len(fr[i]["ox"])}개', fontsize=9.5, pad=4)
        ax[2].set_title(f'오차 = 추정 - 참값   제로필드={z[i]:.3f}', fontsize=9.5, pad=4)
        return im0, im1, im2, sc, mk

    ani = FuncAnimation(fig, upd_f, frames=len(idx), blit=False)
    ani.save(OUT, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    json.dump(dict(episode=ep, path=path, est_seed=seed, seeds=seeds,
                   episode_median_nRMSE=ep_med, corpus_median_nRMSE=corpus_med,
                   Hs=Hs, Tp=loader.Tp_true, depth=loader.depth,
                   n_tracking_frames=len(fr), n_gif_frames=len(idx),
                   avg_nRMSE_this_run=float(n.mean()),
                   zero_field_nRMSE=float(z.mean()),
                   scoring='RMSE / Hs_true (synthetic convention)',
                   gt_kind='HOS M=3 nonlinear',
                   shipped=dict(varpro_steps=cfg.get('varpro_steps'),
                                learned_init_iters=cfg.get('learned_init_iters'),
                                auto_switch_thresh=cfg.get('auto_switch_thresh'),
                                init_method=cfg.get('init_method'),
                                lambda_l1=cfg.get('lambda_l1'))),
              open(INFO, 'w'), indent=1)
    print('wrote', OUT, os.path.getsize(OUT) // 1024, 'KB')


if __name__ == '__main__':
    main()
