"""S9 적용 전/후 비교 GIF — 진실면(GT) 포함.

변경 = 출하 기본값 `lambda_l1` 1e-3 -> 1e-4 (S9 판정 `BOTH`).

## 에피소드 선정 — ⛔체리피킹 아님
각 레짐에서 **짝 Δ(개선량)의 중앙값에 해당하는 에피소드**를 규칙으로 고른다.
최고 성적 에피소드가 아니라 **전형적인** 에피소드다. (선정 근거는 `results.jsonl`.)

## 그림 구성
윗줄: 진실면 | 변경 전 | 변경 후  (같은 색 스케일)
아랫줄 왼쪽: 오차 지도(변경 전 − 진실) / (변경 후 − 진실), 같은 스케일
아랫줄 오른쪽: ★**단면도** — 진실면을 **검은 실선**, 변경 전 점선, 변경 후 실선으로 겹쳐 그린다.

⚠GT 가 있는 **시뮬레이션** 에피소드다. 실측(real) 데이터엔 GT 가 없어 이 그림을 만들 수 없다.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

import wavesplat.run as WR  # noqa: E402

EPISODES = {
    'crowded': 'data_s9_crowded_2026-08-11/episode_BF5_30comp_seed50_uniform_ra_rf_ds.h5',
    'sparse': 'data_s9_sparse_2026-08-11/episode_BF4_5comp_seed54_uniform_ra_rf_ds.h5',
}
SEED = 1575
BEFORE, AFTER = 1e-3, 1e-4
OUTDIR = os.path.join(REPO, 'results', 'gifs')
MAX_FRAMES = 60


def capture(episode, lambda_l1):
    """한 번 돌리면서 프레임별 (t, 예측면, 진실면, 격자) 를 가로챈다."""
    stash = []
    orig = WR.compute_all_metrics

    def wrapped(pred_np, gt_np, gx, gy, *a, **kw):
        stash.append((pred_np.copy(), gt_np.copy(), gx, gy))
        return orig(pred_np, gt_np, gx, gy, *a, **kw)

    WR.compute_all_metrics = wrapped
    try:
        cfg = dict(WR.DEFAULT_CONFIG)
        cfg['lambda_l1'] = lambda_l1
        cfg['seed'] = SEED
        r = WR.run_estimation(episode, cfg)
    finally:
        WR.compute_all_metrics = orig
    return stash, r


def make(regime, episode):
    print(f'\n=== {regime}: {os.path.basename(episode)} ===', flush=True)
    print(f'  변경 전 (lambda_l1={BEFORE:.0e}) 실행...', flush=True)
    sb, rb = capture(episode, BEFORE)
    print(f'  변경 후 (lambda_l1={AFTER:.0e}) 실행...', flush=True)
    sa, ra = capture(episode, AFTER)

    n = min(len(sb), len(sa))
    if n == 0:
        print('  ⛔ 프레임 캡처 실패')
        return None
    idx = np.linspace(0, n - 1, min(MAX_FRAMES, n)).astype(int)

    nb = rb.get('avg_nRMSE', float('nan'))
    na = ra.get('avg_nRMSE', float('nan'))
    print(f'  에피소드 평균 nRMSE: 변경전 {nb:.5f} -> 변경후 {na:.5f} '
          f'({(na-nb)/nb*100:+.1f}%)', flush=True)

    gt0 = sb[0][1]
    vmax = float(np.percentile(np.abs(gt0), 99)) or 1.0
    gx = sb[0][2]
    xs = gx[0] if gx.ndim == 2 else gx
    mid = gt0.shape[0] // 2

    fig = plt.figure(figsize=(13, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.85], hspace=0.32, wspace=0.25)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axe = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    axl = fig.add_subplot(gs[1, 2])

    ims, ime = [], []
    for ax, ttl in zip(axes, ['Ground truth', f'BEFORE  lambda_l1={BEFORE:.0e}',
                              f'AFTER  lambda_l1={AFTER:.0e}']):
        im = ax.imshow(gt0, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower',
                       animated=True)
        ax.set_title(ttl, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ims.append(im)
    for ax, ttl in zip(axe, ['Error: BEFORE - GT', 'Error: AFTER - GT']):
        im = ax.imshow(np.zeros_like(gt0), cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       origin='lower', animated=True)
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ime.append(im)

    lg, = axl.plot(xs, gt0[mid], 'k-', lw=2.4, label='Ground truth (solid)')
    lb, = axl.plot(xs, gt0[mid], '--', color='#d62728', lw=1.5, label='BEFORE (1e-3)')
    la, = axl.plot(xs, gt0[mid], '-', color='#1f77b4', lw=1.8, label='AFTER (1e-4)')
    axl.set_ylim(-vmax * 1.35, vmax * 1.35)
    axl.set_title('Cross-section at mid-y', fontsize=10)
    axl.set_xlabel('x [m]', fontsize=9)
    axl.set_ylabel('η [m]', fontsize=9)
    axl.legend(fontsize=8, loc='upper right')
    axl.grid(alpha=0.3)

    sup = fig.suptitle('', fontsize=12)

    def upd(k):
        i = idx[k]
        pb, g, _, _ = sb[i]
        pa = sa[i][0]
        ims[0].set_data(g); ims[1].set_data(pb); ims[2].set_data(pa)
        ime[0].set_data(pb - g); ime[1].set_data(pa - g)
        lg.set_ydata(g[mid]); lb.set_ydata(pb[mid]); la.set_ydata(pa[mid])
        eb = np.sqrt(np.mean((pb - g) ** 2))
        ea = np.sqrt(np.mean((pa - g) ** 2))
        sup.set_text(f'{regime}  ·  frame {i+1}/{n}  ·  '
                     f'RMSE  before {eb:.4f} m -> after {ea:.4f} m  '
                     f'({(ea-eb)/eb*100:+.1f}%)')
        return ims + ime + [lg, lb, la, sup]

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, f's9_lambda_l1_{regime}.gif')
    ani = animation.FuncAnimation(fig, upd, frames=len(idx), blit=False)
    ani.save(out, writer=animation.PillowWriter(fps=6), dpi=72)
    plt.close(fig)
    sz = os.path.getsize(out) / 1e6
    print(f'  ✅ {out}  ({sz:.1f} MB)', flush=True)
    return out, nb, na


if __name__ == '__main__':
    which = sys.argv[1:] or list(EPISODES)
    for rg in which:
        make(rg, EPISODES[rg])
