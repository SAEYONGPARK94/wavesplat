#!/usr/bin/env python3
"""`paper/figures/fig_pipeline_deployed.pdf` 생성 — **배포되는 경로**의 파이프라인 그림.

⛔이 스크립트가 존재하는 이유(2026-09-02, P9):
초판 그림 `fig_pipeline.pdf` 는 *"applies dispersion-aware beamforming with fine
refinement, and optimises via coarse-to-fine with adaptive N"* 를 그렸는데,
배포 기본값(`init_method='auto'`, `auto_switch_thresh=0`)에서 라우터가 `'learned'` 로
해소되면 **그 경로는 도달 불가**다(`sequential_update.py:832` 의 return 아래).
⟹ 논문의 가장 많이 읽히는 그림이 **돌지 않는 알고리즘**을 그리고 있었다.

★상수는 손으로 적지 않는다 — 소스에서 읽는다. 어긋나면 죽는다.
"""
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from wavesplat.optimizer.sequential_update import (           # noqa: E402
    VARPRO_STEPS_DEFAULT, LEARNED_INIT_ITERS_DEFAULT, LAMBDA_L1_DEFAULT,
    AUTO_SWITCH_THRESH_DEFAULT, DEFAULT_LEARNED_ENSEMBLE_K)
from wavesplat.run import DEFAULT_CONFIG as DC                 # noqa: E402
import wavesplat.optimizer.learned_init as LI                  # noqa: E402

# ── ★소스에서 읽은 상수 (⛔손으로 적지 마라)
VP = VARPRO_STEPS_DEFAULT
JOINT = LEARNED_INIT_ITERS_DEFAULT
TRK_STEPS, TRK_LR, TRK_WIN = DC['tracking_steps'], DC['tracking_lr'], DC['tracking_window']
MAXN = DC['max_n']
# ⚠`nsamp`/`dedup_dk` 는 모듈 상수가 아니라 **함수 서명의 기본값**이다.
#   ⛔손으로 옮겨 적지 말고 서명에서 읽는다 --- 그래야 바뀌면 그림이 따라온다.
import inspect
NSAMP = inspect.signature(LI.propose_atoms).parameters['nsamp'].default
DEDUP = inspect.signature(LI.pool_and_select).parameters['dedup_dk'].default
ETHR = inspect.signature(LI.pool_and_select).parameters['energy_thresh'].default
MOBS = LI.M_OBS_CAP

assert AUTO_SWITCH_THRESH_DEFAULT == 0, '라우터가 무조건 학습을 고르지 않는다 — 그림을 다시 그려라'
assert DEFAULT_LEARNED_ENSEMBLE_K == 1, '학습 경로 앙상블이 켜졌다 — 그림에 median-of-k 를 넣어야 한다'
assert JOINT == 0, '결합 적합이 켜졌다 — 그림의 회색 상자를 살려라'

INK, MUTE, ACC = '#1a1a1a', '#8a8a8a', '#0b5394'
BOXW, BOXH = 1.02, 0.42


def box(ax, x, y, txt, *, w=BOXW, h=BOXH, fc='white', ec=INK, fs=7.0, style='round,pad=0.02',
        tc=INK, lw=1.0, ls='-'):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=style,
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
    ax.text(x, y, txt, ha='center', va='center', fontsize=fs, color=tc, zorder=3, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, *, color=INK, ls='-', lw=1.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=9,
                                 color=color, lw=lw, ls=ls, shrinkA=1, shrinkB=1, zorder=1))


fig, ax = plt.subplots(figsize=(7.6, 3.05))
ax.set_xlim(0, 7.6); ax.set_ylim(0, 3.05); ax.axis('off')

ax.text(0.06, 2.92, 'Phase 1 — initialization  (frames with $t \\leq 2T_p$)',
        fontsize=8.2, weight='bold', color=INK, va='top')
ax.text(0.06, 1.30, 'Phase 2 — tracking  (每 frame)'.replace('每 ', 'per '),
        fontsize=8.2, weight='bold', color=INK, va='top')

y1 = 2.35
xs = [0.62, 1.86, 3.10, 4.34, 5.58, 6.86]
box(ax, xs[0], y1, f'LiDAR frames\n$\\leq{MOBS}$ pts\n(proposal only)')
box(ax, xs[1], y1, 'beamform map\n$64\\times64$\n$\\pm0.6$ rad/m')
box(ax, xs[2], y1, f'conditional\ndiffusion\n{NSAMP} draws $\\times$ 32', ec=ACC, tc=ACC, lw=1.3)
box(ax, xs[3], y1, f'polish + pool\ndrop $\\Delta k<{DEDUP}$\ncap $N\\leq{MAXN}$', ec=ACC, tc=ACC, lw=1.3)
box(ax, xs[4], y1, 'ridge LS\n$(A,\\varphi)$\nclosed form')
box(ax, xs[5], y1, f'VARPRO\n{VP} steps\non $(k_x,k_y)$')
for a, b in zip(xs[:-1], xs[1:]):
    arrow(ax, a + BOXW / 2, y1, b - BOXW / 2, y1)

# 적응 차수 — VARPRO **앞**에서 돈다(D6)
box(ax, xs[4] + 0.15, 1.70, 'adaptive order: prune to 90% energy,\nthen grow if residual RMS $>0.05$ m',
    w=3.0, h=0.42, fs=6.8, fc='#f2f2f2', ec=MUTE, tc=INK)
arrow(ax, xs[4], y1 - BOXH / 2, xs[4], 1.70 + 0.21, color=MUTE, ls=':')
arrow(ax, xs[4] + 1.65, 1.70, xs[5], y1 - BOXH / 2, color=MUTE, ls=':')

# 배포에서 꺼진 것 — **그림에 남기되 꺼졌다고 적는다**
box(ax, 1.60, 1.70, f'joint non-linear fit: disabled ({JOINT} iters)',
    w=2.5, h=0.32, fs=6.8, fc='white', ec=MUTE, tc=MUTE, ls=(0, (3, 2)))

y2 = 0.72
from math import log10
_lam = f'10^{{{int(round(log10(LAMBDA_L1_DEFAULT)))}}}'
box(ax, 0.68, y2, f'window of\n{TRK_WIN} frames', w=1.05)
box(ax, 2.15, y2, f'freeze $(k_x,k_y)$\n{TRK_STEPS} Adam steps\nlr $={TRK_LR:g}$', w=1.45)
box(ax, 3.95, y2, f'loss: weighted MSE\n$+\\ \\lambda_1\\|w\\|_1$, $\\lambda_1={_lam}$\n$+$ spectral smoothness', w=1.95)
box(ax, 5.72, y2, 'steepness\nclamp 0.15', w=1.15)
box(ax, 7.00, y2, 'field at any\n$(x,y,t)$', w=1.05, ec=ACC, tc=ACC, lw=1.3)
for a, b in [(0.68 + 0.525, 2.15 - 0.725), (2.15 + 0.725, 3.95 - 0.975),
             (3.95 + 0.975, 5.72 - 0.575), (5.72 + 0.575, 7.00 - 0.525)]:
    arrow(ax, a, y2, b, y2)
ax.text(3.80, 0.22, 'no explicit phase advance: the forward model\'s $-\\omega_i t$ term carries the propagation',
        ha='center', fontsize=6.6, color=MUTE, style='italic')

# Phase 1 -> Phase 2 전이. ⚠긴 대각선은 'adaptive order' 상자와 Phase 2 제목을
#   가로질러 **그 상자에서 나온 것처럼 보였다** ⟹ 왼쪽 여백으로 우회시킨다.
ax.plot([xs[5] + BOXW / 2 + 0.06, 7.45, 7.45, 0.20, 0.20],
        [y1, y1, 1.44, 1.44, y2 + BOXH / 2 + 0.14],
        color=MUTE, ls='--', lw=0.9, zorder=0, solid_capstyle='round')
arrow(ax, 0.20, y2 + BOXH / 2 + 0.16, 0.60, y2 + BOXH / 2 + 0.01, color=MUTE, ls='--', lw=0.9)

out = os.path.join(ROOT, 'paper', 'figures', 'fig_pipeline_deployed.pdf')
fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
print('wrote', out)
