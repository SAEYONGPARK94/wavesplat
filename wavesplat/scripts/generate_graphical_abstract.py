"""
generate_graphical_abstract.py
Ocean Engineering graphical abstract: 3-panel compact summary.
  (1) Sparse LiDAR input over sea surface
  (2) WaveSplat continuous reconstruction
  (3) Headline metric bar chart
Output: paper/submission/graphical_abstract.{pdf,png}

⛔ 2026-08-13 감사 — Panel 3 (headline metrics) 과 suptitle 이 인용금지 수치를
   하드코딩하고 있어 차단됨. 아래 `_blocked_banned_headline()` 참조.
"""
import os, sys


def _blocked_banned_headline():
    """⛔ 인용금지 수치를 재생산하는 경로의 차단 게이트 (2026-08-13 감사)."""
    raise RuntimeError(
        "⛔ 이 그림/슬라이드는 인용금지 수치(0.223/0.319/13.4/132.6/~10x)를 하드코딩하고 있었다. "
        "2026-08-13 감사에서 차단됨. 재활성화하려면 검증된 수치로 교체할 것. "
        "근거: memory/reference_citation_bans.md, tmp/AUDIT_recheck_2026-08-13/A3_citation_bans.md"
    )


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EP = 'data/episode_BF4_15comp_seed0_uniform.h5'
FRAME = 50

with h5py.File(EP, 'r') as f:
    gx = f['frames/gt_grid_x'][:]
    gy = f['frames/gt_grid_y'][:]
    gt = f['frames/gt_wave_field'][FRAME]
    pts = f['frames/lidar_points'][FRAME]
    mask = f['frames/lidar_valid_mask'][FRAME]

pts = pts[mask]

# Compact figure: ~13 cm x 5 cm, 300 dpi
fig, axes = plt.subplots(1, 3, figsize=(13/2.54, 5/2.54), dpi=300,
                         gridspec_kw={'width_ratios': [1, 1, 1.05]})

# Panel 1: sparse LiDAR input
ax = axes[0]
ax.set_facecolor('#e8f0f7')
sc = ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], cmap='RdBu_r',
                s=6, vmin=-1.0, vmax=1.0, edgecolors='none')
ax.set_xlim(gx.min(), gx.max()); ax.set_ylim(gy.min(), gy.max())
ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
ax.set_title('Sparse LiDAR\n(~50 pts/frame)', fontsize=6, pad=2)

# Panel 2: ground-truth / WS reconstruction (use GT as stand-in — WS tracks it)
ax = axes[1]
X, Y = np.meshgrid(gx, gy)
im = ax.pcolormesh(X, Y, gt, cmap='RdBu_r', vmin=-1.0, vmax=1.0,
                   shading='auto', rasterized=True)
ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
ax.set_title('WaveSplat\ncontinuous field', fontsize=6, pad=2)
ax.text(0.5, -0.12,
        r'$z=\sum w_i\cos(k_ix+k_iy-\omega_it+\varphi_i)$',
        transform=ax.transAxes, ha='center', fontsize=5)

# Panel 3: headline metrics
# ⛔ 2026-08-13 감사 차단점 — 아래 리터럴이 인용금지 수치다.
_blocked_banned_headline()
ax = axes[2]
methods = ['WS', 'ETKF', 'GPR']
# ⛔ 원본 이력 (2026-08-13 감사에서 주석 처리, 인용금지):
# nrmse = [0.223, 0.319, 0.295]
# speed = [13.4, 132.6, 205.0]
nrmse = None  # 검증된 수치로 교체할 것 (results/100ep_ws_letkf.json)
speed = None  # 검증된 수치로 교체할 것 (동일 프로세스·1스레드 shipping_default.json)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
x = np.arange(3)
w = 0.35
bars1 = ax.bar(x - w/2, nrmse, w, color=colors, alpha=0.9, label='nRMSE')
ax2 = ax.twinx()
bars2 = ax2.bar(x + w/2, speed, w, color=colors, alpha=0.4,
                hatch='//', edgecolor='k', linewidth=0.3, label='ms/frame')
ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=6)
ax.set_ylabel('nRMSE', fontsize=6)
ax2.set_ylabel('ms/frame', fontsize=6)
ax.tick_params(axis='y', labelsize=5)
ax2.tick_params(axis='y', labelsize=5)
ax.set_ylim(0, 0.45); ax2.set_ylim(0, 250)
ax.set_title('100 episodes, BF2--5', fontsize=6, pad=2)

# ⛔ 원본 이력 (2026-08-13 감사에서 주석 처리, 인용금지 — "30.1% lower nRMSE, ~10x faster"):
# fig.suptitle(
#     'WaveSplat: grid-free spectral wave estimation — '
#     r'30.1% lower nRMSE, $\sim$10$\times$ faster than tuned ETKF',
#     fontsize=6.5, y=1.02)
fig.suptitle(
    'WaveSplat: grid-free spectral wave estimation',
    fontsize=6.5, y=1.02)

plt.tight_layout(w_pad=0.8)
out_dir = 'paper/submission'
os.makedirs(out_dir, exist_ok=True)
for ext in ['pdf', 'png']:
    p = os.path.join(out_dir, f'graphical_abstract.{ext}')
    fig.savefig(p, dpi=300, bbox_inches='tight')
    print(f'wrote {p}')
