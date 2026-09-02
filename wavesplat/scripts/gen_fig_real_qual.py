#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""실측 **정성 그림** — 레코드 하나(`AA14`, `win=3`)에서 두 팔이 실제로 무엇을 내놓는가.

## 무엇을 돌렸고 무엇을 재집계했는가 (필수 공시)
- **돌린 것 = 추정 2회**(`AA14` × {`SHIP`, `PROP`}). `tmp/B2_REAL_2026-08-23/run_b2.py:23-61`
  의 호출 사슬을 **축자로 복제**한다(장면·팔 구성·시드·채점식 전부 동일).
  새 구성을 만들지 않는다 — 팔은 `gates_b2.ARMS` 단일출처에서 읽는다.
- **재집계한 것 = 0**. 이 그림은 `results.jsonl` 의 수를 다시 계산하지 않고,
  **재현했는지 assert 한다**(아래 §핀).

## ★핀 — 동결 행과 일치하지 않으면 그림을 저장하지 않는다
`tmp/B2_REAL_2026-08-23/results.jsonl` 의 `AA14` 두 행과 대조한다:
`obs_hash`·`q_hash`(같은 자료를 받았나) · `n_atoms`(`G-1`/`G-2` 음성대조) · `nowcast`(종점).

★**핀이 공허하지 않음을 확인했다**(2026-09-02): 동결값을 `1e-4` 만 흔든 **복사본**을
`WS_B2_RESULTS` 로 물리면 `rc=1` 로 죽고 그림을 저장하지 않는다.
⛔그 시험은 **원본을 건드리지 않는다** — 다른 배치가 도는 중일 수 있다.

⚠**왜 `AA14` 인가 — 성능이 아니라 수심 때문이다.** `B-2` 는 수심 **17** 로 돌았고
2026-08-26 에 `BS` 7 레코드가 17→**30** 으로 정정됐다(`AMENDMENT_ZEROREF.md` 철회 배너).
`AA`/`LJ`/`YS` 5 레코드는 수심이 바뀌지 않았으므로 **동결 행과 비트 동일**하다
⟹ 정정에 걸리지 않는 레코드에서만 핀을 걸 수 있다. `BS` 로는 이 assert 자체가 불가능하다.
⛔레코드 선택 사유는 **오직 이것**이며, `AA14` 가 유리해서가 아니다(대형 박스 5 중 4 가
자기 참 제로필드를 이긴다 — 즉 이 레코드는 **쉬운 층**의 한 칸이다. 그림이 그것을 적는다).

## ⛔인용 금지 준수
- **`ZERO_REF = 1.0` 을 쓰지 않는다.** 기준은 `true_zero_field.json` 의 **레코드별 참값**
  (`AA14` = 1.0616). 그림의 "zero field" 선은 예측 `Z ≡ 0` 이 실제로 만드는 잔차다.
- **두 팔의 상대 크기(%)를 인쇄하지 않는다** — 짝 비교에서 허용된 것은 **부호**뿐이다
  (`+52.2%` → `+27.8%` 로 두 번 정정됐다). 절대 nRMSE·산점도·잔차만 그린다.
- **n = 1**. 이 그림은 통계가 아니라 **한 칸의 모양**이다. 12 레코드 승수는 `fig_real.pdf`.
- 관측은 실측 WASS 표면의 **합성 희소화**이지 실제 LiDAR 반환이 아니다(`PREREG.md §6`).

⚠CPU 를 다른 배치와 공유하므로 스레드를 1로 고정한다(`import torch` 보다 먼저).
"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import hashlib                                                     # noqa: E402
import json                                                        # noqa: E402
import warnings                                                    # noqa: E402

warnings.filterwarnings('ignore')
import numpy as np                                                 # noqa: E402
import matplotlib                                                  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
B2DIR = os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23')
for _p in (B2DIR, ROOT, os.path.join(ROOT, 'scripts'),
           os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)

import gates_b2 as G                                               # noqa: E402
from scene_station import scene_station                            # noqa: E402

RECORD = 'AA14'
# 개발 중 그림만 손볼 때 추정을 다시 돌리지 않기 위한 선택적 캐시.
# ⚠기본은 **캐시 없음** = 언제나 추정 2회를 실제로 돌린다(핀이 공허해지지 않도록).
CACHE = os.environ.get('WS_FIG_CACHE', '')


# ⛔동결 배치 파일은 **절대 건드리지 않는다**(다른 배치가 도는 중일 수 있다).
# 핀이 공허하지 않음을 보이려면 이 변수로 **복사본**을 가리켜라 — 원본은 그대로 둔다.
FROZEN = os.environ.get('WS_B2_RESULTS', os.path.join(B2DIR, 'results.jsonl'))


def frozen_rows():
    """`results.jsonl` 에서 이 레코드의 두 팔 행."""
    out = {}
    with open(FROZEN) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('record') == RECORD and 'error' not in r:
                out[r['cell']] = r
    return out


def estimate(sc, arm):
    """`run_b2.run()` 의 사슬을 축자 복제 — 새 구성을 발명하지 않는다."""
    import torch
    from make_gif_bank_deployed import frames_from_scene
    from wavesplat.optimizer.sequential_update import make_updater
    from wavesplat.run import DEFAULT_CONFIG, set_seed
    torch.set_num_threads(1)

    t_shift = float(np.asarray(sc['P'])[:, 2].min())
    t0 = float(sc['Pq'][0, 2]) - t_shift
    frames = frames_from_scene(sc, t_shift, dtype=torch.float32)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(depth=float(sc['depth']), seed=G.SEED, device='cpu')
    cfg.update(G.ARMS[arm])                      # ★팔은 단일출처에서만 온다
    set_seed(G.SEED)
    upd = make_updater(cfg)
    upd.initialize(frames,
                   x_range=(float(sc['gxq'].min()), float(sc['gxq'].max())),
                   y_range=(float(sc['gyq'].min()), float(sc['gyq'].max())),
                   verbose=False)
    gxq, gyq = np.asarray(sc['gxq']), np.asarray(sc['gyq'])
    Z = upd.predict_field(torch.tensor(gxq, dtype=torch.float32),
                          torch.tensor(gyq, dtype=torch.float32),
                          t0 + 0.0).cpu().numpy().reshape(len(gyq), len(gxq))
    n_atoms = getattr(getattr(upd, 'model', None), 'n_gaussians', 0) or 0
    return Z, int(n_atoms)


# ── 장면 (동결 기본 인자 그대로: win=3, cap_mode='uniform') ────────────────────
sc = scene_station(RECORD, win=G.WIN)
P, zobs = np.asarray(sc['P']), np.asarray(sc['z'])
gxq, gyq = np.asarray(sc['gxq']), np.asarray(sc['gyq'])
gt = np.asarray(sc['maps_gt'][0], float)
mm = np.isfinite(gt)
sg = float(sc['sg'])

fields, atoms = {}, {}
if CACHE and os.path.exists(CACHE):
    _c = np.load(CACHE)
    for a in G.ARM_ORDER:
        fields[a], atoms[a] = _c[a], int(_c[a + '_n'])
    print(f'(cache) {CACHE}')
else:
    for a in G.ARM_ORDER:
        fields[a], atoms[a] = estimate(sc, a)
    if CACHE:
        np.savez(CACHE, **fields, **{a + '_n': atoms[a] for a in G.ARM_ORDER})

# ── ★핀: 동결 행과 대조. 어긋나면 그림을 저장하지 않는다 ──────────────────────
FR = frozen_rows()
assert set(FR) == set(G.ARM_ORDER), f'⛔동결 행이 없다: {sorted(FR)}'
h = (lambda a: hashlib.md5(np.ascontiguousarray(a, dtype=np.float64)).hexdigest()[:12])
obs_h, q_h = h(np.column_stack([P, zobs])), h(sc['Pq'])
now = {a: float(np.sqrt(np.mean((fields[a][mm] - gt[mm]) ** 2)) / sg) for a in G.ARM_ORDER}
zero_now = float(np.sqrt(np.mean(gt[mm] ** 2)) / sg)
TZ = json.load(open(os.path.join(B2DIR, 'true_zero_field.json')))

fail = []
for a in G.ARM_ORDER:
    f = FR[a]
    if f['obs_hash'] != obs_h or f['q_hash'] != q_h:
        fail.append(f'{a} scene hash {obs_h}/{q_h} != frozen {f["obs_hash"]}/{f["q_hash"]}')
    if int(f['n_atoms'] or 0) != atoms[a]:
        fail.append(f'{a} n_atoms {atoms[a]} != frozen {f["n_atoms"]}')
    if abs(now[a] - float(f['nowcast'])) > 1e-6:
        fail.append(f'{a} nowcast {now[a]:.9f} != frozen {float(f["nowcast"]):.9f}')
if abs(zero_now - TZ[RECORD]) > 1e-9:
    fail.append(f'zero field {zero_now:.9f} != true_zero_field.json {TZ[RECORD]:.9f}')
if fail:
    raise SystemExit('⛔핀 불일치 — 그림을 저장하지 않는다:\n  ' + '\n  '.join(fail))
print(f'✅핀 통과 ({RECORD}, win={G.WIN}): ' +
      ' · '.join(f'{a} nowcast={now[a]:.6f} N={atoms[a]}' for a in G.ARM_ORDER) +
      f' · true zero field={zero_now:.4f} · obs_hash={obs_h}')

# ── 그림 ─────────────────────────────────────────────────────────────────────
ACC, WARN, MUTE = '#0b5394', '#b45309', '#8a8a8a'
NAME = {'SHIP': 'deployed arm (bank)', 'PROP': 'proposed representation'}
COL = {'SHIP': ACC, 'PROP': WARN}
vmax = float(np.nanmax(np.abs([gt[mm], fields['SHIP'][mm], fields['PROP'][mm]])))

fig = plt.figure(figsize=(9.2, 5.9))
gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.94],
                      hspace=.50, wspace=.16,
                      left=.055, right=.915, top=.885, bottom=.185)
axA = fig.add_subplot(gs[0, 0])
axm = [fig.add_subplot(gs[0, i]) for i in (1, 2, 3)]
axE = fig.add_subplot(gs[1, 0:2])
axH = fig.add_subplot(gs[1, 2:4])


def inside(ax, txt, color, y=.985):
    """패널 안쪽 주석 — 패널 밖에 쓰면 아래 행의 제목과 부딪힌다."""
    ax.text(.02, y, txt, transform=ax.transAxes, ha='left', va='top',
            fontsize=6.2, color=color, linespacing=1.4, zorder=6,
            bbox=dict(fc='white', ec='none', alpha=.78, pad=1.4))


# (a) 관측 구름 + 질의점
Xq, Yq = np.meshgrid(gxq, gyq)
axA.scatter(Xq[mm], Yq[mm], s=15, marker='+', c=MUTE, lw=.55, zorder=2)
axA.scatter(P[:, 0], P[:, 1], c=zobs, s=5.5, cmap='RdBu_r',
            vmin=-vmax, vmax=vmax, lw=0, zorder=3)
axA.set_aspect('equal')
axA.set_title('(a) what the estimator is given', fontsize=8, loc='left')
axA.set_xlabel('x [m]', fontsize=7.2)
axA.set_ylabel('y [m]', fontsize=7.2)
axA.tick_params(labelsize=6.4)
inside(axA, f'{len(zobs)} observations (dots, colour = elevation)\n'
            f'{int(mm.sum())} scored query points (grey +)', '#333')
axA.text(.5, -.27, f'{len(zobs)} samples over a {np.ptp(P[:, 2]):.0f} s memory window\n'
                   f'({len(np.unique(P[:, 2]))} distinct scan times)',
         transform=axA.transAxes, ha='center', va='top', fontsize=6.0,
         color=MUTE, linespacing=1.4)

# (b)-(d) 참값과 두 팔의 nowcast 장 (같은 색축)
notes = ['measured WASS surface\n(same colour scale in (a)-(d))'] + [f'nRMSE {now[a]:.3f}\n{atoms[a]} Gaussian components'
                                     for a in G.ARM_ORDER]
for ax, ttl, F, nt, cl in zip(
        axm,
        ['(b) truth at lead 0 s', '(c) ' + NAME['SHIP'], '(d) ' + NAME['PROP'] + ' (forced)'],
        [gt, fields['SHIP'], fields['PROP']], notes, ['#333', ACC, WARN]):
    im = ax.pcolormesh(gxq, gyq, np.ma.masked_invalid(np.where(mm, F, np.nan)),
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='nearest')
    ax.set_aspect('equal')
    ax.set_title(ttl, fontsize=8, loc='left')
    ax.set_xlabel('x [m]', fontsize=7.2)
    ax.set_xlim(axA.get_xlim())
    ax.set_ylim(axA.get_ylim())
    ax.tick_params(labelsize=6.4, labelleft=False)
    inside(ax, nt, cl)
cb = fig.colorbar(im, ax=axm, fraction=.032, pad=.014, aspect=22)
cb.set_label('surface elevation [m]', fontsize=6.8)
cb.ax.tick_params(labelsize=6.2)

# (e) 참값 대 추정값
lim = vmax * 1.06
axE.plot([-lim, lim], [-lim, lim], '-', color='k', lw=.9, zorder=2)
axE.axhline(0, color=MUTE, ls=':', lw=1.1, zorder=2)
axE.text(-lim * .97, .03 * vmax, 'dotted = a zero field', fontsize=6.2,
         color=MUTE, ha='left', va='bottom', zorder=5,
         bbox=dict(fc='white', ec='none', alpha=.8, pad=1.2))
for a, mk in zip(G.ARM_ORDER, ('o', 's')):
    axE.scatter(gt[mm], fields[a][mm], s=13, marker=mk, facecolors='none',
                edgecolors=COL[a], lw=.75, zorder=3,
                label=f'{NAME[a]}  ·  nRMSE {now[a]:.3f}')
axE.set_xlim(-lim, lim)
axE.set_ylim(-lim, lim)
axE.set_aspect('equal')
axE.set_xlabel('measured elevation [m]', fontsize=7.4)
axE.set_ylabel('estimated [m]', fontsize=7.4)
axE.set_title(f'(e) {int(mm.sum())} query points, nowcast', fontsize=8, loc='left')
axE.tick_params(labelsize=6.6)
axE.legend(fontsize=6.2, loc='lower right', handletextpad=.15,
           borderpad=.3, labelspacing=.3, frameon=True, framealpha=.88,
           edgecolor='none', facecolor='white').set_zorder(6)
axE.grid(alpha=.2, lw=.5)

# (f) 잔차 — 제로필드 잔차(= -참값)를 회색 참조로 함께 그린다
bins = np.linspace(-lim, lim, 34)
axH.hist(-gt[mm], bins=bins, color=MUTE, alpha=.30, zorder=2)
for a in G.ARM_ORDER:
    axH.hist(fields[a][mm] - gt[mm], bins=bins, histtype='step', lw=1.25,
             color=COL[a], zorder=3)
axH.axvline(0, color='k', lw=.8, zorder=4)
axH.set_ylim(0, 48)
axH.set_xlabel('estimate - measured [m]', fontsize=7.4)
axH.set_ylabel('query points', fontsize=7.4)
axH.set_title('(f) residuals at the same query points', fontsize=8, loc='left')
axH.tick_params(labelsize=6.6)
axH.legend(fontsize=6.2, frameon=False, loc='upper right', handletextpad=.5,
           borderpad=.15, labelspacing=.35,
           handles=[plt.Rectangle((0, 0), 1, 1, fc=MUTE, alpha=.30, ec='none')] +
                   [Line2D([], [], color=COL[a], lw=1.25) for a in G.ARM_ORDER],
           labels=[f'a zero field  ·  nRMSE {zero_now:.3f}'] +
                  [f'{NAME[a]}  ·  nRMSE {now[a]:.3f}' for a in G.ARM_ORDER])
axH.grid(axis='y', alpha=.2, lw=.5)

# ★필수 공시 — 라우터가 실측에서 무엇을 하는가, 그리고 이 그림이 무엇이 아닌가
DISCLOSE = [
    'On measured input the shipped router does NOT use the learned prior (WavePrior): it hands the',
    'estimate to a separate bank estimator (c), which builds no Gaussian components at all. The',
    '"proposed representation" arm (d) is that prior forced on.  Both arms receive the identical',
    'observations and are scored at the identical query points.',
    '',
    'One record, one 40 s window, nowcast only: this is what the two arms look like, not how often.',
    'Counts over the 12 measured records are reported separately.  The observations are a synthetic',
    'sparsification of a measured WASS stereo surface, not LiDAR returns.',
]
fig.text(.055, .012, '\n'.join(DISCLOSE[:4]), ha='left', va='bottom',
         fontsize=6.1, color='#444', linespacing=1.45)
fig.text(.525, .012, '\n'.join(DISCLOSE[5:]), ha='left', va='bottom',
         fontsize=6.1, color=MUTE, linespacing=1.45)
fig.suptitle(f'Measured stereo record {RECORD} (fixed station), scored against its own zero field '
             f'(nRMSE {zero_now:.3f}, not the constant 1.0)', fontsize=8.8, x=.055, y=.972,
             ha='left')

OUT = os.path.join(ROOT, 'paper', 'figures', 'fig_real_qual.pdf')
fig.savefig(OUT)
fig.savefig(OUT.replace('.pdf', '.png'), dpi=190)
print(f'✅ {OUT}')
print(f'   {RECORD}: deployed {now["SHIP"]:.4f} · proposed {now["PROP"]:.4f} · '
      f'true zero field {zero_now:.4f}  (all three absolute; ⛔no ratio is printed)')
