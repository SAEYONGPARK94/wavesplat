#!/usr/bin/env python3
"""`paper/figures/fig_results.pdf` --- 결과 그림. **동결 verdict_*.json 에서만** 그린다.

⛔이 스크립트가 존재하는 이유(2026-09-02, 4차 적대적 감사 B8):

> *"논문 52쪽에 그림이 **하나**뿐이고 그것은 개념도다. 재구성 예시 0 · 오차-거리 0 ·
>  스펙트럼 0. 표 넷이 전부 참고문헌 뒤에 인쇄된다. ⟹ 심사 가능한 증거 표면이 없다."*

★그림을 만드는 원칙은 표와 같다: **손으로 적은 수 0개.** 값은 전부 동결 판정에서 읽고,
핀과 어긋나면 죽는다(`gen_paper_constants.py` 와 같은 규약).

패널 A --- 각 배치의 승수와 **정확 이항 신뢰구간**, 그리고 **비교자 자신의 기술**을
  같은 축에 얹는다. ★`cmp.` 열이 비어 있는 행(음성대조가 설계에 없던 배치)이
  **그림에서도 비어 보이게** 한다 --- 그것이 4차 감사 M10 이 지적한 것이다.
패널 B --- `T-4` lead 사다리. 우리 팔의 제로필드 대비 기술은 평평한데
  **비교자가 무너진다**(47/48 -> 34/48). ⟹ *"lead 가 길수록 우리가 낫다"* 가
  **왜 금지인지**를 문장이 아니라 그림으로 보인다(F1 부류 공시의 시각화).
"""
import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TMP = os.path.join(ROOT, 'tmp')


def V(rel):
    p = os.path.join(TMP, rel)
    if not os.path.exists(p):
        raise SystemExit(f'⛔동결 판정이 없다: {p}')
    return json.load(open(p))


def get(d, path):
    for k in path.split('.'):
        d = d[k]
    return d


def ci(k, n, conf=0.95):
    """Clopper--Pearson 정확 이항 구간. ⚠근사 아님 --- n 이 16 인 행이 있다."""
    from scipy.stats import beta
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else beta.ppf(a, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a, k + 1, n - k)
    return lo, hi


# (표시명, verdict 경로, better 경로, n 경로, 비교자 better 경로 or None)
ROWS = [
    ('B-1  strongest eligible baseline', 'B1_BASELINE_2026-08-22/verdict_b1.json',
     'crowded.better', 'crowded.n', None),
    ('B-12 non-linear ref., fresh seeds', 'B12_HOS_EXTEND_2026-08-23/verdict_b12.json',
     'ship_vs_spec.better', 'ship_vs_spec.n', 'spec_vs_zero.better'),
    ('B-15 physically correct sensor', 'B15_RAYDROP_2026-08-24/verdict_b15.json',
     'by_raydrop.slope.ship_beats_spec', 'by_raydrop.slope.n',
     'by_raydrop.slope.spec_beats_zero'),
    ('B-26 sea state of the records', 'B26_SEASTATE_2026-08-27/verdict_b26.json',
     'corners.SS-REAL.ship_vs_spec.better', 'corners.SS-REAL.ship_vs_spec.n',
     'corners.SS-REAL.spec_vs_zero.better'),
    ('B-16 ablate wavenumber continuity', 'B16_KGRID_2026-08-24/verdict_b16.json',
     'primary.better', 'primary.n', None),
    ('B-19  the same, sparse stratum', 'B19_KGRID_SPARSE_2026-08-25/verdict_b19.json',
     'primary.better', 'primary.n', None),
    ('B-22 ablate the learned prior', 'B22_PRIOR_ABLATION_2026-08-27/verdict_b22.json',
     'comparisons.L2.better', 'comparisons.L2.n', 'neg_vs_zero.ANA.better'),
    ('S-ENS ensemble, crowded', 'S_ENS_2026-09-01/verdict_sens.json',
     'contrasts.PRIMARY.better', 'contrasts.PRIMARY.n', None),
    ('S-ENS  the same, sparse', 'S_ENS_2026-09-01/verdict_sens.json',
     'contrasts.SEC-A.better', 'contrasts.SEC-A.n', None),
    ('M-10 inside the observed band', 'M10_GEOMETRY_2026-08-30/verdict_m10.json',
     'primary.primary.better', 'primary.primary.n', 'primary.spec_vs_zero.better'),
    ('T-4  at forecast lead', 'T4_GEOMETRY_2026-08-30/verdict_t4.json',
     'primary.better', 'primary.n', 'comparator.better'),
]

INK, MUTE, ACC, WARN = '#1a1a1a', '#8a8a8a', '#0b5394', '#a33'

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.4, 4.0),
                               gridspec_kw={'width_ratios': [1.65, 1.0]})

# ── 패널 A
# ⚠[2026-09-02] 5차 감사: 표 5 는 세 그룹으로 나뉘고 캡션이 *"크기를 더하거나 빼면 정의된
#   뜻이 없다"* 고 적는다. 초판 그림은 그 구분을 **버리고** 11행을 한 축에 세워
#   본문이 금지한 행간 비교를 초대했다. ⟹ 구분선과 그룹 이름을 넣는다.
GROUPS = {0: 'against an external baseline',
          4: 'ablations of the representation itself',
          9: 'where the gain sits'}
ys, labels = [], []
for i, (name, rel, pb, pn, pc) in enumerate(ROWS):
    d = V(rel)
    k, n = int(get(d, pb)), int(get(d, pn))
    y = len(ROWS) - i
    ys.append(y)
    labels.append(f'{name}  ({k}/{n})')
    lo, hi = ci(k, n)
    axA.plot([lo, hi], [y, y], color=INK, lw=1.1, solid_capstyle='butt', zorder=2)
    axA.plot([k / n], [y], 'o', ms=5.0, color=ACC, zorder=3)
    if pc is not None:
        # ⚠5차 감사: 빈 마커에 CI 가 없어 **비교자만 점추정처럼** 보였다(의도와 반대).
        kc = int(get(d, pc))
        clo, chi = ci(kc, n)
        axA.plot([clo, chi], [y - 0.24, y - 0.24], color=MUTE, lw=0.9, alpha=0.85, zorder=2)
        axA.plot([kc / n], [y - 0.24], 'o', ms=4.5, mfc='none', mec=MUTE, mew=1.1,
                 zorder=3)

for gi, gname in GROUPS.items():
    yv = len(ROWS) - gi + 0.55
    if gi:
        axA.axhline(yv, color='#ccc', lw=0.7)
    # 그룹 이름은 **행 사이 빈 띠**에 놓는다(마커가 사는 y 가 아니다).
    axA.text(1.045, yv - 0.05, gname, fontsize=6.0, color=MUTE, style='italic',
             ha='left', va='top', clip_on=False)
axA.axvline(0.5, color=MUTE, lw=0.8, ls=':')
axA.set_yticks(ys)
axA.set_yticklabels(labels, fontsize=6.6)
axA.set_xlim(0.20, 1.02)
# ⛔[2026-09-02] 6차 감사 N7: 범례가 **마지막 행과 그 빈 마커를 덮었다**(0.708).
#   ⟹ 축 아래쪽에 범례 자리를 만든다(행은 y>=1 에 있고 빈 마커는 y-0.24 = 0.76).
axA.set_ylim(-0.45, len(ROWS) + 0.95)
axA.set_xlabel('fraction of episodes won (95% Clopper-Pearson)', fontsize=7.2)
axA.tick_params(axis='x', labelsize=7)
for sp in ('top', 'right'):
    axA.spines[sp].set_visible(False)
axA.plot([], [], 'o', ms=5, color=ACC, label='our arm vs. its comparator')
axA.plot([], [], 'o', ms=5, mfc='none', mec=MUTE, mew=1.1,
         label='comparator vs. a zero field')
axA.legend(fontsize=6.4, loc='lower left', frameon=False, handletextpad=0.3)
axA.set_title('(a) every comparison this paper rests on', fontsize=8, loc='left')

# ── 패널 B --- T-4 사다리
t4 = V('T4_GEOMETRY_2026-08-30/verdict_t4.json')['ladder']
rungs = sorted(t4, key=float)
x = [float(r) for r in rungs]
ship = [t4[r]['ship_vs_zero']['better'] / t4[r]['ship_vs_zero']['n'] for r in rungs]
spec = [t4[r]['spec_vs_zero']['better'] / t4[r]['spec_vs_zero']['n'] for r in rungs]
axB.plot(x, ship, 'o-', color=ACC, ms=4.5, lw=1.3, label='our arm')
axB.plot(x, spec, 'o--', color=WARN, ms=4.5, lw=1.3, mfc='none', label='comparator')
axB.axhline(0.5, color=MUTE, lw=0.8, ls=':')
axB.set_ylabel('fraction beating a zero field', fontsize=7.2)
axB.set_ylim(0.35, 1.03)
axB.tick_params(labelsize=7)
for sp in ('top', 'right'):
    axB.spines[sp].set_visible(False)
axB.legend(fontsize=6.6, frameon=False, loc='lower left')
axB.set_title('(b) why the growing margin is not a gain', fontsize=8, loc='left')
# ⚠주석이 범례와 겹쳐 둘 다 못 읽혔다 --- 축 아래 여백으로 내린다.
axB.set_xlabel('forecast lead  ($T_p$)\n'
               'our own absolute error worsens across this range', fontsize=7.2)

fig.tight_layout(pad=0.6)
out = os.path.join(ROOT, 'paper', 'figures', 'fig_results.pdf')
fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
print('wrote', out)
