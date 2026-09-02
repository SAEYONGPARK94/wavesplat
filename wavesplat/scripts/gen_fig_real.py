#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""실측 12 레코드 그림 --- ⛔**런 0회**(동결 행 재집계). 사용자 요청 2026-09-02:
*"시뮬레이터 데이터랑 비교한거지? 실측 데이터랑도 뽑아줘"*.

★이 그림이 어려운 이유는 그림이 아니라 **채점 기준**이다. 이 저장소는 실측에서
`ZERO_REF = 1.0` 이라는 **상수**로 채점하다가 2026-08-23 밤에 그것이 거짓임을 발견했다
(참 제로필드는 레코드별 **0.3375–1.1559 = 3.4배**). 그래서 축을 **레코드별 참 제로필드로
정규화**한다 --- 1.0 = *"아무것도 안 하는 것과 같다"*.

⛔그리고 자료를 **수심 30** 으로 잡는다. `B-2` 는 수심 **17** 로 돌았고(정정 전 값),
정정된 대조군이 이미 존재한다(`depth_recheck.py` 가 그 짝을 증명한다: 수심이 닿지
않는 5 레코드가 **전건 비트 동일**, BS 7 개만 다르다 ⟹ 유일한 차이가 수심이다).

⛔**인용 금지 준수**
· 짝 비교 `SHIP` vs `PROP` 의 **크기**를 그리지 않는다(허용은 **부호**뿐;
  +52.2% → +27.8% 로 두 번 정정됐다). 그래서 두 팔을 **서로** 비교하는 패널이 없다.
· 각 팔은 **자기 레코드의 참 제로필드**에 대해서만 채점된다.
· `SHIP` 의 10/12 는 **16팔 Holm(임계 0.0031)에서 탈락**한다 --- 그림이 그것을 적는다.
"""
import os, sys, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
B2DIR = os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23')
sys.path.insert(0, B2DIR)
import gates_b2 as G                                              # noqa: E402


def load(*rel):
    out = []
    for r in rel:
        with open(os.path.join(ROOT, 'tmp', r)) as f:
            out += [json.loads(l) for l in f if l.strip()]
    return [r for r in out if 'error' not in r]


B3 = load('B3_REFIT_2026-08-23/results_b3.jsonl')
B14B = load('B14_OBSMODEL_2026-08-24/results_b14b_0.jsonl',
            'B14_OBSMODEL_2026-08-24/results_b14b_1.jsonl')
TZ = json.load(open(os.path.join(B2DIR, 'true_zero_field.json')))

# 수심 30 (= 정정된 수심) 의 두 팔. 출처는 `depth_recheck.py` 의 표와 **같다**.
SHIP = {r['record']: r['lead0'] for r in B3
        if r['cell'] == 'FROZEN' and r['win'] == G.WIN}
PROP = {r['record']: r['nowcast'] for r in B14B
        if r['obs_model'] == 'IID' and r['noise'] == 0.02}

RECS = list(G.RECORDS)
assert len(RECS) == 12 and set(RECS) == set(TZ), '⛔레코드 집합이 12개가 아니다'
for nm, d in (('SHIP', SHIP), ('PROP', PROP)):
    miss = [k for k in RECS if k not in d]
    assert not miss, f'⛔{nm} 에 없는 레코드: {miss}'

# ── 핀: `depth_recheck.py` 가 수심 30 에서 기록한 참-기준 승수와 **일치해야 한다** ──
PIN = {'SHIP': 10, 'PROP': 7}
wins = {nm: sum(1 for k in RECS if d[k] < TZ[k]) for nm, d in (('SHIP', SHIP), ('PROP', PROP))}
assert wins == PIN, f'⛔핀 불일치 {wins} != {PIN} — depth_recheck.py 와 어긋난다'
p_ship = G.sign_test_one_sided(wins['SHIP'], 12)
p_prop = G.sign_test_one_sided(wins['PROP'], 12)
HOLM16 = 0.05 / 16          # 실측 아크의 16팔 Holm 첫 임계 (금지목록 2026-08-23 밤)

ACC, WARN, MUTE = '#0b5394', '#b45309', '#8a8a8a'
fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.6, 3.15),
                               gridspec_kw={'width_ratios': [2.05, 1]})

# ── (a) 레코드별 기술비 = 팔 / 그 레코드의 참 제로필드 ──────────────────────
order = sorted(RECS, key=lambda k: SHIP[k] / TZ[k])
x = range(len(order))
rs = [SHIP[k] / TZ[k] for k in order]
rp = [PROP[k] / TZ[k] for k in order]
axA.axhline(1.0, color='k', lw=1.0, zorder=1)
axA.text(0.02, 1.012, 'no skill (a zero field)', fontsize=6.6,
         ha='left', va='bottom', transform=axA.get_yaxis_transform())
# ★가장 중요한 공시: 실측 입력에서 배포 라우터는 **WavePrior 를 쓰지 않는다.**
axA.text(0.985, 0.045,
         'on measured input the shipped router does not use WavePrior:\n'
         'it hands the estimate to a separate bank estimator',
         transform=axA.transAxes, ha='right', va='bottom', fontsize=6.3,
         color=MUTE, linespacing=1.35)
axA.plot(x, rs, 'o-', ms=4.6, lw=1.2, color=ACC, zorder=3,
         label=f'deployed arm on measured input ({wins["SHIP"]}/12 below 1)')
axA.plot(x, rp, 's--', ms=4.4, lw=1.1, color=WARN, mfc='none', zorder=3,
         label=f'proposed representation, forced on ({wins["PROP"]}/12)')
axA.set_xticks(list(x))
axA.set_xticklabels(order, rotation=45, ha='right', fontsize=6.4)
axA.set_ylabel('nRMSE / that record\'s true zero field', fontsize=7.6)
axA.set_title('(a) 12 measured records, each scored against its own zero field',
              fontsize=8, loc='left')
axA.tick_params(labelsize=6.8)
axA.legend(fontsize=6.5, frameon=False, loc='upper left')
axA.grid(alpha=.22, lw=.5)

# ── (b) 승수와 다중성 ────────────────────────────────────────────────────
bars = axB.bar([0, 1], [wins['SHIP'], wins['PROP']], width=.56,
               color=[ACC, WARN], alpha=.85, zorder=3)
axB.axhline(10, color=MUTE, ls=':', lw=1.0, zorder=2)
axB.text(1.52, 10.06, 'p < 0.05 needs 10/12', fontsize=6.4, ha='right', va='bottom',
         color=MUTE)
axB.set_xticks([0, 1])
axB.set_xticklabels(['deployed\narm', 'proposed\nrepresentation'], fontsize=7)
axB.set_ylim(0, 13.6)
axB.set_ylabel('records beating a zero field', fontsize=7.6)
axB.set_title('(b) neither survives multiplicity', fontsize=8, loc='left')
axB.tick_params(labelsize=6.8)
for b, p in zip(bars, (p_ship, p_prop)):
    axB.text(b.get_x() + b.get_width() / 2, b.get_height() + .18,
             f'p = {p:.4f}', ha='center', fontsize=6.6)
axB.text(0.5, 0.965, f'16-arm Holm threshold: p < {HOLM16:.4f}',
         ha='center', va='top', fontsize=6.5, color='#b00', transform=axB.transAxes)
axB.grid(axis='y', alpha=.22, lw=.5)

fig.tight_layout(pad=.7)
OUT = os.path.join(ROOT, 'paper', 'figures', 'fig_real.pdf')
fig.savefig(OUT)
fig.savefig(OUT.replace('.pdf', '.png'), dpi=190)
# ★학위논문도 같은 그림을 싣는다 --- **한 생성기에서 두 곳으로**.
#   ⛔복사본을 손으로 두지 마라: 두 문서가 다른 그림을 실은 채 갈라지는 통로다
#     (이 저장소가 오늘 「이름이 셋으로 갈렸다」로 대가를 치른 그 부류).
TOUT = os.path.join(ROOT, 'thesis', 'figures_ws', 'fig12_real_zeroline.pdf')
fig.savefig(TOUT)
print(f'✅ {OUT}')
print(f'✅ {TOUT}')
print(f'   deployed {wins["SHIP"]}/12 p={p_ship:.4f} · proposed {wins["PROP"]}/12 '
      f'p={p_prop:.4f} · Holm(16) = {HOLM16:.4f}')
print('   ⛔둘 다 Holm 임계 위 = 실측 아크에 다중성을 견디는 팔이 0개다.')
