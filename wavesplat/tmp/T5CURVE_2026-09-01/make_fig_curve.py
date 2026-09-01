#!/usr/bin/env python3
"""`T5-CURVE` 논문 그림 — ⛔런 0회. `curve.json` 만 읽는다(손복사 0).

산출 = `thesis/figures_ws/fig11_ensemble_leadcurve.pdf`

★그림이 말해야 하는 것은 **둘**이고, 하나만 그리면 `F1` 부류의 오독이 생긴다:
  (위) 앙상블 이득이 lead 와 함께 **커진다**
  (아래) ⚠**같은 구간에서 비교자(BANK)가 무너진다** — 그리고 PROP 도 문턱을 못 넘는다
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
OUT = os.path.join(ROOT, 'thesis', 'figures_ws', 'fig11_ensemble_leadcurve.pdf')

C = json.load(open(os.path.join(HERE, 'curve.json')))['curve']
lead = [c['lead'] for c in C]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.2, 5.4), sharex=True,
                               gridspec_kw=dict(height_ratios=[1, 1.15], hspace=0.12))

# ── (위) 앙상블 이득
ax1.plot(lead, [c['ens_rel_median_pct'] for c in C], 'o-', color='#1f77b4', lw=1.8, ms=4)
ax1.axhline(0, color='0.6', lw=0.8, ls=':')
ax1.set_ylabel('ensemble gain\n(rel. median, %)')
ax1.invert_yaxis()
ax1.grid(alpha=0.25)
for c in C:                       # 확증 문턱(25/36)을 넘은 lead 를 표시
    if c['ens_better'] >= 25:
        ax1.plot(c['lead'], c['ens_rel_median_pct'], 'o', color='#1f77b4', ms=7, mfc='none')
ax1.set_title('Ensemble gain grows with lead — but so does comparator collapse',
              fontsize=10)

# ── (아래) 참 제로필드를 이기는 셀 수
ax2.plot(lead, [c['bank_beats_zero'] for c in C], 's-', color='#d62728', lw=1.8, ms=4,
         label='BANK-K3 (deployed comparator)')
ax2.plot(lead, [c['prop_med3_beats_zero'] for c in C], '^-', color='#2ca02c', lw=1.8, ms=4,
         label='PROP-M3 (learned + ensemble)')
ax2.plot(lead, [c['prop_m0_beats_zero'] for c in C], 'v--', color='#2ca02c', lw=1.2, ms=3,
         alpha=0.6, label='PROP single member')
ax2.axhline(24, color='0.35', lw=1.0, ls='--')
ax2.text(14.2, 24, ' NOMINAL', va='center', fontsize=7.5, color='0.35')
ax2.axhline(18, color='0.75', lw=0.8, ls=':')
ax2.text(14.2, 18, ' chance', va='center', fontsize=7.5, color='0.6')
ax2.set_xlabel('forecast lead (s)')
ax2.set_ylabel('cells beating\ntrue zero field  (/36)')
ax2.set_ylim(8, 36)
ax2.grid(alpha=0.25)
ax2.legend(fontsize=7.5, loc='upper right', framealpha=0.9)

fig.text(0.005, 0.005,
         'Exploratory post-hoc decomposition of frozen T5-ENS rows (0 new runs). '
         'Not a confirmatory test.', fontsize=6.5, color='0.4')
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches='tight')
print('wrote', os.path.relpath(OUT, ROOT))
sys.exit(0)
