"""AMENDMENT_1 의 모든 수치를 S6 원자료에서 **내가 직접** 재도출한다.

왜: 08-05 감사에서 내 오류 ~23건의 공통원인 1위가 **에이전트 수치 무검증 전달**이었다.
PRO/NEUTRAL/CON 세 감사관이 낸 수치를 그대로 옮기지 않고 전부 여기서 다시 계산한다.

입력 = `tmp/S6_l1_sweep_2026-08-08/results.jsonl` (구 에피소드, 시드 20-23). S7 결과 미참조.
출력 = `verify_amend1.txt`
"""
import json
import os
import re
import sys

import numpy as np
from scipy.stats import binom, binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
S6 = os.path.abspath(os.path.join(HERE, '..', 'S6_l1_sweep_2026-08-08', 'results.jsonl'))
OUT = os.path.join(HERE, 'verify_amend1.txt')


def regime(ep):
    n = int(re.search(r'_(\d+)comp', ep).group(1))
    return 'crowded' if n >= 15 else 'sparse'


def block(ep):
    m = re.search(r'episode_(BF\d)_(\d+)comp', ep)
    return f'{m.group(1)}_{m.group(2)}c'


def main():
    out = []

    def p(s=''):
        print(s, flush=True)
        out.append(s)

    recs = [json.loads(l) for l in open(S6) if l.strip()]
    recs = [r for r in recs if 'error' not in r]
    med = {}
    for r in recs:
        med.setdefault((r['cell'], r['episode']), []).append(r['nRMSE'])
    med = {k: float(np.median(v)) for k, v in med.items()}
    eps = sorted({e for (_, e) in med})

    def pairs(t, b, rg):
        return [(e, med[(t, e)] - med[(b, e)], med[(b, e)])
                for e in eps if regime(e) == rg]

    p('=' * 78)
    p('§1 문턱표 (n=32 양측 정확 부호검정, Holm-2 최악순위 = x2)')
    for k in (22, 23, 24):
        pv = binomtest(k, 32, 0.5).pvalue
        p(f'  {k}/32  p={pv:.5f}  x2={2 * pv:.5f}')
    p('  참고 n=16: 13/16 p=%.5f (x3=%.5f) | 14/16 p=%.5f (x3=%.5f)'
      % (binomtest(13, 16, .5).pvalue, 3 * binomtest(13, 16, .5).pvalue,
         binomtest(14, 16, .5).pvalue, 3 * binomtest(14, 16, .5).pvalue))

    p('')
    p('§1 사전 검정력 P(>=23/32)')
    for r_ in (0.8125, 0.75, 0.70, 0.65):
        p(f'  참 부호율 {r_:.4f}: {1 - binom.cdf(22, 32, r_):.4f}'
          f'   (구 설계 P(>=14/16) = {1 - binom.cdf(13, 16, r_):.4f})')

    p('')
    p('A10 동점표 — 유효 n 별 Holm-2 통과 최소 부호수')
    for n in range(32, 23, -1):
        k = next(k for k in range(n // 2 + 1, n + 1)
                 if 2 * binomtest(k, n, 0.5).pvalue <= 0.05)
        pv = binomtest(k, n, 0.5).pvalue
        p(f'  n={n}: k>={k}  p={pv:.5f}  x2={2 * pv:.5f}')

    p('')
    p('=' * 78)
    p('A3 분기 사전확률 (독립 근사)')
    for p1, p2 in ((0.8125, 0.90), (0.75, 0.85)):
        a, b = 1 - binom.cdf(22, 32, p1), 1 - binom.cdf(22, 32, p2)
        p(f'  P1={p1} P2={p2}: DEPLOY={a * b:.4f} KNOB-REAL={a * (1 - b):.4f} '
          f'BUNDLE-ONLY={(1 - a) * b:.4f} NULL={(1 - a) * (1 - b):.4f}')

    p('')
    p('=' * 78)
    p('S6 주요 대조 재도출 (에피소드당 median-of-3 -> 짝 Δ -> 부호검정)')
    for rg in ('crowded', 'sparse'):
        p(f'--- {rg} ---')
        for t, b in (('AUTO-NT-L1-1e4', 'AUTO-NT'), ('AUTO-NT-L1-1e4', 'AUTO'),
                     ('AUTO-NT', 'AUTO'), ('AUTO-NT-L1-3e4', 'AUTO-NT'),
                     ('AUTO-NT-L1-1e4', 'AUTO-NT-L1-3e4')):
            P = pairs(t, b, rg)
            d = np.array([x[1] for x in P])
            base = np.array([x[2] for x in P])
            imp, ties = int((d < 0).sum()), int((d == 0).sum())
            pv = binomtest(imp, len(d) - ties, 0.5).pvalue
            p(f'  {t:16s} - {b:16s} 개선 {imp}/{len(d)} 동점 {ties} '
              f'medΔ={np.median(d):+.5f} ({np.median(d / base) * 100:+.1f}%) p={pv:.4f}')
        lv = {c: round(float(np.median([med[(c, e)] for e in eps if regime(e) == rg])), 5)
              for c in ('AUTO', 'AUTO-NT', 'AUTO-NT-L1-3e4', 'AUTO-NT-L1-1e4',
                        'AUTO-NT-L1-0')}
        p(f'  수준(중앙): {lv}')

    p('')
    p('=' * 78)
    p('A4 희소 꼬리 구조 (L1-1e4 - AUTO-NT)')
    P = pairs('AUTO-NT-L1-1e4', 'AUTO-NT', 'sparse')
    d = np.array([x[1] for x in P])
    base = np.array([x[2] for x in P])
    rel = d / base
    lt = float(np.median([med[('AUTO-NT-L1-1e4', e)] for e in eps if regime(e) == 'sparse']))
    lb = float(np.median([med[('AUTO-NT', e)] for e in eps if regime(e) == 'sparse']))
    p(f'  수준의 차 = {lt - lb:+.5f} (피해)   |   짝 Δ 중앙 = {np.median(d):+.5f} (개선)')
    p(f'  mean 상대 Δ = {rel.mean() * 100:+.1f}%   중앙 상대 Δ = {np.median(rel) * 100:+.1f}%'
      f'   최대 상대 Δ = {rel.max() * 100:+.1f}%')
    w = int(np.argmax(rel))
    p(f'  악화 >+50%: {int((rel > 0.5).sum())}/16   최악 = {os.path.basename(P[w][0])} '
      f'{P[w][2]:.5f} -> {med[("AUTO-NT-L1-1e4", P[w][0])]:.5f} ({P[w][1]:+.5f})')
    for b_ in sorted({block(e) for e in eps if regime(e) == 'sparse'}):
        sub = [x for x in P if block(x[0]) == b_]
        dd = np.array([x[1] for x in sub])
        p(f'    블록 {b_}: 개선 {int((dd < 0).sum())}/{len(dd)}  medΔ={np.median(dd):+.5f}')
    p('  HARM 게이트 발화확률:')
    for r_ in (5 / 16, 3 / 16, 0.60, 0.6875, 0.70, 0.75):
        p(f'    참 피해율 {r_:.4f}: P(>=23/32)={1 - binom.cdf(22, 32, r_):.3e}  '
          f'P(>=21/32)={1 - binom.cdf(20, 32, r_):.4f}')

    p('')
    p('A8 혼잡 블록 분해 (L1-1e4 - AUTO-NT)')
    P = pairs('AUTO-NT-L1-1e4', 'AUTO-NT', 'crowded')
    meds = {}
    for b_ in sorted({block(e) for e in eps if regime(e) == 'crowded'}):
        sub = [x for x in P if block(x[0]) == b_]
        dd = np.array([x[1] for x in sub])
        meds[b_] = float(np.median(dd))
        p(f'  {b_}: 개선 {int((dd < 0).sum())}/{len(dd)}  medΔ={np.median(dd):+.5f}')
    p(f'  블록 간 |medΔ| 최대/최소 비 = {max(map(abs, meds.values())) / min(map(abs, meds.values())):.1f}배')
    p('  조건 라벨 바닥 p = 2*(1/2)^4 = 0.125  ⟹ 조건 일반화는 α=0.05 원리적 불가')

    p('')
    p('A6 MDE — mde_agg 대 mde_sign80')
    for rg in ('crowded', 'sparse'):
        d = np.array([x[1] for x in pairs('AUTO-NT-L1-1e4', 'AUTO-NT', rg)])
        s = float(1.4826 * np.median(np.abs(d - np.median(d))))
        p(f'  {rg}: robust σ_Δ(에피소드 간) = {s:.5f}   mde_sign80 = 0.7234*σ_Δ = {0.7234 * s:.5f}')
    mj = json.load(open(os.path.join(HERE, 'MDE_prespec_s7.json')))
    for rg in ('crowded', 'sparse'):
        p(f'  {rg}: σ_d(시드잡음만) = {mj["by_regime"][rg]["sigma_d"]:.5f}   '
          f'mde_agg(n=32) = {mj["by_regime"][rg]["mde_agg"]:.5f}')
    p('  ★S6 반례: 혼잡 |Δ|=0.02583 = mde_agg(n=16)=0.014464 의 1.79배였으나 13/16 미달')

    open(OUT, 'w').write('\n'.join(out) + '\n')
    print(f'\n-> {OUT}', file=sys.stderr)


if __name__ == '__main__':
    main()
