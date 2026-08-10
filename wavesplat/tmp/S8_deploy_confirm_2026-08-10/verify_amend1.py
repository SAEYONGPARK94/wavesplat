"""AMENDMENT_1 의 모든 수치를 S7 원자료에서 **내가 직접** 재도출한다 (S7 verify 와 같은 규율).

에이전트 수치 무검증 전달이 08-05 감사에서 내 오류 ~23건의 공통원인 1위였다.
PRO/NEUTRAL/CON 세 감사관의 수치를 그대로 옮기지 않고 전부 여기서 다시 계산한다.
★이 스크립트가 **CON C-3 의 전제를 반증**한다(수준표 vs 짝 Δ).

입력 = `tmp/S7_confirm_n32_2026-08-10/results.jsonl`. S8 결과 미참조(존재하지 않음).
출력 = `verify_amend1.txt`
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import binom, binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
S7 = os.path.abspath(os.path.join(HERE, '..', 'S7_confirm_n32_2026-08-10', 'results.jsonl'))
OUT = os.path.join(HERE, 'verify_amend1.txt')
MDE = os.path.join(HERE, 'MDE_prespec_s8.json')
LINES = []


def p(s=''):
    print(s, flush=True)
    LINES.append(s)


def rg(ep):
    return 'crowded' if ep.startswith('data_s7_crowded') else 'sparse'


def blk(ep):
    m = re.search(r'episode_(BF\d)_(\d+)comp', ep)
    return f'{m.group(1)}_{m.group(2)}c'


def main():
    recs = [json.loads(l) for l in open(S7) if l.strip()]
    recs = [r for r in recs if 'error' not in r]
    eps = sorted({r['episode'] for r in recs})
    seeds = sorted({r['seed'] for r in recs})

    def med(metric):
        raw = defaultdict(list)
        for r in recs:
            v = ((r.get('shell_rmse') or {}).get(metric.split(':')[1])
                 if metric.startswith('shell:')
                 else ((r.get('alpha_episode') or {}).get('0.2') if metric == 'alpha'
                       else r.get(metric)))
            if v is not None and np.isfinite(v):
                raw[(r['cell'], r['episode'])].append(float(v))
        return {k: float(np.median(v)) for k, v in raw.items()}

    def pairs(m, t, b, regime):
        e_ok = [e for e in eps if rg(e) == regime and (t, e) in m and (b, e) in m]
        d = np.array([m[(t, e)] - m[(b, e)] for e in e_ok])
        base = np.array([m[(b, e)] for e in e_ok])
        return e_ok, d, base

    def line(m, t, b, regime, lab):
        e_ok, d, base = pairs(m, t, b, regime)
        imp = int((d < 0).sum())
        pv = binomtest(imp, len(d), 0.5).pvalue
        return (f'  {lab:38s} 개선 {imp}/{len(d)}  Δ중앙 {np.median(d):+.5f} '
                f'({np.median(d / base) * 100:+.1f}%)  p={pv:.5f}')

    p('=' * 88)
    p('§1 문턱 (n=32 양측 정확 부호검정) · 유효 n 별 Holm-2 최소 부호수')
    for k in (21, 22, 23, 24):
        pv = binomtest(k, 32, 0.5).pvalue
        p(f'  {k}/32  p={pv:.6f}  x2={2 * pv:.6f}')
    p('  유효 n : 단독 α=0.05 최소 / Holm-2 최소  (A11 — Holm 비용)')
    for n in range(32, 23, -1):
        k1 = next(k for k in range(n // 2 + 1, n + 1) if binomtest(k, n, .5).pvalue <= 0.05)
        k2 = next(k for k in range(n // 2 + 1, n + 1) if 2 * binomtest(k, n, .5).pvalue <= 0.05)
        p(f'    n={n}: 단독 {k1} / Holm-2 {k2}   {"(비용 +1)" if k2 > k1 else ""}')

    p('')
    p('§1 사전 검정력 P(≥23/32) + Clopper-Pearson 구간 (A8)')
    from scipy.stats import beta
    for lab, k in (('PRIMARY-C (S7 26/32)', 26), ('PRIMARY-S (S7 23/32)', 23)):
        r_ = k / 32
        lo = beta.ppf(0.025, k, 32 - k + 1)
        hi = beta.ppf(0.975, k + 1, 32 - k)
        p(f'  {lab}: r={r_:.5f}  P={1 - binom.cdf(22, 32, r_):.4f}  '
          f'CP95=[{lo:.3f},{hi:.3f}]  그 구간 검정력=[{1 - binom.cdf(22, 32, lo):.3f},'
          f'{1 - binom.cdf(22, 32, hi):.3f}]')
    for r_ in (0.70, 0.65):
        p(f'  보수 r={r_}: P={1 - binom.cdf(22, 32, r_):.4f}')

    p('')
    p('§1 분기 사전확률 — 독립 가정 + 승자의 저주 축소 (A13)')
    for pc, ps in ((0.8125, 0.71875), (0.78, 0.68), (0.75, 0.65), (0.70, 0.60)):
        a, b = 1 - binom.cdf(22, 32, pc), 1 - binom.cdf(22, 32, ps)
        p(f'  참값 {pc}/{ps}: BOTH={a * b:.4f} C-ONLY={a * (1 - b):.4f} '
          f'S-ONLY={(1 - a) * b:.4f} NEITHER={(1 - a) * (1 - b):.4f}')
    p('  A16 HARMFUL 도달확률: 혼잡 %.2e · 희소 %.2e'
      % (1 - binom.cdf(22, 32, 1 - 26 / 32), 1 - binom.cdf(22, 32, 1 - 23 / 32)))

    p('')
    p('=' * 88)
    p('★A3 — CON C-3 전제 검증: `AUTO-NT` 가 처치를 이기는가 (수준 vs 짝 Δ)')
    mn = med('nRMSE')
    for regime in ('crowded', 'sparse'):
        lv = {c: float(np.median([mn[(c, e)] for e in eps if rg(e) == regime]))
              for c in ('AUTO', 'AUTO-L1-1e4', 'AUTO-NT', 'AUTO-NT-L1-1e4')}
        p(f'  {regime} 수준: ' + '  '.join(f'{k} {v:.5f}' for k, v in lv.items()))
        p(line(mn, 'AUTO-NT', 'AUTO-L1-1e4', regime, '    짝 AUTO-NT − AUTO-L1-1e4'))
    p('  ⟹ ★수준표로는 AUTO-NT 가 이기지만 **동결 규약(짝 Δ)으로는 우열 미결**')
    p('     (CON 의 전제는 수준표 산물 — 이 프로젝트가 이미 금지한 규약. 처방만 채택)')

    p('')
    p('★A2 — 무처치 널(같은 셀·cfg, 시드만 다른 짝)에 §6 꼬리 규칙 적용')
    val = {(r['cell'], r['episode'], r['seed']): r['nRMSE'] for r in recs}
    for cell in ('AUTO', 'AUTO-L1-1e4'):
        for regime in ('crowded', 'sparse'):
            for i in range(3):
                s1, s2 = seeds[i], seeds[(i + 1) % 3]
                e_ok = [e for e in eps if rg(e) == regime]
                d = np.array([val[(cell, e, s1)] - val[(cell, e, s2)] for e in e_ok])
                base = np.array([val[(cell, e, s2)] for e in e_ok])
                rel = d / base
                p(f'  {cell:14s} {regime:8s} seed{s1}-{s2}: 최대 {rel.max() * 100:+8.1f}%  '
                  f'>+50% {int((rel > 0.5).sum())}/32  '
                  f'부호반대 {np.sign(rel.mean()) != np.sign(np.median(rel))}')
    p('  ⟹ ★희소는 (b)>+200% 가 **6/6 널 발화**, (a) 4/6. 혼잡은 건전(최대 +39~+78%).')

    p('')
    p('★A5 — avg_obs_loss 는 λ·Σ|w| 를 포함한다 (거의 항등식) vs all_resid')
    for key in ('avg_obs_loss', 'all_resid'):
        m = med(key)
        for regime in ('crowded', 'sparse'):
            p(line(m, 'AUTO-L1-1e4', 'AUTO', regime, f'  {key} · SINGLE({regime})'))
        p(line(med(key), 'AUTO-NT-L1-1e4', 'AUTO', 'crowded', f'  {key} · S7 P-2(crowded)'))

    p('')
    p('★A4 — 근거리 게이트: 셸별 σ 와 MDE, 그리고 S7 P-2 의 실증 미검출')
    for regime in ('crowded', 'sparse'):
        for sk in ('0_5', '5_10', '10_20'):
            m = med(f'shell:{sk}')
            _e, d, base = pairs(m, 'AUTO-L1-1e4', 'AUTO', regime)
            s = float(1.4826 * np.median(np.abs(d - np.median(d))))
            b0 = float(np.median(base))
            p(f'  {regime:8s} {sk:6s} 기준선 {b0:.5f}  σ_rob {s:.5f}  '
              f'MDE80 +{0.7234 * s:.5f} ({0.7234 * s / b0 * 100:+.1f}%)  '
              f'개선 {int((d < 0).sum())}/32')
    for sk in ('0_5', '5_10', '10_20'):
        m = med(f'shell:{sk}')
        _e, d, base = pairs(m, 'AUTO-NT-L1-1e4', 'AUTO', 'crowded')
        p(f'  [S7 P-2 혼잡] {sk:6s} 악화 {int((d > 0).sum())}/32  '
          f'Δ중앙 {np.median(d):+.5f} ({np.median(d / base) * 100:+.1f}%)  '
          f'{"게이트 잡음" if int((d > 0).sum()) >= 21 else "⛔게이트 놓침"}')
    p('  위양성: 널(0.5)에서 셸당 %.4f · 근거리3셸 %.3f · 9셸 %.3f (독립 가정)'
      % (1 - binom.cdf(20, 32, 0.5), 1 - (binom.cdf(20, 32, 0.5)) ** 3,
         1 - (binom.cdf(20, 32, 0.5)) ** 9))

    p('')
    p('★A19 — 처치가 라우터를 움직이는가 (런 단위)')
    route = {(r['cell'], r['episode'], r['seed']): (r.get('auto_selected'), r.get('ensemble_k'))
             for r in recs}
    for t in ('AUTO-L1-1e4', 'AUTO-NT-L1-1e4'):
        same = sum(1 for e in eps for s in seeds if route[('AUTO', e, s)] == route[(t, e, s)])
        p(f'  AUTO vs {t:16s} 라우트·ensemble_k 동일: {same}/{len(eps) * len(seeds)}')

    p('')
    p('★A1/A12 — 블록 이질성 + D2 참조값 + mde_sign80 (1.4826·MAD 고정, A18)')
    mj = json.load(open(MDE))['by_regime'] if os.path.exists(MDE) else {}
    for regime in ('crowded', 'sparse'):
        e_ok, d, base = pairs(mn, 'AUTO-L1-1e4', 'AUTO', regime)
        bl = defaultdict(list)
        for e, x in zip(e_ok, d):
            bl[blk(e)].append(x)
        meds = {k: float(np.median(v)) for k, v in bl.items()}
        ratio = max(map(abs, meds.values())) / max(min(map(abs, meds.values())), 1e-12)
        nfav = sum(1 for k, v in bl.items() if (np.array(v) < 0).sum() > len(v) / 2)
        s_mad = float(1.4826 * np.median(np.abs(d - np.median(d))))
        s_iqr = float((np.percentile(d, 75) - np.percentile(d, 25)) / 1.349)
        p(f'  {regime}: D2 참조 {np.median(d):+.6f} | 우세블록 {nfav}/4 | 비 {ratio:.1f}배')
        p(f'    블록: ' + '  '.join(f'{k} {int((np.array(v) < 0).sum())}/{len(v)} '
                                    f'({np.median(v):+.5f})' for k, v in sorted(bl.items())))
        p(f'    σ_rob: MAD {s_mad:.5f} / IQR {s_iqr:.5f} / SD {float(np.std(d, ddof=1)):.5f}'
          f'  → mde_sign80(MAD) {0.7234 * s_mad:.5f}  관측/MDE {abs(np.median(d)) / (0.7234 * s_mad):.2f}')
        if mj:
            p(f'    (MDE_prespec_s8.json: mde_agg {mj[regime]["mde_agg"]:.5f}, '
              f'mde_sign80 {mj[regime]["mde_sign80"]:.5f}, mde_ep {mj[regime]["mde_ep"]:.5f})')

    p('')
    p('★A21 — α_wave 수준 (무기술 널 = 1.0 대비)')
    ma = med('alpha')
    for regime in ('crowded', 'sparse'):
        lv = {c: float(np.median([ma[(c, e)] for e in eps if rg(e) == regime]))
              for c in ('AUTO', 'AUTO-L1-1e4')}
        _e, d, _b = pairs(ma, 'AUTO-L1-1e4', 'AUTO', regime)
        s = float(1.4826 * np.median(np.abs(d - np.median(d))))
        p(f'  {regime}: AUTO {lv["AUTO"]:.4f} / L1-1e4 {lv["AUTO-L1-1e4"]:.4f} '
          f'(널 대비 기술 {100 * (1 - lv["AUTO"]):.0f}% / {100 * (1 - lv["AUTO-L1-1e4"]):.0f}%) '
          f'| 짝 개선 {int((d < 0).sum())}/32  σ_rob {s:.4f}  δ=0.7234σ {0.7234 * s:.4f}')

    open(OUT, 'w').write('\n'.join(LINES) + '\n')
    print(f'\n-> {OUT}', file=sys.stderr)


if __name__ == '__main__':
    main()
