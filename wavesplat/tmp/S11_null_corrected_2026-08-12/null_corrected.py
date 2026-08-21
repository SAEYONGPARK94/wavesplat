#!/usr/bin/env python3
"""S11-A (런 0회) — **널 보정된 선택 이득**.

S10 DECISION §4-2 가 지정한 계산: `(규칙 − 널에서의 규칙)`.
이것이 양수(=개선)로 남는지가 GT-free 팔 선택기의 진짜 생사다.

정의 (계산 전에 고정, 전부 S10 AUDIT_PRO §1.1 의 규칙 그대로):
  선택 규칙 = 에피소드별로 두 팔의 **3시드 중앙 `window_resid`** 를 비교해
              작은 팔의 **3시드 중앙 nRMSE** 를 취한다.
  처치 팔쌍  T = {AUTO-L1, V2K}   기준선 = AUTO-L1
  널   팔쌍  N = {AUTO, AUTO-SEED-NULL}  기준선 = AUTO
       (cfg 동일·추정 시드만 다름 ⟹ 고를 "더 나은 팔"이 존재하지 않는다)

  에피소드 e 마다
      rel_T(e) = (SELECT_T(e) - AUTO-L1(e)) / AUTO-L1(e)      (음수 = 개선)
      rel_N(e) = (SELECT_N(e) - AUTO(e))    / AUTO(e)
      DiD(e)   = rel_T(e) - rel_N(e)                          (음수 = 널을 넘는 이득)

  PRIMARY  : DiD 의 부호검정 (H0 = 0.5, 동점 제외) + 중앙값
  SECONDARY: 전환 시 승률을 **측정된 널 승률**에 대해 검정 (H0 = p0_null)

⚠한계 (보고에 반드시 병기):
  S9 에는 `AUTO-L1` 의 시드-널 팔이 없다. 널은 `AUTO`(lambda_l1=1e-3) 위에서 측정된다.
  ⟹ 널의 시드잡음 구조가 처치 기준선과 **정확히 같다는 보장은 없다**.
"""
import json, os, math, statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.dirname(HERE)
S9 = os.path.join(TMP, 'S9_lr_x_l1_2026-08-11', 'results.jsonl')
X2 = os.path.join(TMP, 'X2_no_learned_2026-08-11', 'results.jsonl')
S8 = os.path.join(TMP, 'S8_deploy_confirm_2026-08-10', 'results.jsonl')


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def regime(ep):
    """S9 규약: 2/5comp = 희소, 15/30comp = 혼잡 (파일명에서 직접)."""
    b = os.path.basename(ep)
    for n in ('2comp', '5comp'):
        if f'_{n}_' in b:
            return 'sparse'
    for n in ('15comp', '30comp'):
        if f'_{n}_' in b:
            return 'crowded'
    raise ValueError(ep)


def build(recs):
    """(cell, episode) -> {metric: 3시드 중앙값}.  에러 행은 제외하고 개수를 센다."""
    acc = defaultdict(lambda: defaultdict(list))
    nerr = 0
    for r in recs:
        if 'error' in r:
            nerr += 1
            continue
        k = (r['cell'], os.path.basename(r['episode']))
        for m in ('nRMSE', 'window_resid', 'all_resid', 'avg_obs_loss'):
            v = r.get(m)
            if v is not None and math.isfinite(v):
                acc[k][m].append(v)
    out = {k: {m: st.median(v) for m, v in d.items() if v} for k, d in acc.items()}
    return out, nerr


def binom_two_sided(k, n, p=0.5):
    """양측 부호검정 p — 같거나 더 극단인 확률의 합 (Sterne/정확법)."""
    if n == 0:
        return float('nan')
    pmf = [math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    tgt = pmf[k] * (1 + 1e-12)
    return min(1.0, sum(x for x in pmf if x <= tgt))


def binom_one_sided_ge(k, n, p):
    """P(X >= k | n, p) — 널 승률 초과 검정."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def selector(med, armA, armB, eps, stat='window_resid', metric='nRMSE'):
    """에피소드별 선택 결과.  반환: {ep: (chosen_arm, metric_value, baseline_value)}"""
    out = {}
    for e in eps:
        a, b = med.get((armA, e)), med.get((armB, e))
        if not a or not b or stat not in a or stat not in b:
            continue
        chosen = armA if a[stat] <= b[stat] else armB
        out[e] = (chosen, med[(chosen, e)][metric], a[metric])
    return out


def summarise(name, pairs):
    """pairs = [(sel, base)].  개선/악화/동점 + 중앙 상대변화."""
    rel = [(s - b) / b for s, b in pairs if b > 0]
    imp = sum(1 for r in rel if r < 0)
    wor = sum(1 for r in rel if r > 0)
    tie = sum(1 for r in rel if r == 0)
    p = binom_two_sided(imp, imp + wor) if (imp + wor) else float('nan')
    return dict(name=name, n=len(rel), imp=imp, wor=wor, tie=tie, p=p,
                med_rel=st.median(rel) if rel else float('nan'),
                med_rel_nonzero=st.median([r for r in rel if r != 0]) if (imp + wor) else float('nan'))


def main():
    s9, e9 = load(S9), None
    x2 = load(X2)
    med9, err9 = build(s9)
    med2, err2 = build(x2)
    med = {**med9, **med2}

    eps = sorted({e for (c, e) in med9 if c == 'AUTO-L1'})
    eps_x2 = sorted({e for (c, e) in med2 if c == 'V2K'})
    print(f'# 자료  S9 {len(s9)}행(에러 {err9})  X2 {len(x2)}행(에러 {err2})')
    print(f'# 에피소드  S9 {len(eps)}  X2 {len(eps_x2)}  교집합 {len(set(eps) & set(eps_x2))}')
    assert set(eps) == set(eps_x2), '에피소드 집합 불일치 — 짝짓기 불가'

    by_reg = defaultdict(list)
    for e in eps:
        by_reg[regime(e)].append(e)
    print('# 레짐  ' + '  '.join(f'{k}={len(v)}' for k, v in sorted(by_reg.items())))

    rows = []
    for stat in ('window_resid', 'all_resid'):
        print(f'\n{"="*78}\n## 선택 통계량 = {stat}\n{"="*78}')
        for reg, E in sorted(by_reg.items()):
            selT = selector(med, 'AUTO-L1', 'V2K', E, stat)
            selN = selector(med, 'AUTO', 'AUTO-SEED-NULL', E, stat)
            E2 = [e for e in E if e in selT and e in selN]

            relT = {e: (selT[e][1] - selT[e][2]) / selT[e][2] for e in E2}
            relN = {e: (selN[e][1] - selN[e][2]) / selN[e][2] for e in E2}
            did = {e: relT[e] - relN[e] for e in E2}

            sT = summarise('처치', [(selT[e][1], selT[e][2]) for e in E2])
            sN = summarise('널  ', [(selN[e][1], selN[e][2]) for e in E2])

            d = [did[e] for e in E2]
            di = sum(1 for v in d if v < 0)
            dw = sum(1 for v in d if v > 0)
            dt = sum(1 for v in d if v == 0)
            dp = binom_two_sided(di, di + dw)

            # 전환 시 승률
            def switch(sel, arm0):
                w = l = 0
                for e in E2:
                    if sel[e][0] != arm0:
                        if sel[e][1] < sel[e][2]:
                            w += 1
                        elif sel[e][1] > sel[e][2]:
                            l += 1
                return w, l
            wT, lT = switch(selT, 'AUTO-L1')
            wN, lN = switch(selN, 'AUTO')
            p0 = wN / (wN + lN) if (wN + lN) else float('nan')
            pS = binom_one_sided_ge(wT, wT + lT, p0) if (wT + lT) and (wN + lN) else float('nan')

            print(f'\n### {reg}  n={len(E2)}')
            for s in (sT, sN):
                print(f'  {s["name"]}  개선/악화/동점 {s["imp"]:>3}/{s["wor"]:>3}/{s["tie"]:>3}  '
                      f'p={s["p"]:.3e}  중앙rel={s["med_rel"]*100:+7.2f}%  '
                      f'(동점제외 {s["med_rel_nonzero"]*100:+7.2f}%)')
            print(f'  ★DiD  개선/악화/동점 {di:>3}/{dw:>3}/{dt:>3}  p={dp:.3e}  '
                  f'중앙DiD={st.median(d)*100:+7.2f}%  '
                  f'(동점제외 {st.median([v for v in d if v])*100:+7.2f}%)' if (di + dw) else '')
            print(f'  전환승률  처치 {wT}/{wT+lT}'
                  f'{f" = {wT/(wT+lT):.3f}" if (wT+lT) else ""}   '
                  f'널 {wN}/{wN+lN}{f" = {p0:.3f}" if (wN+lN) else ""}   '
                  f'H0=널 단측 p={pS:.4f}' if (wT + lT) and (wN + lN) else '')
            rows.append(dict(stat=stat, regime=reg, n=len(E2),
                             T=sT, N=sN, did_imp=di, did_wor=dw, did_tie=dt, did_p=dp,
                             did_med=st.median(d) if d else float('nan'),
                             sw_T=[wT, lT], sw_N=[wN, lN], p0=p0, p_switch=pS))

    # ------------------------------------------------------------------ #
    # ★C. 짝이 **정말로** 맞는 대조 — 같은 팔 A, 진짜 대안 vs 가짜 대안
    #    SEL_real = min-resid over {AUTO, V2K}        (대안이 실재하는 방법)
    #    SEL_null = min-resid over {AUTO, AUTO-SEED-NULL} (대안이 같은 설정·다른 시드)
    #    둘 다 A=AUTO 를 공유하고 둘 다 "두 번 뽑아 고르기" 이므로,
    #    차이 = **방법을 고른 값** 그 자체. (§A 의 DiD 는 서로 다른 기준선을 빼서
    #    독립 잡음을 더했다 — 검정력이 낮다.)
    # ------------------------------------------------------------------ #
    print(f'\n{"="*78}\n## ★C. 짝맞춤 대조: SELECT{{AUTO,V2K}} vs SELECT{{AUTO,AUTO-SEED-NULL}}'
          f'\n##    (같은 팔 A, 같은 규칙, 같은 "2회 추첨" — 차이 = 방법 선택의 값)\n{"="*78}')
    cc = []
    for stat in ('window_resid', 'all_resid'):
        for reg, E in sorted(by_reg.items()):
            sR = selector(med, 'AUTO', 'V2K', E, stat)
            sN = selector(med, 'AUTO', 'AUTO-SEED-NULL', E, stat)
            E2 = [e for e in E if e in sR and e in sN]
            rel = [(sR[e][1] - sN[e][1]) / sN[e][1] for e in E2]
            imp = sum(1 for r in rel if r < 0)
            wor = sum(1 for r in rel if r > 0)
            tie = sum(1 for r in rel if r == 0)
            p = binom_two_sided(imp, imp + wor) if (imp + wor) else float('nan')
            nz = [r for r in rel if r != 0]
            print(f'  {stat:<13} {reg:<8} n={len(E2):>3}  개선/악화/동점 {imp:>3}/{wor:>3}/{tie:>3}  '
                  f'p={p:.3e}  중앙rel={st.median(rel)*100:+7.2f}%  '
                  f'(동점제외 {st.median(nz)*100:+7.2f}%)' if nz else
                  f'  {stat:<13} {reg:<8} n={len(E2):>3}  전건 동점')
            cc.append(dict(stat=stat, regime=reg, n=len(E2), imp=imp, wor=wor, tie=tie, p=p,
                           med_rel=st.median(rel) if rel else float('nan'),
                           med_rel_nonzero=st.median(nz) if nz else float('nan')))

    # ------------------------------------------------------------------ #
    # ★D. 도달 수준 (레벨) — 무엇을 쓰든 실제 nRMSE 중앙값은 얼마인가
    # ------------------------------------------------------------------ #
    print(f'\n{"="*78}\n## ★D. 도달 수준 = 에피소드 중앙 nRMSE (낮을수록 좋음)\n{"="*78}')
    lv = {}
    for reg, E in sorted(by_reg.items()):
        row = {}
        for arm in ('AUTO', 'AUTO-L1', 'V2K', 'AUTO-SEED-NULL'):
            v = [med[(arm, e)]['nRMSE'] for e in E if (arm, e) in med]
            row[arm] = st.median(v) if v else float('nan')
        for nm, (a, b) in {'SEL{AUTO,V2K}': ('AUTO', 'V2K'),
                           'SEL{AUTO-L1,V2K}': ('AUTO-L1', 'V2K'),
                           'SEL-NULL{AUTO,AUTO-SEED-NULL}': ('AUTO', 'AUTO-SEED-NULL')}.items():
            s = selector(med, a, b, E, 'window_resid')
            row[nm] = st.median([s[e][1] for e in E if e in s])
        # 전지 오라클 = 두 팔 중 nRMSE 가 실제로 낮은 쪽 (GT 필요, 천장)
        orc = []
        for e in E:
            xs = [med[(a, e)]['nRMSE'] for a in ('AUTO-L1', 'V2K') if (a, e) in med]
            if len(xs) == 2:
                orc.append(min(xs))
        row['ORACLE{AUTO-L1,V2K}'] = st.median(orc)
        lv[reg] = row
        print(f'\n### {reg} (n={len(E)})')
        for k, v in row.items():
            print(f'  {k:<32s} {v:.5f}')

    # ------------------------------------------------------------------ #
    # ★E. 에피소드별 선택이 **고정 레짐 규칙**을 이기는가
    #    REGIME-FIXED = 희소면 V2K, 혼잡이면 AUTO-L1 (레짐은 GT-free 로 알 수 없다 —
    #    이것은 선택기가 넘어야 할 **오라클 상한**이지 배포 가능한 규칙이 아니다).
    # ------------------------------------------------------------------ #
    print(f'\n{"="*78}\n## ★E. 에피소드별 선택 vs 고정 레짐 규칙 (희소→V2K / 혼잡→AUTO-L1)\n{"="*78}')
    ee = []
    for reg, E in sorted(by_reg.items()):
        fixed = 'V2K' if reg == 'sparse' else 'AUTO-L1'
        s = selector(med, 'AUTO-L1', 'V2K', E, 'window_resid')
        rel = [(s[e][1] - med[(fixed, e)]['nRMSE']) / med[(fixed, e)]['nRMSE']
               for e in E if e in s and (fixed, e) in med]
        imp = sum(1 for r in rel if r < 0)
        wor = sum(1 for r in rel if r > 0)
        tie = sum(1 for r in rel if r == 0)
        p = binom_two_sided(imp, imp + wor) if (imp + wor) else float('nan')
        nz = [r for r in rel if r != 0]
        print(f'  {reg:<8} vs 고정({fixed:<7})  개선/악화/동점 {imp:>3}/{wor:>3}/{tie:>3}  p={p:.3e}  '
              f'중앙rel={st.median(rel)*100:+7.2f}%  (동점제외 {st.median(nz)*100:+7.2f}%)'
              if nz else f'  {reg:<8} vs 고정({fixed}) 전건 동점')
        ee.append(dict(regime=reg, fixed=fixed, imp=imp, wor=wor, tie=tie, p=p,
                       med_rel=st.median(rel) if rel else float('nan')))

    # ★F. 팔 자체 대조 (X3 §1 재확인) — V2K vs AUTO-L1
    print(f'\n{"="*78}\n## ★F. 팔 대조 V2K − AUTO-L1 (부호검정)\n{"="*78}')
    ff = []
    for reg, E in sorted(by_reg.items()):
        rel = [(med[('V2K', e)]['nRMSE'] - med[('AUTO-L1', e)]['nRMSE']) / med[('AUTO-L1', e)]['nRMSE']
               for e in E if ('V2K', e) in med and ('AUTO-L1', e) in med]
        imp = sum(1 for r in rel if r < 0)
        wor = sum(1 for r in rel if r > 0)
        p = binom_two_sided(imp, imp + wor)
        print(f'  {reg:<8} V2K 승/패 {imp:>3}/{wor:>3}  p={p:.3e}  중앙rel={st.median(rel)*100:+8.2f}%')
        ff.append(dict(regime=reg, v2k_win=imp, v2k_lose=wor, p=p, med_rel=st.median(rel)))

    with open(os.path.join(HERE, 'null_corrected.json'), 'w') as f:
        json.dump({'A_did': rows, 'C_matched': cc, 'D_levels': lv,
                   'E_vs_regime_fixed': ee, 'F_arm_contrast': ff}, f, indent=1, default=float)
    print('\n→ null_corrected.json')


if __name__ == '__main__':
    main()
