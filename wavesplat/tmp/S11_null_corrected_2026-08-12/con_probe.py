#!/usr/bin/env python3
"""S11 CON 감사 — 독립 재도출 + 사망 시도.

⛔`null_corrected.py` 를 import 하지 않는다. 전부 원자료에서 다시 센다.

블록:
  R  재도출        BRIEF §1.2 / §2 / §F 표를 독립 코드로 복제
  A  λ 교락        SEL{AUTO,AUTO-L1}(λ만 다른 두 팔)에 같은 규칙 → −35.8% 중 얼마인가
  B  팔 vs 규칙    ALWAYS-x vs SEL_null,  SEL vs ALWAYS-x,  구조 매칭된 널
  C  꼬리/잭나이프 규칙이 always-V2K 를 이기는 것이 몇 에피소드 때문인가
  D  시드 분산     팔별 시드간 산포 (널 보정의 교정 상수)
  E  팔내 2추첨널  같은 팔 3시드로 만든 "2뽑기 vs 1뽑기" 이득 (팔별)
  F  §4-A 기제     final_n / n_selected / 참 성분수 → 반복예산이 정말 범인인가
  G  비용          wall_s 원자료
"""
import json, os, math, itertools, statistics as st
from collections import defaultdict

TMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S9 = os.path.join(TMP, 'S9_lr_x_l1_2026-08-11', 'results.jsonl')
X2 = os.path.join(TMP, 'X2_no_learned_2026-08-11', 'results.jsonl')
S8 = os.path.join(TMP, 'S8_deploy_confirm_2026-08-10', 'results.jsonl')
S7 = os.path.join(TMP, 'S7_confirm_n32_2026-08-10', 'results.jsonl')

MET = ('nRMSE', 'window_resid', 'all_resid', 'avg_obs_loss', 'nRMSE_p95', 'nRMSE_max',
       'final_n', 'auto_n_selected', 'wall_s', 'init_nrmse', 'init_iters', 'ensemble_k')


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def regime(ep):
    b = os.path.basename(ep)
    if '_2comp_' in b or '_5comp_' in b:
        return 'sparse'
    if '_15comp_' in b or '_30comp_' in b:
        return 'crowded'
    raise ValueError(ep)


def ncomp(ep):
    b = os.path.basename(ep)
    for n in (30, 15, 5, 2):
        if f'_{n}comp_' in b:
            return n
    raise ValueError(ep)


def runs(recs):
    """(cell, ep, seed) -> rec   (에러 제외)"""
    out = {}
    n_err = 0
    for r in recs:
        if 'error' in r:
            n_err += 1
            continue
        out[(r['cell'], os.path.basename(r['episode']), r['seed'])] = r
    return out, n_err


def medians(runmap):
    """(cell, ep) -> {metric: 3시드 중앙값}"""
    acc = defaultdict(lambda: defaultdict(list))
    for (c, e, s), r in runmap.items():
        for m in MET:
            v = r.get(m)
            if isinstance(v, (int, float)) and math.isfinite(v):
                acc[(c, e)][m].append(v)
    return {k: {m: st.median(v) for m, v in d.items() if v} for k, d in acc.items()}


def binom2(k, n, p=0.5):
    if n == 0:
        return float('nan')
    pmf = [math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return min(1.0, sum(x for x in pmf if x <= pmf[k] * (1 + 1e-12)))


def signtest(rel):
    """rel<0 = 개선."""
    imp = sum(1 for r in rel if r < 0)
    wor = sum(1 for r in rel if r > 0)
    tie = sum(1 for r in rel if r == 0)
    nz = [r for r in rel if r != 0]
    return dict(n=len(rel), imp=imp, wor=wor, tie=tie, p=binom2(imp, imp + wor),
                med=st.median(rel) if rel else float('nan'),
                med_nz=st.median(nz) if nz else float('nan'),
                mean=sum(rel) / len(rel) if rel else float('nan'))


def fmt(tag, s):
    return (f'{tag:<52s} n={s["n"]:>3} 개선/악화/동점 {s["imp"]:>3}/{s["wor"]:>3}/{s["tie"]:>3} '
            f'p={s["p"]:.3e} 중앙={s["med"]*100:+8.2f}% 동점제외={s["med_nz"]*100:+8.2f}% '
            f'평균={s["mean"]*100:+9.2f}%')


# ────────────────────────────────────────────────────────────────────────────
r9, e9 = runs(load(S9))
r2, e2 = runs(load(X2))
R = {**r9, **r2}
M = medians(R)
EPS = sorted({e for (c, e, s) in r9 if c == 'AUTO-L1'})
REG = {e: regime(e) for e in EPS}
BY = defaultdict(list)
for e in EPS:
    BY[REG[e]].append(e)
ARMS = sorted({c for (c, e, s) in R})

print(f'# S9 에러 {e9} · X2 에러 {e2} · 에피소드 {len(EPS)} '
      f'(희소 {len(BY["sparse"])} / 혼잡 {len(BY["crowded"])}) · 팔 {ARMS}')

# 결정성 확인 — AUTO-DUP 은 AUTO 와 cfg·시드 동일
dd = [abs(R[('AUTO', e, s)]['nRMSE'] - R[('AUTO-DUP', e, s)]['nRMSE'])
      for e in EPS for s in (1575, 1576, 1577)
      if ('AUTO', e, s) in R and ('AUTO-DUP', e, s) in R]
print(f'# 결정성(AUTO vs AUTO-DUP, 동일 cfg·동일 시드) n={len(dd)} max|Δ|={max(dd):.3e} '
      f'nonzero={sum(1 for x in dd if x>0)}')


def sel(armA, armB, e, stat='window_resid', met='nRMSE'):
    """에피소드 e 에서 잔차 작은 팔을 고르고 그 팔의 met 을 돌려준다."""
    a, b = M.get((armA, e)), M.get((armB, e))
    if not a or not b:
        return None
    c = armA if a[stat] <= b[stat] else armB
    return c, M[(c, e)][met]


# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 100)
print('## R. 재도출 — BRIEF §1.2 짝맞춤 대조  SEL{AUTO,V2K} vs SEL{AUTO,AUTO-SEED-NULL}')
print('=' * 100)
for stat in ('window_resid', 'all_resid'):
    for reg in ('sparse', 'crowded'):
        rel = []
        for e in BY[reg]:
            a = sel('AUTO', 'V2K', e, stat)
            b = sel('AUTO', 'AUTO-SEED-NULL', e, stat)
            if a and b:
                rel.append((a[1] - b[1]) / b[1])
        print(fmt(f'  {stat:<13}{reg:<9} SEL_real vs SEL_null', signtest(rel)))

print('\n' + '=' * 100)
print('## R2. BRIEF §2 도달 수준 (에피소드 중앙 nRMSE)')
print('=' * 100)
LEVEL = {}
for reg in ('sparse', 'crowded'):
    row = {}
    for arm in ('AUTO', 'AUTO-L1', 'V2K', 'AUTO-SEED-NULL'):
        row[arm] = st.median([M[(arm, e)]['nRMSE'] for e in BY[reg]])
    for nm, (a, b) in {'SEL{AUTO,V2K}': ('AUTO', 'V2K'),
                       'SEL{AUTO-L1,V2K}': ('AUTO-L1', 'V2K'),
                       'SEL{AUTO,AUTO-L1}': ('AUTO', 'AUTO-L1'),
                       'SEL-NULL{AUTO,SEEDNULL}': ('AUTO', 'AUTO-SEED-NULL')}.items():
        row[nm] = st.median([sel(a, b, e)[1] for e in BY[reg]])
    row['ORACLE{AUTO-L1,V2K}'] = st.median(
        [min(M[('AUTO-L1', e)]['nRMSE'], M[('V2K', e)]['nRMSE']) for e in BY[reg]])
    LEVEL[reg] = row
    print(f'\n### {reg} n={len(BY[reg])}')
    for k, v in row.items():
        print(f'   {k:<28s} {v:.5f}')

# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 100)
print('## A. ★λ 교락 — 팔 B 를 바꿔가며 **같은 규칙·같은 널**과 대조')
print('   대조군: SEL{AUTO,X} vs SEL_null{AUTO,AUTO-SEED-NULL}, 상대 = (real-null)/null')
print('=' * 100)
for stat in ('window_resid', 'all_resid'):
    print(f'\n--- 선택 통계량 {stat} ---')
    for reg in ('sparse', 'crowded'):
        for armB in ('V2K', 'AUTO-L1', 'AUTO-QTR-L1', 'AUTO-NT', 'AUTO-QTR', 'AUTO-DUP'):
            rel = []
            for e in BY[reg]:
                a = sel('AUTO', armB, e, stat)
                b = sel('AUTO', 'AUTO-SEED-NULL', e, stat)
                if a and b:
                    rel.append((a[1] - b[1]) / b[1])
            print(fmt(f'  {reg:<9}SEL{{AUTO,{armB}}} vs SEL_null', signtest(rel)))

print('\n' + '-' * 100)
print('## A2. ★선택을 아예 빼고 — **항상 X** vs SEL_null (규칙 0회 사용)')
print('-' * 100)
for reg in ('sparse', 'crowded'):
    for armB in ('V2K', 'AUTO-L1', 'AUTO'):
        rel = []
        for e in BY[reg]:
            b = sel('AUTO', 'AUTO-SEED-NULL', e)
            if b:
                rel.append((M[(armB, e)]['nRMSE'] - b[1]) / b[1])
        print(fmt(f'  {reg:<9}ALWAYS-{armB} vs SEL_null', signtest(rel)))

# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 100)
print('## B. ★규칙 vs 고정 팔 — 그리고 **구조가 같은 널**')
print('   규칙은 "항상 팔B" 와 정확히 "팔A 를 고른 에피소드"에서만 다르다.')
print('=' * 100)
for reg in ('sparse', 'crowded'):
    print(f'\n### {reg}')
    for (a, b) in (('AUTO-L1', 'V2K'), ('AUTO', 'V2K'), ('AUTO', 'AUTO-L1'),
                   ('AUTO', 'AUTO-SEED-NULL')):
        for fixed in (a, b):
            rel, nswitch = [], 0
            for e in BY[reg]:
                s_ = sel(a, b, e)
                if not s_:
                    continue
                if s_[0] != fixed:
                    nswitch += 1
                rel.append((s_[1] - M[(fixed, e)]['nRMSE']) / M[(fixed, e)]['nRMSE'])
            print(fmt(f'  SEL{{{a},{b}}} vs ALWAYS-{fixed} (전환 {nswitch})', signtest(rel)))

# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 100)
print('## C. ★꼬리 / 잭나이프 — 규칙이 always-V2K 를 이기는 것이 몇 에피소드 때문인가 (희소)')
print('=' * 100)
for (a, b, fixed) in (('AUTO-L1', 'V2K', 'V2K'), ('AUTO', 'AUTO-SEED-NULL', 'AUTO-SEED-NULL')):
    rel = []
    for e in BY['sparse']:
        s_ = sel(a, b, e)
        rel.append(((s_[1] - M[(fixed, e)]['nRMSE']) / M[(fixed, e)]['nRMSE'], e))
    rel.sort()
    base = signtest([r for r, _ in rel])
    print(f'\n### SEL{{{a},{b}}} vs ALWAYS-{fixed}')
    print(fmt('   전체', base))
    print('   가장 크게 이긴 5개:', [f'{r*100:+.1f}%' for r, _ in rel[:5]])
    print('   가장 크게 진   5개:', [f'{r*100:+.1f}%' for r, _ in rel[-5:]])
    for k in (1, 2, 3, 5, 8, 12):
        sub = [r for r, _ in rel[k:]]            # 가장 크게 이긴 k 개 제거
        s2 = signtest(sub)
        print(f'   상위 {k:>2}개 제거 → 개선/악화 {s2["imp"]:>3}/{s2["wor"]:>3} '
              f'p={s2["p"]:.4f} 중앙={s2["med"]*100:+7.2f}% 평균={s2["mean"]*100:+8.2f}%')
    # 수준(중앙 nRMSE) 기준 잭나이프
    lv_rule = st.median([sel(a, b, e)[1] for e in BY['sparse']])
    lv_fix = st.median([M[(fixed, e)]['nRMSE'] for e in BY['sparse']])
    print(f'   수준: 규칙 중앙 {lv_rule:.5f} vs 항상{fixed} 중앙 {lv_fix:.5f} '
          f'= {(lv_rule-lv_fix)/lv_fix*100:+.2f}%')

# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 100)
print('## D. ★시드 분산 — 널 보정의 교정 상수 (팔마다 "운"의 크기가 다르다)')
print('   상대범위 = (max-min)/median  ·  에피소드별 3시드')
print('=' * 100)
for reg in ('sparse', 'crowded'):
    print(f'\n### {reg}')
    for arm in ('AUTO', 'AUTO-SEED-NULL', 'AUTO-L1', 'V2K', 'AUTO-NT'):
        rr, rw = [], []
        for e in BY[reg]:
            v = [R[(arm, e, s)]['nRMSE'] for s in sorted({s for (c, x, s) in R if c == arm and x == e})]
            w = [R[(arm, e, s)]['window_resid'] for s in sorted({s for (c, x, s) in R if c == arm and x == e})]
            if len(v) >= 2 and st.median(v) > 0:
                rr.append((max(v) - min(v)) / st.median(v))
            if len(w) >= 2 and st.median(w) > 0:
                rw.append((max(w) - min(w)) / st.median(w))
        print(f'  {arm:<16s} nRMSE 상대범위 중앙 {st.median(rr)*100:6.1f}%  '
              f'평균 {sum(rr)/len(rr)*100:6.1f}%   |  window_resid 상대범위 중앙 {st.median(rw)*100:6.1f}%')

# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 100)
print('## E. ★팔내 2추첨 널 — 같은 팔 3시드로 "2뽑기(잔차선택) vs 1뽑기"')
print('   에피소드마다 3가지 회전(참조 1시드 / 후보 2시드). 팔마다 "운"이 얼마나 큰가.')
print('=' * 100)
for reg in ('sparse', 'crowded'):
    print(f'\n### {reg}')
    for arm in ('AUTO', 'AUTO-SEED-NULL', 'AUTO-L1', 'V2K'):
        rel = []
        for e in BY[reg]:
            sds = sorted({s for (c, x, s) in R if c == arm and x == e})
            if len(sds) < 3:
                continue
            for ref in sds:
                cand = [s for s in sds if s != ref]
                pick = min(cand, key=lambda s: R[(arm, e, s)]['window_resid'])
                base = R[(arm, e, ref)]['nRMSE']
                if base > 0:
                    rel.append((R[(arm, e, pick)]['nRMSE'] - base) / base)
        print(fmt(f'  {arm:<16s} 2뽑기 vs 1뽑기', signtest(rel)))

# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 100)
print('## F. ★§4-A 기제 — 반복예산이 범인인가, N 과다가 범인인가 (희소 learned 라우트)')
print('=' * 100)
strat = defaultdict(list)
for e in BY['sparse']:
    rt = st.mode([R[('AUTO-L1', e, s)].get('auto_selected') for s in (1575, 1576, 1577)
                  if ('AUTO-L1', e, s) in R])
    strat[rt].append(e)
for rt, E in sorted(strat.items()):
    rel = [(M[('V2K', e)]['nRMSE'] - M[('AUTO-L1', e)]['nRMSE']) / M[('AUTO-L1', e)]['nRMSE'] for e in E]
    print(fmt(f'  희소/{rt} (n={len(E)})  V2K − AUTO-L1', signtest(rel)))

E53 = strat.get('learned', [])
print(f'\n### 희소·learned 라우트 {len(E53)} 에피소드에서의 N (중앙)')
print(f'{"참성분":>6} {"n_ep":>5} {"n_selected":>11} {"final_n(L1)":>12} {"final_n(V2K)":>13} '
      f'{"med rel(V2K-L1)":>16}')
for nc in (2, 5):
    E = [e for e in E53 if ncomp(e) == nc]
    if not E:
        continue
    ns = st.median([M[('AUTO-L1', e)]['auto_n_selected'] for e in E if 'auto_n_selected' in M[('AUTO-L1', e)]])
    fl = st.median([M[('AUTO-L1', e)]['final_n'] for e in E])
    fv = st.median([M[('V2K', e)]['final_n'] for e in E])
    rl = st.median([(M[('V2K', e)]['nRMSE'] - M[('AUTO-L1', e)]['nRMSE']) / M[('AUTO-L1', e)]['nRMSE'] for e in E])
    print(f'{nc:>6} {len(E):>5} {ns:>11.1f} {fl:>12.1f} {fv:>13.1f} {rl*100:>15.1f}%')

# N 과다와 격차의 상관 (Spearman)
def spearman(x, y):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    a, b = rk(x), rk(y)
    n = len(x)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    den = math.sqrt(sum((a[i] - ma) ** 2 for i in range(n)) * sum((b[i] - mb) ** 2 for i in range(n)))
    r = num / den if den else float('nan')
    z = r * math.sqrt(n - 1)
    p = math.erfc(abs(z) / math.sqrt(2))
    return r, p

xs = [M[('AUTO-L1', e)]['final_n'] - ncomp(e) for e in E53]
ys = [(M[('V2K', e)]['nRMSE'] - M[('AUTO-L1', e)]['nRMSE']) / M[('AUTO-L1', e)]['nRMSE'] for e in E53]
r_, p_ = spearman(xs, ys)
print(f'\n  Spearman( final_n(AUTO-L1) − 참성분수 , rel(V2K−AUTO-L1) )  ρ={r_:+.3f} p={p_:.4f}  n={len(xs)}')
xs2 = [M[('V2K', e)]['final_n'] - ncomp(e) for e in E53]
r2_, p2_ = spearman(xs2, ys)
print(f'  Spearman( final_n(V2K)     − 참성분수 , rel(V2K−AUTO-L1) )  ρ={r2_:+.3f} p={p2_:.4f}')

print('\n### init_nrmse (초기화 직후) — 반복 500 이 아직 수렴 중인가 vs 이미 나쁜 곳인가')
for reg in ('sparse', 'crowded'):
    for arm in ('AUTO-L1', 'V2K'):
        E = BY[reg]
        i0 = st.median([M[(arm, e)]['init_nrmse'] for e in E if 'init_nrmse' in M[(arm, e)]])
        f0 = st.median([M[(arm, e)]['nRMSE'] for e in E])
        print(f'  {reg:<9}{arm:<9} init_nrmse={i0:.5f}  최종 nRMSE={f0:.5f}  '
              f'추적단계 변화 {(f0-i0)/i0*100:+.1f}%')

# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 100)
print('## G. 비용 (wall_s 원자료 중앙)')
print('=' * 100)
for arm in ('AUTO', 'AUTO-L1', 'V2K', 'AUTO-SEED-NULL'):
    for reg in ('sparse', 'crowded'):
        w = [R[(arm, e, s)]['wall_s'] for e in BY[reg]
             for s in sorted({s for (c, x, s) in R if c == arm and x == e})]
        print(f'  {arm:<16s}{reg:<9} 중앙 {st.median(w):6.1f}s  합계 {sum(w)/3600:5.2f} GPU-h  n={len(w)}')

# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 100)
print('## H. S8 독립 표본에서 널 규칙이 재현되는가 (다른 64 에피소드·다른 시드)')
print('=' * 100)
r8, e8 = runs(load(S8))
M8 = medians(r8)
EPS8 = sorted({e for (c, e, s) in r8 if c == 'AUTO'})
BY8 = defaultdict(list)
for e in EPS8:
    BY8[regime(e)].append(e)
print(f'  S8 에러 {e8} · 에피소드 {len(EPS8)} (희소 {len(BY8["sparse"])}/혼잡 {len(BY8["crowded"])})')
for reg in ('sparse', 'crowded'):
    for a, b in (('AUTO', 'AUTO-SEED-NULL'), ('AUTO', 'AUTO-DUP2')):
        rel = []
        for e in BY8[reg]:
            if (a, e) not in M8 or (b, e) not in M8:
                continue
            c = a if M8[(a, e)]['window_resid'] <= M8[(b, e)]['window_resid'] else b
            rel.append((M8[(c, e)]['nRMSE'] - M8[(a, e)]['nRMSE']) / M8[(a, e)]['nRMSE'])
        print(fmt(f'  S8 {reg:<9}SEL{{{a},{b}}} vs {a}', signtest(rel)))

# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — §4-A 기제 / §4-B 결정영향 / 다중성 / 레짐 신호
# ═══════════════════════════════════════════════════════════════════════════
print('\n\n' + '#' * 100)
print('# PART 2')
print('#' * 100)

print('\n## I. ★§4-A 과녁 검증 — 격차는 Phase-1 끝점에 있는가, 추적에 있는가')
print('   init_nrmse = Phase-1 종료 직후 마지막 init 프레임의 전 격자 nRMSE (run.py:186-189)')
for reg, E in (('sparse/learned', E53), ('sparse/v2_konly', strat.get('v2_konly', [])),
               ('crowded', BY['crowded'])):
    if not E:
        continue
    ri = [(M[('V2K', e)]['init_nrmse'] - M[('AUTO-L1', e)]['init_nrmse']) / M[('AUTO-L1', e)]['init_nrmse']
          for e in E if 'init_nrmse' in M[('V2K', e)] and 'init_nrmse' in M[('AUTO-L1', e)]]
    rf = [(M[('V2K', e)]['nRMSE'] - M[('AUTO-L1', e)]['nRMSE']) / M[('AUTO-L1', e)]['nRMSE'] for e in E]
    print(fmt(f'  {reg:<18} init_nrmse  V2K−AUTO-L1', signtest(ri)))
    print(fmt(f'  {reg:<18} 최종 nRMSE  V2K−AUTO-L1', signtest(rf)))
    for arm in ('AUTO-L1', 'V2K'):
        dg = [(M[(arm, e)]['nRMSE'] - M[(arm, e)]['init_nrmse']) / M[(arm, e)]['init_nrmse'] for e in E]
        print(f'    {arm:<8} 추적단계 열화 중앙 {st.median(dg)*100:+8.1f}%  '
              f'(init {st.median([M[(arm,e)]["init_nrmse"] for e in E]):.5f} → '
              f'{st.median([M[(arm,e)]["nRMSE"] for e in E]):.5f})')

print('\n  ★같은 층에서 추적을 끈 팔(AUTO-NT, λ=1e-3)과의 수준 비교 (희소 learned 라우트)')
for arm in ('AUTO', 'AUTO-L1', 'AUTO-NT', 'AUTO-QTR', 'V2K'):
    print(f'    {arm:<12} 중앙 nRMSE {st.median([M[(arm,e)]["nRMSE"] for e in E53]):.5f}  '
          f'init {st.median([M[(arm,e)]["init_nrmse"] for e in E53]):.5f}')

print('\n## J. ★§4-B 널의 상한 — 이미 자료 안에 있다')
print('   구조가 정확히 같은 널(3시드중앙 vs 3시드중앙, 동일 cfg): AUTO 만 존재')
for reg in ('sparse', 'crowded'):
    rel = []
    for e in BY[reg]:
        s_ = sel('AUTO', 'AUTO-SEED-NULL', e)
        rel.append((s_[1] - M[('AUTO', e)]['nRMSE']) / M[('AUTO', e)]['nRMSE'])
    print(fmt(f'  {reg:<9}SEL_null vs ALWAYS-AUTO (=순수 2뽑기 이득)', signtest(rel)))
    rel = []
    for e in BY[reg]:
        s_ = sel('AUTO-L1', 'V2K', e)
        rel.append((s_[1] - M[('AUTO-L1', e)]['nRMSE']) / M[('AUTO-L1', e)]['nRMSE'])
    print(fmt(f'  {reg:<9}SEL{{AUTO-L1,V2K}} vs ALWAYS-AUTO-L1 (=처치)', signtest(rel)))

print('\n## K. 다중성 — §1.2 헤드라인이 고른 셀의 개수')
cells = []
for stat in ('window_resid', 'all_resid'):
    for reg in ('sparse', 'crowded'):
        for armB in ('V2K', 'AUTO-L1', 'AUTO-QTR-L1', 'AUTO-NT', 'AUTO-QTR', 'AUTO-DUP'):
            rel = []
            for e in BY[reg]:
                a, b = sel('AUTO', armB, e, stat), sel('AUTO', 'AUTO-SEED-NULL', e, stat)
                rel.append((a[1] - b[1]) / b[1])
            s_ = signtest(rel)
            cells.append((stat, reg, armB, s_['p'], s_['med']))
sig = [c for c in cells if c[3] < 0.05]
print(f'  이용 가능한 (신호×레짐×팔B) 셀 = {len(cells)},  p<0.05 = {len(sig)}')
for c in sorted(cells, key=lambda z: z[3])[:8]:
    print(f'    {c[0]:<13}{c[1]:<9}{c[2]:<13} p={c[3]:.2e} 중앙={c[4]*100:+7.2f}%')

print('\n## L. 레짐을 GT-free 로 아는가 — auto_n_selected 분포')
for reg in ('sparse', 'crowded'):
    v = sorted(M[('AUTO-L1', e)]['auto_n_selected'] for e in BY[reg] if 'auto_n_selected' in M[('AUTO-L1', e)])
    print(f'  {reg:<9} n_selected  min={v[0]:.0f} p25={v[len(v)//4]:.0f} 중앙={st.median(v):.1f} '
          f'p75={v[3*len(v)//4]:.0f} max={v[-1]:.0f}  (n={len(v)})')
sp = [M[('AUTO-L1', e)]['auto_n_selected'] for e in BY['sparse'] if 'auto_n_selected' in M[('AUTO-L1', e)]]
cr = [M[('AUTO-L1', e)]['auto_n_selected'] for e in BY['crowded'] if 'auto_n_selected' in M[('AUTO-L1', e)]]
best = max(((t, sum(1 for x in sp if x < t) + sum(1 for x in cr if x >= t)) for t in range(1, 31)),
           key=lambda z: z[1])
print(f'  최적 단일 문턱 t={best[0]} 정확도 {best[1]}/{len(sp)+len(cr)} = {best[1]/(len(sp)+len(cr)):.3f} '
      f'(기저율 {max(len(sp),len(cr))/(len(sp)+len(cr)):.3f})')

print('\n## M. §1.2 헤드라인의 꼬리 (희소 window_resid)')
rel = []
for e in BY['sparse']:
    a, b = sel('AUTO', 'V2K', e), sel('AUTO', 'AUTO-SEED-NULL', e)
    rel.append(((a[1] - b[1]) / b[1], e))
rel.sort()
print('  최악 6개:', [f'{r*100:+.0f}%' for r, _ in rel[-6:]])
print('  최선 3개:', [f'{r*100:+.0f}%' for r, _ in rel[:3]])
print(f'  중앙={st.median([r for r,_ in rel])*100:+.2f}%  평균={sum(r for r,_ in rel)/len(rel)*100:+.2f}%  '
      f'p95 팔수준: AUTO-L1 {st.median([M[("AUTO-L1",e)]["nRMSE_p95"] for e in BY["sparse"]]):.4f} '
      f'V2K {st.median([M[("V2K",e)]["nRMSE_p95"] for e in BY["sparse"]]):.4f}')

print('\n## N. §1.2 최악 에피소드 해부')
worst = rel[-1][1]
print(f'  {worst}')
for arm in ('AUTO', 'AUTO-SEED-NULL', 'V2K', 'AUTO-L1'):
    print(f'    {arm:<16} nRMSE={M[(arm,worst)]["nRMSE"]:.5f}  window_resid={M[(arm,worst)]["window_resid"]:.5f}  '
          f'시드별 nRMSE=' + ' '.join(f'{R[(arm,worst,s)]["nRMSE"]:.4f}'
                                     for s in sorted({s for (c,x,s) in R if c==arm and x==worst})))
print(f'  SEL_real 선택={sel("AUTO","V2K",worst)}  SEL_null 선택={sel("AUTO","AUTO-SEED-NULL",worst)}')

print('\n## O. §1.2 를 **출하 팔**(AUTO-L1, 2026-08-11 배포) 위에서 다시 — 널 팔이 자료에 없다')
print('   가능한 것: 처치(=S10 헤드라인) 와, AUTO 위에서 잰 동일구조 널의 크기')
for reg in ('sparse', 'crowded'):
    a = signtest([(sel('AUTO-L1', 'V2K', e)[1] - M[('AUTO-L1', e)]['nRMSE']) / M[('AUTO-L1', e)]['nRMSE']
                  for e in BY[reg]])
    b = signtest([(sel('AUTO', 'AUTO-SEED-NULL', e)[1] - M[('AUTO', e)]['nRMSE']) / M[('AUTO', e)]['nRMSE']
                  for e in BY[reg]])
    print(f'  {reg:<9} 처치 동점제외 {a["med_nz"]*100:+7.2f}% ({a["imp"]}/{a["wor"]})   '
          f'AUTO 위 널 동점제외 {b["med_nz"]*100:+7.2f}% ({b["imp"]}/{b["wor"]})   '
          f'비 {a["med_nz"]/b["med_nz"] if b["med_nz"] else float("nan"):.2f}배')

print('\n## P. S9 실측 처리량 → S11 비용')
print('  S9 autostart.log 12:36:46 → 15:36:46 = 3.00 h / 2016런 = 11.2 런/분 (8워커, 1 GPU)')
for n, lbl in ((612, 'S11 3셀 (288+288+36)'), (576, 'BRIEF 표기 576런')):
    print(f'  {lbl:<24} {n}런 → {n/11.2:.0f} 분  (I900 초기화 1.8배 가정 시 +10~20%)')
