#!/usr/bin/env python3
"""S11 PRO 감사 — 독립 재도출.

⛔`null_corrected.py` 를 읽지도 import 하지도 않는다. 전부 처음부터 짰다.
규약(승계): 에피소드 값 = 추정 시드 3개의 **중앙값** → 같은 에피소드 안에서만 짝 Δ
        → **양측** 정확 이항 부호검정, **동점 제외**(수 병기).
레짐 = 파일명 `_(\\d+)comp_`; ≤5 = sparse, ≥15 = crowded.
선택기 = 두 팔의 3시드 중앙 `selstat` 을 비교해 **작은 팔**의 3시드 중앙 지표를 취한다.

출력 = stdout. 추가 런 0.
"""
import json, math, os, re, statistics as st
from collections import defaultdict

TMP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TMP)  # .../wavesplat/tmp

S9 = os.path.join(ROOT, 'S9_lr_x_l1_2026-08-11', 'results.jsonl')
X2 = os.path.join(ROOT, 'X2_no_learned_2026-08-11', 'results.jsonl')
S8 = os.path.join(ROOT, 'S8_deploy_confirm_2026-08-10', 'results.jsonl')

# ---------------------------------------------------------------- 기초 유틸


def load(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


_COMP = re.compile(r'_(\d+)comp_')


def regime_of(ep):
    m = _COMP.search(ep)
    assert m, ep
    nc = int(m.group(1))
    if nc <= 5:
        return 'sparse'
    if nc >= 15:
        return 'crowded'
    raise ValueError(nc)


def _lchoose(n, i):
    return math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)


def binom_pmf(i, n, p):
    if p <= 0:
        return 1.0 if i == 0 else 0.0
    if p >= 1:
        return 1.0 if i == n else 0.0
    return math.exp(_lchoose(n, i) + i * math.log(p) + (n - i) * math.log1p(-p))


def binom_sf(k, n, p=0.5):
    """P(X >= k), X~Bin(n,p) — 정확(로그공간, 오버플로 없음)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(binom_pmf(i, n, p) for i in range(k, n + 1))


def sign_test(deltas, eps=0.0):
    """양측 정확 이항. delta<0 = 개선(값이 작을수록 좋음)."""
    imp = sum(1 for d in deltas if d < -eps)
    wor = sum(1 for d in deltas if d > eps)
    tie = len(deltas) - imp - wor
    n = imp + wor
    if n == 0:
        return imp, wor, tie, 1.0
    p = min(1.0, 2 * binom_sf(max(imp, wor), n))
    return imp, wor, tie, p


def wilcoxon_z(deltas, eps=0.0):
    """부호순위 정규근사(동점보정). 0 은 제외."""
    xs = [d for d in deltas if abs(d) > eps]
    n = len(xs)
    if n < 2:
        return float('nan'), 1.0
    order = sorted(range(n), key=lambda i: abs(xs[i]))
    ranks = [0.0] * n
    i = 0
    tie_groups = []
    while i < n:
        j = i
        while j + 1 < n and abs(xs[order[j + 1]]) == abs(xs[order[i]]):
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        tie_groups.append(j - i + 1)
        i = j + 1
    W = sum(ranks[i] for i in range(n) if xs[i] > 0)  # 양(=악화) 순위합
    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    var -= sum(t ** 3 - t for t in tie_groups) / 48.0
    if var <= 0:
        return float('nan'), 1.0
    z = (W - mu) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2))
    return z, p


def med(xs):
    return st.median(xs)


def relpct(a, b):
    """(a-b)/b * 100 — a=처치, b=기준."""
    return (a - b) / b * 100.0 if b != 0 else float('nan')


def cp_lower(k, n, alpha=0.05):
    """단측 95% Clopper-Pearson 하한 (이분탐색)."""
    if k == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if binom_sf(k, n, mid) > alpha:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------- 집계


def episode_table(rows, keys=('nRMSE', 'window_resid', 'all_resid')):
    """cell -> episode -> {key: 3시드 중앙}. 시드 수도 검증."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        for k in keys:
            v = r.get(k)
            if v is not None:
                acc[r['cell']][r['episode']][k].append(v)
    out = {}
    for c, eps in acc.items():
        out[c] = {}
        for ep, d in eps.items():
            out[c][ep] = {k: med(v) for k, v in d.items()}
            out[c][ep]['_nseed'] = len(d[keys[0]])
    return out


def shellkeys(rows):
    return sorted(rows[0]['shell_rmse'].keys())


# ---------------------------------------------------------------- 선택기


def select(tab, armA, armB, ep, selstat, metric):
    a = tab[armA][ep]
    b = tab[armB][ep]
    return (a[metric] if a[selstat] <= b[selstat] else b[metric])


def selected_arm(tab, armA, armB, ep, selstat):
    return armA if tab[armA][ep][selstat] <= tab[armB][ep][selstat] else armB


# ================================================================ 본체

def main():
    s9 = load(S9)
    x2 = load(X2)
    s8 = load(S8)
    print(f'# 원자료 행수: S9={len(s9)} X2={len(x2)} S8={len(s8)}')
    print(f'# nonfinite frames 합: S9={sum(r["n_frames_nonfinite"] for r in s9)} '
          f'X2={sum(r["n_frames_nonfinite"] for r in x2)}')

    KEYS = ['nRMSE', 'window_resid', 'all_resid', 'init_nrmse', 'nRMSE_p95',
            'nRMSE_max', 'RMSE_unobs_r5', 'RMSE_unobs_r30', 'RMSE_obs_r5',
            'avg_obs_loss', 'decorr_mean', 'final_n', 'init_iters', 'wall_s',
            'window_n_obs']
    t9 = episode_table(s9, tuple(KEYS))
    t2 = episode_table(x2, tuple(KEYS))
    T = dict(t9)
    T.update(t2)

    EPS = sorted(t9['AUTO'].keys())
    assert len(EPS) == 96
    SP = [e for e in EPS if regime_of(e) == 'sparse']
    CR = [e for e in EPS if regime_of(e) == 'crowded']
    print(f'# 에피소드: sparse={len(SP)} crowded={len(CR)}')
    ns = set(T[c][e]['_nseed'] for c in T for e in T[c])
    print(f'# 셀·에피소드별 시드 수 집합 = {ns}  (3 이어야 함)')

    # cfg 확인
    def cfgset(rows, cell, key):
        return sorted(set(str(r['cfg'].get(key)) for r in rows if r['cell'] == cell))
    print('# cfg 확인:')
    for cell, rows in [('AUTO', s9), ('AUTO-L1', s9), ('AUTO-SEED-NULL', s9), ('V2K', x2)]:
        print(f'   {cell:16s} lambda_l1={cfgset(rows, cell, "lambda_l1")} '
              f'init_method={cfgset(rows, cell, "init_method")} '
              f'init_iters={sorted(set(r["init_iters"] for r in rows if r["cell"]==cell))} '
              f'ens_k={sorted(set(r["ensemble_k"] for r in rows if r["cell"]==cell))}')

    # ------------------------------------------------------------ §1.2
    print('\n' + '=' * 78)
    print('§1.2 재도출 — SEL_real{AUTO,V2K} vs SEL_null{AUTO,AUTO-SEED-NULL}')
    print('=' * 78)
    for selstat in ('window_resid', 'all_resid'):
        for name, eps in (('sparse', SP), ('crowded', CR)):
            ds, rel = [], []
            for e in eps:
                a = select(T, 'AUTO', 'V2K', e, selstat, 'nRMSE')
                b = select(T, 'AUTO', 'AUTO-SEED-NULL', e, selstat, 'nRMSE')
                ds.append(a - b)
                rel.append(relpct(a, b))
            imp, wor, tie, p = sign_test(ds)
            nz = [r for r in rel if abs(r) > 1e-12]
            print(f'  {selstat:13s} {name:8s} n={len(eps):3d}  '
                  f'{imp}/{wor}/{tie}  p={p:.3e}  '
                  f'중앙rel(전체)={med(rel):+.2f}%  중앙rel(동점제외)='
                  f'{med(nz) if nz else float("nan"):+.2f}%  짝평균rel={sum(rel)/len(rel):+.2f}%')

    # ★꼬리 — §1.2 가 안 적은 것
    print('\n§1.2-TAIL ★BRIEF 가 표에 안 적은 것 — 짝 상대 Δ 의 꼬리 (SEL_real − SEL_null)')
    for selstat in ('window_resid', 'all_resid'):
        for name, eps in (('sparse', SP), ('crowded', CR)):
            rel = []
            for e in eps:
                a = select(T, 'AUTO', 'V2K', e, selstat, 'nRMSE')
                b = select(T, 'AUTO', 'AUTO-SEED-NULL', e, selstat, 'nRMSE')
                rel.append(relpct(a, b))
            rel_s = sorted(rel)
            print(f'  {selstat:13s} {name:8s} 평균={sum(rel)/len(rel):+8.2f}%  '
                  f'최선={rel_s[0]:+8.2f}%  최악={rel_s[-1]:+9.2f}%  '
                  f'>+50% 인 에피소드={sum(1 for r in rel if r > 50)}  '
                  f'>+100%={sum(1 for r in rel if r > 100)}')

    # ★§1.2 == 같은-기준선 DiD 임을 수치로 확인
    print('\n§1.2-ID ★수학적 확인: §1.2 의 부호검정 = **같은 기준선(AUTO) DiD** 의 부호검정')
    for name, eps in (('sparse', SP), ('crowded', CR)):
        d12, ddid = [], []
        for e in eps:
            a = select(T, 'AUTO', 'V2K', e, 'window_resid', 'nRMSE')
            b = select(T, 'AUTO', 'AUTO-SEED-NULL', e, 'window_resid', 'nRMSE')
            base = T['AUTO'][e]['nRMSE']
            d12.append(a - b)
            ddid.append(relpct(a, base) - relpct(b, base))
        same = all((x < 0) == (y < 0) and (x > 0) == (y > 0) for x, y in zip(d12, ddid))
        print(f'  {name:8s} 부호 완전일치={same}  §1.2 {sign_test(d12)[:3]}  '
              f'같은기준선DiD {sign_test(ddid)[:3]}')

    # 전환 승률 (§1.3)
    print('\n§1.3 전환 승률 (선택기가 팔을 바꾼 에피소드에서 이겼는가)')
    for selstat in ('window_resid',):
        for name, eps in (('sparse', SP), ('crowded', CR)):
            for lab, (A, B) in (('처치{AUTO,V2K}', ('AUTO', 'V2K')),
                                ('널{AUTO,SEEDNULL}', ('AUTO', 'AUTO-SEED-NULL'))):
                w = l = 0
                for e in eps:
                    pick = selected_arm(T, A, B, e, selstat)
                    if pick == A:
                        continue          # 기준선 유지 = 전환 아님
                    d = T[B][e]['nRMSE'] - T[A][e]['nRMSE']
                    if d < 0:
                        w += 1
                    elif d > 0:
                        l += 1
                n = w + l
                print(f'  {selstat} {name:8s} {lab:20s} 전환 {n:2d}  승/패 {w}/{l} '
                      f'= {w/n if n else float("nan"):.3f}')
    # 처치 vs 널 승률 단측 (2x2 Fisher)
    def fisher_one_sided(a, b, c, d):
        """P(표 이상으로 극단) — 처치 승률 > 널 승률 방향."""
        n = a + b + c + d
        r1, c1 = a + b, a + c
        tot = 0.0
        for x in range(max(0, c1 - (c + d)), min(r1, c1) + 1):
            p = math.exp(_lchoose(r1, x) + _lchoose(n - r1, c1 - x) - _lchoose(n, c1))
            if x >= a:
                tot += p
        return min(1.0, tot)

    print('\n§1.3-B ★기준선을 맞춘 전환 승률 (양쪽 다 팔 A = AUTO), 희소')
    for A, B, lab in (('AUTO', 'V2K', '처치'), ('AUTO', 'AUTO-SEED-NULL', '널')):
        w = l = 0
        for e in SP:
            if selected_arm(T, A, B, e, 'window_resid') == A:
                continue
            d = T[B][e]['nRMSE'] - T[A][e]['nRMSE']
            w += d < 0
            l += d > 0
        print(f'  {lab}: {w}/{w+l} = {w/(w+l):.3f}')
    tw = tl = nw = nl = 0
    for e in SP:
        if selected_arm(T, 'AUTO', 'V2K', e, 'window_resid') != 'AUTO':
            d = T['V2K'][e]['nRMSE'] - T['AUTO'][e]['nRMSE']
            tw += d < 0; tl += d > 0
        if selected_arm(T, 'AUTO', 'AUTO-SEED-NULL', e, 'window_resid') != 'AUTO':
            d = T['AUTO-SEED-NULL'][e]['nRMSE'] - T['AUTO'][e]['nRMSE']
            nw += d < 0; nl += d > 0
    print(f'  Fisher 단측 p (기준선 맞춤, {tw}/{tw+tl} vs {nw}/{nw+nl}) = '
          f'{fisher_one_sided(tw, tl, nw, nl):.4f}')
    # BRIEF 가 쓴 비대칭 버전
    bw = bl = 0
    for e in SP:
        if selected_arm(T, 'AUTO-L1', 'V2K', e, 'window_resid') != 'AUTO-L1':
            d = T['V2K'][e]['nRMSE'] - T['AUTO-L1'][e]['nRMSE']
            bw += d < 0; bl += d > 0
    print(f'  ⚠BRIEF §1.3 이 쓴 비대칭 짝: 처치 {{AUTO-L1,V2K}} {bw}/{bw+bl}='
          f'{bw/(bw+bl):.3f} vs 널 {{AUTO,SEEDNULL}} {nw}/{nw+nl}={nw/(nw+nl):.3f} '
          f'⟹ Fisher 단측 p={fisher_one_sided(bw, bl, nw, nl):.4f}  (팔 A 가 다르다)')

    # ------------------------------------------------------------ §2
    print('\n' + '=' * 78)
    print('§2 도달 수준 (에피소드 중앙 nRMSE 의 에피소드-중앙)')
    print('=' * 78)

    def lvl(fn, eps):
        return med([fn(e) for e in eps])

    defs = [
        ('AUTO(구 기본)', lambda e: T['AUTO'][e]['nRMSE']),
        ('AUTO-L1(현 출하)', lambda e: T['AUTO-L1'][e]['nRMSE']),
        ('V2K', lambda e: T['V2K'][e]['nRMSE']),
        ('SEL{AUTO-L1,V2K}', lambda e: select(T, 'AUTO-L1', 'V2K', e, 'window_resid', 'nRMSE')),
        ('SEL-NULL{AUTO,SEEDNULL}', lambda e: select(T, 'AUTO', 'AUTO-SEED-NULL', e, 'window_resid', 'nRMSE')),
        ('SEL_real{AUTO,V2K}', lambda e: select(T, 'AUTO', 'V2K', e, 'window_resid', 'nRMSE')),
        ('전지오라클{AUTO-L1,V2K}', lambda e: min(T['AUTO-L1'][e]['nRMSE'], T['V2K'][e]['nRMSE'])),
    ]
    print(f'  {"":26s} {"sparse":>10s} {"crowded":>10s}')
    for lab, fn in defs:
        print(f'  {lab:26s} {lvl(fn, SP):10.5f} {lvl(fn, CR):10.5f}')

    # 팔 대조 V2K - AUTO-L1
    print('\n§2-F 팔 대조 V2K − AUTO-L1 (nRMSE)')
    for name, eps in (('sparse', SP), ('crowded', CR)):
        ds = [T['V2K'][e]['nRMSE'] - T['AUTO-L1'][e]['nRMSE'] for e in eps]
        rel = [relpct(T['V2K'][e]['nRMSE'], T['AUTO-L1'][e]['nRMSE']) for e in eps]
        imp, wor, tie, p = sign_test(ds)
        z, pz = wilcoxon_z(ds)
        zr, pzr = wilcoxon_z(rel)
        print(f'  {name:8s} n={len(eps):3d} {imp}/{wor}/{tie} p={p:.4g} | '
              f'중앙rel={med(rel):+.2f}% 평균rel={sum(rel)/len(rel):+.2f}% | '
              f'Wilcoxon(abs) z={z:+.3f} p={pz:.3f} ; (rel) z={zr:+.3f} p={pzr:.3f}')

    # SEL vs 항상 V2K
    print('\n§2 선택기 vs 고정 규칙 "항상 V2K" (nRMSE)')
    for name, eps in (('sparse', SP), ('crowded', CR)):
        ds, rel = [], []
        for e in eps:
            a = select(T, 'AUTO-L1', 'V2K', e, 'window_resid', 'nRMSE')
            b = T['V2K'][e]['nRMSE']
            ds.append(a - b)
            rel.append(relpct(a, b))
        imp, wor, tie, p = sign_test(ds)
        nz = [r for r in rel if abs(r) > 1e-12]
        print(f'  {name:8s} {imp}/{wor}/{tie} p={p:.4g} 중앙rel(동점제외)='
              f'{med(nz) if nz else float("nan"):+.2f}%')

    # ------------------------------------------------------------ §1.1 DiD
    print('\n' + '=' * 78)
    print('§1.1 DiD 재도출 (BRIEF 가 폐기한 설계)')
    print('=' * 78)
    for name, eps in (('sparse', SP), ('crowded', CR)):
        did = []
        for e in eps:
            rt = relpct(select(T, 'AUTO-L1', 'V2K', e, 'window_resid', 'nRMSE'),
                        T['AUTO-L1'][e]['nRMSE'])
            rn = relpct(select(T, 'AUTO', 'AUTO-SEED-NULL', e, 'window_resid', 'nRMSE'),
                        T['AUTO'][e]['nRMSE'])
            did.append(rt - rn)
        imp, wor, tie, p = sign_test(did)
        print(f'  {name:8s} DiD {imp}/{wor}/{tie} p={p:.4g} 중앙={med(did):+.2f}%')
        # 분산 비교: rel_T, rel_N 각각의 IQR
        rt = [relpct(select(T, 'AUTO-L1', 'V2K', e, 'window_resid', 'nRMSE'), T['AUTO-L1'][e]['nRMSE']) for e in eps]
        rn = [relpct(select(T, 'AUTO', 'AUTO-SEED-NULL', e, 'window_resid', 'nRMSE'), T['AUTO'][e]['nRMSE']) for e in eps]
        sd = lambda v: st.pstdev(v)
        print(f'           sd(rel_T)={sd(rt):.2f}  sd(rel_N)={sd(rn):.2f}  sd(DiD)={sd(did):.2f}'
              f'  [독립이면 sqrt(sd_T^2+sd_N^2)={math.sqrt(sd(rt)**2+sd(rn)**2):.2f}]')

    # ------------------------------------------------------------ §4-A 라우트 층화
    print('\n' + '=' * 78)
    print('§4-A 라우트 층화 (AUTO-L1 의 auto_selected)')
    print('=' * 78)
    route = defaultdict(list)
    for r in s9:
        if r['cell'] == 'AUTO-L1':
            route[r['episode']].append(r['auto_selected'])
    for rule in ('majority', 'pure'):
        strata = defaultdict(list)
        for e in SP:
            v = route[e]
            if rule == 'majority':
                lab = 'learned' if v.count('learned') >= 2 else 'v2_konly'
            else:
                lab = v[0] if len(set(v)) == 1 else 'mixed'
            strata[lab].append(e)
        print(f'  [{rule}]')
        for lab in sorted(strata):
            eps = strata[lab]
            ds = [T['V2K'][e]['nRMSE'] - T['AUTO-L1'][e]['nRMSE'] for e in eps]
            rel = [relpct(T['V2K'][e]['nRMSE'], T['AUTO-L1'][e]['nRMSE']) for e in eps]
            imp, wor, tie, p = sign_test(ds)
            z, pz = wilcoxon_z(ds)
            zr, pzr = wilcoxon_z(rel)
            print(f'    {lab:9s} n={len(eps):3d} {imp}/{wor}/{tie} p={p:.4g} '
                  f'중앙rel={med(rel):+.2f}% 평균rel={sum(rel)/len(rel):+.2f}% '
                  f'| AUTO-L1중앙={med([T["AUTO-L1"][e]["nRMSE"] for e in eps]):.5f} '
                  f'V2K중앙={med([T["V2K"][e]["nRMSE"] for e in eps]):.5f} '
                  f'| Wz(abs)={z:+.2f} p={pz:.3f} Wz(rel)={zr:+.2f} p={pzr:.3f}')

    # ★ init_iters 변이 확인 — 반복예산 축의 자료 내 변이가 있는가
    print('\n§4-A2 ★init_iters 의 자료 내 변이 (반복예산 축을 직접 잰 적이 있는가)')
    it = defaultdict(lambda: defaultdict(set))
    for r in s9 + x2:
        it[r['cell']][r['auto_selected'] or 'n/a'].add(r['init_iters'])
    for c in sorted(it):
        print(f'  {c:16s} ' + '  '.join(f'{k}:{sorted(v)}' for k, v in sorted(it[c].items())))
    # 혼잡 선택기 전환 수 (팔 A 별)
    print('\n§5-A ★혼잡에서 선택기가 실제로 개입한 횟수 (팔 A 를 바꾸면 달라진다)')
    for A, B in (('AUTO', 'V2K'), ('AUTO-L1', 'V2K'), ('AUTO', 'AUTO-SEED-NULL')):
        for name, eps in (('sparse', SP), ('crowded', CR)):
            n = sum(1 for e in eps if selected_arm(T, A, B, e, 'window_resid') != A)
            lab = '{%s,%s}' % (A, B)
            print(f'  {lab:32s} {name:8s} 전환 {n}/{len(eps)}')

    # ------------------------------------------------------------ §4-C 대안 PRIMARY 후보
    print('\n' + '=' * 78)
    print('§4-C+ 대안 PRIMARY 후보 — V2K−AUTO-L1 을 I900 의 상한 대리로 (희소)')
    print('=' * 78)
    strata = {'전체 n=64': SP,
              'learned-route(majority)': [e for e in SP if route[e].count('learned') >= 2]}
    cand = ['nRMSE', 'init_nrmse', 'window_resid', 'all_resid', 'nRMSE_p95',
            'nRMSE_max', 'RMSE_unobs_r5', 'RMSE_unobs_r30', 'decorr_mean']
    for sname, eps in strata.items():
        print(f'  --- {sname} (n={len(eps)})')
        for k in cand:
            ds = [T['V2K'][e][k] - T['AUTO-L1'][e][k] for e in eps]
            rel = [relpct(T['V2K'][e][k], T['AUTO-L1'][e][k]) for e in eps]
            imp, wor, tie, p = sign_test(ds)
            print(f'    {k:16s} {imp:3d}/{wor:3d}/{tie:2d} p={p:.3e} '
                  f'부호율={imp/(imp+wor) if imp+wor else float("nan"):.3f} '
                  f'중앙rel={med(rel):+8.2f}% 평균rel={sum(rel)/len(rel):+9.2f}%')

    # 시드 널에서 같은 통계량의 부호검정 (H0=0.5 가 참인지)
    print('\n  --- 무처치 널 대조: AUTO-SEED-NULL − AUTO (같은 cfg, 시드만 다름), 희소 n=64')
    for k in cand:
        ds = [T['AUTO-SEED-NULL'][e][k] - T['AUTO'][e][k] for e in SP]
        imp, wor, tie, p = sign_test(ds)
        rel = [relpct(T['AUTO-SEED-NULL'][e][k], T['AUTO'][e][k]) for e in SP]
        print(f'    {k:16s} {imp:3d}/{wor:3d}/{tie:2d} p={p:.3f} '
              f'중앙rel={med(rel):+8.2f}%  (H0=0.5 검사)')

    # ------------------------------------------------------------ 검정력
    print('\n' + '=' * 78)
    print('검정력 계산 (사전, 정확 이중합)')
    print('=' * 78)

    def primary_k(n, alpha=0.05):
        """양측 정확 이항 p<=alpha 를 주는 최소 개선 수."""
        for k in range(n // 2, n + 1):
            if min(1.0, 2 * binom_sf(k, n)) <= alpha:
                return k
        return None

    def power_conversion(N, q, pw, alpha=0.05):
        """n_eff~Bin(N,q), 개선수~Bin(n_eff,pw), 문턱 primary_k(n_eff)."""
        tot = 0.0
        for ne in range(0, N + 1):
            pn = math.comb(N, ne) * q ** ne * (1 - q) ** (N - ne)
            if pn == 0:
                continue
            k = primary_k(ne, alpha)
            if k is None:
                continue
            tot += pn * binom_sf(k, ne, pw)
        return tot

    def power_plain(N, pw, alpha=0.05):
        """동점 없는 단순 부호검정 검정력."""
        k = primary_k(N, alpha)
        if k is None:
            return 0.0
        # 양측이므로 반대방향도 세어야 하나 방향이 맞으면 무시가능(보수적)
        return binom_sf(k, N, pw)

    print('  primary_k 표:', {n: primary_k(n) for n in
                              [10, 16, 20, 24, 27, 32, 40, 48, 53, 64, 80, 96]})

    print('\n  [A] 선택기 PRIMARY (전환-조건부). 관측 = 아래 재도출값에서 유도')
    # 관측 q, pw 를 재도출 (SEL{AUTO-L1,V2K} vs AUTO-L1, 희소)
    conv = 0
    win = 0
    lose = 0
    for e in SP:
        pick = selected_arm(T, 'AUTO-L1', 'V2K', e, 'window_resid')
        if pick == 'V2K':
            conv += 1
            d = T['V2K'][e]['nRMSE'] - T['AUTO-L1'][e]['nRMSE']
            if d < 0:
                win += 1
            elif d > 0:
                lose += 1
    q_hat = conv / len(SP)
    p_hat = win / (win + lose)
    q_lo = cp_lower(conv, len(SP))
    p_lo = cp_lower(win, win + lose)
    print(f'    관측: 전환 {conv}/{len(SP)} q={q_hat:.4f} (CP95하한 {q_lo:.4f}) ; '
          f'전환시 승 {win}/{win+lose} p={p_hat:.4f} (CP95하한 {p_lo:.4f})')
    for N in (64, 80, 96, 128):
        print(f'    N={N:3d}  검정력@동결(q={q_lo:.3f},p={p_lo:.3f}) = '
              f'{power_conversion(N, q_lo, p_lo):.4f} | @관측 = '
              f'{power_conversion(N, q_hat, p_hat):.4f}')

    print('\n  [B] I900 을 배포후보로 쓸 때의 사전 검정력 (I900 ≡ V2K 가정, 부호검정)')
    for sname, eps in strata.items():
        ds = [T['V2K'][e]['nRMSE'] - T['AUTO-L1'][e]['nRMSE'] for e in eps]
        imp, wor, tie, p = sign_test(ds)
        r = imp / (imp + wor)
        rlo = cp_lower(imp, imp + wor)
        print(f'    {sname}: 관측 부호율 {imp}/{imp+wor} = {r:.3f} (CP95하한 {rlo:.3f})')
        for N in (32, 53, 64, 96, 128, 192, 256, 384, 512):
            print(f'      N={N:4d}  검정력@관측={power_plain(N, r):.3f}  '
                  f'@하한={power_plain(N, rlo):.3f}')

    print('\n  [C] 같은 가정에서 init_nrmse 를 PRIMARY 로 쓰면')
    for sname, eps in strata.items():
        ds = [T['V2K'][e]['init_nrmse'] - T['AUTO-L1'][e]['init_nrmse'] for e in eps]
        imp, wor, tie, p = sign_test(ds)
        r = imp / (imp + wor)
        rlo = cp_lower(imp, imp + wor)
        print(f'    {sname}: 부호율 {imp}/{imp+wor}={r:.3f} (CP하한 {rlo:.3f}) '
              f'| N=32 검정력@하한={power_plain(32, rlo):.3f} '
              f'N=48 {power_plain(48, rlo):.3f} N=64 {power_plain(64, rlo):.3f}')

    print('\n  [D] 필요 N (부호율 r 에서 양측 0.05 검정력 0.80/0.90)')
    for r in (0.55, 0.585, 0.60, 0.65, 0.70, 0.75, 0.80):
        n80 = next((N for N in range(6, 4001) if power_plain(N, r) >= 0.80), None)
        n90 = next((N for N in range(6, 4001) if power_plain(N, r) >= 0.90), None)
        print(f'    r={r:.3f}  N80={n80}  N90={n90}')

    # ---------------- [E] ★비모수 MDE — 측정된 시드-널 잡음분포에서 유도
    print('\n  [E] ★★★비모수 MDE — 무처치 널(AUTO vs AUTO-SEED-NULL)의 실측 log-비 분포에서')
    print('      가정: 처치는 log(nRMSE) 를 상수 log(rho) 만큼 이동. 부호율 r(rho)=F_null(-log rho).')

    def null_logratios(metric, eps):
        out = []
        for e in eps:
            a = T['AUTO-SEED-NULL'][e][metric]
            b = T['AUTO'][e][metric]
            if a > 0 and b > 0:
                out.append(math.log(a / b))
        return out

    LR_ROUTE = [e for e in SP if route[e].count('learned') >= 2]
    for metric, eps, lbl in (('nRMSE', SP, '희소 전체 n=64'),
                             ('nRMSE', LR_ROUTE, '희소 learned-route n=53'),
                             ('init_nrmse', SP, '희소 전체 n=64 (init_nrmse)')):
        L = null_logratios(metric, eps)
        L.sort()
        q = lambda f: L[max(0, min(len(L) - 1, int(round(f * (len(L) - 1)))))]
        print(f'    [{lbl}] 널 log-비: n={len(L)} 중앙={st.median(L):+.4f} sd={st.pstdev(L):.4f} '
              f'IQR=[{q(.25):+.4f},{q(.75):+.4f}] 범위=[{L[0]:+.3f},{L[-1]:+.3f}] '
              f'| 상대 중앙={math.exp(st.median(L))-1:+.2%}')

        def signrate(rho):
            t = -math.log(rho)
            return sum(1 for x in L if x < t) / len(L)

        for N in (32, 53, 64, 96, 128):
            row = []
            for target in (0.80, 0.90):
                # rho 를 낮춰가며(개선 클수록) 검정력 목표 도달하는 최소 개선율
                best = None
                for i in range(1, 900):
                    rho = 1 - i / 1000.0
                    if rho <= 0.02:
                        break
                    if power_plain(N, signrate(rho)) >= target:
                        best = rho
                        break
                row.append('n/a' if best is None else f'{(best-1)*100:.1f}%')
            print(f'      N={N:4d}  MDE80={row[0]:>8s}  MDE90={row[1]:>8s}   '
                  f'(참고 부호율: -10%→{signrate(0.90):.3f}, -20%→{signrate(0.80):.3f}, '
                  f'-30%→{signrate(0.70):.3f}, -50%→{signrate(0.50):.3f})')

    # ---------------- [F] 파일럿의 해상도 (부트스트랩)
    print('\n  [F] 파일럿 해상도 — 널 잡음만으로 16 / 24 / 32 에피소드에서 중앙 log-비의 분포')
    import random
    random.seed(20260812)
    L = null_logratios('nRMSE', LR_ROUTE)
    for npilot in (16, 24, 32, 53):
        meds = []
        for _ in range(20000):
            s = [random.choice(L) for _ in range(npilot)]
            meds.append(st.median(s))
        meds.sort()
        lo, hi = meds[int(.025 * len(meds))], meds[int(.975 * len(meds))]
        signs = []
        for _ in range(20000):
            s = [random.choice(L) for _ in range(npilot)]
            signs.append(sum(1 for x in s if x < 0))
        signs.sort()
        print(f'    n={npilot:3d}: 효과 0 일 때 중앙 상대값 95% = '
              f'[{math.exp(lo)-1:+.1%}, {math.exp(hi)-1:+.1%}] ; '
              f'개선 부호수 95% = [{signs[int(.025*len(signs))]}, {signs[int(.975*len(signs))]}]')

    # ---------------- [G] veto(G-B) 로 쓸 때 필요한 정밀도
    print('\n  [G] I900 을 PRIMARY 가 아니라 거부권(veto) 로 쓸 때')
    print('      veto 규칙 = "I900 이 선택기 이득의 50% 이상"; 이는 검정이 아니라 추정 문제.')
    for N in (32, 53, 64, 96):
        meds = []
        for _ in range(20000):
            s = [random.choice(L) for _ in range(N)]
            meds.append(st.median(s))
        meds.sort()
        lo, hi = meds[int(.025 * len(meds))], meds[int(.975 * len(meds))]
        print(f'    N={N:3d}: 참 효과 0 인 팔의 중앙 상대값이 우연히 '
              f'{math.exp(lo)-1:+.1%}..{math.exp(hi)-1:+.1%} 로 흔들린다 '
              f'⟹ 50% 문턱(선택기 중앙 -17.6% 의 절반 = -8.8%) 위양성 발화확률 = '
              f'{sum(1 for m in meds if math.exp(m)-1 <= -0.088)/len(meds):.3f}')

    # ------------------------------------------------------------ 혼잡 GT-free 레짐 판별
    print('\n' + '=' * 78)
    print('§5 혼잡/희소를 GT-free 로 가를 수 있는가 (AUTO-L1 팔의 배포시점 가용량)')
    print('=' * 78)
    for k in ('window_resid', 'all_resid', 'window_n_obs', 'final_n', 'avg_obs_loss'):
        sp = sorted(T['AUTO-L1'][e][k] for e in SP)
        cr = sorted(T['AUTO-L1'][e][k] for e in CR)
        # 최적 단일 문턱 정확도
        cand_t = sorted(set(sp + cr))
        best = (0, None, None)
        for t in cand_t:
            for direc in (+1, -1):
                acc = (sum(1 for v in sp if (v <= t) == (direc == 1)) +
                       sum(1 for v in cr if (v <= t) != (direc == 1)))
                if acc > best[0]:
                    best = (acc, t, direc)
        print(f'  {k:14s} sparse[min,med,max]=[{sp[0]:.5g},{med(sp):.5g},{sp[-1]:.5g}] '
              f'crowded=[{cr[0]:.5g},{med(cr):.5g},{cr[-1]:.5g}] '
              f'| 최적문턱 정확도 {best[0]}/96 t={best[1]:.5g}')
    # auto_n_selected 중앙
    ans = defaultdict(list)
    for r in s9:
        if r['cell'] == 'AUTO-L1':
            ans[_COMP.search(r['episode']).group(1)].append(r['auto_n_selected'])
    print('  auto_n_selected 중앙(성분수별):',
          {k: med(v) for k, v in sorted(ans.items(), key=lambda x: int(x[0]))})

    # ------------------------------------------------------------ 비용
    print('\n' + '=' * 78)
    print('비용 — wall_s (초/런)')
    print('=' * 78)
    for cell, rows in (('AUTO-L1', s9), ('AUTO', s9), ('V2K', x2)):
        for name, eps in (('sparse', set(SP)), ('crowded', set(CR))):
            w = [r['wall_s'] for r in rows if r['cell'] == cell and r['episode'] in eps]
            print(f'  {cell:8s} {name:8s} n={len(w):4d} 중앙={med(w):6.1f}s 평균={sum(w)/len(w):6.1f}s')

    # ------------------------------------------------------------ 셸 점검 (SEL_real vs SEL_null)
    print('\n' + '=' * 78)
    print('의무 점검 — 널 보정된 선택기(SEL_real{AUTO,V2K} − SEL_null) 의 9셸/미관측, 희소')
    print('=' * 78)
    SH = shellkeys(s9)
    shacc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in s9 + x2:
        for k, v in r['shell_rmse'].items():
            shacc[r['cell']][r['episode']][k].append(v)
    SHT = {c: {e: {k: med(v) for k, v in d.items()} for e, d in eps.items()}
           for c, eps in shacc.items()}
    for k in SH:
        ds = []
        for e in SP:
            a = (SHT['AUTO'][e][k] if T['AUTO'][e]['window_resid'] <= T['V2K'][e]['window_resid']
                 else SHT['V2K'][e][k])
            b = (SHT['AUTO'][e][k] if T['AUTO'][e]['window_resid'] <= T['AUTO-SEED-NULL'][e]['window_resid']
                 else SHT['AUTO-SEED-NULL'][e][k])
            ds.append(a - b)
        imp, wor, tie, p = sign_test(ds)
        print(f'  shell {k:8s} {imp:3d}/{wor:3d}/{tie:2d} p={p:.3e}')
    for k in ('all_resid', 'RMSE_unobs_r5', 'RMSE_unobs_r30', 'nRMSE_p95', 'nRMSE_max'):
        ds = []
        for e in SP:
            a = select(T, 'AUTO', 'V2K', e, 'window_resid', k)
            b = select(T, 'AUTO', 'AUTO-SEED-NULL', e, 'window_resid', k)
            ds.append(a - b)
        imp, wor, tie, p = sign_test(ds)
        print(f'  {k:16s} {imp:3d}/{wor:3d}/{tie:2d} p={p:.3e}')

    # ------------------------------------------------------------ ★사전등록안 검정력
    print('\n' + '=' * 78)
    print('★★★ PRO 사전등록안의 사전 검정력 (PRIMARY = SELECT{A,B} − RESTART{A,A\'} 희소)')
    print('=' * 78)

    def primary_k2(n, alpha=0.05):
        for k in range(n // 2, n + 1):
            if min(1.0, 2 * binom_sf(k, n)) <= alpha:
                return k
        return None

    def power_conv(N, q, pw, alpha=0.05):
        tot = 0.0
        for ne in range(0, N + 1):
            pn = binom_pmf(ne, N, q)
            if pn <= 0:
                continue
            k = primary_k2(ne, alpha)
            if k is None:
                continue
            tot += pn * binom_sf(k, ne, pw)
        return tot

    nont = 41 + 14
    print(f'  S9 실측(팔 A=AUTO): 비동점 {nont}/64 = {nont/64:.3f} ; '
          f'비동점 중 개선 41/{nont} = {41/nont:.3f}')
    q_lo2, p_lo2 = cp_lower(nont, 64), cp_lower(41, nont)
    print(f'  CP95 하한 동결: q={q_lo2:.4f}  p={p_lo2:.4f}')
    scen = [('S9 관측 그대로', nont / 64, 41 / nont),
            ('CP95 하한(동결)', q_lo2, p_lo2),
            ('보수: 팔 A 를 AUTO-L1 로 바꿔 전환이 줄 때', 0.65, 0.70),
            ('최보수', 0.55, 0.65)]
    for lab, q, pw in scen:
        row = '  '.join(f'N={N}:{power_conv(N, q, pw):.3f}' for N in (64, 96, 128))
        print(f'    {lab:42s} q={q:.3f} p={pw:.3f} → {row}')

    print('\n  ★PRO 가 동결을 제안하는 값 (유도 과정을 §안에 적는다)')
    print('    q = 1-(1-0.422)(1-0.55) = 0.740   [0.422 = SELECT{AUTO-L1,V2K} 전환 27/64 MEASURED ;'
          ' 0.55 = RESTART 전환 38/64·20/32 을 보수적으로 내린 값]')
    print('    p = 0.700                        [S9 실측 41/55=0.745 보다 낮게 동결]')
    for N in (64, 80, 96, 112, 128):
        print(f'      N={N:3d} 검정력={power_conv(N, 0.740, 0.700):.4f}')
    for tgt in (0.80, 0.90):
        n = next((N for N in range(16, 400) if power_conv(N, 0.740, 0.700) >= tgt), None)
        print(f'      검정력 {tgt} 필요 N = {n}')

    print('\n  거부권 G-B(I900) 의 사전 민감도 — 비모수 MDE 기준')
    for lbl, eps in (('희소 전체', SP), ('희소 learned-route', LR_ROUTE)):
        L = [math.log(T['AUTO-SEED-NULL'][e]['nRMSE'] / T['AUTO'][e]['nRMSE'])
             for e in eps if T['AUTO'][e]['nRMSE'] > 0]
        sr = lambda rho: sum(1 for x in L if x < -math.log(rho)) / len(L)
        for tgt, nm in ((0.912, '선택기 중앙 -17.6% 의 50% = -8.8%'),
                        (0.824, '선택기 중앙의 100% = -17.6%'),
                        (0.95, '-5%')):
            row = '  '.join(f'N={N}:{power_plain(N, sr(tgt)):.3f}' for N in (64, 96, 128))
            print(f'    [{lbl:18s}] {nm:32s} 부호율={sr(tgt):.3f} → {row}')

    # ------------------------------------------------------------ 꼬리 교정
    print('\n' + '=' * 78)
    print('꼬리 교정 — +4289% 는 널을 넘는가? (무처치 널 자신의 꼬리)')
    print('=' * 78)
    for name, eps in (('sparse', SP), ('crowded', CR)):
        rel = [relpct(T['AUTO-SEED-NULL'][e]['nRMSE'], T['AUTO'][e]['nRMSE']) for e in eps]
        rel_s = sorted(rel)
        print(f'  무처치 널 {name:8s} 최선={rel_s[0]:+.2f}% 최악={rel_s[-1]:+.2f}% '
              f'>+50%={sum(1 for r in rel if r > 50)} >+100%={sum(1 for r in rel if r > 100)} '
              f'평균={sum(rel)/len(rel):+.2f}%')
    # 최악 에피소드 지목
    worst = max(SP, key=lambda e: relpct(select(T, 'AUTO', 'V2K', e, 'window_resid', 'nRMSE'),
                                         select(T, 'AUTO', 'AUTO-SEED-NULL', e, 'window_resid', 'nRMSE')))
    print(f'  ★§1.2 최악 에피소드 = {worst}')
    for c in ('AUTO', 'AUTO-L1', 'V2K', 'AUTO-SEED-NULL'):
        print(f'     {c:16s} nRMSE={T[c][worst]["nRMSE"]:.5f} '
              f'window_resid={T[c][worst]["window_resid"]:.5f}')

    # BRIEF 의 p=0.0031 검증
    print('\n  BRIEF §1.3 의 "H0=널 단측 p=0.0031" 검증:')
    print(f'    binom P(X>=25 | n=27, p0=0.684) = {binom_sf(25, 27, 0.684):.4f}   (BRIEF: 0.0031)')
    print(f'    기준선 맞춤: P(X>=39 | n=46, p0=0.684) = {binom_sf(39, 46, 0.684):.4f}')
    print('    ⚠p0 를 상수로 취급 = 널의 표본오차(n=38) 무시. Fisher 가 옳다 → 위 §1.3-B')

    # ------------------------------------------------------------ 배포 집계(단일 시드)
    print('\n' + '=' * 78)
    print('배포 집계 재도출 — 단일 시드에서도 §1.2 가 살아남는가 (S10 CON B-3 승계)')
    print('=' * 78)
    per = defaultdict(dict)
    for r in s9 + x2:
        per[(r['cell'], r['seed'])][r['episode']] = r
    pairs = [(1575, 1578), (1576, 1579), (1577, 1580)]
    for ts, ns_ in pairs:
        for name, eps in (('sparse', SP), ('crowded', CR)):
            ds = []
            for e in eps:
                A = per[('AUTO', ts)][e]
                B = per[('V2K', ts)][e]
                Z = per[('AUTO-SEED-NULL', ns_)][e]
                a = A['nRMSE'] if A['window_resid'] <= B['window_resid'] else B['nRMSE']
                b = A['nRMSE'] if A['window_resid'] <= Z['window_resid'] else Z['nRMSE']
                ds.append(a - b)
            imp, wor, tie, p = sign_test(ds)
            print(f'  시드 {ts}(널 {ns_}) {name:8s} {imp}/{wor}/{tie} p={p:.4g}')

    # ------------------------------------------------------------ ★재시작-선택기를 후보로
    print('\n' + '=' * 78)
    print('★★★ 널이 아니라 **경쟁 후보** 로 본 SEL_null (= 재시작 선택기 RESTART)')
    print('=' * 78)
    for name, eps in (('sparse', SP), ('crowded', CR)):
        ds, rel = [], []
        for e in eps:
            b = select(T, 'AUTO', 'AUTO-SEED-NULL', e, 'window_resid', 'nRMSE')
            ds.append(b - T['AUTO'][e]['nRMSE'])
            rel.append(relpct(b, T['AUTO'][e]['nRMSE']))
        imp, wor, tie, p = sign_test(ds)
        nz = [r for r in rel if abs(r) > 1e-12]
        print(f'  RESTART{{AUTO,seed\'}} − AUTO  {name:8s} {imp}/{wor}/{tie} p={p:.4g} '
              f'중앙rel(동점제외)={med(nz) if nz else float("nan"):+.2f}% 평균rel={sum(rel)/len(rel):+.2f}%')
    for name, eps in (('sparse', SP), ('crowded', CR)):
        ds = []
        for e in eps:
            a = select(T, 'AUTO-L1', 'V2K', e, 'window_resid', 'nRMSE')
            ds.append(a - T['AUTO-L1'][e]['nRMSE'])
        imp, wor, tie, p = sign_test(ds)
        print(f'  SELECT{{AUTO-L1,V2K}} − AUTO-L1 {name:8s} {imp}/{wor}/{tie} p={p:.4g}')

    print('\n  절사평균 — §1.2 희소 짝 상대 Δ 의 +37.59% 는 몇 개 에피소드인가')
    rel = []
    for e in SP:
        a = select(T, 'AUTO', 'V2K', e, 'window_resid', 'nRMSE')
        b = select(T, 'AUTO', 'AUTO-SEED-NULL', e, 'window_resid', 'nRMSE')
        rel.append(relpct(a, b))
    rs = sorted(rel)
    print(f'    전체평균={sum(rs)/len(rs):+.2f}%  최악1개 제외={sum(rs[:-1])/(len(rs)-1):+.2f}%  '
          f'최악2개 제외={sum(rs[:-2])/(len(rs)-2):+.2f}%  5% 양쪽절사={sum(rs[3:-3])/(len(rs)-6):+.2f}%')

    print('\n  RESTART{AUTO,seed\'} − AUTO 의 꼬리·셸·미관측 (배포 후보로 볼 때 의무 점검)')
    for name, eps in (('sparse', SP), ('crowded', CR)):
        rel = [relpct(select(T, 'AUTO', 'AUTO-SEED-NULL', e, 'window_resid', 'nRMSE'),
                      T['AUTO'][e]['nRMSE']) for e in eps]
        rs = sorted(rel)
        print(f'    {name:8s} 최선={rs[0]:+.2f}% 최악={rs[-1]:+.2f}% '
              f'>+50%={sum(1 for r in rel if r > 50)} 평균={sum(rel)/len(rel):+.2f}%')
    for name, eps in (('sparse', SP), ('crowded', CR)):
        out = []
        for k in SH:
            ds = [(SHT['AUTO'][e][k] if T['AUTO'][e]['window_resid'] <=
                   T['AUTO-SEED-NULL'][e]['window_resid'] else SHT['AUTO-SEED-NULL'][e][k])
                  - SHT['AUTO'][e][k] for e in eps]
            i_, w_, t_, p_ = sign_test(ds)
            out.append(f'{k}:{i_}/{w_}')
        print(f'    {name:8s} 9셸(개선/악화) ' + '  '.join(out))
        for k in ('all_resid', 'RMSE_unobs_r5', 'RMSE_unobs_r30'):
            ds = [select(T, 'AUTO', 'AUTO-SEED-NULL', e, 'window_resid', k) - T['AUTO'][e][k]
                  for e in eps]
            i_, w_, t_, p_ = sign_test(ds)
            print(f'      {k:16s} {i_}/{w_}/{t_} p={p_:.3e}')

    print('\n  비용 (측정 wall_s 중앙, 희소): AUTO-L1 27.5s · V2K 45.1s')
    print('    SELECT{AUTO-L1,V2K} = 27.5+45.1 = 72.6 s/에피소드 (기준선 대비 2.64배)')
    print('    RESTART{AUTO-L1×2}  = 27.5+27.5 = 55.0 s/에피소드 (기준선 대비 2.00배)')

    # ------------------------------------------------------------ S8 널 재확인
    print('\n' + '=' * 78)
    print('S8 교차확인 — 시드 널의 선택 이득 (희소 n=32)')
    print('=' * 78)
    t8 = episode_table(s8, tuple(KEYS))
    E8 = sorted(t8['AUTO'].keys())
    SP8 = [e for e in E8 if regime_of(e) == 'sparse']
    CR8 = [e for e in E8 if regime_of(e) == 'crowded']
    for name, eps in (('sparse', SP8), ('crowded', CR8)):
        ds = []
        for e in eps:
            b = (t8['AUTO'][e]['nRMSE']
                 if t8['AUTO'][e]['window_resid'] <= t8['AUTO-SEED-NULL'][e]['window_resid']
                 else t8['AUTO-SEED-NULL'][e]['nRMSE'])
            ds.append(b - t8['AUTO'][e]['nRMSE'])
        imp, wor, tie, p = sign_test(ds)
        rel = [relpct(b, t8['AUTO'][e]['nRMSE']) for e, b in
               zip(eps, [t8['AUTO'][e]['nRMSE'] if t8['AUTO'][e]['window_resid'] <=
                         t8['AUTO-SEED-NULL'][e]['window_resid'] else t8['AUTO-SEED-NULL'][e]['nRMSE']
                         for e in eps])]
        print(f'  S8 {name:8s} SEL-NULL−AUTO: {imp}/{wor}/{tie} p={p:.4g} 중앙rel={med(rel):+.2f}%')


if __name__ == '__main__':
    main()
