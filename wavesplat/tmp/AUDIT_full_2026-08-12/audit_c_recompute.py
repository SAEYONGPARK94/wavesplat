"""AUDIT-C — 살아있는 통계 주장의 독립 재도출 (2026-08-12).

⚠이 파일은 각 배치의 `analyze_s*.py` 를 **import 하지 않는다**. 규약(에피소드당 중앙,
부호검정, 양측 정확 이항 p, Holm)을 여기서 처음부터 다시 구현하고 `results.jsonl`
원자료만 읽는다. RESULT.md 의 숫자는 **검증 대상**이지 입력이 아니다.

사용: python3 audit_c_recompute.py            (전체)
      python3 audit_c_recompute.py --only s9  (일부)
"""
import argparse
import json
import os
import statistics as st
from collections import defaultdict
from math import comb, erf, sqrt

TMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BATCH = {
    's7': os.path.join(TMP, 'S7_confirm_n32_2026-08-10', 'results.jsonl'),
    's8': os.path.join(TMP, 'S8_deploy_confirm_2026-08-10', 'results.jsonl'),
    's9': os.path.join(TMP, 'S9_lr_x_l1_2026-08-11', 'results.jsonl'),
}

SHELLS = ('0_5', '5_10', '10_20', '20_30', '30_50', '50_75', '75_100', '100_150', '150_inf')
NEAR = ('0_5', '5_10', '10_20')


# ── 통계 원시 구현 (독립) ────────────────────────────────────────────────────
def p_two_sided(k, n):
    """부호 k/n 의 양측 정확 이항 p (q=0.5). 대칭 꼬리 합."""
    if n == 0:
        return float('nan')
    return sum(comb(n, i) for i in range(n + 1)
               if abs(i - n / 2.0) >= abs(k - n / 2.0)) / 2.0 ** n


def binom_ge(n, k, p):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def min_k(n, alpha):
    for k in range(n // 2, n + 1):
        if p_two_sided(k, n) <= alpha:
            return k
    return None


def holm(pvals):
    """Holm-Bonferroni 조정 p (같은 족). 입력 dict(name->p) -> dict(name->adj)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (nm, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[nm] = running
    return out


def boot_ci(vals, B=20000, seed=12345, alpha=0.05):
    """평균의 백분위 부트스트랩 CI.

    ⚠초판은 하위비트 주기가 짧은 LCG 를 n=64(2의 거듭제곱)에 썼다가 CI 가 한 점으로
    퇴화했다. `random.Random`(Mersenne Twister)으로 교체.
    """
    import random
    n = len(vals)
    if n == 0:
        return (float('nan'), float('nan'))
    rng = random.Random(seed)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return (means[int(alpha / 2 * B)], means[int((1 - alpha / 2) * B)])


# ── 적재 ─────────────────────────────────────────────────────────────────────
def load(path):
    rec = defaultdict(lambda: defaultdict(list))
    regimes, errors, n = {}, 0, 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if 'error' in r:
            errors += 1
            continue
        rec[r['cell']][r['episode']].append(r)
        regimes[r['episode']] = r['regime']
        n += 1
    return rec, regimes, errors, n


def G(key):
    return lambda r: r.get(key)


def GS(shell):
    return lambda r: (r.get('shell_rmse') or {}).get(shell)


AGG = {
    'median': lambda v: st.median(v),
    'mean': lambda v: sum(v) / len(v),
}


def ep_value(recs, getter, agg='median'):
    v = [getter(r) for r in recs]
    v = [x for x in v if x is not None and x == x]
    return AGG[agg](v) if v else None


def paired(rec, regimes, treat, base, regime, getter, agg='median'):
    out = []
    if treat not in rec or base not in rec:
        return out
    for ep in rec[treat]:
        if regimes.get(ep) != regime or ep not in rec[base]:
            continue
        a = ep_value(rec[treat][ep], getter, agg)
        b = ep_value(rec[base][ep], getter, agg)
        if a is None or b is None:
            continue
        out.append((ep, a - b, (a - b) / b if b else float('nan')))
    return sorted(out)


def sign_test(deltas, shift=0.0):
    imp = sum(1 for _e, d, _r in deltas if d < shift)
    wor = sum(1 for _e, d, _r in deltas if d > shift)
    tie = len(deltas) - imp - wor
    n = imp + wor
    return dict(imp=imp, wor=wor, tie=tie, n_eff=n, n=len(deltas),
                p=p_two_sided(max(imp, wor), n) if n else float('nan'),
                med_abs=st.median([d for _e, d, _r in deltas]) if deltas else float('nan'),
                med_rel=st.median([r for _e, _d, r in deltas]) if deltas else float('nan'),
                mean_rel=sum(r for _e, _d, r in deltas) / len(deltas) if deltas else float('nan'))


def line(tag, s, k=None):
    v = ''
    if k is not None:
        v = '  PASS' if (s['imp'] >= k and s['med_abs'] < 0) else '  FAIL'
    print(f"  {tag:44s} {s['imp']:3d}/{s['n']:<3d} tie{s['tie']:2d}  "
          f"dmed {s['med_abs']:+.6f} ({s['med_rel']*100:+6.2f}%)  p {s['p']:.6g}{v}")


def per_seed_signs(rec, regimes, treat, base, regime, getter):
    """규약 민감도: 추정기 시드별로 따로 부호검정."""
    seeds = sorted({r['seed'] for eps in rec[treat].values() for r in eps})
    out = {}
    for sd in seeds:
        d = []
        for ep in rec[treat]:
            if regimes.get(ep) != regime or ep not in rec[base]:
                continue
            a = [getter(r) for r in rec[treat][ep] if r['seed'] == sd]
            b = [getter(r) for r in rec[base][ep] if r['seed'] == sd]
            if not a or not b or a[0] is None or b[0] is None:
                continue
            d.append((ep, a[0] - b[0], (a[0] - b[0]) / b[0]))
        out[sd] = sign_test(d)
    return out


# ═══════════════════════════════════════════════════════════════════════════
def audit_s9():
    print('\n' + '=' * 78 + '\nS9 (2016런) — tmp/S9_lr_x_l1_2026-08-11\n' + '=' * 78)
    rec, regimes, err, n = load(BATCH['s9'])
    print(f'  records {n}  errors {err}  cells {sorted(rec)}')
    print(f'  episodes: ' + str({rg: sum(1 for v in regimes.values() if v == rg)
                                 for rg in ('sparse', 'crowded')}))
    K = {'sparse': min_k(64, 0.05), 'crowded': min_k(32, 0.05)}
    print(f'  derived primary_k: {K}   (RESULT claims 41/64 and 23/32)')

    print('\n [1] PRIMARY  AUTO-L1 - AUTO  nRMSE  (median-of-3 seeds)')
    prim = {}
    for rg in ('sparse', 'crowded'):
        s = sign_test(paired(rec, regimes, 'AUTO-L1', 'AUTO', rg, G('nRMSE')))
        prim[rg] = s
        line(f'{rg}', s, K[rg])

    print('\n [1b] 규약 민감도 (같은 자료, 집계만 교체)')
    for rg in ('sparse', 'crowded'):
        sm = sign_test(paired(rec, regimes, 'AUTO-L1', 'AUTO', rg, G('nRMSE'), agg='mean'))
        line(f'{rg} mean-of-3', sm, K[rg])
        ps = per_seed_signs(rec, regimes, 'AUTO-L1', 'AUTO', rg, G('nRMSE'))
        for sd, s in ps.items():
            line(f'{rg} seed={sd} only', s)
        # 런 단위 풀링 (짝을 시드별로 다 세기)
        pooled = []
        for ep in rec['AUTO-L1']:
            if regimes.get(ep) != rg or ep not in rec['AUTO']:
                continue
            bysd = {r['seed']: r for r in rec['AUTO'][ep]}
            for r in rec['AUTO-L1'][ep]:
                if r['seed'] in bysd:
                    b = bysd[r['seed']]['nRMSE']
                    pooled.append((ep, r['nRMSE'] - b, (r['nRMSE'] - b) / b))
        line(f'{rg} pooled-runs (n={len(pooled)})', sign_test(pooled))

    print('\n [1c] 평균-부트스트랩 규약 (per-episode median deltas)')
    for rg in ('sparse', 'crowded'):
        d = paired(rec, regimes, 'AUTO-L1', 'AUTO', rg, G('nRMSE'))
        vals = [x[1] for x in d]
        lo, hi = boot_ci(vals)
        rel = [x[2] for x in d]
        rlo, rhi = boot_ci(rel)
        print(f'   {rg}: mean d {sum(vals)/len(vals):+.6f}  CI95 [{lo:+.6f},{hi:+.6f}]'
              f'  | mean rel {sum(rel)/len(rel)*100:+.2f}%  CI95 [{rlo*100:+.2f},{rhi*100:+.2f}]')

    print('\n [2] 사전 검정력 재계산 (참 부호율 0.6875 가정, RESULT 주장 0.8279)')
    print(f'   binom_ge(64,41,0.6875) = {binom_ge(64, 41, 0.6875):.4f}')
    print(f'   binom_ge(32,23,0.6875) = {binom_ge(32, 23, 0.6875):.4f}  (RESULT 0.4338)')

    print('\n [3] 근거리/먼바다 분해 — 살아있는 배포 헤드라인의 SUPPORT-TRADEOFF 점검')
    for rg in ('sparse', 'crowded'):
        print(f'   --- {rg} ---')
        for sh in SHELLS:
            s = sign_test(paired(rec, regimes, 'AUTO-L1', 'AUTO', rg, GS(sh)))
            line(f'shell {sh}', s)
        s = sign_test(paired(rec, regimes, 'AUTO-L1', 'AUTO', rg, G('all_resid')))
        line('all_resid (GT-free)', s)
        for key in ('RMSE_obs_r5', 'RMSE_unobs_r5', 'RMSE_obs_r30', 'RMSE_unobs_r30'):
            line(key, sign_test(paired(rec, regimes, 'AUTO-L1', 'AUTO', rg, G(key))))

    print('\n [4] 결백 대조 (무처치 널 / DUP)')
    for rg in ('sparse', 'crowded'):
        line(f'{rg} AUTO-SEED-NULL - AUTO', sign_test(
            paired(rec, regimes, 'AUTO-SEED-NULL', 'AUTO', rg, G('nRMSE'))))
        d = paired(rec, regimes, 'AUTO-DUP', 'AUTO', rg, G('nRMSE'))
        print(f'   {rg} AUTO-DUP nonzero pairs: {sum(1 for x in d if x[1] != 0)}/{len(d)}')

    print('\n [5] 선택 후 검정 점검 — AUTO-L1 은 최선의 팔인가 (다중 팔 순위)')
    for rg in ('sparse', 'crowded'):
        rows = []
        for c in sorted(rec):
            if c == 'AUTO':
                continue
            s = sign_test(paired(rec, regimes, c, 'AUTO', rg, G('nRMSE')))
            rows.append((s['med_abs'], c, s))
        rows.sort()
        print(f'   --- {rg} (argmin 순) ---')
        for md, c, s in rows:
            line(c, s)
    return prim


def audit_s8():
    print('\n' + '=' * 78 + '\nS8 (1920런) — tmp/S8_deploy_confirm_2026-08-10\n' + '=' * 78)
    rec, regimes, err, n = load(BATCH['s8'])
    print(f'  records {n}  errors {err}  cells {sorted(rec)}')
    K = min_k(32, 0.05)
    Kh = min_k(32, 0.025)   # Holm-2 최악순위
    print(f'  derived primary_k(32,a=.05)={K}  holm2 worst k={Kh}')

    print('\n [1] PRIMARY  AUTO-L1-1e4 - AUTO  nRMSE')
    ps = {}
    for rg in ('crowded', 'sparse'):
        s = sign_test(paired(rec, regimes, 'AUTO-L1-1e4', 'AUTO', rg, G('nRMSE')))
        ps[rg] = s['p']
        line(rg, s, K)
    print(f'   Holm-2 adjusted: {holm(ps)}')

    print('\n [1b] 규약 민감도')
    for rg in ('crowded', 'sparse'):
        line(f'{rg} mean-of-3', sign_test(
            paired(rec, regimes, 'AUTO-L1-1e4', 'AUTO', rg, G('nRMSE'), agg='mean')), K)
        for sd, s in per_seed_signs(rec, regimes, 'AUTO-L1-1e4', 'AUTO', rg, G('nRMSE')).items():
            line(f'{rg} seed={sd}', s)

    print('\n [2] 사전 검정력 (RESULT: 희소 0.589)')
    for tp in (0.6875, 0.70, 0.72):
        print(f'   binom_ge(32,23,{tp}) = {binom_ge(32, 23, tp):.4f}')

    print('\n [3] S8 셸 분해 (RESULT §2: 악화 셸 0개)')
    for rg in ('crowded', 'sparse'):
        print(f'   --- {rg} ---')
        for sh in SHELLS:
            line(f'shell {sh}', sign_test(
                paired(rec, regimes, 'AUTO-L1-1e4', 'AUTO', rg, GS(sh))))
        line('all_resid', sign_test(
            paired(rec, regimes, 'AUTO-L1-1e4', 'AUTO', rg, G('all_resid'))))

    print('\n [4] S8 §3 DOSE-MONOTONE 재도출 (0_5 셸)')
    for cellname in ('AUTO-NT', 'AUTO-TRK-QTR'):
        for rg in ('crowded',):
            line(f'{cellname}-AUTO 0_5 {rg}', sign_test(
                paired(rec, regimes, cellname, 'AUTO', rg, GS('0_5'))))

    print('\n [5] S8 λ 곡선 — 3e-4 vs 1e-4 (선택 후 검정 여부)')
    for rg in ('crowded', 'sparse'):
        for c in ('AUTO-L1-3e4', 'AUTO-L1-1e4', 'AUTO-L1-0'):
            line(f'{rg} {c}-AUTO', sign_test(paired(rec, regimes, c, 'AUTO', rg, G('nRMSE'))))
    return ps


def audit_s7():
    print('\n' + '=' * 78 + '\nS7 (1152런) — tmp/S7_confirm_n32_2026-08-10\n' + '=' * 78)
    rec, regimes, err, n = load(BATCH['s7'])
    print(f'  records {n}  errors {err}  cells {sorted(rec)}')
    K = min_k(32, 0.05)

    print('\n [1] PRIMARY-1  AUTO-NT-L1-1e4 - AUTO-NT (혼잡)  / PRIMARY-2  vs AUTO')
    pv = {}
    s1 = sign_test(paired(rec, regimes, 'AUTO-NT-L1-1e4', 'AUTO-NT', 'crowded', G('nRMSE')))
    line('P-1 crowded', s1, K)
    pv['P1'] = s1['p']
    s2 = sign_test(paired(rec, regimes, 'AUTO-NT-L1-1e4', 'AUTO', 'crowded', G('nRMSE')))
    line('P-2 crowded', s2, K)
    pv['P2'] = s2['p']
    print(f'   Holm-2: {holm(pv)}')

    print('\n [2] DEPLOY-SINGLE  AUTO-L1-1e4 - AUTO')
    dp = {}
    for rg in ('crowded', 'sparse'):
        s = sign_test(paired(rec, regimes, 'AUTO-L1-1e4', 'AUTO', rg, G('nRMSE')))
        dp[rg] = s['p']
        line(rg, s, K)
    print(f'   Holm-2 (own family): {holm(dp)}')

    print('\n [3] SUPPORT-TRADEOFF 표 (P-2 혼잡 셸별)')
    for sh in SHELLS:
        line(f'shell {sh}', sign_test(
            paired(rec, regimes, 'AUTO-NT-L1-1e4', 'AUTO', 'crowded', GS(sh))))
    line('all_resid', sign_test(
        paired(rec, regimes, 'AUTO-NT-L1-1e4', 'AUTO', 'crowded', G('all_resid'))))

    print('\n [3b] ★DEPLOY-SINGLE 팔의 셸 분해 (살아있는 헤드라인)')
    for rg in ('crowded', 'sparse'):
        print(f'   --- {rg} ---')
        for sh in SHELLS:
            line(f'shell {sh}', sign_test(
                paired(rec, regimes, 'AUTO-L1-1e4', 'AUTO', rg, GS(sh))))
        line('all_resid', sign_test(
            paired(rec, regimes, 'AUTO-L1-1e4', 'AUTO', rg, G('all_resid'))))

    print('\n [4] 사전 검정력 주장 0.937 재계산')
    for tp in (0.8125, 0.80):
        print(f'   binom_ge(32,23,{tp}) = {binom_ge(32, 23, tp):.4f}')

    print('\n [5] 규약 민감도 (P-1)')
    line('P-1 mean-of-3', sign_test(
        paired(rec, regimes, 'AUTO-NT-L1-1e4', 'AUTO-NT', 'crowded', G('nRMSE'), agg='mean')), K)
    for sd, s in per_seed_signs(rec, regimes, 'AUTO-NT-L1-1e4', 'AUTO-NT',
                                'crowded', G('nRMSE')).items():
        line(f'P-1 seed={sd}', s)
    return dp


def cross_batch():
    """크기 재현 — S7 / S8 / S9 의 같은 대조를 원자료에서 나란히."""
    print('\n' + '=' * 78 + '\nCROSS-BATCH: AUTO-L1(1e-4) - AUTO  크기 재현\n' + '=' * 78)
    tab = {}
    for b, treat in (('s7', 'AUTO-L1-1e4'), ('s8', 'AUTO-L1-1e4'), ('s9', 'AUTO-L1')):
        rec, regimes, _e, _n = load(BATCH[b])
        for rg in ('sparse', 'crowded'):
            s = sign_test(paired(rec, regimes, treat, 'AUTO', rg, G('nRMSE')))
            base = st.median([ep_value(rec['AUTO'][ep], G('nRMSE'))
                              for ep in rec['AUTO'] if regimes.get(ep) == rg])
            tab[(b, rg)] = (s, base)
            print(f'  {b} {rg:8s}: {s["imp"]:2d}/{s["n"]:<2d}  dmed {s["med_abs"]:+.6f} '
                  f'({s["med_rel"]*100:+6.2f}%)  p {s["p"]:.6g}  AUTO baseline {base:.5f}')
    for rg in ('sparse', 'crowded'):
        a = tab[('s8', rg)][0]['med_abs']
        c = tab[('s9', rg)][0]['med_abs']
        d = tab[('s7', rg)][0]['med_abs']
        print(f'  {rg}: S9/S8 = {c/a*100:.1f}%   S9/S7 = {c/d*100:.1f}%   S8/S7 = {a/d*100:.1f}%')


def s9_shifted_and_lr():
    """S9 RESULT §2(이동 널)·§4(lr 축)·§5(기제) 재도출."""
    print('\n' + '=' * 78 + '\nS9 부속 — §2 이동 널 · §4 lr 축 · §5 기제\n' + '=' * 78)
    rec, regimes, _e, _n = load(BATCH['s9'])
    mde = json.load(open(os.path.join(TMP, 'S9_lr_x_l1_2026-08-11', 'MDE_prespec_s9.json')))
    K = {'sparse': 41, 'crowded': 23}
    print('\n [§2] 근거리 이동 널 (shift = delta_abs, PREREG 존재증명)')
    for rg in ('sparse', 'crowded'):
        for sh in NEAR:
            info = mde['by_regime'][rg]['shells'][sh]
            shift = info['delta_abs']
            s = sign_test(paired(rec, regimes, 'AUTO-L1', 'AUTO', rg, GS(sh)), shift=shift)
            print(f'   {rg:8s} {sh:6s} shift {shift:+.6f} (rel {info["delta_rel"]*100:.2f}%) '
                  f'-> {s["imp"]}/{s["n"]}  k={K[rg]}  '
                  f'{"PASS" if s["imp"] >= K[rg] else "FAIL"}')

    print('\n [§2b] 9셸 상대변화 범위 (RESULT 주장: 희소 -32%~-35%, 혼잡 -7.1%~-19.4%)')
    for rg in ('sparse', 'crowded'):
        rels = []
        cnts = []
        for sh in SHELLS:
            s = sign_test(paired(rec, regimes, 'AUTO-L1', 'AUTO', rg, GS(sh)))
            rels.append(s['med_rel'] * 100)
            cnts.append(s['imp'])
        print(f'   {rg}: 부호 {min(cnts)}-{max(cnts)}  rel {min(rels):+.2f}% ~ {max(rels):+.2f}%')

    print('\n [§4] lr 축  AUTO-QTR-L1 - AUTO-L1 (혼잡)')
    for key, g in (('nRMSE', G('nRMSE')), ('all_resid', G('all_resid')), ('0_5', GS('0_5'))):
        line(key, sign_test(paired(rec, regimes, 'AUTO-QTR-L1', 'AUTO-L1', 'crowded', g)))
    print('   (희소 참고)')
    for key, g in (('nRMSE', G('nRMSE')), ('all_resid', G('all_resid')), ('0_5', GS('0_5'))):
        line(key, sign_test(paired(rec, regimes, 'AUTO-QTR-L1', 'AUTO-L1', 'sparse', g)))
    print('   AUTO-QTR - AUTO (희소, RESULT: 61/64 -7.39%)')
    line('nRMSE sparse', sign_test(paired(rec, regimes, 'AUTO-QTR', 'AUTO', 'sparse', G('nRMSE'))))

    print('\n [§5] 기제 · 교란')
    for rg in ('sparse', 'crowded'):
        for key in ('window_resid', 'decorr_mean', 'final_n', 'init_iters'):
            line(f'{rg} {key}', sign_test(paired(rec, regimes, 'AUTO-L1', 'AUTO', rg, G(key))))


def audit_direction1():
    """B-1 이 인용한 direction1 자료 재도출 (오늘 오류의 진원)."""
    print('\n' + '=' * 78 + '\nB-1 인용 자료 — results/direction1/*.json\n' + '=' * 78)
    root = os.path.join(os.path.dirname(TMP), 'results', 'direction1')
    d = json.load(open(os.path.join(root, 'traj_step1b_relax.json')))
    rows = d['rows']
    print(f'  traj_step1b_relax rows n={len(rows)}')
    for suf in ('oracle', 'relax'):
        print(f'   --- {suf} (기준 = stationary_{suf}) ---')
        base = [r[f'stationary_{suf}'] for r in rows]
        for arm in ('straight-cross', 'greedy', 'CEM', 'learned'):
            v = [r[f'{arm}_{suf}'] for r in rows]
            dl = [(i, v[i] - base[i], (v[i] - base[i]) / base[i]) for i in range(len(rows))]
            s = sign_test(dl)
            lo, hi = boot_ci([-x[1] for x in dl])   # 부호 = 정지 - 이동 (양수 = 이동 우세)
            rep = d['summary'][f'{arm}_{suf}']
            print(f'    {arm:15s} better {s["imp"]:2d}/{len(rows)} (보고 {rep["nbetter"]})  '
                  f'sign p {s["p"]:.4f}  meanDelta {-sum(x[1] for x in dl)/len(dl):+.5f} '
                  f'(보고 {rep["delta"]:+.5f})  bootCI [{lo:+.5f},{hi:+.5f}] '
                  f'(보고 [{rep["ci"][0]:+.5f},{rep["ci"][1]:+.5f}])')
        # 선택-후-검정: argmin 팔을 고른 뒤 그 팔만 검정하면?
        best = min(('straight-cross', 'greedy', 'CEM', 'learned'),
                   key=lambda a: sum(r[f'{a}_{suf}'] for r in rows))
        print(f'    argmin arm = {best}  (4팔 중 선택 -> 다중비교 보정 필요, '
              f'Bonferroni x4)')
    print('\n  jackknife — CI 하단이 얼마나 얇은가 (straight-cross_oracle)')
    base = [r['stationary_oracle'] for r in rows]
    v = [r['straight-cross_oracle'] for r in rows]
    dl = [base[i] - v[i] for i in range(len(rows))]   # 양수 = 이동 우세
    for drop in range(len(rows)):
        sub = [x for i, x in enumerate(dl) if i != drop]
        lo, hi = boot_ci(sub, B=4000)
        if hi > 0:
            print(f'    drop ep{drop}: mean {sum(sub)/len(sub):+.6f}  CI [{lo:+.6f},{hi:+.6f}] '
                  f'-> 0 포함')
            break
    else:
        print('    모든 1-제거에서 CI 상단 < 0')


def audit_b3():
    """B-3 AUC 재도출 (오늘 배치)."""
    print('\n' + '=' * 78 + '\nB-3 (2026-08-12) — tmp/B3_regime_position_2026-08-12\n' + '=' * 78)
    p = os.path.join(TMP, 'B3_regime_position_2026-08-12', 'b3_result.json')
    d = json.load(open(p))

    def auc(pos, neg):
        """Mann-Whitney AUC: P(pos > neg) + 0.5 P(=)."""
        c = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
        return c / (len(pos) * len(neg))

    for tier in ('T1', 'T2', 'T3'):
        rows = d['synth'][tier]
        sp = [r for r in rows if r['regime'] == 'sparse']
        cr = [r for r in rows if r['regime'] == 'crowded']
        out = []
        for stat in ('PR', 'H', 'E5'):
            out.append(f'{stat} {auc([r[stat] for r in cr], [r[stat] for r in sp]):.3f}')
        print(f'  {tier}: n_sparse {len(sp)} n_crowded {len(cr)}  ' + '  '.join(out))
        for nm, g in (('sparse', sp), ('crowded', cr)):
            v = sorted(r['PR'] for r in g)
            print(f'      {nm:8s} PR med {st.median(v):.2f} [{v[0]:.2f},{v[-1]:.2f}]')
    ap = os.path.join(TMP, 'B3_regime_position_2026-08-12', 'aperture_curve.json')
    if os.path.exists(ap):
        c = json.load(open(ap))
        print('  aperture curve keys:', list(c)[:10] if isinstance(c, dict) else len(c))


def audit_s11():
    """S11 §3 headline 41/14 p=3.6e-04 를 S9+X2 원자료에서 재도출."""
    print('\n' + '=' * 78 + '\nS11 (런 0회) — 널 보정 선택 이득 재도출\n' + '=' * 78)
    s9p = BATCH['s9']
    x2p = os.path.join(TMP, 'X2_no_learned_2026-08-11', 'results.jsonl')
    med = defaultdict(lambda: defaultdict(list))
    reg = {}
    for path in (s9p, x2p):
        for ln in open(path):
            r = json.loads(ln)
            if 'error' in r:
                continue
            ep = os.path.basename(r['episode'])
            for m in ('nRMSE', 'window_resid', 'all_resid'):
                if r.get(m) is not None:
                    med[(r['cell'], ep)][m].append(r[m])
            reg[ep] = r['regime']
    M = {k: {m: st.median(v) for m, v in d.items()} for k, d in med.items()}

    def sel(armA, armB, ep, stat):
        a, b = M.get((armA, ep)), M.get((armB, ep))
        if not a or not b:
            return None
        return armA if a[stat] <= b[stat] else armB

    print('  [A] S11 §3 C_matched = SELECT{AUTO,V2K} vs SELECT{AUTO,AUTO-SEED-NULL}')
    print('      (두 규칙이 팔 A=AUTO 를 공유. 발표값 희소 window_resid 41/14 p=3.55e-04 -35.80%)')
    for stat in ('window_resid', 'all_resid'):
        for rg in ('sparse', 'crowded'):
            eps = [e for e in reg if reg[e] == rg and ('AUTO', e) in M and ('V2K', e) in M
                   and ('AUTO-SEED-NULL', e) in M]
            rel = []
            for e in eps:
                a = M[(sel('AUTO', 'V2K', e, stat), e)]['nRMSE']
                b = M[(sel('AUTO', 'AUTO-SEED-NULL', e, stat), e)]['nRMSE']
                rel.append((e, a - b, (a - b) / b))
            s = sign_test(rel)
            print(f'   {stat:12s} {rg:8s} n={len(eps)}  {s["imp"]}/{s["wor"]}/tie{s["tie"]}  '
                  f'p {s["p"]:.6g}  med_rel {s["med_rel"]*100:+.2f}%')

    print('\n  [B] S11 §2-3 "크기의 60%가 lambda" — 팔 B 를 AUTO-L1 로 갈면?')
    for rg in ('sparse',):
        eps = [e for e in reg if reg[e] == rg and ('AUTO-L1', e) in M and ('V2K', e) in M
               and ('AUTO-SEED-NULL', e) in M]
        for stat in ('window_resid', 'all_resid'):
            rel = []
            for e in eps:
                a = M[(sel('AUTO-L1', 'V2K', e, stat), e)]['nRMSE']
                b = M[(sel('AUTO', 'AUTO-SEED-NULL', e, stat), e)]['nRMSE']
                rel.append((e, a - b, (a - b) / b))
            s = sign_test(rel)
            print(f'   SEL{{AUTO-L1,V2K}} vs SEL-NULL  {stat:12s} {rg} '
                  f'{s["imp"]}/{s["wor"]}  med_rel {s["med_rel"]*100:+.2f}%  p {s["p"]:.4g}')

    print('\n  [C] CON 지적 재확인 — 헤드라인 중앙/평균/최악, 1ep 제거 민감도')
    eps = [e for e in reg if reg[e] == 'sparse' and ('AUTO', e) in M and ('V2K', e) in M
           and ('AUTO-SEED-NULL', e) in M]
    rel = []
    for e in eps:
        a = M[(sel('AUTO', 'V2K', e, 'window_resid'), e)]['nRMSE']
        b = M[(sel('AUTO', 'AUTO-SEED-NULL', e, 'window_resid'), e)]['nRMSE']
        rel.append((e, (a - b) / b))
    vals = sorted(x[1] for x in rel)
    print(f'   median {st.median(vals)*100:+.2f}%  mean {sum(vals)/len(vals)*100:+.2f}%  '
          f'max {max(vals)*100:+.1f}%  min {min(vals)*100:+.1f}%')
    drop = sorted(rel, key=lambda x: -x[1])[0]
    rest = [x[1] for x in rel if x[0] != drop[0]]
    print(f'   최악 1ep({drop[0][:44]}) 제거 -> mean {sum(rest)/len(rest)*100:+.2f}%  '
          f'median {st.median(rest)*100:+.2f}%')


def audit_restart_splits():
    """S11 §6-1 `RESTART` 후보 — 6시드를 3/3 으로 가르는 **10가지 분할 전부**에서 재검정.

    S9 가 쓴 분할(1575-77 vs 1578-80)은 10개 중 하나일 뿐이고, cfg 가 완전히 동일하므로
    나머지 9개도 똑같이 정당하다. 분할을 바꾸면 결과가 얼마나 흔들리는가?
    """
    import itertools
    print('\n' + '=' * 78 + '\nS11 §6-1 RESTART — 6시드 3/3 분할 10가지 전수\n' + '=' * 78)
    rows = [json.loads(l) for l in open(BATCH['s9'])]
    cfgs = {r['cell']: json.dumps(r['cfg'], sort_keys=True)
            for r in rows if r['cell'] in ('AUTO', 'AUTO-SEED-NULL')}
    print(f'  cfg 동일: {cfgs["AUTO"] == cfgs["AUTO-SEED-NULL"]}')
    by = defaultdict(dict)
    for r in rows:
        if r['cell'] in ('AUTO', 'AUTO-SEED-NULL'):
            by[(r['regime'], os.path.basename(r['episode']))][r['seed']] = r
    seeds = sorted({s for d in by.values() for s in d})
    for rg in ('sparse', 'crowded'):
        eps = [k for k in by if k[0] == rg and len(by[k]) == len(seeds)]
        res = []
        for A in itertools.combinations(seeds, 3):
            B = tuple(s for s in seeds if s not in A)
            if A > B:
                continue
            rel = []
            for k in eps:
                d = by[k]
                wa = st.median([d[s]['window_resid'] for s in A])
                wb = st.median([d[s]['window_resid'] for s in B])
                na = st.median([d[s]['nRMSE'] for s in A])
                nb = st.median([d[s]['nRMSE'] for s in B])
                rel.append(((na if wa <= wb else nb) - na) / na)
            i = sum(1 for r in rel if r < 0)
            w = sum(1 for r in rel if r > 0)
            nz = [r for r in rel if r]
            res.append((p_two_sided(max(i, w), i + w) if i + w else 1.0, i, w,
                        len(rel) - i - w, A, st.median(nz) if nz else 0.0))
        res.sort()
        print(f'  --- {rg} n={len(eps)} ---')
        for p, i, w, t, A, m in res:
            s9 = ' <-S9 가 쓴 분할' if A == (1575, 1576, 1577) else ''
            print(f'    A={A}  {i:2d}/{w:2d} tie{t:2d}  p={p:.4f}  '
                  f'med_nonzero={m*100:+.2f}%{"  *" if p <= 0.05 else "   "}{s9}')
        print(f'    유의(p<=0.05) 분할 {sum(1 for r in res if r[0] <= 0.05)}/{len(res)}')


def audit_paper_headline():
    """CLAUDE.md "Current State" 3수 (WS 0.223 / ETKF 0.319 / GPR 0.276, 132.6 ms) 출처 추적."""
    print('\n' + '=' * 78 + '\nCLAUDE.md 헤드라인 3수 — 출처 추적\n' + '=' * 78)
    R = os.path.join(os.path.dirname(TMP), 'results')
    d = json.load(open(os.path.join(R, 'p1_comparison.json')))
    for m in ('WS-P0', 'WS-P1', 'ETKF'):
        v = [e[m]['avg_nRMSE'] for e in d.values() if m in e]
        t = [e[m]['avg_inference_ms'] for e in d.values() if m in e]
        print(f'  p1_comparison  {m:6s} n={len(v)} mean {sum(v)/len(v):.4f} '
              f'median {st.median(v):.4f}  ms {sum(t)/len(t):.1f}')
    g = json.load(open(os.path.join(R, 'gpr_comparison.json')))
    for m in ('WS-P1', 'ETKF', 'GPR'):
        v = [e[m]['avg_nRMSE'] for e in g.values() if m in e]
        t = [e[m].get('avg_inference_ms', 0) for e in g.values() if m in e]
        print(f'  gpr_comparison {m:6s} n={len(v)} mean {sum(v)/len(v):.4f}  '
              f'ms {sum(t)/len(t):.1f}')
    s = json.load(open(os.path.join(R, 'etkf_tuning_sweep.json')))
    rows = []
    for cfg, eps in s.items():
        v = [e['avg_nRMSE'] for e in eps.values() if isinstance(e, dict) and 'avg_nRMSE' in e]
        t = [e['avg_ms'] for e in eps.values() if isinstance(e, dict) and 'avg_ms' in e]
        if v:
            rows.append((sum(v) / len(v), cfg, sum(t) / len(t)))
    rows.sort()
    print(f'  etkf_tuning_sweep: 12 구성 중 argmin = {rows[0][1]}  '
          f'{rows[0][0]:.4f}  {rows[0][2]:.1f} ms   (⚠argmin 선택, 보정 없음)')
    print(f'    전 구성 범위 {rows[0][0]:.4f} .. {rows[-1][0]:.4f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None)
    a = ap.parse_args()
    which = (a.only.split(',') if a.only
             else ['s9', 's9x', 's8', 's7', 'cross', 'd1', 'b3', 's11',
                   'restart', 'paper'])
    if 's9' in which:
        audit_s9()
    if 's9x' in which:
        s9_shifted_and_lr()
    if 's8' in which:
        audit_s8()
    if 's7' in which:
        audit_s7()
    if 'cross' in which:
        cross_batch()
    if 'd1' in which:
        audit_direction1()
    if 'b3' in which:
        audit_b3()
    if 's11' in which:
        audit_s11()
    if 'restart' in which:
        audit_restart_splits()
    if 'paper' in which:
        audit_paper_headline()
