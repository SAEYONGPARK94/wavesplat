"""Rx-7 채점 — PREREG.md §3/§4/§5 축자 구현.

⚠Rx-6 은 채점을 인라인 bash 로 해서 §6.5 산출물이 디스크에 없었다. 이 파일이 그 수정이다.
읽는 것: results.jsonl + nulls_rx7.json 뿐. 규약:
  시드중앙 -> 짝별 부호 있는 Δ -> Δ 의 중앙, 정확 양측 부호검정, 동점은 n 에서 제외.
  ⛔블록중앙의 차 금지 · ⛔절댓값 판정식 금지 · ⛔집계의 비 금지.
"""
import json
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
NEED = {16: 13, 15: 12, 14: 12, 13: 11, 12: 10}          # PREREG §3.3, 결과 전 동결
CELLS = ['AUTO', 'AUTO-DUP', 'AUTO-LR10', 'AUTO-LR100', 'AUTO-LR1000', 'AUTO-NOTRACK']
CANDIDATES = ['AUTO-LR10', 'AUTO-LR100', 'AUTO-LR1000']   # §4 PRIMARY-A 후보
PANEL = ['nRMSE', 'avg_RMSE_observed', 'avg_RMSE_unobserved', 'Pearson_rho',
         'absVR_framewise', 'm_mean', 'decorr_mean', 'shrink_mean']
GOOD_IF_NEG = {k: True for k in PANEL}
GOOD_IF_NEG['Pearson_rho'] = False                        # §3: 클수록 좋다


def sign_p(k, n):
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n)


def main():
    rows = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl')) if l.strip()]
    ok = [r for r in rows if 'error' not in r]
    print(f'rows={len(rows)} ok={len(ok)} err={len(rows)-len(ok)}')
    NULL = {n['episode']: n['zero_null_tracking_nRMSE_mean']
            for n in json.load(open(os.path.join(HERE, 'nulls_rx7.json')))}

    IDX = {}
    for r in ok:
        IDX.setdefault((r['cell'], r['episode']), []).append(r)
    eps = sorted({r['episode'] for r in ok})
    seeds = sorted({r['seed'] for r in ok})

    def med(c, e, k='nRMSE'):
        v = [x[k] for x in IDX.get((c, e), []) if x.get(k) is not None and x[k] == x[k]]
        return st.median(v) if v else None

    # ---------------- §5 무결성 ----------------
    print('\n=== §5 무결성 ===')
    nb = nc = 0
    FIELDS = ['nRMSE', 'init_nrmse', 'avg_RMSE_observed', 'avg_RMSE_unobserved',
              'Pearson_rho', 'm_mean', 'N', 'auto_selected', 'ensemble_k']
    for e in eps:
        for s in seeds:
            a = [x for x in IDX.get(('AUTO', e), []) if x['seed'] == s]
            d = [x for x in IDX.get(('AUTO-DUP', e), []) if x['seed'] == s]
            if a and d:
                for k in FIELDS:
                    nc += 1
                    if a[0].get(k) != d[0].get(k):
                        nb += 1
    print(f'  워커누출 AUTO-DUP: {nc} 비교, 불일치 {nb} -> {"PASS" if nb == 0 else "FAIL"}')
    bad = 0
    for e in eps:
        for s in seeds:
            vals = set()
            for c in CELLS:
                rr = [x for x in IDX.get((c, e), []) if x['seed'] == s]
                if rr:
                    vals.add((round(rr[0]['init_nrmse'], 9), rr[0]['auto_selected'],
                              rr[0]['N'], rr[0]['ensemble_k']))
            if len(vals) > 1:
                bad += 1
    print(f'  Phase-1 불변 (6셀): 불일치 {bad} -> {"PASS" if bad == 0 else "FAIL"}')

    # ---------------- §3.4 널 규칙 ----------------
    kept = [e for e in eps if med('AUTO', e) < NULL[e]]
    excl = [e for e in eps if e not in kept]
    n = len(kept)
    print(f'\n=== §3.4 널 규칙 ===\n  배제 {len(excl)}개: {[e[8:-17] for e in excl]}'
          f'  -> n={n}, 필요 승수 {NEED[n]}')
    print(f'  블록 널 중앙 = {st.median([NULL[e] for e in kept]):.4f}')
    for c in CELLS:
        lv = st.median([med(c, e) for e in kept])
        npass = sum(1 for e in kept if med(c, e) < NULL[e])
        print(f'  {c:14s} medOfMed={lv:.4f}  널통과 {npass}/{n}')

    def paired(a, b, key='nRMSE', blk=None):
        blk = blk or kept
        raw = [med(a, e, key) - med(b, e, key) for e in blk]
        ties = sum(1 for x in raw if x == 0)
        d = [x for x in raw if x != 0]
        if not d:
            return None, 0, 0, 1.0, ties
        wins = sum(1 for x in d if (x < 0) == GOOD_IF_NEG[key])
        return st.median(d), wins, len(d), sign_p(wins, len(d)), ties

    # ---------------- §4 PRIMARY-A ----------------
    print('\n=== ★PRIMARY-A: 줄인 lr 이 "끄기"를 이기는가 (인컴번트 = AUTO-NOTRACK) ===')
    res = {}
    for c in CANDIDATES:
        m, w, nn, p, t = paired(c, 'AUTO-NOTRACK')
        res[c] = (m, w, nn, p)
        print(f'  {c:13s} medΔ={m:+.5f}  개선 {w:2d}/{nn:<2d}  p={p:.4f}')
    # Holm (§4: 후보 3개)
    order = sorted(res, key=lambda c: res[c][3])
    print('\n  Holm 보정 (후보 3개):')
    keep_reduced = False
    for i, c in enumerate(order):
        alpha = 0.05 / (3 - i)
        m, w, nn, p = res[c]
        passed = p <= alpha and m < 0 and w >= NEED[nn]
        keep_reduced |= passed
        print(f'    {c:13s} p={p:.4f} vs α={alpha:.4f}  {"PASS" if passed else "fail"}')
        if p > alpha:
            break
    print(f'\n  ⟹ PRIMARY-A 판정: **{"KEEP-REDUCED" if keep_reduced else "SHIP-OFF"}**')

    # ---------------- §4 PRIMARY-B ----------------
    print('\n=== ★PRIMARY-B: 추적기를 켜면 "보이는 곳"이 나아지는가 ===')
    print('  (판정량 = AUTO − AUTO-NOTRACK. Δ<0 = 추적이 개선)')
    mo, wo, no, po, _ = paired('AUTO', 'AUTO-NOTRACK', 'avg_RMSE_observed')
    mu, wu, nu, pu, _ = paired('AUTO', 'AUTO-NOTRACK', 'avg_RMSE_unobserved')
    print(f'  관측영역   medΔ={mo:+.5f}  개선 {wo:2d}/{no:<2d} p={po:.4f}')
    print(f'  미관측영역 medΔ={mu:+.5f}  개선 {wu:2d}/{nu:<2d} p={pu:.4f}')
    obs_better = mo < 0 and wo >= NEED[no]
    obs_worse = mo > 0 and (no - wo) >= NEED[no]
    unobs_worse = mu > 0 and (nu - wu) >= NEED[nu]
    verdict = ('OVERFIT' if obs_better and unobs_worse else
               'MISSPECIFIED' if obs_worse else 'INCONCLUSIVE-B')
    print(f'  ⟹ PRIMARY-B 판정: **{verdict}**')

    # ---------------- 부차 ----------------
    print('\n=== 부차: 전 셀 vs AUTO / vs 바닥 (nRMSE) ===')
    for c in CELLS[1:]:
        m1, w1, n1, p1, t1 = paired(c, 'AUTO')
        m2, w2, n2, p2, t2 = paired(c, 'AUTO-NOTRACK')
        # a byte-duplicate cell has every paired difference exactly 0; PREREG §3.1
        # removes ties from n, so n collapses to 0 and there is no test to report.
        f1 = ('전부 동점(n=0)' if m1 is None
              else f'{m1:+.4f} ({w1:2d}/{n1} p={p1:.4f})')
        f2 = ('전부 동점(n=0)' if m2 is None
              else f'{m2:+.4f} ({w2:2d}/{n2} p={p2:.4f})')
        tag = (('  ≡AUTO' if (m1 is not None and p1 > 0.05) else '')
               + ('  ≡바닥' if (m2 is not None and p2 > 0.05) else ''))
        print(f'  {c:14s} vs AUTO {f1:28s} | vs바닥 {f2:28s}{tag}')

    print('\n=== 부차: 관측/미관측 패널 (각 셀 − 바닥) ===')
    print(f"  {'cell':14s} {'관측 Δ':>10s} {'승':>7s} | {'미관측 Δ':>10s} {'승':>7s}")
    for c in ['AUTO'] + CANDIDATES:
        a = paired(c, 'AUTO-NOTRACK', 'avg_RMSE_observed')
        b = paired(c, 'AUTO-NOTRACK', 'avg_RMSE_unobserved')
        print(f'  {c:14s} {a[0]:+10.5f} {a[1]:3d}/{a[2]:<3d} | {b[0]:+10.5f} {b[1]:3d}/{b[2]:<3d}')

    print('\n=== 부차: 설정 폴드 n=4 (⛔"유의"·효과크기 인용 금지) ===')
    cfgs = {}
    for e in kept:
        cfgs.setdefault(e.split('comp')[0], []).append(e)
    for c in CANDIDATES + ['AUTO']:
        d = []
        for _, ee in sorted(cfgs.items()):
            v = [med(c, e) - med('AUTO-NOTRACK', e) for e in ee]
            d.append(st.median(v))
        print(f'  {c:14s} vs바닥 폴드Δ 중앙={st.median(d):+.4f}  개선 {sum(1 for x in d if x < 0)}/{len(d)}')


if __name__ == '__main__':
    main()
