"""E-D 채점 — PREREG §3/§4/§5 축자. **처음부터 새로 작성**(E-C 채점기 복사 아님).

⚠E-C 채점기는 `analyze_eb.py` 복사본이라 (a) §5 6항 중 2개만 실행 (b) CROSSOVER 판정을
아예 출력 안 함 (c) 사전등록에 없는 Spearman 을 PRIMARY-2 자리에 남겨뒀다. 이 파일은 그 수정이다.
§3 의 의무 항목(미배제 병기·배제 방향성·널통과/m>1·MDE·셸 면적)을 **전부** 출력한다.
"""
import json
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = ['AUTO', 'AUTO-DUP', 'AUTO-LR10', 'AUTO-LR100', 'AUTO-LR1000', 'AUTO-NOTRACK']
NEED = {48: 31, 47: 31, 46: 30, 45: 30, 44: 29, 43: 29, 42: 28, 41: 28, 40: 27}
LEADS = [('1', '0.1s'), ('5', '0.5s'), ('10', '1.0s'), ('20', '2.0s')]
SHELLS = ['0_5', '5_10', '10_20', '20_30', '30_50', '50_75', '75_100', '100_150', '150_inf']


def sign_p(k, n):
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n)


def main():
    rows = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl')) if l.strip()]
    ok = [r for r in rows if 'error' not in r]
    print(f'=== E-D  rows={len(rows)} ok={len(ok)} err={len(rows)-len(ok)} ===')
    print('⚠ 이 48 에피소드는 네 배치가 이미 채점했다. ⛔풀링·"재현" 금지.')
    NULL = {}
    for nf in ('Rx6_channel_curve_2026-08-07/nulls_rx6.json',
               'Rx7_tracker_deploy_2026-08-07/nulls_rx7.json'):
        p = os.path.join(HERE, '..', nf)
        if os.path.exists(p):
            for n in json.load(open(p)):
                NULL[n['episode']] = n['zero_null_tracking_nRMSE_mean']

    IDX = {}
    for r in ok:
        IDX.setdefault((r['cell'], r['episode']), []).append(r)
    eps = sorted({r['episode'] for r in ok})
    seeds = sorted({r['seed'] for r in ok})

    def med(c, e, path):
        vals = []
        for x in IDX.get((c, e), []):
            v = x
            for k in path:
                v = v.get(k) if isinstance(v, dict) else None
                if v is None:
                    break
            if v is not None and v == v:
                vals.append(v)
        return st.median(vals) if vals else None

    # ---------------- §5 무결성 (6항 전부) ----------------
    print('\n--- §5 무결성 ---')
    nb = nc = 0
    for e in eps:
        for s in seeds:
            a = [x for x in IDX.get(('AUTO', e), []) if x['seed'] == s]
            d = [x for x in IDX.get(('AUTO-DUP', e), []) if x['seed'] == s]
            if a and d:
                for k in a[0]:
                    if k in ('job_id', 'cell', 'wall_s', 'elapsed_ms'):
                        continue
                    nc += 1
                    nb += (a[0].get(k) != d[0].get(k))
    print(f'  1. 워커누출 (전 필드) {nc} 비교, 불일치 {nb} -> {"PASS" if nb == 0 else "FAIL"}')
    bad = 0
    for e in eps:
        for s in seeds:
            v = set()
            for c in CELLS:
                rr = [x for x in IDX.get((c, e), []) if x['seed'] == s]
                if rr:
                    v.add((round(rr[0]['init_nrmse'], 9), rr[0]['auto_selected'],
                           rr[0]['N'], rr[0]['ensemble_k']))
            bad += (len(v) > 1)
    print(f'  2. Phase-1 불변 (6셀) 불일치 {bad} -> {"PASS" if bad == 0 else "FAIL"}')
    cfgs = {c: {json.dumps(x['cfg'], sort_keys=True) for e in eps
                for x in IDX.get((c, e), [])} for c in CELLS}
    print(f'  3. 셀당 cfg 유일성: {all(len(v) == 1 for v in cfgs.values())}')
    fr = [med('AUTO', e, ('shell_frac', k)) for k in SHELLS for e in eps[:1]]
    tot = sum(x for x in fr if x)
    print(f'  5. 셸 가법성: Σfrac = {tot:.6f} -> {"PASS" if abs(tot - 1) < 1e-6 else "FAIL"}')
    d1 = [(med(c, e, ('lead_resid', '1')), med(c, e, ('obs_resid_rms_oos',)))
          for c in CELLS for e in eps]
    mm = sum(1 for a, b in d1 if a is not None and b is not None and abs(a - b) > 1e-9)
    print(f'  6. 리드 배선 (lead[1] == obs_resid_rms_oos): 불일치 {mm} -> '
          f'{"PASS" if mm == 0 else "FAIL"}')

    # ---------------- §3 널 규칙 ----------------
    kept = [e for e in eps if med('AUTO', e, ('nRMSE',)) < NULL[e]]
    excl = [e for e in eps if e not in kept]
    n = len(kept)
    print(f'\n--- §3 널 규칙: 배제 {len(excl)}/{len(eps)} -> n={n}, 필요 {NEED[n]} ---')
    print(f'  배제: {[e[8:-17] for e in excl]}')
    if excl:
        print('  ★배제 방향성 (AUTO − 바닥, nRMSE):')
        for e in excl:
            print(f'     {e[8:-17]:24s} {med("AUTO", e, ("nRMSE",)) - med("AUTO-NOTRACK", e, ("nRMSE",)):+.5f}')
    print('  ★셀별 널통과 / m>1:')
    for c in CELLS:
        np_ = sum(1 for e in eps if med(c, e, ('nRMSE',)) < NULL[e])
        mg = sum(1 for e in eps for x in IDX.get((c, e), [])
                 if x.get('m_mean') is not None and x['m_mean'] > 1)
        print(f'     {c:14s} {np_:2d}/{len(eps)}   m>1 런 {mg}')

    def paired(a, b, path, blk=None, good_neg=True):
        blk = blk or kept
        raw = [med(a, e, path) - med(b, e, path) for e in blk
               if med(a, e, path) is not None and med(b, e, path) is not None]
        d = [x for x in raw if x != 0]
        if not d:
            return None, 0, 0, 1.0
        w = sum(1 for x in d if (x < 0) == good_neg)
        return st.median(d), w, len(d), sign_p(w, len(d))

    # ---------------- ★PRIMARY-1 ----------------
    print(f'\n=== ★PRIMARY-1: 잔차 우위가 리드와 함께 사라지는가 (n={n}, 필요 {NEED[n]}) ===')
    print(f'  {"lead":>6s} {"AUTO":>9s} {"바닥":>9s} {"medΔ":>10s} {"개선":>8s} {"p":>9s}')
    A = {}
    for L, lbl in LEADS:
        lv_a = st.median([med('AUTO', e, ('lead_resid', L)) for e in kept
                          if med('AUTO', e, ('lead_resid', L)) is not None])
        lv_b = st.median([med('AUTO-NOTRACK', e, ('lead_resid', L)) for e in kept
                          if med('AUTO-NOTRACK', e, ('lead_resid', L)) is not None])
        m, w, nn, p = paired('AUTO', 'AUTO-NOTRACK', ('lead_resid', L))
        A[L] = abs(m) if m else 0.0
        print(f'  {lbl:>6s} {lv_a:9.5f} {lv_b:9.5f} {m:+10.5f} {w:3d}/{nn:<3d} {p:9.6f}')
    ratio = A['20'] / A['1'] if A['1'] else float('nan')
    m20, w20, n20, p20 = paired('AUTO', 'AUTO-NOTRACK', ('lead_resid', '20'))
    ok20 = m20 is not None and m20 < 0 and w20 >= NEED[n20]
    verdict = ('WHITE-FITTING' if (ratio < 0.5 and not ok20) else
               'STRUCTURE-REAL' if (ok20 and ratio >= 0.5) else 'PARTIAL')
    print(f'  ★A(2.0s)/A(0.1s) = {ratio:.3f}   L=20 문턱통과 = {ok20}')
    print(f'  ⟹ PRIMARY-1 = **{verdict}**')

    # ---------------- ★PRIMARY-2 ----------------
    print(f'\n=== ★PRIMARY-2: 셸 부호반전 (면적 병기 의무) ===')
    print(f'  {"shell[m]":>10s} {"면적%":>7s} {"AUTO":>8s} {"바닥":>8s} {"medΔ":>10s} '
          f'{"개선":>8s} {"p":>9s}')
    signs = []
    for k in SHELLS:
        fr_k = st.median([med('AUTO', e, ('shell_frac', k)) for e in kept
                          if med('AUTO', e, ('shell_frac', k)) is not None])
        la = st.median([med('AUTO', e, ('shell_rmse', k)) for e in kept
                        if med('AUTO', e, ('shell_rmse', k)) is not None])
        lb = st.median([med('AUTO-NOTRACK', e, ('shell_rmse', k)) for e in kept
                        if med('AUTO-NOTRACK', e, ('shell_rmse', k)) is not None])
        m, w, nn, p = paired('AUTO', 'AUTO-NOTRACK', ('shell_rmse', k))
        passed = w >= NEED[nn] or (nn - w) >= NEED[nn]
        signs.append((k, m, passed))
        print(f'  {k:>10s} {100*fr_k:7.2f} {la:8.4f} {lb:8.4f} {m:+10.5f} '
              f'{w:3d}/{nn:<3d} {p:9.6f}{"  *" if passed else ""}')
    cross = None
    for i in range(len(signs) - 1):
        (k1, m1, p1), (k2, m2, p2) = signs[i], signs[i + 1]
        if m1 is not None and m2 is not None and (m1 < 0 < m2) and p1 and p2:
            cross = (k1, k2)
            break
    if signs[0][1] is not None and signs[0][1] > 0 and signs[0][2]:
        print('  ⟹ PRIMARY-2 = **NO-INNER-BENEFIT**')
    elif cross:
        print(f'  ⟹ PRIMARY-2 = **SHELL-CROSSING-LOCATED** ({cross[0]} → {cross[1]} 사이)')
    else:
        print('  ⟹ PRIMARY-2 = **NO-CROSSING (또는 인접 문턱 불충족)**')

    # ---------------- §3 의무: 미배제 병기 + MDE ----------------
    print(f'\n=== ★§3 의무: 미배제 병기 (n={len(eps)}, 필요 {NEED[len(eps)]}) ===')
    for L, lbl in LEADS:
        m1, w1, n1, p1 = paired('AUTO', 'AUTO-NOTRACK', ('lead_resid', L))
        m2, w2, n2, p2 = paired('AUTO', 'AUTO-NOTRACK', ('lead_resid', L), blk=eps)
        print(f'  lead {lbl:>5s} 배제후 {m1:+.5f} {w1:3d}/{n1:<3d} p={p1:.5f} | '
              f'★미배제 {m2:+.5f} {w2:3d}/{n2:<3d} p={p2:.5f}')
    print('\n=== ★§3 의무: MDE (짝 sd, 80% 검정력) ===')
    inc = abs(paired('AUTO', 'AUTO-NOTRACK', ('nRMSE',))[0])
    for L, lbl in LEADS:
        raw = [med('AUTO', e, ('lead_resid', L)) - med('AUTO-NOTRACK', e, ('lead_resid', L))
               for e in kept
               if med('AUTO', e, ('lead_resid', L)) is not None]
        sd = st.stdev(raw) if len(raw) > 1 else float('nan')
        mde = 2.49 * sd / math.sqrt(len(raw))
        print(f'  lead {lbl:>5s} sd={sd:.5f} MDE={mde:.5f} 관측/MDE={abs(st.median(raw))/mde:5.2f}x')
    print(f'\n  (참고) 추적기 nRMSE 벌점 = {inc:+.5f}')


if __name__ == '__main__':
    main()
