"""E-B 채점 — PREREG §3/§4/§5 축자.

★Rx-7 채점기가 §3 의무 5개(미배제 병기 · 배제 방향성 · m>1 표 · MDE · 런수준 분할)를
빠뜨렸고 그중 하나가 판정을 뒤집었다. 이 파일은 그 수정이며, **의무 항목을 전부 출력한다.**
레인1(PRIMARY)과 레인2(병기)는 **절대 풀링하지 않는다.**
"""
import json
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = ['AUTO', 'AUTO-DUP', 'AUTO-LR10', 'AUTO-LR100', 'AUTO-LR1000', 'AUTO-NOTRACK']
NEED = {48: 31, 47: 31, 46: 30, 45: 30, 44: 29, 43: 29, 32: 22, 31: 22, 30: 21, 29: 21, 28: 20, 27: 20, 26: 19, 25: 19,
        16: 13, 15: 12, 14: 12, 13: 11, 12: 10, 11: 10, 10: 9}
GOOD_NEG = {'nRMSE': True, 'avg_RMSE_observed': True, 'avg_RMSE_unobserved': True,
            'avg_obs_loss': True, 'obs_resid_rms': True, 'obs_resid_rms_oos': True, 'Pearson_rho': False,
            'm_mean': True, 'decorr_mean': True, 'shrink_mean': True,
            'absVR_framewise': True, 'init_nrmse': True}


def sign_p(k, n):
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n)


def load(path, nulls_files):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    ok = [r for r in rows if 'error' not in r]
    NULL = {}
    for nf in nulls_files:
        p = os.path.join(HERE, '..', nf)
        if os.path.exists(p):
            for n in json.load(open(p)):
                NULL[n['episode']] = n['zero_null_tracking_nRMSE_mean']
    return rows, ok, NULL


def score(lane, path, nulls_files, primary):
    rows, ok, NULL = load(path, nulls_files)
    print('\n' + '=' * 78)
    print(f'### {lane}   rows={len(rows)} ok={len(ok)} err={len(rows)-len(ok)}')
    IDX = {}
    for r in ok:
        IDX.setdefault((r['cell'], r['episode']), []).append(r)
    eps = sorted({r['episode'] for r in ok})
    seeds = sorted({r['seed'] for r in ok})

    def med(c, e, k='nRMSE'):
        v = [x[k] for x in IDX.get((c, e), []) if x.get(k) is not None and x[k] == x[k]]
        return st.median(v) if v else None

    # ---- §5 무결성 ----
    nb = nc = 0
    F = ['nRMSE', 'init_nrmse', 'avg_RMSE_observed', 'avg_RMSE_unobserved',
         'avg_obs_loss', 'obs_resid_rms', 'N', 'auto_selected', 'ensemble_k']
    for e in eps:
        for s in seeds:
            a = [x for x in IDX.get(('AUTO', e), []) if x['seed'] == s]
            d = [x for x in IDX.get(('AUTO-DUP', e), []) if x['seed'] == s]
            if a and d:
                for k in F:
                    nc += 1
                    nb += (a[0].get(k) != d[0].get(k))
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
    print(f'  §5 워커누출 {nc} 비교 불일치 {nb} -> {"PASS" if nb==0 else "FAIL"} | '
          f'Phase-1 불변 불일치 {bad} -> {"PASS" if bad==0 else "FAIL"}')

    # ---- §3 널 규칙 (배제 + ★미배제 병기 + ★방향성) ----
    kept = [e for e in eps if med('AUTO', e) < NULL[e]]
    excl = [e for e in eps if e not in kept]
    print(f'  §3 널 배제 {len(excl)}/{len(eps)}: {[e[8:-17] for e in excl]}  -> n={len(kept)}')

    def paired(a, b, key='nRMSE', blk=None):
        blk = blk or kept
        raw = [med(a, e, key) - med(b, e, key) for e in blk
               if med(a, e, key) is not None and med(b, e, key) is not None]
        ties = sum(1 for x in raw if x == 0)
        d = [x for x in raw if x != 0]
        if not d:
            return None, 0, 0, 1.0, ties
        w = sum(1 for x in d if (x < 0) == GOOD_NEG[key])
        return st.median(d), w, len(d), sign_p(w, len(d)), ties

    # ★§3 의무: 배제 방향성
    if excl:
        print('  ★배제 방향성 (배제된 에피소드에서 각 후보 − 바닥):')
        for c in ['AUTO-LR10', 'AUTO-LR100', 'AUTO-LR1000']:
            v = [med(c, e) - med('AUTO-NOTRACK', e) for e in excl]
            print(f'     {c:13s} {[f"{x:+.5f}" for x in v]}')

    # ★§3 의무: 널통과 / m>1 표
    print(f'  ★셀별 널통과 / m>1 (전 {len(eps)} 에피소드):')
    for c in CELLS:
        npass = sum(1 for e in eps if med(c, e) is not None and med(c, e) < NULL[e])
        mg = sum(1 for e in eps for x in IDX.get((c, e), [])
                 if x.get('m_mean') is not None and x['m_mean'] > 1)
        print(f'     {c:14s} 널통과 {npass:2d}/{len(eps)}   m>1 런 {mg}')

    print(f'\n  --- 수준 (블록중앙, n={len(kept)}) ---')
    print(f'  {"cell":14s} {"nRMSE":>8s} {"관측":>9s} {"미관측":>9s} '
          f'{"obs_loss":>10s} {"resid_rms":>10s}')
    for c in CELLS:
        f = lambda k: st.median([med(c, e, k) for e in kept])  # noqa: E731
        print(f'  {c:14s} {f("nRMSE"):8.4f} {f("avg_RMSE_observed"):9.4f} '
              f'{f("avg_RMSE_unobserved"):9.4f} {f("avg_obs_loss"):10.6f} '
              f'{f("obs_resid_rms"):10.5f}')

    if not primary:
        # 병기 레인: Rx-7 PRIMARY-B 재발 여부만
        print(f'\n  --- 병기: 추적 on/off (AUTO − 바닥), n={len(kept)}, 필요 {NEED[len(kept)]} ---')
        for k in ('nRMSE', 'avg_RMSE_observed', 'avg_RMSE_unobserved', 'avg_obs_loss'):
            m, w, n, p, _ = paired('AUTO', 'AUTO-NOTRACK', k)
            print(f'     {k:22s} medΔ={m:+.5f}  개선 {w:2d}/{n:<2d}  p={p:.6f}')
        return

    # ---- ★PRIMARY-1 ----
    print(f'\n=== ★PRIMARY-1: 목적함수가 더 나쁜 필드를 선호하는가 (n={len(kept)}, 필요 {NEED[len(kept)]}) ===')
    for k, lbl in (('avg_obs_loss', '목적함수(정규화 포함)'),
                   ('obs_resid_rms', 'in-sample 잔차 (훈련)'),
                   ('obs_resid_rms_oos', '★★표본외 잔차 (다음 프레임)'),
                   ('nRMSE', '필드 오차')):
        m, w, n, p, _ = paired('AUTO', 'AUTO-NOTRACK', k)
        print(f'  {lbl:24s} medΔ={m:+.6f}  낮아짐 {w:2d}/{n:<2d}  p={p:.6f}')
    mo, wo, no, po, _ = paired('AUTO', 'AUTO-NOTRACK', 'obs_resid_rms_oos')
    print(f'  ⟹ PRIMARY-1 = **{"OOS-ALSO-LOWER" if (mo < 0 and wo >= NEED[no]) else "OOS-NOT-LOWER"}**')
    mi = paired('AUTO', 'AUTO-NOTRACK', 'obs_resid_rms')[0]
    print(f'     비 (oos/in) 점추정 = {mo/mi:.3f}' if mi else '')
    ml, wl, nl, pl, _ = paired('AUTO', 'AUTO-NOTRACK', 'avg_obs_loss')
    mf, wf, nf, pf, _ = paired('AUTO', 'AUTO-NOTRACK', 'nRMSE')
    obj_lower = ml < 0 and wl >= NEED[nl]
    field_worse = mf > 0 and (nf - wf) >= NEED[nf]
    print(f'  ⟹ PRIMARY-1 = **{"OBJECTIVE-PREFERS-WORSE" if obj_lower and field_worse else "NO"}**')

    # ---- ★PRIMARY-2 ----
    print('\n=== ★PRIMARY-2: GT-free 대리 타당성 (에피소드별 5-rung Spearman) ===')
    ladder = ['AUTO-NOTRACK', 'AUTO-LR1000', 'AUTO-LR100', 'AUTO-LR10', 'AUTO']

    def spearman(a, b):
        n = len(a)
        ra = [sorted(a).index(x) for x in a]
        rb = [sorted(b).index(x) for x in b]
        mra, mrb = sum(ra) / n, sum(rb) / n
        num = sum((x - mra) * (y - mrb) for x, y in zip(ra, rb))
        den = math.sqrt(sum((x - mra) ** 2 for x in ra) * sum((y - mrb) ** 2 for y in rb))
        return num / den if den else float('nan')

    rhos = []
    for e in kept:
        a = [med(c, e, 'obs_resid_rms') for c in ladder]
        b = [med(c, e, 'avg_RMSE_observed') for c in ladder]
        if None not in a and None not in b:
            rhos.append(spearman(a, b))
    hi = sum(1 for r in rhos if r >= 0.9)
    print(f'  Spearman 중앙={st.median(rhos):+.4f}  ρ≥0.9 인 에피소드 {hi}/{len(rhos)}  '
          f'필요 {NEED[len(rhos)]}')
    print(f'  ⟹ PRIMARY-2 = **{"SURROGATE-VALID" if hi >= NEED[len(rhos)] else "INVALID"}**')

    # ---- 부차: 교차반경 ----
    print('\n=== 부차: min_dist 교차반경 (AUTO − 바닥, 관측영역) ===')
    print(f'  {"r[m]":>5s} {"medΔ":>10s} {"개선":>8s} {"p":>9s}')
    for rad in ('5', '10', '20', '30', '50', '75', '100', '150'):
        d = []
        for e in kept:
            va = [x['rmse_by_radius'][rad] for x in IDX.get(('AUTO', e), [])
                  if x.get('rmse_by_radius', {}).get(rad) == x.get('rmse_by_radius', {}).get(rad)]
            vb = [x['rmse_by_radius'][rad] for x in IDX.get(('AUTO-NOTRACK', e), [])
                  if x.get('rmse_by_radius', {}).get(rad) == x.get('rmse_by_radius', {}).get(rad)]
            if va and vb:
                d.append(st.median(va) - st.median(vb))
        d = [x for x in d if x != 0]
        if d:
            w = sum(1 for x in d if x < 0)
            print(f'  {rad:>5s} {st.median(d):+10.5f} {w:3d}/{len(d):<3d} {sign_p(w,len(d)):9.5f}')
    mfull, wfull, nfull, pfull, _ = paired('AUTO', 'AUTO-NOTRACK', 'nRMSE')
    print(f'  {"전체":>5s} {mfull:+10.5f} {wfull:3d}/{nfull:<3d} {pfull:9.5f}   <- 부호 반대')

    # ---- ★§3 의무: 미배제 병기 + MDE ----
    print(f'\n=== ★§3 의무: 미배제 병기 (n={len(eps)}, 필요 {NEED[len(eps)]}) ===')
    for c in ['AUTO-LR10', 'AUTO-LR100', 'AUTO-LR1000']:
        m1, w1, n1, p1, _ = paired(c, 'AUTO-NOTRACK', 'nRMSE')
        m2, w2, n2, p2, _ = paired(c, 'AUTO-NOTRACK', 'nRMSE', blk=eps)
        print(f'  {c:13s} 배제후 {m1:+.5f} {w1:2d}/{n1:<2d} p={p1:.4f} | '
              f'★미배제 {m2:+.5f} {w2:2d}/{n2:<2d} p={p2:.4f}')
    print('\n=== ★§3 의무: MDE (짝 sd 기반, 80% 검정력) ===')
    inc = abs(paired('AUTO', 'AUTO-NOTRACK', 'nRMSE')[0])
    for c in ['AUTO-LR10', 'AUTO-LR100', 'AUTO-LR1000']:
        raw = [med(c, e) - med('AUTO-NOTRACK', e) for e in kept]
        sd = st.stdev(raw) if len(raw) > 1 else float('nan')
        mde = 2.49 * sd / math.sqrt(len(raw))
        print(f'  {c:13s} sd={sd:.5f}  MDE={mde:.5f}  MDE/여유={100*mde/inc:5.1f}%  '
              f'관측/MDE={abs(st.median(raw))/mde:.2f}x')


if __name__ == '__main__':
    score('E-C — 독립 48 (⛔신규 아님: 세 배치가 이미 채점)',
          os.path.join(HERE, 'results.jsonl'),
          ['Rx6_channel_curve_2026-08-07/nulls_rx6.json',
           'Rx7_tracker_deploy_2026-08-07/nulls_rx7.json'], primary=True)
