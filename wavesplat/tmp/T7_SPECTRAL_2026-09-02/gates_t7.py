#!/usr/bin/env python3
"""`T7-SPECTRAL` 게이트 + 판정. ⛔런 0회. `--selftest` 로 항진 검사."""
import os
import statistics as st
import sys

N_EXPECTED = 96
N_PAIRS = 48
NOMINAL, CONFIRM, NOT_EARN = 31, 32, 17          # PREREG §3 (n=48, family 2)
SHIPPED = {'lambda_l1': 1e-4, 'varpro_steps': 400, 'tracking_steps': 10,
           'tracking_window': 3, 'tracking_mode': 'w_phi', 'learned_init_iters': 0}
ROUTE = {'LEARN': 'learned', 'ANA': 'v2_konly'}
BLOCKING = ('G-1', 'G-2', 'G-3', 'G-4', 'G-5', 'G-6', 'G-7')


def gate_id(n):
    return n.split()[0]


def evaluate(rows):
    g = []
    ok = lambda c: bool(rows) and all(c(r) for r in rows)          # noqa: E731
    g.append(('G-1 행 수', len(rows) == N_EXPECTED, f'{len(rows)}/{N_EXPECTED}'))
    g.append(('G-2 경로 되읽기', ok(lambda r: r['route_effective'] == ROUTE[r['arm']]),
              f"{sum(1 for r in rows if r['route_effective'] == ROUTE[r['arm']])}/{len(rows)}"))
    pair = {}
    for r in rows:
        pair.setdefault(r['episode'], {})[r['arm']] = tuple(r['scored_idx'])
    same = [e for e, v in pair.items() if len(v) == 2 and len(set(v.values())) == 1]
    g.append(('G-3 채점 프레임 동일', len(same) == len(pair) and len(pair) == N_PAIRS,
              f'{len(same)}/{len(pair)}'))
    g.append(('G-4 배포 상수',
              ok(lambda r: all(abs(r['effective'][k] - v) < 1e-12
                               if isinstance(v, float) else r['effective'][k] == v
                               for k, v in SHIPPED.items())),
              f"{sum(1 for r in rows if r['effective'] == SHIPPED)}/{len(rows)}"))
    # 음성 대조: 제로장의 방향 R² 가 두 팔보다 낮아야 한다
    bad5 = [r for r in rows
            if st.median(r['dir_r2_zero']) >= st.median(r['dir_r2'])]
    g.append(('G-5 제로장 대조', not bad5 and bool(rows),
              f'{len(rows) - len(bad5)}/{len(rows)}'))
    import math
    g.append(('G-6 값 유한',
              ok(lambda r: all(math.isfinite(x) for x in r['dir_r2'] + r['hs_err'])),
              f"{sum(1 for r in rows if all(math.isfinite(x) for x in r['dir_r2'] + r['hs_err']))}/{len(rows)}"))
    g.append(('G-7 짝 완비', len(pair) == N_PAIRS and
              all(len(v) == 2 for v in pair.values()), f'{len(pair)}/{N_PAIRS}'))
    return g


def blocked_by(gates):
    return [gate_id(n) for n, p, _ in gates if not p and gate_id(n) in BLOCKING]


def contrasts(rows):
    by = {}
    for r in rows:
        by.setdefault(r['episode'], {})[r['arm']] = r
    A = B = 0
    relA, relB = [], []
    for e, v in by.items():
        if len(v) != 2:
            continue
        la, an = v['LEARN'], v['ANA']
        # A: LEARN 의 방향 R² 가 더 높은가 (높을수록 좋다)
        a1, a2 = st.median(la['dir_r2']), st.median(an['dir_r2'])
        A += (a1 > a2)
        relA.append(a1 - a2)
        # B: ANA 의 Hs 오차가 더 낮은가 (낮을수록 좋다)
        b1, b2 = st.median(la['hs_err']), st.median(an['hs_err'])
        B += (b2 < b1)
        relB.append(b1 - b2)
    return dict(n=len(relA), A_learn_better=A, B_ana_better=B,
                A_median_delta=st.median(relA) if relA else None,
                B_median_delta=st.median(relB) if relB else None)


def label(c):
    A, B = c['A_learn_better'], c['B_ana_better']
    if A >= CONFIRM and B >= CONFIRM:
        return 'SPECTRAL-SPLIT'
    if A >= CONFIRM:
        return 'SPECTRAL-DIRECTIONAL-EARNS'
    if A <= NOT_EARN:
        return 'SPECTRAL-DIRECTIONAL-LOSES'
    return 'SPECTRAL-UNDECIDED'


def selftest():
    base = dict(arm='LEARN', route_effective='learned', episode='e0',
                scored_idx=[0, 40], n_scored=2, dir_r2=[0.8, 0.8],
                hs_err=[0.13, 0.13], dir_r2_zero=[-0.01, -0.01],
                effective=dict(SHIPPED), final_n=16, seed=42, hs_true=1.2)
    good = []
    for i in range(N_PAIRS):
        for arm in ('LEARN', 'ANA'):
            r = dict(base, episode=f'e{i}', arm=arm, route_effective=ROUTE[arm])
            if arm == 'ANA':
                r['dir_r2'] = [0.6, 0.6]
                r['hs_err'] = [0.09, 0.09]
            good.append(r)
    bad = []
    if blocked_by(evaluate(good)):
        bad.append(('정상', '막혔다: ' + str(blocked_by(evaluate(good)))))
    F = {
        'G-1': ('G-1', lambda rs: rs[:-1]),
        'G-2': ('G-2', lambda rs: [dict(r, route_effective='bank_mmse') for r in rs]),
        'G-3': ('G-3', lambda rs: [dict(r, scored_idx=([0] if r['arm'] == 'ANA'
                                                       else [0, 40])) for r in rs]),
        'G-4': ('G-4', lambda rs: [dict(r, effective=dict(SHIPPED, varpro_steps=0))
                                   for r in rs]),
        'G-5': ('G-5', lambda rs: [dict(r, dir_r2_zero=[0.99, 0.99]) for r in rs]),
        'G-6': ('G-6', lambda rs: [dict(r, dir_r2=[float('nan')] * 2) for r in rs]),
        'G-7': ('G-7', lambda rs: [r for r in rs if r['arm'] == 'LEARN']),
    }
    for name, (expect, forge) in F.items():
        if expect not in blocked_by(evaluate(forge([dict(r) for r in good]))):
            bad.append((name, f'위조가 통과했다 (기대 {expect})'))
    # 라벨 공역
    c = contrasts(good)
    if label(c) != 'SPECTRAL-SPLIT':
        bad.append(('label', f'기대 SPLIT, 실제 {label(c)}'))
    if label(dict(A_learn_better=0, B_ana_better=0)) != 'SPECTRAL-DIRECTIONAL-LOSES':
        bad.append(('label', 'LOSES 미도달'))
    if label(dict(A_learn_better=40, B_ana_better=0)) != 'SPECTRAL-DIRECTIONAL-EARNS':
        bad.append(('label', 'EARNS 미도달'))
    if label(dict(A_learn_better=25, B_ana_better=0)) != 'SPECTRAL-UNDECIDED':
        bad.append(('label', 'UNDECIDED 미도달'))
    uncovered = set(BLOCKING) - {e for e, _ in F.values()}
    if uncovered:
        bad.append(('coverage', f'위조 시험 없는 차단 게이트: {sorted(uncovered)}'))
    for n, why in bad:
        print(f'  [BAD ] {n}: {why}')
    if bad:
        print(f'⛔자가진단 실패 {len(bad)}건')
        return 1
    print(f'✅자가진단 통과 — 정상 통과 · 위조 {len(F)}종 차단 · 라벨 공역 4/4 · '
          '차단 게이트 전건 위조 시험 보유')
    return 0


if __name__ == '__main__':
    sys.exit(selftest() if '--selftest' in sys.argv else 0)
