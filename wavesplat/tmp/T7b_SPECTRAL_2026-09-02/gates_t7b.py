#!/usr/bin/env python3
"""`T7b-SPECTRAL` 게이트 + 판정. ⛔런 0회. `--selftest` 로 항진 검사.

★`T7` 과의 유일한 실질 차이: 음성 대조를 **셋으로 쪼갠다**.
  `G-5a` 지표 생존(제로장이 좁은 대역에) --- 차단
  `G-5b` **우리 팔**이 제로장을 이긴다(>= 90% 행) --- 차단
  `G-5c` 비교자가 제로장을 이기는 비율 --- ⛔**차단 아님, 공시 항목**
⛔사유: 비교자가 어떤 지표에서 무너지는 것은 **측정 결과**이지 자료 무효 사유가 아니다
   (이 저장소가 `F1` 부류를 게이트가 아니라 **동반 공시**로 다루는 방식과 같다).
⚠`T7` 의 게이트를 그 자료 위에서 고쳐 다시 판정하지 **않았다** --- 코퍼스가 다르다.
"""
import os
import statistics as st
import sys

N_EXPECTED = 64
N_PAIRS = 32
NOMINAL, CONFIRM, NOT_EARN = 22, 23, 10          # PREREG §4 (n=32, family 2)
SHIPPED = {'lambda_l1': 1e-4, 'varpro_steps': 400, 'tracking_steps': 10,
           'tracking_window': 3, 'tracking_mode': 'w_phi', 'learned_init_iters': 0}
ROUTE = {'LEARN': 'learned', 'ANA': 'v2_konly'}
BLOCKING = ('G-1', 'G-2', 'G-3', 'G-4', 'G-5a', 'G-5b', 'G-6', 'G-7')
ZERO_BAND = 0.05          # `G-5a`: 제로장 R^2 가 이 대역 안에 있어야 한다
LEARN_MIN_FRAC = 0.90     # `G-5b`: 우리 팔이 제로장을 이기는 행 비율


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
    # ★음성 대조를 셋으로 (PREREG §3)
    zbad = [r for r in rows if abs(st.median(r['dir_r2_zero'])) > ZERO_BAND]
    g.append((f'G-5a 지표 생존(|R2_zero|<={ZERO_BAND})', not zbad and bool(rows),
              f'{len(rows) - len(zbad)}/{len(rows)}'))
    lrn = [r for r in rows if r['arm'] == 'LEARN']
    lok = [r for r in lrn if st.median(r['dir_r2']) > st.median(r['dir_r2_zero'])]
    g.append(('G-5b LEARN>제로장',
              bool(lrn) and len(lok) / len(lrn) >= LEARN_MIN_FRAC,
              f'{len(lok)}/{len(lrn)} ({len(lok) / max(len(lrn), 1):.0%}, '
              f'문턱 {LEARN_MIN_FRAC:.0%})'))
    ana = [r for r in rows if r['arm'] == 'ANA']
    aok = [r for r in ana if st.median(r['dir_r2']) > st.median(r['dir_r2_zero'])]
    g.append(('G-5c ANA>제로장 [공시]', True,
              f'{len(aok)}/{len(ana)} — ⛔차단 아님(보고 항목)'))
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
        # B: **LEARN** 의 Hs 오차가 더 낮은가 (PREREG §4 --- `T7` 관찰 방향)
        b1, b2 = st.median(la['hs_err']), st.median(an['hs_err'])
        B += (b1 < b2)
        relB.append(b2 - b1)
    return dict(n=len(relA), A_learn_better=A, B_learn_better=B,
                A_median_delta=st.median(relA) if relA else None,
                B_median_delta=st.median(relB) if relB else None)


def label(c):
    """PREREG §4. ★등급은 `EXPLORATORY` 로 **고정**한다(노출 공시)."""
    A, B = c['A_learn_better'], c['B_learn_better']
    if A >= CONFIRM and B >= CONFIRM:
        return 'SPECTRAL-BOTH [EXPLORATORY]'
    if A >= CONFIRM:
        return 'SPECTRAL-DIRECTIONAL [EXPLORATORY]'
    if A <= NOT_EARN:
        return 'SPECTRAL-DIRECTIONAL-LOSES [EXPLORATORY]'
    return 'SPECTRAL-UNDECIDED'


def selftest():
    """★항진 검사 --- 정상 통과 · 위조 전건 차단 · 라벨 공역."""
    base = dict(arm='LEARN', route_effective='learned', episode='e0',
                scored_idx=[0, 40], n_scored=2, dir_r2=[0.8, 0.8],
                hs_err=[0.09, 0.09], dir_r2_zero=[-0.01, -0.01],
                effective=dict(SHIPPED), final_n=16, seed=42, hs_true=1.2)
    good = []
    for i in range(N_PAIRS):
        for arm in ('LEARN', 'ANA'):
            r = dict(base, episode=f'e{i}', arm=arm, route_effective=ROUTE[arm])
            if arm == 'ANA':
                r['dir_r2'] = [0.6, 0.6]
                r['hs_err'] = [0.13, 0.13]
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
        'G-5a': ('G-5a', lambda rs: [dict(r, dir_r2_zero=[0.9, 0.9]) for r in rs]),
        'G-5b': ('G-5b', lambda rs: [dict(r, dir_r2=[-0.5, -0.5])
                                     if r['arm'] == 'LEARN' else dict(r) for r in rs]),
        'G-6': ('G-6', lambda rs: [dict(r, dir_r2=[float('nan')] * 2) for r in rs]),
        'G-7': ('G-7', lambda rs: [r for r in rs if r['arm'] == 'LEARN']),
    }
    for name, (expect, forge) in F.items():
        if expect not in blocked_by(evaluate(forge([dict(r) for r in good]))):
            bad.append((name, f'위조가 통과했다 (기대 {expect})'))
    # ★`G-5c` 는 **차단이 아니어야 한다** --- 비교자가 무너져도 라벨이 나와야 한다
    collapsed = [dict(r, dir_r2=[-9.0, -9.0]) if r['arm'] == 'ANA' else dict(r)
                 for r in good]
    if blocked_by(evaluate(collapsed)):
        bad.append(('G-5c', '비교자 붕괴가 **차단**했다 --- 공시여야 한다'))
    # 라벨 공역
    c = contrasts(good)
    if label(c) != 'SPECTRAL-BOTH [EXPLORATORY]':
        bad.append(('label', f'기대 BOTH, 실제 {label(c)}'))
    for A, B, want in ((32, 0, 'SPECTRAL-DIRECTIONAL [EXPLORATORY]'),
                       (0, 0, 'SPECTRAL-DIRECTIONAL-LOSES [EXPLORATORY]'),
                       (16, 0, 'SPECTRAL-UNDECIDED')):
        got = label(dict(A_learn_better=A, B_learn_better=B))
        if got != want:
            bad.append(('label', f'A={A} -> {got}, 기대 {want}'))
    uncovered = set(BLOCKING) - {e for e, _ in F.values()}
    if uncovered:
        bad.append(('coverage', f'위조 시험 없는 차단 게이트: {sorted(uncovered)}'))
    for n, why in bad:
        print(f'  [BAD ] {n}: {why}')
    if bad:
        print(f'⛔자가진단 실패 {len(bad)}건')
        return 1
    print(f'✅자가진단 통과 — 정상 통과 · 위조 {len(F)}종 차단 · '
          '★비교자 붕괴는 **차단하지 않는다** 확인 · 라벨 공역 4/4 · 차단 게이트 전건 보유')
    return 0


if __name__ == '__main__':
    sys.exit(selftest() if '--selftest' in sys.argv else 0)
