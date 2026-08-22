"""W7 단일출처 — 팔 정의 · 게이트 · 판정 규칙.

★러너와 분석기가 **둘 다 이 파일만** 읽는다.
저장소가 08-19~20 에 다섯 번 물린 결함(같은 값을 두 곳에 따로 적어 어긋남)의 처방이다.
`C` 배치는 게이트를 PREREG 에 옳게 적고 분석기에 안 넣어 `S-4` 를 잃었다.
"""

# ── 팔 (전부 λ 를 명시한다. 미명시 러너는 재실행 시 무효가 된다 — CLAUDE.md λ 절) ──
# tracking_lr=0 : S5 와 동일하게 추적기를 끄고 초기화 단계만 본다.
ARMS = {
    # 진실에서 0 스텝. 최적화를 안 하므로 λ 는 원리적으로 무관하나 명시한다.
    'ORA-WPHI-0STEP-R': {'tracking_lr': 0.0, 'lambda_l1': 0.0, 'lambda_smooth': 0.0,
                         'n_iters': 0},
    # S5 `ORA-WPHI-NT` 복제 = 당시 기본값. **양성 대조**(코드 표류 검사).
    'ORA-WPHI-L1E3':    {'tracking_lr': 0.0, 'lambda_l1': 1e-3, 'lambda_smooth': 1e-4},
    # ★★★없던 칸 = 출하 λ. 이 배치의 존재 이유.
    'ORA-WPHI-SHIP':    {'tracking_lr': 0.0, 'lambda_l1': 1e-4, 'lambda_smooth': 1e-4},
}
ARM_ORDER = ['ORA-WPHI-0STEP-R', 'ORA-WPHI-L1E3', 'ORA-WPHI-SHIP']

REGIME = 'crowded'          # 척추의 주장이 사는 층. 희소는 S5 에서 이미 널.
N_EPISODES = 16
SEEDS = [1617]              # 오라클 팔은 시드 무관(S4 확인, C2 가 비트동일로 재확인)
PRIMARY_FIELD = 'window_resid'
SECONDARY_FIELD = 'all_resid'
THRESHOLD = 13              # /16, S5 규약 그대로
S5_L1E3_INWINDOW_RATIO = 3.6648   # 내가 S5 원자료에서 재도출한 값 (2026-08-22)
CONTROL_BAND = (2.5, 5.5)         # 양성 대조 허용 구간 (결과 보기 전 확정)


def sign_test_one_sided(w, n):
    """H1: 처치 > 기준 인 짝의 수 w. 단측 정확 이항 p."""
    from math import comb
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def gates(rows):
    """★게이트 전건. 러너가 런 후 즉시, 분석기가 판정 전 각각 호출한다.

    반환 = [(id, ok, 설명), ...]. `ABORT` 게이트가 하나라도 False 면 판정 금지.
    """
    import collections
    out = []
    ok_rows = [r for r in rows if 'error' not in r]
    out.append(('G-0 런 수', len(ok_rows) == len(ARMS) * N_EPISODES * len(SEEDS),
                f'{len(ok_rows)} / {len(ARMS) * N_EPISODES * len(SEEDS)}'))
    out.append(('G-0b 에러 0', len(rows) - len(ok_rows) == 0,
                f'{len(rows) - len(ok_rows)} 에러'))

    # G-1 λ 가 실제로 먹었나 — ★의도 dict 가 아니라 **객체 되읽기**로 검사한다(S13b 교훈)
    bad = [r['job_id'] for r in ok_rows
           if abs(r.get('lambda_l1_readback', -1) - ARMS[r['cell']]['lambda_l1']) > 1e-12]
    out.append(('G-1 λ 객체 되읽기 일치', not bad, f'불일치 {len(bad)}'))

    # G-2 같은 창을 쟀나
    n_obs = collections.Counter(r.get('window_n_obs') for r in ok_rows)
    per_ep = collections.defaultdict(set)
    for r in ok_rows:
        per_ep[r['episode']].add(r.get('window_n_obs'))
    out.append(('G-2 에피소드별 창 크기 동일', all(len(v) == 1 for v in per_ep.values()),
                f'{sum(1 for v in per_ep.values() if len(v) != 1)} 에피소드 불일치'))

    # G-3 0STEP 이 실제로 진실인가 (음성 대조)
    z = [r for r in ok_rows if r['cell'] == 'ORA-WPHI-0STEP-R']
    out.append(('G-3 0STEP nRMSE ≈ 0', all(r.get('nRMSE', 1) < 0.05 for r in z),
                f'max nRMSE {max((r.get("nRMSE", 9) for r in z), default=float("nan")):.4f}'))

    # G-4 ★양성 대조 — S5 의 벽이 오늘도 재현되는가 (코드 표류 검사). 실패 시 ABORT.
    v = verdict_core(ok_rows, 'ORA-WPHI-L1E3')
    ok = v['wins'] >= THRESHOLD and CONTROL_BAND[0] <= v['ratio'] <= CONTROL_BAND[1]
    out.append((f'G-4 양성대조 L1E3 (ABORT)', ok,
                f'{v["wins"]}/{v["n"]} · ratio {v["ratio"]:.3f} '
                f'(S5 재도출 {S5_L1E3_INWINDOW_RATIO:.3f}, 허용 {CONTROL_BAND})'))
    return out


def verdict_core(rows, arm, field=PRIMARY_FIELD, ref='ORA-WPHI-0STEP-R'):
    """짝(에피소드) 비교. H1 = arm 이 ref 보다 **크다**(= 진실에서 밀려났다)."""
    import statistics as st
    a = {r['episode']: r[field] for r in rows if r['cell'] == arm and field in r}
    b = {r['episode']: r[field] for r in rows if r['cell'] == ref and field in r}
    ks = sorted(set(a) & set(b))
    n = len(ks)
    wins = sum(1 for k in ks if a[k] > b[k])
    ratio = st.median([a[k] / b[k] for k in ks if b[k]]) if ks else float('nan')
    return {'arm': arm, 'field': field, 'n': n, 'wins': wins, 'ratio': ratio,
            'p_one_sided': sign_test_one_sided(wins, n) if n else float('nan')}


# ── ★판정 규칙 (결과 보기 전 확정) ──
def label(primary):
    """PRIMARY = ORA-WPHI-SHIP vs 0STEP-R, 창 안(window_resid), 혼잡 n=16."""
    if primary['wins'] >= THRESHOLD:
        return ('WALL-AT-SHIPPED',
                '출하 λ 에서도 진실에서 밀려난다 ⟹ 척추의 창-안 절반이 회생한다.')
    return ('WALL-IS-PENALTY-ARTIFACT',
            '출하 λ 에서는 진실에 머문다 ⟹ 척추의 창-안 절반은 폐기된 λ=1e-3 의 산물로 확정.')
