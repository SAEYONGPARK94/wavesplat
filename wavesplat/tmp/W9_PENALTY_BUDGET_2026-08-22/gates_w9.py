"""W9 단일출처 — R2-2 의 `PENALTY-BUDGET` 가설을 관측에서 개입으로 올린다."""

# ★W7 의 결함을 고쳤다: 0-스텝의 옳은 노브는 `max_iters_per_stage:[0]` 이다(`n_iters` 아님).
ARMS = {
    'ORA-WPHI-0STEP': dict(extra={'tracking_lr': 0.0, 'max_iters_per_stage': [0],
                                  'loss_thresholds': [None],
                                  'lambda_l1': 0.0, 'lambda_smooth': 0.0},
                           scale_invariant=False),
    # W7 복제 = 양성 대조. 출하 λ, 척도불변 off.
    'ORA-WPHI-SHIP': dict(extra={'tracking_lr': 0.0, 'lambda_l1': 1e-4,
                                 'lambda_smooth': 1e-4},
                          scale_invariant=False),
    # ★개입: 같은 λ, 벌점을 [m^2] 로 만든다.
    'ORA-WPHI-SI':   dict(extra={'tracking_lr': 0.0, 'lambda_l1': 1e-4,
                                 'lambda_smooth': 1e-4},
                          scale_invariant=True),
}
ARM_ORDER = ['ORA-WPHI-0STEP', 'ORA-WPHI-SHIP', 'ORA-WPHI-SI']
REGIME, N_EPISODES, SEEDS = 'crowded', 16, [1619]
FIELD = 'window_resid'
# ★R2-2 가 결과 전에 기재한 규칙 (그대로 옮긴다, 내가 만들지 않았다)
PROMOTE_BAND = (1.00, 1.05)
W7_SHIP_RATIO = 1.5163       # 내가 W7 에서 옳은 기준으로 재도출한 값
CONTROL_BAND = (1.2, 2.0)    # 양성 대조 허용 (결과 전 확정)


def sign_test_one_sided(w, n):
    from math import comb
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def ratio_vs_ref(rows, arm, ref='ORA-WPHI-0STEP', field=FIELD):
    import statistics as st
    a = {r['episode']: r[field] for r in rows if r['cell'] == arm and field in r}
    b = {r['episode']: r[field] for r in rows if r['cell'] == ref and field in r}
    ks = sorted(set(a) & set(b)); n = len(ks)
    w = sum(1 for k in ks if a[k] > b[k])
    return {'arm': arm, 'n': n, 'above': w,
            'ratio': st.median([a[k] / b[k] for k in ks if b[k]]) if ks else float('nan'),
            'p_one_sided': sign_test_one_sided(w, n) if n else None}


def gates(rows):
    ok = [r for r in rows if 'error' not in r]
    out = [('G-0 런 수', len(ok) == len(ARMS) * N_EPISODES * len(SEEDS),
            f'{len(ok)}/{len(ARMS)*N_EPISODES*len(SEEDS)}'),
           ('G-0b 에러 0', len(rows) == len(ok), f'{len(rows)-len(ok)}')]
    # ★G-1 개입이 실제로 먹었나 — 모듈 상수를 워커에서 되읽는다 (의도 아님)
    bad = [r['job_id'] for r in ok
           if r.get('scale_invariant_readback') is not ARMS[r['cell']]['scale_invariant']]
    out.append(('G-1 척도불변 되읽기 일치', not bad, f'불일치 {len(bad)}'))
    bad2 = [r['job_id'] for r in ok
            if abs(r.get('lambda_l1_readback', -1)
                   - ARMS[r['cell']]['extra']['lambda_l1']) > 1e-12]
    out.append(('G-1b λ 되읽기 일치', not bad2, f'불일치 {len(bad2)}'))
    # ★G-2 0STEP 이 진짜 0 스텝인가 — W7 이 여기서 틀렸다
    z = [r for r in ok if r['cell'] == 'ORA-WPHI-0STEP']
    out.append(('G-2 0STEP nRMSE < 0.05 (W7 결함 재발 방지)',
                all(r.get('nRMSE', 9) < 0.05 for r in z),
                f'max {max((r.get("nRMSE", 9) for r in z), default=float("nan")):.4f}'))
    # G-3 양성 대조: SHIP 이 W7 값을 재현하는가
    v = ratio_vs_ref(ok, 'ORA-WPHI-SHIP')
    out.append(('G-3 양성대조 SHIP (ABORT)',
                v['above'] >= 13 and CONTROL_BAND[0] <= v['ratio'] <= CONTROL_BAND[1],
                f'{v["above"]}/{v["n"]} ratio {v["ratio"]:.3f} (W7 {W7_SHIP_RATIO}, 허용 {CONTROL_BAND})'))
    out.append(('G-4 에피소드별 창 크기 동일',
                len({(r['episode'], r.get('window_n_obs')) for r in ok})
                == len({r['episode'] for r in ok}), ''))
    return out


def label(si):
    """PRIMARY = ORA-WPHI-SI vs 0STEP, 창 안. R2-2 가 결과 전에 기재한 규칙."""
    if PROMOTE_BAND[0] <= si['ratio'] <= PROMOTE_BAND[1]:
        return ('PENALTY-BUDGET-CONFIRMED',
                '벌점을 [m^2] 로 만들자 창-안 벽이 사라진다 ⟹ 관측이 개입으로 승격.')
    if si['ratio'] < PROMOTE_BAND[0]:
        return ('OVERSHOOT',
                '개입이 진실보다 좋게 만든다 — 예산 논증 밖. R2-2 §3 재검토.')
    return ('PENALTY-BUDGET-REFUTED',
            f'척도불변으로도 벽이 남는다(ratio {si["ratio"]:.3f}) ⟹ R2-2 §3 철회, 원인은 차원이 아니다.')
