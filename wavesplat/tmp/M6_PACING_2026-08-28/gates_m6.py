"""M-6 게이트 — **페이싱 견고성 용량-반응** (판정 규약, 결과 전 동결 2026-08-28).

⛔이 파일은 결과 산출 전에 동결됐다. 결과가 나온 뒤 이 파일을 고치면 그 판정은 무효다.

# 무엇을 재는가 (정확도 배치가 아니다 — 그 사슬은 B-3/B-9/B-9b 로 닫혔다)
비동기 재적합(`bank_refit_async`)의 효과는 **호출자의 벽시계 페이싱**에 의존한다:
  · 재적합 트리거는 **자료 시간**이다      — sequential_update.py:1292-1294
  · 적합 완료는 **벽시계**다(중앙 ~1.7 s)  — B-9 실측, 주석 :1191-1195
  · 착지는 완료 후 **다음 update()** 에서만 — :1305 → :1228-1232 (`is_alive()` 면 no-op)
  · 진행 중 kick 은 **버려진다**(큐 없음)   — :1188-1190
⟹ 재생 배속 r 에서 적합은 자료시간으로 **r × t_fit 초 늦게** 착지하고,
   지평(마지막 채점 lead)을 넘기면 **전면 no-op** 이 된다(B-9 의 +0.00%).

# 용량-반응 예측 (결과 전 기재, 반증 가능)
  P-a  첫 착지 lead ≈ TAU + r·t_fit  (r 에 단조증가; t_fit 은 **런에서 실측**한 값을 쓴다
       — 1.7 을 손복사하지 않는다. 손복사 기준 상수는 이 저장소의 반복 결함 부류다.)
  P-b  전면 no-op 경계 r_dead ≈ (max_lead − TAU) / t_fit  (t_fit 1.4–2.0 이면 ≈ 4.5–6.4)
  P-c  효과(vs FROZEN 짝 이김 수)는 r 에 단조 감소, r ∈ {0.5, 1} 은 생존(B-9b 33/36),
       r ≥ r_dead 는 사망(적용 0 ⟹ FROZEN 과 같은 장으로 채점).
  P-d  :1197-1208 의 RuntimeWarning 은 **전면 no-op 에서만** 발화한다 —
       부분 열화 대역(착지가 늦지만 0 은 아닌 r)은 **무경고**다. 이것이 기본값 결정의 위험 대역이다.

# ⛔널이면 무엇을 쓰는가 (결과 전 기재)
  · 전 배속 생존(∞ 포함)  ⟹ B-9 와 모순 — 하네스 조사가 먼저다. 어느 쪽이든 "조용한 no-op
    위험" 서술을 철회하게 되므로 기본값 결정의 입력이 바뀐다.
  · 전 배속 사망(r=1 포함) ⟹ B-9b 재현 실패 = 배포 사슬 고리 ④ 재검 — 플립 논의 자체가 정지된다.
  · 예상대로 무릎 발견     ⟹ r† 가 **페이싱 계약**이 된다: "기본값 ON 은 r ≤ r† 호출자에게만
    유효하고, 더 빠른 호출자는 (부분 대역에서) 무경고로 효과를 잃는다."
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
B2 = os.path.abspath(os.path.join(HERE, '..', 'B2_REAL_2026-08-23'))
B3 = os.path.abspath(os.path.join(HERE, '..', 'B3_REFIT_2026-08-23'))
for p in (HERE, B2, B3):
    if p not in sys.path:
        sys.path.insert(0, p)
from gates_b3 import RECORDS, WINDOWS, SEED, NOISE, N_PAIRS, sign_test_one_sided, key, paired  # noqa

# ── 팔 (⛔빈 dict 금지 — F-7. 2026-08-22 배포 3키는 **명시 핀**: dict(DEFAULT_CONFIG)
#    런타임 스냅샷이 미래 기본값을 흡수하는 부류(S13·HOSGATE)를 막는다) ──────────────
BASE_PINS = {'varpro_steps': 400, 'learned_init_iters': 0, 'auto_switch_thresh': 0}
TAU = 5.0                                   # 후보 구성의 재적합 주기 (자료 초)
ASYNC_CFG = {'bank_refit_every_s': TAU, 'bank_refit_async': True}    # = recommended_real_refit_config()
FROZEN_CFG = {'bank_refit_every_s': None, 'bank_refit_async': False}  # = 출하 기본값 (명시!)

PACES = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, float('inf')]   # 재생 배속 r (∞ = 무페이싱 = B-9 조건)
ARMS = ({f'ASYNC5_P{p:g}': dict(ASYNC_CFG) for p in PACES}
        | {'FROZEN_P1': dict(FROZEN_CFG), 'FROZEN_PInf': dict(FROZEN_CFG)})
PACE_OF = ({f'ASYNC5_P{p:g}': p for p in PACES}
           | {'FROZEN_P1': 1.0, 'FROZEN_PInf': float('inf')})
ARM_ORDER = list(ARMS)
FIELD = 'fc_mean'            # PRIMARY 점수 = lead 2–14 평균 (B-3 사슬과 동일, 결과 전 확정)
LATE_FIELD = 'late_mean'     # SECONDARY = lead 8–14 평균 (무릎 근방 분해능용; 승격 금지)
PAIR_BASE = 'FROZEN_P1'      # 짝 기준 팔 (NC-1 통과가 전제)
NOMINAL, CONFIRM = 24, 26    # 사슬(B-3/B-9b)과 같은 문턱
N_TESTS = len(PACES)         # Holm 가족 = 배속 8개
FAMILY_ALPHA = 0.05
T_FIT_BAND_RUN = (0.8, 3.5)      # G-6a 런별 중앙 t_fit 허용 대역 (초) — 경합 감시
T_FIT_BAND_BATCH = (1.0, 2.8)    # G-6b 전 런 중앙 대역 (B-6 실측 1.35–2.53 포함)


def holm(pvals):
    """Holm 단계하강: {이름: p} → {이름: 통과여부}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    out, m = {}, len(items)
    alive = True
    for i, (name, p) in enumerate(items):
        if alive and p <= FAMILY_ALPHA / (m - i):
            out[name] = True
        else:
            alive = False
            out[name] = False
    return out


def gates(rows):
    ok = [r for r in rows if 'error' not in r]
    by = {a: [r for r in ok if r['cell'] == a] for a in ARMS}
    asy = [r for r in ok if r['cell'].startswith('ASYNC5_')]
    out = [('G-0 런 수 전건', all(len(by[a]) == N_PAIRS for a in ARMS),
            f"{ {a: len(by[a]) for a in ARMS if len(by[a]) != N_PAIRS} or 'all 36' }"),
           ('G-0b 에러 0', len(rows) == len(ok), f'{len(rows) - len(ok)}')]
    # ★NC-1 — FROZEN 은 페이싱과 무관해야 한다(비트 동일). 깨지면 배치 전체 해석 불가.
    f1 = {key(r): r[FIELD] for r in by['FROZEN_P1']}
    fi = {key(r): r[FIELD] for r in by['FROZEN_PInf']}
    same = sum(1 for k in f1 if k in fi and f1[k] == fi[k])
    out.append(('★NC-1 FROZEN 페이싱 불변(비트 동일)', same == N_PAIRS, f'{same}/{N_PAIRS}'))
    # ★G-2 — 트리거는 자료시간 ⟹ kick 수는 배속과 무관하게 결정적이어야 한다
    out.append(('★G-2 ASYNC kick 수 = 자료시간 결정값(전 배속 동일)',
                bool(asy) and len({r['async_kicks'] for r in asy}) == 1
                and all(r['async_kicks'] >= 2 for r in asy),
                f"kicks {sorted({r.get('async_kicks') for r in asy})}"))
    # ★G-3 — 페이싱이 실제로 걸렸는가 (배속별 벽시계 하한/상한)
    bad = 0
    for r in ok:
        p, w = PACE_OF[r['cell']], r['wall_s']
        lead_span = max(float(k) for k in r['per_lead'])
        if p == float('inf'):
            bad += (w > 12.0)                 # 무페이싱인데 실시간급이면 실패
        else:
            bad += (w < 0.9 * lead_span / p)  # 페이싱인데 그만큼 안 걸렸으면 실패
    out.append(('★G-3 페이싱 체결(배속별 벽시계 대역)', bad == 0, f'위반 {bad}'))
    # G-4 — 되읽기 = 의도 (객체 되읽기, 의도표 아님 — S13b 의 교훈)
    mism = [r['cell'] for r in ok if not (
        r['readback'].get('bank_refit_async') == ARMS[r['cell']]['bank_refit_async']
        and r['readback'].get('bank_refit_every_s') == ARMS[r['cell']]['bank_refit_every_s']
        and all(r['readback'].get(k) == v for k, v in BASE_PINS.items()))]
    out.append(('G-4 구성 되읽기 일치(3핀 포함)', not mism, f'{sorted(set(mism))[:3]}'))
    out.append(('G-5 36 창 전건·팔별', all(len({key(r) for r in by[a]}) == N_PAIRS for a in ARMS), ''))
    # G-6 — 경합 감시: t_fit 실측 대역 (무릎 예측이 이 값에서 유도되므로 게이트다)
    tf_runs = [r['t_fit_median_s'] for r in asy if r.get('t_fit_median_s')]
    import statistics as st
    med = st.median(tf_runs) if tf_runs else float('nan')
    out.append(('G-6a t_fit 런별 대역', all(T_FIT_BAND_RUN[0] <= t <= T_FIT_BAND_RUN[1] for t in tf_runs),
                f'이탈 {sum(1 for t in tf_runs if not T_FIT_BAND_RUN[0] <= t <= T_FIT_BAND_RUN[1])}'))
    out.append(('G-6b t_fit 전런 중앙 대역', T_FIT_BAND_BATCH[0] <= med <= T_FIT_BAND_BATCH[1],
                f'{med:.2f} s'))
    # G-7 — 팔별 필드 존재 (S-2a 부류: 기준선 셸 0행 재발 방지)
    need = [FIELD, LATE_FIELD, 'per_lead', 'applied_pre_wait', 'first_apply_lead',
            'noop_warning_fired', 'wall_s', 'readback', 'async_kicks', 'async_skipped',
            'health_pre_wait', 'loadavg1']
    miss = [k for r in ok for k in need if k not in r]
    out.append(('G-7 필드 존재 전건', not miss, f'{sorted(set(miss))[:4]}'))
    # ★G-8 — 경보 정합: RuntimeWarning ⟺ 채점 중 적용 0 (진단 장치 자체를 검증한다 —
    #   자기진단이 통과해도 판정 불변이 아니라는 부류가 두 번 나왔다)
    inc = [r['cell'] for r in asy
           if bool(r['noop_warning_fired']) != (r['applied_pre_wait'] == 0 and r['async_kicks'] >= 2)]
    out.append(('★G-8 no-op 경보 ⟺ 적용 0', not inc, f'{sorted(set(inc))[:3]}'))
    return out


def label(gate_rows, alive_by_pace, mech_monotone, rdead_hat):
    g_ok = all(ok for _, ok, _ in gate_rows)
    ncs = [ok for name, ok, _ in gate_rows if name.startswith('★NC-1')]
    if not ncs[0]:
        return ('VACUOUS-PACING-LEAK',
                '⛔FROZEN 이 페이싱에 따라 달라졌다 — 비동기 기제 밖의 누수. 전체 해석 불가.')
    alive = [p for p in PACES if alive_by_pace.get(p)]
    dead = [p for p in PACES if not alive_by_pace.get(p)]
    if len(alive) == len(PACES):
        lab = 'PACING-INDEPENDENT (CONTRADICTS-B9 — investigate harness)'
    elif not alive:
        lab = 'EFFECT-DEAD-EVERYWHERE (B-9b REPLICATION FAILURE — no flip discussion)'
    else:
        r_alive = max(alive)
        lab = f'EFFECT-ALIVE-UPTO-{r_alive:g}x'
        if dead and min(dead) < r_alive:
            lab += ' (NON-MONOTONE — read per-pace table, do not summarize)'
    tag = 'CONFIRMATORY' if g_ok and mech_monotone else 'EXPLORATORY'
    return (f'{lab} [{tag}]',
            f'생존 배속 {alive} · 사망 배속 {dead} · 실측 유도 r_dead≈{rdead_hat:.1f} · '
            f'기제 단조 {mech_monotone} · 게이트 {"전건" if g_ok else "⚠부분"}')
