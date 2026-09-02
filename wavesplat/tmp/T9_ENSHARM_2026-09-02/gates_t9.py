#!/usr/bin/env python3
"""`T9-ENSHARM` 게이트·문턱·라벨 — **단일출처**.

★질문: *"앙상블이 희소 층을 **해치는가**"* — `T8` 과 **같은 질문, 다른 자**.
`T8` 은 **상대** 여유(2%)로 물었고 문턱에 미달했다. 이 배치는 **절대** 여유로 묻는다.

⛔`T8` 의 자료 위에서 다시 자르지 않는다 — 그 꼬리를 이미 보았다(사후 완화 금지).
⟹ 코퍼스가 다르다(시드 1720-1735).
⛔팔 정의·집계 규약은 `gates_t8` 을 **임포트**한다. 재선언하면 두 배치가 조용히 갈린다.
"""
import glob
import json
import os
import statistics as st
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
T8 = os.path.abspath(os.path.join(HERE, '..', 'T8_ENSDEFAULT_2026-09-02'))
for _p in (HERE, T8, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gates_t8 as T8G  # noqa: E402

BATCH = 'T9-ENSHARM'
FIELD = T8G.FIELD

# ── 코퍼스 (PREREG §1) ───────────────────────────────────────────────────────
DATA = 'data_t9_sparse_2026-09-02'
SEEDS_SCENE = tuple(range(1720, 1736))
EPISODES = tuple(f'BF{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds'
                 for bf in (4, 5) for nc in (2, 5) for s in SEEDS_SCENE)
N_PAIRS = len(EPISODES)

ANCHOR_EPISODES = T8G.ANCHOR_EPISODES
ANCHOR_SEED = T8G.ANCHOR_SEED
EST_SEEDS = T8G.EST_SEEDS
ARMS, ARM_ORDER, TREATMENT_KEY = T8G.ARMS, T8G.ARM_ORDER, T8G.TREATMENT_KEY
arm_config = T8G.arm_config


def expected_rows():
    return len(ARM_ORDER) * len(EPISODES) * len(EST_SEEDS)


# ── PRIMARY: **절대** 비열등성 (PREREG §3) ──────────────────────────────────
# ★여유는 자료가 아니라 **센서**에서 왔다: nRMSE = RMSE/Hs 이고 이 코퍼스는 Hs 1.2·2.0 m
#   ⟹ δ=0.005 는 표면 고도 6·10 mm 이고, 센서 잡음 σ=0.02 m(=20 mm)보다 **작다**.
#   ⛔`T8` 의 꼬리(0.0035-0.0119)를 보고 고른 값이 아니다.
# ★[2026-09-02] **개정 1**(결과 0행): 여유를 하나 고르지 않고 **격자 전체에서 곡선**을
#   보고한다. 사유 = `T8` 감사가 실측으로 보였듯 여유가 판정을 가르고(0.001↔0.002에서
#   뒤집힌다) 나는 그 곡선을 볼 수 있는 자료를 이미 갖고 있었다.
#   ⟹ 판정 = `K(δ) >= 문턱` 인 **가장 작은 δ**. 출력이 참/거짓이 아니라 **수**다.
DELTA_GRID = (0.001, 0.002, 0.005, 0.010)
# ⚠센서 눈금(해석용, ⛔판정 문턱 아님): nRMSE=RMSE/Hs, Hs 1.2·2.0 m ⟹ 0.005 는 표면
#   고도 6·10 mm 이고 센서 잡음 σ=0.02 m(20 mm)보다 작다.
DELTA_SENSOR = 0.005
DELTA_ABS = DELTA_SENSOR   # 하위 호환(보고용). ⛔판정에 쓰지 않는다.
P0 = T8G.P0          # 0.75 — `T8` 과 같은 틀
ALPHA = T8G.ALPHA    # 0.025


def _upper_tail(n, k, p):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def ni_threshold(n=N_PAIRS, p0=P0, alpha=ALPHA):
    return min(k for k in range(n + 1) if _upper_tail(n, k, p0) <= alpha)


def ni_power(p_true, n=N_PAIRS):
    return _upper_tail(n, ni_threshold(n), p_true)


def ni_mde(n=N_PAIRS, power=0.80):
    k, p = ni_threshold(n), P0
    while p <= 1.0 and _upper_tail(n, k, p) < power:
        p += 0.001
    return round(p, 3)


def thresholds():
    return dict(n=N_PAIRS, delta_grid=list(DELTA_GRID),
                delta_sensor=DELTA_SENSOR, p0=P0, alpha=ALPHA,
                K_MIN=ni_threshold(), NOT_EARN=N_PAIRS - ni_threshold(),
                power_at_090=round(ni_power(0.90), 3),
                power_at_0906=round(ni_power(0.906), 3),
                power_at_095=round(ni_power(0.95), 3),
                mde_p80=ni_mde(),
                predicted='delta* = 0.002 또는 0.005 (AMENDMENT_1 §2.3, 결과 전 기록)')


# ── 집계 ────────────────────────────────────────────────────────────────────
def per_episode(rows, arm, field=FIELD):
    eps = set(EPISODES)
    by = {}
    for r in rows:
        if r.get('arm') != arm or r.get('episode') not in eps:
            continue
        if r.get(field) is not None:
            by.setdefault(r['episode'], []).append(float(r[field]))
    return {k: st.median(v) for k, v in by.items()}


def noninferiority(rows):
    a, b = per_episode(rows, 'M3'), per_episode(rows, 'M1')
    ks = sorted(set(a) & set(b))
    n = len(ks)
    if n == 0:
        return dict(n=0, K=None, curve=None, delta_star=None)
    d = {k: a[k] - b[k] for k in ks}
    thr = ni_threshold(n)
    curve = {f'{dd:g}': sum(1 for k in ks if d[k] <= dd) for dd in DELTA_GRID}
    passing = [dd for dd in DELTA_GRID if curve[f'{dd:g}'] >= thr]
    star = min(passing) if passing else None
    return dict(n=n, curve=curve, delta_star=star, K_min=thr,
                K=curve[f'{DELTA_SENSOR:g}'],          # 센서 눈금에서의 값(보고용)
                delta_sensor=DELTA_SENSOR,
                delta_median=st.median(d.values()),
                worst_abs=max(d.values()),
                better=sum(1 for k in ks if d[k] < 0),
                rel_median_pct=st.median([(a[k] - b[k]) / b[k] * 100 for k in ks]),
                p_at_star=(_upper_tail(n, curve[f'{star:g}'], P0)
                           if star is not None else None))


def cost(rows, timing_rows):
    """⛔라벨에 들어가지 않는다(PREREG §4) — **값만 보고**한다."""
    out = {}
    for arm in ARM_ORDER:
        v = [r['init_time_s'] for r in rows if r.get('arm') == arm
             and isinstance(r.get('init_time_s'), (int, float))]
        f = [x for r in timing_rows if r.get('arm') == arm
             for x in (r.get('frame_ms') or [])]
        out[arm] = dict(init_s_median=st.median(v) if v else None,
                        frame_ms_median=st.median(f) if f else None,
                        n_init=len(v), n_frame=len(f))
    out['note'] = ('⛔문턱 없음 — 사용자가 예산을 100 ms 로 밝혔고 그 판단은 '
                   '측정이 아니라 결정이다(PREREG §4).')
    return out


# ── 라벨 (PREREG §5) ────────────────────────────────────────────────────────
READING = {
    'ENS-NO-HARM-SPARSE': ('절대 자로는 해치지 않는다 ⟹ ★`T8` 의 미달은 **자의 문제**였다. '
                           '⛔`T8` 을 뒤집지 않는다 — 다른 자·다른 코퍼스. 쓸 수 있는 문장은 '
                           '*"상대 자로는 미달, 절대 자로는 통과"* 이고 그 차이 자체가 측정이다.'),
    'ENS-HARMS-SPARSE': '⛔앙상블이 절대 자로도 해친다 — 반증.',
    'ENS-HARM-UNDECIDED': '⛔*"차이 없다"* 금지 — MDE 병기 의무.',
}


def label_for(ni):
    """⛔개정 1: 판정은 **가장 작은 통과 여유** `δ*` 다(참/거짓이 아니다).

    ⚠`REVERSE` 는 격자의 **가장 느슨한** 여유에서조차 `NOT_EARN` 이하일 때만 낸다 ---
      그 조건은 사전등록 §5 의 *"≤ 16"* 을 격자로 옮긴 것이고, 이 규칙이 코드에만
      있지 않도록 `AMENDMENT_1 §2.1` 에 함께 적었다(`T8` 감사 지적 #1 의 재발 방지).
    """
    if ni.get('curve') is None:
        return None, 'NOROWS'
    t = thresholds()
    if ni['delta_star'] is not None:
        return (f"{BATCH} :: ENS-NO-HARM-SPARSE (CONFIRMATORY, "
                f"delta*={ni['delta_star']:g})", 'CONFIRMATORY')
    loosest = ni['curve'][f'{max(DELTA_GRID):g}']
    if loosest <= t['NOT_EARN']:
        return f'{BATCH} :: ENS-HARMS-SPARSE', 'REVERSE'
    return f'{BATCH} :: ENS-HARM-UNDECIDED', 'UNDECIDED'


DISCLOSURE = [
    '⚠blind 아님 — `T8` 의 방향·크기(상대 중앙 -8.14%, 40/64, 꼬리 다섯 셀)를 알고 있다.',
    '⚠여유의 **기각된 대안**(시드 스프레드 0.0163)은 `T8` 의 M1 행에서 계산했다 — '
    '기준선 팔의 성질이지 대조가 아니지만, 본 것은 본 것이다.',
    '⛔`T8` 을 갱신·철회하지 않는다 — 다른 자, 다른 코퍼스.',
    '⛔혼잡 층을 재지 않는다 — `S-ENS` 의 55/64 는 그대로다.',
    '⛔비용에 문턱이 없다(PREREG §4) ⟹ 비용으로 어떤 결론도 내지 않는다.',
    '⛔기본값 전환 권고 아님 — 전환은 사용자 결정이다.',
    '⚠단일 층(희소)·단일 센서 모형·단일 참값 생성기·한 기계.',
]

# ── 게이트 ──────────────────────────────────────────────────────────────────
BLOCKING = ('G-1', 'G-2', 'G-3', 'G-4', 'G-5', 'G-6', 'G-7', 'G-8', 'G-9', 'G-10',
            'G-ANCHOR')


def gates(rows, anchor_rows=(), timing_rows=()):
    G = []

    def A(name, ok, detail=''):
        G.append((name, bool(ok), detail))

    ok = [r for r in rows if 'error' not in r]
    have = bool(ok)
    prim = [r for r in ok if r.get('episode') in set(EPISODES)]

    A('G-1 스레드=1 · 단일 PID',
      have and all(r.get('threads') == 1 for r in ok)
      and len({r.get('pid') for r in ok}) == 1,
      f"pids={len({r.get('pid') for r in ok})}")

    bad = [f"{r.get('arm')}:{r.get('ensemble_k')}" for r in ok
           if r.get('ensemble_k') != ARMS.get(r.get('arm'), {}).get('K')]
    A('G-2 패키지 ensemble_k 되읽기 == 팔의 기대값', have and not bad, ';'.join(bad[:3]))

    A('G-3 라우터가 학습으로 해소 전건',
      have and all(r.get('auto_selected') == 'learned' for r in ok),
      f"{sorted({r.get('auto_selected') for r in ok})}")

    A('G-4 출하 기본값 트립와이어',
      have and all(r.get('deploy_pin_ok') is True for r in ok))

    cov = {(a, s): sum(1 for r in prim if r.get('arm') == a and r.get('seed') == s)
           for a in ARM_ORDER for s in EST_SEEDS}
    A('G-5 행 완비 (64 x 2팔 x 3시드)',
      have and len(prim) == expected_rows() and all(v == N_PAIRS for v in cov.values()),
      f'rows={len(prim)}/{expected_rows()}')

    seen = {r['episode'] for r in ok}
    scene = {int(e.split('seed')[1].split('_')[0]) for e in seen if 'seed' in e}
    stray = seen - set(EPISODES)
    A('G-6 결과 파일이 신선 코퍼스만 담는다 (시드 >= 1720)',
      bool(scene) and min(scene) >= 1720 and not stray,
      f'min_seed={min(scene) if scene else None} stray={sorted(stray)[:3]}')

    ca, cb = arm_config('M1', EST_SEEDS[0]), arm_config('M3', EST_SEEDS[0])
    diff = {k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k)}
    A(f'G-7 두 팔이 {TREATMENT_KEY} 하나만 다르다', diff == {TREATMENT_KEY} and bool(ca),
      f'diff={sorted(diff)}')

    vals = [r.get(FIELD) for r in prim]
    A('G-8 값 유효 (유한·양수)',
      bool(vals) and all(isinstance(v, (int, float)) and v == v and v > 0 for v in vals))

    # ★신설 — `T8` 코퍼스와 서로소인가 (사후 완화 방지의 기계적 증명).
    # ⚠이것은 **행이 아니라 설계**를 본다 — 지금 자료로는 절대 실패하지 않는다.
    #   막는 대상은 **미래의 편집**이다(누가 `SEEDS_SCENE` 를 T8 쪽으로 옮기면 죽는다).
    #   자료 쪽 누출은 `G-6`(stray) 이 잡고, 자가진단이 그것을 위조로 확인한다.
    overlap = set(EPISODES) & set(T8G.EPISODES)
    A('G-9 T8 코퍼스와 서로소 (설계 검사)', not overlap, f'overlap={sorted(overlap)[:3]}')

    anc = T8G._anchor_sens()
    a_n, a_bad = 0, []
    for r in anchor_rows:
        if 'error' in r or r.get('arm') != 'M1':
            continue
        ref = anc.get(r.get('episode'))
        if ref is None:
            continue
        a_n += 1
        if ref != r.get(FIELD):
            a_bad.append(f"{r['episode']}: {r.get(FIELD)} != {ref}")
    A('G-ANCHOR M1 == 동결 S-ENS LEARN-M1 (비트)', a_n > 0 and not a_bad,
      f'compared={a_n} bad={a_bad[:2]}')

    # ★[2026-09-02] 개정 1 · `T8` 감사 지적 #4 — **결정을 만들 수 있는 축의 측정에
    #   차단 게이트가 하나도 없었다**(`H-TIMING` 이 `True` 하드코딩이었다).
    #   ⚠`T9` 에서 비용은 라벨에 안 들어가지만, *"측정이 유효한가"* 는 별개다.
    #   ⛔값이 결정을 만들지 않는다는 것이 **측정을 검사하지 않아도 된다는 뜻이 아니다.**
    tk = [r for r in timing_rows if 'error' not in r]
    per_arm = {a: sum(1 for r in tk if r.get('arm') == a) for a in ARM_ORDER}
    fms = [x for r in tk for x in (r.get('frame_ms') or [])]
    A('G-10 계측 행 유효 (스레드=1 · 학습 경로 · 팔 균형 · 값 유한)',
      bool(tk)
      and all(r.get('threads') == 1 for r in tk)
      and all(r.get('route_effective') == 'learned' for r in tk)
      and len(set(per_arm.values())) == 1 and min(per_arm.values()) > 0
      and bool(fms) and all(isinstance(v, (int, float)) and v == v and v > 0
                            for v in fms),
      f'rows={len(tk)} per_arm={per_arm} frames={len(fms)}')

    w = {a: [r.get('wall_s') for r in prim if r.get('arm') == a and r.get('wall_s')]
         for a in ARM_ORDER}
    A('H-COST 단가 공시', True,
      ' · '.join(f'{a} 중앙 {st.median(v):.1f}s' for a, v in w.items() if v))
    A('H-TIMING 프레임 측정 행', True, f'{len(timing_rows)} 행')
    return G


def gate_id(name):
    return name.split()[0]


def blocked_by(G):
    return [gate_id(n) for n, o, _ in G if not o and gate_id(n) in BLOCKING]
