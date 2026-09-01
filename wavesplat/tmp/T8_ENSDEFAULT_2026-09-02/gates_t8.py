#!/usr/bin/env python3
"""`T8-ENSDEFAULT` 게이트·문턱·라벨 — **단일출처**. 러너·분석기는 여기를 호출만 한다.

사전등록 = `PREREG.md`. ⛔수를 문서에서 베끼지 마라 — 전부 재도출이다.

★질문: *"학습 경로 앙상블을 **배포 기본값으로 켜도 되는가**"* — 두 조건을 묻는다.
  ① 희소를 **해치지 않는가**(비열등성) ② 켜도 **해석적 대안보다 싼가**(비용)

⛔이 배치는 *"희소에서 이득이 있는가"* 를 묻지 않는다 --- `S-ENS` 가 `UNDECIDED` 를 냈고
같은 질문을 다시 하면 결과를 본 뒤 규칙을 바꾸는 것이 된다(`M-4` 선례).
"""
import glob
import json
import os
import statistics as st
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SENS = os.path.abspath(os.path.join(HERE, '..', 'S_ENS_2026-09-01'))
T6 = os.path.abspath(os.path.join(HERE, '..', 'T6_BUDGET_2026-09-02'))
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BATCH = 'T8-ENSDEFAULT'
FIELD = 'avg_nRMSE'

# ─────────────────────────────────────────────────────────────────────────────
# 1. 코퍼스 (PREREG §2) — ⛔여기가 단일출처다
# ─────────────────────────────────────────────────────────────────────────────
DATA = 'data_t8_sparse_2026-09-02'
SEEDS_SCENE = tuple(range(1704, 1720))            # 16 -> n=64
EPISODES = tuple(f'BF{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds'
                 for bf in (4, 5) for nc in (2, 5) for s in SEEDS_SCENE)
N_PAIRS = len(EPISODES)                           # 64

# 앵커 — `S-ENS` 코퍼스에서 4개. ⛔**PRIMARY 표본이 아니다**(G-6 이 검사).
ANCHOR_EPISODES = ('BF4_2comp_seed48_uniform_ra_rf_ds',
                   'BF4_2comp_seed49_uniform_ra_rf_ds',
                   'BF5_5comp_seed48_uniform_ra_rf_ds',
                   'BF5_5comp_seed49_uniform_ra_rf_ds')
ANCHOR_SEED = 1626                                # `S-ENS` 가 쓴 추정기 시드

EST_SEEDS = (42, 43, 44)                          # PREREG §3 — 에피소드 값은 이 셋의 중앙값

# ─────────────────────────────────────────────────────────────────────────────
# 2. 팔 — ⛔전 키 명시. 두 팔은 `learned_ensemble_k` **하나만** 다르다.
# ─────────────────────────────────────────────────────────────────────────────
_BASE = dict(
    init_method='auto', auto_switch_thresh=0,
    varpro_steps=400, learned_init_iters=0, lambda_l1=1e-4,
    k_snap_dk=None, adaptive_n=True, max_n=30,
    tracking_mode='w_phi', tracking_steps=10, tracking_lr=0.005, tracking_window=3,
    use_confidence=True, analytic_ensemble_k=3, real_domain_gate=True,
)
ARMS = {
    'M1': dict(K=1),          # 배포 기본값
    'M3': dict(K=3),
}
ARM_ORDER = ['M1', 'M3']
TREATMENT_KEY = 'learned_ensemble_k'


def arm_config(arm, seed):
    return dict(_BASE, learned_ensemble_k=ARMS[arm]['K'], seed=int(seed))


def expected_rows():
    return len(ARM_ORDER) * len(EPISODES) * len(EST_SEEDS)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PRIMARY — 비열등성 (PREREG §5.1). ⛔부호검정이 아니다.
# ─────────────────────────────────────────────────────────────────────────────
DELTA_PCT = 2.0          # 여유: 이보다 나빠지지 않으면 「실질적으로 해치지 않음」
P0 = 0.75                # H0: 「δ 안에 드는 비율」 <= 0.75
ALPHA = 0.025            # 단측


def _upper_tail(n, k, p):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def ni_threshold(n=N_PAIRS, p0=P0, alpha=ALPHA):
    """H0: p<=p0 을 기각하는 최소 K."""
    return min(k for k in range(n + 1) if _upper_tail(n, k, p0) <= alpha)


def ni_power(p_true, n=N_PAIRS):
    return _upper_tail(n, ni_threshold(n), p_true)


def ni_mde(n=N_PAIRS, power=0.80):
    """검정력 `power` 를 내는 최소 참 비율. ⛔설계를 동결하기 전에 이걸 봐라."""
    k = ni_threshold(n)
    p = P0
    while p <= 1.0 and _upper_tail(n, k, p) < power:
        p += 0.001
    return round(p, 3)


def thresholds():
    return dict(n=N_PAIRS, delta_pct=DELTA_PCT, p0=P0, alpha=ALPHA,
                K_MIN=ni_threshold(),
                power_at_090=round(ni_power(0.90), 3),
                power_at_095=round(ni_power(0.95), 3),
                mde_p80=ni_mde(),
                predicted_K='55-61 (PREREG §5.1, 결과 전 기록)')


# ─────────────────────────────────────────────────────────────────────────────
# 4. 비용·프레임 문턱 — ⛔**동결 `T6` 에서 읽는다**(손으로 적지 않는다)
# ─────────────────────────────────────────────────────────────────────────────
def t6_cells():
    p = os.path.join(T6, 'verdict_t6.json')
    if not os.path.exists(p):
        raise SystemExit(f'⛔동결 T6 판정이 없다: {p}')
    v = json.load(open(p))
    assert v['gates_pass'] == '10/10', f"⛔T6 게이트가 통과 상태가 아니다: {v['gates_pass']}"
    return v['cells']


def cost_threshold_s():
    """SEC-A 문턱 = `T6` 의 **해석적 희소 초기화 중앙값**.
    ★임의 상수가 아니다: 앙상블을 켠 학습 경로가 *대신 배포했을 대안*보다 비싸지면
      그 경로를 기본값으로 둘 논거가 사라진다."""
    return float(t6_cells()['CPU|ANA|sparse']['init_s_median'])


def frame_budget_ms(hz=10.0):
    return 1000.0 / hz


# ─────────────────────────────────────────────────────────────────────────────
# 5. 집계
# ─────────────────────────────────────────────────────────────────────────────
def per_episode(rows, arm, field=FIELD, episodes=None):
    """에피소드 -> 추정기 시드 **중앙값**. ⛔한 시드로 판정하지 않는다."""
    eps = set(episodes if episodes is not None else EPISODES)
    by = {}
    for r in rows:
        if r.get('arm') != arm or r.get('episode') not in eps:
            continue
        v = r.get(field)
        if v is None:
            continue
        by.setdefault(r['episode'], []).append(float(v))
    return {k: st.median(v) for k, v in by.items()}


def noninferiority(rows):
    a, b = per_episode(rows, 'M3'), per_episode(rows, 'M1')
    ks = sorted(set(a) & set(b))
    n = len(ks)
    if n == 0:
        return dict(n=0, K=None, rel_median_pct=None, p_one_sided=None, worst_pct=None)
    rel = {k: (a[k] - b[k]) / b[k] * 100 for k in ks}
    K = sum(1 for k in ks if rel[k] <= DELTA_PCT)
    return dict(n=n, K=K, delta_pct=DELTA_PCT,
                rel_median_pct=st.median(rel.values()),
                worst_pct=max(rel.values()),
                better=sum(1 for k in ks if rel[k] < 0),
                p_one_sided=_upper_tail(n, K, P0))


def cost(rows):
    out = {}
    for arm in ARM_ORDER:
        v = [r['init_time_s'] for r in rows
             if r.get('arm') == arm and r.get('episode') in set(EPISODES)
             and isinstance(r.get('init_time_s'), (int, float))]
        out[arm] = dict(n=len(v), init_s_median=st.median(v) if v else None)
    thr = cost_threshold_s()
    m3 = out['M3']['init_s_median']
    out['threshold_s'] = thr
    out['pass'] = (m3 is not None and m3 < thr)
    return out


def frames(timing_rows):
    out = {}
    for arm in ARM_ORDER:
        fms = [x for r in timing_rows if r.get('arm') == arm
               for x in (r.get('frame_ms') or [])]
        out[arm] = dict(n=len(fms), frame_ms_median=st.median(fms) if fms else None)
    a, b = out['M3']['frame_ms_median'], out['M1']['frame_ms_median']
    out['budget_ms'] = frame_budget_ms()
    out['pass'] = (a is not None and b is not None
                   and abs(a - b) / b <= 0.10 and a < 0.10 * frame_budget_ms())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 6. 라벨 — 결과 전 확정 (PREREG §5.4) · 공역 전건
# ─────────────────────────────────────────────────────────────────────────────
LABELS = {
    (True, True): 'ENS-DEFAULT-READY',
    (True, False): 'ENS-DEFAULT-COSTLY',
    (False, True): 'ENS-DEFAULT-HARMS-SPARSE',
    (False, False): 'ENS-DEFAULT-NO',
}
READING = {
    'ENS-DEFAULT-READY': ('전환의 **측정 조건이 전부 충족**됐다. ⛔그래도 전환은 '
                          '**사용자 결정**이다 — 이 배치는 코드를 바꾸지 않는다.'),
    'ENS-DEFAULT-COSTLY': ('⛔전환 권고 아님 — 해치지는 않으나 **초기화가 해석적 대안만큼 '
                           '비싸다**. 앙상블을 켤 논거(더 싸면서 더 정확)가 무너진다.'),
    'ENS-DEFAULT-HARMS-SPARSE': ('⛔전환 권고 아님 — 희소 무해가 **반증**됐다.'),
    'ENS-DEFAULT-NO': '⛔전환 권고 아님(둘 다 실패).',
    'ENS-DEFAULT-UNDECIDED': ('문턱에 도달하지 못했다. ⛔*"차이 없다"* 금지 — '
                              'MDE 병기 의무.'),
}


def label_for(ni, cost_res):
    t = thresholds()
    if ni['K'] is None:
        return None, 'NOROWS'
    if ni['K'] < t['K_MIN']:
        # 문턱 미달 = 비열등성 미확립. ⚠반증(해친다)과 미결을 구별한다.
        harmed = ni['n'] - ni['K']
        if harmed > ni['n'] * (1 - P0) * 2:      # 해로운 비율이 H0 의 두 배를 넘는다
            return f'{BATCH} :: {LABELS[(False, cost_res["pass"])]}', 'REVERSE'
        return f'{BATCH} :: ENS-DEFAULT-UNDECIDED', 'UNDECIDED'
    return f'{BATCH} :: {LABELS[(True, cost_res["pass"])]}', 'CONFIRMATORY'


DISCLOSURE = [
    '⚠blind 아님(부분) — S-ENS 의 희소 중앙값(-1.56%)과 승수(35/64)를 알고 있다. '
    '★단 코퍼스가 신선하고 통계량이 다르다(꼬리 개수 vs 승수).',
    f'⚠여유 δ={DELTA_PCT}% 는 내가 골랐다 — 정당화는 PREREG §8.2 가 전부다.',
    '⛔기본값 전환 권고가 자동으로 따라오지 않는다 — 라벨과 무관하게 **사용자 결정**이다.',
    '⛔실측 판정 갱신 아님 — 출하 라우터는 실측을 bank_mmse 로 보낸다(합성 트래픽만 닿는다).',
    '⛔혼잡 이득을 재측정하지 않았다 — S-ENS 55/64 를 인용할 뿐이다.',
    '⚠단일 층(희소)·단일 센서 모형(uniform)·단일 참값 생성기(선형).',
    '⚠비용 문턱은 한 기계(AMD Ryzen 7 7800X3D)의 값이다.',
]

# ─────────────────────────────────────────────────────────────────────────────
# 7. 게이트
# ─────────────────────────────────────────────────────────────────────────────
BLOCKING = ('G-1', 'G-2', 'G-3', 'G-4', 'G-5', 'G-6', 'G-7', 'G-8', 'G-ANCHOR')


def _anchor_sens():
    """동결 `S-ENS LEARN-M1` 의 episode -> avg_nRMSE."""
    out = {}
    for p in glob.glob(os.path.join(SENS, 'results_sens_*.jsonl')):
        for ln in open(p):
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r.get('arm') == 'LEARN-M1':
                out[r['episode']] = r.get('avg_nRMSE')
    return out


def gates(rows, anchor_rows=(), timing_rows=()):
    G = []

    def A(name, ok, detail=''):
        G.append((name, bool(ok), detail))

    ok = [r for r in rows if 'error' not in r]
    have = bool(ok)
    prim = [r for r in ok if r.get('episode') in set(EPISODES)]

    A('G-1 스레드=1 · 단일 PID',
      have and all(r.get('threads') == 1 for r in ok) and len({r.get('pid') for r in ok}) == 1,
      f"threads={sorted({r.get('threads') for r in ok})} pids={len({r.get('pid') for r in ok})}")

    bad = [f"{r.get('arm')}/{r.get('episode')}:{r.get('ensemble_k')}"
           for r in ok if r.get('ensemble_k') != ARMS.get(r.get('arm'), {}).get('K')]
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
      f'rows={len(prim)}/{expected_rows()} cov={sorted(set(cov.values()))}')

    # ⛔[2026-09-02] **자가진단이 이 게이트의 설계 결함을 잡았다**: 초판은 `prim`
    #   (= EPISODES 로 이미 걸러낸 행) 위에서 누출을 찾았다 ⟹ 앵커 행이 결과 파일에
    #   섞여도 필터가 먼저 지워서 **게이트가 영원히 발화하지 않는다**.
    #   ★`blocked_by` 가 항상 비었던 `T5`/`T7` 결함과 **같은 부류**다(장식 게이트).
    #   ⟹ **거른 뒤가 아니라 거르기 전(`ok`)** 을 본다.
    seen_eps = {r['episode'] for r in ok}
    scene = {int(e.split('seed')[1].split('_')[0]) for e in seen_eps if 'seed' in e}
    stray = seen_eps - set(EPISODES)
    A('G-6 결과 파일이 신선 코퍼스만 담는다 (시드 >= 1704 · 앵커 0)',
      bool(scene) and min(scene) >= 1704 and not stray,
      f'min_seed={min(scene) if scene else None} stray={sorted(stray)[:3]}')

    ca, cb = arm_config('M1', EST_SEEDS[0]), arm_config('M3', EST_SEEDS[0])
    diff = {k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k)}
    A(f'G-7 두 팔이 {TREATMENT_KEY} 하나만 다르다',
      diff == {TREATMENT_KEY} and bool(ca), f'diff={sorted(diff)}')

    vals = [r.get(FIELD) for r in prim]
    A('G-8 값 유효 (유한·양수)',
      bool(vals) and all(isinstance(v, (int, float)) and v == v and v > 0 for v in vals))

    anc = _anchor_sens()
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
    A('G-ANCHOR M1 == 동결 S-ENS LEARN-M1 (비트)',
      a_n > 0 and not a_bad, f'compared={a_n} bad={a_bad[:2]}')

    # ── 범위 축소(⛔차단 아님)
    w = {a: [r.get('wall_s') for r in prim if r.get('arm') == a and r.get('wall_s')]
         for a in ARM_ORDER}
    A('H-COST 단가 공시', True,
      ' · '.join(f'{a} 중앙 {st.median(v):.1f}s' for a, v in w.items() if v))
    A('H-TIMING 프레임 측정 행', True, f'{len(timing_rows)} 행')
    return G


def gate_id(name):
    return name.split()[0]


def blocked_by(G):
    """⛔`T5`·`T7` 에서 두 번 난 결함: 이름 전체를 비교해 **항상 빈 리스트**였다."""
    return [gate_id(n) for n, o, _ in G if not o and gate_id(n) in BLOCKING]
