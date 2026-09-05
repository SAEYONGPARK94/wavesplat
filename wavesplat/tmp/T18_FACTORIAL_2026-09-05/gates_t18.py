"""`T18-FACTORIAL` 단일출처 — **사전 × 파수 양자화 2×2 교호작용** (`PREREG.md` 동결 `601ef1b`).

⛔이 파일은 **러너보다 먼저** 쓰였다(사전등록 §12-2). 판정 문턱·라벨·확률·설계상수는 **전부
여기서 계산**되고, 러너·분석기는 이 모듈만 부른다(⛔리터럴 손복사 0).

★이 배치의 특이점 (사전등록 §0.4 · §3.6 을 코드로 옮긴 것)
  네 칸 중 **셋은 동결 행에 이미 있다**(`B-16 SHIP-NOW == B-22 SHIP` 48/48 비트,
  `B-22 LEARN == B-22 SHIP` 48/48 비트 — 설계 시점 실측, `verify_registered()` 가 재확인).
  그러나 **동결 행은 「앵커」로만** 쓰고 네 칸을 **전부 새로 돌린다**(`T13` 사고 방지).
  그리고 사전등록은 **결과 전에** 이 설계가 자기 질문에 답하지 못할 공산이 크다고 적었다:
  가법 널이 결측 칸에 예측하는 값이 **관측 범위 밖**이라 `P(초가법) = 0.000000` 이다.
  ⟹ ★**등록된 산출물은 「`A-Q` 가 어디에 착지하는가」**(축 II = `T_INERT`·`T_CEIL`)다.
  ⛔결과를 보고 이것을 「조합 주장」으로 되돌리지 마라.

⛔**금지**: 외부 기준선 우위 주장(외부 행 0개) · 희소/실측/비혼잡 외삽 · `SEC-*` 승격 ·
  `B-16`/`B-22`/`B-20b` 판정 대체 · 배포 권고 · `d_i` 중앙값 단독 효과크기 인용.
"""
import glob
import json
import math
import os
import statistics as st
import sys
from math import comb, erf, sqrt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
B16DIR = os.path.abspath(os.path.join(HERE, '..', 'B16_KGRID_2026-08-24'))
B22DIR = os.path.abspath(os.path.join(HERE, '..', 'B22_PRIOR_ABLATION_2026-08-27'))
for _p in (HERE, ROOT, B16DIR, B22DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gates_b16 as B16                                            # noqa: E402
import gates_b22 as B22                                            # noqa: E402

# ── 표본·시드 (⛔손목록 금지 — 형제 배치에서 임포트) ────────────────────────
N_EP = B22.N_EP                          # 48 (혼잡 HOS, `B-12` 조건절)
EST_SEEDS = tuple(B22.EST_SEEDS)         # (42, 43, 44) — ★앵커의 **요구조건**
SECC_SEEDS = (45, 46, 47)                # `SEC-C` 신선 추정기 시드
SECC_N_EP = 16                           # 정렬된 48 중 앞 16
SMOKE_N_EP = 2                           # 사전등록 §7
LAYER_PRIMARY = 'PRIMARY'
LAYER_SECC = 'SEC-C'
MESH_EXPECTED = dict(B22.MESH_EXPECTED)  # {'ADEQUATE': 31, 'DEGRADED': 17}
TP_FACTOR = B22.TP_FACTOR

# ── 개입 노브 둘 (⛔`wavesplat/**` 은 한 바이트도 고치지 않는다) ──────────────
DK1 = B16.DK1                            # 0.02618 = 2π/240 (기준선 격자의 고운 축)
DK1_REDERIVE = 2.0 * math.pi / 240.0     # ⚠사전등록 §9.3-5: `DK1` 은 손 반올림 리터럴이다
DK_REDERIVE_TOL = 1e-5                   # 실측 편차 6.12e-08 의 163배 (`B-19` 규약 상속)


# ─────────────────────────────────────────────────────────────────────────────
# §1 팔 (5) — ⛔빈 dict `{}` 로 팔을 정의하지 않는다 (`gates_b7.py:46` 사고)
# ─────────────────────────────────────────────────────────────────────────────
#  · `prior`   True = 학습된 사전(확산 제안) · False = v2 빔포밍 원자, **개수 일치**(`ANA-N`)
#  · `quant`   True = `k_snap_dk = DK1`(격자 스냅) · False = `None`(연속)
#  · `b22`     골격을 가져오는 `gates_b22` 팔 — 배포 상수는 **패키지에서** 채워진다
#  · `atoms`   러너가 주입할 원자 출처. `None` = 인라인(배포 경로)
#  ★`SHIP` 은 칸이 아니라 **앵커**다(인라인 배포 경로). 빼지 않는다(사전등록 §1.3).
ARMS = {
    'SHIP': dict(prior=True,  quant=False, b22='SHIP',   atoms=None,
                 cell=None,           anchors=(('B22', 'SHIP'), ('B16', 'SHIP-NOW'))),
    'L-C':  dict(prior=True,  quant=False, b22='LEARN',  atoms='learned',
                 cell=('ON', 'CONT'), anchors=(('B22', 'LEARN'),)),
    'L-Q':  dict(prior=True,  quant=True,  b22='LEARN',  atoms='learned',
                 cell=('ON', 'SNAP'), anchors=(('B16', 'KGRID1'),)),
    'A-C':  dict(prior=False, quant=False, b22='ANA-N',  atoms='analytic_n',
                 cell=('OFF', 'CONT'), anchors=(('B22', 'ANA-N'),)),
    # ★결측 칸 — 저장소 전체에 0행. 이 배치가 처음 잰다 ⟹ 앵커 **없음**.
    'A-Q':  dict(prior=False, quant=True,  b22='ANA-N',  atoms='analytic_n',
                 cell=('OFF', 'SNAP'), anchors=()),
}
ARM_ORDER = ['SHIP', 'L-C', 'L-Q', 'A-C', 'A-Q']
CELL_ARMS = ('L-C', 'L-Q', 'A-C', 'A-Q')          # 2×2 칸 (⛔`SHIP` 은 앵커다)
INJECTED_ARMS = tuple(a for a in ARM_ORDER if ARMS[a]['atoms'] is not None)
QUANT_ARMS = tuple(a for a in ARM_ORDER if ARMS[a]['quant'])
CONT_ARMS = tuple(a for a in ARM_ORDER if not ARMS[a]['quant'])
# ★열 안의 유일한 차이가 `k_snap_dk` 임을 되읽기로 증명하는 짝 (`G-PAIRSRC`)
SNAP_PAIRS = (('L-C', 'L-Q'), ('A-C', 'A-Q'))
CTRL_ARM = 'L-C'                                  # 양성대조 = 배포 표현이 무기술을 이긴다
ROUTE_EXPECTED = B22.ROUTE_LEARNED                # 'learned' — 다섯 팔 전건
DEPLOY_EXPECTED = dict(B22.DEPLOY_EXPECTED)


def arm_config(arm, atoms=None):
    """★팔의 **완성된 설정 dict** — 러너·자가진단이 같은 함수를 부른다(단일출처).

    골격은 `gates_b22.arm_config` 가 **패키지 배포값으로** 채우고, 이 배치가 더하는 키는
    `k_snap_dk` **하나뿐**이다(사전등록 §1). ⛔`(x or 기본값)` 금지 — `None` 은 falsy 다.
    """
    spec = ARMS[arm]
    cfg = B22.arm_config(spec['b22'], atoms)
    cfg['k_snap_dk'] = DK1 if spec['quant'] else None
    return cfg


def expected_dk(arm):
    """이 팔에서 **발화해야 하는** 스냅 간격. `None` = 발화하면 안 된다(`G-SNAPFIRE`)."""
    return DK1 if ARMS[arm]['quant'] else None


PRIOR_OFF_ARMS = tuple(a for a in ARM_ORDER if not ARMS[a]['prior'])
# ★★사전등록 §1.2 가 **결과 전에** 적은 **등록된 이탈**:
#   *"`ANA-N` 은 `learned_energy_thresh = 0.0` 으로 돈다(`B-22 §3` 규약) ⟹ 사전 인자는
#     「원자 출처」와 「에너지 문턱」의 묶음이다. ⛔단일 노브라고 쓰지 마라."*
#   ⟹ 사전-OFF 두 팔은 배포 상수 9종 중 **이 한 키에서만** 핀을 벗어난다.
#   ⚠사전등록 §6.1 의 `G-DEPLOY` 는 이 이탈을 예외로 적지 않아 §1.2 와 **모순**이었다
#     (스모크가 잡았다). 개정 = `AMENDMENT_2_PREDATA.md` — 예외를 **허용**하는 것이 아니라
#     **요구**한다(빠져 있으면 그것도 FAIL 이다).
DEPLOY_REGISTERED_DEVIATION = {'learned_energy_thresh': 0.0}


def arm_deploy_expected(arm):
    """★이 팔이 **등록한** 배포 상수 9종 — ⛔`DEPLOY_EXPECTED` 를 그대로 쓰지 않는다."""
    cfg = arm_config(arm)
    return {k: cfg[k] for k in DEPLOY_EXPECTED}


def deploy_deviation_audit():
    """팔별로 배포 핀을 벗어나는 키 — ★**등록된 이탈과 정확히 같아야 한다**.

    반환 `{arm: {key: (팔의 값, 핀)}}`. 이 함수는 행을 읽지 않는다(팔 정의만 본다).
    """
    out = {}
    for a in ARM_ORDER:
        got = arm_deploy_expected(a)
        dev = {k: (got[k], v) for k, v in DEPLOY_EXPECTED.items() if got[k] != v}
        if dev:
            out[a] = dev
    return out


def deploy_deviation_registered():
    """등록된 이탈의 정본 — `deploy_deviation_audit()` 이 이것과 **집합으로 일치**해야 한다."""
    return {a: {k: (v, DEPLOY_EXPECTED[k])
                for k, v in DEPLOY_REGISTERED_DEVIATION.items()}
            for a in PRIOR_OFF_ARMS}


# ─────────────────────────────────────────────────────────────────────────────
# §2 부호검정 · 문턱 · MDE — ⛔리터럴 32/31 을 판정 코드에 적지 않는다(§12-4)
# ─────────────────────────────────────────────────────────────────────────────
ALPHA = 0.05
FAMILY_SIZE = 1          # ★PRIMARY 가족 = **1주장**(교호작용). 주효과 둘은 앵커/양성대조다.


def p_ge(k, n, q=0.5):
    return sum(comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(k, n + 1))


def p_le(k, n, q=0.5):
    return sum(comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(0, k + 1))


def sign_p_one(w, n):
    """단측 부호검정 `P(W >= w | q=0.5)`."""
    return None if n <= 0 else p_ge(w, n, 0.5)


def sign_p_two(w, n):
    """양측 부호검정 (대칭 이항이므로 `2·min(꼬리)`, 1 로 자른다)."""
    if n <= 0:
        return None
    return min(1.0, 2.0 * min(p_ge(w, n, 0.5), p_le(w, n, 0.5)))


def k_one(n, alpha=ALPHA):
    """단측 문턱 = `P(W >= k | .5) <= alpha` 인 최소 k. 없으면 `None`."""
    c = [k for k in range(n + 1) if p_ge(k, n, 0.5) <= alpha]
    return min(c) if c else None


def k_two(n, alpha=ALPHA / FAMILY_SIZE):
    """양측 문턱 `k*(n)` = 각 꼬리 `alpha/2`. ⛔없으면 `None`(= 원리적 판정 불가)."""
    return k_one(n, alpha / 2.0)


def mde_q(k, n, power):
    """검정력 `power` 를 주는 최소 승률 q (이분탐색)."""
    if k is None:
        return None
    lo, hi = 0.5, 1.0
    for _ in range(200):
        m = 0.5 * (lo + hi)
        if p_ge(k, n, m) >= power:
            hi = m
        else:
            lo = m
    return hi


THRESH_I = k_two(N_EP)                    # 축 I (교호작용, 양측 α=0.05) — 32
THRESH_SEC = k_one(N_EP, ALPHA)           # SEC-A/SEC-B (무보정 단측)   — 31
THRESH_CTRL = THRESH_I                    # ★양성대조 `W[L-C < ZERO]` (사전등록 §3.3 다리 -0)
THRESH_SECC = k_two(SECC_N_EP)            # `SEC-C` 축 I (n=16)
# ⛔`k*(n_eff)` 가 **존재하지 않는** n = 어떤 자료로도 양측 기각 불가 ⟹ UNDER-POWERED
N_EFF_UNPOWERED = [n for n in range(1, N_EP + 1) if k_two(n) is None]
N_EFF_UNPOWERED_MAX = max(N_EFF_UNPOWERED) if N_EFF_UNPOWERED else 0

MDE = {
    'I_q50': round(mde_q(THRESH_I, N_EP, 0.50), 4),
    'I_q80': round(mde_q(THRESH_I, N_EP, 0.80), 4),
    'I_q90': round(mde_q(THRESH_I, N_EP, 0.90), 4),
    'SEC_q50': round(mde_q(THRESH_SEC, N_EP, 0.50), 4),
    'SEC_q80': round(mde_q(THRESH_SEC, N_EP, 0.80), 4),
    'SEC_q90': round(mde_q(THRESH_SEC, N_EP, 0.90), 4),
    'SECC_q80': round(mde_q(THRESH_SECC, SECC_N_EP, 0.80), 4),
}
Q80 = MDE['I_q80']                        # 축 I 의 MDE — ⛔SEC·SEC-C 와 **섞지 마라**
EQ_BAND = math.log(1.05)                  # 등가 밴드 (`T14 §4.4` → `T16 §3.3` 계승)
TIE_ABS = 1e-12                           # 동점 = `|d_i| < 1e-12` (사전등록 §3.1)


def n_eff_table(ns=(48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 6)):
    """`n_eff` 별 축 I 문턱·q80 — ⚠단조가 아니다(부호검정의 이산성). ⛔오류가 아니다."""
    out = {}
    for n in ns:
        k = k_two(n)
        out[n] = dict(k=k, q80=(None if k is None else round(mde_q(k, n, 0.80), 4)))
    return out


def family_cost_table():
    """★2×2 vs 2×2×2 의 대가 (사전등록 §5.2 블록 3) — 선언이 아니라 수치."""
    rows = {}
    for nm, fam, runs in (('2x2_family1', 1, 720), ('2x2_family3', 3, 720),
                          ('2x2x2_family7', 7, 1296)):
        a = ALPHA / fam
        k = k_two(N_EP, a)
        rows[nm] = dict(tail_alpha=a / 2.0, k=k, q80=round(mde_q(k, N_EP, 0.80), 4),
                        primary_runs=runs)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# §3 동결 행 — **앵커**이자 **설계 상수의 출처**. ⛔비교자로 쓰지 않는다(§2.2).
# ─────────────────────────────────────────────────────────────────────────────
B16_GLOB = os.path.join(B16DIR, 'results_b16_*.jsonl')
B22_GLOB = os.path.join(B22DIR, 'results_b22_*.jsonl')
# ★동결 「10팔」 = `B-16` 3 + `B-22` 8 − 중복 1(`SHIP-NOW` == `SHIP`, 48/48 비트).
FROZEN_ARMS = (('B16', 'KGRID1'), ('B16', 'KGRID2'),
               ('B22', 'SHIP'), ('B22', 'LEARN'), ('B22', 'ANA'), ('B22', 'ANA-N'),
               ('B22', 'LEARN-NOVP'), ('B22', 'ANA-NOVP'), ('B22', 'ANA-N-NOVP'),
               ('B22', 'ANA-SHIP'))
FROZEN_DUP = ('B16', 'SHIP-NOW')          # ⚠`B22 SHIP` 과 비트 동일 — 세지 않는다
_FROZEN_CACHE = {}


def _load_jsonl(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        with open(p, encoding='utf-8') as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'error' not in r:
                    rows.append(r)
    return rows


def frozen_rows(batch):
    """동결 jsonl 전건 (`batch ∈ {'B16','B22'}`)."""
    if batch not in _FROZEN_CACHE:
        _FROZEN_CACHE[batch] = _load_jsonl(B16_GLOB if batch == 'B16' else B22_GLOB)
    return _FROZEN_CACHE[batch]


def med3(rows, key='avg_nRMSE', seeds=EST_SEEDS):
    """★에피소드별 **median-of-3** — `B-16`·`B-22` 규약 축자 계승.

    ⛔시드가 셋이 아닌 (arm, episode) 는 **버리지 않고 빼놓는다**(`incomplete`) — 조용한
    표본 손실이 이 저장소가 반복해서 물린 부류다.
    반환 `(값 dict, 불완전 목록)`.
    """
    bag = {}
    for r in rows:
        if r.get('est_seed') not in seeds:
            continue
        v = r.get(key)
        if v is None:
            continue
        bag.setdefault((r['arm'], r['episode']), {})[r['est_seed']] = float(v)
    out, bad = {}, []
    for k, d in bag.items():
        if len(d) == len(seeds):
            out[k] = st.median([d[s] for s in sorted(d)])
        else:
            bad.append((k, sorted(d)))
    return out, bad


def frozen_med3(batch):
    m, bad = med3(frozen_rows(batch))
    assert not bad, f'동결 {batch} 에 불완전 셀 {bad[:3]}'
    return m


def frozen_zero():
    """에피소드별 참 제로필드 기준 (`B-22` 동결 행의 `zero_nRMSE`). ⛔상수 1.0 아님."""
    out = {}
    for r in frozen_rows('B22'):
        z = r.get('zero_nRMSE')
        if z is None:
            continue
        out.setdefault(r['episode'], set()).add(float(z))
    bad = {e: sorted(v) for e, v in out.items() if len(v) > 1}
    assert not bad, f'동결 ZERO 가 에피소드 안에서 갈린다 {list(bad)[:3]}'
    return {e: next(iter(v)) for e, v in out.items()}


def ceil_by_episode():
    """`CEIL_i` = 동결 **10팔**에서 관측된 에피소드별 최대 `nRMSE / ZERO`.

    ★축 II 의 `T_CEIL` 이 이 값을 쓴다. 동결 값이므로 **핀**이고, 결과를 보고 바꾸지 않는다.
    """
    m16, m22, z = frozen_med3('B16'), frozen_med3('B22'), frozen_zero()
    out = {}
    for e in sorted(z):
        vals = []
        for b, a in FROZEN_ARMS:
            v = (m16 if b == 'B16' else m22).get((a, e))
            if v is not None:
                vals.append(v / z[e])
        if vals:
            out[e] = max(vals)
    return out


def design_constants():
    """★§0.4 의 설계 상수를 **동결 행에서 재계산**한다(⛔손복사 금지 · 추정기 런 0회).

    `q1` = 양자화 효과(사전 ON 열) · `pr` = 사전 효과(k 연속 열) · `RHO` = 둘의 상관.
    ⚠`SD` 는 **모표준편차**(ddof=0) — 사전등록의 인쇄값이 그것이다(`verify_registered` 가 댄다).
    """
    m16, m22, z = frozen_med3('B16'), frozen_med3('B22'), frozen_zero()
    ceil = ceil_by_episode()
    eps = sorted(e for e in z if ('LEARN', e) in m22)
    q1 = [math.log(m16[('KGRID1', e)]) - math.log(m22[('LEARN', e)]) for e in eps]
    pr = [math.log(m22[('ANA-N', e)]) - math.log(m22[('LEARN', e)]) for e in eps]
    # 가법 널이 결측 칸에 예측하는 값 (ZERO 대비)
    pred = [m22[('ANA-N', e)] * m16[('KGRID1', e)] / m22[('LEARN', e)] / z[e] for e in eps]
    allrel = []
    for e in eps:
        for b, a in FROZEN_ARMS:
            v = (m16 if b == 'B16' else m22).get((a, e))
            if v is not None:
                allrel.append(v / z[e])

    def _sd0(x):
        mu = st.mean(x)
        return math.sqrt(sum((v - mu) ** 2 for v in x) / len(x))

    def _corr(a, b):
        ma, mb = st.mean(a), st.mean(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
        return num / den

    return dict(
        n=len(eps), episodes=eps,
        MED_Q1=st.median(q1), SD_Q1=_sd0(q1), Q1_POS=sum(1 for v in q1 if v > 0),
        MED_PR=st.median(pr), SD_PR=_sd0(pr), PR_POS=sum(1 for v in pr if v > 0),
        RHO=_corr(q1, pr),
        CEIL_MED=st.median([ceil[e] for e in eps]),
        CEIL_MAX=max(allrel),
        SOFT_CEIL=math.sqrt(2.0),
        ADD_NULL_MED=st.median(pred), ADD_NULL_MIN=min(pred), ADD_NULL_MAX=max(pred),
        ADD_OVER_SOFT=sum(1 for p in pred if p > math.sqrt(2.0)),
        ADD_OVER_EPCEIL=sum(1 for p, e in zip(pred, eps) if p > ceil[e]),
        ADD_OVER_GLOBAL=sum(1 for p in pred if p > max(allrel)))


def unit_costs():
    """★§8 의 단가 = 동결 행의 `wall_s` **중앙**. ⚠`A-Q` 만 가산 외삽이다."""
    med = {}
    for batch, glob_arm in (('B16', ('KGRID1', 'SHIP-NOW')),
                            ('B22', ('SHIP', 'LEARN', 'ANA-N'))):
        bag = {}
        for r in frozen_rows(batch):
            if r['arm'] in glob_arm and r.get('wall_s') is not None:
                bag.setdefault(r['arm'], []).append(float(r['wall_s']))
        for a, v in bag.items():
            med[(batch, a)] = st.median(v)
    ship, lc = med[('B22', 'SHIP')], med[('B22', 'LEARN')]
    lq, ac = med[('B16', 'KGRID1')], med[('B22', 'ANA-N')]
    return {'SHIP': ship, 'L-C': lc, 'L-Q': lq, 'A-C': ac,
            # ⚠추정: 사전-OFF 열에서 스냅이 더하는 비용을 사전-ON 열에서 빌렸다
            'A-Q': ac + (lq - ship), 'A-Q_is_extrapolated': True}


def projected_cost():
    """런 수 × 단가 (⛔`ZERO` 는 산술이라 런 0회)."""
    c = unit_costs()
    prim = {a: len(EST_SEEDS) * N_EP for a in ARM_ORDER}
    secc = {a: len(SECC_SEEDS) * SECC_N_EP for a in CELL_ARMS}
    s_prim = sum(prim[a] * c[a] for a in ARM_ORDER)
    s_secc = sum(secc[a] * c[a] for a in CELL_ARMS)
    return dict(primary_runs=sum(prim.values()), secc_runs=sum(secc.values()),
                total_runs=sum(prim.values()) + sum(secc.values()),
                per_arm_primary={a: round(prim[a] * c[a], 1) for a in ARM_ORDER},
                seconds=s_prim + s_secc, hours=(s_prim + s_secc) / 3600.0,
                unit_costs={k: (round(v, 2) if isinstance(v, float) else v)
                            for k, v in c.items()})


N_RUNS_PRIMARY = len(ARM_ORDER) * N_EP * len(EST_SEEDS)        # 720
N_RUNS_SECC = len(CELL_ARMS) * SECC_N_EP * len(SECC_SEEDS)     # 192
N_RUNS_TOTAL = N_RUNS_PRIMARY + N_RUNS_SECC                    # 912
N_RUNS_SMOKE = len(ARM_ORDER) * SMOKE_N_EP * len(EST_SEEDS)    # 30


# ─────────────────────────────────────────────────────────────────────────────
# §4 판정 사다리 (사전등록 §3.3) — ⛔결과를 보고 순서·문턱을 바꾸지 않는다
# ─────────────────────────────────────────────────────────────────────────────
LABELS = ('ANCHOR-DID-NOT-REPRODUCE',
          'CONTROL-DID-NOT-REPLICATE',
          'INTERACTION-UNDER-POWERED',
          'COMBINATION-SUPERADDITIVE',
          'SUBADDITIVE-QUANT-BELOW-SCALE-WITHOUT-PRIOR',
          'SUBADDITIVE-AT-CEILING',
          'SUBADDITIVE-PARTIAL',
          'NO-INTERACTION-DETECTED')
# ⛔양성 주장은 **하나뿐**이다. 나머지는 구조(3) · 미검출(2) · 차단(2).
LABEL_KIND = {'COMBINATION-SUPERADDITIVE': 'CONFIRMATORY',
              'SUBADDITIVE-QUANT-BELOW-SCALE-WITHOUT-PRIOR': 'STRUCTURAL',
              'SUBADDITIVE-AT-CEILING': 'STRUCTURAL',
              'SUBADDITIVE-PARTIAL': 'STRUCTURAL-EXPLORATORY',
              'NO-INTERACTION-DETECTED': 'NOT-DETECTED',
              'INTERACTION-UNDER-POWERED': 'NOT-DETECTED',
              'ANCHOR-DID-NOT-REPRODUCE': 'BLOCKED',
              'CONTROL-DID-NOT-REPLICATE': 'BLOCKED'}


def interaction_contrast(y_aq, y_ac, y_lq, y_lc):
    """★교호작용 대비 `d_i` — **단일출처 정의**(사전등록 §0.2). ⛔다른 식으로 계산하지 않는다.

        d_i = [ln y(A-Q) − ln y(A-C)] − [ln y(L-Q) − ln y(L-C)]
            = ln( y(A-Q)·y(L-C) / (y(A-C)·y(L-Q)) )

    `d_i > 0` = **초가법**(두 절제를 겹치면 합보다 더 나빠진다).
    로그 척도라 에피소드별 임의의 곱셈 재척도에 **불변**이다.
    """
    return (math.log(y_aq) - math.log(y_ac)) - (math.log(y_lq) - math.log(y_lc))


def t_inert(med_ln_r):
    """축 II 판별자 ①: *"사전이 없으면 양자화가 자보다 작게 움직인다"* (`|median ln r| < ln 1.05`)."""
    return abs(med_ln_r) < EQ_BAND


def t_ceil(med_z, med_ceil):
    """축 II 판별자 ②: *"착지가 이미 관측된 최악 대역이다"* = 천장 교락."""
    return med_z >= med_ceil


def label(anchor_ok, w_ctrl, w_plus, n_eff, inert, ceil_hit):
    """★인쇄 라벨 (사전등록 §3.3 발화 순서 그대로).

    ⛔`SUBADDITIVE-QUANT-BELOW-SCALE-WITHOUT-PRIOR` 를 `NO-INTERACTION-DETECTED` 로 접지
    마라(`M8b` 가 대가를 치른 부류) — **왜 그 방향인지가 라벨의 내용**이다.
    ⛔`NO-INTERACTION-DETECTED` 를 *"가법적이다/차이가 없다"* 로 읽지 마라 — **검정력의 진술**이다.
    ★2번 다리를 `w <= n_eff − k` 가 아니라 `(n_eff − w) >= k` 로 적는다: 반대 방향의 **이김 수**를
    세는 것이 정직한 서술이고, 정적 검사기도 그렇게 읽는다(`T16 label_D` 규약).
    """
    if not anchor_ok:
        return 'ANCHOR-DID-NOT-REPRODUCE'
    if w_ctrl < THRESH_CTRL:
        return 'CONTROL-DID-NOT-REPLICATE'
    kstar = k_two(n_eff)
    if kstar is None:
        return 'INTERACTION-UNDER-POWERED'
    if w_plus >= kstar:
        return 'COMBINATION-SUPERADDITIVE'
    if (n_eff - w_plus) >= kstar:
        if inert:
            return 'SUBADDITIVE-QUANT-BELOW-SCALE-WITHOUT-PRIOR'
        if ceil_hit:
            return 'SUBADDITIVE-AT-CEILING'
        return 'SUBADDITIVE-PARTIAL'
    return 'NO-INTERACTION-DETECTED'


def ladder_table(n=N_EP):
    """★사다리를 **전 조합에서** 돌린다 — `UNDECIDED` 구멍이 없음을 기계로 보인다.

    반환 `[(입력 dict, 라벨), …]`. ⛔라벨이 `None` 이거나 등록 목록 밖이면 `selftest` 가 죽는다.
    """
    k = k_two(n)
    rows = []
    for anchor_ok in (False, True):
        for w_ctrl in (THRESH_CTRL - 1, THRESH_CTRL):
            for w_plus, n_eff in ((k, n), (n - k, n), (n // 2, n),
                                  (N_EFF_UNPOWERED_MAX, N_EFF_UNPOWERED_MAX)):
                for inert in (True, False):
                    for ceil_hit in (True, False):
                        kw = dict(anchor_ok=anchor_ok, w_ctrl=w_ctrl, w_plus=w_plus,
                                  n_eff=n_eff, inert=inert, ceil_hit=ceil_hit)
                        rows.append((kw, label(**kw)))
    # ★경계 하나 더 — `n_eff = 6` 에서는 `k*` 가 **존재한다**(UNDER-POWERED 아님)
    kw = dict(anchor_ok=True, w_ctrl=THRESH_CTRL, w_plus=6,
              n_eff=N_EFF_UNPOWERED_MAX + 1, inert=True, ceil_hit=True)
    rows.append((kw, label(**kw)))
    return rows


# ── 결과 전 예측 (사전등록 §3.5) — ⛔손타이핑 금지, **코드가 계산한다** ────────
P_ANCH = 0.98        # ⚠이항 아님 — `S-ENS` 128/128 비트 · 그러나 배포 상수가 바뀐 전례가 있다
P_CTRL = 0.9999      # 동결 48/48 이 근거
P_INERT = 0.75       # 축 II 갈래 발화 예측 (결정적 통계량 — ⛔이항 확률로 인용하지 마라)
P_CEIL = 0.15


def _phi(z):
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def q_of(delta, rho, sd=None):
    """`d_i > 0` 승률 `q = Φ(delta / sd_d)`, `sd_d = sd(q1)·√(2(1−ρ))`.

    ⚠`ρ` 는 **미관측 대리값**이다(사전-ON 열의 상관을 사전-OFF 열에 빌렸다) ⟹ 민감도를 4열로 적는다.
    """
    if sd is None:
        sd = PRED_INPUTS['SD_Q1']
    return _phi(delta / (sd * sqrt(2.0 * (1.0 - rho))))


# ★★**예측의 입력은 「등록된 4자리 값」이다** — ⛔전정밀도 재계산값이 아니다.
#   사유(구현 시점 실측, 결과 0행): 사전등록 §3.5 의 스크립트가
#   `MED_Q1, SD_Q1, MED_PR, RHO = 0.4879, 0.3290, 0.6655, 0.7138` 을 **리터럴로** 고정하고
#   그 위에서 `Q_REG`·`Q_SYN` 을 냈다. 동결 jsonl 에서 전정밀도로 재계산하면
#   `Q_REG` 가 0.003752 → 0.0037468 로 **5자리째에서** 갈린다(라벨 확률은 인쇄 정밀도로 불변).
#   ⟹ **등록된 예측은 등록된 입력의 함수**이므로 그것을 쓰고, ⛔재계산은 버리지 않고
#      `verify_registered()` 가 4자리 일치를 **강제**한다(§12-1 "재계산으로 핀한다").
PRED_INPUTS = {'MED_Q1': 0.4879, 'SD_Q1': 0.3290, 'MED_PR': 0.6655, 'RHO': 0.7138}


def delta_map(rhos=(0.0, 0.5, None, 0.9)):
    """§3.5 블록 4 — `delta -> q` 사상. `None` = 등록된 `RHO`."""
    d = PRED_INPUTS
    rho_live = d['RHO']
    cases = (('H_sub/중복', -d['MED_PR']), ('H_sub/중간', -0.50 * d['MED_Q1']),
             ('H_add/독립', 0.0), ('H_add/약함', +0.25 * d['MED_Q1']),
             ('H_syn/반쪽', +0.50 * d['MED_Q1']), ('H_syn/강함', +1.00 * d['MED_Q1']))
    out = {}
    for nm, dd in cases:
        out[nm] = dict(delta=dd, q={('rho_reg' if r is None else f'rho_{r}'):
                                    q_of(dd, rho_live if r is None else r, d['SD_Q1'])
                                    for r in rhos})
    return out


def q_registered():
    """★등록 예측 `Q_REG`(= H_sub, 중복) 와 반대 가설 `Q_SYN`(= 반쪽 synergy)."""
    d = PRED_INPUTS
    return (q_of(-d['MED_PR'], d['RHO'], d['SD_Q1']),
            q_of(+0.50 * d['MED_Q1'], d['RHO'], d['SD_Q1']))


def label_probs(q=None, n=None, p_anch=P_ANCH, p_ctrl=P_CTRL,
                p_inert=P_INERT, p_ceil=P_CEIL):
    """★등록 예측 하의 **모든 라벨 확률** — §3.5 블록 5 를 재생산한다.

    ⚠**종속 공시**: 네 칸이 같은 48 에피소드·같은 관측점을 공유하므로 독립이 아니다.
    ⛔사후 검정력 주장에 쓰지 마라. ⚠`p_inert`·`p_ceil` 은 **선언적 예측**이고 이항이 아니다 —
    그 셋의 **합(아가법 버킷)** 만이 이항에서 나온 값이다.
    """
    n = N_EP if n is None else n
    q = q_registered()[0] if q is None else q
    k = k_two(n)
    base = p_anch * p_ctrl
    if k is None:
        sup = sub = 0.0
        nul = 0.0
        unp = 1.0
    else:
        sup, sub, unp = p_ge(k, n, q), p_le(n - k, n, q), 0.0
        # ⛔`1 − sup − sub` 이 부동소수로 **음수**가 될 수 있다(q 가 극단이면 두 꼬리 합이
        #   1 을 미세하게 넘는다) ⟹ 0 으로 자른다. `-0.000000` 을 인쇄하지 않기 위해서다.
        nul = max(0.0, 1.0 - sup - sub)
    out = {'ANCHOR-DID-NOT-REPRODUCE': 1.0 - p_anch,
           'CONTROL-DID-NOT-REPLICATE': p_anch * (1.0 - p_ctrl),
           'COMBINATION-SUPERADDITIVE': base * sup,
           'SUBADDITIVE-QUANT-BELOW-SCALE-WITHOUT-PRIOR': base * sub * p_inert,
           'SUBADDITIVE-AT-CEILING': base * sub * p_ceil,
           'SUBADDITIVE-PARTIAL': base * sub * (1.0 - p_inert - p_ceil),
           'NO-INTERACTION-DETECTED': base * nul,
           'INTERACTION-UNDER-POWERED': base * unp}
    out['_meta'] = dict(q=q, n=n, k_star=k, sum=sum(v for kk, v in out.items()
                                                    if not kk.startswith('_')))
    return out


def null_label_probs(n=None):
    """귀무 `q=0.5` 아래 오라벨률 (§3.5 블록 7)."""
    return label_probs(q=0.5, n=n)


def detect_power_syn():
    """★반대 가설(반쪽 synergy) 아래 `COMBINATION-SUPERADDITIVE` 검출확률.

    ⚠앵커·양성대조 통과를 **곱한** 값이다(사전등록의 인쇄값이 그것이다).
    """
    return label_probs(q=q_registered()[1])['COMBINATION-SUPERADDITIVE']


def sensitivity(qs=(0.004, 0.05, 0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.837, 0.9), n=None):
    """§3.5 블록 6 — 축 I **주변확률**(앵커/대조 조건부)."""
    n = N_EP if n is None else n
    k = k_two(n)
    out = {}
    for q in qs:
        sup, sub = p_ge(k, n, q), p_le(n - k, n, q)
        out[q] = dict(superadditive=sup, subadditive=sub, null=1.0 - sup - sub)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# §5 게이트 이름 · 차단 (사전등록 §6)
# ─────────────────────────────────────────────────────────────────────────────
GATES_BLOCKING = (
    'G-ANCHOR-B22', 'G-ANCHOR-B16', 'G-LATTICE', 'G-SNAPFIRE', 'G-ATOMSRC',
    'G-PAIRSRC', 'G-FRAME', 'G-PROV', 'G-DEPLOY', 'G-DEPTH', 'G-MAXN',
    'G-CORPUS', 'G-MESH', 'G-SIBLING', 'G-FINITE', 'G-N', 'G-ENV',
    'G-BLOCK', 'G-COV',
)
# ★기록 전용 — ⛔막지 않는다. 이름을 등록해 두는 이유는 **나중에 조건이 붙어도 장식이 되지
#   않게** 하기 위해서다(`T5-ENS` 는 차단 게이트 14개가 이름 비교 결함으로 장식이 될 뻔했다).
GATES_RECORD = ('G-DOSE', 'G-NSEL', 'G-VARPRO', 'G-CEIL', 'G-WALL', 'G-GROW')
GATE_NAMES = GATES_BLOCKING + GATES_RECORD


def blocked_by(rows):
    """⛔**긴 이름부터** 접두 비교 (`T5-ENS` 의 *"항상 빈 리스트"* 재발 방지).

    이 규약이 없으면 `G-ANCHOR-B22` 가 `G-ANCHOR-B16` 과, `G-N` 이 `G-NSEL` 과 접힌다.
    `rows` = `(이름, ok, 상세)` 목록.
    """
    order = sorted(set(GATE_NAMES), key=len, reverse=True)
    out = []
    for name, ok, _d in rows:
        if ok:
            continue
        nm = name.replace('★', '').replace('⚠', '').replace('⛔', '').strip()
        for b in order:
            if nm.startswith(b):
                out.append(b)
                break
    return out


def selfcheck_blocked_by(names=None):
    """★`G-BLOCK` 자가진단 — 등록된 **모든** 이름이 합성 위반 튜플로 실제로 잡히는가.

    반환 `(못 잡은 이름, 다른 이름으로 접힌 것)`. ⛔둘 다 비어야 한다.
    """
    names = list(GATE_NAMES) if names is None else list(names)
    missed, folded = [], []
    for nm in names:
        got = blocked_by([(f'★{nm} 합성 위반', False, '')])
        if not got:
            missed.append(nm)
        elif got[0] != nm:
            folded.append(f'{nm}→{got[0]}')
    if blocked_by([('★G-COV 합성 통과', True, '')]):
        missed.append('<통과 행이 차단됐다>')
    return missed, folded


# ─────────────────────────────────────────────────────────────────────────────
# §6 앵커 · 형제 교집합
# ─────────────────────────────────────────────────────────────────────────────
ANCHOR_T1_TOL = 0.0        # 등급 T1 = **비트 동일**
ANCHOR_T2_RTOL = 1e-9      # 등급 T2 = 상대차 ≤ 1e-9 (⛔어느 등급인지 verdict 에 인쇄)
ZERO_RTOL = 1e-5


def anchor_grade(a, b):
    """`None` = 재현 실패 · `'T1'` = 비트 동일 · `'T2'` = 상대차 ≤ 1e-9."""
    if a is None or b is None:
        return None
    if a == b:
        return 'T1'
    if b != 0 and abs(a / b - 1.0) <= ANCHOR_T2_RTOL:
        return 'T2'
    return None


def lattice_check():
    """★`G-LATTICE` — **동결끼리의 교차 등식** `B16 SHIP-NOW == B22 SHIP` (런 0회).

    ⛔거짓이면 2×2 격자 자체가 정합하지 않는다. 부수로 `B22 LEARN == B22 SHIP` 도 댄다
    (주입 경로 == 인라인 경로 — `L-C` 앵커가 성립하는 근거).
    """
    m16, m22 = frozen_med3('B16'), frozen_med3('B22')
    eps = sorted({e for (a, e) in m22 if a == 'SHIP'})
    g1 = [anchor_grade(m16.get(('SHIP-NOW', e)), m22.get(('SHIP', e))) for e in eps]
    g2 = [anchor_grade(m22.get(('LEARN', e)), m22.get(('SHIP', e))) for e in eps]
    return dict(n=len(eps),
                ship_now_vs_ship=dict(T1=g1.count('T1'), T2=g1.count('T2'),
                                      fail=g1.count(None)),
                learn_vs_ship=dict(T1=g2.count('T1'), T2=g2.count('T2'),
                                   fail=g2.count(None)))


def anchor_map():
    """이 배치의 팔 → 동결 팔 (비트 재현 요구). ⛔`A-Q` 는 앵커가 **없다**."""
    return {a: ARMS[a]['anchors'] for a in ARM_ORDER}


def sibling_key(episode, arm_alias, est_seed, k_snap_dk, atoms_md5):
    """`G-SIBLING` 의 교집합 키 (사전등록 §2.4). ★팔은 **앵커 별칭으로 사상**한 뒤 비교한다."""
    return (episode, arm_alias, int(est_seed),
            None if k_snap_dk is None else round(float(k_snap_dk), 12), atoms_md5)


# ─────────────────────────────────────────────────────────────────────────────
# §7 등록값 대조 — ⛔불일치 시 `AssertionError` (사전등록 §12-3)
# ─────────────────────────────────────────────────────────────────────────────
REGISTERED = {
    # §3.1 문턱
    'THRESH_I': 32, 'THRESH_SEC': 31, 'P_ONE_TAIL': 0.014652, 'P_TWO_TAIL': 0.029305,
    'P_SEC': 0.029732, 'EQ_BAND': 0.048790, 'CEIL_MED': 1.306,
    'N_EFF_UNPOWERED': [1, 2, 3, 4, 5],
    # §4 MDE
    'I_q50': 0.6552, 'I_q80': 0.7109, 'I_q90': 0.7384,
    'SEC_q50': 0.6345, 'SEC_q80': 0.6911, 'SEC_q90': 0.7193, 'SECC_q80': 0.8529,
    # §0.4 설계 상수 (동결 행에서 **재계산**)
    'MED_Q1': 0.4879, 'SD_Q1': 0.3290, 'MED_PR': 0.6655, 'SD_PR': 0.3615,
    'RHO': 0.7138, 'Q1_POS': 46, 'PR_POS': 48,
    'CEIL_MAX': 2.136, 'ADD_NULL_MED': 1.696, 'ADD_NULL_MIN': 0.695,
    'ADD_NULL_MAX': 3.930, 'ADD_OVER_SOFT': 37, 'ADD_OVER_EPCEIL': 43,
    'ADD_OVER_GLOBAL': 15,
    # §3.5 예측
    'Q_REG': 0.003752, 'Q_SYN': 0.836473, 'DETECT_SYN': 0.978864,
    'P_SUPER': 0.000000, 'P_INERT_LABEL': 0.734926, 'P_CEIL_LABEL': 0.146985,
    'P_PARTIAL': 0.097990, 'P_ANCHOR_FAIL': 0.020000, 'P_CTRL_FAIL': 0.000098,
    'P_NULL_NOINT': 0.951186, 'P_NULL_SUPER': 0.014358,
    # §5.2 가족 크기의 대가
    'FAM1_q80': 0.7109, 'FAM3_q80': 0.7305, 'FAM7_q80': 0.7500,
    # §8 단가 (초, 중앙)
    'COST_SHIP': 11.97, 'COST_LC': 8.55, 'COST_LQ': 11.87, 'COST_AC': 10.45,
    # §6.1 · §2.2 격자
    'LATTICE_T1': 48, 'MESH': {'ADEQUATE': 31, 'DEGRADED': 17},
    'N_RUNS_TOTAL': 912, 'N_RUNS_SMOKE': 30,
}


def verify_registered(verbose=False):
    """★사전등록에 인쇄된 **모든 수**를 재생산한다. ⛔불일치 = `AssertionError`.

    ⚠이것이 「손타이핑 0」의 기계적 근거다 — 문서와 코드가 갈리면 배치가 시작되지 않는다.
    """
    bad = []

    def eq(name, got, want, tol):
        ok = (got == want) if tol is None else (abs(got - want) <= tol)
        if not ok:
            bad.append(f'{name}: 구현 {got!r} vs 등록 {want!r}')
        elif verbose:
            print(f'  [OK] {name:16s} {got!r}')

    R = REGISTERED
    eq('THRESH_I', THRESH_I, R['THRESH_I'], None)
    eq('THRESH_SEC', THRESH_SEC, R['THRESH_SEC'], None)
    eq('P_ONE_TAIL', p_ge(THRESH_I, N_EP, 0.5), R['P_ONE_TAIL'], 5e-7)
    eq('P_TWO_TAIL', 2 * p_ge(THRESH_I, N_EP, 0.5), R['P_TWO_TAIL'], 1e-6)
    eq('P_SEC', p_ge(THRESH_SEC, N_EP, 0.5), R['P_SEC'], 5e-7)
    eq('EQ_BAND', EQ_BAND, R['EQ_BAND'], 5e-7)
    eq('N_EFF_UNPOWERED', N_EFF_UNPOWERED, R['N_EFF_UNPOWERED'], None)
    for k, v in (('I_q50', 'I_q50'), ('I_q80', 'I_q80'), ('I_q90', 'I_q90'),
                 ('SEC_q50', 'SEC_q50'), ('SEC_q80', 'SEC_q80'),
                 ('SEC_q90', 'SEC_q90'), ('SECC_q80', 'SECC_q80')):
        eq(k, MDE[v], R[k], 5e-5)
    d = design_constants()
    eq('n_episodes', d['n'], N_EP, None)
    for k in ('MED_Q1', 'SD_Q1', 'MED_PR', 'SD_PR', 'RHO'):
        eq(k, d[k], R[k], 5e-5)
    for k in ('Q1_POS', 'PR_POS', 'ADD_OVER_SOFT', 'ADD_OVER_EPCEIL', 'ADD_OVER_GLOBAL'):
        eq(k, d[k], R[k], None)
    eq('CEIL_MED', d['CEIL_MED'], R['CEIL_MED'], 5e-4)
    eq('CEIL_MAX', d['CEIL_MAX'], R['CEIL_MAX'], 5e-4)
    for k in ('ADD_NULL_MED', 'ADD_NULL_MIN', 'ADD_NULL_MAX'):
        eq(k, d[k], R[k], 5e-4)
    q_reg, q_syn = q_registered()
    eq('Q_REG', q_reg, R['Q_REG'], 5e-7)
    eq('Q_SYN', q_syn, R['Q_SYN'], 5e-7)
    eq('DETECT_SYN', detect_power_syn(), R['DETECT_SYN'], 5e-7)
    lp = label_probs()
    eq('P_SUPER', lp['COMBINATION-SUPERADDITIVE'], R['P_SUPER'], 5e-7)
    eq('P_INERT_LABEL', lp['SUBADDITIVE-QUANT-BELOW-SCALE-WITHOUT-PRIOR'],
       R['P_INERT_LABEL'], 1e-6)
    eq('P_CEIL_LABEL', lp['SUBADDITIVE-AT-CEILING'], R['P_CEIL_LABEL'], 5e-7)
    eq('P_PARTIAL', lp['SUBADDITIVE-PARTIAL'], R['P_PARTIAL'], 5e-7)
    eq('P_ANCHOR_FAIL', lp['ANCHOR-DID-NOT-REPRODUCE'], R['P_ANCHOR_FAIL'], 5e-7)
    eq('P_CTRL_FAIL', lp['CONTROL-DID-NOT-REPLICATE'], R['P_CTRL_FAIL'], 5e-7)
    eq('label_prob_sum', lp['_meta']['sum'], 1.0, 1e-9)
    nlp = null_label_probs()
    eq('P_NULL_NOINT', nlp['NO-INTERACTION-DETECTED'], R['P_NULL_NOINT'], 5e-7)
    eq('P_NULL_SUPER', nlp['COMBINATION-SUPERADDITIVE'], R['P_NULL_SUPER'], 5e-7)
    fam = family_cost_table()
    eq('FAM1_q80', fam['2x2_family1']['q80'], R['FAM1_q80'], 5e-5)
    eq('FAM3_q80', fam['2x2_family3']['q80'], R['FAM3_q80'], 5e-5)
    eq('FAM7_q80', fam['2x2x2_family7']['q80'], R['FAM7_q80'], 5e-5)
    c = unit_costs()
    eq('COST_SHIP', c['SHIP'], R['COST_SHIP'], 5e-3)
    eq('COST_LC', c['L-C'], R['COST_LC'], 5e-3)
    eq('COST_LQ', c['L-Q'], R['COST_LQ'], 5e-3)
    eq('COST_AC', c['A-C'], R['COST_AC'], 5e-3)
    lat = lattice_check()
    eq('LATTICE_T1', lat['ship_now_vs_ship']['T1'], R['LATTICE_T1'], None)
    eq('LATTICE_LEARN_T1', lat['learn_vs_ship']['T1'], R['LATTICE_T1'], None)
    eq('N_RUNS_TOTAL', N_RUNS_TOTAL, R['N_RUNS_TOTAL'], None)
    eq('N_RUNS_SMOKE', N_RUNS_SMOKE, R['N_RUNS_SMOKE'], None)
    eq('MESH', MESH_EXPECTED, R['MESH'], None)
    eq('DK1_REDERIVE', abs(DK1 - DK1_REDERIVE), 0.0, DK_REDERIVE_TOL)
    if bad:
        raise AssertionError('사전등록 ↔ 구현 불일치:\n  ' + '\n  '.join(bad))
    return True


def selftest(verbose=True):
    """★런 전 자가진단 — ⛔실패하면 러너가 **한 런도 돌리지 않는다**.

    ① 등록값 재생산 ② `blocked_by` 자가진단 ③ 사다리 전 조합 소진 ④ 라벨 이름 위생.
    """
    problems = []
    try:
        verify_registered(verbose=verbose)
    except AssertionError as e:                                   # noqa: BLE001
        problems.append(str(e))
    missed, folded = selfcheck_blocked_by()
    if missed or folded:
        problems.append(f'blocked_by 미포착 {missed} · 접힘 {folded}')
    lad = ladder_table()
    holes = [kw for kw, lb in lad if lb not in LABELS]
    if holes:
        problems.append(f'사다리 구멍 {holes[:3]}')
    emitted = {lb for _kw, lb in lad}
    if emitted != set(LABELS):
        problems.append(f'사다리가 못 내는 라벨 {sorted(set(LABELS) - emitted)}')
    # ★라벨 이름에 `G-` 부분문자열이 있으면 `check_prereg.GATE_RE` 가 **없는 게이트**로 읽는다
    #   (사전등록 §13 이 그 사유로 라벨을 한 번 개명했다).
    ghost = [lb for lb in LABELS if 'G-' in lb]
    if ghost:
        problems.append(f'라벨에 `G-` 부분문자열 {ghost}')
    ghost_g = [g for g in GATE_NAMES if GATE_NAMES.count(g) > 1]
    if ghost_g:
        problems.append(f'게이트 이름 중복 {sorted(set(ghost_g))}')
    if verbose:
        print(f'  [OK] 사다리 조합 {len(lad)} · 라벨 {len(emitted)}종 · 구멍 0')
        print(f'  [OK] 게이트 {len(GATE_NAMES)}개 (차단 {len(GATES_BLOCKING)} · '
              f'기록 {len(GATES_RECORD)}) · blocked_by 전건 포착')
    if problems:
        for p in problems:
            print('  [FAIL] ' + p)
        return False, problems
    return True, []


if __name__ == '__main__':
    ok, _ = selftest(verbose=True)
    d = design_constants()
    print('\n══ 설계 상수 (동결 행 재계산 · 추정기 런 0회) ═══════════════════')
    print(f"  MED_Q1 {d['MED_Q1']:.4f}  SD_Q1 {d['SD_Q1']:.4f}  >0 {d['Q1_POS']}/{d['n']}")
    print(f"  MED_PR {d['MED_PR']:.4f}  SD_PR {d['SD_PR']:.4f}  >0 {d['PR_POS']}/{d['n']}")
    print(f"  RHO {d['RHO']:.4f}  CEIL_MED {d['CEIL_MED']:.3f}  CEIL_MAX {d['CEIL_MAX']:.3f}")
    print(f"  가법 널 예측  median {d['ADD_NULL_MED']:.3f}  "
          f"> 에피소드별 최대 {d['ADD_OVER_EPCEIL']}/{d['n']}")
    q_reg, q_syn = q_registered()
    print(f'\n══ 라벨 확률 (q_I = {q_reg:.6f}) ══════════════════════════════')
    for k, v in label_probs().items():
        if not k.startswith('_'):
            print(f'  {k:45s} {v:.6f}')
    print(f'  ★H_syn(q={q_syn:.6f}) 아래 검출확률 = {detect_power_syn():.6f}')
    print(f"\n══ 비용 ══════════════════════════════════════════════════════")
    pc = projected_cost()
    print(f"  {pc['total_runs']} 런 · {pc['seconds']:.0f} s = {pc['hours']:.2f} 시간 "
          f"(1스레드 직렬) · 단가 {pc['unit_costs']}")
    sys.exit(0 if ok else 1)
