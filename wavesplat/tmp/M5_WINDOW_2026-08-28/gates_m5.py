"""M-5 단일출처 — ★**관측 창 길이 `Tobs`(= 시간축 개구)가 실측 경계를 매개하는가.**

소거표(2026-08-28)에서 남은 다섯 후보 중 **② 창 길이/시간축**을 잰다.
⛔이 파일이 상수·팔·문턱·게이트·라벨·**결과 전 해석 규칙**의 **유일 정의**다.
`PREREG.md` 는 이 파일의 산문 번역이고 어긋나면 **코드가 이긴다**.

★**손복사 상수 0건**: `W`·`m_scan`·`m_cap` 는 `scene_station()` 시그니처에서 **되읽고**,
배포 창 규약 `tp_factor` 는 `wavesplat/run.py` 소스에서, 사전의 학습 창 분포 `TOBS_FACTOR`
와 `M_OBS` 는 `gen_mixed.py`·`d3_beamform_net.py` 소스에서 **정적 파싱**한다
(⛔`gen_mixed`·`d3_beamform_net` 은 **임포트하지 않는다** — 최상위에서 CUDA 를 건드린다).
레코드·창·시드·잡음은 `gates_b3.py` 에서 **임포트**한다(B-7 의 선례, `gates_b7.py:43`).

────────────────────────────────────────────────────────────────────────────
★**왜 이 축인가 — 코드에서 도출한 미공시 불일치**

  · 배포 **합성** 경로:  `wavesplat/run.py` → `loader.get_init_window(tp_factor=2.0)`
        ⟹ 관측 창 = **2.0 × Tp**
  · 배포 사전의 **학습** 분포: `gen_mixed.py:TOBS_FACTOR = (1.9, 2.3)`, `Tobs = U(1.9,2.3)·Tp`
        ⟹ 학습 창 = **1.9–2.3 × Tp**
  · 실측 하네스: `scene_station(W=40.0)` **초 고정**
        ⟹ 12 레코드 중앙 **11.80 × Tp** (범위 6.54–20.43; `premise_m5.json:P2`)

⟹ ★**실측 팔은 사전이 한 번도 본 적 없는 시간 개구에서 평가돼 왔다**(중앙 5.4배 밖,
   12/12 레코드 전부). **어느 배치도 이 값을 움직인 적이 없다** — `scene_station` 의
   호출자 15곳이 전부 기본값 `W=40.0` 을 쓴다.

★**도달성(기제까지 확인)**: 사전의 조건자는 `d3_beamform_net.beamform()` 이고
`phi = kx·x + ky·y − ω(k)·t` 를 쌓는 정합 필터다. 그 지도의 k-공간 주엽 폭은
**1/T_obs 로 좁아진다** ⟹ 창 길이는 CNN 입력 영상의 점퍼짐함수를 바꾼다.
★그리고 `learned_init.propose_atoms` 는 지도를 만들기 **직전에**
`t0 = P[:,2].min(); Ps[:,2] -= t0` 으로 **원점을 스스로 정규화한다** ⟹
***조건자가 시간축에서 보는 양은 오직 관측 시간 **폭** 하나이고, 그것이 바로 이 배치가
움직이는 변수다.*** (학습은 `ot = uniform(0, Tobs)`, `Tobs = U(1.9,2.3)·Tp`.)

★**단일요인이 구조적으로 보장된다** (이 배치의 최대 강점):
`scene_station` 안에서 `W` 는 **오직** `m = P_all[:,2] >= t0 - W` 한 줄에만 들어간다.
`Zmem`(따라서 `Hs`·`kp`·**채점 분모 `sg`**)·`Zfut`·`maps_gt`·`Pq`·`leads`·`gxq/gyq` 는
`W_max=40.0` 로 고정돼 **전혀 움직이지 않는다**. 그리고
  · 실측 팔(bank): `bank_max_obs=600` — 장면이 이미 `m_cap=600` 으로 자른다
  · 제안 팔(learned): `learned_init.py:65 M_OBS_CAP=400` 이 **다시** 400으로 자른다
⟹ **모든 수준에서 `n_obs` 가 정확히 600(사전은 400)으로 불변**이고, 움직이는 것은
   **그 점들이 깔린 시간 폭 하나뿐**이다. `G-2`·`G-2b` 가 이것을 행에서 강제한다.
   (전제: 잘라낸 뒤에도 cap 이전 후보가 600 이상 — `premise_m5.py:P1` 이 미리 확인한다.)

────────────────────────────────────────────────────────────────────────────
⛔**이 배치가 주장하지 않는 것 (결과 전 공시)**

 1. ⛔**기제 지목 금지.** `DIRECTION = ELIMINATION-OR-LOCATION`. 양성이 나와도 산출은
    *"경계가 이 축 위에 놓여 있다"* 까지이고 *"원인은 X 다"* 로 올리지 않는다.
 2. ⛔**배포 권고 금지.** 용량(dose)을 정하는 `Tp` 는 **조밀한 참 표면**에서 온
    `kp`(`scene_station:101 directional_spread(Zmem[-1])`)로 계산된다 = **GT 유래**다.
    ⟹ *"배포에서 W 를 2Tp 로 하라"* 는 이 배치로 말할 수 **없다**. (`B-25` 의 교훈.)
    ✅그 대신 **GT-free 거울** `SEC8`(초 고정)을 같이 돌려 용량 정의가 결과를 만들지
    않았음을 보인다.
 3. ⛔**②(사전 지지집합)와 ③(정상성)을 분리한다고 주장 금지.** 창을 줄이면 학습 지지집합에
    가까워지는 **동시에** 비정상성 노출이 준다. `SHIP`(사전 없는 bank 경로)이 부분
    판별자일 뿐이고, 그것도 M-2(시간 원점 거리)와는 분리하지 못한다 — `ORIG2` 가 그 몫이다.
 4. ⛔**사이트 수준 일반화 금지**(사이트 4개). ⛔**외부 기준선 추가 금지**(정지규칙).
 5. ⛔`ZERO_REF=1.0` 을 쓰지 않는다. 살아 있는 것은 **짝 비교**이고, 참 제로필드는
    `premise_m5.json:P5` 에서 (record,win) 별로 다시 계산해 **부수 서술**로만 쓴다.
"""
import ast
import json
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT,
           os.path.join(ROOT, 'scripts'),
           os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'B3_REFIT_2026-08-23')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 표본·창·시드·잡음: `gates_b3.py` 단일출처 (손복사 0) ─────────────────────
from gates_b3 import (RECORDS, WINDOWS, SEED, NOISE,          # noqa: E402
                      sign_test_one_sided)

BATCH = 'M5_WINDOW_2026-08-28'
DIRECTION = 'ELIMINATION-OR-LOCATION'
FIELD = 'nowcast'                 # lead 0. `leadmean` 은 사전등록된 부차 (SECONDARY-LEAD)
G_ACC = 9.81


# ── 배포·학습 규약 상수: 소스에서 되읽는다 ──────────────────────────────────
def _sig_defaults():
    """`scene_station()` 시그니처에서 `W`·`m_scan`·`m_cap`·`noise` 기본값을 되읽는다."""
    import inspect
    from scene_station import scene_station
    p = inspect.signature(scene_station).parameters
    return {k: p[k].default for k in ('W', 'm_scan', 'm_cap', 'noise')}


def _static_const(path, name, kind='num'):
    """소스를 **임포트하지 않고** 최상위 대입 `name = <literal>` 을 읽는다.

    ⛔`gen_mixed.py`·`d3_beamform_net.py` 는 최상위에서 CUDA/에피소드를 건드리므로
    임포트 금지다. `ast` 로 정적 파싱한다."""
    src = open(os.path.join(ROOT, path)).read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(node.value)
    raise KeyError(f'{path}: 최상위 상수 {name} 없음')


def _deploy_tp_factor():
    """`wavesplat/run.py` 의 배포 초기화 창 = `get_init_window(tp_factor=?)`."""
    src = open(os.path.join(ROOT, 'wavesplat', 'run.py')).read()
    m = re.findall(r'get_init_window\(tp_factor=([0-9.]+)\)', src)
    if not m:
        raise KeyError('run.py: get_init_window(tp_factor=..) 호출 없음')
    if len(set(m)) != 1:
        raise ValueError(f'run.py 안에 tp_factor 가 여러 값이다: {sorted(set(m))}')
    return float(m[0])


SCENE_DEFAULTS = _sig_defaults()
W_ANCHOR = float(SCENE_DEFAULTS['W'])              # 40.0  (실측 하네스 규약)
M_SCAN = int(SCENE_DEFAULTS['m_scan'])             # 12    (⛔건드리지 않는다)
M_CAP = int(SCENE_DEFAULTS['m_cap'])               # 600
assert float(SCENE_DEFAULTS['noise']) == float(NOISE), 'scene 기본 잡음 != gates_b3.NOISE'

TP_FACTOR_DEPLOY = _deploy_tp_factor()                                        # 2.0
TOBS_FACTOR_TRAIN = tuple(_static_const(
    'tmp/mixed_retrain_2026-07-26/gen_mixed.py', 'TOBS_FACTOR'))              # (1.9, 2.3)
M_OBS_TRAIN = int(_static_const(
    'tmp/mixed_retrain_2026-07-26/gen_mixed.py', 'M_OBS'))                    # 400
M_OBS_COND = int(_static_const('scripts/d3_beamform_net.py', 'M_OBS'))        # 400
M_OBS_CAP_LEARNED = int(_static_const(
    'wavesplat/optimizer/learned_init.py', 'M_OBS_CAP'))                      # 400

# 배포 창 계수가 학습 지지집합 안에 있는지 — 전제 확인(문서 주장의 근거)
assert TOBS_FACTOR_TRAIN[0] <= TP_FACTOR_DEPLOY <= TOBS_FACTOR_TRAIN[1]
assert M_OBS_TRAIN == M_OBS_COND == M_OBS_CAP_LEARNED


# ── 수준(dose) 정의 ─────────────────────────────────────────────────────────
# `kind`: 'sec' = W 를 초로 고정 · 'tp' = W = factor × Tp(record,win) · 'noop' = 무필터
# `origin`: 'min' = `run_b2.py:29` 규약(t_shift = P[:,2].min())
#           'pred_minus'= t_shift 를 **예측 시각에서 factor·Tp 앞**에 둔다(개구는 그대로)
LEVELS = {
    'ANCHOR': dict(kind='sec', value=W_ANCHOR, origin='min',
                   why='실측 하네스 규약(B-2…B-10 전부). 기준 팔.'),
    'TP6':    dict(kind='tp',  value=6.0, origin='min', why='용량-반응 중간'),
    'TP4':    dict(kind='tp',  value=4.0, origin='min', why='용량-반응 중간'),
    'TP2':    dict(kind='tp',  value=TP_FACTOR_DEPLOY, origin='min',
                   why='★배포 합성 규약 = 학습 지지집합 중앙. PRIMARY 대비.'),
    'SEC8':   dict(kind='sec', value=8.0, origin='min',
                   why='★GT-free 거울 — 용량이 GT 유래 Tp 로 정의된 것이 결과를 '
                       '만들지 않았음을 보인다. 8 s 는 레코드 중앙 Tp 의 약 2.4배.'),
    'ORIG2':  dict(kind='sec', value=W_ANCHOR, origin='pred_minus',
                   why='★시간 **원점** 불변 대조 — 관측 집합은 ANCHOR 와 비트동일하고 '
                       '예측 시각까지의 거리만 TP2 와 같게 맞춘다. 조건자는 원점을 스스로 '
                       '정규화하므로(learned_init.propose_atoms) 이 수준은 **적합·외삽 '
                       '단계**만 건드린다. 자유 위상 φ 가 흡수해야 하므로 ≈0 이 기대값이고, '
                       '움직이면 그 자체가 결함이다.'),
}
ORIG2_TP_FACTOR = TP_FACTOR_DEPLOY   # ORIG2 의 원점 거리 = TP2 와 같은 2.0 Tp
LEVEL_ORDER = ['ANCHOR', 'TP6', 'TP4', 'TP2', 'SEC8', 'ORIG2']
DOSE_LADDER = ['ANCHOR', 'TP6', 'TP4', 'TP2']        # 창이 짧아지는 순서

# win=3 에서만 도는 **무연산 대조**(비트동일이어야 한다)
NOOP_LEVEL = 'NOOP'
NOOP_SPEC = dict(kind='noop', value=1e9, origin='min',
                 why='W 배관이 작동하지 않아야 할 때 정말 작동하지 않는가.')
NOOP_WIN = 3

ARMS = {
    'SHIP': {},                                                     # 실측 게이트 → bank_mmse
    'PROP': {'real_domain_gate': False, 'init_method': 'learned'},   # ★제안 표현 강제
}
ARM_ORDER = ['SHIP', 'PROP']
PRIMARY_ARM = 'PROP'
PRIMARY_WIN = 3                        # ⛔n=36 을 PRIMARY 로 쓰지 않는다(가짜복제 금지)
PRIMARY_LEVEL = 'TP2'
REF_LEVEL = 'ANCHOR'

N_PRIMARY = len(RECORDS)                                   # 12
THRESHOLD_12 = 10                      # /12 단측 부호검정 p=0.0193 (결과 전 확정)
N_SECONDARY = len(RECORDS) * len(WINDOWS)                  # 36
THRESHOLD_36 = 24                      # /36 단측 p=0.0326 (결과 전 확정)

N_RUNS = (len(LEVELS) * len(ARMS) * len(RECORDS) * len(WINDOWS)
          + len(ARMS) * len(RECORDS))                      # + NOOP(win=3)

# ORIG2 의 bank-팔 불변 허용오차 (위상 자유도로 흡수되어야 한다). 결과 전 고정.
ORIG2_SHIP_TOL = 1e-3                  # 상대. 넘으면 `G-7` FAIL (SCOPING — 아래)
BITEQ_KEYS = ('obs_hash', 'q_hash')    # 장면 수준 비트동일에 쓰는 키

# ★결과 전 선언: 게이트를 **차단(BLOCKING)** 과 **범위축소(SCOPING)** 로 나눈다.
#   · BLOCKING 이 하나라도 FAIL 이면 `analyze_m5.py` 는 **판정을 내지 않는다.**
#   · SCOPING 이 FAIL 이면 **그 수준만 해석에서 빠지고** PRIMARY 는 그대로 읽는다.
#   ⛔결과를 보고 이 분류를 바꾸지 않는다.
SCOPING = {'G-7': 'ORIG2 수준만 해석 불가로 선언한다 (PRIMARY 는 TP2 vs ANCHOR 라 무관)'}

# 외부 동결 참조 (★새 교훈 ①: 사전등록 시점에 **실제 파일에 대고** 비지 않음을 확인했다.
#  확인 방법 = `python3 gates_m5.py --refcheck` 및 `tmp/_audit/check_gate_refs.py`)
FROZEN_B2 = os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23', 'results.jsonl')
FROZEN_B2_WIN = 3                      # `gates_b2.WIN`


def _load_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def b2_scene_ref():
    """B-2 동결 행에서 (record → obs_hash) 를 낸다. ★비면 빈 dict 을 낸다(공허 PASS 금지).

    ⛔`bool(ref) and …` 로만 쓴다 — 참조가 비면 게이트는 **FAIL** 이어야 한다."""
    ref = {}
    for r in _load_jsonl(FROZEN_B2):
        if 'error' in r or r.get('win') != FROZEN_B2_WIN:
            continue
        h = r.get('obs_hash')
        if h:
            ref.setdefault(r['record'], h)
    return ref


# ── 집계 ────────────────────────────────────────────────────────────────────
def key(r):
    return (r.get('level'), r.get('cell'), r.get('record'), r.get('win'))


def _pick(rows, level, arm, field=FIELD, wins=None):
    wins = WINDOWS if wins is None else wins
    return {(r['record'], r['win']): r[field] for r in rows
            if 'error' not in r and r.get('level') == level and r.get('cell') == arm
            and r.get('win') in wins and r.get(field) is not None}


def paired(rows, level, ref_level, arm, field=FIELD, wins=None):
    """`level` 이 `ref_level` 보다 **낮으면**(오차가 작으면) 이긴 것으로 센다."""
    a = _pick(rows, level, arm, field, wins)
    b = _pick(rows, ref_level, arm, field, wins)
    ks = sorted(set(a) & set(b))
    n = len(ks)
    w = sum(1 for k in ks if a[k] < b[k])
    return {'level': level, 'ref': ref_level, 'arm': arm, 'field': field,
            'n': n, 'better': w,
            'rel_median_pct': (st.median([(a[k] - b[k]) / b[k] * 100 for k in ks])
                               if ks else None),
            'p_one_sided': sign_test_one_sided(w, n) if n else None}


def dose_curve(rows, arm, field=FIELD, wins=None):
    """창이 짧아지는 순서로 ANCHOR 대비 승수를 낸다."""
    return [paired(rows, lv, REF_LEVEL, arm, field, wins) for lv in DOSE_LADDER[1:]]


# ── 게이트 ──────────────────────────────────────────────────────────────────
def gates(rows):
    ok = [r for r in rows if 'error' not in r]
    out = []
    A = out.append

    A(('G-0 런 수', len(ok) == N_RUNS, f'{len(ok)}/{N_RUNS}'))
    A(('G-0b 에러 0', len(rows) == len(ok), f'{len(rows) - len(ok)} 에러'))
    A(('G-0c 셀 중복 0', len({key(r) for r in ok}) == len(ok),
       f'{len(ok) - len({key(r) for r in ok})} 중복'))

    # ── G-1 개입 되읽기 ⛔항등식 아님: 설정값이 아니라 **관측 배열**에서 다시 잰다 ──
    bad = []
    for r in ok:
        lv, w_used, span = r.get('level'), r.get('W_used'), r.get('t_span')
        if lv is None or w_used is None or span is None:
            bad.append((lv, r.get('record'), 'missing')); continue
        if lv == NOOP_LEVEL:
            continue
        dt = r.get('dt') or 0.0
        # 관측이 실제로 그 폭 안에 있는가 (위) + 그 폭을 실제로 채웠는가 (아래)
        if not (span <= w_used + 2 * dt + 1e-9 and span >= 0.90 * min(w_used, W_ANCHOR)):
            bad.append((lv, r.get('record'), round(span, 3), round(w_used, 3)))
    A(('G-1 개입 되읽기: 관측 시간폭이 W 를 따라 움직였다(설정값 아님)',
       not bad, str(bad[:4])))

    # G-1b 자르기는 **과거 쪽에서만** 일어났다 — 끝 시각은 모든 수준에서 같다
    bad = []
    for (rec, w) in {(r['record'], r['win']) for r in ok}:
        ends = {round(r.get('t_end_abs', float('nan')), 9) for r in ok
                if r['record'] == rec and r['win'] == w}
        if len(ends) > 1:
            bad.append((rec, w, sorted(ends)))
    A(('G-1b 자르기는 과거 쪽에서만 (관측 끝 시각 불변)', not bad, str(bad[:2])))

    # ── G-2 ★단일요인 보장: 관측 **개수**가 모든 수준에서 같다 ──────────────
    counts = sorted({r.get('n_obs') for r in ok})
    A((f'G-2 n_obs 가 전 수준에서 {M_CAP} 불변(개수 교락 없음)',
       counts == [M_CAP], str(counts)))
    A((f'G-2a 제안 팔 조건자 입력 {M_OBS_CAP_LEARNED} 불변',
       all(r.get('n_obs_learned') in (None, M_OBS_CAP_LEARNED)
           for r in ok if r.get('cell') == 'PROP'), ''))

    # ── G-2b ★채점·질의·참값이 수준에 걸쳐 비트동일 ────────────────────────
    bad = []
    for (rec, w) in {(r['record'], r['win']) for r in ok}:
        grp = [r for r in ok if r['record'] == rec and r['win'] == w]
        for k in ('q_hash', 'gt_hash', 'sigma_bits', 'kp_bits'):
            if len({r.get(k) for r in grp}) > 1:
                bad.append((rec, w, k))
    A(('G-2b 채점 분모·질의·참값·kp 가 수준에 걸쳐 비트동일', not bad, str(bad[:4])))

    # ── G-3 팔이 서로 달랐다 (B-2 의 G-1/G-2 계승) ──────────────────────────
    prop = [r for r in ok if r['cell'] == 'PROP']
    ship = [r for r in ok if r['cell'] == 'SHIP']
    A(('G-3 PROP 이 성분을 만들었다', bool(prop) and all((r.get('n_atoms') or 0) > 0
                                                     for r in prop),
       f'{sum(1 for r in prop if (r.get("n_atoms") or 0) > 0)}/{len(prop)}'))
    A(('G-3b PROP 전건 VARPRO 발화', bool(prop) and all(r.get('varpro_fired')
                                                      for r in prop),
       f'{sum(1 for r in prop if r.get("varpro_fired"))}/{len(prop)}'))
    A(('G-3c SHIP 은 bank 경로(성분 0 · real_gate True)',
       bool(ship) and all((r.get('n_atoms') or 0) == 0 and r.get('real_gate') is True
                          for r in ship), f'{len(ship)}'))

    # ── G-4 ★양성대조(비트동일): NOOP == ANCHOR ────────────────────────────
    bad = []
    for arm in ARM_ORDER:
        for rec in RECORDS:
            a = [r for r in ok if r['level'] == NOOP_LEVEL and r['cell'] == arm
                 and r['record'] == rec and r['win'] == NOOP_WIN]
            b = [r for r in ok if r['level'] == 'ANCHOR' and r['cell'] == arm
                 and r['record'] == rec and r['win'] == NOOP_WIN]
            if len(a) != 1 or len(b) != 1:
                bad.append((arm, rec, 'missing')); continue
            for k in BITEQ_KEYS + ('nowcast_bits', 'leadmean_bits'):
                if a[0].get(k) != b[0].get(k):
                    bad.append((arm, rec, k))
    A(('G-4 양성대조: 무연산 W 는 ANCHOR 와 **비트동일**', not bad, str(bad[:4])))

    # ── G-5 ★외부 동결 참조 (⛔`bool(ref) and …` — 참조가 비면 FAIL) ────────
    ref = b2_scene_ref()
    mine = {r['record']: r.get('obs_hash') for r in ok
            if r['level'] == 'ANCHOR' and r['win'] == FROZEN_B2_WIN}
    common = sorted(set(ref) & set(mine))
    A(('G-5 외부 참조가 비지 않았다 (B-2 동결 행)',
       bool(ref) and len(ref) == len(RECORDS), f'ref={len(ref)}/{len(RECORDS)}'))
    A(('G-5b ANCHOR 장면이 B-2 동결 장면과 비트동일',
       bool(ref) and bool(common) and len(common) == len(RECORDS)
       and all(ref[k] == mine[k] for k in common),
       f'{sum(1 for k in common if ref.get(k) == mine.get(k))}/{len(RECORDS)}'))

    # ── G-6 배포 상수 되읽기 (⛔의도표 금지 — 객체에서 잡은 값) ──────────────
    bad = [(r.get('level'), r.get('record'), r.get('cap_init_method'))
           for r in ok if r.get('cell') == 'PROP'
           and r.get('cap_init_method') != 'learned']
    A(('G-6 PROP 의 init_method 를 **객체**에서 되읽으면 learned', not bad, str(bad[:3])))
    lam = {r.get('cap_lambda_l1') for r in ok}
    A(('G-6b lambda_l1 이 전 런에서 하나', len(lam) == 1, str(lam)))

    # ── G-7 ORIG2 의 bank-팔 불변 대조 (위상 자유도가 흡수해야 한다) ─────────
    a = _pick(rows, 'ORIG2', 'SHIP')
    b = _pick(rows, 'ANCHOR', 'SHIP')
    ks = sorted(set(a) & set(b))
    dev = max((abs(a[k] - b[k]) / b[k] for k in ks), default=None)
    A(('G-7 ORIG2 는 bank 팔에서 시간원점 불변', bool(ks) and dev is not None
       and dev <= ORIG2_SHIP_TOL, f'max rel dev={dev}'))

    # ── G-8 표본 완전성 ─────────────────────────────────────────────────────
    A(('G-8 12 레코드 × 3 창 전건',
       {(r['record'], r['win']) for r in ok if r['level'] == 'ANCHOR'}
       == {(rec, w) for rec in RECORDS for w in WINDOWS}, ''))
    A(('G-8b 전제 파일 존재', os.path.exists(os.path.join(HERE, 'premise_m5.json')), ''))
    A(('G-8c 전제가 전 수준 허용성을 미리 확인했다', _premise_admissible(), ''))
    return out


def _premise_admissible():
    p = os.path.join(HERE, 'premise_m5.json')
    if not os.path.exists(p):
        return False
    d = json.load(open(p)).get('P1_level_admissibility') or {}
    return bool(d) and bool(d.get('all_admissible'))


def blocking_failures(gate_rows):
    """SCOPING 으로 선언된 것을 뺀 FAIL 목록. 비어 있지 않으면 판정 금지."""
    return [gid for gid, ok, _ in gate_rows
            if not ok and gid.split()[0] not in SCOPING]


# ── 결과 전 해석 규칙 (네 갈래 전부 결과 전에 적는다) ────────────────────────
def label(primary, discrim, secondary, dose):
    """PRIMARY = `PROP`, win=3, n=12, `TP2` vs `ANCHOR`, `nowcast`, 문턱 10/12.

    `discrim` = 같은 대비의 `SHIP` 팔. `secondary` = 같은 대비의 n=36.
    ⛔`secondary` 는 **PRIMARY 를 승격시키지 못한다** — 무효화(UNDECIDED)만 시킨다."""
    w, n = primary['better'], primary['n']
    ws = discrim['better']
    mono = [d['better'] for d in dose]
    dose_ok = all(x >= y for x, y in zip(mono, mono[1:])) if len(mono) > 1 else False

    if w >= THRESHOLD_12:
        if ws >= THRESHOLD_12:
            return ('WINDOW-EARNS-BOTH-ARMS',
                    f'창을 배포 규약({TP_FACTOR_DEPLOY}·Tp)으로 줄이면 **두 팔 다** '
                    f'좋아진다(PROP {w}/{n} p={primary["p_one_sided"]:.4f}, '
                    f'SHIP {ws}/{discrim["n"]}). ⟹ 경계는 창 길이 축 위에 놓여 있으나 '
                    '**사전에 특이하지 않다** — ⛔사전 지지집합(②)과 비정상성(③)을 '
                    '가르지 못한다. 용량-반응 단조=' + str(dose_ok) + '.')
        return ('WINDOW-EARNS-PRIOR-ARM',
                f'★제안 표현 팔만 좋아진다(PROP {w}/{n} p={primary["p_one_sided"]:.4f}; '
                f'SHIP {ws}/{discrim["n"]} 은 같은 문턱을 **넘지 못했다**). '
                '⟹ 경계가 창 길이 축 위에 놓여 있고 효과가 **사전을 지닌 팔에 몰려 있다**. '
                '⛔"SHIP 은 영향이 없다" 로 읽지 마라(등가검정 안 했다). '
                '용량-반응 단조=' + str(dose_ok) + '.')
    if (n - w) >= THRESHOLD_12:
        return ('WINDOW-LONGER-IS-BETTER',
                f'창을 줄이면 **나빠진다**(PROP {w}/{n}). ⟹ 실측 팔은 긴 시간 개구를 '
                '쓰고 있고, *"학습 지지집합 밖 창이 손해"* 가설은 **반증**된다. '
                '소거표에 그 방향으로 한 줄이 추가된다.')
    if secondary['better'] >= THRESHOLD_36 or (secondary['n'] - secondary['better']) >= THRESHOLD_36:
        return ('WINDOW-UNDECIDED (UNDERPOWERED)',
                f'PRIMARY 는 갈리지 않았는데({w}/{n}) n=36 층은 갈린다'
                f'({secondary["better"]}/{secondary["n"]}). ⛔n=36 은 창 3개를 겹쳐 센 '
                '가짜복제라 승격 근거가 아니다 ⟹ **소거로 읽지 않는다**. '
                '다음은 창을 늘리는 것이 아니라 **레코드**를 늘리는 새 사전등록이다.')
    return ('WINDOW-DOES-NOT-MEDIATE',
            f'어느 수준에서도 갈리지 않는다(PROP {w}/{n}, SHIP {ws}/{discrim["n"]}, '
            f'n=36 {secondary["better"]}/{secondary["n"]}, 용량-반응 단조={dose_ok}). '
            '⟹ ★**소거표에 「관측 창 길이/시간축」 행이 추가된다.** '
            '그리고 현재 살아 있는 교락 하나가 은퇴한다 — 실측 음성은 '
            f'*"하네스가 사전에게 학습 지지집합의 {W_ANCHOR / 2:.0f}배 창을 먹였기 때문"* '
            '이 아니다. ⛔"차이가 없다" 로 쓰지 마라(등가검정 안 했다).')


# ── 자가진단 + 변형 하네스 ──────────────────────────────────────────────────
def _fixture(prop_better, ship_better, n_rec=None, seed=0):
    """네 라벨 분기를 전부 때리는 최소 픽스처. ⛔판정 자료가 아니다."""
    recs = RECORDS if n_rec is None else RECORDS[:n_rec]
    rows = []
    for wi, w in enumerate(WINDOWS):
        for ri, rec in enumerate(recs):
            for arm in ARM_ORDER:
                better = prop_better if arm == 'PROP' else ship_better
                for lv in list(LEVELS) + [NOOP_LEVEL]:
                    if lv == NOOP_LEVEL and w != NOOP_WIN:
                        continue
                    base = 1.0 + 0.01 * ri
                    if lv in ('TP2', 'SEC8'):
                        base -= 0.05 if ri < better else -0.05
                    elif lv == 'TP4':
                        base -= 0.03 if ri < better else -0.03
                    elif lv == 'TP6':
                        base -= 0.01 if ri < better else -0.01
                    rows.append(dict(
                        level=lv, cell=arm, record=rec, win=w, nowcast=base,
                        leadmean=base * 1.1, n_obs=M_CAP,
                        n_obs_learned=(M_OBS_CAP_LEARNED if arm == 'PROP' else None),
                        W_used=(W_ANCHOR if lv in ('ANCHOR', 'ORIG2', NOOP_LEVEL)
                                else 8.0), t_span=W_ANCHOR - 0.1 if lv in
                        ('ANCHOR', 'ORIG2', NOOP_LEVEL) else 7.9,
                        dt=0.1, t_end_abs=100.0,
                        q_hash=f'q{rec}{w}', gt_hash=f'g{rec}{w}',
                        sigma_bits=f's{rec}{w}', kp_bits=f'k{rec}{w}',
                        obs_hash=f'o{rec}{w}{lv}', nowcast_bits=f'n{rec}{w}{lv}{arm}',
                        leadmean_bits=f'l{rec}{w}{lv}{arm}',
                        n_atoms=(20 if arm == 'PROP' else 0),
                        varpro_fired=(arm == 'PROP'),
                        real_gate=(True if arm == 'SHIP' else None),
                        cap_init_method=('learned' if arm == 'PROP' else 'bank_mmse'),
                        cap_lambda_l1=1e-4))
    # ★ANCHOR(win=3) 의 장면 해시는 **실제 B-2 동결 행**에서 가져온다 — 그래야 픽스처가
    # `G-5b`(외부 참조 비트동일)를 진짜로 통과/실패시킨다(가짜 통과 금지).
    _ref = b2_scene_ref()
    for r in rows:
        if r['level'] == 'ANCHOR' and r['win'] == FROZEN_B2_WIN and r['record'] in _ref:
            r['obs_hash'] = _ref[r['record']]
    # NOOP 은 ANCHOR 와 비트동일해야 한다
    _anchor = {(r['record'], r['win'], r['cell']): r for r in rows
               if r['level'] == 'ANCHOR'}
    for r in rows:
        if r['level'] == NOOP_LEVEL:
            src = _anchor[(r['record'], r['win'], r['cell'])]
            for k in BITEQ_KEYS + ('nowcast_bits', 'leadmean_bits'):
                r[k] = src[k]
    # ORIG2 의 SHIP 팔은 ANCHOR 와 같아야 한다(G-7)
    anch = {(r['record'], r['win']): r['nowcast'] for r in rows
            if r['level'] == 'ANCHOR' and r['cell'] == 'SHIP'}
    for r in rows:
        if r['level'] == 'ORIG2' and r['cell'] == 'SHIP':
            r['nowcast'] = anch[(r['record'], r['win'])]
    return rows


def _verdict_from(rows, wins=None):
    p = paired(rows, PRIMARY_LEVEL, REF_LEVEL, PRIMARY_ARM, wins=[PRIMARY_WIN])
    d = paired(rows, PRIMARY_LEVEL, REF_LEVEL, 'SHIP', wins=[PRIMARY_WIN])
    s = paired(rows, PRIMARY_LEVEL, REF_LEVEL, PRIMARY_ARM, wins=WINDOWS)
    dose = dose_curve(rows, PRIMARY_ARM, wins=[PRIMARY_WIN])
    return label(p, d, s, dose), p, d, s, dose


def selftest():
    fails = []

    def chk(cond, msg):
        if not cond:
            fails.append(msg)

    # ① 네 라벨 분기가 전부 도달 가능한가
    chk(_verdict_from(_fixture(12, 0))[0][0] == 'WINDOW-EARNS-PRIOR-ARM', 'branch A')
    chk(_verdict_from(_fixture(12, 12))[0][0] == 'WINDOW-EARNS-BOTH-ARMS', 'branch B')
    chk(_verdict_from(_fixture(0, 0))[0][0] == 'WINDOW-LONGER-IS-BETTER', 'branch C')
    chk(_verdict_from(_fixture(6, 6))[0][0] == 'WINDOW-DOES-NOT-MEDIATE', 'branch D')

    # ② 문턱이 정확히 그 자리에 있다 (9 는 안 되고 10 은 된다)
    chk(_verdict_from(_fixture(9, 0))[0][0] != 'WINDOW-EARNS-PRIOR-ARM', 'thresh 9')
    chk(_verdict_from(_fixture(10, 0))[0][0] == 'WINDOW-EARNS-PRIOR-ARM', 'thresh 10')
    chk(abs(sign_test_one_sided(10, 12) - 0.01928710937) < 1e-9, 'p(10,12)')
    chk(sign_test_one_sided(24, 36) < 0.05 <= sign_test_one_sided(23, 36), 'p(24,36)')

    # ③ 게이트가 깨끗한 픽스처를 통과한다
    good = _fixture(6, 6)
    g = dict((k.split()[0], ok) for k, ok, _ in gates(good))
    # G-8b/G-8c 는 전제 파일의 **존재·내용**을 보는 진짜 검사다 — 픽스처가 만들 수 없다.
    _env = ('G-8b', 'G-8c')
    chk(all(v for k, v in g.items() if k not in _env),
        'clean fixture gates: ' + str([k for k, v in g.items() if not v and k not in _env]))
    chk(g['G-8b'] == os.path.exists(os.path.join(HERE, 'premise_m5.json')),
        'G-8b must mirror premise_m5.json existence')
    chk(g['G-8c'] == _premise_admissible(), 'G-8c must mirror premise admissibility')

    # ④b SCOPING 분류가 실제로 작동한다 (G-7 은 판정을 막지 않는다)
    rows7 = _fixture(6, 6)
    [r.__setitem__('nowcast', 9.0) for r in rows7
     if r['level'] == 'ORIG2' and r['cell'] == 'SHIP'][:1]
    gr = gates(rows7)
    chk('G-7' not in [x.split()[0] for x in blocking_failures(gr)],
        'G-7 must be SCOPING, not blocking')
    chk(any(gid.startswith('G-7') and not ok for gid, ok, _ in gr), 'G-7 must FAIL here')

    # ④ ★변형 하네스 — 결함을 하나씩 심으면 **그 게이트가 반드시 FAIL** 해야 한다
    def mutate(fn, gate_id):
        rows = _fixture(6, 6)
        fn(rows)
        gg = dict((k.split()[0], ok) for k, ok, _ in gates(rows))
        chk(gg.get(gate_id) is False, f'mutation not caught by {gate_id}')

    mutate(lambda rs: rs.pop(), 'G-0')
    mutate(lambda rs: rs.append(dict(rs[0])), 'G-0c')
    mutate(lambda rs: rs[0].__setitem__('t_span', 39.0)
           if rs[0]['level'] != 'ANCHOR' else rs[0].__setitem__('t_span', 1.0), 'G-1')
    mutate(lambda rs: rs[0].__setitem__('t_end_abs', 7.0), 'G-1b')
    mutate(lambda rs: rs[0].__setitem__('n_obs', 599), 'G-2')
    mutate(lambda rs: rs[0].__setitem__('q_hash', 'X'), 'G-2b')
    mutate(lambda rs: rs[0].__setitem__('sigma_bits', 'X'), 'G-2b')
    mutate(lambda rs: [r.__setitem__('n_atoms', 0) for r in rs if r['cell'] == 'PROP'],
           'G-3')
    mutate(lambda rs: [r.__setitem__('nowcast_bits', 'X') for r in rs
                       if r['level'] == NOOP_LEVEL][:1], 'G-4')
    mutate(lambda rs: [r.__setitem__('cap_init_method', 'v2_konly') for r in rs
                       if r['cell'] == 'PROP'][:1], 'G-6')
    mutate(lambda rs: [r.__setitem__('nowcast', 9.0) for r in rs
                       if r['level'] == 'ORIG2' and r['cell'] == 'SHIP'][:1], 'G-7')

    # ⑤ ★공허 참조 방어 — 참조가 비면 G-5/G-5b 는 **FAIL** 이어야 한다(PASS 아님)
    global FROZEN_B2
    saved, FROZEN_B2 = FROZEN_B2, os.path.join(HERE, '__no_such_file__.jsonl')
    gg = dict((k.split()[0], ok) for k, ok, _ in gates(_fixture(6, 6)))
    FROZEN_B2 = saved
    chk(gg.get('G-5') is False and gg.get('G-5b') is False, 'empty ref must FAIL')

    # ⑥ 소스 되읽기 상수가 문서의 주장을 지탱한다
    chk(W_ANCHOR == 40.0 and M_SCAN == 12 and M_CAP == 600, 'scene defaults')
    chk(TP_FACTOR_DEPLOY == 2.0, 'deploy tp_factor')
    chk(TOBS_FACTOR_TRAIN == (1.9, 2.3), 'train TOBS_FACTOR')
    chk(M_OBS_CAP_LEARNED == 400, 'learned M_OBS_CAP')
    chk(N_RUNS == 456, f'N_RUNS={N_RUNS}')

    print(f'selftest: {"PASS" if not fails else "FAIL"}  ({len(fails)} 실패)')
    for f in fails:
        print('  -', f)
    if not fails:
        print(f'  levels={LEVEL_ORDER}+{NOOP_LEVEL}  arms={ARM_ORDER}  '
              f'N_RUNS={N_RUNS}  thresholds={THRESHOLD_12}/12, {THRESHOLD_36}/36')
        print(f'  W_ANCHOR={W_ANCHOR}  deploy={TP_FACTOR_DEPLOY}xTp  '
              f'train={TOBS_FACTOR_TRAIN}xTp  m_cap={M_CAP}  learned_cap={M_OBS_CAP_LEARNED}')
    return 0 if not fails else 1


def refcheck():
    """★새 교훈 ①: 외부 동결 참조를 **실제 파일에 대고** 돌려 비지 않음을 확인한다."""
    ref = b2_scene_ref()
    print(f'FROZEN_B2 = {FROZEN_B2}')
    print(f'  존재={os.path.exists(FROZEN_B2)}  ref 레코드={len(ref)}/{len(RECORDS)}')
    miss = [r for r in RECORDS if r not in ref]
    print(f'  누락={miss}')
    print(f'  샘플={list(ref.items())[:3]}')
    print(f'gates_b3 임포트: RECORDS={len(RECORDS)} WINDOWS={WINDOWS} SEED={SEED} NOISE={NOISE}')
    return 0 if (ref and not miss) else 1


if __name__ == '__main__':
    if '--refcheck' in sys.argv:
        sys.exit(refcheck())
    sys.exit(selftest())
