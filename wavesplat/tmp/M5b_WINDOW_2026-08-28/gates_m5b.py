"""M-5b 단일출처 — ★**관측 창 길이 `Tobs`(= 시간축 개구)가 실측 경계를 매개하는가.**

★**`M5_WINDOW_2026-08-28` 의 후속.** 그 배치는 차단 게이트 `G-1b` FAIL 로 판정을 내지
못했고 **`VOID`** 로 닫혔다(`tmp/M5_WINDOW_2026-08-28/AMENDMENT_1.md`, 결과 미열람).
⛔**결과 0행에서 새로 동결한다.** 이 파일이 상수·팔·문턱·게이트·라벨·**결과 전 해석 규칙**의
**유일 정의**다. `PREREG.md` 는 산문 번역이고 어긋나면 **코드가 이긴다**.

────────────────────────────────────────────────────────────────────────────
★**`M5` 에서 무엇이 바뀌었나 — 「불변」과 「강화」를 결과 전에 나눠 적는다**

**불변**(= 사후 자유도가 아니라는 증거. 글자 그대로 유지된다):
  · 팔 2개(`SHIP`·`PROP`) · 수준 7개(`ANCHOR`·`TP6`·`TP4`·`TP2`·`SEC8`·`ORIG2`·`NOOP`)
  · `PRIMARY` = `PROP` · win 3 · `TP2` vs `ANCHOR` · `nowcast` · n=12 · 문턱 **10/12**
  · 부차 n=36 문턱 **24/36** · 라벨 **네 갈래** · `DIRECTION` · MDE(q=0.90) · 금지 목록
  · `N_RUNS = 456`

**강화**(전부 `M5` 결과를 **보지 않은 상태**에서, 기제와 편향 방향만으로 결정됐다):
  1. ★**장면 규약** `cap_mode='end_anchored'` — *"가장 최신의 비어있지 않은 스캔은 통째로
     남기고 남은 예산만 균등추첨"*. `scene_station` 에 **기본값-꺼짐**으로 들어갔다
     (기본 `'uniform'` = 2026-08-28 이전과 비트동일; 회귀 = `regress_scene_station.py`).
     ⟹ ★`W` 의 **두 번째 경로**(cap 추첨의 모집단 크기)가 닫힌다.
  2. ★**`G-1b` 를 절대형으로** — 수준 간 *비교*(그 형태는 수준이 하나면 못 잡는다)가 아니라
     ***`t_end_abs == (그 (record,win) 의 최신 비어있지 않은 프레임 시각)`*** 을 요구한다.
     앵커가 **둘**이다: 행 안의 `t0_abs`(`G-1b`) · 전제가 동결한 `t_end_true`(`G-1c`).
  3. ★**`G-5b` 2겹** — `end_anchored` 는 `B-2` 동결 장면과 **비트동일하지 않다**(포기 공시).
     대신 **(a) `G-5b1`** 기본값 경로가 `B-2` 를 12/12 비트재현(파일 수정이 기본에서 무해)
     **(b) `G-5b2`** 판정 행의 **cap 이전 pool 해시**가 **수정 전 소스**(git blob,
     md5 대조)가 낸 pool 해시와 비트동일(= 바뀐 것이 **cap 뿐**).
  4. ★**밀도 공변 공시** — `n_obs` 를 600 으로 고정했으므로 `W` 를 줄이면 **`t0` 부근
     시간 밀도가 처치와 함께 움직인다**(≈600/`W` 점/초). `M5` 가 미공시했던 것이고
     이제 행에 싣고 판정문에 붙인다. ⛔*"개구 그 자체"* 로 읽지 않는다.
  5. ★**`G-9` 커버리지 메타게이트**(오늘 `M4 G-13` 의 부류 = **필터 항진**).
     각 게이트가 **실제로 몇 행/몇 짝을 보았는지** 세어 **0 이면 FAIL**.
     자가진단이 깨끗한 픽스처에서 **전 게이트 커버리지 > 0** 을 요구한다.
     ⚠`tmp/_audit/check_gate_refs.py` 는 **키 부재**를 잡지 **필터 항진**은 못 잡는다.
  6. `G-0d`(유한성) · `G-2a` 를 **비어있지 않음까지** 요구(⛔`in (None, 400)` 는
     shim 이 죽으면 **항등적으로 참**이었다 — `M5` 에 남아 있던 같은 부류다).

────────────────────────────────────────────────────────────────────────────
★**왜 이 축인가 — 코드에서 도출한 미공시 불일치** (`M5` 에서 그대로 계승)

  · 배포 **합성** 경로:  `wavesplat/run.py` → `loader.get_init_window(tp_factor=2.0)`
  · 배포 사전의 **학습** 분포: `gen_mixed.py:TOBS_FACTOR = (1.9, 2.3)`, `Tobs = U(1.9,2.3)·Tp`
  · 실측 하네스: `scene_station(W=40.0)` **초 고정** ⟹ 12 레코드 중앙 **≈11.8 × Tp**

⟹ ★**실측 팔은 사전이 한 번도 본 적 없는 시간 개구에서 평가돼 왔다**(12/12 레코드가 밖).

★**도달성**: 조건자 `d3_beamform_net.beamform()` 의 k-공간 주엽 폭이 `1/T_obs` 로 좁아지고,
`learned_init.propose_atoms` 는 지도 직전 `t0 = P[:,2].min(); Ps[:,2] -= t0` 으로 원점을
스스로 정규화한다 ⟹ ***조건자가 시간축에서 보는 양은 오직 관측 시간 폭 하나***다.

★**단일요인**: `scene_station` 안에서 `W` 는 `m = P_all[:,2] >= t0 - W` 한 줄에만 들어가고,
`Zmem`(⟹ `Hs`·`kp`·채점 분모 `sg`)·`Zfut`·`maps_gt`·`Pq`·`leads`·`gxq/gyq` 는 `W_max=40`
고정이라 움직이지 않는다. ★그리고 `end_anchored` 가 **cap 추첨의 모집단 의존**을 없앴다
⟹ 남는 공변은 **밀도 하나뿐이고 그것은 공시된다**(위 5항 · 아래 ⛔6).

────────────────────────────────────────────────────────────────────────────
⛔**이 배치가 주장하지 않는 것 (결과 전 공시)**

 1. ⛔**기제 지목 금지.** `DIRECTION = ELIMINATION-OR-LOCATION`. 양성이 나와도 산출은
    *"경계가 이 축 위에 놓여 있다"* 까지이고 *"원인은 X 다"* 로 올리지 않는다.
 2. ⛔**배포 권고 금지.** 용량을 정하는 `Tp` 는 **조밀한 참 표면**의 `kp` 에서 온다 = GT 유래
    (`B-25` 의 교훈). ✅대신 GT-free 거울 `SEC8`(초 고정)을 같이 돌린다.
 3. ⛔**②(사전 지지집합)와 ③(정상성)을 분리한다고 주장 금지.**
 4. ⛔**사이트 수준 일반화 금지**(사이트 4개) · ⛔**외부 기준선 추가 금지**(정지규칙:
    `S17` + `B-5` + `B2_WANGFIX`) · ⛔**세 번째 LETKF 수리 금지**.
 5. ⛔`ZERO_REF=1.0` 을 쓰지 않는다. 살아 있는 것은 **짝 비교**다.
 6. ★⛔**밀도 공변 (신규 공시)** — `n_obs=600` 고정이므로 `W` 를 줄이면 `t0` 부근 시간
    밀도가 **함께** 커진다. 정확한 문장 = ***"관측 **개수** 를 고정했으므로 시간 폭과
    `t0` 부근 시간 밀도가 함께 움직인다"***. ⟹ 양성이 나와도 ⛔*"개구(aperture) 그 자체"*
    로 읽지 않는다. 분리는 **별도 사전등록**(균등 시간간격 솎기)의 몫이다.
 7. ⛔**`M5` 의 `VOID_unread/` 수치를 인용하지 않는다.** 이 배치의 판정과 섞지 않는다.
 8. ⛔*"`B-2` 와 같은 장면"* 이라 쓰지 않는다 — `end_anchored` 는 `B-2`…`B-10` 실측 아크와
    **비트동일한 장면이 아니다**(`ANCHOR` 포함). 유지되는 것은 **같은 WASS 표면 · 같은
    창 선택 · 같은 잡음 규약 · 같은 채점**이고, 바뀐 것은 **cap 추첨 규약 하나**다.
"""
import ast
import hashlib
import json
import os
import re
import statistics as st
import subprocess
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

BATCH = 'M5b_WINDOW_2026-08-28'
PRIOR_VOID_BATCH = 'M5_WINDOW_2026-08-28'      # ⛔그 결과는 인용 금지(VOID_unread/)
DIRECTION = 'ELIMINATION-OR-LOCATION'
FIELD = 'nowcast'                 # lead 0. `leadmean` 은 사전등록된 부차
G_ACC = 9.81

PREMISE_JSON = os.path.join(HERE, 'premise_m5b.json')

# ── ★장면 규약 (이 배치의 유일한 코드 변경) ─────────────────────────────────
CAP_MODE = 'end_anchored'
POOL_NO_CAP = 10 ** 9             # `m_cap` 을 껐을 때 = cap 이전 pool 을 그대로 받는다
# `scene_station.py` 의 **수정 전** 판본. `M5`·`B-2`…`B-10` 30여 배치가 이 md5 를 동결했다.
# ⟹ pool 불변(=바뀐 것이 cap 뿐)을 이 blob 으로 증명한다. ⛔HEAD 가 아니라 **blob** 을 핀한다.
SCENE_PRE_BLOB = '27c35dc2b20c6f13bdfd714225d910005114b39f'
SCENE_PRE_MD5 = '18183f898008d239db002a3634f19686'
T_EQ_TOL = 1e-9                   # 절대 시각 동일 판정 허용오차(초). 결과 전 고정.


# ── 배포·학습 규약 상수: 소스에서 되읽는다 ──────────────────────────────────
def _sig_defaults():
    """`scene_station()` 시그니처에서 기본값을 되읽는다 (`cap_mode` 포함)."""
    import inspect
    from scene_station import scene_station
    p = inspect.signature(scene_station).parameters
    return {k: p[k].default for k in ('W', 'm_scan', 'm_cap', 'noise', 'cap_mode')}


def _static_const(path, name):
    """소스를 **임포트하지 않고** 최상위 대입 `name = <literal>` 을 읽는다.

    ⛔`gen_mixed.py`·`d3_beamform_net.py` 는 최상위에서 CUDA/에피소드를 건드린다."""
    src = open(os.path.join(ROOT, path)).read()
    for node in ast.parse(src).body:
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
# ★기본값은 **현행 유지**여야 한다 — 이 배치는 기본값을 바꾸지 않는다(호출자 30곳 보호).
assert SCENE_DEFAULTS['cap_mode'] == 'uniform', \
    f"scene_station 의 cap_mode 기본값이 {SCENE_DEFAULTS['cap_mode']!r} — 'uniform' 이어야 한다"
assert CAP_MODE != SCENE_DEFAULTS['cap_mode'], '이 배치의 규약이 기본값과 같다(처치 없음)'

TP_FACTOR_DEPLOY = _deploy_tp_factor()                                        # 2.0
TOBS_FACTOR_TRAIN = tuple(_static_const(
    'tmp/mixed_retrain_2026-07-26/gen_mixed.py', 'TOBS_FACTOR'))              # (1.9, 2.3)
M_OBS_TRAIN = int(_static_const(
    'tmp/mixed_retrain_2026-07-26/gen_mixed.py', 'M_OBS'))                    # 400
M_OBS_COND = int(_static_const('scripts/d3_beamform_net.py', 'M_OBS'))        # 400
M_OBS_CAP_LEARNED = int(_static_const(
    'wavesplat/optimizer/learned_init.py', 'M_OBS_CAP'))                      # 400

assert TOBS_FACTOR_TRAIN[0] <= TP_FACTOR_DEPLOY <= TOBS_FACTOR_TRAIN[1]
assert M_OBS_TRAIN == M_OBS_COND == M_OBS_CAP_LEARNED


# ── 수준(dose) 정의 ─────────────────────────────────────────────────────────
LEVELS = {
    'ANCHOR': dict(kind='sec', value=W_ANCHOR, origin='min',
                   why='실측 하네스 규약(B-2…B-10 전부). 기준 팔.'),
    'TP6':    dict(kind='tp',  value=6.0, origin='min', why='용량-반응 중간'),
    'TP4':    dict(kind='tp',  value=4.0, origin='min', why='용량-반응 중간'),
    'TP2':    dict(kind='tp',  value=TP_FACTOR_DEPLOY, origin='min',
                   why='★배포 합성 규약 = 학습 지지집합 중앙. PRIMARY 대비.'),
    'SEC8':   dict(kind='sec', value=8.0, origin='min',
                   why='★GT-free 거울 — 용량이 GT 유래 Tp 로 정의된 것이 결과를 '
                       '만들지 않았음을 보인다.'),
    'ORIG2':  dict(kind='sec', value=W_ANCHOR, origin='pred_minus',
                   why='★시간 **원점** 불변 대조 — 관측 집합은 ANCHOR 와 비트동일하고 '
                       '예측 시각까지의 거리만 TP2 와 같게 맞춘다.'),
}
ORIG2_TP_FACTOR = TP_FACTOR_DEPLOY
LEVEL_ORDER = ['ANCHOR', 'TP6', 'TP4', 'TP2', 'SEC8', 'ORIG2']
DOSE_LADDER = ['ANCHOR', 'TP6', 'TP4', 'TP2']        # 창이 짧아지는 순서

NOOP_LEVEL = 'NOOP'
NOOP_SPEC = dict(kind='noop', value=1e9, origin='min',
                 why='W 배관이 작동하지 않아야 할 때 정말 작동하지 않는가.')
NOOP_WIN = 3

# ★`W >= W_ANCHOR` 인 수준 = cap 이전 pool 이 `B-2` 시절 pool 과 **같아야 하는** 수준.
#   `G-5b2` 가 이 수준들에서만 pool 해시를 대조한다(다른 수준은 pool 이 당연히 다르다).
POOL_LEVELS = ('ANCHOR', 'ORIG2', NOOP_LEVEL)

# ★⛔**빈 dict `{}` 로 팔을 정의하지 않는다** (`gates_b7.py:46` BLOCKER, 2026-08-23).
SHIPPED_2026_08_22 = dict(varpro_steps=400, learned_init_iters=0, auto_switch_thresh=0)
ARMS = {
    'SHIP': dict(SHIPPED_2026_08_22, real_domain_gate=True,
                 real_estimator='bank_mmse'),
    'PROP': dict(SHIPPED_2026_08_22, real_domain_gate=False, init_method='learned'),
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
          + len(ARMS) * len(RECORDS))                      # + NOOP(win=3) = 456

ORIG2_SHIP_TOL = 1e-3                  # 상대. 넘으면 `G-7` FAIL (SCOPING)
BITEQ_KEYS = ('obs_hash', 'q_hash')

# ★결과 전 선언: BLOCKING / SCOPING. ⛔결과를 보고 이 분류를 바꾸지 않는다.
SCOPING = {'G-7': 'ORIG2 수준만 해석 불가로 선언한다 (PRIMARY 는 TP2 vs ANCHOR 라 무관)'}

# 외부 동결 참조
FROZEN_B2 = os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23', 'results.jsonl')
FROZEN_B2_WIN = 3                      # `gates_b2.WIN`

# 각 `gates()` 호출이 남기는 **게이트별 커버리지**(= 실제로 들여다본 행/짝 수).
COVERAGE = {}


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
    """B-2 동결 행에서 (record → obs_hash). ★비면 빈 dict 을 낸다(공허 PASS 금지).

    ⛔`bool(ref) and …` 로만 쓴다 — 참조가 비면 게이트는 **FAIL** 이어야 한다."""
    ref = {}
    for r in _load_jsonl(FROZEN_B2):
        if r.get('win') != FROZEN_B2_WIN:
            continue
        h = r.get('obs_hash')
        if h:
            ref.setdefault(r['record'], h)
    return ref


def hash_obs(a, z=None):
    """★러너와 전제가 **같은 함수**로 해시한다 (⛔두 곳에 복사하면 그게 결함 부류다).

    `run_b2.py:50` 의 규약 축자: `md5(ascontiguousarray(a, float64))[:12]`.
    ⟹ `B-2` 동결 `obs_hash` 와 **직접 비교 가능**하다."""
    import numpy as np
    arr = (np.column_stack([np.asarray(a, float), np.asarray(z, float)])
           if z is not None else np.asarray(a, float))
    return hashlib.md5(np.ascontiguousarray(arr, dtype=np.float64)).hexdigest()[:12]


def premise():
    return json.load(open(PREMISE_JSON)) if os.path.exists(PREMISE_JSON) else {}


def premise_cells(pre=None):
    """(record, win) → P8 셀 dict. **비면 빈 dict** (공허 PASS 금지)."""
    d = ((pre if pre is not None else premise()).get('P8_pool_and_end') or {})
    out = {}
    for rec, per in (d.get('cells') or {}).items():
        for w, v in per.items():
            out[(rec, int(w))] = v
    return out


def scene_pre_source():
    """`scene_station.py` 의 **수정 전** 소스를 git blob 에서 꺼내 md5 로 대조한다.

    ⛔HEAD 를 쓰지 않는다(다른 세션이 커밋하면 흔들린다). blob 을 직접 핀한다."""
    src = subprocess.run(['git', '-C', ROOT, 'cat-file', 'blob', SCENE_PRE_BLOB],
                         capture_output=True, check=True).stdout
    got = hashlib.md5(src).hexdigest()
    if got != SCENE_PRE_MD5:
        raise RuntimeError(f'수정 전 scene_station md5 불일치: {got} != {SCENE_PRE_MD5}')
    return src.decode()


# ── 집계 ────────────────────────────────────────────────────────────────────
def key(r):
    return (r.get('level'), r.get('cell'), r.get('record'), r.get('win'))


def _finite(x):
    try:
        return x is not None and float(x) == float(x) and abs(float(x)) != float('inf')
    except (TypeError, ValueError):
        return False


def _pick(rows, level, arm, field=FIELD, wins=None):
    wins = WINDOWS if wins is None else wins
    return {(r['record'], r['win']): r[field] for r in rows
            if 'error' not in r and r.get('level') == level and r.get('cell') == arm
            and r.get('win') in wins and _finite(r.get(field))}


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
    return [paired(rows, lv, REF_LEVEL, arm, field, wins) for lv in DOSE_LADDER[1:]]


def density_disclosure(rows, wins=None):
    """★공시 전용 — 수준별 `t0` 부근 시간 밀도. ⛔판정에 쓰지 않는다.

    `n_obs` 를 600 으로 고정했으므로 창을 줄이면 밀도가 **함께** 오른다."""
    wins = [PRIMARY_WIN] if wins is None else wins
    out = []
    for lv in LEVEL_ORDER + [NOOP_LEVEL]:
        v = [r for r in rows if 'error' not in r and r.get('level') == lv
             and r.get('win') in wins and r.get('cell') == PRIMARY_ARM]
        if not v:
            continue
        def med(k):
            xs = [float(r[k]) for r in v if _finite(r.get(k))]
            return round(st.median(xs), 4) if xs else None
        out.append(dict(level=lv, n=len(v), W_s=med('W_used'),
                        W_over_Tp=med('W_over_Tp'), t_span=med('t_span'),
                        dens_per_s=med('dens_per_s'), dens_last_1s=med('dens_last_1s'),
                        dens_last_02Tp=med('dens_last_02Tp')))
    return out


# ── 게이트 ──────────────────────────────────────────────────────────────────
def gates(rows):
    """★모든 게이트가 **커버리지**(실제로 본 행/짝 수)를 함께 남긴다.

    `G-9` 가 커버리지 0 을 FAIL 시킨다 ⟹ *"그 팔을 구조적으로 비우는 필터"* 가
    조용히 통과하지 못한다(오늘 `M4 G-13` 이 그 부류로 죽었다)."""
    ok = [r for r in rows if 'error' not in r]
    out, cov = [], {}

    def A(gid, good, msg, n):
        out.append((gid, bool(good), str(msg)))
        cov[gid.split()[0]] = int(n)

    A('G-0 런 수', len(ok) == N_RUNS, f'{len(ok)}/{N_RUNS}', len(ok))
    A('G-0b 에러 0', len(rows) == len(ok), f'{len(rows) - len(ok)} 에러', len(rows))
    A('G-0c 셀 중복 0', len({key(r) for r in ok}) == len(ok),
      f'{len(ok) - len({key(r) for r in ok})} 중복', len(ok))
    # G-0d ★유한성 — NaN 은 `a < b` 에서 조용히 "패배" 로 세어진다.
    nf = [(r.get('level'), r.get('cell'), r.get('record'), r.get('win'))
          for r in ok if not (_finite(r.get('nowcast')) and _finite(r.get('leadmean')))]
    A('G-0d nowcast·leadmean 이 전 행에서 유한', not nf, str(nf[:4]), len(ok))

    # ── G-1 개입 되읽기 ⛔항등식 아님: 설정값이 아니라 **관측 배열**에서 다시 잰다 ──
    bad, seen = [], 0
    for r in ok:
        lv, w_used, span = r.get('level'), r.get('W_used'), r.get('t_span')
        if lv is None or w_used is None or span is None:
            bad.append((lv, r.get('record'), 'missing')); continue
        if lv == NOOP_LEVEL:
            continue
        seen += 1
        dt = r.get('dt') or 0.0
        if not (span <= w_used + 2 * dt + 1e-9 and span >= 0.90 * min(w_used, W_ANCHOR)):
            bad.append((lv, r.get('record'), round(span, 3), round(w_used, 3)))
    A('G-1 개입 되읽기: 관측 시간폭이 W 를 따라 움직였다(설정값 아님)',
      not bad, str(bad[:4]), seen)

    # ── ★G-1b **절대형** — 끝 시각이 그 셀의 최신 비어있지 않은 프레임 시각이다 ──
    #    ⛔`M5` 의 수준 간 비교형은 수준이 하나면 못 잡는다. 여기서는 **행 안의 앵커**
    #    `t0_abs`(= 장면의 `t0` = 메모리 창의 마지막 프레임 시각)와 직접 맞춘다.
    pre = premise()
    cells = premise_cells(pre)
    nonempty_ok = bool(cells) and all(c.get('last_frame_nonempty') for c in cells.values())
    bad, seen = [], 0
    for r in ok:
        te, t0a = r.get('t_end_abs'), r.get('t0_abs')
        if not (_finite(te) and _finite(t0a)):
            bad.append((r.get('record'), r.get('level'), 'missing')); continue
        seen += 1
        if abs(float(te) - float(t0a)) > T_EQ_TOL:
            bad.append((r.get('record'), r.get('win'), r.get('level'),
                        round(float(te) - float(t0a), 6)))
    A('G-1b ★절대형: 관측 끝 시각 == 그 셀의 최신 프레임 시각 (행 내부 앵커)',
      nonempty_ok and not bad,
      ('전제가 최신 프레임 비어있음을 보고했다' if not nonempty_ok else str(bad[:3])),
      seen)

    # ── ★G-1c 두 번째 앵커: 전제가 **동결한** t_end_true 와 일치 ──────────────
    bad, seen = [], 0
    for r in ok:
        c = cells.get((r.get('record'), r.get('win')))
        if not c or not _finite(r.get('t_end_abs')):
            continue
        seen += 1
        if abs(float(r['t_end_abs']) - float(c['t_end_true'])) > T_EQ_TOL:
            bad.append((r.get('record'), r.get('win'), r.get('level')))
    A('G-1c ★절대형(외부 앵커): 끝 시각 == 전제가 동결한 t_end_true',
      bool(cells) and not bad and seen == len(ok), str(bad[:3]), seen)

    # ── G-2 ★단일요인 보장: 관측 **개수**가 모든 수준에서 같다 ──────────────
    counts = sorted({r.get('n_obs') for r in ok})
    A(f'G-2 n_obs 가 전 수준에서 {M_CAP} 불변(개수 교락 없음)',
      counts == [M_CAP], str(counts), len(ok))
    # ⛔`in (None, 400)` 은 shim 이 죽으면 **항등적으로 참**이다 ⟹ 비어있지 않음까지 요구.
    prop = [r for r in ok if r.get('cell') == 'PROP']
    seen = [r for r in prop if r.get('n_obs_learned') is not None]
    A(f'G-2a 제안 팔 조건자 입력 {M_OBS_CAP_LEARNED} 불변 (⛔None 허용 안 함)',
      bool(prop) and len(seen) == len(prop)
      and all(r['n_obs_learned'] == M_OBS_CAP_LEARNED for r in seen),
      f'{len(seen)}/{len(prop)} 기록됨', len(seen))

    # ── G-2b ★채점·질의·참값이 수준에 걸쳐 비트동일 ────────────────────────
    bad, seen = [], 0
    for (rec, w) in {(r['record'], r['win']) for r in ok}:
        grp = [r for r in ok if r['record'] == rec and r['win'] == w]
        seen += 1
        for k in ('q_hash', 'gt_hash', 'sigma_bits', 'kp_bits'):
            if len({r.get(k) for r in grp}) > 1:
                bad.append((rec, w, k))
    A('G-2b 채점 분모·질의·참값·kp 가 수준에 걸쳐 비트동일', not bad, str(bad[:4]), seen)

    # ── ★G-2c 장면 규약 되읽기 (⚠이것만으로는 항등식이다 — 관측 가능한 귀결은 G-1b/G-1c) ──
    modes = {r.get('cap_mode_used') for r in ok}
    sigs = {r.get('cap_mode_sig') for r in ok}
    A(f'G-2c cap_mode 가 전 런에서 {CAP_MODE!r} · 시그니처 기본값은 uniform',
      modes == {CAP_MODE} and sigs == {'uniform'}, f'used={modes} sig={sigs}', len(ok))

    # ── G-3 팔이 서로 달랐다 ────────────────────────────────────────────────
    ship = [r for r in ok if r['cell'] == 'SHIP']
    A('G-3 PROP 이 성분을 만들었다',
      bool(prop) and all((r.get('n_atoms') or 0) > 0 for r in prop),
      f'{sum(1 for r in prop if (r.get("n_atoms") or 0) > 0)}/{len(prop)}', len(prop))
    A('G-3b PROP 전건 VARPRO 발화',
      bool(prop) and all(r.get('varpro_fired') for r in prop),
      f'{sum(1 for r in prop if r.get("varpro_fired"))}/{len(prop)}', len(prop))
    A('G-3c SHIP 은 bank 경로(성분 0 · real_gate True)',
      bool(ship) and all((r.get('n_atoms') or 0) == 0 and r.get('real_gate') is True
                         for r in ship), f'{len(ship)}', len(ship))

    # ── G-4 ★양성대조(비트동일): NOOP == ANCHOR ────────────────────────────
    bad, seen = [], 0
    for arm in ARM_ORDER:
        for rec in RECORDS:
            a = [r for r in ok if r['level'] == NOOP_LEVEL and r['cell'] == arm
                 and r['record'] == rec and r['win'] == NOOP_WIN]
            b = [r for r in ok if r['level'] == 'ANCHOR' and r['cell'] == arm
                 and r['record'] == rec and r['win'] == NOOP_WIN]
            if len(a) != 1 or len(b) != 1:
                bad.append((arm, rec, 'missing')); continue
            seen += 1
            for k in BITEQ_KEYS + ('nowcast_bits', 'leadmean_bits'):
                if a[0].get(k) != b[0].get(k):
                    bad.append((arm, rec, k))
    A('G-4 양성대조: 무연산 W 는 ANCHOR 와 **비트동일**', not bad, str(bad[:4]), seen)

    # ── G-5 ★외부 동결 참조가 비지 않았다 (⛔`bool(ref) and …`) ─────────────
    ref = b2_scene_ref()
    p7 = (pre.get('P7_default_bitidentical') or {})
    A('G-5 외부 참조가 비지 않았다 (B-2 동결 행 · 전제도 같은 수를 봤다)',
      bool(ref) and len(ref) == len(RECORDS)
      and int(p7.get('n_ref') or 0) == len(RECORDS),
      f'ref={len(ref)}/{len(RECORDS)} premise_n_ref={p7.get("n_ref")}', len(ref))

    # ── ★G-5b1 2겹 (a): **기본값 경로**가 B-2 를 비트재현한다 ────────────────
    #    ⟹ `scene_station.py` 수정이 30여 호출자의 기본 동작에 무해함을 증명한다.
    A('G-5b1 ★기본값(cap_mode=uniform) 이 B-2 동결 장면을 비트재현',
      bool(p7) and int(p7.get('n_equal') or 0) == len(RECORDS)
      and int(p7.get('n_checked') or 0) == len(RECORDS),
      f'{p7.get("n_equal")}/{p7.get("n_checked")}', int(p7.get('n_checked') or 0))

    # ── ★G-5b2 2겹 (b): 판정 행의 **cap 이전 pool** 이 수정 전 소스와 비트동일 ──
    #    ⟹ 바뀐 것이 **cap 뿐**임을 판정 자료 위에서 증명한다.
    bad, seen, covered = [], 0, set()
    for r in ok:
        want_pool = r.get('level') in POOL_LEVELS
        if want_pool != (r.get('pool_hash') is not None):
            bad.append((r.get('record'), r.get('level'), 'pool_hash 존재 규약 위반'))
            continue
        if not want_pool:
            continue
        c = cells.get((r.get('record'), r.get('win')))
        if not c:
            bad.append((r.get('record'), r.get('win'), 'no premise cell')); continue
        seen += 1
        covered.add((r['record'], r['win']))
        if r['pool_hash'] != c['pool_hash'] or int(r.get('pool_n') or -1) != int(c['pool_n']):
            bad.append((r['record'], r['win'], r['level'], r['pool_hash'], c['pool_hash']))
    A('G-5b2 ★cap 이전 pool 이 수정 전 소스의 pool 과 비트동일 (바뀐 것은 cap 뿐)',
      bool(cells) and not bad
      and covered == {(rec, w) for rec in RECORDS for w in WINDOWS},
      f'{seen} 행 · {len(covered)}/{len(RECORDS) * len(WINDOWS)} 셀 · {str(bad[:2])}', seen)

    # ── G-6 배포 상수 되읽기 (⛔의도표 금지 — 객체에서 잡은 값) ──────────────
    bad = [(r.get('level'), r.get('record'), r.get('cap_init_method'))
           for r in prop if r.get('cap_init_method') != 'learned']
    A('G-6 PROP 의 init_method 를 **객체**에서 되읽으면 learned', not bad,
      str(bad[:3]), len(prop))
    lam = {r.get('cap_lambda_l1') for r in ok}
    A('G-6b lambda_l1 이 전 런에서 하나', len(lam) == 1, str(lam), len(ok))
    drift = {}
    for k, v in SHIPPED_2026_08_22.items():
        seenv = {r.get('cap_' + k) for r in ok}
        if seenv != {v}:
            drift[k] = sorted(str(x) for x in seenv)
    A('G-6c 배포 3키가 선언값으로 전 런 고정 (조용한 흡수 없음)', not drift,
      str(drift), len(ok))

    # ── G-7 ORIG2 의 bank-팔 불변 대조 (SCOPING) ────────────────────────────
    a = _pick(rows, 'ORIG2', 'SHIP')
    b = _pick(rows, 'ANCHOR', 'SHIP')
    ks = sorted(set(a) & set(b))
    dev = max((abs(a[k] - b[k]) / b[k] for k in ks), default=None)
    A('G-7 ORIG2 는 bank 팔에서 시간원점 불변',
      bool(ks) and dev is not None and dev <= ORIG2_SHIP_TOL,
      f'max rel dev={dev}', len(ks))

    # ── G-8 표본 완전성 ─────────────────────────────────────────────────────
    anch = {(r['record'], r['win']) for r in ok if r['level'] == 'ANCHOR'}
    A('G-8 12 레코드 × 3 창 전건',
      anch == {(rec, w) for rec in RECORDS for w in WINDOWS}, f'{len(anch)}', len(anch))
    A('G-8b 전제 파일 존재', os.path.exists(PREMISE_JSON), PREMISE_JSON, 1)
    A('G-8c 전제가 전 수준 허용성 + 최신프레임 비어있지않음을 미리 확인했다',
      _premise_admissible(pre) and nonempty_ok, f'cells={len(cells)}', len(cells))

    # ── ★G-10 밀도 공변이 **행에 기록**됐다 (공시 의무. 값은 판정에 쓰지 않는다) ──
    miss = [(r.get('record'), r.get('level')) for r in ok
            if not all(_finite(r.get(k)) for k in
                       ('dens_per_s', 'dens_last_1s', 'dens_last_02Tp', 'n_last_scan'))]
    A('G-10 ★밀도 공변이 전 행에 기록됐다 (⛔값은 판정에 쓰지 않는다 — 공시용)',
      not miss, str(miss[:3]), len(ok))

    # ── ★G-9 메타: 커버리지 0 인 게이트가 없다 (필터 항진 금지) ──────────────
    zero = sorted(k for k, v in cov.items() if v <= 0)
    A('G-9 ★커버리지 0 게이트 없음 (필터 항진 = 항등적으로 참인 검사 금지)',
      not zero, str(zero), len(cov))

    COVERAGE.clear(); COVERAGE.update(cov)
    return out


def _premise_admissible(pre=None):
    d = ((pre if pre is not None else premise()).get('P1_level_admissibility') or {})
    return bool(d) and bool(d.get('all_admissible'))


def blocking_failures(gate_rows):
    """SCOPING 으로 선언된 것을 뺀 FAIL 목록. 비어 있지 않으면 판정 금지."""
    return [gid for gid, ok, _ in gate_rows
            if not ok and gid.split()[0] not in SCOPING]


# ── 결과 전 해석 규칙 (네 갈래 전부 결과 전에 적는다) ────────────────────────
DENSITY_CAVEAT = ('⚠**밀도 공변 공시**: `n_obs` 를 600 으로 고정했으므로 창을 줄이면 '
                  '`t0` 부근 시간 밀도가 **함께** 커진다(≈600/W 점/초). '
                  '⛔따라서 이 결과를 *"개구(aperture) 그 자체"* 로 읽지 마라 — '
                  '개구와 밀도의 분리는 **별도 사전등록**의 몫이다.')


def label(primary, discrim, secondary, dose):
    """PRIMARY = `PROP`, win=3, n=12, `TP2` vs `ANCHOR`, `nowcast`, 문턱 10/12."""
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
                    f'가르지 못한다. 용량-반응 단조={dose_ok}. ' + DENSITY_CAVEAT)
        return ('WINDOW-EARNS-PRIOR-ARM',
                f'★제안 표현 팔만 좋아진다(PROP {w}/{n} p={primary["p_one_sided"]:.4f}; '
                f'SHIP {ws}/{discrim["n"]} 은 같은 문턱을 **넘지 못했다**). '
                '⟹ 경계가 창 길이 축 위에 놓여 있고 효과가 **사전을 지닌 팔에 몰려 있다**. '
                '⛔"SHIP 은 영향이 없다" 로 읽지 마라(등가검정 안 했다). '
                f'용량-반응 단조={dose_ok}. ' + DENSITY_CAVEAT)
    if (n - w) >= THRESHOLD_12:
        return ('WINDOW-LONGER-IS-BETTER',
                f'창을 줄이면 **나빠진다**(PROP {w}/{n}). ⟹ 실측 팔은 긴 시간 개구를 '
                '쓰고 있고, *"학습 지지집합 밖 창이 손해"* 가설은 **반증**된다. '
                '소거표에 그 방향으로 한 줄이 추가된다. ' + DENSITY_CAVEAT)
    if (secondary['better'] >= THRESHOLD_36
            or (secondary['n'] - secondary['better']) >= THRESHOLD_36):
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
def _fixture(prop_better, ship_better, n_rec=None, anchored=True):
    """네 라벨 분기를 전부 때리는 최소 픽스처. ⛔판정 자료가 아니다.

    ★`anchored=True` 면 **실제 전제 파일**의 `t_end_true`·`pool_hash` 를 싣는다 —
    그래야 `G-1b`·`G-1c`·`G-5b2` 가 **자료에 대고 만족 가능**한지 시험된다
    (= 오늘 `M4 G-13` 을 죽인 *필터 항진* 을 사전등록 시점에 잡는 방법)."""
    recs = RECORDS if n_rec is None else RECORDS[:n_rec]
    cells = premise_cells() if anchored else {}
    rows = []
    for w in WINDOWS:
        for ri, rec in enumerate(recs):
            c = cells.get((rec, w)) or {}
            t_end = float(c.get('t_end_true', 100.0))
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
                    is_pool = lv in POOL_LEVELS
                    rows.append(dict(
                        level=lv, cell=arm, record=rec, win=w, nowcast=base,
                        leadmean=base * 1.1, n_obs=M_CAP,
                        n_obs_learned=(M_OBS_CAP_LEARNED if arm == 'PROP' else None),
                        W_used=(W_ANCHOR if lv in ('ANCHOR', 'ORIG2', NOOP_LEVEL)
                                else 8.0),
                        W_over_Tp=4.0,
                        t_span=W_ANCHOR - 0.1 if lv in
                        ('ANCHOR', 'ORIG2', NOOP_LEVEL) else 7.9,
                        dt=0.1, t_end_abs=t_end, t0_abs=t_end,
                        cap_mode_used=CAP_MODE, cap_mode_sig='uniform',
                        pool_hash=(c.get('pool_hash', 'POOL') if is_pool else None),
                        pool_n=(c.get('pool_n', 7200) if is_pool else None),
                        dens_per_s=15.0, dens_last_1s=12.0, dens_last_02Tp=6.0,
                        n_last_scan=12,
                        q_hash=f'q{rec}{w}', gt_hash=f'g{rec}{w}',
                        sigma_bits=f's{rec}{w}', kp_bits=f'k{rec}{w}',
                        obs_hash=f'o{rec}{w}{lv}', nowcast_bits=f'n{rec}{w}{lv}{arm}',
                        leadmean_bits=f'l{rec}{w}{lv}{arm}',
                        n_atoms=(20 if arm == 'PROP' else 0),
                        varpro_fired=(arm == 'PROP'),
                        real_gate=(True if arm == 'SHIP' else None),
                        cap_init_method=('learned' if arm == 'PROP' else 'bank_mmse'),
                        cap_lambda_l1=1e-4,
                        **{'cap_' + k: v for k, v in SHIPPED_2026_08_22.items()}))
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

    # ③ 게이트가 깨끗한 픽스처를 통과한다 (⚠전제 파일이 **있어야** 한다)
    good = _fixture(6, 6)
    g = dict((k.split()[0], ok) for k, ok, _ in gates(good))
    chk(all(g.values()), '깨끗한 픽스처 FAIL: ' + str([k for k, v in g.items() if not v]))

    # ③b ★★**만족가능성 / 필터 항진 방지** — 모든 게이트가 실제로 무언가를 보았다.
    #    (오늘 `M4 G-13` 은 `0/12` 이 **항등적으로 참**인 실패였다.)
    zero = sorted(k for k, v in COVERAGE.items() if v <= 0)
    chk(not zero, f'커버리지 0 게이트: {zero}')
    chk(g.get('G-9') is True, 'G-9 메타게이트가 깨끗한 픽스처에서 PASS 해야 한다')
    for k in ('G-1b', 'G-1c', 'G-5b1', 'G-5b2', 'G-2a', 'G-10'):
        chk(COVERAGE.get(k, 0) > 0, f'{k} 커버리지 0 — 자료에 대고 만족 불가')

    # ③c ★전제 파일이 **실제로** 있고 앵커를 싣고 있는가 (사전등록 시점 확인)
    cells = premise_cells()
    chk(len(cells) == len(RECORDS) * len(WINDOWS),
        f'전제 P8 셀 {len(cells)} != {len(RECORDS) * len(WINDOWS)} '
        '(premise_m5b.py 를 먼저 돌려라)')
    chk(all(c.get('last_frame_nonempty') for c in cells.values()) if cells else False,
        '전제: 최신 프레임이 비어있는 셀이 있다 ⟹ G-1b 의 앵커가 무효')
    p7 = (premise().get('P7_default_bitidentical') or {})
    chk(int(p7.get('n_equal') or 0) == len(RECORDS),
        f'전제 P7 기본값 비트재현 {p7.get("n_equal")}/{len(RECORDS)}')

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
    mutate(lambda rs: rs[0].__setitem__('nowcast', float('nan')), 'G-0d')
    mutate(lambda rs: rs[0].__setitem__('t_span', 39.0)
           if rs[0]['level'] != 'ANCHOR' else rs[0].__setitem__('t_span', 1.0), 'G-1')
    # ★끝 시각 위조 — 수준이 **하나뿐이어도** 잡혀야 한다(M5 의 비교형은 못 잡았다)
    mutate(lambda rs: rs[0].__setitem__('t_end_abs', rs[0]['t0_abs'] + 0.2), 'G-1b')
    mutate(lambda rs: [r.__setitem__('t_end_abs', r['t_end_abs'] + 0.2)
                       for r in rs], 'G-1c')
    mutate(lambda rs: rs[0].__setitem__('n_obs', 599), 'G-2')
    mutate(lambda rs: [r.__setitem__('n_obs_learned', None) for r in rs
                       if r['cell'] == 'PROP'], 'G-2a')     # ★shim 사망 = 항등 통과 금지
    mutate(lambda rs: rs[0].__setitem__('q_hash', 'X'), 'G-2b')
    mutate(lambda rs: rs[0].__setitem__('sigma_bits', 'X'), 'G-2b')
    mutate(lambda rs: rs[0].__setitem__('cap_mode_used', 'uniform'), 'G-2c')
    mutate(lambda rs: [r.__setitem__('n_atoms', 0) for r in rs if r['cell'] == 'PROP'],
           'G-3')
    mutate(lambda rs: [r.__setitem__('nowcast_bits', 'X') for r in rs
                       if r['level'] == NOOP_LEVEL][:1], 'G-4')
    mutate(lambda rs: [r.__setitem__('pool_hash', 'X') for r in rs
                       if r['level'] in POOL_LEVELS][:1], 'G-5b2')
    mutate(lambda rs: [r.__setitem__('pool_hash', None) for r in rs
                       if r['level'] == 'ANCHOR'], 'G-5b2')  # 규약 위반(있어야 하는데 없다)
    mutate(lambda rs: [r.__setitem__('pool_hash', 'X') for r in rs
                       if r['level'] == 'TP2'], 'G-5b2')     # 규약 위반(없어야 하는데 있다)
    mutate(lambda rs: [r.__setitem__('dens_last_1s', None) for r in rs][:1], 'G-10')
    mutate(lambda rs: [r.__setitem__('cap_init_method', 'v2_konly') for r in rs
                       if r['cell'] == 'PROP'][:1], 'G-6')
    mutate(lambda rs: [r.__setitem__('nowcast', 9.0) for r in rs
                       if r['level'] == 'ORIG2' and r['cell'] == 'SHIP'][:1], 'G-7')
    mutate(lambda rs: rs[0].__setitem__('cap_varpro_steps', 0), 'G-6c')
    mutate(lambda rs: [r.__setitem__('cap_auto_switch_thresh', 8) for r in rs][:1],
           'G-6c')

    # ⑤ ★공허 참조 방어 — 참조가 비면 **FAIL** 이어야 한다(PASS 아님)
    global FROZEN_B2, PREMISE_JSON
    saved, FROZEN_B2 = FROZEN_B2, os.path.join(HERE, '__no_such_file__.jsonl')
    gg = dict((k.split()[0], ok) for k, ok, _ in gates(_fixture(6, 6)))
    FROZEN_B2 = saved
    chk(gg.get('G-5') is False, 'empty B-2 ref must FAIL')

    savedp, PREMISE_JSON = PREMISE_JSON, os.path.join(HERE, '__no_premise__.json')
    gg = dict((k.split()[0], ok) for k, ok, _ in gates(_fixture(6, 6, anchored=False)))
    PREMISE_JSON = savedp
    for k in ('G-1b', 'G-1c', 'G-5b1', 'G-5b2', 'G-8b', 'G-8c'):
        chk(gg.get(k) is False, f'전제가 없으면 {k} 는 FAIL 이어야 한다(공허 PASS 금지)')

    # ⑥ 소스 되읽기 상수가 문서의 주장을 지탱한다
    chk(W_ANCHOR == 40.0 and M_SCAN == 12 and M_CAP == 600, 'scene defaults')
    chk(SCENE_DEFAULTS['cap_mode'] == 'uniform', 'cap_mode 기본값이 uniform 이어야 한다')
    chk(TP_FACTOR_DEPLOY == 2.0, 'deploy tp_factor')
    chk(TOBS_FACTOR_TRAIN == (1.9, 2.3), 'train TOBS_FACTOR')
    chk(M_OBS_CAP_LEARNED == 400, 'learned M_OBS_CAP')
    chk(N_RUNS == 456, f'N_RUNS={N_RUNS}')

    # ⑦ ★수정 전 소스가 blob 에서 꺼내지고 md5 가 맞는다
    try:
        src = scene_pre_source()
        chk('cap_mode' not in src, '수정 전 소스에 cap_mode 가 있으면 blob 핀이 틀렸다')
    except Exception as e:                                   # noqa: BLE001
        chk(False, f'scene_pre_source: {e!r}')

    # ⚠`COVERAGE` 는 **마지막 `gates()` 호출**의 것이다 — 위에서 참조를 비운 채 부른 뒤라
    #   그대로 찍으면 0 이 잔뜩 나온다(공허한 자기보고). 깨끗한 픽스처로 다시 채운다.
    gates(_fixture(6, 6))
    clean_cov = dict(COVERAGE)

    print(f'selftest: {"PASS" if not fails else "FAIL"}  ({len(fails)} 실패)')
    for f in fails:
        print('  -', f)
    if not fails:
        print(f'  levels={LEVEL_ORDER}+{NOOP_LEVEL}  arms={ARM_ORDER}  '
              f'N_RUNS={N_RUNS}  thresholds={THRESHOLD_12}/12, {THRESHOLD_36}/36')
        print(f'  cap_mode={CAP_MODE!r} (기본값={SCENE_DEFAULTS["cap_mode"]!r})  '
              f'W_ANCHOR={W_ANCHOR}  deploy={TP_FACTOR_DEPLOY}xTp  '
              f'train={TOBS_FACTOR_TRAIN}xTp')
        print('  커버리지(깨끗한 픽스처): '
              + ', '.join(f'{k}={v}' for k, v in sorted(clean_cov.items())))
    return 0 if not fails else 1


def refcheck():
    """★교훈 ①: 외부 참조를 **실제 파일에 대고** 돌려 비지 않음을 확인한다.
    ★교훈 ②(오늘 신설): 게이트가 **자료에 대고 만족 가능**한지 커버리지로 확인한다."""
    ref = b2_scene_ref()
    print(f'FROZEN_B2 = {FROZEN_B2}')
    print(f'  존재={os.path.exists(FROZEN_B2)}  ref 레코드={len(ref)}/{len(RECORDS)}')
    miss = [r for r in RECORDS if r not in ref]
    print(f'  누락={miss}')
    cells = premise_cells()
    pre = premise()
    print(f'PREMISE = {PREMISE_JSON}  존재={os.path.exists(PREMISE_JSON)}')
    print(f'  P8 셀={len(cells)}/{len(RECORDS) * len(WINDOWS)}  '
          f'최신프레임 비어있지않음={sum(1 for c in cells.values() if c.get("last_frame_nonempty"))}')
    print(f'  P7 기본값 비트재현={(pre.get("P7_default_bitidentical") or {}).get("n_equal")}'
          f'/{(pre.get("P7_default_bitidentical") or {}).get("n_checked")}')
    gates(_fixture(6, 6))
    zero = sorted(k for k, v in COVERAGE.items() if v <= 0)
    print('  커버리지: ' + ', '.join(f'{k}={v}' for k, v in sorted(COVERAGE.items())))
    print(f'  커버리지 0 게이트={zero}')
    print(f'gates_b3 임포트: RECORDS={len(RECORDS)} WINDOWS={WINDOWS} '
          f'SEED={SEED} NOISE={NOISE}')
    okc = bool(ref) and not miss and len(cells) == len(RECORDS) * len(WINDOWS) and not zero
    return 0 if okc else 1


if __name__ == '__main__':
    if '--refcheck' in sys.argv:
        sys.exit(refcheck())
    sys.exit(selftest())
