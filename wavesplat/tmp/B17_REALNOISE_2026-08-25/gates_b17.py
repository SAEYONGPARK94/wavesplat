"""B-17 게이트·팔·문턱·라벨 — **단일 출처**. 러너와 분석기는 이 파일만 읽는다.

⛔이 파일은 **결과가 하나라도 나오기 전에** 커밋된다(git 앵커). 문턱·라벨은 여기에만 있고
사후에 바꾸지 않는다.

## 왜 이 배치인가 (외부 심사 처방 `P-5`, 2026-08-25) — 축자
> *"실측 잡음을 상수 σ 에서 **거리·입사각 의존**으로 바꿔 12 레코드를 재측정하라 —
>  σ(d) 는 논문 곡선을 그대로 쓰면 **손으로 고를 것이 없다**.
>  판정 기준: `B-2` 의 **층 분해(대형 박스 5/5 vs 소형 2/7)가 유지되는가.**
>  유지되면 `F-2` 가 닫히고 음성이 **방법의 성질**로 승격된다.
>  무너지면 ***음성은 관측 모형의 성질***이고, 그 문장이 논문에 들어가야 한다."*

## ⚠처방의 수치 두 개를 **결과 전에** 정정한다 (둘 다 저장소 동결 기록에서 도출)
1. `5/5 vs 2/7` 은 **폐기된 기준** `ZERO_REF = 1.0` 에서 나온 값이다.
   참 제로필드로 재채점하면 **4/5 vs 1/7** 이다(`B2_REAL_2026-08-23/AMENDMENT_ZEROREF.md` §2 표).
2. 그 `1/7` 은 **BS 수심 17.0 m** 에서 나왔고, `scene_station.py:37-39` 가 그 값을
   **30.0 m 로 정정**했다. 정정된 코드에서 같은 팔·같은 잡음(σ=0.02)은
   `B-14b` 의 `IID σ=0.02` 행으로 이미 존재하며, 그 층 분해는 **4/5 vs 3/7** 이다.
   ⟹ ★**이 배치의 기준선은 `B-2` 의 숫자가 아니라 이 배치 자신의 `CONST` 팔**이고,
   `CONST` 는 `B-14b` 와 비트 동일해야 한다(게이트 `G-1b`).
   ⛔그래서 층 규칙은 **절대 문턱이 아니라 `CONST` 대비 비율**로 쓴다(§ `stratum_label`).

## 두 물음 (결과 전 기재, 분리 선언)
- **Q1 (심사의 문자 그대로)**: 논문 곡선을 **그대로** 썼을 때 층 분해가 유지되는가?
  → 팔 `RANGEANG`. 이것이 **PRIMARY** 이고 `F-2` 판정을 낸다.
- **Q2 (귀속)**: 무언가 바뀌면 그것은 잡음의 **형태** 때문인가 **수준** 때문인가?
  → `RANGEANG-MM`(형태만, 평균분산 고정) + `CONST-010`(수준만, 형태 없음).
  ⚠논문 곡선을 실측 기하에 얹으면 **수준이 층마다 반대 방향으로 움직인다**
  (소형 박스는 r≪d_ref 라 σ→σ_0=0.01 로 **조용해지고**, 대형 박스는 원거리·근접입사라
  **시끄러워진다**). `B-2b` 가 이미 σ 0.02→0.005 에서 짝 10/12 개선을 쟀으므로
  **수준 교락은 가정이 아니라 측정된 위험**이다. Q2 가 없으면 Q1 은 해석 불가다.
"""
import hashlib
import json
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
B2 = os.path.abspath(os.path.join(HERE, '..', 'B2_REAL_2026-08-23'))
if B2 not in sys.path:
    sys.path.insert(0, B2)

# ── 표본·층: **동결 파일에서 import 한다** (손으로 옮기지 않는다) ─────────────
# ⛔`gates_b2` 의 md5 를 함께 못박는다 — 그 파일이 바뀌면 여기서 **시끄럽게** 실패한다.
GATES_B2_MD5 = 'a5d71fafa08f50fa1b6fda9650863392'
B2_RESULTS_MD5 = 'fcf93699fe7cf6b81632a12f185097c1'
SCENE_STATION_MD5 = '18183f898008d239db002a3634f19686'


def _md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


assert _md5(os.path.join(B2, 'gates_b2.py')) == GATES_B2_MD5, \
    'gates_b2.py 가 바뀌었다 — 층 정의의 출처가 흔들린다'

import gates_b2 as B2G                                        # noqa: E402

RECORDS = list(B2G.RECORDS)          # 12 독립 레코드 (동결)
LARGE_BOX = list(B2G.LARGE_BOX)      # 40 m 이상 — AA×3 · LJ · YS
SMALL_BOX = list(B2G.SMALL_BOX)      # 13–19 m, 개구/λ≈1 — BS×7
WIN = B2G.WIN                        # 3
SEED = B2G.SEED                      # 42
N = len(RECORDS)
FIELD = 'nowcast'                    # PRIMARY 종점 (B-2 와 동일)

# ── 논문의 잡음 곡선 ── ⛔**줄번호로 인용하지 않는다** (2026-08-26 수리) ──────
# 초판은 `thesis_main.tex:1490/:1492/:1493/:1545/:5134-5136` 을 **줄번호로** 인용했다.
# 2026-08-25~26 편집으로 그 포인터가 전부 약 +60 줄 어긋났고 `:5134` 는 지금
# **RETRACTED 주석 블록**(인용금지 수치)을 가리킨다. ⚠값 자체는 실측 확인됐으므로
# **측정은 무손상**이고 틀린 것은 **포인터**다. ⟹ 포인터를 두 가지로 갈아 끼운다:
#   (a) **코드 단일출처가 있는 상수는 코드에서 파싱해 읽는다** (손복사 0회)
#   (b) 없는 상수는 선언하고 **정규식 앵커**로 런타임에 논문과 대조한다 (`G-10`)
# ⚠이 수리 도중 실측: 같은 세션의 두 호출 사이에 `thesis_main.tex` 의 줄번호가 움직였다
#   (다른 세션이 편집 중) ⟹ **줄번호는 이 저장소에서 원리적으로 못 쓴다.**
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
THESIS = os.path.join(ROOT, 'thesis', 'thesis_main.tex')


def _arg_default(relpath, func, arg):
    """소스에서 함수 인자 기본값을 **파싱**해 읽는다(무거운 import 부작용 없음)."""
    import ast
    src = open(os.path.join(ROOT, relpath), encoding='utf-8').read()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == func:
            names = [a.arg for a in node.args.args]
            defs = ([None] * (len(names) - len(node.args.defaults))
                    + list(node.args.defaults))
            for nm, dv in zip(names, defs):
                if nm == arg and dv is not None:
                    return ast.literal_eval(dv)
    raise KeyError(f'{relpath}::{func}({arg}=) 의 기본값을 찾지 못했다')


# ✅σ_r · d_ref 는 **패키지 코드에 단일출처가 이미 있다** ⟹ 거기서 읽는다.
#   · σ_r   = `wavesim/noise_model.py` `add_noise(sigma_range=...)`  (LiDAR 거리 잡음)
#   · d_ref = `wavesplat/model/confidence.py` `observation_weights(d_ref=...)`
SIGMA_R = float(_arg_default('wavesim/noise_model.py', 'add_noise', 'sigma_range'))
D_REF = float(_arg_default('wavesplat/model/confidence.py',
                           'observation_weights', 'd_ref'))
# ⛔σ_0 만은 코드에 단일출처가 **없다** — 잡음 함수에 기저항 자체가 없다.
#   ⟹ 여기서 선언하고 `G-10` 이 런타임에 논문 본문과 대조한다. (손복사 1건, 명시)
SIGMA0 = 0.01
# 배포 실측 잡음 = 동결 장면 빌더의 기본값 그 자체 (손복사 금지).
SIGMA_DEPLOY = float(_arg_default('tmp/B2_REAL_2026-08-23/scene_station.py',
                                  'scene_station', 'noise'))

# 논문 본문 앵커 — ⛔줄번호가 아니라 **정규식**이다(편집에 견딘다).
# (부록 하이퍼파라미터 표가 같은 세 값을 다시 싣지만 그 표는 앵커로 걸지 않는다.)
THESIS_ANCHORS = (
    ('sigma0', r'\$\\sigma_0\s*=\s*([0-9.]+)\$\\,m은\s*기저', lambda: SIGMA0),
    ('sigma_r', r'\$\\sigma_r\s*=\s*([0-9.]+)\$\\,m은\s*기준', lambda: SIGMA_R),
    ('d_ref', r'd_\{\\mathrm\{ref\}\}\s*=\s*([0-9.]+)\\,\\text\{m\}',
     lambda: D_REF),
)


def thesis_anchors(path=THESIS):
    """세 상수를 논문에서 **정규식으로** 찾아 *현재* 줄번호와 값을 낸다.

    반환 dict 은 분석기가 `verdict` 에 그대로 싣는다 ⟹ 포인터가 **결과와 함께 굳는다**.
    파일이 없으면 `None`(조판 환경 밖에서도 러너가 죽지 않게).
    """
    import re
    if not os.path.exists(path):
        return None
    lines = open(path, encoding='utf-8').read().splitlines()
    out = {}
    for name, pat, want_fn in THESIS_ANCHORS:
        want, rx, hit = float(want_fn()), re.compile(pat), None
        for i, ln in enumerate(lines, 1):
            m = rx.search(ln)
            if m:
                hit = (i, float(m.group(1)))
                break
        out[name] = {'line_now': None if hit is None else hit[0],
                     'found': None if hit is None else hit[1],
                     'want': want,
                     'ok': hit is not None and abs(hit[1] - want) < 1e-12}
    return out

# ── 센서 기하 ───────────────────────────────────────────────────────────────
# ✅**수평 위치는 자료에서 도출됐다** — `sensor_geom.py` 참조. WASS 유효마스크가
#   카메라 시야의 해면 투영이고, 그 쐐기 꼭짓점이 카메라 연직 아래점이다.
#   실측(2026-08-25): 쐐기 적합 3건 |apex| = 6.18 / 6.55 / 7.88 m (r² 0.987–1.000,
#   패치 반대각선 48–163 m) · 직사각 크롭 8건 격자중심 |c| = 0.02–0.08 m ·
#   1건(AA15b) `inconclusive`. ⟹ **12 중 11 이 (0,0) 을 가리킨다.**
SENSOR_XY = (0.0, 0.0)
SENSOR_XY_TOL = 10.0          # G-2: 도출된 원점이 이 안에 있어야 한다 [m]
# ⚠**`G-2` 는 [기술적] 게이트다 — 판정 근거로 세지 않는다.** 이 허용오차(10 m)는
#   `sensor_origin.json` 의 관측값(|apex| = 6.18/6.55/7.88 m)을 **본 뒤에** 정해졌고
#   그 파일은 동결이므로 **산술적으로 통과가 보장**된다. 즉 이것은 가설 검정이 아니라
#   *"기하 도출 파이프라인이 돌았고 선언값과 같은 자리를 가리킨다"* 는 **도구 검사**다.
#   ⛔이 게이트의 PASS 를 증거로 인용하지 마라.
# ⛔**높이는 도출되지 않는다** — .nc 전역 속성에 카메라 위치가 없다(기선 2.03 m 뿐).
#   ⟹ **결과 전에 선언**하고 민감도 팔로 노출한다. 자료에서 나온 값이 **아니다**.
SENSOR_H = 12.0               # 선언값. 네 사이트 전부 고정 상부구조물(타워·등대·플랫폼).
SENSOR_H_ALT = 30.0           # 민감도 팔. ⛔확증 가족 밖(탐색적).
COS_MIN = 0.05                # 입사각 클램프(≈87°). 결과 전 고정. 클램프 비율을 행에 싣는다.
MM_TARGET = SIGMA_DEPLOY      # 평균분산 정합 목표 = 배포 수준

# ── 팔 (⛔빈 dict 금지 — 전부 명시) ─────────────────────────────────────────
_S = dict(x_s=SENSOR_XY[0], y_s=SENSOR_XY[1])
ARMS = {
    # 배포 = 양성대조. B-2 `PROP` · B-14b `IID σ=0.02` 와 **비트 동일**해야 한다.
    'CONST':        dict(name='CONST', model='const', sigma=SIGMA_DEPLOY, mm=False),
    # 수준 통제: 논문 곡선이 소형 박스(r≪d_ref)에서 수렴하는 값 그 자체.
    'CONST-010':    dict(name='CONST-010', model='const', sigma=SIGMA0, mm=False),
    # 논문 분산식(`THESIS_ANCHORS['sigma0'/'sigma_r']`) 축자, 거리만.
    'RANGE':        dict(name='RANGE', model='range', h=SENSOR_H, mm=False, **_S),
    # ★PRIMARY — 논문 분산식 × 논문 신뢰가중(`THESIS_ANCHORS['d_ref']` 식)의 θ 의존.
    'RANGEANG':     dict(name='RANGEANG', model='range_angle', h=SENSOR_H, mm=False, **_S),
    # 형태 통제: 같은 형태, 평균분산을 배포 수준에 고정.
    'RANGEANG-MM':  dict(name='RANGEANG-MM', model='range_angle', h=SENSOR_H, mm=True, **_S),
    # h 민감도. ⛔확증 가족 밖.
    'RANGEANG-H30': dict(name='RANGEANG-H30', model='range_angle', h=SENSOR_H_ALT,
                         mm=False, **_S),
}
ARM_ORDER = ['CONST', 'CONST-010', 'RANGE', 'RANGEANG', 'RANGEANG-MM', 'RANGEANG-H30']
REF_ARM = 'CONST'
PRIMARY_ARM = 'RANGEANG'
# 확증 가족 = `CONST` 대비 짝 부호검정을 거는 팔들. `RANGEANG-H30` 은 **밖**이다.
FAMILY = ['CONST-010', 'RANGE', 'RANGEANG', 'RANGEANG-MM']
EXPLORATORY = ['RANGEANG-H30']

# ── 추정기 팔 (B-2 `PROP` = 제안 표현 강제) + ★배포 기본값 셋을 **명시** ──────
# ⚠2026-08-22 배포 변경 3건을 dict(DEFAULT_CONFIG) 흡수에 맡기지 않고 직접 적는다.
#   되돌리기는 `sequential_update.pre_2026_08_22_config()`.
ESTIMATOR_CFG = {
    'real_domain_gate': False,      # 실측 게이트 우회 (B-2 `PROP`)
    'init_method': 'learned',       # 제안 표현 강제 (B-2 `PROP`)
    'varpro_steps': 400,            # 배포 기본값 (2026-08-22)
    'learned_init_iters': 0,        # 배포 기본값 (2026-08-22)
    'auto_switch_thresh': 0,        # 배포 기본값 (2026-08-22)
    'lambda_l1': 1e-4,              # LAMBDA_L1_DEFAULT (2026-08-11 배포)
}
# ── ★개입 노브의 **증거** ── ⛔`upd.config` 되읽기는 항등식이라 금지 ──────────
# BLOCKER (2026-08-26, 런 전 수리). 초판은 `READBACK_KEYS` 를 `upd.config` 에서 되읽어
# `ESTIMATOR_CFG` 와 대조했다. 그러나 `sequential_update.py:399` 가
#     self.config = config
# 로 **입력 dict 을 복사 없이 그대로 붙든다** ⟹ 그 대조는 방금 쓴 것을 다시 읽는
# `cfg[k] == cfg[k]` 이고 **실패할 수 없다.** 게다가 그 항등식은 **한 번도 읽히지 않는
# 키까지 통과**시킨다(아래 `INERT_KEYS`) ⟹ *방벽을 붙였다고 믿게 만든다*
# (= `W8` `G-4` 공허한 통과의 재현이고, 문언이 `W7` 의 *"설정을 넣었다 ≠ 설정이 먹었다"*
#  를 인용하고 있어 **더 나쁘다**).
# ⟹ 증거를 셋으로 나눈다. **config 와 독립적으로 존재하는 것만 증거로 센다.**
#   (1) DERIVED — 생성자가 config 와 **따로 스냅샷한 속성** (`sequential_update.py:398-413`)
#   (2) COUNTER — **실행이 남긴 수** (`initialize()` 반환 dict 안의 계수기)
#   (3) INERT   — 이 팔에서 **도달 불가**. 값 대조를 하지 않고, 대신 라우터가 발화하지
#                 않았음을 **양성 증명**한다(`G-7c`).
DERIVED_READBACK = {          # ESTIMATOR_CFG 키 → SequentialUpdater 속성명
    'init_method': 'init_method',      # sequential_update.py:413 (config.get 결과를 스냅샷)
    'lambda_l1': 'lambda_l1',          # sequential_update.py:401
}
COUNTER_READBACK = {          # ESTIMATOR_CFG 키 → 행의 `counters` 키 + 그 계수의 출처
    # `_initialize_learned` 의 결합적합 루프가 **실제로 돈 횟수** = len(loss_history)
    # (`sequential_update.py:889` max_iters → `:890-899` 루프 → `:931` loss_history)
    'learned_init_iters': ('n_loss_steps', 'len(init_result["loss_history"])'),
    # VARPRO 가 **실제로 집행한 스텝** (`varpro.py:151 steps_executed` → `:249 vp_steps`)
    'varpro_steps': ('vp_steps_executed', 'init_result["varpro"]["vp_steps"]'),
}
# ⛔이 팔에서 **도달 불가한 키** — 둘 다 `if self.init_method == 'auto'`(`:689`) 블록
#   아래에서만 읽힌다: `real_domain_gate` 는 `:706`, `auto_switch_thresh` 는
#   `_auto_select_init`(`:1008`) 안이고 그 함수는 `auto` 경로에서만 불린다.
#   이 배치의 팔은 `init_method='learned'` 를 **명시**하므로 두 키는 **평생 읽히지 않는다.**
#   ⟹ ★**팔 정의에는 문서용으로 남기되 게이트에서 값 대조를 하지 않는다**
#      (대조해 봐야 항등식이고, PASS 가 *"먹었다"* 를 뜻하지 않는다).
#      대신 `G-7c` 가 `_auto_selected`/`_auto_real_gate` 가 **끝까지 None** 임을 확인해
#      *"라우터가 발화하지 않았다 = 이 두 키가 읽히는 코드가 실행되지 않았다"* 를 **양성 증명**한다.
INERT_KEYS = ('real_domain_gate', 'auto_switch_thresh')
ROUTER_ATTRS = ('_auto_selected', '_auto_real_gate', '_auto_n_selected')
ATTR_MISSING = '<<ATTR-MISSING>>'   # getattr 실패를 None 과 구별하는 표식

assert (set(DERIVED_READBACK) | set(COUNTER_READBACK) | set(INERT_KEYS)
        == set(ESTIMATOR_CFG)), \
    'ESTIMATOR_CFG 의 모든 키는 파생·계수·도달불가 셋 중 하나로 분류돼야 한다'

# ── 문턱 (결과 전 `comb` 로 계산) ───────────────────────────────────────────
# 단측 부호검정, n=12:
#   10/12 = 0.019287 · 11/12 = 0.003174 · 12/12 = 0.000244  (바닥, 단측)
#   ⚠기록의 "바닥 p=0.0005" 는 **양측값**(2·0.000244=0.000488)이다. 여기서는 **단측**을 쓴다
#     (`gates_b2.sign_test_one_sided` 와 같은 규약).
# 가족 = FAMILY 4개 ⟹ α = 0.05/4 = 0.0125 ⟹ **11/12 라야 확증**(0.003174 ≤ 0.0125).
NOMINAL = 10
CONFIRM = 11
FAMILY_ALPHA = 0.05 / len(FAMILY)


def sign_test_one_sided(w, n=N):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


assert abs(sign_test_one_sided(NOMINAL) - 0.019287) < 1e-6
assert abs(sign_test_one_sided(CONFIRM) - 0.003174) < 1e-6
assert sign_test_one_sided(CONFIRM) <= FAMILY_ALPHA < sign_test_one_sided(NOMINAL)

# ── 층 규칙 (결과 전 확정) ──────────────────────────────────────────────────
# `gap(arm) = wins_large/|LARGE| − wins_small/|SMALL|`.
# 기준은 **이 배치 자신의 `CONST` 팔**이다(위 §정정 2 참조).
# ⛔절대 문턱을 쓰지 않는 이유: `B-2` 의 `1/7` 은 수심 17.0 m 의 산물이고 코드가 30.0 m 로
#   정정됐다. 정정된 코드의 기준선은 이 배치 안에서만 정직하게 얻어진다.
STRATUM_SURVIVE_RATIO = 0.50    # gap 의 **절반 이상**이 남으면 유지
STRATUM_COLLAPSE_RATIO = 0.25   # gap 의 **4분의 1 이하**만 남으면 붕괴
STRATUM_SMALL_CAP = 6           # 소형 박스에서 이 이상 이기면(=6/7) 그것만으로 붕괴
STRATUM_LARGE_SLACK = 1         # 대형 층이 `CONST` 보다 이 이상 떨어지면 유지로 치지 않는다
STRATUM_SMALL_SLACK = 1         # 소형 층이 `CONST` 보다 이 이상 오르면 유지로 치지 않는다
# ★m-10 (결과 전 고정): `STRATUM-COLLAPSES` 가 **공시된 취약 레코드**에 기대는지
#   `meaning` 에 **자동으로** 적는다. PREREG §5 가 결과 전에 `YS17 RANGEANG` 의
#   클램프 **32.8%** · σ_rms 가 `CONST` 의 5.8배를 수치까지 공시했고
#   `STRATUM_LARGE_SLACK = 1` 이 그 한 건을 흡수하도록 설계됐다. 남은 결함은
#   `verdict.label` 이 **단일 문자열**이라 그 조건이 하류로 전달되지 않는 것뿐이다
#   ⟹ 라벨이 아니라 `meaning` 에 실어 보낸다.
CLAMP_FRAC_THRESH = 0.20        # `frac_cos_clamped` 가 이보다 크면 '취약 레코드'로 표시
# ★층은 **두 방향으로** 무너질 수 있고 둘 다 같은 결론을 낸다:
#   (a) 소형이 이기기 시작한다  (b) 대형이 지기 시작한다
#   어느 쪽이든 *"잡음 모형만 바꿨는데 층이 흔들렸다"* ⟹ **층은 관측 모형의 성질**이다.
#   ⛔`gap` 만 보면 (b) 를 놓친다(대형 1/5·소형 0/7 도 gap=+0.2 다) ⟹ 대형 하한을 함께 건다.
# ⚠**층 대비에는 검정력이 없다**(대형 5 · 소형 7). 2026-08-23 야간 감사가 층간 대비 3검정을
#   전부 널로 되돌린 전례가 있다 ⟹ ★**p 값을 주장하지 않는다.** Fisher 는 **기술적으로만** 낸다.


def fisher_one_sided(a, b, c, d):
    """2×2 [[a,b],[c,d]] 단측(대형이 더 잘함) 정확검정. ⚠**기술용**이다."""
    n = a + b + c + d
    r1, r2, c1 = a + b, c + d, a + c
    lo, hi = max(0, c1 - r2), min(r1, c1)
    tot = sum(comb(r1, k) * comb(r2, c1 - k) for k in range(lo, hi + 1))
    num = sum(comb(r1, k) * comb(r2, c1 - k) for k in range(a, hi + 1))
    return num / tot if tot else None


# ── 집계 ────────────────────────────────────────────────────────────────────
def _vals(rows, arm, field=FIELD, subset=None):
    return {r['record']: r[field] for r in rows
            if r['cell'] == arm and 'error' not in r and r.get(field) is not None
            and (subset is None or r['record'] in subset)}


def beats_zero(rows, arm, subset=None):
    """★참 제로필드 기준(`zero_ref`, 행에 실린 값). ⛔상수 1.0 금지."""
    z = {r['record']: r['zero_ref'] for r in rows if r['cell'] == arm and 'error' not in r}
    v = _vals(rows, arm, subset=subset)
    win = sorted(k for k, x in v.items() if x < z[k])
    loss = sorted(k for k, x in v.items() if not x < z[k])
    w = len(win)
    return {'arm': arm, 'n': len(v), 'wins': w,
            'win_records': win, 'loss_records': loss,
            'p_one_sided': sign_test_one_sided(w, len(v)) if v else None,
            'median': (sorted(v.values())[len(v) // 2] if v else None)}


def paired(rows, arm, ref=REF_ARM, field=FIELD, subset=None):
    """짝 비교(같은 레코드). 기준값이 소거되므로 `zero_ref` 와 무관하다."""
    import statistics as st
    a, b = _vals(rows, arm, field, subset), _vals(rows, ref, field, subset)
    ks = sorted(set(a) & set(b))
    w = sum(1 for k in ks if a[k] < b[k])
    return {'arm': arm, 'ref': ref, 'n': len(ks), 'better': w,
            'rel_median_pct': (st.median([(a[k] - b[k]) / b[k] * 100 for k in ks])
                               if ks else None),
            'p_one_sided': sign_test_one_sided(w, len(ks)) if ks else None}


def stratum(rows, arm):
    lg, sm = beats_zero(rows, arm, LARGE_BOX), beats_zero(rows, arm, SMALL_BOX)
    fl = lg['wins'] / lg['n'] if lg['n'] else float('nan')
    fs = sm['wins'] / sm['n'] if sm['n'] else float('nan')
    return {'arm': arm, 'large': lg, 'small': sm, 'frac_large': fl, 'frac_small': fs,
            'gap': fl - fs,
            'fisher_p_descriptive': fisher_one_sided(lg['wins'], lg['n'] - lg['wins'],
                                                     sm['wins'], sm['n'] - sm['wins'])}


def clamp_map(rows, arm):
    """레코드 → `frac_cos_clamped` (행의 `sigma_stats` 에서 **그대로** 읽는다)."""
    return {r['record']: (r.get('sigma_stats') or {}).get('frac_cos_clamped')
            for r in rows if r['cell'] == arm and 'error' not in r}


def _st_wins(st):
    return set(st['large']['win_records']) | set(st['small']['win_records'])


def _st_losses(st):
    return set(st['large']['loss_records']) | set(st['small']['loss_records'])


def _clamp_note(arm_st, ref_st, clamp, thresh=CLAMP_FRAC_THRESH):
    """★m-10: 붕괴 판정이 **공시된 취약 레코드**(클램프 과다)에 기댔는지 자동 기재."""
    if clamp is None:
        return ('\n  ⚠클램프 진단이 넘어오지 않았다(`clamp_map()` 미전달) '
                '⟹ 취약 레코드 의존 여부 **미확인**.')
    heavy = sorted(k for k, v in clamp.items() if v is not None and v > thresh)
    # `CONST` 는 이겼는데 이 팔에서 진 레코드 = 붕괴를 실제로 만든 것들
    regress = sorted(_st_wins(ref_st) & _st_losses(arm_st))
    if not heavy:
        return (f'\n  ✅클램프>{thresh:.0%} 레코드 **0개** '
                f'⟹ 이 붕괴는 `cos θ` 클램프 인공물에 기대지 않는다.')
    flipped = sorted(set(heavy) & set(regress))
    return (f'\n  ⚠클램프>{thresh:.0%} 레코드 {heavy} (PREREG §5 가 결과 전 공시) · '
            f'승패를 가른 레코드 {regress} ⟹ 그중 취약 레코드 '
            f'**{len(flipped)}/{max(len(regress), 1)}건** {flipped} '
            + ('⟹ ⛔**붕괴가 공시된 취약 레코드에 기댄다 — 그만큼 깎아 읽어라.**'
               if flipped else
               '⟹ ✅**붕괴는 취약 레코드 밖에서 일어났다.**'))


def stratum_label(arm_st, ref_st, clamp=None):
    """★`F-2` 판정. 규칙은 **결과 전** 이 함수에 있다. 기준 = 같은 배치의 `CONST`.

    `clamp` = `clamp_map(rows, arm)`; 넘기면 붕괴 판정에 **m-10 취약 레코드 주석**이 붙는다.
    """
    import math
    g, g0 = arm_st['gap'], ref_st['gap']
    small_w = arm_st['small']['wins']
    if (math.isnan(g) or math.isnan(g0) or arm_st['large']['n'] != len(LARGE_BOX)
            or arm_st['small']['n'] != len(SMALL_BOX)):
        return ('STRATUM-NO-DATA',
                f'표본이 모자란다 (대형 {arm_st["large"]["n"]}/{len(LARGE_BOX)} · '
                f'소형 {arm_st["small"]["n"]}/{len(SMALL_BOX)}) ⟹ 판정하지 않는다')
    if g0 <= 0:
        return ('STRATUM-BASELINE-ABSENT',
                f'`CONST` 자신이 층을 만들지 않는다(gap={g0:+.3f}) ⟹ `F-2` 를 이 배치로 물을 수 없다')
    large_w, rl, rs = arm_st['large']['wins'], ref_st['large']['wins'], ref_st['small']['wins']
    large_ok = large_w >= rl - STRATUM_LARGE_SLACK       # 대형이 여전히 이긴다
    small_ok = small_w <= rs + STRATUM_SMALL_SLACK       # 소형이 여전히 진다
    tag = (f'대형 {large_w}/{arm_st["large"]["n"]} (CONST {rl}) · '
           f'소형 {small_w}/{arm_st["small"]["n"]} (CONST {rs}) · '
           f'gap {g:+.3f} vs CONST {g0:+.3f}')
    if large_ok and small_ok and g >= STRATUM_SURVIVE_RATIO * g0:
        return ('STRATUM-SURVIVES',
                f'{arm_st["arm"]} 에서도 층이 남는다 ({tag}) '
                f'⟹ `F-2` 가 닫히고 음성이 **방법의 성질**로 승격된다')
    if not large_ok or small_w >= STRATUM_SMALL_CAP or g <= STRATUM_COLLAPSE_RATIO * g0:
        why = ('대형이 무너져서' if not large_ok else
               '소형이 회복해서' if small_w >= STRATUM_SMALL_CAP else 'gap 이 사라져서')
        return ('STRATUM-COLLAPSES',
                f'{arm_st["arm"]} 에서 층이 {why} 무너진다 ({tag}) '
                f'⟹ ★**실측 음성은 관측 모형의 성질이다** — 그 문장이 논문에 들어가야 한다'
                + _clamp_note(arm_st, ref_st, clamp))
    return ('STRATUM-UNDECIDED',
            f'{arm_st["arm"]} — 두 문턱 사이 ({tag}). ⛔어느 쪽으로도 승격하지 않는다')


def paired_label(pv):
    """부차 — 잡음 모형 교체가 짝으로 무엇을 하는가 (가족단위 α=0.0125)."""
    w, n = pv['better'], pv['n']
    if w >= CONFIRM:
        return ('REALNOISE-HELPS', f'{pv["arm"]} 이 CONST 를 {w}/{n} 로 이긴다 (가족단위 통과)')
    if n - w >= CONFIRM:
        return ('REALNOISE-HURTS', f'{pv["arm"]} 이 CONST 에 {n-w}/{n} 로 진다 (가족단위 통과)')
    if w >= NOMINAL or n - w >= NOMINAL:
        return ('REALNOISE-NOMINAL', f'{pv["arm"]} {w}/{n} — 명목만. 탐색적으로 읽는다')
    return ('REALNOISE-NULL', f'{pv["arm"]} {w}/{n} — 짝으로 갈리지 않는다')


# ── 게이트 ──────────────────────────────────────────────────────────────────
def gates(rows, sensor_origin=None, b2_rows=None, b14b_rows=None):
    ok = [r for r in rows if 'error' not in r]
    out = [('G-0 오류 0행', len(ok) == len(rows), f'{len(rows) - len(ok)} 행 오류')]
    need = len(ARMS) * N
    out.append(('G-0b 표본 수', len(ok) == need, f'{len(ok)}/{need}'))

    # ★G-1 양성대조 — CONST 가 **동결 B-2 `PROP` 행과 비트 동일**해야 한다.
    #   ⚠BS 7 레코드는 `scene_station.py:37-39` 의 수심 정정(17.0→30.0) 때문에 **원리적으로
    #     불가능**하다(B-14b `RESULT_B14B.md` 마지막 줄이 같은 사실을 적었다).
    #     ⟹ 정정에 영향받지 않는 **5 레코드**(AA×3·LJ·YS)에서 비트 동일을 요구한다.
    unaffected = [r for r in RECORDS if not r.startswith('BS')]
    if b2_rows is not None:
        ref = {r['record']: r['nowcast'] for r in b2_rows
               if r.get('cell') == 'PROP' and 'error' not in r}
        got = {r['record']: r[FIELD] for r in ok if r['cell'] == 'CONST'}
        hit = [k for k in unaffected if k in ref and k in got and got[k] == ref[k]]
        bad = [k for k in unaffected if k in ref and k in got and got[k] != ref[k]]
        out.append((f'G-1 양성대조: CONST ≡ B-2 PROP (수심 정정 무관 {len(unaffected)} 레코드, 비트)',
                    bool(ref) and not bad and all(k in got for k in unaffected),
                    (f'불일치 {bad}' if bad else
                     f'{len(hit)}/{len(unaffected)} 비트 동일 '
                     f'(CONST 행 {sum(1 for k in unaffected if k in got)}개)')))
    # ★G-1b 양성대조 2 — 정정된 수심으로 돌린 `B-14b IID σ=0.02` 와 **12/12 비트 동일**.
    if b14b_rows is not None:
        ref2 = {r['record']: r['nowcast'] for r in b14b_rows
                if r.get('obs_model') == 'IID' and r.get('noise') == 0.02 and 'error' not in r}
        got = {r['record']: r[FIELD] for r in ok if r['cell'] == 'CONST'}
        hit2 = [k for k in RECORDS if k in ref2 and k in got and got[k] == ref2[k]]
        bad2 = [k for k in RECORDS if k in ref2 and k in got and got[k] != ref2[k]]
        out.append(('G-1b 양성대조: CONST ≡ B-14b IID σ=0.02 (12 레코드, 비트)',
                    bool(ref2) and not bad2 and len(got) == N,
                    (f'불일치 {bad2}' if bad2 else
                     f'{len(hit2)}/{N} 비트 동일 (CONST 행 {len(got)}개)')))

    # G-2 [기술적] 센서 원점 도출 파이프라인이 돌았고 선언값과 같은 자리를 가리킨다.
    #   ⛔**판정 근거로 세지 않는다** — 허용오차 10 m 가 동결 관측값(|apex| ≤ 7.88 m)을
    #   본 뒤에 정해졌으므로 통과가 산술적으로 보장된다(도구 검사다). 상수 옆 주석 참조.
    if sensor_origin is not None:
        offs = {k: (v['x_s'], v['y_s']) for k, v in sensor_origin.items()
                if v.get('x_s') is not None}
        far = {k: v for k, v in offs.items()
               if ((v[0] - SENSOR_XY[0]) ** 2 + (v[1] - SENSOR_XY[1]) ** 2) ** .5
               > SENSOR_XY_TOL}
        out.append((f'G-2 [기술적] 도출 센서 원점 ⊂ {SENSOR_XY} ±{SENSOR_XY_TOL} m '
                    f'(⛔통과 보장 — 판정 근거 아님)',
                    len(offs) >= 11 and not far,
                    f'도출 {len(offs)}/{N} · 초과 {list(far)}'))

    # ★G-3 **짝 개입** — 모든 팔이 정확히 같은 관측점 P 를 받았다 (난수 소비 동수의 증거).
    ph = {}
    for r in ok:
        ph.setdefault(r['record'], set()).add(r['p_hash'])
    out.append(('G-3 팔 간 관측점 P 비트 동일 (짝 개입)',
                bool(ph) and all(len(s) == 1 for s in ph.values()),
                f'{sum(1 for s in ph.values() if len(s) == 1)}/{len(ph)} 레코드'))

    # ★G-4 개입이 실제로 발화했다 — 잡음 팔의 z 해시가 CONST 와 다르고 σ 가 상수가 아니다.
    zh = {}
    for r in ok:
        zh.setdefault(r['cell'], {})[r['record']] = r['z_hash']
    fired = all(zh.get(a, {}) != zh.get(REF_ARM, {}) for a in ARM_ORDER if a != REF_ARM)
    varying = all(r['sigma_stats']['max'] > r['sigma_stats']['min'] * (1 + 1e-9)
                  for r in ok if ARMS[r['cell']]['model'] != 'const')
    out.append(('G-4 개입 발화 (z 해시 상이 + σ 가 점마다 다르다)', fired and varying,
                f'z상이 {fired} · σ분산 {varying}'))

    # G-5 참 제로필드가 상수가 아니다 (⛔ZERO_REF=1.0 재발 방지)
    zs = {round(r['zero_ref'], 9) for r in ok}
    out.append(('G-5 참 제로필드가 상수 1.0 이 아니다', len(zs) > 1 and 1.0 not in zs,
                f'{len(zs)} 종'))

    # G-6 제안 표현이 실제로 돌았다 (성분 > 0, VARPRO 발화) — B-2 `G-1`/`G-1b` 계승
    out.append(('G-6 전건 Gaussian 성분 > 0 · VARPRO 발화',
                bool(ok) and all((r.get('n_atoms') or 0) > 0 and r.get('varpro_fired')
                                 for r in ok),
                f'atoms>0 {sum(1 for r in ok if (r.get("n_atoms") or 0) > 0)}/{len(ok)} · '
                f'vp {sum(1 for r in ok if r.get("varpro_fired"))}/{len(ok)}'))

    # ★G-7a **파생 속성** 일치 — 생성자가 config 와 따로 스냅샷한 값만 센다.
    #   ⛔초판의 `upd.config` 되읽기(항등식)를 대체한다. 사유는 DERIVED_READBACK 주석.
    bad7a = [(r['cell'], r['record'], k, (r.get('derived') or {}).get(k))
             for r in ok for k in DERIVED_READBACK
             if (r.get('derived') or {}).get(k) != ESTIMATOR_CFG[k]]
    out.append((f'G-7a 파생 속성 일치 ({", ".join(sorted(DERIVED_READBACK))}) '
                f'— ⛔config 되읽기 아님',
                not bad7a and bool(ok),
                str(bad7a[:3]) if bad7a else f'{len(ok)}/{len(ok)} 행 일치'))

    # ★G-7b **개입 계수기** — 실행이 남긴 수가 선언값과 같다 (설정 echo 가 아니다).
    bad7b = []
    for r in ok:
        c = r.get('counters') or {}
        for k, (ck, _src) in COUNTER_READBACK.items():
            if c.get(ck) != ESTIMATOR_CFG[k]:
                bad7b.append((r['cell'], r['record'], ck, c.get(ck), ESTIMATOR_CFG[k]))
    out.append((f'G-7b 개입 계수기 = 선언값 '
                f'({", ".join(v[0] for v in COUNTER_READBACK.values())})',
                not bad7b and bool(ok),
                str(bad7b[:3]) if bad7b else f'{len(ok)}/{len(ok)} 행 일치'))

    # ★G-7c **라우터 미발화** — `INERT_KEYS` 가 이 팔에서 읽히지 않았음의 양성 증거.
    #   `_auto_selected`/`_auto_real_gate` 는 `auto` 경로에서만 채워진다(그 외 None).
    #   ⚠속성이 아예 없으면 `ATTR_MISSING` 이 들어오고 그 또한 FAIL 이다.
    bad7c = [(r['cell'], r['record'], r.get('router'))
             for r in ok
             if (r.get('router') or {'_auto_selected': ATTR_MISSING}).get(
                 '_auto_selected') is not None
             or (r.get('router') or {'_auto_real_gate': ATTR_MISSING}).get(
                 '_auto_real_gate') is not None]
    out.append((f'G-7c 라우터 미발화 ⟹ {INERT_KEYS} 는 이 경로에서 무효(문서용)',
                not bad7c and bool(ok),
                str(bad7c[:2]) if bad7c else
                f'{len(ok)}/{len(ok)} 행 _auto_selected=_auto_real_gate=None'))

    # G-8 채점 창이 행에 실려 있고 팔마다 같다
    wk = {}
    for r in ok:
        wk.setdefault(r['record'], set()).add(json.dumps(r['window'], sort_keys=True))
    out.append(('G-8 채점 창 기록 + 팔 간 동일',
                bool(wk) and all(len(s) == 1 for s in wk.values()),
                f'{sum(1 for s in wk.values() if len(s) == 1)}/{len(wk)} 레코드'))

    # G-9 팔 정의가 비어 있지 않다 (⛔빈 dict 로 팔을 정의하지 않는다)
    out.append(('G-9 팔 정의 비어있지 않음', all(bool(v) for v in ARMS.values()),
                f'{len(ARMS)} 팔'))

    # ★G-10 [기술적] 잡음 곡선 상수가 논문 본문과 **런타임에** 일치한다.
    #   ⛔줄번호 대신 정규식 앵커를 쓴다. 논문이 편집돼도 어긋나지 않고,
    #   값이 바뀌면 여기서 시끄럽게 실패한다. σ_r·d_ref 는 코드 단일출처에서 읽었으므로
    #   이 게이트는 *"코드와 논문이 같은 값을 말하는가"* 도 함께 검사한다.
    ta = thesis_anchors()
    if ta is None:
        out.append(('G-10 [기술적] 논문 상수 앵커', False, f'{THESIS} 없음 — 대조 불가'))
    else:
        out.append(('G-10 [기술적] 논문 상수 앵커 (σ_0·σ_r·d_ref, 정규식)',
                    all(v['ok'] for v in ta.values()),
                    ' · '.join(f"{k}={v['found']}@:{v['line_now']}" for k, v in ta.items())))
    return out
