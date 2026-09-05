"""`T17-REALSCALE` 판정 단일출처 — **분석면**. ⛔`PREREG.md`(동결 `601ef1b`)가 정본이고
이 파일이 그것을 코드로 옮긴다.

★러너보다 **먼저** 쓰였다(2026-09-04 규율: 사전등록 → 게이트 → 분석기 → 러너 → 스모크 → 본실행).
⛔이 파일은 **판정만** 한다 — 규칙(`scale_rules_t16`)·장면(`scene_station`)·러너는 측정면이다.

★★**이 파일의 규율 넷** (전부 저장소가 대가를 치른 부류다)
1. ⛔**손타이핑 0.** 문턱·MDE·라벨 확률은 **계산한다**. `PREREG` 가 인쇄한 수는 `REGISTERED`
   에 **핀**으로만 들어가고 `registration_drift()` 가 계산값과 대조한다 ⟹ 어긋나면 `G-COV` 가 죽는다.
2. ⛔**배포·규칙 상수는 임포트한다.** `S_MIN/S_MAX/L_TRAIN/K_TRAIN` 은 `scale_rules_t14`,
   `N1/ZOOM/FLOOR` 은 `scale_rules_t16`, 표본·시드·창은 `gates_b2`, 배포 상수는
   `wavesplat.optimizer.sequential_update`·`wavesplat.run` 에서 온다.
3. ⛔**`(x or 기본값)` 금지** — `0.0` 은 falsy 다(`T14 AMENDMENT_1 §1`). 전부 `is None` 으로 가른다.
4. ⛔**라벨 문자열에 `…G-` 을 만들지 마라** — `PREREG §16 R-1/R-2` 가 하루에 두 번 당했다
   (`LANDIN`+`G-CHANGE…` · `SI`+`G-WORSE`). `_no_phantom_gate_tokens()` 가 임포트 시점에 막는다.

⚠**MDE 반올림 규약**: `PREREG §4` 가 인쇄한 16개 값은 전부 **4자리 올림**(ceil)이다
(실측 대조: `round()` 는 16개 중 8개가 어긋나고 `ceil` 은 16/16 일치).
⟹ `mde_q()` 는 **올림**한다. 사유는 규약이 아니라 방향이다 — ***MDE 를 과소보고하지 않는다.***
"""
import ast
import json
import math
import os
import re
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
B2 = os.path.abspath(os.path.join(HERE, '..', 'B2_REAL_2026-08-23'))
B3 = os.path.abspath(os.path.join(HERE, '..', 'B3_REFIT_2026-08-23'))
B14 = os.path.abspath(os.path.join(HERE, '..', 'B14_OBSMODEL_2026-08-24'))
T14 = os.path.abspath(os.path.join(HERE, '..', 'T14_AUTOSCALE_2026-09-04'))
T16 = os.path.abspath(os.path.join(HERE, '..', 'T16_BANDFIX_2026-09-05'))
BANKPORT = os.path.abspath(os.path.join(HERE, '..', 'bankmmse_port_2026-07-29'))
for _p in (HERE, ROOT, os.path.join(ROOT, 'scripts'), B2, T14, T16, BANKPORT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gates_b2 as GB2                                              # noqa: E402
import scale_rules_t14 as SR14                                      # noqa: E402
import zero_ref as ZR                                               # noqa: E402

try:                        # ★측정면 규칙. 게이트를 **러너보다 먼저** 쓰므로 늦게 깨질 수 있다.
    import scale_rules_t16 as SR16                                  # noqa: E402
except ImportError:         # ⛔없으면 `G-RULEID` 가 FAIL 한다. 조용히 통과시키지 않는다.
    SR16 = None

# ── 표본 · 관측 규약 — ★`gates_b2`/`scene_station` 에서 **임포트**한다 (PREREG §2) ─────
RECORDS = tuple(GB2.RECORDS)                 # 12 (⛔`LJ Grid2` 배제는 `B-2` 가 결과 전에 했다)
WINS = (0, 1, 2, 3, 4, 5)                    # `scene_station` 의 `linspace(lo, hi, 6)`
ANCHOR_WIN = GB2.WIN                         # = 3 (`B-2`·`B-3`·`B-14b` 의 창)
SEED = GB2.SEED                              # = 42 ★앵커가 성립하려면 같아야 한다
SEED17 = 20261400                            # `SEC-SEED`(Phase B) 전용 밑변 → +1..+72
LARGE_BOX = tuple(GB2.LARGE_BOX)             # ⛔층은 라벨을 만들지 않는다(기술통계)
SMALL_BOX = tuple(GB2.SMALL_BOX)
W_OBS = 40.0                                 # `scene_station` 기본 관측창 [s]
M_SCAN, M_CAP, NOISE = 12, 600, 0.02         # `scene_station` 기본값 (B-2 와 비트 동일)
CAP_MODE = 'uniform'
FIELD = GB2.FIELD                            # 'nowcast' — 종점은 `B-2` 계승

# ── 팔 — ⛔**되읽기로 적는다. `{}` 로 팔을 정의하지 않는다** (PREREG §1) ────────────────
#   ⛔`gates_b2.py:24` 의 `SHIP` 팔과 `gates_b7.py:46` 의 `SHIP-TRACK` 팔이 **빈 dict** 으로
#   정의돼 대가를 치렀다. ⚠그 자구를 여기 그대로 옮겨 적지 않는다 —
#   `check_batch_artifacts.py` 의 빈-팔 정규식이 **주석의 인용도 위반으로 센다**(실측 확인).
#   저장소가 이름 붙인 「N 곳 전부의 N 에는 주석이 포함된다」 부류다.
#   ★빈 dict 팔은 합성에서는 출하 기본값이지만 **실측에서는 `bank_mmse`** 다.
#   ⟹ 아래는 전부 키가 있고, `G-ARM` 이 `arm_provenance()` 되읽기로 정체를 증명한다.
ARM_CFG = {
    'BANK':       {'init_method': 'auto', 'real_estimator': 'bank_mmse'},
    'PROP':       {'real_domain_gate': False, 'init_method': 'learned'},
    'PROP-BANDR': {'real_domain_gate': False, 'init_method': 'learned'},
    'PROP-S025':  {'real_domain_gate': False, 'init_method': 'learned'},
    'PROP-AP':    {'real_domain_gate': False, 'init_method': 'learned'},
    'PROP-FIX2':  {'real_domain_gate': False, 'init_method': 'learned'},
}
ARM_ORDER = ('BANK', 'PROP', 'PROP-BANDR', 'PROP-S025', 'PROP-AP', 'PROP-FIX2')
PROP_ARMS = ('PROP', 'PROP-BANDR', 'PROP-S025', 'PROP-AP', 'PROP-FIX2')
RULE_ARMS = {'PROP-BANDR': 'AUTO-BANDR', 'PROP-AP': 'AUTO-AP'}
# ★규칙의 **프로세스 간 신원** — ⛔`id()` 는 프로세스마다 다르므로 러너↔분석기 대조에 쓸 수 없다
#   (스모크가 잡았다: `G-RULEID` 가 `rule_ids 불일치` 로 FAIL 했다). 신원은 `module.qualname`
#   으로 적고, **객체 동일성(`is`)** 은 분석기가 **자기 프로세스 안에서** 단언한다.
RULE_QUALNAMES = {'s_bandr': 'scale_rules_t16.s_bandr',
                  's_bandr_raw': 'scale_rules_t16.s_bandr_raw',
                  's_ap': 'scale_rules_t14.s_ap',
                  's_ap_raw': 'scale_rules_t14.s_ap_raw',
                  's_band_raw': 'scale_rules_t14.s_band_raw',
                  'kp_hat_rel': 'scale_rules_t16.kp_hat_rel',
                  'clip_s': 'scale_rules_t16.clip_s'}


def qualname_of(fn):
    return f'{fn.__module__}.{fn.__qualname__}'
# ⛔`PROP-S025` 의 상수는 손타이핑이 아니라 **규칙 모듈의 레일**이다(`SR14.S_MIN`).
S_FIXED = {'BANK': 1.0, 'PROP': 1.0, 'PROP-S025': SR14.S_MIN, 'PROP-FIX2': 2.0}
PRIMARY_ARM = 'PROP-BANDR'
# ⚠`ZERO` 는 **행이 아니라 열**이다 — `PREREG §1` 이 "런 0", `§6.1 G-N` 이 "6팔 × 72셀 = 432행",
#   `§6.4` 가 `zero_true` 를 **행 스키마의 열**로 등록했다. ⟹ 채점 기준은 모든 행이 들고 다닌다.
ZERO_REF_NAME = 'ZERO'

# ── 규칙 상수 — ★전부 **임포트** (PREREG §1 · §12-2) ──────────────────────────────────
S_MIN, S_MAX = SR14.S_MIN, SR14.S_MAX
L_TRAIN, K_TRAIN = SR14.L_TRAIN, SR14.K_TRAIN
N1 = 120 if SR16 is None else SR16.N1
ZOOM = 3.5 if SR16 is None else SR16.ZOOM
FLOOR = 1e-3 if SR16 is None else SR16.FLOOR
TWO_PI = 2.0 * math.pi

# ── 배포 상수 6종 — ★임포트 (PREREG §1) ⛔손복사 금지 ──────────────────────────────────
#   ⚠지금은 여섯 개 모두 `DEFAULT_CONFIG` 와 같다 ⟹ `G-DEPLOY` 는 **구조적으로 참(항진)**이다.
#   그래도 **명시**하는 이유는 `B-7` BLOCKER 다: `{}` 팔이 그날의 런타임 기본값을 스냅샷했다.
def deploy_consts():
    """배포 상수 6종의 **단일 출처** — 패키지에서 읽는다(⛔리터럴 6개를 적지 않는다)."""
    from wavesplat.optimizer import sequential_update as SU
    return {
        'varpro_steps': int(SU.VARPRO_STEPS_DEFAULT),
        'learned_init_iters': int(SU.LEARNED_INIT_ITERS_DEFAULT),
        'auto_switch_thresh': int(SU.AUTO_SWITCH_THRESH_DEFAULT),
        'lambda_l1': float(SU.LAMBDA_L1_DEFAULT),
        'bank_refit_every_s': SU.BANK_REFIT_EVERY_S_DEFAULT,
        'bank_refit_async': False,
    }


DEPLOY_KEYS = ('varpro_steps', 'learned_init_iters', 'auto_switch_thresh',
               'lambda_l1', 'bank_refit_every_s', 'bank_refit_async')

# ── 표본 크기 (PREREG §0.1 · §1 · §6.1 G-N) ───────────────────────────────────────────
N_REC = len(RECORDS)                         # 12
N_WIN = len(WINS)                            # 6
N_CELL = N_REC * N_WIN                       # 72  ★PRIMARY 단위
N_ARM = len(ARM_ORDER)                       # 6
N_ROWS = N_ARM * N_CELL                      # 432
N_RUNS = N_ROWS                              # 432 (ZERO 는 행이 아니다)
N_DATE = 10                                  # `PREREG §2` 의 일자 10개
N_SITE = 4

# ── 허용오차 (PREREG §3.1 · §5 · §12-2) ───────────────────────────────────────────────
TIE_REL = 1e-6               # 동점 = `|a−b|/max(|b|,1e-12) < TIE_REL`
EQ_TOL = 1e-9                # `G-S-READBACK` (⛔사전등록 §12-2 등록값 — 그대로 통과한다)
DEPTH_TOL = 1e-12            # `G-DEPTH-EXACT` 의 `depth_used == depth/s` (등록값, 통과)
FLOAT32_EPS = 2.0 ** -23     # 1.1921e-07 — `bank_mmse._beamform_top` 의 격자·누산 dtype
# ★★`KD_TOL` — `G-DEPTH-EXACT` 의 **kd 항만** 여기로 옮겼다 (`AMENDMENT_1_PREDATA.md §2`).
#   ⛔사전등록 §6.1 은 `1e-9·k̂d` 를 등록했는데 그 자는 **원리적으로 도달 불가**다:
#   `_beamform_top` 이 `torch.linspace(-klim, klim, 120)` 을 **float32** 로 만들고 float32 로
#   누산하므로, `klim' = s·klim` 격자의 첨두는 `s·k̂` 와 float32 eps 규모로만 일치한다.
#   ⚠스모크 실측(2셀 12행): `s` 가 이진 정확수(1.0·0.25·2.0)면 상대차 **정확히 0**,
#   임의 실수(0.320944·0.300155)면 **4.28e-08 · 7.59e-07** ⟹ 1e-9 은 구조적으로 못 넘는다.
#   ★값의 유도는 **관측이 아니라 dtype** 이다: 격자 2회 + hypot + argmax 지터 ≈ 10·eps ≈ 1.2e-6,
#   거기에 한 자릿수 여유 ⟹ `1e-5`. ⛔관측에 맞춰 조율한 값이 아니다(관측 최대의 13배).
#   ⚠이것은 **완화다.** 그러나 이 게이트가 잡아야 할 고장(수심 미분할 · 시간 지수 오류)은
#   `kd` 를 **O(1) 상대**로 움직인다(≥ √s 배) ⟹ 1e-5 는 그보다 네 자릿수 아래다.
KD_TOL = 1e-5
INVERT_TOL_REL = 1e-4        # `max|P_implicit − P_explicit| <= 1e-4·sg`
FOLD_MAJORITY = 7            # `C-FOLD` — 레코드 접기 다수 문턱 (⌈12/2⌉+1)
N_EFF_MIN_FOR_FOLD = 2       # 접기에서 창 부호가 하나뿐이면 중앙값이 의미 없다
DRIVE_T1_LN = math.log(1.05)  # `G-DRIVE` 의 자 (⛔이 배치의 축이 아니다 — 기록만)

# ── 다중성 (PREREG §3.1) ──────────────────────────────────────────────────────────────
ALPHA = 0.05
FAMILY_SIZE = 2              # PRIMARY 가족 = **2축**(`S`·`D`) ⟹ Bonferroni α/2
ALPHA_FAM = ALPHA / FAMILY_SIZE


# ── 이항 산수 ─────────────────────────────────────────────────────────────────────────
def p_ge(k, n, q=0.5):
    return sum(comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(k, n + 1))


def p_le(k, n, q=0.5):
    return sum(comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(0, k + 1))


def sign_p(w, n):
    """이김 다리의 단측 p (⛔`n=0` 에서 죽지 않는다)."""
    return p_ge(w, n, 0.5) if n else None


def sign_p_hurt(w, n):
    """해침 쪽 단측 p — 라벨이 아니라 `SIG_WORSE` 기록용."""
    return p_le(w, n, 0.5) if n else None


def k_one_sided(n, alpha=ALPHA):
    """단측 문턱 `min{k : P(W>=k|n,.5) <= alpha}` (없으면 `None`)."""
    c = [k for k in range(n + 1) if p_ge(k, n, 0.5) <= alpha]
    return min(c) if c else None


def k_two_sided(n, alpha=ALPHA_FAM):
    """★양측 문턱 `k⁺(n) = min{k : P(W>=k|n,.5) <= alpha/2}` — **동점 제외 후** `n_eff` 로 부른다.

    ⛔`None` 이면 그 `n` 에서는 **어떤 자료로도** 양측 기각이 불가능하다 ⟹ `D-UNPOWERED`.
    """
    c = [k for k in range(n + 1) if p_ge(k, n, 0.5) <= alpha / 2.0]
    return min(c) if c else None


def mde_q(k, n, power, nd=4):
    """문턱 `w>=k` 를 주어진 검정력으로 넘기는 **최소 q** — ★4자리 **올림**.

    ⛔`round()` 가 아니다: `PREREG §4` 의 16개 값이 전부 올림이고(대조 16/16),
    방향도 그것이 옳다 — ***MDE 를 과소보고하지 않는다.***
    """
    lo, hi = 0.5, 1.0
    for _ in range(300):
        mid = (lo + hi) / 2.0
        if mid == lo or mid == hi:
            break
        if p_ge(k, n, mid) < power:
            lo = mid
        else:
            hi = mid
    scale = 10.0 ** nd
    return math.ceil(((lo + hi) / 2.0) * scale) / scale


# ── 문턱 (PREREG §3.1 · §4) — ★전부 **계산한다** ──────────────────────────────────────
THRESH_F = k_one_sided(N_CELL, ALPHA)            # 44 · 양성대조(가족 밖 · 무보정 단측)
THRESH_S = k_one_sided(N_CELL, ALPHA_FAM)        # 45 · 축 S (Bonferroni 단측)
THRESH_D = k_two_sided(N_CELL)                   # 46 · 축 D (Bonferroni 양측)
SIG_WORSE = N_CELL - THRESH_S                    # 27 · ⛔라벨 아님 · 기록만
N_EFF_UNPOWERED = max(n for n in range(1, N_CELL + 1) if k_two_sided(n) is None)   # 6
THRESH_S_REC = k_one_sided(N_REC, ALPHA_FAM)     # 10
THRESH_D_REC = k_two_sided(N_REC)                # 11
THRESH_DATE = k_one_sided(N_DATE, ALPHA)         # 9
THRESH_SITE = k_one_sided(N_SITE, ALPHA)         # ⛔None — 바닥 p 0.0625


def _mde_table():
    """★MDE 를 **계산**한다 — ⛔`PREREG §4` 의 수를 손으로 옮기지 않는다."""
    out = {}
    for tag, k, n in (('S', THRESH_S, N_CELL), ('D', THRESH_D, N_CELL),
                      ('F', THRESH_F, N_CELL),
                      ('REC_S', THRESH_S_REC, N_REC), ('REC_D', THRESH_D_REC, N_REC),
                      ('DATE', THRESH_DATE, N_DATE)):
        for lab, pw in (('q50', 0.50), ('q80', 0.80), ('q90', 0.90)):
            out[f'{tag}_{lab}'] = mde_q(k, n, pw)
    # ⛔기각 불가 대역 (PREREG §3.1 · §4) — 「과반이지만 문턱 미달」이 아니라 **전 대역**이다
    out['unrejectable_S'] = (0, THRESH_S - 1)
    out['unrejectable_D'] = (N_CELL - THRESH_D + 1, THRESH_D - 1)
    out['unrejectable_S_rec'] = (0, THRESH_S_REC - 1)
    out['unrejectable_D_rec'] = (N_REC - THRESH_D_REC + 1, THRESH_D_REC - 1)
    return out


MDE = _mde_table()
Q80_S, Q80_D, Q80_F = MDE['S_q80'], MDE['D_q80'], MDE['F_q80']
Q80_REC_S, Q80_REC_D, Q80_DATE = MDE['REC_S_q80'], MDE['REC_D_q80'], MDE['DATE_q80']
# ⛔**섞지 마라**(`M8b` 선례): `S` 널이면 `S_q80`, `D` 널이면 `D_q80`, 접기는 `REC_*`.

# ── 등록 예측 (PREREG §3.5) ───────────────────────────────────────────────────────────
PRED_Q = {'S': 0.60, 'D': 0.65}
# `q_D=0.65` 근거: 제3자 코퍼스의 유사 짝(`T16 SEC-A`)이 58/60 이었으나 실측은 (i) 레일 4/12
#   (ii) 1단 여유 5/12 가 1 미만 (iii) 처치 방향이 반대 ⟹ **크게 내려 잡았다**.
# `q_S=0.60` 근거: 동결 `PROP`(항등)이 7/12=0.583. ⛔`BANK` 의 10/12 는 다른 팔이라 안 쓴다.

# ── ★결과 전 프로브 값 (PREREG §0.2 · §6.3 · §12-2) — **핀**이다 ─────────────────────
#   ⛔이 수들은 win=3 에서 **결과 전에** 측정됐다. 분석기가 같은 양을 재측정해 나란히 인쇄하고
#     어긋나면 `probe_drift` 로 드러낸다(⛔막지는 않는다 — `§6.3` 이 "막으면 배치가 죽는다").
RAIL_BANDR_W3 = 4
RAIL_AP_W3 = 8
EDGE_W3 = 1
MARGIN_W3 = (0.364, 226.136)
MARGIN_LT1_W3 = 5
DRIVE_MED_W3 = 0.009815
DRIVE_GT_T1_W3 = 1
DEEP_LT_PI_W3 = 3
ANCHOR_BANK_W3 = 10                # `BANK < ZERO_true` (동결 재집계)
ANCHOR_PROP_W3 = 7                 # `PROP < ZERO_true`
ANCHOR_PAIR_W3 = 2                 # `PROP < BANK`
PROBE_W3 = {'rail_bandr': RAIL_BANDR_W3, 'rail_ap': RAIL_AP_W3, 'edge': EDGE_W3,
            'margin_lt1': MARGIN_LT1_W3, 'drive_gt_t1': DRIVE_GT_T1_W3,
            'deep_lt_pi': DEEP_LT_PI_W3}

# ── ⛔`PREREG` 가 인쇄한 수는 **핀**으로만 들어간다 ───────────────────────────────────
#   `registration_drift()` 가 위의 **계산값**과 대조하고 분석기 `G-COV` 가 그것으로 죽는다.
#   ⟹ 문서와 코드가 갈리면 **배치가 라벨을 못 낸다**.
#   ⛔반대로 문서를 코드에 맞춰 고치지 마라 — 사전등록은 결과 0행에서 동결됐다(`601ef1b`).
REGISTERED = {
    'THRESH_F': 44, 'THRESH_S': 45, 'THRESH_D': 46, 'SIG_WORSE': 27,
    'N_EFF_UNPOWERED': 6, 'THRESH_S_REC': 10, 'THRESH_D_REC': 11, 'THRESH_DATE': 9,
    'THRESH_SITE': None,
    'N_CELL': 72, 'N_REC': 12, 'N_WIN': 6, 'N_ARM': 6, 'N_ROWS': 432, 'N_RUNS': 432,
    'S_q50': 0.6176, 'S_q80': 0.6647, 'S_q90': 0.6885,
    'D_q50': 0.6314, 'D_q80': 0.6781, 'D_q90': 0.7016,
    'F_q50': 0.6037, 'F_q80': 0.6513, 'F_q90': 0.6754,
    'REC_S_q80': 0.8693, 'REC_D_q80': 0.9308, 'DATE_q80': 0.9168,
    'unrejectable_S': (0, 44), 'unrejectable_D': (27, 45),
    'S_MIN': 0.25, 'S_MAX': 4.0, 'L_TRAIN': 220.55, 'K_TRAIN': 0.06779,
    'N1': 120, 'ZOOM': 3.5, 'FLOOR': 1e-3, 'TIE_REL': 1e-6,
    'FOLD_MAJORITY': 7, 'INVERT_TOL_REL': 1e-4, 'EQ_TOL': 1e-9,
    'SEED': 42, 'SEED17': 20261400, 'ANCHOR_WIN': 3,
    'P_S': 0.380148, 'P_D_up': 0.630054, 'P_D_down': 0.000001, 'P_D_null': 0.369946,
    'P_any_positive': 0.770688, 'P_both_null': 0.229312,
}


def registration_drift():
    """★사전등록이 인쇄한 수 ↔ 이 파일이 계산한 수. 어긋난 항목의 목록(빈 목록 = 일치)."""
    lp = label_probs()
    live = {'THRESH_F': THRESH_F, 'THRESH_S': THRESH_S, 'THRESH_D': THRESH_D,
            'SIG_WORSE': SIG_WORSE, 'N_EFF_UNPOWERED': N_EFF_UNPOWERED,
            'THRESH_S_REC': THRESH_S_REC, 'THRESH_D_REC': THRESH_D_REC,
            'THRESH_DATE': THRESH_DATE, 'THRESH_SITE': THRESH_SITE,
            'N_CELL': N_CELL, 'N_REC': N_REC, 'N_WIN': N_WIN, 'N_ARM': N_ARM,
            'N_ROWS': N_ROWS, 'N_RUNS': N_RUNS,
            'S_MIN': S_MIN, 'S_MAX': S_MAX, 'L_TRAIN': L_TRAIN, 'K_TRAIN': K_TRAIN,
            'N1': N1, 'ZOOM': ZOOM, 'FLOOR': FLOOR, 'TIE_REL': TIE_REL,
            'FOLD_MAJORITY': FOLD_MAJORITY, 'INVERT_TOL_REL': INVERT_TOL_REL,
            'EQ_TOL': EQ_TOL, 'SEED': SEED, 'SEED17': SEED17, 'ANCHOR_WIN': ANCHOR_WIN,
            'P_S': round(lp['P_S'], 6), 'P_D_up': round(lp['P_D_up'], 6),
            'P_D_down': round(lp['P_D_down'], 6), 'P_D_null': round(lp['P_D_null'], 6),
            'P_any_positive': round(lp['P_any_positive'], 6),
            'P_both_null': round(lp['P_both_null'], 6)}
    live.update({k: MDE[k] for k in ('S_q50', 'S_q80', 'S_q90', 'D_q50', 'D_q80', 'D_q90',
                                     'F_q50', 'F_q80', 'F_q90', 'REC_S_q80', 'REC_D_q80',
                                     'DATE_q80', 'unrejectable_S', 'unrejectable_D')})
    bad = []
    for k, want in REGISTERED.items():
        got = live.get(k)
        if isinstance(want, tuple):
            ok = tuple(got) == want if got is not None else False
        elif want is None:
            ok = got is None
        elif isinstance(want, float):
            ok = got is not None and abs(got - want) <= 1e-9 * max(1.0, abs(want))
        else:
            ok = got == want
        if not ok:
            bad.append(f'{k}: 등록 {want} vs 계산 {got}')
    return bad


# ── 표본 (⛔손목록 금지 — 계산한다) ───────────────────────────────────────────────────
def cells():
    """72셀 = 12 레코드 × 6창. 실행 순서 = 셀 순(record × win) — `PREREG §8`."""
    return [(r, w) for r in RECORDS for w in WINS]


def cell_id(record, win):
    return f'{record}|w{int(win)}'


def site_of(record):
    return record[:2]


def date_of(record):
    """레코드의 관측 일자 — ★`scene_station.RECORDS_12` 의 **파일 경로에서** 뽑는다."""
    from scene_station import RECORDS_12
    m = re.search(r'Surfaces_(\d{8})_', RECORDS_12[record]['path'])
    return m.group(1) if m else None


def dates():
    return sorted({date_of(r) for r in RECORDS})


def frozen_sample():
    """동결된 표본 파일(측정면). 없으면 `None` — `G-N` 이 그것으로 FAIL 한다."""
    p = os.path.join(HERE, 'sample_t17.json')
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding='utf-8'))


def seed_of_phase_b(record, win):
    """`SEC-SEED`(Phase B) 셀 시드 = `SEED17 + cell_index` ⟹ 20261401 … 20261472.

    ⛔Phase A 는 **`SEED`(=42)** 다 — 흔드는 축은 창이지 시드가 아니고,
    §6.2 앵커가 성립하려면 같아야 한다.
    """
    return SEED17 + cells().index((record, int(win))) + 1


# ── 규칙이 소비자에게 넘겨야 하는 값 (분석기가 **독립 재계산**한다) ────────────────────
def klim1_new(l_obs):
    """새 규칙의 1단 격자 반경 `0.5·(N1−1)·2π/L_obs` ⟹ `Δk₁ = 2π/L_obs`(Rayleigh 한계)."""
    return 0.5 * (N1 - 1) * (TWO_PI / l_obs)


def dk1_new(l_obs):
    return TWO_PI / l_obs


def s_of_arm(arm, s_rule_bandr, s_rule_ap):
    """팔의 `s` — ⛔러너가 리터럴을 적는 것이 아니라 이 함수가 단일출처로 낸다."""
    if arm in S_FIXED:
        return float(S_FIXED[arm])
    if arm == 'PROP-BANDR':
        return float(s_rule_bandr)
    if arm == 'PROP-AP':
        return float(s_rule_ap)
    raise KeyError(arm)


# ── 판정 사다리 (PREREG §3.3) — ★**2축** · 발화 순서 고정 ─────────────────────────────
S_LABELS = ('REAL-SKILL-WITH-AUTOSCALE', 'NO-REAL-SKILL-DETECTED')
D_LABELS = ('SCALE-CHANGES-REAL-LANDING', 'SCALE-WORSENS-REAL-LANDING',
            'SCALE-SHIFT-UNDETECTED-ON-REAL', 'D-UNPOWERED')
ANCHOR_LABEL = 'ANCHOR-DID-NOT-REPLICATE'
FOLD_SUFFIX = '[CELL-ONLY · EXPLORATORY]'


def label_S(w_s):
    """축 S (기술 축, 단측, Bonferroni α/2) = `W[PROP-BANDR < ZERO_true]`.

    ⛔`REAL-SKILL-WITH-AUTOSCALE` 는 `PROP`(항등 스케일) 값 **병기 없이 인쇄 금지** —
      그것 없이는 **규칙의 공로인지** 알 수 없다(§3.3).
    ⛔`NO-REAL-SKILL-DETECTED` 는 이름이 「미검출」이다. *"불가능"*·*"도달 불가"* 로 읽지 마라.
      MDE `q80` 병기가 의무다(§3.4).
    """
    if w_s >= THRESH_S:
        return 'REAL-SKILL-WITH-AUTOSCALE'
    return 'NO-REAL-SKILL-DETECTED'


def label_D(w_d, n_eff):
    """축 D (착지 축, 양측, Bonferroni α/2) = `W[PROP-BANDR < PROP]`.

    사전등록 발화 순서(⛔결과를 보고 순서·문턱을 바꾸지 않는다):
      0. `k⁺(n_eff)` 부재(= `n_eff <= 6`) → `D-UNPOWERED`
      1. `W >= k⁺`                        → `SCALE-CHANGES-REAL-LANDING`
      2. `n_eff − W >= k⁺`                → `SCALE-WORSENS-REAL-LANDING`
      3. 그 밖                            → `SCALE-SHIFT-UNDETECTED-ON-REAL`
    ★2번 다리를 `W <= n_eff − k⁺` 가 아니라 `n_eff − W >= k⁺` 로 적는다: **반대 방향의
      이김 수**를 세는 것이 이 검정의 정직한 서술이고, 그래야 「양성 이름이 기각-실패 다리에
      붙었는가」를 보는 정적 검사가 이 다리를 옳게 읽는다.
    ⛔`T16` 의 `D-TREATMENT-BELOW-SCALE` 갈래는 이 배치에 **없다** — 발화 조건이 결과 전에
      이미 거짓이다(`|ln s_bandr|` 중앙 1.2254 ≫ ln1.05, 초과 12/12).
    """
    kplus = k_two_sided(n_eff)
    if kplus is None:                       # ⛔어떤 자료로도 양측 기각 불가 (n_eff <= 6)
        return 'D-UNPOWERED'
    if w_d >= kplus:
        return 'SCALE-CHANGES-REAL-LANDING'
    if (n_eff - w_d) >= kplus:
        return 'SCALE-WORSENS-REAL-LANDING'
    return 'SCALE-SHIFT-UNDETECTED-ON-REAL'


def fold_sign(rel_by_record):
    """★`C-FOLD` 의 접기 — 레코드 안 6창의 짝별 상대차 `(a−b)/b` 의 **중앙값 부호**.

    반환 `(w_rec, n_rec, n_tied, per_record)`.
    `중앙값 < 0` = 처치가 이긴 레코드 · `== 0` 은 동점(제외).
    ⛔이것은 검정이 아니라 **강등 규칙의 입력**이다(§3.4 `C-FOLD`).
    """
    import statistics as st
    per, w, tied = {}, 0, 0
    for rec in sorted(rel_by_record):
        vals = [v for v in rel_by_record[rec] if v is not None]
        if len(vals) < N_EFF_MIN_FOR_FOLD:
            per[rec] = None
            continue
        med = float(st.median(vals))
        per[rec] = med
        if med == 0.0:
            tied += 1
        elif med < 0.0:
            w += 1
    n = sum(1 for v in per.values() if v is not None and v != 0.0)
    return w, n, tied, per


def fold_demote(cell_positive, w_rec, n_rec):
    """★`C-FOLD` (PREREG §3.4) — **강등만 하고 승격하지 않는다**.

    셀 단위 검정이 양성인데 레코드 접기의 다수(`>= FOLD_MAJORITY`)에 못 미치면
    확증 등급을 박탈하고 접미사를 붙인다.
    선례 = `T5-ENS AMENDMENT_1` 의 「단위 바꿔치기」(셀 36 이 레코드 12 로 접으니 깨졌다).
    """
    if not cell_positive:
        return False, '해당 없음 (셀 단위가 양성이 아니다)'
    if n_rec == 0:
        return True, '레코드 접기가 전건 동점이다 ⟹ 강등'
    if w_rec >= FOLD_MAJORITY:
        return False, f'레코드 접기 {w_rec}/{n_rec} >= {FOLD_MAJORITY} ⟹ 유지'
    return True, f'레코드 접기 {w_rec}/{n_rec} < {FOLD_MAJORITY} ⟹ 강등'


def label(w_f, w_s, w_d, n_eff, demote_S=False, demote_D=False):
    """★두 축을 합친 인쇄 라벨. ⛔양성대조 `F` 가 거짓이면 **다른 라벨을 인쇄하지 않는다**(§3.2).

    ⛔어떤 라벨도 *"우리 방법이 경쟁자보다 낫다"* 로 읽지 마라 — 외부 기준선 행이 **0개**다
      (`B2_WANGFIX`·`B-5` 가 실측 외부 기준선을 두 번 연속 죽였다). 라벨은 **귀속**이다.
    """
    if w_f < THRESH_F:
        return ANCHOR_LABEL
    ls, ld = label_S(w_s), label_D(w_d, n_eff)
    if demote_S and ls in ('REAL-SKILL-WITH-AUTOSCALE',):
        ls = f'{ls} {FOLD_SUFFIX}'
    if demote_D and ld in ('SCALE-CHANGES-REAL-LANDING', 'SCALE-WORSENS-REAL-LANDING'):
        ld = f'{ld} {FOLD_SUFFIX}'
    return f'{ls} ∧ {ld}'


def label_probs(q=None, n=None):
    """★등록 예측 하의 **모든 라벨 확률** — ⛔손타이핑 금지, **계산한다**(§3.5 · §12-3).

    ⚠**종속 공시**: `S`·`D`·`F` 는 같은 72셀·같은 관측을 공유하므로 독립이 아니다.
      아래는 곱셈으로 계산했고 ⛔**사후 검정력 주장에 쓰지 마라**.
    ⚠`F` 는 **예측이 아니라 복제 주장**이므로 결합에 곱하지 않는다(§3.2).
    """
    q = dict(PRED_Q) if q is None else dict(q)
    n = N_CELL if n is None else n
    k_s, k_d = k_one_sided(n, ALPHA_FAM), k_two_sided(n)
    p_s = p_ge(k_s, n, q['S'])
    if k_d is None:                       # ⛔n 이 너무 작아 D 축이 원리적으로 판정 불가
        p_up = p_dn = 0.0
    else:
        p_up, p_dn = p_ge(k_d, n, q['D']), p_le(n - k_d, n, q['D'])
    p_null = 1.0 - p_up - p_dn
    s_ax = {'REAL-SKILL-WITH-AUTOSCALE': p_s, 'NO-REAL-SKILL-DETECTED': 1 - p_s}
    d_ax = {'SCALE-CHANGES-REAL-LANDING': p_up,
            'SCALE-WORSENS-REAL-LANDING': p_dn,
            'SCALE-SHIFT-UNDETECTED-ON-REAL': p_null if k_d is not None else 0.0,
            'D-UNPOWERED': 0.0 if k_d is not None else 1.0}
    joint = {f'{a} ∧ {b}': pa * pb for a, pa in s_ax.items() for b, pb in d_ax.items()}
    both_null = (1 - p_s) * p_null
    return {'q': q, 'n': n, 'thresholds': {'S': k_s, 'D': k_d, 'F': k_one_sided(n, ALPHA)},
            'P_S': p_s, 'P_D_up': p_up, 'P_D_down': p_dn, 'P_D_null': p_null,
            'S_axis': s_ax, 'D_axis': d_ax,
            'joint': joint, 'joint_sum': sum(joint.values()),
            'P_any_positive': 1.0 - both_null, 'P_both_null': both_null,
            '_note': ('⚠축 독립 가정 · 같은 72셀을 공유하므로 독립이 아니다 · '
                      '⛔사후 검정력 주장에 쓰지 마라 · F 는 복제 주장이라 곱하지 않았다')}


def null_label_probs(n=None):
    """귀무 `q=0.5` 아래 오라벨률 (§3.5)."""
    return label_probs(q={'S': 0.5, 'D': 0.5}, n=n)


def sensitivity_table(n=None):
    """§3.5 의 민감도 표를 **재생산**한다."""
    out = {}
    for qq in (0.50, 0.55, 0.60, 0.65, 0.70, 0.80):
        lp = label_probs(q={'S': qq, 'D': qq}, n=n)
        out[f'{qq:.2f}'] = {'P_S': lp['P_S'], 'P_D_up': lp['P_D_up'],
                            'P_D_down': lp['P_D_down'], 'P_both_null': lp['P_both_null']}
    return out


def rejected_design_probs():
    """⛔채택하지 않은 설계 — 레코드 12 단독(§3.5). **왜 안 골랐는지가 기록이다.**"""
    return label_probs(n=N_REC)


def ladder_table(n=N_CELL):
    """★사다리를 **전 조합에서** 돌려 본다 — `UNDECIDED` 구멍이 없음을 기계로 보인다.

    ⛔한 축이 지는 조합도 반드시 **라벨**을 낸다. 반환 = `(설명, 라벨)` 목록.
    """
    kd = k_two_sided(n)
    rows = []
    for w_f in (THRESH_F - 1, THRESH_F):
        for w_s in (THRESH_S - 1, THRESH_S):
            for w_d, n_eff in ((kd, n), (n - kd, n), (n // 2, n),
                               (N_EFF_UNPOWERED, N_EFF_UNPOWERED)):
                for dS in (False, True):
                    for dD in (False, True):
                        rows.append(({'w_F': w_f, 'w_S': w_s, 'w_D': w_d, 'n_eff': n_eff,
                                      'demote_S': dS, 'demote_D': dD},
                                     label(w_f, w_s, w_d, n_eff, dS, dD)))
    # ★경계 하나 더: `n_eff = 7` 에서는 `k⁺` 가 **존재한다**
    rows.append(({'w_F': THRESH_F, 'w_S': THRESH_S, 'w_D': 7,
                  'n_eff': N_EFF_UNPOWERED + 1, 'demote_S': False, 'demote_D': False},
                 label(THRESH_F, THRESH_S, 7, N_EFF_UNPOWERED + 1)))
    return rows


# ── 차단 (PREREG §6) ──────────────────────────────────────────────────────────────────
# ★**등록된 게이트 이름 전부**가 여기 있다 — 기록 전용도 포함한다.
#   ⛔그것이 「기록 전용이 막는다」는 뜻은 **아니다**: 기록 전용 게이트의 조건은 분석기에서
#   리터럴 `True` 라 **구조적으로 FAIL 할 수 없다** ⟹ `blocked_by` 가 그 이름을 볼 일이 없다.
#   이름을 넣어 두는 이유는 반대다 — 나중에 누가 조건을 붙이면 **조용히 장식이 되지 않는다**.
GATES_BLOCKING = (
    # §6.1 차단 게이트 (사전등록 표 그대로)
    'G-ARM', 'G-NOEMPTY', 'G-ZEROREF', 'G-S-READBACK', 'G-S-BLIND', 'G-S-SRC',
    'G-RULEID', 'G-DEPTH-EXACT', 'G-RAILID', 'G-OBS', 'G-ATOMS', 'G-PURE',
    'G-INVERT', 'G-N', 'G-ENV', 'G-BLOCK',
    # ★구현 추가 (아래 `GATES_ADDED` 가 사유를 적는다)
    'G-SEED', 'G-COV',
)
GATES_RECORD = ('G-RAIL', 'G-EDGE', 'G-MARGIN', 'G-DRIVE', 'G-ADAPT', 'G-WIN',
                'G-DEEP', 'G-XBATCH', 'G-DEPLOY',
                # ★구현 추가 (기록·등급)
                'G-ANCHOR', 'G-SIBLING')
GATE_NAMES = GATES_BLOCKING + GATES_RECORD

# ★★**사전등록에 없는데 구현이 추가한 게이트** — ⛔조용히 늘리지 않는다. 사유를 적는다.
#   ⚠`G-COV` 가 「구현에만 있는 이름」을 FAIL 시키므로, 여기 **선언되지 않은** 추가는 배치를 죽인다.
GATES_ADDED = {
    'G-ANCHOR': '사전등록 §3.2 가 이름으로 부르지만(⟨G-ANCHOR·G-WIN 이 어디가 다른지⟩) '
                '§6 게이트 표에는 없다. §6.2 의 2등급 앵커(A-BANK·A-PROP)를 실행하는 게이트다. '
                '⛔막지 않는다 — T3 은 **등급 강등**이다.',
    'G-SIBLING': '사전등록 §11 R11 이 "`G-SIBLING` 기록" 이라 적었으나 게이트 표에 없다. 기록 전용.',
    'G-SEED': '사전등록에 없다. `§2 시드` 절이 "PRIMARY 추정기 시드 = 42" 를 못박았는데 그것을 '
              '**되읽는** 게이트가 표에 없었다 ⟹ `T15` 의 `set_seed` 후크 관용구로 추가했다. '
              '★차단이다 — 추가는 **엄하게만** 한다.',
    'G-COV': '사전등록에 없다. 게이트 표 ↔ 구현 1:1 · 등록 상수 드리프트 · 병기 의무(§3.4) '
             '자가검사. `T16`·`T11 A2 §7` 선례. ★차단이다.',
}


def blocked_by(rows):
    """⛔**긴 이름부터** 접두 비교.

    ★이 규약이 없으면 `G-S-BLIND` 가 `G-S-...` 로, `G-NOEMPTY` 가 `G-N` 으로 접힌다
    (`T5-ENS` 는 **이름 전체를 비교해** 차단 게이트 14개를 장식으로 만들 뻔했다).
    """
    order = sorted(set(GATE_NAMES), key=len, reverse=True)
    out = []
    for row in rows:
        if isinstance(row, dict):
            name = row.get('name')
            ok = row.get('pass_', row.get('ok'))
        else:
            name, ok = row[0], row[1]
        if ok:
            continue
        nm = str(name).replace('★', '').replace('⚠', '').replace('⛔', '').strip()
        for b in order:
            if nm.startswith(b):
                out.append(b)
                break
    return out


def selfcheck_blocked_by(names=None):
    """★자가진단 — 등록된 **모든** 이름이 합성 위반 튜플로 실제로 잡히는지.

    반환 `(못 잡은 이름, 다른 이름으로 접힌 것)`. ⛔둘 다 비어야 한다(`G-BLOCK`).
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


# ── ⛔유령 게이트 방지 (PREREG §16 R-1·R-2) ───────────────────────────────────────────
PHANTOM_RE = re.compile(r'[A-Za-z0-9]G-[A-Za-z0-9]+')


def _no_phantom_gate_tokens():
    """★라벨·접미사 문자열이 `check_prereg` 의 게이트 정규식에 걸리는지 **임포트 시점에** 본다.

    ⛔`LANDIN`+`G-CHANGE-DETECTED` · `SI`+`G-WORSE` · `SUBADDITIVE-CEILIN`+`G-C…` 가
      **오늘 하루에 세 번** 유령 게이트를 만들었다. 같은 정규식으로 잡는다.
    """
    bad = []
    for s in (list(S_LABELS) + list(D_LABELS) + [ANCHOR_LABEL, FOLD_SUFFIX]
              + [f'{a} ∧ {b}' for a in S_LABELS for b in D_LABELS]
              + [f'{a} {FOLD_SUFFIX}' for a in list(S_LABELS) + list(D_LABELS)]):
        m = PHANTOM_RE.search(s)
        if m:
            bad.append(f'{s} → 유령 {m.group(0)}')
    return bad


_PHANTOM = _no_phantom_gate_tokens()
assert not _PHANTOM, f'⛔라벨 문자열이 유령 게이트를 만든다: {_PHANTOM}'


# ── 소스 위생 (분석기 `G-ZEROREF` 가 쓴다) ────────────────────────────────────────────
BATCH_SOURCES = ('gates_t17.py', 'analyze_t17.py', 'run_t17.py', 'blind_worker_t17.py')


def zero_ref_literal_hits():
    """★`ZERO_REF = 1.0` 형태의 **상수 채점 기준**이 이 배치의 소스에 있는가 (AST).

    ⛔`gates_b2.ZERO_REF = 1.0` 이 여섯 문서의 계수를 동시에 틀리게 했다. 이 배치는
      그 이름을 **쓰지도 임포트하지도** 않는다. 참조(`X.ZERO_REF`)까지 잡는다.
    ⚠범위 = **이 배치의 소스만**(`B2_REAL` 의 옛 상수를 고치라는 뜻이 아니다).
    """
    hits = []
    for f in BATCH_SOURCES:
        p = os.path.join(HERE, f)
        if not os.path.exists(p):
            continue
        try:
            tree = ast.parse(open(p, encoding='utf-8').read())
        except SyntaxError as e:
            hits.append(f'{f}:파싱실패 {e}')
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute) and n.attr == 'ZERO_REF':
                hits.append(f'{f}:참조 .ZERO_REF')
            if isinstance(n, ast.Name) and n.id == 'ZERO_REF':
                hits.append(f'{f}:이름 ZERO_REF')
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    nm = getattr(t, 'id', None)
                    if nm and 'ZERO' in nm and isinstance(n.value, ast.Constant) \
                            and isinstance(n.value.value, (int, float)) \
                            and not isinstance(n.value.value, bool):
                        hits.append(f'{f}:{nm} = {n.value.value}')
    return sorted(set(hits))


def zero_true_cached():
    """`B2_REAL/true_zero_field.json` (win=3 캐시) — ⛔이 파일은 **읽기 전용**이다."""
    p = os.path.join(B2, 'true_zero_field.json')
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else None


def zero_true(record, win, use_cache=True):
    """★채점 기준의 **단일 출처** — 함수다(⛔상수가 아니다). `PREREG §2`."""
    return float(ZR.zero_ref_nowcast(record, win=int(win), noise=NOISE, use_cache=use_cache))


# ── 앵커 (PREREG §6.2) — 2등급 ─────────────────────────────────────────────────────────
ANCHOR_REL_TOL = 1e-6


def anchor_bank():
    """`B3_REFIT/results_b3.jsonl` `cell=FROZEN, win=3` 의 `lead0` (12값)."""
    p = os.path.join(B3, 'results_b3.jsonl')
    if not os.path.exists(p):
        return None
    out = {}
    for ln in open(p, encoding='utf-8'):
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r.get('cell') == 'FROZEN' and r.get('win') == ANCHOR_WIN and 'error' not in r:
            out[r['record']] = float(r['lead0'])
    return out or None


def anchor_prop():
    """`B14_OBSMODEL/results_b14b_*.jsonl` `axis=A, obs_model=IID, noise=0.02` 의 `nowcast`.

    ⛔`B-14b` 는 **런 전 동결이 없는 배치**다(자기 공시) ⟹ 앵커로만 쓰고 라벨을 인용하지 않는다.
    """
    import glob
    out = {}
    for p in sorted(glob.glob(os.path.join(B14, 'results_b14b_*.jsonl'))):
        for ln in open(p, encoding='utf-8'):
            if not ln.strip():
                continue
            r = json.loads(ln)
            if (r.get('axis') == 'A' and r.get('obs_model') == 'IID'
                    and r.get('noise') is not None and abs(r['noise'] - NOISE) < 1e-12
                    and 'error' not in r):
                out[r['record']] = float(r['nowcast'])
    return out or None


def anchor_grade(got, want):
    """T1 비트 동일 · T2 `rel < 1e-6` · T3 그 밖 (⛔차단이 아니라 **등급 강등**)."""
    if not want:
        return 'T3 (대조 자료 없음)', []
    common = sorted(set(got) & set(want))
    if not common:
        return 'T3 (교집합 0)', []
    bits = [k for k in common if got[k] == want[k]]
    rels = [(k, abs(got[k] / want[k] - 1.0) if want[k] != 0 else float('inf'))
            for k in common]
    worst = max((v for _, v in rels), default=float('inf'))
    if len(bits) == len(common):
        return f'T1 (비트 동일 {len(bits)}/{len(common)})', []
    if worst < ANCHOR_REL_TOL:
        return f'T2 (rel<{ANCHOR_REL_TOL} · 비트 {len(bits)}/{len(common)})', []
    off = sorted((k for k, v in rels if v >= ANCHOR_REL_TOL),
                 key=lambda k: -dict(rels)[k])
    return (f'T3 (패키지 드리프트 · 최대 rel {worst:.3e} · 비트 {len(bits)}/{len(common)})',
            off[:6])


# ── 형제 배치 · 시드 대역 (`G-XBATCH`·`G-SIBLING`, 기록 전용) ─────────────────────────
SEED_DECL_RE = re.compile(r'\b(SEED[A-Z0-9_]*)\s*=\s*(\d{7,})')


def sibling_seed_bases():
    """형제 배치의 **선언 밑변** — `tmp/*/{*.py,*.md}` 의 `SEED… = <7자리 이상>`.

    ⛔이 배치 자신은 뺀다. ⚠인용도 선언으로 센다(`check_seed_bands` 와 같은 규약).
    """
    import glob
    base = os.path.dirname(HERE)
    out = {}
    for d in sorted(glob.glob(os.path.join(base, '*'))):
        if not os.path.isdir(d) or os.path.abspath(d) == HERE:
            continue
        for p in sorted(glob.glob(os.path.join(d, '*.py'))
                        + glob.glob(os.path.join(d, '*.md'))):
            try:
                txt = open(p, encoding='utf-8', errors='replace').read()
            except OSError:
                continue
            for nm, val in SEED_DECL_RE.findall(txt):
                out.setdefault(os.path.basename(d), set()).add((nm, int(val)))
    return out


def sibling_outputs():
    """형제 결과 파일의 존재·mtime (`G-SIBLING` 기록 — CPU 경합·드리프트 단서)."""
    import glob
    base = os.path.dirname(HERE)
    out = {}
    for d in ('T14_AUTOSCALE_2026-09-04', 'T15_HOMEREG_2026-09-05',
              'T16_BANDFIX_2026-09-05', 'B2_REAL_2026-08-23', 'B3_REFIT_2026-08-23',
              'B14_OBSMODEL_2026-08-24'):
        for p in sorted(glob.glob(os.path.join(base, d, 'results*.jsonl'))):
            out[f'{d}/{os.path.basename(p)}'] = round(os.path.getmtime(p), 1)
    return out


if __name__ == '__main__':                      # 자기 점검 (⛔런 0회 · 순수 산수)
    print('══ gates_t17 자기 점검')
    print(f'  표본 {N_REC}레코드 × {N_WIN}창 = {N_CELL}셀 · {N_ARM}팔 = {N_ROWS}행')
    print(f'  문턱  F {THRESH_F} · S {THRESH_S} · D {THRESH_D} · SIG_WORSE {SIG_WORSE} '
          f'· n_eff 무검정력 <= {N_EFF_UNPOWERED}')
    print(f'  접기  레코드 S {THRESH_S_REC} · D {THRESH_D_REC} · 일자 {THRESH_DATE} '
          f'· 사이트 {THRESH_SITE}(⛔기각 불가)')
    print(f'  MDE   {MDE}')
    lp = label_probs()
    print(f"  P(S)={lp['P_S']:.6f} P(D위)={lp['P_D_up']:.6f} P(D아래)={lp['P_D_down']:.6f} "
          f"P(D널)={lp['P_D_null']:.6f}")
    for k, v in sorted(lp['joint'].items(), key=lambda kv: -kv[1]):
        if v > 0 or 'UNPOWERED' not in k:
            print(f'    {k:<72s} {v:.6f}')
    print(f"  ★양성 라벨 {lp['P_any_positive']:.6f} · 두 축 모두 널 {lp['P_both_null']:.6f}")
    print(f'  민감도 {sensitivity_table()}')
    print(f"  귀무 {null_label_probs()['P_any_positive']:.6f}")
    rd = rejected_design_probs()
    print(f"  ⛔채택 안 한 n=12 설계: 양성 {rd['P_any_positive']:.6f} "
          f"· 두 축 널 {rd['P_both_null']:.6f}")
    drift = registration_drift()
    print(f'  등록 드리프트 {drift if drift else "없음 (0)"}')
    miss, fold = selfcheck_blocked_by()
    print(f'  blocked_by 자가진단 미포착 {miss} · 접힘 {fold}')
    lt = ladder_table()
    print(f'  사다리 {len(lt)}조합 · 라벨 {len(set(l for _, l in lt))}종 · '
          f'None {sum(1 for _, l in lt if not l)}')
    print(f'  유령 게이트 {_PHANTOM if _PHANTOM else "없음"}')
    print(f'  ZERO_REF 리터럴 {zero_ref_literal_hits() or "없음"}')
