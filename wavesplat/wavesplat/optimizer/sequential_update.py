"""
sequential_update.py
Two-phase sequential wave field estimation.

Phase 1 (Initialization): Joint optimization over first 2*Tp frames
    - Coarse-to-fine: N=3 → 8 → 15
    - FFT-based initial values
    - Full window optimization

Phase 2 (Tracking): Per-frame warm-start update
    - Predict: φᵢ += ωᵢ·Δt
    - Update: 5-10 gradient steps
    - Target: 50ms/frame
"""

import os
import torch
import torch.nn as nn
import numpy as np
import time
from typing import List, Dict, Optional

from wavesplat.model.spectral_gaussian import SpectralGaussianModel
from wavesplat.model.dispersion import dispersion_omega
from wavesplat.optimizer.initializer import initialize_from_frames
from wavesplat.optimizer.coarse_to_fine import CoarseToFineSchedule
from wavesplat.data_loader import FrameData

# ── Shipping default for the L1 sparsity weight — SINGLE SOURCE ──────────────────
# 1e-3 -> 1e-4 on 2026-08-11 (S9, prereg verdict BOTH: sparse 50/64 -33.0% p=7e-06,
# crowded 27/32 -9.9%; all 9 shells improved, GT-free residual improved in both regimes).
#
# Why this constant lives HERE and not in run.py: it was previously written in THREE
# places -- configs/estimation.yaml, run.py DEFAULT_CONFIG, and this `.get()` fallback.
# The S9 deploy updated the first two and left this one at 1e-3, so an estimator built
# without an explicit key silently disagreed with the shipping default.  run.py now
# imports this name, so there is one definition and the yaml simply mirrors it.
# (The same class of defect -- a constant written twice and going stale -- has now cost
# this project three separate incidents; see tmp/S9_lr_x_l1_2026-08-11/PREREG.md §7.)
LAMBDA_L1_DEFAULT = 1e-4

# ── VARPRO 배포 (2026-08-22) ────────────────────────────────────────────────
# 그전까지 이 처방은 `tmp/W4_SOLVER_2026-08-19/varpro.py` 의 **몽키패치로만** 존재했고
# 패키지에는 없었다 = "측정은 됐는데 배포는 안 된" 상태. 아래 두 상수가 그 배포다.
# 근거: W4 1,344런(52/64, -17.53%, p=2.28e-07) · W5 640런(결합 적합은 잉여) ·
#       B 1,152런(배포 라우터 그대로 ITT, 혼잡 55/60 -25.28% p=5.19e-12).
# B 의 `A-VP` 팔 = {'learned_init_iters': 0} + VARPRO 400 스텝. 그것을 그대로 배포한다.
# ⚠ 이 값을 바꾸면 그 세 배치의 재현이 끊긴다. 바꿀 때는 기준선 팔이 명시하는지 확인할 것.
VARPRO_STEPS_DEFAULT = 400
# 0 = 학습 초기화 뒤의 결합 경사하강을 돌리지 않는다. W5 P-2 가 그것을 잉여로 측정했고
# (36/64 널) 벽시계를 29.7 s -> 14.5 s 로 줄였다. 2026-08-22 이전 값은 500 이었다.
LEARNED_INIT_ITERS_DEFAULT = 0

# ── bank-MMSE 재적합 주기 배포 (2026-08-23) — SINGLE SOURCE ──────────────────────
# `None`(적합 동결) -> **5.0 s**.  실측 전용 경로이므로 합성 결과는 원리적으로 불변이다
# (`_bank` 는 실측 게이트 안에서만 채워진다; 회귀 시험
#  `tests/test_config_exposure.py::test_bank_refit_cannot_affect_synthetic_runs` 가 고정).
#
# 근거(전부 2026-08-23, 전부 사전등록 · 고정 관측소 12 레코드):
#   정확도 · B-3   72런  refit@2s 가 동결을 36/36 창에서 이긴다(p=1.5e-11)
#           · B-3b 108런 주기 1·2·5·10 s **전부** 통과 ⟹ 조율된 상수가 아니다
#           · B-3c  72런 **한 번도 안 본 창 {0,2,4}** 에서 36/36 복제(-20.4%, 같은 크기)
#   기술   · 참 제로필드(레코드·lead 별) 대비 tau=5 는 28/36, p=6.0e-04
#            ⟹ 그날 가족단위 alpha=0.05/10=0.005 를 **통과하는 가장 긴 주기**
#   예산   · B-6  재적합 단가 중앙 1.714 s ⟹ tau=5 에서 주기의 0.34(최악 0.51)
#           · B-6b 렌더 137 us/점(R^2=0.9991) ⟹ 0.1 s 프레임에 약 720점
#           · B-6c 제어 경로 질의(8점, `d3_d17_ctrl_closedloop.build_query` 유래)로
#                  **12/12 레코드가 프레임 예산 통과, 100 Hz 예산도 통과**,
#                  14 s 창 점유율 중앙 0.385(최악 0.571)
#
# ⛔왜 2 s 가 아닌가: 정확도는 가장 좋지만(32/36, p=9.7e-07) **점유율 중앙 0.997·최악 1.474**
#   = 실시간이 아니다.  ⛔왜 10 s 가 아닌가: 예산은 좋지만 기술이 **24/36, p=0.033** 으로
#   가족단위 문턱(0.005)에서 죽는다.  ⟹ **5.0 이 둘을 동시에 만족하는 유일한 측정점.**
#
# ⚠범위: 합성 잡음 0.02 m · 4 사이트 · CPU 1코어(7800X3D)에서만 쟀다.  실제 센서 잡음에서는
#   미측정이고, `B-2b` 가 성능이 관측 잡음에 지배적으로 의존함을 보였다.
# 되돌림: `bank_refit_every_s=None` (또는 `pre_2026_08_23_config()`).
# ⛔⛔**2026-08-23 밤 되돌림: 5.0 -> None.**  아래 근거는 유효하지만 **배포를 지탱하지 못한다.**
# 배포 검증(RV-6)이 두 가지를 찾았다:
#   ① **결정 규칙이 사전등록 밖이었다** — 여섯 게이트 파일 어디에도 "참 제로필드 + 가족단위 α"
#      기준이 없다.  전부 "vs FROZEN" 이었고 거기서는 **네 주기가 다 통과**한다.
#      Holm 단계하강이면 tau=10 도 통과하고, 14 s 지평이 긴 주기를 구조적으로 벌한다
#      (tau=10 은 채점 lead 의 69%가 FROZEN 과 비트 동일).  ⟹ "tau=5 가 유일한 배포점" 철회.
#   ② ★**동기 정지를 안 쟀다** — 재적합은 `update()` 안에서 동기로 돈다.
#      `t_refit / dt_frame` = **중앙 20.8배**(최악 25.3) ⟹ tau=5 면 약 60프레임마다
#      **1.35–2.53 s 블로킹**.  그리고 `B-6c` 의 `CONTROL-FRAME-OK 12/12` 는 **배포 전
#      (기본값 None, 재적합 0회) 구성**에서 나온 판정이다 ⟹ **예산 근거가 배포 구성을 안 덮는다.**
# ⟹ 출하 기본값은 **자기 증거 위에** 서야 한다.  되돌린다.
#
# ✅**측정된 권고로는 남긴다**(정확도 이득은 실재한다):
#   `bank_refit_every_s=5.0` 은 실측 12 레코드에서 참 제로필드 대비 28/36(p=6.0e-04),
#   동결 대비 35/36(-16.6%).  **실시간 제어 루프에 넣으려면 비동기 재적합이 선행돼야 한다.**
#   상세: tmp/B6_BUDGET_2026-08-23/{RESULT_B6.md,RESULT_B6B.md,RESULT_B6C.md,RV6_deploy_audit.md}
#         tmp/B3_REFIT_2026-08-23/{RESULT_B3.md,RESULT_B3B.md,RESULT_B3C.md}
BANK_REFIT_EVERY_S_DEFAULT = None


def recommended_real_refit_config() -> dict:
    """★**측정된 권고**(배포 기본값이 **아니다**) — 재적합 5 s + **비동기**.

    2026-08-23 에 `bank_refit_every_s=5.0` 을 기본값으로 배포했다가 **같은 날 되돌렸다**:
    동기 재적합이 프레임 예산을 **9–13배** 깼고(`B-9` 음성대조 36/36, 최악 774–867 ms 대 dt
    67–100 ms), 내가 근거로 든 예산 판정은 **재적합이 0회인 구성**의 것이었다.
    그 뒤 **비동기 재적합**(`bank_refit_async`)을 구현하고 **후보 구성 그대로** 다시 쟀다:

    | | 측정 |
    |---|---|
    | 최악 프레임(런별 최악의 **중앙**) | ★**2.8 ms** — 동기 **774 ms** 대비 **274배** 감소 |
    | ⚠최악 프레임(**전 런 최대**) | **6.51 ms** — 동기 **1352.9 ms** 대비 208배 (여전히 dt 67-100 ms 안) |
    | 정확도 vs **동결** | ★**33/36, −10.39%, p=1.0e-07** (**실시간 페이싱**, `B-9b`) |
    | ⚠정확도 vs **무기술 기준** | 비동기 **23/36 (p=0.066, 명목 미달)** · 동기 **28/36 (p=0.0006)** |
    | 비동기의 대가 | 동기 대비 +5.05%, 그리고 ★**무기술 문턱을 넘느냐 마느냐** |

    ⛔**중요(2026-08-23 R2 토론 F-14)**: 위 `33/36` 의 비교자 `FROZEN` 은 **스스로 무기술**이다
    (참 제로필드 대비 16/36, p=0.80). ⟹ *"동결보다 낫다"* 가 *"기술이 있다"* 를 뜻하지 않는다.
    **비동기는 무기술 기준 대비 명목 문턱에 미달하고 동기는 통과한다** --- 그것이 지연 착지의
    정직한 대가다.

    ⛔**그럼에도 기본값으로 만들지 않는다.  사유는 성능이 아니라 *조용한 조건 의존성* 이다**:
    ★***비동기의 효과는 호출자의 벽시계 페이싱에 의존한다.***  재생 하네스가 실시간보다 빠르면
    (실측 중앙 **6배**) 적합이 채점 뒤에 착지해 **설정은 켜져 있는데 실질은 꺼진** 상태가 된다.
    기본값으로 만들면 그 함정이 **모든 배치에 조용히 깔린다** — 이 저장소가 *"개입이 개입하지
    않았다"* 로 여러 번 대가를 치른 바로 그 부류다.
    ⟹ **명시적으로 켜고, `bank_refit_health()['effective']` 를 게이트로 검사하라.**
        cfg.update(recommended_real_refit_config())
        ...
        assert upd.bank_refit_health()['effective'], '비동기 재적합이 실질적으로 안 돌았다'
    상세: `tmp/B9_ASYNC_2026-08-23/{RESULT_B9.md,RESULT_B9B.md}`
    """
    return {'bank_refit_every_s': 5.0, 'bank_refit_async': True}


def pre_2026_08_23_config() -> dict:
    """~~2026-08-23 배포 변경을 되돌린다~~ — ⚠**그 배포는 같은 날 밤 철회됐다.**

    이 함수는 **이제 항등**(기본값이 다시 `None`)이지만, 이 이름을 쓰는 기록과 러너가
    있으므로 **남겨 둔다**.  새 코드는 쓸 필요가 없다.

    원래 문언 (스위치 **하나**).

    | 키 | 이전 | 2026-08-23 |
    |---|---|---|
    | `bank_refit_every_s` | `None`(동결) | **5.0** |

    ⚠이 키는 **실측 경로에서만** 의미가 있다.  합성 배치를 재현할 때는 이 함수가 필요 없다
    (`_bank` 가 `None` 으로 남으므로 값이 무엇이든 도달 불가).
    2026-08-22 의 세 스위치까지 함께 되돌리려면 `pre_2026_08_22_config()` 를 같이 쓰라.
    """
    return {'bank_refit_every_s': None}

# 2026-08-22 이전의 값. ⛔`varpro_steps=0` 하나만으로는 배포 전 거동으로 돌아가지 않는다.
# 그 조합은 W4 의 `L-ITER0` 팔(VARPRO 없이 결합 적합도 없음)이고 배포 전보다 **나쁘다**
# (W4 재집계: `L-ITER0` vs `L-SHIP` 7/64, +23.34%). 배포 전 거동은 두 값을 **함께** 준다.
LEARNED_INIT_ITERS_PRE_VARPRO = 500

# ── 경로 통일 (2026-08-22) ──────────────────────────────────────────────────
# 라우터 임계. `_auto_select_init` 이 학습 제안의 `n_selected >= thresh` 일 때 학습 경로를
# 고른다. **0 = 항상 학습** = 합성 경로 통일. 기본값을 `init_method='learned'` 로 바꾸는
# 대신 이 값을 0 으로 두는 이유는 ★**실측 게이트가 `init_method=='auto'` 안에 있기 때문**이다
# (기본값을 바꾸면 게이트가 조용히 사라진다 — `initialize()` 의 ⚠2026-08-22 함정 주석 참조).
# 근거 (같은 `B` 코퍼스 128 에피소드, 강제 두 팔, median-of-3, 신선 시드):
#   · 전체   학습이 더 좋음 **90/128**, p=4.9e-06, 상대 중앙 **-38.5%**
#   · 혼잡   63/64, -50.98%  (해석적은 제로필드에 **25/64 패배**; 30성분에선 18/32)
#   · 희소   27/64 부호 널, +49.2%  (절대값 0.021 vs 0.010, 제로필드 0.25)
#   · ★제로필드 이김: 학습 **128/128** vs 해석적 99/128(에피소드별 실측 기준)
# ⛔★**위 수치의 범위 단서(2026-08-22 검수)**: 그 비교의 해석적 팔은 `init_method` 를
#   **명시**해 median-of-3 앙상블을 받지 못했다 = **약해진 비교자**. 배포 해석적(k=3)은
#   그보다 37/44 · -14.5% 낫고, 그 표본에서 헤드라인이 뒤집힌다(희소 8/33 · +110%).
#   ⟹ **합산 `90/128`·`-38.5%` 인용 금지**(사전등록에 없는 합산이고 어느 층도 근처가 아니다).
#   앙상블을 포함한 재측정 = `tmp/W15_DEPLOY_VERIFY_2026-08-22`.
# ⚠**부수효과**: 임계 0 이면 라우터가 해석적을 안 고르므로 `analytic_ensemble_k` 앙상블이
#   합성에서 **구조적으로 죽는다**. 그 상태는 배포된 적도 측정된 적도 없다.
#   · 라우터는 시드에 불안정했다 — 128 중 34(26.6%)에서 시드만 바꿔도 판정이 뒤집혔다
#   (tmp/W11_UNIFY_2026-08-22 · W13_SPARSE_HEAD2HEAD · W14_CROWDED_HEAD2HEAD)
# ⛔실측은 통일에서 제외된다 — tmp/W12_UNIFY_REAL_2026-08-22 가 학습 강제 시 예보 지평에서
#   제로필드에 **0/5** 패배를 측정했다. 실측 게이트가 그것을 막는다.
# ⚠되돌리기: 8 (2026-08-22 이전 값). 과거 배치 재현 시 명시할 것.
AUTO_SWITCH_THRESH_DEFAULT = 0

# ── 표현 절제 노브 (2026-08-24, `B-16`) ──────────────────────────────────────
# ★**기본값 None = 완전한 no-op(비트 동일).**  배포 거동을 바꾸지 않는다.
# 왜 있는가: 저장소의 헤드라인은 *"격자 비의존 표현"* 을 기여로 말하는데,
# **표현만 절제한 팔이 저장소에 0건**이었다(2026-08-24 방법론 심사 A2).
# ⟹ 이 노브는 배포 파이프라인을 그대로 두고 **초기화 끝(VARPRO 직후)에 파수만**
#    간격 `k_snap_dk` 격자에 스냅하고 (w, φ) 를 LS 로 다시 맞춘다.
#    추적·가지치기·성장은 그대로다 ⟹ 바뀌는 것은 **파수의 연속성 하나뿐**이다.
# ⛔이것은 "격자 기반 방법"의 재현이 아니다 — 격자 기준선은 상태·동화 방식이 다르다.
#   이 노브가 재는 것은 **우리 팔 안에서 파수 연속성이 얼마를 벌어 주는가**이다.
K_SNAP_DK_DEFAULT = None
AUTO_SWITCH_THRESH_PRE_UNIFY = 8


def recommended_learned_ensemble_config(k: int = 3) -> dict:
    """★**측정된 권고**(배포 기본값이 **아니다**) — 학습 경로 median-of-k.

    ⛔**왜 기본값이 아닌가**(2026-09-01, 결정 `D3`, 다중 에이전트 토론):

    | | |
    |---|---|
    | 확증 | `T3-NOW` 나우캐스트 **29/36 · −8.65%** · `T5-ENS` 예보 **NEAR 28/36 −8.53% / FAR 35/36 −11.10%** (셋 다 Holm 생존) |
    | ⛔그런데 | **셋 다 패키지가 소유하지 않은 연산자**로 쟀다 — 하네스가 median 을 만들었고 행의 `ens_k_readback` 은 전부 `None` 이다 |
    | ⛔그리고 | 셋 다 **`real_domain_gate=False`** 로 쟀는데, 배포는 실측에서 게이트를 켜고 `bank_mmse` 로 보낸다 |
    | ⛔합성 근거 | **0건.** 기본값이 실제로 바꾸는 트래픽은 **합성**인데 개입 확증은 실측뿐이다 |
    | ⚠NEAR 비대칭 | 28/36 을 돕지만 **8/36 을 해친다**(최악 **+40.56%**, `T5` BS11a w5). FAR 은 1/36(+0.99%) |

    ⟹ 08-23 교훈(***"측정한 구성과 배포한 구성이 다르면 그 측정은 배포를 지탱하지 못한다"***)에
    따라 **경로만 구현하고 기본값은 1 로 둔다.** 이 함수가 「명시 경로」다
    (선례 = :func:`recommended_real_refit_config`).

    ★**이 경로를 구현한 이유**: 그전까지 확증된 팔이 **패키지 안에 존재하지 않았다.**
    이제 존재하므로 다음 배치는 **배포 코드 경로 자체를 잰다**(하네스 연산자가 아니라).
    기본값 전환은 **합성 사전등록 배치 뒤의 별도 결정**이다.

    사용::

        cfg.update(recommended_learned_ensemble_config())
        upd = make_updater(cfg)
        assert upd._effective_k('learned', real_gate=False) == 3

    ⛔★**초판(2026-09-01)이 `init_method='learned'` 를 돌려줬다 — 그것이 결함이었다.**
    실측 도메인 게이트는 `init_method == 'auto'` **분기 안에만** 있으므로(`initialize()`),
    명시 `learned` 는 게이트를 **아예 통과하지 않는다**. ⟹ 초판 권고를 실측에 쓰면
    **게이트 없이 학습 멤버 3개**가 돌았다 --- `W-12` 가 측정으로 금지한 그 팔이다
    (실측 학습 강제 = 예보 지평 0/5 패배). 이 파일의 `:742` 주석이 이미 축자로
    *"`init_method` 를 'learned' 로 바꾸면 실측 게이트가 조용히 사라진다"* 고 경고해 뒀는데도
    같은 함정에 빠졌다. 독립 감사가 잡았다.
    ⟹ **권고는 `auto` 를 유지한다.** 라우터가 합성에서 학습으로 해소하고, 실측에서는
    게이트가 여전히 bank-MMSE 로 보낸다(그쪽은 자체 장 앙상블을 돈다).
    """
    return {'init_method': 'auto', 'learned_ensemble_k': int(k)}


def pre_varpro_config() -> dict:
    """VARPRO 배포(2026-08-22 오전) **직전** 거동. ⚠**부분 복원**이다.

    ⛔이 함수만으로는 2026-08-22 **이전** 거동이 되지 않는다 — 같은 날 오후에
    `auto_switch_thresh` 도 8 → 0 으로 바뀌었다(경로 통일). 전체 복원은
    `pre_2026_08_22_config()` 를 쓰라.
    (초판 독스트링이 *"스위치는 둘"* 이라 단언했는데 **셋**이다 — 2026-08-22 검수 V3 가 잡았다.)
    """
    return {'varpro_steps': 0,
            'learned_init_iters': LEARNED_INIT_ITERS_PRE_VARPRO}


def pre_2026_08_22_config() -> dict:
    """2026-08-22 **하루 전체**의 배포 변경을 되돌리는 설정 조각 — 스위치 **셋**.

    ★한둘만 끄면 배포된 적도 측정된 적도 없는 제3·제4의 팔이 된다.
    과거 배치를 재현할 때는 반드시 이 함수를 쓰라: `cfg.update(pre_2026_08_22_config())`.

    | 키 | 이전 | 2026-08-22 |
    |---|---|---|
    | `varpro_steps` | (없음=0) | 400 |
    | `learned_init_iters` | 500 | 0 |
    | `auto_switch_thresh` | 8 | 0 |

    ⚠**주의**: `dict(DEFAULT_CONFIG)` 를 런타임에 스냅샷하는 러너는 이 셋을 **자동으로 흡수**한다.
    사전등록에 *"explicitly, no fallback"* 이라 적혀 있어도 그 부류는 막지 못한다
    (S13·S13b·HOSGATE 가 그렇게 쓰였다 — 2026-08-22 검수 V3).
    """
    return {'varpro_steps': 0,
            'learned_init_iters': LEARNED_INIT_ITERS_PRE_VARPRO,
            'auto_switch_thresh': AUTO_SWITCH_THRESH_PRE_UNIFY}

# The PRE-2026-08-11 shipping value of `lambda_l1`, kept ONLY so the S8 probe gates below
# (`S8_L1_TRACK_ONLY` / `S8_L1_PHASE1_ONLY`, 2026-08-10) reproduce bit-exactly.  It was
# written as a bare `0.001` literal at two call sites whose comments claimed it was "the
# SHIPPED default"; that claim silently became FALSE on 2026-08-11 when S9 deployed 1e-4,
# so the literal changed meaning from "current default" to "old default" without anyone
# editing it.  ⛔DO NOT retune this to LAMBDA_L1_DEFAULT -- the recorded S8 batch is only
# reproducible at 1e-3.  Naming it is the whole fix: the value is now stated once and its
# meaning is stated with it.
LAMBDA_L1_PRE_S9 = 1e-3

# ── Shipping default for the initialisation route — SINGLE SOURCE ────────────────
# 'auto' since 2026-07-26 (user decision): the GT-free regime router.  run.py imports
# this name into DEFAULT_CONFIG and its yaml-loader fallback reads DEFAULT_CONFIG, so
# there is exactly one definition and configs/estimation.yaml is a mirror -- the same
# layout LAMBDA_L1_DEFAULT uses above, for the same reason.
INIT_METHOD_DEFAULT = 'auto'

# ── Class-level fallback of SequentialUpdater.__init__ — DELIBERATELY NOT THE ABOVE ──
# ⚠These two constants are NOT duplicates of one another and must not be merged.
#   * INIT_METHOD_DEFAULT ('auto') is a POLICY that SequentialUpdater cannot execute on
#     its own: the router's median-of-k analytic ensemble lives in `make_updater` /
#     `MedianEnsembleUpdater`.  A bare `SequentialUpdater({})` carrying 'auto' would be a
#     THIRD behaviour -- routed but un-ensembled -- that has never shipped and has never
#     been measured.
#   * INIT_METHOD_CLASS_FALLBACK ('v2_konly') is what a directly-constructed updater has
#     always done, and it is what every research runner that omits the key has recorded.
# MEASURED 2026-08-21: this fallback is REACHED, i.e. it is NOT a dead default.  Call
# sites that construct the class with a config lacking 'init_method' and then run it
# include wavesplat/tests/test_phase_convention.py:125 and eight tmp/ research runners
# (tmp/G1_waveprior_power/*.py, tmp/tracker_ampdecay_2026-07-25/*.py,
# tmp/COMPARE_REAL_SYNTH_2026-08-19/real_estimate.py via make_updater).  Changing it
# would silently re-route those recorded batches.
INIT_METHOD_CLASS_FALLBACK = 'v2_konly'


# `_adaptive_prune_grow` historically compares an ABSOLUTE residual RMS in metres against
# `grow_residual_threshold` (0.05 m) with no normalisation by sea state. This is the same
# defect class as coarse_to_fine.SCALE_INVARIANT_ADVANCE, but it lives in a different
# function, so that flag never reached it. Measured over 8 episodes spanning Hs 0.3-5.0 m
# (tmp/SCALE_AUDIT_fable_2026-07-23.md): residual_rms rises monotonically with Hs
# (0.069 -> 0.107 -> 0.175 m), so on crowded high-Hs episodes the threshold is *guaranteed*
# to fire and the 30% rollback guard fires along with it -- an outcome previously read as
# incidental rather than structural.
# When True, the residual RMS is divided by the observation RMS of the same frames, so the
# threshold means "fraction of observed signal amplitude still unexplained". That normaliser
# needs no ground truth and is therefore deployable. False preserves shipped behaviour.
#
# MEASURED ON REAL DATA, 2026-08-23 (tmp/B2_REAL_2026-08-23/RESULT_B2F.md, 12 fixed-station
# stereo records, sigma 0.09-2.51 m).  The synthetic reading above does NOT carry over:
#   * With the flag False the growth branch fired 0/12 -- the absolute 0.05 m threshold is
#     never crossed on these records, so shipped `adaptive_n` is PRUNE-ONLY on real input.
#   * With it True the components that grew were exactly the 7 records pruning had cut below
#     max_n, and every one returned to exactly 30; the 5 records pruning left alone were
#     bit-identical.  So on real data this flag's whole effect is "undo the pruning", not
#     "a better-scaled threshold".  Skill was unchanged (7/12 -> 8/12 beating zero-field,
#     paired 3/12 better, relative median +0.00%, p=0.98).
# Do not cite the synthetic "guaranteed to fire" sentence as a statement about real input.
SCALE_INVARIANT_GROW = False

# The L1 sparsity term adds `lambda_l1 * sum|w|`, a quantity in metres, to a mean-squared
# loss in metres squared. lambda_l1 therefore carries a hidden unit, and its effective
# strength is sea-state dependent: measured 3-62% of the total loss across Hs 0.3-5.0 m,
# non-monotonically, with a 58% swing in total recovered amplitude in a BF3-2comp ablation
# (tmp/SCALE_AUDIT_fable_2026-07-23.md; the nRMSE impact was not isolated there).
# When True the penalty is multiplied by the observation RMS so both terms are [m^2].
# False preserves shipped behaviour.
SCALE_INVARIANT_L1 = False

# The tracking warm-start historically advances every component's phase by omega*dt before the
# per-frame gradient update (`self.model.phi.add_(omega * dt)` in `update`). This DOUBLE-counts
# time: the forward model already evolves the wave through its own -omega*t term
# (spectral_gaussian.forward), so evaluating forward(t) with the init-time phase already gives
# the correctly-propagated field. Adding omega*dt on top cancels the -omega*t evolution and
# FREEZES the field at the initialization instant; the 10 Adam steps/frame then have to re-slew
# the phase back every frame, and under Adam momentum + L1 that per-frame tug-of-war bleeds
# amplitude toward zero. Measured (tmp/TRACKER_AMPDECAY_2026-07-25.md): starting from a
# parameter-exact init (field nRMSE 0.000), shipped tracking decorrelates to nRMSE ~0.31 with no
# gradient (pure frozen-wave propagation) and to ~0.16 with gradient while sum|w| decays -32%;
# setting lambda_l1=0 does NOT stop the decay (0.1631 vs 0.1633), so L1 is not the cause.
# When False, the redundant advance is skipped and forward(t) tracks the true time evolution
# directly, leaving only the genuine slow drift from any wavenumber error for the gradient to
# correct. True restores the pre-2026-07-26 legacy behaviour bit-for-bit.
# DEFAULT FLIPPED to False on 2026-07-26 (user decision) after the 100-episode gate:
# 0.1990 -> 0.1145 mean nRMSE, improves 100/100 episodes, Wilcoxon p=3.9e-18
# (tmp/rerun_2026-07-25/PHASEFIX_100EP_2026-07-25.md). Every headline number published
# before that date was measured with the legacy True behaviour unless stated otherwise.
TRACK_PHASE_PREDICT = False

# Deployed size of the analytic median-of-k field ensemble (see MedianEnsembleUpdater).
# k=3 is the pre-registered, confirmed value; it applies ONLY to the analytic fallback of
# the `auto` router. k=1 disables the ensemble.
DEFAULT_ANALYTIC_ENSEMBLE_K = 3

# Deployed member count of the real-domain bank-MMSE field ensemble (see
# SequentialUpdater._initialize_bank): pointwise-median FIELD across fits whose only
# difference is the val-split seed {s, s+1000, s+2000}. Measured 2026-07-29 on the three
# frozen GIF-ledger windows: never worse than the median single seed, and it repairs the
# one bad-seed lead blow-up observed (AA15a w5 lead-mean 0.824 -> 0.621,
# tmp/bankmmse_port_2026-07-29/m3_field.json). 1 disables the ensemble.
DEFAULT_REAL_BANK_ENSEMBLE_K = 3

# ★Member count of the LEARNED-path field ensemble. **Default 1 = OFF.**
#
# Why the path exists but is off by default (2026-09-01, decision D3):
#   The gain is confirmed twice on real data -- `T3-NOW` nowcast 29/36, -8.65%, and
#   `T5-ENS` forecast NEAR 28/36 -8.53% / FAR 35/36 -11.10% (both Holm-surviving).
#   But *both* were measured with an operator the package did not own: the harness
#   built the median itself. Shipping a default on the strength of a measurement whose
#   configuration the package cannot reproduce is exactly the 2026-08-23 mistake
#   (`bank_refit_every_s` deployed and withdrawn the same day).
#   So the path is implemented here -- which makes the measured arm reproducible -- and
#   the default stays 1 until a preregistered SYNTHETIC batch measures it on this code
#   (the deployed default routes synthetic traffic; the confirmations are real-data only).
#
# ⚠Known asymmetry at NEAR: the pointwise median helps 28/36 cells but HURTS 8/36,
#   worst +40.56% (`T5` BS11a w5). At FAR only 1/36 is hurt (worst +0.99%).
#   ⛔The wrapper docstring's "harming no slice (max deviation +0.0025)" is the
#   ANALYTIC arm's 2026-07-27 measurement and does not transfer to the learned arm.
#
# Use `recommended_learned_ensemble_config()` to turn it on explicitly.
DEFAULT_LEARNED_ENSEMBLE_K = 1


def _rx1_gate(name: str) -> bool:
    """Rx-1 (2026-08-05) RESEARCH gate. Unset or '0' == shipped behaviour, bit-for-bit.

    The environment is read at CALL time, not import time, because the Rx-1 driver reuses
    multiprocessing pool workers across cells; an import-time read would freeze whichever
    cell the worker happened to run first.

    RX1_CONF_NORM=1
        Divide the confidence-weighted data term by sum(weights) instead of n_obs in
        `_compute_loss`. The shipped form divides a WEIGHTED sum by an UNWEIGHTED count,
        so the data term is scaled by mean(confidence) (~0.86 measured) while the L1 and
        smoothness penalties are not -- i.e. the penalties are relatively amplified.
    RX1_PHASE2_NOREG=1
        Drop the L1 + smoothness penalties from the TRACKING objective only; Phase 1
        (initialisation) keeps them. This isolates "the tracking loop calls the same
        penalised objective as Phase 1" from "the penalties exist at all".

    Neither gate is a deployment knob: both are off by default and are not read by any
    shipped config path.
    """
    return os.environ.get(name, '0') == '1'


def _frames_have_gt(frames) -> bool:
    """True if ANY frame carries a non-degenerate ground-truth field.

    Real-data front-ends (wavesplat/real_data.py, the scene_real harnesses) build
    FrameData with gt_field=None; EpisodeDataLoader always attaches a non-zero field.
    Mirrors the `has_gt` test in run.py, widened from the last init frame to the window.
    """
    for f in frames:
        g = getattr(f, 'gt_field', None)
        if g is None:
            continue
        try:
            if float(torch.abs(g).max()) > 1e-10:
                return True
        except (RuntimeError, ValueError, TypeError):
            continue
    return False


class SequentialUpdater:
    """
    Two-phase sequential wave field estimator.

    Usage:
        updater = SequentialUpdater(config)
        updater.initialize(init_frames)
        for frame in tracking_frames:
            result = updater.update(frame)
    """

    def __init__(self, config: dict):
        """
        Args:
            config: dict with keys:
                n_gaussians: list [3, 8, 15] or int
                depth: water depth [m]
                lr: learning rate (default 0.01)
                lambda_l1: L1 sparsity weight (default LAMBDA_L1_DEFAULT = 1e-4,
                    deployed 2026-08-11 by S9; was 1e-3 before that)
                lambda_smooth: smoothness weight (default 0.0001)
                init_max_iters: max iterations for initialization (default 500)
                tracking_steps: gradient steps per frame (default 10)
                device: 'cpu' or 'cuda'
                loss_thresholds: list of loss thresholds per stage (default [0.1, 0.05, None])
                max_iters_per_stage: list of max iters per stage (default [200, 200, 500])
                tracking_mode: 'phi_only', 'w_phi', or 'all' (default 'w_phi')
                tracking_lr: learning rate for tracking phase (default 0.005)
                max_steepness: wave steepness clamp (default 0.15)
        """
        self.config = config
        self.depth = config.get('depth', 50.0)
        self.lr = config.get('lr', 0.01)
        self.lambda_l1 = config.get('lambda_l1', LAMBDA_L1_DEFAULT)
        self.lambda_smooth = config.get('lambda_smooth', 0.0001)
        self.tracking_steps = config.get('tracking_steps', 10)
        self.device = config.get('device', 'cpu')
        self.loss_thresholds = config.get('loss_thresholds', None)
        self.max_iters_per_stage = config.get('max_iters_per_stage', None)
        self.tracking_mode = config.get('tracking_mode', 'w_phi')
        self.tracking_lr = config.get('tracking_lr', 0.005)
        # NOTE the fallback is INIT_METHOD_CLASS_FALLBACK ('v2_konly'), NOT the shipping
        # default INIT_METHOD_DEFAULT ('auto') -- see the constants at the top of this
        # module for why they differ and why merging them would be wrong.
        self.init_method = config.get('init_method', INIT_METHOD_CLASS_FALLBACK)
        # WavePrior UNIFIED (init_method='auto'): diagnostics of the GT-free regime
        # router, populated only when init_method=='auto' (see _auto_select_init).
        # Default (v2_konly) leaves these None -> no behavioural effect.
        self._auto_selected = None
        self._auto_n_selected = None
        # WavePrior UNIFIED-GUARD (init_method='auto'): GT-free disagreement guard that
        # vetoes a learned route on a single-clean-swell over-proposal (see
        # _auto_select_init). RETIRED 2026-07-27 (default auto_guard=False), kept for
        # reproducibility. Populated only on the 'auto' path; None otherwise.
        self._auto_guard_fired = None
        self._auto_bf_peak_ratio = None
        # REAL-DOMAIN GATE (init_method='auto'): True when the router was bypassed because
        # the input carries no ground-truth field, i.e. real measured data (see initialize).
        self._auto_real_gate = None
        # REAL-DOMAIN ESTIMATOR (2026-07-29): what runs behind the gate. 'bank_mmse'
        # (deployed default) = the ported research estimator, ~2x better on the frozen
        # real windows (see _initialize_bank); 'v2_konly' = the pre-port 2026-07-27
        # behaviour, kept as an escape hatch for reproducing the older real-data runs.
        self.real_estimator = str(config.get('real_estimator', 'bank_mmse'))
        # RESEARCH-ONLY bank knobs (2026-08-28, tmp/M8_LADDER_2026-08-28/PATCH.md).
        # ⛔"설정을 넣었다 != 설정이 먹었다" -- these two are read ONLY by
        # _bank_fit_window, so any other real_estimator would swallow them silently.
        if (config.get('bank_dk_frac') is not None
                or config.get('bank_k_set') is not None) and \
                str(config.get('real_estimator', 'bank_mmse')) != 'bank_mmse':
            raise ValueError("bank_dk_frac / bank_k_set require "
                             "real_estimator='bank_mmse'")
        self._bank = None   # list[BankMMSEFit] once _initialize_bank has run
        self.use_dispersion = config.get('use_dispersion', True)
        self.use_confidence = config.get('use_confidence', True)
        # Was documented above and set in configs/estimation.yaml, but never read: every
        # clamp_steepness() call site used the function default. The shipped YAML value
        # happens to equal that default, so behaviour is unchanged -- but the knob was
        # silently inert and would not have taken effect if changed. (2026-07-24)
        self.max_steepness = config.get('max_steepness', 0.15)
        self.seed = config.get('seed', None)

        # Neural corrector (nonlinear correction)
        self.neural_corrector_path = config.get('neural_corrector_path', None)

        n_gaussians = config.get('n_gaussians', [3, 8, 15])
        if isinstance(n_gaussians, int):
            self.stages = [n_gaussians]
        else:
            self.stages = list(n_gaussians)

        # Adaptive N parameters
        self.adaptive_n = config.get('adaptive_n', True)
        self.prune_threshold = config.get('prune_threshold', 0.10)  # energy fraction to discard
        self.grow_residual_threshold = config.get('grow_residual_threshold', 0.05)
        self.max_n = config.get('max_n', 30)

        # Multi-frame tracking
        self.tracking_window = config.get('tracking_window', 3)
        self._frame_buffer = []

        self.model = None
        self.optimizer = None
        self.tracking_optimizer = None
        self._last_dt = 0.1

    def _build_model(self, n: int) -> SpectralGaussianModel:
        # Build model WITHOUT neural corrector (optimization should be clean)
        # NN corrector is loaded but disabled during optimization, enabled only for prediction
        model = SpectralGaussianModel(
            n, depth=self.depth, use_dispersion=self.use_dispersion,
            use_neural_corrector=False)
        if self.neural_corrector_path is not None:
            # Preserve RNG state: NN corrector creation changes torch random state
            rng_state = torch.random.get_rng_state()
            model.load_neural_corrector(self.neural_corrector_path)
            torch.random.set_rng_state(rng_state)
            # Disable during optimization — will be re-enabled for prediction
            model.use_neural_corrector = False
        model.to(self.device)
        return model

    def _build_optimizer(self):
        """Build optimizer for initialization phase (all params unfrozen).

        RESEARCH-ONLY knob `init_k_lr_scale` (default 1.0 = shipped behaviour, bit-identical):
        multiplies the (kx, ky) group learning rate during the INITIALISATION phase only.
        0.0 freezes the wavenumbers at whatever the initializer placed them, which is what
        experiment S2 (2026-08-05) needs to ask "given TRUE (kx, ky), can amplitude+phase
        alone reconstruct the crowded field?". Tracking already freezes kx/ky under the
        shipped tracking_mode='w_phi'; this extends that freeze backwards over init.
        """
        k_lr = self.lr * 0.5 * float(self.config.get('init_k_lr_scale', 1.0))
        param_groups = [
            {'params': [self.model.w, self.model.phi], 'lr': self.lr},
            {'params': [self.model.kx, self.model.ky], 'lr': k_lr},
        ]
        if not self.use_dispersion:
            param_groups.append(
                {'params': [self.model.omega_free], 'lr': self.lr})
        self.optimizer = torch.optim.Adam(param_groups)

    def _build_tracking_optimizer(self):
        """Build optimizer for tracking phase based on tracking_mode.

        Freezes parameters not being updated and creates a fresh Adam optimizer
        with tracking_lr for the remaining parameters.
        """
        mode = self.tracking_mode

        if mode == 'phi_only':
            self.model.kx.requires_grad_(False)
            self.model.ky.requires_grad_(False)
            self.model.w.requires_grad_(False)
            self.model.phi.requires_grad_(True)
            params = [{'params': [self.model.phi], 'lr': self.tracking_lr}]
        elif mode == 'w_phi':
            self.model.kx.requires_grad_(False)
            self.model.ky.requires_grad_(False)
            self.model.w.requires_grad_(True)
            self.model.phi.requires_grad_(True)
            params = [{'params': [self.model.w, self.model.phi],
                        'lr': self.tracking_lr}]
        else:  # 'all' — original behavior
            self.model.kx.requires_grad_(True)
            self.model.ky.requires_grad_(True)
            self.model.w.requires_grad_(True)
            self.model.phi.requires_grad_(True)
            params = [
                {'params': [self.model.w, self.model.phi],
                 'lr': self.tracking_lr},
                {'params': [self.model.kx, self.model.ky],
                 'lr': self.tracking_lr * 0.5},
            ]

        # A1 ablation: omega_free needs gradient in tracking too
        if not self.use_dispersion:
            self.model.omega_free.requires_grad_(True)
            params.append({'params': [self.model.omega_free],
                           'lr': self.tracking_lr})

        self.tracking_optimizer = torch.optim.Adam(params)

    def _compute_loss(self, frames: List[FrameData],
                      reg: bool = True) -> torch.Tensor:
        """Compute weighted MSE + regularization over multiple frames.

        `reg=False` drops the L1 and smoothness penalties (Rx-1 K4, tracking-only).
        Default True == shipped behaviour.
        """
        total_loss = torch.tensor(0.0, device=self.device)
        n_obs = 0
        w_sum = 0.0
        conf_norm = _rx1_gate('RX1_CONF_NORM')     # Rx-1 K3; False == shipped

        for frame in frames:
            if len(frame.x_obs) == 0:
                continue

            z_pred = self.model(frame.x_obs, frame.y_obs, frame.t)
            residual = z_pred - frame.z_obs
            weights = frame.confidence if self.use_confidence \
                else torch.ones_like(frame.confidence)
            weighted_sq = weights * residual ** 2
            total_loss = total_loss + weighted_sq.sum()
            n_obs += len(frame.x_obs)
            if conf_norm:
                w_sum += float(weights.sum())

        if conf_norm and w_sum > 1e-12:
            total_loss = total_loss / w_sum
        elif n_obs > 0:
            total_loss = total_loss / n_obs

        # L1 sparsity on weights. Note the shipped form adds a [m] quantity to an [m^2] loss,
        # so lambda_l1 carries a hidden unit and its effective strength moves with sea state:
        # measured contribution ranges over 3-62% of the total loss across Hs, non-monotonically
        # (tmp/SCALE_AUDIT_fable_2026-07-23.md). SCALE_INVARIANT_L1 multiplies by the
        # observation RMS to make the penalty [m^2] like the data term; False keeps shipped
        # behaviour. The scale is computed under no_grad -- it is a normaliser, not a target.
        if reg and self.lambda_l1 > 0:
            l1 = torch.sum(torch.abs(self.model.w))
            if SCALE_INVARIANT_L1:
                with torch.no_grad():
                    zs = [f.z_obs for f in frames if len(f.z_obs) > 0]
                    scale = torch.sqrt(torch.mean(torch.cat(zs) ** 2)) if zs else None
                if scale is not None and float(scale) > 1e-9:
                    l1 = l1 * scale
            total_loss = total_loss + self.lambda_l1 * l1

        # Spectral smoothness: penalize weight variation (sorted by |k|)
        if reg and self.lambda_smooth > 0 and self.model.n_gaussians > 1:
            with torch.no_grad():
                k_mag = torch.sqrt(self.model.kx ** 2 + self.model.ky ** 2)
                sort_idx = torch.argsort(k_mag)
            w_sorted = self.model.w[sort_idx]
            smooth_penalty = torch.sum((w_sorted[1:] - w_sorted[:-1]) ** 2)
            total_loss = total_loss + self.lambda_smooth * smooth_penalty

        return total_loss

    def _compute_residuals(self, model, frames):
        """Compute observation residuals for coarse-to-fine refinement.

        Returns (x, y, residual_z, t) when init_method is v2/v2_konly,
        or (x, y, residual_z) for v1 (backward compatible).
        """
        all_x, all_y, all_res, all_t = [], [], [], []
        with torch.no_grad():
            for frame in frames:
                if len(frame.x_obs) == 0:
                    continue
                z_pred = model(frame.x_obs, frame.y_obs, frame.t)
                res = (frame.z_obs - z_pred).cpu().numpy()
                n = len(frame.x_obs)
                all_x.append(frame.x_obs.cpu().numpy())
                all_y.append(frame.y_obs.cpu().numpy())
                all_res.append(res)
                all_t.append(np.full(n, frame.t))

        # 'oracle' is the RESEARCH-ONLY S2 arm (2026-08-05). It differs from 'v2_konly'
        # only in where the initial (kx, ky) come from, so every downstream stage --
        # residual form, prune, grow -- must behave identically to 'v2_konly'; hence it
        # joins the beamform family here. Additive: no shipped init_method changes.
        if all_x:
            x = np.concatenate(all_x)
            y = np.concatenate(all_y)
            r = np.concatenate(all_res)
            if self.init_method in ('v2', 'v2_konly', 'learned', 'oracle'):
                return x, y, r, np.concatenate(all_t)
            return x, y, r
        if self.init_method in ('v2', 'v2_konly', 'learned', 'oracle'):
            return np.array([]), np.array([]), np.array([]), np.array([])
        return np.array([]), np.array([]), np.array([])

    def initialize(self, frames: List[FrameData],
                   x_range=(10.0, 250.0), y_range=(-100.0, 100.0),
                   verbose: bool = False) -> Dict:
        """
        Phase 1: Joint optimization over initialization window.

        Args:
            frames: list of FrameData from init window
            x_range, y_range: spatial domain
            verbose: print progress

        Returns:
            dict with 'loss_history', 'n_iters', 'elapsed_s'
        """
        # S8 (2026-08-10) opt-in probe, the mirror image of S8_L1_PHASE1_ONLY (see update()).
        # `lambda_l1` acts in Phase 1 AND in tracking, so decomposing "which locus does the
        # work" from one direction alone is an accounting identity, not evidence. With
        # S8_L1_TRACK_ONLY=1 PHASE 1 is pinned to LAMBDA_L1_PRE_S9 while tracking keeps
        # the configured value. Gate unset (the default) touches nothing -- byte-identical.
        # ⚠CORRECTION 2026-08-21: this comment used to say "pinned to the SHIPPED default".
        # That was true when S8 ran (2026-08-10) and became FALSE the next day, when S9
        # deployed lambda_l1=1e-4.  The pin is to the OLD default and must stay there for
        # S8 to reproduce; only the wording and the bare literal were wrong.
        _l1_p1_saved = None
        if os.environ.get('S8_L1_TRACK_ONLY') == '1':
            _l1_p1_saved = self.lambda_l1
            self.lambda_l1 = LAMBDA_L1_PRE_S9
        try:
            return self._initialize_inner(frames, x_range, y_range, verbose)
        finally:
            if _l1_p1_saved is not None:
                self.lambda_l1 = _l1_p1_saved

    def _initialize_inner(self, frames: List[FrameData],
                          x_range=(10.0, 250.0), y_range=(-100.0, 100.0),
                          verbose: bool = False) -> Dict:
        # WavePrior UNIFIED: init_method='auto' routes learned vs v2_konly from a
        # GT-free signal (the learned proposal's n_selected) computed on the same init
        # window both arms see, so one config value handles every regime. Additive; only
        # entered when init_method=='auto'. It mutates self.init_method to the chosen arm
        # and then falls through to the normal dispatch below. The shipped default
        # (v2_konly) never enters this branch.
        # ⚠2026-08-22 함정 (읽고 넘어가지 마라). 아래 실측 게이트는 `init_method == 'auto'`
        # **안에만** 있다. 이것은 의도된 계약이다 — 명시적 `init_method` 는 "연구자가 일부러
        # 고른 것"으로 존중되고(`tests/test_bank_mmse_port.py::test_explicit_init_method_never_banks`),
        # 게이트는 **기본 경로**를 지킨다.
        # ⛔**그러므로 "한 방법으로 통일" 을 `init_method` 기본값을 'learned' 로 바꿔서 하면
        #    실측 게이트가 조용히 사라진다.** 그 상태가 무엇인지는 측정됐다 —
        #    `tmp/W12_UNIFY_REAL_2026-08-22`: 실측에 학습을 강제하면 nowcast 가 제로필드를
        #    2/5 만 이기고 **예보 지평에서는 0/5**(중앙 1.69배 = 아무것도 안 하는 것보다 나쁘다).
        # ✅**안전한 통일 방법**: 기본값을 `'auto'` 로 **유지**하고 라우터 임계
        #    (`auto_switch_thresh`)를 0 으로 두어 합성에서 항상 학습을 고르게 한다.
        #    그러면 실측 게이트는 그대로 작동한다. (근거: `tmp/W11_UNIFY_2026-08-22`)
        if self.init_method == 'auto':
            # REAL-DOMAIN GATE (2026-07-27). The deployment authorisation covers SYNTHETIC
            # inputs only: "실측 경로 별도 게이트 -- 실측 입력에서 auto는 여전히 비활성이어야 함"
            # (tmp/confirm2_2026-07-27/RESULT_CONFIRM2.md §4, residual condition, unreleased).
            # WHY IT IS NOT OPTIONAL: on all 30 real windows the proposal's n_selected is
            # >= 8 (the sparse observation geometry makes it over-propose), so the th=8
            # router would route 30/30 to the learned arm -- the arm measured to be no
            # better and sometimes worse on real seas -- and the retired guard fired 0/30
            # (docs/HANDOFF_OPUS_2026-07-26.md §"Blind auto (W5 official)"). Without this
            # gate the shipped default silently applies an unauthorised policy to real data.
            # SIGNAL: absence of a ground-truth field. It is the one property that separates
            # every real path (wavesplat/real_data.py builds FrameData with gt_field=None,
            # as do the scene_real harnesses) from every synthetic path (EpisodeDataLoader
            # always attaches a non-zero gt_field). It needs no filename convention, no
            # loader flag and no metadata key, so it cannot be bypassed by a new real
            # front-end. Same test run.py uses for `has_gt`, applied to the init window.
            # Escape hatch for deliberate research use: real_domain_gate=False.
            if frames and self.config.get('real_domain_gate', True) \
                    and not _frames_have_gt(frames):
                import sys as _sys
                import warnings
                if self.real_estimator not in ('bank_mmse', 'v2_konly'):
                    raise ValueError(
                        f"real_estimator must be 'bank_mmse' or 'v2_konly', "
                        f"got {self.real_estimator!r}")
                self._auto_real_gate = True
                if self.real_estimator == 'bank_mmse':
                    # BANK-MMSE PORT (2026-07-29, user-directed deployment). The router
                    # stays bypassed on real input (the learned arm's n_selected is >= 8
                    # on every measured window and it has no demonstrated real-sea
                    # benefit); what changed is the fallback: the single v2_konly run was
                    # measured WORSE than the research bank-MMSE on the frozen real
                    # windows (nowcast nRMSE 0.858/0.927/1.048 vs 0.498/0.666/0.718), so
                    # the gate now routes to the ported estimator under an
                    # observation-only parametrization.
                    #   CORRECTION 2026-08-11 (P1 watchdog, re-derived from raw):
                    #   this comment used to say "~2x worse" and cite m3_field.json for
                    #   BOTH arms.  Both were wrong.  (a) The ratios are 1.723/1.392/1.460
                    #   -- median 1.46x, and NO window reaches 2x.  (b) The two arms live
                    #   in DIFFERENT files: bank = tmp/bankmmse_port_2026-07-29/
                    #   m3_field.json, v2_konly = tmp/confirm2_2026-07-27/real_gif_info.json.
                    #   The routing decision itself stands (bank wins all 3 windows); only
                    #   the stated magnitude was inflated.  Do not cite "2x".
                    msg = ("real (ground-truth-free) input detected: the learned/router "
                           "path is not authorised here; routing to the ported bank-MMSE "
                           "estimator (observation-only parameters, median-of-k field). "
                           "Set real_estimator='v2_konly' for the pre-2026-07-29 "
                           "fallback, or real_domain_gate=False to reach the router "
                           "deliberately.")
                    warnings.warn(msg, RuntimeWarning, stacklevel=2)
                    print(f"  [Init/auto] REAL-DOMAIN GATE -> bank_mmse. {msg}",
                          file=_sys.stderr, flush=True)
                    self._auto_selected = 'bank_mmse'
                    self.init_method = 'bank_mmse'
                    return self._initialize_bank(frames, verbose)
                msg = ("init_method='auto' is NOT authorised on real (ground-truth-free) "
                       "input: the learned arm's n_selected is >= 8 on every measured "
                       "window, so the th=8 router would route 100% to an arm with no "
                       "demonstrated real-sea benefit. Forcing init_method='v2_konly' "
                       "(single run, no median-of-k ensemble). Set real_domain_gate=False "
                       "to override deliberately.")
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                print(f"  [Init/auto] REAL-DOMAIN GATE -> v2_konly (single). {msg}",
                      file=_sys.stderr, flush=True)
                self._auto_selected = 'v2_konly'
                self.init_method = 'v2_konly'
            else:
                self._auto_real_gate = False
                self.init_method = self._auto_select_init(frames, verbose)

        # WavePrior Stage-1: learned set-proposal supplies the component set, then the
        # SAME joint fit / adaptive-N / tracker run on top. Additive; the shipped
        # default (v2_konly) never enters this branch.
        if self.init_method == 'learned':
            return self._initialize_learned(frames, x_range, y_range, verbose)

        t0 = time.time()
        loss_history = []

        # Coarse-to-fine schedule (configurable thresholds)
        default_thresholds = [0.1, 0.05, None] if len(self.stages) == 3 else [None]
        default_max_iters = [200, 200, 500] if len(self.stages) == 3 else [500]
        schedule = CoarseToFineSchedule(
            stages=self.stages,
            loss_thresholds=self.loss_thresholds or default_thresholds,
            max_iters_per_stage=self.max_iters_per_stage or default_max_iters,
            init_method=self.init_method,
            depth=self.depth,
        )

        # Observation-variance scale for the (optional) scale-invariant stage-advance test.
        # Set unconditionally; it is only consulted when
        # coarse_to_fine.SCALE_INVARIANT_ADVANCE is True.
        try:
            _z = np.concatenate([f.z_obs.detach().cpu().numpy().ravel() for f in frames])
            schedule.loss_scale = float(np.mean(_z ** 2)) if _z.size else None
        except Exception:
            schedule.loss_scale = None

        # Initialize first stage
        n_init = schedule.current_n()
        init_kx, init_ky, init_w, init_phi = initialize_from_frames(
            frames, n_init, self.depth, x_range, y_range,
            init_method=self.init_method, seed=self.seed
        )

        self.model = self._build_model(n_init)
        with torch.no_grad():
            self.model.kx.copy_(init_kx.to(self.device))
            self.model.ky.copy_(init_ky.to(self.device))
            self.model.w.copy_(init_w.to(self.device))
            self.model.phi.copy_(init_phi.to(self.device))
        self._build_optimizer()

        stage_iters = 0
        total_iters = 0
        max_total = sum(schedule.max_iters)

        for iteration in range(max_total):
            self.optimizer.zero_grad()
            loss = self._compute_loss(frames)
            loss.backward()
            self.optimizer.step()
            self.model.clamp_steepness(self.max_steepness)

            loss_val = loss.item()
            loss_history.append(loss_val)
            stage_iters += 1
            total_iters += 1

            if verbose and iteration % 50 == 0:
                print(f"  [Init] iter={iteration}, N={self.model.n_gaussians}, "
                      f"loss={loss_val:.6f}")

            # Check coarse-to-fine advancement
            if schedule.should_advance(loss_val, stage_iters):
                if verbose:
                    print(f"  [Refine] N={schedule.current_n()} → "
                          f"{schedule.stages[schedule.current_stage + 1]}")
                schedule.advance(
                    self.model, frames, self._compute_residuals,
                    x_range, y_range
                )
                self._build_optimizer()
                stage_iters = 0

                if not schedule.has_next():
                    # Final stage: continue until max_iters
                    continue

        elapsed = time.time() - t0
        if verbose:
            print(f"  [Init complete] {total_iters} iters, "
                  f"N={self.model.n_gaussians}, "
                  f"loss={loss_history[-1]:.6f}, {elapsed:.1f}s")

        # Set _last_t for tracking phase
        if frames:
            self._last_t = frames[-1].t

        # Adaptive N: prune weak components, grow if residual high
        if self.adaptive_n:
            self._adaptive_prune_grow(frames, x_range, y_range, verbose)

        # Build tracking-specific optimizer (freezes params per tracking_mode)
        self._build_tracking_optimizer()

        return {
            'loss_history': loss_history,
            'n_iters': total_iters,
            'elapsed_s': elapsed,
            'final_n': self.model.n_gaussians,
        }

    def _initialize_learned(self, frames: List[FrameData],
                            x_range=(10.0, 250.0), y_range=(-100.0, 100.0),
                            verbose: bool = False) -> Dict:
        """Phase 1 for init_method='learned' (WavePrior Stage-1).

        The learned set-proposal supplies the FULL component set at once (kx, ky and,
        unless learned_konly, LS-fit w/phi), so the coarse-to-fine N-staging is
        bypassed -- there is nothing to grow into. Everything after the install is the
        shipped pipeline: the same joint-fit optimizer / loss, the same adaptive
        prune+grow, the same tracking optimizer. Only the kx/ky *source* changes.
        """
        from wavesplat.optimizer.learned_init import learned_init_params

        t0 = time.time()
        init_kx, init_ky, init_w, init_phi, n = learned_init_params(
            frames, self.depth, self.config, seed=self.seed)

        self.model = self._build_model(n)
        with torch.no_grad():
            self.model.kx.copy_(init_kx.to(self.device))
            self.model.ky.copy_(init_ky.to(self.device))
            self.model.w.copy_(init_w.to(self.device))
            self.model.phi.copy_(init_phi.to(self.device))
        self._build_optimizer()

        # Joint fit at full N (analog of the shipped final-stage polish budget).
        max_iters = int(self.config.get('learned_init_iters', LEARNED_INIT_ITERS_DEFAULT))
        loss_history = []
        for iteration in range(max_iters):
            self.optimizer.zero_grad()
            loss = self._compute_loss(frames)
            loss.backward()
            self.optimizer.step()
            self.model.clamp_steepness(self.max_steepness)
            loss_history.append(loss.item())
            if verbose and iteration % 100 == 0:
                print(f"  [Init/learned] iter={iteration}, N={self.model.n_gaussians}, "
                      f"loss={loss.item():.6f}")

        if frames:
            self._last_t = frames[-1].t

        if self.adaptive_n:
            self._adaptive_prune_grow(frames, x_range, y_range, verbose)

        self._build_tracking_optimizer()

        # ── VARPRO (2026-08-22 배포) ─────────────────────────────────────────
        # 원 몽키패치(`tmp/W4_SOLVER_2026-08-19/varpro.py:install`)는 `_initialize_learned`
        # 전체가 끝난 **뒤** 적용했다. 배선 지점을 그대로 맞춘다 = 그 배치들과 같은 순서.
        vp_steps = int(self.config.get('varpro_steps', VARPRO_STEPS_DEFAULT))
        vp_rec = None
        if vp_steps > 0:
            from wavesplat.optimizer import varpro as _varpro
            vp_rec = _varpro.apply_varpro(self, frames, steps=vp_steps,
                                          weighting='confidence', verbose=verbose)

        # ── 표현 절제: 파수 격자 스냅 (기본 None = no-op) ────────────────────
        snap_rec = self._snap_k_to_grid(frames, verbose)

        elapsed = time.time() - t0
        if verbose:
            print(f"  [Init/learned complete] {max_iters} iters, "
                  f"N={self.model.n_gaussians}, "
                  f"loss={loss_history[-1] if loss_history else float('nan'):.6f}, "
                  f"{elapsed:.1f}s")
        return {
            'loss_history': loss_history,
            'n_iters': max_iters,
            'elapsed_s': elapsed,
            'final_n': self.model.n_gaussians,
            'varpro_steps': vp_steps,
            'varpro': vp_rec,
            'k_snap': snap_rec,
        }

    def _snap_k_to_grid(self, frames, verbose: bool = False):
        """파수를 간격 `k_snap_dk` 격자에 스냅하고 (w, φ) 를 LS 로 다시 맞춘다.

        ★`k_snap_dk` 가 None/0 이면 **아무것도 하지 않고 None 을 반환**한다
        (모델을 건드리지 않으므로 배포 경로는 비트 동일하다).
        반환 dict 은 **개입 계수기**다 — *"설정을 넣었다 ≠ 설정이 먹었다"* 를 막는다.
        """
        dk = self.config.get('k_snap_dk', K_SNAP_DK_DEFAULT)
        if not dk:
            return None
        dk = float(dk)
        import numpy as _np
        from wavesplat.optimizer.learned_init import ls_fit_wphi
        with torch.no_grad():
            kx0 = self.model.kx.detach().cpu().numpy().astype(float).copy()
            ky0 = self.model.ky.detach().cpu().numpy().astype(float).copy()
        kx1 = _np.round(kx0 / dk) * dk
        ky1 = _np.round(ky0 / dk) * dk
        w1, p1 = ls_fit_wphi(kx1, ky1, frames, float(self.config.get('depth', 50.0)))
        with torch.no_grad():
            self.model.kx.copy_(torch.as_tensor(kx1, dtype=self.model.kx.dtype,
                                                device=self.model.kx.device))
            self.model.ky.copy_(torch.as_tensor(ky1, dtype=self.model.ky.dtype,
                                                device=self.model.ky.device))
            self.model.w.copy_(torch.as_tensor(_np.asarray(w1, float),
                                               dtype=self.model.w.dtype,
                                               device=self.model.w.device))
            self.model.phi.copy_(torch.as_tensor(_np.asarray(p1, float),
                                                 dtype=self.model.phi.dtype,
                                                 device=self.model.phi.device))
        self._build_tracking_optimizer()      # 파라미터를 바꿨으므로 옵티마이저 재구성
        shift = _np.hypot(kx1 - kx0, ky1 - ky0)
        uniq = len({(round(a / dk), round(b / dk)) for a, b in zip(kx1, ky1)})
        rec = dict(dk=dk, n_atoms=int(len(kx0)), n_unique_cells=int(uniq),
                   shift_median=float(_np.median(shift)) if len(shift) else 0.0,
                   shift_max=float(shift.max()) if len(shift) else 0.0,
                   fired=True)
        if verbose:
            print(f"  [k-snap] dk={dk:.4f}  atoms={rec['n_atoms']} -> "
                  f"{rec['n_unique_cells']} cells, |Δk| med {rec['shift_median']:.4f}")
        return rec

    def _auto_select_init(self, frames: List[FrameData], verbose: bool = False) -> str:
        """WavePrior UNIFIED: GT-free regime router for init_method='auto'.

        Computes the learned set-proposal's selected-component count (n_selected) on the
        same init window both arms see, and returns 'learned' when
        n_selected >= auto_switch_thresh, else 'v2_konly'.
        ⚠**기본값은 2026-08-22 에 8 -> 0 으로 바뀌었다**(합성 경로 통일: 임계 0 이면
        성분 수와 무관하게 항상 learned). 그전 문구는 8 을 *"nRMSE 로 튜닝하지 않은
        Stage-1 혼잡 신호"* 라 정당화했는데 **그 정당화는 0 에 대해 성립하지 않는다** —
        0 은 라우팅을 끄는 값이고 W11/W13/W14 의 정확도 비교로 골랐다.
        ⛔그 비교의 해석적 팔은 median-of-3 앙상블을 받지 못한 **약해진 비교자**였다
        (`make_updater` 는 init_method=='auto' 일 때만 앙상블을 준다). 재측정 = W15.
        ⚠**임계 0 의 부작용**: n_selected==0 도 0>=0 이라 learned 로 간다 = 축퇴 폴백.

        The raw proposal atoms are cached into config['learned_atoms'] so a 'learned' route reuses them
        without re-running the (GPU) diffusion proposal. Purely additive; only called when
        init_method=='auto'. No GT is used -- the decision is made from observations alone.

        Config knobs (all optional): auto_switch_thresh (int, 기본 AUTO_SWITCH_THRESH_DEFAULT=0),
        learned_atoms (precomputed raw atoms dict, skips the GPU proposal),
        learned_ckpt / learned_nsamp / learned_diffusion_seed (proposal),
        learned_energy_thresh / learned_dedup_dk / max_n (pooling, must match the
        learned arm's so the signal and the installed set agree).
        """
        from wavesplat.optimizer.learned_init import (
            propose_atoms, pool_and_select, DEFAULT_CKPT)
        cfg = self.config
        thresh = int(cfg.get('auto_switch_thresh', AUTO_SWITCH_THRESH_DEFAULT))
        cache = cfg.get('learned_atoms')
        if cache is not None:
            rkx = np.asarray(cache['kx'], float)
            rky = np.asarray(cache['ky'], float)
            ren = np.asarray(cache['energy'], float)
        else:
            dseed = int(cfg.get('learned_diffusion_seed',
                                0 if self.seed is None else self.seed))
            try:
                rkx, rky, ren = propose_atoms(
                    frames, self.depth,
                    ckpt=cfg.get('learned_ckpt', DEFAULT_CKPT),
                    nsamp=int(cfg.get('learned_nsamp', 4)),
                    diffusion_seed=dseed)
            except (FileNotFoundError, OSError, ImportError) as e:
                # 'auto' is the shipped default: a missing/unreadable diffusion checkpoint
                # must degrade to the analytic path, not crash the estimator.
                import warnings
                warnings.warn(f"init_method='auto': learned proposal unavailable "
                              f"({e}); falling back to v2_konly.")
                self._auto_n_selected = None
                self._auto_guard_fired = None
                self._auto_selected = 'v2_konly'
                return 'v2_konly'
            # Cache so a 'learned' route reuses these exact atoms (no double proposal).
            cfg['learned_atoms'] = dict(kx=rkx, ky=rky, energy=ren)
        sk, _, _ = pool_and_select(
            rkx, rky, ren,
            energy_thresh=float(cfg.get('learned_energy_thresh', 0.03)),
            dedup_dk=float(cfg.get('learned_dedup_dk', 0.03)),
            max_n=int(cfg.get('max_n', 30)))
        n_sel = int(len(sk))
        chosen = 'learned' if n_sel >= thresh else 'v2_konly'
        self._auto_n_selected = n_sel

        # WavePrior UNIFIED-GUARD (additive): the proposal over-proposes ~30 near-degenerate
        # atoms on a single clean swell (the characterized BF2_1comp misroute), so n_selected
        # says 'crowded' when the sea is not. The learned atoms cannot self-diagnose this (the
        # slope over-proposals are atom-level indistinguishable from a crowded spectrum). The
        # v2 beamform power map, computed from the observations alone, does: a single swell
        # yields one dominant peak (bf_peak_ratio ~ 0). When the base decision is 'learned' and
        # the beamform is that single-dominant, veto -> v2_konly. GT-free; never promotes
        # v2->learned. Off by setting auto_guard=False (then this block is inert).
        # Threshold 0.05 sits below the crowded-sea beamform-ratio floor measured on the
        # design set (tmp/unified_guard_2026-07-26/), so no crowded route is flipped there.
        #
        # ── RETIRED 2026-07-27: default flipped True -> False ────────────────────────
        # The guard was tuned on the design set and did NOT survive the fresh held-out 40
        # (docs/HANDOFF_OPUS_2026-07-26.md S2 FAIL): out of sample it fired on episodes it
        # should not have and stayed silent on the one that exploded (bfr 0.065 > 0.05),
        # i.e. it was WORSE than no-guard. It is also uninformative on real seas
        # (bf_peak_ratio on all 30 real windows carries no signal, ibid.). The confirmation
        # round that authorised deployment specifies "router th=8 + guard retired"
        # (tmp/confirm2_2026-07-27/RESULT_CONFIRM2.md §4, PREREG_CONFIRM2.md L41).
        # The code is kept, not deleted, so the historical configuration is reproducible by
        # passing auto_guard=True; nothing in the shipped path sets it.
        guard = bool(cfg.get('auto_guard', False))
        guard_ratio = float(cfg.get('auto_guard_ratio', 0.05))
        fired = False
        bfr = None
        if guard and chosen == 'learned':
            from wavesplat.optimizer.initializer import beamform_peak_ratio
            bfr = float(beamform_peak_ratio(frames, self.depth))
            if bfr < guard_ratio:
                chosen = 'v2_konly'
                fired = True
        self._auto_guard_fired = fired
        self._auto_bf_peak_ratio = bfr
        self._auto_selected = chosen
        if verbose:
            gtag = (f" guard(bf_ratio={bfr:.3f}<{guard_ratio})->veto" if fired
                    else (f" guard(bf_ratio={bfr:.3f})ok" if bfr is not None else ""))
            print(f"  [Init/auto] n_selected={n_sel} (thresh={thresh}) -> {chosen}{gtag}")
        return chosen

    @staticmethod
    def _bank_obs_from_frames(frames: List[FrameData]):
        """(n,3) obs points P=(x,y,t) and elevations z from FrameData, time-major."""
        P_list, z_list = [], []
        for f in frames:
            if len(f.x_obs) == 0:
                continue
            x = f.x_obs.detach().cpu().numpy().ravel()
            y = f.y_obs.detach().cpu().numpy().ravel()
            zz = f.z_obs.detach().cpu().numpy().ravel()
            P_list.append(np.column_stack([x, y, np.full(len(x), float(f.t))]))
            z_list.append(zz)
        if not P_list:
            return np.zeros((0, 3)), np.zeros(0)
        return np.vstack(P_list).astype(float), np.concatenate(z_list).astype(float)

    def _initialize_bank(self, frames: List[FrameData],
                         verbose: bool = False) -> Dict:
        """Phase 1 for the real-domain route: median-of-k bank-MMSE field ensemble.

        The estimator is the ported research bank-MMSE (optimizer/bank_mmse.py --
        the call behind every published real-sea field number), fitted on the init
        window's observations with OBSERVATION-ONLY parameters: sigma0 = 0.05*Hs_obs
        (Hs_obs = 4*std(z)) and kp from the estimator's own two-stage beamform zoom.
        Members differ only in the val-split seed {s, s+1000, s+2000, ...} (the
        MedianEnsembleUpdater offsets); prediction is the pointwise-median field.
        Measured 2026-07-29 on the frozen GIF-ledger windows against the previous
        single-v2_konly route (nowcast nRMSE): 0.858 -> 0.498 (AA15a w5),
        0.927 -> 0.666 (AA15b w3), 1.048 -> 0.718 (YS17 w0)
        (tmp/bankmmse_port_2026-07-29/{blind_vs_frozen,m3_field}.json).

        No gradient tracking on this route: the fit is closed-form linear in z.
        update() maintains a rolling observation window and, only if
        bank_refit_every_s is set, refits from it; the default (None) keeps the fit
        frozen, which is exactly the protocol the real-sea numbers were measured
        under (pure forecast after the init window).
        """
        t0 = time.time()
        P, z = self._bank_obs_from_frames(frames)
        if len(z) < 32:
            raise ValueError(f"bank-MMSE route needs >= 32 observations in the "
                             f"init window, got {len(z)}")
        k = int(self.config.get('real_bank_ensemble_k',
                                DEFAULT_REAL_BANK_ENSEMBLE_K))
        if not 1 <= k <= 5:
            raise ValueError('real_bank_ensemble_k must be 1..5')
        base = 0 if self.seed is None else int(self.seed)
        self._bank_seeds = [base + off for off in (0, 1000, 2000, 3000, 4000)[:k]]
        self._bank_fit_window(P, z)
        self._bank_obs = (P, z)
        self._bank_memory_s = float(self.config.get('bank_memory_s', 40.0))
        self._bank_refit_every_s = self.config.get('bank_refit_every_s',
                                                   BANK_REFIT_EVERY_S_DEFAULT)
        self._bank_last_fit_t = float(np.max(P[:, 2]))
        self._last_t = frames[-1].t
        elapsed = time.time() - t0
        if verbose:
            print(f"  [Init/bank] median-of-{k} bank-MMSE, seeds={self._bank_seeds}, "
                  f"kp={[round(f.kp, 4) for f in self._bank]}, "
                  f"B={[f.n_candidates for f in self._bank]}, "
                  f"val_misfit={[round(f.vr, 4) for f in self._bank]}, "
                  f"{elapsed:.1f}s")
        return {
            # per-member held-out predictive misfit: the GT-free fit-quality signal
            'loss_history': [f.vr for f in self._bank],
            'n_iters': 0,
            'elapsed_s': elapsed,
            'final_n': 0,               # no Gaussian components on this route
            'bank_members': k,
            'bank_seeds': list(self._bank_seeds),
            'bank_kp': [f.kp for f in self._bank],
            'bank_n_candidates': [f.n_candidates for f in self._bank],
            'bank_val_misfit': [f.vr for f in self._bank],
            'Hs_obs': self._bank_Hs_obs,
        }

    def _bank_fit_window(self, P, z):
        """Fit the k-member ensemble from one observation window (GT-free).

        The window is capped at bank_max_obs points with the SAME deterministic
        subsample the frozen research harness used (scene_real m_cap=600,
        default_rng(0)): every published real-sea bank number was measured on
        600-obs windows, and the cap also bounds the O(n^2 B) fit cost."""
        from wavesplat.optimizer.bank_mmse import fit_bank_mmse

        cap = int(self.config.get('bank_max_obs', 600))
        if len(z) > cap:
            ii = np.random.default_rng(0).choice(len(z), cap, replace=False)
            P, z = P[ii], z[ii]
        self._bank_Hs_obs = 4.0 * float(np.std(z))
        sigma0 = 0.05 * self._bank_Hs_obs
        # RESEARCH-ONLY (2026-08-28, tmp/M8_LADDER_2026-08-28/PATCH.md): both keys
        # default to None = the pre-patch call, BIT-IDENTICAL (regression
        # tmp/M8_LADDER_2026-08-28/regress_bank_mmse.py).  Not on the deployment
        # surface (run.py DEFAULT_CONFIG unchanged); the guard in __init__ refuses
        # them on a route that would silently ignore them.
        self._bank = [fit_bank_mmse(P, z, self.depth, sigma0, kp_hint=None,
                                    seed=s,
                                    dk_frac=self.config.get('bank_dk_frac'),
                                    k_set=self.config.get('bank_k_set'))
                      for s in self._bank_seeds]

    # ── 비동기 재적합 (2026-08-23, 기본 OFF) ──────────────────────────────────
    # 왜: 재적합 단가 중앙 1.714 s 대 프레임 간격 0.067–0.100 s = **약 20배**.  동기로 돌면
    # 그 프레임이 실시간 예산을 확정적으로 깬다(B-6/RV-6).  적합은 **관측 창의 순수 함수**이고
    # 결과를 `self._bank` 에 **한 번에** 대입할 뿐이므로 배경 스레드로 옮겨도 의미가 안 바뀐다.
    # 계약: ①`kick` 은 즉시 반환한다 ②적합이 도는 동안 **현재 앙상블을 계속 쓴다**
    #      ③`collect` 는 완료된 결과만 원자적으로 교체한다 ④진행 중이면 새 kick 을 **무시**한다
    #      (큐를 쌓지 않는다 -- 늦은 적합은 어차피 낡은 창이다).
    def _bank_refit_async_kick(self, P, z):
        import threading
        fut = getattr(self, '_bank_async', None)
        if fut is not None and fut['thread'].is_alive():
            self._bank_async_skipped = getattr(self, '_bank_async_skipped', 0) + 1
            return
        # ★2026-08-23 진단 (B-9 의 교훈): 비동기의 **효과는 호출자의 벽시계 페이싱에 의존한다.**
        # 재생 하네스가 실시간보다 빠르면(`B-9` 실측: 런 벽시계 기준 **중앙 6배**;
        # ⚠종전 주석의 "494배" 는 **`update()` 점유율** 기준이라 틀렸다 --- 2026-08-23 R2 토론 F-15)
        # 적합(≈1.7 s)이 **채점이 끝난 뒤에** 착지해 `bank_refit_async=True` 가 **켜져 있는데 실질은 꺼진**
        # 상태가 된다.  그것이 `B-9` 에서 정확도를 "다 잃은 것처럼" 보이게 했다(상대중앙 +0.00%).
        # ⟹ 그 상황을 **조용히 넘기지 않는다**: 두 번째 kick 인데 적용이 0회면 경고한다.
        if getattr(self, '_bank_async_kicks', 0) >= 1 \
                and getattr(self, '_bank_async_applied', 0) == 0 \
                and not getattr(self, '_bank_async_warned', False):
            import warnings
            self._bank_async_warned = True
            warnings.warn(
                'bank_refit_async is ON but no background fit has been APPLIED yet after '
                f'{getattr(self, "_bank_async_kicks", 0)} kick(s).  The fit takes ~seconds of '
                'WALL time; if the caller replays frames faster than real time the refit lands '
                'after scoring and the flag is effectively a no-op.  Pace the loop to real time '
                'or call bank_refit_wait().  (measured: tmp/B9_ASYNC_2026-08-23/RESULT_B9.md)',
                RuntimeWarning)
        P, z = np.array(P, copy=True), np.array(z, copy=True)
        box = {'result': None, 'error': None}

        def _work():
            try:
                # 임시 인스턴스에 적합해 self 를 건드리지 않는다(스레드 안전).
                import copy as _copy
                shim = _copy.copy(self)
                shim._bank = None
                type(self)._bank_fit_window(shim, P, z)
                box['result'] = (shim._bank, getattr(shim, '_bank_Hs_obs', None))
            except Exception as e:                      # noqa: BLE001
                box['error'] = e

        th = threading.Thread(target=_work, daemon=True, name='bank_refit')
        th.start()
        self._bank_async = {'thread': th, 'box': box}
        self._bank_async_kicks = getattr(self, '_bank_async_kicks', 0) + 1

    def _bank_refit_async_collect(self):
        """완료된 배경 적합이 있으면 원자적으로 교체한다. 없으면 no-op."""
        fut = getattr(self, '_bank_async', None)
        if fut is None or fut['thread'].is_alive():
            return False
        self._bank_async = None
        if fut['box']['error'] is not None:
            # ⚠적합 실패는 **조용히 넘어가지 않는다** -- 현재 앙상블을 유지하고 경고한다.
            import warnings
            warnings.warn(f'async bank refit failed, keeping current fit: '
                          f'{fut["box"]["error"]!r}', RuntimeWarning)
            return False
        bank, hs = fut['box']['result']
        if bank is not None:
            self._bank = bank
            if hs is not None:
                self._bank_Hs_obs = hs
            self._bank_async_applied = getattr(self, '_bank_async_applied', 0) + 1
            return True
        return False

    def bank_refit_health(self):
        """비동기 재적합이 **실질적으로 작동했는지** 되읽는다 (배치 게이트용).

        `B-9`/`B-9b` 가 보인 것: 같은 설정이 **호출자의 페이싱에 따라 작동하기도, 하지 않기도** 한다
        (재생 6배에서는 applied 가 채점 뒤에야 올랐다).  ⟹ 배치는 **의도표가 아니라 이것을** 봐야 한다.
        """
        k = getattr(self, '_bank_async_kicks', 0)
        a = getattr(self, '_bank_async_applied', 0)
        return {'enabled': bool(self.config.get('bank_refit_async', False)),
                'kicks': k, 'applied': a,
                'skipped': getattr(self, '_bank_async_skipped', 0),
                'in_flight': bool(getattr(self, '_bank_async', None) is not None
                                  and self._bank_async['thread'].is_alive()),
                'effective': bool(k > 0 and a > 0)}

    def bank_refit_wait(self, timeout=None):
        """시험·배치용: 배경 적합이 끝날 때까지 기다렸다가 교체한다."""
        fut = getattr(self, '_bank_async', None)
        if fut is not None:
            fut['thread'].join(timeout)
        return self._bank_refit_async_collect()

    def _bank_update(self, frame: FrameData) -> Dict:
        """Phase 2 on the bank-MMSE route: rolling window + optional refit.

        Each frame's observations join the rolling memory window (bank_memory_s,
        default 40 s = the frozen protocol's window). If bank_refit_every_s is set,
        the ensemble refits from that window once enough time has accumulated (a
        full fit -- seconds, schedule accordingly); the default None keeps the init
        fit frozen, matching the validated pure-forecast protocol. 'loss' is the
        GT-free observation misfit of the current median field at this frame's
        points (the analogue of the gradient path's obs-fit loss).
        """
        t0 = time.time()
        self._last_t = frame.t
        if len(frame.x_obs) == 0:
            return {'loss': float('nan'), 'n_pts': 0, 'elapsed_ms': 0.0}
        Pf, zf = self._bank_obs_from_frames([frame])
        P, z = self._bank_obs
        P = np.vstack([P, Pf])
        z = np.concatenate([z, zf])
        keep = P[:, 2] >= float(frame.t) - self._bank_memory_s
        self._bank_obs = (P[keep], z[keep])
        if self._bank_refit_every_s is not None and \
                float(frame.t) - self._bank_last_fit_t \
                >= float(self._bank_refit_every_s):
            # ★2026-08-23: 동기 재적합은 프레임 간격의 **약 20배**(실측 1.35–2.53 s) 동안
            # 이 호출을 붙잡는다 -- 그것이 배포를 되돌린 사유다(tmp/B6_BUDGET_2026-08-23/
            # REVERT_2026-08-23.md).  `bank_refit_async=True` 면 **배경 스레드**에서 적합하고
            # 그동안 **현재 앙상블을 계속 쓴다**; 완료된 다음 프레임에서 원자적으로 교체한다.
            # ⚠기본값은 False = **비트 동일한 종전 거동**.
            if self.config.get('bank_refit_async', False):
                self._bank_refit_async_kick(*self._bank_obs)
            else:
                self._bank_fit_window(*self._bank_obs)
            self._bank_last_fit_t = float(frame.t)
        self._bank_refit_async_collect()
        preds = np.stack([f.render(Pf) for f in self._bank], axis=0)
        med = np.median(preds, axis=0)
        return {'loss': float(np.mean((med - zf) ** 2)),
                'n_pts': int(len(zf)),
                'elapsed_ms': (time.time() - t0) * 1000.0}

    def _bank_predict_grid(self, grid_x, grid_y, t: float) -> torch.Tensor:
        """(Ny, Nx) pointwise-median field of the bank ensemble at time t --
        same grid convention as SpectralGaussianModel.predict_grid."""
        gx = np.asarray(grid_x.detach().cpu().numpy() if torch.is_tensor(grid_x)
                        else grid_x, dtype=float).ravel()
        gy = np.asarray(grid_y.detach().cpu().numpy() if torch.is_tensor(grid_y)
                        else grid_y, dtype=float).ravel()
        Xg, Yg = np.meshgrid(gx, gy)
        Q = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, float(t))])
        preds = np.stack([f.render(Q) for f in self._bank], axis=0)
        med = np.median(preds, axis=0).reshape(len(gy), len(gx))
        return torch.as_tensor(med, dtype=torch.float32)

    def update(self, frame: FrameData) -> Dict:
        """
        Phase 2: Single-frame tracking update.

        1. Predict: φᵢ += ωᵢ·Δt
        2. Update: gradient descent on single-frame loss
        3. Return timing and loss info

        Args:
            frame: single FrameData

        Returns:
            dict with 'loss', 'n_pts', 'elapsed_ms'
        """
        if self._bank is not None:
            return self._bank_update(frame)

        t0 = time.time()

        if self.model is None:
            raise RuntimeError("Must call initialize() before update()")

        # Phase prediction: advance phases by ω·Δt.
        # NOTE: this is redundant with the forward model's own -ω·t term and, when enabled,
        # double-counts time (see TRACK_PHASE_PREDICT note above). Gated so shipped behaviour
        # (advance ON) is the default; TRACK_PHASE_PREDICT=False skips it.
        dt = frame.t - getattr(self, '_last_t', frame.t)
        if TRACK_PHASE_PREDICT and abs(dt) > 1e-6:
            with torch.no_grad():
                omega = self.model._omega()
                self.model.phi.add_(omega * dt)
        self._last_t = frame.t

        if len(frame.x_obs) == 0:
            return {'loss': float('nan'), 'n_pts': 0, 'elapsed_ms': 0.0}

        # Multi-frame window: keep recent frames for more stable gradients
        self._frame_buffer.append(frame)
        if len(self._frame_buffer) > self.tracking_window:
            self._frame_buffer.pop(0)

        # Gradient update using tracking optimizer on window
        opt = self.tracking_optimizer if self.tracking_optimizer is not None \
            else self.optimizer
        # Rx-1 K4: `reg=False` makes the TRACKING objective the data term alone while
        # Phase 1 keeps its penalties. Default (gate unset) is reg=True == shipped.
        track_reg = not _rx1_gate('RX1_PHASE2_NOREG')
        # S8 (2026-08-10) opt-in probe.  `lambda_l1` is a single config key but it acts in
        # TWO places: the Phase-1 fit (`_compute_loss` via `initialize`) and this tracking
        # loop.  So "change one line in estimation.yaml" moves two loci at once, and a
        # deployment claim cannot name a single mechanism.  With S8_L1_PHASE1_ONLY=1 the
        # TRACKING objective is pinned to LAMBDA_L1_PRE_S9 while Phase 1 keeps the
        # configured value, which separates them.  Gate unset (the default) touches
        # nothing -- byte-identical to before.
        # ⚠CORRECTION 2026-08-21: this comment used to read "pinned to the SHIPPED default
        # (configs/estimation.yaml:12, lambda_l1=0.001)".  Both halves are now false --
        # S9 deployed 1e-4 on 2026-08-11 and that yaml line reads 0.0001 -- but the PIN
        # itself is correct and deliberate: S8 only reproduces at the pre-S9 value.
        _l1_saved = None
        if os.environ.get('S8_L1_PHASE1_ONLY') == '1':
            _l1_saved = self.lambda_l1
            self.lambda_l1 = LAMBDA_L1_PRE_S9
        try:
            for step in range(self.tracking_steps):
                opt.zero_grad()
                loss = self._compute_loss(self._frame_buffer, reg=track_reg)
                loss.backward()
                opt.step()
        finally:
            if _l1_saved is not None:
                self.lambda_l1 = _l1_saved

        self.model.clamp_steepness(self.max_steepness)

        elapsed_ms = (time.time() - t0) * 1000

        return {
            'loss': loss.item(),
            'n_pts': len(frame.x_obs),
            'elapsed_ms': elapsed_ms,
        }

    def _adaptive_prune_grow(self, frames, x_range, y_range, verbose=False):
        """Adaptive N: prune weak components, grow if residual is still high.

        Pruning uses energy-based selection: keep fewest components that
        capture (1 - prune_threshold) fraction of total energy (Σ wᵢ²).
        """

        # --- Prune: energy-based selection ---
        with torch.no_grad():
            energy = self.model.w ** 2  # energy per component
            total_energy = energy.sum().item()

            if total_energy < 1e-12:
                return

            # Sort by energy descending, keep enough for target fraction
            target_fraction = 1.0 - self.prune_threshold  # e.g., keep 95%
            sort_idx = torch.argsort(energy, descending=True)
            cumulative = torch.cumsum(energy[sort_idx], dim=0)
            n_keep = int((cumulative / total_energy < target_fraction).sum().item()) + 1
            n_keep = max(1, min(n_keep, self.model.n_gaussians))

            n_before = self.model.n_gaussians
            if n_keep < n_before:
                keep_idx = sort_idx[:n_keep]
                keep_idx = keep_idx.sort()[0]

                # Safety check: compute residual before and after pruning
                # Rollback if pruning significantly increases residual
                result_before = self._compute_residuals(self.model, frames)
                res_before = result_before[2] if len(result_before) >= 3 else np.array([])
                rms_before = float(np.sqrt(np.mean(res_before ** 2))) if len(res_before) > 0 else 0.0

                # Temporarily prune
                import copy
                saved_state = {
                    'w': self.model.w.data.clone(),
                    'kx': self.model.kx.data.clone(),
                    'ky': self.model.ky.data.clone(),
                    'phi': self.model.phi.data.clone(),
                    'n': self.model.n_gaussians,
                }
                if not self.model.use_dispersion:
                    saved_state['omega_free'] = self.model.omega_free.data.clone()

                self.model.remove_gaussians(keep_idx)

                result_after = self._compute_residuals(self.model, frames)
                res_after = result_after[2] if len(result_after) >= 3 else np.array([])
                rms_after = float(np.sqrt(np.mean(res_after ** 2))) if len(res_after) > 0 else 0.0

                # Rollback if residual increases by more than 30%
                if rms_before > 1e-12 and rms_after > 1.3 * rms_before:
                    # Restore original model
                    self.model.n_gaussians = saved_state['n']
                    self.model.w = torch.nn.Parameter(saved_state['w'])
                    self.model.kx = torch.nn.Parameter(saved_state['kx'])
                    self.model.ky = torch.nn.Parameter(saved_state['ky'])
                    self.model.phi = torch.nn.Parameter(saved_state['phi'])
                    if not self.model.use_dispersion:
                        self.model.omega_free = torch.nn.Parameter(saved_state['omega_free'])
                    if verbose:
                        print(f"  [Adaptive] Pruning rolled back: residual would increase "
                              f"{rms_before:.4f} → {rms_after:.4f}")
                else:
                    if verbose:
                        frac = cumulative[n_keep-1].item() / total_energy * 100
                        print(f"  [Adaptive] Pruned: {n_before} → {self.model.n_gaussians} "
                              f"(kept {frac:.1f}% energy)")

        # --- Grow: add components if residual energy is high ---
        # NOTE (2026-08-23): this early return makes `max_n` and any growth-side intervention
        # NON-ORTHOGONAL.  Lowering max_n to the pooling cap (learned_init.pool_and_select
        # returns exactly max_n components) disables the whole growth branch, so an arm that
        # sets both `max_n=10` and SCALE_INVARIANT_GROW=True is bit-identical to `max_n=10`
        # alone (measured 12/12, tmp/B2_REAL_2026-08-23/RESULT_B2F.md).  A per-call counter on
        # this function does NOT detect that -- the prune half still runs.  Count the BRANCH.
        if self.model.n_gaussians >= self.max_n:
            return

        # Compute residual energy
        result = self._compute_residuals(self.model, frames)
        if self.init_method in ('v2', 'v2_konly', 'learned', 'oracle') and len(result) == 4:
            all_x, all_y, all_res, all_t = result
        else:
            all_x, all_y, all_res = result[:3]
            all_t = None

        if len(all_res) == 0:
            return

        residual_rms = float(np.sqrt(np.mean(all_res ** 2)))

        # Quantity actually compared against grow_residual_threshold. Absolute metres by
        # default; a dimensionless fraction of observed signal amplitude when
        # SCALE_INVARIANT_GROW is set (see the module-level note).
        grow_metric = residual_rms
        if SCALE_INVARIANT_GROW:
            obs = [f.z_obs.detach().cpu().numpy() for f in frames if len(f.z_obs) > 0]
            if obs:
                obs_rms = float(np.sqrt(np.mean(np.concatenate(obs) ** 2)))
                if obs_rms > 1e-9:
                    grow_metric = residual_rms / obs_rms

        if grow_metric < self.grow_residual_threshold:
            if verbose:
                print(f"  [Adaptive] Residual RMS={residual_rms:.4f} "
                      f"(metric={grow_metric:.4f}) < threshold "
                      f"{self.grow_residual_threshold}, no growth needed")
            return

        # Decide how many to add (proportional to residual, capped)
        n_add = min(
            max(3, int(grow_metric / self.grow_residual_threshold * 3)),
            self.max_n - self.model.n_gaussians
        )

        if n_add <= 0:
            return

        # Use beamforming on residuals to find new peaks
        if self.init_method in ('v2', 'v2_konly', 'learned', 'oracle') and all_t is not None:
            # Same non-subtracting greedy as coarse_to_fine._beamform_residuals; the
            # RX2_GROW_* research gates (initializer._rx2_gate, read at CALL time) are
            # applied at BOTH residual sites so an intervention is not diluted to one of
            # them. All gates off == shipped behaviour, bit-for-bit.
            from wavesplat.optimizer.initializer import (
                _beamform_coarse, _beamform_refine, _linear_projection,
                MIN_DK, RECOMPUTE_MIN_DK, _rx2_gate, G as _G_INIT,
            )
            _recompute = _rx2_gate('RX2_GROW_RECOMPUTE')
            _subtract = _recompute or _rx2_gate('RX2_GROW_SUBTRACT')
            min_dk = (RECOMPUTE_MIN_DK if _rx2_gate('RX2_GROW_TIGHT_SUPPRESS')
                      else MIN_DK)
            z_work = (np.array(all_res, copy=True) if _subtract else all_res)

            power, kx_axis, ky_axis = _beamform_coarse(
                all_x, all_y, z_work, all_t, self.depth,
                kx_range=(-4.0, 4.0), ky_range=(-4.0, 4.0),
                n_kx=61, n_ky=61)

            # Zero DC
            cx = np.argmin(np.abs(kx_axis))
            cy = np.argmin(np.abs(ky_axis))
            power[max(0, cy-1):cy+2, max(0, cx-1):cx+2] = 0.0

            kx_list, ky_list, amp_list, phi_list = [], [], [], []
            dkx = kx_axis[1] - kx_axis[0]
            _suppressed = []

            for _i_add in range(n_add):
                flat_idx = np.argmax(power.ravel())
                iy_p, ix_p = np.unravel_index(flat_idx, power.shape)
                if power[iy_p, ix_p] < 1e-12:
                    break
                kx_c, ky_c = kx_axis[ix_p], ky_axis[iy_p]
                kx_f, ky_f, _ = _beamform_refine(
                    all_x, all_y, z_work, all_t, self.depth,
                    kx_c, ky_c, delta=0.15, n_fine=31)
                amp, _ph_fit = _linear_projection(
                    all_x, all_y, z_work, all_t, kx_f, ky_f, self.depth)
                kx_list.append(kx_f)
                ky_list.append(ky_f)
                amp_list.append(amp)
                phi_list.append(_ph_fit)

                suppress = int(min_dk / dkx) + 1
                y_lo = max(0, iy_p - suppress)
                y_hi = min(power.shape[0], iy_p + suppress + 1)
                x_lo = max(0, ix_p - suppress)
                x_hi = min(power.shape[1], ix_p + suppress + 1)
                power[y_lo:y_hi, x_lo:x_hi] = 0.0
                _suppressed.append((y_lo, y_hi, x_lo, x_hi))

                if _subtract:
                    _kmag = np.sqrt(kx_f ** 2 + ky_f ** 2 + 1e-12)
                    _om = np.sqrt(_G_INIT * _kmag * np.tanh(_kmag * self.depth))
                    _ph = kx_f * all_x + ky_f * all_y - _om * all_t + _ph_fit
                    z_work = z_work - amp * np.cos(_ph)
                if _recompute and _i_add + 1 < n_add:
                    power, kx_axis, ky_axis = _beamform_coarse(
                        all_x, all_y, z_work, all_t, self.depth,
                        kx_range=(-4.0, 4.0), ky_range=(-4.0, 4.0),
                        n_kx=61, n_ky=61)
                    power[max(0, cy-1):cy+2, max(0, cx-1):cx+2] = 0.0
                    for (_a, _b, _c, _d) in _suppressed:
                        power[_a:_b, _c:_d] = 0.0

            n_found = len(kx_list)
        else:
            kx_list, ky_list, amp_list, phi_list = [], [], [], []
            n_found = 0

        # Pad with random if needed
        if n_found < n_add:
            extra = n_add - n_found
            kx_list.extend(np.random.randn(extra) * 0.1)
            ky_list.extend(np.random.randn(extra) * 0.1)
            amp_list.extend(np.ones(extra) * 0.01)
            phi_list.extend(np.random.uniform(0, 2 * np.pi, extra))

        n_actual = min(len(kx_list), n_add)
        if n_actual > 0:
            device = self.model.w.device
            from wavesplat.optimizer.initializer import _rx2_gate as _rx2
            if _rx2('RX2_GROW_FITPHI') and len(phi_list) >= n_actual:
                _init_phi = torch.tensor(
                    np.asarray(phi_list[:n_actual], dtype=np.float64),
                    dtype=torch.float32, device=device)
            else:
                _init_phi = torch.rand(n_actual, device=device) * 2 * np.pi
            self.model.add_gaussians(
                n_actual,
                init_kx=torch.tensor(kx_list[:n_actual], dtype=torch.float32, device=device),
                init_ky=torch.tensor(ky_list[:n_actual], dtype=torch.float32, device=device),
                init_w=torch.tensor(amp_list[:n_actual], dtype=torch.float32, device=device),
                init_phi=_init_phi,
            )

            # Fine-tune the new components with a few iterations
            self._build_optimizer()
            for _ in range(50):
                self.optimizer.zero_grad()
                loss = self._compute_loss(frames)
                loss.backward()
                self.optimizer.step()
                self.model.clamp_steepness(self.max_steepness)

            if verbose:
                print(f"  [Adaptive] Grew: +{n_actual} → N={self.model.n_gaussians} "
                      f"(residual RMS={residual_rms:.4f})")

    def _ls_w_update(self, frames: List[FrameData]):
        """Analytically update w via least-squares with current kx/ky/phi/omega.

        Builds design matrix A[row,i] = cos(kx_i*x + ky_i*y - omega_i*t + phi_i)
        and solves w = argmin ||A w - z||^2.
        """
        n = self.model.n_gaussians
        if n == 0:
            return

        with torch.no_grad():
            kx = self.model.kx.cpu().numpy()
            ky = self.model.ky.cpu().numpy()
            phi = self.model.phi.cpu().numpy()
            omega = self.model._omega().cpu().numpy()

        rows_A = []
        rows_z = []
        for frame in frames:
            if len(frame.x_obs) == 0:
                continue
            x = frame.x_obs.cpu().numpy()
            y = frame.y_obs.cpu().numpy()
            z = frame.z_obs.cpu().numpy()
            t = frame.t
            phase = (kx[None, :] * x[:, None]
                     + ky[None, :] * y[:, None]
                     - omega[None, :] * t
                     + phi[None, :])
            rows_A.append(np.cos(phase))
            rows_z.append(z)

        if not rows_A:
            return

        A = np.vstack(rows_A)
        z_vec = np.concatenate(rows_z)

        w_ls, _, _, _ = np.linalg.lstsq(A, z_vec, rcond=None)

        with torch.no_grad():
            self.model.w.copy_(torch.tensor(w_ls, dtype=torch.float32,
                                            device=self.model.w.device))

    def amplitude_refit(self, frames: List[FrameData],
                        verbose: bool = False) -> Dict:
        """
        Phase 3: Post-hoc least-squares amplitude+phase calibration.

        Freezes kx, ky and solves for optimal w and phi via linear LS
        using cos/sin decomposition:
            z = Σ [a_i cos(θ_i) + b_i sin(θ_i)]
        where θ_i = kx_i·x + ky_i·y - ω_i·t, then
            w_i = sqrt(a_i² + b_i²), phi_i = atan2(-b_i, a_i)
        The minus sign is required by the shipped forward model
        w·cos(θ+φ) = (w cosφ)·cosθ − (w sinφ)·sinθ, i.e. b = −w·sinφ.
        Same convention as model/tnse.py:216 and optimizer/initializer.py:448.

        This corrects both amplitude decay and phase drift caused by
        tracking oscillation in Phase 2.

        Args:
            frames: list of recent FrameData (typically last 5-10 frames)
            verbose: print progress

        Returns:
            dict with 'w_before', 'w_after', 'residual_before', 'residual_after'
        """
        if self._bank is not None:
            # bank-MMSE route: the fit is already the exact closed-form linear
            # solution on its window -- there are no drifting amplitudes to refit
            return {}
        if self.model is None:
            raise RuntimeError("Model not initialized")

        n = self.model.n_gaussians
        if n == 0:
            return {}

        with torch.no_grad():
            kx = self.model.kx.cpu().numpy()
            ky = self.model.ky.cpu().numpy()
            omega = self.model._omega().cpu().numpy()
            w_before = self.model.w.cpu().numpy().copy()

        # Build design matrix with cos/sin pairs:
        # A = [cos(θ_1) sin(θ_1) cos(θ_2) sin(θ_2) ... cos(θ_N) sin(θ_N)]
        # where θ_i = kx_i·x + ky_i·y - ω_i·t  (no phi — we solve for it)
        rows_A = []
        rows_z = []
        for frame in frames:
            if len(frame.x_obs) == 0:
                continue
            x = frame.x_obs.cpu().numpy()
            y = frame.y_obs.cpu().numpy()
            z = frame.z_obs.cpu().numpy()
            t = frame.t

            # base phase without phi: (M, N)
            theta = (kx[None, :] * x[:, None]
                     + ky[None, :] * y[:, None]
                     - omega[None, :] * t)
            # Interleave cos/sin columns: (M, 2N)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            A_frame = np.empty((len(x), 2 * n), dtype=np.float64)
            A_frame[:, 0::2] = cos_theta
            A_frame[:, 1::2] = sin_theta
            rows_A.append(A_frame)
            rows_z.append(z)

        if not rows_A:
            return {}

        A = np.vstack(rows_A)   # (total_obs, 2N)
        z_vec = np.concatenate(rows_z)  # (total_obs,)

        # Solve [a_1, b_1, a_2, b_2, ...] = argmin ||A c - z||^2
        coeffs, _, _, _ = np.linalg.lstsq(A, z_vec, rcond=None)

        a = coeffs[0::2]  # cos coefficients
        b = coeffs[1::2]  # sin coefficients
        w_refit = np.sqrt(a ** 2 + b ** 2)
        phi_refit = np.arctan2(-b, a)

        # Compute residuals for reporting
        # Before: use old w, phi
        phi_before = self.model.phi.detach().cpu().numpy()
        A_before = np.empty((len(z_vec), n))
        idx = 0
        for frame in frames:
            if len(frame.x_obs) == 0:
                continue
            m = len(frame.x_obs)
            x = frame.x_obs.cpu().numpy()
            y = frame.y_obs.cpu().numpy()
            theta = (kx[None, :] * x[:, None]
                     + ky[None, :] * y[:, None]
                     - omega[None, :] * frame.t
                     + phi_before[None, :])
            A_before[idx:idx + m] = np.cos(theta)
            idx += m
        residual_before = float(np.sqrt(np.mean((A_before @ w_before - z_vec) ** 2)))
        residual_after = float(np.sqrt(np.mean((A @ coeffs - z_vec) ** 2)))

        # Update model weights and phases
        with torch.no_grad():
            self.model.w.copy_(torch.tensor(w_refit, dtype=torch.float32,
                                            device=self.model.w.device))
            self.model.phi.copy_(torch.tensor(phi_refit, dtype=torch.float32,
                                              device=self.model.phi.device))

        if verbose:
            print(f"  [Refit] LS refit (w+phi): "
                  f"residual {residual_before:.4f} → {residual_after:.4f}")
            print(f"  [Refit] |w| mean: {np.mean(np.abs(w_before)):.4f} → "
                  f"{np.mean(np.abs(w_refit)):.4f}")

        return {
            'w_before': w_before.tolist(),
            'w_after': w_refit.tolist(),
            'residual_before': residual_before,
            'residual_after': residual_after,
        }

    def predict_field(self, grid_x: torch.Tensor, grid_y: torch.Tensor,
                      t: float) -> torch.Tensor:
        """Predict wave field on regular grid at time t.

        Temporarily enables neural corrector (if loaded) for prediction only.
        """
        if self._bank is not None:
            return self._bank_predict_grid(grid_x, grid_y, t)
        if self.model is None:
            raise RuntimeError("Model not initialized")
        with torch.no_grad():
            # Enable NN corrector for prediction if loaded
            had_nn = self.model.use_neural_corrector
            if self.neural_corrector_path and self.model._neural_corrector is not None:
                self.model.use_neural_corrector = True
            result = self.model.predict_grid(grid_x, grid_y, t)
            self.model.use_neural_corrector = had_nn
            return result


# ─────────────────────────────────────────────────────────────────────────────
# Median-of-k analytic field ensemble  (deployed 2026-07-27)
# ─────────────────────────────────────────────────────────────────────────────

def _rng_snapshot():
    return (torch.get_rng_state(), np.random.get_state())


def _rng_restore(state):
    torch.set_rng_state(state[0])
    np.random.set_state(state[1])


def _seed_globals(seed: int):
    """Same global seeding run.set_seed does; inlined to avoid a circular import."""
    torch.manual_seed(seed)
    np.random.seed(seed)


class MedianEnsembleUpdater:
    """`auto` router + median-of-k ensemble on the ANALYTIC fallback.

    WHY. The analytic arm initialises component phases at random in two places, so a
    fixed (episode, config) still gives a *bimodal* run-to-run spread, CV up to 53%
    (tmp/ESTIMATOR_VARIANCE_RESULT_2026-07-23.md). A few percent of sparse lanes land in
    the bad mode and blow up 5-10x. The documented prescription is median-of-k, and the
    confirmation round applied it at the level of the predicted FIELD (not of the scalar
    score), which makes it deployable: no ground truth is used to combine members. On the
    fresh reserve-seed 40 it removed every sparse-lane violation (>0.10: 0/80, q95 0.0552)
    while harming no slice (max deviation +0.0025), and it demonstrably fired -- 28 lanes
    with member disagreement >0.05, 8 genuine repairs
    (tmp/confirm2_2026-07-27/RESULT_CONFIRM2.md §2).

    SCOPE -- deliberately narrow, three exclusions:
      1. Only the ANALYTIC fallback of `init_method='auto'`. An explicit
         `init_method='v2_konly'` (what the paper scripts pass) never reaches this class
         and stays bit-identical -- reproducibility of published numbers is protected.
      2. If the router picks `learned`, the ensemble is NOT built (one member): the
         authorised configuration ensembles the analytic arm only.
      3. If the real-domain gate fires, the ensemble is NOT built (spec: the real path is
         outside the authorisation).

    Members are the standard estimator run with est seeds {s, s+1000, s+2000}; the output
    of `predict_field` is the pointwise median across members. For odd k the median IS one
    of the member values, so k=1 is bit-identical to a single SequentialUpdater run.
    Latency is additive and was measured: 6.5 ms/frame per member -> 19.7 ms for k=3,
    inside the 100 ms budget (ibid.); initialisation is 3x and parallelisable.

    Each member owns its RNG state (snapshot/restore around every call), so interleaving
    the members frame-by-frame is equivalent to running them one after another.

    The public surface mirrors SequentialUpdater (`initialize`, `update`, `predict_field`,
    `amplitude_refit`, `.model`); unknown attributes are forwarded to member 0.
    """

    MEMBER_OFFSETS = (0, 1000, 2000, 3000, 4000)   # frozen arm = first 3

    def __init__(self, config: dict):
        self.config = config
        # ★Which knob applies is decided by the requested route, not by the router's
        #   later decision: `learned` reads `learned_ensemble_k`, everything else reads
        #   `analytic_ensemble_k`. Keeping them as two named constants (rather than one
        #   merged 'ensemble_k') is deliberate -- the repo has been bitten by merging two
        #   constants whose deployment histories differ.
        if config.get('init_method') == 'learned':
            self.k_key = 'learned_ensemble_k'
            k = int(config.get('learned_ensemble_k', DEFAULT_LEARNED_ENSEMBLE_K))
        else:
            self.k_key = 'analytic_ensemble_k'
            k = int(config.get('analytic_ensemble_k', DEFAULT_ANALYTIC_ENSEMBLE_K))
        if k < 1 or k > len(self.MEMBER_OFFSETS):
            raise ValueError(f'{self.k_key} must be 1..{len(self.MEMBER_OFFSETS)}')
        self.k_requested = k
        self._primary = SequentialUpdater(config)
        self._members = [self._primary]
        self._rng = [None]
        self.member_seeds = [config.get('seed')]
        # populated by initialize(): how many members were actually built
        self.ensemble_k = 1

    # ── attribute forwarding ────────────────────────────────────────────────
    def __getattr__(self, name):
        # Only reached for attributes NOT found on the instance/class.
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            primary = self.__dict__['_primary']
        except KeyError:
            raise AttributeError(name)
        return getattr(primary, name)

    @property
    def model(self):
        return self._primary.model

    @property
    def members(self):
        return list(self._members)

    ENSEMBLED_ROUTES = ('v2_konly', 'learned')

    def _effective_k(self, route: str, real_gate: bool) -> int:
        """해소된 route 로 멤버 수를 **다시 읽는다**. 반환 1 이면 앙상블 없음.

        ⛔**두 노브는 끝까지 섞이지 않는다.** `__init__` 은 요청된 `init_method` 로
        노브를 골랐지만, `auto` 는 그 시점에 어디로 갈지 모른다 --- 라우터가 `learned` 로
        해소하면 `analytic_ensemble_k` 를 물려받아서는 안 된다.

        ★초판(2026-09-01)이 정확히 그 결함을 냈다: 출하 기본 구성(`auto` + 임계 0 ⟹ 항상
        `learned`)이 `analytic_ensemble_k=3` 을 그대로 써서 **배포 기본값이 조용히 학습
        멤버 3개를 만들었다.** 결정 `D3` 가 명시적으로 하지 않기로 한 것이고, 토론의 반대
        측이 경고한 「조용한 재라우팅」 그 자체다. ⟹ 이 함수가 그 경계다.

        실측 도메인 게이트는 두 앙상블을 모두 누른다 --- 게이트는 bank-MMSE 로 보내고
        그쪽이 자체 장 앙상블(`DEFAULT_REAL_BANK_ENSEMBLE_K`)을 돌린다.
        """
        if real_gate or route not in self.ENSEMBLED_ROUTES:
            return 1
        # ⛔★결정을 여기로 모았으면 **검증도 여기 있어야 한다**(독립 감사 지적).
        #   `__init__` 은 *요청된* init_method 로 고른 노브 하나만 검증한다 ⟹ `auto` 로
        #   들어오면 `learned_ensemble_k` 는 **어디서도 검증되지 않았고**, `k=99` 가
        #   조용히 5 로 잘렸다(요청 99 · 실제 5 · 경고 0). 「같은 기능 두 곳」이 절반만
        #   닫힌 상태였다.
        if route == 'learned':
            k = self.config.get('learned_ensemble_k', DEFAULT_LEARNED_ENSEMBLE_K)
            name = 'learned_ensemble_k'
        else:
            k = self.config.get('analytic_ensemble_k', DEFAULT_ANALYTIC_ENSEMBLE_K)
            name = 'analytic_ensemble_k'
        try:
            k = int(k)
        except (TypeError, ValueError):
            raise ValueError(f'{name} must be an int 1..{len(self.MEMBER_OFFSETS)}, '
                             f'got {k!r}')
        if k < 1 or k > len(self.MEMBER_OFFSETS):
            raise ValueError(f'{name} must be 1..{len(self.MEMBER_OFFSETS)}, got {k}')
        return k

    # ── phase 1 ─────────────────────────────────────────────────────────────
    def initialize(self, frames: List[FrameData],
                   x_range=(10.0, 250.0), y_range=(-100.0, 100.0),
                   verbose: bool = False) -> Dict:
        # Member 0 runs first and unchanged: it carries the router decision and the
        # ambient RNG stream, so a single-member outcome is bit-identical to the
        # non-ensemble path.
        result = self._primary.initialize(frames, x_range, y_range, verbose)
        self._rng[0] = _rng_snapshot()

        route = self._primary.init_method
        real_gate = bool(getattr(self._primary, '_auto_real_gate', False))
        # ★Two ensembled routes, each with its own knob (see DEFAULT_LEARNED_ENSEMBLE_K):
        #   · `v2_konly`  -- the analytic fallback of the `auto` router (deployed 2026-07-27)
        #   · `learned`   -- the learned prior (path added 2026-09-01, **default off**)
        #   The real-domain gate still suppresses both: it routes to bank-MMSE, which runs
        #   its own internal field ensemble (`DEFAULT_REAL_BANK_ENSEMBLE_K`).
        #
        # ⛔★**초판(2026-09-01) 결함 — 배포 기본값이 조용히 앙상블을 얻었다.**
        #   `init_method='auto'` 는 __init__ 시점에 `analytic_ensemble_k`(=3) 를 읽는데,
        #   라우터가 `learned` 로 해소되면 초판 조건절이 그 3 을 그대로 써서 **출하 기본
        #   구성이 학습 멤버 3개를 만들었다**. 그것은 D3 가 명시적으로 하지 않기로 한 것이고,
        #   반대 측이 경고한 「조용한 재라우팅」 그 자체다.
        #   ⟹ **k 는 해소된 route 로 다시 읽는다.** 두 노브는 끝까지 섞이지 않는다.
        k_eff = self._effective_k(route, real_gate)
        self.k_effective = k_eff
        if k_eff <= 1:
            self.ensemble_k = 1
            result['ensemble_k'] = 1
            # model is None when the real-domain gate routed to bank-MMSE (which has
            # no Gaussian components and runs its own internal field ensemble)
            result['member_n'] = [int(self._primary.model.n_gaussians)
                                  if self._primary.model is not None else 0]
            result['member_elapsed_s'] = [float(result['elapsed_s'])]
            return result

        base = self.config.get('seed')
        base = 0 if base is None else int(base)
        elapsed = [float(result['elapsed_s'])]
        for off in self.MEMBER_OFFSETS[1:k_eff]:
            s = base + off
            cfg = dict(self.config)
            if route == 'learned':
                # ★Learned members are split by the diffusion draw **and by the global
                #   torch stream that draw reseeds** -- `scripts/d3_beamform_cond_diffusion.py`
                #   calls `torch.manual_seed(seed)` inside `sample()`, so every later torch
                #   draw (growth phase, ...) differs too. Only the numpy stream is shared.
                #   ⛔초판 주석의 *"ONLY in the diffusion draw"* 는 거짓이었다(독립 감사).
                #   ⚠팔 자체는 `T3`/`T5` 하네스 규약과 같으므로 **판정은 무효가 아니다** —
                #   틀린 것은 서술이다.
                #   `seed` stays at
                #   `base` (so the ambient RNG stream is the same for every member) and
                #   `learned_diffusion_seed` carries the offset -- this is the convention
                #   the confirmations were measured under (`T3 gates_t3.member_overrides`,
                #   `T5-ENS`), so member 0 is bit-identical to the k=1 path.
                # ⚠`learned_atoms` must not be inherited or every member reuses one draw.
                cfg['init_method'] = 'learned'
                cfg['seed'] = base
                cfg['learned_diffusion_seed'] = s
                cfg.pop('learned_atoms', None)
                _seed_globals(base)
            else:
                cfg['init_method'] = 'v2_konly'  # explicit: no second router call
                cfg['seed'] = s
                _seed_globals(s)
            m = SequentialUpdater(cfg)
            r = m.initialize(frames, x_range, y_range, verbose=False)
            self._members.append(m)
            self._rng.append(_rng_snapshot())
            self.member_seeds.append(s)
            elapsed.append(float(r['elapsed_s']))

        self.ensemble_k = len(self._members)
        result['ensemble_k'] = self.ensemble_k
        result['member_n'] = [int(m.model.n_gaussians) for m in self._members]
        result['member_elapsed_s'] = elapsed
        result['elapsed_s'] = float(sum(elapsed))   # honest total initialisation cost
        if verbose:
            print(f"  [Init/ensemble] median-of-{self.ensemble_k} {route} members, "
                  f"seeds={self.member_seeds}, N={result['member_n']}, "
                  f"{result['elapsed_s']:.1f}s total")
        return result

    # ── phase 2 ─────────────────────────────────────────────────────────────
    def update(self, frame: FrameData) -> Dict:
        if len(self._members) == 1:
            return self._primary.update(frame)

        outs = []
        for i, m in enumerate(self._members):
            _rng_restore(self._rng[i])
            outs.append(m.update(frame))
            self._rng[i] = _rng_snapshot()

        losses = [o['loss'] for o in outs]
        finite = [v for v in losses if np.isfinite(v)]
        return {
            # median of the member observation-fit losses: the ensemble's GT-free quality
            # signal (see RESULT_CONFIRM2 §3 R7 -- selecting on it is a separate, not-yet
            # pre-registered idea; reporting it is not).
            'loss': float(np.median(finite)) if finite else float('nan'),
            'n_pts': outs[0]['n_pts'],
            # additive by construction: members run sequentially in one deployment thread
            'elapsed_ms': float(sum(o['elapsed_ms'] for o in outs)),
            'member_loss': losses,
            'member_elapsed_ms': [o['elapsed_ms'] for o in outs],
        }

    def amplitude_refit(self, frames: List[FrameData], verbose: bool = False):
        out = self._primary.amplitude_refit(frames, verbose=verbose)
        for i, m in enumerate(self._members[1:], start=1):
            _rng_restore(self._rng[i])
            m.amplitude_refit(frames, verbose=False)
            self._rng[i] = _rng_snapshot()
        return out

    # ── prediction ──────────────────────────────────────────────────────────
    def predict_field(self, grid_x: torch.Tensor, grid_y: torch.Tensor,
                      t: float) -> torch.Tensor:
        if len(self._members) == 1:
            return self._primary.predict_field(grid_x, grid_y, t)
        preds = [m.predict_field(grid_x, grid_y, t) for m in self._members]
        # Pointwise (element-wise) median. k is odd, so the result is one of the member
        # values at every point -- no interpolation, no dtype change.
        return torch.median(torch.stack(preds, dim=0), dim=0).values


def make_updater(config: dict):
    """Build the estimator the deployed policy calls for.

    `init_method='auto'` (the shipped default) gets the median-of-k analytic ensemble
    wrapper; every explicit init_method -- in particular the `v2_konly` the paper scripts
    pass -- gets the plain SequentialUpdater and is therefore bit-identical to before.
    """
    if config.get('init_method') == 'auto' and \
            int(config.get('analytic_ensemble_k', DEFAULT_ANALYTIC_ENSEMBLE_K)) > 1:
        return MedianEnsembleUpdater(config)
    # ★`learned` is wrapped ONLY when the caller asks for it explicitly
    #   (`DEFAULT_LEARNED_ENSEMBLE_K = 1`), so `make_updater({'init_method': 'learned'})`
    #   stays a plain SequentialUpdater and every recorded batch is unchanged.
    if config.get('init_method') == 'learned' and \
            int(config.get('learned_ensemble_k', DEFAULT_LEARNED_ENSEMBLE_K)) > 1:
        return MedianEnsembleUpdater(config)
    return SequentialUpdater(config)
