"""Regression tests that pin the SHIPPING defaults.

Why this file exists
--------------------
2026-08-13 audit finding (BLOCKER): the three deployment knobs could all be
reverted at once -- ``lambda_l1`` 1e-4 -> 1e-3, ``init_method`` auto ->
v2_konly, ``tracking_mode`` w_phi -> all -- and the whole suite still reported
``111 passed``.  No test referenced ``DEFAULT_CONFIG`` or ``configs/`` at all.

That is the exact failure class that invalidated the S7 and S8 re-runs: a
default moved underneath batch runners whose baseline arms did not pin it, so
the baseline silently received the treatment value.  The defence is cheap --
assert the deployed values, and assert that the places which are supposed to
mirror one definition actually agree.

If a test here fails you are changing a deployed default.  That is allowed, but
it must be a deliberate act: update the value here, in
``configs/estimation.yaml``, and in the auto-memory deployment record together,
and check that every batch runner's BASELINE arm names the old value
explicitly.
"""

from pathlib import Path

import os
import pytest
import yaml

from wavesplat.optimizer.sequential_update import (
    INIT_METHOD_CLASS_FALLBACK,
    INIT_METHOD_DEFAULT,
    LAMBDA_L1_DEFAULT,
    LAMBDA_L1_PRE_S9,
    SequentialUpdater,
)
from wavesplat.run import DEFAULT_CONFIG, load_yaml_config


REPO_ROOT = Path(__file__).resolve().parents[2]
ESTIMATION_YAML = REPO_ROOT / "configs" / "estimation.yaml"

# The deployed operating point.  Each entry carries the date and the batch that
# authorised it so a future reader can find the evidence.
SHIPPING_DEFAULTS = {
    # 2026-08-11, S9 (verdict BOTH); single-sourced in commit 5f0846c.
    "lambda_l1": 1e-4,
    # 2026-07-26 flip (user decision): GT-free regime router.
    "init_method": "auto",
    # kx/ky frozen during tracking; only w and phi are updated.
    "tracking_mode": "w_phi",
    # 2026-07-27 confirm2: median-of-3 on the analytic fallback.
    # ⚠2026-08-22: 임계 0 통일로 라우터가 해석적을 안 고르므로 이 앙상블은 합성에서
    #   구조적으로 죽었다. 값은 유지하나 **작동하지 않는다**(검수 V1/V3 실측 ensemble_k=1).
    "analytic_ensemble_k": 3,
    # ── 2026-08-22 배포 변경 셋 (배포값 적는 자리는 여기 하나여야 한다) ──
    "varpro_steps": 400,
    "learned_init_iters": 0,
    "auto_switch_thresh": 0,
    # ── 2026-08-23: 실측 전용 경로. 5.0 으로 배포했다가 **같은 날 철회**했다 ──
    # 정확도 이득은 측정됐으나(참 제로필드 28/36 p=6.0e-04) 재적합이 `update()` 안에서
    # **동기**로 돌아 프레임 간격의 **약 20배**(1.35–2.53 s) 블로킹한다.  근거로 쓴 예산 판정은
    # **재적합이 0회인 구성**에서 나온 것이었다.  ⟹ 기본값은 **동결(None)**.
    # 권고는 `sequential_update.recommended_real_refit_config()` 로 남아 있고,
    # 선행 조건(비동기 재적합)은 `bank_refit_async=True` 로 구현돼 있다(기본 False).
    # 상세: tmp/B6_BUDGET_2026-08-23/REVERT_2026-08-23.md
    "bank_refit_every_s": None,
    "bank_refit_async": False,
}

# ★[2026-09-02] `learned_ensemble_k` 는 위 dict 에 **넣을 수 없다** --- 저 dict 는
#   `run.py DEFAULT_CONFIG` 의 핀이고, 이 값은 거기 없다.  코드 감사가 그것을 결함으로
#   지목했다(2026-09-02, D2): 9월 1일에 배포 상수를 만들면서 **회귀 보호를 빠뜨렸다**.
#   ⟹ 아래에서 **모듈 상수 자체**를 핀한다(설정 표면은 건드리지 않는다 --- 표면을 바꾸면
#   CLI/YAML 화이트리스트와 `dict(DEFAULT_CONFIG)` 스냅샷 규약이 같이 바뀌고,
#   그것은 사전등록 없이 일요일에 할 일이 아니다).
#   ⚠남는 비대칭은 기록해 둔다: `analytic_ensemble_k` 는 `DEFAULT_CONFIG` 에 있고
#   이것은 없다.  런의 실효 k 는 **객체 되읽기**로만 확인된다(`S-ENS` 의 `G-ENSK`).
LEARNED_ENSEMBLE_K_SHIPPED = 1


def test_learned_ensemble_k_ships_off():
    """⛔1 에서 올리려면 별도 사전등록이 필요하다.

    `S-ENS/PREREG.md` §5 가 결과 **전에** 남은 조건 셋을 고정했다:
    희소 층 `UNDECIDED` · 단가 2.6배 · ★**학습 경로 프레임 예산이 한 번도 측정된 적 없다**.
    `S-ENS` 는 그중 **필요조건 하나**만 충족했다.
    """
    from wavesplat.optimizer.sequential_update import DEFAULT_LEARNED_ENSEMBLE_K
    assert DEFAULT_LEARNED_ENSEMBLE_K == LEARNED_ENSEMBLE_K_SHIPPED, (
        "학습 경로 앙상블 기본값이 바뀌었다.  이것은 배포 동작 변경이다 --- "
        "별도 사전등록 배치 없이 바꾸지 마라(S-ENS PREREG §5)."
    )


def test_learned_ensemble_k_is_pinned_in_default_config():
    """★[2026-09-02] **결정 `N-2`(페르소나 토론)로 비대칭을 닫았다.**

    종전 시험은 이 키가 `DEFAULT_CONFIG` 에 **없음**을 고정했고, 그 독스트링이 축자로
    적었다 --- *"넣는 것 자체가 나쁘다는 뜻이 아니라 **그것이 결정이라는 뜻**이다;
    결정을 하려면 이 시험을 먼저 지워야 한다."* 그 결정이 내려졌다.

    ⛔**바꾼 것은 가시성이지 값이 아니다.** 반대 측 논거가 실측으로 보였다:
    이 노브가 설정 표면에도 배치 트립와이어에도 없어서 **기본값이 바뀌어도 기계가
    탐지하지 못하고**, `dict(DEFAULT_CONFIG)` 스냅샷 러너가 그것을 조용히 흡수한다
    (저장소가 λ 에서 이미 치른 대가). 이제 값이 표면에 있으므로 드리프트가 보인다.
    """
    assert DEFAULT_CONFIG.get("learned_ensemble_k") == LEARNED_ENSEMBLE_K_SHIPPED, (
        "설정 표면의 learned_ensemble_k 가 바뀌었다. 3 으로 올리는 것은 배포 동작 "
        "변경이며 별도 사전등록이 필요하다(S-ENS PREREG §5 · T9 RESULT §8)."
    )


def test_learned_ensemble_k_two_sources_agree():
    """★두 출처가 갈리면 그 자체가 결함이다 --- 저장소의 반복 부류(λ 3중 기재)."""
    from wavesplat.optimizer.sequential_update import DEFAULT_LEARNED_ENSEMBLE_K
    assert DEFAULT_CONFIG.get("learned_ensemble_k") == DEFAULT_LEARNED_ENSEMBLE_K, (
        "run.py DEFAULT_CONFIG 와 sequential_update 의 모듈 상수가 갈렸다."
    )
