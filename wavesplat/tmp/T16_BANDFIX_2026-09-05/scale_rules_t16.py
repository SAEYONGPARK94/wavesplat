"""`T16-BANDFIX` **스케일 규칙** — 측정면. ⛔`PREREG §1` · `REACHABILITY §3` 가 정본.

★이 배치가 새로 등록하는 규칙은 **하나뿐**이다: `s_bandr` — 2단 빔포머의 **1단 격자를
절대 상수 `2.0` 에서 관측 개구 유래 `0.5·(N1−1)·2π/L_obs` 로** 바꾼 것.
그러면 `Δk₁ = 2π/L_obs`(Rayleigh 한계)가 되어 규칙이 재라벨에 대해 **등변**이 된다.

⛔`AUTO-AP`·`AUTO-BAND` 는 **손복사하지 않는다** — `scale_rules_t14` 에서 임포트하고
`G-RULEID` 가 `SR16.s_ap is SR14.s_ap` … 를 **객체 동일성**으로 단언한다.

⛔이 모듈은 관측 5배열 `(x, y, t, z, depth)` 밖의 것을 **받지도 보지도 않는다**
(`G-S-SRC` 정적 AST · `G-S-BLIND` 별도 프로세스 재실행).
⛔결과를 본 뒤 `K_TRAIN`·`L_TRAIN`·`N1`·`ZOOM`·레일을 바꾸는 것은 금지 — 새 규칙은 새 사전등록으로만.
"""
import numpy as np

from scale_rules_t14 import (S_MAX, S_MIN, K_TRAIN, L_TRAIN,      # noqa: F401
                             s_ap, s_ap_raw, s_band, s_band_raw)

N1 = 120                  # `bank_mmse.py:160`·`:163` 의 `n=120`
ZOOM = 3.5                # `bank_mmse.py:163` 의 `3.5 * kp`
FLOOR = 1e-3              # `bank_mmse.py:161`·`:164` 의 `max(·, 1e-3)`
TWO_PI = 2.0 * np.pi


def clip_s(value):
    """안전 레일 `[S_MIN, S_MAX]` — ★**규칙 모듈이 단일출처로 낸다**.

    ⛔러너가 레일을 재구현하면 손복사가 된다. `s_ap`·`s_band`·`s_bandr` 의 내부 표현과
    **비트 동일**한 식이다(`float(min(max(raw, S_MIN), S_MAX))`).
    """
    return float(min(max(float(value), S_MIN), S_MAX))


def klim1_rel(x, y):
    """★새 규칙의 1단 반경 — **관측 개구에서 유도**한다.

    `linspace(-klim, klim, N1)` 의 간격이 `2·klim/(N1−1)` 이므로
    `klim = 0.5·(N1−1)·2π/L_obs` ⟹ `Δk₁ = 2π/L_obs` = Rayleigh 한계.
    `L_obs` 는 `SR14.s_ap_raw` 가 쓰는 것과 **같은 양**이다(`max(ptp x, ptp y)`).
    """
    lobs = max(float(np.ptp(np.asarray(x))), float(np.ptp(np.asarray(y))))
    return 0.5 * (N1 - 1) * (TWO_PI / lobs)


def kp_hat_rel(x, y, t, z, depth):
    """2단 빔포머 줌 — ⛔`SR14.kp_hat` 과 **1단 반경만** 다르고 나머지는 축자 동일."""
    from wavesplat.optimizer.bank_mmse import _beamform_top
    P = np.column_stack([x, y, t]).astype(np.float32)
    r = np.asarray(z, dtype=np.float32)
    bx, by = _beamform_top(P, r, depth, klim=klim1_rel(x, y), n=N1)
    kp = max(float(np.hypot(bx, by)), FLOOR)
    bx, by = _beamform_top(P, r, depth, klim=ZOOM * kp, n=N1)
    return max(float(np.hypot(bx, by)), FLOOR)


def s_bandr_raw(x, y, t, z, depth):
    """규칙 ③ **비클리핑** — `K_TRAIN / k̂_p^REL`. ★이 배치의 처치."""
    return float(K_TRAIN / kp_hat_rel(x, y, t, z, depth))


def s_bandr(x, y, t, z, depth):
    """규칙 ③ + 안전 레일."""
    return clip_s(s_bandr_raw(x, y, t, z, depth))


RULES = {'AUTO-AP': s_ap, 'AUTO-BAND': s_band, 'AUTO-BANDR': s_bandr}
# ★`G-EQUIVAR` 는 **레일이 아니라 규칙**을 시험한다 — 클리핑은 등변성을 깬다
#   (`s_ap` 2.81 을 c=0.5 로 재라벨하면 5.62 가 되어 `S_MAX=4.0` 에 걸린다).
#   ⛔그리고 `G-EQUIVAR-RAW` 가 바로 그 사실을 **행동 증거**로 쓴다.
RULES_RAW = {'AUTO-AP': s_ap_raw, 'AUTO-BAND': s_band_raw, 'AUTO-BANDR': s_bandr_raw}
