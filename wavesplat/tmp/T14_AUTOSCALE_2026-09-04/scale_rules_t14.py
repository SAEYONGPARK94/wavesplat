"""`T14-AUTOSCALE` **스케일 규칙** — ⛔`SCALE_ESTIMATORS.md §2` 가 정본.

★이 모듈은 **관측 5배열만** 받는다: `(x, y, t, z, depth)`.
⛔참값(`target`·`eta`·`Ew`·`sigma4`)·메타(`hs`·`angle`·`block`)를 **받지도 보지도 않는다**.
`G-S-SRC`(정적) 와 `G-S-BLIND`(별도 프로세스 재실행) 가 그것을 기계로 확인한다.
★규칙은 **결과 0행 시점에 동결**됐다 — 결과를 본 뒤 `K_TRAIN` 의 중심 통계나 `L_TRAIN` 의
정의를 바꾸는 것은 금지. 새 규칙은 **새 사전등록으로만**.
"""
import numpy as np

S_MIN, S_MAX = 0.25, 4.0
L_TRAIN = 220.55      # 학습 풀 최대 변 [m] — pool_xy() 재생성으로 대조(G-LTRAIN)
K_TRAIN = 0.06779     # 사전 학습 첨두 파수의 **기하평균** [rad/m] — 재생성 대조(G-KTRAIN)


def kp_hat(x, y, t, z, depth):
    """2단 빔포머 줌 — `bank_mmse.py:158-165` 절차와 축자 동일. ★관측만 본다."""
    from wavesplat.optimizer.bank_mmse import _beamform_top
    P = np.column_stack([x, y, t]).astype(np.float32)
    r = np.asarray(z, dtype=np.float32)
    bx, by = _beamform_top(P, r, depth, klim=2.0, n=120)
    kp = max(float(np.hypot(bx, by)), 1e-3)
    bx, by = _beamform_top(P, r, depth, klim=3.5 * kp, n=120)
    return max(float(np.hypot(bx, by)), 1e-3)


def s_ap_raw(x, y, t, z, depth):
    """규칙 ① **비클리핑** — 개구/픽셀 비를 학습 범위로: `L_obs / L_TRAIN`."""
    L = max(float(np.ptp(np.asarray(x))), float(np.ptp(np.asarray(y))))
    return float(L / L_TRAIN)


def s_band_raw(x, y, t, z, depth):
    """규칙 ② **비클리핑** — 첨두 파수를 사전 밴드로: `K_TRAIN / k̂_p`."""
    return float(K_TRAIN / kp_hat(x, y, t, z, depth))


def s_ap(x, y, t, z, depth):
    """규칙 ① + 안전 레일."""
    return float(min(max(s_ap_raw(x, y, t, z, depth), S_MIN), S_MAX))


def s_band(x, y, t, z, depth):
    """규칙 ② + 안전 레일."""
    return float(min(max(s_band_raw(x, y, t, z, depth), S_MIN), S_MAX))


RULES = {'AUTO-AP': s_ap, 'AUTO-BAND': s_band}
# ★`G-EQUIVAR` 는 **레일이 아니라 규칙**을 시험해야 한다 — 클리핑은 등변성을 깬다
#   (`s_ap` 2.81 을 c=0.5 로 재라벨하면 5.62 가 되어 `S_MAX=4.0` 에 걸린다).
RULES_RAW = {'AUTO-AP': s_ap_raw, 'AUTO-BAND': s_band_raw}
