"""
raydrop_uniform.py
Model A: Distance/Fresnel-based diffuse dropout.

Physics (Dropout A):
  Each hit beam independently returns with probability:
    P_i = η · R_fresnel(θ_i) · cos(θ_i) / (d_i/d_ref)²

  η is chosen so that E[N_returns] = target_n_pts (calibration-free).
  Each beam is sampled independently (Bernoulli), so the actual return
  count varies frame-to-frame — matching real LiDAR behaviour on water.

  Physical interpretation:
    R_fresnel(θ_i)  — reflectance of the water surface at 1550nm
    cos(θ_i)        — Lambertian projected-area factor
    1/d²            — inverse-square radiometric falloff

Reference: Hecht (2017) Optics, §4.6; Fresnel equations.

⛔★**2026-08-24 — 이 모형의 입사각 의존성은 학위논문 제2장의 서술과 반대다.**
   저장소 함수로 직접 계산한 `R_fresnel(θ)·cos θ` (법선 대비 배율):
       0° 1.00 · 40° 0.93 · 60° 1.50 · 70° 2.33 · **80° 3.12** · 85° 2.64
   ⟹ **입사각 80° 가 법선보다 3.12배 더 잘 반환한다.**
   그런데 `thesis/thesis_main.tex:913-916` 은 이렇게 적는다(축자):
       *"고반사율에서 입사광의 대부분이 정반사(specular)되어 LiDAR 수신기 방향으로
         반환되지 않으므로, 유효 관측점이 소실(raydrop)된다."*
   두 문장은 같이 참일 수 없다. **물리적 사유**: 단상 LiDAR 가 받는 것은 *정반사 방향의 세기*가
   아니라 **후방산란**이다. 준정반사면에서 후방산란은 법선 근처에서 강하고 스침각에서 무너진다.
   `R_fresnel` 은 **정반사 방향으로 나가는 비율**이라 후방산란의 대리 변수로 쓰면 부호가 뒤집힌다.
   ⟹ ★**물리적으로 옳은 구현은 이미 저장소에 있다**: `raydrop_slope.py`
      (반사 광선이 센서로 되돌아오는 기하 조건 = 정반사 후방산란).

   ⚠**그럼에도 이 파일을 고치지 않는다.** 기록된 코퍼스 전체(`B-1`·`B-11`·`B-12` 포함)가
   이 모형 위에서 생성됐고, 지금 바꾸면 **기록과 코드가 어긋난다.**
   대신 **개입으로 잰다** — `tmp/B15_RAYDROP_2026-08-24/` 가 같은 파랑장 20짝에서
   `uniform` ↔ `slope` 를 바꿔 헤드라인이 견디는지 판정한다.
   ⛔**그 결과가 무엇이든 이 불일치 자체는 논문에서 수정해야 한다** — *"모형이 틀린 것"* 과
   *"결론이 바뀌는 것"* 은 다른 문제다.
   출처: 2026-08-24 외부 심사(해양공학) `F-7`, `tmp/DEBATE_PHD_2026-08-24/B2_external_ocean.md`.
"""

import numpy as np


N_WATER_1550NM = 1.321  # refractive index of water at 1550nm (Seyond Falcon K1)


def fresnel_reflectance(theta_i: np.ndarray, n: float = N_WATER_1550NM) -> np.ndarray:
    """
    Fresnel reflectance for unpolarized light (average of s and p polarizations).
    θ_i: incidence angle [radians]  (0 = normal incidence)

    Returns R in [0, 1].
    """
    cos_i = np.cos(theta_i)
    sin_i = np.sin(theta_i)
    sin_t_sq = np.clip((sin_i / n)**2, 0.0, 1.0)
    cos_t = np.sqrt(1.0 - sin_t_sq)

    Rs = ((cos_i - n * cos_t) / (cos_i + n * cos_t))**2
    Rp = ((n * cos_i - cos_t) / (n * cos_i + cos_t))**2

    return np.clip(0.5 * (Rs + Rp), 0.0, 1.0)


def return_probability(incidence_angles: np.ndarray,
                        distances: np.ndarray,
                        eta: float = 1.0,
                        d_ref: float = 125.0) -> np.ndarray:
    """
    Unnormalized return likelihood for each beam.

    P_raw = R_fresnel(θ_i) · cos(θ_i) / (d/d_ref)²

    Args:
        incidence_angles: (N,) [radians]
        distances:        (N,) [m]
        eta:              multiplicative scale (set externally for calibration)
        d_ref:            reference distance [m] — Falcon K1: 125m (half of 250m spec)

    Returns:
        P: (N,) values clipped to [0, 1]
    """
    R = fresnel_reflectance(incidence_angles)
    cos_i = np.cos(incidence_angles)
    d_norm = distances / d_ref
    P = eta * R * cos_i / (d_norm**2 + 1e-9)
    return np.clip(P, 0.0, 1.0)


def apply_dropout(hit_mask: np.ndarray,
                   hit_points: np.ndarray,
                   hit_dist: np.ndarray,
                   incidence_angles: np.ndarray,
                   target_n_pts: int,
                   rng: np.random.Generator,
                   eta: float = 1.0,
                   d_ref: float = 125.0) -> np.ndarray:
    """
    Apply Fresnel-based dropout via independent Bernoulli sampling.

    Strategy:
      1. Compute P_raw_i = R_fresnel(θ_i)·cos(θ_i) / (d_i/d_ref)² for every hit.
      2. Set η so that Σ P_raw_i · η ≈ target_n_pts  (expected count calibration).
      3. Clip P_i = clip(η·P_raw_i, 0, 1).
      4. Each beam independently returns with probability P_i  (Bernoulli draw).

    The actual return count N ~ Binomial and varies naturally frame-to-frame,
    reflecting the changing wave surface geometry.  target_n_pts is the *expected*
    number of returns, not a hard quota.

    Args:
        hit_mask:         (N_rays,) bool — geometric hit mask
        hit_points:       (N_rays, 3)
        hit_dist:         (N_rays,)
        incidence_angles: (N_rays,)
        target_n_pts:     expected number of returns per frame
        rng:              numpy random Generator
        eta, d_ref:       Fresnel model parameters

    Returns:
        return_mask: (N_rays,) bool — final return mask (variable count per call)
    """
    return_mask = np.zeros(len(hit_mask), dtype=bool)
    if not np.any(hit_mask):
        return return_mask

    hit_idx = np.where(hit_mask)[0]
    P_raw = return_probability(incidence_angles[hit_idx],
                                hit_dist[hit_idx], eta=1.0, d_ref=d_ref)

    P_sum = P_raw.sum()
    if P_sum < 1e-12:
        return return_mask

    # Scale so E[N_returns] = target_n_pts, then cap each P at 1
    eta_scaled = target_n_pts / P_sum
    P = np.clip(eta_scaled * P_raw, 0.0, 1.0)

    # Independent Bernoulli draw per beam
    draws = rng.random(len(hit_idx))
    selected = hit_idx[draws < P]
    return_mask[selected] = True

    return return_mask
