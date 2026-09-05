"""시간 원점(time origin) 민감도 — **측정한 것을 고정한다**.

⛔**이 파일은 「평행이동 불변성이 성립한다」는 가정을 적은 것이 아니다.**
2026-09-05 에 배치 `T17` 이 같은 규칙을 **두 시간 원점**에서 부르고 그 둘을 비교했다:
한쪽은 **원시 `t`**(레코드 절대시각, 최대 **1745.8 s**), 다른 쪽은 `t - t.min()`(0–40 s).
`scale_rules_t16.py:48` 이 `float32` 로 캐스트하는데 `t≈1744 s` 의 float32 ulp 는
**1.22e-4 s**, `t∈[0,40]` 은 **~4e-6 s** ⟹ **위상 정밀도가 약 30배 다르다**.
한 셀에서 항등 스케일(두 호출이 **동일해야 하는** 지점)의 첨두 파수 추정이 **2.126배** 갈렸다.
상세 = `tmp/T17_REALSCALE_2026-09-05/AMENDMENT_2_POSTRESULT.md` §C1.

**시험 구멍**: `test_bank_mmse_port.py:48-56,97-98` 은 `t0=0.0` / `rng.uniform(0,30,n)` 만 쓴다
⟹ 배포 포트의 회귀 시험이 **0 이 아닌 시간 원점을 한 번도 밟지 않았다**.
그리고 배포 패키지는 `learned_init.py:197-199` 을 빼면 **어디서도 시간을 재기준화하지 않는다**
(`sequential_update.py:1230` · `initializer.py:288` · `bank_mmse.py:58,159` 전부 호출자의
원점을 그대로 물려받는다). 불변성은 **호출 규약만으로** 유지되고 있었고 기계가 없었다.

★**그래서 먼저 쟀다**(2026-09-05, CPU, 런 0회의 배치 아님). 실측 요지:

| 팔 | 원점 의존성 | 실측 |
|---|---|---|
| `bank_mmse._beamform_top` (fp32 argmax) | **없음** (t0 ≤ 1e7) | argmax 비트 동일 |
| `bank_mmse.fit_bank_mmse` (fp64 렌더) | float64 반올림뿐 | t0=1745.8 에서 rel 2.9e-13 |
| `SequentialUpdater.predict_field` (실측 게이트 → bank) | **없음** | t0=1745.8 에서 **비트 동일** |
| `SpectralGaussianModel.predict_grid` (fp32 원자) | **있다, t 에 비례** | t=40 4.2e-6 → t=1745.8 3.0e-4 (**70배**) |

⟹ 아래 시험은 그 **측정값**을 고정한다. 허용오차는 전부 **dtype 에서 유도**했고
관측한 수에 맞춘 것이 아니다(유도는 각 시험의 주석에 적었다).
⛔**배포 코드는 한 줄도 고치지 않았다** — 거동 변경은 별도 사전등록 대상이다.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest
import torch

from wavesplat.data_loader import FrameData

G = 9.81
DEPTH = 30.0

EPS32 = float(np.finfo(np.float32).eps)      # 1.1921e-07
EPS64 = float(np.finfo(np.float64).eps)      # 2.2204e-16

# 실측 레코드가 실제로 도달하는 절대시각 (T17 `t_start = 1743.99999…`).
T_REAL_MAX = 1745.8


def _sea(n=300, seed=7, t0=0.0, tau_span=40.0):
    """분산관계에 맞는 2성분 바다. **(x, y, z) 는 t0 와 무관하게 고정**하고
    시간 라벨만 t0 만큼 옮긴다 = 순수한 「재라벨」."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 80, n)
    y = rng.uniform(0, 80, n)
    tau = rng.uniform(0, tau_span, n)

    def om(k):
        return np.sqrt(G * k * np.tanh(k * DEPTH))

    k1, th1, k2, th2 = 0.12, 0.3, 0.2, -0.8
    z = (0.8 * np.cos(k1 * np.cos(th1) * x + k1 * np.sin(th1) * y - om(k1) * tau + 0.5)
         + 0.4 * np.cos(k2 * np.cos(th2) * x + k2 * np.sin(th2) * y - om(k2) * tau + 1.7)
         + rng.normal(0, 0.03, n))
    return np.column_stack([x, y, tau + t0]), z


# ── 0. 기제 자체를 dtype 사실로 고정 ────────────────────────────────────────

def test_float32_ulp_ratio_is_the_t17_mechanism():
    """T17 이 물린 「약 30배」가 순수한 dtype 사실임을 고정한다 (런 0회).

    `t∈[0,40]` 의 ulp 대 `t≈1745.8` 의 ulp 의 비. float32 는 2진 지수마다 ulp 가
    2배가 되므로 이 비는 **정확히 2의 거듭제곱**이고 플랫폼에 의존하지 않는다.
    """
    ulp_local = np.spacing(np.float32(40.0))
    ulp_real = np.spacing(np.float32(T_REAL_MAX))
    assert ulp_local == pytest.approx(3.8147e-06, rel=1e-4)
    assert ulp_real == pytest.approx(1.2207e-04, rel=1e-4)
    assert ulp_real / ulp_local == 32.0, (
        'T17 §C1 이 인용한 「약 30배」는 2^5 = 32 배다')
    # float64 로 같은 것을 하면 그 비는 그대로지만 절대 크기가 9 자리 작다
    assert np.spacing(np.float64(T_REAL_MAX)) < 1e-12


# ── 1. fp32 beamform argmax = 이산 출력이므로 **비트**로 고정한다 ───────────

@pytest.mark.parametrize('t0', [100.0, T_REAL_MAX, 10000.0])
def test_beamform_top_argmax_is_time_origin_invariant(t0):
    """`bank_mmse._beamform_top` (**배포 실측 경로의 fp32 단계**) 의 argmax 가
    시간 원점에 불변임을 **비트로** 고정한다.

    ★**왜 비트인가**: 출력이 격자 위의 argmax = 이산량이다. 「거의 같다」는 의미가 없고
    움직였다면 그 자체가 판정이다.

    ★**허용오차 유도(왜 움직이면 안 되는가)** — 관측한 수가 아니라 dtype 에서 나온다.
      · `bank_mmse.py:58` 이 t 를 float32 로 담고 `:69-70` 이 위상
        `ψ = kx·x + ky·y − ω·t` 를 float32 로 만든다.
      · 절대시각 t0 는 **모든 관측점에 공통**이므로 `|Σ z e^{iψ}|²` 에서 상쇄된다.
        남는 것은 점마다 다른 **반올림 잔차**뿐이고 그 상한은
            δψ ≈ EPS32 · ω_max · t_max                                    … (a)
        `klim=2.0` 격자의 모서리 |k| = 2√2 ⟹ ω_max = √(g·2√2) ≈ 5.27 rad/s.
        t0 = 1745.8 → δψ ≲ 1.19e-7 × 5.27 × 1785.8 ≈ **1.1e-3 rad**.
      · argmax 를 움직이려면 δψ 가 **격자 한 칸이 만드는 위상차**에 견줄 만해야 한다.
        창 안에서 k 를 한 칸(dk = 2·klim/(n−1) = 0.0336) 옮기면 위상은
            |∂ψ/∂k| · dk ≈ (dω/dk) · T_span · dk = 4.58 × 40 × 0.0336 ≈ **6.2 rad**
        만큼 바뀐다(dω/dk 는 kp≈0.12 에서, T_span = 창 40 s).
      · ⟹ **δψ / (한 칸 위상차) ≈ 1.8e-4** = 격자 분해능의 **5천분의 1**.
        ulp 가 격자보다 이만큼 아래면 argmax 는 **원리적으로 움직일 수 없다** ---
        단 **상위 두 첨두가 그 1.8e-4 안에서 동점일 때**는 예외이고,
        ★T17 이 물린 것이 바로 그 예외(2단 격자 **모서리**)다. 아래 셋째 시험 참조.
    """
    from wavesplat.optimizer.bank_mmse import _beamform_top

    P0, z = _sea(t0=0.0)
    P, _ = _sea(t0=t0)
    assert np.array_equal(P[:, :2], P0[:, :2]), '재라벨인데 공간이 움직였다'

    b0 = _beamform_top(P0.astype(np.float32), z.astype(np.float32), DEPTH,
                       klim=2.0, n=120)
    b = _beamform_top(P.astype(np.float32), z.astype(np.float32), DEPTH,
                      klim=2.0, n=120)
    assert b == b0, (
        f'fp32 beamform argmax 가 시간 원점 t0={t0} 에서 움직였다: {b0} -> {b}. '
        f'예측 위상 지터 δψ≈{EPS32 * 5.27 * (t0 + 40):.2e} rad 는 격자 한 칸(≈6.2 rad)의 '
        f'{EPS32 * 5.27 * (t0 + 40) / 6.2:.1e} 배에 불과하다 --- 움직였다면 근사 동점이거나 '
        f'다른(더 심각한) 원인이다')


def test_beamform_top_breaks_down_once_ulp32_swallows_the_window():
    """★**비공허성 증거**: 위 시험이 그냥 둔감한 것이 아니라, fp32 시간축이
    실제로 창을 삼키면 **깨진다**는 것을 같은 코드로 보인다.

    유도: 창 길이 T_span = 40 s. `ulp32(t0) > T_span` 이면 `float32(t0 + τ)` 는
    **τ 전체에 대해 한 값**으로 접힌다 ⟹ 시간축이 소멸하고 beamform 은
    순수 공간 주기도(periodogram)가 된다(공통 위상 `ω·t0` 는 `|·|²` 가 지운다).
    `ulp32(1e10) = 1024 s ≫ 40 s` ⟹ 접힘이 **보장**된다.
    실측(2026-09-05): t0 ≥ **1e8**(ulp 8 s ≈ 파주기) 부터 argmax 가 움직인다.
    """
    from wavesplat.optimizer.bank_mmse import _beamform_top

    P0, z = _sea(t0=0.0)
    P, _ = _sea(t0=1e10)
    P32 = P.astype(np.float32)
    assert len(np.unique(P32[:, 2])) == 1, (
        'ulp32(1e10)=1024 s > 창 40 s 인데 시간축이 접히지 않았다')

    b0 = _beamform_top(P0.astype(np.float32), z.astype(np.float32), DEPTH,
                       klim=2.0, n=120)
    b = _beamform_top(P32, z.astype(np.float32), DEPTH, klim=2.0, n=120)
    assert b != b0, (
        '시간축이 한 점으로 접혔는데도 argmax 가 같다 --- 이 시험이 fp32 시간 경로를 '
        '실제로 밟고 있는지 확인하라(공허한 시험 경보)')


# ── 2. bank-MMSE 적합 전체 (fp64 렌더) ──────────────────────────────────────

def test_fit_bank_mmse_is_time_origin_invariant():
    """`fit_bank_mmse` + `BankMMSEFit.render` 의 평행이동 불변성.

    ★**수학적 근거**(왜 정확 산술에서 **정확히** 불변인가):
      원자가 k 마다 cos/sin **쌍**이므로 t→t+t0 은 각 k 블록의 2×2 **회전** R(ω t0) 이다.
      · `gam` 은 `|Σ z e^{iψ}|²` 에서 나오므로 공통 위상에 불변,
        그리고 cos/sin 두 블록에 **같은 값**이 들어간다(`bank_mmse.py:213` `concat([g, g])`).
      · ⟹ `K = Φ diag(gam) Φᵀ` 에서 `Φ → Φ R` 이고 `R diag(g,g) Rᵀ = diag(g,g)` ⟹ **K 불변**.
      · 렌더 `Φ_q gam Φᵀ α` 도 같은 이유로 불변.
      ⟹ 관측되는 어떤 차이도 **순수 부동소수 잔차**다.

    ★**허용오차 유도** (관측값 적합 금지):
      설계·사전전력·해·렌더가 전부 float64 (`bank_mmse.py:149,191-194,202-204,106-108`).
      위상의 지배항은 `ω·t` 이고 그 반올림 상한은
          δψ ≲ 2 · EPS64 · ω_max · t_max
      이며 `|d cos/dψ| ≤ 1` 이라 설계행렬 원소의 섭동도 같은 크기.
      선형해를 통과하며 조건수만큼 증폭될 수 있으므로 여유계수 **C = 100** 을 곱한다
      (C 는 관측값이 아니라 「조건수 여유」로 고른 상수다).
      t0 = 1745.8 에서 이 상한은 상대 **8e-11** 이고, 실측은 2.9e-13 = **280배 여유**.
    """
    from wavesplat.optimizer.bank_mmse import fit_bank_mmse

    n = 200
    P0, z = _sea(n=n, t0=0.0)
    P, _ = _sea(n=n, t0=T_REAL_MAX)
    rq = np.random.default_rng(3)
    Q_xy = np.column_stack([rq.uniform(0, 80, 150), rq.uniform(0, 80, 150)])
    Q0 = np.column_stack([Q_xy, np.full(150, 41.0)])
    Q = np.column_stack([Q_xy, np.full(150, 41.0 + T_REAL_MAX)])
    sigma0 = 0.05 * 4.0 * float(np.std(z))

    f0 = fit_bank_mmse(P0, z, DEPTH, sigma0, kp_hint=None, seed=5)
    f1 = fit_bank_mmse(P, z, DEPTH, sigma0, kp_hint=None, seed=5)

    # 이산량은 비트로 --- 이것들이 갈리면 격자 자체가 다른 격자다
    assert f1.kp == f0.kp, 'fp32 두단 zoom 의 kp 가 시간 원점에서 갈렸다'
    assert f1.dk_c == f0.dk_c and f1.n_candidates == f0.n_candidates
    assert (f1.pexp, f1.smult) == (f0.pexp, f0.smult), \
        'val 선택 초매개변수가 갈렸다 = 격자가 아니라 판정이 움직였다'

    r0, r1 = f0.render(Q0), f1.render(Q)
    om_max = float(np.max(f0.om))
    t_max = float(max(P[:, 2].max(), Q[:, 2].max()))
    tol_rel = 100.0 * EPS64 * om_max * t_max
    rel = float(np.max(np.abs(r1 - r0))) / float(np.std(r0))
    assert rel <= tol_rel, (
        f'fp64 렌더가 float64 반올림 상한을 넘었다: rel={rel:.3e} > {tol_rel:.3e} '
        f'(ω_max={om_max:.3f}, t_max={t_max:.1f})')
    # 공허하지 않음: 상한 자체가 float32 출력 ulp 보다 한참 아래여야 의미가 있다
    assert tol_rel < EPS32, '허용오차가 float32 해상도보다 크면 아무것도 고정하지 못한다'


# ── 3. 배포 공개 표면 (실측 게이트 → bank) ──────────────────────────────────

def test_shipping_bank_route_predict_field_is_time_origin_invariant():
    """`SequentialUpdater`(실측 게이트가 `bank_mmse` 로 라우팅)의 **공개 표면**.

    ★**허용오차 유도**: `sequential_update.py:1519` 가 결과를 **float32** 로 캐스트한다.
    ⟹ 출력이 표현할 수 있는 최소 눈금은 `ulp32(|field|_max)` ≈ 상대 6e-8 인데,
    위 시험이 보인 float64 잔차는 상대 **8e-11 이하**다 ⟹ 5 자리 아래 ⟹
    ***공개 표면은 비트 동일해야 한다***. 그래서 허용치를 「출력 dtype 의 1 ulp」로 잡는다
    (관측값이 아니라 **출력 dtype** 에서 나온 수다). 실측(2026-09-05)은 **정확히 0.0**.
    """
    import warnings
    from wavesplat.optimizer.sequential_update import SequentialUpdater

    def _frames(t0, n_frames=15, n_per=40, dt=2.0, seed=11):
        r = np.random.default_rng(seed)

        def om(k):
            return np.sqrt(G * k * np.tanh(k * DEPTH))

        k1, th1, k2, th2 = 0.12, 0.3, 0.2, -0.8
        out = []
        for j in range(n_frames):
            tau = j * dt
            xx = r.uniform(0, 80, n_per)
            yy = r.uniform(0, 80, n_per)
            zz = (0.8 * np.cos(k1 * np.cos(th1) * xx + k1 * np.sin(th1) * yy
                               - om(k1) * tau + 0.5)
                  + 0.4 * np.cos(k2 * np.cos(th2) * xx + k2 * np.sin(th2) * yy
                                 - om(k2) * tau + 1.7)
                  + r.normal(0, 0.03, n_per))
            out.append(FrameData(
                t=float(tau + t0),
                x_obs=torch.tensor(xx, dtype=torch.float32),
                y_obs=torch.tensor(yy, dtype=torch.float32),
                z_obs=torch.tensor(zz, dtype=torch.float32),
                confidence=torch.ones(n_per, dtype=torch.float32),
                incidence_angles=torch.zeros(n_per, dtype=torch.float32),
                gt_field=None))
        return out

    gx, gy = torch.linspace(0, 80, 16), torch.linspace(0, 80, 12)
    fields = {}
    for t0 in (0.0, T_REAL_MAX):
        upd = SequentialUpdater({'init_method': 'auto', 'depth': DEPTH, 'seed': 7,
                                 'real_bank_ensemble_k': 1})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            upd.initialize(_frames(t0), x_range=(0, 80), y_range=(0, 80))
        # ★팔 정체는 의도가 아니라 **되읽기**로 확인한다 (2026-09-03 규율)
        assert upd.arm_provenance()['is_bank_mmse'], \
            '실측 게이트가 발화하지 않았다 --- 이 시험은 bank 경로를 재는 것이다'
        fields[t0] = upd.predict_field(gx, gy, 30.0 + t0).numpy()

    d = float(np.max(np.abs(fields[T_REAL_MAX].astype(np.float64)
                            - fields[0.0].astype(np.float64))))
    tol = float(np.spacing(np.float32(np.max(np.abs(fields[0.0])))))   # 출력 1 ulp
    assert d <= tol, (
        f'공개 표면이 시간 원점에서 움직였다: max|Δ|={d:.3e} > 출력 1 ulp {tol:.3e}')


# ── 4. ★해석 팔은 **민감하다** --- 그 민감도를 고정한다 ─────────────────────

def test_analytic_model_phase_error_grows_with_absolute_time():
    """⚠**여기서는 불변성이 성립하지 않는다. 그 사실을 고정한다.**

    `model/spectral_gaussian.py:105-124` 의 `forward` 는 파라미터가 전부 float32 이고
    `t` 를 **절대시각 파이썬 실수**로 받아 `omega[None,:] * t` 를 **float32** 로 만든다
    (`sequential_update.py:2032` `predict_field` → `predict_grid` 가 그대로 넘긴다).
    ⟹ 위상 오차가 **t 에 비례해서 자란다**:

        |Δz| ≲ C · EPS32 · (|k|_max·L_max + π + ω_max·t) · Σ|w|          … (b)

    (공간항 `|k|·x` 와 φ 의 반올림이 t 무관 항, `ω·t` 가 t 비례 항. C 는 여유계수 4.)
    ⛔**이 상한은 float64 로 넓혀도 사라지지 않는다** --- 원자 자체가 float32 로 저장돼
    ω 가 상대 6e-8 밖에 안 알려져 있고, 그 오차가 t 만큼 증폭되기 때문이다.

    실측(2026-09-05, 8원자): t=40 → 4.2e-6 · t=1745.8 → **3.0e-4** (= **70배**).
    ⟹ 이 시험은 (i) 상한 (b) 가 지켜지는지와 (ii) **민감도가 실제로 t 와 함께 자라는지**를
    둘 다 본다. (ii) 가 깨지면 누군가 원자 dtype 이나 위상 규약을 바꾼 것이다.
    """
    from wavesplat.model.spectral_gaussian import SpectralGaussianModel

    rng = np.random.default_rng(0)
    N = 8
    kx = rng.uniform(-0.3, 0.3, N).astype(np.float32)
    ky = rng.uniform(-0.3, 0.3, N).astype(np.float32)
    w = rng.uniform(0.05, 0.5, N).astype(np.float32)
    phi = rng.uniform(0, 2 * np.pi, N).astype(np.float32)

    m = SpectralGaussianModel(n_gaussians=N, depth=DEPTH)
    with torch.no_grad():
        m.kx.copy_(torch.tensor(kx))
        m.ky.copy_(torch.tensor(ky))
        m.w.copy_(torch.tensor(w))
        m.phi.copy_(torch.tensor(phi))

    gx, gy = torch.linspace(0, 80, 17), torch.linspace(0, 80, 13)
    Xg, Yg = np.meshgrid(gx.numpy().astype(np.float64),
                         gy.numpy().astype(np.float64))
    kx64, ky64, w64, phi64 = (a.astype(np.float64) for a in (kx, ky, w, phi))
    kk = np.sqrt(kx64 ** 2 + ky64 ** 2 + 1e-12)
    om64 = np.sqrt(G * kk * np.tanh(kk * DEPTH))
    om_max, k_max, sw = float(om64.max()), float(kk.max()), float(np.abs(w64).sum())
    L_max = 80.0

    def _dev(t):
        f32 = m.predict_grid(gx, gy, float(t)).detach().numpy().astype(np.float64)
        ps = (kx64[None, None, :] * Xg[:, :, None]
              + ky64[None, None, :] * Yg[:, :, None]
              - om64[None, None, :] * t + phi64[None, None, :])
        f64 = (w64[None, None, :] * np.cos(ps)).sum(2)
        return float(np.max(np.abs(f32 - f64)))

    def _bound(t):
        return 4.0 * EPS32 * (k_max * L_max + np.pi + om_max * t) * sw

    d_short, d_real = _dev(40.0), _dev(T_REAL_MAX)
    for t, d in ((40.0, d_short), (T_REAL_MAX, d_real), (10000.0, _dev(10000.0))):
        assert d <= _bound(t), (
            f'fp32 해석 팔의 위상 오차가 유도 상한 (b) 를 넘었다: t={t} '
            f'max|Δ|={d:.3e} > {_bound(t):.3e}')
    # ★민감도가 **실재**한다는 것을 고정한다 (예측비 ≈ 30배, 실측 ≈ 70배; 하한 8배로 잡는다)
    assert d_real > 8.0 * d_short, (
        f'절대시각 민감도가 사라졌다: t=40 에서 {d_short:.3e}, t={T_REAL_MAX} 에서 '
        f'{d_real:.3e}. 원자 dtype 이나 시간 규약이 바뀌었다면 이 파일의 서술을 갱신하라')
