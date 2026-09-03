"""위상무작위화 대체표본 — **한 곳에서만 구현한다**.

★왜 공유 모듈인가: `R2B_STEREONOISE_2026-09-02/run_r2b.py:130` 이 1D 판을 만들었고
`R2C_STEREONOISE_2026-09-03/run_r2c.py:151` 이 **import 대신 사본을 재구현**했다.
CLAUDE.md 가 이 저장소의 반복 결함으로 적어 둔 부류
(*"같은 기능이 두 파일에 중복 구현돼 한쪽만 계측"*)의 재발이고, `T1` 이 세 번째가 될
참이었다. 새 배치는 **여기를 import** 하고 게이트가 이 파일의 md5 를 행에 핀한다.

⛔**여기 있는 함수는 진폭 스펙트럼을 보존한다**(= 귀무 「랜덤위상 선형 중첩」).
그 성질이 통계량 선택을 구속한다 — `|FFT|` 만의 함수인 통계량은 **검정력이 정확히 0** 이다
(`T1` 사전등록 v1 이 그 함정에 빠져 자료 도착 전에 폐기됐다).
"""
import numpy as np

__all__ = ['ft_surrogate', 'ft_surrogate2d', 'skewness', 'crest_trough']


def ft_surrogate(x, rng):
    """1D 푸리에 위상무작위화. `R2B` 의 구현을 축자 이식(거기서 검증됐다).

    ⛔IAAFT 는 채택하지 않는다 — 진폭 **분포**까지 보존하면 우리가 재려는 위상 결합의
    일부를 함께 보존해 검정력을 깎는다.
    """
    x = np.asarray(x, float)
    n = x.size
    F = np.fft.rfft(x)
    ph = rng.uniform(0, 2 * np.pi, F.size)
    ph[0] = 0.0
    if n % 2 == 0:
        ph[-1] = 0.0                       # Nyquist 는 실수여야 한다
    return np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=n)


def ft_surrogate2d(z, rng):
    """2D 푸리에 위상무작위화 — `|FFT2|` 를 보존하고 실수 장을 돌려준다.

    ★**구현**: 백색잡음 실수 장의 FFT 위상을 빌린다. 그것은 에르미트 대칭을 **구성상**
    만족하고 주변분포가 균일하다.

    ★★**두 번의 함정을 내가 실제로 밟았다** — 둘 다 이 저장소의 기록으로 남긴다:
    · **(1) 되감기** — 에르미트 짝은 `F[(-i) % N, (-j) % M]` 이지 `F[N-1-i, M-1-j]` 가
      아니다. `ph[::-1, ::-1]` 로 반대칭화하면 한 칸 어긋나 진폭이 보존되지 않는다
      (`|FFT|` 최대차 **1.46e+03**). ⟹ **`G-NULL`(진폭 보존)이 즉시 잡았다.**
    · **(2) 위상 분포** — 되감기를 고쳐 `(ph − ph_mirror)/2` 로 만들면 진폭은 완벽히
      보존되지만 위상이 **삼각분포**가 되어 0 근처에 몰린다 ⟹ 성분들이 정렬해
      **대체표본 자신이 비가우스**가 된다(왜도 귀무 평균이 0 이 아니라 **+0.18**).
      ⛔**`G-NULL` 은 이것을 못 잡는다** — 진폭은 완벽하다.
      ⟹ ★**`G-NULLMOMENT`**(귀무 모멘트가 이론값에 있는지)가 따로 필요하다.
      이 결함은 **양성 대조가 잡았다**(우리 HOS 가 0/48 로 검출되지 않아 이상을 봤다).
    """
    z = np.asarray(z, float)
    F = np.fft.fft2(z)
    ph = np.angle(np.fft.fft2(rng.normal(size=z.shape)))
    return np.real(np.fft.ifft2(np.abs(F) * np.exp(1j * ph)))


def skewness(z):
    """표본 왜도. 랜덤위상 선형 중첩의 기대값은 **0** 이다."""
    z = np.asarray(z, float).ravel()
    z = z[np.isfinite(z)]
    s = z.std()
    return float(((z - z.mean()) ** 3).mean() / s ** 3) if s > 0 else np.nan


def crest_trough(z):
    """마루/골 비 = `max(z-z̄) / |min(z-z̄)|`. 대칭 장의 기대값은 **1** 이다."""
    z = np.asarray(z, float).ravel()
    z = z[np.isfinite(z)]
    d = z - z.mean()
    lo = float(np.abs(d.min()))
    return float(d.max() / lo) if lo > 0 else np.nan
