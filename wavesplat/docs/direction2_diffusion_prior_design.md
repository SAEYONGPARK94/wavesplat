# Direction II — Physics-Constrained Diffusion Prior for Directional Wave Spectra

> 박사 이론 혁신 방향 II. 생성: 2026-06-26.
> 동기: T-NSE(feedforward amortization)는 input-invariant 붕괴. 교훈 = "amortize 하지 말고
> **prior를 학습**하고 추론은 **per-episode 물리결합**으로." 이건 그 설계.

## 1. 핵심 정식화 (the key idea)

추정 대상을 **방향 진폭 스펙트럼 S(kx,ky)** (2D 격자, 비음수)로 둔다. 위상 φ_n은
per-component 랜덤이므로 **GP로 주변화(marginalize)** 한다 — 이게 T-NSE의 위상 비선형성
문제를 우회하는 핵심.

- **Forward / likelihood (closed-form Gaussian):**
  스펙트럼 S가 정의하는 space-time GP 커널
  `K_S((x,t),(x',t')) = Σ_grid S(k)·cos(k·Δx − ω(k)·Δt)`, `ω(k)=√(g|k|tanh|k|d)`.
  희소 LiDAR 관측 z에 대한 marginal likelihood
  `p(z|S) = N(z; 0, K_S(X,X) + σ²I)` — **S에 대해 미분 가능**, 위상은 해석적으로 주변화됨.

- **Prior:** 시뮬레이터로 무한 생성한 현실적 S(kx,ky) 이미지들에 학습한 **diffusion/score model** `p_θ(S)`.

- **Posterior sampling (DPS/DPnP 류):** `p(S|z) ∝ p(z|S)·p_θ(S)`.
  diffusion denoising 단계(prior score)와 likelihood gradient `∇_S log p(z|S)` 단계를 교대.

→ 학습된 prior가 "현실적 바다 스펙트럼"의 구조로 방향 모호성(aperture problem)을 해소하되,
추론은 per-episode 물리결합이라 amortization 붕괴가 없음. full posterior UQ를 원리적으로 제공.

## 2. 왜 novel (문헌 확인)
- score-based diffusion posterior sampling은 성숙(NeurIPS24 DPnP, FWI, SAR)하나
  **방향 파랑스펙트럼 prior + 물리 posterior는 미존재**.
- GP marginal likelihood로 위상 주변화 + diffusion prior 결합은 이 도메인 최초.

## 3. 데이터 표현
- 스펙트럼 격자: kx,ky ∈ [−1.1, 1.1], **64×64** (Δk≈0.034 rad/m, 인접성분 분해 가능).
- 라벨: `config/wave_components` (A,ω,kx,ky,φ,spread) → 각 성분을 (kx,ky)에 A²(에너지) 배치,
  약간의 가우시안 splat으로 rasterize → S_true(kx,ky). 100 에피소드 + 회전/반사 증강으로 prior 학습셋.
- log1p 스케일 + [0,1] 정규화로 diffusion 학습 안정화.

## 4. 단계 (차례로 학습·기록·검토)
- **Step 1**: 스펙트럼 데이터셋 rasterize + GP-marginal-likelihood forward 미분가능성 sanity. ← 지금
- **Step 2**: diffusion prior 학습 (작은 U-Net, 64×64). 샘플이 현실적 스펙트럼인지 검토.
- **Step 3**: per-episode posterior sampling → 방향복원·field nRMSE를 WaveSplat/oracle과 비교.
- **판정**: oracle gap(0.205→0.126) 좁히면 성공 / 못 좁히면 깨끗한 음성결과(왜 prior로도 안 되는가).

## 5. 위험
- diffusion posterior sampling 튜닝 까다로움(step size, guidance weight).
- GP marginal likelihood 비용: K_S는 grid 합이라 무겁다 → 관측 subsample, 저랭크 근사 검토.
- 스펙트럼 격자 분해능 vs 성분 밀도 trade-off.

## 6. 결과 기록 위치
`results/direction2/` (스펙트럼셋 통계, prior 샘플, posterior 비교).
