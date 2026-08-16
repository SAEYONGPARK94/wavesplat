# WaveSplat PhD 방향 — 팩트 기반 재검토
Date: 2026-05-19

> ⛔**2026-08-13 3라운드 감사 — 이 문서의 §1.3·§1.4 수치는 철회됐다. 인용 금지.**
> 이 문서는 2026-05-19 스냅숏이고 그 이후 기본값·수치가 전부 바뀌었다.
> 근거: `memory/reference_citation_bans.md` · `tmp/AUDIT_recheck_2026-08-13/R3_regression.md §4(ii)`
>
> | 이 문서의 문장 | 상태 |
> |---|---|
> | `v2_konly (default)` (§1.3) | ⛔**낡음** — 2026-07-26 flip 이후 출하 기본값은 `init_method: auto` (GT-free 라우터, `run.py:43` · `configs/estimation.yaml:24`). `v2_konly` 는 명시 지정 시에만 쓰인다 |
> | `WS 0.223 vs ETKF 0.319 vs GPR 0.276` (§1.4) | ⛔**RETRACTED** — 세 값이 **서로 다른 코퍼스**다. GPR `0.276` 은 `results/gpr_comparison.json` 의 **n=10** 산출(같은 파일 동반 팔은 WS-P1 0.2325 · ETKF 0.4499)이고, 100-에피소드 표에 섞을 수 없다. 100-ep 에서 같은 WS 열(0.2229)은 **LETKF 0.2278 과 51/49 p=0.9204 = 널** 이다 |
> | `WS 13.4 ms / ETKF 132.6 ms` (§1.4) | ⛔**RETRACTED** — 16스레드 unpinned 아티팩트. 허용되는 서술은 **동일 프로세스·1스레드 1.5–2.6×** (CPU = AMD Ryzen 7 7800X3D) |
> | `57 tests all passing` (§1.4) | ⛔**낡음** — 실측 **196**(wavesim 85 + wavesplat 111, `pytest --collect-only`) |
> | `13ms, nRMSE 0.223` (§5 리스크 불릿, `:231`) | ⛔**위와 동일 사유로 인용 금지** |

## Part 1: WaveSplat 실제 코드 팩트 체크

### 1.1 Forward Model (spectral_gaussian.py:66-88)
```python
z(x, y, t) = Σ wᵢ · cos(kxᵢ·x + kyᵢ·y - ωᵢ·t + φᵢ)
```
- **선형 Airy 파 모델** (plane wave superposition)
- 자유 파라미터: wᵢ (amplitude), kxᵢ, kyᵢ (wavenumber), φᵢ (phase) — 컴포넌트당 4개
- ωᵢ = √(g·|kᵢ|·tanh(|kᵢ|·d)) — hard dispersion constraint (dispersion.py:22-33)
- "Spectral Gaussian"이라는 이름이지만, 실제로는 **sinusoidal basis의 weighted sum**
- Gaussian kernel이 아님 — 3D Gaussian Splatting과 구조적 유사성은 이름뿐

### 1.2 Optimizer (sequential_update.py)
- **Phase 1 (Init)**: Beamforming → coarse-to-fine (N=3→8→15) → Adam gradient descent
- **Phase 2 (Tracking)**: phase prediction (φ += ω·Δt) + Adam 10 steps (w, φ만 update)
- **Phase 3 (optional)**: Least-squares amplitude refit
- Loss = MSE(obs) + λ₁·L1(w) + λ_smooth·smoothness
- **Neural network 없음, training 없음, pretraining 없음**

### 1.3 초기화 (initializer.py)
- ~~v2_konly (default)~~ ⛔**낡음 (2026-08-13 감사)** — 출하 기본값은 `init_method: auto`.
  v2_konly: dispersion-aware beamforming on raw observations
- Greedy peak extraction → fine refinement → linear projection
- **순수 고전 신호처리** (classical beamforming + LS projection)

### 1.4 현재 성능
- ~~WS avg nRMSE: 0.223 vs ETKF 0.319 vs GPR 0.276~~ ⛔**RETRACTED (2026-08-13 감사)** — 상단 배너 참조
- ~~Speed: WS 13.4ms/frame (tracking), ETKF 132.6ms, GPR >200ms~~ ⛔**RETRACTED (2026-08-13 감사)**
- ~~57 tests all passing~~ ⛔**낡음** — 실측 **196**

---

## Part 2: 이전 제안의 오류 정정

### ❌ 오류 1: "Analytical FSBC가 WaveSplat의 고유 장점"

**사실**: Linear Airy wave에서 dispersion relation ω² = g|k|tanh(|k|d)는
**linearized FSBC 자체**이다.

FSBC (Free Surface Boundary Conditions):
- Kinematic: ∂η/∂t = w|_{z=0} - u·∂η/∂x - v·∂η/∂y
- Dynamic: ∂Φ/∂t + ½|∇Φ|² + gη = 0

**선형화하면** (2차 이상 항 무시):
- ∂η/∂t = ∂Φ/∂z|_{z=0}
- ∂Φ/∂t + gη = 0

이 두 조건을 결합하면 → **ω² = g|k|tanh(|k|d)** = 우리의 dispersion constraint.

**결론**: 선형 모델에서 FSBC loss를 추가하면 **이미 만족되는 조건을 다시 검사하는 것**.
추가 정보 제공 없음. 완전히 **redundant**.

FSBC가 의미 있으려면:
- **비선형 항**이 필요 (Stokes 2nd order 이상)
- 현재 모델은 1st order linear → FSBC 잔차 = 0 (by construction)

### ❌ 오류 2: "Ehlers PINO 대비 analytical FSBC가 정확도 우위"

**사실**: Ehlers의 FSBC loss가 유용한 이유는 그들의 모델이 **FNO (neural operator)**라서
dispersion을 자동으로 만족시키지 않기 때문. FNO 출력을 FSBC로 regularize해야 한다.

WaveSplat은 이미 hard dispersion constraint로 linearized FSBC를 **정확히** 만족.
따라서 "우리가 analytical로 FSBC를 계산할 수 있다"는 것은 사실이지만,
**계산할 필요가 없다** (이미 0).

### ⚠️ 수정 필요: "Gabor-PINN과의 차별화"

Gabor function: G(x) = A·exp(-|x-c|²/σ²)·cos(k·x + φ)
WaveSplat: z = Σ wᵢ·cos(kxᵢ·x + kyᵢ·y - ωᵢ·t + φᵢ)

차이점:
- Gabor: **spatially localized** (Gaussian envelope) → compact support
- WaveSplat: **spatially global** (no envelope) → plane waves
- Gabor: forward PDE solving (known equation, solve for field)
- WaveSplat: inverse estimation (sparse observations, estimate parameters)

이 차이는 유효하지만, "analytical physics" 장점은 과장이었음.

### ⚠️ 수정 필요: "Unrolled optimizer가 DCSpecNet과 다르다"

실제로는 같은 위험을 가짐:
- DCSpecNet 실패 원인: train 분포 ≠ test 분포 (generalization gap)
- Unrolled optimizer도 HOS 시뮬레이터로 training → 같은 generalization 위험
- 다만 unrolled optimizer는 기존 optimizer 구조를 보존하므로 약간 안전

---

## Part 3: WaveSplat의 실제 차별화 (팩트 기반)

### 확실한 차별화 (코드로 확인 가능)

| # | 차별화 포인트 | 근거 (코드 위치) | 경쟁자 대비 |
|---|-------------|----------------|------------|
| 1 | Grid-free: 임의 (x,y)에서 평가 | spectral_gaussian.py:65-88 | FNO, ETKF: regular grid 필요 |
| 2 | Hard dispersion: ω = f(k) in architecture | dispersion.py:22-33 | Ehlers: soft (loss), PINN: soft |
| 3 | Training-free: 순수 optimization | run.py 전체 — 학습 루프 없음 | PINO: pretrain 필요, PINN: per-instance train |
| 4 | Adaptive N: 컴포넌트 수 자동 조절 | sequential_update.py:375-551 | ETKF: 고정 grid, FNO: 고정 latent |
| 5 | Online sequential: frame-by-frame update | sequential_update.py:321-373 | Ehlers PINO: batch, PINN: batch |
| 6 | 실시간 CPU: ~13ms/frame tracking | 측정값 (논문 결과) | ETKF: 133ms, GPR: >200ms |
| 7 | Phase prediction: φ += ω·Δt analytical | sequential_update.py:342-345 | ETKF: forward model 필요 |

### 주장할 수 없는 것

| 주장 | 왜 불가능한지 |
|------|-------------|
| Analytical FSBC | 선형 모델에서 redundant (dispersion = linearized FSBC) |
| Physics-exact representation | Spectral bias 없음은 맞지만, Gabor-PINN도 동일 |
| Novelty in representation | Sum-of-sinusoids는 Fourier decomposition — 매우 고전적 |
| Zero-cost forecasting | Phase advance는 trivial — "novelty"라 부르기 어려움 |

---

## Part 4: 진짜 남은 Gap은 무엇인가?

### Gap 1: 비선형 파 (Nonlinear waves)
- 현재 모델: 1st order Airy (선형)
- GT simulator: Gerstner waves (약비선형)
- HOS simulator: M=1~3 perturbation (비선형)
- BF5+ 해상에서 비선형 효과 무시 불가 (steepness > 0.1)
- **아무도 grid-free explicit representation으로 비선형 파를 다루지 않음**

### Gap 2: 학습 기반 최적화 가속
- 현재: hand-tuned Adam (lr=0.01, tracking_lr=0.005, λ₁=0.001, ...)
- 이 hyperparameters가 모든 해상 조건에서 최적인지 보장 없음
- Algorithm unrolling으로 per-layer step size/regularization을 학습하면
  해상 조건별 adaptive optimization 가능
- **Off-Grid ADMM (MSSP 2025)가 1D에서 성공 → 2D+t 확장은 미개척**

### Gap 3: 실해역 적용
- 현재: 합성 데이터만 (wavesim)
- LiDAR 노이즈, 선체 운동, partial occlusion 등 미처리
- Engineering contribution이지 algorithmic novelty는 아님

### Gap 4: Uncertainty quantification
- 현재: point estimate만 (w, kx, ky, φ의 최적값)
- Bayesian extension (posterior over spectral params)
- KOSS (Kalman-Optimal SSM)가 이 방향의 참고

---

## Part 5: 방향 재평가 (정직한 버전)

### 방향 A: Nonlinear Spectral Gaussian (★★★★★ — 최고 추천)

**핵심**: 1st order Airy → 2nd/3rd order Stokes 확장

현재 모델:
```
η(x,y,t) = Σ Aⱼ cos(θⱼ)     where θⱼ = kxⱼx + kyⱼy - ωⱼt + φⱼ
```

2nd order Stokes 확장:
```
η(x,y,t) = Σ Aⱼ cos(θⱼ) + Σᵢ Σⱼ Bᵢⱼ cos(θᵢ + θⱼ) + Σᵢ Σⱼ Cᵢⱼ cos(θᵢ - θⱼ)
```

여기서 Bᵢⱼ, Cᵢⱼ는 Aᵢ, Aⱼ, kᵢ, kⱼ로부터 **해석적으로** 결정됨 (Dean & Dalrymple).
→ 추가 자유 파라미터 없음! 기존 (w, kx, ky, φ)만으로 비선형 보정 가능.

**이때 FSBC가 의미 있어짐**:
- 2nd order 모델은 FSBC를 **근사적으로** 만족 (정확하지 않음)
- Nonlinear FSBC 잔차를 loss에 추가하면 → 3rd order 효과까지 간접 포착
- **이것이 진짜 "analytical FSBC"의 가치**

**왜 이 방향인가**:
1. **아무도 안 함**: grid-free explicit nonlinear spectral estimation = 0 papers
2. **수학적으로 clean**: Stokes expansion은 well-established theory
3. **기존 코드에 추가 가능**: spectral_gaussian.py의 forward()에 2nd order 항 추가
4. **측정 가능한 개선**: BF5+ episode에서 nRMSE 개선 예상
5. **FSBC loss가 진짜 도움**: 비선형 잔차를 regularization으로 활용
6. **Hard dispersion은 그대로**: 1st order ω = f(k), 2nd order corrections는 해석적

**Advisor fit**:
- "비선형 파랑 물리를 representation에 직접 인코딩"
- CFD 배경과 자연스럽게 연결
- ICNN처럼 "구조적 보장"이라는 동일 철학

**구현 난이도**: 중
- Stokes 2nd order interaction coefficients: 교과서에 있는 공식
- Forward model 확장: O(N²) 추가 항 (N=15~30이면 225~900개)
- Gradient는 PyTorch autograd가 처리
- 핵심 어려움: 비선형 항에 대한 efficient backprop

### 방향 B: Dispersion-Constrained Algorithm Unrolling (★★★★☆)

**핵심**: 기존 optimizer를 K-layer unrolled network로 변환

```
Layer k: θ_{k+1} = θ_k - α_k · ∇L(θ_k) + β_k · proximal_correction
         followed by: ω = √(g|k|tanh(|k|d))  (dispersion projection)
```

여기서 α_k (step size), β_k (regularization weight)는 learnable.

**장점**:
- Off-Grid ADMM (MSSP 2025)의 2D ocean wave 확장 → 명확한 novelty gap
- 기존 optimizer 구조를 보존 → 해석 가능
- Deep Unfolding Survey (2026)가 설계 가이드라인 제공
- CDU (2026)가 hard constraint unrolling 이론 제공

**리스크**:
- DCSpecNet과 같은 generalization gap 가능
  - 완화 방법: UTOPY (2025)의 homotopy training (dense→sparse progressive)
  - 완화 방법: few-shot fine-tuning per episode
- Training data 의존: HOS simulator 필요
- 성능 개선 보장 없음: 현재 Adam이 이미 잘 작동

**Advisor fit**:
- "Optimization algorithm을 학습한다" — Bayesian optimization과 유사 철학
- Physics constraint (dispersion projection) 포함

### 방향 C: A+B 결합 — Nonlinear + Unrolling (★★★★★ 하지만 scope 문제)

Phase 1 논문: Nonlinear Spectral Gaussian (방향 A)
Phase 2 논문: Unrolled optimizer for nonlinear model (방향 B를 A 위에)

PhD 전체로는 이상적이지만, 각 phase가 1편 분량이므로 시간 문제.

### 방향 D: Mamba-based Sequential Tracker (★★★☆☆)

**핵심**: tracking phase의 Adam 10 steps를 STREAM+Mamba SSM으로 대체

**장점**:
- STREAM (2024): irregular point cloud SSM — LiDAR에 직접 적용
- KOSS (2025): Kalman-optimal SSM — ETKF의 학습된 대안
- 트렌디: Mamba가 scientific computing에서 급부상

**리스크**:
- 현재 tracking이 이미 잘 작동 ~~(13ms, nRMSE 0.223)~~ ⛔**두 수치 RETRACTED (2026-08-13 감사)** — 상단 배너 참조
- Mamba를 적용해도 큰 개선이 있을지 불확실
- "Mamba를 wave estimation에 적용" = application paper 위험
- 구현 복잡도 매우 높음

**Verdict**: 보조 기여 수준. 핵심 방향으로는 부적합.

---

## Part 6: 최종 추천

### 1순위: 방향 A — Nonlinear Spectral Gaussian

**이유**:
1. 팩트에 기반한 진짜 gap (비선형 파의 grid-free explicit estimation)
2. 수학적 기반이 확실 (Stokes expansion = 200년 이론)
3. FSBC loss가 진짜 의미 있어짐 (비선형 잔차 regularization)
4. 기존 코드의 자연스러운 확장 (spectral_gaussian.py forward 수정)
5. 합성/실데이터 모두에서 측정 가능한 개선 (BF5+ 해상)
6. 구현 난이도가 관리 가능

**논문 제목 후보**:
"Nonlinear Spectral Gaussian Wave Estimation: Grid-Free Second-Order Reconstruction from Sparse Ship-Mounted LiDAR"

**기여 목록**:
1. 최초의 grid-free explicit nonlinear (2nd order) spectral wave estimation
2. Stokes interaction coefficients를 architectural constraint로 인코딩
3. Nonlinear FSBC 잔차를 regularization loss로 활용 (이때 비로소 의미 있음)
4. BF5+ 해상에서 linear 대비 정량적 개선 입증
5. Hard dispersion (1st order) + analytical Stokes correction (2nd order) = dual physics constraint

### 2순위: 방향 B — Algorithm Unrolling (A와 병행 가능)

**이유**: A가 representation 기여, B가 algorithm 기여. 독립적으로 진행 가능.

### 비추천: 이전 추천 방향들
- ~~PI-WaveSplat (FSBC only)~~: 선형 모델에서 redundant
- ~~Physics-Guided Spectral Mamba~~: 과도한 복잡성, 불확실한 개선
- ~~DCSpecNet 계열~~: 실험적 실패 확인

---

## Part 7: 구현 로드맵 (방향 A)

### Phase 1 (1-2주): Stokes 2nd Order Theory 정리
- Dean & Dalrymple Ch.9에서 interaction coefficients 유도
- Multi-directional 2nd order interaction: Bᵢⱼ⁺ (sum), Bᵢⱼ⁻ (difference)
- 공식 검증: 단일 파에서 known Stokes 2nd order와 비교

### Phase 2 (1-2주): Forward Model 확장
- spectral_gaussian.py에 2nd order forward 추가
- η₂(x,y,t) = Σᵢ Σⱼ Bᵢⱼ⁺ cos(θᵢ+θⱼ) + Bᵢⱼ⁻ cos(θᵢ-θⱼ)
- Efficient computation: O(N²·M) where N=components, M=obs points
- N=15, M=500 → 112,500 operations per eval (still fast)

### Phase 3 (1주): Nonlinear FSBC Loss
- Velocity potential Φ를 Stokes 2nd order로 유도
- Kinematic + Dynamic FSBC 잔차 계산
- L_total = L_obs + λ_FSBC · L_FSBC

### Phase 4 (2주): Evaluation
- 기존 linear WaveSplat vs nonlinear WaveSplat
- BF3 (mild): 차이 작을 것 (비선형 효과 미미)
- BF5+ (steep): 유의미한 개선 예상
- Ablation: 2nd order terms only vs 2nd order + FSBC loss

### Phase 5 (1주): HOS Simulator 기반 검증
- HOS simulator (M=2,3)가 GT → nonlinear WaveSplat이 더 정확해야
- Linear WaveSplat vs nonlinear WaveSplat vs ETKF on HOS episodes

---

## Part 8: 위험 요소 및 대응

| 위험 | 심각도 | 대응 |
|------|--------|------|
| O(N²) 2nd order 항이 너무 느림 | 중 | N=15면 225항, 허용 범위. N>30이면 cutoff 적용 |
| Stokes 2nd order가 BF5에서도 부족 | 낮음 | 3rd order 확장 가능, 또는 FSBC loss로 간접 보정 |
| 비선형 항의 gradient가 불안정 | 중 | Gradient clipping + warm-start from linear solution |
| BF3-4에서 개선 미미 → reviewer 불만 | 중 | BF5-7 에피소드 추가 생성, 비선형 효과가 큰 조건 집중 |
| 이론이 이미 너무 알려져 있음 | 낮음 | Stokes theory는 200년, 하지만 grid-free estimation에의 적용은 없음 |
