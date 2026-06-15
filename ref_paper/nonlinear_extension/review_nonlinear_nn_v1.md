# Nonlinear Extension + NN: Literature Review v1

**Date**: 2026-05-19
**목적**: WaveSplat 비선형 확장 + NN 방향의 선행연구 조사 및 novelty 검증

---

## 1. 핵심 경쟁 논문 (본문 확인)

### 1.1 Ehlers et al. — PINO for Wave Reconstruction (arXiv 2508.03315, Aug 2025)

**아키텍처**: Physics-Informed Neural Operator (PINO)
- 3 Fourier layers, 128 modes, 32-64 latent channels
- **~201M parameters** (매우 큰 모델)
- Zakharov formulation + free surface BC residuals in loss

**Input**:
- Case A: 5 buoys (equidistant, ~246m spacing)
- Case B: X-band radar snapshots + 1 calibration buoy
- **NO LiDAR, NO point cloud**

**Physics**: HOS M=4 reference, nonlinear free surface BCs (soft constraint)

**Dimensions**: **1D+t only** — "Extending to 2D+t is future work"

**Speed**: 0.014s inference per sample (1D)

**Accuracy**: SSP ~0.10 (buoy), ~0.13 (radar)

**한계**:
- 1D only
- Synthetic data only
- 201M params — 대형 GPU 필요
- Soft physics constraint (해 정확도 보장 안됨)

### 1.2 Ehlers et al. — PINN for Wave DA & Prediction (Phys. Rev. Fluids 10, 094901, 2025)

**아키텍처**: Dual-network PINN (eta network + Phi network)
- 4 hidden layers × 200 neurons, tanh activation
- Fourier feature embedding (trainable F matrix, n_f=10)
- Hard constraint via pre-trained M(x,t) + R(x,t) networks (frozen during main training)
- Physics: fully nonlinear potential flow (Laplace + KFSBC + DFSBC + BBC)

**Input**: Sparse surface elevation — 3 buoys (synthetic) or 4 spatial snapshots (wave flume)

**Dimensions**: **1D+t only**

**Training**: ~30 min per instance on RTX 3090, AmsGrad Adam 5K epochs + L-BFGS 40K epochs

**Results**:
- Synthetic linear: SSP=0.024 (elevation), 0.021 (potential)
- Wave flume assimilation (ε=0.09): SSP=0.049
- Wave flume prediction: SSP=0.084 overall, degrades to 0.141 at t+1.6s

**한계**:
- **No generalization** — each wave instance requires full retraining (~30 min)
- 1D only; 2D+t noted as future work
- Prediction degrades with horizon
- No comparison with HOS or ETKF

### 1.3 Klein, Ehlers et al. — ML for Sparse Radar Reconstruction (arXiv 2305.11913, 2023)

**아키텍처**: U-Net + FNO 비교
- U-Net: 5-depth encoder-decoder
- FNO: 3 layers, 64 modes

**Input**: X-band radar (512 spatial points, 1-10 snapshots)

**Physics**: HOS M=4 for GT generation

**Dimensions**: **1D only**

**Accuracy**:
- U-Net (10 snapshots): nL2=0.123, SSP=0.061
- FNO (9 snapshots): nL2=0.153, SSP=0.077

**한계**:
- 1D only
- Synthetic data only
- No parametric representation
- No LiDAR

### 1.4 Desmars et al. — Nonlinear Deterministic Reconstruction (JFM 975, 2023)

**Method**: HOS-OCM (HOS-Observation Coupling Method) + analytical baselines
- LWT, LWT-CDR (corrected dispersion), ICWM (Lagrangian), HOSM1, HOSM3
- Variational optimization: simultaneous control of all time steps
- LGMRES solver with FFT-accelerated matrix-free operations

**Input**:
- Synthetic: random elevations, shadowed radar, radar intensity (tilt + shadowing)
- Experimental: 520 wave gauges at 7m spacing (Draupner wave at 1:70 scale)

**Dimensions**: **1D** (formalism supports 2D but all results 1D)

**Key Results**:
- HOSM3 prediction (irregular, shadowed): SSP=0.036 vs HOSM1 SSP=0.498 (**14x improvement**)
- Regular wave prediction: HOSM3 SSP=0.014 vs LWT SSP=0.287 (**20x improvement**)
- **Only HOSM3 predicted correct amplitude+phase of Draupner extreme wave**
- Prediction horizon: ~6 peak periods forward

**핵심 발견 — 2nd vs 3rd order 차이**:
- **Shape nonlinearity** (2nd order): sharper crests, flatter troughs → improves reconstruction
- **Nonlinear dispersion correction** (3rd order): frequency-dependent phase velocity → improves prediction
- "Second-order models have limited impact on prediction accuracy because they miss nonlinear phase effects"
- ICWM phase correction은 frequency-independent → HOSM3 대비 부족

**우리 실험과의 일치**: 3-way gap에서 M=2가 10s에서 69% 설명하지만 60s에서 37%로 감소 — 이는 Desmars의 발견과 정확히 일치 (3rd order dispersion correction이 시간에 따라 누적)

**한계**:
- 1D only; 2D 확장은 future work
- Computation time not reported (not real-time)
- No NN involved (purely physics-based)
- Per-instance optimization (no generalization)

### 1.5 1D-FNO-ResNet (Ocean Eng, Aug 2024)

**Method**: FNO + ResNet autoregressive model for deep water wave evolution
- Training data: HOS simulation
- Correction model for error accumulation control

**Dimensions**: 1D deep water wave trains

**한계**: 1D only, spatial domain prediction (not parametric)

---

## 2. 경쟁 환경 요약

| 논문 | 방법 | 비선형 | 차원 | Input | 실시간 | NN |
|------|------|--------|------|-------|--------|-----|
| Ehlers PINO 2025 | PINO | M=4 (soft) | **1D** | Buoy/Radar | 0.014s | 201M |
| Ehlers PINN 2025 | PINN | Fully NL | **1D** | Surface elev | 30min | Per-instance |
| Klein+Ehlers 2023 | U-Net/FNO | M=4 (GT) | **1D** | Radar | Not stated | ~10M |
| Desmars 2023 | HOS/ICWM | M=3 | **1D** | Radar/Gauges | Not RT | No |
| 1D-FNO-ResNet 2024 | FNO+ResNet | M=? | **1D** | HOS data | Fast | ~? |
| **WaveSplat (현재)** | Parametric | **Linear** | **2D** | **Sparse LiDAR** | **13ms** | **No** |
| **WaveSplat+NL (제안)** | Parametric+NN | **2nd+NN** | **2D** | **Sparse LiDAR** | **?ms** | **Small** |

---

## 3. 핵심 발견: Gap 분석

### 모든 경쟁자가 1D+t
Ehlers 그룹(Hamburg)이 이 분야를 주도하고 있으나, **모든 연구가 1D+t에 국한**.
2D+t 확장을 "future work"으로 반복 언급 (Ehlers PINO 2025, Desmars 2023).

### Sparse LiDAR input을 다룬 논문 = 0
- Buoy (time series at fixed points)
- X-band radar (spatial snapshots with shadowing)
- Wave gauges (dense array)
- **LiDAR point cloud → 0편**

### Parametric spectral representation + nonlinear = 0
- 모든 경쟁 방법은 **gridded spatial field** 입출력
- **Parametric (kx, ky, w, φ) representation에 비선형 보정** → 선행 연구 없음

### NN으로 Stokes 잔차를 학습 = 0
- PINN/PINO는 PDE residual을 loss에 넣음 (soft constraint)
- Analytical Stokes 2nd order + NN higher-order residual → **이 조합은 없음**

---

## 4. Novelty 평가

### WaveSplat 비선형 확장의 고유한 점

1. **2D+t**: 유일한 2D 비선형 파도 추정 시스템 (모든 경쟁자 1D)
2. **Sparse LiDAR**: 새로운 sensor modality (기존: buoy, radar, gauge)
3. **Parametric + Physics + NN 하이브리드**:
   - 1차: analytical linear (현재 WaveSplat)
   - 2차: analytical Stokes (Sharma & Dean)
   - 고차: NN residual learning
   - **이 3단계 접근은 선행 연구 없음**
4. **Hard dispersion constraint 유지**: PINO/PINN은 soft constraint (loss에 넣음)
   → WaveSplat은 dispersion을 architecture에 hard-code

### PhD Novelty 수준: **충분**

| 기준 | 평가 |
|------|------|
| 방법론 차별성 | **상** — parametric + analytical + NN hybrid (unique) |
| 문제 설정 차별성 | **상** — 2D + sparse LiDAR (unique) |
| 경쟁자 대비 이점 | **상** — 실시간 2D vs 비실시간 1D |
| 독립 논문 가능성 | **상** — "Nonlinear WaveSplat" 단독 논문 |
| 교수님 수용 가능성 | **상** — 물리(Stokes) + NN = 익숙한 조합 |

---

## 5. 위험 요소

### 5.1 Ehlers PINO의 2D 확장 선점 위험
- Ehlers 그룹이 2D 확장을 명시적으로 future work으로 언급
- 그러나 PINO 2D 확장은 계산 비용 문제 (201M→수십억 params?)
- 우리의 parametric 접근은 근본적으로 다른 방법론 → 선점되어도 차별화 가능

### 5.2 비선형 효과 크기 — **실험 완료 (2026-05-19)**
- HOS gap 실험: 10s에서 5%, 30s에서 9%, 60s에서 15%
- **3-way gap 실험 결과**: HOS M=2가 총 gap의 69%(10s), 47%(30s), 37%(60s) 설명
- Desmars 2023과 일치: 2nd order는 shape correction만 제공, 3rd order dispersion correction은 시간에 따라 누적

### 5.3 NN이 학습할 신호 크기 — **정량화 완료**
- Total gap(linear vs HOS M=3): 5-17% nRMSE
- 2nd order가 설명하는 부분: 10s에서 ~3.5%, 30s에서 ~4.5%, 60s에서 ~5.5%
- **NN 타겟 (3rd+ order residual)**: 10s에서 ~1.5%, 30s에서 ~5%, 60s에서 ~9%
- → NN 필요성 확인: 특히 30-60s 예측에서 NN residual이 실질적
- → 10s에서도 1.5%는 WaveSplat 정확도(22% nRMSE) 대비 의미 있는 개선
- **Desmars 핵심 인사이트**: NN이 학습해야 할 것은 주로 frequency-dependent phase correction

---

## 6. 추천 진행 순서

```
Step 1: Stokes 2nd order forward model 구현 (2주)
  → Sharma-Dean interaction coefficients
  → WaveSplat forward model에 통합
  → 기존 episodes에서 reconstruction 비교

Step 2: 3-way gap 실험 (1주)
  → HOS GT vs Linear vs Stokes2 비교
  → Stokes2가 gap의 몇 %를 설명하는지 정량화
  → NN residual의 학습 타겟 크기 결정

Step 3: Neural residual correction 구현 (2-3주)
  → HOS training data 생성
  → MLP: spectral params → δη(x,y)
  → WaveSplat tracking에 통합

Step 4: Prediction extension (2주)
  → Neural ODE: d(w,φ)/dt = physics + NN
  → 30-60s prediction 정확도 개선 측정
```

---

## 참고 문헌

1. Ehlers, Stender, Hoffmann (2025). "Bridging ocean wave physics and deep learning: PINO for nonlinear wavefield reconstruction." arXiv:2508.03315
2. Ehlers et al. (2025). "PINN for phase-resolved DA and prediction of nonlinear ocean waves." arXiv:2501.08430
3. Klein, Ehlers et al. (2023). "ML for phase-resolved reconstruction of nonlinear ocean waves from sparse remote sensing data." arXiv:2305.11913
4. Desmars et al. (2023). "Nonlinear deterministic reconstruction and prediction of remotely measured ocean surface waves." JFM 975
5. 1D-FNO-ResNet (2024). Ocean Engineering, ScienceDirect
