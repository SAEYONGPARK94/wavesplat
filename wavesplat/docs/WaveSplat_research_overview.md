# WaveSplat: Grid-free Spectral Gaussian Wave Field Estimation

## 연구 개요 (Research Overview)

**한 줄 요약**: 선박 탑재 LiDAR의 희소 3D 포인트로부터, 물리 기반 Spectral Gaussian으로 전체 해수면 파고장을 실시간 복원하는 grid-free 추정 프레임워크.

**논문 제목**: "WaveSplat: Grid-free Wave Field Estimation from Sparse Ship-mounted LiDAR via Spectral Gaussian Mixture"

**타겟**: Ocean Engineering (IF~5) 또는 IROS 2027

---

## 1. 문제 정의 (Problem Statement)

### 1.1 왜 이 문제가 중요한가?

선박 전방의 파고장(wave field)을 미리 파악하면:
- **하이드로포일 MPC**: 2~5초 preview로 선제적 제어 가능
- **안전**: 위험 파랑 조기 감지
- **연료 효율**: 파도 대응 경로/속도 최적화

### 1.2 기존 방법의 한계

| 기존 방법 | 한계 |
|-----------|------|
| 부이(Buoy) | 고정 점, 공간 해상도 없음 |
| 레이더(X-band) | 대형·고비용, 소형선 부적합 |
| ETKF-grid ("From Points to Waves") | **정규 격자 위에서만 추정** → 관측점-격자 보간 오차 |
| PINO (Physics-Informed Neural Operator) | 정규 격자 입력 전제, 불규칙 관측 불가 |

**핵심 갭**: LiDAR가 생성하는 **불규칙·희소 포인트 클라우드**를 직접 활용하면서, 물리를 만족하는 연속 파고장 추정 방법이 없다.

### 1.3 LiDAR 데이터의 특성

- **센서**: Seyond Falcon K1 (1550nm, 152채널, 120°×25° FOV)
- **출력**: 불규칙 간격의 3D 점군 (x, y, z)
- **난점**:
  - Fresnel 반사 → 입사각 의존적 점 밀도
  - 거리 증가 → 점 밀도·정확도 감소
  - 단일 프레임: 관측 영역 ≪ 추정 영역

---

## 2. 제안 방법 (Proposed Method)

### 2.1 핵심 아이디어

파고장을 **정규 격자**가 아닌, **스펙트럼 도메인의 Gaussian 혼합**으로 표현한다:

```
z(x, y, t) = Σᵢ wᵢ · cos(kxᵢ·x + kyᵢ·y - ωᵢ·t + φᵢ)
```

- **wᵢ**: 진폭 (amplitude) [m]
- **(kxᵢ, kyᵢ)**: 파수 벡터 (wavenumber) [rad/m]
- **ωᵢ**: 각주파수 — **분산 관계로 자동 결정** (자유 파라미터 아님)
- **φᵢ**: 위상 [rad]

**Gaussian 당 자유 파라미터: 4개 (wᵢ, kxᵢ, kyᵢ, φᵢ)**

### 2.2 물리 제약: 분산 관계 (Dispersion Relation)

```
ωᵢ = √(g · |kᵢ| · tanh(|kᵢ| · d))
```

- g = 9.81 m/s², d = 수심 [m]
- |kᵢ| = √(kxᵢ² + kyᵢ²)
- **완전 분산 관계** (심해 근사 ω=√(gk) 사용하지 않음)
- Optimizer가 (kx, ky)를 갱신하면 ω가 자동으로 결정됨 → **물리가 구조에 내장**

### 2.3 2단계 Sequential Update

#### Phase 1: Initialization (첫 2Tp ≈ 14초)

```
[관측 프레임들] → FFT → 초기 (kx, ky, w, φ) → Coarse-to-Fine: N=3→8→15
```

1. 초기 관측 윈도우의 점들을 격자화 → 2D FFT → 주 파수 피크 추출
2. **Coarse-to-Fine**: N=3(대규모 성분)으로 시작, loss 수렴 시 N=8, N=15로 점진 확대
3. 새 Gaussian은 **잔차(residual)의 FFT**로 초기화 → 기존 성분이 놓친 주파수 보완

#### Phase 2: Tracking (이후 프레임, 목표 50ms/frame)

```
매 프레임:
  1. Predict: φᵢ += ωᵢ · Δt   (물리 기반 위상 전진)
  2. Update: 10 gradient steps (w, φ만 업데이트 — kx/ky 고정)
```

- Warm-start: 이전 프레임의 최적 파라미터에서 시작
- 위상 전진이 좋은 초기값을 제공 → 소수 gradient step으로 수렴
- **tracking_mode**: `w_phi` (기본값) — kx/ky를 고정하여 스펙트럼 구조 보존
  - `phi_only`: φ만 업데이트 (Oracle에 가장 가까운 구조)
  - `all`: 전부 업데이트 (기존 방식, 과적합 위험)
- **별도 optimizer**: Init과 Tracking에 서로 다른 Adam optimizer 사용 (tracking_lr=0.005)

### 2.4 관측 신뢰도 가중치 (Confidence Weighting)

```
cⱼ = cos(θⱼ) / (dⱼ / d_ref)
```

- θⱼ: LiDAR 빔과 해수면 법선의 입사각 (시뮬: GT 법선, 실측: $\hat z$ 근사)
- dⱼ: 센서-관측점 거리
- cos(θⱼ): Lambertian 투영 인자 + 입사각 의존 specular 손실의 단조 근사
- **관측 노이즈의 이분산성(heteroscedasticity)** 모델링 — 가중치, likelihood 아님
- "이 관측점의 z 측정이 얼마나 정확한가"를 가중치로 반영
- 정확한 Fresnel R(θ)는 raydrop 시뮬에서만 사용

### 2.5 손실 함수 (Loss Function)

```
L = (1/M) Σⱼ cⱼ · (z_pred - zⱼ)² + λ₁ · ||w||₁ + λ₂ · smoothness
```

| 항 | 역할 |
|----|------|
| cⱼ·(z_pred - zⱼ)² | 가중 MSE (신뢰도 반영) |
| λ₁·‖w‖₁ | L1 정규화 (스펙트럼 희소성) |
| λ₂·smoothness | 인접 성분 간 진폭 변화 제한 |

### 2.6 물리적 안전 장치

- **Steepness clamp**: ε = |k|·|w| ≤ 0.15 (쇄파 방지)
- 선형 모델에는 수평 변위가 없으므로 Gerstner의 Q-constraint는 구조적으로 불필요
- NaN loss 감지 시 자동 stage 전진

---

## 3. 시뮬레이션 (WaveSim)

### 3.1 왜 시뮬레이션인가?

- 실해역 LiDAR 데이터의 GT(ground truth) 파고장 불가능
- 물리 기반 시뮬레이터로 controlled 실험 가능
- Parametric sweep (해상, 관측 수, 모델) 체계적 비교

### 3.2 WaveSim 구성

```
JONSWAP 스펙트럼 → Gerstner 파도 → 메쉬 생성 → LiDAR 레이캐스팅 → 레이드롭
```

| 모듈 | 설명 |
|------|------|
| **Wave Generator** | JONSWAP 스펙트럼 + Gerstner 파면 (Q-constraint 적용) |
| **Mesh Generator** | 파면 → 삼각형 메쉬 (250m × 200m) |
| **Raycast** | Möller-Trumbore 알고리즘 (ray-triangle intersection) |
| **Raydrop Model A** | Fresnel 반사 기반 uniform dropout (n=1.321 @1550nm) |
| **Raydrop Model B** | 경사각 기반 specular dropout (α_threshold=5°) |
| **GT Field** | 고해상도 격자 위 파면 진리값 |
| **Noise Model** | 거리·입사각 의존 측정 노이즈 + confidence |

### 3.3 데이터셋

- **10개 시나리오** × 5 random seeds × 2 raydrop 모델 = **100 에피소드**
- 각 에피소드: 30초, 301 프레임 (Δt = 0.1s)
- 해상: BF3~BF5 (Hs = 0.3~3.25m, Tp = 3~11s)
- 관측 수: 50~300 pts/frame

### 3.4 의도된 Model Mismatch

| | GT (시뮬레이터) | 추정 알고리즘 |
|--|----------------|--------------|
| 파도 모델 | Gerstner (회전, 수평변위) | Linear sinusoidal (비회전) |
| 오차 | — | O(ε²) ≈ 0.25% at BF4 |

→ 추정 알고리즘이 약한 비선형성에도 robust함을 보이는 것이 실험 목표 중 하나.

---

## 4. Baseline 비교 설계

| Method | Physics | Temporal | Grid-free | 역할 |
|--------|---------|----------|-----------|------|
| GPR (single frame) | ✗ | ✗ | ✓ | 하한: 순수 보간 |
| GPR (5-frame window) | ✗ | ✓ | ✓ | temporal 기여 분리 |
| ETKF-grid | ✓ | ✓ | ✗ | **grid 한계 증명** |
| **WaveSplat** | ✓ | ✓ | ✓ | 제안 방법 |
| WaveSplat-oracle | ✓ | ✓ | ✓ | 상한: true (k,A) + φ만 추정 |

→ 이 표로 physics/temporal/grid-free 각각의 기여를 정량적으로 분리 가능.

### 4.1 Ablation Study 설계 (TODO)

| Ablation | 변경 내용 | 증명 대상 |
|----------|----------|-----------|
| A1: No dispersion | ω를 자유 파라미터로 (k와 독립) | C2 기여 |
| A2: No confidence | uniform weight (cⱼ=1) | C3 기여 |
| A3: Grid-based eval | 격자 위에서만 loss 계산 | C1 기여 |

---

## 5. 평가 메트릭 (11개)

### Primary (논문 메인 테이블)
| 메트릭 | 목표 |
|--------|------|
| RMSE_z [m] | < 0.05 (BF4) |
| nRMSE (RMSE/Hs) | < 5% |
| Hs_error [%] | < 5% |
| Tp_error [s] | < 0.5 |
| Spectral_R² | > 0.90 |

### Secondary (분석용)
- RMSE_near (10-80m) / RMSE_far (80-250m)
- RMSE_unobserved (관측되지 않은 영역 → physics prior 기여 핵심 증거)
- T_converge (수렴 시간)
- T_inference [ms/frame] (실시간 가능성)
- Directional_R² (방향 추정 정확도)
- Dispersion 집중도 (ablation 전용)

---

## 6. Novelty / Contributions

### C1: Grid-free Spectral Estimation Framework
**기존**: ETKF-grid는 정규 격자 위에서 상태를 정의 → 불규칙 관측을 격자에 보간해야 함 → **보간 오차가 추정 정확도의 병목**.

**WaveSplat**: 스펙트럼 도메인 Gaussian 표현은 **임의의 (x, y) 좌표에서 직접 평가** 가능. 격자 보간 불필요. 관측점이 불규칙하게 분포해도 정보 손실 없이 활용.

### C2: Dispersion-Constrained Spectral Parameterization
**기존**: ETKF는 격자 상태 + 외부 dispersion propagation. 물리와 표현이 분리.

**WaveSplat**: `ω = f(k, d)` 분산 관계가 **모델 구조에 내장**. (kx, ky)를 자유 파라미터로 최적화하면 ω가 자동 결정. 물리 위반 불가능. Projected gradient descent로 hard constraint 유지.

### C3: Geometry-aware Observation Confidence
**기존**: 관측 노이즈를 등방(isotropic)으로 가정하거나 무시.

**WaveSplat**: 입사각·거리에 의존하는 기하 신뢰 가중치 `cⱼ = cos(θⱼ) / (dⱼ/d_ref)`. cos(θⱼ)는 Lambertian 투영 인자 + Fresnel R(θ)의 단조 surrogate, 1/(d/d_ref)은 정규화된 역거리 감쇠. 해수면 관측 특화. 정확한 Fresnel은 raydrop 시뮬에서만 사용 (편광·미시 거칠기 가정 회피).

### 부가 Novelty
- **WaveSim**: 완전 물리 기반 LiDAR 해수면 시뮬레이터 (Gerstner + Möller-Trumbore + Fresnel raydrop)
- **Intentional model mismatch 분석**: linear estimator vs Gerstner GT → robustness 정량화
- **Coarse-to-Fine spectral refinement**: FFT 잔차 기반 progressive Gaussian 추가

### 6.1 Novelty 위험 요소 (2026-04-06 검토)

| 위험 | 대응 |
|------|------|
| "Grid-free인데 ETKF보다 정확도 낮으면?" | ETKF 직접 비교 필수. slope(불균일) 데이터에서 grid-free 장점 강조 |
| "Dispersion constraint의 실질적 기여?" | Ablation A1 (ω free)으로 정량 증명 필요 |
| "Init 불안정으로 성능 분산 큼" | Init 개선(시공간 FFT) or 다중 seed 앙상블 |

**가장 유사한 선행연구**:
- **E3 (ETKF, Jiang & Li 2024)**: 직접 비교 대상. 격자 보간 한계 → WaveSplat의 C1 근거
- **G2 (FluidGS, 2025)**: 이름 유사하나 도메인(Navier-Stokes/렌더링 vs dispersion/LiDAR) 완전 상이
- **M2 (PINO, 2025)**: 학습 데이터 필요 + 격자 전제 → WaveSplat과 구조적 차이

---

## 7. 시스템 아키텍처

```
wavesplat/
├── model/
│   ├── spectral_gaussian.py   # 핵심 forward model: z = Σ wᵢcos(k·r - ωt + φ)
│   ├── dispersion.py          # ω = √(g|k|tanh(|k|d)) — PyTorch differentiable
│   └── confidence.py          # cⱼ = cos(θ)/d — observation weighting
├── optimizer/
│   ├── initializer.py         # FFT-based (kx, ky, w, φ) 초기화
│   ├── coarse_to_fine.py      # N=3→8→15 progressive refinement
│   └── sequential_update.py   # Phase 1 (init) + Phase 2 (tracking, w_phi mode)
├── evaluation/
│   └── metrics.py             # 11개 평가 메트릭 (directional_r2, dispersion_conc 포함)
├── baselines/
│   ├── gpr_baseline.py        # GPR (Matérn ν=5/2)
│   ├── etkf_grid.py           # ETKF-grid baseline
│   └── oracle_baseline.py     # WaveSplat-oracle (true kx/ky/A, optimize φ only)
├── visualization/
│   └── plot_results.py        # bar chart, convergence, heatmap, text table
├── data_loader.py             # HDF5 → FrameData
├── run.py                     # 메인 추정 실행 (YAML config + CLI override)
├── evaluate.py                # 배치 평가 + 비교표
└── __main__.py                # python -m wavesplat 진입점

wavesim/
├── wave_generator.py          # JONSWAP + Gerstner (GT)
├── raycast.py                 # Möller-Trumbore LiDAR simulation
├── raydrop_uniform.py         # Model A: Fresnel-based dropout
├── raydrop_slope.py           # Model B: specular dropout
├── noise_model.py             # measurement noise + confidence
├── pipeline.py                # end-to-end episode generation
└── config.py                  # YAML config dataclasses

configs/
└── estimation.yaml            # 전체 추정 설정 (model/optimizer/schedule/eval)
```

---

## 8. 검증 현황

| 항목 | 상태 |
|------|------|
| WaveSim 코드 | ✅ 완료 (39 tests pass) |
| WaveSplat 모델 | ✅ 완료 (12 unit tests pass) |
| WaveSplat 통합 | ✅ 완료 (6 integration tests pass) |
| 데이터셋 생성 | 🔄 63/100 에피소드 완료 (BF2-BF4, 1-15comp) |
| Oracle baseline | ✅ 완료 — nRMSE 4.5% (BF3_2comp, 모델 충분성 증명) |
| Tracking 개선 | ✅ w_phi mode 구현 — 기존 대비 최대 45% 개선 |
| Tracking 비교 실험 | ✅ 9개 에피소드 × 3 모드 비교 완료 |
| Init 진단 | ✅ FFT k-mismatch 분석 완료 → **Init 개선 필요** 확인 |
| Ablation study | ⏳ 미실시 (dispersion/confidence 제거 실험) |
| ETKF 직접 비교 | ⏳ 미실시 |
| Init 개선 | ⏳ 시공간 FFT 또는 해상도 향상 |
| 논문 작성 | ⏳ 실험 완료 후 |

**전체 테스트**: 57/57 pass (wavesim 39 + model 12 + integration 6)

### 8.1 현재 성능 (2026-04-06)

#### Tracking 모드 비교 (9개 대표 에피소드 평균)

| 모드 | Tracking nRMSE | 속도 | 비고 |
|------|---------------|------|------|
| **w_phi** (기본) | **13~26%** | ~9ms | kx/ky 고정, w+φ 업데이트 |
| phi_only | 26~34% | ~8ms | w/kx/ky 고정, φ만 업데이트 |
| all (기존) | 27~32% | ~13ms | 전부 업데이트, 과적합 발생 |
| Oracle | ~4.5% | - | true kx/ky/A, φ만 추정 |

#### 에피소드별 상세 (w_phi mode)

| Episode | Init nRMSE | Track nRMSE | all(old) 대비 개선 |
|---------|-----------|-------------|-------------------|
| BF3_2comp (uniform) | 4.8% | 17.3% | +38% |
| BF3_2comp (slope) | 4.9% | **13.2%** | **+45%** |
| BF4_1comp (uniform) | 5.0% | 22.9% | +19% |
| BF4_5comp (slope) | 4.9% | **19.6%** | **+26%** |

**slope(불균일 관측) 데이터에서 w_phi 효과 극대화** → Grid-free (C1) 장점 증거

#### 핵심 병목: Init FFT의 k-mismatch

```
Init이 모든 파도 성분을 잡으면 → tracking nRMSE ~7% (Oracle에 근접)
Init이 부 성분을 놓치면 → tracking nRMSE ~20% (미매칭 성분이 노이즈로 작용)
```

| Episode | True 성분 | 매칭된 성분 | 미매칭 에너지 |
|---------|----------|------------|-------------|
| BF3_2comp | 2 | 1/2 | 3.1% |
| BF3_5comp | 5 | 3/5 | 6.3% |
| BF4_1comp | 1 | 1/1 | 0.0% |
| BF4_5comp | 5 | 3/5 | 24.4% |

근본 원인: 현재 `initializer.py`가 **spatial-only 2D FFT** (32x24 grid) → 시간 정보 미활용, 고주파 성분 Nyquist 한계

---

## 9. 핵심 설계 원칙 (8개)

1. **First Principles Physics**: 모든 물리량은 기본 방정식에서 도출
2. **Independent Verification**: wavesim ↔ wavesplat 독립 구현, 교차 검증
3. **Full Dispersion**: tanh 항상 포함, 심해 근사 사용 금지
4. **Q-constraint**: GT 시뮬레이터에 필수, 추정 모델은 linear이라 구조적 불필요
5. **Parametric Sweep**: 해상·관측수·모델 전수 조합
6. **Two Raydrop Models**: Fresnel(물리) + Specular(경험), 비교 분석
7. **No Paper Delay**: 시뮬레이션으로 완결, 실험은 supplement
8. **Open Code + Data**: 재현 가능성 보장

---

## 10. 기대 성과

- **nRMSE < 10%** (BF3-4) — Init 개선 + w_phi tracking 조합
- **~9ms/frame** 실시간 추정 — 100Hz+ 처리 가능 (10Hz LiDAR 대비 여유)
- **Grid-free**: 관측 밀도·분포에 무관하게 연속 파고장 복원
- slope(불균일) 관측에서 grid-based 대비 우위 정량 증명
- 후속 연구: MPC preview, bound harmonic correction, 실선 검증

---

## 11. TODO — 우선순위순 (2026-04-06 기준)

### P0: 논문 제출 전 필수

| # | 작업 | 예상 효과 | 상태 |
|---|------|----------|------|
| 1 | **Init 개선**: FFT 해상도 향상 (64×48) 또는 시공간 3D FFT | nRMSE 20% → 10% 이하 기대 | ⏳ |
| 2 | **Ablation A1**: Dispersion 제거 (ω free) | C2 contribution 정량 증명 | ⏳ |
| 3 | **Ablation A2**: Confidence 제거 (uniform weight) | C3 contribution 정량 증명 | ⏳ |
| 4 | **ETKF 직접 비교**: 같은 에피소드에서 WaveSplat vs ETKF | 논문 메인 테이블 | ⏳ |
| 5 | **데이터셋 완성**: 나머지 37개 에피소드 생성 | 통계적 유의성 | 🔄 |

### P1: 성능 강화

| # | 작업 | 설명 |
|---|------|------|
| 6 | Multi-frame tracking window | 단일 프레임 대신 최근 3-5 프레임으로 loss 계산 |
| 7 | Adaptive N (Gaussian 수) | 잔차 기반 동적 N 증가/감소 |
| 8 | EMA smoothing | 파라미터 지수이동평균으로 진동 억제 |
| 9 | Hyperparameter sweep | tracking_lr, tracking_steps, lambda 최적화 |

### P2: 논문 작성

| # | 작업 |
|---|------|
| 10 | 실험 결과 정리 + 비교 테이블/그래프 |
| 11 | 논문 초고 작성 (Introduction, Method, Experiments, Discussion) |
| 12 | 연구개요 문서 → 논문 구조 변환 |
