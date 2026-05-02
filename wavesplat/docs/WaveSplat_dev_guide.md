# WaveSplat 개발 가이드

## 연구 설계 토론 결과 및 구현 계획 | 2026.03.29

---

# 1. 연구 개요

## 1.1 연구 주제

선박 전방 조사(forward-looking) LiDAR의 극도로 sparse한 수면 포인트클라우드로부터
연속적 wave height field z(x,y,t)를 복원하는 Grid-free Spectral Gaussian Estimation 방법론 개발.
하류 응용으로 hydrofoil 예측 제어.

## 1.2 핵심 연구 질문

- **RQ1**: 프레임당 수십~수백 개의 극도로 sparse한 LiDAR 수면 포인트로부터 연속적 wave field를 복원할 수 있는가?
- **RQ2**: 파랑 물리 prior(dispersion relation)가 미관측 영역의 복원 정확도를 얼마나 향상시키는가?
- **RQ3**: 복원된 wave field를 hydrofoil 예측 제어에 활용하면 기존 reactive 제어 대비 얼마나 개선되는가? (논문 2)

## 1.3 확정된 Contributions (논문 1, 방법론 중심 — 2026.04.06 4인 토론 확인)

- **C1**: Grid-free spectral estimation framework — 기존 ETKF의 grid 보간 한계를 극복하여 sparse irregular 관측에서 직접 wave field 복원
- **C2**: Dispersion-constrained spectral Gaussian parameterization — 물리 법칙(dispersion relation)을 구조적 제약으로 내장
- **C3**: Geometry-aware confidence model — LiDAR 수면 관측의 입사각/거리 의존 신뢰도 가중 (Lambertian 투영 × 역거리; 정확한 Fresnel R(θ)는 raydrop 시뮬에만 사용)
- **C4**: (선택적, 후속 연구) 2nd order bound harmonic correction

---

# 2. 방법론 결정 과정 요약

## 2.1 3DGS → Spectral Gaussian Estimation 전환

### 전환 이유 (8라운드 검토 결과)

- 3DGS의 핵심(photometric loss, tile-based rasterization, SH color)을 모두 제거하면 "Gaussian basis + gradient descent"만 남아 3DGS라 부를 수 없음
- FluidGS(ACM MM 2025)가 이미 "physics-informed 3DGS for fluids" 수행 → 물리만 바꾸면 application
- "From Points to Waves"(Coastal Eng. 2024)가 ETKF + sparse 3D points로 wave field 추정을 이미 수행 → grid 보간 한계를 극복하는 것이 핵심 차별점
- "Gaussian Splatting"이라는 이름은 misleading → 논문 제목에서 제거

### 최종 방법론: Grid-free Spectral Gaussian Estimation

- 파랑 스펙트럼을 (k, ω) 공간의 Gaussian mixture로 표현
- Dispersion relation을 dynamics model에 하드코딩 (soft loss가 아님)
- Sequential update로 시간에 따른 관측 누적
- Geometry-aware observation noise model (입사각/거리 의존 신뢰도)

### 논문 제목

"WaveSplat: Grid-free Wave Field Estimation from Sparse Ship-mounted LiDAR via Spectral Gaussian Estimation"

## 2.2 Isaac Sim → Pure Python 전환

### 전환 이유

- 지도교수가 시뮬레이터 자체를 신뢰하지 않음. "Isaac Sim이 잘 만들어져 있다"는 근거 불충분
- Isaac Sim의 RTX LiDAR 수면 반사 모델을 우리가 통제할 수 없음
- materialId 반환 버그(#254), reflectance 데이터 접근 불가(#309)
- 재현성: pip install만으로 실행 가능 vs Isaac Sim + RTX GPU 필요
- 해양공학 reviewer에게 "Fresnel 방정식을 직접 구현"이 "NVIDIA 시뮬레이터 사용"보다 설득력 있음
- 데이터셋 생성 속도 차이(18분)는 무시 가능

### Pure Python의 장점

- 모든 물리가 명시적 수식으로 코드에 구현
- 6개 가정이 투명하게 나열 가능
- Unit test로 각 구성요소 독립 검증
- 코드 공개 시 pip install로 재현 가능

## 2.3 Calibration-free 접근

### 결정 이유

- Return probability를 특정 값으로 고정하면 특정 LiDAR/조건에 종속됨
- NEUBOAT calibration은 "장애물 감지용 LiDAR의 우연한 수면 return"에 overfitting 위험
- 통신공학의 SNR sweep처럼, N_pts를 sweep하면 일반적 결론 도출 가능

### 전략

- N_pts = 5, 10, 20, 50, 100, 200, 500을 sweep
- 결과를 "N_pts vs RMSE" 곡선으로 제시
- NEUBOAT 데이터는 이 곡선 위의 "현실 참조점(reality anchor)"으로만 사용 (1~2문장)

---

# 3. 물리적 분석 결과

## 3.1 LiDAR 수면 Return 메커니즘

### NEUBOAT 관측 사실

- 파도가 있는 실해역에서 전방 LiDAR에 수면 return이 간헐적으로 관측됨
- 정수면에서의 테스트는 수행한 적 없음 (정수면에서 나오는지 불확실)

### 두 가지 가능한 메커니즘

**(a) Diffuse scattering (capillary wave 기반)**
- 수면의 미세 거칠기가 입사광을 여러 방향으로 산란
- 풍속에 크게 의존 (Cox-Munk: σ² ≈ 0.003 + 0.00512U)
- 정수면에서도 약간의 return 가능

**(b) Specular backscatter (gravity wave slope 기반)**
- 파봉의 센서 향 경사면에서 수면 법선이 빔 방향과 정렬
- 파의 존재에 결정적으로 의존
- 정수면에서는 return 불가

### 전방 조사에서의 specular backscatter 조건

```
설치 높이 h=5m에서:
  R=200m → grazing 1.4° → 필요 slope 0.025
  R=100m → grazing 2.9° → 필요 slope 0.050
  R=50m  → grazing 5.7° → 필요 slope 0.100
  R=30m  → grazing 9.5° → 필요 slope 0.167

BF4(Hs=1.2m, Tp=7s)에서 파 경사 ε ≈ 0.05
→ 전방 50~200m 범위에서 필요 slope(0.025~0.100)과 파 경사(0.03~0.07)가 겹침
→ 파봉 근처에서 specular backscatter 조건 성립
```

### 방향 의존성

- 파가 센서 방향으로 전파(head sea) → return 최대
- 파가 횡방향(beam sea) → 빔 방향 slope 성분 0 → return 최소

### 대응: 두 가지 raydrop 모델 병렬 구현

어느 메커니즘인지는 NEUBOAT 데이터 분석 또는 수조 실험에서 확인.
그 전까지 두 모델을 모두 구현하고, 비교 자체를 ablation study로 활용.

## 3.2 Fresnel 반사 물리

```
물(n=1.321 @1550nm, Seyond Falcon K1)의 Fresnel 반사율 (비편광 평균):
  θ_i = 0°  (수직 입사): R ≈ 0.02 (2%)
  θ_i = 75°:              R ≈ 0.17
  θ_i = 80°:              R ≈ 0.26
  θ_i = 85°:              R ≈ 0.47
  θ_i = 89°:              R ≈ 0.87

주의: 높은 반사율 ≠ 높은 return rate
  반사율이 높아도 반사 방향이 센서를 향하지 않으면 return 없음
```

## 3.3 비선형 파랑 영향

| BF | ε (steepness) | 2차 파봉 보정 | 판정 |
|----|--------------|-------------|------|
| 3 | ~0.03 | 1~2cm | 무시 가능 |
| 4 | ~0.05 | 5~10cm | 관리 가능 |
| 5 | ~0.05~0.07 | 10~20cm | 2차 보정 필요 (후속) |
| 6+ | >0.06 | >20cm | 적용 범위 외 (쇄파) |

- Hydrofoil operational envelope(Hs<1.5m, BF3~4)과 선형 모델 유효 범위가 겹침
- 비선형 보정(2차 bound harmonics)은 후속 연구로 이동
- 적용 범위: BF3~5, non-breaking waves

## 3.4 Gerstner Wave 중첩의 물리적 충실도

### 계층적 분석

- Level 0: 개별 Gerstner wave = rotational exact solution → 문제 없음
- Level 1: 중첩 = approximate solution → ε ≪ 1에서 유효 (BF3~4)
- Level 2: 국소 경사 한계 초과 가능 → 성분 수 30개 제한으로 확률 극히 낮음
- Level 3: Wave-wave interaction 없음 → 후속 연구
- Level 4: 쇄파 없음 → 범위 외

### 필수 구현: Q 파라미터 제약

```
Particle crossing 방지 조건: ΣQᵢ·ωᵢ·Aᵢ ≤ 1

이 조건이 깨지면 multi-valued surface 발생 → 물리적으로 불가능한 GT
BF3~4에서는 여유 충분 (Qᵢ ≤ 1 가정 시 상한: ΣωᵢAᵢ ≈ 0.3~0.5 << 1)
```

### 시뮬레이션 성능 = 실해역 상한

- Gerstner 중첩에는 bound harmonics가 없음 → 모든 에너지가 dispersion shell 위
- 실해역에서는 에너지 일부가 dispersion 벗어남 → 복원이 더 어려움
- 논문에서 이 사실을 정직하게 명시

## 3.5 물리적 오류 검토 결과 (9개 이슈)

| # | 이슈 | 심각도 | 대응 | 상태 |
|---|------|--------|------|------|
| 1 | Grazing/incidence angle 혼용 | 중간 | θ_g = π/2 - θ_i 관계 명시, θ_g 주로 사용 | 논문 작성 시 |
| 2 | 높은 반사율 ≠ 높은 return | 낮음 | 수식적 유도 보완 | 논문 작성 시 |
| 3 | Gerstner는 rotational (irrotational 아님) | 낮음 | "rotational exact solution" 명시 | 논문 작성 시 |
| 4 | Particle crossing | **높음** | Q 제약 필수 구현 | **코드에서 구현** |
| 5 | Spectral Gaussian 역변환 조건 | 중간 | narrow Gaussian 조건 명시 | 논문 작성 시 |
| 6 | 2차 전달함수 방향 의존성 | 중간 | 후속 연구 (2D full 전달함수) | 스킵 |
| 7 | "깊은 물" 가정 과도 | 중간 | full dispersion(tanh) 항상 사용 | **코드에서 구현** |
| 8 | Return probability 불확실 | **높음** | parametric sweep으로 해소 | **실험 설계** |
| 9 | 시간 전파 비선형 오차 | 낮음 | BF4에서 8초는 안전, 해류는 limitation | 논문 작성 시 |

---

# 4. 선행연구 및 차별성

## 4.1 직접적 위협 논문

| 논문 | 핵심 | WaveSplat 차별점 |
|------|------|----------------|
| FluidGS (ACM MM 2025) | Physics-informed 3DGS for fluids, 4 RGB views | WaveSplat은 3DGS 아님. LiDAR-only. 해양 파랑 전용 물리. |
| From Points to Waves (Coastal Eng. 2024) | ETKF + sparse stereo 3D → wave field, grid 보간 | **핵심 차별점**: grid 보간 없이 직접 estimation. 극한 sparse 대응. |
| Kim et al. (2023) | Adaptive Kalman filter, 선박 운동 → 파랑 역추정 | 선박 운동 센서 기반, 공간 정보 없음. WaveSplat은 공간 wave field 제공. |
| Ehlers et al. (2025) | PINO for nonlinear wave reconstruction | Regular grid 입력(X-band radar). WaveSplat은 irregular sparse LiDAR. |
| EnKF-PFL (Phys. Rev. Fluids, 2025) | EnKF + pseudospectral, 고정 센서 시계열 | 고정 wave gauge. WaveSplat은 이동 센서의 불규칙 포인트. |

## 4.2 핵심 Novelty 논증

기존 연구의 공통 한계:
1. 고정 센서 또는 항공 관측
2. Grid 기반 또는 시계열 기반 관측
3. 충분한 관측 밀도

WaveSplat의 고유 특성:
1. **이동하는 센서** (관측 영역 변화)
2. **Irregular sparse point cloud** (grid 보간 불가)
3. **극한 sparsity** (프레임당 수십 개)
4. **입사각·거리 의존적 신뢰도** (geometry-aware: Lambertian × inverse range)

---

# 5. 시뮬레이션 설계 (WaveSim)

## 5.1 아키텍처

```
wavesim/
├── wave_generator.py        # Gerstner wave (NumPy)
│   ├── single_gerstner()    # 단일 성분
│   ├── jonswap_components() # JONSWAP → 성분 리스트
│   ├── superpose()          # 다중 성분 중첩
│   └── check_q_constraint() # Q 파라미터 제약 검증
│
├── lidar_simulator.py       # LiDAR 시뮬레이션
│   ├── generate_rays()      # 빔 패턴 생성 (기하학)
│   ├── build_mesh()         # wave vertices → trimesh
│   ├── raycast()            # Ray-mesh intersection (Open3D)
│   └── compute_normals()    # Hit point에서 수면 법선
│
├── raydrop_uniform.py       # Model A: 거리/Fresnel 기반 균일 dropout
│   └── apply_dropout()      # P = η × R_fresnel × cos(θ) / d²
│
├── raydrop_slope.py         # Model B: 파 slope 기반 specular dropout
│   └── apply_dropout()      # 반사 방향이 센서를 향하는지 판정
│
├── gt_generator.py          # GT 생성
│   └── evaluate_field()     # Gerstner 수식을 정규 그리드에서 평가
│
├── noise_model.py           # 센서 노이즈
│   └── add_noise()          # Gaussian range/angular noise
│
├── data_writer.py           # HDF5 저장
│
├── config.yaml              # 모든 파라미터
│
└── tests/                   # Unit tests
    ├── test_wave_generator.py
    ├── test_raycast.py
    ├── test_raydrop.py
    └── test_gt.py
```

## 5.2 Trust Chain (물리 법칙 → 코드 → 검증)

| 구성요소 | 물리 법칙 | 출처 | 검증 방법 |
|---------|----------|------|----------|
| Wave mesh | Gerstner exact solution | Gerstner (1802), Dean & Dalrymple (1991) | 해석해 수치 비교, RMSE < 1e-6m |
| JONSWAP | 경험적 스펙트럼 | Hasselmann et al. (1973) | 스펙트럼 형태/Hs/Tp 비교 |
| Q 제약 | Jacobian > 0 조건 | Tessendorf (2001), Finch (2004) | J 값 모니터링, 항상 > 0 |
| Ray intersection | Möller-Trumbore | Möller & Trumbore (1997) | 알려진 곡면에서 해석해 비교 |
| Surface normal | Mesh 미분기하 | 표준 수학 | Gerstner 해석적 법선과 비교 |
| Fresnel 반사율 | Fresnel 방정식 | Hecht (2017, Optics) | n=1.321(@1550nm)에서 교과서 값 비교 |
| Dropout A | 거리/Fresnel/Lambertian | 표준 광학 | 문헌 return rate 범위 확인 |
| Dropout B | Specular backscatter 기하 | 반사 법칙 | 정수면에서 return=0 확인 |
| Sensor noise | Gaussian noise | LiDAR datasheet | 스펙 값 직접 적용 |
| Dispersion | ω = √(gk·tanh(kd)) | Airy wave theory | 해석해 일치 확인 |

## 5.3 WaveSim에서 확실히 스킵하는 것

| 요소 | 스킵 이유 |
|------|----------|
| Isaac Sim / 시각적 렌더링 | 불필요 (LiDAR-only) |
| 바람/조류/안개/비 | parametric sweep으로 커버 |
| 비선형 파랑 (HOS) | 후속 연구 |
| 쇄파 | 범위 외 |
| 6-DOF 선박 운동 | 후속 (정지+직진만) |
| MDL material 커스터마이징 | Isaac Sim 불사용 |
| 수면 물보라/거품 | BF3~5에서 미미 |

## 5.4 데이터 형식: HDF5

```
episode_BF4_30comp_seed42.h5
├── /config/
│   ├── sea_state: "BF4"
│   ├── Hs: 1.2
│   ├── Tp: 7.0
│   ├── gamma: 3.3
│   ├── n_components: 30
│   ├── wave_components: (30, 6)  # [A, omega, kx, ky, phi, Q]
│   ├── depth: 50.0
│   └── lidar_config/
│       ├── height: 5.0
│       ├── channels: 152
│       └── scan_rate: 10.0
│
├── /frames/
│   ├── timestamps: (N_frames,)
│   ├── lidar_points: (N_frames, max_pts, 3)
│   ├── lidar_confidence: (N_frames, max_pts)
│   ├── lidar_valid_mask: (N_frames, max_pts)
│   ├── lidar_incidence_angles: (N_frames, max_pts)
│   ├── gt_wave_field: (N_frames, Ny, Nx)
│   ├── gt_grid_x: (Nx,)
│   └── gt_grid_y: (Ny,)
│
├── /raydrop_model/
│   ├── model_type: "uniform" or "slope_dependent"
│   └── parameters: {...}
│
└── /gt_params/
    ├── Hs_true: float
    ├── Tp_true: float
    └── spectrum_1d: (N_freq, 2)
```

## 5.5 데이터셋 구성

10개 시나리오 × 5 random seeds × 2 raydrop models = 100 에피소드 (각 30초, 10Hz = 301프레임)

| 시나리오 | 해상상태 | 성분 수 | 선박 | 목적 |
|---------|---------|--------|------|------|
| S01 | BF2 (Hs=0.3m) | 1 | 정지 | 단일파 기본 |
| S02 | BF3 (Hs=0.6m) | 1 | 정지 | 단일파 |
| S03 | BF3 | 2 | 정지 | Bi-chromatic |
| S04 | BF3 | 5 | 정지 | 소수 성분 |
| S05 | BF4 (Hs=1.2m) | 1 | 정지 | 단일파 |
| S06 | BF4 | 5 | 정지 | 소수 성분 |
| S07 | BF4 | 15 | 정지 | 중간 복잡도 |
| S08 | BF4 | 30 | 정지 | 전체 스펙트럼 |
| S09 | BF5 (Hs=2.0m) | 15 | 정지 | 높은 해상 |
| S10 | BF5 | 30 | 정지 | 높은 해상 |

**이동 시나리오 (Phase 2 이후 추가 예정)**:
| S11 | BF4 | 15 | 직진 5kts | 선박 이동 |
| S12 | BF4 | 30 | 직진 5kts | 선박 이동 |
| S13 | BF5 | 15 | 직진 5kts | 선박 이동+높은 해상 |
| S14 | BF4 | 30 | 직진 10kts | 고속 |
| S15 | BF5 | 30 | 직진 10kts | 극한 |

---

# 6. WaveSplat 알고리즘 설계 (2026.04.05 4인 토론 반영)

## 6.1 Forward Model: Linear Sinusoidal

추정 알고리즘은 **linear sinusoidal model**을 사용 (Gerstner의 horizontal displacement 무시):

```
z(x, y, t) = Σᵢ Aᵢ · cos(kᵢ·r - ωᵢ·t + φᵢ)
```

**의도된 model mismatch**: 시뮬레이터(GT)는 Gerstner wave이므로 O(ε²) 차이 존재.
BF4에서 ε ≈ 0.05 → 오차 ~0.25%. 논문 Discussion에 명시하되, "linear estimator가
weakly nonlinear GT에 대해 robust함을 보임"으로 오히려 강점으로 활용.

## 6.2 Spectral Gaussian Representation

파랑 스펙트럼 S(k, ω)를 (k, ω) 공간의 Gaussian mixture로 표현:

```
S(k, ω) = Σᵢ wᵢ · G(k, ω | μᵢ, Σᵢ)
```

각 Gaussian의 중심 μᵢ = (kₓᵢ, kyᵢ, ωᵢ)는 dispersion relation **hard constraint**:

```
ωᵢ = √(g·|kᵢ|·tanh(|kᵢ|·d))   [projection step으로 강제]
```

→ 자유 파라미터: wᵢ, kxᵢ, kyᵢ, φᵢ (Gaussian당 4개)
→ N=15 → 60 파라미터

## 6.3 Observation Model

LiDAR 포인트 (pⱼ, zⱼ)에서:

```
z_model(pⱼ, t) = Σᵢ Aᵢ · cos(kᵢ·pⱼ - ωᵢ·t + φᵢ)
   where Aᵢ = √(2wᵢ)
```

**Confidence weight cⱼ = cos(θᵢ) / (d/d_ref)**:
- 이것은 **observation noise의 heteroscedasticity**를 반영
- "이 포인트의 z 측정이 얼마나 정확한가" (NOT wave model의 정확도)

## 6.4 최적화: Projected Gradient Descent (PyTorch)

```
min_{wᵢ, kᵢ, φᵢ} Σⱼ cⱼ · (z_model(pⱼ,t) - zⱼ)² + λ₁·||w||₁ + λ₂·smoothness

λ₁·||w||₁:    L1 sparsity (불필요 Gaussian 비활성화)
λ₂·smoothness: 인접 Gaussian 간 weight 변화 제한 (실제 스펙트럼의 smoothness)
```

- EM 부적합 (dispersion constraint가 closed-form M-step 깨뜨림)
- 각 step: gradient → **projection** (ωᵢ = f(kᵢ) 재계산)

## 6.5 Sequential Update: 2단계 설계

### Phase A: Initialization (첫 2 peak periods)

- BF4 (Tp=7s)에서 ~14초 = 140프레임
- **Coarse-to-fine**: N=3 → 8 → 15 (점진적 Gaussian 추가)
- FFT-based spectral analysis로 초기값 설정
- Joint optimization over full window
- 오프라인 수준 (수초)

### Phase B: Tracking (이후 매 프레임)

- **Warm-start**: 이전 θ* = (w*, k*, φ*)를 초기값으로 사용
- **Predict**: φᵢ(t+Δt) = φᵢ(t) + ωᵢ·Δt
- **Update**: 5~10 gradient steps per frame
- 목표: 50ms/frame 이내

### Phase 결정 논거

- 단일 프레임에서 phase(φ) 결정 불가 → temporal window 필수
- Initialization에서 joint optimization으로 phase 확정 후, tracking은 미세 조정만
- **수렴 시간(T_converge) 자체를 실험 결과로 보고**

## 6.6 Gaussian 수 N

- **논문 1**: 고정 N, ablation으로 N=5 / 10 / 15 비교
- 물리적 가이드라인: N ≈ spectral bandwidth / mean Gaussian width
- **Adaptive densification = future work** (3DGS 유사성 혼란 방지)

## 6.7 코드 구조

```
wavesplat/
├── model/
│   ├── spectral_gaussian.py  # Spectral Gaussian representation (PyTorch)
│   ├── dispersion.py         # ω = √(gk·tanh(kd)), projection step
│   └── confidence.py         # Observation noise weighting (NOT model accuracy)
│
├── optimizer/
│   ├── sequential_update.py  # 2-phase: initialization + tracking
│   └── coarse_to_fine.py     # Progressive N=3→8→15
│
├── evaluation/
│   ├── metrics.py            # Primary 5 + Secondary 6 metrics
│   ├── spectral_analysis.py  # Welch, 2D FFT, coherence
│   └── visualize.py          # matplotlib 시각화
│
├── baselines/
│   ├── gpr_baseline.py       # GPR: single-frame + window (scikit-learn)
│   └── etkf_grid.py          # ETKF with grid interpolation
│
├── run.py                    # 추론/복원 메인 스크립트
├── evaluate.py               # 평가 메인 스크립트
└── config.yaml               # 알고리즘 하이퍼파라미터
```

---

# 7. 실험 설계

## 7.1 실험 매트릭스

### Part A: Progressive Complexity

성분 수 증가에 따른 각 방법의 성능 변화 (정지 선박, Model A+B):

| 실험 | 성분 수 | N_pts | 핵심 관찰 |
|------|--------|-------|----------|
| A1 | 1 | 100 | 모든 방법 성공 (sanity check) |
| A2 | 2 | 100 | Beat pattern, spectral estimation 시작 |
| A3 | 5 | 100 | GPR 한계 시작 |
| A4 | 15 | 100 | WaveSplat 우위 emerge |
| A5 | 30 | 100 | Main result |

### Part B: Sparsity Sensitivity

동일 파 조건(30성분, BF4)에서 밀도 변화:

| 실험 | N_pts | 핵심 관찰 |
|------|-------|----------|
| B1 | 5 | 극한 sparse, 모든 방법의 하한 |
| B2 | 10 | |
| B3 | 20 | |
| B4 | 50 | |
| B5 | 100 | |
| B6 | 200 | |
| B7 | 500 | Dense, 모든 방법의 상한 |

### Part C: Observation Model 비교

동일 조건에서 Model A vs Model B:

| 실험 | 성분 | N_pts | Model | 핵심 관찰 |
|------|------|-------|-------|----------|
| C1 | 30 | 50 | A (uniform) | 균일 분포 |
| C2 | 30 | 50 | B (slope) | 파봉 집중 분포 |
| C3 | 30 | 100 | A | |
| C4 | 30 | 100 | B | |

### Part D: Baseline 비교 (2026.04.05 확정)

| Method | Physics | Temporal | Grid-free | 구현 |
|--------|---------|----------|-----------|------|
| GPR (single frame) | ✗ | ✗ | ✓ | scikit-learn GP, Matérn ν=5/2 |
| GPR (window, 5 frames) | ✗ | ✓ | ✓ | 5프레임 누적 → GP fitting |
| ETKF-grid | ✓ | ✓ | ✗ | "From Points to Waves" 재구현 |
| **WaveSplat** | ✓ | ✓ | ✓ | 제안 방법 |
| WaveSplat-oracle | ✓ | ✓ | ✓ | True (k,ω) init, φ만 최적화 |

**제외**: PINO (regular grid 전제), EnKF-PFL (고정 센서 전제) — 논문에 사유 명시

GPR 설정:
- Kernel: Matérn ν=5/2 (주), RBF (비교) — Matérn은 C¹, 파도에 적합
- length_scale grid search: [5, 10, 20, 50m]
- noise: σ=0.02m
- Window GPR: 직전 5프레임 누적, 시간 정보 무시 → temporal accumulation만의 효과 측정

ETKF-grid 설정:
- State: η(grid) + ∂η/∂t, prediction: dispersion 시간 진화
- Observation operator: bilinear interpolation (grid → obs points)
- Grid 해상도 3단계: 20×15 / 40×30 / 80×60, 최적 결과 보고
- 보간: thin-plate spline, ensemble=50
- 원저자 코드 요청 시도

### Part E: Ablation Study

| 실험 | 구성 | 목적 |
|------|------|------|
| E1 | WaveSplat (full) | 전체 성능 |
| E2 | - dispersion 제약 (soft/없음) | Dispersion의 기여 + dispersion 집중도 메트릭 |
| E3 | - confidence weighting | Confidence의 기여 |
| E4 | - sequential update (single frame) | Temporal accumulation의 기여 |

## 7.2 평가 ��표 (2026.04.05 확정)

### Primary Metrics

| 메트릭 | 정의 | 목표 |
|--------|------|------|
| RMSE_z | √(mean(z_pred - z_gt)²) [m] | < 0.05m (BF4) |
| nRMSE | RMSE_z / Hs | < 5% |
| Hs_error | \|Hs_pred - Hs_gt\| / Hs_gt [%] | < 5% |
| Tp_error | \|Tp_pred - Tp_gt\| [s] | < 0.5s |
| Spectral_R² | R² of S_pred(ω) vs S_gt(ω) | > 0.90 |

### Secondary Metrics

| 메트릭 | 목적 |
|--------|------|
| RMSE_near (10-80m) / RMSE_far (80-250m) | 거리별 성능 분리 |
| RMSE_unobserved | LiDAR 포인트 없는 영역 → **physics prior 기여 핵심 증거** |
| T_converge [s] | nRMSE < threshold 도달 시간, init/tracking 구분 |
| T_inference [ms/frame] | 실시간 가능성 |
| Directional_R² | 2D (kx,ky) spectrum 비교 → 방향 추정 정확도 |
| Dispersion 집중도 | Ablation E2에서만 사용 (hard constraint면 항상 1.0) |

### 보고 방식

- 5 seeds, mean ± std
- Effect size 포함
- 근거리/원거리/미관측 영역 분리 보고
- Uncertainty estimation: 논문 1 scope 밖 (future work)

## 7.3 핵심 Figure 목록 (논문용)

1. **N_pts vs nRMSE 곡선** — 방법별 비교 (가장 중요한 그림)
2. **성분 수 vs nRMSE 곡선** — progressive complexity
3. **복원 시각화** — sparse 포인트 + GT + 복원된 surface (3D)
4. **1D 스펙트럼 비교** — GT vs WaveSplat vs baselines
5. **Observation model 비교** — Model A vs B의 포인트 분포 + 성능
6. **Ablation** — 각 구성요소 제거 시 성능 변화
7. **수렴 시간** — T_converge vs N_pts (initialization phase)
8. **RMSE_near vs RMSE_far** — 거리별 성능 + RMSE_unobserved

---

# 8. NEUBOAT 데이터 분석 계획

## 8.1 목적

- "실제 선박 LiDAR에서 수면 return이 존재한다"는 feasibility 입증
- Return 통계(프레임당 수, 거리 분포)로 parametric sweep의 현실 범위 확인
- 논문의 reality anchor (1~2문장 + 통계 표/그래프)

## 8.2 작업 절차

```
Step 1: 수면 포인트 추출 (1~2주)
  입력: LiDAR point cloud + GNSS/IMU
  처리:
    1. 선박 자세 보정 (기존 코드 활용)
    2. 높이 필터: 해수면 추정 높이 ± 3m
    3. 거리 필터: 전방 10~200m
    4. 선박 구조물 제거 (기존 마스킹)
    5. 남은 포인트 = 수면 후보
  출력: 수면 후보 포인트 + 메타데이터

Step 2: 통계 분석 (1주)
  - 프레임당 수면 포인트 수 분포 (평균, 중앙값, 범위)
  - 거리별 포인트 밀도
  - Intensity 분포
  - 가능하면 기상 데이터 매칭
  출력: 통계 표, return rate vs distance 그래프

Step 3: 논문용 정리
  - "NEUBOAT LiDAR에서 프레임당 평균 X개(범위 Y~Z)의 수면 return 관측"
  - 이 값을 parametric sweep 곡선 위에 표시
```

## 8.3 병렬 진행

NEUBOAT 데이터 분석과 WaveSim 개발은 **동시에 진행 가능**.
WaveSim은 두 가지 raydrop 모델을 모두 구현해두고,
NEUBOAT 분석 결과에 따라 main result의 모델을 선택.

---

# 9. 논문 구조

## 9.1 논문 1 구조

```
Section 1: Introduction
  - 선박 전방 LiDAR의 수면 관측 한계
  - 기존 방법의 한계 (grid 기반, dense 관측 전제)
  - WaveSplat 제안 및 contribution 요약

Section 2: Problem Formulation
  - LiDAR 수면 반사 물리 (Fresnel, specular/diffuse)
  - Forward-looking geometry와 극한 sparsity
  - 문제의 수학적 정의

Section 3: WaveSplat Algorithm
  3.1: Spectral Gaussian representation
  3.2: Dispersion-constrained parameterization
  3.3: Sequential update with temporal coherence
  3.4: Confidence-weighted observation model

Section 4: WaveSim — Physics-based Simulation
  4.1: 설계 철학 및 trust chain
  4.2: Wave generation (Gerstner + JONSWAP + Q 제약)
  4.3: LiDAR simulation (ray casting + two raydrop models)
  4.4: 구성요소별 검증 결과

Section 5: Experimental Results
  5.1: Progressive complexity (성분 1→30)
  5.2: Sparsity sensitivity (N_pts sweep)
  5.3: Baseline 비교 (GPR, ETKF-grid)
  5.4: Observation model 비교 (uniform vs slope-dependent)
  5.5: Ablation study
  5.6: NEUBOAT 실데이터 분석 (reality anchor)

Section 6: Discussion
  - 적용 범위 및 한계 (BF3~5, 선형, non-breaking)
  - 시뮬레이션 성능 = 실해역 상한
  - Hydrofoil 응용 가능성 (future work 예고)

Section 7: Conclusion
```

타겟: **Ocean Engineering** (IF ~5) 또는 **IROS 2027**

## 9.2 논문 2 구조 (후속)

```
"Wave-Preview Model Predictive Control for Hydrofoil Vessels
 Using LiDAR-based Wave Field Estimation"

- 논문 1의 WaveSplat을 black box로 사용
- Dispersion 기반 시간 전파 (미래 예측)
- Hydrofoil MPC 설계
- Closed-loop 시뮬레이션: no control / reactive PID / WaveSplat+MPC / perfect preview
```

타겟: **IROS** 또는 **Control Engineering Practice**

## 9.3 학위 논문 구조

```
Chapter 1: 서론
Chapter 2: 선행연구
Chapter 3: LiDAR 수면 관측 물리 + WaveSim [논문 1 확장]
Chapter 4: Spectral Gaussian Estimation [논문 1 핵심]
Chapter 5: 시뮬레이션 검증 [논문 1 실험]
Chapter 6: 비선형 보정 + 선박 운동 확장 [미발표]
Chapter 7: 실해역 검증 [미발표, 수조/실선]
Chapter 8: Hydrofoil 예측 제어 [논문 2]
Chapter 9: 결론
```

---

# 10. 타임라인

| Phase | 기간 | Track A (데이터) | Track B (개발) |
|-------|------|----------------|---------------|
| 1 | M1~M2 | NEUBOAT 수면 return 분석 | WaveSim 구축 (wave_generator, lidar_simulator, 두 raydrop 모델) |
| 2 | M3~M4 | 분석 결과 반영 → main raydrop 모델 선택 | WaveSim 검증 + 데이터셋 생성 |
| 3 | M5~M8 | — | WaveSplat 알고리즘 개발 + 시뮬레이션 실험 |
| 4 | M9~M11 | — | 논문 1 작성 + 투고 |
| (수조 가능 시) | M12 | 수조 실험 (1일) | revision 반영 |
| (revision/버퍼) | M13 | — | 논문 1 revision 대응 및 버퍼 |
| 5 | M14~M17 | — | 논문 2 (Hydrofoil MPC) |
| 6 | M18~M24 | 실선/수조 | 학위 논문 |

---

# 11. 즉시 시작할 작업 (우선순위)

## 11.1 Phase 1-A: NEUBOAT 데이터 분석 (M1, 1~2주)

```bash
# 작업 디렉토리
mkdir -p ~/wavesplat/neuboat_analysis
cd ~/wavesplat/neuboat_analysis

# 필요한 스크립트
# extract_surface_points.py  — 수면 포인트 추출
# analyze_statistics.py      — 통계 분석
# plot_results.py             — 시각화
```

## 11.2 Phase 1-B: WaveSim 기본 구축 (M1~M2, 3~4주)

```bash
# 프로젝트 구조
mkdir -p ~/wavesplat/{wavesim,wavesplat,experiments,data,docs}

# 구현 순서:
# 1. wave_generator.py (1주)
#    - single_gerstner() → 해석해 비교 unit test
#    - jonswap_components() → 스펙트럼 형태 검증
#    - Q 제약 구현
#
# 2. lidar_simulator.py + raydrop 두 모델 (1주)
#    - Open3D raycasting
#    - raydrop_uniform.py
#    - raydrop_slope.py
#
# 3. gt_generator.py + noise_model.py + data_writer.py (0.5주)
#
# 4. 전체 파이프라인 검증 + 데이터셋 생성 (0.5주)
```

## 11.3 Python 의존성

```
# requirements.txt
numpy>=1.24
open3d>=0.17
h5py>=3.8
scipy>=1.10
matplotlib>=3.7
torch>=2.0
scikit-learn>=1.2  # GPR baseline
pyyaml>=6.0
pytest>=7.0
```

---

# 12. 핵심 원칙 (개발 중 항상 참조)

1. **모든 물리는 first principles에서 유도한다.** "라이브러리가 해준다"는 근거가 아니다.
2. **모든 구성요소는 독립적으로 검증한다.** Unit test로 해석해와 비교.
3. **Full dispersion relation(tanh 포함)을 항상 사용한다.** 깊은 물 근사 금지.
4. **Q 파라미터 제약(ΣQᵢωᵢAᵢ ≤ 1)을 항상 강제한다.**
5. **Parametric sweep으로 결론을 일반화한다.** 특정 값에 종속된 주장 금지.
6. **두 가지 raydrop 모델을 모두 유지한다.** 데이터가 답을 준다.
7. **논문 1의 투고를 수조/실선 때문에 지연시키지 않는다.**
8. **코드와 데이터를 공개한다.** 재현성이 신뢰도의 핵심.
