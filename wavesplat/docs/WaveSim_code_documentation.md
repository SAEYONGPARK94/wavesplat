# WaveSim 코드 문서

## 모듈 구조 및 구현 현황 | 2026.03.30

---

## 1. 전체 구조

```
wavesplat/
├── wavesim/                      # 시뮬레이션 패키지
│   ├── __init__.py
│   ├── wave_generator.py         # Gerstner wave + JONSWAP 스펙트럼
│   ├── lidar_simulator.py        # LiDAR 빔 생성 + Ray-mesh intersection
│   ├── raydrop_uniform.py        # Model A: Fresnel/거리 기반 dropout
│   ├── raydrop_slope.py          # Model B: Specular backscatter dropout
│   ├── gt_generator.py           # Ground truth 파 필드 생성
│   ├── noise_model.py            # 센서 노이즈 모델
│   ├── data_writer.py            # HDF5 저장/로드
│   ├── pipeline.py               # 엔드투엔드 파이프라인
│   └── tests/
│       ├── __init__.py
│       ├── test_wave_generator.py   # 10 tests
│       ├── test_raycast.py          # 8 tests
│       ├── test_raydrop.py          # 9 tests
│       ├── test_gt.py               # 8 tests
│       └── test_pipeline.py         # 4 tests
├── wavesplat/                    # 추정 알고리즘 (Phase 3, 미구현)
├── experiments/
├── data/                         # 생성된 HDF5 에피소드
└── docs/
    ├── WaveSplat_개발가이드.md
    └── WaveSim_코드문서.md
```

**테스트 현황: 39/39 통과**

---

## 2. 모듈별 상세

### 2.1 `wave_generator.py`

파랑 스펙트럼 생성 및 Gerstner wave 중첩.

#### 주요 클래스 / 함수

**`WaveComponent`** (dataclass)
```python
@dataclass
class WaveComponent:
    A: float      # 진폭 [m]
    omega: float  # 각주파수 [rad/s]
    kx: float     # 파수 x성분 [rad/m]
    ky: float     # 파수 y성분 [rad/m]
    phi: float    # 초기 위상 [rad]
    Q: float      # Gerstner steepness 파라미터
```

**`full_dispersion(k, depth)`**
- 완전 선형 분산 관계: `ω = √(g·k·tanh(k·d))`
- 깊은 물 근사 금지 — 항상 `tanh` 포함

**`wavenumber_from_omega(omega, depth)`**
- 역 분산 관계: Newton-Raphson iteration으로 `k` 계산
- Roundtrip 정확도: `|ω_back - ω| < 1e-6` 보장

**`single_gerstner(components, X, Y, t)`**
- 다중 성분 Gerstner wave 중첩
- 반환: `(X_disp, Y_disp, Z, nx, ny, nz)` — 변위 + 수면 법선
- Q=0이면 순수 sinusoidal, 단일 성분 RMSE < 1e-10 m (해석해 비교)

**`check_q_constraint(components)`**
- 입자 교차 방지 조건 검증: `ΣQᵢ·ωᵢ·Aᵢ ≤ 1`
- 위반 시 스케일 다운 자동 적용

**`jonswap_components(Hs, Tp, n_components, ...)`**
- JONSWAP 스펙트럼 (Hasselmann et al. 1973)에서 성분 생성
- 진폭 정규화: `Hs = 4·√(Σ Aᵢ²/2)` 정확도 보장
- 방향 spreading: Gaussian (기본 σ=30°)
- Q 파라미터: 최대 허용값의 50%로 보수적 설정

#### 검증된 물리 (Trust Chain)

| 검증 항목 | 방법 | 결과 |
|----------|------|------|
| 분산 관계 정확도 | 깊은 물/얕은 물 해석해 비교 | < 0.01% 오차 |
| 역 분산 roundtrip | `ω → k → ω` | < 1e-6 rad/s |
| 단일 성분 해석해 | RMSE vs analytic | < 1e-10 m |
| 수면 법선 방향 | nz > 0 전역 확인 | ✓ |
| Q 제약 | 모든 시나리오 | ΣQωA ≤ 1 ✓ |
| JONSWAP Hs 정확도 | 5개 seed 평균 | < 20% 오차 |

---

### 2.2 `lidar_simulator.py`

전방 조사 LiDAR 빔 생성 및 파 표면과의 교차점 계산.

#### 주요 클래스 / 함수

**`LidarConfig`** (dataclass)
```python
@dataclass
class LidarConfig:
    height: float = 5.0           # 센서 높이 [m]
    channels: int = 32            # 수직 채널 수
    scan_rate: float = 10.0       # 스캔 속도 [Hz]
    fov_h_deg: float = 120.0      # 수평 FOV [°]
    fov_v_min_deg: float = -5.0   # 수직 최소 각도 [°]
    fov_v_max_deg: float = 5.0    # 수직 최대 각도 [°]
    range_min: float = 10.0       # 최소 탐지 거리 [m]
    range_max: float = 200.0      # 최대 탐지 거리 [m]
    n_h_beams: int = 360          # 수평 빔 수
```

**`build_mesh(X, Y, Z)`**
- 파 표면 그리드 → 삼각형 메시 변환
- 각 그리드 셀에 2개 삼각형 생성

**`raycast_surface(ray_origins, ray_dirs, vertices, triangles, ...)`**
- Möller-Trumbore 알고리즘 벡터화 구현 (Möller & Trumbore 1997)
- 범위 필터 `[range_min, range_max]` 적용
- 반환: `(hit_mask, hit_points, hit_dist)`

**`generate_rays(lidar_cfg, ship_x, ship_y, ship_heading_deg)`**
- 선박 위치/방향 기반 빔 패턴 생성
- `N_rays = channels × n_h_beams`개의 단위 벡터 반환

**`compute_incidence_angles(hit_points, ray_dirs, surface_normals)`**
- 입사각 `θ_i = arccos(|ray_dir · normal|)` 계산
- Grazing angle `θ_g = π/2 - θ_i` 동시 반환

**`interpolate_normals(hit_points, X, Y, NX, NY, NZ)`**
- 쌍선형 보간으로 hit point에서 수면 법선 추정
- `scipy.interpolate.RegularGridInterpolator` 사용

#### 검증 항목

| 테스트 | 결과 |
|-------|------|
| 수직 빔 → 평면 교차 위치 정확도 | |z_hit| < 0.5m ✓ |
| 상향 빔 → miss | ✓ |
| range_min 필터 | ✓ |
| 빔 방향 단위벡터 확인 | ‖d‖ = 1.0 ± 1e-10 ✓ |
| 법선 입사각 θ_i = 0° 확인 | ✓ |

---

### 2.3 `raydrop_uniform.py` (Model A)

**물리 모델**: 거리 · Fresnel · Lambertian 기반 균일 dropout

```
P_return = η · R_fresnel(θ_i) · cos(θ_i) / (d/d_ref)²
```

- `η`: 센서 효율 (calibration-free: default=1)
- `R_fresnel`: Fresnel 반사율 (물, n=1.321 @1550nm, Seyond Falcon K1)
- `cos(θ_i)`: Lambertian factor
- `d²`: 역제곱 거리 감쇄

**`fresnel_reflectance(theta_i, n=1.321)`**
- s/p 편광 평균
- θ=0°: R≈2%, θ=89°: R>80%
- 단조 증가 검증 ✓

**`apply_dropout(..., target_n_pts)`**
- P를 가중치로 importance sampling
- 정확히 `target_n_pts`개 반환 보장

---

### 2.4 `raydrop_slope.py` (Model B)

**물리 모델**: 파 경사 기반 specular backscatter dropout

- 입사 빔의 반사 방향이 센서를 향하는지 판정
- 조건: `dot(reflected_ray, -ray_dir) ≥ cos(α_threshold)`
- `α_threshold = 5°` (default)

**`specular_alignment(ray_dirs, surface_normals, alpha_threshold_deg=5.0)`**
- 반사 방향 계산: `r = d - 2(d·n)n`
- alignment 값 반환 (높을수록 센서를 향함)

**`apply_dropout(...)`**
- Specular 후보가 없을 경우 fallback: 균일 랜덤 샘플링
  - 물리적 근거: diffuse scattering이 항상 일부 return 제공
- 후보 있을 경우: Fresnel·1/d²·alignment 가중치 sampling

#### 두 모델 비교

| 특성 | Model A (uniform) | Model B (slope) |
|-----|-------------------|-----------------|
| 포인트 분포 | 전방향 균일 | 파봉 집중 |
| 메커니즘 | Diffuse scattering | Specular backscatter |
| 정수면 return | 있음 (약함) | 없음 (fallback 제외) |
| 매개변수 | η, d_ref | α_threshold, d_ref |

---

### 2.5 `gt_generator.py`

Ground truth 파 필드를 정규 그리드에서 평가.

**`make_grid(x_range, y_range, nx, ny)`**
- `(ny, nx)` meshgrid 생성

**`evaluate_field(components, X, Y, t)`**
- Gerstner 중첩으로 Z, NX, NY, NZ 계산

**`compute_spectrum_1d(components, n_freq)`**
- 성분의 에너지를 주파수 빈으로 집계
- 반환: `(n_freq, 2)` — [ω, S(ω)]

---

### 2.6 `noise_model.py`

**`add_noise(hit_points, ray_dirs, sigma_range=0.02, sigma_angle_deg=0.05)`**
- Range noise: ray 방향으로 Gaussian 교란 (σ=2cm)
- Angular noise: ray 법선 방향으로 Gaussian 교란 (σ=0.05°)
- 수직 빔 처리: `|perp| < 1e-6`일 때 대체 기저 벡터 사용

**`compute_confidence(incidence_angles, distances, d_ref=50.0)`**
- `c = cos(θ_i) / (d/d_ref)` — 입사각·거리 기반 신뢰도
- 범위: (0, 1]

---

### 2.7 `data_writer.py`

**HDF5 파일 구조**

```
episode_BF4_30comp_seed42_uniform.h5
├── /config/
│   ├── attrs: sea_state, Hs, Tp, gamma, n_components, depth
│   ├── wave_components: (N_comp, 6)  [A, omega, kx, ky, phi, Q]
│   └── lidar_config/: height, channels, scan_rate
├── /frames/
│   ├── timestamps:              (N_frames,)
│   ├── lidar_points:            (N_frames, max_pts, 3)   [gzip]
│   ├── lidar_valid_mask:        (N_frames, max_pts)       [gzip]
│   ├── lidar_incidence_angles:  (N_frames, max_pts)       [gzip]
│   ├── lidar_confidence:        (N_frames, max_pts)       [gzip]
│   ├── gt_wave_field:           (N_frames, Ny, Nx)        [gzip]
│   ├── gt_grid_x:               (Nx,)
│   └── gt_grid_y:               (Ny,)
├── /raydrop_model/
│   └── attrs: model_type, parameters
└── /gt_params/
    ├── attrs: Hs_true, Tp_true
    └── spectrum_1d: (N_freq, 2)
```

Round-trip 무결성 검증: 저장/로드 후 GT 파 필드 값 정확 일치 ✓

---

### 2.8 `pipeline.py`

엔드투엔드 파이프라인. WaveComponent 생성부터 HDF5 저장까지.

#### 해상 상태 사전 정의

| 코드 | Hs [m] | Tp [s] |
|------|--------|--------|
| BF2 | 0.3 | 4.0 |
| BF3 | 0.6 | 5.5 |
| BF4 | 1.2 | 7.0 |
| BF5 | 2.0 | 9.0 |

#### 실행 방법

```bash
# 기본 실행 (BF4, 30성분, 50pts/frame)
python3 -m wavesim.pipeline --sea_state BF4 --n_components 30 --n_pts 50 --seed 42

# 전체 옵션
python3 -m wavesim.pipeline \
    --sea_state BF4 \
    --n_components 30 \
    --n_pts 50 \
    --seed 42 \
    --raydrop uniform \   # or 'slope'
    --duration 30.0 \     # 30초 에피소드
    --dt 0.1 \            # 10Hz
    --output_dir data
```

#### 파이프라인 흐름

```
jonswap_components()
        ↓
check_q_constraint()  ← 항상 검증
        ↓
[프레임 루프]
  evaluate_field()       → GT wave field (80×60 grid)
  build_mesh()           → 삼각형 메시 (40×30)
  generate_rays()        → (N_rays, 3)
  raycast_surface()      → hit_mask, hit_pts, hit_dist
  interpolate_normals()  → 수면 법선
  compute_incidence_angles()
  apply_dropout()        → Model A or B
  add_noise()
  compute_confidence()
        ↓
save_episode()  → HDF5
```

#### 실행 결과 예시 (BF4, 30성분, 50pts)

```
[Wave] BF4: Hs=1.2m, Tp=7.0s, 30 components, Q-sum=0.950
  [t=0.0s] 50 LiDAR points, GT Hs=1.270m
  [t=0.1s] 50 LiDAR points, GT Hs=1.272m
  ...
[Saved] data/episode_BF4_30comp_seed42_uniform.h5
  Frames: 11, Pts/frame: 50, Grid: (60, 80)
```

---

## 3. 테스트 실행

```bash
cd /home/saeyong/reserch/wavesplat

# 전체 테스트
python3 -m pytest wavesim/tests/ -v

# 모듈별 실행
python3 -m pytest wavesim/tests/test_wave_generator.py -v
python3 -m pytest wavesim/tests/test_raycast.py -v
python3 -m pytest wavesim/tests/test_raydrop.py -v
python3 -m pytest wavesim/tests/test_gt.py -v
python3 -m pytest wavesim/tests/test_pipeline.py -v
```

**테스트 결과 (2026.03.30)**

| 파일 | 테스트 수 | 결과 |
|------|---------|------|
| test_wave_generator.py | 10 | 10 passed |
| test_raycast.py | 8 | 8 passed |
| test_raydrop.py | 9 | 9 passed |
| test_gt.py | 8 | 8 passed |
| test_pipeline.py | 4 | 4 passed |
| **합계** | **39** | **39 passed** |

---

## 4. Python 의존성

```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
h5py>=3.8
pyyaml>=6.0
pytest>=7.0
open3d>=0.17    # 선택 (현재 순수 NumPy 구현 사용)
torch>=2.0      # wavesplat/ 알고리즘 구현 시 필요
scikit-learn>=1.2  # GPR baseline 구현 시 필요
```

설치:
```bash
pip install numpy scipy matplotlib h5py pyyaml pytest open3d
```

---

## 5. 구현 상태 및 다음 단계

### 완료 (Phase 1-B)
- [x] `wave_generator.py` — Gerstner + JONSWAP + Q 제약
- [x] `lidar_simulator.py` — Ray-mesh intersection + 법선 보간
- [x] `raydrop_uniform.py` — Model A
- [x] `raydrop_slope.py` — Model B
- [x] `gt_generator.py` — GT 파 필드
- [x] `noise_model.py` — 센서 노이즈
- [x] `data_writer.py` — HDF5 저장/로드
- [x] `pipeline.py` — 엔드투엔드 파이프라인
- [x] 39개 단위/통합 테스트

### 진행 예정
- [ ] Phase 2: WaveSim 검증 + 75개 에피소드 데이터셋 생성
- [ ] Phase 3: WaveSplat 추정 알고리즘 (`wavesplat/` 패키지)
  - `model/spectral_gaussian.py`
  - `optimizer/sequential_update.py`
  - `baselines/gpr_baseline.py`, `etkf_grid.py`
- [ ] Phase 4: 시뮬레이션 실험 + 논문 작성

---

## 6. 알려진 한계 및 주의사항

1. **Raycast 속도**: 순수 NumPy 구현으로 BVH 없음 → 대규모 메시에서 느림. 30초 에피소드 전체 생성에 수 분 소요 예상. Open3D raycasting으로 대체 가능.

2. **Slope 모델 fallback**: BF2~3 등 낮은 해상에서 specular 후보가 없을 경우 균일 샘플링으로 자동 전환. 이 동작을 ablation에서 명시할 것.

3. **Q 파라미터**: 각 성분에 최대 허용값의 50%로 설정 (보수적). 더 큰 Q는 더 뾰족한 파봉을 생성하나 particle crossing 위험 증가.

4. **선박 이동 미구현**: 현재 `ship_x=0, ship_y=0` 고정. S11~S15 시나리오(직진 선박)는 추후 `pipeline.py`에 이동 로직 추가 필요.
