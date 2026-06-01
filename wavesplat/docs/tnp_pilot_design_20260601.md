# TNP Pilot Design — Pre-Training Investigation (2026-06-01)

## 0. 배경

### 검토 의견 (랩미팅 후속)
> "기존 시도는 모두 작은 갈래. EnKF, PINN 처럼 *큰 갈래* 가 필요"

### 이미 시도 + 실패한 *작은 갈래* 7가지
- DCSpecNet, USGO, Neural corrector, Stokes2, DU-RELAX, PINO/FNO/DeepONet, RELAX standalone
- 공통 실패: **"기존 iterative 파이프라인 위에 NN 덧붙이기"** — Adam optimizer 의 local minimum (nRMSE 0.22) 을 회피 못함
- 특히 **optimizer 자체의 NN 화 (USGO, DU-RELAX)** 도 실패 — *10 unrolled layers 가 500 Adam iter 를 못 따라감*

### Direction B 의 진짜 *큰 갈래* — Paradigm 전환

```
기존: iterative refinement (Adam 500 iter, per-instance optimization)
         ↓
큰 갈래: Amortized inference (한 번 학습 → 즉시 답, meta-learning)
```

이 paradigm 전환의 1순위 후보 = **TNP (Transformer Neural Process)**.

---

## 1. NP 계열 이론 흐름

| Year | Model | 핵심 기여 | 비고 |
|---|---|---|---|
| 2018a | **CNP** (Garnelo) | DeepSets encoder + mean aggregation, deterministic | 가장 simple baseline |
| 2018b | **NP** (Garnelo) | + global latent z (factorized Gaussian) | 불확실성 정량화 |
| 2019 | **ANP** (Kim) | Cross-attention fixes underfitting at context | Attention 도입 시작 |
| 2020 | **ConvCNP** (Gordon, Cambridge MLG) | SetConv + CNN → *translation equivariance* | Zero-shot OOD |
| 2022 | **TNP** (Nguyen & Grover) | Full transformer; SOTA but O(n²) | Scalability 문제 |
| 2023 | **LBANP** (Feng) | Latent bottleneck tokens → linear in N_c | Scalable variant |
| 2024 | **Gridded TNP / PT-TE-TNP** (Ashman, Cambridge MLG) | *Pseudo-token grid + Swin attention* for spatio-temporal scale | **본 검토 1순위** |

### NP 의 task 정의 (공통)
```
Task = (Context set, Target set)
  Context: {(x_c, y_c)}_i  — observations
  Target:  {x_t}            — query points
  Prediction: p(y_t | x_t, context)
Meta-learning: 여러 task 에서 학습 — task 분포 일반화
```

---

## 2. Gridded TNP 의 구조 (Ashman et al. 2024)

### 왜 "Gridded" 인가
- 일반 TNP 는 *(N_c + N_t)² self-attention* — weather-scale (10⁶ points) 에 infeasible
- *Regular pseudo-token grid* 를 encoder ↔ decoder 사이 *중간 representation* 으로 도입
- Attention 은 *grid cell 들 사이* 에서만 — *quadratic cost 제거*

### 3-stage Architecture
```
┌─ Off-grid context observations  ──────────────────┐
│   {(x_c, y_c, t_c, η_c)} — 50 LiDAR pts × T frames │
└──────────────┬────────────────────────────────────┘
               ↓
┌─ Grid Encoder (PT-GE) ─────────────────────────────┐
│  각 grid cell m 에 대해 k-NN context 점에 MHCA 적용 │
│  u_m ← MHCA(u_m⁰, {z_c,n}_{n ∈ N(v_m; k)})         │
│  Complexity: O(M · k)                              │
└──────────────┬────────────────────────────────────┘
               ↓
┌─ Processor (Swin attention on grid) ──────────────┐
│  Windowed self-attention over grid tokens          │
│  Spatial + temporal windows (ERA5 의 4×24×60 처럼) │
│  Complexity: O(M · w²) — quadratic cost 제거       │
└──────────────┬────────────────────────────────────┘
               ↓
┌─ OOTG Decoder ─────────────────────────────────────┐
│  Off-grid target 각각 k-nearest grid cell 로 MHCA  │
│  z_t ← MHCA(z_t⁰, {u_m}_{m ∈ N(x_t; k)})           │
│  Complexity: O(k · N_t)                            │
└──────────────┬────────────────────────────────────┘
               ↓
┌─ Output: p(η | x_q, y_q, t_q, context) ───────────┐
│  Gaussian likelihood: μ, σ                         │
└────────────────────────────────────────────────────┘
```

### Time 처리
*별도 temporal module 없음* — *4D 격자 (T × H × W × C)* 로 처리. Swin window 가 space + time 모두 cover.

### 검증
ERA5 (climate data, ~10⁶ points) 에서 TNP/ConvCNP 베이스라인 outperform.

### 오픈소스 (github.com/cambridge-mlg/gridded-tnp)
- **PyTorch + Lightning + Hydra** (NetCDF 기본)
- **PyTorch Geometric 의존성 없음** (위험 ↓)
- Config 으로 PT-GE / KI-GE / Swin-TNP variants 전환 가능
- 학습 script: `lightning_train.py`

---

## 3. WaveSplat ↔ Gridded TNP 매핑 설계

### 3.1 데이터 호환성 — *거의 직접 호환*

WaveSplat HDF5 구조 (`frames/`):
```
gt_grid_x:      (80,)            # x grid coords
gt_grid_y:      (60,)            # y grid coords
gt_wave_field:  (301, 60, 80)    # T × H × W ← Gridded TNP 의 ERA5 4D 와 동일 구조
lidar_points:   (301, 71, 3)     # T × N_max × (x, y, z)
lidar_valid_mask: (301, 71)      # valid flag
timestamps:     (301,)
```

→ ERA5 의 *station observations + reanalysis grid* 와 일대일 대응:
- Context = LiDAR points (sparse, off-grid)
- Target = GT grid (dense, on-grid)
- Grid = 80 × 60 = 4,800 cells, T = 301 frames

### 3.2 Task 정의 (3 options)

| Option | Task = | 장점 | 단점 |
|---|---|---|---|
| **A. Single-frame** | 한 frame (50 pts → 4,800 grid) | 가장 단순 | Temporal info 무시 — multi-frame 우위 못 살림 |
| **B. Short window** | 3-5 frames (150-250 pts → 4,800×N grid) | WaveSplat 의 multi-frame 과 부합 | 중간 복잡도 |
| **C. Full episode** | 301 frames (15,000 pts → 1.4M grid) | 가장 풍부한 temporal | OOM 위험, single episode 의 task variety 없음 |

**권장: B (Short window, 3-5 frames)**
- WaveSplat 의 init window (multi-frame) 와 같은 paradigm
- ERA5 의 4×24×60 4D grid 와 유사 (4 frame × 60 × 80 ≈ 19,200 cells)
- Swin window size 와 잘 맞음

### 3.3 Train / Test Split 전략

```
Dataset: 110 episodes
  ├─ Train: 70 episodes (BF2-BF5 의 main sweep, seed 0-3)
  ├─ Val:   20 episodes (seed 4 of BF2-BF5)
  └─ Test:  20 episodes (BF6-BF7 + held-out seeds, OOD)

각 episode 에서:
  Task = (3-frame window) — 한 episode 에서 약 300 / 3 = 100 tasks
  Train 총 tasks ≈ 70 × 100 = 7,000 (meta-learning 에 충분)
```

**Data leakage 방지**:
- 같은 episode 의 frame 들을 train/val/test 에 동시에 두면 안 됨
- *Episode level split* 으로 *완전 분리*

---

## 4. Architecture 선택

### 비교

| Architecture | Pros | Cons | 선택? |
|---|---|---|---|
| Vanilla TNP | 가장 단순 | O((N_c + N_t)²) — 4,800 target 에 OOM | ✗ |
| ConvCNP | Translation equivariance | Grid-based, sparse irregular obs 처리 한계 | ✗ |
| LBANP | Linear in N_c | Latent bottleneck 정보 손실 가능 | △ |
| **Gridded TNP (PT-GE + Swin)** | *Scalable + spatio-temporal natively* | 구현 복잡도 (Cambridge code 학습 곡선) | **✓ 1순위** |
| BSA-TNP | 1M+ points scalable | Overkill (우리는 50 × 5 = 250 pts) | △ |

### Gridded TNP 선택 이유

1. *데이터 구조 일치* — ERA5 4D = (T, H, W, C) ≈ WaveSplat 4D
2. *Sparse irregular context* 가 PT-GE 의 강점 — LiDAR 50pts 와 정확 일치
3. *Dense grid target* 도 자연 처리 — OOTG decoder 생략 가능 (grid 자체가 target)
4. *Spatio-temporal Swin* — multi-frame coherence 자동 학습
5. *오픈소스 + Lightning* — 빠른 prototyping

### Specific config (시작점)

```yaml
grid_size: [4, 16, 16]   # T × H × W (downsampled from 4 × 60 × 80)
embed_dim: 128           # token embedding
n_heads: 4
n_processor_layers: 4    # Swin attention 깊이
window_size: [4, 4, 4]   # Swin window
grid_encoder: PT-GE      # pseudo-token, k=8 nearest context
grid_decoder: skip       # target = grid 자체, OOTG 생략
```

---

## 5. Physics 통합 — *Dispersion 어떻게 넣을까*

이게 *PhD novelty 의 핵심*. NP 분야에 *dispersion-aware NP* 가 0편이라 우리 contribution.

### Option A: PINN-style PDE Residual Loss
```
Loss = L_NLL + λ · L_dispersion
  L_NLL = Gaussian likelihood at target points
  L_dispersion = ‖∂_tt η − c²(k)·∇²η‖²
    where c(k) = ω/k = √(g·tanh(k·d)/k)
```
- **장점**: 구현 직관적, gradient 직접 backprop
- **단점**: *soft constraint* — 위반 가능. WaveSplat 의 hard constraint 보다 약함
- 참고: Ehlers 2025 의 PFT-PINN

### Option B: Spectral Projection Layer (post-processing)
```
Decoder output η̂(x, y, t)
        ↓
FFT2D + dispersion filter ω(k)
        ↓
Inverse FFT  → η̂_physics
```
- **장점**: *Hard constraint* 유지 (분산관계 위반 0)
- **단점**: 미분 가능성 확보 필요 (FFT 는 가능)
- WaveSplat 의 dispersion 과 같은 강도

### Option C: Spectral-Parameterized Latent (hybrid)
- Latent tokens 을 *(A_n, k_xn, k_yn, φ_n) 형태로 강제*
- Decoder 가 wavesplat-style η(x, y, t) = Σ A·cos(...) 생성
- ω 는 dispersion 으로 자동
- **장점**: WaveSplat 의 hard constraint + NP 의 amortized inference 결합
- **단점**: Decoder 가 fixed function — TNP 의 expressivity 일부 손실

### **권장: Option A (Pilot) → Option B/C (full study)**
- Pilot 에서는 *Option A 가 가장 빠른 구현*
- Loss = L_NLL + λ·L_dispersion 만 추가
- Pilot 성공 후 Option B/C 로 고도화

---

## 6. 데이터 흐름 설계

### 6.1 HDF5 → PyTorch Dataset

```python
class WaveSplatTNPDataset(torch.utils.data.Dataset):
    def __init__(self, episodes, window=3, stride=1):
        self.tasks = []
        for ep in episodes:
            n_frames = len(ep['timestamps'])
            for t0 in range(0, n_frames - window, stride):
                self.tasks.append((ep, t0, window))

    def __getitem__(self, idx):
        ep, t0, w = self.tasks[idx]
        # Context: LiDAR points in [t0, t0+w]
        context_x, context_y, context_t, context_eta = self._collect_lidar(ep, t0, w)
        # Target: GT grid (정해진 격자)
        target_grid = ep['gt_wave_field'][t0:t0+w]  # (w, 60, 80)
        target_x, target_y, target_t = self._grid_coords(ep, t0, w)
        return dict(
            x_context=context_x, y_context=context_eta,
            x_target=target_grid_coords, y_target=target_grid_values,
        )
```

### 6.2 Coordinate normalization
- (x, y) → [-1, 1] 정규화 (grid 범위 기준)
- t → [0, T_max] / T_max (episode 시작 기준)
- η → standardize (mean 0, std 1) — episode 별 std 로

### 6.3 Batch 구성
- Batch size 16~32 tasks
- 각 task 의 context 크기 가변 (50~250 pts) → padding + mask
- Target 크기 고정 (4,800 × w)

### 6.4 Sanity check
- Toy task: same-episode train/val → overfit 확인 (loss 떨어지는가?)
- 그 후 cross-episode generalization

---

## 7. Pilot 실험 설계 (Go/No-Go)

### Stage 1: Setup (Day 1-2)
- [ ] Gridded TNP repo clone + dependency 설정
- [ ] Lightning + Hydra 환경 검증 (`lightning_train.py` 토이 데이터로)
- [ ] HDF5 → PyTorch Dataset 변환 코드 작성
- [ ] 1 episode 의 task list 생성 (~100 tasks)

**Go/No-go #1**: 토이 데이터로 학습 돌아가는가?

### Stage 2: Overfit Test (Day 3)
- [ ] 1 episode 의 50 task 로 *overfit* 시도 (validation = train)
- [ ] Loss 가 떨어지는가? (NLL → 0 근처)
- [ ] 학습된 모델로 reconstruction 시각화

**Go/No-go #2**: TNP 가 우리 데이터를 *학습 가능* 한가?

### Stage 3: Generalization (Day 4-5)
- [ ] 3 episodes 사용: 2 train, 1 test
- [ ] Train task → val task generalization
- [ ] Test episode 에서 nRMSE 측정
- [ ] WaveSplat baseline 과 비교

**Go/No-go #3**: TNP 가 *meta-learning 가능* 한가?

### Stage 4: Physics Integration (Day 6-7)
- [ ] Option A (dispersion PDE residual loss) 추가
- [ ] λ tuning (0.0 vs 0.1 vs 1.0)
- [ ] Test episode 에서 nRMSE 재측정

**최종 판단**: 본격 진행 vs DKL-GP fallback

---

## 8. 예상 Challenge + Mitigation

| 위험 | 가능성 | Mitigation |
|---|---|---|
| Cambridge code 학습 곡선 | HIGH | Lightning + Hydra 익숙해지기, 2일 buffer |
| Context/target split 설계가 wave 도메인 부적합 | MED | 3가지 option 사전 준비 (A/B/C) |
| OOM (4,800 target × batch) | MED | Grid 16×16 으로 downsample, batch 줄임 |
| Train/test data leakage | MED | Episode level split 엄수 |
| Hyperparameter tuning 시간 과다 | HIGH | 기본 config 부터 시작, grid search 최소화 |
| TNP 가 학습 안 됨 (Go/No-go #2 실패) | LOW-MED | Vanilla TNP 로 sanity, ConvCNP fallback |
| Generalization 안 됨 (#3 실패) | MED-HIGH | DKL-GP 로 fallback |

---

## 9. 결론

### 핵심 발견
1. *Optimizer NN 응용 (USGO, DU-RELAX)* 은 *작은 갈래* — iterative paradigm 유지로 한계 명확
2. *큰 갈래* = paradigm 전환 = **TNP (Transformer Neural Process)**
3. *Gridded TNP* 가 WaveSplat 데이터 (ERA5 와 같은 4D 격자) 와 *완벽 호환*
4. *NP 분야에 dispersion-aware physics 통합 = 0편* → PhD contribution

### 학습 전까지의 단계 (Pre-Training)

| 단계 | 작업 | 예상 시간 |
|---|---|---|
| 1 | Repo + dependency 설정 | 0.5일 |
| 2 | 코드 구조 파악 | 0.5일 |
| 3 | HDF5 → PyTorch Dataset 변환 | 1일 |
| 4 | Toy sanity test (학습 돌아가는지) | 0.5일 |
| 5 | Single-episode task list 생성 | 0.5일 |
| **합계** | **Pre-training 준비** | **3일** |

### 추천 다음 행동
1. 본 보고서 review + 사용자 승인
2. Stage 1 시작 (Gridded TNP repo clone)
3. Stage 2 Go/No-go 후 본격 진행 결정

### Fallback Plan
- TNP 실패 시: **DKL-GP** (Sound Field DKL, arXiv:2407.04417 기반)
- DKL-GP 도 실패 시: **SIREN + Dispersion PDE** (per-scene, 데이터 불필요)
