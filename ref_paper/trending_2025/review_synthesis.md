# Trending Network Survey — Review & Synthesis for WaveSplat PhD
Date: 2026-05-19

## Evaluation Criteria
1. **Problem fit**: Sparse LiDAR point cloud → 2D+t wave field (spectral estimation)
2. **Advisor fit**: SNU 이신형 교수 — CFD, towing tank, ICNN, Bayesian opt, surrogate modeling
3. **PhD novelty**: Must go beyond "apply existing method to new domain"
4. **DCSpecNet lessons**: Non-convex loss failed, generalization gap, beamforming baseline is strong
5. **WaveSplat context**: Grid-free spectral Gaussians, dispersion hard constraint, w_phi tracking

---

## 1. FR-Mamba (2025) — Dual-Branch FNO1d + Mamba SSM

### Summary
Sparse sensor → full field reconstruction via dual-branch architecture. FNO1d handles spatial correlations, Mamba SSM handles temporal dynamics. Gating fusion merges both branches. Tested on cylinder flow (Re=100).

### Strengths
- **직접적 문제 매칭**: sparse sensor → dense field = 우리 문제와 동일 구조
- MAE 0.31e-4 (8개 방법 중 최고), 시간 간격 T1-T5에서 일관된 성능
- Mamba SSM: O(L) 시간 복잡도 (Transformer의 O(L²) 대비)
- FNO1d로 spatial, Mamba로 temporal — 역할 분리가 명확

### Weaknesses / WaveSplat 적용 한계
- **Regular grid 필수**: FNO는 고정 격자에서 동작 → WaveSplat의 grid-free 철학과 충돌
- **센서 위치 고정**: 16개 fixed sensor 가정. LiDAR point cloud는 매 프레임 위치가 변함
- **2D flow만 검증**: cylinder wake (192×112 grid). Ocean wave의 spectral structure와 다름
- **Physics 미포함**: 순수 data-driven, dispersion constraint 없음

### WaveSplat 적용 가능성: ★★☆☆☆
FNO의 regular grid 요구사항이 근본적 한계. LiDAR의 불규칙 sparse input을 처리하려면 아키텍처 대폭 수정 필요. Mamba의 temporal modeling은 유용하지만, 단독으로는 novelty 부족.

### 가능한 활용
- Mamba SSM을 WaveSplat의 sequential tracking에 적용 (w_phi update를 SSM으로)
- 하지만 이는 component-level 차용이지 전체 방법론이 아님

---

## 2. KANO (ICLR 2026) — Kolmogorov-Arnold Neural Operator

### Summary
KAN 기반 neural operator. Pseudo-differential symbol p(x,ξ)를 spatial+spectral 양쪽에서 KAN으로 parameterize. FNO의 한계(position-dependent dynamics에서 super-exponential scaling)를 이론적으로 극복.

### Strengths
- **이론적 기반 탄탄**: Theorem 1 (FNO curse), Theorem 2 (KANO polynomial scaling)
- **극한 효율**: 152 params로 FNO 566K와 동등 이상 성능
- **Symbolic recovery**: 4th decimal까지 수식 복원 가능
- **Dual-domain**: spatial + spectral 동시 처리 → wave estimation에 자연스러움
- ICLR 2026 — 최신 top venue

### Weaknesses / WaveSplat 적용 한계
- **1D PDE만 검증**: Burgers, advection, Helmholtz. 2D+t ocean wave는 미검증
- **Fixed grid input**: neural operator의 표준 한계 (regular grid 가정)
- **KAN 자체의 불안정성**: B-spline extrapolation 문제, 학습 속도 느림
- **구현 복잡도**: pseudo-differential symbol + Kohn-Nirenberg quantization → 구현 난이도 높음
- **확장성 미지수**: 152 params가 complex wave field에 충분한지 불확실

### WaveSplat 적용 가능성: ★★★☆☆
Dual-domain (spatial+spectral) 철학은 wave estimation에 적합. 하지만 fixed grid 한계와 2D+t 미검증이 리스크. "KANO for ocean waves"는 논문 하나 될 수 있으나, 구현 난이도가 높고 novelty가 "application" 수준에 머물 위험.

### 가능한 활용
- KAN의 symbolic recovery 능력을 dispersion relation 학습에 활용
- Pseudo-differential operator로 wave propagation을 모델링
- WaveSplat optimizer의 spectral parameter update를 KANO로 대체

---

## 3. KAN (ICLR 2025) — Kolmogorov-Arnold Networks

### Summary
MLP의 fixed activation + learnable weights를 뒤집음: learnable activation (B-spline) on edges + summation on nodes. Scaling law α=k+1=4 (cubic spline) vs MLP α≈1. Grid extension, sparsification, symbolification pipeline.

### Strengths
- **Interpretability**: sparsification → pruning → symbolification으로 수식 추출 가능
- **Scaling law 우위**: α=4 vs MLP α=1 — 작은 모델로 높은 정확도
- **Grid extension**: coarse→fine progressive refinement (WaveSplat의 coarse-to-fine과 유사!)
- **Science applications**: PDE fitting, symbolic regression에 강점

### Weaknesses / WaveSplat 적용 한계
- **Small-scale only**: 큰 모델에서는 MLP 대비 이점 불명확 (ICLR reviewers도 지적)
- **학습 느림**: B-spline 계산 비용, MLP 대비 10x 느림
- **기본 building block**: 자체로는 연구 주제가 아닌 도구
- **Extrapolation 약함**: B-spline은 grid 밖에서 불안정
- **이미 2025년 논문**: 2026년 PhD에서 KAN 적용만으로는 시의성 부족

### WaveSplat 적용 가능성: ★★☆☆☆
단독 연구 주제로는 부적합. "KAN for wave estimation" = 도구 적용 수준. 하지만 다른 방법론의 sub-component로 KAN을 사용하는 것은 유효 (e.g., KANO에서 이미 사용).

### 가능한 활용
- Dispersion relation의 symbolic discovery
- WaveSplat optimizer 내부의 MLP를 KAN으로 교체 (minor contribution)

---

## 4. VIRSO (2026) — Graph Neural Operator for Sparse-to-Dense

### Summary
Graph 기반 neural operator. Sparse boundary → dense interior field. Spectral GNN (graph Fourier convolution) + spatial GNN (aggregation) collaboration. V-KNN으로 mesh density adaptive graph 구성. Edge deployable (Jetson Nano).

### Strengths
- **Irregular geometry 지원**: graph 기반이므로 grid-free 가능!
- **Sparse-to-dense**: 정확히 우리 문제 구조 (sparse LiDAR → dense wave field)
- **Spectral + spatial collaboration**: wave의 spectral 특성과 잘 맞음
- **V-KNN**: sensor density에 따라 adaptive graph — LiDAR의 불균일 분포에 적합
- **Edge deployment**: 실시간 선박 탑재 가능성 (Jetson Nano에서 동작)
- **EDP 10.1 J·ms**: 에너지 효율 우수

### Weaknesses / WaveSplat 적용 한계
- **Static field만 검증**: 정상상태 열전달, LDC flow. Time-varying wave에 미검증
- **Physics constraint 미포함**: 순수 data-driven, dispersion 미적용
- **Nuclear engineering 도메인**: ocean wave와 physics가 상이
- **Graph construction overhead**: V-KNN 매 프레임 재구성 필요 → latency
- **PhD novelty**: "VIRSO를 wave에 적용" = domain transfer 수준

### WaveSplat 적용 가능성: ★★★☆☆
Grid-free + sparse-to-dense는 문제 구조가 가장 잘 맞음. 하지만 temporal dynamics 미지원과 physics constraint 부재가 한계. Graph-based approach에 dispersion constraint를 결합하면 novelty 가능성 있음.

### 가능한 활용
- Graph neural operator + dispersion hard constraint 결합
- V-KNN adaptive graph를 LiDAR point cloud에 적용
- 하지만 time-varying 처리를 위해 추가 아키텍처 필요

---

## 5. TTT Done Right (2025) — Test-Time Training with Large Chunks

### Summary
Inference 시점에 fast weights를 self-supervised loss로 update하는 패러다임. SwiGLU-MLP fast weights + Muon optimizer. NVS (3D Gaussian Splatting), language modeling, video diffusion에 적용.

### Strengths
- **3D Gaussian Splatting 연결**: WaveSplat이 spectral Gaussians 사용 → 직접 연관
- **Test-time adaptation**: 새 episode마다 online adaptation → wave 특성 변화에 대응
- **O(1) state/compute**: 메모리/계산량 고정
- **Self-supervised**: ground truth 없이 학습 가능 (PINO wave와 유사 철학)
- **312M params, 38.7 FPS**: 실시간 가능

### Weaknesses / WaveSplat 적용 한계
- **NVS 문제와 구조 상이**: 3DGS의 multi-view → novel view ≠ sparse LiDAR → wave field
- **Fast weight update의 안정성**: wave estimation에서 발산 가능성
- **구현 복잡도**: Muon optimizer, context parallelism, gradient sharding → 구현 난이도 높음
- **Physics 미고려**: 순수 학습 기반
- **PhD scope**: TTT 자체가 연구 주제 — wave estimation에 TTT 적용은 side contribution

### WaveSplat 적용 가능성: ★★☆☆☆
TTT의 online adaptation 아이디어는 매력적이지만, wave estimation의 핵심 문제(spectral estimation + dispersion constraint)와 직접 연결이 약함. "TTT for wave estimation"은 너무 broad한 주제.

### 가능한 활용
- WaveSplat의 sequential update를 TTT 프레임워크로 재해석
- Episode-level adaptation (meta-learning 관점)

---

## 6. PINO Wave (Ehlers 2025) — Physics-Informed Neural Operator for Ocean Waves

### Summary
FNO backbone에 FSBC (free surface boundary condition) physics loss를 적용. Sparse buoy/radar → full wave field reconstruction. Ground truth 불필요. 1D+t only (long-crested). Hamburg UT + TU Berlin + Imperial College.

### Strengths
- **정확히 같은 문제**: sparse sensor → ocean wave field reconstruction
- **No ground truth**: physics loss만으로 학습 → 실측 데이터에 바로 적용 가능
- **FSBC physics**: Eqs 1-2 (kinematic + dynamic FSBC) — 해양공학적으로 rigorous
- **Fast inference**: 0.014s
- **도메인 전문성**: 해양파 전문 그룹 (Ehlers, Stender, Hoffmann)

### Weaknesses / WaveSplat 적용 한계
- **1D+t only**: 2D+t 미구현 (long-crested만). 우리는 2D+t 필수
- **FNO backbone**: regular grid 필요 → sparse LiDAR 불규칙 input과 충돌
- **경쟁자 논문**: Ehlers group이 이 분야 선점. 따라가면 2nd mover
- **코드 미공개**: 재현 불가
- **HOSM 합성 데이터**: 실제 해상 검증 없음

### WaveSplat 적용 가능성: ★★★★☆ (가장 높음, 하지만 주의 필요)
문제 자체가 동일하므로 적용성 최고. 하지만 **직접 경쟁** 관계이므로 차별화가 핵심:
- Ehlers: 1D+t, FNO, fixed grid, buoy/radar
- WaveSplat: 2D+t, grid-free, spectral Gaussian, LiDAR
이 차이가 곧 novelty.

### 가능한 활용 (핵심!)
- **FSBC physics loss**를 WaveSplat에 도입 (현재 WaveSplat은 observation MSE만 사용)
- WaveSplat의 grid-free spectral Gaussian이 FNO보다 ocean wave에 더 적합한 이유를 이론적으로 증명
- 2D+t extension이 Ehlers의 future work → 선점 가능
- **Positioning**: "PINO Wave의 2D+t grid-free extension with hard dispersion constraint"

---

## 7. Spectral Bias in PINNs (2025) — Karniadakis Group

### Summary
PINNs의 spectral bias 원인 규명. First-order optimizer: ė_k = -η·λ_k·e_k (고주파 느림). Second-order: ė_k = -e_k (주파수 무관). Tanh → high-freq 지수적 감쇠, Sin → decay 없음. KAN/PIKAN 변형 비교.

### Strengths
- **근본 원인 규명**: spectral bias = optimization issue, not architecture
- **해결책 제시**: second-order optimizer, sin activation, spectral-aware loss
- **KAN/PIKAN 분석**: KAN 기반 PINN 변형들의 장단점 체계적 비교
- **Karniadakis group**: PINN 원조 그룹의 analysis → 권위 있음
- **실용적 가이드**: 어떤 activation/optimizer 조합이 최적인지 제시

### Weaknesses / WaveSplat 적용 한계
- **분석 논문**: 새로운 방법론이 아닌 기존 방법론 분석
- **직접 적용 불가**: WaveSplat은 PINN이 아님 (explicit spectral representation)
- **PhD 주제 부적합**: 분석/survey 성격

### WaveSplat 적용 가능성: ★★☆☆☆
직접 적용은 어렵지만, 중요한 insight 제공:
- WaveSplat의 spectral Gaussian representation이 PINN보다 spectral bias에 강한 이유 설명 가능
- Second-order optimizer를 WaveSplat에 적용 가능
- Sin activation의 이점 → WaveSplat의 spectral parametrization이 이를 내재적으로 해결

### 가능한 활용
- Related work에서 WaveSplat의 design choice 정당화에 활용
- Optimizer 개선 (Adam → L-BFGS 등) 근거

---

## Synthesis: PhD Direction Recommendation

### 문제 재정의
WaveSplat의 핵심 과제: **sparse, irregular LiDAR → 2D+t wave field estimation with hard physics constraints**

### 방향별 비교

| 방향 | Novelty | 구현 난이도 | Advisor fit | 문제 적합성 | Risk |
|------|---------|------------|-------------|-------------|------|
| A. PINO-style physics loss + WaveSplat | ★★★★ | 중 | ★★★★ | ★★★★★ | 중 |
| B. DMU (ADMM unfolding) | ★★★★ | 상 | ★★★ | ★★★★ | 상 |
| C. Graph Neural Operator + dispersion | ★★★ | 상 | ★★★ | ★★★★ | 상 |
| D. KANO for wave spectral estimation | ★★★ | 상 | ★★ | ★★★ | 상 |
| E. TTT-style online adaptation | ★★ | 상 | ★★ | ★★ | 상 |

### 추천: Direction A — Physics-Informed WaveSplat (PI-WaveSplat)

**핵심 아이디어**: Ehlers 2025의 FSBC physics loss를 WaveSplat의 grid-free spectral Gaussian framework에 통합

**왜 이 방향인가**:
1. **직접적 경쟁 우위**: Ehlers는 1D+t + FNO (fixed grid). 우리는 2D+t + grid-free + LiDAR
2. **Physics constraint 이중화**: dispersion hard constraint (이미 있음) + FSBC soft loss (추가)
3. **Advisor fit**: "Physics-Constrained Optimization" 프레이밍 — ICNN 경험과 직결
4. **구현 가능성**: WaveSplat optimizer에 loss term 추가 수준 — 아키텍처 재설계 불필요
5. **No ground truth 가능**: FSBC loss는 관측값만으로 계산 가능 → 실측 적용 가능
6. **논문 스토리**: "Grid-free spectral Gaussians + physics-informed optimization for 2D+t wave estimation from sparse ship-mounted LiDAR"

**구체적 추가 사항**:
- L_total = L_obs (현재) + λ₁·L_FSBC_kinematic + λ₂·L_FSBC_dynamic + λ_reg·L_reg
- FSBC: ∂η/∂t + ∇Φ·∇η = ∂Φ/∂z (kinematic), ∂Φ/∂t + ½|∇Φ|² + gη = 0 (dynamic)
- Spectral Gaussian → 해석적 FSBC 계산 가능 (FNO와 달리 자동미분 불필요)

**차별화 포인트 (vs Ehlers 2025)**:
- 2D+t (Ehlers: 1D+t only)
- Grid-free spectral Gaussian (Ehlers: FNO on regular grid)
- Ship-mounted LiDAR (Ehlers: fixed buoy/radar)
- Hard dispersion constraint in architecture (Ehlers: soft in loss)
- Online sequential estimation (Ehlers: batch reconstruction)

### 보조 방향: Option B (DMU) 병행 검토
ADMM unrolling은 독립적으로 진행 가능. PI-WaveSplat이 주(主)이고, DMU는 optimizer 개선으로 부(副) contribution 가능.

### 비추천 방향
- **FR-Mamba**: regular grid 종속, physics 없음
- **KAN standalone**: tool-level, PhD novelty 부족
- **TTT**: wave estimation 핵심 문제와 연결 약함
- **DCSpecNet 계열 재시도**: 이미 실험적으로 실패 확인

---

## 다음 단계 (추천)
1. FSBC loss를 WaveSplat spectral Gaussian에서 해석적으로 유도
2. 기존 L_obs에 L_FSBC 추가하는 prototype 구현
3. 기존 episode에서 ablation (L_obs only vs L_obs + L_FSBC)
4. Ehlers 2025와의 positioning 명확화 (Related Work section 초안)
