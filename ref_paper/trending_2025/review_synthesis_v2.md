# Trending Network Survey v2 — 종합 검토 및 방향 추천
Date: 2026-05-19

## 조사 범위
5개 방향 × 총 ~60편 논문 (2025-2026)
1. Neural Operators (irregular grid, point cloud)
2. State Space Models / Mamba (scientific computing)
3. Ocean Wave Estimation ML (직접 경쟁자)
4. Differentiable/Explicit Representations (3DGS + physics)
5. Deep Unfolding / Algorithm Unrolling (inverse problem)

---

## Part 1: 핵심 발견 — WaveSplat 관점에서 중요한 논문들

### A. 가장 가까운 기존 연구 (Representation 관점)

#### Gabor-Enhanced PINNs (2024-2025)
- **표현**: G(x) = A · exp(-|x-c|²/σ²) · exp(i·k·x) — Gaussian-windowed sinusoid
- **용도**: acoustic/seismic wavefield **forward simulation** (PDE residual loss)
- **vs WaveSplat**: 우리의 spectral Gaussian과 표현 구조가 거의 동일!
  - 차이 1: Gabor = forward problem (알려진 PDE → 풀기), WaveSplat = **inverse** (sparse obs → 추정)
  - 차이 2: Gabor = soft PDE constraint (loss), WaveSplat = **hard** dispersion (architecture)
  - 차이 3: Gabor = acoustic/seismic, WaveSplat = **ocean surface waves**
  - 차이 4: Gabor = offline batch, WaveSplat = **online sequential**
- **시사점**: 표현 자체의 novelty는 약화됨. 대신 inverse + hard constraint + online이 차별화 핵심

#### SV-SNN (Separated-Variable Spectral NN, 2025)
- **표현**: learnable frequency parameters + separation of variables
- **성능**: PINNs 대비 1-3 orders of magnitude 정확도 향상, 90% parameter 감소
- **vs WaveSplat**: 유사한 철학 (learnable spectral parameters), 하지만 PDE solver, 비 wave domain

### B. 가장 가까운 기존 연구 (Application 관점)

#### Ehlers PINO (Physics of Fluids, 2025)
- **확인**: 이전 조사와 동일. 1D+t, FNO, soft FSBC, buoy/radar
- **신규 발견**: PINN 버전도 존재 (Phys. Rev. Fluids 2025) — per-instance optimization
- **코드**: 미공개

#### Hard-BC PINN for Free Surface Waves (Physics of Fluids, 2025)
- **핵심 결론**: hard constraint > soft constraint for wave BCs
- **시사점**: **WaveSplat의 hard dispersion 설계를 직접 지지하는 독립 연구**

#### LiDAR Point Cloud Wave Analysis (Ocean Engineering, 2025)
- **내용**: LiDAR point cloud에서 wave characteristics 추출 (geometric fitting)
- **한계**: spectral decomposition 없음, physics constraint 없음
- **시사점**: LiDAR 기반 wave 연구가 있지만 ML/physics 결합은 없음

### C. Point Cloud 입력을 지원하는 최신 아키텍처

| 방법 | Venue | Point Cloud | Physics | Temporal |
|------|-------|-------------|---------|----------|
| RIGNO | NeurIPS 2025 | ✓ (multi-scale GNN) | ✗ | ✗ |
| GAOT | NeurIPS 2025 | ✓ (attention + GNN) | ✗ | ✗ |
| LaMO | ICML 2025 | ✓ (Mamba latent) | ✗ | ✓ (SSM) |
| STREAM | arXiv 2024 | ✓ (geometric SSM) | ✗ | ✓ (SSM) |
| FieldFormer | arXiv 2025 | ✓ (mesh-free) | ✓ (PDE residual) | ✗ |
| VIRSO | arXiv 2026 | ✓ (V-KNN graph) | ✗ | ✗ |

→ Point cloud 지원은 이제 보편화. **하지만 physics constraint + temporal + wave domain 조합은 없음**

### D. Deep Unfolding 핵심 발견

#### Off-Grid Frequency Estimation via ADMM+Learning (MSSP 2025)
- **가장 구조적으로 유사한 문제**: off-grid spectral estimation from limited measurements
- **방법**: ADMM unroll + learned dictionary with unitary constraint
- **차이**: 1D 진동/음향 신호, dispersion relation 없음, point cloud 입력 아님
- **시사점**: WaveSplat의 optimizer를 ADMM unroll로 대체하는 것이 기술적으로 validated

#### CDU — Constrained Dual Unrolling (2026)
- **핵심**: primal-dual 네트워크로 hard constraint optimization을 unroll
- **시사점**: dispersion constraint를 dual variable로 enforce하는 이론적 근거

#### Adaptive Vekua Cascade (2025)
- **핵심**: spectral coefficients를 closed-form으로 계산하는 differentiable solver
- **시사점**: WaveSplat의 amplitude/phase estimation을 closed-form layer로 대체 가능

### E. SSM/Mamba 핵심 발견

#### STREAM (2024) — SSM for Sparse Geometric Data
- **핵심**: relative coordinate → SSM step size 주입, O(N) point cloud 처리
- **시사점**: LiDAR point cloud의 불규칙 spacing을 SSM이 자연스럽게 처리

#### KOSS (2025) — Kalman-Optimal SSM
- **핵심**: selective SSM = Kalman gain의 learnable version
- **시사점**: WaveSplat의 ETKF 대안으로, Kalman-SSM 하이브리드 가능

#### PINNMamba (ICML 2025)
- **핵심**: data-free physics-informed SSM, 86.3% error reduction over PINNsFormer
- **시사점**: training data 없이 physics constraint만으로 학습 가능 → 실해역 적용

#### PGMNO (Neural Networks, 2026) — Physics-Guided Mamba Neural Operator
- **핵심**: governing equation을 Mamba 아키텍처에 deep integration
- **시사점**: dispersion relation을 Mamba operator 내부에 embedding 가능

### F. 3DGS + Physics 동향

| 방법 | Venue | Physics 유형 | 적용 |
|------|-------|-------------|------|
| PhysGaussian | CVPR 2024 | MPM continuum mechanics | deformable solids |
| NGFF | ICLR 2026 | learned force fields | rigid/soft bodies |
| GausSim | ICCV 2025 | mass/momentum conservation | elastic objects |
| D-Prism | arXiv 2026 | deformation network | dynamic scenes |
| PIDG | 여러 2025 | constitutive models | deformable solids |

→ 3DGS + physics는 매우 활발하지만, **전부 visual scene dynamics (rendering)**.
→ **Field estimation from sparse sensors는 아무도 안 함** ← 우리의 gap

---

## Part 2: 경쟁 현황 지도 (Landscape Map)

```
                     Representation
           Implicit/Grid          Explicit/Parametric
          ┌─────────────────────┬──────────────────────┐
Physics   │ Ehlers PINO (2025)  │                      │
Soft      │ FieldFormer (2025)  │ Gabor-PINN (2025)    │
(loss)    │ PhysBR (2026)       │ SV-SNN (2025)        │
          ├─────────────────────┼──────────────────────┤
Physics   │ Hard-BC PINN (2025) │                      │
Hard      │                     │  ★ WaveSplat (ours)  │
(arch.)   │                     │                      │
          └─────────────────────┴──────────────────────┘

                     Input Type
           Fixed Grid/Buoy       Sparse Point Cloud
          ┌─────────────────────┬──────────────────────┐
Offline   │ Ehlers PINO/PINN    │ RIGNO, GAOT          │
(batch)   │ SHRED, AUWave       │ FieldFormer           │
          ├─────────────────────┼──────────────────────┤
Online    │ Wang ETKF (2024)    │                      │
(seq.)    │                     │  ★ WaveSplat (ours)  │
          └─────────────────────┴──────────────────────┘
```

**WaveSplat은 두 지도 모두에서 유일한 위치를 점유:**
1. Explicit representation + Hard physics constraint
2. Online sequential + Sparse point cloud input

---

## Part 3: 방향 재평가

이전 추천 (Direction A: PI-WaveSplat — FSBC loss 추가)을 **새로운 발견들을 반영해** 재평가합니다.

### 방향 1: PI-WaveSplat (FSBC loss 추가만)
- **장점**: 구현 간단, 기존 코드 활용
- **단점**: Gabor-PINN이 이미 spectral representation + physics loss 조합을 함. **차별화 약화**
- **평가**: ★★★☆☆ — 논문 1편 수준, PhD 핵심은 아님

### 방향 2: DMU (Dispersion Manifold Unfolding — ADMM unroll)
- **장점**: Off-Grid ADMM (MSSP 2025)가 1D에서 성공 → 2D+t ocean wave로 확장은 명확한 novelty
- **장점**: Deep Unfolding Survey (2026)가 설계 가이드라인 제공
- **장점**: CDU가 hard constraint unrolling 이론 제공
- **단점**: 구현 복잡도 높음, training data 필요
- **평가**: ★★★★☆ — PhD-level novelty, 하지만 implementation risk

### 방향 3: Physics-Guided Spectral Mamba (NEW!)
- **핵심**: STREAM (sparse geometric SSM) + PGMNO (physics-guided Mamba) + spectral Gaussian representation
- **구조**:
  - Input: sparse LiDAR point cloud {(x,y,z)}_t (STREAM 방식으로 encoding)
  - Temporal: Mamba SSM으로 sequential state update (KOSS의 Kalman-optimal 방식)
  - Output: spectral Gaussian parameters {(kx,ky,A,φ)}_j
  - Constraint: hard dispersion ω = √(g|k|tanh(|k|d)) (architectural)
  - Loss: observation MSE + analytical FSBC (only possible with explicit spectral repr.)
- **장점**:
  - STREAM+Mamba: point cloud native + O(N) temporal → 실시간 가능
  - PGMNO: physics를 아키텍처에 deep integration
  - Spectral Gaussian output: analytical FSBC 계산 가능 (FNO/PINN 불가)
  - Kalman-SSM 연결 (KOSS): 기존 ETKF 대비 학습된 Kalman gain
- **단점**: 구현 매우 복잡, 여러 최신 기술 결합 필요
- **평가**: ★★★★★ — 최고 novelty, 하지만 최고 risk

### 방향 4: Unrolled Spectral Gaussian Optimizer (추천!)
- **핵심**: WaveSplat의 기존 optimizer를 algorithm unrolling으로 학습 가능하게 만듦
- **구조**:
  - Layer 0 (Init): Beamforming → spectral Gaussian params (기존 v2_konly)
  - Layer 1~K (Unrolled update): 각 layer = 1 iteration of constrained optimization
    - Learnable step sizes, regularization weights, proximal operators
    - Hard dispersion projection after each layer (Beucler 2021 + CDU 2026)
    - Off-grid spectral correction (MSSP 2025 방식)
  - Output: refined spectral Gaussian params {(kx,ky,A,φ)}_j
  - Loss: observation MSE + analytical FSBC + sparsity
- **장점**:
  - 기존 WaveSplat optimizer 구조를 그대로 unroll → 구현 연속성
  - Off-Grid ADMM (MSSP 2025)의 2D ocean wave 확장 → 명확한 novelty
  - Hard dispersion을 projection layer로 enforce (CDU 이론 적용)
  - Analytical FSBC가 가능한 유일한 이유: explicit spectral Gaussian representation
  - Training data는 HOS simulator로 무한 생성 가능
  - Inference는 unrolled network forward pass → 실시간
- **단점**: unrolling 구현 필요, hyperparameter tuning
- **평가**: ★★★★★ — 최고 novelty + 관리 가능한 risk

---

## Part 4: 최종 추천 — Direction 4: Unrolled Spectral Gaussian Optimizer

### 왜 이것이 최선인가?

#### Novelty 검증 (5가지 차별화)

| # | 우리의 기여 | 가장 가까운 기존 연구 | 차이점 |
|---|------------|---------------------|--------|
| 1 | Spectral Gaussian for ocean wave estimation | Gabor-PINN (2025) | Inverse estimation (not forward), ocean wave (not acoustic), online (not batch) |
| 2 | Hard dispersion in architecture | Hard-BC PINN (PoF 2025) | Explicit spectral repr. (not implicit NN), dispersion (not Laplace BC) |
| 3 | Analytical FSBC computation | Ehlers PINO (2025) | Closed-form (not autograd), explicit repr. (not FNO), 2D+t (not 1D+t) |
| 4 | Unrolled off-grid spectral estimator | Off-Grid ADMM (MSSP 2025) | 2D ocean wave (not 1D vibration), dispersion constraint (없음), point cloud input (not 1D signal) |
| 5 | Adaptive component count in unrolled net | D-Prism (2026) | Physics-based grow/prune (not geometric), spectral domain (not spatial) |

→ 각 기여가 기존 연구와 명확히 구별됨. 그리고 **5가지가 하나의 coherent framework로 통합**.

#### 교수님 프레이밍

> "ICNN이 convexity를 architecture로 보장하듯, 우리는 dispersion을 architecture로, FSBC를 analytical computation으로 보장합니다. 기존 optimizer를 Deep Unfolding으로 학습 가능하게 만들어, physics-constrained optimization의 수렴 속도와 정확도를 동시에 개선합니다."

연결 고리:
- ICNN (Amos 2017, 연구실 경험) → hard constraint in architecture
- Bayesian optimization (연구실 경험) → surrogate-based optimization → unrolled optimizer
- CFD (연구실 core) → wave physics (FSBC, dispersion)

#### Implementation Roadmap

```
Phase 1 (2주): Analytical FSBC
  - Spectral Gaussian에서 velocity potential Φ 유도
  - FSBC 잔차 closed-form 구현
  - 기존 optimizer에 L_FSBC 추가, ablation

Phase 2 (3주): Unrolled Optimizer
  - 기존 coarse-to-fine + sequential_update를 K-layer로 unroll
  - Learnable step sizes, regularization weights
  - Dispersion projection layer
  - HOS simulator로 training data 생성

Phase 3 (2주): Off-Grid Extension
  - Off-grid spectral correction (MSSP 2025 방식)
  - Adaptive N integration (grow/prune within unrolled layers)

Phase 4 (2주): Evaluation
  - 기존 WaveSplat vs Unrolled WaveSplat vs Ehlers PINO (재현) vs ETKF
  - Ablation: hard vs soft dispersion, analytical vs autograd FSBC
  - Speed benchmark: inference time comparison
```

#### vs 경쟁자 논문 Table

| Method | Input | Representation | Physics | Training | Speed | 2D+t |
|--------|-------|---------------|---------|----------|-------|------|
| Ehlers PINO (2025) | Buoy/Radar | FNO (implicit) | Soft FSBC | Physics-only | ~1s GPU | ✗ (1D+t) |
| Ehlers PINN (2025) | Wave gauge | NN (implicit) | Full PFT | Per-instance | Slow | ✗ (1D+t) |
| Wang ETKF (2024) | Radar | Grid ensemble | Forward model | None | ~130ms | ✓ |
| Hard-BC PINN (2025) | BC/IC | NN (implicit) | Hard Laplace+FSBC | Per-instance | Slow | ✗ |
| Off-Grid ADMM (2025) | 1D signal | Grid + correction | None | Supervised | Fast | ✗ (1D) |
| **Ours (proposed)** | **Ship LiDAR** | **Spectral Gaussian** | **Hard disp. + analytical FSBC** | **Supervised (HOS)** | **~13ms CPU** | **✓** |

---

## Part 5: 비추천 방향 및 이유

| 방향 | 비추천 이유 |
|------|------------|
| DCSpecNet 재시도 | 실험적으로 실패 확인 (beamforming 대비 2-10x 열등) |
| KAN standalone | tool-level, PhD novelty 부족 (ICLR reviewers도 large-scale 이점 의문 제기) |
| FR-Mamba 적용 | regular grid 종속, wave physics 없음 |
| TTT for waves | core problem과 연결 약함 |
| RIGNO/GAOT 직접 적용 | "기존 neural operator를 wave에 적용" = application paper |
| Pure Mamba operator | PGMNO, LaMO 등 이미 많음 — wave domain application만으로는 부족 |

---

## Part 6: 필수 인용 논문 (Related Work용)

### Representation
1. Gabor-Enhanced PINNs (2024/2025) — 가장 가까운 representation 선행연구
2. PhysGaussian (CVPR 2024) — Gaussian + physics 패러다임
3. SV-SNN (2025) — learnable spectral parameters

### Wave Estimation
4. Ehlers PINO (Physics of Fluids 2025) — 직접 경쟁자
5. Ehlers PINN (Phys. Rev. Fluids 2025) — per-instance wave estimation
6. Wang & Pan ETKF (2024) — ensemble Kalman baseline
7. Hard-BC PINN (Physics of Fluids 2025) — hard > soft 검증

### Method
8. Off-Grid ADMM (MSSP 2025) — unrolled spectral estimation 선행연구
9. CDU (2026) — constrained dual unrolling 이론
10. Deep Unfolding Survey (Shlezinger+ 2026) — 설계 가이드라인
11. Beucler (PRL 2021) — hard physics constraint in architecture

### Input Processing
12. STREAM (2024) — sparse geometric SSM (future extension용)
13. FieldFormer (2025) — physics-informed sparse sensor reconstruction
