# Direction B: 문헌 조사 및 방향 결정 (2026-05-22)

## 1. 배경

### 교수님 피드백 (2026-05-13 랩미팅)
> "EnKF, PINN처럼 큰 갈래가 필요하다"

### 문제 정의
- **Task**: sparse irregular ship-mounted LiDAR (27-75 points/frame) → 2D ocean wave field z(x,y,t) 추정
- **현재 최선**: RELAX init + Adam optimizer → avg nRMSE 0.22
- **핵심 bottleneck**: spectral estimation accuracy (추정 오차 ~20%)

---

## 2. 지금까지 시도한 방향 (7개 — 전부 실패/부족)

| # | 방향 | 기법 | 결과 | 실패 원인 |
|---|------|------|------|-----------|
| A | DCSpecNet | Set Transformer encoder → spectral params | nRMSE >> BF | novelty 부족 + 학습 부족 |
| B | USGO | Algorithm unrolling (WaveSplat optimizer) | val 0% 개선 | init→GT 거리 과대, 10 layers << 500 Adam |
| C | Neural corrector | DeepSets/Set Transformer residual correction | +0.15%~+77% 악화 | 추정오차(20%) >> 비선형효과(5%) |
| D | Stokes2 prediction | 2nd order analytical + HOS gap | 0% gap closed at K=20 | spectral truncation 지배 |
| E | DU-RELAX | Deep Unfolded RELAX (3 versions) | nRMSE ≈ 1.0 | coarse search 대체 불가 |
| F | PINO/FNO/DeepONet | Neural operators | 구조 비호환 | uniform grid/fixed sensor 요구 |
| G | RELAX standalone | Classical RELAX as Direction B | e2e +1.9% | init 71%↑이 optimizer에 흡수됨 |

### 공통 실패 패턴
- **"기존 파이프라인 위에 neural 덧붙이기"** → 모두 실패
- 원인: 제한된 데이터 (110 episodes), classical method가 이미 상당히 강력

### RELAX 성과 (WaveSplat 논문에 편입)
- **Init-only**: nRMSE 0.488 → 0.141 (71% 개선), 6/6 에피소드 승리
- **End-to-end**: avg +1.9% 개선, 6/9 승리 (complex sea 15+ comp에서 +8~17%)
- `init_method='relax_konly'`로 파이프라인 통합 완료
- **결론**: WaveSplat 논문 init chapter 업그레이드로 충분, 단독 Direction B로는 부족

---

## 3. 새로운 방향 문헌 조사 (50+ 논문)

### 3.1 주요 경쟁자: Ehlers 그룹 (TU Hamburg)

| 논문 | 연도 | 기법 | 특징 |
|------|------|------|------|
| Ehlers et al., Ocean Eng. | 2023 | U-Net, FNO | data-driven, 1D radar |
| Ehlers et al., Phys Rev Fluids | 2025a | PINN | per-scene, potential flow PDE, 1D buoy |
| Ehlers et al., Phys Fluids | 2025b | PINO | real-time, free surface BC, 1D |

**우리의 차별점**:
1. **2D** field (Ehlers는 1D buoy/flume)
2. **Ship-mounted LiDAR** (sparse, moving, irregular)
3. **Hard dispersion constraint** (Ehlers는 soft PDE residual)
4. **Parametric spectral representation** (explicit kx, ky, w, phi)

### 3.2 INR / Neural Fields

| 논문 | 연도/Venue | 핵심 | 관련성 |
|------|-----------|------|--------|
| SIREN (Sitzmann) | NeurIPS 2020 | sin activation INR, PDE loss | HIGH |
| MMGN (Luo) | ICLR 2024 | factorized INR, sparse→continuous | HIGH |
| STRIDE (Tong) | arXiv 2026 | temporal encoder + modulated INR | HIGH |
| Senseiver (Santos) | Nat Mach Intell 2023 | cross-attention, variable sensors | HIGH |
| PINF Smoke (Chu) | SIGGRAPH 2022 | neural field + Navier-Stokes PDE | MED-HIGH |

### 3.3 Diffusion / Flow Matching

| 논문 | 연도/Venue | 핵심 | 관련성 |
|------|-----------|------|--------|
| Spatially-Aware Diffusion (Zhuang) | CMAME 2024 | cross-attention diffusion, sparse obs | HIGH |
| PiRD | arXiv 2024 | physics-informed residual diffusion | HIGH |
| **DDIS** | arXiv 2025 | **decoupled prior + PDE**, data-scarce | **VERY HIGH** |
| CoCoGen (Jacobsen) | SIAM J Sci Comp 2025 | PDE-consistent score model | HIGH |
| Flow Matching + PDE (Baldan) | arXiv 2025 | ODE path, PDE constraints | MED-HIGH |

### 3.4 GNN / Transformer / Neural Process

| 논문 | 연도/Venue | 핵심 | 관련성 |
|------|-----------|------|--------|
| **Gridded TNP** (Ashman) | arXiv 2024 | efficient attention NP, open-source | **HIGH** |
| BSA-TNP | arXiv 2025 | scalable NP (1M+ points) | HIGH |
| VIRSO | arXiv 2026 | GNN sparse→dense, V-KNN | HIGH |
| FieldFormer | arXiv 2025 | physics-informed transformer, PDE | HIGH |
| Senseiver for tsunami (McDugald) | GRL 2025 | ~30 buoys → tsunami field | HIGH |
| **Sound Field DKL** | arXiv 2024 | wave eq + deep kernel GP | **HIGH** |
| LEVDA (Si & Chen) | arXiv 2026 | latent ensemble 4DEnVar + NN surrogate | HIGH |
| KalmanNet (Revach) | IEEE TSP 2022 | neural Kalman gain | MED |

---

## 4. Top 5 후보 평가

### 4.1 Neural Process (TNP) — **1순위 추천**

**구조**: LiDAR {(x,y,z,t)} → cross-attention encoder → latent → decoder → z(x_q, y_q, t_q)

| 항목 | 평가 |
|------|------|
| Novelty | HIGH — NP for ocean wave estimation = 0편 |
| 실현성 | HIGH — Gridded TNP 오픈소스 (Cambridge MLG) |
| 데이터 요구 | 중간 — 110 ep × 50 frames = 5,500 meta-learning pairs |
| Dispersion 호환 | 설계 필요 — spectral projection post-processing |
| 이전 실패와의 차이 | DCSpecNet은 single-shot spectral prediction, TNP는 per-query field prediction |
| 교수님 프레이밍 | "Physics-Constrained Meta-Learning for Real-Time Wave Estimation" |

**핵심 참고 논문**:
- Gridded TNP (Ashman et al., 2024): github.com/cambridge-mlg/gridded-tnp
- Senseiver (Santos et al., Nat Mach Intell 2023): ~30 buoys → field reconstruction
- Senseiver for tsunami (McDugald et al., GRL 2025): wave physics + sparse obs

### 4.2 SIREN + Dispersion PDE Loss — **2순위**

**구조**: z(x,y,t) = SIREN_θ(x,y,t), Loss = MSE_obs + λ·dispersion_PDE_residual

| 항목 | 평가 |
|------|------|
| Novelty | HIGH — SIREN for 2D ocean wave + hard dispersion = 0편 |
| 실현성 | HIGH — SIREN ~50 lines, per-scene optimization |
| 데이터 요구 | **없음** — per-scene (학습 데이터 불필요!) |
| Dispersion 호환 | 자연스러움 — PDE residual loss로 통합 |
| 이전 실패와의 차이 | 표현 자체를 바꿈 (spectral Gaussian → neural field) |
| 위험 | per-scene optimization = 느림, spectral Gaussian보다 나은 이유 불분명 |

### 4.3 Deep Kernel GP (DKL-GP) — **안전한 선택**

| 항목 | 평가 |
|------|------|
| Novelty | MED-HIGH — ocean wave dispersion kernel = novel |
| 실현성 | HIGH — GPyTorch, 27-75점 exact inference |
| 데이터 요구 | 낮음 — physics regularization으로 보충 |
| Dispersion 호환 | EXCELLENT — kernel에 직접 encoding |
| 교수님 연결 | Bayesian 경험 (Bayesian opt for UUV hull, 2024) |
| 핵심 참고 | Sound Field DKL (arXiv:2407.04417) — wave eq + sparse sensors |

### 4.4 Senseiver/STRIDE — **공학적 선택**

| 항목 | 평가 |
|------|------|
| Novelty | MED-HIGH — domain application |
| 실현성 | HIGH — Perceiver-IO 기반, well-documented |
| 데이터 요구 | 중간 |
| 장점 | ~30 buoys에서 검증 (우리 27-75점과 유사) |
| 단점 | 방법론적 novelty 낮음 (기존 arch 적용) |

### 4.5 DDIS (Decoupled Diffusion) — **이론적 최강**

| 항목 | 평가 |
|------|------|
| Novelty | HIGH — decoupled prior + PDE for ocean waves = 0편 |
| 실현성 | LOW — 복잡한 구현 (unconditional prior + neural operator + 4DEnVar) |
| 데이터 요구 | 높음 (borderline: 11K snapshots) |
| 장점 | data-scarce에서 이론적 우위 (40% advantage in l2 error) |
| 위험 | 구현 복잡도, inference 느림 (20-1000 reverse steps) |

---

## 5. 페르소나 토론 요약

### TNP vs SIREN

| 관점 | TNP | SIREN |
|------|-----|-------|
| Paradigm | Amortized (한번 학습 → 즉시 적용) | Per-scene (매번 최적화) |
| Inference 속도 | 빠름 (forward pass) | 느림 (수백 iteration) |
| 데이터 필요 | 5,500+ pairs | 없음 |
| 물리 통합 | Post-processing projection | PDE residual loss |
| DCSpecNet 차이 | per-query prediction (근본적으로 다름) | 표현 자체가 다름 |
| Real-time | 가능 | 어려움 |

### 결론
- **TNP**: 높은 novelty + amortized inference + real-time 가능 → PhD 방향으로 최적
- **SIREN**: 데이터 불필요 장점이 강력하나, per-scene = 현재 WaveSplat과 같은 paradigm
- **DKL-GP**: 실패 리스크 최소, 교수님 Bayesian 연결, but novelty 상대적 부족

---

## 6. 추천 방향

### 1순위: TNP (Transformer Neural Process)
- **논문 제목**: *"Dispersion-Constrained Neural Process for Real-Time Ocean Wave Field Estimation from Sparse Ship-Mounted LiDAR"*
- **PhD contribution**:
  1. NP for 2D ocean wave estimation (0편)
  2. Dispersion-aware attention kernel / spectral projection layer
  3. Meta-learning across sea states with uncertainty quantification
  4. Real-time inference from sparse irregular LiDAR
- **오픈소스 기반**: Gridded TNP (Cambridge MLG)
- **Fallback**: 실패 시 DKL-GP로 전환

### 2순위: SIREN + Dispersion PDE (fallback)
- 학습 데이터 불필요, 구현 간단
- Per-scene optimization이므로 real-time 제약

---

## 7. 다음 단계

1. Gridded TNP 코드 분석 및 우리 데이터에 적응
2. Dispersion constraint 통합 방법 설계
3. Pilot 실험: 1개 에피소드에서 TNP vs WaveSplat 비교
4. 실패 시 DKL-GP fallback

---

## 참고 문헌 (핵심)

### Neural Processes
- Ashman et al. (2024). Gridded Transformer Neural Processes. arXiv:2410.06731
- Santos et al. (2023). Senseiver. Nature Machine Intelligence.
- McDugald et al. (2025). Senseiver for tsunami. GRL. arXiv:2411.12948

### INR / SIREN
- Sitzmann et al. (2020). SIREN. NeurIPS. arXiv:2006.09661
- Luo et al. (2024). MMGN. ICLR. arXiv:2401.11611
- Tong & Chen (2026). STRIDE. arXiv:2602.04201

### Diffusion / Flow Matching
- DDIS (2025). Decoupled Diffusion Inverse Solver. arXiv:2601.23280
- Shu et al. (2023). Physics-informed Diffusion. JCP. arXiv:2211.14680
- PiRD (2024). Physics-informed Residual Diffusion. arXiv:2404.08412

### Deep Kernel GP
- Sound Field DKL (2024). arXiv:2407.04417
- Wang & Xing (2022). PI-DKL. AISTATS.

### Ocean Wave (경쟁자)
- Ehlers et al. (2025a). PINN for ocean waves. Phys Rev Fluids. arXiv:2501.08430
- Ehlers et al. (2025b). PINO for real-time waves. Phys Fluids. arXiv:2508.03315

### RELAX
- Li & Stoica (1996). An adaptive filtering approach to spectral estimation and SAR imaging. IEEE TSP.

### Data Assimilation
- LEVDA (2026). Si & Chen. arXiv:2602.19406
- VIRSO (2026). arXiv:2604.01802
- FieldFormer (2025). arXiv:2510.03589
