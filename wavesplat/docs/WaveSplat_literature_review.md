# WaveSplat 선행연구 정리 (Literature Review)

> 최종 업데이트: 2026.04.06
> Q1 저널 및 Top-tier 학회 중심 정리
> Novelty 검증 완료 (2026.04.06) — 구조적 차별점 굳건, 정량 증명 필요

---

## 1. 파고장 추정 및 복원 (Wave Field Estimation & Reconstruction)

### 1.1 결정론적(Deterministic) 파고장 복원

| # | 논문 | 저널/학회 | 연도 | 핵심 내용 |
|---|------|-----------|------|-----------|
| **W1** | Nouguier, Grilli & Guérin, "Nonlinear Ocean Wave Reconstruction Algorithms Based on Simulated Spatiotemporal Data Acquired by a Flash LIDAR Camera" | **IEEE Trans. Geoscience & Remote Sensing** (Q1) | 2014 | Flash LiDAR 카메라로 취득한 시공간 데이터에서 비선형 파면 복원. 선박 전방 탑재 LiDAR 가정, 최소자승법 기반 진폭 추정. **WaveSplat의 직접적 선행연구** |
| **W2** | Desmars, Pérignon, Ducrozet & Grilli, "Experimental and Numerical Assessment of Deterministic Nonlinear Ocean Waves Prediction Algorithms" | **Ocean Engineering** (Q1) | 2020 | 비균등 샘플링 ��측에서 결정론적 비선형 파도 예측. 최적화 기반 초기조건 반전의 한계(실시간 불가) 분석 |
| **W3** | Qi, Wu, Liu & Kim, "Nonlinear Deterministic Reconstruction and Prediction of Remotely Measured Ocean Surface Waves" | **Journal of Fluid Mechanics** (Q1) | 2023 | 원격 측정 데이터에서 비선형 해수면 복원·예측. HOS(High-Order Spectral) 기반. 분산 관계를 활용한 시공간 일관성 보장 |
| **W4** | Fisher, Thomson & Schwendeman, "Rapid Deterministic Wave Prediction Using a Sparse Array of Buoys" | **Ocean Engineering** (Q1) | 2021 | 희소 부이 배열로 위상 분해 파도 예측. 스펙트럼 대비 63% 정확도 향상. 예측 수평선 1-3 Tp |

### 1.2 앙상블 기반 자료동화 (Ensemble-based Data Assimilation)

| # | 논문 | 저널/학회 | 연도 | 핵심 내용 |
|---|------|-----------|------|-----------|
| **E1** | Wang & Pan, "Phase-Resolved Ocean Wave Forecast with Ensemble-Based Data Assimilation" | **Journal of Fluid Mechanics** (Q1) | 2021 | EnKF + HOS 결합. 관측 데이터를 시뮬레이션에 동화시켜 위상 분해 예측 성능 향상. 합성 데이터 + 실제 레이더 검증 |
| **E2** | Wang & Pan, "Phase-Resolved Ocean Wave Forecast with Simultaneous Current Estimation through Data Assimilation" | **Journal of Fluid Mechanics** (Q1) | 2022 | E1 확장: 파도 예��과 동시에 해류 추정. 자료동화 분석방정식 수정으로 예측 가능 영역과 측정 영역 불일치 처리 |
| **E3** | ⭐ Jiang & Li, "From Points to Waves: Fast Ocean Wave Spatial-Temporal Fields Estimation Using ETKF with Optical Measurement" | **Coastal Engineering** (Q1) | 2024 | **WaveSplat의 직접 비교 대상.** ETKF로 스테레오 3D 점군에서 시공간 파고장 추정. 희소 점 → 격자 보간 → 칼만 필터. fuzzy logic 이상치 제거. **한계: 격자 보간 필수** |

### 1.3 기계학습 기반 파면 복원

| # | 논문 | 저널/학회 | 연도 | 핵심 내용 |
|---|------|-----------|------|-----------|
| **M1** | Ehlers, Klein, Heinlein et al., "Machine Learning for Phase-Resolved Reconstruction of Nonlinear Ocean Wave Surface Elevations from Sparse Remote Sensing Data" | **Ocean Engineering** (Q1) | 2024 | U-Net + FNO 아키텍처로 희소 레이더 데이터에서 위상 분해 파면 복원. HOS 시뮬레이션 + 기하학적 레이더 모델로 학습 데이터 생성 |
| **M2** | Ehlers et al., "Bridging Ocean Wave Physics and Deep Learning: Physics-Informed Neural Operators for Nonlinear Wavefield Reconstruction in Real-Time" | **arXiv** (preprint) | 2025 | PINO(Physics-Informed Neural Operator)로 GT 없이 실시간 비선형 파면 복원. 부이 시계열 + 레이더 스냅샷 모두 처리. **한계: 정규 격자 입력 전제** |

---

## 2. LiDAR 기반 해수면 측정 (LiDAR-based Ocean Surface Measurement)

| # | 논문 | 저널/학회 | 연도 | 핵심 내용 |
|---|------|-----------|------|-----------|
| **L1** | ⭐ Nouguier, Grilli & Guérin (=W1 재등록) | **IEEE TGRS** (Q1) | 2014 | Flash LiDAR + 선박 탑재, forward-looking. 거리 증가에 따른 점 밀도 ��소 분석. **WaveSplat 센서 구성의 직접 근거** |
| **L2** | Blenkinsopp, Mole, Turner & Peirson, "High-Resolution Monitoring of Wave Transformation in the Surf Zone Using a LiDAR Scanner Array" | **Coastal Engineering** (Q1) | 2012 | LiDAR 스캐너 배열로 쇄파대 고해상도 파도 변형 모니터링 |
| **L3** | Martins, Blenkinsopp, Deigaard & Power, "Observation of Surf Zone Wave Transformation Using LiDAR" | **Applied Ocean Research** (Q1) | 2018 | 연안 LiDAR로 파도 변형 관측. 시공간 해상도 분석 |
| **L4** | Xue, Wang & Qiu, "Wave Characteristics Analysis by LiDAR Point Cloud" | **Ocean Engineering** (Q1) | 2025 | LiDAR 점군에서 파도 특성 추출 프레임워���. Welch/MTM/AR 스펙트럼 분석 + EMEP 방향 스펙트럼. **점군→파도 파라미터 파이프라인의 최신 연구** |
| **L5** | Ma et al., "Retrieval of Directional Power Spectral Density and Wave Parameters from Airborne LiDAR Point Cloud" | **Ocean Engineering** (Q1) | 2023 | 항공 LiDAR 점군에서 방향 파워 스펙트럼 역산. JONSWAP 파라미터 추정 |

---

## 3. 스펙트럼 파도 추정 (Spectral Wave Estimation)

| # | 논문 | 저널/학회 | 연도 | 핵심 내용 |
|---|------|-----------|------|-----------|
| **S1** | Nielsen, "Non-Parametric Wave Spectral Estimation Using Vessel Motions" | **Applied Ocean Research** (Q1) | 2008 | 선박 운동 데이터에서 비모수적 파도 스펙트럼 추정 |
| **S2** | Montazeri, Nielsen & Jensen, "Estimation of Wind Sea and Swell Using Shipboard Measurements" | **Applied Ocean Research** (Q1) | 2016 | ��상 측정에서 풍파/너울 분리 추정 |
| **S3** | Grainger, Jonathan & Sherring, "Estimating the Parameters of Ocean Wave Spectra" | **Ocean Engineering** (Q1) | 2021 | JONSWAP 스펙트럼 파라미터 추정 방법론. 최적화 기반 파라미터 식별 |

---

## 4. 가우시안 스플래팅 및 유체 표현 (3D Gaussian Splatting & Fluid)

| # | 논문 | 저널/학회 | 연도 | 핵심 내용 |
|---|------|-----------|------|-----------|
| **G1** | Kerbl, Kopanas, Leimkühler & Drettakis, "3D Gaussian Splatting for Real-Time Radiance Field Rendering" | **ACM TOG / SIGGRAPH** (Q1) | 2023 | 3DGS 원 논문. 실시간 렌더링을 위한 Gaussian 기반 장면 표현 |
| **G2** | ⭐ Wu, Yang, Zhu et al., "FluidGS: Physics Informed Gaussian Splatting for Dynamic Fluid Reconstruction from Sparse Views" | **ACM Multimedia** (A*) | 2025 | 3DGS + Navier-Stokes 물리 제약으로 희소 뷰에서 동적 유체 복원. Lagrangian-Eulerian 하이브리드. **WaveSplat과 가장 유사한 접근이나 차별점 명확** |
| **G3** | Feng, Zhang, Wu et al., "Gaussian Splashing: Dynamic Fluid Synthesis with Gaussian Splatting" | **AAAI** (A*) | 2024 | 3DGS + PBD(Position-Based Dynamics)로 유체 합성. 렌더링-물리 시뮬레이션 통합 |
| **G4** | Wu et al., "4D Gaussian Splatting for Real-Time Dynamic Scene Rendering" | **CVPR** (A*) | 2024 | 4DGS: 동적 장면 실시간 렌더링. temporal Gaussian deformation |

---

## 5. 파도 예측 기반 선박 제어 (Wave Prediction → Ship Control)

| # | 논문 | 저널/학회 | 연도 | 핵심 내용 |
|---|------|-----------|------|-----------|
| **C1** | Haring, Hassani & Experiment, "This is the Way: Mitigating Roll of an Autonomous USV in Wavy Conditions Using MPC" | **IEEE/RSJ IROS** (A*) | 2024 | MPC로 파도 환경 자율 USV 롤 저감. **"next step: wave prediction model 통합"** → WaveSplat 논문2 동기 |
| **C2** | Zheng, Negenborn et al., "MPC-based Motion Planning and Control Enables Smarter and Safer Autonomous Marine Vehicles" | **IEEE/CAA J. Automatica Sinica** (Q1) | 2023 | MPC 기반 자율 해상 차량 운동 계획·제어 종합 서베이 |
| **C3** | Jin, Oh, Kim & Lee, "Advanced Heave Control of a Hydrofoil-Based Autonomous Surface Vehicle" | **Ocean Engineering** (Q1) | 2025 | 하이드로포일 ASV 상하동요 제어. CFD + 가변 프로펠러 혁명수 제어 |

---

## 6. 레이더 기반 파고장 추정 (Radar-based Wave Estimation) — 참고

| # | 논문 | 저널/학회 | 연도 | 핵심 내용 |
|---|------|-----------|------|-----------|
| **R1** | Hessner, Wallbridge & Dolphin, "Wave-by-Wave Forecasting via Assimilation of Marine Radar Data" | **J. Atmos. Oceanic Technol.** (Q1) | 2020 | X-band 레이더 + 자료동화 파도별 예측 |
| **R2** | Lee, Nam, Liu & Yang, "Study on Predictability of Ocean Wave Fields Based on Marine Radar Measurement Data" | **Proc. IMechE Part M** (Q1) | 2024 | 해양 레이더 데이터 기반 파고장 예측 가능성 연구 |
| **R3** | Qi et al., "Phase-Resolved Wave Prediction from Radar Observations Based on a Multi-Scale FNO" | **Physics of Fluids** (Q1) | 2025 | 다중 스케일 FNO로 레이더 관측에서 위상 분해 파도 예측 |

---

## ⭐ 필독 논문 (Must-Read) — 우선순위순

### Tier 1: 반드시 정독해야 할 논문 (WaveSplat 핵심 관련)

| 우선순위 | 논문 코드 | 왜 읽어야 하는가 |
|----------|-----------|-------------------|
| **1** | **E3** — Jiang & Li (2024), Coastal Engineering | **직접 비교 대상 (primary baseline)**. ETKF 구현 세부사항, 격��� 보간 한계, 실험 설계를 WaveSplat과 1:1 비교해야 함. WaveSplat의 "grid-free" novelty가 이 논문의 한계를 극��하는 구조 |
| **2** | **W1/L1** — Nouguier, Grilli & Guérin (2014), IEEE TGRS | **LiDAR + 파면 복원의 원조**. Flash LiDAR 선박 탑재 가정, 최소자승법 기반 진폭 계수 추정. WaveSplat의 "spectral estimation from LiDAR" 접근의 직접적 출발점 |
| **3** | **E1** — Wang & Pan (2021), J. Fluid Mech. | **EnKF + 파도 자료동화의 이론적 기초**. WaveSplat의 sequential update와 비교할 수 있는 대안적 프레임워크. 예측 가능 영역 vs 측정 영역 불일치 처리 방법 |
| **4** | **W3** — Qi et al. (2023), J. Fluid Mech. | **비선형 결���론적 복원의 최신 이론**. HOS 기반 비선형 예측 한계 분석. WaveSplat의 linear model mismatch 논의에 필요한 이론적 배경 |

### Tier 2: 깊이 이해를 위해 읽어야 할 논문

| 우선순위 | 논문 코드 | 왜 읽어야 하는가 |
|----------|-----------|-------------------|
| **5** | **M1** — Ehlers et al. (2024), Ocean Engineering | **ML 기반 희소 데이터 복원의 최신 연구**. FNO 접근과 WaveSplat의 spectral Gaussian 접근 비교 논의에 필요 |
| **6** | **G2** — FluidGS (2025), ACM MM | **Gaussian + physics-informed 유체 복원에서 가장 유사한 연구**. 차별점: FluidGS는 Navier-Stokes/렌더링, WaveSplat은 해수면 분산 관계/LiDAR 관측 |
| **7** | **L4** — Xue et al. (2025), Ocean Engineering | **LiDAR 점군 → 파도 특성 추출의 최신 파이프라인**. WaveSplat의 전처리 및 스펙트럼 분석 비교에 유용 |
| **8** | **W4** — Fisher et al. (2021), Ocean Engineering | **희소 관측에서 결정론적 예측의 실험적 검증**. 예측 수평선, 관측 밀도-성능 관계를 WaveSplat 실험 설계에 참고 |

### Tier 3: 논문 작성 시 인용/참고

| 논문 코�� | 용도 |
|-----------|------|
| **G1** — 3DGS 원논문 | Related Work — "Gaussian representation" 계보 설명 |
| **C1** — MPC + 파도 | Motivation — "왜 파고장 추정이 필요한가" 논문 서두 |
| **S3** — 스펙트럼 추정 | Method — JONSWAP 파라미터 추정 방법론 참고 |
| **E2** — EnKF + 해류 | Discussion — 향후 해류 동시 ��정 확장 언급 |
| **R3** — FNO 레이�� | Discussion — 다른 센서 모달리티 확장 가능성 |

---

## 7. WaveSplat 포지셔닝 맵

```
                    Physics-based          Data-driven
                         │                      │
     Grid-based     E3(ETKF)               M1(FNO)
                    E1(EnKF+HOS)           M2(PINO)
                    R1(Radar+DA)
                         │                      │
                         │    ┌─────────────┐   │
                         └────│  WaveSplat  │───┘
                              │ Grid-free + │
                              │ Physics     │
                              └─────────────┘
                         │                      │
     Grid-free      W1(LiDAR+LS)           G2(FluidGS)
                    W4(Buoy array)          G3(GS+PBD)
```

**WaveSplat의 고유 위치**: Grid-free + Physics-constrained + LiDAR-specific confidence.
기존 연구 중 이 세 가지를 동시에 만족하는 방법은 없음.

---

## 8. Novelty 검증 결과 (2026-04-06)

### 실험 기반 위협 분석

| Contribution | 구조적 차별점 | 정량 증명 | 상태 |
|---|---|---|---|
| **C1: Grid-free** | ETKF/FNO/PINO 모두 격자 필수 → WaveSplat은 임의 좌표 직접 평가 | slope(불균일) 데이터에서 w_phi 45% 개선 확인 | **굳건** |
| **C2: Dispersion** | ω=f(k,d) 구조 내장 vs ETKF 외부 적용 | Ablation 미실시 (TODO) | 구조적 유효, 증명 필요 |
| **C3: Confidence** | cos(θ)/d 가중치 — 기존 연구는 등방 가정 | Ablation 미실시 (TODO) | 구조적 유효, 증명 필요 |

### 1:1 비교 위험도

| 선행연구 | 위험 수준 | WaveSplat 차별점 | 대응 |
|---------|---------|----------------|------|
| **E3 (ETKF, 2024)** | **높음** | Grid-free, 보간 불필요 | 직접 비교 실험 필수 |
| G2 (FluidGS, 2025) | 낮음 | 도메인 완전 상이 (NS vs dispersion) | 인용만으로 충분 |
| M1/M2 (FNO/PINO) | 중간 | 학습 불필요, 불규칙 관측 직접 처리 | 구조적 차이 명시 |
| W1 (LiDAR+LS, 2014) | 낮음 | WaveSplat의 직접 출발점, 개선 관계 | 계보 설명 |

### 핵심 결론

> **Novelty는 굳건하나, Ablation과 ETKF 직접 비교로 정량 입증이 필수.**
> 특히 Init 개선으로 tracking nRMSE를 10% 이하로 달성해야 "grid-free가 실질적으로 유효하다"는 주장에 설득력이 생김.

---

## 참고 자료 출처

- [From Points to Waves — Jiang & Li, Coastal Engineering 2024](https://www.sciencedirect.com/science/article/abs/pii/S0378383924002382)
- [Flash LiDAR Wave Reconstruction — Nouguier et al., IEEE TGRS 2014](https://ieeexplore.ieee.org/document/6521365/)
- [EnKF Phase-Resolved Forecast — Wang & Pan, J. Fluid Mech. 2021](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/phaseresolved-ocean-wave-forecast-with-ensemblebased-data-assimilation/1C553ED4AD796F97DCB895F09E3BF89C)
- [Nonlinear Wave Reconstruction — Qi et al., J. Fluid Mech. 2023](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/nonlinear-deterministic-reconstruction-and-prediction-of-remotely-measured-ocean-surface-waves/0D492ABAE4B99782612F5CE4AC890B6D)
- [ML Phase-Resolved Reconstruction — Ehlers et al., Ocean Engineering 2024](https://www.sciencedirect.com/science/article/pii/S0029801823024435)
- [PINO Wavefield Reconstruction — Ehlers et al., arXiv 2025](https://arxiv.org/abs/2508.03315)
- [FluidGS — Wu et al., ACM MM 2025](https://dl.acm.org/doi/10.1145/3746027.3755500)
- [Rapid Wave Prediction from Buoys — Fisher et al., Ocean Engineering 2021](https://www.sciencedirect.com/science/article/pii/S0029801821003061)
- [Wave Characteristics by LiDAR — Xue et al., Ocean Engineering 2025](https://www.sciencedirect.com/science/article/abs/pii/S0029801825026782)
- [LiDAR Directional Spectrum — Ma et al., Ocean Engineering 2023](https://www.sciencedirect.com/science/article/abs/pii/S0029801822019771)
- [MPC for USV Roll — Haring et al., IROS 2024](https://arxiv.org/html/2408.15349v1)
- [3DGS Original — Kerbl et al., SIGGRAPH 2023](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [4DGS — Wu et al., CVPR 2024](https://github.com/hustvl/4DGaussians)
- [Gaussian Splashing — Feng et al., AAAI 2024](https://arxiv.org/abs/2401.15318)
- [MPC Marine Survey — Zheng et al., IEEE/CAA JAS 2023](https://www.ieee-jas.net/article/doi/10.1109/JAS.2022.106016)
