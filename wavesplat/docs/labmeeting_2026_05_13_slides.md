# Lab Meeting 2026-05-13 — WaveSplat Presentation Draft

**발표자**: Saeyong Park
**발표 시간**: 20분 (1 slide ≈ 1분, 18 slides + Q&A 2분)
**대상**: 외부 청자 (도메인 가정 최소화)
**Date**: 2026-05-13 (Wed)

---

## 19-Slide Outline (selective prior art detail)

**Decisions** (2026-05-08):
- **Q1**: Top 5 prior art detail (1 paper/slide) + backup slides (Q&A) for나머지 7 paper
- **Q2**: 우리가 PDF figure 우선 추출 → 사용자가 PowerPoint 에서 다듬기
- **Q3**: Wang 2025 는 PPT 에서 빼고 진행 (paper 작성 단계에서 정독). 단, 받으면 즉시 detail slide 추가.

| # | Slide | Part | Day |
|---|---|---|---|
| 1 | Title + outline | — | 1 ✅ |
| 2 | Motivation: phase-resolved wave estimation | Part 1 | 1 ✅ |
| 3 | Problem: sparse + real-time + 2D directional | Part 1 | 1 ✅ |
| 4 | Prior art landscape — 5 categories overview | Part 1 | 2 |
| 5 | **Desmars 2023 (JFM) — HOS-OCM** ⭐ | Part 1 | 2 |
| 6 | **Wang & Pan 2021/22 (JFM) — EnKF-HOS** ⭐ | Part 1 | 2 |
| 7 | **Ehlers et al. 2023 (Ocean Eng) — FNO ML** ⭐ | Part 1 | 2 |
| 8 | **Ehlers et al. 2025 (PRF) — PFT-PINN** ⭐ | Part 1 | 2 |
| 9 | **Yan et al. 2025 (arXiv) — VOF-EnKF** ⭐ | Part 1 | 2 |
| 10 | 응용 매트릭스 + 우리 niche | Part 1 | 2 |
| 11 | WaveSim — physics + raydrop sensor model | Part 2 | 3 |
| 12 | WaveSim — 100 episodes overview | Part 2 | 3 |
| 13 | WaveSplat 알고리즘 (exploration phase) | Part 3 | 3 |
| 14 | Prior result — P1 12-ep comparison + real-time | Part 3 | 3 |
| 15 | 본질 발견 — spatial Nyquist 한계 | Part 4 | 4 |
| 16 | Main framing — Dispersion-Coupled Spatio-Temporal Inversion ⭐⭐ | Part 4 | 4 |
| 17 | Mechanism + Identifiability decomposition | Part 4 | 4 |
| 18 | Linear 한계 + Stokes future + PFT-PINN 위협 대응 | Closing | 5 |
| 19 | Timeline + venue + Q&A | Closing | 5 |

### Backup slides (A1-A7, Q&A 시 참조)
- A1: Desmars 2018 (OMAE) — LIDAR + ICWM concept
- A2: Desmars 2020 (Ocean Eng) — wave gauge experiment
- A3: Klein 2020 (Fluids) — HOSM/NLSE/LWT 비교
- A4: Hunt 2007 (Tellus A) — LETKF foundational
- A5: Tang-Bhaskar-Recht 2013 (IEEE TIT) — atomic norm CS
- A6: Möller-Trumbore 1997 (JGT) — ray-triangle (코드 검증)
- A7: Nouguier 2014 (IEEE TGRS) — flash LIDAR historical

### 5-day timeline (재조정)
- **Day 1 (오늘)**: Slide 1-3 ✅ 완료
- **Day 2 (내일)**: Slide 4-10 (7 slides, 가장 무거움) — *외부 서버 접속 day*
- **Day 3 (수)**: Slide 11-14 (WaveSim 2 + exploration phase 2)
- **Day 4 (목)**: Slide 15-17 (main framing — 가장 중요)
- **Day 5 (금)**: Slide 18-19 + backup + presenter notes + rehearsal

---

## Wang 2025 정독 체크리스트 (내일 오전 작업)

받자마자 §8.2 9-항목 확인:
1. State vector — η + ∂η/∂t? 또는 η + spectral coefficients?
2. Observation operator H — obs→grid (interpolation BEFORE) or grid→obs?
3. Propagation — FFT exact dispersion? WASSfast model 정의?
4. Inflation — multiplicative? additive?
5. Localization — r_loc 사용?
6. Outlier handling — fuzzy logic 알고리즘
7. Test dataset — simulator/실측?
8. Hyperparameter tuning protocol — sweep? sensitivity?
9. Code release — GitHub 공개?

→ **Case A/B/C 결정** → §32 추가 → Slide 4-5 / Slide 11-12 의 ETKF baseline framing 결정

---

# Part 1 — Motivation & Prior Art

## Slide 1 — Title

### Title
**Phase-Resolved Ocean Wave Estimation from Sparse Shipboard LiDAR**
*Exploration phase results & a path forward*

### Subtitle (1-line)
Real-time, 2D-directional, training-free wave field reconstruction

### Author / affiliation
Saeyong Park
[Lab name placeholder]
2026-05-13

### Outline preview (4 bullets)
- Why phase-resolved waves matter
- Where the field stands (12 paper review)
- What we built so far (WaveSim + WaveSplat exploration phase)
- Where we are headed (new framing + publication plan)

### Visual
- Center: ship + LiDAR + wave field schematic (재사용 가능 — sea_trial 사진 + lidar overlay)

### Presenter notes
오늘 발표는 두 부분으로 나뉩니다. **앞 절반**은 우리가 6개월 동안 진행한 *exploration phase* 의 결과 — 어떤 prior art 가 있고, 우리가 만든 simulator (WaveSim) 와 estimator (WaveSplat) 이 어떻게 작동하는지. **뒤 절반**은 그 과정에서 발견한 본질적 mechanism — vessel motion 과 dispersion 의 결합 이 어떻게 *spatial Nyquist 한계를 우회* 하는지, 그리고 이를 main contribution 으로 격상시킬 publication plan. 외부 청자라 해서 도메인 background 부터 시작하겠습니다 — phase-resolved wave estimation 이 왜 필요한지부터.

---

## Slide 2 — Motivation: Why Phase-Resolved Waves Matter

### Title
**Phase-resolved wave estimation enables look-ahead control of marine systems**

### Key bullets
- **Statistical descriptions are not enough**: H_s, T_p 같은 spectrum 통계만으로는 *언제 큰 파도가 올지* 모름
- **Phase-resolved**: 각 파도의 위치·시점을 *결정론적* 으로 추정 → look-ahead window (0–30 s) 확보
- **Direct applications**:
  - **Hydrofoil ride control**: 파도 정보 미리 알면 feed-forward control → ride RMS ↓
  - **Autonomous vessel navigation**: 위험 파도 회피, course planning
  - **Offshore operations**: 인양 작업, helicopter landing window 결정
  - **Wave-energy converter control**: optimal damping tuning
- **Common requirement**: *real-time + look-ahead window > horizon of control loop*

### Visual
- Left: H_s/T_p spectrum (statistical) — only mean info
- Right: η(x, t) phase-resolved field (deterministic) — where each crest is
- Bottom: hydrofoil 응용 schematic (preview → FF → ride RMS reduction)
- (재사용 가능 — hydrofoil_controller_flow.png)

### Presenter notes
파도 추정 분야에는 두 layer 가 있습니다. 첫째 **statistical estimation** — significant wave height H_s, peak period T_p, JONSWAP spectrum 같은 *통계량*. 둘째 **phase-resolved estimation** — η(x,y,t) 자체를 시공간상에서 *결정론적* 으로 추정. 통계는 엔지니어링 design 에 필수지만, *실시간 control* 에는 부족합니다. 특정 시점에 정확히 어디에 큰 파도가 오는지 알아야 ride control, hydrofoil, autonomous navigation 같은 응용이 가능. 이 발표는 후자, phase-resolved 추정 문제에 집중합니다. 응용으로는 hydrofoil ride control 이 가장 직접적 — 우리 연구실에서 진행 중인 hydrofoil 연구의 motivation 이기도 합니다.

---

## Slide 3 — Problem: Three Compounding Challenges

### Title
**Shipboard LiDAR-based phase-resolved estimation is hard for three reasons**

### Key bullets — 3개 challenge

#### Challenge 1 — **Sparse, non-uniform measurements**
- Single LiDAR scan: ~50 valid points / frame (raydrop, shadowing 후)
- Distance-dependent density: 근거리 dense, 원거리 sparse
- Range limited: ~250 m (vs ocean wavelength 50–300 m)

#### Challenge 2 — **Real-time constraint**
- Control loop horizon: ~10 Hz update
- Per-frame budget: < 100 ms
- 오프라인 CFD/PINN/EnKF 들은 분~시간 단위 소요

#### Challenge 3 — **2D directional sea state**
- Real ocean: short-crested, multi-directional
- 1D unidirectional 가정은 부족 (Klein 2020, Desmars 2023 둘 다 대부분 1D)
- 2D (k_x, k_y) 추정 필요

### Visual
- 3-panel schematic:
  - (좌) sparse LiDAR scan-line 점 분포 (이미 있는 figure 재사용 — wavesim_BF4_5c_wave_lidar.png)
  - (중) timing diagram: control loop 10 Hz
  - (우) 2D directional spectrum vs 1D (kx, ky) heatmap

### Presenter notes
이 3가지가 동시에 만족되어야 하는 setup 은 prior art 에 거의 없습니다. (1) **Sparse non-uniform** — LiDAR 가 흔히 쓰이는 wave gauge 나 dense radar 와 다르게 frame 당 50점 정도, 거리 따라 density 다름. (2) **Real-time** — control loop 에 들어가야 하므로 < 100 ms. (3) **2D directional** — 실제 바다는 한 방향 파도가 아니라 directional spectrum. 다음 슬라이드에서 보겠지만, prior art 12편을 검토한 결과 *세 조건을 동시에 만족하는 method 가 없습니다*. 우리 연구의 niche 는 정확히 이 3-조건 교집합입니다.

---

# (Day 1 작업 종료 — Slide 1-3 draft 완료)

---

# Day 2 — Slide 4-10 (Prior Art Detail + Niche)

> **참고**: Slide 5-9 의 paper figure 는 `docs/figures/prior_art/` 디렉토리에 추출 예정. 추출 후 각 slide 의 `[FIGURE_PLACEHOLDER]` 자리에 삽입.

## Slide 4 — Prior Art Landscape (5 Categories)

### Title
**Phase-resolved wave estimation: 5 categories of prior art**

### Key bullets — 5 카테고리
| 카테고리 | 대표 paper | 핵심 | 한계 |
|---|---|---|---|
| **(1) Variational / grid-based** | Desmars 2018-2023, Aragh-Nwogu 2008 | predefined k-grid + cost function inversion | on-grid, 1D 위주 |
| **(2) Ensemble Kalman filter** | Wang & Pan 2021/22, Yan 2025 | HOS/VOF + EnKF + inflation | offline, 1D xz-plane |
| **(3) ML supervised** | Ehlers 2023 (FNO), Duan 2020, Liu 2022 | pre-trained NN, fast inference | training data, generalization |
| **(4) Physics-informed unsupervised** | Ehlers 2025 (PFT-PINN), Lu 2024 | PINN + PFT loss, no training data | ~30 min/instance, 1D+t |
| **(5) Parametric off-grid** ⭐ | **WaveSplat (우리)** | continuous (k_x, k_y, ω, A, φ) + dispersion hard | linear approximation |

### Visual
- 5-카테고리를 2D 평면에 배치:
  - X축: **fidelity** (linear → fully nonlinear)
  - Y축: **inference speed** (slow → real-time)
- 각 paper 를 점으로 plot
- **우리 위치**: top-left (linear, real-time) + 추가 차원 = **2D directional + LiDAR**

### Presenter notes
12 paper 정독 결과 5 카테고리로 분류 가능합니다. 첫 4개는 기존 paradigm — variational on-grid, EnKF ensemble, ML supervised, physics-informed unsupervised. 우리는 **5번째 카테고리**, parametric off-grid 입니다. 다른 카테고리들은 모두 *grid 또는 NN 으로 wave field 를 표현* 하지만, 우리는 *spectral parameter 자체를 직접 추정*. 이게 가능한 이유는 dispersion 제약을 hard constraint 로 강제하기 때문 — 다음 슬라이드부터 가장 가까운 prior art 5 편을 detail 로 보여드리겠습니다.

---

## Slide 5 — Desmars 2023 (JFM) ⭐ 가장 가까운 prior art

### Title
**Desmars et al. 2023 (JFM 975, A8) — HOS-OCM with vessel-mounted optical**

### Key message (3 bullets)
- **첫 vessel-mounted optical phase-resolved wave reconstruction + prediction** with HOS (full nonlinear potential flow)
- **Setup**: shipborne LiDAR / camera, 1D longitudinal, predefined k-grid
- **Method**: variational data assimilation (cost function inversion) → initialize HOS solver → predict

### Method diagram
[FIGURE_PLACEHOLDER: desmars2023_fig2_setup.png]
*Vessel-mounted optical sensor + 1D wave reconstruction schematic*

### Result figure
[FIGURE_PLACEHOLDER: desmars2023_fig_results.png]
*Reconstruction + prediction accuracy (paper Fig. 8 or similar)*

### 한계 (우리 관점, 3 bullets)
- **On-grid k-band assumption** — wavenumber 가 predefined grid 에 묶임 (off-grid 가 아님)
- **1D unidirectional** — 2D directional sea state 처리 안 함
- **Optical sensor model** 명시 없음 — raydrop, shadowing, Fresnel 가정 미상

### Presenter notes
12 paper 중 **우리와 가장 가까운 prior art**. Setup 자체가 거의 동일 — 선박 mounted optical sensor 로 phase-resolved wave 를 추정하고 HOS 로 predict. 차이는 (1) 1D vs 우리 2D, (2) on-grid vs 우리 off-grid, (3) sensor model detail. 우리 paper 의 main reference 가 됩니다.

---

## Slide 6 — Wang & Pan 2021/22 (JFM) ⭐ 우리 ETKF baseline 의 family

### Title
**Wang & Pan 2021 (JFM 918, A19) + 2022 (JFM 949, A31) — EnKF-HOS**

### Key message (3 bullets)
- **First EnKF integration with HOS** — ensemble-based DA on phase-resolved waves
- **2022**: extension with current estimation (EnKF-HOS-C)
- **GPU implementation, near-real-time** (단 dense observation 가정)

### Method diagram
[FIGURE_PLACEHOLDER: wangpan2021_enkf_hos_diagram.png]
*EnKF-HOS pipeline*

### Result figure
[FIGURE_PLACEHOLDER: wangpan2021_predictable_zone.png]
*Predictable zone (group velocity bound) + reconstruction error*

### 한계 (우리 관점)
- **Dense observation 가정** — sparse LiDAR setup 직접 적용 어려움
- **1D wave** — 2D directional 처리 안 함
- **Ensemble cost** — 50-100 members × HOS 시간 통합 (still slower than parametric)

### 우리와 관계
- 우리 [wavesplat/baselines/etkf_grid.py](wavesplat/baselines/etkf_grid.py) 가 이 family 의 simplified implementation
- *Note*: Wang et al. 2025 (Coastal Eng 197) 는 더 최근 grid-based ETKF 변형 — paper 작성 단계에서 정독 예정 (Case A/B/C 결정)

### Presenter notes
EnKF-HOS family 는 우리 ETKF baseline 의 직접적 prior art 입니다. Wang & Pan 2021 이 처음으로 HOS 와 EnKF 를 결합했고, 2022 에서 current estimation 으로 확장. 한계는 (1) ensemble 비용으로 real-time 어렵고, (2) dense obs 가정이라 sparse LiDAR setup 에 직접 적용 안 됨. 우리는 이 baseline 을 sparse-input 으로 tune 해서 비교 했습니다.

---

## Slide 7 — Ehlers et al. 2023 (Ocean Eng) — FNO ML radar

### Title
**Ehlers et al. 2023 — Fourier Neural Operator for radar wave reconstruction**

### Key message (3 bullets)
- **First ML-based phase-resolved reconstruction** from sparse X-band radar
- **FNO architecture** (Fourier global mapping) outperforms U-Net (CNN local)
- **Real-time inference**: 1.9 ms / instance (단 supervised pre-training 필요)

### Method diagram
[FIGURE_PLACEHOLDER: ehlers2023_fno_arch.png]
*FNO architecture (lifting + Fourier layers + projection)*

### Result figure
[FIGURE_PLACEHOLDER: ehlers2023_ssp_results.png]
*SSP metric across n_s snapshots, FNO vs U-Net*

### 한계 (우리 관점)
- **Supervised** — 3120 HOS samples 학습 필요, distribution shift risk
- **1D unidirectional** — 2D directional 처리 안 함
- **Black-box** — physical interpretability 부족

### 우리와 차별점
- 우리: **unsupervised** (no training data), interpretable parametric model
- 우리: **2D directional**, FNO 1D
- Speed: 우리 ~13 ms vs FNO 1.9 ms — *비슷한 영역*, 단 우리는 training-free

### Presenter notes
ML supervised 카테고리의 SOTA. 1.9 ms 의 inference 속도가 인상적이지만 *3120 HOS sample 학습* 이라는 비용이 있고 *out-of-distribution sea state* 에 약합니다. 우리는 training 자체가 없어 generalization 문제 없음. 또한 1D 만 처리, 우리는 2D directional. 둘 다 sparse sensor + real-time 영역인 점에서 paper 1 의 main competitor.

---

## Slide 8 — Ehlers et al. 2025 (PRF) — PFT-PINN ⭐ 가장 위협적

### Title
**Ehlers et al. 2025 (Phys. Rev. Fluids 10, 094901) — Physics-Informed NN with Potential Flow Theory**

### Key message (3 bullets)
- **PFT-PINN**: 두 NN 병렬 (η-net + Φ-net), Laplace + 3 BCs (kinematic/dynamic/bottom) AD-enforced
- **Hard-constraint trick**: η̃ = M(x,t) + R(x,t)·N_η̃ — measurements 를 architecture 에 hard-encode
- **Validated on MARATHON wave flume** (T_p=1.2s, ε=0.09 JONSWAP) — SSP=0.049 (assimilation), SSP=0.084 (prediction)

### Method diagram
[FIGURE_PLACEHOLDER: ehlers2025_pftpinn_arch.png]
*Two parallel NNs + collocation points + AD-based loss components*

### Result figure
[FIGURE_PLACEHOLDER: ehlers2025_marathon_results.png]
*Wave flume reconstruction, SSP < 0.05*

### 한계 (우리 관점)
- **~30 min training / instance** (NVIDIA RTX 3090, 40K+ epochs)
- **1D + t**, future work 가 "2D + t extension" 명시
- **Buoy time-series only** — sparse spatial snapshot LiDAR setup 처리 안 함

### 우리와 차별점 (정량)
- **Speed**: 우리 ~13 ms vs PFT-PINN ~30 min = **약 130,000× 차이** (응용 영역 분리)
- 우리: **2D + t** 이미 가능
- 우리: **LiDAR raydrop sensor model** 포함

### 우리 niche 위협
- 같은 그룹 (TUHH+Imperial) 이 *PFT-PINN-2D-realtime* 을 1-3년 내 발표 가능 → **우리 publish 속도 중요**

### Presenter notes
가장 *위협적인* prior art. Physics-informed unsupervised 라는 점에서 우리와 근본 철학 같습니다 — training data 없이 dispersion + BCs 를 enforce. 다만 1D + t 만 다루고, 30분 training 필요. Future work 가 명시적으로 "2D extension + real-time speedup" 인데, 같은 그룹이 1-3년 안에 이걸 발표하면 우리 niche 가 좁아짐. 따라서 우리는 (1) **빨리 publish**, (2) **응용 영역 차별화** (PINN = design-time analysis, 우리 = real-time control loop) 으로 대응.

---

## Slide 9 — Yan et al. 2025 (arXiv) — VOF-EnKF

### Title
**Yan, Che, Li 2025 (arXiv:2510.14356) — VOF-NS + EnKF for plunging breakers**

### Key message (3 bullets)
- **First VOF-NS + EnKF integration** — full Navier-Stokes + ensemble DA
- **POD reduction**: M=4.8e4 cells → ~30 modes (level-set), 4 modes (interface)
- **Plunging breaker reconstruction** — assimilate upstream linear, CFD evolves breaking

### Method diagram
[FIGURE_PLACEHOLDER: yan2025_pipeline.png]
*VOF + EnKF + POD pipeline*

### Result figure
[FIGURE_PLACEHOLDER: yan2025_plunging_breaker.png]
*Plunging breaker reconstruction sequence (paper Fig. 18)*

### 한계 (우리 관점)
- **Offline** — 50 members × full OpenFOAM = orders of magnitude slower
- **1D xz-plane only** — 2D directional 처리 안 함
- **Inlet boundary 처리 어려움** — near-inlet RMS 0.228 H_s vs downstream 0.090

### 우리와 차별점 (정량)
- **Speed**: 우리 ~13 ms vs VOF-EnKF (offline)
- 우리: **2D directional**, VOF-EnKF 는 1D xz
- **응용 영역 분리**: VOF-EnKF = high-fidelity offline digital twin, 우리 = real-time control

### Presenter notes
다른 극단의 prior art — *full physics fidelity* 추구. Plunging breaker 처리 가능한 유일한 method 지만 50 ensemble × OpenFOAM 으로 *fundamentally offline*. 우리와 응용 영역 다름 — 우리는 real-time control loop, VOF-EnKF 는 design-time digital twin. Limitation 으로 인정할 점은 우리가 wave breaking 못 처리한다는 점, future work 로.

---

## Slide 10 — 응용 매트릭스 + 우리 Niche

### Title
**Application matrix: where does WaveSplat fit?**

### Application matrix table

| 응용 요구 | WaveSplat | PFT-PINN | VOF-EnKF | EnKF-HOS | FNO | Desmars |
|---|---|---|---|---|---|---|
| Real-time control loop (선박/hydrofoil) | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| Phase-resolved 2D directional | ✅ | ❌ | ❌ | ❌ (1D) | ❌ | ❌ |
| Wave breaking, plunging | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Wave-structure interaction (loads) | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Velocity field full inference | ❌ | ✅ | ✅ | ✅ | ❌ | partial |
| Bathymetry inversion | ❌ | ✅ ext. | ❌ | ❌ | ❌ | ❌ |
| Training-free unsupervised | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| Sparse LiDAR sensor model | ✅ | ❌ | ❌ | ❌ | ❌ | partial |

### 우리 niche (3 bullets)
- **Real-time + 2D directional + sparse LiDAR + unsupervised** = **1순위 unique 조합**
- 응용: **선박 ride control, hydrofoil FF preview, autonomous navigation**
- *Trade-off*: linear approximation, no breaking, no velocity inference (limitation 으로 인정)

### Visual
- 매트릭스 table 강조 (우리 column 색칠)
- 응용 영역 segmentation: **real-time control vs design-time analysis** 두 영역으로 분리, 우리는 real-time 영역 specialist

### Presenter notes
이 매트릭스가 *우리 paper 의 positioning 핵심*. 모든 항목을 만족하는 method 는 없습니다 — 각자의 specialty 가 있어요. 우리 specialty 는 *real-time + 2D directional + sparse LiDAR* 의 unique 조합. 응용 영역으로 보면 PFT-PINN/VOF-EnKF 와 *경쟁 안 함* — 그들은 *design-time analysis tool*, 우리는 *real-time control component*. 한계도 명시 — wave breaking, velocity inference, full nonlinear 는 우리 영역 밖. 다음 슬라이드부터는 *우리가 만든 것* 을 보여드립니다.

---

# (Day 2 작업 종료 시 update)

다음: Day 3 — Slide 11-14 (WaveSim + exploration phase)

---
