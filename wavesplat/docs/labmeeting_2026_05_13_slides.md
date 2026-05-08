# Lab Meeting 2026-05-13 — WaveSplat Presentation Draft

**발표자**: Saeyong Park
**발표 시간**: 20분 (1 slide ≈ 1분, 18 slides + Q&A 2분)
**대상**: 외부 청자 (도메인 가정 최소화)
**Date**: 2026-05-13 (Wed)

---

## 19-Slide Outline (selective prior art detail)

**Decisions** (2026-05-08 updated):
- **Q1**: Top 5 prior art detail (1 paper/slide) + backup slides (Q&A) for 나머지
- **Q2**: 우리가 PDF figure 우선 추출 → 사용자가 PowerPoint 에서 다듬기
- **Q3**: Wang 2025 정독 완료 → S6에 반영, 3-way 비교 결과 S14에 반영
- **Q4**: WASSfast (Bergamasco 2021) 코드 분석 완료 → S4 카테고리 + S6 관련 언급

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

## Wang 2025 정독 체크리스트 — ✅ 완료 (2026-05-08)

1. State vector: η + ∂η/∂t (2-field) ✅
2. Observation operator: Delaunay interpolation BEFORE (obs→grid) ✅
3. Propagation: FFT exact dispersion ✅
4. Inflation: fuzzy R adaptation (not multiplicative inflation) ✅
5. Localization: sub-state ETKF (Hunt 2007) ✅
6. Outlier handling: 9-rule fuzzy logic (Table 1) ✅
7. Test dataset: stereo camera field data ✅
8. Hyperparameter: Ne=300 optimal (Fig. 12) ✅
9. Code release: GitHub 공개 ✅

→ **Case B 결정**: 같은 ETKF family지만 enhanced (fuzzy + Delaunay + large Ne)
→ Faithful reproduction 구현 완료 → 12-episode 3-way 비교 완료
→ S6, S14 업데이트 완료

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
| **(2) Ensemble Kalman filter** | Wang & Pan 2021/22, Wang 2025, Yan 2025 | HOS/VOF + EnKF + inflation | offline~slow, grid 한계 |
| **(3) Spectral grid fitting** | WASSfast (Bergamasco 2021) | sparse pts → NUFFT spectral grid + dispersion predict | NxN grid 표현, stereo camera |
| **(4) ML supervised** | Ehlers 2023 (FNO), Duan 2020, Liu 2022 | pre-trained NN, fast inference | training data, generalization |
| **(5) Physics-informed unsupervised** | Ehlers 2025 (PFT-PINN), Lu 2024 | PINN + PFT loss, no training data | ~30 min/instance, 1D+t |
| **(6) Parametric off-grid** ⭐ | **WaveSplat (우리)** | continuous (k_x, k_y, ω, A, φ) + dispersion hard | linear approximation |

### Visual
- 5-카테고리를 2D 평면에 배치:
  - X축: **fidelity** (linear → fully nonlinear)
  - Y축: **inference speed** (slow → real-time)
- 각 paper 를 점으로 plot
- **우리 위치**: top-left (linear, real-time) + 추가 차원 = **2D directional + LiDAR**

### Presenter notes
12+ paper 정독 결과 6 카테고리로 분류 가능합니다. 특히 카테고리 3 WASSfast (Bergamasco 2021)가 우리와 가장 가까운 접근 — sparse 3D points에서 dispersion 활용한 spectral fitting. 차이는 WASSfast가 NxN spectral grid coefficients를 추정하는 반면, 우리는 N개 continuous parametric component를 직접 추정한다는 점. 또한 WASSfast는 stereo camera (수천 점) 전제, 우리는 scanning LiDAR (~50 점)로 극도로 sparse한 상황. 우리는 **6번째 카테고리**, parametric off-grid입니다.

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
- **Wang 2025 (Coastal Eng 197)**: 같은 ETKF family의 최신 변형 (stereo camera, fuzzy R, Ne=300)
  - 우리가 faithful reproduction 구현: [wavesplat/baselines/etkf_wang2025.py](wavesplat/baselines/etkf_wang2025.py)
  - **3-way 비교 완료**: WS 0.211/13ms vs Wang2025 0.253/632ms vs ETKF-orig 0.382/59ms
  - Wang2025는 기본 ETKF 대비 34% 개선되지만, WaveSplat보다 16% 열위 + 47x 느림

### Presenter notes
EnKF-HOS family는 우리 ETKF baseline의 직접적 prior art입니다. Wang & Pan 2021이 처음으로 HOS와 EnKF를 결합했고, 2022에서 current estimation으로 확장. **최신 Wang 2025 (Coastal Eng)는 fuzzy logic + Delaunay interpolation + Ne=300으로 enhanced ETKF를 제안** — 이를 faithful하게 재현해서 비교한 결과, 기본 ETKF보다 확실히 좋지만(0.253 vs 0.382) WaveSplat(0.211)에는 못 미침. 다만 고복잡도(15-30 성분)에서는 Wang2025가 WaveSplat과 비등하거나 역전 — grid 방법의 robustness. 속도는 632ms vs 13ms로 47배 차이.

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

# (Day 2 작업 종료 — Slide 4-10 draft 완료)

---

# Day 3 — Slide 11-14 (WaveSim + Exploration Phase)

## Slide 11 — WaveSim: Physics + LiDAR Sensor Model

### Title
**WaveSim — physics + LiDAR sensor model**

### Subtitle
선형 중첩 + 분산 + JONSWAP 스펙트럼, 그리고 LiDAR raydrop 물리까지

### Key bullets

#### Physics (수면 모델)
- η(x, y, t) = Σ_n A_n cos(k_xn·x + k_yn·y − ω_n·t + φ_n)
- N개 plane wave의 선형 중첩 (Linear Wave Theory, LWT)

#### Dispersion relation (분산 관계)
- ω_n = √(g·|k_n|·tanh(|k_n|·d)) — 주파수와 wavenumber의 1-to-1 매핑
- 깊은 물 근사: ω² ≈ g·|k| (d → ∞)

#### Spectrum
- JONSWAP-2D (Hasselmann 1973 + 방향 spreading)
- N_components 가변 (1, 2, 5, 15, 20, 30)

#### LiDAR sensor model (unique contribution)
- **Ray-triangle intersection**: Möller-Trumbore 1997 (그래픽 분야 표준, 검증된 알고리즘)
- **Raydrop probability**: Fresnel reflectance (specular vs diffuse), 입사각 + 파면 경사에 의존
- **Geometric shadowing**: 가까운 파도가 먼 파도를 가림 (occlusion)
- **Vessel motion**: V=11.3 m/s 이동 + ego-velocity correction

#### 검증
- 39 unit test 통과, 100+ episode 합성 dataset 생성

### Visual
- 오른쪽: wave field 시각화 (wavesim_BF4_5c_wave_field.png)

### Presenter notes
WaveSim은 우리가 만든 simulator입니다. 두 부분으로 나뉩니다.

(1) 수면 model (physics):
  - Linear Wave Theory — N개 plane wave의 선형 중첩으로 수면 표현
  - Dispersion relation — frequency ω와 wavenumber k의 관계 (깊은 물에서 ω² = g·k·tanh(kd))
  - JONSWAP-2D 스펙트럼 — 실제 바다의 통계적 특성 모방

한계는 정직하게 인정합니다 — Klein 2020이 정량 비교한 HOSM > NLSE > LWT 중 우리는 LWT family로 가장 단순. ε (steepness) < 0.07 영역에서만 신뢰 가능.

(2) LiDAR sensor model — 이게 우리의 unique contribution:
  - Ray-triangle intersection: LiDAR ray가 mesh와 만나는 점을 계산. Möller-Trumbore 1997 알고리즘 사용 (그래픽 분야 표준)
  - Raydrop: 파도 표면에서 ray가 반사되어 LiDAR에 못 돌아오는 확률. Fresnel reflectance 사용 (specular = 거울 반사, diffuse = 산란)
  - Shadowing: 앞쪽 파도가 뒤쪽 파도를 가림
  - Vessel motion: 선박이 V=11.3 m/s로 움직이므로 매 frame 다른 위치에서 관측

12편 prior art 중 이 정도 sensor model을 simulate한 것은 없습니다. 39개 unit test로 simulator 정확성 검증, 110 episode를 합성 dataset으로 생성했습니다.

---

## Slide 12 — WaveSim: 100+ Episodes Overview

### Title
**110 episode — 통제된 benchmark**

### Subtitle
재현 가능, 다양한 복잡도, ground-truth η + LiDAR 점 모두 보존

### Key bullets

#### Episode 변수
| 변수 | 값 | 의미 |
|---|---|---|
| Sea state (BF) | BF2, BF3, BF4, BF5, BF6, BF7 | 잔잔 → 거친 바다 |
| N_components | 1, 2, 5, 15, 20, 30 | 저 → 고 복잡도 |
| Sensor type | uniform, slope | LiDAR scan 분포 |
| Seeds | 0–4 (BF2-5), 0 (BF6-7) | 재현성 보장 |

#### Episode당 데이터
- T = 30 s, Δt = 0.1 s → 300 frames
- Ground-truth η field: 50×50 격자 (참값으로 평가용)
- LiDAR scan: ~50 valid pts/frame (raydrop 후)
- HDF5 format, 110 files

#### Test coverage
- 39 unit test (wavesim/) + 18 (wavesplat/) = **57 total, 모두 통과**
- Simulator + estimator의 정확성 보증

### Visual
- 오른쪽: BF4 LiDAR scan 시각화 (wavesim_BF4_5c_wave_lidar.png)

### Presenter notes
110 episode의 통제된 benchmark를 구축했습니다.

Episode 설계:
  - Sea state: BF2 (아주 잔잔) ~ BF7 (매우 거친) — 6단계 Beaufort scale
  - N_components: 1 (단순 단일 파) ~ 30 (복잡 다중 파) — 알고리즘이 component 수에 robust한지 검증
  - Sensor type: uniform (균등 분포), slope (거리 비례 분포)
  - Seeds: BF2-BF5는 seed 0-4 (5개씩), BF6-BF7은 seed 0 (1개씩)

Episode당 데이터:
  - 30초 시뮬레이션, 0.1초 간격 → 300 frame
  - Ground-truth η field 50×50 격자 — 평가 시 참값
  - LiDAR scan은 raydrop/shadowing 적용 후 frame당 ~50점

전체 110 episode를 HDF5 파일로 저장. Reproducibility 보장 — seed 저장, 동일 seed로 재실행 시 동일 결과.

이 dataset 자체가 paper의 contribution 일부입니다. Ehlers 2023도 비슷한 합성 benchmark를 만들었지만 1D였고, 우리는 2D directional + LiDAR sensor model까지 포함.

---

## Slide 13 — WaveSplat Algorithm (Exploration Phase)

### Title
**WaveSplat — parametric off-grid estimator**

### Subtitle
5N 파라미터 (k_x, k_y, ω, A, φ) × N개 component를 직접 추정

### Key bullets

#### State 표현
- N개 component × {A_n, k_xn, k_yn, ω_n, φ_n}
- Dispersion hard constraint: ω_n = √(g·|k_n|·tanh(|k_n|·d))
- → 자유도 감소: 5N → 4N (ω가 k에 의해 결정됨)

#### Two-stage estimation (main framing의 motivation)
- **Init phase** — multi-frame (2×T_p ≈ 1초):
  - 2-stage beamforming (v2_konly): 81×81 → 41×41 grid search
  - (k_x, k_y)만 추정, amplitude/phase는 optimizer에 맡김
  - Coarse-to-fine schedule: N = 3 → 8 → 15
- **Track phase** — per-frame (~13 ms):
  - (k_x, k_y) 고정, (ω, A, φ)만 Adam으로 update
  - Phase prediction warm-start: φ += ω·Δt

#### Adaptive N (component 수 동적 조정)
- **Energy-based pruning**: amplitude 작은 component 제거 (90% energy 보존)
- **Residual-triggered growth**: 잔여 오차 > 5cm이면 새 component 추가
- **Safety rollback**: RMSE > 130%면 이전 상태로 복귀
- Hard cap: N ≤ 30

#### Loss function
- L = Σ c_j·(η_LiDAR − η_model)² + λ_L1·Σ|A_n| + λ_smooth·Σ(A_{n+1}−A_n)²
- c_j = cos(θ)/d_norm — geometry-aware confidence weight

### Visual
- 오른쪽: pipeline figure (fig_pipeline.png)

### Presenter notes
WaveSplat의 탐색 단계 알고리즘입니다. 핵심 아이디어:

(1) State 표현 — N개 component × 5 파라미터 = 5N 자유도. Dispersion 관계로 ω가 k에 의해 결정되므로 실제 자유도는 4N. 매우 compact.

(2) Two-stage 추정 — 단일 frame에서 (k_x, k_y)는 ill-conditioned (정보 부족). 그래서 init 단계는 multi-frame으로 (k_x, k_y)만 정확히 추정하고, track 단계는 (k_x, k_y) 고정 후 매 frame (ω, A, φ)만 update. 이게 후반부 main framing 'identifiability decomposition'의 motivation입니다.

(3) Adaptive N — 처음에는 N을 모르고, episode마다 N이 다를 수 있음.
  - Pruning: amplitude 작은 component 제거 (90% energy 유지)
  - Growth: residual 크면 새 component 추가
  - Safety rollback: 새 component가 잘못되어 RMSE 폭증하면 원상복구

(4) Loss — 측정치-모델 차이 weighted squared error + L1 sparsity + spectral smoothness. Dispersion은 항상 만족하도록 hard constraint. Confidence weight는 입사각과 거리에 따라 측정 신뢰도를 반영.

이 알고리즘이 *왜 작동하는지*에 대한 분석이 다음 슬라이드부터의 main framing입니다.

---

## Slide 14 — Exploration Phase Results: vs EnKF-HOS (Wang-Pan 2021/22)

### Title
**탐색 단계 결과 — WaveSplat vs EnKF-HOS baseline**

### Subtitle
실존 논문 기반 baseline과 비교: Wang & Pan 2021/22 (JFM) family ETKF

### Result table (12-episode 3-way comparison)

| Method | Paper | Avg nRMSE | Per-frame time | 비고 |
|---|---|---|---|---|
| **WaveSplat (ours)** | — | **0.211** | **13.4 ms** | adaptive N + dispersion hard |
| ETKF-Wang2025 | Wang 2025 (Coastal Eng 197) | 0.253 | 632 ms | Ne=300, 80×60 grid, fuzzy R |
| ETKF (ours, simplified) | Wang & Pan 2021 family | 0.382 | 58.6 ms | Ne=100, 40×30 grid |

**에피소드별 핵심 패턴:**
| 조건 | WaveSplat | Wang2025 | 승자 |
|------|-----------|----------|------|
| 저복잡도 (1-5 comp) | 0.05-0.17 | ~0.25 | **WS 압도** |
| 중복잡도 (5-15 comp) | 0.17-0.30 | ~0.25 | 비등 |
| 고복잡도 (15-30 comp) | 0.29-0.31 | ~0.25 | **Wang 우위** |

### Key takeaways
- **평균 정확도**: WaveSplat 0.211 vs Wang2025 0.253 — **16% 개선**
- **속도**: 13.4 ms vs 632 ms — **47× 빠름**
- **핵심 발견**: WaveSplat은 저복잡도에서 압도적 (0.05!), 고복잡도에서 역전
  - Parametric representation의 강점: 소수 성분에서 exact fitting
  - Grid 방법의 robustness: 성분 수에 무관하게 ~0.25 수렴
- **Important**: 이 결과는 *탐색 단계* — main contribution은 *왜 이렇게 잘 되는가*의 분석

### Presenter notes
3-way 비교 설명:
  - ETKF-orig: Wang & Pan 2021 family의 simplified implementation. Ne=100, 40x30 grid.
  - ETKF-Wang2025: Wang 2025 (Coastal Eng 197)의 faithful reproduction. Ne=300, 80x60 grid, fuzzy R adaptation, Delaunay interpolation, sub-state localized ETKF. 논문 그대로 재현.
  - WaveSplat: adaptive N + multi-frame + dispersion hard constraint.

핵심 발견 — 에피소드별 패턴:
  - BF3_2comp: WS 0.052 vs Wang 0.250 — **5배 차이**! 2개 성분만 있으면 spectral Gaussian이 거의 exact fitting 가능. Grid는 아무리 정교해도 이 수준 불가.
  - BF4_15comp: WS 0.302 vs Wang 0.249 — 역전. 15개 성분이면 adaptive N이 다 따라가지 못함.
  - Wang2025의 nRMSE ~0.25 수렴 패턴 — grid 방법의 "안정된 바닥" vs WaveSplat의 "가변 성능"

이 trade-off가 곧 novelty: "새로운 representation이 어디서 좋고 어디서 한계인지 최초 규명"

---

# Part 4 — Main Framing: Why Does WaveSplat Work?

## Slide 15 — 본질 발견: Sparse Observation의 난제

### Title
**핵심 질문: 왜 50개 점으로 30-성분 파도를 추정할 수 있는가?**

### Content
- **문제**: 단일 frame에서 ~50개 LiDAR 점 → 30개 wave component 추정 (4N = 120 파라미터)
- **Spatial-only 한계**: FFT로 binning하면 grid 해상도에 의해 k_Nyq = 0.38 rad/m
  - GT k_max = 0.4-1.0 rad/m → 일부 성분 해상 불가
- **핵심 인사이트**: 시간축 + 물리법칙이 underdetermined system을 풀 수 있게 만듦

**4-way ablation 결과** (12 episodes):

| Variant | Init | Dispersion | Window | Avg nRMSE |
|---|---|---|---|---|
| A | v1 (FFT) | Free w | 1-frame | 0.213 |
| B | v2 (beam) | Free w | 3-frame | 0.208 |
| C | v2 (beam) | Hard | 1-frame | 0.226 |
| **D (WaveSplat)** | **v2 (beam)** | **Hard** | **3-frame + adaptive** | **0.211** |

- A와 D의 평균 차이는 작으나, per-episode 행동이 다름 (다음 slide)

### Presenter notes
- 청자에게 먼저 "50개 점으로 120개 파라미터를 어떻게 추정하는가"라는 직관적 질문을 던짐
- FFT grid Nyquist 설명: v1 init은 32x24 grid (240m x 200m) → dx=7.5m, dy=8.3m → k_Nyq = pi/8.3 ≈ 0.38 rad/m
- 이 한계를 넘으려면 1) 시간축 정보 활용 (multi-frame), 2) 물리 제약 (dispersion) 필요
- Ablation 결과의 핵심: 각 메커니즘이 특정 조건에서 기여하며, 결합이 robust한 성능을 제공

---

## Slide 16 — Dispersion Coupling: 물리적 정규화

### Title
**Dispersion-Coupled Estimation: 물리법칙이 정규화 역할**

### Content
- **핵심 메커니즘**: w = sqrt(g|k|tanh(|k|d)) → k 추정 시 w가 자동 결정
  - 4N 자유 파라미터 → 3N으로 감소 (25% DOF 절감)
  - 물리적으로 의미 없는 (k, w) 조합 배제

- **Ablation 세부 분석**:
  - **D가 A보다 확실히 좋은 경우** (>5% 개선):
    - BF4_1comp (+41.5%), BF3_5comp (+38.8%), BF3_2comp (+18.7%)
    - 공통점: 저~중 복잡도, 성분 수 적음 → dispersion이 정규화로 작동
  - **A가 D보다 좋은 경우** (>5%):
    - BF5_15comp (-31.8%), BF3_5comp_slope (-23.0%), BF4_15comp (-18.9%)
    - 공통점: 고복잡도 또는 고해상도 → dispersion 제약이 init 오류를 전파

- **결론**: Dispersion coupling은 "마법"이 아니라 **물리 기반 정규화**
  - 적은 성분에서 강력, 많은 성분에서는 정확한 init이 더 중요

### Presenter notes
- Dispersion 제약의 이중성을 솔직하게 설명
- "왜 항상 좋지 않은가?" — dispersion은 k → w 매핑. k가 틀리면 w도 틀림. Free w (variant A, B)에서는 w가 보상 가능
- 이것이 "init이 중요하다"는 결론으로 이어짐 → beamforming v2의 필요성
- 현재 한계: 고복잡도(30comp)에서 beamforming init 자체의 k 추정이 불완전
- 향후 작업: init 개선 (더 정밀한 beamforming, 또는 iterative refinement)

---

## Slide 17 — 메커니즘 분해: Identifiability-aware Sequential Estimation

### Title
**Init-Track 분리의 근거: Identifiability 비대칭**

### Content
- **관찰**: k_x, k_y는 multi-frame에서만 추정 가능 → Init phase에서 고정
- **Tracking에서는 w, phi만 업데이트** (tracking_mode = 'w_phi')
  - k를 tracking에서 업데이트하면 overfitting (기존 문제, 2026-04-06 fix)

- **Beamforming Init (v2_konly)**:
  1. Multi-frame 관측 수집 (2Tp window)
  2. (kx, ky) grid에서 dispersion-aware beamforming
  3. Coarse → Fine refinement
  4. k 고정 후 tracking phase 진입

- **Tracking Phase**:
  1. Phase prediction: phi += w * dt (dispersion으로 w 계산)
  2. Gradient descent: w, phi만 10 steps
  3. Adaptive N: energy 기반 pruning + residual 기반 growth

- **결과**: 이 분리 덕분에 13.4 ms/frame 실시간 달성

### Figure placeholder
[fig_beamforming_2stage.png — 이미 paper/figures/ 에 존재]

### Presenter notes
- Fisher Information 관점: 단일 frame에서 k의 FIM eigenvalue는 w, phi보다 1000배 작음
- 이는 k를 단일 frame에서 추정하면 불안정하다는 의미 → multi-frame init 필요
- w_phi tracking mode의 발견 경위: 처음에는 all-param tracking → overfitting 문제 발생
- k-freeze + w_phi로 전환 후 tracking 안정화 (2026-04-06 fix)
- 이 설계가 "identifiability-aware"라고 부를 수 있는 근거

---

# Part 5 — Limitations & Future

## Slide 18 — 한계 + Nonlinear 확장 + PFT-PINN 대응

### Title
**현재 한계와 향후 확장 방향**

### Content

**인정해야 할 한계:**
1. **Linear model** — z(x,y,t) = Sum cos() → Stokes 2nd order 이상 미포함
   - BF6-7 고해상도에서 nonlinear 효과 커짐
2. **고복잡도에서 grid 방법에 역전** — 15-30 성분에서 Wang2025(0.25)에 WaveSplat(0.30) 패배
   - Adaptive N이 많은 성분을 동시에 추정하기 어려움
   - 향후: hybrid 접근 (spectral Gaussian + grid refinement)
3. **Synthetic only** — 실측 데이터 미검증
4. **No breaking / No velocity field**

**향후 확장 경로:**
1. **고복잡도 대응**: hybrid 접근 (WaveSplat coarse + grid refinement)
2. **Stokes 확장**: 2nd order nonlinear terms
3. **Hybrid PINN refinement**: WaveSplat → coarse init → PFT-PINN fine-tune
4. **실측 검증**: 선박 LiDAR 데이터 (Avikus)

**PFT-PINN 대응** (Ehlers 2025):
- PFT-PINN: 30분/instance, offline → 우리와 응용 영역 다름
- 가능한 협업: WaveSplat으로 real-time coarse estimation → PFT-PINN으로 offline refinement

### Presenter notes
- "linear model"이 가장 큰 한계. 하지만 우리의 target application (real-time control)에서는 충분
- Stokes 확장은 model에 2nd-order term 추가로 비교적 straightforward
- PFT-PINN과의 관계: 경쟁이 아니라 보완. 시간축이 다름 (real-time vs offline)
- 실측 데이터 검증은 PhD 과정에서 반드시 필요한 단계
- Wang 2025는 받는 즉시 9-item checklist로 분석하고 baseline 추가

---

## Slide 19 — Timeline + Venue + Q&A

### Title
**향후 일정 및 목표**

### Content

**논문 1: Ocean Engineering 투고**
- 현재: 코드/실험 완료, 논문 초안 1500줄
- 목표: 2026년 6-7월 투고
- 내용: WaveSplat algorithm + 110-episode validation + EnKF-HOS comparison

**논문 2 (PhD 확장):**
- Nonlinear extension (Stokes 2nd)
- 실측 LiDAR 데이터 검증
- Hydrofoil controller와 연동

**학회:**
- OCEANS 2026: 발표 목표
- ISOPE 2027: 확장 논문

**이번 발표의 핵심 정리:**
1. Sparse LiDAR + real-time + 2D directional → 기존 방법 없음
2. Parametric off-grid + dispersion hard → 물리적 정규화
3. Init-Track 분리 → identifiability 기반 설계
4. Ablation: 물리 제약 + 시간 융합의 상보적 기여 확인

**Q&A**

### Presenter notes
- Ocean Engineering 논문은 IEEE OCEANS format에서 변환 중
- 핵심 reviewer concern 예상: (1) linear only, (2) synthetic only → Future work에서 이미 대응
- PhD timeline: 논문 1 (2026 H2) → 논문 2 실측+nonlinear (2027) → 논문 3 controller integration (2027-2028)
- Q&A 예상 질문:
  - "왜 linear?" → real-time 제약 + Stokes 확장 계획
  - "실측은?" → Avikus LiDAR 데이터 접근 가능, 다음 논문에서
  - "PFT-PINN보다 뭐가 좋은가?" → 실시간 (13ms vs 30min), 2D directional, training-free
  - "Oracle과의 gap은?" → 0.211 vs 0.087 = k 추정 오차, init 개선이 핵심

---

# (Day 4 작업 — Slide 15-19 draft 완료, ablation 결과 반영)

---

# PPT 생성 완료 (2026-05-08)

**Generator script**: [scripts/generate_labmeeting_2026_05_13.py](../scripts/generate_labmeeting_2026_05_13.py)
**Output PPT**: [docs/labmeeting_2026_05_13.pptx](labmeeting_2026_05_13.pptx)
**Slides**: 19 main + 7 backup = 26 total
**Size**: ~514 KB

## 재생성 명령어

```bash
cd /home/saeyong/reserch/wavesplat
python3 scripts/generate_labmeeting_2026_05_13.py
```

## 19 main + 7 backup 모두 PPT 에 들어감

| # | Status | Content |
|---|---|---|
| S1-S3 | ✅ Day 1 full content | Title + Motivation + Problem |
| S4-S10 | ✅ Day 2 full content | 5-category + 5 prior art detail + matrix |
| S11-S14 | ✅ Day 3 full content | WaveSim + exploration phase |
| S15-S17 | ✅ Day 4 full content (ablation 반영) | Ablation + mechanism + identifiability |
| S18-S19 | ✅ Day 4 full content | Limitations + Plan |
| A1-A7 | ✅ Backup outline | Supporting prior art |

> **Note**: Day 3-5 의 slides 도 generator script 에 outline + key bullets + presenter notes 포함되어 있음. 각 day 작업은 *content refinement* (figure 추가, content 강화) 임. PowerPoint 에서 직접 다듬어도 되고, generator script 를 수정 후 재생성해도 됨.

## Figure 자리 (placeholder 들)

생성된 PPT 에서 placeholder 로 남은 figure (PowerPoint 에서 paper PDF 캡처해서 수동 삽입):

- S5: Desmars 2023 setup + result figures (paper Fig. 2, 8)
- S6: Wang-Pan EnKF-HOS pipeline + predictable zone
- S7: Ehlers 2023 FNO architecture + SSP results
- S8: Ehlers 2025 PFT-PINN architecture + MARATHON results
- S9: Yan 2025 VOF-EnKF pipeline + plunging breaker (Fig. 18)

이미 삽입된 figure (PNG path 존재 시):
- S2: hydrofoil_controller_flow.png
- S3: wavesim_BF4_5c_wave_lidar.png
- S11: wavesim_BF4_5c_wave_field.png
- S12: wavesim_BF4_5c_wave_lidar.png
- S13: paper/figures/fig_pipeline.png
- S17: paper/figures/fig_beamforming_2stage.png

---

# 다음 작업

## 완료 (2026-05-08)
- Wang 2025 정독 + faithful 구현 + 12-episode 3-way 비교 ✅
- WASSfast 코드 분석 + novelty 평가 ✅
- Slide md 업데이트 (S4, S6, S14, S18) ✅

## 남은 작업 (랩미팅 전)
1. **PPT 재생성**: `scripts/generate_labmeeting_2026_05_13.py` 수정 후 재실행
2. **Paper figure 추출 + 삽입** (PowerPoint에서 직접)
3. **Rehearsal** (5/12 또는 5/13 오전)

## 남은 작업 (랩미팅 후 — 논문)
1. Related Work에 WASSfast, Wang 2025 추가 + novelty 프레이밍 강화
2. Wang 2025 비교 결과 Table 반영
3. "Sparse LiDAR 문제 설정의 새로움" Introduction에 명시

---
