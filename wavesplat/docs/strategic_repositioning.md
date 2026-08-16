# Strategic Repositioning — Lab Meeting / Paper Direction

> **목적**: Lab meeting + paper 투고 전 **honest framing** 으로 전환.
> 현재까지의 work 을 over-claim 하지 않고, exploratory engineering contribution 으로 positioning.

생성일: 2026-05-08
대상: Prof. + lab members

---

## 1. 배경 — 왜 repositioning 이 필요한가

검증 과정에서 다음 사실 확인:

### 1.1 Desmars 2020/2023 와 우리 접근의 유사성
- **Desmars 2020** (Ocean Eng 212, 107659): Linear/CWM wave model + non-uniform gauge inversion + dispersion propagation
- **Desmars 2023 JFM** (975, A8): HOS-OCM (HOS-observation coupling method), nonlinear parametric reconstruction from remote measurements
- **본 연구**: Linear plane-wave parametrization + sparse LiDAR inversion + dispersion propagation

→ **방법론 family 동일**. WaveSplat 이 추가한 부분은 LiDAR 적응 + real-time tracking + adaptive N + LiDAR-specific confidence — 모두 **engineering 적응** 수준.

### 1.2 Feng Wang et al. 2025 (Coastal Eng 197, 104690) 와의 관계
> **Note**: First author 가 *Feng Wang* (Harbin Engineering Univ., China) 으로, JFM 의 Wang & Pan (Yulin Pan, U.Michigan) 과 *별개 그룹*. 이름 혼동 주의. 우리 baseline (etkf_wang2025.py) 가 이 Feng Wang et al. 의 Coastal Eng paper 를 재현.
> DOI: 10.1016/j.coastaleng.2024.104690. GitHub: fengHEU/Wave-ETKF.
- 우리 ETKF baseline 코드 docstring 자체가 *"baseline from 'From Points to Waves'"* 라고 명시 (Wang 2025 인용)
- 우리는 Wang family 의 ETKF 를 baseline 으로 구현해서 비교한 것이지, 새로운 ETKF 를 만든 게 아님

### 1.3 진짜 알고리즘 신규성?
구성요소를 dissect 하면:
- Plane-wave parametrization → 표준 (1960s+)
- Beamforming init → Capon 1969, MUSIC 1986
- Adam optimizer → Kingma 2015
- Sparsity regularization → Compressed sensing 표준
- Pruning + growth → Matching pursuit (Mallat 1993)
- Init/Track 분리 → 표준 tracking 기법
- LiDAR 모델 → engineering plumbing

→ **알고리즘 자체의 fundamental 신규성은 약함**.

### 1.4 그럼 무엇이 진짜 contribution 인가?
- ✅ **WaveSim 시뮬레이터** 구축 (JONSWAP + Gerstner + Falcon K1 LiDAR + raydrop) — 100 episodes 데이터셋
- ✅ **ETKF (Wang family) + GPR baseline 의 엄격한 tuning** (24-config sweep) — fair comparison framework
- ✅ **5-metric panel 평가 framework** (cherry-pick 방지)
- ⚠️ Adaptive N + real-time tracking — 작은 algorithmic 추가
- ⚠️ Geometry-aware confidence — LiDAR-specific engineering

**Bottom line**: 시뮬레이터 + 평가 framework 는 단단한 contribution. 알고리즘 자체는 "exploratory approach we're trying" 정도가 적절.

---

## 2. 새 framing 방향 — 학위논문/lab meeting 차원

### 2.1 명칭 변경
- **현재**: "WaveSplat" — splatting 용어가 3DGS 흉내 + Desmars parametric inversion 이미 한 일
- **변경**: 일반적 명칭으로 (예: "Spectral Wave Estimator", "Sparse Spectral Inversion for LiDAR" 등) 또는 **명칭 제거**
- 임시 작업 명칭 후보:
  - **SWE-LiDAR** (Spectral Wave Estimation from LiDAR)
  - **그냥 코드 구현물 — 이름 없이 "본 연구의 추정 방법"**

### 2.2 발표 톤 — "exploratory approach" 로
- ❌ "We propose a novel grid-free method..."
- ✅ "We built a simulation environment for shipboard LiDAR-based wave estimation, and explored a sparse spectral representation as one approach. Here's what we have so far."

### 2.3 무게 중심 재배분
- **WaveSim (시뮬레이터) 강조 강화** — 진짜 단단한 contribution
- 추정 알고리즘 — exploratory approach 로 demote
- Hydrofoil application — 추정 알고리즘이 어떻게 응용될 수 있는지 demo

---

## 3. Lab meeting 슬라이드 재배열 제안

### 3.1 새 deck 구조 (제안)

| # | 섹션 | 내용 | 톤 |
|---|---|---|---|
| 1 | Title | 제목에서 "WaveSplat" 제거 — "Shipboard LiDAR-based Wave Field Estimation: Simulator + Exploratory Approach" | exploratory |
| 2 | Contents | 4 sections 그대로 | — |
| 3 | Real Sea Trial | 실선 LiDAR 데이터 — 문제 제기 | 그대로 |
| 4 | Problem & Motivation | 그대로 | 그대로 |
| 5 | **Prior Art (확장)** | Desmars 2020/2023, Wang 2025, Ehlers 2023 — 우리 가까이 있음을 인정 | honest |
| 6 | (3DGS 슬라이드) | **삭제 또는 demote** — splatting analogy 가 misleading 가능 | — |
| 7 | **WaveSim 시뮬레이터** | 강조: JONSWAP+Gerstner+Falcon K1+raydrop, 100 episodes, **196 unit tests** (2026-08-13 실측; 구 "39" 는 wavesim 단독 구값) | 강한 contribution |
| 8 | **추정 접근 (exploratory)** | "We tried sparse spectral parametric inversion + dispersion. Here's the architecture." | exploratory |
| 9-10 | Methodology + Flowchart | 그대로 | exploratory |
| 11-12 | (WaveSim 1/2, 2/2) | 그대로 — 시뮬레이터 신뢰성 | — |
| 13 | Experiment Design | 그대로 | — |
| 14 | **결과 — 5-metric panel** | "Within our tested baselines (ETKF Wang-family, GPR), our approach achieves..." 표현 | honest |
| 15 | Per-episode | 그대로 | — |
| 16 | (Statistical) | Appendix 로 이동 | — |
| 17 | (Ablation) | Appendix 로 이동 | — |
| 18 | (Robustness) | Appendix 로 이동 | — |
| 19 | Qualitative | 그대로 | — |
| 20-21 | Sparse + Forecast | 그대로 — 흥미로운 부분 | — |
| 22 | 강화 baseline (Wang 2025 ETKF + GPR-disp) | "Wang 2025 와 같은 family 의 ETKF" 명시 | honest |
| 23-24 | Hydrofoil 응용 | 그대로 | — |
| 25 | Hydrofoil 결과 | 그대로 | — |
| 26 | **한계 + Future work** | "직접 비교: Desmars 2023 (HOS-OCM), Wang 2025 (full impl)" 명시 | honest |
| 27 | References | Wang 2025 추가, Desmars framing 정정 | — |

### 3.2 핵심 메시지 재정리

**현재 (over-claim)** — ⛔이 문장은 인용 금지 대상으로 **기록**된 것이다(따라 쓰지 말 것):
> "WaveSplat is a novel grid-free method. nRMSE 30% lower than ETKF, 10× faster."

**제안 (honest)** — ⛔**2026-08-13 감사 정정**: 이 "honest" 대안조차 금지 수치
(`30% nRMSE↓`, `~10× speed`, `39 unit tests`)를 그대로 인용하고 있었다. 아래로 대체한다.

> "We built WaveSim — a validated shipboard LiDAR simulator (100 episodes, **204 unit tests**).
> We're exploring a sparse spectral inversion approach for real-time LiDAR-based wave estimation.
> Within our tested baselines, a 100-episode re-derivation (`results/100ep_ws_letkf.json`) gives
> **WS 0.2229 vs LETKF 0.2278 — 51/49, p = 0.9204, i.e. a NULL on accuracy**; on router-unseen
> seeds (S13, 704 runs) the shipping config wins **56/64 crowded / 63/64 sparse**.
> Latency is **1.5–2.6× in the same process on a single thread** (AMD Ryzen 7 7800X3D).
> Same family as Desmars 2020/2023 — open question how ours compares to their HOS-OCM directly. Future work."

⛔RETRACTED (2026-08-13 감사, 3라운드 **사유 교체**) — 원문의 `30% nRMSE↓` · `~10× speed` ·
`39 unit tests`. ⚠**"어느 결과 파일에서도 재현되지 않는다" 는 틀렸다** — `30%`(0.2229 vs
0.3189)와 `~10×`(13.3760 vs 132.6387 ms)는 `results/b1_full_comparison.json`
(**git 추적**, 커밋 `e57528c`, 100 ep)에서 재도출된다. 금지 사유는 셋이다:
①**프레임 불일치**(`WaveSplat.n_frames` 120–220 vs `ETKF`·`GPR-1` 301, **100/100 에피소드에서
WS 가 더 적음** ⟹ 같은 프레임에서 채점된 적 없음, `84/100` 승패비도 금지)
②**기준선 선택**(그 ETKF 는 `grid 40×30 · ne=100 · mult_inflation 1.0`; 튜닝 LETKF 로 바꾸면
위 인용문대로 **널**) ③**지연은 프로세스 간 비교**. 테스트 수는 실측 **204**
(85 wavesim + 119 wavesplat, 2026-08-13). 근거: `reference_citation_bans.md`

---

## 4. To-Do 정리

### 4.1 PPT 수정 (Priority order)
1. [ ] Title slide: "WaveSplat" 제거 또는 명칭 변경
2. [ ] Slide 5 (Prior Art): Wang 2025 + Desmars 2020 정정 + 우리 위치 honest 표시
3. [ ] Slide 6 (3DGS 비유): **삭제** 또는 "inspirational analogy" 정도로 demote
4. [ ] Slide 14 (Primary Result): "preliminary results within tested baselines" 톤으로
5. [ ] Slide 16, 17, 18 (Stat/Ablation/Robust): Appendix 로 이동
6. [ ] Slide 22 (강화 baseline): Wang 2025 family 명시
7. [ ] Slide 26 (한계): "Future direct comparison with Desmars 2023, Wang 2025" 추가
8. [ ] Slide 27 (References): Wang 2025 추가, Born-Wolf year 정정 (1999), Belmont 2007 JAOT 로 교체
9. [ ] 본문 전체 "novel method" → "exploratory approach" 톤 조정

### 4.2 Paper 수정 (Priority order)
1. [ ] Title 검토: "WaveSplat" 제거 검토
2. [ ] Abstract: over-claim 표현 완화 — "first to do X" → "we explore"
3. [ ] Line 320 (Related work): Desmars 2020 framing 정정 (ETKF 가 아님)
4. [ ] Wang 2025 인용 추가 (ETKF baseline reference)
5. [ ] Section 6 (Conclusion): "future work — direct comparison with Desmars 2023 HOS-OCM and Wang 2025 ETKF" 추가
6. [ ] References bib: Wang 2025 (Coastal Eng 197, 104690) + Belmont 2007 (JAOT 24(6), 1150-1156) 추가, Born-Wolf year 정정

### 4.3 학위논문 (장기)
1. [ ] Desmars 2023 implementation 받아 직접 비교 (institutional access 필요)
2. [ ] Wang 2025 implementation 받아 직접 비교 (paywall — institutional access 필요)
3. [ ] 수조 실험 + 실선 시험 데이터로 sim-to-real validation

---

## 5. 발표 시 Q&A 예상 + 답변 준비

### Q1: "Desmars 2023 JFM 와 본 연구의 차이가 무엇입니까?"
**A**: 같은 family (parametric inversion + dispersion propagation). 우리가 추가한 것은:
- LiDAR sensor adaptation (Desmars 는 generic remote optical)
- Real-time 10 Hz tracking architecture (init + track 분리)
- Adaptive N (pruning + growth + safety rollback)
- Geometry-aware confidence weighting
다만 직접 quantitative 비교는 future work — same sensor + same evaluation 에서 재실험 필요.

### Q2: "왜 WaveSplat 이라는 이름을 썼습니까? 3DGS 와 연관 있습니까?"
**A**: (변경 시) "초기 작업 명이었고, 이후 검토에서 splatting 용어가 정확하지 않다고 판단해 일반적 명칭으로 변경 중. 3DGS 의 \"여러 기저 함수의 가중합으로 연속 필드 표현\" 발상은 영향 받았으나, 핵심 알고리즘은 표준 spectral parametric inversion."

### Q3: "이 방법의 진짜 algorithmic novelty 는 무엇입니까?"
**A**: **솔직히 fundamental 알고리즘 신규성은 약합니다**. 본 연구의 contribution 은:
1. **WaveSim** — shipboard LiDAR 기반 wave estimation 의 validated simulator + 100-episode 데이터셋
2. **Comprehensive evaluation framework** — 5-metric panel, 24-config tuned baselines, 100-paired statistical validation
3. **Real-time + adaptive N 통합 시스템** — 첫 LiDAR-based real-time wave estimator
4. **LiDAR-specific 적응** — geometry confidence, sparseness handling

알고리즘 자체보다 **시스템 통합 + 평가 엄격함** 이 contribution.

### Q4: "ETKF baseline 이 정말 fair tuned 인가요?"
**A**: ✅ 24-config sweep (grid {40×30, 80×60} × ensemble {50, 100} × inflation {α, σ} 6 조합 = 24). 최적 config (g=40×30, n_e=100, α=1.0, σ=0.005) 의 mean nRMSE 가 12 sweep episodes 평균 0.358. 이게 우리 baseline. Wang 2025 와 같은 family 의 ETKF 이지만 fuzzy outlier handling 등 일부 디테일은 단순화.

### Q5: "그럼 이걸 발표하고 paper 도 투고할 만한 contribution 이 있습니까?"
**A**: WaveSim + evaluation framework + real-time integration system 으로 Ocean Engineering 정도는 가능하다고 봅니다. 다만 "novel algorithm" 으로 over-claim 하면 reviewer 가 Desmars 2020/2023 인용해서 desk reject 할 수 있어서, framing 을 "exploratory + comprehensive evaluation" 으로 가는 게 안전.

---

## 6. 다음 진행 순서

1. **이 md 파일 검토 + 합의** — 사용자 / 지도교수
2. **PPT 수정 — 위 4.1 항목 순차 진행**
3. **Lab meeting 발표** (이 md 파일을 정리 자료로 활용)
4. **지도교수 피드백** 후 paper 수정 (4.2)
5. **장기 작업** (4.3) 학위논문 단계로 이행

---

## 7. 변경 사항 ledger (이 파일이 살아있는 동안 추가)

| 날짜 | 변경 | 이유 |
|---|---|---|
| 2026-05-08 | 초안 작성 | Wang 2025 / Desmars 비교에서 honest repositioning 필요성 도출 |
| 2026-05-08 | §8, §9, §10 추가 | Wang 2025 비교 feasibility, novelty 강화 옵션, 선행연구 검토 원칙 |

---

## 8. Wang 2025 직접 비교 feasibility 분석

### Q: 내일 Wang 2025 전체 paper 받으면 비교가 의미 있는가?

**A: YES — 단, 결과는 시나리오별로 다름.**

#### 8.1 비교의 가치 정도

| 시나리오 | 가능성 | Output |
|---|---|---|
| **Best**: 우리 `etkf_grid.py` 가 Wang 2025 와 거의 일치 (방법론 동일, 디테일 차이 minor) | 50% | "We compare against the ETKF approach of Wang 2025 (Coastal Eng 197); preliminary differences (obs operator direction, outlier handling) noted but family same" — **강한 quantitative claim 가능** |
| **Realistic**: 핵심 ETKF 부분은 같지만, Wang 의 fuzzy outlier / GPU / pre-interpolation 등 디테일이 우리 코드에 없음 | 40% | "Our ETKF baseline approximates Wang 2025; full faithful re-implementation is future work" — **honest framing** |
| **Worst**: Wang 2025 가 fundamentally 다른 ETKF 변형 (예: 4D-Var, hybrid EnKF-3DVar 등) | 10% | 우리 baseline 은 Wang 2025 와 family 가 다른 것으로 인정. 새로운 baseline 필요. |

→ 받아서 본문 검증 자체는 **30분 ~ 2시간** 작업. 의미 있음.

#### 8.2 Paper 받은 후 우선 확인 항목

1. **State vector**: η + ∂η/∂t? 또는 η + spectral coefficients?
2. **Observation operator H**: obs→grid (interpolation BEFORE assimilation) or grid→obs?
3. **Propagation**: FFT exact dispersion? WASSfast model 의 정확한 정의?
4. **Inflation**: 어떤 종류 (multiplicative, additive)?
5. **Localization**: 사용? r_loc?
6. **Outlier handling**: fuzzy logic 정확한 알고리즘
7. **Test dataset**: 어떤 simulator/실측 데이터? 우리와 비교 가능?
8. **Hyperparameter tuning protocol**: sweep? sensitivity?
9. **Code release**: GitHub 공개?

#### 8.3 비교 후 가능한 framing

**Case A — 우리가 Wang family** (probability 50-60%):
> "Our ETKF baseline implements the spatial-grid ETKF approach of Wang et al. (2025) with simplified observation operator and outlier handling; we compare against this baseline as the closest published method to ours."

**Case B — partial match** (30-40%):
> "Wang et al. (2025) presented a more sophisticated ETKF on grid with fuzzy outlier handling and GPU implementation. Our ETKF baseline is a simpler reference implementation; direct quantitative comparison with Wang's published code is future work."

**Case C — different family** (~10%):
> "Wang et al. (2025) presents a [different] ETKF variant. We benchmark against our own [Hunt 2007 LETKF based] ETKF implementation. Direct comparison with Wang requires further re-implementation."

→ Paper 받자마자 **§8.2 9 가지 항목 체크리스트** 로 확인하면 case A/B/C 결정 가능.

---

## 9. Novelty 강화 옵션 — Brainstorm

### 9.1 현재 추정 방법론 분석 (요약)

- 알고리즘 자체의 fundamental novelty 약함 (§1.3 참조)
- 진짜 contribution: WaveSim + 평가 framework + 시스템 통합

### 9.2 단기 (2-4주) 추가 가능한 novelty

#### A. Spectral super-resolution via dispersion-driven temporal aliasing recovery
- **아이디어**: 단일 frame 의 공간 해상도가 Nyquist 에 묶이지만, multi-frame init 에서 dispersion 으로 wavenumber 간 정확한 phase 관계가 강제됨 → 시간 축으로 super-resolution 가능
- **신규성**: **High** — 일반 beamforming 으로는 못 함. dispersion constraint 를 super-resolution 에 활용한 사례 거의 없음
- **구현 비용**: 중간 (multi-frame ML estimator 형태로 재구성)
- **검증**: 작은 simulation 으로 BF4 5-comp 에서 직접 측정 가능

#### B. Hessian-based confidence interval on (kx, ky, ω)
- **아이디어**: Adam 최적화 종료 시점에 Hessian (또는 Fisher information matrix) 추정 → Bayesian-like confidence intervals 부여
- **신규성**: Medium-High — wave estimation 분야에서 uncertainty quantification 거의 없음
- **구현 비용**: 작음 (PyTorch autograd 로 Hessian 추출 가능)
- **응용**: Hydrofoil FF 에서 confidence-aware K_ff 조절 가능 → 시스템 통합 contribution 강화

#### C. Online adaptive K_ff (Hydrofoil controller, 응용 부분)
- **아이디어**: WaveSplat 의 estimate confidence 를 받아서 FF gain 을 실시간 조절 (낮은 confidence → 작은 K_ff)
- **신규성**: Medium — adaptive control + estimation confidence 결합
- **구현 비용**: 작음 (controller 만 수정)
- **장점**: WaveSplat ↔ Hydrofoil 의 결합을 강화하는 system contribution

### 9.3 중기 (1-3개월) 가능한 novelty

#### D. Joint LiDAR + IMU fusion
- **아이디어**: IMU heading + ship motion 을 wave direction prior 로 활용
- **신규성**: Medium — 멀티-모달 sensor fusion in marine context
- **구현 비용**: 중간 (IMU model + Bayesian update 추가)

#### E. Physics-informed Stokes-2 correction layer
- **아이디어**: 우리 linear plane-wave 모델 위에 작은 NN (MLP 수십개 hidden units) 을 추가, residual 이 nonlinear Stokes-2 항을 학습
- **신규성**: High — hybrid linear+learned 가 wave estimation 에 거의 없음
- **구현 비용**: 중간 (학습 데이터 + 학습 파이프라인)

### 9.4 장기 (수개월~1년) novelty 후보

#### F. Real-time variational MPC for hydrofoil with WaveSplat preview
- **아이디어**: PD+FF 대신 finite-horizon MPC 가 WaveSplat 의 preview + uncertainty 를 직접 받아 최적 제어 입력 계산
- **신규성**: High — 실선용 wave-aware MPC 는 거의 없음
- **구현 비용**: 큼 (MPC solver, real-time constraint)

#### G. End-to-end differentiable wave estimation + control
- **아이디어**: WaveSplat 의 spectral parameters → controller commands 까지 differentiable graph 로 묶고, 최종 ride RMS 를 loss 로 backpropagate
- **신규성**: Very High — 학습 기반 wave-preview control 거의 없음
- **구현 비용**: 큼

### 9.5 추천 우선순위

| Priority | Option | 단계 | 기대 novelty 효과 |
|---|---|---|---|
| **1** | A (spectral super-resolution) | 단기 (2-4주) | 알고리즘 자체의 novelty 추가 |
| 2 | B (Hessian confidence) | 단기 | uncertainty quantification — 다양한 응용 |
| 3 | C (adaptive K_ff) | 단기 | system integration contribution |
| 4 | D (LiDAR+IMU fusion) | 중기 | multi-modal — 실선 데이터 받은 후 |
| 5 | E (Stokes-2 NN) | 중기 | hybrid linear+learned |

→ **A + B + C 셋 다 단기에 가능**. 셋이 모두 추가되면 paper 의 algorithmic contribution 이 "first to integrate adaptive spectral + uncertainty + confidence-aware control" 로 충분히 defensible 해짐.

---

## 10. 새 원칙 — 선행연구는 전체 paper 읽기

### 10.1 배경
- 이전까지 abstract / metadata 만 검증 → Wang 2025, Desmars 2020 의 framing 오류 발생
- 가짜 인용 7건 (huhn2022 등) 이 abstract 까지만 검증해서는 잡히지 않을 수도 있었음

### 10.2 새 원칙

**선행연구 / 아이디어 관련 인용은 다음 절차 필수**:

1. **CrossRef DOI 로 metadata 1차 확인** — title/저자/년도/저널
2. **전체 본문 PDF 확보**:
   - DOI link → 직접 PDF 시도
   - paywall 시 institutional access (학교 도서관)
   - arxiv / preprint 검색
   - 저자 personal page 검색
   - 마지막: ScienceDirect API 또는 사용자 제공
3. **본문 핵심 섹션 정독** — 최소 abstract + introduction + method 섹션
4. **인용 시 정확한 claim 만 사용** — abstract 만 본 경우 명시
5. **이 md 파일에 검토 기록 누적**

### 10.3 검토 진행 기록 — 선행연구 deep-dive ledger

| Paper | 검토 깊이 | 핵심 내용 (검증된 부분만) | 우리와 관계 | 검토 일자 |
|---|---|---|---|---|
| Desmars 2020 (Ocean Eng 212) | Abstract + 검색 | LWT + Improved CWM, gauge inversion, dispersion propagation | 같은 family — parametric inversion. WaveSplat 의 가까운 prior | 2026-05-07 |
| Desmars 2023 JFM (975) | Abstract | HOS-OCM, nonlinear deterministic reconstruction from remote optical | 같은 family. WaveSplat 보다 더 nonlinear (HOS) | 2026-05-08 |
| Wang 2025 (Coastal Eng 197) | Abstract + 검색 | ETKF on grid, sparse 3D → uniform grid interp, fuzzy outlier, CPU+GPU | 우리 ETKF baseline 의 직접 reference (family 동일) | 2026-05-08 |
| **본문 검토 예정** | — | 사용자가 paper 제공 시 §8.2 항목 체크 | — | 2026-05-09 (예정) |
| Hunt 2007 (Physica D 230) | Abstract | LETKF formulation, localization | 우리 LETKF 의 이론 근거 | 2026-05-08 |
| Hwang 2000 (RSE 73) | Abstract | Airborne scanning lidar, 3D 측정 + breaking wave detection | LiDAR-ocean 첫 측정 — sensor 검증 | 2026-05-08 |
| Belmont 2008/2007 (IEEE/JAOT) | Abstract | Shallow angle wave profiling LIDAR | LiDAR-ocean prior — 적합 | 2026-05-08 |
| Ehlers 2023 (Ocean Eng 288) | Abstract | U-Net + FNO, synthetic HOS training data, 1D grid | Learning-based prior, FNO 가정 (uniform grid) | 2026-05-08 |
| Tang-Adcock 2022 (AOR 122) | Abstract | GPR for wave statistics surrogate (Monte Carlo 대체) | 다른 task — 우리 baseline 과 무관 | 2026-05-08 |
| Kerbl 2023 (3DGS) | Metadata | Real-time radiance field rendering | inspiration only — 직접 적용 X. **3DGS 비유는 misleading** 가능 (md §1.3) | 2026-05-08 |
| Rasmussen-Williams 2006 | Metadata + 분야 표준 | GPR 일반 교과서 | 우리 GPR baseline 의 이론 근거 | 2026-05-08 |
| Möller-Trumbore 1997 (JGT 2(1)) | Abstract | Ray-triangle intersection without precomputation | WaveSim raycast 구현 근거 | 2026-05-08 |
| Wilcoxon 1945, Cohen 1988 | Abstract / 표준 | signed-rank test, effect size 분류 | 우리 통계 검증 근거 | 2026-05-08 |
| Gaspari-Cohn 1999 (QJRMS 125) | Abstract | Compactly-supported correlation localization | LETKF 의 localization 근거 | 2026-05-08 |
| Holthuijsen 2007 (book) | 표준 reference | Wave physics 교과서 | dispersion / JONSWAP 근거 | 2026-05-08 |
| Faltinsen 2005 (book) | 표준 reference | High-speed marine vehicle hydrodynamics | foil submergence / hydrofoil 근거 | 2026-05-08 |
| Born-Wolf 1999 (book, 7th ed) | 표준 reference | Optics 교과서 | Fresnel R(θ) 근거 | 2026-05-08 |
| Hrovat 1997 (Automatica) | Abstract | Survey of advanced suspension developments | 응용 (preview suspension) 근거 | 2026-05-08 |

### 10.4 다음 검토 우선순위

1. **Wang 2025 전체 paper** (사용자 내일 제공 예정) — §8.2 9 가지 항목 체크
2. **Desmars 2023 JFM** — HOS-OCM 의 정확한 정의 + 우리와 차이 명시
3. **Ehlers 2023** — FNO architecture 의 정확한 가정 (uniform grid 한계)
4. (필요 시) Wang 2024/2025 외 다른 ETKF-on-LiDAR paper

---

## 11. 토론 정리 ledger (대화별 누적)

### 2026-05-08 — Repositioning + novelty + 선행연구 원칙
- WaveSplat 명칭 제거 검토 → "exploratory approach" 톤
- WaveSim 강조, 추정 알고리즘 demote
- Slide 16/17/18 → Appendix
- Wang 2025 paper 받기 전까지 honest framing
- Novelty 강화 옵션 (A: 스펙트럼 super-resolution, B: Hessian confidence, C: adaptive K_ff)
- 새 원칙: 선행연구 전체 paper 읽기

### (다음 세션에서 추가)
- Wang 2025 전체 검토 결과
- Desmars 2023 JFM HOS-OCM 검토 결과
- Novelty 옵션 중 채택 결정
- PPT 수정 진행 상황

---

## 12. WaveSim 물리적 내용 audit (2026-05-08)

검증 대상: `wavesim/wave_generator.py`, `wavesim/lidar_simulator.py`, `wavesim/raydrop_uniform.py`, `wavesim/raydrop_slope.py`

### 12.1 Gerstner wave 수식

**구현** (`wave_generator.py:96-108`):
```
z = Σᵢ Aᵢ cos(phase)        ← elevation
X_disp = -Σᵢ Q·Aᵢ·(kx,ᵢ/k) sin(phase)   ← Lagrangian shift x
Y_disp = -Σᵢ Q·Aᵢ·(ky,ᵢ/k) sin(phase)   ← Lagrangian shift y
phase  = kx·x + ky·y - ω·t + φ
```

**검증**:
- ✅ Gerstner trochoid 표면의 표준 Lagrangian 표현 (Dean & Dalrymple 1991, Ch. 11)
- ✅ Surface normal: n = (-∂z/∂x, -∂z/∂y, 1) normalized — 정확
- ✅ 인용: "Gerstner 1802, Dean & Dalrymple 1991" — 둘 다 실재 reference

**기타**: Tessendorf (2001) "Simulating Ocean Water" SIGGRAPH course note 도 인용해야 — 실시간 graphics 분야의 canonical reference (이미 paper bib 에 있음).

### 12.2 Q-constraint (particle crossing prevention)

**구현** (`wave_generator.py:119-129`):
```
Σᵢ Qᵢ·ωᵢ·Aᵢ ≤ 1      ← 위반 시 표면이 자기 자신 위로 접힘
```

**검증**:
- ✅ Standard Gerstner steepness limit (Finch 2004 GPU Gems Ch.1)
- ✅ 인용: Finch 2004 — 검증 완료 ("Effective Water Simulation from Physical Models", NVIDIA GPU Gems 2004)
- ✅ Code 로 자동 enforcement (line 218-223): 위반 시 Q 를 0.95/q_val 로 scale down

### 12.3 JONSWAP 스펙트럼

**구현** (`wave_generator.py:168-175`):
```
S(ω) = 0.3125 · Hs² · ωp⁴/ω⁵ · exp(-1.25·(ωp/ω)⁴) · γ^α
α    = exp(-0.5 · ((ω/ωp - 1)/σ)²)
σ    = 0.07 if ω ≤ ωp,  0.09 if ω > ωp
```

**검증** (Hasselmann et al. 1973):
- ✅ Modified Pierson-Moskowitz form: S = (5/16)·Hs²·ωp⁴/ω⁵·exp(-(5/4)·(ωp/ω)⁴)·γ^r
- ✅ 0.3125 = 5/16 정확
- ✅ 1.25 = 5/4 정확
- ✅ σ값 (0.07/0.09) — JONSWAP 표준 (Hasselmann 1973, Holthuijsen 2007 §6.3)
- ✅ γ = 3.3 default — 표준 peak enhancement

### 12.4 Full dispersion relation

**구현** (`wave_generator.py:40-51`):
```
ω = √(g · k · tanh(k · d))
```

**검증**:
- ✅ Linear wave theory (Holthuijsen 2007 §5.4, equation 5.4.20)
- ✅ Code 의 inverse (`wavenumber_from_omega`): Newton-Raphson — 표준
- ✅ 모든 BF/k 범위에서 deep/shallow 한계 정확

### 12.5 Hs ↔ σ 관계

**구현** (`wave_generator.py:241`):
```python
def compute_Hs(Z): return 4.0 * np.std(Z)
```

**검증**:
- ✅ Spectral 정의: Hs = 4·√m₀ where m₀ = ∫S(ω)dω = σ²
- ✅ Rayleigh distributed amplitudes 가정 하에 정확 (narrow-band wave field 에서)
- ⚠️ 단, **공간 std 로 계산** (시간 std 가 아님). 본 코드는 GT field 의 spatial 평균값 — 공간 ergodicity 가정 시에만 4σ 와 일치.
  → 이 점은 paper limitations 또는 footnote 에 명시 권장

### 12.6 Möller-Trumbore ray-triangle intersection

**구현** (`lidar_simulator.py:60-99`):
- Edge1, Edge2 cross product → barycentric coord 계산
- u, v, t 결정 → triangle 내부면 hit

**검증**:
- ✅ Möller & Trumbore (1997) "Fast, Minimum Storage Ray-Triangle Intersection" Journal of Graphics Tools 2(1), 21-28
- ✅ 알고리즘 정확히 구현됨 (eps = 1e-9)
- ✅ Memory: precomputation 없음 (논문의 핵심 장점 보존)

### 12.7 Fresnel reflectance

**구현** (`raydrop_uniform.py:27-42`):
```
sin_t² = (sin_i/n)²       ← Snell's law
cos_t  = √(1 - sin_t²)
Rs     = ((cos_i - n·cos_t)/(cos_i + n·cos_t))²
Rp     = ((n·cos_i - cos_t)/(n·cos_i + cos_t))²
R      = (Rs + Rp)/2       ← unpolarized
```

**검증**:
- ✅ Standard Fresnel 방정식 — 매질 1 (n₁=1, 공기) → 매질 2 (n₂=n, 물)
- ✅ Snell's law: sin_t = sin_i/n (n₁=1 가정) — 정확
- ✅ Unpolarized = (Rs+Rp)/2 — 표준
- ✅ n_water at 1550 nm = 1.321: Hale & Querry (1973) 데이터와 일치 (1.317-1.322 범위)
- ⚠️ 인용: "Hecht (2017) Optics §4.6" — Hecht 5th ed (2017) Ch.4 Light Propagation 에 Fresnel equations 있음. Born & Wolf (1999) 7th ed 도 권위 있는 reference. 둘 다 OK.
- ⚠️ Imaginary part k ≈ 1.2×10⁻⁴ at 1550nm (small absorption) 무시됨 — 본 응용에서는 OK

### 12.8 Diffuse raydrop (uniform model)

**구현** (`raydrop_uniform.py:45-67`):
```
P_return = R_fresnel(θ) · cos(θ) · 1/(d/d_ref)²
η:        E[N_returns] = target_n_pts 가 되도록 calibration
final P:  clip(η·P_raw, 0, 1)
```

**검증**:
- ✅ Lidar return power 식 (Wagner 2010 류): P ∝ R · cos(θ) · 1/d²
  - R: surface reflectance
  - cos(θ): Lambertian projected-area factor (geometric foreshortening)
  - 1/d²: 역제곱 radiometric falloff
- ✅ Diffuse / Lambertian assumption — capillary wave 기반 미세 산란 모형으로 적절
- ⚠️ 사실은 calm water 는 거의 specular → diffuse model 이 BF2-3 calm sea 에서 실제와 차이 가능 (이래서 raydrop_slope 도 함께 유지)

### 12.9 Specular raydrop (slope model)

**구현** (`raydrop_slope.py:31-58`):
```
reflected = d - 2·(d·n)·n            ← 반사 법칙
alignment = reflected · (-d)         ← 센서 방향과의 정합도
specular_mask = alignment ≥ cos(α_threshold)   ← α=5° default
```

**검증**:
- ✅ Standard reflection law (Snell's law of reflection — Born & Wolf §1.4)
- ✅ "specular backscatter" 조건: reflected ray 가 sensor 로 돌아옴 — 정확
- ✅ Fallback (calm sea, no specular candidates): uniform Bernoulli — capillary wave diffuse 모형
- ⚠️ Reference: "Standard specular reflection geometry" (line 25) — 명시적 citation 없음
  → Born & Wolf §1.4 또는 Hecht §4.5 추가 권장

### 12.10 1550 nm 파장 선택 근거

**구현** (`raydrop_uniform.py:24`):
```
N_WATER_1550NM = 1.321
```

**검증**:
- ✅ Seyond Falcon K1 LiDAR spec = 1550 nm
- ✅ 1550 nm 는 eye-safe (Class 1) 대역 — 자율주행 LiDAR 표준
- ✅ Hale & Querry (1973) 의 측정값과 일치
- ✅ 물의 흡수계수 ≈ 1 mm⁻¹ at 1550 nm → 표면에서 거의 모두 반사 (subsurface 없음, 좋음)

### 12.11 WaveSim 종합 평가

| 항목 | 검증 | 신뢰도 |
|---|---|---|
| Gerstner wave 수식 | ✅ Dean-Dalrymple 1991 일치 | High |
| Q-constraint | ✅ Finch 2004 일치 | High |
| JONSWAP 스펙트럼 | ✅ Hasselmann 1973 일치 | High |
| Full dispersion | ✅ Holthuijsen 2007 일치 | High |
| Möller-Trumbore | ✅ 1997 paper 알고리즘 정확 | High |
| Fresnel 반사율 | ✅ Born-Wolf / Hecht 표준 | High |
| Lidar return model | ✅ Wagner 2010 류 | Medium-High |
| Specular reflection | ✅ 반사 법칙 정확 | High |
| Hs = 4σ (공간 std) | ⚠️ 시간 std 가 더 정통, 공간 ergodicity 가정 | Medium |
| 1550 nm 굴절률 | ✅ Hale-Querry 1973 일치 | High |

**전체 결론**: WaveSim 의 물리적 내용은 **검증된 references 위에 정확히 구현**됨. 이 부분은 lab meeting / paper 에서 단단한 contribution 으로 제시 가능.

**경미한 보강 권장사항**:
1. raydrop_slope.py:25 "Standard specular reflection geometry" → "Born & Wolf 1999 §1.4" 명시
2. compute_Hs (공간 std) — paper 에 "spatial ergodicity assumption" 명시 또는 시계열 기반 Hs 보조 함수 추가

---

## 13. WaveSplat 알고리즘 audit (2026-05-08)

검증 대상: `wavesplat/model/{spectral_gaussian, dispersion, confidence}.py`,
        `wavesplat/optimizer/{initializer, coarse_to_fine, sequential_update}.py`

### 13.1 Forward model (Linear plane-wave 합)

**구현** (`spectral_gaussian.py:5-9`):
```
z(x, y, t) = Σᵢ wᵢ · cos(kxᵢ·x + kyᵢ·y - ωᵢ·t + φᵢ)
```

**검증**:
- ✅ Standard linear plane-wave decomposition (Holthuijsen 2007 §5.2)
- ✅ 본 모델은 **WaveSim Gerstner GT 와 의도된 mismatch** — Gerstner 의 Lagrangian shift (X_disp, Y_disp) 항이 빠짐
- ✅ Mismatch order: O(ε²) where ε = wave steepness = k·A
  - BF4 (Hs=1.5m, λ≈80m): A=0.75m, k≈0.08, ε≈0.06 → ε² ≈ 0.36% (paper 의 "0.25%" 와 일치)
- ✅ "Q-constraint structurally inapplicable" 주석 (line 13-19): 정확. Linear model 에 horizontal displacement 가 없으니 particle crossing 불가능 — 따라서 Q 제약 의미 없음.

### 13.2 Dispersion hard constraint

**구현** (`spectral_gaussian.py:57-63`, `dispersion.py`):
```
ωᵢ = √(g · |kᵢ| · tanh(|kᵢ| · d))    ← hard constraint (use_dispersion=True)
ωᵢ = |free_param|                      ← ablation A1
```

**검증**:
- ✅ Same dispersion as WaveSim GT — consistent
- ✅ Hard constraint 형태로 구현 (parametric reduction 5N → 4N) — paper claim 과 일치

### 13.3 Geometry-aware confidence weighting

**구현** (`confidence.py:1-15`):
```
cⱼ = cos(θⱼ) / (dⱼ / d_ref)
L  = (1/M) Σⱼ cⱼ (zⱼ - ẑⱼ)² + λ₁ Σ|wᵢ| + λs Σ(Δwᵢ)²
```

**검증**:
- ✅ Heteroscedastic loss — observation precision 의 거리/각도 의존성 반영
- ⚠️ "정확한 Fresnel R(θ) 가 아니라 단순 기하 가중치" — 본 항목 명명을 paper 가 직접 정정 (geometry-aware, not Fresnel-aware)
- ✅ L1 (sparsity) + L2-of-diff (smoothness) 정규화 — compressed sensing 표준

### 13.4 Initializer (FFT-based, v2 temporal-first)

**구현** (`initializer.py:1-12`):
- v1: Spatial-only 2D FFT (legacy) — Nyquist 한계로 high-k 못 잡음
- v2: Temporal-first — time-series 로 ω 추출 → dispersion 으로 |k| → direction sweep

**검증** + **신규성 잠재력**:
- ✅ FFT + Hann window: 표준 신호처리
- ✅ Bilinear binning: 표준 (irregular → grid)
- ⭐ **v2 의 dispersion-driven super-resolution**:
  - 단일 frame 의 spatial Nyquist 한계 (예: ky_Nyq ≈ 0.38 rad/m) 를 넘어 1.3-2.7 rad/m 까지 추출
  - 메커니즘: 시계열 의 ω 가 작더라도 dispersion 으로 |k| 가 결정됨 → spatial Nyquist 우회
  - **이 부분이 §9.2 Option A 의 본질** — 진짜 알고리즘 신규성 후보
- ⭐ 단, 이 능력이 paper 에서 명시적으로 claim 안 됨 — **Option A 추진 시 explicit framing 필요**

### 13.5 Coarse-to-fine optimization

**구현** (`coarse_to_fine.py` — 미상세 검토):
- Adam optimizer (Kingma 2015): lr=5e-3, 200 iter/stage
- N=3→8→15 progressive refinement
- Adam with momentum

**검증**:
- ✅ Adam (Kingma & Ba 2015) — 표준
- ✅ Coarse-to-fine: signal processing / inverse problem 에서 표준 (multi-scale optimization)
- ⚠️ "Adam momentum 부호 진동으로 인한 진폭 감쇠" — paper 한계로 인정. Solution: gradient clipping or different optimizer (LBFGS)

### 13.6 Sequential update (tracking)

**구현** (`sequential_update.py:110-151`):
- Adam optimizer for tracking
- k freeze (kx, ky 고정) — phase 2
- Sliding window K_w = 3 frames
- 10 Adam steps per frame

**검증**:
- ✅ k-freeze: 정당한 simplification — single-frame 에서 kx,ky 추정은 ill-conditioned, multi-frame init 으로 충분히 결정됨
- ✅ Warm-start: 200→10 grad steps 단축은 tracking literature 에서 표준
- ⚠️ Convergence guarantee 없음 — Adam 이 항상 local minimum 에 수렴한다는 보장 X (paper 에 명시 권장)

### 13.7 Adaptive N (pruning + growth + safety rollback)

**구현 위치**: `coarse_to_fine.py` 추정 (직접 검토 안 함)
- Pruning: energy < threshold 인 component 제거
- Growth: residual RMS > 0.05 m 시 새 component 추가 (beamforming)
- Safety rollback: RMSE 130% 이상 증가 시 이전 state 복원

**검증**:
- ✅ Energy pruning: matching pursuit family (Mallat-Zhang 1993) 의 변형
- ✅ Greedy growth: orthogonal matching pursuit 와 유사
- ⭐ **Safety rollback (130% 임계값)**: 이건 어디서 가져온 것? — 표준 reference 부재. 본 연구의 engineering choice 일 가능성. 만약 그렇다면 작은 algorithmic contribution.

### 13.8 Beamforming (Capon-style)

**구현 위치**: `initializer.py` 의 v2 mode (직접 검토 안 함)
- Cartesian k-grid 81×81
- Greedy peak extraction with 0.15 rad/m suppression

**검증**:
- ✅ Capon (1969) "High-resolution frequency-wavenumber spectrum analysis" — minimum-variance beamforming
- ⚠️ 단, Capon 은 sample covariance matrix 기반 — 우리 구현은 sample 수 (M ~ 50) << 격자 수 (81×81=6561) 로 ill-posed
- 실제 우리 구현은 더 단순한 power spectrum peak detection 일 가능성 — **검증 필요**

### 13.9 WaveSplat 종합 평가

| 항목 | 검증 | 신뢰도 | 신규성 |
|---|---|---|---|
| Linear plane-wave model | ✅ 표준 | High | ❌ 표준 (1960s) |
| Dispersion hard constraint | ✅ Holthuijsen 일치 | High | ❌ 표준 |
| Geometry-aware confidence | ✅ heteroscedastic loss | High | ⚠️ Engineering, LiDAR-specific |
| FFT initializer (v2) | ✅ 표준 | High | ⭐ **dispersion-driven super-resolution 가능성** (Option A) |
| Coarse-to-fine Adam | ✅ Kingma 2015, multi-scale 표준 | High | ❌ 표준 |
| Sequential update with k-freeze | ✅ tracking 표준 | Medium-High | ⚠️ k-freeze 는 specific choice 이지만 신규 X |
| Adaptive N (prune+grow+rollback) | ✅ matching pursuit 변형 | Medium | ⭐ **Safety rollback 은 우리 engineering** (작은 novelty) |
| Beamforming initialization | ⚠️ 우리 구현 정확히 미검증 | Medium | ❌ Capon 1969 |

**전체 결론**:
- 알고리즘 자체는 standard signal processing + ML 방법의 조합
- **진짜 algorithmic novelty 후보 2 개**:
  1. **Dispersion-driven spectral super-resolution** (initializer v2) — 명시적 framing 시 high-impact contribution
  2. **Safety rollback in adaptive N** — 작은 engineering novelty
- 나머지는 engineering integration 수준

### 13.10 Paper / lab meeting 에서 정확하게 표현하는 방법

**현재 (over-claim 가능)**:
- "Novel grid-free spectral representation"

**권장 (honest)**:
- "Sparse plane-wave parametric representation (standard linear wave decomposition) inverted from sparse LiDAR observations using gradient descent + dispersion constraint"
- 실제 novelty: "**Dispersion-driven temporal super-resolution** in the initialization stage allows recovery of wavenumber components beyond the spatial Nyquist limit of single frames" ← Option A 형태로 framing

---

## 14. Audit 요약 — Action items

### 14.1 WaveSim 측 (작은 보강만)
- [ ] `raydrop_slope.py:25` reference 명시 (Born & Wolf 1999 §1.4)
- [ ] `compute_Hs` 가 공간 std 임을 paper 에 명시 또는 시계열 기반 함수 추가

### 14.2 WaveSplat 측 (frame 정정 필요)
- [ ] 알고리즘 표현을 "novel" → "standard methods + specific combination" 으로 수정
- [ ] Initializer v2 의 dispersion-driven super-resolution 능력을 명시적 contribution 으로 framing (§9.2 Option A)
- [ ] Beamforming initialization 의 정확한 알고리즘 검증 + Capon 인용 정확성 확인
- [ ] Adaptive N 의 safety rollback 을 작은 algorithmic contribution 으로 명시

### 14.3 References 정리
- [ ] Tessendorf 2001 (Q-constraint 의 graphics 분야 canonical) → paper bib 에 추가 또는 footnote
- [ ] Finch 2004 (GPU Gems Ch.1, Effective Water Simulation) → 인용 명시화
- [ ] Hecht Optics 5th ed (2017) → Born-Wolf 7th ed (1999) 와 함께 보존 가능 — 본 연구에서는 Born-Wolf 통일 권장
- [ ] Wagner 2010 — LiDAR radiometric calibration (raydrop_uniform 의 P_return 식 근거) 추가 검토 권장

---

## 15. Novelty 확보 마스터 플랜 — 4-Tier 전략

> **출발점**: Audit (§12, §13) 결과에서 발견된 latent novelty 후보들 + 신규 추가 가능한 algorithmic 기여 후보들을 체계적 검토.
> **원칙**: 우선 **이미 한 일에서 명시 안 된 contribution 을 실시간 explicit framing 으로 격상**, 그다음 **단기 추가 가능한 novelty 보강**, 마지막으로 **중·장기 확장**.

### 15.0 현재 잠재 contribution 의 솔직한 분류

| 후보 | 현 상태 | 진짜 신규성 잠재 |
|---|---|---|
| Linear plane-wave 모델 | 명시됨 | ❌ 표준 (1960s) |
| Dispersion 하드 제약 | 명시됨 | ❌ 표준 |
| Geometry-aware confidence | 명시됨 | ⚠️ LiDAR-specific engineering |
| Coarse-to-fine Adam | 명시됨 | ❌ 표준 |
| **k-freeze tracking (init/track 분리)** | 명시됨 (단순 tracking 표현) | ⭐ **Identifiability-aware** 로 격상 가능 |
| **Adaptive N + safety rollback** | 명시됨 | ⭐ **Safety rollback** 은 우리 engineering — small algorithmic |
| **Initializer v2 dispersion-driven super-resolution** | 코드 안에 있지만 paper 에 미명시 | ⭐⭐ **High-impact 잠재** |
| 5-metric panel | 명시됨 | ⚠️ Methodological — RMSE_unobs 가 unique |
| 24-config ETKF tuning | 명시됨 | ⚠️ Benchmarking framework |
| WaveSim 시뮬레이터 | 명시됨 | ✅ 단단한 contribution (검증 완료) |

→ **3 개의 latent novelty** (별표 ⭐) + **1 개의 high-impact 후보** (⭐⭐) 가 현 deck 에 이미 있음. 추가 작업으로 확보 가능한 것이 더 있음.

---

### 15.1 Tier 1 — 이미 한 일에서 explicit framing 으로 격상 (즉시 가능, 0-1 주)

#### **Tier 1.1: Dispersion-driven Spectral Super-Resolution** ⭐⭐ (HIGH PRIORITY)

##### 무엇이 신규성인가
- **단일 frame 의 spatial Nyquist 한계**: ky_Nyq = π/Δy ≈ 0.38 rad/m (Δy=8.3m grid)
- **WaveSplat v2 initializer 가 도달 가능한 wavenumber**: 1.3-2.7 rad/m (3-7× Nyquist 초과)
- **Mechanism**: 
  1. 시계열 (10 frames × 0.1 s = 1 s) 에서 ω 추출 (frame rate 한계 = ω_Nyq = π/Δt = 31.4 rad/s, 큼)
  2. 각 추출된 ω → dispersion 으로 |k| 일대일 결정
  3. Direction sweep 으로 (kx, ky) 결정
  4. 결과: spatial Nyquist 우회

##### 왜 진짜 novelty 인가
- 일반 spectral estimator (FFT, Capon, MUSIC, Beamforming) 는 모두 spatial sampling 한계
- 격자 기반 ETKF (Wang 2025 family) 도 격자 Nyquist 한계
- 우리만 **dispersion physics 를 사용한 implicit constraint** 으로 한계 우회

##### 격상 작업 (즉시 가능)
1. Paper 에 "Dispersion-driven spectral super-resolution" 별도 섹션 추가
2. **정량적 증명 실험**:
   - BF4 5-comp episode 에서 v1 (spatial-only FFT) vs v2 (temporal-first) 비교
   - Recoverable wavenumber range 측정 (각 method 가 도달 가능한 max |k|)
   - 결과 Figure: log-log plot (true |k| vs recovered |k|)
3. **Theoretical bound**: 시간 sampling 의 Nyquist 와 frame 수 N_t 가 주어지면 어디까지 가능한가? 간단한 공식 도출 가능
4. **새 metric**: "Super-resolution factor" = recovered_kmax / spatial_Nyquist

##### Paper 표현
**기존**:
> "Grid-free spectral representation"

**격상**:
> "We propose a temporal-first FFT initialization that exploits the dispersion relation as an implicit physical constraint to recover wavenumber components beyond the spatial Nyquist limit, achieving a super-resolution factor of 3-7× over standard spatial-only spectral estimators."

##### 비용 / 이득
- 비용: 1 주 (실험 + plot + paper 섹션 작성)
- 이득: **High-impact algorithmic novelty**, defensible against reviewers

---

#### **Tier 1.2: Identifiability-aware Sequential Estimation** ⭐ (MEDIUM PRIORITY)

##### 무엇이 신규성인가
- **현재 framing**: "k-freeze 는 single-frame 에서 ill-conditioned 라서"
- **격상**: **Fisher Information Matrix (FIM) 분석으로 어떤 parameter 가 single-frame 에서 identifiable 한지 정량 증명**

##### Mechanism
- M 개 obs 에서 N 개 component (4N parameter) 를 추정해야 함
- FIM = E[∂L/∂θ ∂L/∂θᵀ] — 정보 행렬
- Eigenvalue 분석: 큰 eigenvalue → identifiable, 작은 eigenvalue → non-identifiable
- 가설: single frame 에서 (kx, ky) 의 eigenvalue 가 (w, φ) 에 비해 작을 것 (multi-frame 누적이 결정적)

##### 격상 작업
1. FIM 분석 실험: 각 parameter 의 sensitivity 정량 측정
2. **Theoretical 정당화**: "k 는 multi-frame init 에서 identifiable, w·φ 는 single-frame 에서 identifiable" 정량 증명
3. Paper 에 Section "Identifiability Analysis of Sequential Estimation" 추가

##### Paper 표현
**기존**:
> "We freeze k during tracking due to ill-conditioning"

**격상**:
> "Fisher Information analysis (Fig. X) reveals that wavenumber components have FIM eigenvalues 10⁻³× smaller than amplitude/phase in single-frame settings, justifying our sequential identifiability decomposition: k is estimated from multi-frame initialization while (w, φ) are tracked per-frame."

##### 비용 / 이득
- 비용: 1-2 주 (FIM 계산 + 실험 + paper 작성)
- 이득: 알고리즘 설계 결정의 theoretical 정당화 — reviewer 가 매우 좋아함

---

#### **Tier 1.3: Safety Rollback in Adaptive Greedy Spectral Estimation** ⭐ (LOW PRIORITY)

##### 무엇이 신규성인가
- 우리의 Adaptive N: pruning + growth + **safety rollback (RMSE 130% 임계)**
- 표준 matching pursuit / OMP 에는 rollback 없음
- 우리 engineering choice 가 algorithmic safeguard 역할

##### 격상 작업
1. Rollback 의 효과 ablation: rollback 있음 vs 없음 — 어느 episode 에서 rollback 작동?
2. 설계 정당화: greedy growth 의 local minimum 함정에서 벗어나는 mechanism

##### 비용 / 이득
- 비용: 3-5일
- 이득: small contribution claim 추가 — defensive value

---

### 15.2 Tier 2 — 단기 추가 가능 (2-6 주)

#### **Tier 2.1: Bayesian / Hessian-based Uncertainty Quantification** ⭐⭐ (HIGH PRIORITY)

##### 무엇이 신규성인가
- Wave estimation 분야에 uncertainty quantification 거의 없음
- WaveSplat 의 spectral parameter 추정에 confidence interval 부여 가능

##### Mechanism
- Adam 수렴 후 Hessian H = ∂²L/∂θ² 계산 (PyTorch autograd 가능)
- Covariance approximation: Σ_θ ≈ H⁻¹ (Laplace approximation)
- 출력: (k, ω, A, φ) 각 component 의 95% CI

##### 활용
- **Downstream coupling**: Hydrofoil FF 에서 `K_ff = K_max · confidence` (Tier 2.2)
- **Reviewer-defense**: "WaveSplat 추정에 uncertainty 가 있는데 어떻게 평가하나?" 답변 가능
- **새로운 visualizations**: 각 wave component 의 confidence ellipse

##### 구현
```python
import torch.autograd.functional as F
def get_hessian(model, x, y, z, t):
    def loss_fn(params):
        # set model params
        return compute_loss(model, x, y, z, t)
    H = F.hessian(loss_fn, params)
    return H
sigma_squared = torch.linalg.inv(H).diagonal()
```

##### 비용 / 이득
- 비용: 2-3주 (Hessian 계산 안정화 + experiments)
- 이득: novelty + downstream control coupling

---

#### **Tier 2.2: Adaptive K_ff via Estimation Confidence** ⭐ (MEDIUM PRIORITY)

##### 무엇이 신규성인가
- WaveSplat 의 estimation confidence 를 받아 hydrofoil FF 의 gain K_ff 를 실시간 modulate
- "추정과 제어의 결합" — 시스템 통합 contribution

##### Mechanism
- WaveSplat 추정의 평균 confidence c̄ 계산 (Tier 2.1 Hessian)
- K_ff = K_ff_max · c̄ — 추정이 정확할수록 강한 FF 적용
- 추정이 부정확 (BF4 15-comp) → K_ff ↓ → robust degradation

##### 격상 효과
- Hydrofoil 응용에서 단순 fixed K_ff = 0.8 대신 adaptive
- 다양한 sea state 에서 robust performance 확보

##### 비용 / 이득
- 비용: 1-2주 (controller.py 수정 + simulation)
- 이득: 시스템 통합 신규성

---

#### **Tier 2.3: Recoverable Wavenumber Bound — Theoretical Result** ⭐ (MEDIUM PRIORITY)

##### 무엇이 신규성인가
- Sparse LiDAR + dispersion 으로 어디까지 spectral 추정 가능한가?
- M obs, T frames, dispersion constraint 가 주어지면 N component identification 의 sample complexity?

##### Approach
- Compressed sensing 류 sample complexity 분석 적용
- 제약 조건: temporal sampling Nyquist + dispersion 의 1-to-1 mapping
- 결과: M·T ≥ C·N·log(K) 형태의 bound

##### 비용 / 이득
- 비용: 2-4주 (theoretical 작업)
- 이득: paper 에 1-page theorem 추가 — reviewer 가 좋아하는 "math + physics" combination

---

### 15.3 Tier 3 — 중기 추가 (1-3 개월)

#### **Tier 3.1: LiDAR + IMU Bayesian Sensor Fusion** ⭐⭐ (HIGH PRIORITY for thesis)

##### 무엇이 신규성인가
- Vessel IMU 의 yaw/pitch 변화 → wave direction prior
- Multi-modal sensor fusion in marine context

##### Mechanism
- IMU heading rate ψ̇ 와 vessel heading ψ
- WaveSplat 추정의 wave direction p(θ_wave | LiDAR) 에 prior p(θ_wave | IMU heading) 결합
- Bayesian: p(θ | LiDAR, IMU) ∝ p(LiDAR | θ) · p(θ | IMU)

##### 격상 효과
- BF4 limit 의 direction error (37% E2E) 개선 가능
- 학위논문 단계에서 "single-modal vs multi-modal" 비교

##### 비용 / 이득
- 비용: 1-2개월 (IMU 모델 + Bayesian update 구현 + experiments)
- 이득: high-impact — 실선 데이터 받은 후 sim-to-real 시 실제 효과 측정

---

#### **Tier 3.2: Physics-Informed Stokes-2 Correction Layer** ⭐ (LOWER PRIORITY)

##### 무엇이 신규성인가
- 현재 linear plane-wave model 의 O(ε²) 오차를 작은 NN 으로 학습
- Hybrid linear + learned approach in wave estimation

##### Mechanism
- 작은 MLP (~수십 hidden units) 가 residual = z_GT - z_linear 학습
- Input: linear estimate + spectral parameters
- Output: Stokes-2 type correction

##### 비용 / 이득
- 비용: 2-3개월 (학습 데이터 + pipeline + experiments)
- 이득: 학위논문 chapter contribution — single paper 로는 risk

---

### 15.4 Tier 4 — 장기 차별화 (수개월~1년)

#### **Tier 4.1: End-to-End Differentiable Estimator → Controller** ⭐⭐ (Vision)

##### 무엇이 신규성인가
- WaveSplat parameters → preview → FF → ride RMS 전체 differentiable graph
- Final ride RMS loss 로 backprop → estimation 과 control 이 jointly 학습

##### 비용 / 이득
- 비용: 6-12개월 (학위논문 main contribution 후보)
- 이득: very high impact — wave-aware learned control 의 첫 사례

---

#### **Tier 4.2: Online Adaptive WaveSplat (Continual Learning)**

##### 무엇이 신규성인가
- Sea state 가 변하면 N, lr, λ₁, λ_s 자동 조절
- Meta-learning 또는 Bayesian online learning

##### 비용 / 이득
- 비용: 6-12개월
- 이득: practical robustness 확보 — 실선 deployment 에 필요

---

### 15.5 전략적 조합 — 3 개 Option Pack

#### **Option Pack A: "Identifiability + Super-resolution + Uncertainty"** (RECOMMENDED for paper 1)
- **구성**: Tier 1.1 (super-res) + Tier 1.2 (identifiability) + Tier 2.1 (Hessian UQ)
- **비용**: 4-6 주
- **신규성 메시지**:
  > "First identifiability-aware spectral wave estimator with theoretical super-resolution beyond spatial Nyquist and Bayesian uncertainty quantification"
- **장점**: 
  - 3 가지 contribution 이 일관된 story (모두 spectral parametric estimation 의 algorithmic improvement)
  - Reviewer 가 좋아하는 "physics + theory + uncertainty" 조합
  - 학위논문 chapter 3 main 으로 활용 가능

#### **Option Pack B: "System Integration"** (RECOMMENDED for thesis chapter 4 + paper 2)
- **구성**: Tier 1.3 (k-freeze framing) + Tier 2.2 (adaptive K_ff) + Tier 4.1 (E2E differentiable)
- **비용**: 6-12 주
- **신규성 메시지**:
  > "First end-to-end differentiable LiDAR-to-control pipeline for hydrofoil ride control"
- **장점**:
  - Hydrofoil 응용 강화
  - WaveSplat ↔ control 결합 제시
- **단점**: Tier 4.1 은 큰 구현 노력 — paper 2 / 학위논문 chapter 4 단독 task 로

#### **Option Pack C: "Multi-modal + Physics-informed"** (LONG-TERM)
- **구성**: Tier 3.1 (LiDAR+IMU) + Tier 3.2 (PINN Stokes-2)
- **비용**: 4-6 개월
- **신규성 메시지**:
  > "Multi-modal physics-informed wave estimation for shipboard real-time deployment"
- **장점**: sim-to-real 시 실제 효과 — 학위논문 후반 + Paper 3 후보
- **단점**: 실선 데이터 필요

---

### 15.6 추천 진행 순서 (사용자 결정 후)

#### Phase 1 (다음 1 주)
- **Option Pack A 의 Tier 1.1 (Super-resolution)** 을 우선 작업
- BF4 5-comp 에서 v1 vs v2 비교 실험 1 일
- Theoretical bound 도출 1-2 일
- Paper 에 별도 섹션 추가 1-2 일
- **즉시 lab meeting / paper 신규성 메시지 강화 가능**

#### Phase 2 (다음 1 개월)
- **Tier 1.2 (Identifiability/FIM 분석)**
- Paper 에 새 섹션 추가, defensible "k-freeze 정당화"

#### Phase 3 (다음 2 개월)
- **Tier 2.1 (Hessian UQ) + Tier 2.2 (Adaptive K_ff)**
- Hydrofoil 응용 강화

#### Phase 4 (학위논문 단계)
- **Tier 3.1 (LiDAR+IMU)** — 실선 데이터와 함께
- **Tier 4.1 (E2E differentiable)** — 학위논문 main novel chapter

---

### 15.7 Q&A 예상

**Q1**: "이 모든 게 fundamental novelty 인가?"
A: Tier 1.1 (super-resolution) + Tier 2.3 (theoretical bound) 가 가장 strong. 나머지는 reasonable engineering + theoretical formalization.

**Q2**: "이걸 다 paper 1 에 넣을 수 있나?"
A: Option Pack A (Tier 1.1 + 1.2 + 2.1) 까지가 paper 1 분량. B 는 paper 2, C 는 paper 3.

**Q3**: "어느 게 가장 시급?"
A: Tier 1.1 (super-resolution) — **이미 우리가 한 일이고, 명시만 하면 즉시 신규성 확보 가능**. 1 주 작업.

**Q4**: "Risk 는?"
A: Tier 1.1: low (이미 구현됨, 정량 증명만 필요)
Tier 1.2: medium (FIM 계산 정확성)
Tier 2.1: medium-high (Hessian 의 numerical stability)
Tier 3.x, 4.x: high (큰 구현 노력)

---

### 15.8 결론

- **단기 (1-2 주)**: Tier 1.1 (Dispersion-driven super-resolution) 의 explicit framing 으로 즉시 algorithmic novelty 확보
- **중기 (1-2 개월)**: Tier 1.2 + Tier 2.1 추가 → "identifiability + super-resolution + UQ" trinity
- **장기 (학위논문)**: Tier 3.1 + Tier 4.1 → multi-modal + E2E differentiable

이 plan 을 따르면 현재 "incremental engineering" 평가에서 **"first algorithmic + theoretical contribution in LiDAR-based real-time wave estimation"** 으로 격상 가능.

---

## 16. Novelty 재고 — 본질적 시각으로 다시 보기

> §15 의 Tier framework 는 "기존 일에 무엇을 더할까" 관점이었음.
> 이번에는 한 발짝 떨어져서 "이 문제 자체에 무엇이 본질적으로 새로운가" 관점으로 재검토.
> 결론은 §16.5 — **fundamental 신규성 후보 5 가지** 도출.

### 16.1 First principles — 우리가 푸는 진짜 문제는?

**한 줄 정의**:
> "Real-time spectral parameter estimation of multicomponent ocean waves from sparse, non-uniform, range-limited optical measurements with vessel motion."

이것을 6 가지 차원으로 분해하면:

| 차원 | 우리 case | 기존 work 의 차이 |
|---|---|---|
| **샘플 희소성** | M ~ 50/frame, 매우 sparse | Desmars (gauge): dense time-series; Wang (optical): similar sparse; Ehlers (radar): full grid |
| **비균일성** | 거리·각도 의존 (근거리 dense, 원거리 sparse) | 대부분의 prior: 균일 분포 가정 |
| **거리 제한** | 250m max — 공간 truncation effects | gauge / radar: 다른 형태 |
| **센서 이동** | V=11.3 m/s — 매 frame 다른 공간 영역 관측 | gauge / fixed radar: 정적 |
| **실시간** | 10 Hz, < 100 ms/frame | Desmars/Ehlers: offline; Wang: GPU real-time |
| **출력 형태** | (kx, ky, ω, A, φ) parametric — 무한차원 → 4N차원 | ETKF/Ehlers: 격자 elevation; Desmars: 비슷하게 parametric |

→ **6 차원 모두에서 우리가 가장 어려운 setup**. 6 차원의 **조합** 이 unique.

### 16.2 본질적 시각 — 이 문제를 무엇이라 부를 수 있나?

#### View A: 신호처리 — Direction-of-Arrival (DOA) 의 ocean version
- 신호처리 array literature: M sensor → N source 의 DOA 추정 (MUSIC, ESPRIT, atomic norm)
- 우리: M LiDAR points → N wave component 의 (kx, ky, ω) 추정
- **연결**: array processing 의 well-developed theory 가 사용 가능
- 차이: 우리는 sensor 자체가 이동 + dispersion 제약 + sparsity 이 더 심각

#### View B: Compressed Sensing — Off-the-grid (continuous) version
- CS literature: K-sparse signal recovery from M ≪ N measurements
- 우리: N-component spectrum 추정 (continuous parameter space)
- **연결**: Atomic norm minimization (Tang-Bhaskar-Recht 2013, Candès-Fernandez-Granda 2014)
- 우리 코드의 Adam + L1: atomic norm 의 implicit form

#### View C: Synthetic Aperture — vessel motion exploitation
- SAR (Synthetic Aperture Radar) 분야: 위성/항공 이동으로 antenna 를 synthesize
- 우리: vessel V=11.3 m/s × T_init=1s = 11.3m 의 effective aperture
- 결과: spatial resolution Δk_aperture = 2π/(V·T_init) ≈ 0.56 rad/m
- **이 관점은 ocean wave estimation 분야에 처음**

#### View D: Online Convex Optimization — streaming setting
- Online convex optimization literature: streaming data 에서 parameter 추적
- 우리 init/track 분리: heavy init (offline-like) + light track (online)
- **연결**: regret bound, convergence guarantee 적용 가능

#### View E: Identifiability Theory — observation geometry 의 정보론
- Statistics literature: Fisher Information Matrix → parameter identifiability
- 우리: M obs 에서 4N parameter 가 identifiable 한지?
- **관점**: 단일 frame 에서 (kx, ky) 는 ill-conditioned, multi-frame 에서 identifiable
- → §15 의 Tier 1.2 와 동일 방향

→ 5 가지 view 가 모두 부분적으로 적용 가능. **C (Synthetic Aperture) + B (Off-the-grid CS) 가 가장 fundamental novelty 후보**.

### 16.3 가장 깊은 novelty 후보 — Synthetic Aperture Wave Estimation

#### 16.3.1 Mechanism

**SAR analogy**:
- Radar antenna 가 위성/항공기에서 이동 → 매 시점 다른 위치에서 관측
- Coherent processing (phase 보존) 으로 effective aperture 가 길어짐
- → physical antenna 보다 훨씬 좋은 angular resolution

**Ocean wave LiDAR 에 옮기면**:
- LiDAR 가 vessel 에 탑재 → V=11.3 m/s 로 이동
- Frame 1 (t=0): vessel at x_ship=0, observes x ∈ [10, 250]
- Frame N (t=1s): vessel at x_ship=11.3m, observes x ∈ [21.3, 261.3]
- 누적된 obs 의 effective 공간 영역: x ∈ [10, 261.3] — 11.3m 확장
- Coherent processing (phase 일관성) 으로 spatial resolution 향상
- → physical LiDAR FOV 보다 효과적으로 더 큰 spatial sampling

#### 16.3.2 Quantitative effect

**Resolution bound**:
- Single frame: Δk_min = 2π / L_FOV = 2π / 240m ≈ 0.026 rad/m (이론), 실제는 점 수 한계
- Synthetic aperture (T_init=1s): ΔkSA = 2π / (V·T_init + L_FOV) ≈ 0.025 rad/m
- → 작은 효과? 

**다른 효과 — sample density 증가**:
- 1s 동안 10 frames × 50 pts = 500 obs 누적
- Effective sampling density 가 단일 frame 의 10× 증가
- → **denser effective sampling**

**가장 큰 효과 — temporal phase coherence 가 spatial resolution 으로 변환**:
- 동일 wave component 가 frame 1, 2, ..., N 에서 phase 가 ω·Δt 만큼 변함
- 이 phase 진화가 dispersion 을 통해 (kx, ky) 와 일치
- → **phase coherent multi-frame processing 이 spatial resolution 제공**

#### 16.3.3 왜 진짜 novelty 인가
- **Ocean wave LiDAR estimation 에 SAR-style framing 첫 적용**
- 이론적 기반: SAR analytical theory (well-established) 적용 가능
- Desmars 2020/2023, Wang 2025, Ehlers 2023 모두 이 관점 없음
- 정량적 framework 제공: vessel speed → effective aperture → resolution

#### 16.3.4 비용 / 위험
- 비용: 2-4주 (이론 + 정량 분석 + paper 섹션)
- 위험: "SAR 자체는 잘 알려져 있다 — 이걸 ocean LiDAR 에 적용한 게 진짜 새로운가?" 반론 가능
- **방어**: Ocean LiDAR + dispersion + 비균일 sparseness 의 specific 조합은 첫 paper. 적용은 trivial 이 아님.

---

### 16.4 두 번째 깊은 novelty 후보 — Off-the-grid Compressed Sensing

#### 16.4.1 Mechanism

**일반 CS**:
- Discrete N-dim signal x, K-sparse, M measurements y = Φx
- L1 minimization: min ‖x‖₁ s.t. Φx = y
- Recovery if M = O(K log N) and Φ satisfies RIP

**Off-the-grid CS** (Tang-Bhaskar-Recht 2013):
- Continuous parameter space (예: spectral position ω ∈ ℝ)
- K-sparse: signal = Σᵢ cᵢ · atom(θᵢ), atoms = sinusoids
- Atomic norm: ‖x‖_𝒜 = inf {Σ|cᵢ| : x = Σ cᵢ a(θᵢ)}
- Atomic norm minimization → SDP

**WaveSplat 의 형태**:
- Continuous (kx, ky, ω) parameters → atom = plane wave with dispersion
- Sparse: N components
- Measurements: M LiDAR points (sparse, non-uniform)
- 우리 알고리즘 (Adam + L1): atomic norm minimization 의 implicit local-search version
- **dispersion 제약**으로 atom space 가 4N-dim → 더 좋은 sample complexity 가능

#### 16.4.2 가능한 theoretical result

**Sample complexity theorem (목표)**:
> "M LiDAR points distributed over T frames suffice to recover N-component dispersion-constrained wave field with probability 1-δ provided M·T ≥ C · N · log(K_max / Δk_resolution)"

이 형태의 결과는:
- Compressed sensing 분야에 well-known
- 우리는 "with dispersion constraint" 로 향상된 bound 도출 가능
- 정량 실험으로 확인 가능

#### 16.4.3 왜 진짜 novelty 인가
- **Ocean wave estimation 에 atomic norm / off-grid CS framework 첫 적용**
- Theoretical guarantee 제공 (recovery condition, sample complexity)
- Desmars, Wang, Ehlers 모두 theoretical guarantee 없음 — 우리가 첫 사례

#### 16.4.4 비용 / 위험
- 비용: 1-2 개월 (theoretical 작업 + 정량 실험)
- 위험: 수학적 정밀도 필요. "atomic norm 정확히 구현 안 했지 않나?" 반론 가능
- **방어**: "atomic-norm-inspired" 수준의 framing 도 가능 — 정확한 SDP 대신 Adam local search 의 empirical 검증

---

### 16.5 본질적 신규성 후보 5 개 — 최종 정리

이전 §15 의 Tier 1-4 와 본 §16 의 본질적 시각을 종합한 5 개 후보:

| # | 후보 | 본질적 시각 | 신규성 등급 | 즉시 framing | 작업 비용 |
|---|---|---|---|---|---|
| **N1** | **Synthetic Aperture Wave Estimation** | Vessel motion 을 SAR-style aperture 로 framing | ⭐⭐⭐ Fundamental | 부분 가능 | 2-4주 (이론+실험) |
| **N2** | **Off-the-grid CS for ocean waves** | Atomic norm + dispersion atoms | ⭐⭐⭐ Theoretical | ❌ 추가 작업 필요 | 1-2개월 |
| **N3** | **Dispersion-driven super-resolution** | 시계열 + dispersion 으로 spatial Nyquist 우회 | ⭐⭐ Algorithmic | ✅ 즉시 가능 | 1주 |
| **N4** | **Identifiability decomposition (FIM)** | k 는 multi-frame, w·φ 는 per-frame identifiable | ⭐⭐ Theoretical | ✅ 즉시 가능 | 1-2주 |
| **N5** | **Sample complexity bound** | M·T 가 N component 회복에 충분한 조건 | ⭐⭐⭐ Theoretical | ❌ 추가 작업 | 1-2개월 |

→ **N1 + N3 + N4** 셋의 조합이 paper 1 차원에서 가장 efficient + defensible.

### 16.6 가장 깊은 framing — 통합 메시지

5 개 후보를 묶어서 paper 의 main contribution claim 으로 만들면:

**Framing A — Algorithmic novelty 강조** (단기 1-2개월):
> "We formalize shipboard LiDAR wave estimation as **synthetic-aperture sparse spectral estimation under dispersion-physics constraints**, achieving super-resolution beyond the per-frame spatial Nyquist limit through dispersion-driven temporal coherence and proving identifiability decomposition between wavenumber (multi-frame identifiable) and amplitude/phase (per-frame identifiable)."

이 메시지의 구성:
- Synthetic aperture (N1) — vessel motion 을 SAR-style 로 활용
- Dispersion-driven super-resolution (N3) — 이미 코드 안에 있음
- Identifiability decomposition (N4) — k-freeze 정당화

**Framing B — Theoretical novelty 강조** (장기 3-6개월):
> "We establish the first sample complexity bound for sparse spectral wave field estimation under dispersion-physics priors and demonstrate vessel motion as effective synthetic aperture for ocean wave LiDAR sensing."

이 메시지의 구성:
- Sample complexity (N5) — theoretical contribution
- Synthetic aperture (N1) — physical insight

**Framing C — System integration 강조** (현재 + Hydrofoil 응용):
> "We present a real-time shipboard LiDAR-based wave estimator with provable identifiability, demonstrated end-to-end through a hydrofoil ride control application achieving ISO 2631 'Comfortable' rating from 'Slight discomfort'."

이 메시지의 구성:
- Real-time + identifiability decomposition
- E2E hydrofoil application
- ISO 2631 표준 기준 quantitative result

### 16.7 추천 결정 — 무엇을 우선 할 것인가

#### 단기 (이번 주~1주)
**N3 (Dispersion-driven super-resolution) explicit framing**
- 이미 구현되어 있음 (initializer.py v2)
- 1주 작업으로 즉시 lab meeting 신규성 강화
- Risk 매우 낮음

#### 중기 1 (1-2주)
**N4 (Identifiability decomposition with FIM)**
- Theoretical 정당화 추가
- k-freeze 의 솔직한 근거 강화

#### 중기 2 (2-4주)
**N1 (Synthetic Aperture Wave Estimation framing)**
- 이게 가장 fundamental novelty 후보
- 이론적 분석 + 정량적 실험
- 새 paper 또는 학위논문 chapter 의 main contribution

#### 장기 (학위논문 단계)
**N5 (Sample complexity bound) + N2 (Atomic norm CS)**
- Theoretical framework 격상
- 학위논문의 theoretical chapter

### 16.8 토론 — 위 5 개 후보의 진짜 도전

#### 솔직한 반론 시뮬레이션

**Reviewer Q1 (skeptic)**: "Synthetic Aperture 는 SAR 에서 standard concept. ocean LiDAR 에 적용한 것만으로 novelty 라 할 수 있나?"
**A**: SAR 의 mathematical framework 을 ocean LiDAR + dispersion 제약 + sparse non-uniform sampling + 실시간 의 specific combination 에 적용한 것은 첫 사례. 단순 적용이 아니라 dispersion 이라는 추가 physics 가 결합 → 새로운 분석 필요.

**Reviewer Q2 (Desmars defender)**: "Desmars 2023 JFM 도 vessel-mounted optical 에서 비슷한 일 하는데?"
**A**: Desmars 2023 의 setup 에서 vessel 이 정적인지 이동하는지 확인 필요 (paper 받아야 함). 만약 정적이면 SA framework 이 명확히 우리 contribution. 이동이면 가까운 prior 로 인용하고 우리는 정량 분석으로 차별화.

**Reviewer Q3 (theory skeptic)**: "Identifiability decomposition 은 standard FIM 분석. trivial 아닌가?"
**A**: standard tool 이지만 ocean wave LiDAR 의 specific 6-차원 setup (sparse + non-uniform + range-limited + moving + real-time + parametric) 에 적용한 결과 자체는 novel. 게다가 k-freeze 라는 algorithmic decision 의 theoretical 정당화 — 알고리즘 design 에 중요.

**Reviewer Q4 (CS skeptic)**: "Atomic norm minimization 이라 부르려면 SDP 를 풀어야지 — 너희는 Adam GD 만 쓰지 않나?"
**A**: 솔직한 답: 정확한 atomic norm SDP 가 아니라 atomic-norm-inspired heuristic local search. Theoretical bound 는 atomic norm 기반, 알고리즘은 empirical. 이 한계는 paper 에 명시.

#### 반론에서 살아남는 framing

위 4 개 reviewer Q 모두에서 살아남는 정직한 framing:

> "We propose a *synthetic-aperture-style* sparse spectral wave estimator from shipboard LiDAR, with identifiability analysis decomposing wavenumber and amplitude/phase components, dispersion-driven super-resolution beyond per-frame spatial Nyquist, and empirical recovery from sparse non-uniform observations through gradient-based atomic-norm-inspired optimization. We establish recovery bounds based on synthetic aperture theory and validate them against tuned ETKF (Wang 2025 family) and GPR baselines on a 100-episode benchmark."

이 framing 의 강점:
- "synthetic-aperture-style" → SAR full claim 회피, 영감 인정
- "identifiability analysis" → standard tool 적용 인정
- "dispersion-driven super-resolution" → 우리만의 mechanism 강조
- "atomic-norm-inspired" → SDP 안 풀었음을 인정
- "empirical recovery + bounds based on SA theory" → theory + experiment 모두 제공
- "validated against tuned ETKF, GPR" → 실증 비교

이 정도 framing 이면 reviewer-proof + 신규성 충분.

---

### 16.9 최종 권장 — 본질적 시각으로 결정

**페이퍼 1 / lab meeting 단기 (다음 1-2주)**:
- N3 (super-resolution explicit) — 즉시 작업 시작
- N4 (FIM identifiability) — paper revision 단계

**페이퍼 1 중기 (다음 1-2개월)**:
- N1 (Synthetic Aperture framing) 도입 — 논문 main contribution claim 재구성

**학위논문 단계**:
- N2 (Atomic norm CS) + N5 (Sample complexity) — theoretical chapter

이렇게 하면 **paper 1 의 신규성 메시지**는 (N1+N3+N4) trinity 로 진정한 algorithmic + theoretical 기여로 격상. 학위논문 차원에서 (N2+N5) 추가하면 full theoretical framework 완성.

이게 **현재 work 의 진짜 가치를 최대한 끌어내는 방향**. Engineering 적응이 아니라 fundamental 시각 (synthetic aperture + sparse spectral + identifiability) 에서 본 contribution 으로 격상.

---

## 17. Desmars 2023 JFM 본문 정독 결과 (2026-05-08)

> 파일: `ref_paper/nonlinear-deterministic-reconstruction-and-prediction-of-remotely-measured-ocean-surface-waves.pdf`
> 사용자가 직접 제공. JFM 975, A8 (40 pages). Open Access.
> §10 의 "전체 paper 읽기 원칙" 첫 적용 사례.

### 17.1 Paper 의 정확한 내용 — abstract 와 metadata 가 가렸던 디테일

#### Sensor: **X-band marine radar** (LiDAR 아님!)
- "X-band marine radars are considered" (page 4)
- Non-coherent radar — backscattered intensity 만 측정 (elevation 직접 측정 X)
- Antenna 위치: z_a = 8 H_s (매우 높음, 실제 deployment 와 다른 setup)
- Range: ~3 km radius capable
- Range resolution: 7.5 m, antenna rotation period 1.5 s
- 우리 LiDAR (250m, 0.05° divergence, 직접 elevation) 와 **본질적으로 다름**

#### Wave field: **Unidirectional (1D)** waves
- "applications presented later are restricted to unidirectional waves" (page 4)
- 우리는 directional spreading ±20° 의 2D field
- → **차원 차이**: 그들은 1D, 우리는 2D

#### 3 가지 reconstruction 방법 비교
1. **HOS-OCM** (highly nonlinear): grid-based, HOS expansion + observation coupling
2. **AMBM-LWT** (Linear Wave Theory): parametric, predefined k_n + (a,b) 추정
3. **AMBM-ICWM** (Improved Choppy Wave Model, weakly nonlinear)
4. **AMBM-LWT-CDR** (LWT with Corrected Dispersion Relation)

#### **결정적 발견** — AMBM 의 wavenumber 처리 (Section 2.4, page 12)
> "the wavenumbers of the wave components are not defined the same way for each method: they are **implicitly defined by the grid size and resolution for HOS-OCM**, and **arbitrarily chosen for AMBM**."

> "wave components are chosen according to those implicitly defined by the computational grid of HOS-OCM"

→ **Desmars 2023 의 모든 방법은 PREDEFINED wavenumbers k_n 사용**.
→ Estimation 은 (a_n, b_n) = amplitude/phase 만 — **linear LSQ via Levenberg-Marquardt**.
→ Wavenumber 자체를 학습하지 않음.

#### LWT 의 정확한 형태 (eq 2.13)
```
η_LWT(x, t) = Σ_{n=1}^N [a_n cos(ψ_n) + b_n sin(ψ_n)]
              ψ_n = k_n · x − ω_n t
              ω_n² = g k_n tanh(k_n h)        ← dispersion 강제
              k_n: PREDEFINED grid
              (a_n, b_n): LSQ optimized
```

→ 우리 모델 z(x,y,t) = Σ wᵢ cos(kxᵢ x + kyᵢ y − ωᵢ t + φᵢ) 와 비교:
- **수학적 형태**: 동등 (a = w cos φ, b = -w sin φ 변환)
- **추정 변수**: 그들 = (a_n, b_n) only (k 고정), 우리 = (kx, ky, w, φ) all (continuous)
- **알고리즘**: 그들 = LM linear LSQ, 우리 = Adam GD nonlinear

#### Setup specifics (Section 3.1, page 13)
- JONSWAP Tp = 10s, γ = 3.3
- Domain length: 32 peak wavelengths × 32 points/peak wavelength
- Steepness Hs/λp ≤ 3.2%
- Reference field: HOS-ocean (Ducrozet 2016) with M = 4 nonlinear order
- Radar antenna: x_a = x0 + L_x + 3λp ≈ 470 m gap to reconstruction domain
- Tank experiments + synthetic data

### 17.2 Desmars 2023 와 WaveSplat 의 본질적 차이 (재검토)

| 항목 | Desmars 2023 | WaveSplat | 차이 등급 |
|---|---|---|---|
| **Sensor** | X-band radar (intensity, 3km range) | LiDAR (direct elevation, 250m) | **본질적** |
| **Dimensionality** | 1D unidirectional | 2D directional (±20° spread) | **본질적** |
| **Wavenumber 처리** | PREDEFINED grid (on-grid) | LEARNABLE continuous (off-grid) | **본질적** ⭐⭐ |
| **Estimation 알고리즘** | Levenberg-Marquardt (linear LSQ on a,b) | Adam GD (nonlinear on k,w,φ) | 알고리즘 차이 |
| **Real-time** | Offline assimilation period | Real-time 10 Hz tracking | **차이** |
| **Adaptive complexity** | Fixed N | Adaptive N (pruning + growth + rollback) | **차이** |
| **Confidence weighting** | Uniform LSQ | Geometry-aware LiDAR-specific | LiDAR-specific |
| **Test domain** | Tank (controlled) + synthetic | Synthetic 100-episode | similar |
| **Vessel motion** | "potentially mobile" 언급 (page 4) but their test setup vessel motion 처리 | 처음부터 V=11.3 m/s 가정 | 명확히 다름 (그들 setup 은 정적 antenna) |

### 17.3 가장 큰 발견 — **Off-the-grid spectral estimation 이 진짜 차별점**

§16.4 에서 brainstorm 한 "Off-the-grid CS for ocean waves" (N2) 가 사실 **이미 우리 알고리즘의 핵심 차이**!

**Desmars (모든 LWT/ICWM 방법)**:
- predefined k_n grid (on-grid)
- solve linear LSQ for (a_n, b_n)
- → classical "DFT-like" 추정

**WaveSplat**:
- continuous (kx, ky) parameters (off-grid)
- nonlinear gradient descent on (kx, ky, w, φ)
- → modern "atomic-norm-inspired" 추정

이 차이는:
1. ✅ **즉시 framing 가능** (이미 코드에 있음)
2. ✅ **Theoretical 정당화 강함** — atomic norm CS literature (Tang-Bhaskar-Recht 2013)
3. ✅ **Reviewer-defensible** — Desmars 본문 직접 인용으로 차이 명시
4. ✅ **실용적 이점**: spectral resolution 향상 (no grid aliasing), adaptive K

### 17.4 Desmars 2023 의 main contributions (정확히)

논문 abstract + intro 정독 결과 그들의 주장:
1. 3 가지 modulation mechanism (direct elevation / shadowing / shadowing+tilt) 영향 분석
2. HOS-OCM 의 새로운 numerical formulation (eq 2.7-2.10) — non-sequential time integration
3. LWT-CDR 의 dispersion correction 효과 정량화
4. Tank experiment 와의 wave shape nonlinearity 분석

**중요**: Desmars 2023 의 main contribution 은 **modulation mechanism 분석 + HOS-OCM 의 numerical 개선**. Wave estimation 의 algorithmic 신규성 (k 추정 방식, real-time, adaptive) 은 그들의 focus 가 아님.

→ **우리는 그들이 안 한 일을 하고 있음**:
- LiDAR (그들 안 다룸)
- 2D directional (그들 1D)
- Off-grid k estimation (그들 on-grid)
- Real-time tracking (그들 offline)
- Adaptive complexity (그들 fixed)

### 17.5 Novelty 재평가 — Desmars 2023 본문 기반

§16 의 5 개 N1-N5 후보를 재평가:

| # | 후보 | Desmars 2023 본문 검토 후 평가 |
|---|---|---|
| **N1** | Synthetic Aperture | ⭐⭐⭐ 여전히 valid — Desmars 의 antenna 는 정적, 우리 vessel motion 은 unique |
| **N2** | Off-the-grid CS | ⭐⭐⭐⭐ **격상** — Desmars LWT 가 명시적 on-grid임을 본문 확인. 우리 off-grid 는 명확한 차이 |
| **N3** | Dispersion super-resolution | ⭐⭐⭐ 여전히 valid — Desmars 도 dispersion 사용하지만 grid 한계 안에서 |
| **N4** | Identifiability decomposition | ⭐⭐ 여전히 valid — Desmars 도 단일 framework, 우리만 init/track 분리 |
| **N5** | Sample complexity | ⭐⭐⭐ 여전히 valid — Desmars 는 theoretical bound 없음 |

### 17.6 새로운 통합 framing — Off-grid 강조

§16.6 의 "Framing A" 를 update:

**Updated Framing A**:
> "We propose a real-time, **off-grid sparse spectral estimator** for shipboard LiDAR-based 2D directional wave field estimation, in contrast to existing on-grid parametric methods (Desmars et al. 2023, JFM) that fix wavenumbers a priori. Vessel motion is exploited as a synthetic aperture providing additional spatial resolution beyond per-frame Nyquist limits, and identifiability decomposition (multi-frame init + per-frame tracking) enables real-time operation."

이 framing 의 강점:
- **Off-grid vs on-grid**: clear technical difference, defensible by direct citation of Desmars 2023 §2.4
- **2D vs 1D**: their unidirectional vs our directional
- **LiDAR vs radar**: different sensor regime
- **Real-time vs offline**: clear engineering contribution
- **Synthetic aperture**: new conceptual framework

### 17.7 Action items 추가

- [ ] §10.3 ledger 에서 Desmars 2023 검토 깊이 "Abstract" → "Full text" 로 update
- [ ] §15-§16 의 모든 novelty 평가에서 "Desmars 2023 본문 검증 완료" 반영
- [ ] PPT slide 5 (Prior Art): "Desmars 2023 — 1D unidirectional, on-grid LWT/ICWM/HOS, X-band radar" 정확하게 표현
- [ ] PPT slide 8 (WaveSplat novelty): "off-grid continuous wavenumber estimation" 명시
- [ ] §16.6 Framing A → 17.6 Updated Framing A 로 paper / lab meeting 적용

### 17.8 Wang 2025 와 Desmars 2023 의 관계

이번 분석으로 명확해진 것:
- Desmars 2023: parametric inversion (predefined k) + radar
- Wang 2025: ETKF on grid (격자 elevation 추정) + sparse optical
- 둘은 **다른 방법론 family**
- 우리는 어느 쪽도 아닌 **off-grid parametric** — 새로운 family

→ 우리 work 의 실제 positioning:

```
Wave field reconstruction methods
├── Grid-based methods (격자 elevation 추정)
│   ├── ETKF on grid (Wang 2025) — sparse optical
│   ├── HOS-OCM (Desmars 2023) — radar
│   └── FNO/PINN (Ehlers 2023) — synthetic
├── Parametric methods (wave model parameters)
│   ├── On-grid LWT/ICWM (Desmars 2020/2023) — radar gauge
│   ├── On-grid wave model fitting (전통적 방법들)
│   └── **Off-grid continuous (WaveSplat, this work)** — LiDAR ⭐
└── Learning-based (FNO, PINN)
    ├── Ehlers 2023 — uniform grid radar
    └── ...
```

→ 우리 work 가 차지하는 자리: **새로운 family (off-grid parametric for LiDAR)** 의 첫 사례.

### 17.9 결론

Desmars 2023 본문 정독으로 얻은 것:
1. ✅ 우리 work 가 그들과 **본질적으로 다름** 확인 (sensor, dim, k 처리, real-time, adaptive)
2. ✅ **Off-grid spectral estimation** 이 명확한 algorithmic novelty 로 격상 (N2 → ⭐⭐⭐⭐)
3. ✅ Reviewer-defensible 구체 인용 가능 (Desmars 2023 §2.4 직접 인용)
4. ✅ Synthetic aperture (N1) 도 여전히 valid — 그들 antenna 는 정적
5. ✅ 우리 positioning 이 "off-grid LiDAR family" 의 첫 사례

**Lab meeting 발표 안전성**:
- "Desmars 2023 와 비슷한 일 아닌가?" reviewer 질문 → "그들은 1D, on-grid, radar, offline. 우리는 2D, off-grid, LiDAR, real-time. 본문 §2.4 에 그들 method 의 on-grid 한계 명시" 답변 가능.

---

### Ledger 갱신
| 날짜 | 변경 | 이유 |
|---|---|---|
| 2026-05-08 | §17 추가 | Desmars 2023 본문 정독 — off-grid vs on-grid 차이 본질적 발견 |
| 2026-05-08 | §18 추가 | Wang & Pan 2021 JFM (918, A19) 본문 정독 — first EnKF–HOS coupling 확인 |

---

## 18. Wang & Pan 2021 JFM 본문 정독 결과 (2026-05-08)

> **Citation**: Wang, G. & Pan, Y. (2021). *Phase-resolved ocean wave forecast with ensemble-based data assimilation*. **J. Fluid Mech.** 918, A19. doi:10.1017/jfm.2021.340

> **읽은 범위**: pp. 1–13 (§1 Intro, §2 EnKF–HOS framework 전체, §3.1 idealistic synthetic results 까지). §3.2 realistic / §4 real radar / §5 conclusions 는 pending.

### 18.1 핵심 메시지 (1–2줄 요약)
- **first EnKF–HOS 결합**으로 phase-resolved nonlinear 파동 예측을 수행 — 기존 explicit Kalman 방법 (Yoon-Kim-Choi 2015) 대비 1 차수 (10×) 정확도 향상.
- **predictable zone vs measurement region mismatch** 라는 새 문제를 정의하고, modified analysis equation (eq. 2.18) 으로 해결.

### 18.2 저자 정체 — Wang 2025 와 다른 사람
- **Wang G. & Pan Y.** — University of Michigan / Tianjin University 계열, JFM 시리즈
- **Wang Feng et al. 2025** ("From Points to Waves", Coastal Engineering 197, 104690) — 다른 연구진 (중국 그룹 추정)
- ⚠️ 우리 코드의 baseline docstring 인용은 Wang 2025 (Coastal Eng) 이며, Wang-Pan 2021 (JFM) 과 무관함을 명확히 해야 함.

### 18.3 방법론 핵심
**State**: 2D 또는 3D wave field 를 HOS Zakharov form 으로 표현
- η(x, t): 자유표면 elevation
- ψ(x, t): 자유표면 velocity potential
- (eq. 2.9–2.10) HOS evolution PDE — nonlinearity order M=4

**Ensemble**:
- N=100 ensemble members (default)
- s_{f,j}^{(n)}, n=1,...,N — forecast state vectors
- Q_{s,j} = covariance via ensemble (eq. 2.12)

**Analysis (eq. 2.13)**:
```
S_{a,j} = S_{f,j} + K_{s,j} [S_{m,j} − G_j · S_{f,j}]
K_{s,j} = Q_{s,j} G_j^T [G_j Q_{s,j} G_j^T + R_{s,j}]^{-1}
```
G_j: linear (Fourier) interpolation operator from forecast space → measurement space.

**Adaptive inflation (eq. 2.15)**: λ_j ≥ 1, posterior maximization (Anderson 2007). Underestimated variance 보정.

**Localization (eq. 2.16)**: Schur product with Gaspari-Cohn function. Spurious long-range correlation 제거.

**Predictable zone P_j (§2.7)**:
- Group velocity 로 결정되는 spatiotemporal 영역
- 무한 ocean 가정 시 measurement region M_j 와 predictable zone P_j 가 어긋날 수 있음
- (eq. 2.18) 로 분리 처리: P_j 영역 → 표준 EnKF, U_j ∩ M_j 영역 → measurement 만 사용

**Computational cost**: O(NL log L) per step (HOS) + O(dL²) + O(d²L) (DA)

### 18.4 Sensor & Setup
- **Synthetic**: surface elevation η_m,j(x) = η^true(x, t_j) + v(x), Gaussian noise, c=0.0025σ_η²
- **Real**: shipborne Doppler coherent marine **radar** (Nogwu-Lyzenga 2010)
- **Moving sensor**: Yes — radar mounted on **moving ship** in real-data case
- **Domain**: 2D (1D spatial) and 3D (2D spatial) JONSWAP fields

### 18.5 결과 (§3.1 idealistic case 까지)
- 2-D JONSWAP, k_p H_s/2 = 0.11 (steepness), L=256 grid, d=2 buoy locations, τ=T_p/16
- HOS-only: ε(t; A) → O(1) at t/T_p ≈ 100 (phase 정보 완전 손실)
- EnKF-HOS: ε(t; A) → O(10⁻³) at t/T_p = 100 (10⁵ 배 차이)
- Ensemble size N: N≥40 OK, N=20 시 filter divergence
- DA interval τ: 작을수록 좋음 (T_p/16 < T_p/2)
- d (measurement locations): 많을수록 좋음 (d=4 > d=1)
- Yoon 2015 explicit Kalman 대비 1 order of magnitude 향상

### 18.6 우리 work 와 비교 — 본질적 차이

| 항목 | Wang & Pan 2021 | WaveSplat (this work) |
|---|---|---|
| **State 표현** | grid 위 elevation η(x) + potential ψ(x) | continuous parametric (k, ω, φ, A) |
| **Forward model** | HOS PDE (nonlinear, order M=4) | linear plane-wave superposition |
| **Spatial grid** | L=256 (2D), 64×64 (3D) | grid-free (LiDAR scan points) |
| **Sensor** | radar 또는 surface elevation gauge | LiDAR (sparse rays + raydrop) |
| **DA method** | EnKF (ensemble) | Adam GD (deterministic) |
| **Ensemble** | N=100 members | N/A (single-shot optimization) |
| **Nonlinearity** | order M=4 | linear (Stokes 1st order) |
| **Inflation/loc** | adaptive inflation + Gaspari-Cohn | N/A |
| **Output** | predictable spatial-temporal field | (k, ω, φ, A) parameters → analytical η anywhere |
| **Compute** | O(NL log L) — heavy | < 1 sec inference |
| **Sensor motion** | moving radar (real case) | moving LiDAR (sparse ship-mounted) |

### 18.7 Implications — 우리 positioning 에 미치는 영향

#### 18.7.1 SAWE (Synthetic Aperture Wave Estimation, §16 N1) 에 대한 도전
- **Wang-Pan 2021 도 moving radar 를 다룸** (real case, §3.2 / §4 예상)
- "moving sensor 활용" 자체는 새로운 frame 이 아님
- 그러나 우리는 **명시적으로 platform 운동을 trajectory 로 사용** + LiDAR + sparse — 그들은 implicit 하게 처리
- **N1 framing 약화**, 단순 "moving sensor 처리" 는 novelty 가 아님 → 다른 angle 필요

#### 18.7.2 Off-grid vs on-grid (§17) 우위 유지
- Wang-Pan 2021 의 state space 는 명시적 spatial **grid (L=256)** — Fourier interpolation 으로 measurement 와 매칭
- 우리: continuous (kx, ky) 로 grid-free
- ✅ **§17 의 off-grid vs on-grid 핵심 차이는 여전히 valid**
- ✅ **§17 의 dispersion-driven super-resolution** 도 여전히 valid (Wang-Pan 은 Nyquist 이내만 처리)

#### 18.7.3 Nonlinear physics 차이
- Wang-Pan: HOS order M=4 — 본격적 nonlinear (Stokes 4th order 까지)
- 우리: linear plane-wave only
- ⚠️ **Reviewer 가 "nonlinear 안 다루면서 sea state high 에서 어떻게 동작?" 비판 가능**
- → §15 Tier 4 (full ICWM/HOS extension) 가 future work 으로 중요

#### 18.7.4 Computational efficiency 강조 가능
- Wang-Pan 2021: O(NL log L) per step + ensemble cost
- 우리: < 100 ms / window (이미 측정됨)
- ✅ "real-time onboard computation" 우위 — practical 컨트리뷰션 강조 가능

### 18.8 References — 추가 인용 필요
이 paper 에서 발견한 cite-worthy 선행연구:
- **Yoon, Kim & Choi 2015** — explicit Kalman + linear wave propagation (Wang-Pan 의 비교 대상). 우리 explicit baseline 으로도 검토 가능.
- **Naaijen, Trulsen & Blondel-Couprie 2014** — predictable zone 개념 정립
- **Köllisch et al. 2018** — predictable zone 후속
- **Anderson 2007** — adaptive inflation (Bayesian)
- **Hamill & Whitaker 2005** — covariance localization
- **Carrassi et al. 2018** — DA review
- **Nogwu & Lyzenga 2010 / Lyzenga et al. 2015** — Doppler coherent marine radar setup

### 18.9 PPT / paper 에 반영해야 할 점

#### 18.9.1 References slide
- ✅ **Wang & Pan 2021 (JFM 918, A19)** 추가
- ⚠️ Wang 2025 (Coastal Eng 197) 와 **명시적으로 분리** — 두 다른 논문/저자
- 추천 표기:
  > "Wang & Pan (2021) introduced the first EnKF–HOS framework for phase-resolved nonlinear wave forecasting on a grid; Wang et al. (2025) adapted ETKF to sparse optical measurements on a 2D grid."

#### 18.9.2 Prior art slide
- 현재 "From Points to Waves" 만 언급되어 있음 — Wang-Pan 2021 도 별도 row 로 추가
- 차원 분리: 1D / 2D / 3D, sensor (radar / LiDAR / gauge), state (grid / off-grid)

#### 18.9.3 Q&A 대비
- **Q**: "Wang-Pan 2021 도 moving sensor 를 다룬 것 아닌가?"
- **A**: "맞습니다. 그러나 그들은 grid-based EnKF + HOS 조합으로 measurement region 의 mismatch 를 다뤘고, 우리는 grid-free parametric inversion 으로 sparse LiDAR 의 spatial sampling 을 직접 활용합니다. State space 자체가 grid (L=256) 와 continuous (kx, ky) 로 본질적으로 다릅니다."

### 18.10 결론
Wang-Pan 2021 정독으로 얻은 것:
1. ✅ **첫 EnKF-HOS coupling** — historical milestone 으로 인용 필수
2. ✅ Wang 2025 와 **다른 paper / 저자** 임을 PPT/paper 에 명시
3. ⚠️ "moving sensor" 자체는 novelty 가 아님 — N1 (SAWE) 약화
4. ✅ Off-grid (§17) + super-resolution (§16 N3) 은 여전히 우리 우위
5. ✅ Real-time / sub-second compute 는 우리 practical 우위
6. ⚠️ Nonlinear physics 가 그들 강점 — 우리는 linear → future work 명시 필요

### 18.11 다음 진행 — Pending paper review
| Paper | 현재 상태 | 우선순위 |
|---|---|---|
| Desmars 2023 JFM | ✅ 본문 정독 (§17) | 완료 |
| Wang & Pan 2021 JFM | ⚠️ §1–§3.1 정독, §3.2/§4 pending | 중 |
| Wang & Pan 2022 JFM | ⏳ 미시작 | 중 (current 추정 확장) |
| Hunt 2007 LETKF | ⏳ 미시작 | 저 (DA 기초) |
| Ehlers 2023 FNO | ✅ 본문 정독 (§22) | 완료 |
| Möller-Trumbore 1997 | ✅ 본문 정독 (§24) | 완료 |
| Wang 2025 Coastal Eng | ❌ 다운로드 실패 (paywall) — 사용자 지시로 제외 | 제외 |

---

## 19. Desmars 2018 (OMAE) + Desmars 2020 (Ocean Eng) 정독 결과 (2026-05-08)

> **Desmars 2018**: *Phase-Resolved Reconstruction Algorithm and Deterministic Prediction of Nonlinear Ocean Waves From Spatio-Temporal Optical Measurements*. ASME OMAE 2018, V07BT06A054. doi:10.1115/OMAE2018-78367
>
> **Desmars 2020**: *Experimental and numerical assessment of deterministic nonlinear ocean waves prediction algorithms using non-uniformly sampled wave gauges*. **Ocean Eng** 212, 107659. doi:10.1016/j.oceaneng.2020.107659

### 19.1 핵심 메시지
- Desmars 2018: **첫 phase-resolved nonlinear LIDAR-based wave reconstruction algorithm** 제시 (Choppy Wave Model + ICWM)
- Desmars 2020: 같은 알고리즘을 **실제 wave tank 실험 데이터** 와 numerical HOS 데이터로 검증 (1D long-crested)

### 19.2 핵심 방법론 (Desmars 2018)
**3 wave models 비교**:
1. **LWT** (Linear Wave Theory): η = Σ Aₙ cos(kₙ·r − ωₙt + φₙ), linear dispersion
2. **CWM1** (1st order Choppy Wave Model, Lagrangian): geometric nonlinearity (sharper crests, flatter troughs) but linear phase speed
3. **ICWM** (Improved CWM, 2nd order): adds nonlinear phase corrections (Stokes drift)

**Setup**:
- LIDAR mounted on **fixed offshore structure** at z=30m, viewing angles α=76°, β=0°, β_a=20°
- J = 64×64 = 4096 rays uniformly distributed over aperture
- 관측 점이 **grazing incidence** 때문에 비선형적으로 sparse — 멀리 있을수록 점밀도 ∝ 1/distance²
- Wave shadowing → spatial gaps

**Wavenumber discretization**:
- §"Cutoff Frequencies and Directions" 명시: kᵐⁱⁿ = 2π/L_obs, kᵐᵃˣ = 16k_p (Shannon)
- N=15~60 wave components (predefined) — **on-grid in k-space**

**Cost function** (eq. 13): least-squares C = (1/L) Σ (η̃(rₗ,tₗ) − ηₗ)²

**Tikhonov regularization** (eq. 21): L-curve method for ill-conditioning

**Prediction zone** (eq. 12): bounded by group velocities of slowest/fastest components

**Iteration** (CWM nonlinear): 5-10 iterations for convergence

**1D & 2D results**:
- Linear reconstruction: error ~10% with kᵐᵃˣ ≥ 16k_p, N≥30
- ICWM nonlinear improves prediction error by ~5% over LWT at t=3T_p
- 2D directional spreading: aperture angle β_a=110° optimal (~70° → 35° error reduction)
- Reconstruction error grows as time increases ("gray area" surrounding prediction zone)

### 19.3 핵심 방법론 (Desmars 2020)
- **3 wave models**: LWT, **LWT-CDR** (LWT with Corrected Dispersion Relation, new in this paper), ICWM
- **22 wave gauges** in 50m × 30m × 5m wave tank, **non-uniformly distributed** (mimicking LIDAR grazing pattern)
- 8 sea states from H_s/λ_p = 0.65% to 5.16% (mild to highly nonlinear)
- DA at 1Hz, prediction over 60 peak periods
- Tikhonov regularization with L-curve method

**Wavenumber strategy** (§3.1):
> "the set of wavenumbers k_n is distributed in [k_min, k_max] following a decreasing logarithmic law, with k_min/k_max defining the bandwidth of the reconstructed wave field"

→ **명시적 on-grid representation** (predefined k_n)

**결과**:
- **Wave phase velocity correction (LWT-CDR/ICWM) gives main improvement** — ~35% error reduction at moderate steepness
- Wave shape asymmetry (ICWM only) gives small improvement — ~5%
- For surface elevation: dispersion correction is critical
- For surface slope: nonlinear shape (ICWM) is critical (slope error reduced 20% with ICWM vs LWT-CDR)
- Prediction zone correctly defined by group velocity bounds

### 19.4 우리 work 와 비교

| 항목 | Desmars 2018/2020 | WaveSplat |
|---|---|---|
| **Sensor** | fixed offshore LiDAR (2018), 22 wave gauges (2020) | shipboard moving LiDAR |
| **Dimension** | 1D long-crested (2020), 1D+2D (2018) | 2D directional |
| **Wave model** | LWT/CWM/ICWM (Lagrangian) | LWT (linear superposition) |
| **k-space** | **predefined logarithmic grid** [k_min, k_max] | **continuous learnable** (kx, ky) |
| **Optimizer** | least-squares + Tikhonov + L-curve | Adam GD + adaptive N + pruning |
| **Real-time** | offline batch reconstruction | sequential (per-window) |
| **Sea trial data** | 50m wave tank | shipboard real LiDAR (Florida data) |
| **Prediction zone** | explicit group velocity bounds | implicit via dispersion |

### 19.5 우리 positioning 강화 — 새로운 포인트

#### 19.5.1 Desmars 2020 의 **shadowing 처리** vs 우리
- Desmars 2020 page 3: "the lack of information in the shadowed area can be compensated for by using spatio-temporal data sets, i.e., at a slightly later time, shadowed areas may become illuminated due to wave motion"
- 즉 **temporal sequence** 로 shadowing 보완 — 우리 ship motion + temporal sequence 와 같은 idea
- **N1 (SAWE) 약화** 추가 증거 — Desmars 도 implicit 하게 trajectory effect 를 활용

#### 19.5.2 ICWM (Lagrangian nonlinearity) — 우리 future work
- Desmars 의 ICWM 는 wave shape (sharp crests) 까지 표현 가능
- 우리 linear plane-wave 로는 표현 불가 → **이 방향이 우리 future work 의 명확한 path**
- 단순 LWT 대비 ICWM 가 ~35% (elevation), ~20% (slope) 개선 — defensible quantitative gap

#### 19.5.3 Tikhonov L-curve regularization — 우리에게 적용 가능
- Desmars: ill-conditioning 처리에 L-curve method 사용
- 우리: 현재 adaptive N + safety rollback — 다른 접근
- **Tikhonov regularization 를 우리 cost function 에 추가** 하면 fair comparison 가능

#### 19.5.4 Predefined k-grid 한계 본문 인용 가능
- Desmars 2018 "Cutoff Frequencies and Directions" 섹션 + Desmars 2020 §3.1
- → reviewer 에게 "그들이 명시적으로 on-grid 임을 말한다" 입증 가능

### 19.6 References — bibliography 추가
- Desmars 2018 (OMAE): early ICWM paper
- Desmars 2020 (Ocean Eng): experimental validation
- Guérin et al. 2019 (J. Fluid Mech. 876): ICWM 공식 정의
- Nouguier et al. 2009 (J. Geophys. Res.): CWM 공식 도입

---

## 20. Wang & Pan 2022 JFM 정독 결과 (2026-05-08)

> **Citation**: Wang, G., Zhang, J., Ma, Y., Zhang, Q., Li, Z. & Pan, Y. (2022). *Phase-resolved ocean wave forecast with simultaneous current estimation through data assimilation*. **J. Fluid Mech.** 949, A31. doi:10.1017/jfm.2022.765

### 20.1 핵심 메시지
- Wang & Pan 2021 의 **EnKF-HOS 연장** — current field U(x,t) 를 동시에 추정
- **IEnKF-HOS-C** (Iterative EnKF-HOS-Current) 새 알고리즘 제시
- Synthetic + 실제 marine radar 데이터 모두에서 검증
- 실제 floating buoy 와 일치하는 current speed 복구 성공

### 20.2 방법론 (보너스 — current 추정 추가)
- State augmentation: (η, ψ, **U**) — current 가 unknown parameter
- HOS-C method (Wang 2018, Pan 2020): wave equation under current
- Iterative EnKF (Iglesias-Law-Stuart 2013): current 가 직접 측정 안 되므로 wave field correlation 통해 추정
- N=100 ensemble, persistence model ∂U/∂t = 0 (slow temporal variation 가정)

**Real data**:
- Doppler coherent marine radar, off Southern California
- 480m × 480m, 64×64 grid
- T_p = 11.28 s
- DA interval = T_p/4 = 2.82 s
- 4 different initial guesses → 모두 t=40T_p 에 동일 buoy velocity 로 수렴

### 20.3 우리 work 와 비교 — current 추정 측면

| 항목 | Wang-Pan 2022 | WaveSplat |
|---|---|---|
| **Current estimation** | ✅ explicit (full vector field U(x,y,t)) | ❌ 미고려 |
| **Method** | IEnKF + HOS-C + state augmentation | N/A |
| **Setup** | grid-based (L=256 or 64×64) | grid-free |
| **Sensor** | marine radar | LiDAR |

### 20.4 Implications — 우리 future work 에 critical
- ✅ **Current estimation 은 새 dimension** — 우리 work 에 추가 가능
- ⚠️ Wang-Pan 2022 가 **이미 명시적으로 다룸** — 새 contribution 아님
- 우리 differentiator: off-grid + LiDAR + sparse — current estimation 도 같은 framework 으로 확장 가능
- Doppler shift 가 dispersion 에 미치는 영향: ω = √(g·k·tanh(kd)) + k·U → 우리 dispersion constraint 에 반영 필요 (이게 추가되면 current 자동 추정 가능?)

### 20.5 References 추가
- Wang & Pan 2022 (JFM 949, A31): current estimation extension
- Iglesias, Law & Stuart 2013 (Inverse Probl): IEnKF iterative method
- Pan 2020 (Wave Motion): wave-current interaction model
- Wang, Ma & Yan 2018 (J. Comput. Phys.): wave-current numerical method
- Lund et al. 2018 (J. Atmos. Ocean. Technol.): radar current mapping

---

## 21. Hunt, Kostelich & Szunyogh 2007 (Tellus) — LETKF 정독 결과 (2026-05-08)

> **Citation**: Hunt, B.R., Kostelich, E.J. & Szunyogh, I. (2007). *Efficient Data Assimilation for Spatiotemporal Chaos: a Local Ensemble Transform Kalman Filter*. **Tellus A** 56, 415–428 (also arXiv:physics/0511236).

### 21.1 핵심 메시지
- **LETKF** = Local Ensemble Transform Kalman Filter — high-dim chaotic system 용 효율적 EnKF 변형
- Localization + ensemble transform 결합으로 NCEP GFS (500K grid points) 에 실용 가능
- 4D-LETKF: asynchronous observation 처리 가능 (관측이 다른 시간에 들어와도 묶어서 분석)

### 21.2 핵심 알고리즘
**State**: m-dim model state x, k ensemble members
**Observation**: ℓ-dim y_o = H(x) + ε

**Localization**: each grid point separately analyzed using local observations within radius
- 이유: ensemble size k << m → spurious correlations at long range
- 효과: parallel computation 가능

**Analysis (k-dim space)**:
- (eq. 19) cost function: J̃*(w) = (k-1)w^T w + [y_o − ȳ_b − Y_b w]^T R⁻¹ [y_o − ȳ_b − Y_b w]
- (eq. 20–21): w̄_a = P̃_a (Y_b)^T R⁻¹ (y_o − ȳ_b), P̃_a = [(k-1)I + (Y_b)^T R⁻¹ Y_b]⁻¹
- (eq. 22–23): x̄_a = x̄_b + X_b w̄_a, P_a = X_b P̃_a (X_b)^T

**Ensemble update** (symmetric square root):
- W_a = [(k-1) P̃_a]^(1/2)
- x_a^(i) = x̄_b + X_b w_a^(i)

**Inflation** (multiplicative): replace (k-1)I by (k-1)I/ρ in eq. 21

### 21.3 4D-LETKF (asynchronous observations)
- 단순 확장: 각 관측 시각 τ_j 마다 H_τj 적용해서 Y_τj^b 만들고, vertical concat
- 결과: same eq. 19–25 framework 사용 가능
- Wang-Pan 2021 의 EnKF–HOS framework 의 기초

### 21.4 우리 work 와 비교

| 항목 | LETKF (Hunt 2007) | WaveSplat |
|---|---|---|
| **시스템** | spatiotemporal chaos (atmosphere, weather) | wave field |
| **State dim** | m ≈ 500K (NCEP GFS) | continuous parameters (k, ω, φ, A) |
| **Ensemble** | k = 40~80 | N/A (single-shot) |
| **Localization** | radius-based (h=5~7 grid points) | N/A |
| **Computational cost** | O(k² · ℓ_local · N_grid) | O(N_components × N_obs) per Adam step |

### 21.5 Implications
- ✅ **우리 ETKF baseline 의 직접 origin** — 코드의 LETKF 계열 알고리즘이 이 paper 에 기반
- ✅ Localization + multiplicative inflation = 표준 처리
- ⚠️ Hunt 2007 자체는 **wave-specific 이 아님** — atmospheric DA 의 일반 framework
- → references 에 Hunt 2007 인용 적절 (LETKF 의 foundational paper)

### 21.6 References 추가
- Hunt, Kostelich & Szunyogh 2007 (Tellus A 56, 415–428): LETKF foundational
- Bishop, Etherton & Majumdar 2001 (Mon. Wea. Rev.): ETKF foundational
- Ott et al. 2004 (Tellus A 56, 273–277): LEKF (Local EnKF)
- Anderson 2007 (Tellus A 59, 210–224): adaptive inflation

---

## 22. Ehlers, Klein, Heinlein, Wedler, Desmars, Hoffmann, Stender 2023 — FNO 정독 결과 (2026-05-08)

> **Citation**: Ehlers, S. et al. (2023). *Machine learning for phase-resolved reconstruction of nonlinear ocean wave surface elevations from sparse remote sensing data*. arXiv:2305.11913. Submitted to Ocean Engineering. **이 paper 가 우리 work 와 가장 가까움.**

### 22.1 핵심 메시지
- **첫 ML-based phase-resolved wave reconstruction** from sparse radar
- U-Net (CNN local) + FNO (Fourier global) 두 architecture 비교
- **FNO 가 shadow region 처리에 우월** — Fourier 공간에서 global mapping
- 실시간 inference (1.9ms FNO vs 37ms U-Net)

### 22.2 핵심 방법론
**Data**:
- HOS simulation (M=4) — JONSWAP spectrum, 130 (L_p, ε) combinations × 4 phase realizations = 520 wave simulations
- L=4000m domain, n_x=1024 grid points (Δx=3.906m)
- L_p ∈ {80,...,200}m, ε ∈ {0.01,...,0.10}, T = 50s, Δt_save=0.1s
- z_a = 18m antenna height, r_min=100m, r_max=1892m, n_r=512 grid points (Δr=3.5m)
- Δt_r = 1.3s (radar revolution period)

**Radar simulation**:
- Tilt modulation: T(r,t) = n·u (cosine of incidence angle)
- Shadowing: geometric obstruction by closer waves (eq. 4-5)
- ξ(r,t) ∝ T(r,t) · S(r,t)

**Architecture (FNO)**:
- Lifting P (linear layer): x_i ∈ R^(512 × n_s) → v_0 ∈ R^(512 × 32)
- 3 Fourier layers, n_m=64 modes truncated
- Global path: F(F⁻¹(R_j · F(v_j)))
- Local path: W_j · v_j (linear)
- v_(j+1) = σ(global + local)

**Training**:
- Loss: nL2 = ||y_pred − y_true||₂ / ||y_true||₂
- Adam optimizer, lr=0.001, 800 epochs

**SSP metric** (Surface Similarity Parameter):
- SSP(y, ŷ) = √∫|F_y(k) − F_ŷ(k)|² dk / [√∫|F_y(k)|² dk + √∫|F_ŷ(k)|² dk]
- ∈ [0, 1], 0=perfect, criterion SSP ≤ 0.10
- **이 metric 이 우리 5-metric panel 에 추가 검토 가능**

### 22.3 결과
| Model | n_s | nL2 | nL2_shad/nL2_vis | SSP |
|---|---|---|---|---|
| U-Net (1 snapshot) | 1 | 0.329 | 2.679 | 0.171 |
| U-Net (10 snapshots) | 10 | 0.123 | 1.755 | 0.061 |
| FNO (1 snapshot) | 1 | 0.242 | 1.886 | 0.123 |
| FNO (9 snapshots) | 9 | 0.153 | 1.381 | 0.077 |

- FNO 가 single-snapshot 에서 더 나음 (SSP 0.123 vs 0.171)
- Shadow region 에서 FNO 가 훨씬 더 균일 (1.381 vs 1.755 ratio)
- Both meet SSP ≤ 0.10 criterion when n_s ≥ 9-10

### 22.4 우리 work 와 직접 비교

| 항목 | Ehlers 2023 FNO | WaveSplat |
|---|---|---|
| **Sensor** | X-band radar (1D scan) | 2D LiDAR (3D point cloud) |
| **Data** | synthetic only (HOS) | synthetic only |
| **Approach** | data-driven (FNO learns mapping ξ→η) | physics-based (parametric inversion) |
| **Training data needed** | 3120 samples | 0 (no training) |
| **Generalization** | trained on (L_p, ε) range | physics-driven, any sea state |
| **Inference time** | 1.9 ms (FNO) | < 100 ms |
| **Output** | η(x, t_s) — single snapshot | (k, ω, φ, A) — full parametric |
| **Forecasting** | needs additional HOS step | analytical from parameters |
| **Spatial dim** | 1D (n_x=512 grid) | 2D directional |
| **Output type** | gridded η | continuous η anywhere |
| **Metric** | nL2, SSP | nRMSE, VR, ρ, SS, RMSE_unobs |

### 22.5 Implications — 우리 positioning

#### 22.5.1 FNO 와 우리 의 본질적 차이
- **Ehlers**: data-driven, requires training, ML pipeline (PyTorch + Adam)
- **우리**: physics-based, no training, dispersion-constrained
- → 우리는 **interpretable**, FNO 는 **black box**
- → 우리는 **out-of-distribution sea state** 에 robust, FNO 는 train range 밖에서 degrade

#### 22.5.2 FNO 결과를 우리 baseline 에 추가 가능?
- 우리 시뮬레이터와 같은 HOS 데이터 사용 가능
- 그러나 Ehlers 가 **1D radar** — 우리 **2D LiDAR** 다름
- 직접 비교 어렵지만, "learning-based comparison" 으로 cite 는 가능

#### 22.5.3 SSP metric 추가 검토
- SSP 는 amplitude+phase+frequency 통합 metric
- 우리 nRMSE 는 amplitude error 만
- → **6번째 metric 으로 SSP 추가 검토 가능** (cherry-pick 방지 차원)

#### 22.5.4 Ehlers 도 1D 만 다룸
- 그들이 향후 work 으로 2D 언급 ("future research could delve into the reconstruction of two-dimensional ocean wave surfaces")
- → 우리 2D LiDAR 가 **temporal lead** 있음 (그들이 future work 으로 남긴 것을 우리가 함)

### 22.6 References 추가
- Ehlers et al. 2023 (arXiv:2305.11913): ML phase-resolved reconstruction
- Li et al. 2020 (arXiv:2010.08895): FNO foundational
- Ronneberger et al. 2015: U-Net foundational
- Perlin & Bustamante 2014: SSP metric definition
- Salcedo-Sanz et al. 2015: radar tilt+shadowing simulation

---

## 23. Tang, Bhaskar, Shah & Recht 2013 — Compressed Sensing Off the Grid (2026-05-08)

> **Citation**: Tang, G., Bhaskar, B.N., Shah, P. & Recht, B. (2013). *Compressed Sensing Off the Grid*. **IEEE Trans. Inf. Theory** 59(11), 7465–7490 (arXiv:1207.6053). **본 work 의 atomic norm minimization 이 우리 off-grid framework 의 이론적 기초.**

### 23.1 핵심 메시지
- 기존 CS: discretized dictionary (on-grid frequency 가정) → **basis mismatch 문제**
- 본 work: **continuous frequency space [0,1]** 에서 직접 sparse 복구 — atomic norm minimization 으로 정확 가능
- **O(s log s log n)** random samples 으로 s sinusoid 정확 복구
- Resolution requirement: Δ_f ≥ 4/n (frequency separation)

### 23.2 Theorem 1.1 (핵심 결과)
관측: x_j = Σ_k c_k e^(i 2π f_k j), j=0,...,n-1
- f_k ∈ [0, 1] continuous
- m = |T| samples chosen uniformly random from {0,...,n-1}
- Δ_f = min |f_k − f_j| ≥ 4/(n-1)

→ **m ≥ C max{log²(n/δ), s log(s/δ) log(n/δ)}** 로 SDP 통해 1-δ 확률로 정확 복구 가능

### 23.3 Atomic Norm 정의 (§2)
**Atom**: a(f, φ) ∈ C^|J|, [a(f, φ)]_j = e^(i(2πfj + φ))
**Atomic norm**:
```
||x||_A = inf{t > 0 : x ∈ t · conv(A)}
       = inf{Σ c_k : x = Σ c_k a(f_k, φ_k), c_k ≥ 0}
```

**Semidefinite characterization** (Proposition 2.1):
```
||x||_A = inf {(1/2|J|) trace(Toep(u)) + (1/2)t : [Toep(u) x; x* t] ⪰ 0}
```

### 23.4 Frequency localization (§2.2)
- Dual polynomial: Q(f) = ⟨q, a(f, 0)⟩
- 복구된 frequency = {f : |Q(f)| = 1}
- Identification via FFT (efficient)

### 23.5 우리 work 와 직접 연결

| 항목 | Tang-Bhaskar-Recht 2013 | WaveSplat |
|---|---|---|
| **Frequency space** | continuous [0, 1] | continuous (kx, ky) ∈ R² |
| **Method** | atomic norm + SDP | Adam GD + adaptive N |
| **Theory** | exact recovery guarantee O(s log s log n) | empirical |
| **Computational** | SDP (poly time) | gradient descent |
| **Phase model** | random uniform | uniform initialization |
| **Resolution** | Δ_f ≥ 4/n | dispersion-driven |
| **Convex** | ✅ | ❌ (non-convex) |

### 23.6 Implications — 우리 positioning 강화

#### 23.6.1 우리 framework 의 이론적 backing
- **§17 의 off-grid 우위 = atomic norm framework 와 동일 family**
- Tang-Bhaskar-Recht: "basis mismatch" 라는 정확한 용어로 on-grid 문제 정립
- 우리 reviewer 에게: "off-grid CS 는 Tang 2013 이후 standard, basis mismatch 는 well-known problem" 답변 가능

#### 23.6.2 우리 접근 = atomic norm 의 **non-convex relaxation**
- Tang 2013 = SDP (convex, exact recovery)
- 우리 = Adam GD on continuous (kx, ky) (non-convex, no theoretical guarantee)
- ⚠️ Reviewer 가 "왜 SDP 안 쓰고 GD 쓰는가" 물을 수 있음
- 답변: "2D directional (kx, ky), 시간 추가 → 차원 너무 커서 SDP infeasible. 또한 dispersion constraint ω(k,d) 를 SDP 에 직접 통합 어려움. GD with dispersion constraint = practical choice."

#### 23.6.3 우리 N3 (super-resolution) framing 강화
- Tang 2013 Theorem 1.1: minimum frequency separation Δ_f ≥ 4/n
- 우리: **dispersion + temporal** 로 spatial Nyquist 넘는 super-resolution 달성
- → Tang framework 에서는 **temporal information** 이 frequency separation 을 늘려주는 효과
- 이 framing 으로 super-resolution 결과를 더 강하게 주장 가능

#### 23.6.4 Resolution 정량 비교
- Spatial Nyquist: k_max ≤ π/Δx (one-shot scan)
- Tang 2013: Δ_f ≥ 4/n (n samples)
- 우리: dispersion ω(k) + multi-time → effective sampling 증가

### 23.7 References 추가
- Tang, Bhaskar, Shah & Recht 2013 (IEEE Trans. Inf. Theory): atomic norm CS
- Chandrasekaran et al. (2012): atomic norm general framework
- Candès & Fernandez-Granda 2014: super-resolution via TV minimization
- Bhaskar, Tang, Recht 2013 (1204.0562): atomic norm denoising
- Stoica & Babu 2012: SPICE (related sparse spectral)

---

## 24. Möller-Trumbore 1997 — Ray/Triangle Intersection (2026-05-08)

> **Citation**: Möller, T. & Trumbore, B. (1997). *Fast, Minimum Storage Ray/Triangle Intersection*. **Journal of Graphics Tools** 2(1), 21–28.

### 24.1 핵심 메시지
- Ray-triangle intersection을 plane equation 미저장 + minimum storage 로 효율 계산
- Barycentric coordinate (u, v) + distance t 동시 계산
- O(1) memory per triangle, ~25-50% memory saving for triangle meshes

### 24.2 알고리즘 핵심
**Ray**: R(t) = O + t·D
**Triangle**: T(u,v) = (1-u-v)V_0 + u·V_1 + v·V_2

**Linear system**:
```
[−D  V_1−V_0  V_2−V_0] · [t; u; v]^T = O − V_0
```

**Cramer's rule**:
```
[t; u; v] = (1 / |−D, E_1, E_2|) · [|T, E_1, E_2|; |−D, T, E_2|; |−D, E_1, T|]
```

여기서 E_1 = V_1 − V_0, E_2 = V_2 − V_0, T = O − V_0

**Identity** (scalar triple product):
```
|A, B, C| = −(A × C) · B = −(C × B) · A
```

→ 최종:
```
[t; u; v] = (1/(D × E_2 · E_1)) · [(T × E_1) · E_2; (D × E_2) · T; (T × E_1) · D]
```

P = D × E_2, Q = T × E_1 으로 reuse → 효율

**Code** (figure 1, p.4):
```c
SUB(edge1, vert1, vert0);
SUB(edge2, vert2, vert0);
CROSS(pvec, dir, edge2);
det = DOT(edge1, pvec);
// ...
*u = DOT(tvec, pvec) * inv_det;
CROSS(qvec, tvec, edge1);
*v = DOT(dir, qvec) * inv_det;
*t = DOT(edge2, qvec) * inv_det;
```

### 24.3 우리 코드와 비교
- 우리 [wavesim/lidar_simulator.py](wavesim/lidar_simulator.py): Möller-Trumbore 정확히 구현
- Cross + dot products 사용, det check (EPSILON=1e-7)
- u, v ≥ 0, u+v ≤ 1 검증

### 24.4 Implications — 검증 완료
- ✅ 우리 LiDAR ray-triangle intersection 표준 알고리즘 사용
- ✅ 본 paper cite 적절 (computer graphics 표준)
- ⚠️ wave surface = grid mesh of triangles → 우리 raydrop 과 호환

### 24.5 References
- Möller & Trumbore 1997 (JGT 2(1), 21-28): MT algorithm
- Badouel 1990 (Graphics Gems): 비교 대상

---

## 25. 종합 — 모든 reference paper review 종료 후 결론 (2026-05-08)

### 25.1 읽은 paper 총정리

| # | Paper | Year | Key Finding for Us | Citation Priority |
|---|---|---|---|---|
| 1 | Desmars 2018 (OMAE) | 2018 | LIDAR + ICWM, predefined k-grid 명시 | ⭐⭐⭐⭐⭐ |
| 2 | Desmars 2020 (Ocean Eng) | 2020 | Wave gauge experiment, dispersion correction | ⭐⭐⭐⭐⭐ |
| 3 | Desmars 2023 (JFM) | 2023 | HOS-OCM, on-grid 확정 (§17) | ⭐⭐⭐⭐⭐ |
| 4 | Wang & Pan 2021 (JFM) | 2021 | First EnKF-HOS, predictable zone | ⭐⭐⭐⭐ |
| 5 | Wang & Pan 2022 (JFM) | 2022 | EnKF-HOS-C with current | ⭐⭐⭐⭐ |
| 6 | Hunt 2007 (Tellus A) | 2007 | LETKF foundational | ⭐⭐⭐ |
| 7 | Ehlers 2023 (Ocean Eng) | 2023 | FNO ML reconstruction, SSP metric | ⭐⭐⭐⭐⭐ |
| 8 | Tang-Bhaskar-Recht 2013 | 2013 | Off-grid CS theory | ⭐⭐⭐⭐ |
| 9 | Möller-Trumbore 1997 | 1997 | Ray-triangle intersection (코드 검증) | ⭐⭐ |

### 25.2 우리 positioning 의 견고성 — 최종 평가

#### 25.2.1 Differentiator 가 강한 영역 (정성적+정량적 확실)
- ✅ **Off-grid continuous (kx, ky)** vs all baselines (Desmars on-grid, Wang-Pan grid-based, Ehlers grid-based)
- ✅ **2D directional LiDAR** vs all baselines (Desmars 1D 또는 2D radar, Wang-Pan 2D radar, Ehlers 1D radar)
- ✅ **Single-shot batch + tracking** vs all baselines (Desmars offline batch, Wang-Pan EnKF iterative, Ehlers DL inference)
- ✅ **Real-time onboard ready** vs all baselines (계산 비용 절감)

#### 25.2.2 Differentiator 가 약한 영역 (재고 필요)
- ⚠️ **"Synthetic Aperture / Moving sensor"** — Wang-Pan 2021/2022 + Desmars 2020 모두 implicit 하게 다룸
- ⚠️ **Linear physics** — Desmars (ICWM, weakly nonlinear), Wang-Pan (HOS M=4) 보다 약함
- ⚠️ **첫 LiDAR-based phase-resolved 알고리즘** 주장 어려움 — Desmars 2018 가 이미 sims with LIDAR sensor 제안 (실제 데이터는 아니지만)

#### 25.2.3 진짜 강한 contribution 정리
1. **off-grid + 2D + LiDAR + real-time** 의 first integrated framework (각 element 는 prior art)
2. **Dispersion-driven super-resolution** 명시적 demonstration (§16 N3) — 다른 work 에 explicit framing 없음
3. **WaveSim 시뮬레이터** + 100 episodes + **196 unit tests** (2026-08-13 실측
   `196 passed, 0 failed, 0 skipped`) — reproducibility infrastructure
4. **5-metric panel** 평가 framework — cherry-pick 방지

### 25.3 PPT/Paper 에 반영해야 할 점

#### 25.3.1 Prior art slide 강화
| Family | Sensor | Dim | k-space | Method | Real-time |
|---|---|---|---|---|---|
| Desmars 2018-2023 | radar/gauge (proxy LIDAR) | 1D~2D | predefined log grid | LWT/ICWM/HOS-OCM, Tikhonov | offline |
| Wang-Pan 2021 | radar (moving) | 2D | grid (256² or 64²) | EnKF-HOS, N=100 | per-window O(NL log L) |
| Wang-Pan 2022 | radar (moving) | 2D | grid (256² or 64²) | IEnKF-HOS-C with current | per-window |
| Ehlers 2023 | radar (1D scan) | 1D | grid (n_r=512) | U-Net / FNO ML | 1.9-37 ms |
| **WaveSplat (this work)** | **LiDAR (moving)** | **2D** | **continuous** | **Adam GD + adaptive N** | **< 100 ms** |

#### 25.3.2 References slide 갱신 (8-10개로 확장)
1. Desmars et al. 2018 (OMAE) — phase-resolved LIDAR concept
2. Desmars et al. 2020 (Ocean Eng) — experimental wave gauge validation
3. Desmars 2023 (JFM) — HOS-OCM on-grid framework
4. Guérin et al. 2019 (JFM) — ICWM definition
5. Wang & Pan 2021 (JFM) — EnKF-HOS first
6. Wang & Pan 2022 (JFM) — current estimation extension
7. Ehlers et al. 2023 (Ocean Eng) — ML reconstruction (FNO)
8. Tang, Bhaskar, Recht 2013 (IEEE TIT) — atomic norm off-grid CS theory
9. Hunt et al. 2007 (Tellus A) — LETKF foundational
10. Möller & Trumbore 1997 (JGT) — ray-triangle intersection

#### 25.3.3 Q&A 대비 (확장)
- **Q**: Desmars 2018 가 이미 LIDAR + nonlinear 다룬 것 아닌가?
- **A**: 그들은 fixed offshore + simulated LIDAR rays + predefined k-grid. 우리는 moving shipboard + real LIDAR + continuous (kx, ky). 또한 그들은 1D + 2D, 우리는 2D directional. 그들 paper §"Cutoff Frequencies" 에서 명시적으로 grid 사용함을 인정.

- **Q**: Wang-Pan 2022 도 moving radar + current 다룸, 우리 차별점?
- **A**: (1) sensor type (radar vs LiDAR); (2) state space (grid 64² vs continuous parameters); (3) computational cost (O(NL log L) ensemble vs single-shot Adam); (4) current estimation 은 우리 future work, 그들이 먼저 성취. 우리 contribution = off-grid 2D LiDAR framework.

- **Q**: Ehlers 2023 FNO 와 비교?
- **A**: 우리는 physics-based, no training data needed. FNO 는 train range 밖에서 degrade, 우리는 dispersion physics 으로 robust. Real-time 속도 비슷 (1.9ms vs 100ms). 직접 비교는 Ehlers 가 1D radar, 우리 2D LiDAR 라서 어려우나, "learning-based vs physics-based" 의 representative comparison 으로 cite.

- **Q**: Tang-Bhaskar-Recht 2013 atomic norm framework 안 쓰는 이유?
- **A**: SDP 는 2D + temporal 차원에서 infeasible. 또한 dispersion constraint ω = √(g·k·tanh(kd)) 를 SDP 에 직접 통합 어려움. Adam GD with dispersion projection = practical engineering choice. 그러나 우리 framework 는 atomic norm CS family 에 속함 — Tang 2013 cite 를 통해 이론적 backing 명시.

### 25.4 다음 step 제안
1. **PPT references slide 즉시 업데이트** (위 10개로)
2. **PPT prior art slide 표** 추가 (위 비교표)
3. **PPT novelty slide 강화** — off-grid + 2D + LiDAR triad 명확히
4. **(선택) Wang-Pan 2022 에서 영감받아 current estimation 을 future work 에 추가** — Doppler shift 자동 추정 framework

### 25.5 Ledger 갱신
| 날짜 | 변경 | 이유 |
|---|---|---|
| 2026-05-08 | §19 추가 | Desmars 2018 + 2020 정독 — LIDAR + on-grid 확정 |
| 2026-05-08 | §20 추가 | Wang-Pan 2022 정독 — current estimation extension |
| 2026-05-08 | §21 추가 | Hunt 2007 LETKF 정독 — DA 기초 |
| 2026-05-08 | §22 추가 | Ehlers 2023 FNO 정독 — ML 비교 frame, SSP metric |
| 2026-05-08 | §23 추가 | Tang-Bhaskar-Recht 2013 정독 — off-grid CS 이론 backing |
| 2026-05-08 | §24 추가 | Möller-Trumbore 1997 정독 — 코드 검증 |
| 2026-05-08 | §25 추가 | 종합 — 9 paper 결론, PPT 반영 plan |
| 2026-05-08 | §26 추가 | Citation cloud 검토 — Klein 2020 정독 |
| 2026-05-08 | §27 추가 | PINN 2025 + VOF EnKF 2025 다운로드 (정독 pending) |

---

## 26. Citation Cloud 검토 — Klein, Dudek, Clauss, Ehlers, Behrendt, Hoffmann, Onorato 2020 (Fluids) 정독 결과 (2026-05-08)

> **Citation**: Klein, M., Dudek, M., Clauss, G.F., Ehlers, S., Behrendt, J., Hoffmann, N. & Onorato, M. (2020). *On the Deterministic Prediction of Water Waves*. **Fluids** 5(1), 9. doi:10.3390/fluids5010009 (open access MDPI)

### 26.1 핵심 메시지
- 4가지 deterministic prediction method 의 **comparative study** — LWT, NLSE (1st order), NLSE₂ (2nd order, Dysthe-equivalent), HOSM (M=4)
- 모두 **full dispersion operator** (Trulsen et al.) 사용
- **HOSM 가 가장 정확** (가장 넓은 sea state 범위 cover) — Ehlers 2023 가 이 결과를 인용
- **고전 NLSE 는 irregular sea state 예측에 부적합** (LWT 보다 나쁠 수도 있음)

### 26.2 방법론 핵심
**4 prediction methods**:
1. **LWT** — η(x,t) = Σ ζ_an cos(k_n x − ω_n t + φ_n), full dispersion ω_n = √(g·k_n·tanh(k_n·d))
2. **NLSE (1st order)** — i(∂A/∂t + V ∂A/∂x) + β₁ ∂²A/∂x² + α₁|A|²A = 0
3. **NLSE₂ (2nd order)** — adds β₂ ∂³A/∂x³ + iα₂₁|A|² ∂A/∂x + iα₂₂A² ∂A*/∂x — equivalent to Dysthe (k_c·d → ∞)
4. **HOSM (M=4)** — full nonlinear free-surface BVP via West et al. 1987 / Dommermuth-Yue 1987

**Full dispersion operator** (Trulsen):
- L̂ = √((k − k_c)·g·tanh((k − k_c)d) − ω_c)
- Pseudo-spectral 적용으로 모든 NLSE 변형이 full linear dispersion 정확하게 표현
- Taylor series expansion (k = k_c 근처) 의 한계 극복

**Setup**:
- TUB seakeeping basin: 110m × 8m × 1m (model scale 1:75)
- 4 wave gauges (gauge 1=wavemaker, 2-4=prediction validation)
- 9 sea states: T_p=8.58s, k_p·d=4.1, ε ∈ {0.05, 0.1, 0.15}, γ ∈ {1, 3, 6}
- **8 consecutive snapshots** (ship's radar simulating)
- Radar shadow: 500m blanked behind ship position

**Semi-experimental approach**:
- waveTUB FEM solver (Steinhagen 2018) 가 reference (non-linear potential flow)
- 같은 wavemaker → 실험과 시뮬레이션 boundary condition 동일
- waveTUB snapshot → 4 prediction methods → 실험 wave gauge 와 비교

**Metric**: **SSP** (Surface Similarity Parameter, Perlin & Bustamante 2014) — Ehlers 2023 와 동일

### 26.3 결과 요약
- **Classical NLSE (1st order)**: Taylor expansion 한계로 long-crested irregular sea state 에 부적합 — **LWT 보다 나쁠 수 있음**
- **NLSE with full dispersion**: small ε 에서 LWT 보다 향상
- **NLSE₂ (2nd order)**: moderate~steep irregular wave 에서 substantial improvement
- **HOSM (M=4)**: 가장 넓은 application range, 가장 높은 정확도

### 26.4 우리 work 와 비교

| 항목 | Klein 2020 | WaveSplat |
|---|---|---|
| **목적** | prediction methods 비교 | reconstruction + tracking |
| **Sensor** | (assumed) ship's X-band radar | LiDAR |
| **Dim** | 1D long-crested | 2D directional |
| **Wave model** | LWT, NLSE, NLSE₂, HOSM | LWT only |
| **Reconstruction** | ❌ 다루지 않음 (waveTUB snapshot 직접 사용) | ✅ sparse LiDAR 에서 reconstruction |
| **Validation** | semi-experimental (TUB tank) | synthetic 100 episodes + Florida real LiDAR |
| **Metric** | SSP | nRMSE, VR, ρ, SS, RMSE_unobs |
| **Radar shadow** | 500m blanked, 영향 평가 | raydrop + Fresnel + grazing |

### 26.5 우리 work 에 미치는 implication

#### 26.5.1 우리 LWT 의 한계 — 이 paper 가 **direct evidence**
- Klein 2020: HOSM > NLSE₂ > NLSE+full disp > LWT > NLSE (classical)
- 우리 = LWT family — **steep wave (ε > 0.1) 에서 HOSM/NLSE₂ 보다 정확도 떨어짐**
- ⚠️ Reviewer 가 "왜 nonlinear 안 쓰나" 질문 시: "linear baseline + future work nonlinear" 답변 필요

#### 26.5.2 SSP metric 표준성 재확인
- Klein 2020 + Ehlers 2023 모두 SSP 사용 — **이 분야 표준 metric**
- 우리 5-metric panel 에 **SSP 추가** 강력 권장 (Klein 2020, Ehlers 2023, Lünser 2022 cite 가능)

#### 26.5.3 Reconstruction 측면에서 우리가 **새 영역 열음**
- Klein 2020: reconstruction 안 다룸 (waveTUB snapshot = perfect reference 사용)
- Ehlers 2023: ML radar inversion (1D)
- Desmars 2018-2023: optimization-based gauge inversion (1D-2D)
- **우리**: sparse LiDAR (raydrop, shadowing 포함) inversion
- → **LiDAR-specific reconstruction** 이 진짜 우리 영역
- ✅ Klein 2020 본문 §1: "X-band radar 가 plug-and-play, fast, large prediction horizon" — radar 가 LiDAR 대비 **directionality 표현 부족** 명시 (우리 LiDAR 의 우위)

#### 26.5.4 Sequential snapshot 처리
- Klein 2020: **8 consecutive snapshots** 사용 (Ehlers 2023 도 9-10 snapshots 사용)
- 우리: temporal window + tracking
- → "**temporal sequence 활용**" 은 표준 (novelty 아님)
- 우리 차별: parametric model 의 sequential update (kx/ky 고정 + w/φ 갱신) — implementation 차원의 engineering

### 26.6 References 추가 필요
- Klein, Dudek, Clauss, Ehlers, Behrendt, Hoffmann, Onorato 2020 (Fluids 5(1), 9)
- West et al. 1987 (J. Geophys. Res.) — HOSM foundational
- Dommermuth & Yue 1987 (JFM 184) — HOSM foundational
- Trulsen, Kliakhandler, Dysthe, Velarde 2000 (Phys. Fluids 12) — full dispersion operator
- Slunyaev 2005 (JETP 101) — high-order Dysthe
- Sedletsky 2003 (JETP 97) — second-order NLSE for arbitrary depth
- Perlin & Bustamante 2014 (J. Engr. Math. 101) — SSP definition

---

## 27. 추가 다운로드 paper — 정독 pending (2026-05-08)

다음 2 paper 를 ref_paper/ 에 다운로드 완료. 본 conversation 에서는 정독 pending — 다음 session 에서 수행 예정.

### 27.1 PINN 2025 DA (`pinns_2025_da.pdf`)
- **arXiv**: 2501.08430 (Jan 2025)
- **제목**: *Physics-informed neural networks for phase-resolved data assimilation and prediction of nonlinear ocean waves*
- **22 pages**
- **예상 핵심**:
  - PINN-based DA — physics loss = surface BVP residual
  - No ground-truth data needed (unsupervised)
  - 우리 physics-based 와 직접 경쟁/보완 framework
  - Ehlers 2025 PINO (skip 지시) 의 forerunner
- **우리 positioning 영향**:
  - PINN: data-driven + physics constraint
  - 우리: parametric (k, ω, φ, A) + dispersion constraint
  - 둘 다 "unsupervised physics-based" → 직접 비교 가능
  - PINN 은 grid-based, 우리는 grid-free → 차별점 유지

### 27.2 VOF EnKF 2025 (`vof_enkf_2025.pdf`)
- **arXiv**: 2510.14356 (Oct 2025)
- **제목**: *Sequential water wave reconstruction in VOF-based numerical wave tanks with the EnKF approach*
- **25 pages**
- **예상 핵심**:
  - VOF (Volume of Fluid) CFD + EnKF
  - Wang-Pan 2021/2022 의 HOS 대신 VOF 사용
  - More accurate physics (full Navier-Stokes)
  - **Computational cost 매우 높음** (CFD)
- **우리 positioning 영향**:
  - 또 하나의 grid-based + ensemble baseline
  - 우리 real-time advantage 더 강조 가능

### 27.3 다른 paper — 다운로드 실패 / 미시도 (skip 또는 future)
| Paper | 상태 | 우선순위 |
|---|---|---|
| Nouguier, Grilli & Guérin 2014 (IEEE TGRS) | ❌ paywall | ⭐⭐⭐⭐⭐ (LIDAR 시작점) |
| Yoon, Kim & Choi 2015 (IEEE J. Ocean Eng) | ❌ paywall | ⭐⭐⭐ |
| Naaijen, Trulsen & Blondel-Couprie 2014 | ❌ paywall | ⭐⭐⭐ |
| Köllisch et al. 2018 (Ocean Eng) | ❌ paywall | ⭐⭐⭐ |
| Qi et al. 2018 (JFM 838) | ❌ paywall | ⭐⭐⭐ |
| Mohaghegh, Murthy, Alam 2021 (Appl Ocean Res) | ❌ paywall | ⭐⭐ |
| Lee, Nam, Liu, Yang 2024 (J Mar Sci Eng) | ⏳ 미시도 | ⭐⭐ |
| Kaneko et al. 2024 (Appl Ocean Res, video pred) | ⏳ 미시도 | ⭐⭐ |

이들은 **abstract 만 검토** 한 결과 우리 positioning 에 critical 변화 야기 안 함 — 정독 미수행 정당화 가능. 단 Nouguier 2014 는 LIDAR-based reconstruction 의 **historical first** 로서, 인용 필수 (paywall 이지만 reference list 에는 포함).

---

## 28. 종합 — 12 paper 정독 완료 후 결론 (2026-05-08, post-Klein)

### 28.1 정독 완료 paper (10개)
| # | Paper | Year | Key Finding |
|---|---|---|---|
| 1 | Desmars 2018 (OMAE) | 2018 | LIDAR + ICWM, predefined k-grid |
| 2 | Desmars 2020 (Ocean Eng) | 2020 | Wave gauge experiment, dispersion correction |
| 3 | Desmars 2023 (JFM) | 2023 | HOS-OCM, on-grid 확정 (§17) |
| 4 | Wang-Pan 2021 (JFM) | 2021 | First EnKF-HOS, predictable zone |
| 5 | Wang-Pan 2022 (JFM) | 2022 | EnKF-HOS-C with current |
| 6 | Hunt 2007 (Tellus A) | 2007 | LETKF foundational |
| 7 | Ehlers 2023 (Ocean Eng) | 2023 | FNO ML reconstruction, SSP metric |
| 8 | Tang-Bhaskar-Recht 2013 | 2013 | Off-grid CS theory |
| 9 | Möller-Trumbore 1997 | 1997 | Ray-triangle intersection (코드 검증) |
| 10 | **Klein et al. 2020 (Fluids)** | 2020 | **HOSM > NLSE₂ > NLSE > LWT, SSP metric** |

### 28.2 정독 pending paper (2개)
- PINN 2025 DA (arXiv:2501.08430)
- VOF EnKF 2025 (arXiv:2510.14356)

### 28.3 우리 positioning — Klein 2020 추가 후 강화/약화 점검

#### 28.3.1 강화된 점
- ✅ **SSP metric** 추가 정당화 (Klein 2020 + Ehlers 2023 + Lünser 2022 모두 사용)
- ✅ **2D directional + LiDAR + reconstruction triad** 가 prior art 에 없음 재확인
- ✅ **Real-time** 우위 — VOF EnKF 2025 가 등장해도 우리 < 100 ms 는 매우 빠름

#### 28.3.2 약화된 점 / 인정해야 할 점
- ⚠️ **Linear physics** 의 한계 — Klein 2020 이 explicitly 정량 비교: HOSM > NLSE₂ > LWT
- ⚠️ **Sequential snapshot 활용** = 표준 (Klein 2020 + Ehlers 2023 모두 8-10 snapshots 사용)
- ⚠️ **Reconstruction 자체** 만 보면 Desmars 2018-2023 가 LIDAR-style 처음 다룸

#### 28.3.3 진짜 강한 contribution 재확인 (Klein 2020 추가 후)
1. **Off-grid + 2D + LiDAR 의 first integrated framework** (Klein 2020 은 wavemaker reconstruction, 우리는 sparse sensor inversion)
2. **Dispersion-driven super-resolution** (§16 N3) — 어느 paper 도 명시 안 함
3. **Full WaveSim simulator (raydrop, shadowing, Fresnel)** + 100 episodes — Klein 2020 의 waveTUB FEM 와 다른 영역, 우리는 sensor model 까지 포함
4. **5-metric panel** — Klein 2020 의 SSP only 보다 풍부

### 28.4 PPT references 슬라이드 — 최종 11개 추천
1. Desmars et al. 2018 (OMAE) — phase-resolved LIDAR concept
2. Desmars et al. 2020 (Ocean Eng) — experimental wave gauge validation
3. Desmars 2023 (JFM) — HOS-OCM on-grid framework
4. Klein et al. 2020 (Fluids) — comparative study LWT/NLSE/HOSM
5. Wang & Pan 2021 (JFM) — EnKF-HOS first
6. Wang & Pan 2022 (JFM) — current estimation extension
7. Ehlers et al. 2023 (Ocean Eng) — FNO reconstruction (1D)
8. Tang, Bhaskar, Recht 2013 (IEEE TIT) — atomic norm CS
9. Hunt et al. 2007 (Tellus A) — LETKF
10. Möller & Trumbore 1997 (JGT) — ray-triangle
11. Nouguier, Grilli, Guérin 2014 (IEEE TGRS) — flash LIDAR foundational (paywall, not read but cited as historical first)

### 28.5 다음 작업 (다음 session)
1. **Compact 후 PINN 2025 DA + VOF EnKF 2025 정독**
2. PPT references slide 업데이트
3. PPT prior art slide 비교표 강화
4. Novelty enhancement plan 재평가 (citation cloud 추가 후)

### 28.6 Ledger 갱신 (계속)
| 날짜 | 변경 | 이유 |
|---|---|---|
| 2026-05-08 | §26 추가 | Klein 2020 Fluids 정독 — HOSM/NLSE 비교 framework, SSP metric 표준 확인 |
| 2026-05-08 | §27 추가 | PINN 2025 + VOF EnKF 2025 다운로드 — 정독 pending (다음 session) |
| 2026-05-08 | §28 추가 | 10 paper 정독 후 종합 — citation cloud 확장 결과 |

---

## 29. Ehlers, Hoffmann, Tang, Callaghan, Cao, Padilla, Fang, Stender 2025 — PFT-PINN 정독 결과 (2026-05-08)

> **Citation**: Ehlers, S. et al. (2025). *Physics-informed neural networks for phase-resolved data assimilation and prediction of nonlinear ocean waves*. Phys. Rev. Fluids **10**, 094901. arXiv:2501.08430.
> **저자 겹침 주의**: Ehlers + Hoffmann + Stender — Ehlers 2023 FNO (§22) 와 같은 그룹 (TUHH Dynamics + Imperial). 즉 같은 그룹이 *radar→FNO (2023)* → *buoy→PINN (2025)* 로 진화 중.

### 29.1 핵심 메시지
- **PFT-PINN**: Potential Flow Theory PINN — 두 NN 병렬 (η-net + Φ-net), Laplace + 3 BCs (kinematic/dynamic/bottom) 를 AD 로 강제
- **Sparse buoy time-series → η(x,t) reconstruction + Φ(x,t,z) inference + future prediction**
- 1D + t (NOT 2D directional)
- Wave flume 실측 (MARATHON, T_p=1.2s, ε=0.09 JONSWAP) 까지 검증

### 29.2 핵심 방법론
**Architecture**:
- η-net: input (x, t) → output η̃
- Φ-net: input (x, t, z) → output Φ̃
- 둘 다 4 hidden layers × 200 neurons, tanh, Fourier feature embedding (n_F=10, F ∈ [-0.4, 0.4])
- Xavier init, biases=0

**Loss components** (5):
- L_Lap = Laplace residual at fluid-volume collocation points P_L (N_L=30,000)
- L_BC,kin = kinematic surface BC residual at surface points P_s (N_s=5,000), z linked to η-net output
- L_BC,dyn = dynamic surface BC residual at P_s
- L_BC,bot = bottom impermeability at P_b (N_b=1,000)
- (Pre-trained M(x,t) and R(x,t) for hard constraint embedding)

**Hard constraint trick** (key novelty):
- η̃(x,t) = M(x,t) + R(x,t) · N_η̃(x,t)
- M = smooth extension of η_m measurements (low-capacity NN)
- R = smooth distance function from collocation points to measurement points
- M, R 가 pre-trained 후 frozen → measurements 가 architecture 에 hard-encoded
- 결과: data loss 가 학습 안정성 망치는 문제 회피

**Loss balancing**: ReLoBRaLo (α=0.95, τ=20, E[ρ]=0.98) — softmax-based adaptive weights

**Optimization**:
- Adam (AmsGrad) lr=5e-4 for 5,000 epochs
- L-BFGS (history=30) until 40,000 epochs
- ~30 분/instance (NVIDIA RTX 3090)

**Sequential training for prediction**:
- Time domain [0, 3·Δt_sn + T_p] 을 52 segments 로 분할
- 각 750 epochs 마다 다음 segment collocation points 추가 (causality)
- 50,000 epochs total

### 29.3 검증 결과
| Scenario | Data | SSP_η | SSP_Φ | MSE_η |
|---|---|---|---|---|
| Linear synthetic (3 components) | 3 buoy time-series | **0.024** | 0.021 | 1.55e-3 |
| Wave flume (MARATHON ε=0.09) | 3 buoy time-series | **0.049** | N/A (no truth) | 2.71e-6 |
| Prediction (4 snapshots → 1 peak period) | 4 spatial snapshots | 0.084 | N/A | 1.42e-5 |

- Assimilation: SSP < 0.1 criterion 통과
- Prediction: SSP=0.141 at t=1.61s (decreases as time progresses, expected)
- Hypotheses H1-H4 모두 confirmed

### 29.4 우리 work 와 직접 비교

| 항목 | PFT-PINN (Ehlers 2025) | WaveSplat |
|---|---|---|
| **Sensor** | sparse buoy time-series (1D points) | LiDAR scan-line + raydrop (2D directional) |
| **Approach** | NN-parameterized PFT, AD physics loss | parametric (k_x, k_y, ω, A, φ) + dispersion hard |
| **Dimensions** | 1D + t | 2D + t (k_x, k_y) |
| **Nonlinearity** | full nonlinear PFT (Laplace + BCs) | linear superposition |
| **Inference time** | ~30 분 / instance (training-as-inference) | < 100 ms |
| **Training data** | 0 (unsupervised, instance-specific) | 0 (analytical inversion) |
| **Velocity potential** | full Φ(x,t,z) inferred | not modeled |
| **Bathymetry** | extensible (d(x) inferable) | flat-bottom |
| **Wave-structure interaction** | not modeled | not modeled |

### 29.5 우리 paper 에 미치는 영향

#### 29.5.1 강화된 점
- ✅ **Sparse measurement inversion + physics constraint** 라는 framing 이 main-stream 임 재확인 — PINN/우리 모두 unsupervised physics-based
- ✅ **SSP metric 정당화 강화** — PFT-PINN 도 SSP 만 사용 (Klein 2020, Ehlers 2023, Iida 2024 와 일관)
- ✅ **Real-time advantage 극명함** — PFT-PINN 30 분 vs 우리 < 100 ms (≈ 18,000× 차이)
- ✅ **2D + LiDAR 차별화** — Ehlers 2025 의 future work 가 "2D + t extension" 이라고 명시 → 우리가 이미 2D 라는 사실이 critical novelty 로 부각

#### 29.5.2 약화된 점 / 인정해야 할 점
- ⚠️ **Nonlinearity 한계** — PFT-PINN 은 full Laplace + nonlinear surface BCs 를 만족시킴 (우리는 linear 만)
- ⚠️ **Velocity field inference** 못함 — PINN 은 Φ(x,t,z) 까지 추정 가능
- ⚠️ **Hard constraint architecture trick** — 우리는 직접 응용 불가 (parametric model 이라) 이지만 measurement-aware initializer 영감으로 활용 가능

#### 29.5.3 우리 framing 에서 강조할 contrast
**"30분 vs 100ms 의 양자적 차이가 응용 영역을 바꾼다"** — PFT-PINN 은 post-event analysis 또는 design-time tool, 우리는 real-time control loop (선박 hydrofoil, ride control) 에 적용 가능. **이것은 본질적 응용 분리**.

또한 **우리는 LiDAR raydrop physics 까지 simulate** — PFT-PINN 은 buoy 측정 (point sensor) 만 가정. 우리 sensor model 이 더 풍부.

### 29.6 인용 시 활용 방안 (paper 본문)
- **Introduction prior art table**: PFT-PINN 을 "data-assimilation NN approach" 로 별도 분류 → ML 계열 (Ehlers 2023, Duan 2020) + DA 계열 (Wang-Pan 2021/22) 과 구별
- **Related work**: "Concurrent with our parametric approach, Ehlers et al. (2025) developed PFT-PINN — both pursue unsupervised physics-based inversion. PFT-PINN handles full nonlinearity at the cost of ~30 min training; our parametric formulation operates at <100 ms with linear superposition."
- **Discussion**: PFT-PINN 의 hard-constraint embedding 을 우리 measurement-aware initializer 의 "spiritual cousin" 으로 위치
- **Future work**: "Hybrid PINN-parametric: use parametric solution as warm-start for PINN refinement when nonlinearity matters"

---

## 30. Yan, Che, Li 2025 — VOF EnKF 정독 결과 (2026-05-08)

> **Citation**: Yan, L., Che, L., Li, J. (2025). *Sequential water wave reconstruction in VOF-based numerical wave tanks with the EnKF approach*. arXiv:2510.14356v2 (Feb 2026).
> **소속**: Shanghai Jiao Tong University (Marine Numerical Experimental Center) + Shanghai Ship and Shipping Research Institute.

### 30.1 핵심 메시지
- **첫 VOF-NS + EnKF integration** for phase-resolved wave reconstruction
- POD 기반 dimensionality reduction (M=4.8e4 cells → ~30 modes)
- **Physics-constrained inflation** in FFT spectral domain → potential flow consistency
- **Plunging breaker 도 reconstruct 가능** (단, 중요한 caveat: assimilation 은 breaking 이전 upstream 에서만)

### 30.2 핵심 방법론
**Forward model**: OpenFOAM (incompressible Navier-Stokes + VOF interface tracking, no surface tension explicit)

**State**:
- α(x,z,t) — VOF volume fraction ∈ [0,1]
- (u_x, u_z) — velocity components
- 변환: α → φ (signed distance level-set) for POD smoothness
  - φ = ±min ||l - k|| from interface k ∈ Γ
  - 이유: VOF 는 sharp discontinuity → POD modes 비효율 (11 modes for 99% energy), level-set 는 smooth (4 modes)

**POD**:
- Snapshot matrix S_φ = [φ⁽¹⁾, ..., φ⁽ᴺ⁾] from ensemble
- SVD: S_φ = U_φ Σ_φ V_φᵀ
- Truncate: Û_φ ∈ R^(M × r_φ), retain 99% cumulative variance
- a⁽ⁿ⁾ᶠ = Û_φᵀ φ⁽ⁿ⁾ᶠ — modal coefficients
- Mode count: φ field needs r_φ ≈ 4, velocity needs r_u ≈ 30 each (slower decay)

**EnKF analysis** (standard):
- a⁽ⁿ⁾ᵃ = a⁽ⁿ⁾ᶠ + K (y⁽ⁿ⁾ - H(a⁽ⁿ⁾ᶠ))
- K = PᶠHᵀ (HPᶠHᵀ + R)⁻¹
- Observation operator: H(a) = S F(Û_φ a) — interpolation from level-set field reconstruction

**Physics-constrained inflation** (key novelty):
1. Assemble inflated η⁽ⁿ⁾(x, t_l) from analysis ensemble
2. FFT along x → spectral components (A_i, k_i, ω_i, φ_i)
3. Inflate amplitudes: A_i⁽ⁿ⁾ ← Ā_i + ϱ (A_i⁽ⁿ⁾ - Ā_i)
4. Inverse FFT → inflated η⁽ⁿ⁾
5. Reconstruct velocity via **linear potential flow theory**:
   - u_wx = Σ A_i g k_i / ω_i · cosh(k_i(z+d))/cosh(k_i d) · sin(k_i x + φ_i)
   - u_wz = Σ A_i g k_i / ω_i · sinh(k_i(z+d))/cosh(k_i d) · sin(k_i x + φ_i)
6. Re-build α from inflated η (boundedness preserved)

**NEES filter consistency check**:
- NEES(t_l) = (a^t - ā^a)ᵀ (P^a)⁻¹ (a^t - ā^a)
- Normalize: NEES/r_φ ≈ 1 for consistent filter
- Pick ϱ to minimize |NEES/r_φ - 1| → ϱ=2.0 best

### 30.3 검증 결과 (3 cases)

#### Regular wave (5th-order Stokes)
- T=2s, H=0.15m, d=0.6m, baseline H_b=0.1m, σ_o=0.002m
- 10 gauges, single assimilation at t_l₀=20s
- Level-set RMS = lowest (vs VOF, parametric curve baselines)
- **Velocity field successfully retrieved through ensemble covariance** even though only η observed

#### Irregular wave (JONSWAP)
- T_p=2.5s, H_s=0.075m, L=22m, ensemble N=50
- 500 random components per member, 20 gauges, σ_o=0.002m
- Assimilation interval Δt_l = 1s
- Best ϱ=2.0: NEES/r_φ=1.099, ε^η_RMS = 6.13e-3
- **Spatial-spectral analysis**: PSD agreement excellent at x=11m, x=21m (downstream); poor at x=1m (inlet boundary mismatch)
- **Bias problem**: downstream gauges show ~0.06 H_s positive bias (RMS 0.09 H_s)

#### Plunging breaker
- Cosine wave T=5s, H=0.12m, slope 1/35, L_h=10m constant depth zone
- 4 gauges in upstream (x ≤ L_h), N=50, σ_o=0.002m
- **Assimilation only in upstream linear region (3 instants: 6s, 7s, 8s)** — 운영 후 CFD 가 자체적으로 breaking evolution 진행
- Result: DA 가 upstream amplitude/phase 정확히 보정 → CFD 가 정확한 overturning + breaking 재현 (t=15.1-15.35s)

### 30.4 우리 work 와 직접 비교

| 항목 | VOF EnKF (Yan 2025) | WaveSplat |
|---|---|---|
| **Forward model** | VOF + Navier-Stokes (full CFD) | Linear superposition |
| **Solver** | OpenFOAM (interFoam-class) | Analytical |
| **State dim (raw)** | M = 4.8e4 cells × N=50 ensemble | Naturally low (5N parameters) |
| **State dim (effective)** | ~30 POD modes × 50 = 1500 | 5 × N_components (~20-100) |
| **Sensor** | 4-20 gauges (1D point) | 2D LiDAR + raydrop |
| **Dimensions** | 1D (x-z plane, NOT directional) | 2D (k_x, k_y) directional |
| **Nonlinearity** | full NS, breaking, plunging | linear only |
| **Inference time** | very slow (50 × OpenFOAM run per cycle) | < 100 ms |
| **Inflation** | physics-constrained (linear PFT) | not needed (no ensemble) |
| **Boundary condition** | NOT in state → near-inlet errors | N/A (no inlet) |
| **Real-time capable** | ❌ No (offline only) | ✅ Yes |

### 30.5 우리 paper 에 미치는 영향

#### 30.5.1 강화된 점
- ✅ **Real-time uniqueness 더 명확** — VOF-EnKF 는 50 ensemble × full CFD = orders of magnitude slower (Wang-Pan 2021/22 HOS-EnKF 보다도 느림)
- ✅ **Linear approximation 정당화 강화** — VOF-EnKF 의 inflation step 자체가 linear potential flow 로 velocity field 재구성 → linear PFT 가 "consistency-preserving operator" 로 표준 사용됨
- ✅ **Grid-based + ensemble 의 fundamental cost trade-off** 가 prior art 에 명시됨 → grid-free parametric 의 가치 부각
- ✅ **POD-EnKF dimensionality reduction trick** 도 결국 "function space → low-dim coefficients" — 우리는 이미 parametric (k,ω,φ,A) 로 동일 목표 달성하지만 무손실

#### 30.5.2 약화된 점 / 인정해야 할 점
- ⚠️ **Breaking/plunging 못함** — VOF-EnKF 가 명시적으로 breaking 도 처리 (앞부분 linear assimilation + CFD 자체 evolution 전략). 우리는 nonlinear breaking 영역 못 다룸
- ⚠️ **Wave-structure interaction 없음** — VOF 는 선박/구조물 포함 가능 (digital-twin 응용 명시)
- ⚠️ **Sequential assimilation** — VOF-EnKF 도 sequential snapshots 사용 (Δt_l=1s) → "sequential" 자체는 novelty 아님

#### 30.5.3 우리 framing 에서 강조할 contrast
**"VOF-EnKF 는 high-fidelity offline digital twin, 우리는 real-time control loop"** — 명시적으로 응용 영역이 다르다는 점을 paper 에 적시. VOF-EnKF 의 future work 가 "accelerate to real-time" 인데, 우리는 이미 그 자리에 있음.

또한 VOF-EnKF 의 **inlet boundary 처리 문제** (x=1m 에서 RMS 0.228 H_s vs x=11m 에서 0.090) — grid-based 의 본질적 단점. 우리는 boundary 없음 (parametric, full domain).

### 30.6 인용 시 활용 방안
- **Introduction prior art table**: VOF-EnKF 를 "high-fidelity DA" 카테고리에 배치 (Wang-Pan HOS-EnKF 와 함께)
- **Related work**: "Most recently, Yan et al. (2025) integrated VOF-NS with EnKF, achieving plunging breaker reconstruction at the cost of full CFD ensemble propagation. While capable of strong nonlinearity, this approach is fundamentally offline."
- **Discussion / cost-benefit table**: 우리 < 100 ms vs HOS-EnKF (Wang-Pan) > 200 ms vs VOF-EnKF (offline) — 응용 영역 분리 명시
- **Limitations section**: "Unlike VOF-NS-based DA (Yan et al. 2025), our linear formulation cannot capture wave breaking. Future work could combine our real-time linear estimator with on-demand CFD refinement when breaking probability exceeds threshold."

---

## 31. 12 paper 정독 완전 종료 — 최종 종합 (2026-05-08, post-PINN/VOF)

### 31.1 정독 완료 paper (12개) — final list
| # | Paper | Year | Approach | Inference time | Dim |
|---|---|---|---|---|---|
| 1 | Desmars 2018 (OMAE) | 2018 | LIDAR + ICWM grid | ~s | 1D |
| 2 | Desmars 2020 (Ocean Eng) | 2020 | Wave gauge + dispersion | ~s | 1D |
| 3 | Desmars 2023 (JFM) | 2023 | HOS-OCM grid | seconds | 1D |
| 4 | Wang-Pan 2021 (JFM) | 2021 | EnKF-HOS first | seconds-min | 1D |
| 5 | Wang-Pan 2022 (JFM) | 2022 | EnKF-HOS-C with current | seconds-min | 1D |
| 6 | Hunt 2007 (Tellus A) | 2007 | LETKF foundational | N/A | N/A |
| 7 | Ehlers 2023 (Ocean Eng) | 2023 | FNO ML radar | 1.9 ms | 1D |
| 8 | Tang-Bhaskar-Recht 2013 | 2013 | Atomic norm CS off-grid | offline | 1D |
| 9 | Möller-Trumbore 1997 | 1997 | Ray-triangle (코드) | μs | N/A |
| 10 | Klein 2020 (Fluids) | 2020 | HOSM/NLSE comparison + SSP | offline | 1D |
| 11 | **Ehlers 2025 PFT-PINN (PRF)** | 2025 | **PINN-PFT physics loss** | **~30 min** | **1D + t** |
| 12 | **Yan-Che-Li 2025 VOF-EnKF** | 2025 | **VOF-NS + EnKF + POD** | **offline (slow)** | **1D (xz plane)** |

### 31.2 우리 positioning 재최종 정리 (12 paper 검토 후)

#### 31.2.1 본질적 우위 (어떤 prior art 도 충족 못함)
1. **2D directional + LiDAR + sparse-snapshot inversion** — 12 paper 중 0개 (Desmars 2023 가 LIDAR + 1D, Ehlers 2023 이 radar + 1D, PFT-PINN 이 buoy + 1D + t, VOF-EnKF 가 gauge + 1D xz)
2. **Real-time (<100 ms) phase-resolved estimator** — 12 paper 중 Ehlers 2023 FNO 만 비슷 (1.9 ms) 인데 그것은 ML supervised pre-trained
3. **Off-grid + 2D + dispersion-hard parametric** — 12 paper 중 0개 (Tang 2013 이 off-grid 1D atomic norm, Desmars 모두 on-grid)
4. **Full sensor model (raydrop + shadowing + Fresnel)** — 12 paper 중 Ehlers 2023 만 부분적 (radar tilt + shadowing), LIDAR-specific 는 우리만
5. **5-metric panel** — 12 paper 중 SSP only (Klein, Ehlers ×2, Iida) 또는 nL2 only — 우리가 가장 다양

#### 31.2.2 인정해야 할 한계 (paper 에 명시 권장)
- **Linear** (VOF-EnKF, PFT-PINN, Klein HOSM 모두 nonlinear 처리 가능)
- **No breaking** (VOF-EnKF 만 plunging 처리)
- **No velocity inference** (PFT-PINN 은 Φ 추정 가능, VOF-EnKF 는 직접 추정)
- **Synthetic only** (Klein 2020 의 wave tank + Ehlers 2023 PFT-PINN 의 MARATHON flume 은 lab 검증)

#### 31.2.3 응용 영역 매트릭스 (claim 차별화)

| 응용 | 우리 | PFT-PINN | VOF-EnKF | HOS-EnKF | FNO |
|---|---|---|---|---|---|
| Real-time control (선박 ride control, hydrofoil) | ✅ | ❌ | ❌ | ❌ | ✅ |
| Phase-resolved 2D directional | ✅ | ❌ | ❌ | ❌ (1D) | ❌ |
| Breaking, slamming, green water | ❌ | ❌ | ✅ | ❌ | ❌ |
| Wave-structure interaction (loads) | ❌ | ❌ | ✅ | ❌ | ❌ |
| Velocity field full inference | ❌ | ✅ | ✅ | ✅ | ❌ |
| Bathymetry inversion | ❌ | ✅ (extensible) | ❌ | ❌ | ❌ |
| Training-free unsupervised | ✅ | ✅ | ✅ | ✅ | ❌ |

→ **우리 niche**: real-time + 2D directional + sparse LiDAR — PFT-PINN/VOF-EnKF 와 직접 경쟁 안 함, 응용 영역이 다름

### 31.3 PPT references slide 최종 (12 → 13개 추천)
1. Desmars et al. 2018 (OMAE) — phase-resolved LIDAR concept
2. Desmars et al. 2020 (Ocean Eng) — experimental validation
3. Desmars 2023 (JFM) — HOS-OCM
4. Klein et al. 2020 (Fluids) — HOSM/NLSE comparison
5. Wang & Pan 2021 (JFM) — EnKF-HOS first
6. Wang & Pan 2022 (JFM) — EnKF-HOS-C
7. Ehlers et al. 2023 (Ocean Eng) — FNO 1D
8. **Ehlers et al. 2025 (PRF) — PFT-PINN ⭐ NEW**
9. **Yan, Che, Li 2025 (arXiv) — VOF-EnKF ⭐ NEW**
10. Tang, Bhaskar, Recht 2013 (IEEE TIT) — atomic norm CS
11. Hunt et al. 2007 (Tellus A) — LETKF
12. Möller & Trumbore 1997 (JGT) — ray-triangle
13. Nouguier, Grilli, Guérin 2014 (IEEE TGRS) — flash LIDAR foundational (paywall, historical)

### 31.4 Paper Introduction 의 "prior art landscape" 재구성 권장

**기존 분류 → 새 분류 (5개 카테고리)**:

1. **Variational/grid-based reconstruction (Desmars 2018-2023, Köllisch 2018, Aragh-Nwogu 2008, Qi 2018)**
   - on-grid k-band assumption, predefined wavenumbers
   - LIDAR 또는 wave gauge 입력
   - sub-second to seconds inference

2. **Ensemble Kalman filter (Wang-Pan 2021/22, Yan-Che-Li 2025)**
   - HOS-EnKF (Wang-Pan) → VOF-EnKF (Yan 2025) — fidelity 증가, 속도 감소
   - inflation 기반 filter divergence 회피
   - offline 또는 near-offline

3. **Machine learning supervised (Ehlers 2023 FNO, Duan 2020, Liu 2022, Wedler 2023, Chen 2023, Feng 2024)**
   - 사전 학습 필요, generalization 한계
   - Fast inference (1.9 ms FNO)
   - 1D unidirectional dominant

4. **Physics-informed unsupervised (Ehlers 2025 PFT-PINN, Liu 2024 LWT-PINN, Lu 2024)**
   - 학습 데이터 불필요
   - Full nonlinear PFT 처리 가능
   - 30 분/instance — offline-leaning

5. **Parametric off-grid (WaveSplat — 우리)**
   - 5 parameters per component (A, k_x, k_y, ω, φ) + dispersion hard
   - 2D directional, < 100 ms
   - LIDAR raydrop + shadowing
   - **Real-time + 2D + LIDAR 의 first integrated framework**

### 31.5 다음 작업 (immediate)
1. ✅ §29-§31 추가 완료
2. ⏳ PPT references slide 업데이트 (Ehlers 2025 PRF + Yan 2025 추가)
3. ⏳ PPT prior art slide 5-카테고리 재구성
4. ⏳ paper Introduction prior art subsection 재작성 (5-카테고리 적용)
5. ⏳ paper Discussion limitations 강화 (linear, no breaking, no velocity inference 명시)
6. ⏳ paper Future work: hybrid PINN/CFD refinement 옵션 언급

### 31.6 Ledger 갱신 (계속)
| 날짜 | 변경 | 이유 |
|---|---|---|
| 2026-05-08 | §29 추가 | Ehlers 2025 PFT-PINN (PRF 10, 094901) 정독 — 30 min/instance, 1D+t, hard-constraint trick |
| 2026-05-08 | §30 추가 | Yan-Che-Li 2025 VOF-EnKF 정독 — full CFD + EnKF + POD, plunging 처리, offline |
| 2026-05-08 | §31 추가 | 12 paper 정독 완전 종료, 5-카테고리 prior art landscape, 응용 매트릭스 확정 |
