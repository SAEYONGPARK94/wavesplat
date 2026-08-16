# E1 — 방법론/알고리즘 축 탐색 (2026-08-15)

**역할**: novelty 가 *실재하는* 방법론 방향 탐색. READ-ONLY (이 파일 외 수정 0). 하위 에이전트 0.
**문헌 검색**: `WebSearch`/`WebFetch` 로 12 회 수행 (전부 수행됨, "미확인" 항목은 아래에 명시).
**추정기 런**: 0 회. 아래 §1 의 수치는 **저장된 관측점 위 numpy 계산**이다.

---

## 0. 한 문단 요약

배정된 네 축(**①분산-다양체 ANM ②이동 궤적 off-grid ③synthetic aperture ④bound-wave 사전**)은
**네 개 다 죽었다** — 셋은 저장소가 **이미 측정해서** 죽였고(재발견), 하나는 **개입 지점이 없다**
(`ship_speed_kts` 는 dead config). 대신 탐색 중 **런 0회 측정으로 새 축 하나가 나왔다**:
**저장소의 지지집합 탐색이 강제하는 분리 문턱(`MIN_DK=0.15`)이 등방(isotropic)인데,
실제 관측 기하의 분해능은 2.4–2.8배 비등방**이다. 이 하나가 저장소에 기록된 **미해명 이상현상**
(`MIN_DK` 를 0.05 로 내리면 혼잡 5/5 개선 · 중밀도 0/5 악화)의 **양쪽 부호를 동시에 예측한다.**

---

## 1. ★★★ MEASURED — 오늘 새로 잰 것 (런 0회, 저장된 관측점)

### 1.1 k-공간 분해능 셀은 비등방이다 (2.4–2.8배)

출하 관측 기하(정지 센서, 49점/프레임, 301프레임, T=30 s)에서 **정합필터 응답**
`A(k) = |Σ_i exp(i[k·x_i − ω(k)t_i]) · exp(−i θ_i^true)| / N` 을 **참 성분의 반경(radial)·방위(azimuthal)
축을 따라** 스캔하고 주엽 HWHM 을 측정했다. 스크립트는 §6 에 전문 수록(재현 가능).

| 에피소드 | kp [rad/m] | HWHM_반경 | HWHM_방위 | **비등방비** | MIN_DK/HWHM_반경 |
|---|---|---|---|---|---|
| BF4_15comp_seed48 (S9 혼잡) | 0.089 | 0.0227 | 0.0625 | **2.75** | 6.6 |
| BF4_15comp_seed49 (S9 혼잡) | 0.078 | 0.0208 | 0.0559 | **2.69** | 7.2 |
| BF4_2comp_seed48 (S9 희소) | 0.242 | 0.0370 | 0.0893 | **2.42** | 4.1 |
| BF4_2comp_seed49 (S9 희소) | 0.229 | 0.0350 | 0.0894 | **2.55** | 4.3 |
| BF4_15comp_seed0_slope (출하) | 0.096 | 0.0218 | 0.0535 | **2.45** | 6.9 |
| BF4_15comp_seed0_uniform (출하) | 0.096 | 0.0236 | 0.0622 | **2.64** | 6.4 |

**전 6/6 에피소드, 두 레짐 · 두 raydrop 모델 · 두 코퍼스에서 방향 일치.**

**기제(런 0회 산수, 위 측정과 정합)**: 반경 분해능은 **분산관계 + 기록 길이**가 산다
(Δω = 2π/T = 0.21 rad/s, c_g ≈ 5.5 m/s ⟹ Δ|k| ≈ 0.038). 방위 분해능은 **발자국 개구**만 산다
(측정된 5–95 pct 개구 = x 27.9 m × y 42.5 m ⟹ Δk ≈ 0.15–0.23).
⟹ **시간축이 |k| 를 사고, 개구가 방향을 산다. 둘의 환율이 다르다.**

### 1.2 출하 상수 `MIN_DK=0.15` 는 방위 한계에 맞춰져 있고 반경에 등방으로 부과된다

Rayleigh 환산(2·HWHM): **반경 한계 0.042–0.074 · 방위 한계 0.107–0.179.**
`MIN_DK = 0.15` (`wavesplat/optimizer/initializer.py:63`) 는 **방위 한계와 거의 같고 반경 한계의 2–3.5배**다.
그리고 억제는 **정사각 상자**로 부과된다 (`initializer.py:362-363`, `:552-553`,
`optimizer/learned_init.py:257`: `supx = int(MIN_DK/dkx)+1; supy = int(MIN_DK/dky)+1`).
저장소 어디에도 비등방·방향 의존 처리가 **없다** (`optimizer/`·`model/` grep, 0 hit).

### 1.3 ★★★ 이 기제가 저장소의 미해명 이상현상 양쪽 부호를 예측한다

`initializer.py:70-73` 주석(=`tmp/MIN_DK_RESULT_2026-07-23.md`)이 스스로 적어 놓은 것:

> *"at 0.05 the crowded regime improves 5/5 but the mid-density regime degrades 0/5 via duplicate peaks"*

비등방 셀 가설의 예측:
- 0.05 > 반경 한계 0.042 ⟹ **반경 방향으로는 진짜 분해능이 풀린다** ⟹ 혼잡 개선 (관측: 5/5 ✔)
- 0.05 ≪ 방위 한계 0.107–0.179 ⟹ **방위 방향으로 유령/중복 봉우리** ⟹ 중밀도 악화 (관측: 0/5 ✔)

저장소에 게시된 다른 어떤 설명도 **두 부호를 동시에** 설명하지 못한다. (기존 설명은
"static_suppress 라 지도가 참 성분을 아직 갖고 있어서 재선택" — 이건 악화만 설명하고 개선을 설명 못 한다.)

### 1.4 여유 정량 — 지금 억제되는 참 쌍의 71% 는 자기 방향에서는 분해 가능하다

12 에피소드(S9 혼잡 6 · S9 희소 3 · 출하 30comp 3), 참 성분 쌍 **1,938개**:

| 기준 | 차단되는 쌍 |
|---|---|
| 등방 `MIN_DK = 0.15` (출하) | **450 (23.2%)** |
| 기하 정합 타원 (반경 0.045 · 방위 0.125) | **130 (6.7%)** |
| **⟹ 풀리는 쌍** | **320 = 현재 차단분의 71.1%** |

30comp 에서 특히 크다(101/435 → 32/435). 희소 2comp 는 0 → 0 (예상대로 무영향, 음성 대조로 쓸 수 있음).

⚠ **이건 "이득"이 아니라 "여유"다.** 차단 해제가 곧 필드 오차 개선이라는 보장은 없다
(S5 가 측정했듯 λ shrinkage 가 그 이득을 먹을 수 있다 — 그래서 아래 C2 가 C1 의 **선행조건**이다).

---

## 2. 후보 3개

### ★ C1 — 관측 연산자에 정합된 **비등방 분리 셀**

**한 문장**: 출하 지지집합 탐색은 실제 분해능이 2.4–2.8배 비등방인 k-공간에 **등방 분리 문턱**을
부과하고 있고, 셀을 기하에 맞추면 현재 억제되는 참 성분 쌍의 71% 가 풀린다.

**선점 여부 (문헌 검색 완료)**
- **일반 원리는 선점됨**: 지지집합 회복이 **연산자가 유도하는 계량**에 지배된다는 것은
  Duval & Peyré, *Exact Support Recovery for Sparse Spikes Deconvolution*, FoCM **15(5) 1315–1355 (2015)**
  (arXiv:1306.6909) 와 Candès–Fernandez-Granda (2014) 의 분리조건.
- **해양 인스턴스는 미발견**: "phase-resolved wave reconstruction 을 다차원 line-spectral estimation 으로
  놓고 분산관계가 유도하는 **비등방 분리조건**을 쓴 논문"은 **검색으로 찾지 못했다**
  (검색어 4종: line spectral estimation + dispersion / ANM manifold / phase-resolved + LSE + lidar /
  dispersion-constrained CS). 가장 가까운 것은 **Jiang·Kalisch·Benoit·Holand·Sprenger,
  arXiv:2605.25074 (2026-05)** — 부이 3성분 **격자 기반** CS + group sparsity + 에너지 제약이고
  **분산관계를 원자 사전 제약으로 쓰지 않으며 희소 공간표본·이동센서를 다루지 않는다.**
- ⟹ **정직한 novelty 등급 = "알려진 원리의 새 인스턴스 + 진단".** ML 학회급 아님.
  **Ocean Engineering / JAOT 급 방법-측정 기여**이고, 자산은 *알고리즘*이 아니라 *§1.3 의 기제 설명*이다.

**킬러 4개**

| 킬러 | 판정 | 근거 |
|---|---|---|
| ① 개입 지점이 코드에 실재 | **PASS** | `initializer.py:63`(상수) · `:357-363` · `:505-515` · `:552-553` · `learned_init.py:257`. 스칼라 1개 → 타원 2개 파라미터. |
| ② 공정한 기준선 | **PASS** | 기준선 = 출하 `MIN_DK=0.15` 그 자체 + **이미 측정된 등방 0.05 팔**이 요인정합 대조(=M2 의 BRV 를 만족). |
| ③ 표현 아티팩트 아님 | **PARTIAL** ⚠ | 합성 GT 가 분산정합 평면파 합이라 **지지집합 이득은 모델클래스 안**에서 측정된다. **완화**: 비등방은 *관측 연산자*의 성질이지 GT 의 성질이 아니고(§1.1 은 GT 진폭을 안 씀), HOS 코퍼스와 실측 13레코드가 off-model 확인 경로로 존재. **PREREG 에 명시할 것.** |
| ④ `direction_deg=0` 반인과 기하 회피 | **PASS** | 비등방은 발자국 기하 × T 의 함수이고 조우속도와 무관. `180°` 에서는 **강해질 것으로 예측**(위상 회전이 늘면 반경 분해능이 더 좋아진다) — 죽지 않는다. |

**사전등록 가능한 사망조건 (3개, 결과 전 동결)**
1. **PRIMARY**: 신선 혼잡 시드 n=32, `ELLIPSE` 팔 vs 출하 `MIN_DK=0.15` 팔, 단측 부호검정,
   문턱 **≥21/32**(저장소 기존 게이트 규약). 미달 ⟹ **DEAD**.
2. **기제 사망조건**: `ELLIPSE` 의 이득이 **방위 한계 하나로 맞춘 등방 반지름**(1-노브 대조)의 이득을
   초과하지 못하면 **비등방이 기제가 아니다 ⟹ DEAD** (이득이 있어도 이름을 못 산다).
3. **거래 검사**: 혼잡 통과 + 희소 악화면 `CROWDED-ONLY` 로 인쇄하고 **배포 주장 금지**
   (`SUPPORT-TRADEOFF` 8번째 재현 가능성). 근/원 셸 분해 **필수**.
   ⚠ 희소 2comp 는 §1.4 에서 차단 쌍 0 ⟹ **희소는 사전에 널을 예측**한다. 널이 나오면 그건
   반증이 아니라 **양성 대조 통과**다 — PREREG 에 이 방향성을 미리 못박아야 한다.

**최소 실험과 비용**
- 신규 계산: 3팔(SHIP / ELLIPSE / ISO-0.05 재현) × 2레짐 × 32시드 = **192 런**.
  S7(1,152런)·S13(704런) 대비 소규모 ⟹ 기존 배치 속도로 **수 시간**.
- 데이터 생성 **0회**(기존 `data_s9_*` + 신선 시드는 잔여 [1584–1818] 에서).
- §1 의 특성화는 **이미 끝났다**(런 0회, 위 표).
- 코드 변경: 억제 상자를 타원으로 = 4개 사이트, ~10줄. **단일 출처 상수로 유도**할 것
  (n-상수 하드코딩 재발 3회 전례).

---

### ★ C2 — VARPRO + 비벌점 재적합 (**C1 의 선행조건**, novelty 는 0)

**한 문장**: 지지집합이 주어지면 남은 문제는 닫힌형 볼록 LS 인데(오늘 측정: 전역 ρ=0.999964),
출하 추정기는 극좌표 `(w,φ)` 에서 1차 최적화기 + L1 벌점을 돌려 자기 목적함수 기준 **72배** 위에 선다.

**선점 여부**: **완전 선점, 방법 novelty 0.** VARPRO = Golub & Pereyra (1973), 리뷰
*Inverse Problems* **19 (2003) R1**. post-lasso 재적합 = Belloni–Chernozhukov (2013).
⟹ ⛔**"새 방법"으로 절대 팔지 말 것.**
단 **미청구 응용은 있다** — VARPRO 를 phase-resolved 해양파 재구성에 쓴 논문을 검색으로 찾지 못했고
(HOR: Qi·Wu·Liu·Yue, *JFM* 2018; Desmars, *JFM* **975 A8** 2023 은 진폭·위상을 **동시** 최적화),
"분리가능 구조를 안 쓰고 있다"는 지적은 문헌에 **한 문단**어치 값은 있다.

**킬러**: ①PASS(`model/spectral_gaussian.py:118-124` 가 이미 `(a,b)` 에 선형;
`optimizer/sequential_update.py`·`coarse_to_fine.py` 가 개입 지점) ②PASS ③PASS(솔버 교체라 GT 무관)
④PASS.

**사망조건**: 오늘 측정된 오라클-k 격차의 **≥50%** 를 혼잡에서 닫지 못하면 *"벽은 위상 하위공간의
최적화 실패"* 척추가 **자기 처방을 못 산다** ⟹ 척추 재검토.

**비용**: 낮음. **이건 08-15 토론의 P0/P1 그 자체**이고 몇 주째 미실행이다.
**C1 보다 먼저 해야 한다** — S5 가 *"정규화항이 해를 밀어낸다(오라클 팔 오차의 84%)"* 를 이미
측정했으므로, λ 를 안 고치고 C1 을 재면 C1 의 이득이 shrinkage 에 먹혀 **거짓 널**이 나온다.

---

### ▽ C3 — 표본 선택(shadowing/specular)을 우도에 넣기 (**권고: 지금은 하지 마라**)

**한 문장**: LiDAR 반환은 MNAR 이다 — `wavesim/raydrop_slope.py` 는 정반사 정렬이 성립하는
**특정 경사에서만** 반환을 내주는데 추정기는 모든 반환을 등가중한다.

**선점**: X-band 레이더 shadowing modulation 은 광범위하게 연구됨(Hs 를 shadowing 으로 추정하는 계열).
LiDAR + 격자없음 조합은 덜 다뤄졌으나 **프레이밍이 새롭지 않다.**

**킬러**: ①**FAIL 쪽** — `noise_model.compute_confidence` 는 있으나 **기여 #3(신뢰 가중치)은
DEAD 확정**(S13). 선택모형은 신규 코드가 필요. ②PASS ③**FAIL 위험 큼** — 선택 기제를
시뮬레이터가 만들었으므로 시뮬 안에서만 검증한 "수리"는 **시뮬레이터 역산 아티팩트**가 된다.
④PASS.

**판정**: C1·C2 가 둘 다 착지하기 전에는 **투자 금지.**

---

## 3. ⛔ 버린 후보와 이유 (다음 사람이 같은 길을 안 가게)

### D1 — **분산-다양체 gridless atomic norm / BLASSO** (배정 항목 ①) = **삼중 사망**

1. **이론 선점**: 임의 파라미터화 원자집합의 atomic norm 일반론 = **Chandrasekaran·Recht·Parrilo·Willsky,
   FoCM 2012**. 가장 가까운 동형 = **Yang & Xi, arXiv:2503.15164 (2025-03),
   "Gridless Chirp Parameter Retrieval via Constrained Two-Dimensional Atomic Norm Minimization"** —
   *2D 주파수 공간의 곡선 궤적 위로 원자를 구속*하고 근사 SDP + **dual polynomial 최적성 증명서**까지
   제시한다. 해양 분산 버전은 **corollary**. (이 판정은 `project_novelty_rereview_4agent_2026-07-23`
   Round 3 가 이미 내렸고, 오늘 원논문 초록으로 **재확인**했다.)
2. **저장소 경험 선점 = 이득 0**: ANM 게이트가 **이미 실행됐다** —
   `wavesplat/tmp/blasso_feasibility.py` (2026-07-23). sliding-Frank-Wolfe vs greedy+joint-refine:
   5comp **0.0747 vs 0.0748**, 15comp **0.110 vs 0.109** ⟹ **필드 nRMSE 개선 0.**
   naive greedy 대비 이득은 **전부 joint-refine 몫이고 출하 파이프라인이 이미 그걸 한다.**
3. **킬러 ③ 정면 위반**: 합성 GT 가 **분산정합 평면파의 합**이다 ⟹ 다양체 구속 사전은 정답을
   받아 들고 시작한다. `−93% 희소` 가 표현 아티팩트로 판정난 것과 **같은 함정**.

⚠ **정직한 잔여 1건 (누구도 답 안 함, 그러나 이 학위논문의 일이 아님)**:
`ω = √(g|k|tanh(|k|d))` 는 **초월함수**라 Yang–Xi 식 대수적 궤적 SDP 구성이 **적용되지 않는다.**
"비대수적 다양체 구속 ANM 의 SDP 특성화"는 진짜 열린 수학 문제로 보이나, **(2)가 상금이 0 이라고
이미 측정했으므로** 그 문제를 풀어도 이 저장소에서 살 것이 없다. 신호처리 이론 논문 소재.

### D2 — **이동 궤적 위 희소 표본 off-grid 복원** (배정 항목 ②) = **개입 지점 부재로 사망**

★**`ship_speed_kts` 는 dead config 다.** `wavesim/config.py:76` 에 정의돼 있고 **어디서도 읽히지 않는다**
(grep 전수). `wavesim/pipeline.py` 는 시간 루프 **바깥**에서 ray 를 1회 생성하고 ship_x=ship_y=0 로 고정.
⟹ **출하 110 에피소드와 모든 S-배치 코퍼스는 정지 센서다.** 시뮬레이터 신규 코드 + 데이터 전량 재생성
없이는 **잴 수 없다** = 킬러 ① FAIL.
프레이밍도 선점: 이동-프로브 predictable zone (Qi·Wu·Liu·Yue 2018), 비균일 gridless 는 STAP 계열의
성숙 분야(reweighted-ANM STAP, arXiv:2004.04559).

### D3 — **synthetic aperture / 시간이 파수를 산다** (배정 항목 ③) = **이미 소진 + 질문 자체가 오답**

★**"synthetic aperture 와 같은가?" 의 답 = 다르다. 그리고 저장소가 이미 쟀다.**
Synthetic aperture 는 플랫폼 **병진**이 필요하다. 여기 센서는 **정지**인데도 T 가 늘면 분해능이 는다 —
분산관계가 ω 를 |k| 에 묶기 때문이다. 이건 **개구 합성이 아니라 분산 기반 반경 분해능**이고,
**§1.1 이 그걸 정량화한 것이다**(T=30 s → 반경 0.021 vs 43 m 발자국 → 방위 0.062).
이동 버전은 7월에 실측으로 완주해서 죽었다: `D_eff/λ` 보편법칙 **기각**(10 레코드, 0.7λ 위 6개 음수),
"해로움=정보" 해석 2회 번복, 최종 승자는 **식별을 전혀 안 하는 정규화 선형 MMSE**
(`project_synthetic_aperture_2026-07-03`). **재방문 금지.**

### D4 — **bound-wave 인지 희소 복원** (배정 항목 ④) = **삼중 사망**

1. **코드가 이미 있고 널이었다**: `wavesplat/model/stokes2.py` 가 Sharma & Dean (1981) 유한수심
   2차 전달함수를 구현하고 `spectral_gaussian.py:45,87-95,130-131` 에 `use_stokes2` 로 배선돼 있다.
   저장소 프로브 측정: **0.97708 → 0.97711** (널). 기록: *"Nonlinearity is not a learning opportunity"*.
2. **문헌 선점**: HOR 이 이미 bound harmonic 을 추정·제거하고 자유파 진폭·위상을 최적화한다
   (Qi·Wu·Liu·Yue, *JFM* 2018 "Nonlinear phase-resolved reconstruction of irregular water waves";
   2차 sum/difference bound 성분을 **정확 2차 이론**으로 계산해 검증). 비선형 축 삼중 선점 확정
   (Desmars *JFM* 975 A8 2023 · Ehlers *Ocean Eng.* 288 2023 = arXiv:2305.11913 · Ehlers PINO arXiv:2508.03315).
3. ★**브랜치 이름이 가리키는 과거 시도는 이미 끝났고 음성이다** — `learn-boundwave-retrain` 의
   커밋 `099e627` ("v8bw 체크포인트 — bound-wave bank task-loss FT") → `e7522d2`
   ("**v8bw 실측 평가 — 재학습이 zero-shot 못 이김 (음성)**"). 브랜치 이름은 **살아있는 계획이 아니라
   닫힌 음성 아크의 화석**이다.

### D5 — 부수 확인 (선점 논문 3건의 정체)

- **Katende, arXiv:2606.07153** = *"No-Harm Physics-Informed Inverse Learning with Residual-Calibrated
  Uncertainty"* — 잔차 기반 **증명서/선택** 프레임워크. ⟹ **GT-free 선택기**를 선점하지, 추정기를 선점하지 않는다.
- **Sprunck·Pereyra·Liaudat, arXiv:2510.27663** = *"Bayesian model selection and misspecification testing
  in imaging inverse problems only from noisy and partial measurements"* — 역시 **GT-free 모형선택**.
  파랑 재구성 논문이 **아니다**. ⟹ 이 둘은 기여 "held-out 신호/선택기" 를 선점하고 **방법 축은 안 건드린다.**
- **FluidSplat, arXiv:2605.18866** = 희소 센서에서 K 개 **비등방 Gaussian primitive** 예측(partition-of-unity).
  ⟹ **"Gaussian splatting 이름 공간"을 선점**하지 분산관계 물리를 선점하지 않는다.
  (저장소는 이미 *"Gaussian Splatting 이름 = cosmetic"* 으로 기각해 뒀다 — 그 기각이 옳았음이 재확인됨.)

---

## 4. MEASURED vs ARGUED 분리

| 진술 | 등급 |
|---|---|
| 정합필터 주엽이 방위/반경 2.42–2.75배 비등방 (6 에피소드, 2 레짐, 2 raydrop) | **MEASURED** (런 0, §6 스크립트) |
| `MIN_DK=0.15` 가 반경 HWHM 의 4.1–7.2배 | **MEASURED** |
| 참 쌍 1,938 중 450 차단 → 타원 시 130 (320 해제, 71.1%) | **MEASURED** |
| 발자국 5–95 pct 개구 = 27.9 × 42.5 m, T=30 s, 49 pts/frame | **MEASURED** |
| `ship_speed_kts` 미사용 dead config | **MEASURED** (grep 전수) |
| `stokes2.py` 존재·배선됨 | **MEASURED** (코드 직독) |
| 비등방 셀이 `MIN_DK` 0.05 이상현상의 **원인**이다 | **ARGUED** — 부호 양쪽을 예측하나 대안설명 배제 안 됨. C1 사망조건 #2 가 이걸 시험한다. |
| 셀을 고치면 필드 nRMSE 가 개선된다 | **ARGUED** — §1.4 는 **여유**이지 이득이 아니다. |
| ANM 이 이 문제에서 상금 0 | **MEASURED**(2026-07-23 게이트) — 단 λ=1e-3 시대 · 4에피소드 · 단발 적합 |
| 비대수 다양체 ANM 의 SDP 부재 | **ARGUED** (Yang–Xi 초록만 확인, 본문 미독) |

## 5. ⚠ 내 탐색의 한계 (열람 범위 = 주장 범위)

- 읽은 코드: `wavesim/pipeline.py`(1–80) · `wavesim/config.py:76` · `wavesim/raydrop_slope.py`(1–40) ·
  `wavesplat/optimizer/initializer.py`(47–95, grep 전역) · `cs_initializer.py`(1–30) ·
  `wavesplat/model/stokes2.py`(1–50) · `tmp/blasso_feasibility.py`(1–60, 221행 시그니처).
  **`sequential_update.py`(1,589행) 와 `coarse_to_fine.py` 는 안 열었다** ⟹ C2 의 개입 지점은
  기존 메모 인용이지 내 직독이 아니다.
- 문헌: 초록 수준. **Yang–Xi 2503.15164 본문 미독** ⟹ "비대수 궤적엔 SDP 없음" 은 추론이다.
- ANM 게이트(2026-07-23)는 **λ=1e-3 배포 이전**의 측정이다. λ=1e-4 배포 후 재측정된 적 **없다**.
  ⟹ ⛔"ANM 상금 0" 을 **현재 구성에서** 재확인하려면 8 런이 필요하다(추천하진 않는다 — D1 의
  사망 근거는 (1)(3) 만으로도 충분하다).
- 실측 13레코드·HOS 경로에서는 **아무것도 새로 재지 않았다.**

## 6. 재현 스크립트 (스크래치패드 소실 방지를 위해 전문 수록)

### 6.1 비등방 측정 (§1.1)

```python
"""cd wavesplat/ && python3 <this>   — 런 0회, 저장된 관측점만 사용"""
import h5py, glob, numpy as np
G = 9.81

def omega(kx, ky, d):
    km = np.hypot(kx, ky)
    return np.sqrt(G*km*np.tanh(km*d) + 1e-12)

def load(fn, T_use=30.0, stride=1):
    with h5py.File(fn,'r') as f:
        lp=f['frames/lidar_points']; vm=f['frames/lidar_valid_mask'][:]
        ts=f['frames/timestamps'][:]; d=float(f['config'].attrs['depth'])
        wc=f['config/wave_components'][:]
        X=[];Y=[];T=[]
        for i in range(0,len(ts),stride):
            if ts[i]>T_use: break
            p=lp[i][vm[i]]; X.append(p[:,0]); Y.append(p[:,1]); T.append(np.full(len(p),ts[i]))
    return np.concatenate(X),np.concatenate(Y),np.concatenate(T),d,wc

def ambiguity(x,y,t,d,kx0,ky0,nr=161,extent=0.30):
    k0=np.hypot(kx0,ky0); u=np.array([kx0,ky0])/k0
    v=np.array([-u[1],u[0]])
    ref=np.exp(-1j*(kx0*x+ky0*y-omega(kx0,ky0,d)*t))
    def scan(vec):
        s=np.linspace(-extent,extent,nr); out=np.empty(nr)
        for j,dd in enumerate(s):
            kx=kx0+dd*vec[0]; ky=ky0+dd*vec[1]
            ph=kx*x+ky*y-omega(kx,ky,d)*t
            out[j]=abs(np.sum(np.exp(1j*ph)*ref))/len(x)
        return s,out
    return scan(u),scan(v)

def hwhm(s,a):
    c=np.argmin(abs(s)); pk=a[c]
    r=c
    while r+1<len(a) and a[r+1]<a[r] and a[r]>0.5*pk: r+=1
    l=c
    while l-1>=0 and a[l-1]<a[l] and a[l]>0.5*pk: l-=1
    def cross(i0,i1):
        if a[i1]>=0.5*pk: return abs(s[i1])
        f=(a[i0]-0.5*pk)/(a[i0]-a[i1]+1e-12)
        return abs(s[i0]+f*(s[i1]-s[i0]))
    return 0.5*(cross(c,min(r,len(a)-1))+cross(c,max(l,0)))

files=[]
for pat in ['data_s9_crowded_2026-08-11/*.h5','data_s9_sparse_2026-08-11/*.h5','data/episode_BF4_*.h5']:
    files+=sorted(glob.glob(pat))[:2]
for fn in files:
    x,y,t,d,wc=load(fn); i=int(np.argmax(wc[:,0]))
    kx0,ky0=float(wc[i,2]),float(wc[i,3]); k0=np.hypot(kx0,ky0)
    (sr,ar),(sa,aa)=ambiguity(x,y,t,d,kx0,ky0)
    hr,ha=hwhm(sr,ar),hwhm(sa,aa)
    print(f"{fn.split('/')[-1][:52]:52s} kp {k0:6.3f} rad {hr:.4f} azi {ha:.4f} "
          f"ratio {ha/hr:5.2f} MINDK/rad {0.15/hr:5.1f}")
```

### 6.2 여유 정량 (§1.4)

```python
import h5py, glob, numpy as np
MIN_DK=0.15
files=(sorted(glob.glob('data_s9_crowded_2026-08-11/*.h5'))[:6]
     + sorted(glob.glob('data_s9_sparse_2026-08-11/*.h5'))[:3]
     + sorted(glob.glob('data/episode_BF4_30comp*.h5'))[:3])
tot_iso=tot_ell=tot_pairs=0
for fn in files:
    with h5py.File(fn,'r') as f: wc=f['config/wave_components'][:]
    kx,ky=wc[:,2],wc[:,3]; N=len(kx); iu=np.triu_indices(N,1)
    dx=(kx[:,None]-kx[None,:])[iu]; dy=(ky[:,None]-ky[None,:])[iu]
    mx=(kx[:,None]+kx[None,:])[iu]/2; my=(ky[:,None]+ky[None,:])[iu]/2
    m=np.hypot(mx,my)+1e-12; ux,uy=mx/m,my/m
    dr=np.abs(dx*ux+dy*uy); da=np.abs(-dx*uy+dy*ux)
    iso=np.hypot(dx,dy)<MIN_DK                       # 출하 등방
    ell=((dr/0.045)**2+(da/0.125)**2)<1.0            # 측정된 기하 정합 타원
    tot_iso+=iso.sum(); tot_ell+=ell.sum(); tot_pairs+=len(dx)
    print(f"{fn.split('/')[-1][:46]:46s} N{N:3d} pairs{len(dx):5d} iso{iso.sum():4d} ell{ell.sum():4d}")
print(f"TOTAL {tot_pairs} | iso {tot_iso} ({100*tot_iso/tot_pairs:.1f}%) "
      f"| ell {tot_ell} ({100*tot_ell/tot_pairs:.1f}%) | freed {tot_iso-tot_ell}")
```

---

## 7. 권고 순서

1. **C2 (VARPRO + 비벌점 재적합)** 먼저. novelty 0 이지만 **C1 의 선행조건**이고, 안 하면 C1 이
   λ shrinkage 에 먹혀 거짓 널이 난다(S5 근거).
2. **C1 (비등방 분리 셀)** — 사망조건 3개 동결 후 192 런. **`FROZEN_md5.txt` 를 만들고 커밋+push**
   (50개 PREREG 중 7개만 암호학적 동결 — "preregistered" 대신 "self-attested freeze" 표기 문제 해결).
3. **C3 는 보류.**
4. ⛔ **D1–D4 재방문 금지.** 특히 브랜치 이름 `learn-boundwave-retrain` 이 살아있는 계획처럼 보이지만
   **닫힌 음성 아크**다.

**정직한 총평**: 이 저장소에 **알고리즘 계열 수준의 novelty 는 없다** — 8월 novelty 토론과 방법론
토론의 결론이 오늘 탐색으로 **다시 재현됐다**(3라운드 13축 소거 + 오늘 4축 소거 = 17축).
살 수 있는 것은 **"이 관측 기하에서 지지집합 분리조건은 비등방이고, 출하 상수가 그걸 어긴다"**
라는 **진단 하나**이며, 그 값어치는 방법이 아니라 **§1.3 이 기존 이상현상 양쪽 부호를 설명한다**는
데 있다.
