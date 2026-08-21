# P4 — 문헌·novelty 검증자 판정: **B-1 을 지금 돌릴 조건이 충족되었는가**

**2026-08-12.** 렌즈 = *"제안된 B-1 기여가 이미 선점되었는가."*
방법 = ① BRIEF/EVIDENCE 를 읽고 ② **원자료에서 전부 재도출**한 뒤 ③ 웹 검색으로 선행연구 확인.
⚠**외부 인용은 전부 검증 표시를 달았다**(`[V]`=검증, `[P]`=부분검증, `[U]`=검증 실패).

---

## 0. 한 줄

**조건부 YES — 단 지금 제안된 어떤 PRIMARY 도 novelty 를 만들지 못한다.**
B-1 이 살 수 있는 유일한 자격은 **novelty 가 아니라 타당성 수리(validity repair)** 이며,
**BRIEF §1 의 "부활 논거"는 원자료를 열어보니 성립하지 않는다** — `direction1` 은 출하 시스템이 아니다.

---

## 1. ★★★내부 재도출 — BRIEF 가 놓친 것 (원자료 직접 확인)

### 1.1 EVIDENCE.md §1–§3 의 숫자는 **전건 정확**

`results/direction1/traj_step1b_relax.json` 을 직접 파싱해 대조: 표의 10개 nRMSE·nbetter 값,
verdict 문자열, `geom` 5쌍 **전부 일치**. `platform_aperture.json` 의 CRB 7×5 표도 **전부 일치**.
⟹ **EVIDENCE.md 는 신뢰할 수 있다. 무너지는 것은 BRIEF 의 해석층이다.**

### 1.2 ★★★**결정적 오류 — `direction1` 의 정지 기준선은 20×20 m 인데, 실제 시스템의 정지 기준선은 이미 150–200 m 다**

| | 정지 팔 공간범위 | 이긴 이동 팔 |
|---|---|---|
| `direction1`(BRIEF §1 의 증거) | **19.97 × 19.96 m** | CEM 114×103 · learned 112×106 |
| **실제 WaveSplat 생성기** (`data_s9_crowded_2026-08-11`, 내가 3 에피소드 직접 측정) | **173×197 · 149×183 · 203×185 m** | — |

⟹ ★**`direction1` 이 "이동이 이긴다(21/24)"고 잰 것은 개구가 34배 결핍된 허수아비 정지 기준선에 대해서다.**
⟹ ★★★**그것도 모자라, `direction1` 의 *이긴 이동 팔*(112–116 m)조차 실제 시스템의 *정지 팔*(150–200 m)보다 작다.**
⟹ BRIEF §1 의 **"정보 통로는 닫히지 않았다"는 WaveSplat 으로 전이되지 않는다.** BRIEF 는 이것을 표시하지 않았다.

(같은 파일 `d74_closed_loop_trajectory.py:41` `L_FOOT = 20.0`. 실제 **단일 프레임** 발자국만 해도
중앙값 40–49 × 67–69 m 로 이미 2–3배 크다.)

### 1.3 ★★★**두 번째 결정적 오류 — `direction1` 은 출하 생성기도 출하 추정기도 아니다**

- **생성기**: `fly_path()`(`scripts/d74_closed_loop_trajectory.py:67`) = 배 위치 주변 **20 m 정사각형에
  균등난수 30점**을 뿌리고 `z = Σ A cos(ψ) + N(0, 0.02)` 를 **해석식에서 직접** 만든다.
  **LiDAR 광선·차폐·grazing angle·raydrop·거리의존 밀도 전부 없다.**
- **추정기**: `relax_step` + `fit_amplitudes_phases` + `render_wave_field`(`wavesplat.model.tnse`).
  **출하 경로(`wavesplat/run.py`, spectral-gaussian + coarse_to_fine + sequential_update)가 아니다.**
- **CRB**: `d72_platform_aperture.py` 의 `realistic_sea()` 도 `M=600` 랜덤 (x₀,y₀,t) 점구름 + σ=0.02 로,
  스캔라인도 차폐도 없다.

⟹ ★**BRIEF §1 은 K6(출하 구성, "완벽한 k 는 가치 0")와 `direction1`(토이) 을 대칭으로 맞세우지만
대칭이 아니다.** 오히려 **S1(2026-08-08)의 교훈 "출하 팔에서도 재라"가 정확히 여기 다시 걸린다.**
BRIEF §1 은 그 교훈을 **거꾸로 적용**해서 출하 측정을 토이 측정으로 기각했다.

### 1.4 ★★★**세 번째 오류 — CRB 헤드라인 "9.61° → 2.16° = 4.4배"는 단일 시드 추출이다**

`realistic_sea()` 는 `seed=1` 한 번만 돌아간다. 내가 **시드 1–8 로 재도출**했다:

| | v=0 | cross-track v=10 | along-track v=5 | along-track v=10 |
|---|---|---|---|---|
| 발표값(seed=1) | **9.61** | **2.16** | 23.55 | 18.97 |
| **8시드 중앙값** | **16.12** | **5.81** | **43.30** | 20.80 |
| 8시드 범위 | [7.6, **79.1**] | [1.9, **21.2**] | [14.4, 116.0] | [7.3, 71.1] |

⟹ ⛔**"4.4배" 인용 금지.** 시드 중앙값 배율은 **2.8배**이고, 발표된 두 값은 8개 중 **각각 2번째로 작은** 추출이다.
⟹ ✅**단 부호는 견고하다**: cross-track v=10 이 v=0 보다 나은 시드 **8/8**. along-track v=5 가 나쁜 시드 **8/8**.
⟹ 정리: **`CONFIRMED` 판정 자체는 살아남고, 그 크기는 살아남지 못한다.**
(메모리 `feedback_prespecify_aggregation_and_direction.md` 가 경고한 것과 같은 물림 — **크기를 단일 추출로 발표**.)

### 1.5 **네 번째 오류 — along-track 이 나빠지는 기제 설명의 상수가 틀렸다**

`d72_platform_aperture.py:97` 이 인쇄하는 설명: *"platform near-surfs the wave, **c_g=5.2**: frozen field"*.
그러나 파면이 정지(frozen)하는 조건은 **U = c_p** 이지 c_g 가 아니다. k=0.09 에서 c_p=√(g/k)=**10.44 m/s**,
c_g=5.22 m/s. 게다가 내 재도출에서 along-track 은 **v=5 에서 최악(8/8 악화)이고 v=10 에서 오히려 회복**(5/8)
⟹ *"c_g 에서 서핑해서 나쁘다"* 는 서사와 **데이터가 반대 방향**이다.
⟹ **정합 상수 오류의 재발**(메모리에 Rx-5 2π 오류·Rx-6 정합 상수 오류 기록). **네 번째 재발로 기록한다.**

### 1.6 사소한 오류 2건

- EVIDENCE §5 의 **"정지 배도 173×197 m"는 n=1**(seed48). 내가 잰 seed49=149×183, seed50=203×185.
  단일 추출을 상수처럼 인용했다.
- EVIDENCE.md 첫 줄이 *"생성 스크립트가 이 파일을 만들었다"* 라고 적었으나 **그 스크립트가 디렉터리에 없다**
  (`BRIEF.md`, `EVIDENCE.md` 2개 파일뿐). §4·§5 의 재현 경로가 문서상 끊겨 있다.
  (§1–§3 은 내가 원자료로 독립 재도출해 일치 확인했으므로 실질 피해는 없다.)

### 1.7 BRIEF 가 **옳게** 잡은 것

- §1-3 **"이긴 것은 이동이 아니라 최적화된 궤적"** — 원자료 확인 결과 **맞다**.
  `straight-cross_oracle` delta = **−0.0148, 11/24**(정지보다 나쁨), `greedy_oracle` +0.0046, 12/24(널).
  오라클에서도 이기는 것은 **CEM(+0.0205, 21/24)과 learned(+0.0182, 22/24)뿐**이다.
- §1-4 붕괴 지점이 k 추정이라는 것 — `CEM_relax` delta **−0.0101, 12/24** = 널 맞다.
- B-1 의 동기(배가 정지) — `wavesim/pipeline.py:141-142` 주석 + `ship_speed_kts` dead config 확인.
  이미 2026-05-13 랩미팅 문서에서 **3중 검증**돼 있었다(`scripts/generate_labmeeting_2026_05_13.py:1383`).

---

## 2. 문헌 — **B-1 의 각 프레이밍이 이미 점유되었는가**

### 2.1 "플랫폼 운동 = 합성 개구 ⟹ 방향 분해능" → **완전 점유 (교과서)**

이것은 SAR/SAS 의 **정의 그 자체**다. `[V]` 검색 확인: *"SAR synthesises a large antenna aperture via
platform motion"*, *"synthetic aperture sonar uses the motion of a platform to synthetically create a much
larger aperture than the physical array"*. ⟹ `platform_aperture.json` 의 `CONFIRMED` 는
**알려진 물리의 재측정**이며, 논문에서 기여로 쓸 수 없다. **배경(background)으로만 쓸 수 있다.**

### 2.2 "조우기하 — following vs crossing, U/c_p" → **완전 점유 (해양·조선 양쪽에서)**

- `[V]` **Colosi, L., Pizzo, N., Grare, L., Statom, N., Lenain, L. (2023)**, *"Observations of Surface Gravity
  Wave Spectra from Moving Platforms"*, **J. Atmos. Oceanic Technol. 40(10):1153–1169**,
  doi:10.1175/JTECH-D-23-0022.1. (제목·저자·권·페이지·DOI 다중 출처 일치. 본문은 출판사 403 으로 미열람.)
  → 플랫폼 속도로 인한 **Doppler 변조**가 속도·주파수·**파-플랫폼 상대각**에 의존함을 관측으로 다룸.
- `[V]` **Collins, C. O. III, et al. (2017)**, *"Doppler Correction of Wave Frequency Spectra Measured by
  Underway Vessels"*, **J. Atmos. Oceanic Technol. 34(2):429**, doi:10.1175/JTECH-D-16-0138.1.
  (제목·저널·권·호·연도 확인. **전체 저자 목록은 미확인** `[P]`.)
- `[V]` **following seas 모호성**: *"for sufficiently large U cos(θr), we are faced with an ambiguity in
  determining how to allocate energy from one observed frequency to two intrinsic frequencies"* —
  위 문헌군의 표준 서술. 조선 쪽에서는 **"triple-value problem" / 조우스펙트럼 "folding"** 으로 불린다 `[V]`.
- `[V]` **U cosχ ≈ c_p 에서 조우스펙트럼이 수학적으로 특이(singular)** 하다는 것, 그리고 그 지점이
  **surf-riding 임계**라는 것은 **교과서 seakeeping**(IMO 2세대 복원성 기준의 근거)이다.

⟹ ★★★**EVIDENCE §5 의 "U/c_p = 1.049 = 서프라이딩 임계, following 은 위상이 거의 안 돈다"는
새 발견이 아니라 교과서 상수를 우리 배치에서 다시 읽은 것이다.** 논문 기여로 쓰면 **리뷰어가 즉시 잡는다.**

### 2.3 "이동이 예측가능 영역/재구성을 바꾼다" → **점유**

- `[P]` **Qi, Y. et al. (2018)**, *predictable zone* — Wave Motion **77:195–213** 으로 프로젝트 메모리에 기록됨.
  ⚠**나는 이 정확한 권·페이지를 외부에서 확인하지 못했다** `[U]`. 검색은 *"Qi et al. (2018) studied the
  spatial-temporal predictable domain based on the linear wave model in theory"* 라는 2차 서술만 확인해 줬다.
  ⟹ **인용 전에 원문 확인 필요.**
- `[P]` *"Nonlinear wave reconstruction and prediction by a shipborne radar with a dynamic averaging
  algorithm"*, **Phys. Rev. Fluids 10, 094801 (2025)** — 제목·저널·권·논문번호 확인, **저자 미확인**(APS 403).
  → 핵심: *"When the ship slows down in head waves or speeds up in following waves, the relative velocity
  between the waves and the ship decreases, which **extends both the time required for reconstructing waves
  in the blind zone and the length of the predictable wave field ahead of the ship**."*
  ⟹ ★★★**BRIEF §3 의 "면적 vs 위상다양성 상충"과 정확히 같은 축이 이미 정량화되어 출판되어 있다**
  (게다가 **위상분해(phase-resolved) HOS** 로, 즉 스펙트럼이 아니라 우리와 같은 결정론적 재구성으로).
- `[P]` 이동 프로브가 예측가능 영역에 미치는 영향에 대한 보정 스킴이 이미 존재함 — 검색 요약
  *"The effects of moving probe, ambient current and finite water depth on the predictable region have been
  studied"*. 특정 논문 귀속은 **미확인** `[U]`(Desmars 계열로 추정되나 확정 못 함).

### 2.4 "선박 탑재 LiDAR 로 움직이면서 파를 잰다" → **계측 자체가 이미 존재**

`[P]` Scripps AIRSEA(Lenain / Grare / Statom) 그룹이 **2D 스캐닝 LiDAR + pan/tilt + bore-sighted 카메라 +
선속용 GPS + 시간동기 6-DOF 모션센서**를 **선박에 탑재**해 운용 중이라는 기술 서술을 확인했다.
⚠**이 구성을 기술한 특정 논문을 특정하지 못했다** `[U]` — 인용 전에 확정 필요.
관련 `[P]`: Grare, L. et al. (2021), *Frontiers in Marine Science*, doi:10.3389/fmars.2021.664728
(instrumented Wave Gliders; 발행 2021-08-16 확인, 정확한 제목 미확인).

⟹ **"움직이는 배에서 LiDAR 로 파를 재는 것"은 novelty 가 아니다. 우리가 안 한 것일 뿐이다.**

### 2.5 ★"오라클 파라미터에서 얻은 이득이 추정 파라미터에서 사라진다" → **이름이 이미 셋 있다**

이것이 이 세션의 **진짜 발견**이라고 BRIEF 가 지목한 항목이다. 판정: **현상은 표준 지식이고, 이름도 있다.**

1. **통계 실험설계**: **국소최적설계(locally optimal design)의 비효율**. Chernoff (1953) 가 도입.
   *"For nonlinear models the information matrix depends on the unknown parameters… locally optimal designs
   usually become inefficient when the replaced estimates are far from their true unknown values."* `[V]`(개념)
   ⚠Chernoff 원문의 정확한 서지(Ann. Math. Statist. 24(4):586–602)는 **직접 확인 못 함** `[U]`.
   리뷰: `[V]` **Chaloner, K. & Verdinelli, I. (1995)**, *"Bayesian Experimental Design: A Review"*,
   **Statistical Science 10(3):273–304** — 제목·저널·권·호·페이지 확인.
2. **제어**: **dual effect / certainty-equivalence 실패**. Feldbaum 의 1960–61 4부작이 원전 `[P]`
   (개념은 EOLSS *Adaptive Dual Control* 챕터에서 확인, **원 논문 서지는 미확인** `[U]`).
   `[P]` Bar-Shalom & Tse, *"Dual effect, certainty equivalence, and separation in stochastic control"*
   — 제목 확인, **IEEE TAC 19(5):494–500 (1974)** 라는 권·호·연도는 **미확인** `[U]`.
3. **로보틱스 active perception**: `[V]` **He, S., Tao, Y., Spasojevic, I., Kumar, V., Chaudhari, P.**,
   *"An Active Perception Game for Robust Information Gathering"*, **arXiv:2404.00769** (2024).
   → **추정된 information gain 과 참 information gain 의 괴리**를 게임이론으로 모델링하고
   **sub-linear regret** 를 증명. ⚠제목이 버전에 따라 *"…for Robust Exploration"* 으로도 나타난다 — 인용 시 확인.

⟹ ★★★**"oracle gain vanishes under estimated parameters"는 명명 가능한 신규 현상이 아니다.
세 분야에서 이미 명명되어 있고, 그중 하나는 regret bound 까지 있다.**
⟹ 우리가 가질 수 있는 것은 **현상이 아니라 크기**다: *"이 역문제에서 certainty-equivalence 벌점이
설계 이득의 100% 를 먹는다"* 는 **정량**은 점유되지 않았다.

### 2.6 프로젝트가 이미 기록한 선점 4건 — **전부 검증됨**

| 인용 | 검증 |
|---|---|
| Katende, R., *"No-Harm Physics-Informed Inverse Learning with Residual-Calibrated Uncertainty"*, arXiv:**2606.07153** (2026) | `[V]` 제목·저자(Kabale Univ.) 확인 |
| Sprunck, T., Pereyra, M., Liaudat, T. I., *"Bayesian model selection and misspecification testing in imaging inverse problems only from noisy and partial measurements"*, arXiv:**2510.27663** (2025, v2 2025-12-18) | `[V]` 제목·저자 확인 |
| Gomes, C. P. & Selman, B., *"Algorithm portfolios"*, **Artificial Intelligence 126(1–2):43–62** (2001) | `[V]` 제목·저널·권·호·페이지 확인 |
| Houba, N., Giarda, G., Speri, L., *"SlotFlow: Amortized Trans-Dimensional Inference with Slot-Based Normalizing Flows"*, arXiv:**2511.23228** (2025) | `[V]` 제목·저자 확인 |

⟹ **A 배치의 "3중 선점" 판정은 문헌적으로 정확하다.** 기각 사유가 아니라 **범위 제한 사유**로 유지.

---

## 3. §7 답변 (⛔"판정 없음" 없음)

> §7 은 번호 1–6 의 **6개 항목**이다. 내 브리핑이 요구한 7번째(= novelty 판정)를 §3.7 로 추가한다.

### Q1. B-1 을 지금 돌릴 조건이 충족되었는가 → **조건부 YES**

조건(실행가능 문장으로):

- **C1.** 사전등록 문서 첫 줄에 **"이 배치는 novelty 배치가 아니라 타당성 수리(validity repair) 배치다"**
  라고 명시하고, §2.1–§2.4 의 선행연구를 **배경으로 인용**한다. novelty 주장 문장은 **한 줄도 쓰지 않는다.**
- **C2.** 정지 기준선을 **실제 개구(에피소드별 실측, 대략 150–200 m)** 로 명시하고,
  **`direction1` 의 어떤 수치도 사전등록·해석·논문에 인용하지 않는다**(§1.2–§1.5).
  특히 ⛔**"9.61°→2.16° = 4.4배"** ⛔**"오라클에서 21/24"** ⛔**"이동은 정보를 만든다(CRB 확증)"** 인용 금지.
- **C3.** 팔은 **출하 생성기 + 출하 추정기**(`wavesplat.run`, 출하 기본값)로만 만든다.
  토이(`d74`/`tnse`/`relax_step`) 경로는 **이 배치에 등장하지 않는다.**
- **C4.** 운동 팔을 **면적/피복 m² 로 지정하지 않고** `U/c_p` 와 heading `χ` **두 축으로** 사전 지정한다
  (`c_p` 는 에피소드의 피크 성분에서 계산; §3.2 참조).
- **C5.** **GPU 런은 A 배치 다음이다.** B-1 의 생성(CPU 16코어, ~4 h)만 A 와 동시에 돌린다.

### Q2. §4 의 4개 조건이 옳은가 → **1·2 는 옳다(수정 필요). 3 은 BRIEF 의 반증이 틀렸다. 4 도 틀렸다.**

- **조건 1(면적 vs 위상다양성 분리)** — 취지는 옳으나 **이름이 틀렸다.**
  이 둘은 독립 변수가 아니라 **조우주파수 ω_e = ω − **k**·**U**** 라는 한 변수의 두 얼굴이다.
  ⟹ **"면적 vs 위상다양성"이 아니라 `U/c_p` × `χ`(heading) 2차원 격자로 사전 지정하라.**
  이렇게 써야 §2.2 의 선행연구와 같은 좌표계에서 비교되고, 리뷰어가 재현할 수 있다.
- **조건 2(이길 조합을 숫자로 지목)** — **옳다, 유지.** 단 예언 단위를 **피복 m²가 아니라
  "정지 기준선 대비 개구 증분비"** 로 바꿔라. 2.4× 피복이 0/24 로 진 이유의 절반은
  §1.2 가 보여주듯 **정지 기준선이 다르게 정의됐기 때문**이다.
- **조건 3(k 통로 우회 근거)** — BRIEF 가 *"§1 에서 반증됨"* 이라고 삭제선을 그었으나
  ⛔**그 반증이 틀렸다**(§1.2–§1.3: 반증 근거인 `direction1` 이 토이).
  ⟹ **대체 조건 3′**: *"`direction1` 의 어떤 결과도 출하 팔로 전이되지 않는다.
  정보 통로 질문은 출하 생성기 + 출하 추정기에서 **처음부터 다시** 열려야 하며,
  K6 (출하 팔, p=0.79) 은 여전히 **유일하게 출하 구성에서 잰 값**이다."*
- **조건 4("도움 안 됨" vs "못 씀"을 가를 팔)** — **유지.** BRIEF 는 *"이미 답이 나와 있다"*고 했지만
  그 답(CRB 확증 + 오라클 21/24)이 전부 토이에서 나왔으므로 **출하 팔에서 다시 갈라야 한다.**
  이것이 아래 Q3 의 PRIMARY 가 된다.

### Q3. ★B-1 의 PRIMARY 는 무엇이어야 하는가 → **"이동이 이기는가"도 "왜 못 쓰는가"도 아니고, 제3의 것**

**novelty 관점에서 점유되지 않은 유일한 프레이밍 = 벌점의 *크기* 를 출하 팔에서 재는 것.**

| 후보 프레이밍 | novelty 판정 |
|---|---|
| (a) "이동이 이긴다" | ⛔**점유 + 자체 데이터로 이미 부정.** SAR/SAS 교과서(§2.1) + `traj_step1_netmmse` 0/24. **기각.** |
| (b) "조우기하(following vs crossing)가 다르다" | ⛔**점유.** Colosi 2023 · Collins 2017 · PRF 10:094801 (2025) · seakeeping 교과서(§2.2–§2.3). **기각.** |
| (c) "왜 추정기가 정보를 못 쓰는가" (현상 명명) | ⛔**개념 점유.** Chernoff/Chaloner–Verdinelli · Feldbaum dual control · arXiv:2404.00769 (§2.5). **현상으로는 기각.** |
| (d) ★**"그 벌점이 이 문제에서 얼마나 큰가"(정량)** | ✅**미점유.** 유일한 생존 프레이밍. |
| (e) ★"측정한 시스템이 주장한 시스템인가"(타당성) | ✅**미점유하나 novelty 가 아님** — 학위논문 방어 자산. |

**사전등록 가능한 형태 (제안):**

- **설계**: 2 × 3 완전요인 · 대응표본(paired by episode).
  - 요인 K: `oracle_k` / `estimated_k` (출하 초기화)
  - 요인 M: `stationary` / `following (χ=0°, U=0.5·c_p)` / `crossing (χ=90°, U=0.5·c_p)`
  - `U=0.5·c_p` 로 고정하는 이유: **U/c_p→1 은 조우주파수 특이점**(§2.2)이라 교락이며,
    특이점 자체는 교과서라 잴 가치가 없다. 특이점은 SECONDARY 로만 만진다.
- **표본**: **신선 48 에피소드**(혼잡 24 + 희소 24). 시드 잔여 [1584–1818] 에서 연속 할당.
- **PRIMARY 통계량 — CE 벌점비**
  `Δ_oracle(e) = nRMSE(stationary, oracle) − min_M nRMSE(M, oracle)`
  `Δ_est(e)    = nRMSE(stationary, est)    − nRMSE(M*, est)`  (M* = 오라클에서 이긴 팔, **에피소드 무관하게 사전 고정**)
  `ρ(e) = 1 − Δ_est(e) / Δ_oracle(e)` — **ρ=1 이면 벌점이 이득 전부를 먹었다.**
- **방향·문턱(사전 동결)**: H0: `ρ ≥ 0.5`. 대립: `ρ < 0.5`(벌점이 절반 미만 = 이동이 실사용 가능).
  검정 = 에피소드별 `ρ(e) < 0.5` 부호검정, 단측 이항 α=0.05, n=48 ⟹ **문턱 ≥ 31/48**.
  **검정력**: 참 비율 0.75 이면 0.90; 0.70 이면 0.79. ⟹ **참 비율 0.75 를 사전 선언**한다.
- **SECONDARY(novelty 주장 금지, 교과서 재측정임을 명시)**:
  `U/c_p ∈ {0, 0.5, 1.05}` × `χ ∈ {0°, 90°}` 격자에서 nRMSE 와 조우주파수 스펙트럼 폭.
- **부재증명 보호**: `Δ_oracle` 자체가 널이면(부호검정 미달) ρ 는 정의되지 않는다
  ⟹ **G-0 게이트**: `Δ_oracle` 이 ≥31/48 를 못 넘으면 **PRIMARY 는 `UNDECIDED` 로 종료**하고
  ρ 를 계산하지 않는다(0/0 을 해석하는 4번째 재발 방지).

### Q4. **실패했을 때의 값** (⛔"통과하면 좋아진다" 금지)

두 분기 모두 **문서 산출물이 사전에 결정되어 있다**:

- **ρ ≥ 0.5 로 실패(= 벌점이 이득 대부분을 먹음)** ⟹
  ★**학위논문 제1장의 "선박 탑재" 주장을 "정지 가정 하에서만 유효"로 정직하게 제한할 근거가 생긴다.**
  현재 제1장 기여 7개 중 4개가 무효인 상태에서, **무효 주장을 "측정된 한계"로 대체**하는 것은
  삭제보다 나은 처리다. 또한 §2.5 의 세 문헌군에 대해
  *"이 역문제에서 CE 벌점은 설계 이득의 X% 다"* 라는 **인용 가능한 숫자 한 개**가 남는다.
- **ρ < 0.5 로 통과** ⟹ 궤적 최적화가 배포 후보로 승격(단 §2.1–§2.3 때문에 **novelty 가 아니라 공학 기여**).

**그리고 어느 분기도 아닌 세 번째 값 — 이것이 실은 가장 크다:**
★**"측정한 시스템 = 주장한 시스템" 이 처음으로 참이 된다.**
현재 학위논문/논문 제목은 *선박 탑재*인데 **100 에피소드 전부 배가 정지**이고,
이 사실은 2026-05-13 에 이미 3중 검증돼 문서화돼 있다. **이것은 방어 당일의 급소다.**
B-1 이 실패해도 이 한 가지는 **반드시** 회수된다 — 그래서 B-1 의 값은 결과에 조건부가 아니다.

### Q5. **B-1 vs A vs 집필 순위** → **집필(부분 해제) > A > B-1**. **단 동시실행이 순위를 바꾼다.**

1. **집필/철회 (1순위).** 학위논문 `thesis/thesis_main.tex` **3,992줄이 이미 존재**하고
   **제1장 기여 7개 중 4개가 무효 · 제5장은 미실행 계획서 · 제4장은 죽은 제어 축**이다.
   이것은 **측정으로 줄어들지 않는 부채**이고, 11,000 런/12일에 배포 양성 1개라는 환금율(5%)을 보면
   **측정을 더 해서 이 부채가 갚아질 확률이 낮다.** ⟹ 논문 편집 정지의 **부분 해제(무효 주장 철회 한정)**
   를 사용자에게 다시 올린다.
2. **A (2순위).** novelty 0(§2.6 3중 선점 검증됨)이지만 **"독립 표본 0→1"** 은 실재하는 값이다.
   07-26 이 자산 1위로 꼽은 **음성-기제**는 독립 재현 없이는 인용 가능한 자산이 아니다.
   검정력 0.918, 7.4 h, GPU. ⟹ **먼저 GPU 를 점유한다.**
3. **B-1 (3순위).** §2 가 보여주듯 **이길 수 있는 novelty 프레이밍이 전부 점유**돼 있고,
   유일한 생존 프레이밍(Q3-d)은 이득이 **정량 한 개**다. 11 h + 생성 4 h.

★**동시실행이 순위를 바꾸는가 — 부분적으로 YES.**
생성은 CPU(`wavesim` 에 torch 없음), 런은 GPU ⟹ **B-1 의 생성 4 h 를 A 의 GPU 7.4 h 와 겹쳐라.**
그러면 B-1 의 한계비용이 **11 h → 실질 7 h(런만)** 로 떨어지고, **A 결과를 보고 B-1 런 여부를 결정**할 수 있다.
⟹ 권고 실행 순서: **(집필 해제 요청) ∥ A@GPU ∥ B-1 생성@CPU → A 결과 확인 → B-1 런 여부 결정.**

### Q6. **메인 세션의 오류를 더 찾아라** → **6건 추가 발견** (§1 에 상세)

| # | 오류 | 심각도 |
|---|---|---|
| 1 | `direction1` 정지 팔 = **20×20 m**, 실제 정지 팔 = **150–200 m** ⟹ BRIEF §1 의 "정보 통로 열림" 논거 **전이 불가** | ★★★ |
| 2 | `direction1` 은 **출하 생성기도 출하 추정기도 아님**(해석식 점구름 + `relax_step`) ⟹ K6 와 **비대칭 대비** | ★★★ |
| 3 | CRB 헤드라인 **"9.61°→2.16° = 4.4배"가 단일 시드**. 8시드 중앙값은 16.12→5.81 = **2.8배**, 범위 [7.6,79.1]/[1.9,21.2] | ★★★ |
| 4 | along-track 악화 기제 설명의 **상수 오류**(`c_g` 로 썼으나 freezing 은 `c_p`), 그리고 데이터가 서사와 반대(v=5 최악, v=10 회복) | ★★ |
| 5 | EVIDENCE §5 **"173×197 m"가 n=1**(seed49=149×183, seed50=203×185) | ★ |
| 6 | EVIDENCE.md 가 주장한 **생성 스크립트가 디렉터리에 없다** ⟹ §4·§5 재현 경로 단절 | ★ |

**과장 점검 (BRIEF 문장 단위):**
- ⛔*"정보 통로는 닫히지 않았다 — CONFIRMED 양성이 이미 있다"* → **과장.** 토이 CRB 의 단일 시드 결과.
  올바른 문장: *"토이 관측모형에서 cross-track 운동이 방향 CRB 를 낮춘다(8시드 8/8, 중앙값 2.8배).
  이것은 SAR/SAS 교과서와 일치하며, 출하 관측모형에서는 측정된 바 없다."*
- ⛔*"K6 는 정지 구성에서 잰 것이고 이동 구성으로 확장되지 않는다"* → **참이나 대칭이 누락됨.**
  `direction1` 도 **출하 구성으로 확장되지 않는다.** 그리고 **K6 쪽이 출하 팔이다.**
- ⛔*"§3 조우기하 = 면적과 위상다양성이 상충한다 (과거 실험은 이 둘을 안 갈랐다)"* → **후단은 맞고 전단은 선점.**
  PRF 10:094801 (2025) 가 **위상분해 재구성에서** 같은 상충을 이미 정량화했다(§2.3).
- ⛔*"B-3 이 개구 요구량을 숫자로 줬다 ⟹ B-1 의 근거"* → **연결 오류.**
  B-3 의 AUC 곡선은 **레짐 분류** 지표이고, B-1 이 바꾸는 것은 **재구성 정확도**다. 같은 "개구"라는 단어를 쓸 뿐
  **같은 양이 아니다.** BRIEF §2 는 이 둘을 이어 붙였다. (BRIEF 가 "B-1 은 실측 개구를 못 키운다"고
  단서를 단 것은 옳다.)

### Q7 (추가). ★**novelty 총평 — B-1 은 논문 기여를 만들 수 없다. 학위논문 방어를 구할 수는 있다.**

- **점유 지도**: 물리(§2.1 SAR/SAS) · 조우기하(§2.2 Colosi/Collins/교과서) · 이동-예측가능영역(§2.3 PRF 2025) ·
  계측(§2.4 Scripps 선박 LiDAR) · 음성기제의 이름(§2.5 Chernoff/Feldbaum/2404.00769)
  ⟹ **다섯 층 전부 점유.**
- **미점유 = 두 칸뿐**: ① **이 역문제에서 CE 벌점의 크기**(수치 한 개), ② **우리 측정의 타당성**(방어 자산).
- ⟹ 07-26 의 만장일치 판정(*"ML 기여 REJECT / 측정과학 PARTIAL PASS, 논지 대상 = 경계·실패 지도"*)이
  **B-1 에 대해서도 그대로 유효**하다. B-1 은 **경계 지도에 점 하나를 찍는 배치**이지 축을 여는 배치가 아니다.

---

## 4. 참고문헌 (검증 표시 포함)

`[V]` 완전검증(제목·저자·저널/arXiv id·연도 확인) · `[P]` 부분검증 · `[U]` 확인 실패

1. `[V]` Gomes, C. P. & Selman, B. (2001). *Algorithm portfolios.* **Artificial Intelligence 126(1–2):43–62.**
2. `[V]` Chaloner, K. & Verdinelli, I. (1995). *Bayesian Experimental Design: A Review.* **Statistical Science 10(3):273–304.**
3. `[V]` Colosi, L., Pizzo, N., Grare, L., Statom, N., & Lenain, L. (2023). *Observations of Surface Gravity Wave Spectra from Moving Platforms.* **J. Atmos. Oceanic Technol. 40(10):1153–1169.** doi:10.1175/JTECH-D-23-0022.1 — *본문 미열람(403), 서지는 다중 출처 일치.*
4. `[P]` Collins, C. O. III, et al. (2017). *Doppler Correction of Wave Frequency Spectra Measured by Underway Vessels.* **J. Atmos. Oceanic Technol. 34(2):429.** doi:10.1175/JTECH-D-16-0138.1 — *전체 저자 목록 미확인.*
5. `[P]` (저자 미확인) (2025). *Nonlinear wave reconstruction and prediction by a shipborne radar with a dynamic averaging algorithm.* **Phys. Rev. Fluids 10, 094801.** — *APS 403, 저자 미확인. 내용 요약은 검색 스니펫에서.*
6. `[V]` He, S., Tao, Y., Spasojevic, I., Kumar, V., & Chaudhari, P. (2024). *An Active Perception Game for Robust Information Gathering.* **arXiv:2404.00769.** — *버전에 따라 제목이 "…for Robust Exploration" 으로도 표기됨.*
7. `[V]` Katende, R. (2026). *No-Harm Physics-Informed Inverse Learning with Residual-Calibrated Uncertainty.* **arXiv:2606.07153.**
8. `[V]` Sprunck, T., Pereyra, M., & Liaudat, T. I. (2025). *Bayesian model selection and misspecification testing in imaging inverse problems only from noisy and partial measurements.* **arXiv:2510.27663** (v2, 2025-12-18).
9. `[V]` Houba, N., Giarda, G., & Speri, L. (2025). *SlotFlow: Amortized Trans-Dimensional Inference with Slot-Based Normalizing Flows.* **arXiv:2511.23228.**
10. `[U]` Chernoff, H. (1953). *Locally optimal designs…* — **개념은 다수 2차 출처에서 확인, 정확한 서지 미확인.**
11. `[U]` Feldbaum, A. A. (1960–61). dual control 4부작 — **개념 확인(EOLSS *Adaptive Dual Control*), 원 논문 서지 미확인.**
12. `[U]` Bar-Shalom, Y. & Tse, E. *Dual effect, certainty equivalence, and separation in stochastic control.* — **제목 확인, 저널/권/연도 미확인.**
13. `[U]` **Qi, Y. et al. (2018). Wave Motion 77:195–213 — 프로젝트 메모리의 이 서지를 외부에서 확인하지 못했다.
    2차 서술("Qi et al. 2018 studied the spatial-temporal predictable domain")만 확인. ⚠인용 전 원문 확인 필요.**
14. `[U]` Scripps AIRSEA **선박 탑재 2D 스캐닝 LiDAR** 시스템 — 기술 서술은 확인, **특정 논문 미확정.**
15. `[P]` Grare, L. et al. (2021). *Frontiers in Marine Science*, doi:10.3389/fmars.2021.664728 — 발행일 확인, 제목 미확인.
16. `[P]` (저자 미확인) (2025). *Wave-field reconstruction using stereo cameras on a floating platform in a synthetic environment.* **Ocean Engineering** (S0029801825006717) — 제목만 확인, 403.
