# P4 = 문헌 매파 — A/B/C 의 novelty 는 **선점 여부**가 정한다

작성 2026-08-12. 규율: **열람 범위 = 주장 범위**([[feedback-scope-of-search-equals-scope-of-claim]]).
아래 §5 에 **검색어·열람 범위·못 연 문**을 전부 적었다. 못 찾은 것을 "없다"고 쓰지 않았다.

전력 공시: 나는 이 아크에서 SlotFlow(arXiv 2511.23228)로 성분집합 prior 신규성을,
Katende 로 라우터 기여를 죽였다. **오늘 Katende 를 다시 읽었고, 전보다 더 정확히 A 를 덮는다.**

--------------------------------------------------------------------------------
## 0. 한 줄 판정

> **A 는 문헌이 이미 답한 질문이다(기제·대조설계 양쪽 다 선점).
> B-1 은 문헌이 "오염"이라 부르는 것을 "자원"으로 뒤집는 유일한 열린 각도다.
> C 는 A 의 검정력 없는 판이다 — 단 "B-1 파일럿"으로 재정의하면 2위가 된다.**

novelty 순위 **B(=B-1) ≫ C(재정의 시) > A**. 원안 C(=A 축소) 고정이면 **B > A > C**.

--------------------------------------------------------------------------------
## 1. ★A 의 주장은 선점됐는가 — **그렇다. 3중으로.**

A 의 PRIMARY = *"GT 없이 잔차로 여러 실행 중 하나를 고르는 규칙이, 같은 설정을 재시드해
고르는 것보다 나은가"*. 이 문장은 **기제 · 널 · 맞대결 형식**의 세 조각으로 쪼개진다.
**세 조각이 전부 선점돼 있다.**

### 1.1 기제(= 팔 B, "잔차로 고르는 GT-free 선택층") — **직격 선점**

| 문헌 | 무엇을 했나 | A 와의 겹침 |
|---|---|---|
| **Katende, arXiv 2606.07153** (2026-06-05) "No-Harm Physics-Informed Inverse Learning with Residual-Calibrated Uncertainty" | 학습 재구성을 **잔차로 교정된 반경**이 기준선보다 나쁘지 않을 때만 채택(`R_learn ≤ R_base + ε_safe`), 아니면 기준선 반환. 저자 자신이 *"a certification-and-selection layer, **not another reconstruction architecture**"* 라고 못 박음. limited-angle tomography 포함 5개 역문제에서 시험. | **A 의 팔 B 그 자체.** 게다가 우리보다 **강하다** — 조건부 안정성 가정 아래 **사후 오차 상한**까지 유도했다. 우리는 상한이 없다. |
| **Sprunck·Pereyra·Liaudat, arXiv 2510.27663** (2025-10-31, v3 2026-05-28) "Bayesian model selection and misspecification testing in imaging inverse problems only from noisy and partial measurements" | **GT 없이, 잡음 섞인 부분 관측만으로** 후보 모델 선택 + 오지정 검출. 방법 = Bayesian cross-validation + **data fission(무작위 관측 분할)**. | **S11 §6-2 의 `obs_resid_rms_oos` held-out 신호와 같은 부류**이고 더 원리적(분할이 무작위화돼 있어 편향이 통제됨). 07-23 이 처방한 신호는 **이미 논문이 돼 있다.** |
| 고전 축 | **discrepancy principle**(Morozov) · **GCV** · **SURE / SUGAR**(arXiv 1405.1164) · **UNSURE**(arXiv 2409.01985, 잡음 수준조차 몰라도 됨) | "관측만으로 모델·정규화 파라미터를 고른다"는 **반세기 문헌**. A 는 이 문헌에 도메인 사례 1건을 더한다. |

⛔ 이미 `reference_citation_bans.md` 에 **"라우터를 기여로 주장 금지"**(07-26, Katende 선점)가 있다.
A 가 통과해도 **그 문장을 쓸 수 없다.** 이건 새 제약이 아니라 **이미 걸린 제약**이다.

### 1.2 널(= "두 번 돌려 좋은 쪽 고르면 그냥 이긴다") — **ML 방법론에선 표준 지식**

S10 이 잡은 *"널이 무(無)에서도 이긴다"* 는 이 아크의 큰 발견이었다. **문헌에선 이름이 있다.**

- **Picard, arXiv 2109.08203** (2021) — CIFAR-10 에서 시드 10⁴개를 훑어 **best-of-N 이 겉보기 이득을 만든다**를 실증. ImageNet 에서도 min–max 격차 ~0.5%.
- **Bethard, arXiv 2210.13393** "We need to talk about random seeds" (2022) — 시드 사용의 **안전/위험** 분류. *"varying only the random seed to create score distributions for performance comparison"* 을 **위험**으로 분류하고, **ACL 논문 85편 중 50% 초과가 위험 패턴**임을 보고.
- **Bouthillier et al., arXiv 2103.03098 / MLSys 2021** "Accounting for Variance in ML Benchmarks" — 시드·데이터·초기화·하이퍼파라미터가 벤치마크에 미치는 분산을 전부 모형화하고 **비교 방법론 권고**를 제시.
- **Dodge et al., EMNLP 2019** "Show Your Work" — **expected max performance as a function of the number of trials**(시행 수를 늘리면 기대 최댓값이 오른다).
- 통칭 **"seed hacking"** — 제안 방법에만 유리한 시드 부분집합을 고르면 개선이 없어도 개선처럼 보인다.

⟹ **S10/S11 의 "널이 이긴다"는 도메인 최초일 수는 있으나 현상으로는 최초가 아니다.**
심사위원이 ML 출신이면 *"그건 Picard 2021 입니다"* 한 줄로 끝난다.

### 1.3 맞대결 **형식**(`RESTART{같은 팔, 다른 시드}` vs `METHOD-SELECT{두 방법}`) — **25년 된 질문**

- **Gomes & Selman, "Algorithm portfolios", Artificial Intelligence 126(1–2):43–62, 2001.**
  포트폴리오를 **(a) 같은 알고리즘 여러 복사본을 다른 난수 시드로**(= 우리의 `RESTART`) 와
  **(b) 서로 다른 알고리즘 혼합**(= 우리의 `METHOD-SELECT`) 으로 나누고, **어느 쪽이 언제 이기는지**를
  묻는다. 결론의 형태도 우리가 기대하는 것과 같다 — **분산이 큰(risk-seeking) 팔이 포트폴리오
  안에서 유리해질 수 있다**, 그리고 **혼합 포트폴리오가 단일-알고리즘 포트폴리오를 이기는 조건이 있다**.
  ⚠**검증 등급 = 2차.** 원문 PDF 3종(ScienceDirect·Cornell·arXiv 1302.1541)이 전부 403 또는
  파싱 실패 ⟹ 위 문장은 **검색 스니펫과 2차 인용**에 근거한다. 인용 전 원문 확인 필요.
- 후속: **"On algorithm portfolios and restart strategies"**(Oper. Res. Lett. 2010) ·
  **"On Maximum Speedup Ratio of Restart Algorithm Portfolios"**(INFORMS J. Computing) ·
  SATzilla(arXiv 1111.2249) 계열 per-instance algorithm selection 문헌 전체.

### 1.4 ★그리고 **왜 잔차 선택이 실패하는가**에도 이미 이름이 있다

S11 §4b 가 n=1 로 관찰한 *"참 오차는 좋아졌는데 잔차는 반대로 움직였다"* 와
E-B 의 *"nRMSE 최적과 잔차 최적이 어긋난다"* 는 **semiconvergence** 의 교과서 서명이다:
반복 정규화에서 **잔차는 계속 내려가는데 참 오차는 어느 지점부터 올라간다.**
그래서 discrepancy principle 이 **"잡음 수준에서 멈춰라"** 라는 처방으로 존재한다.

⟹ 우리가 **발견처럼 다루던 것이 문헌에선 처방까지 딸린 기성품**이다.

### 1.5 A 에 **남는 것**

내 열람 범위에서 **못 찾은 것 하나**: *"역문제 실무 파이프라인에서 GT-free 선택 규칙을
**시드-재시작 널과 사전등록으로 맞대결**시킨 사례."* Katende 도 Sprunck 도 **seed-null 대조를
쓰지 않는다**(2606.07153 초록·2510.27663 초록 확인 범위 내).

**그러나 이것은 방법 novelty 가 아니라 프로토콜 novelty 이고**, 08-01 이 이미
**"자기반증은 기여가 아니라 바닥"**(#3)으로 강등했다. **A 의 novelty ≒ 0.**

--------------------------------------------------------------------------------
## 2. ★B-1(플랫폼 이동 = 선박 운동에 의한 합성 개구)은 선점됐는가 — **반쯤. 핵심 각도는 열려 있다.**

### 2.1 이미 있는 것 (⛔"최초" 금지)

| 문헌 | 내용 |
|---|---|
| **Phys. Rev. Fluids (2025-09), DOI 10.1103/jsd9-4kx1** — "Nonlinear wave reconstruction and prediction by a shipborne radar with a dynamic averaging algorithm" | **정지 또는 이동 선박**의 레이더 + HOS(3차 비선형) 로 재구성·예측. blind zone 으로 들어오는 파를 **점진적으로 재구성**(= 시간에 걸친 도메인 축적). ⚠APS 원문 403 — 제목·저널·시기·"stationary or moving ships"·dynamic averaging 은 **검색 스니펫 2건 교차 확인**, 초록 전문·저자명 **미확인**. |
| **DSGP-Net**, Ocean Engineering 2026 (pii `S0029801826009200`) | 희소 스테레오(**표면의 5–20%만 커버**) → dense 파고. **명시적으로** *"기존 sparse-to-dense 방법은 시간열에 의존해 **이동 플랫폼에서 실패한다**"* 를 문제로 지목하고 **단일 프레임**으로 회피. ⚠ScienceDirect 403 — 초록 요약만 열람. |
| **Qi, Wu, Liu, Yue, Wave Motion 77:195–213 (2018)** "Predictable zone for phase-resolved reconstruction and forecast of irregular waves" | 예측가능 영역을 **군속도 범위 · 방향 확산 · 측정의 공간·시간 범위**로 결정. ⟹ **"이동이 측정 도메인을 넓힌다"는 이 이론의 연역적 따름정리**이지 발견이 아니다. 실험 확인 = JMST 2021(장파정). |
| **Remote Sensing 16(21):3929 (2024)** "An RD-Domain Virtual Aperture Extension Method for Shipborne HFSWR" | 선박 운동으로 **가상 개구를 확장**하는 발상 자체가 인접 분야에 있다(단 DOA 용, 파랑장 재구성 아님). 합성개구 소나/레이더 전체도 마찬가지. |

### 2.2 ★그러나 **아직 못 찾은 것** = B-1 의 델타

내 검색 범위에서 **"선속을 개구 자원으로 놓고, 이동이 위상해상 재구성의 식별가능성 / 예측가능
영역을 얼마나 사는지 정량화한" 연구는 찾지 못했다.**

★그리고 **문헌의 지배적 태도가 정반대**라는 것이 결정적이다:

- DSGP-Net(2026): 이동 = **시간열 방법을 죽이는 것** ⟹ 회피 설계.
- 합성개구 소나/레이더: 이동 = **sub-wavelength 로 보정해야 할 것**(micronavigation).
- 선박 X-band: 이동 = **선체운동 보정 항목**(Ocean Eng. 2025, pii `S0029801825020967`).

⟹ **B-1 = "문헌이 부채로 적는 항목을 자산으로 재평가한다."** 이건 A 에 없는 종류의 novelty다.
⚠단 **PRF 2025 가 이미 "moving ships" 를 다루므로 "이동을 다룬다"는 델타가 아니다.**
델타는 **"이동의 값을 CRB / 예측가능 영역 회수율로 값매김한다"** 여야 하고,
그건 07-18 이 정한 척추(**CRB-앵커 운용성 envelope**)와 **정확히 같은 문법**이다.
⟹ **B-1 은 새 축이 아니라 척추의 미납 조각이다.** 그래서 강하다.

--------------------------------------------------------------------------------
## 3. ★"음성-기제 지도"를 논지로 삼는 선례 + `SUPPORT-TRADEOFF` 에 이름이 있는가

### 3.1 `SUPPORT-TRADEOFF` — **단일 이름은 못 찾았다. 그러나 조각별로는 전부 이름이 있다.**

| 조각 | 확립된 이름 | 출처 |
|---|---|---|
| 잔차↓ 인데 참 오차↑ | **semiconvergence** (+ 처방 = **discrepancy principle**) | 반복 정규화 교과서(Landweber/CGLS 계열) |
| 측정 영역 밖에서 예측이 무너진다 | **predictable zone / deterministic predictable zone (DPZ)** | Qi et al. Wave Motion 2018 · JMST 2021 |
| 관측점은 맞추고 먼 곳을 망친다 | **spurious long-range correlations** → 처방 = **covariance localization** | 앙상블 DA 표준(우리 ETKF 기준선이 쓰는 바로 그 문헌) |
| 관측이 못 보는 성분은 원리적으로 못 잡는다 | **null space of the forward operator** / **limited-angle artifacts** | CT·역문제 |

⟹ ⛔**`SUPPORT-TRADEOFF` 를 "새 현상"으로 파는 순간 즉사한다.**
✅팔 수 있는 형태 = **"이 네 이름이 같은 벽의 네 얼굴이고, 우리 설정에서 6회 독립 재현되며,
GT-free 게이트가 바로 그 벽 때문에 원리적으로 오작동한다"** 는 **통합 + 정량**.
(= 07-26 이 이미 1위 자산으로 지정한 **음성-기제**. 새 발견이 아니라 **이미 가진 자산의 정확한 이름**.)

### 3.2 음성을 논지로 삼는 선례 — **있다. 그것도 최상위 저널에.**

- **McGreivy & Hakim, *Nature Machine Intelligence* 2024 (arXiv 2407.07218)** —
  "Weak baselines and reporting biases lead to overoptimism in machine learning for
  fluid-related PDEs": ML 우위를 주장한 논문의 **79%(60/76)가 약한 기준선과 비교**,
  outcome/publication bias 만연, 원인 = **researcher degrees of freedom + 양성 편향**.
- **"Position: Embracing Negative Results in Machine Learning", arXiv 2406.03980.**
- 워크숍 "I Can't Believe It's Not Better!" 계열.

⟹ **두 방향의 함의가 동시에 성립한다.**
① ✅**음성·감사 서사는 출판된다** — 우리 자산 배치(07-26 ①음성-기제)는 문헌적으로 **타당**하다.
② ⛔**그 자리엔 이미 임자가 있다.** McGreivy&Hakim 은 **유체 PDE ML** 을 이미 감사했다.
   우리의 델타는 "감사한다"가 아니라 **"이 도메인의 물리적 벽(개구·예측가능 영역)을 정량한다"** 여야 한다.
③ ⏱**늦을수록 나빠진다.** 감사 문헌은 선착순이다.

--------------------------------------------------------------------------------
## 4. 08-01 이후 **새로 나온 선점** (2025 하반기 ~ 2026)

**찾은 것 7건** — 굵은 항목은 이 아크의 주장과 직접 충돌한다.

1. ★**Katende, arXiv 2606.07153** (2026-06-05) — no-harm 잔차 인증 선택층. **A 직격.**
2. ★**Sprunck·Pereyra·Liaudat, arXiv 2510.27663** (2025-10-31 / v3 2026-05-28) — GT 없이 부분 관측
   분할(data fission)로 모델 선택·오지정 검출. **A 의 held-out 신호(`obs_resid_rms_oos`) 직격.**
3. ★**DSGP-Net**, Ocean Engineering 2026 (pii `S0029801826009200`) — 희소 스테레오 5–20% → dense
   파고, **이동 플랫폼 명시**. **B-1 · 실측 스테레오(Bergamasco) 영역 측면 타격.**
4. **Phys. Rev. Fluids 2025-09** (DOI 10.1103/jsd9-4kx1) — shipborne radar, **정지 또는 이동 선박**. **B-1 측면.**
5. ★**"Fusing SWAN priors with Tucker decomposition: a *training-free* framework for *sparse wave
   field reconstruction*"**, Ocean Engineering 2026 (pii `S0029801826006207`) — **"학습 없이 희소에서
   파랑장 재구성"이 이미 논문 제목이다.** X1/X2/X3 의 "bank 라우팅이 옳았다" 서사와 같은 영토.
6. ★**Jiang·Kalisch·Benoit·Holand·Sprenger, arXiv 2605.25074** (2026-05-24) — 희소 데이터에서 방향
   스펙트럼 압축센싱 복원. **group sparsity + 에너지 제약**을 도입한 이유가
   *"ℓ₁-shrinkage 는 희소성이 엄밀히 성립하지 않을 때 신호 에너지를 과소추정한다"* ⟹
   **S5–S9 의 `lambda_l1` 아크(= 정규화항이 해를 밀어냈다)와 같은 병을 진단하고 처방했다.**
   ⚠우리의 λ 발견은 **기제로서는 이미 알려진 것**일 수 있다.
7. **He·Chabchoub·Wang, arXiv 2608.09584** (2026-08-10) — 파랑장을 **희소 결맞음 패킷 앙상블**로 표현.
   SlotFlow 에 이은 **성분집합 표현의 두 번째 선점**.

**못 찾은 것**: "분산관계 하드제약 + **이동** 희소 표고 관측 + 온라인 추적"의 **조합 전체**를 그대로
하는 논문은 못 찾았다. ⚠단 §5 스코프 한계를 반드시 병기할 것.

--------------------------------------------------------------------------------
## 5. ⚠열람 범위 = 주장 범위 (**여기까지만 봤다**)

**검색어(실제 사용, 15회)**: unsupervised/self-supervised model selection without ground truth ·
SURE/SUGAR/UNSURE hyperparameter selection · winner's curse model selection · random seed best-of-k
inflation · algorithm portfolio vs restart portfolio · multi-start NLS residual selection ·
semiconvergence + discrepancy principle · shipborne/moving-platform wave reconstruction ·
synthetic aperture ocean wave estimation · predictable zone Qi/Wu/Yue · DA spurious long-range
correlations + localization · null space limited-angle · negative results as contribution ·
2026 arXiv sparse/grid-free wave spectral estimation · LiDAR moving ship wave reconstruction.

**열지 못한 문(= 이 범위 밖은 주장하지 않는다)**
- **APS(Phys. Rev. Fluids) 403** ⟹ PRF 2025 는 **초록 전문·저자명 미확인**.
- **ScienceDirect 403** ⟹ DSGP-Net · "Fusing SWAN priors" · Gomes&Selman AIJ 원문 **전문 미열람**
  (제목·요약 스니펫만).
- **Gomes&Selman PDF 파싱 실패 3회**(Cornell·arXiv 1302.1541) ⟹ §1.3 의 "restart vs mixed 명시 비교"는
  **2차 인용 등급**. 인용 전 원문 확인 필요.
- **미열람 코퍼스**: OMAE/ISOPE 회의록 · IEEE Xplore 전문 · ProQuest 등 **학위논문 저장소** ·
  특허 전문(검색엔 걸렸으나 안 읽음) · **비영어권 문헌** · 유료 저널 전문 일반.
- ⟹ **"선점이 없다"고 쓴 곳은 이 문서에 하나도 없다.** 전부 **"이 범위에서 못 찾았다"** 로 썼다.

--------------------------------------------------------------------------------
## 6. BRIEF §4 질문 1–4 답 (문헌 관점 한정)

### Q1. A/B/C 가 통과·실패했을 때 novelty 는 어떻게 변하는가

| | **통과 시** | **실패 시** |
|---|---|---|
| **A** | novelty **음(−)**. Katende(기제)·Gomes&Selman(형식)이 이미 있는 자리에 **합성 확증 1건**을 더한다. 게다가 ⛔"라우터/선택층을 기여로" 가 이미 금지돼 있어 **쓸 문장이 없다**. 통과는 오히려 **팔 수 없는 것을 팔고 싶어지는 압력**을 만든다(08-05 실패 모드). | novelty **양(+), 단 작다.** "GT-free 선택 이득의 상당분이 seed hacking" 의 **도메인 최초 정량** = McGreivy&Hakim 계열의 각주. ★**그러나 S10 이 이미 널을 쟀고 S11 이 정정 범위를 확정했다** ⟹ A 는 **이미 아는 답에 n 을 더한다.** |
| **B-1** | novelty **최대**. 문헌이 "보정 대상"으로 적는 선박 운동을 **개구 자원으로 값매김** ⟹ `APERTURE-LIMITED` 가 지목한 **유일한 남은 레버**이자 CRB-앵커 척추의 **미납 조각**. NAOE 얼굴(운용성)과도 정합. | novelty **여전히 큼**. 이동해도 skill 이 안 사면 **개구 축이 정보급으로 닫힌다** — 재학습·λ·추적기·k·채널·배치가 전부 죽은 뒤 **마지막 물리 레버까지 죽었다**는 지도의 완성 = *"추정기 문제가 아니라 센서 배치 문제"* 라는 **논지의 결론부**. ★**A 와 달리 B-1 은 실패해도 한 章이 된다.** |
| **C** | A 와 **같은 축**에 **검정력만 낮춘다** ⟹ 문헌 관점에서 **A 의 나쁜 버전**. | 검정력 부족과 널을 못 가른다 ⟹ **S6 재방문**(13/16 이 원리적으로 통과 불가였던 그 설계 한계). **가치 ≈ 0.** |

⟹ **비대칭이 결정적이다**: A 는 **실패해야만** 값이 있고 그 값도 이미 부분적으로 샀다.
**B-1 은 통과해도 실패해도 값이 있다.**

### Q2. 박사 심사에서 어디에 들어가고, 가장 아픈 질문은 무엇인가

| | 章 | 심사위원의 **가장 아픈 질문** | 그 선택지가 답이 되는가 |
|---|---|---|---|
| **A** | Ch. 부록(방법론·프로토콜) | *"Gomes & Selman 이 2001년에 restart portfolio vs mixed portfolio 를 물었고, Katende 가 2026년에 잔차 인증 선택층을 냈습니다. **당신의 질문은 무엇이 새롭습니까?**"* | ❌**답이 안 된다.** 답할 수 있는 유일한 형태는 *"우리는 새 규칙을 제안하지 않는다 — **기존 규칙류를 시드 널로 무효화한다**"* 이고, 그건 **음성 형태**여야 하며 그러면 **A 를 돌릴 이유가 약해진다**(S10 이 이미 부분 확보). |
| **B-1** | Ch.4 CRB-앵커 운용성 envelope 의 **결론 절** + Ch.5 | *"PRF 2025 가 이미 이동 선박에서 재구성·예측을 했습니다. **당신의 델타는 무엇입니까?**"* | ✅**답이 된다** — *"그들은 이동을 **오염**으로 보고 보정했고, 우리는 이동을 **개구**로 보고 값매김한다(예측가능 영역·CRB 대비 회수율)."* ⚠**단 CRB 백본에 붙일 때만.** 안 붙이면 "우리도 이동을 다룬다"가 되어 **즉사**. |
| **C** | Ch. 부록 | *"검정력이 무너진 실험을 **왜 돌렸습니까?**"* | ❌ |

### Q3. A 의 결정적 약점(실측 미도달 · 논문 정지 · 배포 경로 부재) 반박 — **못 한다. 게다가 넷째를 더한다.**

문헌 매파로서 CON 을 **반박하지 못한다.** 반대로 **네 번째 사망 요인 = 선점**을 추가한다.

- A 가 통과해도 **쓸 자리가 없다**: 선택층은 ⛔인용금지 목록에 이미 있고, Katende 2606.07153 이
  **오차 상한까지 딸린 형태로** 자리를 잡았다.
- A 를 정당화하는 유일한 논리는 **"음성을 사기 위해 돈을 낸다"** 인데,
  그 음성(널이 이긴다)은 **S10 이 이미 샀고 S11 이 희소에서 정정했다** ⟹ **이미 산 물건을 다시 사는 것.**
- ⟹ **문헌 관점에서 A 는 novelty 로 정당화되지 않는다.** 정당화하려면 **결정영향**(배포 문장 또는
  논문 문장)이 필요한데 **둘 다 막혀 있다**(bank 게이트 · 08-04 편집 정지).
- ✅A 를 살리는 **유일한 재구성**: PRIMARY 를 *"선택이 재시작을 이기나"* 가 아니라
  ***"우리 도메인에서 GT-free 선택 이득 중 seed-hacking 분율은 얼마인가"*** 로 바꾸고,
  결과를 **McGreivy&Hakim 형식의 방법론 노트**로 출하하는 것. 그러면 novelty 는 **작지만 0 이 아니다.**
  ⚠단 그건 **11.5 h 배치가 아니라 S10 재분석**으로 대부분 얻는다.

### Q4. B 를 고른다면 무엇인가 — **B-1 > B-3 > B-5(실행 불가) > B-4 > B-2**

1. **B-1 (1위)** — 유일하게 **통과·실패 양쪽이 章이 되는** 축. 문헌 델타 생존(§2.2).
   척추(CRB-앵커 envelope)의 미납 조각. `APERTURE-LIMITED` 판정문이 **직접 지목**한 레버.
2. **B-3 실측의 레짐 위치 측정 (2위, 런 0회)** — 비용 0 이고, **DSGP-Net·PRF 2025 대비 스코프
   문장**을 만들어 준다. ★그리고 **B-2 의 내부 모순 해소의 전제**다(아래).
3. **B-5 논문화 (3위이나 ⛔사용자 지시로 정지 중)** — 실행 불가. 단 기록: McGreivy&Hakim 이
   NMI 2024 에 자리를 잡았으므로 **감사 서사는 선착순이고 늦을수록 나빠진다.**
4. **B-4 `bank_max_obs` (4위)** — 07-29 `mcap1200` 반증 있음, novelty 0.
5. **B-2 학습의 실측 전이 (최하)** — ★**모순 처리**: X1/X2 는 "학습의 실측 전이가 핵심 레버"라
   했으나 X3 가 근거 절반(표현 바닥 아티팩트)을 철회했고 재학습 축은 `APERTURE-LIMITED` 로 닫혔다.
   **이 모순은 논증으로 못 푼다 — B-3(실측이 혼잡형인가 희소형인가)을 먼저 재야 푼다.**
   ⟹ **B-2 는 B-3 뒤에 놓이며, 그 전엔 못 돈다.** 게다가 "학습 prior 가 희소 파랑 복원서 고전법을
   이긴다"는 **Ehlers OE2023(2305.11913) · Desmars JFM2023 · PINO(2508.03315) 로 이미 선점**이다.

--------------------------------------------------------------------------------
## 7. 최종 순위 (novelty 관점 **한정**)

| 순위 | 선택지 | 근거 |
|---|---|---|
| **1** | **B (= B-1 플랫폼 이동)** | 유일한 열린 문헌 델타. 문헌이 부채로 적는 것을 자산으로 뒤집는다. **실패해도 章이 된다.** |
| **2** | **C — 단 "A 축소"가 아니라 "B-1 파일럿"으로 재정의할 때만** | 원안 C(=A 축소)는 novelty 0 + 검정력 붕괴 ⟹ **3위**. |
| **3** | **A** | 기제(Katende) · 널(Picard/Bethard) · 형식(Gomes&Selman) **3중 선점**. 통과 시 novelty 음, 실패 시 양이나 **이미 부분 구매됨**. |

⛔**이 순위는 novelty 축 한정이다.** 실행 위험·시간 예산·검정력은 P3/P5 의 몫이다.

## 8. 신규 ⛔인용 금지 후보 (P4 제안)

- ⛔**"GT-free 잔차 선택은 새 아이디어다"** → Katende 2606.07153 · Sprunck 2510.27663 · discrepancy
  principle/GCV/SURE. 허용 = "이 도메인에서 **시드 널과 맞대결시킨** 사례는 우리가 찾은 범위엔 없다".
- ⛔**"널이 무에서도 이긴다는 것이 새 발견"** → Picard 2109.08203 · Bethard 2210.13393 ·
  Dodge EMNLP2019 · "seed hacking". 허용 = "**이 도메인 최초 정량**".
- ⛔**"`SUPPORT-TRADEOFF` 는 새 현상"** → semiconvergence · predictable zone · covariance
  localization · null space. 허용 = "**네 이름이 같은 벽의 네 얼굴임을 6회 재현으로 묶었다**".
- ⛔**"이동 선박 파랑 재구성 최초"** → PRF 2025(정지 **또는 이동**). 허용 = "이동을 **개구 자원으로
  값매김**한 사례는 우리가 찾은 범위엔 없다".
- ⛔**"희소에서 학습 없이 재구성"을 신규로** → Ocean Eng. 2026 "training-free ... sparse wave field
  reconstruction"(pii S0029801826006207).
- ⛔**"λ 가 해를 민다는 것이 새 기제"** → arXiv 2605.25074 가 *"ℓ₁-shrinkage 는 신호 에너지를
  과소추정한다"* 를 명시하고 에너지 제약을 처방한다.

--------------------------------------------------------------------------------
## 9. 참고문헌 (내가 **실제로 확인한 것만**, 확인 등급 표기)

**1차 확인(초록 전문 열람)**
- Katende, *No-Harm Physics-Informed Inverse Learning with Residual-Calibrated Uncertainty*, arXiv **2606.07153** (2026-06-05).
- Sprunck, Pereyra, Liaudat, *Bayesian model selection and misspecification testing in imaging inverse problems only from noisy and partial measurements*, arXiv **2510.27663** (2025-10-31; v3 2026-05-28).
- McGreivy & Hakim, *Weak baselines and reporting biases…*, arXiv **2407.07218** / **Nature Machine Intelligence** 2024, doi 10.1038/s42256-024-00897-5.
- Bethard, *We need to talk about random seeds*, arXiv **2210.13393** (2022).
- Jiang, Kalisch, Benoit, Holand, Sprenger, *Recovery of directional wave spectrum from sparse data with compressed sensing*, arXiv **2605.25074** (2026-05-24).
- He, Chabchoub, Wang, *Rogue Wave Statistics from a Sparse Coherent Structure Decomposition*, arXiv **2608.09584** (2026-08-10).

**2차 확인(검색 스니펫·2차 인용만; 원문 미열람)**
- Gomes & Selman, *Algorithm portfolios*, **Artificial Intelligence 126(1–2):43–62, 2001**.
- Picard, *torch.manual_seed(3407) is all you need*, arXiv **2109.08203** (2021).
- Bouthillier et al., *Accounting for Variance in ML Benchmarks*, arXiv **2103.03098** / MLSys 2021.
- Qi, Wu, Liu, Yue, *Predictable zone…*, **Wave Motion 77:195–213 (2018)**.
- *Nonlinear wave reconstruction and prediction by a shipborne radar with a dynamic averaging algorithm*, **Phys. Rev. Fluids** (2025-09), doi 10.1103/jsd9-4kx1.
- **DSGP-Net**, Ocean Engineering 2026, pii `S0029801826009200`.
- *Fusing SWAN priors with Tucker decomposition…*, Ocean Engineering 2026, pii `S0029801826006207`.
- *An RD-Domain Virtual Aperture Extension Method for Shipborne HFSWR*, **Remote Sensing 16(21):3929 (2024)**.
- UNSURE, arXiv **2409.01985**; SUGAR, arXiv **1405.1164**; SATzilla, arXiv **1111.2249**;
  *Position: Embracing Negative Results in ML*, arXiv **2406.03980**.
- Dodge et al., *Show Your Work*, EMNLP 2019 (제목·주장만 확인, 원문 미열람).

**기존 목록에서 재확인(신규 아님)**: SlotFlow arXiv 2511.23228 · Ehlers arXiv 2305.11913 ·
PINO arXiv 2508.03315 · Desmars JFM 2023 · Mamandipoor arXiv 1509.01942.
