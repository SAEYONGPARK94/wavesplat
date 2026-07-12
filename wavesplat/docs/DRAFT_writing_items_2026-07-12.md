# 집필 항목 5건 초안 (B-7 · B-8 · B-9 · B-10 · B-13) — 2026-07-12

> Opus 집필 에이전트 초안. 논문에 바로 이식 가능한 프로즈/표 형태.
> 검수·삽입 위치 배정은 메인 에이전트가 한다. **커밋 금지.**
> 톤 규율: "최초/유일/SOTA/winning" 금지, 겸손·정확(기존 `PAPER_CH_C_draft` 톤 준수).
> 모든 수치는 각 절 끝 **[출처]** 줄에 재확인 경로를 명기했다.
> 다른 에이전트 작업 영역(`results/direction3/figs_appendix/`,
> `scripts/d3_appendix_figs.py`)은 건드리지 않았다.

---

## B-7. 용어 규율 절 — "실측 검증(real-data validation)"의 정의 + Limitations 1번

### B-7.1 정의 문단 (Methods 또는 Data 절 서두에 삽입)

**국문 초안**

본 논문에서 "실측 검증(real-data validation)"이라는 용어는 하나의 고정된
의미로만 사용한다: **공개 스테레오 카메라로 재구성된 실해역 해수면장에,
사양-충실(spec-faithful) LiDAR 샘플링 연산자를 zero-shot으로 적용한 평가**를
가리킨다. 구체적으로 관측 연산자는 Seyond Falcon LiDAR의 공표 사양(사용자
매뉴얼 p.12: 시야각 120°×25°, 각분해능 0.18°×0.24°, 250 m @ 10 % 반사율,
거리 정밀도 2 cm 1σ, 주사율 10 FPS)을 모사한 전방-부채꼴 희소 샘플링 +
거리 잡음(N(0, 2 cm))으로 구성되며, 이 연산자를 스테레오 재구성 표면 위에
얹어 그 표면 자체를 참값으로 채점한다. 즉 참 해수면은 실측(스테레오)이되,
LiDAR 관측은 사양-모사이며, 추정기는 실측 표면에 대해 사전 미세조정 없이
(zero-shot) 적용된다.

이 정의를 고정하는 이유는 두 가지다. 첫째, 실해역 해수면의 통계(방향
분산, 협대역성, 비선형 마루-골 비대칭, 재구성 잡음)는 시뮬레이션이 완전히
재현하지 못하므로, 관측 연산자를 실측 표면에 적용하는 것은 추정기의
분포-외(out-of-distribution) 강건성에 대한 유의미한 시험이다. 둘째,
동시에 이것은 선박 탑재 LiDAR로 취득한 실물 거리 데이터가 아니며, 그
차이를 흐리지 않기 위해 "실측 검증"의 외연을 위와 같이 명시적으로
한정한다. 본문에서 이 용어가 등장할 때는 항상 이 의미이며, 실물 LiDAR
취득을 함의하지 않는다.

**English (병기, 이식용)**

Throughout this paper, "real-data validation" denotes one thing only: the
zero-shot application of a spec-faithful LiDAR sampling operator to
stereo-reconstructed real-sea surfaces. The observation operator emulates
the published Seyond Falcon specification (manual p.12: FOV 120°×25°,
angular resolution 0.18°×0.24°, 250 m @ 10 % reflectivity, 2 cm 1σ range
precision, 10 FPS) as a forward-fan sparse sampler with additive N(0, 2 cm)
range noise, applied on top of the reconstructed surface, which itself
serves as ground truth. The true surface is measured (stereo); the LiDAR
observation is spec-emulated; the estimator is applied without any
fine-tuning on the real surfaces. We fix this scope so that the term is
never read as implying ship-mounted LiDAR field data, which we do not
possess (Limitation L1).

### B-7.2 Limitations 항목 1 (Limitations 절 첫 항목)

**국문 초안**

**L1 — 실물 선박 탑재 LiDAR 실측 데이터의 부재.** 본 연구의 실측 검증은
전부 공개 스테레오 재구성 표면(IFREMER Bergamasco 아카이브)에 사양-모사
LiDAR 연산자를 적용한 것이며, 실제 선박에 탑재한 LiDAR로 취득한 거리
데이터는 포함하지 않는다. 우리는 공개 가능한 스테레오 레코드를 전수
조사하여 12–13개 레코드(4개 사이트)가 접근 가능한 전량임을 확인했고, 이를
사용했다. 따라서 (i) 실제 LiDAR의 반사율·입사각 의존 결측 패턴,
(ii) 선체 운동·진동에 의한 자세 잡음, (iii) 물보라·강우 등 대기-해면
간섭은 본 검증에 직접 포함되지 않으며 사양-모사 연산자의 단순화(부채꼴 내
균일 랜덤 샘플링)로 근사된다. 이 한계를 메우기 위한 최소 실물 실증
계획은 §[B-8 로드맵 절]에 기술한다. 본 논문의 결론은 이 범위 안에서만
보증되며, 특히 성능의 절대 수치(상관계수·nRMSE)는 사양-모사 조건에서의
값임을 명시한다.

**[출처]** Falcon 사양: `CLAUDE.md`, `docs/HANDOFF_BRIEF_2026-07-11.md` §0
(매뉴얼 p.12). 스테레오 아카이브 전량 12–13 레코드/4 사이트:
`docs/LIMITATIONS_ledger_2026-07-10.md` C1, `PERSONA_DEBATE_phd_value_2026-07-12.md`
R4. 사양-모사 단순화(부채꼴 균일 랜덤): `scripts/d3_falcon_real_operator.py`
lines 17–20.

---

## B-8. 수조 최소 실증 로드맵 절 (Discussion 또는 Future Work)

> 선행 확인: `docs/snu_towing_tank_experiment_plan.md`(2026-06-27 작성) **존재**.
> 본 절은 그 계획서를 학위 스케줄 내 "최소 1캠페인"으로 압축하고, 목적을
> "관측 연산자의 실물 검증"으로 고정한다.

**국문 초안**

앞 절(L1)의 한계에 대한 직접적 대응으로, 우리는 서울대학교 예인수조(towing
tank)에서 단일 최소 실증 캠페인을 계획한다. 이 캠페인의 목적은 성능
헤드라인의 경신이 아니라 **관측 연산자의 실물 검증** — 즉 사양-모사 LiDAR
샘플링이 실제 고체-상태 LiDAR의 거리 측정과 어느 정도 일치하는지, 그리고
그 위에서 추정 파이프라인이 실물 센서-실물 참값 쌍에 대해 재현되는지를
확인하는 것이다.

캠페인의 핵심 구성은 다음과 같다. 선박급 고체-상태 LiDAR(예: Seyond
Falcon, σ_range ≈ 2 cm, ≥ 10 Hz)를 수조 갠트리/캐리지에 스치는 입사각으로
전방-하향 장착하여 선박 탑재 기하를 모사하고, 참값은 알려진 x(및 소수의
across-tank y) 위치의 저항식/정전용량식 파고계(wave probe) 배열로 취득한다.
LiDAR·파고계·조파기 신호·캐리지 엔코더는 공통 시간축(트리거)으로 동기화하고,
조건별 최소 5회 반복(조파기 위상 시드 변경)으로 불확실성·통계를 확보한다.
최소 캠페인에 반드시 포함하는 실험은 다음 네 가지다:

- **E1 (규칙파 센서 정합, 최우선):** 낮은 급경사 규칙파에서 LiDAR가
  복원한 해수면 표고를 파고계와 직접 비교(RMSE_z)하고 분산관계 ω(k)의
  일치를 확인한다 — 이것이 관측 연산자 검증의 본체다.
- **E3 (불규칙 1-D nowcast):** 단방향 JONSWAP 스펙트럼에서 희소 관측만으로
  주파수-스펙트럼과 장을 복원(장 nRMSE, 스펙트럼 R², Hs/Tp 오차).
- **E7 (예보/예측 가능 영역):** 캐리지 지점 nowcast에서 하류 파고계 지점의
  표고를 예보하여 리드-거리별 예보 skill과 UQ 커버리지를 실물로 측정.
- **E9 (UQ 보정):** ≥ 5회 반복으로 ±1σ/±2σ 커버리지와 신뢰도 곡선 측정.

**중요한 범위 고지(honest scoping).** 예인수조는 본질적으로 단방향
조파이므로, 본 논문에서 가장 어려운 다방향(short-crested) 복합 해상 주장은
수조로 검증할 수 없다. 따라서 수조 캠페인은 **1-D·주파수 영역·시간 융합·
관측 연산자 충실도·UQ 보정**의 물리적 검증에 한정하고, 다방향 일반화는
시뮬레이션 + 향후 해양공학수조(예: KRISO 다방향 basin) 캠페인으로 명시적
분리한다. 또한 수조 파는 Froude 축척(H ~ 0.05–0.3 m, T ~ 0.6–2 s)이므로
모형 축척과 실선 등가 축척을 함께 보고한다. 이 캠페인은 "실측 검증 완료"의
외연을 실물 센서로 확장하되, 그 주장은 위 네 실험이 다루는 축(센서 충실도·
시간 융합·예보·UQ)에 한정됨을 분명히 한다.

**English (한 문장 요지, 이식용)**

We plan a single minimal towing-tank campaign whose purpose is validation
of the *observation operator* — not a performance headline: it checks how
closely the spec-emulated LiDAR sampler matches a physical solid-state
LiDAR against wave-probe ground truth (E1), and whether the estimation
pipeline reproduces on physical sensor / physical truth pairs for 1-D
nowcast (E3), forecast (E7), and UQ calibration (E9). Multi-directional
claims remain in simulation and are deferred to a future ocean-basin
campaign, since a towing tank is uni-directional.

**[출처]** 계획서 존재·실험표: `docs/snu_towing_tank_experiment_plan.md`
(2026-06-27; §2 setup, §3 표 E1–E10, §1 단방향 제약, §4 청구항 매핑).
"실물 검증이 목적, 헤드라인 아님" 프레임: `PERSONA_DEBATE_phd_value_2026-07-12.md`
R4 (G 발언, 행동항목 #3).

---

## B-9. 'CNN = 최소 충분 인스턴스' 문단 (Ch.C 학습 챕터, 방법론 방어 절)

**국문 초안**

우리의 학습 배치는 소형 CNN이 물리 사후분포의 하이퍼파라미터(진폭 스펙트럼
prior γ, 잡음바닥 배율 smult)를 예측하도록 되어 있다. 이 선택은 아키텍처가
주장의 대상이어서가 아니라, **이 과제에서 표현 용량이 병목이 아님을 세
차례 측정으로 확인**했기 때문이다. 첫째, 자기지도 인코더의 용량을 13층에서
1,600만-파라미터급까지 바꾸어도 읽기-헤드의 미관측-바다 성능은 사실상
불변이었다(+0.508 vs +0.527; ΔR² ≈ 0.019). 둘째, 수십 개 파라미터의 대각
가중(wiener2) 구조가 자유 CNN 잔차를 전 구간에서 앞섰다. 셋째, 결정론적
물리 사상인 속박파(bound-wave) 헤드는 고정 물리 연산자(0.195)가 자유
CNN(0.111)을 앞섰다. 나아가 선형-가우시안·올바른-은행·MSE 무대에서는 어떤
학습기도 사후평균을 이길 수 없다는 정리가 성립하므로, 인코더를 현대적
아키텍처로 교체해도 그 무대의 천장은 뚫리지 않는다. 병목은 표현이 아니라
정보(개구)다.

이 진술은 point-native 아키텍처 교체 실험으로 직접 검정되었다. 동일한 학습
데이터(bwtask 8,000 셔드, 동일 분할)·동일한 미분가능 MMSE 목적함수·동일한
동결 판정창(60창)에서, 현직 CNN(beamform 파워맵 입력)을 정보-동등한 원시
점군을 입력받는 perceiver-형 set-encoder로 교체한 결과, 실측 60창 짝지음
상관은 **point +0.472 vs CNN +0.620 (Δ −0.148, t −3.90)**, 실패 창
(hand < 0.5, n = 25)에서도 **point +0.279 vs CNN +0.372 (Δ −0.093)**로,
동등성 문턱에 크게 못 미쳤다. 원인 위치 진단이 이 판정을 완성한다: 동일
검증 표본에서 CNN 0.745 vs point 0.815 — 즉 격차는 실측 전이가 아니라
**합성 훈련 도메인에서부터** 존재한다. 정보가 동등한 원시 점군에서
attention 인코더가 재발견하지 못한 것을, beamform-맵 + kp-정규화 격자라는
물리-정규화 귀납 편향이 실제 성능 기여로 제공하고 있다는 뜻이다. 이것이
G1(용량 무관)·wiener2(구조 > 자유)·속박파(물리 > 자유)에 이은 네 번째
동일-방향 판정이다.

따라서 우리는 CNN을 학습이 꽂히는 슬롯(γ, σ)의 **최소 충분 인스턴스**로
서술한다: 이 인터페이스는 아키텍처-불가지론적이며(어떤 예측기든 γ·σ를 내면
된다), CNN은 111× 가속(amortization)과 100 ms 계산 예산을 만족하는 가장
단순한 구현이다. 아키텍처 자체는 본 논문의 기여가 아니며, 학습의 기여는
"물리 사후분포의 어느 슬롯에 학습을 배치해야 실측에서 이기는가 + 그 주장을
사전등록 게이트로 인증하는 절차"에 있다.

**아키텍처 민감도 표 (논문 자산; 승패 무관)**

| Arm | 전창 corr (n=60) | 실패창 corr (hand<0.5, n=25) |
|---|---|---|
| hand-MMSE (무학습 물리) | +0.532 | +0.145 |
| CNN 현직 (beamform 맵, γ+smult 헤드) | +0.620 | +0.372 |
| point-native (perceiver-lite set-encoder) | +0.472 | +0.279 |

*비고:* 두 학습 팔(CNN·point) 모두 무학습 hand를 실패 창에서 앞선다
(+0.372, +0.279 vs +0.145) — 점군 인코더도 zero-shot 실측 전이는 성립한다.
표현 축의 판정은 "point-native가 불가능"이 아니라 "동일 데이터·목적함수
예산에서 맵-CNN 동등에 미달(Δ −0.148)"이다. 단일 아키텍처
(perceiver-lite 0.29 M vs CNN 1.06 M), 콜드스타트(3× 스텝으로 보상),
아키텍처 탐색 미수행을 한계로 공시한다.

**English (핵심 문장, 이식용)**

We describe the CNN as the *minimal sufficient instance* of the slot into
which learning is inserted (γ, σ): the interface is architecture-agnostic,
and the CNN is the simplest realization meeting the 111× amortization and
100 ms compute budget. Representational capacity is not the bottleneck
here — a point-native perceiver encoder on information-equivalent raw
points, trained on the same shards with the same differentiable-MMSE loss,
underperformed the map-CNN by Δ −0.148 (t −3.90), a gap already present in
the synthetic training domain (0.815 vs 0.745), indicating the
physics-normalized inductive bias (beamform map + kp-normalized bank)
carries the effective contribution.

**[출처]** point vs CNN 판정 수치 (재확인 완료, `results/direction3/m3_point_eval.json`,
n=60): 전창 point 0.4719 / cnn 0.6195 / hand 0.5320; 짝지음 Δ(point−cnn)
−0.1476, t −3.897; 실패창(hand<0.5, n=25) point 0.2789 / cnn 0.3716 /
hand 0.1450, Δ −0.093. 동일-val 0.745 vs 0.815: `docs/M3_POINT_prereg_2026-07-12.md`
판정 결과 §(훈련 로그 유래, eval JSON에는 미포함 — 문서 값 채택).
G1 enc13/1600 (+0.508/+0.527), wiener2, 속박파(0.195/0.111), 선형 천장
정리: `docs/LIMITATIONS_ledger_2026-07-10.md` B1·B4,
`PERSONA_DEBATE_cnn_outdated_2026-07-12.md` R1. 111×: `HANDOFF_BRIEF` §7.

---

## B-10. 커버리지 매트릭스 표 (Ch.C 또는 Data/Evaluation 절)

**국문 초안 (표 도입 문단)**

보유 실측 레코드마다 가능한 평가 프로토콜이 다르다 — 스테레오 패치의
물리적 크기가 전방-부채꼴 조우 기하를 담을 수 있는지, 그리고 참값 표면의
품질이 채점을 견디는지에 따라 갈린다. 아래 매트릭스는 각 레코드가 어느
증거층에 기여하며, 어떤 프로토콜로 채점되었고, 특정 층에서 제외된 경우 그
사유를 한눈에 정리한다. 이 표는 "전체 데이터"라는 표현이 층마다 다른
부분집합을 가리킬 수 있다는 점을 투명화하기 위한 것이다.

**커버리지 매트릭스**

레코드(사이트·패치 대각):
- **BS11a / BS11b** (Bergamasco 2011, 수심 30 m, 패치 대각 ~16 m)
- **BS13a / BS13b** (Bergamasco 2013, 수심 30 m, 패치 대각 ~20–22 m)
- **AA14** (2014, 수심 17 m, ~65 m), **AA15a / AA15b** (2015, 수심 17 m, ~70–77 m)
- **YS17** (2017, 수심 50 m, ~242 m), **LJ18g / LJ18n** (2018, 수심 45 m, ~250–300 m 폭풍 너울)

| 레코드 | ① 비관측영역 nowcast (8+2 rec) | ② 전장 60창 | ③ 조우 트랙 60창 | ④ 폭풍 확장 | 제외 사유 |
|---|:---:|:---:|:---:|:---:|---|
| BS11a | ✔ | — | — | — | 패치 ~16 m < 부채꼴(r 8–40 m): 전방-fan 기하 불가 |
| BS11b | ✔ | — | — | — | 동일(패치 ~16 m) |
| BS13a | ✔ | — | — | — | 동일(패치 ~22 m) |
| BS13b | ✔ | — | — | — | 동일(패치 ~20 m) |
| AA14  | ✔ | ✔ | ✔ | — | — |
| AA15a | ✔ | ✔ | ✔ | — | — |
| AA15b | ✔ | ✔ | ✔ | — | — |
| YS17  | ✔ | ✔ | ✔ | — | — |
| LJ18g | ✔(폭풍) | ✔ | ✔ | ✔ | — |
| LJ18n | ✔(폭풍) | — | — | ✔ | 전장/트랙 60창 프로토콜 미편입(폭풍 확장 판정에만 사용) |

프로토콜·판정 요약(층별):

- **① 비관측영역 nowcast** (10 레코드; 8 풍파 + 2 폭풍): 희소 관측
  창에서 관측되지 않은 영역의 표고를 복원, γ-net vs hand-MMSE 짝지음.
  창-짝 이득 **+0.143 (t 4.42, 8/8 레코드 부호일치, 부트스트랩 CI
  [0.087, 0.206]); 상대 ~+36 %.** 폭풍 2 레코드는 층 ④와 상보(LJ18g +0.146,
  LJ18n −0.054 = 폭풍 확장은 동률/혼재).
- **② 전장 60창** (5 레코드 × 12창): 채점 부채꼴 전역의 장 상관.
  learned +0.583 vs hand +0.550, **이득 +0.033 (t 3.22, 5/5, CI [0.018, 0.048]);
  진폭비 learned 0.69× vs hand 0.89×.**
- **③ 조우 트랙 60창** (5 레코드 × 12창): 배가 지나는 미래 궤적 위의
  위상. head(learned) +0.620 vs hand +0.532, **전창 +0.088 (t 3.14, 5/5),
  실패창(hand<0.5, n=25) +0.227 (t 4.75, 5/5).**
- **④ 폭풍 확장** (LJ18g / LJ18n): 긴 너울(λ ~ 300 m, 부채꼴 << 파장)
  스트레스 영역에서의 재판정 — 확정 이득 없음(동률).

**English (한 문장 요지)**

Each real record admits a different set of evaluation protocols, gated by
whether its stereo patch is large enough to host the forward encounter fan;
the matrix makes explicit which records feed which evidence layer, so that
"the full dataset" is never conflated across layers.

**[출처]** 재확인 완료.
- 패치 대각(L_diag): `sa_multirecord_mmse.json` 창별 `L_diag` 평균 —
  BS11a 15.9, BS11b 16.3, BS13a 22.1, BS13b 19.7, AA14 65.2, AA15a 70.3,
  AA15b 76.8, YS17 241.9 m. 부채꼴 기하·제외 사유: `scripts/d3_falcon_real_operator.py`
  lines 5–20, 66–86 (RECORDS: YS17/AA15a/AA14/AA15b/LJ18g만 fan 가능;
  r_fan (8–40) 또는 (20–100)).
- 층 ① nowcast: `sa_multirecord_mmse.json` / `sa_multirecord_mmse_net.json`
  (풍파 8) + `_LJ` 짝 (폭풍 2); 종합 통계 `cluster_bootstrap_2026-07-12.json`
  contrast A (mean +0.1428, t_win 4.420, t_rec 4.321, sign 8/8,
  boot_ci [0.0874, 0.2062]). 상대 +36 %는 프레임 수치(레코드-수준
  common_unobs 재계산: 8 풍파 +40 %, 10 레코드 +37 % — 절대 +0.143을
  1차 인용, 상대는 ~+36 % 근사로 표기).
- 층 ② 전장: `field_wide_eval.json` (field_hand 0.5497, field_learned
  0.5832, Δ +0.0334; amp_hand 0.887, amp_learned 0.692) + `cluster_bootstrap`
  contrast E (t_win 3.219, CI [0.018, 0.048], 5/5).
- 층 ③ 트랙: `gate_promotion_eval.json` (hand 0.5320, bw_s0_head 0.6195,
  Δ +0.0875; bad25 hand 0.1450 head 0.3716 Δ +0.2266) + `cluster_bootstrap`
  contrast B (all60 t_win 3.135; bad25 t_win 4.750, 둘 다 5/5).
- 층 ④ 폭풍: `sa_multirecord_mmse{,_net}_LJ.json` (common_unobs: LJ18g
  hand 0.108 → net 0.254; LJ18n hand 0.361 → net 0.307).

---

## B-13. 프레임 고정 문단 (초록·서론용 + C.2b 정렬)

**국문 초안 (초록/서론용 핵심 문단)**

우리 문제의 본체는 **전장(全場) 파랑 추정**이다 — 배가 지나는 조우 트랙은
그 추정된 장에 대한 하나의 질의 패턴일 뿐이며, 특정 궤적이 목적을 정의하지
않는다. 이 프레임은 두 가지 불변성을 요구한다. 첫째 **방향 불변성**:
추정은 지구-고정 좌표에서 이루어져 선박 침로에 종속되지 않는다. 둘째
**속도 불변성**: 파수 은행이 등방(isotropic)이어서 선속·조우 주파수가 장
표현을 편향시키지 않는다. 조우 예보는 이 좌표-불변 장에 대한 시공간 질의로
자연스럽게 유도된다.

이 프레임 위에서 학습의 몫은 하나의 기울기(gradient)로 정리된다:
**정보가 옅어지는 곳일수록 학습 prior의 몫이 커진다.** 관측되지 않은 영역의
nowcast에서 학습은 상대 ~+36 %(창-짝 절대 +0.143, t 4.42, 8/8 레코드)를
얻고, 채점 부채꼴 전역의 전장 지표에서는 손해 없이 소폭 우위(+0.033,
t 3.22, CI 하한 +0.018 > 0)를 유지하며, 배가 실제로 지나는 조우 트랙에서는
+0.088(t 3.14)로, 물리 기준이 실패하는 창(hand corr < 0.5)에서는
+0.227(t 4.75)로 이득이 커진다. 즉 물리 추정기가 이미 잘 맞는 곳에서는
학습이 거의 중립이고, 물리가 확신 있게 틀리는 곳에서 학습이 겸손하게
맞춘다. 이 단조 기울기 자체가 "학습은 물리가 모르는 것을 채울 때만
이긴다"는 중심 명제의 정량적 증거다.

**English (병기, 이식용)**

The object of estimation is the *whole wave field*; the encounter track a
ship follows is one query pattern over that field, not the objective
itself. This requires two invariances — directional (estimation in
earth-fixed coordinates, independent of heading) and speed (an isotropic
wavenumber bank, so vessel speed / encounter frequency does not bias the
field representation) — from which encounter forecast follows as a
space-time query. On this frame the role of learning forms a single
gradient: the thinner the information, the larger learning's share. In the
unobserved-region nowcast learning gains ~+36 % (paired +0.143, t 4.42,
8/8 records); over the whole scoring fan it holds a small, harmless edge
(+0.033, t 3.22, CI lower bound +0.018 > 0); on the encounter track it
gains +0.088 (t 3.14), and where the physics baseline fails (hand corr
< 0.5) the gain rises to +0.227 (t 4.75). This monotone gradient is the
quantitative form of the thesis: learning wins only where the physics does
not already know the answer.

**C.2b 정렬 노트 (기존 C.2b 절 앞에 연결 문장으로).** 위 기울기의 마지막
칸(실패 창)이 C.2b "어떻게 잘 맞는가"의 대상이다. 그곳에서 "장-겉보기"와
"트랙-정확도"가 갈라진다: 물리 기준은 실패 창에서 장 질감을 그럴듯하게
보존하되(진폭비 0.89×) 조우 트랙 위상이 반대로 갈 수 있고, 학습 팔은 장을
흐리는 대가로(진폭비 0.69×) 트랙 위상을 산다. 따라서 전장 지표의 소폭
우위(+0.033)와 트랙·실패창의 큰 이득(+0.088, +0.227)은 모순이 아니라 같은
기제(비구속 성분의 수축 → 구속 구조의 위상 보존)의 두 단면이다.

**[출처]** 재확인 완료. 기울기 4칸:
- nowcast +0.143 (상대 ~+36 %): `cluster_bootstrap_2026-07-12.json` A
  (mean 0.1428, t_win 4.420, sign 8/8, CI [0.0874, 0.2062]).
- 전장 +0.033: `cluster_bootstrap` E (mean 0.0334, t_win 3.219,
  CI [0.018, 0.0483], 5/5); 원본 `field_wide_eval.json` (0.5497 → 0.5832).
- 트랙 +0.088: `cluster_bootstrap` B all60 (mean 0.0875, t_win 3.135, 5/5);
  원본 `gate_promotion_eval.json` (0.5320 → 0.6195).
- 실패창 +0.227: `cluster_bootstrap` B bad25 (mean 0.2266, t_win 4.750, 5/5);
  원본 `gate_promotion_eval.json` bad25 (0.1450 → 0.3716).
- 진폭비 0.89× / 0.69×: `field_wide_eval.json` (amp_hand 0.887, amp_learned
  0.692). 프레임(전장 본체·트랙은 질의·불변성): `PERSONA_REVIEW_fullfield_worklist_2026-07-12.md`
  R3, B-13; `HANDOFF_BRIEF` §0.

---

## 부록: 재확인한 수치 목록 (감사용)

| 수치 | 재확인 값 | 출처 | 상태 |
|---|---|---|---|
| M3 전창 point/cnn/hand | 0.4719 / 0.6195 / 0.5320 | `m3_point_eval.json` (직접 계산) | 일치 |
| M3 point−cnn Δ, t | −0.1476, t −3.897 | 동상 | 일치 (문서 −0.148/−3.90) |
| M3 실패창 point/cnn/hand | 0.2789 / 0.3716 / 0.1450 | 동상 (hand<0.5, n=25) | 일치 |
| M3 동일-val cnn/point | 0.745 / 0.815 | `M3_POINT_prereg` §판정 (훈련 로그) | 문서 값 채택(JSON 미포함) |
| nowcast 창-짝 이득 | +0.1428, t_win 4.420, 8/8, CI [0.087,0.206] | `cluster_bootstrap` A | 일치(JSON 채택) |
| nowcast 상대 % | ~+36 % (재계산 8풍파 +40 %, 10rec +37 %) | `sa_multirecord_mmse{,_net}.json` 재계산 | **경미 불일치**: 절대 +0.143 1차 인용, 상대는 프레임 값 ~+36 % |
| 전장 이득 | +0.0334, t 3.219, CI [0.018,0.048], 5/5 | `cluster_bootstrap` E / `field_wide_eval.json` | 일치 |
| 전장 learned/hand | 0.5832 / 0.5497 | `field_wide_eval.json` | 일치 |
| 진폭비 learned/hand | 0.692 / 0.887 | `field_wide_eval.json` | 일치 |
| 트랙 이득 all60 | +0.0875, t 3.135, 5/5 | `cluster_bootstrap` B / `gate_promotion_eval.json` | 일치 |
| 트랙 이득 bad25 | +0.2266, t 4.750, 5/5 | 동상 | 일치 |
| 트랙 hand/head | 0.5320 / 0.6195 | `gate_promotion_eval.json` | 일치 (M3 hand/cnn과 동일 팔) |
| BS 패치 대각 | 15.9 / 16.3 / 22.1 / 19.7 m | `sa_multirecord_mmse.json` L_diag | 일치(~14–22 m) |
| fan 가능 레코드 | YS17/AA15a/AA14/AA15b/LJ18g | `d3_falcon_real_operator.py` RECORDS | 일치 |
| G1 enc13/1600 | +0.508 / +0.527 | `LIMITATIONS_ledger` B1 (문서) | 문서 값 채택 |
| 속박파 물리/CNN | 0.195 / 0.111 | `PERSONA_DEBATE_cnn_outdated` R1 (문서) | 문서 값 채택 |

**발견한 문제 / 주의:**
1. **nowcast 상대 % 불일치(경미):** 프레임에 쓰인 "+36 %"는 절대 창-짝
   +0.143을 상대화한 근사값이다. 재계산 시 정의에 따라 달라진다 — 8 풍파
   레코드 record-level common_unobs = +40 %, 10 레코드(폭풍 포함) = +37 %.
   규칙(불일치 시 JSON 채택)에 따라 **절대값 +0.143 (t 4.42, 8/8)을 1차
   인용**하고, 상대는 "~+36 %"로 표기하되 정의 의존성을 각주 처리 권고.
2. **창-짝 절대 미세 불일치:** `cluster_bootstrap` A의 +0.1428은 내
   독립 재계산(common_unobs 영역, +0.121, 36/47창)과 소폭 다르다 — 영역
   키/포함 창 정의 차이로 추정. 규칙대로 committed JSON 값(+0.1428, 39/47
   [HANDOFF §7])을 채택.
3. **트랙 팔 = M3 CNN 팔 확인:** `gate_promotion_eval.json`의 hand/bw_s0_head
   (0.5320/0.6195)가 `m3_point_eval.json`의 hand/cnn과 정확히 일치 —
   즉 B-9의 "CNN 현직"과 B-10 층③의 "head(learned)"는 동일 배치. 교차
   일관성 확인됨(오류 아님, 문서에 반영).
4. **BS 패치 "14–20 m":** 실측 L_diag는 15.9–22.1 m로, 임무 명세의 "14–20 m"는
   근사. 표에는 실측값(~16 / ~20–22 m)을 사용.
