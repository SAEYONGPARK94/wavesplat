# C2 사전등록 — envelope-게이트 제어 데모 (결과 前 동결) 2026-07-22

> 방향(페르소나 수렴): envelope의 검증 = 통계 skill(§8, modest) 아니라 **제어 결정가치.** 이 문서는 **결과 보기 전**
> 동결. 규율([[feedback_direction_flip_discipline]], [[feedback_validation_nonlinearity]]): 지표·분할·가설·양방향 발표 사전고정.

## 1. 제어 과제 (NAOE-native)
**예측적 heave 보상** (heave-compensated crane/gangway/helideck의 축소판): 물체가 body 위치 (x,y)에서 예보 지평
(leads = 예보 선행시간) 동안 파-유도 heave η_true(x,y,lead)를 겪는다. Feedforward 제어기가 phase-resolved 예보
η_hat로 액추에이터를 선-보상: **잔차 heave = η_true − g·η_hat.** 목표 = **잔차 분산(=느끼는 운동) 최소화.**
- 물리: η_hat이 신뢰할 때 g↑(적극 보상), 신뢰 못 할 때 g↓(안 하느니만 못함 회피). **틀린 예보로 full 보상은 해악**
  (ρ<0.5서 잔차 분산 > 원 분산). ⟹ **envelope(σ_CRB, truth 없이 아는 신뢰도)이 gain을 조절.**

## 2. 제어기 (query point (x,y,lead)마다 gain g 정책)
- **naive-full**: g=1 (예보 항상 완전 신뢰).
- **★naive-best-const (주 baseline)**: g = 단일 상수, TRAIN서 총 잔차분산 최소화로 tune (최선의 비적응 제어기).
- **envelope-gated (우리)**: g = g(σ_CRB). 사상 σ_CRB→g를 **TRAIN 바다서 보정**(단조/Wiener 형: g=v/(v+σ_CRB²·s),
  v·s는 train fit), **TEST 바다에 적용.** truth 미사용.
- **oracle (상한, 비현실)**: g = point별 Wiener 최적 = cov(η_true,η_hat)/var(η_hat)의 국소추정, TEST-truth 사용.

## 3. 지표
- **R = 1 − Σvar(잔차)/Σvar(η_true)** (분산-저감), 제어기별, **TEST 바다서.** 클수록 좋음(R<0=해악).
- 분산은 body 위치별 leads(지평) 걸쳐. (각 위치 = 지평 시계열; ensemble = 위치×바다.)

## 4. 가설 (사전등록·단측)
- **주 H1**: **R_gated > R_naive-best-const** (envelope 적응이 최선 상수 이김), TEST서. 
  보고 = Δ = R_gated − R_naive_const + **부트스트랩 CI**(바다-블록).
- **부 H2**: oracle-gap-closed = (R_gated−R_naive)/(R_oracle−R_naive) ∈ [0,1] — envelope이 oracle 적응의 몇 %?
- **부 H3**: R_naive-full 보고 (틀린-창서 full 보상 해악 정량 = envelope의 방어가치).

## 5. 분할·검정력
- 바다: SEAS[:N], **TRAIN=짝수 idx, TEST=홀수 idx** (사전고정). 보정·best-const는 TRAIN만, oracle는 TEST-truth.
- N≥16 바다 목표(TEST≥8), 위치 ≥300/바다 → 지평 걸친 충분 표본. 부트스트랩 = 바다-블록 2000.

## 6. 판정 규칙 (양방향, 사전고정)
- **WIN**: Δ>0 & 부트 CI 하한>0 (2000, 바다-블록). ⟹ **결정-가치 헤드라인 인가**(envelope이 제어 개선).
- **NO-WIN**: Δ≤0 또는 CI 0 포함. ⟹ **정직 폴백 = C1(공통천장) 헤드라인 + envelope=통계 UQ**, 결정-가치 미주장.
- 어느 쪽이든 발표 (음성도). oracle-gap로 "천장이 낮아서 무이김"인지 "envelope이 약해서"인지 구분.

## 7. 급소 (사전 인지)
- σ_CRB rank-corr(§8 ρ0.30)이 약하면 gated≈best-const 위험 — 단 **E-59 LEVEL 보정(level-ρ 0.89~1.00)**이 gain 스케일
  설정엔 충분할 수 있음(rank보다 level이 gain에 중요). 이게 H1의 실낱 근거.
- 循環 방지: 보정 TRAIN-fit·TEST 적용. best-const도 TRAIN-fit ⟹ 공정(gated는 상수 넘는 신뢰정보 있어야만 이김).
- inverse-crime 방지: **HOS 비선형 truth**(합성 아님), η_hat은 RELAX 적합(오라클 θ 아님).

## 8. 산출
`tmp/c2_control_demo.py` → `results/direction3/c2_control_demo.json`. 결과 후 헤드라인 확정.
