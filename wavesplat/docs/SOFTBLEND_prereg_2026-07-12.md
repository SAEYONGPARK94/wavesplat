# 사전등록 — 소프트 결합 셀 + 트랙 RMSE 감사 (2026-07-12)

> **동결 시점**: 60창 트랙 RMSE·결합 팔 수치를 보기 전. (8레코드 나우캐스트
> 축의 nrmse_bl은 판정 전 감사에서 이미 열람 — net 6/8 승 — 이 문서의 판정
> 대상이 아님을 공시.)

## 동기

1. 사용자 관찰(2026-07-12): "RMSE가 보기에 hand가 더 좋아 보인다" — 60창
   조우-트랙 축에서는 RMSE가 저장된 적 없음. 검증 필요. 가설: learned의
   진폭 수축(0.61× vs hand 0.73×)이 트랙 RMSE에서 비용일 수 있다
   (nRMSE² = 1 + a² − 2aρ; a=진폭비, ρ=corr).
2. 선택 상금 실재(+0.107 전체/+0.145 나쁜 창, critic_v2_judge 재도출)하나
   비평가 선택은 2회 FAIL로 폐쇄. **고르지 않고 섞기**(무학습 고정-가중
   결합)는 미판정 인접 셀 — 규칙 3(무학습 경쟁자 먼저)의 정공법.

## 프로토콜

`d3_gate_promotion_eval`의 동결 60창(5레코드 × 12) verbatim 재구성(동일
창 시작·관측 rng·시드·kp_hint). 예측은 protocol 질의점 Pq(1 s 리드,
lead ≤ 10 s, finite gt)에서 산출. 팔 예측 벡터는 JSON에 저장(재도출 가능).

## 팔

| 팔 | 정의 | 역할 |
|---|---|---|
| hand | mmse_gpu(kp_hint) | 기존 물리 팔 |
| learned | net_gamma_mmse + smult_head (bw_s0) | 현직 |
| **blend50** | 0.5·hand + 0.5·learned | **주 판정 팔 (동결)** |
| blend25/75 | w·hand+(1−w)·learned, w∈{0.25,0.75} | 진단 전용(판정 비대상) |
| oracle-select | 창별 max(hand, learned) | 회계 천장(선택) |
| oracle-blend | 창별 최적 w∈[0,1] (corr 기준) | 회계 천장(결합) |
| amp-oracle | 창별 RMSE-최적 스케일 a\*=⟨p,t⟩/⟨p,p⟩ (각 팔) | RMSE 재교정 회계 천장 |

모든 결합·스케일 팔은 이미 산출된 두 예측의 결정론 함수 — 학습·선택 누출
없음. w 스윕과 oracle은 회계 전용이며 판정에 쓰지 않는다.

## 지표

- **corr** (주 지표, 프로젝트 표준·창-수준 짝지음)
- **track nRMSE** = RMS(pred−truth)/RMS(truth), protocol 점에서.
  truth 비중심화(표고 ~0-평균 공시). 부 지표.

## 게이트 (동결 — 숫자 보기 전)

- **G-SB1 결합가치(corr)**: 짝지음 Δcorr(blend50 − learned) 전체 60창
  평균 **≥ +0.03 AND t ≥ 2.5**.
- **G-SB2 무해**: hand-좋은 창(hand ≥ 0.5, n≈35) Δcorr(blend50 − learned)
  평균 **≥ −0.01** AND 레코드별 평균 Δ의 최악값 **≥ −0.05**.
- **G-SB3 RMSE 무비용**: 짝지음 ΔnRMSE(blend50 − learned) 전체 평균
  **≤ +0.005**.

**판정 분기 (동결)**: 3게이트 전부 PASS → blend50을 현직 후보로 승격하되
즉시 채택 금지 — 훈련 시드 s1/s2의 learned로 재현 확인 후 승격(G-H 방식).
하나라도 FAIL → 소프트 결합 셀 폐쇄, "선택 상금은 섞기로도 미회수"를
수치와 함께 기록하고 L4로 이동.

## RMSE 감사 (게이트 아님 — 서술 판정, 사전 질문 동결)

1. 트랙 nRMSE에서 hand vs learned 승수·평균차(60창 짝지음) — 사용자 관찰의
   사실 여부.
2. 지면 어디서: Δcorr>0인데 ΔnRMSE>0(corr 승·RMSE 패)인 창 수, a vs ρ
   분해로 진폭 수축 몫 정량.
3. amp-oracle 회계: 창별 최적 스케일로 재교정 시 각 팔의 nRMSE 하한 —
   "learned의 RMSE 열세(있다면)가 재교정으로 사라지는 양"을 수치화.
   배포-가능 재교정(레코드-인과 게인)은 **이번 판정 아님** — 회계만 하고,
   회수분이 ≥0.02면 후속 사전등록 대상으로 기록.

## 산출물

`results/direction3/softblend_rmse_eval.json` (창별 행: 팔별 corr/nRMSE/
진폭비 a + 예측 벡터), 판정 로그, 본 문서에 판정 결과 추기.
