# 사전등록 — 궤적정책 완결 (d74 → 연속/HOS + net-MMSE 실현) 2026-07-22

> 사용자 "추천대로 수행". fable #1 = 유일하게 정보벽을 옮긴 축(학습 궤적정책 greedy ×2.5~4.4). 완결 = d74를
> 이 프로젝트 최대 함정(**이산→연속 붕괴**)과 **oracle→realistic estimator** 두 realism 축에 노출.
> [[project_methods_landscape_2026-07-22]]. 규율: 사전등록·이중검증·MEASURED/ARGUED.

## 배경 (코드확인)
- d74: 정책(stationary/straight-along/straight-cross/greedy/CEM/learned)이 **field-CRB 최소화**로 heading 선택,
  linear 8-comp 바다. Stage C 실현 = **RELAX + TRUE wavenumbers(oracle ID)** = "aperture 효과 격리".
- ⚠**핵심 관찰**: d74 achievability가 **oracle 파수** 사용 = M-3/B-2와 **똑같은 oracle 이슈**. ⟹ 궤적 우위도
  oracle 아티팩트일 위험. **realistic estimator(net-MMSE, beamform kp 비-oracle)로 재검이 필수.**

## 결정 질문 (staged)
- **Q1(Step 1, realism-b·값쌈)**: 궤적정책 achievability 우위가 **비-oracle 추정기(net-MMSE beamform kp)** 에서
  살아남나? (현 oracle-RELAX → net-MMSE 교체, linear 바다 유지.)
- **Q2(Step 2, realism-a, Q1 생존시만)**: HOS 비선형 바다(`d63` HOSSimulator 재사용) + net-MMSE서도 살아남나?

## 방법 (Step 1)
- d74 정책 계획(CRB-기반, learned=CEM 모방훈련) **그대로**. 각 정책 heading으로 fly_path→obs(x,y,t,z) 수집(linear 바다).
- **신규 achievability = net_gamma_mmse**(production net, beamform kp=kp_hint None, smult_fixed) 로 그리드 재구성
  → nRMSE vs linear GT. 기존 oracle-RELAX도 병기(대조).
- 에피소드 n≥24, headline speed. **모든 정책 동일 net·동일 obs 예산** → 상대비교 공정.

## ★사전 판정 규칙 (동결)
- **PASS(Step1)**: net-MMSE achieved nRMSE서 **제어정책(straight-cross/CEM/learned) < stationary** 유의 마진
  (Δ≥0.02 AND paired 부호 다수 AND 95%CI 0제외) ⟹ 궤적 우위가 realistic 추정기서 생존 → Step 2 진행.
- **FAIL(Step1)**: 마진 소멸(Δ<0.01 or CI∋0) ⟹ **궤적 우위=oracle-파수 아티팩트**(B-2와 동형) → 학습 궤적정책도
  realistic서 marginal, 정직 기록.
- **PARTIAL**: 그 외.
- **CEM vs learned**: learned가 CEM에 근접(achieved nRMSE 차 ≤0.01)해야 "학습이 planner 대체" 주장 유효.

## 사전 예측 (M-3 학습 반영, 중립)
M-3서 oracle-의존 우위가 realistic서 축소된 전례 ⟹ **Step1 washout(FAIL) 개인확률 ~50%.** 궤적은 CRB를
직접 옮기므로(정보 자체 생성) B-2(추정 효율)와 달리 생존 여지도 실재 ~50%. **억지 금지.**

## 캐비엇·이중검증
- net-MMSE는 특정 net 의존 → 상대비교만(절대 nRMSE 아님). smult_fixed 고정.
- **이산→연속 붕괴**(이 프로젝트 최대 함정, ≥3 사례)가 Step2서 발동할 수 있음 — 그게 최유력 반증 시나리오.
- fable 설계 red-team(사전) + 결과 독립감사(사후).
