# 세션 마스터 정리 — 2026-07-22 (Opus×fable, 전체 종합)

> 하루 세션 전체(브레인스톰→열린실마리 전건→kp-robust설계+방법리뷰→궤적/SBI→⓪①②)의 **주제별 정리.**
> 규율 준수(사전등록·이중검증·MEASURED/ARGUED·대칭감사). 모든 값은 세션 내 검증됨. 파일 인덱스는 §7.

## 0. 한 줄 (HOS-준수 최종판)
> **학습 "돌파" positive를 전부 realistic/HOS서 검증 → B-2·궤적·①better-net 전부 oracle-washout 또는 inverse-crime
> synthetic 아티팩트로 사망(HOS 확정). 살아남는 학습주장 = 겸손·HOS-검증(현 net-MMSE가 HOS서 classical 이김,
> 0.79 vs 0.60). thesis 척추 = spine A(CRB-앵커 판별+정직음성+HOS 학습>classical). "더 나은 net" 불필요·불가.**
> ★HOS 재검(사용자 지시)이 ① inverse-crime 확정: synthetic "+0.45 헤드룸"=oracle-LS 1.0 컨닝 인위산물, HOS 실제
> 천장 0.52서 net 이미 위(room −0.27). "net이 좋아진"게 아니라 판타지 천장→현실 천장. 권위 `tmp/HOS_LADDER_RESULT_2026-07-22.md`.

## 1. ★핵심 성과 = C8 (A 척추 강화)
**"estimator-independent operability envelope"** — 브레인스톰서 나·fable이 독립적으로 지목한 **#1 과대주장**
(헤드라인엔 있으나 E-59가 "무증거"라 명시)을 **직접 시험·해소**.
- e58 sandbox서 σ_actual 추정기만 3클래스 교체. **under-det 레짐(운용상 중요)**: σ_CRB이 ridge **+0.72**·relax
  (다른 클래스) **+0.68** 예측·상호 **+0.55** 동의·거리보정 후에도 강함(pd +0.54/+0.57). well-det선 약함.
- **판정 = 한정형 SUPPORT**: "ML twin만 예측" 반증(relax 강예측) + 무조건형도 반증(단일모드 GP 미예측) ⟹
  **"센서 파수정보 resolve하는 추정기 클래스에 독립, under-det서 강함"**(무조건형 인용 금지).
- **이중검증 2/2**: 독립집계+LOO+regime-MW / **fable 독립 재구현 CONFIRM**(자체 OMP, 상호동의 +0.506 일치).
- ⟹ **E-59 최대 미검증 갭 봉인 = A 척추 실질 강화.** (파일: C8_RESULT·F6_c8_verify)

## 2. ★반복 패턴 = 학습 positive의 oracle-washout
| positive | oracle 조건 | realistic 조건 | 판정 |
|---|---|---|---|
| **B-2 loss-swap** (실패창 T−L) | +0.139 (참 스펙트럼 피크 먹임) | **+0.022** (beamform kp) | **oracle-inflated** |
| **B-2 재훈련 gate** (M-3G) | — | 훈련 kp가 **처음부터 beamform**(생성코드 확인) ⟹ **재훈련=no-op** | **회복 불가** |
| **궤적 제어** (M-3G류) | CEM/learned가 stationary 이김(+0.02, far-field 개구이득 실재) | **소멸**(추정 파수, CEM 동률·learned 열위) | **oracle-washout** |
| **궤적 살리기** (②) | — | near-obs 회복도 실패, 퍼진 obs가 파수추정 악화 | **FAILED** |
- **기제 공통**: 이득이 **참 파수(oracle)** 를 아는 데 의존. realistic 추정(beamform/RELAX)이면 소멸.
  ⟹ **B는 헤드라인 아님·별도 챕터 확정, "+0.139" 인용금지→"+0.022"(beamform).** (M-3·M-3G·TRAJ·TRAJ_salvage)

## 3. 학습 지형 지도 (fable ground-up 리뷰) — dead-end 회피 참조
- **~40 아키텍처 이미 시도.** **패턴**: FAIL=전부 "raw 표현→점추정"(다봉 posterior MSE최적=평균=mean-collapse,
  ≥15건) / SUCCESS=전부 "네트워크는 저차원 **물리-정규화 prior만**, 재구성은 **닫힌형 물리 solve**". ⟹ 관건=목적함수/분해.
- **레짐별 벽**: well-det=**진짜 정보벽**(≥15확증, 네트워크 무의미) / under-det=**알고리즘 갭**(학습 이김) /
  센서궤적=CRB상 wall-mover였으나 **realistic서 죽음(위 §2)**.
- **★정정**: "set-diffusion이 RELAX 전승"은 초기 d43 수치, 최종검증선 **비학습 baseline과 동률**(0.234 vs 0.233).
  정확도 유의격파 유일 사례 = **d56 fast amortized posterior**(실시간 제약 안에서만).
- **★3번째 범주**: 조우예보 0.10→0.90은 정보/표현 아니라 **검증-프로토콜 버그(n-독약: 창-안 held-out)** — 학습 없이 뛰었음.
  = B-2/M-3 창-안vs창-밖과 같은 축(대칭경고).
- **재고금지(이중선점)**: FNO(Ehlers OE2023 벤치 FNO승)·PINO(같은 그룹 2025-08)·SIREN·점군 transformer·track-critic.
  (파일: GROUNDUP_methods_review·SYNTHESIS_design_review)

## 🛑 4-CORRECTION (세션 말미, fable red-team+Opus 확인) — §4 아크 전체 **철회·미검증**
아래 §4 및 파수-ID CRB/challenge/better-net 결과 **인용 금지**: (a)**inverse crime**(obs=predict(θ)+noise, oracle가
동일 유한-atom fit ⟹ oracle 1.0·+0.45 헤드룸·realizable 인위부풀림, HOS 0=상시지시 위반) (b)**"15comp=swell" 거짓**
(n_components=이산화밀도, 스펙트럼 동일 Hs=1.2) (c)**pseudo-replication**(n=16 실제 n=5) (d)**sort 버그**(15comp만).
**better-net=FAIL**(supervised Δ−0.14; task-loss fine-tune 후에도 v3와 parity 0.559 vs 0.568, 무개선). ⟹ **"under-det
학습 forward path"=철회, HOS 재검 전 헤드라인 불가.** 살아남음=B-2·궤적 oracle-washout(B-2=HOS)·C8(synthetic 캐비엇)·
spine A. **4번째 낙관오독**(유일생로 서사가 HOS지시 가림, fable가 잡음). 권위 `tmp/BETTERNET_fable_2026-07-22.md`.

## 4. ~~★under-det 갭 = 파수 ID (오늘 유일 생존 학습축)~~ [철회 — 위 4-CORRECTION 참조]
sparse under-det 바다 재구성 사다리(n=16): ridge 0.12 < RELAX 0.39 < **net-MMSE 0.55** < **oracle-LS(참 파수) 1.00 ≈ CRB천장**.
- **갭 전체 = 파수 식별**(참 파수면 완벽재구성). 학습(net-MMSE)이 RELAX보다 +0.16, 완벽-ID까지 **+0.45 헤드룸**.
- **★oracle-의존 아님**(net-MMSE는 beamform) = B-2/궤적과 대조되는 **진짜 비-oracle 학습 기회.**
- **★realizability 해결 = REALIZABLE**(파수-ID CRB, n=16): (1)파수 식별가능(각도 CRB 0.5°≪성분간격 2.6°, ratio
  0.21) (2)참 파수집합이 유일 최적(잔차 0.018≈잡음, RELAX 5× 열위 = 우도 단봉). ⟹ **헤드룸=비볼록 탐색의 알고리즘
  실패, 원리상 학습 식별기가 실현 가능 = 오늘 유일 CRB-지지 forward path.**
- **★도전(갭 닫기 시도, n=16)**: RELAX 0.40 → off-grid refine 0.51(탐색 +0.11) → net-MMSE 0.57(refine 미달) →
  EM 무이득 → oracle 1.00. **⟹ quick levers(off-grid·EM)론 못 닫음, 닫는 길=더 나은 LEARNED 식별기(실질 연구).**
  **CRB이 천장 도달 가능 보증 ⟹ B-2·궤적(oracle-washout)과 결정적 차이 = 논문 "학습 기여"의 정직한 자리.**
  ⚠"원리상 realizable"≠"시연"·일부 marginal·synthetic 낙관(실측 천장<1). (파일: GAP_LADDER·WAVENUMBER_ID_CRB·CHALLENGE_RESULT)

## 5. ★인용 정정 / 집필 전 문서청소 (확정 리스트)
1. **"estimator-independent" → 한정형**(C8) 전 문서 재서술.
2. **§4↔§5 "하나의 엔진" 재서술** — F-2: §4 envelope-CRB(d1_crb 공유)와 §5 파티션-CRB(e51, 별개구현)은 **다른 엔진**.
3. **다채널 조건② = 이미 CLOSED**(F-3, E-55/56) — 새 실험 불필요, THESIS_FRAMING_v2 §9/10만 stale, 문서정합 4건.
4. **§8을 level-calibration 1차훅으로 재구성**(F-5, MEASURED 재도출 level-ρ 0.89~1.00).
5. **"census ≥4λ 3개" 인용정정 → YS17 유일**(M-4, 실측 under-det는 YS17 상한, 이동 LiDAR만 경로).
6. **RAW_SINK 훅 버그**(F-3·F-5서 json 미덤프 — 재실행으로 우회).
7. **B-2 "+0.139" 인용금지 → "+0.022"(beamform-eval)**·F1 "훈련=oracle kp" 정정(M-3G). E-B2K 아레나=M-3로 재판정.
8. **⓪ keystone③ CRB "단위버그" = false alarm**(경험검증, e37:475이 deg2rad 저장) — 인용 변경 없음, keystone③ 정상.

## 6. 정직한 현 위치 + 권고
- **A 척추(CRB-앵커 운용성 envelope) = CONDITIONAL PASS·오늘 강화**(C8 봉인). keystone③ 정상(⓪), 단 GUARD2
  (achievability/random-phase) 여전히 미납(SBI는 미보정으로 INCONCLUSIVE — 보정된 NPE 필요).
- **B 챕터(loss-swap) = 헤드라인 아님 확정**(oracle-washout, M-3/M-3G) — 별도 챕터·재훈련 관문·"+0.022" 인용.
- **학습의 미래**: 유일 생존축 = under-det 파수-ID 갭(realizability 미해결). 궤적·센싱설계축은 realistic서 죽음.
- **★권고**: 학습 지형 철저 매핑 완료 ⟹ **최고가치 = spine A 논문화**(오늘 음성들이 "학습 천장=oracle-한계"를
  반복확인 = 정직한 음성 자산) + 지도교수 difference-first 합의. 학습 더 밀면 값싼 다음관문=파수-ID CRB.

## 7. 프로세스 교훈 (규율 작동)
- **discipline harness가 예비 낙관 오독 3+건 차단**: ①궤적 n=2 smoke "생존"→full n=24 반전 ②SBI 넓은-posterior
  "CRB 반증?"→calibration이 NPE 미보정 적발 ③CRB "단위버그"→경험 before/after가 false-alarm+정상코드 파괴 방지.
- **fable red-team이 net-MMSE FAIL의 OOD confound 적발** → 깨끗한 RELAX 재검 유도(그래도 FAIL=진짜).
- **양방향 교정**: 내 비관오류(M-3 FAIL 오선언 아님, 여기선 낙관오류)도 잡힘. 코드추적만으론 fable·Opus 둘 다
  false-positive였고 **경험검증이 진실**. ⟹ [[feedback_direction_flip_discipline]]·[[feedback_dual_verification_default]] 재확인.

## 8. 파일 인덱스 (repo tmp/, 오늘)
- **브레인스톰**: BRAINSTORM_opus_fable
- **C8**: C8_prereg·C8_RESULT·F6_c8_verify
- **B-2 realism**: M1_RESULT(kp↔hand)·M3_RESULT(센서-kp FLAG)·M3G_RESULT(gate VOID)·M3G_verify·M3G_redteam
- **fable F-1~5**: F1_gaussianity·F2_crb_engine·F3_multichannel·F4_b2_mechanism·F5_level_calibration
- **M-4**: M4_ge4lambda_finding
- **설계/리뷰**: DESIGN_kp_robust·GROUNDUP_methods_review·SYNTHESIS_design_review·EXECUTION_SYNTHESIS
- **궤적**: TRAJ_gate_prereg·TRAJ_redteam·TRAJ_RESULT·TRAJ_salvage_RESULT
- **SBI**: SBI_npe_crb_verify·SBI_RESULT_summary
- **under-det**: GAP_LADDER_RESULT
- **메모리**: project_brainstorm_next·project_execution·project_methods_landscape (2026-07-22)
