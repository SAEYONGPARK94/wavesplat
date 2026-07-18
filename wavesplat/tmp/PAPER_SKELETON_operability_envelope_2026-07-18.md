# 논문 골격 초안 — 센서-인지 CRB-앵커 운용성 Envelope (NAOE) 2026-07-18

> fable 박사심사 = **CONDITIONAL PASS**(조건 7개, 관건=envelope skill 검증 실험 미실행). 사용자 "괜찮으면
> 추천대로 작업" → 골격 초안. **규율 적용**([[feedback_direction_flip_discipline]]): 각 절 내용을 **MEASURED**
> (검증됨) / **PENDING**(조건 실험) 태그. 헤드라인은 검증 전까지 **설계(keystone-to-be-sealed)**로 표기.

## 제목 (후보)
"A Sensor-Aware Operability Envelope for Phase-Resolved Wave Forecasting from Sparse Moving Elevation
Observations: An Information-Theoretic (Cramér–Rao) Approach"
- ⛔ "LiDAR" 금지(미관측) · ⛔ "information wall" 금지(NAOE 부적합). 얼굴 = **운용성 지도**, CRB = 엔진.

## Abstract (fable 초안 기반)
선상 희소 이동 표고 관측으로부터의 phase-resolved 파랑 예보에서, 예보 신뢰 시공간 영역(**운용성 envelope**)을
학습 추정기의 경험적 불확실성이 아니라 **센서 사양·궤적·해상에 대한 정보이론 하한(Cramér–Rao)에 앵커하여
계산·보정·검증하는 방법**을 제시. envelope의 내부 엔진 = 보조-정보 채널을 5요소(가치·회수성·공급·달성성·
정보한계)로 회계하고 CRB로 "회수 실패가 정보벽인가 추정기 효율인가"를 판별하는 프로토콜. 두 채널(레짐
파티션·표면 해류)에 적용해 **채널마다 다른 사망요인 판별**을 실증. 보완 = 우도-실패 판정 동기의 비-우도
task-loss로 추정기를 제어가치에 정렬(제어전달 t=−4.02). 모든 판정 = 사전등록·군집보정·HOS 재현·대칭감사로
봉인. 스코프 = 모사 센서 연산자(스테레오 실측 anchor 병기).

## 절 구조 + MEASURED/PENDING 태그

**§1 Introduction** — phase-resolved 예보의 운용 질문("언제·어디서 믿나"). gap: 기존 envelope = 기하-해석적
(Naaijen) 또는 경험 ML-UQ(Zhao), **정보이론 앵커 부재**(litcheck 확인, MEASURED).

**§2 Related work / difference-first** — DSWP·predictable zone·ML-UQ. 포지셔닝: "그들 = 추정기-종속 경험 UQ /
우리 = 추정기-독립 정보이론 하한, 정보벽 vs 효율갭 판별 가능"(MEASURED: litcheck). ⚠경쟁 Zhao 그룹·시간리스크.

**§3 Method — 추정기(vehicle)** — 분산제약 스펙트럴 GP: z=Φw+noise, ω=√(g|k|tanh), 사후평균 장 = 선형 solve.
prior γ(k)는 빔포밍+학습(MEASURED: 확립된 기계, novelty 아님·difference-first만).

**§4 Method — ★CRB-앵커 운용성 envelope (기여)** — FIM=(1/σ²)JᵀJ, D_eff=L+c_g·T_obs, envelope=CRB<임계 시공간
영역, 사후분산 보정. (MEASURED: CRB 백본 direction1_crb·2클래스 FIM 성분급+파티션급 E-51/52 / **PENDING:
envelope→skill 예측 검증 = 조건①**).

**§5 정보-채널 회계 프로토콜 (내부 엔진)** — 5요소 + CRB 판별(정보한계 vs 효율갭). (MEASURED: E-52 결과 B
판별 — regime은 정보벽 아니라 효율갭 +0.046, 정보잔여 +0.040).

**§6 채널 연구 (판별타당도)** — regime: 가치 +0.043(n=957)·blind≈net(arena X·Y, 전수열거)·achievability 사망
(부이급 부재). current: value 사망(kp-fix, radar급 무가치, base-robust). **두 채널 다른 사망요인 = 프로토콜
판별 실증**(MEASURED: E-42~E-56 3중검증 / PENDING: 조건② 개구 채널 추가 = 3채널 판정표).

**§7 B-2 task-loss 학습 (효율갭 닫기)** — 우도-실패 동기 비-우도 task-loss, 미분가능 물리-사후 관통, 제어정렬.
차별=물리 추정기 유지(vs end-to-end PoF2023)+검증사슬(MEASURED: 제어전달 t−4.02·군집·시드·HOS·zero-shot).

**§8 ★Validation (KEYSTONE — 조건①③, PENDING)** — envelope 예측 신뢰영역 vs 실제 phase-resolved skill
(regime·HOS 포함), 경험-UQ baseline(무보정 사후분산·이상적 VB-NN 재현) 대비, 보정 피복 재확인, **스테레오
실측 anchor 1건**. **이 절이 헤드라인을 봉인 — 미실행이 최대 리스크(fable).**

**§9 정직한 음성 + 방법론** — 사전등록·7규율(검정력 게이트·MEASURED/ARGUED·대칭감사)·척추 동결·스코프 정직화.
= "결과 믿을 수 있나"에 대한 실질 답(MEASURED: 방법론적 성숙).

**§10 Conclusion** — 성능은 정보-속박. **센서+채널마다 "추정기 만들지 vs 외부정보 공급할지" 결정규칙** =
운용성 지도(NAOE 산출물).

## ★학위 성립 조건 (fable 7개, 로드맵)
1. **[관건] §8 envelope skill 검증 사전등록 실행·통과** — 실패 시 헤드라인이 회계/진단으로 후퇴(NAOE 부적합).
2. **다채널 완주**: 개구 채널을 5요소+CRB 판별로 정식 서술(baseline-특이성 이식 前 사전등록). 3채널 판정표.
3. **실측 anchor 1건**(스테레오, 이동-센서 미검은 캐비엇).
4. **§7 random-phase 캐비엇 ARGUED→MEASURED**(값싼 후속).
5. **지도교수 논문 유형 명시 수용**(v1 "조합" 지적에 difference-first+판별타당도 선제, 문서화).
6. **속도**: envelope 논문(채널1+프로토콜) 6개월 내 투고(Zhao 창 방어).
7. **명명·스코프 규율**: LiDAR/information-wall 금지·배포주장 삭제 유지.

## 정직한 현 위치
- **논문-급 확정 / 박사-급 = 다채널 일반성 완주 조건부**(fable). 자산 = 저널 2~3편, 통합 서사 존재 = 박사 하한 만족.
- **단일 최대 리스크 = §8 검증 실패**. ⟹ 다음 실제 작업 = 골격 위에 **§8 envelope-skill 검증 실험 설계·실행**
  (조건①). 이게 통과하면 CONDITIONAL→PASS. **운명이 하나의 사전등록 실험에 걸림 = 건강한 상태(fable).**
