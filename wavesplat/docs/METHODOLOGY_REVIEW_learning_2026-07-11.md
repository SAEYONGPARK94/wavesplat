# 학습 방법론 총검토 — 잠긴 방향("무조건 학습") 하의 후보 6종 GO/NO-GO

날짜 2026-07-11 밤 · 사용자 지시("설계 외에도 방법론을 더 검토") · litcheck 병행
· 판정 기준: 100 ms 예산 · UQ/무해 보증 · 실측 전이 · 한계 원장과의 정합

| # | 방법론 | 선행 (litcheck) | 우리 차별점 | 판정 |
|---|---|---|---|---|
| M1 | **비평가-선택기** (커밋 512fba0, 실행 중) | selective prediction/L2D 프레임워크 확립; MRIP 신뢰도; ★인접: 위상분해 파랑예보 불확실성-유도 지평 (JMSE 2025) | 기권이 아니라 **인증 후보-가족 선택**, 라벨=조우-트랙(창-밖), 실측 zero-shot, 최악 보증(B1) 유지 | **GO — 진행 중** |
| M2 | **L4 결정-지향 학습** (마루 도착시각·제어 손실) | loss-calibrated inference(2201.03128), decision-focused learning(SPO 계열) 확립 | wave-by-wave 제어 손실로 위상분해 예보 학습은 미개척 추정(실행 시 재검증); posterior-mean 비최적성은 정리 = 방 보장 | **GO — 다음 수** |
| M3 | **점군-native 연산자** (88k 점 → 예보; L3 현대화) | CloudLSTM, point-cloud neural operator(PDE), 파-구조 상호작용 점군 복원(OE 2025), Ehlers/PINO는 격자 기반 | 원시 시공간 점군 직입력 + 고정-T 작동점; 단 H7이 시뮬 표고-정확도 방 ~0 증명 → 승부처는 **계산(88k↔100ms)과 실측**뿐 | **HOLD** — M1·M2 뒤 공정성 팔로 |
| M4 | 신경과정/TNP amortized posterior | AHGP 선행 + **내부 음성**(T-NSE input-invariant 붕괴) | — | **NO-GO** (문서화됨) |
| M5 | 생성 prior 고속화 (DPS→flow matching/consistency) | DPS 내부 양성(HOS 승)·느림; consistency 계열로 100 ms 가능성 | UQ/posterior 샘플 3-제품 체계의 배포판 — 정확도가 아니라 **제품 라인** | **HOLD** — UQ 챕터 보조 |
| M6 | 적응 관측 정책 (폐루프) | active sensing 확립 | H4: 단발 headroom 얇음 + 주어진-센서 제약(지향 고정) | **NO-GO** (논문 범위 밖) |

## 우선순위 (고정)

**M1 (실행 중) → M2 (다음 수) → M3 (공정성 팔 + 계산 서사)**, M5는 UQ 보조.
셋 다 "학습이 척추"이되 자리가 다르다: M1=검증·선택, M2=목적함수, M3=표현.
한계 원장 정합: M1은 B9/B7의 처방, M2는 지표 전제(MSE)를 깨는 유일 셀,
M3는 A5(시간 개구)·계산 제약에서만 이길 수 있음을 사전 명시.

## litcheck 메모 (novelty 문장용)

- selective prediction/L2D: [survey](https://www.emergentmind.com/topics/selective-prediction),
  [selective recalibration](https://arxiv.org/html/2410.05407v1),
  [L2D top-k](https://arxiv.org/pdf/2505.10160) — 프레임 선점, 도메인 조합은 열림
- ★인접 최근접: [Uncertainty-Guided Prediction Horizon, phase-resolved
  (JMSE 2025)](https://www.mdpi.com/2077-1312/13/7/1250) — **정독 필수**;
  차별점 후보 = 지평 조절 vs 후보-가족 선택+게이트, 합성→실측 zero-shot
- 점군 예보: [CloudLSTM](https://arxiv.org/pdf/1907.12410),
  [파-구조 점군 복원 (OE 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0029801825019079)
- 결정-지향: loss-calibrated EP(2201.03128) + SPO 계열 — M2 실행 시 정밀 검증
