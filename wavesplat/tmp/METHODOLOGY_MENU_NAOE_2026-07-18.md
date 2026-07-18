# 방법론 메뉴 — 조선해양공학과(NAOE) 적합 방법 프레임 (2026-07-18, litcheck 접지)

> 사용자 방향수정: "정보벽 논문은 NAOE 주제로 부적합 — 방법론 더 검토 필요." 진단/음성 프레임 → **방법 기여** 필요.
> 방법론 검토(fable, WebSearch 9회) + 코디 flip. [[THESIS_FRAMING_v2]]·[[PAPER_LOGIC_structure]] 갱신 방향.

## 확정: 정보-채널 회계 단독 = NAOE 최하 적합 (사용자 판단 litcheck 확인)
진단/음성-결과·신호처리/응용수학 언어. **폐기 아니라 "envelope 내부 엔진"으로 강등-재배치.**

## litcheck — 방법 후보별 선점 상태
| 후보 | 상태 | 근거(선행) |
|---|---|---|
| A 시공간 분산-융합 GP | **방법 사망**(vehicle 생존) | Ehlers OE2023·SWAN-Tucker OE2026·ensemble DA JFM; 커널=파랑 공분산=DSWP-DA 선형MMSE |
| B B-2 task-loss 학습 | **좁은 진짜 틈** | DFL 일반론 확립(LODL'22 등)이나 파랑-추정 인스턴스 미발견; 경쟁=end-to-end 선체운동 NN(PoF2023) |
| C 배포 시스템 | **사망(치명 결격)** | FutureWaves 상용·sea-trial 有 vs 우리 실물센서 0회 |
| D 해류-강건 kp | **사망** | NOAA Doppler보정·dispersion 해류fit 표준 + 사내 맹목복원 5/5 실패 |
| E predictable/운용성 envelope | **틈 有(좁아지는 중)** | 개념=NAOE 정통(Naaijen/Desmars); **근접선행 2건**(Applied Energy 2022 VB-UQ·2025 universal zone·JMSE2025) — 전부 **고정프로브·경험적 NN-UQ**. 살아남는 틈=**센서-인지+이동궤적+희소점군+CRB-앵커** envelope |

## NAOE 적합도
E 상 · B 중상(제어전달 t−4.02로 seakeeping 접지) · A 중(vehicle) · C 상이나 결격 · D 중 · 회계단독 하.

## ★권고 골격 (1+2 혼합)
1. **[헤드라인] 센서-인지 예측가능/운용성 envelope 방법** (E, A=vehicle): 주어진 센서 사양·궤적·해상에서
   phase-resolved 예보의 신뢰 시공간 영역을 **계산·보정(calibrate)·판별**. D_eff=L+c_g·T_obs 확장 + CRB 앵커 +
   사후분산 보정(피복 90→91% 실증). **difference-first**: Naaijen(기하만·완전관측) / Applied Energy(특정 NN
   경험 UQ·고정프로브) / 우리(sensor-operator-aware + bound-anchored + calibrated + 이동-희소점군).
2. **[장3] B-2 task-loss 학습**: 우도-실패 판정이 동기 된 비-우도 task-loss로 미분가능 물리-사후 관통, 추정기를
   제어가치에 정렬. 차별=물리 추정기 유지(vs PoF2023 end-to-end)+검증사슬(군집·시드·HOS·제어전달·zero-shot).
- **정보-채널 5요소 회계 + CRB 판별 = envelope의 내부 엔진**("왜 envelope이 거기 있나"=정보한계 vs 효율갭).
  같은 결과수치 전부 재사용, headline만 진단→방법 flip.
- **차선(보험)**: 방법 delta 붕괴 시 ADVISOR_MEMO 선택지 A(system+science) — 단 교수 명시 수용 선결(v1 지적 재현).

## ⚠ 커밋 前 관문 (규율: 검증 전 확정 금지)
**Applied Energy 2025 · JMSE 2025 원문 정독**으로 1위 delta 확정 (fable은 초록만 확인). 이동센서+CRB앵커까지
했다면 delta 붕괴 → B-2 유일 방법축(골격 얇음 → 교수 합의). **이 관문 통과 전 "envelope 방법" 확정 인용 금지.**

## 남은 실험 (1위 채택 시)
envelope 예측 vs 실제 skill 사전등록 검증(regime·HOS 포함)·보정 피복 재확인·스테레오 실측 anchor 1건·
근접선행 2편 직접 대비. "LiDAR" 명명 금지→"sparse moving elevation observations". 틈 시간민감(2022→2025 좁아짐).

## URL (litcheck)
Ehlers OE2023 S0029801823024435 · SWAN-Tucker OE2026 S0029801826006207 · ensemble DA arXiv2012.00677 ·
Desmars OE2020 S0029801820306557 · radar sea-trial S0029801820303395 · JMSE2025 mdpi 2077-1312/13/7/1250 ·
JMST2021 predictable-zone s00773-021-00830-0 · LODL NeurIPS2022 arXiv2203.16067 · PoF2023 end-to-end
pof/35/9/097127 · FutureWaves futurewavesradar.com · Applied Energy 2022 S0306261922010042 · 2025 S0306261925016265
