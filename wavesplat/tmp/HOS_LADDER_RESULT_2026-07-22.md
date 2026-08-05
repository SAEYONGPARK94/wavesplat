# HOS 재검 결과 — ① inverse-crime 확인 + 학습의 진짜 HOS-검증 positive (2026-07-22)

> 사용자 "Hos포함해서 다시 봐야". fable inverse-crime 비판을 HOS(비선형·실제 z·oracle 없음)로 직접 재검.
> 스크립트 `tmp/hos_ladder.py`, 원자료 `results/direction3/hos_ladder.json`. n=8 HOS 바다(build_sea, ~5분/바다).

## 결과 (corr vs 실제 HOS 참 필드, n=8, oracle/inverse-crime 없음)
| 방법 | corr median |
|---|---|
| RELAX (classical) | 0.601 |
| **net-MMSE (v3, 학습)** | **0.788** ⟵ 최선 |
| net-MMSE (better-net cd_task) | 0.761 |
| rich 선형-fit 천장 (oracle 없이 최선 선형) | 0.516 |

## Q1: HOS서 학습가능 room? = **없음 (−0.273)** — fable 비판 정확 확인
- **synthetic ①: rich-ceil(=oracle-LS 1.0) − net = +0.33 room.** HOS: rich-ceil(0.52) − net(0.79) = **−0.27** = **net이 천장 위.**
- **부호 뒤집힘.** synthetic "+0.45 헤드룸"은 **inverse crime**(oracle-LS가 참 파수 컨닝 → 1.0)의 인위산물.
  HOS(실제)선 선형-atom 천장이 0.52뿐(비선형 model mismatch)이고 net-MMSE는 이미 그 위. ⟹ **학습가능 room 없음.**
- ★사용자 질문("net이 왜 좋아졌나") 답: **net 안 변함(같은 v3). 천장이 판타지(1.0, 컨닝)→현실(0.52)로 바뀜.**
  "+0.45 room"은 도달불가 판타지까지의 거리였음. (부수: HOS obs 600 vs synthetic 250 → 절대값 0.55↔0.79 비직접비교.)

## Q2: better-net(cd_task)이 HOS서 v3 이기나? = **아니오 (−0.027)** — 철회 확정
comp-diverse better-net이 HOS서 v3보다 근소 열위. synthetic parity와 정합. **"더 나은 net" forward-path 사망 확정.**

## ★그러나 진짜 HOS-검증 positive (살아남음)
**현 학습 net-MMSE(v3)가 HOS서 classical을 확실히 이김**: 0.79 vs RELAX 0.60 vs 선형천장 0.52 (전 8바다 일관).
= **"학습된 물리-정규화 prior가 실제 비선형 바다서 classical 추정을 이긴다"의 HOS-검증 실물 근거.** synthetic 아니고
inverse crime 아님. **spine A의 "학습이 이기는 곳" 주장을 HOS로 뒷받침.** (단 "더 나은 net으로 큰 갭 닫기"는 아님 —
net은 이미 realistic 천장 근처.)

## ⟹ 세션 최종 (HOS-준수·정직)
- **① "better net으로 +0.45 닫기" = 사망**(inverse crime 인위 헤드룸, HOS서 −0.27, better-net 무이득). 상시지시 이제 준수.
- **B-2·궤적·① = 전부 realistic서 washout/synthetic 아티팩트.** 학습 "돌파" 서사 전부 사망.
- **살아남는 학습 주장 = 모듈적·검증됨**: 현 net-MMSE가 HOS서 classical 이김(0.79 vs 0.60). = spine A 정합, 정직·겸손·HOS-검증.
- **thesis 척추 = spine A**(CRB-앵커 판별 + 정직한 음성 + HOS서 학습>classical). **"더 나은 net" 안 필요·불가.**
- 교훈: 사용자의 "HOS로 다시 보자"가 정확했고, inverse crime의 정체를 HOS가 확정. [[feedback_validation_nonlinearity]] 재확인.
