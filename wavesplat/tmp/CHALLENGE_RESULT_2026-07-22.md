# 도전 결과 — under-det 갭 닫기 시도 = **부분성공/FAIL(quick levers): 더 나은 학습 식별기 필요** (2026-07-22)

> 사용자 "도전해보자". 파수-ID CRB이 헤드룸을 REALIZABLE(알고리즘 탐색갭)이라 판정 → 더 나은 탐색기로 실제 닫기 시도.
> 스크립트 `tmp/challenge_refine.py`, 원자료 `results/direction3/challenge_refine.json`. under-det(15comp) n=16.

## 판정 = quick levers로는 닫히지 않음 (realizable but 쉽지 않음)
achieved 재구성 corr median (n=16):
| 방법 | corr | vs net-MMSE |
|---|---|---|
| RELAX (그리드 greedy) | 0.395 | — |
| **off-grid refine** (autograd variable-projection) | 0.514 | **−0.079** (CI[−.127,−.029], 3/16) |
| **net-MMSE (학습)** | **0.568** | (기준) |
| net-MMSE + EM refine | 0.578 | +0.010 (n.s.) |
| oracle-LS (참 파수) | 0.999 | — |

- **off-grid 탐색은 classical RELAX를 +0.11 개선**(격자 이산화·국소최적 일부 해소) — 탐색이 갭의 일부임을 확인.
- **그러나 학습 net-MMSE를 못 넘음**(Δ−0.079 유의). **EM refine도 무이득**(Δ 0.000).
- ⟹ **quick levers(off-grid 탐색·EM)로는 net-MMSE 위 +0.34(→oracle) 못 닫음.**

## ⟹ 결론 (정직)
- **realizability는 CRB이 확인(정보 있음, 정답 유일)** — 그러나 **닫으려면 classical 탐색·refine이 아니라 더 나은
  LEARNED 파수 식별기가 필요.** net-MMSE(학습 dense-γ bank)가 classical 탐색을 이미 이기고(0.40→0.57),
  나머지 +0.34는 **더 나은 학습법**(참 sparse 성분에 더 가까운 γ prior/식별기)만 회수 가능.
- = **오늘 세션 방법-지도와 정합**: 승자는 항상 "물리-정규화 학습 prior + 닫힌형 solve"; classical 탐색만으론 부족.
- **도전 결과**: 갭이 realizable임은 CRB이 승인, quick 시도는 못 닫음 ⟹ **진짜 forward path = 더 나은 학습 식별기
  (실질 연구/논문 작업), 값싼 트릭 아님.** 단, CRB이 그 노력이 헛되지 않음(천장 도달 가능)을 보증.

## 시도한 lever (전부 기록)
- off-grid autograd refine(coarse+fine init, 60 iter, variable-projection): RELAX +0.11, net-MMSE 미달.
- net-MMSE + EM(max_em=6): 무이득.
- (미시도, 다음): net-MMSE 학습 prior + off-grid bank 증강 하이브리드 / 참-성분 근접 재훈련 net / multi-restart 강탐색.

## ⟹ 세션 최종 위치
- under-det 파수-ID 갭 = **정체 규명(파수 ID)·헤드룸 실재(+0.45)·realizable(CRB)·닫으려면 더 나은 학습 식별기**.
- = **학습을 논문 챕터로 밀 유일 정직한 forward path**(CRB-검증). 나머지(B-2·궤적)는 oracle-washout.
- **권고**: 이 축(더 나은 학습 파수 식별기)이 논문 "학습 기여"의 정직한 자리 — 단 realistic/HOS서 재검·model
  mismatch로 실측 천장<1 유의. 병행/대안 = spine A 집필. [[project_methods_landscape_2026-07-22]].
