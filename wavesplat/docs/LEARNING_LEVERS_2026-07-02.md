# 학습 개선 레버 면밀 검토 (2026-07-02)

> 확정된 병목: **어려운 레짐(8+comp/sparse)의 detection**. RELAX adaptive greedy(잔차 구동, 1개씩) 우세.
> A1(self-sup 1-shot grid net + refine): 8/300 0.64 vs RELAX 0.92. grid-only 수렴(plateau), 8-seed로 gap 진짜.
> "덜 학습"이 아니라 방법 한계 → 남은 레버 검토.

## 시도한 것 (요약)
- 연속 (kx,ky) 회귀 + obs-fit → 비볼록 정체(fail).
- grid-spectral 1-shot net + obs-fit → 볼록, 학습되나 grid-cap 0.44.
- grid peak + physics refine (A1) → 0.69 avg, 6-comp RELAX 근접, 8+ 미달.
- v2(촘촘/혼합/장학습) → 유사(avg 0.69). **grid+refine 계열 한계 확인.**

## 남은 레버 (win 잠재력 순)
1. **★ Deep-unrolled matching pursuit (LISTA/ISTA 언롤, self-sup)** — 1-shot 대신 **반복 sparse 복원**(잔차 구동
   구조를 층으로). RELAX greedy를 학습으로 모사. detection 병목 직격. DOA서 검증. → **프로토타입 대상.**
2. **Supervised-init + self-sup 실측 fine-tune** — component_distill(supervised, sim서 RELAX 이미 능가) 자산 활용
   → 실측서 obs-fit test-time 적응. **실측 전이(진짜 상금)** 겨냥. 단 net raw가 실측서 나쁘면 국소최적 위험.
3. **Iterative/recurrent 제안 net** — net이 잔차서 1개씩 제안·subtract(RELAX 모사). #1과 유사 계열.
4. **더 나은 peak 추출** — soft-argmax 서브그리드 보간, 잔차-재검출. 증분.
5. **Reweighted-L1 / atomic-norm(gridless)** — sparse 복원 개선. 증분.
6. **Net warm-start RELAX** — net peak로 RELAX greedy 초기화. ≥RELAX지만 학습기여=속도뿐(RELAX가 본체).
7. **더 큰 net/장학습** — grid-only 수렴했으니 grid-metric엔 무효; peak엔 미미.

## 검증 전략
- **#1(LISTA self-sup) 프로토타입** — 어려운 레짐(8/300, 12/300)서 peak+refine이 A1(0.64)을 넘어 RELAX(0.92)로
  가는지. **넘으면 학습 headroom 있음(계속). ~A1이면 detection이 근본한계 → 학습 exhausted → 피벗.**
- 결과로 continue-vs-pivot 결정. (사용자: 개선 없으면 다른 방향 OK.)

## 레버 #1 (LISTA unrolled) 결과 + 종합 판정 (2026-07-02)
LISTA(deep-unrolled ISTA, self-sup, Lipschitz-정규화 step): 학습 성공(heldout-rmse 0.34→0.12).
| NC/MO | LISTA | A1(1-shot) | RELAX |
|---|---|---|---|
| 6/600 | 0.851 | 0.92 | 0.939 |
| 8/300 | **0.809** | 0.64 | 0.944 |
| 12/300 | 0.559 | 0.65 | 0.794 |
| **평균** | **0.74** | 0.74 | **0.89** |
- LISTA는 8/300(어려운 detection) A1 크게 개선(0.64→0.81)하나 타 구성 하락 → **평균 A1 동급, RELAX 미달.**
- numerics 교훈: 512원자 coherent dict → ISTA step 안정영역 매우 좁음(dead-zone↔divergence). Lipschitz 정규화+live init
  필요. **이 ill-conditioning 자체가 RELAX가 joint sparse 아닌 greedy 쓰는 이유.**

## ★ 종합 판정 — 학습 레버 면밀 검토 완료
시도한 4가지 label-free 학습 모두 RELAX 미달(평균 0.69~0.74 vs RELAX 0.89):
① 연속회귀(정체) ② 1-shot grid+refine(0.69) ③ v2 촘촘/혼합/장학습(0.69, 수렴확정) ④ unrolled ISTA(0.74).
- **결론: label-free 학습은 RELAX를 robust하게 못 넘음.** RELAX greedy가 coherent dispersion dict에 near-optimal.
  supervised는 RELAX 능가하나 라벨 필요(sim2real). 데이터량 아님(무한 on-the-fly), 학습량 아님(수렴확정), 방법 한계.
- **데이터 종류는 단일(clean JONSWAP)** — 실측 전이 gap의 원인(양이 아니라 종류).
- **→ 피벗 (사용자 허락)**: 학습으로 RELAX 능가 추구 중단. 기여축 전환:
  **(A) RELAX=추정 본체 확정 + (B2) 정보한계 이론(학습이 벽까지 saturate) + (B4) forecast operability + (B3) UQ.**
  = PINO/Ehlers(방법론문)와 안 겹치고 이미 강한 자산(CRB·operability·실측) 활용.
