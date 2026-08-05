# C2b 사전등록 — C2 rescue 시도 (power + 운용레짐), 결과 前 동결 2026-07-22

> 사용자 "C B A 순으로하자" = (c)검정력↑ → (b)운용레짐 재검 → (a)C1 폴백. **NO-WIN 후 second-bite임을 명시.**
> 규율([[feedback_direction_flip_discipline]]): p-hacking 방지 위해 **두 검정 사전고정 + honor 규칙 + 다중성 인지.**
> ⚠**정직 disclosure**: n=16/32서 음성 2바다가 저-Hs임을 이미 봄. 아래 Hs≥2 제한은 **운용 근거(제어 수요=거친 바다)**로
> 독립 정당화되나, 내가 힌트를 본 후 설계했음을 명시 — 그래서 **사전고정 + 단 1회 검정**으로 규율.

## 대상 효과 (동결값)
현 n=32 결과: Δ(gated−const) = +0.038, sea-block 부트 95% CI **[−0.028, +0.107]** = NO-WIN. per-sea 14/16 양성.

## (c) 검정력 — seed-replicate (사전고정 배치)
- ⚠**배치 갱신 (2026-07-22, 복제결과 보기 前)**: 검정력 계산상 seed 1개론 부족(현 16 test-block·CI반폭0.067 → 0넘기려
  ~50 block 필요). ⟹ **seed=1,2,3 (3회 복제)로 사전고정** = 64 test-block. **이게 최종 배치·더 안 늘림(open-ended 금지).**
  사용자 "그래도 c로 해보자". 사이즈 결정은 **검정력 근거(결과 peek 아님)** — 표준 sample-size 산정.
- 32개 (Tp,Hs) 바다를 seed 1,2,3로 복제(새 위상 실현). 분할: si-parity로 train(even)/test(odd), **같은 (Tp,Hs)의 모든
  seed는 같은 split**(누수 방지). block = (si,seed). 정책(Wiener·best-const)은 train fit, test 적용(원 사전등록과 동일).
- **중간 peek 금지**: seed-1만 보고 유의하면 멈추기 = p-hacking. **3개 다 빌드 후 단 1회 분석.**

## (b) 운용레짐 — Hs≥2 제한 (사전고정)
- **1차-2 검정**: 위 test 집합을 **Hs≥2 바다로 제한**(운용 근거: 제어 수요는 거친 바다; Hs<2=near-calm·제어 덜 필요).
- 같은 R·sea-block 부트 CI.

## 검정 (사전고정, 단 1회씩)
- **Primary-1 (power)**: seed-복제 포함 전체 test pooled, sea-block 부트 CI 하한>0?
- **Primary-2 (operational)**: Hs≥2 제한 test, sea-block 부트 CI 하한>0?
- **다중성**: 2 primary → **둘 중 하나라도 통과 시 "부분 rescue(맥락 명시)"**, 둘 다면 "rescue", 둘 다 실패면 (a).
  Bonferroni 인지(2검정): 강주장엔 α/2 권장하나, 본 문서는 **정직 라벨링**(어느 검정이 통과했는지·다중성)으로 대체.

## honor 규칙 (동결)
- 통과해도 효과는 **small(+0.04~0.09)** — "강 헤드라인" 아니라 **"조건부(운용레짐)·정직 보조"**까지만 승격 허용.
- 실패면 **즉시 (a)**: C1(공통천장) 헤드라인 + C2 = "시사적·미확립".
- **어느 쪽이든 발표.** 결과가 헤드라인 강도 결정.

## 실행
- (b)는 기존 seed-0 캐시로 **즉시 무료 계산**(먼저 정보). (c)는 seed-1 빌드 ~2.7hr 백그라운드 → 완료 후 결합 power 검정.
- 스크립트 `tmp/c2_power.py` → `results/direction3/c2_power.json`.

## 정직 예상 (사전 기록)
- (c) 단독: 효과 small이라 **borderline 가능성**(48+ test-sea 필요 추정). (b) 운용제한: 음성 저-Hs 제거로 **더 유망한 rescue.**
- 성공해도 = **modest·조건부.** landmark 아님. 이 시도의 목적 = "정직하게 rescue 가능한지 소진" (사용자 지시).
