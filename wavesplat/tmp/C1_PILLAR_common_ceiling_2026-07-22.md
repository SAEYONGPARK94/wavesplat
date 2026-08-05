# C1 지지기둥 — "공통천장: 더 나은 방법이 아니라 envelope이 산출" 2026-07-22

> 방향-C의 이론 기둥. C2 (c)결과 무관하게 안정(C2 rescue면 지지기둥, 실패면 헤드라인). 정직·스코프 엄수.
> 규율: MEASURED만 인용, "정보-한계" 단어는 CRB 강화 전 금지 → **"estimator-급 불변 천장"까지만.**

## 주장 (한 문장)
**잘 설계된(물리-제약) 추정기들의 재구성 skill은 수렴한다 — 학습 prior가 좋은 해석 prior를 못 이기고(WaveSplat≈
WavePrior), 모두 약한 matching-pursuit(RELAX) 위에 있다. CRB 운용성 envelope이 이 추정기-family 전반의 skill 고저를
예측하므로(estimator-급 불변, 분산-제약 family 한정), 유용한 산출은 "약간 더 나은 추정기"가 아니라 envelope이다.**

## 증거 (MEASURED, 기존 결과)
### E1. 학습 prior ≈ 해석 prior (같은 MMSE, 같은 셋업) — `compare_metrics.json`
| case | WaveSplat(해석 prior) | WavePrior(학습 prior) | 승자 |
|---|---|---|---|
| linear | 0.883 | 0.872 | 해석 근소 |
| HOS | 0.832 | 0.877 | 학습 근소 |
| real stereo | 0.653 | 0.624 | 해석 근소 |
⟹ **TIE (case마다 승자 다름·차 ≤0.045).** "학습이 고전 이긴다" = 이 축서 **거짓**(방법 novelty 0, [[project_novelty_phd_verdict_2026-07-15]]).

### E2. 물리-제약 추정기 클러스터 vs 약한 RELAX — `hos_ladder.json` (n=8 HOS)
- net-MMSE 0.755 · better-net 0.741 · (해석 prior WaveSplat 0.83@compare) — **클러스터**
- 약한 RELAX(matching-pursuit) 0.562 · 오지정 linear-fit "ceiling" 0.491 — **아래**
⟹ net−RELAX +0.193. **천장은 "물리-제약 잘 설계"면 도달; 특정 방법(해석 vs 학습)은 그 위서 무차별.**
⚠주의: "linear-fit ceiling(rich) 0.491 < net 0.755"은 **inverse-crime ceiling의 오지정** 때문(HOS서 linear-fit이 천장
아님) — 이 "ceiling" 수치는 **천장 주장에 인용 금지**(오늘 inverse-crime 감사서 확인).

### E3. envelope = estimator-급 불변 (단, 스코프) — `c8_cross_estimator.json`
- σ_CRB이 skill 예측: **ridge**(crb-skill ρ~0.29) · **relax**(ρ~0.59) — 분산-제약 클래스서 작동.
- **GP(무제약)엔 실패**(ρ~−0.02). ⟹ verdict = REFUTE/PARTIAL. **"estimator-독립"은 무조건형 금지 → "분산-제약 family
  내 estimator-급 불변"까지만** ([[project_execution_2026-07-22]] C8 한정형 봉인과 정합).

### E4. envelope이 경험UQ보다 skill 예측 우위 (HOS) — `hos_s8_envelope.json` (n=8)
- Spearman(σ_CRB, actual skill) median **+0.30** vs 경험UQ(Zhao proxy) **+0.21** vs 거리 +0.12. (8/8 바다서 CRB>emp.)
- ⚠**PARTIAL** (PASS bar 0.5 미달). = "예측력 modest이나 경험UQ보다 나음". **강주장 금지.**

## C1이 지지하는 thesis 논리
1. 방법 경쟁은 무의미(E1·E2: 잘 설계된 추정기 수렴) ⟹ **novelty는 "더 나은 estimator"가 아님.**
2. 대신 **design-time·estimator-free 신뢰지도(envelope)**가 산출 — estimator-급 불변(E3, 스코프)·경험UQ 우위(E4).
3. envelope의 **검증** = 통계 skill(E4, modest §8) + **제어 결정가치(C2)**. *(C2 = 작지만 방향-신뢰 이득; (c) 검정력 진행 중.)*

## 격상 경로 (선택, 미완)
- **"estimator-급 불변" → "정보-한계"**: CRB 강화(random-phase·achievability, GUARD2) 성공 시. 미실행.
- **envelope §8 PARTIAL(0.30) → PASS(0.5)**: 미해결(정보-한계 modest가 진짜 천장일 수도).
- **실물 데이터**: 유일한 판도 변경자, 현재 불가 가정.

## 정직 총평
C1은 **견고한 지지기둥**(E1~E4 다 MEASURED)이나 **강 novelty 아님** — "방법들 수렴 + envelope이 modest하게 유용."
thesis = **정직한 진단 + 운용 신뢰지도**(modest-solid). 강화는 CRB(#7)/실물(#10)만.
