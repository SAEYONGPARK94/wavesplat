# HOS §8 envelope 키스톤 재검 = PARTIAL·생존 + CRB>경험UQ HOS-검증 (2026-07-22)

> 사용자 "남은 일 시작". §8(CONDITIONAL→PASS 키스톤)을 HOS로 재검(synthetic §8 inverse-crime 정정).
> 스크립트 `tmp/hos_s8_envelope.py`, 원자료 `results/direction3/hos_s8_envelope.json`. n=8 HOS 바다.
> σ_CRB = fitted θ̂(RELAX 성분)의 field CRB(oracle 아님·운용 현실). σ_actual = ridge-LS 재구성 오차 vs 실제 HOS 참.

## 결과 (n=8 HOS 바다, corr = Spearman)
| 지표 | HOS median | synthetic §8 |
|---|---|---|
| **CRB ↔ actual** | **+0.296** | +0.33 |
| partial \| distance | **+0.275** (생존) | +0.21 |
| **emp-UQ ↔ actual** | **+0.206** | +0.33(동률) |
| dist ↔ actual | +0.124 | +0.32 |
- **CRB > emp-UQ 8바다 전부**(per-sea: 0.24>−0.03, 0.32>0.28, 0.31>0.19, 0.28>0.24, 0.29>0.12, 0.38>0.21, 0.45>0.25, 0.28>0.20).

## 판정 = **PARTIAL (생존) + 핵심 delta HOS-검증**
1. **§8 키스톤 HOS서 생존**: CRB↔actual +0.30(양수·거리 confound 넘어 partial +0.28). **①(HOS서 −0.27 죽음)과 정반대** —
   envelope은 실제 비선형 바다서도 skill 위치를 (약하게) 예측. **inverse-crime 아님(실제 HOS z·θ̂ CRB·oracle 없음).**
2. **★CRB envelope > 경험UQ HOS-검증(핵심)**: HOS서 CRB(0.30) > emp(0.21), **8바다 전부.** synthetic선 **동률**
   (둘 다 0.33 = "예측 우위 없음")이었으나 **HOS서 CRB 우위 출현.** 기제: HOS 비선형이라 선형모델 경험UQ(ridge
   posterior)는 miscalibrated, CRB(정보기하 at θ̂)는 상대 견고. ⟹ **thesis 핵심 novelty("추정기-독립 CRB-앵커 >
   추정기-종속 경험UQ")가 HOS서 실증·강화**(synthetic 동률보다 나음). = Zhao difference-first의 HOS 근거.
3. **여전히 PARTIAL(정직)**: +0.30 < 0.5 PASS바. "예측 우위" 헤드라인 금지, "estimator-독립+판별+경험UQ 대비 HOS 우위"로.

## ⚠캐비엇
- n=8 · **단일 HOS 실현**(σ_actual 위상평균 없음=노이지, synthetic e58은 R=12 위상평균) · emp=ridge posterior
  대리(Zhao VB-NN 아님) · θ̂=RELAX 적합(적합오차 포함) · nowcast/단리드.
- **이중검증 미실행**(fable 독립감사 권장) — 단 8바다 전부 CRB>emp 일관은 견고.

## ⟹ 세션 종합 갱신 (HOS-준수)
- **①under-det/better-net = HOS서 사망**(inverse crime). **§8 envelope = HOS서 생존(PARTIAL)+경험UQ 우위 HOS-검증.**
  = 대조가 명확: **오늘 죽은 건 "학습 돌파", 살아남은 건 "CRB-앵커 운용성 envelope"** = spine A 핵심.
- **박사 함의**: 페르소나 debate의 "결정적 급소=CRB envelope을 HOS서 살려내기"가 **부분 충족**(생존+경험UQ 우위).
  헤드라인은 PARTIAL이나 **핵심 delta(CRB>경험UQ)가 HOS-검증**된 게 CONDITIONAL→PASS 방향 진전.
- **다음**: C8 HOS 재검(estimator-독립) · fable 독립감사 · 위상평균 σ_actual(노이즈↓) · 지도교수 합의 · 실물센서.
