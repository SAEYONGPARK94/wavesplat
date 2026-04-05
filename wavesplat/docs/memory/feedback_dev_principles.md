---
name: 8 development principles from discussion
description: 4인 토론에서 합의된 개발 8원칙 — first principles, full dispersion, Q constraint 등
type: feedback
---

1. 모든 물리는 first principles에서 유도. "라이브러리가 해준다"는 근거 아님.
2. 모든 구성요소 독립 검증 (unit test + 해석해 비교).
3. Full dispersion(tanh) 항상 사용. 깊은 물 근사 금지.
4. Q 파라미터 제약(ΣQᵢωᵢAᵢ ≤ 1) 항상 강제.
5. Parametric sweep으로 결론 일반화. 특정 값 종속 주장 금지.
6. 두 raydrop 모델 모두 유지. 데이터가 답을 준다.
7. 논문 1 투고를 수조/실선 때문에 지연시키지 않음.
8. 코드 + 데이터 공개. 재현성 = 신뢰도.

**Why:** 지도교수가 물리적 엄밀성, 외부위원이 실용성, 리뷰어가 재현성을 각각 강하게 요구.
**How to apply:** 새 코드 작성/리뷰 시 이 8원칙 준수 확인. 특히 #3, #4는 코드 레벨에서 반드시 체크.
