---
name: Code-document discrepancies found in review
description: 코드↔문서 불일치 9건 — α_threshold 버그, 905nm/1550nm, Contribution 번호 등
type: project
---

## 발견된 불일치 (2026.04.05 반복 검토)

### 코드 버그 (즉시 수정 필요)
- **#3: α_threshold 불일치** — `raydrop_slope.py` 내 `specular_alignment` 기본값 2° vs `apply_dropout` 기본값 5°. 5°로 통일 필요.

### 문서 업데이트 필요
- **#2: 905nm → 1550nm** — 개발가이드 여러 곳에서 "905nm" 잔존. Falcon K1은 1550nm.
- **#5: Contribution 번호** — 개발가이드 C1~C4 (논문 구성 중심) vs 토론 기록 C1~C4 (방법론 중심). **방법론 중심으로 통일**: C1=Grid-free estimation, C2=Dispersion-constrained parameterization, C3=Fresnel confidence, C4=Bound harmonics (optional)
- **#9: 개발가이드 §6~7 outdated** — 알고리즘 설계/메트릭/baseline이 이번 토론에서 구체화됨. 반영 필요.

### 논문 작성 시 반영
- **#1: 토론 기록 "32채널"** — frozen 문서이므로 수정 안 함. 실제 152채널.
- **#4: Raycast mesh 해상도 (40×30)** — GT mesh (80×60)와 다름. 의도된 설계이나 논문에 명시.
- **#6: Linear sinusoidal vs Gerstner mismatch** — 추정 알고리즘은 linear, GT는 Gerstner. O(ε²) 차이. Discussion에 포함.
- **#7: Confidence = observation noise weight** — wave model 정확도와 혼동 방지. 논문에서 명확 구분.
- **#8: GPR temporal variant** — GPR(window, 5 frames) 추가로 physics/temporal 기여 분리.

**How to apply:** #3은 코드 수정 시 즉시 처리. #2, #5, #9는 개발가이드 업데이트 시 반영. #6~#8은 논문 작성 시 참조.
