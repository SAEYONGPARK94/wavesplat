---
name: Code-document discrepancies found in review
description: 코드↔문서 불일치 9건 — α_threshold 버그, 905nm/1550nm, Contribution 번호 등
type: project
---
> ⛔**ARCHIVED / 낡은 사본 (2026-08-13 감사).** 현행 연구 기록의 권위는 auto-memory(`~/.claude/projects/-home-vexit-wavesplat/memory/MEMORY.md`)와 `tmp/<배치>/RESULT.md` 다. 이 디렉터리의 수치는 다수가 이후 측정에서 무효화됐다 — 인용 금지.


## 발견된 불일치 (2026.04.06 4인 토론 후 업데이트)

### 코드 버그
- **#3: α_threshold 불일치** — ✅ **해결 완료** (2026.04.06 확인). `specular_alignment()` 기본값 5.0°, `apply_dropout()` 기본값 5.0°로 이미 통일됨.

### 문서 업데이트 필요
- **#2: 905nm → 1550nm** — ✅ **수정 완료** (2026.04.06). 개발가이드 전체 1550nm으로 통일.
- **#5: Contribution 번호** — ✅ **수정 완료** (2026.04.06). 방법론 중심 C1~C4로 통일.
- **#9: 개발가이드 §6~7 outdated** — ✅ **수정 완료** (2026.04.06). 4인 토론 결과 반영.
- **#NEW-1: 데이터셋 구성 불일치** — ✅ **수정 완료** (2026.04.06). "15 시나리오 × 75 에피소드" → "10 시나리오 × 100 에피소드"로 수정.
- **#NEW-2: HDF5 구조 오류** — ✅ **수정 완료** (2026.04.06). `lidar_intensities` 제거, `channels: 32` → `152` 수정.

### 논문 작성 시 반영
- **#1: 토론 기록 "32채널"** — frozen 문서이므로 수정 안 함. 실제 152채널.
- **#4: Raycast mesh 해상도 (40×30)** — GT mesh (80×60)와 다름. 의도된 설계이나 논문에 명시. Phase 3에서 mesh sensitivity test 수행.
- **#6: Linear sinusoidal vs Gerstner mismatch** — 추정 알고리즘은 linear, GT는 Gerstner. O(ε²) 차이. Discussion에 **정량 수치 포함**.
- **#7: Confidence = observation noise weight** — wave model 정확도와 혼동 방지. 논문에서 명확 구분.
- **#8: GPR temporal variant** — GPR(window, **3/5/10 frames**) 모두 테스트 후 최적 결과 보고. Phase 3에서 구현.

**How to apply:** 코드/문서 이슈 모두 해결됨. #4, #6~#8은 논문 작성 시 참조.
