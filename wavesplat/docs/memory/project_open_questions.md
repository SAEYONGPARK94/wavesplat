---
name: WaveSplat remaining open questions (revised)
description: 반복 검토 후 남은 미해결 항목 — 구현 단계에서 확인 필요
type: project
---

## 코드 수정 필요 (즉시)
- **raydrop_slope.py α_threshold**: specular_alignment 기본값 2° → 5°로 통일
- **pipeline.py raycast mesh 해상도**: 하드코딩(40×30) → config 파라미터화 검토

## 문서 수정 필요
- **개발가이드 905nm → 1550nm** 통일
- **개발가이드 C1~C4** 방법론 중심으로 수정
- **개발가이드 §6~7** 알고리즘/메트릭/baseline 반영

## 구현 단계에서 실험적 결정
1. Coarse-to-fine 전환 residual threshold 구체값
2. Initialization 2Tp 충분성 — BF별 실험
3. 정규화 λ₁, λ₂ — grid search / cross-validation
4. ETKF-grid inflation factor — 원논문 확인 또는 실험
5. GPR window 프레임 수 — 5 frame 외에 3, 10도 비교?

## 데이터/실험 기회에 확인
6. 수조에서 실제 return 여부 (일정 미정)
7. NEUBOAT 수면 포인트 분리 가능성
8. 해류 Doppler shift 실제 영향 크기

**How to apply:** 코드 수정은 다음 개발 세션에서 즉시. 문서 수정은 개발가이드 업데이트 시. #1~#5는 Phase 3 초기에.
