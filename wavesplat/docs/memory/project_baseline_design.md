---
name: Baseline comparison design (revised)
description: GPR(single+window), ETKF-grid, Oracle baseline — physics/temporal 기여 분리 가능
type: project
---

## Baseline 비교 표 (핵심 — 논문 Table 후보)

| Method | Physics | Temporal | Grid-free | 역할 |
|--------|---------|----------|-----------|------|
| GPR (single frame) | ✗ | ✗ | ✓ | 하한: 물리/시간 없이 순수 보간 |
| GPR (window, 5 frames) | ✗ | ✓ | ✓ | temporal 기여 분리 |
| ETKF-grid | ✓ | ✓ | ✗ | grid 한계 증명 (WaveSplat motivation) |
| **WaveSplat** | ✓ | ✓ | ✓ | 제안 방법 |
| WaveSplat-oracle | ✓ | ✓ | ✓ | 상한: true (k,ω) init |

## 구현 상세

### GPR (single frame)
- scikit-learn GP, Matérn ν=5/2 (주) + RBF (비교)
- length_scale grid search: [5, 10, 20, 50m]
- noise: σ=0.02m

### GPR (window, 5 frames)
- 직전 5프레임 관측 누적 → single GP fitting
- 시간 정보 무시 (모든 관측을 동일 시점으로 취급)
- physics 없이 temporal accumulation만의 효과 측정

### ETKF-grid
- State: η(grid) + ∂η/∂t
- Prediction: dispersion 기반 시간 진화
- Observation operator: bilinear interpolation (grid → obs points)
- Grid 해상도 3단계: 20×15 / 40×30 / 80×60
- 보간: thin-plate spline, ensemble=50
- 원저자 코드 요청 시도

### WaveSplat-oracle
- True (kx, ky, ω) 초기화, φ만 최적화
- 최적화의 이론적 상한

## 제외 (논문에 사유 명시)
- PINO: regular grid 전제
- EnKF-PFL: 고정 센서 시계열 전제

## 공정성
- 모든 baseline 하이퍼파라미터 grid search → 최적 결과 보고
- 동일 관측 데이터, 동일 메트릭, 계산 시간도 비교

**Why:** Prof.K "physics/temporal 기여를 분리 가능한 표"가 논문 contribution을 시각적으로 보여줌.
**How to apply:** 이 표를 논문 Table 1 후보로. baselines/ 구현 시 4개 모두 포함.
