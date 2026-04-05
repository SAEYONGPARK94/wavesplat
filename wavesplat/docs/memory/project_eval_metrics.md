---
name: Evaluation metrics design
description: 평가 메트릭 설계 — primary/secondary, 공간 분리, 수렴 시간, 통계
type: project
---

## Primary Metrics
| 메트릭 | 정의 | 목표 |
|--------|------|------|
| RMSE_z | √(mean(z_pred - z_gt)²) [m] | < 0.05m (BF4) |
| nRMSE | RMSE_z / Hs | < 5% |
| Hs_error | |Hs_pred - Hs_gt| / Hs_gt [%] | < 5% |
| Tp_error | |Tp_pred - Tp_gt| [s] | < 0.5s |
| Spectral_R² | R² of S_pred(ω) vs S_gt(ω) | > 0.90 |

## Secondary Metrics
| 메트릭 | 목적 |
|--------|------|
| RMSE_near (10-80m) / RMSE_far (80-250m) | 거리별 성능 |
| RMSE_unobserved | Physics prior 기여 핵심 증거 |
| T_converge (nRMSE < threshold 도달 시간) | 수렴 속도, init vs tracking 구분 |
| T_inference [ms/frame] | 실시간 가능성 |
| Directional_R² (2D kx,ky spectrum) | 방향 추정 정확도 |
| Dispersion 집중도 | Ablation E2(dispersion 제거)에서만 사용 |

## 보고 방식
- 5 seeds, mean ± std
- Effect size 포함
- 근거리/원거리/미관측 영역 분리
- 수렴 시간을 별도 figure로
- Uncertainty: 논문 1 scope 밖

**Why:** Prof.E의 "시간 지연/공간 분해능" 지적, Rev.2의 "통계적 유의성" 요구 반영.
**How to apply:** evaluation/metrics.py 구현 시 이 목록 전부 포함. 논문 Table/Figure 매핑.
