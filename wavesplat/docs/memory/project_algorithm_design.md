---
name: Phase 3 algorithm design decisions (revised)
description: Spectral Gaussian Estimation 구체 설계 — projected GD, init/tracking, linear forward model
type: project
---

## Forward Model: Linear Sinusoidal (not Gerstner)
```
z(x,y,t) = Σᵢ Aᵢ·cos(kᵢ·r - ωᵢ·t + φᵢ)
```
- 추정 알고리즘은 linear model, GT(시뮬레이터)는 Gerstner
- **의도된 model mismatch**: O(ε²) ≈ 0.25% (BF4)
- 논문 Discussion에 명시: "linear estimator가 weakly nonlinear GT에 대해 robust함을 보임"

## 최적화: Projected Gradient Descent (PyTorch)
- 자유 파라미터: wᵢ, kxᵢ, kyᵢ, φᵢ (Gaussian당 4개)
- Projection step: ωᵢ = √(g|kᵢ|tanh(|kᵢ|d)) — hard constraint
- Loss: Σⱼ cⱼ·(z_model - zⱼ)² + λ₁·||w||₁ + λ₂·smoothness

## Confidence Weight 명확화
- cⱼ = cos(θᵢ) / (d/d_ref) → **observation noise의 heteroscedasticity**
- "이 포인트의 z 측정이 얼마나 정확한가" (NOT wave model의 정확도)

## 2단계 Sequential Update
1. **Initialization** (첫 2Tp, BF4≈14초/140프레임)
   - Coarse-to-fine: N=3→8→15
   - Joint optimization over full window
   - FFT-based spectral analysis로 초기값
2. **Tracking** (이후 프레임)
   - Warm-start: 이전 θ* 초기값
   - Predict: φᵢ += ωᵢ·Δt
   - Update: 5~10 gradient steps
   - 목표 50ms/frame

## Gaussian 수 N
- 논문 1: 고정 N, ablation N=5/10/15
- 물리적 가이드: N ≈ spectral bandwidth / mean Gaussian width
- Adaptive densification = future work

## 정규화
- L1 on weights (sparse spectrum)
- Spectral smoothness (인접 weight 변화 제한)
- λ₁, λ₂: grid search 또는 cross-validation

**Why:** 반복 검토에서 linear vs Gerstner mismatch, confidence 해석 명확화 추가됨.
**How to apply:** spectral_gaussian.py 구현 시 linear sinusoidal forward model 사용. confidence가 observation noise weight임을 코드 주석에 명시.
