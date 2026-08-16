---
name: Algorithm design decisions (updated post-P1)
description: Spectral Gaussian Estimation 설계 — projected GD, init/tracking, adaptive N, multi-frame — 구현 완료
type: project
---
> ⛔**ARCHIVED / 낡은 사본 (2026-08-13 감사).** 현행 연구 기록의 권위는 auto-memory(`~/.claude/projects/-home-vexit-wavesplat/memory/MEMORY.md`)와 `tmp/<배치>/RESULT.md` 다. 이 디렉터리의 수치는 다수가 이후 측정에서 무효화됐다 — 인용 금지.


## Forward Model: Linear Sinusoidal (not Gerstner)
```
z(x,y,t) = Σᵢ wᵢ·cos(kxᵢ·x + kyᵢ·y - ωᵢ·t + φᵢ)
```
- 추정 알고리즘은 linear model, GT(시뮬레이터)는 Gerstner
- **의도된 model mismatch**: O(ε²) ≈ 0.25% (BF4)

## 최적화: Projected Gradient Descent (PyTorch)
- 자유 파라미터: wᵢ, kxᵢ, kyᵢ, φᵢ (Gaussian당 4개)
- Projection step: ωᵢ = √(g|kᵢ|tanh(|kᵢ|d)) — hard constraint
- Loss: Σⱼ cⱼ·(z_model - zⱼ)²
- Tracking mode: `w_phi` — kx, ky frozen, only w and φ updated

## Confidence Weight
- cⱼ = cos(θᵢ) / (d/d_ref) → **observation noise의 heteroscedasticity**

## 2단계 Sequential Update
1. **Initialization** (첫 2Tp)
   - Beamforming v2_konly → coarse-to-fine N=3→8→15
   - 평균 18% tracking 개선 (vs random init)
2. **Tracking** (이후 프레임)
   - Predict: φᵢ += ωᵢ·Δt
   - Update: gradient steps with sliding window k=3 frames
   - 13.4 ms/frame 달성

## Adaptive N (P1에서 구현 완료)
- **Pruning**: Energy-based (Σwᵢ²), keep 90% total energy
- **Growth**: Residual > 0.05 → beamforming on residual → add new components
- **Safety**: Rollback if residual increases >30% after pruning
- **Max N**: 30
- 결과: BF4_1comp 15→1, BF3_2comp 15→2, BF5_15comp 15→21

## Multi-frame Tracking (P1에서 구현 완료)
- Sliding window k=3 recent frames for gradient computation
- 단일 프레임 대비 gradient stability 개선, speed cost 미미

**Status:** 구현 및 실험 완료 (P0-P1). 논문 초안 작성 완료 (P2).
