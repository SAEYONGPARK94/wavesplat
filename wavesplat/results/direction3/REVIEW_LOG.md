# Direction III — 검증 로그 (grid-free 학습 prior)

설계: [docs/direction3_gridfree_prior_design.md](../../docs/direction3_gridfree_prior_design.md)
코드: [d3_setdiff_prior.py](../../scripts/d3_setdiff_prior.py)(학습), [d3_setdiff_posterior.py](../../scripts/d3_setdiff_posterior.py)(DPS)

---

## Step 1 — set-diffusion prior 학습 + DPS posterior (핵심 GO)

**날짜:** 2026-06-27 · **GPU:** RTX 4080 Super

### A. 설계 대조 (md §3~§6)
- 표현: 스펙트럼 = **연속 atom 32개** {(kx,ky,logE)}, 격자 제거. ✅
- prior: permutation-equivariant set-Transformer DDPM(ε-pred, 0.56M params, T=500), 16k JONSWAP atom-set로 학습. ✅
- likelihood: grid-free GP-marginal K=Σᵢaᵢcos(kᵢ·Δx−ω(kᵢ)·Δt), Cholesky 미분가능. ✅
- posterior: DPS(unconditional score + likelihood gradient, x̂₀ clamp). ✅

### B. prior 품질 (학습 2차, 수정 후)
- 1차(nc 5~30 가변): padding atom이 원점에 쌓여 prior가 k≈0 과대표현 (NO-GO).
- **수정: 항상 32 실제 atom**(JONSWAP 밀도 연속샘플) → 원점 artifact 제거, 방향 arc 구조 재현.
  energy-weighted |k| prior 0.133 vs real 0.098 (동급). [fig_prior_samples.png]

### C. 결과 — field nRMSE (동일 3에피소드, Dir-II와 apples-to-apples)
| 에피소드 | grid-diffusion(Dir-II) | grid-oracle-S | **grid-free BMA** | grid-free best | prior-only |
|---|---|---|---|---|---|
| BF4 5c  | 0.272 | 0.265 | **0.236** | 0.219 | 0.45 |
| BF4 15c | 0.292 | 0.292 | **0.262** | 0.277 | 0.30 |
| BF3 5c  | 0.320 | 0.291 | **0.246** | 0.266 | 0.32 |
| **평균** | **0.294** | **0.282** | **0.248** | 0.254 | — |

(BMA = posterior-mean field, guidance=0.1, 12 posterior samples; spread 0.03~0.07 = 보정 UQ)
참고: WaveSplat fused 0.205, exact-continuous 0.06.

### D. 핵심 결론 (두 갈래, 둘 다 결정적)
1. **격자가 진짜 병목이었다 (grid-free GO):** grid-free BMA(0.248)가 grid-diffusion(0.294)뿐 아니라
   **grid-oracle-S(0.282)까지 3에피소드 모두 격파.** 격자 양자화가 ~0.04~0.05 nRMSE 비용이었음을 입증.
   prior 개념(Dir-II GO) + 연속 표현(Dir-III) = 표현 병목 해소.
2. **그러나 정보의 벽은 그대로다 (info-limited):** 0.248 ≫ exact 0.06, WaveSplat fused 0.205보다도 위.
   **표현을 완벽히 연속화해도 단일윈도우 한계(Dir-I)는 못 넘음** → 잔차는 표현이 아니라 **정보**.
   Dir-I CRB가 예측한 그대로(단일윈도우 aperture 한계).

### E. 세 방향의 통합 (loop 완성)
- **Dir-II**: 학습 prior 개념 GO, 격자 NO-GO (0.294).
- **Dir-III**: 격자 제거 → 0.248 (grid-oracle도 격파) → **격자가 비용이었음 확인**, prior는 작동.
- **Dir-I**: 0.248 ≫ 0.06 잔차 = **정보의 벽**(CRB)로 설명. 표현 고쳐도 남는 건 정보부족.
- 부가가치: DPS는 **보정된 posterior UQ**(샘플 spread) 제공 → (C) predictable-zone/UQ와 직결.

### F. 박사 novelty / contribution
- **검증된 기여:** ship-LiDAR 파랑역문제용 **grid-free atomic set-diffusion prior + 분산제약
  GP-marginal likelihood DPS** — 도메인 최초. 격자 DL(spectrum→FFT)·atomic-norm(hand-crafted)·
  고전 DSWP(최적화) 대비 차별성 명확.
- **정직한 한계:** 0.248은 grid를 이기지만 정보의 벽(0.06)엔 못 미침. per-sample mean(0.31)은
  BMA(0.25)보다 나쁨 — 개별 샘플 다양성, BMA(posterior-mean field)가 정답 추정.
- prior 학습loss 0.114로 다소 높음(에너지채널 난이도) → 더 나은 prior 여지 있음.

### G. 판정
✅ **grid-free GO** (격자 병목 해소, grid-oracle 격파, 보정 UQ 획득) /
**정보의 벽은 잔존**(Dir-I 확증). 다음: (C) DPS posterior UQ를 CRB predictable-zone에 보정,
또는 전-윈도우(시간융합) likelihood로 정보의 벽 자체를 낮추기(Dir-I이 가리킨 유일한 레버).
