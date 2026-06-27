# Direction III — Grid-free learned spectral prior (design)

> **RESULT (2026-06-27): grid-free GO.** set-diffusion prior + DPS, posterior-mean
> (BMA) field nRMSE **0.248** vs grid-diffusion 0.294 vs grid-oracle 0.282 — beats
> the grid (incl. grid-oracle) on all 3 episodes, confirming the grid cost
> ~0.04–0.05. Still ≫ exact 0.06 ⇒ residual is the **information wall** (Dir-I),
> not representation. Full log: [results/direction3/REVIEW_LOG.md](../results/direction3/REVIEW_LOG.md).

**Date:** 2026-06-27 · status: design → implementation → **evaluated (GO)**
Motivated by Direction I (continuous (kx,ky) is *required* — grid quantization
Δk≈0.034 costs +0.22 nRMSE; even grid-oracle 0.28 ≫ exact 0.06) and Direction II
(diffusion prior **concept GO** at 0.294, but **grid NO-GO**). This direction
removes the grid.

## 1. The one-sentence idea
Represent the directional wave spectrum as a **set of continuous atoms**
{(kxᵢ, kyᵢ, aᵢ)} instead of a 64×64 image; learn a **permutation-invariant
set-diffusion prior** over such atom sets; recover the spectrum from sparse LiDAR
by **diffusion posterior sampling (DPS)** guided by the phase-marginalized
GP-marginal likelihood. Atoms keep the likelihood **exact and continuous** (no
grid quadrature), directly curing the grid NO-GO.

## 2. Why atoms (not an INR or a finer grid)
The phase-marginalized likelihood (Direction II) is

    K_S(P,P') = Σ_k S(k)·cos(k·Δx − ω(k)·Δt),   p(z|S)=N(z; 0, K_S+σ²I).

- **Grid S:** the sum is a quadrature over 64×64 cells → quantizes (kx,ky) → the
  Direction-I precision wall. Finer grid ⇒ O(N²) cost blow-up.
- **Atomic S = Σᵢ aᵢ δ(k−kᵢ):** the sum becomes Σᵢ aᵢ·cos(kᵢ·Δx − ω(kᵢ)·Δt) — an
  **exact, continuous, cheap** kernel, differentiable in (kxᵢ,kyᵢ,aᵢ). This is the
  same functional form as the WaveSplat forward model, so the representation is
  *native* to the problem. An INR S=f_ψ(k) would still need quadrature to form K_S
  → back to a grid. **Atoms are the principled grid-free choice.**

## 3. Representation
- Fixed cardinality **M = 32 atoms** (≥ max true components 30). Atom feature:
  `(kx, ky, ℓ)` with ℓ = log-energy. Active atoms carry energy; padding atoms sit
  at the origin (k=0, non-physical) with floor energy ℓ_min → contribute ~0 to K_S
  and are a learnable "inactive" convention.
- Normalize: kx,ky ∈ [−1.1,1.1] → /1.1; ℓ = log(A²+ε) min-max to [−1,1]. Diffusion
  runs in this normalized atom space (32×3 tensor).

## 4. Prior model — permutation-equivariant set diffusion
- **Data:** `wavesim.wave_generator.jonswap_components(Hs,Tp,nc,γ,dir,spread)` — the
  *same generator the episodes are drawn from* (no train/test spectrum mismatch).
  Sample (Hs∈[0.8,3.2], Tp∈[5,11], nc∈[5,30], dir∈[−50,50]°, spread∈[12,40]°,
  γ∈[1.5,4]); take atoms (A,kx,ky); pad to 32. ~16k samples.
- **Score net:** DDPM ε-prediction; a small **set Transformer** (self-attention is
  permutation-equivariant) over the 32 atoms + sinusoidal diffusion-time embedding.
  d_model≈128, 4 attention blocks. MSE ε-loss, cosine β-schedule, T=500.
- Sanity: unconditional samples must reproduce energy / dispersion-shell placement /
  directional spreading of held-out JONSWAP draws.

## 5. Posterior — DPS with physics likelihood (grid-free)
Reverse diffusion from noise; at step t:
1. ε̂ = score(x_t, t) → x̂₀ (denoised atoms) → decode (kxᵢ,kyᵢ,aᵢ).
2. Likelihood guidance (Chung et al. 2023):
   `∇_{x_t} log p(z | atoms(x̂₀))`, with log p = −½ zᵀ(K+σ²I)⁻¹z − ½ log|K+σ²I|,
   K = Σᵢ aᵢ cos(kᵢ·Δx − ω(kᵢ)·Δt). Autograd through Cholesky.
3. x_{t−1} = DDPM step(x_t, ε̂) + ζ_t · ∇ (guidance), ζ_t ∝ 1/‖∇‖ (DPS scaling).
No projection onto a grid anywhere → fully continuous.

## 6. Evaluation (decisive comparisons)
Field nRMSE at the next frame on BF4_5c/15c, BF3_5c (same episodes as Dir-II):

| method | reference |
|---|---|
| MLE (no prior) | 1.000 (fails) |
| **grid diffusion (Dir-II)** | **0.294** ← must beat |
| grid-oracle | 0.282 |
| exact continuous | ~0.06 ← target |
| WaveSplat fused | 0.205 |

**Go criterion:** grid-free DPS < 0.294 (beats grid prior) and ideally approaches
exact-continuous, demonstrating the grid was the bottleneck (not the prior idea).
Also report posterior-sample spread as **calibrated UQ** (ties to Direction I
predictable-zone / (C)).

## 7. Honest risks
- Direction I proved single-window has little *fine-direction* information → DPS
  recovers prior-typical directions, not necessarily the true fine direction. The
  win is expected in **field nRMSE + calibrated UQ**, not in pinpointing direction.
  Framed accordingly.
- DPS guidance scale / Cholesky stability / atom collapse (mode-seeking) are the
  practical failure modes; mitigations: jitter, guidance annealing, energy floor,
  multiple posterior samples.
- If grid-free DPS does **not** beat 0.294, that is itself a strong result: it would
  show the bottleneck is information (Direction I), not representation — closing the
  loop honestly either way.

## 8. Positioning vs prior art
- vs **classical DSWP / CN-RLS** (optimization, expensive in multidirectional seas):
  learned prior replaces hand-tuned regularization, amortizes the expensive init.
- vs **atomic-norm gridless** (DOA): data-driven learned prior instead of a generic
  sparsity/atomic-norm penalty.
- vs **grid DL spectrum estimators** (radar-spectrum→directional-spectrum→FFT): we
  are grid-free + physics-likelihood, avoiding the fatal quantization.
- vs **point-cloud diffusion** (Luo&Hu 2021) / **DPS** (Chung 2023): we instantiate
  set diffusion on spectral atoms with a dispersion-constrained GP-marginal
  likelihood — a domain-novel inverse-problem solver.

Refs: Luo&Hu CVPR2021 (2103.01458); Chung et al. ICLR2023 DPS (2209.14687);
atomic-norm super-resolution (Candès–Fernandez-Granda; Tang et al.); DSWP:
Desmars/Qi phase-resolved, CN-RLS S-band radar (RemoteSens 2024), Flash-LIDAR
nonlinear reconstruction.
