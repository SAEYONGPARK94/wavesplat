# Set-diffusion component prior — field→component switch, closed (2026-07-03)

**Ask (psy9412):** move the generative prior from the *field* to the *sparse component set*
(RELAX structure) → resolve underdetermination + get RELAX-level constraint + generative UQ.
Asset: `results/direction3/setdiff_prior_long.pt`.

## What this is
- **Prior:** permutation-invariant set-diffusion (DDPM, ε-pred) over `M=32` continuous
  spectral atoms `(kx, ky, logE)`, trained on JONSWAP component sets from the episode
  generator. Asset `setdiff_prior_long.pt` (d=256, 8 heads, 8 layers, 4.35M params).
  Code: `scripts/d3_setdiff_prior.py`.
- **Posterior:** grid-free DPS — reverse-diffuse the atom set, guided at each step by the
  phase-marginalized **GP-marginal likelihood evaluated on continuous atoms** (no grid):
  `K = Σ aᵢ cos(kxᵢΔx + kyᵢΔy − ω(kᵢ)Δt)`, `log p = −½ zᵀ(K+σ²I)⁻¹z − ½log|·|`.
  Code: `scripts/d3_setdiff_posterior.py`; head-to-head harness `scripts/d30_symmetric_dps.py`.

## Operating point — guidance sweep (`tmp/g_sweep.py` → `tmp/g_sweep_long.json`)
Clean inverted-U; **guid = 0.05 optimal** for both accuracy and UQ:

| guid | 0.00 | 0.05 | 0.10 | 0.20 | 0.35 |
|------|------|------|------|------|------|
| BMA nRMSE | 0.269 | **0.222** | 0.237 | 0.268 | 0.275 |
| UQ pix-corr | 0.15 | **0.29** | 0.24 | 0.21 | 0.11 |

Too-strong guidance over-fits sparse/noisy obs and leaves the prior manifold.

## Head-to-head, LINEAR seas (30 episodes, 2.2·Tp window, **same subsampled obs m≤200 to both**)
`results/direction3/symmetric_dps_long.json`

| sea      | DPS (BMA) | RELAX3 (same obs) | RELAX3 (full, m≈3000) | ST-DF-GP | oracle |
|----------|-----------|-------------------|-----------------------|----------|--------|
| BF3_5c   | 0.232     | **0.159**         | 0.086                 | 0.106    | 0.074  |
| BF4_5c   | 0.196     | **0.098**         | 0.051                 | 0.073    | 0.044  |
| BF4_15c  | **0.201** | 0.251             | 0.102                 | 0.232    | 0.034  |
| BF4_30c  | **0.213** | 0.270             | 0.194                 | 0.287    | 0.035  |
| BF5_15c  | **0.167** | 0.195             | 0.115                 | 0.255    | 0.016  |
| BF5_30c  | **0.175** | 0.218             | 0.164                 | 0.274    | 0.022  |
| **OVERALL** | **0.197** | **0.199**      | 0.119                 | 0.205    | 0.037  |

- At equal observation budget, **DPS ties RELAX overall (0.197 vs 0.199)** and **beats RELAX
  on every complex sea (≥15 components)**, where RELAX's greedy identification struggles.
- RELAX_full 0.119 uses **~15× more observations** (m≈3000). DPS is capped at m≈200 because its
  GP-marginal likelihood is **O(m³) per diffusion step** — the ceiling is the observation
  budget, not the prior.

## Head-to-head, NONLINEAR HOS seas (order M=3, 6 settings) — the decisive win
`tmp/hos_dps_relax.py` → `results/direction3/hos_dps_relax.json`

| setting | DPS | RELAX3 (same obs) | RELAX3 (full 5× obs) | UQ pix-corr |
|---------|-----|-------------------|----------------------|-------------|
| BF3 ×2  | 0.242 | 0.283 | 0.302 | +0.23 |
| BF4 ×2  | 0.201 | 0.261 | 0.264 | +0.34 |
| BF5 ×2  | 0.182 | 0.232 | 0.217 | +0.35 |
| **OVERALL** | **0.208** | **0.258** | **0.261** | **+0.30** |

**DPS beats RELAX on all 6 nonlinear settings at both budgets.** On nonlinear seas RELAX does
*not* improve with 5× more obs — the linear-superposition model saturates against bound
harmonics, while the component prior regularizes against that over-fit. The learned prior's
advantage is clearest exactly where the physics baseline's linearity assumption breaks.

## Generative UQ — calibrated (the thing the field-prior lacked)
Figure `results/direction3/uq_calibration_BF4_15c_s0.png`.
- **Within-episode:** mean per-pixel corr(|err|, std) = **+0.28** (linear) / **+0.30** (HOS);
  +0.40 on the shown complex episode. High-uncertainty pixels are high-error pixels.
- **Across-episode:** Hs-normalized posterior spread vs nRMSE = **+0.80** — the posterior is
  more spread-out exactly on the harder episodes. (Raw meters-vs-normalized gave a spurious
  −0.65; `d30` now stores `spread_hs` and reports the fair number.)

## Honest caveats
- **RELAX wins on simple/sparse linear seas** (few components → greedy physics is near-optimal).
  The component prior's regime is complex/high-order/nonlinear seas.
- The **O(m³)** GP-marginal likelihood is the binding computational limit (m≈200). Reaching
  RELAX_full (0.119) needs a cheaper likelihood (e.g. Nyström / inducing points / RFF).
- Prior trained on **linear JONSWAP**; it generalizes to nonlinear HOS (robustness), but has
  **not** been fine-tuned on real broadband LiDAR — consistent with the earlier sim2real
  finding, real deployment would need realistic-distribution retraining.

## Option A (2026-07-03) — exact Woodbury likelihood, lift the m-cap

**Motivation:** the DPS guidance evaluated the m×m GP-marginal likelihood by dense
Cholesky = O(m³)/step, capping m≈200. But `K_ij = Σ_m a_m cos(φ_i^m − φ_j^m)` is
**exactly rank ≤ 2M**: `K = ΦΦᵀ`, `Φ = [√a_m cosφ^m, √a_m sinφ^m] ∈ R^{m×2M}`,
`φ_i^m = k_m·r_i − ω_m t_i`. So `(K+σ²I)` is low-rank-plus-diagonal and Woodbury /
matrix-determinant-lemma give the marginal likelihood **exactly** in O(m·M²).
`scripts/d3_setdiff_posterior.py::loglik_atoms` rewritten.

**Verified (`tmp/woodbury_test.py`):** vs dense reference, value rel-err ~1e-13,
**gradient** rel-err ~1e-11 (double). Speed (fwd+bwd, GPU): dense m=2000 = 35.75 ms
(m=3000 OOM); Woodbury **~2 ms flat** in m. So **m=3000 costs the same as the old m=200.**

**Head-to-head at m=3000 (`tmp/dps_highm.py` → `results/direction3/dps_highm.json`), guid 0.05:**

| sea      | DPS@3000 | DPS@200 | RELAX@3000(=full) | UQ pix |
|----------|----------|---------|-------------------|--------|
| BF3_5c   | 0.215    | 0.232   | **0.086**         | +0.27  |
| BF4_5c   | 0.191    | 0.196   | **0.051**         | +0.38  |
| BF4_15c  | 0.175    | 0.201   | **0.102**         | +0.40  |
| BF5_15c  | 0.130    | 0.167   | **0.115**         | +0.42  |
| BF4_30c  | **0.190**| 0.213   | 0.194             | +0.33  |
| BF5_30c  | **0.155**| 0.175   | 0.164             | +0.37  |
| **OVERALL** | **0.176** | 0.197 | 0.119          | +0.36  |

- Lifting m 200→3000: overall **0.197→0.176 (−11%)**, UQ **+0.28→+0.36**. At **equal**
  m=3000, DPS **beats RELAX on both 30-component seas**; RELAX wins 15c and simple.
- **Refined thesis:** the learned prior's edge is largest when most **underdetermined**
  (few obs OR high component count OR nonlinear). The m=200 "DPS beats RELAX on all
  complex seas" was partly because RELAX was obs-starved at m=200; with full obs RELAX
  recovers on 15c but still loses on 30c (greedy can't resolve 30 components).

**What the residual simple-sea gap is NOT (ruled out, honest negatives):**
- **Not the observation budget** — m=3000 barely helps simple seas (0.23→0.21).
- **Not the guidance** — sweep at m=3000 (`tmp/g_sweep_highm.py`) keeps **0.05 optimal**;
  higher guidance raises variance and hurts the BMA.
- **Not the posterior summary** — deployable **likelihood-selection overfits**
  (`tmp/mapselect_test.py`): the max-obs-likelihood sample is *worse* than the BMA
  (BF4_5c 0.362 vs 0.234; BF3_5c 0.393 vs 0.218); softmax-weighting collapses to it.
  BMA (uniform mean) is already the robust choice.
- ∴ the simple-sea gap is the **generative posterior's sharpness on sparse spectra**
  (fixed M=32 over-parameterises a 5-spike sea; RELAX adapts cardinality). Fixing it
  would need a variable-cardinality / sparsity-aware prior — a retrain, not a knob.

**Verdict:** A is a clean permanent estimator win (exact O(m³)→O(m·M²)) and helps most
on the hardest seas (30c, where it now beats full-obs RELAX). It does **not** make DPS
win on easy seas — three obvious fixes ruled out. This is strong evidence to pivot to B
(lean on the underdetermined/nonlinear regime + calibrated UQ), not chase RELAX on easy
cases.

## Novelty framing (lit-checked, measured)
Generative DPS for inverse problems is established (image/field domain; NO spectral-bias,
SAR). Not found: a **permutation-invariant set-diffusion prior over continuous wave-spectral
atoms** (CLEAN/RELAX-structured) with a **grid-free GP-marginal physics likelihood** for ocean
nowcasting + calibrated UQ. Contribution = that specific combination; do **not** claim
"first/only". Discriminative baselines (PINO, Ehlers) are point estimators without UQ.
