# Learned off-grid component estimator — design (2026-06-28)

**Motivation.** Symmetric experiments (`symmetric_dps_VERDICT.md`, `symmetric_gp_VERDICT.md`,
`dps_density.json`) showed: at a fixed information budget every *smoothing* estimator
plateaus (~0.20) — the learned set-diffusion prior (BMA) and the dispersion-kernel GP
alike — while only explicit **off-grid component identification** (RELAX) converts
observation density into accuracy (0.20 -> 0.12 as m: 200 -> 3000). The DPS density probe
isolated the cause: DPS barely improves with m (0.235 -> 0.216 over 200 -> 700), so its
limit is *posterior-mean over-smoothing*, not the O(m^3) likelihood cap.

**Goal.** A learned estimator that outputs SHARP off-grid atoms directly — an *amortized
RELAX* — so it (a) is sharp by construction (renders a sparse cosine sum, no smoothing
kernel) and (b) can exploit observation density. If it works, it gives RELAX-level accuracy
at NN speed; if it does not, it is an informative negative that further hardens "the lever
is off-grid identification, not learning."

## Architecture
Input: observation window O = {(xᵢ, yᵢ, tᵢ, zᵢ)}, i=1..M (sparse LiDAR over 2.2·Tp).

1. **Set encoder** (permutation-invariant): per-point token = MLP(normalize[x,y,t,z]);
   L self-attention blocks over tokens -> context H ∈ R^{M×d}. The time channel t carries
   the synthetic-aperture information (each obs acts at r − c_g·k̂·t).
2. **Atom decoder** (DETR-style set prediction): N_q=32 learnable query embeddings;
   cross-attention(queries -> H) + self-attention; per query head ->
   (kx, ky, log A, φ, presence_logit). Off-grid (kx,ky) are continuous regressions
   (bounded by a tanh·K_LIM), NOT grid bins.
3. **Dispersion hard constraint:** ω = √(g|k|·tanh(|k|d)), k=√(kx²+ky²). ω is never free.
4. **Render head (differentiable, no learned params):**
   η̂(x,y,t) = Σ_q σ(presence_q)·A_q·cos(kx_q·x + ky_q·y − ω_q·t + φ_q).

## Training (synthetic, on-the-fly)
- Sample sea state: Hs∈[0.6,2.0], Tp∈[5.5,9.0]; n_comp∈{5,15,30} via `jonswap_components`
  -> true atoms (kx,ky,A,φ). (Same linear free-wave family as the benchmark/CRB.)
- Render sparse obs at random (x,y) in the footprint and random t in [0, 2.2·Tp]
  (multi-time = synthetic aperture), add σ=0.02 noise -> the input window.
- Targets: (i) GT field η at a target time on a small grid; (ii) the true atom set.

**Loss** = field-reconstruction MSE (rendered η̂ vs GT η at the target frame)
        + λ_match · Chamfer/Hungarian over (kx,ky) weighted by A (correct placement,
          anti-collapse)
        + λ_obs · MSE(η̂ at obs points vs z) (data consistency).
Field-recon makes the model sharp where it matters; the matching term blocks the
mean-collapse that sank earlier amortized regressors.

## Why this can avoid the earlier amortized collapse
Earlier NN estimators (T-NSE / CNP / KalmanNet variants) reverted to the mean because
(a) they regressed a *field/spectrum* (smooth target) and (b) were trained at/below the
information wall (no stable target). Here: (a) the output is a *sparse atom set* rendered
by physics — the loss optimum is sharp, not an average; (b) training is *above the wall*
(multi-time window with enough aperture); (c) Hungarian set loss explicitly penalizes
hedged averaging.

## Evaluation
Feed the SAME observation window as DPS/RELAX on the benchmark complex episodes
(BF4_15c, BF4_30c, BF5_15c, BF5_30c), render the field at the target frame, report nRMSE /
Pearson vs DPS (~0.20), RELAX (0.10–0.16), and oracle. Success = breaks the ~0.20 plateau.

## Status
Prototype: `scripts/d34_learned_offgrid.py` (encoder+decoder+render, on-the-fly train, eval
on benchmark). First feasibility pass; not a tuned final model.
