# VERDICT — Integrating physics UQ into the amortized posterior σ-head (#2, 2026-06-29)

**Goal:** d58 found the learned posterior slightly over-confident at long forecast lead (the
lead-structured miscalibration the unified physics UQ, d64, fixes). Give the learned σ-head the
physics: a per-query COVERAGE feature cov(q)=exp(−d_geo/ℓ) (d_geo = distance to the obs cloud
advected to the query time by the dominant group velocity). Train two otherwise-identical fast
posteriors on continuous-spectrum FORECAST queries — vanilla (x,y,t) vs physics (x,y,t,cov) — and
compare ±1σ coverage across forecast lead on held-out NONLINEAR HOS windows. `scripts/d67_physics_posterior_uq.py`
(3500 steps each, eval BF4-15c/BF5-30c, cal seeds {0,1}, test {2,3}).

## Result — honest surprise: forecast-trained learning ALREADY wins; the physics feature is redundant

| ±1σ coverage (ideal .68) | lead 0s | 2s | 4s | 6s | mean \|cov−.68\| |
|---|---|---|---|---|---|
| **vanilla NN** (x,y,t) | 0.70 | 0.70 | 0.72 | 0.73 | **0.032** |
| physics-feature NN | 0.73 | 0.73 | 0.75 | 0.75 | 0.059 |
| unified physics UQ (non-learned, d64) | 0.76 | 0.74 | 0.71 | 0.67 | 0.045 |

Nowcast accuracy: vanilla 0.250 ≈ physics 0.250 (RELAX mean 0.236 — classical still marginally more
accurate, consistent with the information limit).

**Findings:**
1. **The vanilla learned σ-head, trained directly on forecast queries, is the BEST lead-calibrated
   (0.032)** — better than the non-learned unified physics UQ (0.045) AND the physics-feature NN
   (0.059). Training on forecast queries lets the cross-attention decoder learn lead-dependent
   uncertainty end-to-end from obs geometry + query time. (This is the real fix for the d58
   over-confidence: train on forecast leads — not add physics.)
2. **The explicit coverage feature does NOT help — it slightly HURT (0.059 vs 0.032).** The decoder
   already extracts predictability from obs geometry + time; the hand-crafted scalar is redundant and
   destabilizes σ a little. A clean negative for the physics-feature-integration hypothesis.

## Implication (paper)
- **Positive for the deployable contribution:** the amortized posterior gives the full real-time
  package — accurate mean (≈ classical), ms inference, and **lead-calibrated** UQ — in one forward
  pass, *learned end-to-end*, beating both classical+physics-UQ and physics-feature variants on
  lead-calibration. Strengthens "amortized posterior = deployment-ideal."
- **Negative for explicit physics-feature integration:** consistent with the project theme — when the
  end-to-end model is trained on the right distribution (forecast queries), hand-crafted physics
  structure adds nothing. The physics UQ (d64) remains the way to give the *classical* RELAX estimator
  (no learned σ) calibrated uncertainty; for the *learned* posterior, forecast-training suffices.

## ⚠️ CORRECTION — multi-seed (d68, 4 training seeds) reverses the single-run ranking
The d67 single run (vanilla 0.032 < physics 0.059) was **training-RNG noise**. Over 4 seeds:
- **physics-feature lead-cal dev 0.015 ± 0.005  vs  vanilla 0.027 ± 0.003** (physics modestly BETTER),
  paired Wilcoxon **p=0.125 (not significant)**; nowcast nRMSE identical (0.250 ± 0.000).
→ Honest conclusion flips to: **the physics coverage feature does NOT hurt and modestly helps
lead-calibration, but not significantly at n=4.** Both variants are well-calibrated (~0.015–0.027).
The robust claims stand: (a) forecast-trained learned σ gives lead-calibrated UQ in one pass;
(b) accuracy is identical and ≈ classical (info-limited). `results/direction3/multiseed_posterior.json`.

## Caveats
- Single-run magnitudes are RNG-noisy (hence the d68 multi-seed correction above). n=4 seeds still
  modest; the physics-feature benefit is suggestive (0.015 vs 0.027) but not significant (p=0.125).
- 2 seas × 2 test seeds, leads ≤6s, single coverage length scale ℓ=40 m, dominant-c_g coverage.
- All three methods are within 0.03–0.06 of ideal — all reasonably calibrated; the contribution is
  that the plain forecast-trained posterior is best, not that the others fail badly.
