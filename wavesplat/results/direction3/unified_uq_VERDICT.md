# VERDICT — Unified physics UQ for deployable phase-resolved forecasting (#2, 2026-06-29)

**Idea (from `predictable_zone_test_VERDICT.md`):** give the deployable RELAX forecast a calibrated
predictive uncertainty it never had (RELAX/F1 provide NO UQ), grounded in physics:

    σ_total²(x,y,t*) = c·σ_FIM² + (1−c)·σ_sea²,   c = exp(−d_geo/ℓ)

σ_FIM = CRB field-std from the estimated components' Fisher info (near-field), σ_sea = Hs/4 =
zero-predictor std (far-field ceiling), c = group-velocity coverage. One length scale ℓ (+ a global
temperature) calibrated on held-out seeds. `scripts/d64_unified_uq.py`, validated on HOS forecasts.

## Result — modest but honest win: better calibration ACROSS forecast lead
Calibrated on seeds {0,1}, tested on {2,3} (BF4-15c, BF5-30c, ℓ=26 m, 92 160 test cells). Both
models globally temperature-matched to ±1σ=0.68 on calibration.

| | ±1σ (ideal .68) | ±2σ (ideal .95) | corr(σ,\|e\|) |
|---|---|---|---|
| FIM-only (temperature-scaled) | 0.765 | **0.955** | +0.348 |
| **unified FIM⊕geo** | **0.710** | 0.927 | +0.337 |

**Coverage vs forecast lead (the structured miscalibration the physics fixes):**

| lead | FIM ±1σ | UNIFIED ±1σ |
|---|---|---|
| 0s | 0.75 | 0.75 |
| 2s | 0.76 | 0.72 |
| 4s | 0.77 | 0.70 |
| 6s | 0.78 | 0.67 |

- **FIM-only drifts increasingly OVER-covered with lead (0.75→0.78); the unified model stays near
  ideal (0.75→0.67).** Mean |±1σ−0.68| across lead: **FIM 0.082 vs UNIFIED 0.034 (~2.4× better)**.
- The unified model also **generalizes cal→test better on ±1σ** (test 0.710 vs FIM 0.765 — a single
  global temperature on FIM over-covers out of sample; the physics structure transfers).
- It gives the classical deployable estimator a **calibrated predictive σ it previously lacked**.

## Honest caveats / nuance (per standing fact-check directive)
- The win is **modest, not dramatic**: on ±2σ overall, FIM-only is actually slightly better (0.955
  vs 0.927); spatial corr(σ,|e|) is ~tied (0.35 vs 0.34). The unified model's clear advantage is
  **lead-calibration and out-of-sample ±1σ**, not a blanket improvement.
- The predicted mechanism was subtler than the #1 verdict guessed: with ~24 ESTIMATED components the
  FIM field-σ is not uniformly over-confident — after a global temperature it OVER-covers and the
  miscalibration is **lead-structured** (grows with lead). The physics-coverage term removes most of
  that lead drift, which is the actual, honest contribution.
- 2 seas × 4 seeds, forecast leads ≤6s, 1-D length scale ℓ + global temperature. Coverage uses the
  Gaussian |e|<σ ≈ 0.68 convention (forecast bias not separated).

## Where it fits the paper
A small, defensible, physics-grounded UQ result: the deployable estimator gets calibrated
uncertainty that **stays calibrated across forecast lead** by unifying parameter-uncertainty (FIM)
with unmodeled-energy transport (geometric coverage). Consistent with the project theme — physics
grounding buys honest, modest gains; the information limit dominates the rest. Feeds the
amortized-posterior UQ (the learned σ-head can be regularized toward / benchmarked against this
physics target).
