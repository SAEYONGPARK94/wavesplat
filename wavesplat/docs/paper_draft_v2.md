# Paper draft scaffold (v2) — 2026-06-28

Working title: **"The information limits of phase-resolved ocean-wave estimation from
sparse ship-borne LiDAR — and a deployable, calibrated estimator that respects them."**

Target: Ocean Engineering / IEEE J. Oceanic Eng. / OCEANS. Framing per
`results/direction3/FACT_CHECK_2026-06-27.md` §C — *limits → why → beat-by-design →
deployable estimator + calibrated UQ.* NOT a "new single-window SOTA" paper.

## Abstract (draft)
Phase-resolved nowcasting of the ocean surface from a sparse, ship-mounted LiDAR is
central to short-horizon wave forecasting for motion control. We derive the
Cramér–Rao bound (CRB) for this inverse problem under the dispersion constraint and
a space-time observation model, exposing (i) an *information wall* — a minimum
observation time T_wall that grows with sea-state complexity, below which no unbiased
estimator beats a trivial predictor; (ii) an *effective synthetic aperture*
D_eff = L_footprint + c_g·T_obs that governs wavenumber resolution; and (iii) a
*predictable zone* set by spectral identifiability rather than the geometric
group-velocity cone. We show, both theoretically and empirically, that the bottleneck
is information, not algorithm: amortized neural estimators collapse, scaling a learned
prior is inert, and post-hoc sharpening fails — all predicted by the CRB. The two
levers that move the bound are (a) CRB-optimal active sensing (scan design) and
(b) sequential time fusion. We realize the latter in a deployable recursive estimator
(dispersion-constrained beamforming + RELAX component management + off-grid
refinement) that, from observations only and with a period-scaled fusion window,
reconstructs the field with field nRMSE 0.13 (Pearson 0.81) across BF3–BF5 and 5–30
components — a 37% improvement over a tuned space-time GP — and provides calibrated
posterior uncertainty. We delimit honest scope (simulation; multi-directional
generalization; a towing-tank validation plan) and position the contribution as the
limits of the problem plus an uncertainty-aware estimator that respects them.

## Section outline (with content + assets)
1. **Introduction** — DSWP/phase-resolved prediction for ship control; sparse
   ship-LiDAR; the gap: limits + honest UQ are missing; contributions list.
2. **Observation model & problem** — z=Σ wᵢcos(kxᵢx+kyᵢy−ωᵢt+φᵢ), dispersion
   ω=√(g|k|tanh|k|d); sparse LiDAR; sim setup (wavesim); metric panel
   (nRMSE/Pearson/VR/skill, zero-predictor baseline).
3. **CRB & information limits (core theory)** — FIM derivation (Jacobian verified
   1e-15); D_eff = L + c_g·T_obs (verified slope −1.09); T_wall(complexity)
   [0.1/4.3/9.7 s]; predictable-zone = identifiability (≠ geometric cone);
   not self-diagnosable from one window. *Figs:* information wall, synthetic aperture,
   predictable zone.
4. **The bottleneck is information (why algorithms don't suffice)** — NN-as-estimator
   collapse; learned prior is mandatory but scaling it is inert (val-loss floor);
   post-hoc sharpening fails. Grid-free atomic set-diffusion + DPS (beats grid; complexity-
   robust; calibrated UQ) — *the prior provides the mandatory bias.* *Figs:* grid-free
   vs grid, UQ calibration.
5. **Beating the wall by design** —
   5.1 **Active sensing** (D-optimal scan): −69% CRB, −32% achieved on complex seas.
   5.2 **Sequential time fusion**: information accumulates; period-scaled window
   T_obs ≈ k·Tp; non-stationarity is not a practical limit (sea coherence ~20 min).
6. **Deployable recursive estimator** — beamform → RELAX add/prune → off-grid refine →
   warm-start; monotone, self-adapting order. **Main result** (Table 1, Fig main):
   0.130 vs ST-DF-GP 0.205 (−37%) across all sea-states, Pearson 0.81, oracle 0.037.
   CRB-gated hybrid (−15%) as a routing alternative; calibrated UQ.
7. **Forecast skill** — propagate components; skill vs lead-time & distance vs
   persistence; tie to predictable zone. *Fig:* forecast skill; GIFs (supp).
8. **Scope, limitations, validation plan** — simulation; linear vs Gerstner (small);
   multi-directional needs ocean basin; **SNU towing-tank plan** (E1–E10); real-data
   future work.
9. **Conclusion** — limits + design + deployable calibrated estimator.

## Main table (Table 1) — to fill from main_results.json
| sea-state | ST-DF-GP | deployable RELAX (P) | oracle |
|---|---|---|---|
| BF3 5c | 0.106 | 0.084 (0.95) | 0.074 |
| BF4 5c | 0.073 | 0.055 (0.98) | 0.044 |
| BF4 15c | 0.232 | 0.131 (0.84) | 0.034 |
| BF4 30c | 0.287 | 0.209 (0.57) | 0.035 |
| BF5 15c | 0.255 | 0.125 (0.86) | 0.016 |
| BF5 30c | 0.274 | 0.178 (0.66) | 0.022 |
| **overall** | **0.205** | **0.130 (0.81)** | **0.037** |

## Figure inventory (all generated, in results/direction1, results/direction3)
fig_information_wall, fig_synthetic_aperture, fig_predictable_zone (Dir-I);
fig_gridfree_vs_grid, fig_uq_calibration, fig_main_results, fig_forecast,
fig_nonstationary, GIFs (supp).

## Honesty checklist (carry into every claim — from fact-check)
- oracle = bound, deployable = realization; state both.
- period-scaled window vs 1s: advantage = time accumulation (operationally free,
  stationarity-safe); say it.
- report full metric panel, never nRMSE alone.
- CRB-gated hybrid gate = sea-state metadata (window-only gate fails).
- simulation only; multi-directional → ocean basin.
