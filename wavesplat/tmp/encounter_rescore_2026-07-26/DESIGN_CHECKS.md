# Design checks run before/alongside the encounter re-score (W9-c)

All of these were computed **before any encounter-scored number was read**, and none of them is
an endpoint. They exist so the RESULT can distinguish "the information wall" from "an artifact
of my own scoring geometry".

## 1. Ground truth is exact, not interpolated
`η(x,y,t) = Σ Aᵢ cos(kxᵢx + kyᵢy − ωᵢt + φᵢ)` from the episode's own stored
`config/wave_components` reproduces the stored `gt_wave_field` to **max |Δ| ≤ 2.2e-15 m**
(cells a / c / cbig, frames 0 and 157). Q enters only the Gerstner horizontal displacement,
never Z. ⟹ scoring off-grid distances and off-raster leads is exact, not interpolated.

## 2. Neither arm has a spatial envelope
- analytic: `SpectralGaussianModel.forward` = `Σ wᵢ cos(kxᵢx + kyᵢy − ωᵢt + φᵢ)` (global plane
  waves; the "Gaussian" is spectral, not spatial).
- learned: `wavesplat.model.tnse.render_wave_field` = the same plane-wave sum.

⟹ a point at x = footprint + 200 m is a **legitimate extrapolation of the fitted spectrum**, not
a region where the representation is structurally zero. Any far-field skill decay is physical
(information / phase error), not a modelling artifact.

## 3. The +x encounter axis vs the realized propagation direction
The axis is the generator design constant (`direction_deg = 0`). Realized energy-weighted mean
directions and the resulting lateral offset of the +x ray from the true down-wave ray:

| cell | mean dir | per-episode range | mean \|dir\| | lateral offset at d = 0 / 10 / 22 / 50 / 100 / 200 m |
|---|---|---|---|---|
| **C** | −0.80° | −4.7 … +4.0° | 3.39° | 0.0 / 0.6 / 1.3 / 3.0 / 5.9 / 11.8 m |
| **C-big** | −2.15° | −6.4 … +0.4° | 2.29° | 0.0 / 0.4 / 0.9 / 2.0 / 4.0 / 8.0 m |
| A | −2.82° | −14.8 … +18.7° | 12.04° | 0.0 / 2.1 / 4.7 / 10.7 / 21.3 / 42.7 m |
| A-big | −8.71° | −21.6 … +5.7° | 10.99° | 0.0 / 1.9 / 4.3 / 9.7 / 19.4 / 38.8 m |

**Consequence, stated in advance:** the fixed +x ray is well aligned on the **decision
(continuum) cells** — off-track by ≤ 12 m even at 200 m, i.e. ≤ 0.2 λ_p. On the **discrete
reference cells** a 15-atom realization's mean direction wanders ±15°, so the far-field points
are up to ~40 m off the true down-wave ray. Discrete-cell far-field skill is therefore a
**lower bound**; continuum-cell numbers are not affected in any material way. This is a
consequence of holding the axis fixed (the alternative — a per-episode GT-derived axis — would
leak ground truth into the geometry).

## 4. Observation support beyond each encounter distance
`x_edge` = p95 of the window's valid-return x ≈ **39 m**. Because the grazing-incidence tail
reaches 200+ m, a small fraction of returns lies beyond each encounter point (measured on
cell C, seed 10 uniform, i_ref 157): 5.0 % beyond 0 m, 2.4 % beyond 10 m, 1.0 % beyond 22 m,
0.37 % beyond 50 m, 0.04 % beyond 100 m, 0.0 % beyond 200 m. Reported per run in the JSONL.
⟹ the 22 m point sits at the ~1 % tail of the footprint, the 100 m and 200 m points are
effectively unobserved.

## 5. ⚠ Parent-run arm identity (this is a finding about W9, not about this run)
`tmp/discreteness_2026-07-26/run_ws.py` passes `config={'adaptive_n','seed','device'}` and
inherits the **shipped** `init_method`. That default was flipped `v2_konly → 'auto'` in commit
`aa8e4c3` at **2026-07-26 11:53**; the W9 WS arm ran **14:16–14:42** and the aperture WS arm at
**18:43**. Bit-level reproduction:

| cell / seed | stored `ws_runs.jsonl` init_nRMSE | re-run, DEFAULT (`auto`) | re-run, `v2_konly` |
|---|---|---|---|
| a / 42 | 0.148491 (n_g 12) | **0.148491 (n_g 12) — identical** | 0.234347 (n_g 9) |
| c / 42 | 0.252451 (n_g 18) | **0.252451 (n_g 18) — identical** | 0.284619 (n_g 19) |

Live routing on `cell_c/seed10_uniform`: `auto` → **learned** (n_selected 18, bf_peak_ratio 0.44,
guard did not veto). Corroborating pattern: `n_gaussians == max_n == 30` occurs in
**0/50** cell-A runs but **16/50** cell-C and **16/50** cell-C-big runs, mirroring the learned
proposal's measured `n_selected` saturation on continuum seas.

⟹ **the W9 "analytic arm" was the `auto` router, not the analytic estimator.** Both W9 result
files' `ws_best` / `R` / analytic-skill numbers are affected. This experiment pins
`init_method='v2_konly'` so its analytic arm is analytic, and its lead-0 full-grid anchor
supplies an uncontaminated replacement for the W9 global-map skill.
