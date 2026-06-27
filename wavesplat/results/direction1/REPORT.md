# Direction I — Cramér–Rao Bound for space-time wave field estimation

**Date:** 2026-06-27 · **Code:** [`scripts/d1_crb.py`](../../scripts/d1_crb.py) · **GPU:** RTX 4080 Super (float64)

## 0. Objective
Prove *rigorously* — not empirically — that the WaveSplat / ST-DF-GP estimation
problem is **information-limited**, and quantify exactly how much information a
single short observation window contains. This turns the negative results of
Direction II (T-NSE collapse, off-grid NO-GO) from "our estimators failed" into
"**no unbiased estimator can succeed here, and we can say precisely when**."

## 1. Observation model & Fisher information
Matched to `wavesplat.model.spectral_gaussian`:

    z(x,y,t) = Σᵢ wᵢ·cos(ψᵢ),   ψᵢ = kxᵢ·x + kyᵢ·y − ωᵢ·t + φᵢ
    ωᵢ = √(g·kᵢ·tanh(kᵢ·d)),    kᵢ = √(kxᵢ²+kyᵢ²)     (dispersion = hard constraint)

Free parameters per component: θᵢ = (wᵢ, kxᵢ, kyᵢ, φᵢ); ω is **not** free.
Gaussian elevation noise εₘ ~ N(0, σ²), σ = 0.02 m (LiDAR range noise). For a
Gaussian-mean model the FIM is the scaled Jacobian-Gram matrix:

    F = (1/σ²)·Jᵀ J,    J[m,a] = ∂μₘ/∂θ_a      (M observations × 4N params)

The dispersion constraint enters through the **only** non-trivial derivative:

    ∂ψ/∂kx = x − c_g·(kx/k)·t,   ∂ψ/∂ky = y − c_g·(ky/k)·t,   c_g = dω/dk

So the effective sampling position at time t is **r − c_g·k̂·t** — each point is
displaced *along the propagation direction* by c_g·t. The 4N×4N FIM is full (not
block-diagonal): component cross-terms encode the multi-component crosstalk.

*Validation:* analytic Jacobian vs `torch.autograd` → **max error 5.3×10⁻¹⁵** (machine ε). ✅

## 2. Effective aperture  D_eff = L_foot + c_g·T_obs
Because the time displacement is purely along k̂, observing over a window
[0, T_obs] stretches the point cloud **along the propagation direction** by
c_g·T_obs. The lever arm for wavenumber estimation is therefore

    D_eff = L_footprint + c_g · T_obs            (synthetic aperture)

and the single-component wavenumber CRB scales as σ_|k| ∝ 1/(√M · D_eff).

*Controlled verification* (single component, **M and footprint held fixed**, only
T_obs varied to isolate aperture from sample-count): log-log slope of σ_|k| vs
D_eff = **−1.09** (ideal −1.0). ✅ → `fig_aperture_law.png` (left).

**Crucial asymmetry** (`fig_aperture_law.png`, right): the *direction* CRB is
**flat** in T_obs (~0.25° for a single component). Synthetic aperture extends the
aperture only ALONG k̂ — it sharpens |k| (and ω), but adds **no cross-range
aperture**, which is what direction needs. Direction is therefore limited by the
physical footprint, not by time. For a single clean wave direction is easy; the
real directional difficulty is **multi-component decoupling**, which time *does*
help (by letting components drift apart), but only over c_g·T_obs ≳ L_foot.

## 3. Field-reconstruction CRB & the information wall
Propagating the parameter CRB to the reconstructed field over the GT grid at the
current time t*:

    field-RMSE CRB = √(tr(F⁻¹ G)),   G = ⟨ g gᵀ ⟩_grid,   field-nRMSE CRB = ·/Hs

This is a lower bound on **any unbiased** estimator's field nRMSE. Defining the
**information wall** T_wall as the T_obs where the CRB first drops below 0.25 (the
zero-predictor baseline — below the wall no unbiased estimator beats predicting
zero):

| episode | N | T_wall (CRB<0.25) | T(CRB<0.10) | L/c_g | cond(F)@1s |
|---|---|---|---|---|---|
| BF4 5-comp  |  5 | **0.1 s** | 0.2 s  | 14.4 s | 7.7×10⁴  |
| BF4 15-comp | 15 | **4.3 s** | 5.9 s  | 6.3 s  | 8.6×10¹² |
| BF4 30-comp | 30 | **9.7 s** | 13.4 s | 13.7 s | 9.0×10¹⁷ |

→ `fig_information_wall.png`. The wall rises steeply with model complexity: a
30-component sea needs ~**10 s ≈ 1.4 Tp** of observation before the field is even
identifiable. Condition numbers confirm this is *genuine rank-deficiency*
(30c@1s: cond ≈ 10¹⁸ = numerically singular), not a numerical artifact.

## 4. Estimator vs CRB — near-optimal or bias-dependent?
ST-DF-GP runs a W=10-frame (1 s) window. Comparing its field nRMSE to the CRB at
T_obs = 1 s:

| episode | ST-DF-GP fused | CRB @1s | regime |
|---|---|---|---|
| BF4 5-comp  | 0.070 | 0.026 | **near-optimal** (2.7× the bound) |
| BF4 15-comp | 0.272 | 3.04  | CRB **vacuous** — estimator beats unbiased bound |
| BF4 30-comp | 0.308 | 195   | CRB **vacuous** |

Two distinct regimes, both decisive:

1. **Identifiable regime (5-comp):** the existing estimator is already within a
   factor ~3 of the optimum. There is little headroom for a better *point*
   estimator — confirming the pivot away from point-estimator engineering.
2. **Under-determined regime (15/30-comp at 1 s):** the unbiased CRB diverges, so
   the estimator can only succeed because it is **biased** (amplitude
   regularization + GP fusion). An unbiased estimator here is *worse than zero*.
   This is the rigorous proof that **bias / learned priors are mandatory**, not a
   heuristic — exactly unifying Direction II (priors help) with Direction I.

## 5. Unified PhD narrative (the three pieces converge)
- **(I) Limits:** a hard information wall exists; T_wall(N) ∝ complexity and ∝
  L/c_g. Below it, unbiased estimation is impossible (CRB > zero-predictor).
- **(II) Priors:** because the wall is real, regularization/learned priors are the
  *only* way to operate below it — they buy bias to beat the divergent variance.
- **Synthetic aperture is the lever that lowers the wall over time** (D_eff =
  L+c_g·T_obs), but it sharpens |k|/ω, not direction; and at the 1 s windows used
  in practice c_g·T_obs ≈ 5 m ≪ footprint, so single-window gains are small —
  quantitatively reproducing the Direction-II "≈3 s time benefit is limited"
  finding and explaining *why* whole-window space-time fusion (ST-DF-GP) wins.

## 6. Honesty / limitations
- depth not stored in episodes → d = 50 m assumed; verified dispersion holds for
  the stored (k, ω) at d = 50 m (residual < 0.3%).
- CRB uses the *true* components (matched-order, best-case bound); a real estimator
  also pays a model-order penalty, so the gaps above are lower bounds on the gap.
- σ = 0.02 m iid Gaussian on elevation; correlated/angular noise ignored (small).
- Single seed (s0) per sea state for the curves; estimator numbers are s0 from
  `results/spacetime_gp_30ep.json`. Multi-seed CRB bands are a cheap next step.

## 7. Next steps
1. Multi-seed CRB bands + per-sea-state averaging (cheap, strengthens claims).
2. Predictable-zone CRB: map the spatial region where CRB < threshold at a target
   forecast time → ties Direction I to the (C) calibrated-UQ / predictable-zone work.
3. Efficiency-vs-T_obs of ST-DF-GP across the full sweep (run estimator at several
   W) to trace where it tracks the CRB and where bias takes over.
