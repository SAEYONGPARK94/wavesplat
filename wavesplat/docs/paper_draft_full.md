# Full draft prose (v2) — 2026-06-28

> Draft sections for the paper scaffolded in `paper_draft_v2.md`. Prose is
> publication-oriented but to be edited; all numbers trace to committed JSON/logs in
> `results/`. Honesty checklist applied throughout.

## 1. Introduction
Short-horizon, phase-resolved prediction of the ocean surface ahead of a vessel
underpins feed-forward control of ship and offshore-platform motions, wave-energy
converters, and safe crane/helideck operations. A ship-mounted LiDAR offers dense
but spatially **sparse and non-uniform** elevation returns over a limited footprint,
from which one must reconstruct — and forecast — the surrounding wave field. Existing
deterministic sea-wave-prediction (DSWP) methods (radar/LiDAR optimization, grid
spectral fits, ensemble assimilation) report point accuracy but rarely (i) state the
*fundamental limits* of what is recoverable from such sparse data, or (ii) provide
*calibrated uncertainty*. This paper supplies both. We derive the Cramér–Rao bound
(CRB) for the dispersion-constrained, space-time, multi-component inverse problem,
use it to explain *why* a large class of estimators (including modern neural ones)
cannot do better, identify the two physical levers that move the bound, and build a
deployable recursive estimator that realizes them while reporting calibrated
uncertainty. Contributions: (1) a CRB/identifiability theory for ship-LiDAR wave
estimation, including an information wall, an effective synthetic aperture, and an
identifiability-based predictable zone; (2) evidence — theoretical and empirical —
that the bottleneck is information, not algorithm; (3) two CRB-derived levers
(active sensing, sequential time fusion); (4) a deployable estimator beating a tuned
space-time GP by 37% across BF3–BF5 with calibrated UQ and multi-second forecast
skill; (5) an honest scope and a towing-tank validation plan.

## 2. Observation model & problem setup
The surface is modelled as a sum of linear free waves
z(x,y,t)=Σᵢ wᵢ cos(kxᵢx+kyᵢy−ωᵢt+φᵢ) with the dispersion constraint
ωᵢ=√(g|kᵢ|tanh|kᵢ|d) (a hard projection, not a free parameter). Sparse LiDAR returns
{(xₘ,yₘ,tₘ,zₘ)} carry Gaussian elevation noise σ≈0.02 m. We evaluate against a
pre-registered panel — nRMSE, Pearson ρ, variance ratio (VR), skill score — to guard
against the over-smoothing that lets a near-zero predictor score nRMSE≈0.25 on a
zero-mean field. Simulation episodes span BF3–BF5 and 5–30 components.

## 3. CRB and the information limits (core theory)
For the Gaussian-mean model the Fisher information is F=(1/σ²)JᵀJ over θ=(w,kx,ky,φ)
per component (ω=ω(k) not free). The only non-trivial derivative,
∂ψ/∂kx = x − c_g(kx/k)t, shows that each observation at time t acts at the displaced
position r−c_g·k̂·t. Over a window this stretches the effective aperture along the
propagation direction:

  **D_eff = L_footprint + c_g·T_obs**  (verified: σ_|k| ∝ 1/D_eff, log-log slope −1.09).

Propagating F⁻¹ to the reconstructed field gives a CRB on field nRMSE. Defining the
information wall T_wall as the observation time at which the unbiased CRB first drops
below the zero-predictor (nRMSE 0.25), we find T_wall ≈ 0.1 s (5-comp), 4.3 s
(15-comp), 9.7 s (30-comp) — it grows steeply with complexity, and the FIM condition
number separates complexity by ~15 orders of magnitude. The predictable zone — where
the field can be reconstructed at a target time — is set by *spectral identifiability*
(which components the geometry resolves), **not** by the geometric group-velocity cone
of classical DSWP. Crucially, identifiability is **not self-diagnosable from a single
window**: five data-only signals fail to separate complexity, so the unbiased problem
cannot even tell it is under-determined.

## 4. The bottleneck is information, not algorithm
Three findings, all predicted by the CRB. (a) **Amortized neural estimators collapse**:
input→spectrum/field regressors (transformer/CNP/neural-process/KalmanNet variants)
revert to the mean — there is no stable target below the wall. (b) **A learned prior
is mandatory but scaling it is inert**: a grid-free atomic *set-diffusion* prior used
inside a physics posterior (diffusion posterior sampling on the phase-marginalized
GP-marginal likelihood) takes the failed MLE (nRMSE 1.0) to 0.25 and beats a grid
prior (0.29) and even the grid oracle (0.28); but a 10× scale-up (model, data, EMA)
leaves the field error essentially unchanged — the ε-loss hits an irreducible floor.
(c) **Post-hoc sharpening fails**: max-likelihood-sample selection and off-grid
single-window refinement restore amplitude but wreck structure. One cannot sharpen
past the wall; one must add information. The learned prior's role is exactly the
*mandatory bias* the CRB demands, and it ships calibrated posterior uncertainty
(one global temperature → ±1σ coverage 0.68), whose spatial structure matches the
CRB predictable zone (corr 0.65).

## 5. Beating the wall by design
**Active sensing.** Because the field-CRB depends on *where* one samples, a D-optimal
choice of returns reduces the CRB by up to 69% on identifiable seas and the *achieved*
error by 32% on a complex sea (a cross-range-spread heuristic, needing only the
dominant direction, keeps −20%). Scan/return design is thus a real, deployable lever.
**Sequential time fusion.** Accumulating a track supplies the missing information: with
an efficient estimator the complex-sea field-error falls from ~0.13 (1 s) to 0.02–0.03
(15 s). The required window scales with the wave period (synthetic aperture in
wavelengths), and — critically — this is *stationarity-safe*: the sea-state coherence
time is ~20 min (≈120 wave periods), orders of magnitude longer than the window, so
the period-scaled window (≈2 Tp, 12–20 s) sits deep in the stationary regime
(non-stationarity error <0.02). Single-window GPs fail on complex seas precisely
because they fuse only ~1 s, below the wall.

## 6. Deployable recursive estimator & main result
We realize the time-fusion lever from observations only: accumulate the track →
dispersion-constrained beamforming → RELAX-style matching-pursuit component add with
amplitude pruning → off-grid atomic refinement → warm-start across the growing window.
Across **all 30 benchmark episodes × 5 frames**, with a period-scaled window, the
estimator reconstructs the field with overall **nRMSE 0.130 (Pearson 0.81, skill
0.64)** versus a tuned space-time GP at **0.205 (−37%)**, beating it on **every**
sea-state (BF3-5c…BF5-30c) with real reconstruction (Pearson 0.57–0.98, VR 0.77–0.84),
approaching the information bound (oracle 0.037). A CRB-gated hybrid (route precise-GP
vs robust-prior by the FIM condition) gives −15% as an alternative when a sea-state
estimate is available. *(Honest: the period-scaled window uses more observation time
than the 1-s GP; the advantage is time accumulation — operationally free under
continuous scanning, stationarity-safe, and unavailable to the over-smoothing GP. The
oracle gap is component-identification headroom.)*

## 7. Forecast skill
Propagating the estimated components by dispersion yields a phase-resolved forecast.
Its skill is **flat over 0–5 s lead** (Pearson 0.95–0.98, skill 0.90–0.96), whereas a
persistence baseline collapses after ~1 s (ρ→negative by 1.5–2 s). Forecast error
grows with downstream distance, consistent with the predictable zone. This multi-second
phase-resolved forecast is the practical payoff for feed-forward motion control.

## 8. Scope, limitations, validation
Simulation-based; the linear model differs from Gerstner GT by O(ε²)≈0.25% at BF4.
Multi-directional generalization is demonstrated in simulation; a towing tank is
uni-directional, so a multi-directional ocean basin is required for that regime. We
provide a 10-experiment SNU towing-tank plan (sensor/forward-model fidelity, the
information wall vs window, active sensing, forecast skill, calibrated UQ). Real-data
validation is future work.

## 9. Conclusion
The limits of phase-resolved ship-LiDAR wave estimation are governed by information,
which the CRB quantifies; the way past them is information-gathering by design
(sensing + time fusion), realized by a deployable recursive estimator that beats a
tuned GP by 37%, forecasts multiple seconds ahead, and reports calibrated uncertainty.
