# PRE-REGISTRATION — Current-Doppler phase-rate term on REAL stereo data (analytic estimator)

STAGE-3a of the WavePrior program. Frozen 2026-07-25, BEFORE the full 30-window run.
Author: Claude (agent). Branch: learn-boundwave-retrain. No commits, no paper edits.

## Hypothesis (the July attribution under test)
The July GP space-time analysis attributed the sim-real gap to current-induced Doppler:
`omega_observed = omega_intrinsic + k . U`. If that attribution is correct, giving the
analytic bank estimator (`mmse_bank_estimate`) a per-window ambient current U should
REDUCE the real-data nowcast error relative to U=(0,0).

Honest stance: this is a hypothesis TEST, not a demo. A null (no improvement) is a
publishable negative that BOUNDS the Doppler explanation and will be reported as such.

## Implementation (additive, default-off — already present)
The Doppler term already exists in `scripts/d3_sa_nomp.py::mmse_bank_estimate` via the
optional `Uc=(Ux,Uy)` argument (default `None`): when given, `om = om + kxb*Ux + kyb*Uy`
is applied to BOTH the design matrix and the beamform-power prior. Bank candidate
wavenumbers (kxb,kyb) are U-independent; U only shifts each candidate's temporal
frequency. Verified bit-identical: default `Uc=None` == `Uc=(0,0)` (max|diff|=0.0) on
one synthetic scene (linear sim seed 0) and one real window (YS17 w0). No other code in
the estimator/protocol is modified; defaults untouched.

## U estimation (self-contained per window, observation-only — NOT learning)
Per window, U is estimated from the observations of THAT window alone (no training, no
GT, no cross-window fitting):
- Criterion: the estimator's own internal held-out observation residual. Each candidate
  U is scored by calling `mmse_bank_estimate(P, z, depth, Pq_tiny, sigma0=0.05*Hs,
  kp_hint=kp, seed=int(i0), Uc=U, diag_out=gate)` and reading `gate['vr']` = RMSE on the
  25% observation validation split held out inside the solver. The split is seeded by
  `seed=int(i0)`, identical across all U candidates for a window, so the comparison is
  fair. This residual is computed purely on the LiDAR observations; the evaluation GT
  (maps_gt / encounter targets) is NEVER touched during U selection. ("held-in
  observation residual" in the task brief is realized as this in-sample-from-observations,
  GT-free criterion.)
- Search: coarse Cartesian grid Ux,Uy in {-1.5,...,+1.5} step 0.3, restricted to
  |U|<=1.5 m/s (81 points incl. origin) -> pick coarse argmin(vr) -> local refine on a
  7x7, step-0.1 box centred on the coarse winner, clipped to |U|<=1.5 (final resolution
  0.1 m/s). U_hat = argmin(vr) over coarse union refine. |U|<=1.5 m/s is the physical
  plausibility envelope; U_hat pinned at the boundary is flagged.
- Budget: obs-only solve ~0.5 s; ~130 candidates/window x 30 windows ~= 42 min
  single-thread, run with <=3 workers, nice -10, OMP=1 (projected < 20 min wall). The
  full 0.1 grid (709 pts) would be ~3 h (>2.5 h) so it is replaced by coarse+refine, as
  the brief permits.

## Final metric (analytic arm ONLY)
For each window, two full-Pq solves: baseline `Uc=(0,0)` and Doppler `Uc=U_hat`. Metric =
`map_nrmse_nowcast`, computed with the EXACT `d3_compare_gifs.metrics` normalization:
map nRMSE at lead 0, normalizer `sg = std(all finite GT pooled over all leads)`. This is
the identical primary metric used by `run_wp_realdata` (WaveSplat arm). WavePrior (learned
arm) is NOT run here.

Windows: 5 records x 6 windows = 30 (YS17, AA15a, AA14, AA15b, LJ18g; wins 0-5), via
`d3_compare_gifs.scene_real(record, win)` with default W=40, m_scan=12, m_cap=600,
noise=0.02 (same as run_wp_realdata).

## PRIMARY test (frozen)
- Quantity: paired per-window Delta = nRMSE(U=0) - nRMSE(U_hat)  [positive = Doppler helps].
- Population: the 27 NORMAL windows, PRE-EXCLUDING the 3 known collapse windows
  (WS>=1.0 in wp_realdata_fixed): AA14_w0, AA15a_w3, LJ18g_w3.
- Test: one-sided Wilcoxon signed-rank, H1: median(Delta) > 0, alpha = 0.05.
  Zero-Delta windows (U_hat=0 -> bit-identical solves) handled by Wilcoxon's default
  zero_method; count reported.
- VERDICT rule:
  - CONFIRM: Wilcoxon p < 0.05 AND median(Delta) > 0 (Doppler materially helps).
  - REFUTE:  median(Delta) <= 0 (Doppler does not help / hurts).
  - BOUND:   p >= 0.05 with median(Delta) > 0 but n.s. (effect too small to establish;
             bounds the Doppler explanation by the observed effect size + CI).

## SECONDARY (reported, not gating)
- ALL-30 paired Wilcoxon (incl. the 3 collapse windows).
- Per-record splits of Delta.
- The U_hat values themselves: magnitude distribution, physical plausibility (<=1.5 m/s),
  within-record stability (std of |U_hat| and direction spread per record).
- Lead-averaged map nRMSE (mean over leads), U_hat vs U=0, paired.
- Effect on the 3 collapse windows specifically (does Doppler fix any?).

## Deliverable
`tmp/doppler_real_2026-07-25/DOPPLER_REAL_2026-07-25.md` — prereg verdict, paired table
(per window: U_hat, nRMSE before/after), U-stability, MEASURED/ARGUED separation,
implication for the sim-real gap attribution (confirm/bound/refute), and a recommendation
on whether the Doppler term should enter the Stage-3b learned-arm real test.
