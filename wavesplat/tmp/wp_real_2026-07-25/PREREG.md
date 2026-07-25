# PRE-REGISTRATION — STAGE-3b: WavePrior learned set-proposal, REAL-DATA transfer gate

Frozen 2026-07-25, BEFORE the full 30-window run. Branch `learn-boundwave-retrain`.
Author: Claude (agent). No commits, no paper edits, no training on real data, defaults
untouched, checkpoints read-only. Companion result: `WP_REAL_2026-07-25.md`.

## 0. Context and framing (confirmatory-with-known-direction)
Stage-1/2 established the learned set-proposal (beamform-conditioned set-diffusion + 80-iter
polish, `scripts/d3_beamform_cond_diffusion.py`, ckpt `results/direction3/beamform_cond_diffusion.pt`)
wins crowded SYNTHETIC seas decisively and IN-DISTRIBUTION (39/40, median -0.105 nRMSE/Hs,
`WP_CROWDED_HEADTOHEAD_2026-07-25.md`) but LOSES sparse synthetic seas (0/12). Real Bergamasco
stereo records are the held-out transfer test. The standing caveat
(`d3_beamform_cond_diffusion.py:12`) is that synthetic-clean nets fail to transfer to real
broadband+noisy data. Real records are swell-dominated => closer to the SPARSE regime where the
proposal LOSES; therefore a pooled LOSS here is the EXPECTED outcome under the niche hypothesis
and does NOT invalidate the synthetic crowded claim. A WIN would EXTEND the niche to real data.

## 1. Arms (frozen)
- **Hand arm** = `mmse_bank_estimate` (the analytic-prior "WaveSplat" arm of `d3_compare_gifs.solve`):
  single deterministic solve per window, `sigma0=0.05*Hs`, `kp_hint=sc['kp']`, `seed=sc['seed']`
  (= i0). This is the identical estimator/config used by `run_wp_realdata` and `doppler_real`;
  its per-window nowcast is bit-identical to `doppler_real.json:nrmse_now_u0`.
- **Learned arm** = proposal components from the REAL-obs beamform map -> polish -> field.
  Obs subsampled to the net's M_OBS=400 (seeded rng(0), matching the crowded head-to-head and
  `learned_init.M_OBS_CAP`); times shifted so min=0 for the beamform map / polish. Per diffusion
  seed: draw n=4 posterior atom sets, 80-iter Gauss-Newton polish each, LS-fit A,phi on the
  window obs (`atoms_to_field`, `fit_amplitudes_phases` convention, ridge 0.5), render field,
  average the 4 fields (BMA, no oracle selection). **Median-of-5 diffusion sampling seeds
  (0,1,2,3,4).** No training, no GT in the loop, no per-window tuning.

## 2. Primary endpoint (frozen)
- Metric = **paired per-window nowcast map nRMSE** = the established `d3_compare_gifs.metrics`
  `map_nrmse_nowcast`: RMSE at lead 0 normalized by `sg = std(all finite GT pooled over all
  leads)`. IDENTICAL normalization for both arms, on the same GT, same query grid. (Note: this
  is /gt_std, the real-data metric of `run_wp_realdata`/`doppler_real` -- NOT the /Hs of the
  crowded run.)
- Comparison unit: per-window paired `Delta = nRMSE(hand) - nRMSE(learned)`; positive = learned
  better. Learned value = median over the 5 diffusion seeds; hand value = the single solve.
- **Population = 30 windows** (5 records x 6 windows: YS17, AA15a, AA14, AA15b, LJ18g; win 0-5).
  **Decision set = the 27 NORMAL windows**, pre-excluding the 3 known collapse windows
  (hand nowcast >= 1.0 in wp_realdata_fixed / doppler_real): **AA14_w0, AA15a_w3, LJ18g_w3.**
- Test: **two-sided Wilcoxon signed-rank on the 27 normal deltas, alpha = 0.05** (direction
  genuinely unknown). Report median Delta, win count, rank-biserial effect size, p.

### Interpretation rules for all four outcome cells (FROZEN, before seeing numbers)
1. **LOSS on normal-27 (median Delta<0), collapse NOT fixed** => niche is synthetic-crowded only;
   real swell-dominated seas behave like the sparse regime; transfer caveat CONFIRMED. Thesis
   niche claim = "in-distribution crowded synthetic"; real data reported as an honest negative.
2. **LOSS on normal-27, collapse FIXED** => the proposal does not help typical real windows but
   the learned prior repairs the analytic path's catastrophic basin failures. Report as a
   conditional robustness benefit (failure-mode insurance), not a headline accuracy win.
3. **WIN on normal-27 (median Delta>0, p<0.05), collapse fixed-or-not** => the niche EXTENDS to
   real data: the learned prior beats the analytic prior on held-out real seas. This is the
   strong outcome and would be a genuine thesis headline (subject to the same-solver, same-obs,
   zero-shot controls).
4. **WIN in point estimate but p>=0.05** => BOUND: suggestive but underpowered at n=27; report
   effect size + CI, do not claim.

## 3. Secondary endpoints (reported, NOT decision-primary)
- **(a) Collapse windows** (AA14_w0 hand 2.31, AA15a_w3 1.12, LJ18g_w3 2.93): does the learned
  arm avoid the hand arm's collapses? Report per-window hand vs learned; "fixed" = learned < 1.0.
- **(b) Doppler-gated 2x2 on records passing the 3a U-stability gate (YS17 ALONE)**: apply the
  record-MEDIAN U_hat from `doppler_real.json` (YS17 median = (-0.95,+0.60), |U|=1.12 m/s) to
  BOTH arms. Report the 2x2 {hand,learned} x {U=0, U=U_med} mean nowcast nRMSE over YS17's 6
  windows, and the paired deltas. Doppler applied as `omega = disp(k) + k.U` in both the bank
  design (hand, existing `Uc=` arg) and the learned LS-fit+render (additive helper, verified ==
  `atoms_to_field` at U=0 to ~1e-5). Bank/atom wavenumbers are U-independent (only frequency
  shifts), per 3a.
- **(c)** per-record splits; **effective component count** of the proposals (n_eff after the
  rel-energy 0.03 / max-30 selection) -- are real seas proposed as sparse (few atoms) or crowded?
- Lead-averaged map nRMSE (mean over leads) as context.

## 4. Execution (frozen)
- Learned arm: GPU (RTX 4080S), diffusion T=500, n=4 BMA, polish 80 iters, seeds 0-4.
- Hand arm: CPU, <=4 workers, OMP=1, nice -10.
- Beamform-map validity CHECK (mission-required): per window log the real-obs beamform peak
  (kx,ky) and whether it hits the k-grid edge [-0.6,0.6]; if peaks systematically clip, the net
  is out-of-domain and that is reported as a transfer limitation.
- Budget ~2h; coarsen to 3 diffusion seeds only if projecting over (not expected).
- All outputs under `tmp/wp_real_2026-07-25/` only. `data/` episodes and checkpoints untouched.

## 5. Scope this test can / cannot license
- A WIN licenses: "the learned set-proposal transfers to held-out real stereo seas, beating the
  analytic prior with the same solver / same obs / zero-shot." A LOSS licenses: "the niche is
  in-distribution crowded synthetic; real swell seas fall in the sparse regime where the analytic
  path already wins" -- and does NOT weaken the synthetic crowded PASS.
- Neither outcome resurrects the refuted gamma/bank-MMSE-prior superiority axis
  (`project_waveprior_promotion_2026-07-24`); this is a different object/endpoint/comparison.
- No default change, no paper edit, no commit is authorized by this test.
