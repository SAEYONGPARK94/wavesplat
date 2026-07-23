# PRE-REGISTRATION — ETKF fine-mesh regression diagnosis + retune (2026-07-24)

Written **before** any diagnostic number was produced this session. Only two numbers were
looked at first, and they are stated here so they cannot be smuggled in later as
"predictions":

- (F1) `gt_wave_field` is **bit-identical** between `data/` and `data_finemesh_2026-07-23/`
  for the four episodes checked (BF2_1c, BF3_1c, BF3_5c, BF4_15c seed0 uniform), and the
  GT grid is 80x60 in BOTH datasets. The task brief's premise "the GT grid changed from
  40x30 to 200x167" is **false** — 200x167 is the *raycast* mesh, the GT grid never moved.
- (F2) Observation amplitude changed: `std(z_obs)` old -> new = 0.0924 -> 0.1471 (BF3_1c;
  GT std 0.1500), 0.0635 -> 0.0698 (BF2_1c; GT std 0.0750), 0.1350 -> 0.1482 (BF3_5c),
  0.2462 -> 0.2605 (BF4_15c).

So the ONLY thing that changed for any estimator is the **z value of the LiDAR returns**;
the scoring target is unchanged.

## Hypotheses

- **H-REP (representation/shrinkage mismatch).** ETKF's analysis grid (40x30 over
  x in [10,250], y in [-100,100] => dx 6.15 m, dy 6.90 m) cannot represent the GT field.
  Its best possible output has correlation rho < 1 with GT, so the *error-minimising*
  amplitude is rho * (GT amplitude), i.e. shrunk. The old data's raycast mesh attenuated
  the observed z by roughly the same factor, so the old ETKF was accidentally pushed to
  approximately the right (shrunken) amplitude. Correct-amplitude observations push it to
  full amplitude, which over-shoots relative to what the grid can represent => higher RMSE.
  **The old number was flattered by a coincidence, and the fix is a finer analysis grid.**
- **H-STALE (pure hyperparameter staleness).** Ne / inflation / obs_noise / max_obs were
  set against old-mesh statistics; retuning restores parity.
- **H-NORM (normalisation / operator difference).** `Hs_true`, obs counts, confidence or
  the observation operator differ between datasets in a way that biases the metric.
- **H-NOISE (sensor-noise realisation).** Point geometry / return counts differ enough to
  change the effective information content.

## Pre-registered read-out rules (fixed before looking)

1. **R1 — representation floor.** For each ETKF grid resolution G, define the *oracle-grid
   floor*: sample the analytic GT at G's nodes, bilinearly interpolate to the 80x60 GT
   grid, score nRMSE. This floor is identical for both datasets (GT is identical).
   *Read:* if the fine-mesh ETKF nRMSE at 40x30 is at or near this floor while the
   old-mesh ETKF is BELOW it, the old number is provably impossible for an honest
   estimator at that grid and was flattered by the attenuated observations => **H-REP is
   established and the regression is NOT a handicap, it is the removal of a lucky
   cancellation.** ("Below the floor" is possible only via shrinkage that trades bias for
   variance in a way an oracle grid-sampler does not.)
2. **R2 — decisive shrinkage test.** For every frame compute the optimal global scale
   alpha* = <pred,gt>/<pred,pred> and the correlation rho(pred,gt). *Read:* if
   (a) rho is EQUAL OR HIGHER on fine-mesh data, and (b) alpha* is materially < 1 on
   fine-mesh but ~1 on old data, and (c) nRMSE at alpha* removes most of the regression,
   then the regression is an **amplitude/shrinkage artifact**, not a loss of information.
   Threshold: "removes most" = >= 60 % of the coarse->fine nRMSE gap closed.
   If instead rho DROPS on fine-mesh data, the fine-mesh observations genuinely carry less
   usable structure for ETKF and H-REP is refuted.
3. **R3 — grid-resolution sweep.** Sweep the ETKF analysis grid from coarser than shipped
   to much finer. *Read:* if a finer analysis grid monotonically reduces the fine-mesh
   nRMSE and the coarse->fine regression shrinks or reverses at the finer grids, **the
   regression is a configuration artifact** (ETKF was being handicapped) and the retuned
   number is the one to report. If no grid helps, the regression is not representational.
4. **R4 — low-pass control.** Score fine-mesh ETKF against a GT low-passed to the ETKF
   grid's Nyquist. *Read:* if the regression vanishes against low-passed GT, it is
   representation mismatch by construction.
5. **R5 — tuning honesty.** Tuning episodes are declared BEFORE the sweep and are disjoint
   from the reporting episodes. Tuning set = seed-1 episodes of the same 12 scenarios
   (never used for reporting). Reporting set = the paper's 12 seed-0 episodes. The winner
   is chosen by mean nRMSE on the tuning set alone; no reporting-set number may influence
   the choice. Selection metric fixed in advance: mean nRMSE over the tuning set.
6. **R6 — seed handling.** ETKF is stochastic (ensemble init, obs thinning). Every
   reported ETKF number is the **median of 3 filter seeds** (0,1,2), consistent with the
   project's median-of-k convention. Spread (min..max) reported alongside.

## Falsifiers

- H-REP is refuted if the oracle-grid floor at 40x30 is well BELOW the old-mesh ETKF
  number (then the old number was achievable honestly), or if rho drops on fine-mesh data.
- H-STALE is refuted if the best fine-mesh config still regresses vs the best old-mesh
  config on paired episodes.
- H-NORM is refuted by the F1/F2 checks above plus an explicit Hs_true comparison.

## What will NOT be claimed

No claim that WaveSplat's margin over ETKF survives unless the retuned, honestly-held-out
fine-mesh ETKF number is measured against fine-mesh WaveSplat numbers produced under the
same protocol. If WaveSplat fine-mesh numbers are not re-measured here, the margin
statement is explicitly deferred.
