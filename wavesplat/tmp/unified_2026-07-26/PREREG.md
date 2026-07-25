# UNIFIED SYSTEM — pre-registration (frozen before scoring)

Date 2026-07-26. Branch `learn-boundwave-retrain`. Additive code only, defaults untouched
(bit-identity re-verified), no commits, no paper edits, no training, `data/`/checkpoints
untouched. Frozen BEFORE running the Part-B and Part-C scoring scripts.

Mission: turn the measured regime boundary into ONE unified estimator (`init_method='auto'`)
that decides learned-vs-analytic from observations alone, and measure it.

--------------------------------------------------------------------------------
## PART A — code (already implemented, additive)
`init_method='auto'`: at init time, computes the learned proposal, pools it
(`pool_and_select`, identical knobs to the learned arm), and routes to `learned` if
`n_selected >= auto_switch_thresh` (default **8**), else to `v2_konly`. The raw atoms are
cached so a `learned` route does not re-run the GPU proposal. Purely additive; the
`v2_konly` default path is bit-identical (verified 3/3 to 1e-10 by the Stage-1 control).
Router decision surfaced in the run summary as `auto_selected` / `auto_n_selected`.

--------------------------------------------------------------------------------
## PART B — synthetic 100ep switch (GENUINELY PRE-REGISTERED — extra-48 learned
## results NOT yet inspected at freeze time)

### Data / arms (all median-of-5, tracked avg_nRMSE/Hs, phase-fixed tracker)
- **pure-v2** = `tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl` (100ep x 5 est_seeds).
  REUSED. (Verified bit-identical to Stage-1 v2_konly on the 52 overlap.)
- **pure-learned** = Stage-1 `arm_runs.jsonl` learned (52ep) + NEW extra-48 learned lanes
  (`tmp/unified_2026-07-26/arm_runs_extra.jsonl`), same protocol: lanes
  (est_seed,diff_seed) = (42,0),(1,1),(2,2),(3,3),(4,4); atoms from the GPU precompute.
- **switch** = **per-lane assembly** (PRIMARY, faithful to live `init_method='auto'`): for
  each (episode, lane) route by that lane's `n_selected` (>=8 -> learned else v2), take the
  routed arm's tracked nRMSE for that lane, then median-of-5 across lanes. Validated by
  ~10 live `init_method='auto'` runs reproducing the assembly (routing + tracked nRMSE).
- **oracle** = per-episode min(pure-learned, pure-v2) median (NOT deployable; reference).
- Secondary robustness: per-episode-median-signal assembly (one route per episode from
  median n_selected, the Stage-1 analyze.py convention).

### Regime slices (frozen definitions)
- **crowded** = 15comp + 30comp (n=40). Non-inf reference = pure-learned.
- **sparse**  = 1comp + 2comp (n=34). Non-inf reference = pure-v2.
- **moderate** = 5comp (n=16). Reported as context (not a pass/fail slice).

### Frozen success bar (PRIMARY)
1. **Aggregate**: switch 100ep mean <= min(pure-learned mean, pure-v2 mean).
2. **Crowded non-inferiority**: switch vs pure-learned, paired two-sided Wilcoxon (n=40).
   PASS iff NOT (median Δ(switch−pure-learned) > 0 AND p < 0.05) — i.e. switch not
   significantly worse than pure-learned on crowded.
3. **Sparse non-inferiority**: switch vs pure-v2, paired two-sided Wilcoxon (n=34).
   PASS iff NOT (median Δ(switch−pure-v2) > 0 AND p < 0.05).
DECISION = PASS iff all three hold.

### Sensitivity (reported, not gating)
- Thresholds {6, 8, 10} (per-lane): aggregate + slice non-inf for each.
- Switch-signal distribution: per-episode median n_selected; count near-boundary episodes
  (n_sel in [6,10]).
- **Misroutes**: episodes where the per-episode route picked the worse arm (vs oracle),
  with cost = routed median − oracle median.

--------------------------------------------------------------------------------
## PART C — real-data collapse fallback (detector FROZEN here; design is IN-SAMPLE on
## these 30 windows — stated plainly)

**In-sample caveat (explicit).** The detector below was designed by inspecting the GT-free
signals on the same 30 Bergamasco windows it is scored on (n is what it is: 3 known collapse
+ 27 normal). The falsifiable claim is **specificity + no-harm on these 30** (fires on the
collapses, does not fire on normals); generalization to *future* windows is the true test and
is NOT claimed here.

### Candidate GT-free signals evaluated (max 2, per mission)
1. **Held-out obs-residual level** `vr_u0` (from `doppler_real.json`): discriminates the
   LJ18g storm record globally (all 6 windows vr>0.75) but NOT collapse-specifically —
   AA14_w0 and AA15a_w3 collapses sit at the normal median (~0.185). As an OR term it would
   add false fires on LJ18g normal windows. **Rejected as a firing signal.**
2. **Predicted-vs-observed amplitude ratio** `amp_ratio_grid` = std(analytic nowcast field
   over the query grid) / std(observed z) (re-scored, hand arm bit-identical to wp_real,
   max|diff|=0). Fires on the amplitude-explosion signature of a basin collapse in the
   (extrapolated) reconstruction. **SELECTED.**

### FROZEN detector
**Fire iff `amp_ratio_grid > 1.5`.** (Single signal. Threshold 1.5 = reconstruction carries
>50% more amplitude than observed, physically implausible for a linear fit; it sits well
above the highest normal window observed in design.)

### FROZEN fallback arm
hand (`mmse_bank_estimate`) on every window; **if the detector fires, use the learned-proposal
nowcast** (`learned_now` from wp_real.json) for that window instead.

### Frozen success criteria (report all three)
(i) **Detection**: fires on >= 2/3 known collapse windows.
(ii) **Specificity / no-harm**: false-fire rate on the 27 normal windows, and its cost —
     fallback vs pure-hand on the 27 normal windows must be paired n.s. (a zero false-fire
     rate makes this an exact tie).
(iii) **Aggregate**: fallback all-30 mean vs pure-hand all-30 mean.

--------------------------------------------------------------------------------
## Compute plan
- Extra-48 learned arm: GPU precompute (240 proposals) + CPU tracker (240 lanes, 4 workers,
  OMP=1, nice-10). Same `_worker` as Stage-1 run_arms.py, learned arm only.
- ~10 live `init_method='auto'` demo runs on crowded episodes (routing+result validation).
- Part-C: hand-arm re-score already done (`hand_amp.json`, bit-identical control passed).
- All reused jsonls: batchC, Stage-1 arm_runs/atoms_cache, wp_real, doppler_real.
