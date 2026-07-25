# UNIFIED GUARD + SPARSE-MERGE — pre-registration (frozen before scoring)

Date 2026-07-26. Branch `learn-boundwave-retrain`. Additive code only, production defaults
untouched (bit-identity re-verified), no commits, no paper edits, no training,
`data/`/checkpoints untouched. Frozen BEFORE running any Task-1 re-assembly score or any
Task-2 tracker run. Builds on `tmp/unified_2026-07-26/` (the `init_method='auto'` router).

Two coupled, pre-registered improvements to the unified estimator:
- TASK 1 (ordered first): a GT-free routing guard that vetoes the learned route on the
  characterized single-clean-swell over-proposal edge (BF2_1comp misroute cluster).
- TASK 2: GT-free atom merge (proposal cardinality calibration) to attempt sparse parity
  for the learned path without retraining.

--------------------------------------------------------------------------------
## DESIGN JUSTIFICATION (done pre-freeze, from recorded diagnostics — see explore_*.py/json)

Characterization of the misroute (why one signal, this signal):
- The learned proposal over-proposes ~30 atoms on BF2_1comp. The DAMAGING misroutes are the
  `slope` variants, where the proposal atoms are **spread across k-space with no dominant
  energy peak** (dom_frac ~0.06-0.13). At the ATOM level these are INDISTINGUISHABLE from a
  genuinely crowded sea: high `n_selected` (~30), high kp-scaled merge count (21-30), low
  dominant-energy fraction. So NO atom-only signal (dom_frac, energy-weighted merge count)
  separates them from crowded (measured: `explore_signals.py`, `explore_falseveto.py`).
- The v2 beamform power map, computed directly from the init-window observations
  (independent of the diffusion proposal), DOES see the truth: a single clean swell yields
  ONE dominant beamform peak. Signal = **`bf_peak_ratio` = P2/P1**, the 2nd-strongest to
  strongest beamform-peak power ratio (v2 coarse beamformer, MIN_DK-suppressed greedy peaks).
- Measured on the 100 finemesh episodes (`explore_beamform.json`): the minimum `bf_peak_ratio`
  among ALL crowded (15+30comp) episodes is **0.0553**. The 4 worst damaging BF2_1comp
  misroutes have `bf_peak_ratio` 0.024-0.030, strictly below that floor. Therefore a threshold
  **< 0.0553** vetoes the clean-single-swell over-proposals with **zero crowded flips by
  construction**. This design is IN-SAMPLE on these 100 episodes (the crowded floor is measured
  on them), stated plainly; generalization to new seas is not claimed.
- Known residual (declared before scoring): `BF2_1comp_seed3_slope` has `bf_peak_ratio`=0.186
  with genuine multi-peak beamform structure (npk>=10% = 3), indistinguishable from crowded;
  it is NOT caught by any 0-crowded-flip threshold and remains misrouted. Its cost (~+0.26 on
  that episode) is the main reason the sparse-mean criterion below may or may not clear +0.005.

--------------------------------------------------------------------------------
## TASK 1 — routing guard (code additive; only affects `init_method='auto'`)

### FROZEN guard
In `_auto_select_init` (entered ONLY when `init_method=='auto'`, which is not the production
default): after computing the learned proposal's `n_selected`, when the base decision would be
`learned` (`n_selected >= auto_switch_thresh`, default 8), compute `bf_peak_ratio` on the init
window and **override to `v2_konly` iff `bf_peak_ratio < auto_guard_ratio` (default 0.05)**.
Additive config: `auto_guard` (bool, default True for the 'auto' path only),
`auto_guard_ratio` (float, default 0.05). Guard never promotes v2->learned; it only vetoes a
learned route. `bf_peak_ratio` convention (frozen): v2 `_beamform_coarse` on collected init
obs, kx/ky range (-4,4), n_coarse=81, zero the 3x3 DC block, greedy peak extraction with
MIN_DK (0.15) box suppression; `bf_peak_ratio = power[2nd peak]/power[1st peak]`, and 0.0 if
fewer than 2 peaks are found. Diagnostics surfaced: `_auto_guard_fired`, `_auto_bf_peak_ratio`.

### FROZEN scoring (re-assembly + live validation)
Arms/data identical to `tmp/unified_2026-07-26` (batchC pure-v2, Stage-1+extra learned lanes,
atom caches; LANES est_seed/diff_seed = (42,0),(1,1),(2,2),(3,3),(4,4); median-of-5).
- Current-auto (baseline) = per-lane route learned iff `n_selected>=8` else v2 (= 0.0979).
- Guard-auto = same, but a learned lane is overridden to v2 iff `bf_peak_ratio<0.05`.
- Since every guard flip is learned->v2 and the pure-v2 lane value already exists (batchC,
  est_seeds {1,2,3,4,42}), the guard-auto score is a pure RE-ASSEMBLY from existing lanes.
- LIVE VALIDATION: run `init_method='auto'` (guard on) on the flipped episodes (5 lanes each);
  require the router to (a) fire the guard and (b) reproduce the batchC pure-v2 lane value
  bit-identically (the guard routes to v2_konly, which is the shipped path).

### FROZEN success bar (PRIMARY, all three required)
1. **Sparse-slice mean** (1+2comp, n=40, per-lane median-of-5) returns to **within +0.005 of
   pure-v2** (i.e. guard-auto sparse mean <= 0.0204 + 0.005 = **0.0254**).
2. **Crowded routing unchanged**: **0 flips** on any 15/30comp episode-lane vs current-auto.
3. **100ep aggregate** guard-auto mean **< 0.0979** (strictly improves current auto).
DECISION = PASS iff all three hold; each reported individually (honest partial if one fails).
Sensitivity (reported, not gating): `auto_guard_ratio` in {0.04, 0.05, 0.055}.

--------------------------------------------------------------------------------
## TASK 2 — sparse-parity attempt via GT-free atom merge (no retraining)

### FROZEN merge (code additive; only when `learned_merge_frac` is set)
After `pool_and_select` yields the energy-sorted selected atoms, single-linkage agglomerate in
(kx,ky) within Euclidean radius `r = learned_merge_frac * kp`, where `kp` = |k| of the
highest-energy selected atom (repo component-separation convention = adaptive kp-scaled radius,
NOT the absolute MIN_DK which chains across the spectrum — measured in `explore_signals.py`).
Replace each cluster with its ENERGY-WEIGHTED centroid (kx,ky) and summed energy; re-sort by
energy. This is the calibrated proposal set; `w,phi` then LS-fit as usual. Default
`learned_merge_frac=None` => no merge => bit-identical to shipped learned path.
FROZEN variants (max 2): **V1 = 0.25*kp**, **V2 = 0.375*kp** (x1 / x1.5).

### FROZEN measurement
- Arm: pure-learned-with-merge, SAME protocol as pure-learned (TRACK_PHASE_PREDICT=False,
  learned_init_iters=500, cached raw atoms, LANES, median-of-5), adding `learned_merge_frac`.
- SPARSE 40 (1+2comp), live tracker, both variants, vs pure-v2 sparse (0.0204) and
  pure-learned sparse (0.0534).
- NO-HARM spot check: 10 crowded episodes = BF4_15comp_seed{0..4}_uniform +
  BF4_30comp_seed{0..4}_uniform, merged-learned (each variant) vs CURRENT learned (paired
  Wilcoxon over 10 episode medians); require **n.s.** (p>=0.05) — merge must not harm crowded.

### FROZEN success tiers (report which, honestly; best variant)
- **FULL PARITY** = merged-learned sparse mean within +0.005 of v2 (<= **0.0254**).
- **PARTIAL** = halves the pure-learned->v2 gap (merged-learned sparse mean <= **0.037**).
- **FAIL** = otherwise.
Honesty rule (frozen): NO claim that learning "beats" v2 on sparse unless a paired test on the
sparse slice shows merged-learned significantly below v2. Parity is the realistic bar (v2 is
near-init-exact on sparse; headroom is small) — stated explicitly.

### Conditional (only if FULL PARITY): implied unified scoreboard
If merge reaches FULL PARITY, additionally re-assemble the 100ep `auto` with guard + merged
proposals (needs merged-learned lanes on the crowded/moderate slices too) and report the
"one method covers everything" scoreboard. If PARTIAL/FAIL, this is skipped (stated).

--------------------------------------------------------------------------------
## Oracle-analysis (no compute; from existing jsonls)
List the SPARSE episodes where CURRENT learned already beats v2 (per-episode median-of-5), and
what distinguishes them (comp count, slope/uniform, raw-init nrmse / init quality). Descriptive.

--------------------------------------------------------------------------------
## Compute plan
- Task 1: re-assembly (no compute) + ~30 live guard-validation runs (flipped eps x 5 lanes).
- Task 2: merged-learned tracker: sparse 40 x 5 x 2 variants (400) + crowded spot 10 x 5 x 2
  (100) = ~500 runs, 4 workers, OMP=1, nice-10 (~30-40 min projected). If >2h, drop V2.
- All proposals from existing caches (no GPU). Reused: batchC, Stage-1/extra arm+atom jsonls,
  explore_beamform.json (deterministic beamform, re-derivable).
