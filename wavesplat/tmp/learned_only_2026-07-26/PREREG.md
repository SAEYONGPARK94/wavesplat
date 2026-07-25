# LEARNED-ONLY SPARSE PARITY — FINAL ROUND — pre-registration (frozen before scoring)

Date 2026-07-26. Branch `learn-boundwave-retrain`. Additive code only, production defaults
untouched (bit-identity re-verified), no commits, no paper edits, no training,
`data/`/checkpoints untouched. Frozen BEFORE running any candidate-1 tracker score.

FINAL attempt in the "learning-only covers everything" line ordered by the user. Prior round
(kp-scaled atom MERGE, `tmp/unified_guard_2026-07-26` TASK 2) FAILED sparse parity
(0.0546-0.0548 vs bar 0.0254). Root cause it recorded: on the catastrophic BF2_1comp slope
episodes the proposal atoms are genuinely scattered — merging reduces the count (final_N→1) but
the energy-weighted centroid lands at the wrong k, so error does not drop. The SEPARATE routing
guard (`init_method='auto'` + bf_peak_ratio veto) already PASSED and gives a safe unified system
(auto+guard 0.0875). THIS round tries to make the PURE LEARNED pipeline competitive on sparse
WITHOUT switching to the analytic selector.

## LEGITIMACY LINE (stated, enforced in code)
The proposal net's INPUT is already the beamform power map. Using beamform-derived quantities to
CONDITION / FILTER / SELECT / REWEIGHT the proposed atoms is input-conditioning WITHIN the one
learned pipeline — not method switching. The line NOT crossed: the final component *locations*
come from the learned proposal machinery (atoms from the net; w/phi LS is shared infra). We never
substitute the v2 greedy selector's peak location as a component. Concretely: peak-anchored
filtering only KEEPS or DISCARDS net atoms by proximity to a beamform peak; when nothing survives
it returns the original net atoms unchanged (a no-op) — it never injects a beamform-located atom.

================================================================================
## DESIGN-PHASE (in-sample on the 100 finemesh eps + read-only GT; done PRE-freeze)
See `design_diag.py` → `design_diag_q1.json`, `design_diag_q2.json`. In-sample, stated plainly.

Ground truth from `config/wave_components` [A, ω, kx, ky, φ, Q]; the dominant true component's
|k| = kp is the anchor scale. Distances in kp-units, median over the 5 diff_seed lanes.

**Q1 — is there a net atom near truth to anchor onto? (candidate-1 feasibility)**
- Catastrophic BF2_1comp_slope (n=5): nearest selected atom to the TRUE swell = **0.742 kp**
  [0.563, 0.800]; nearest to the beamform DOMINANT peak = 0.743 kp; beamform peak error to truth
  = **0.058 kp** (the beamform SEES truth). ⟹ the diffusion net proposes **no atom near truth**;
  the atoms are a scatter ~0.74 kp away. (BF2_1comp_seed1_uniform is the same failure, 0.748 kp.)
- All other sparse eps: nearest atom to truth = **0.002 kp** (proposal spot-on).
- CONSEQUENCE (recorded before scoring): peak-anchored filtering can COLLAPSE scatter onto a good
  atom only where a good atom exists. On the catastrophic eps none is within any kp-scaled radius,
  so the filter is a NO-OP there (by design, to guarantee no-harm) — it cannot reach truth without
  substituting the beamform-located peak, which the legitimacy line forbids. Candidate 1 is
  therefore expected to leave the catastrophic eps ≈ pure-learned; it is run to MEASURE that and
  to verify no-harm on crowded, not because the diagnostic predicts parity.

**Q2 — does held-out residual pick a good sample? (candidate-2 feasibility)**
Per-sample (32 atoms) LS w/phi fit on 2/3 of the init obs, residual on the held-out 1/3, on the
25 catastrophic lanes: held-out residual is FLAT (~0.069) across the 4 samples EXCEPT the single
lane (BF2_1comp_seed2_slope ds4) where one sample is genuinely good (d_true 0.00) — there the
residual drops to 0.007 and is correctly picked. Lanes with ANY good sample = **1/25**; lanes
where the min-residual sample is the good one = **1/25**. ⟹ residual DOES separate gross garbage
(0.069 vs 0.007) but a good sample essentially never exists to select. **CANDIDATE 2 = DEAD at
design time** (consistent with the known obs-residual caveat): selection cannot recover a swell
that no sample proposed. Proceed with candidate 1 only. NOT run.

================================================================================
## CANDIDATE 1 — PEAK-ANCHORED FILTERING (code additive; only when opt-in flag set)

### FROZEN mechanism (`learned_init.peak_anchor_filter`, called after `pool_and_select`)
Given the pool_and_select'd atoms (kx,ky,energy) and the init-window observations:
1. Compute significant beamform peaks with the v2 convention (`_beamform_coarse` on collected
   init obs, kx/ky (-4,4), n_coarse=81, zero the 3×3 DC block, greedy MIN_DK=0.15 box-suppressed
   peak extraction). Significant = power ≥ `sig_frac`·P1 (P1 = dominant peak power).
2. kp = |k| of the DOMINANT beamform peak (anchor scale = repo kp-scaled separation convention).
3. Keep each atom iff its Euclidean (kx,ky) distance to the NEAREST significant beamform peak
   ≤ `anchor_frac`·kp. Re-sort survivors by energy.
4. If ≥1 atom survives → survivors are the calibrated set. If 0 survive → return the ORIGINAL
   pool_and_select'd atoms unchanged (NO-OP; never inject a beamform-located atom → no-harm).
GT-free, atoms-from-net only. `w,phi` LS-fit as usual afterwards. Default flag off → the shipped
learned path is bit-identical.

Additive config keys (opt-in only; `auto`/`v2_konly`/default learned untouched):
`learned_peak_anchor` (bool, default False), `learned_anchor_frac` (float), `learned_anchor_sigfrac`
(float, default 0.1 = the npk_10 convention already shown to separate sparse vs crowded).

### FROZEN variants (max 2) — anchor radius, kp-scaled
- **V1 = anchor_frac 0.375** (matches the merge V2 radius; tight enough to collapse scatter).
- **V2 = anchor_frac 0.60** (looser; maximizes no-harm on crowded where true peaks are dense).
`sig_frac` fixed at 0.1 for both. No other knobs varied. No hyperparameter fishing beyond these.

### FROZEN measurement (identical protocol to pure-learned)
`TRACK_PHASE_PREDICT=False`, `learned_init_iters=500`, cached raw atoms, LANES
(est/diff)=(42,0),(1,1),(2,2),(3,3),(4,4), median-of-5, live tracker runs.
- SPARSE 40 (1+2comp), both variants, vs pure-v2 sparse (0.0204) and pure-learned sparse (0.0534).
- NO-HARM spot check: 10 crowded = BF4_15comp_seed{0..4}_uniform + BF4_30comp_seed{0..4}_uniform,
  anchored-learned (each variant) vs CURRENT learned, paired Wilcoxon over 10 episode medians;
  require **n.s. (p≥0.05) OR medianΔ≤0** (filter must not harm crowded).

### FROZEN success tiers (best variant; reported honestly)
- **FULL PARITY** = anchored-learned sparse mean within +0.005 of v2 (≤ **0.0254**).
- **PARTIAL** = halves the pure-learned→v2 gap (sparse mean ≤ **0.037**).
- **FAIL** = otherwise.
Honesty (frozen): no claim learning "beats" v2 on sparse absent a paired test showing anchored
sparse significantly below v2. If FAIL, the final answer for tonight is: learning-only sparse
parity is NOT achievable without retraining/architecture change (per two pre-registered attempts:
merge + peak-anchor), with the future-requirements list.

### Conditional (only if FULL PARITY): implied full-100ep learned-only scoreboard vs auto+guard
0.0875 (needs anchored lanes on crowded/moderate too). If PARTIAL/FAIL, skipped (stated).

================================================================================
## Compute plan
Candidate 1 only (candidate 2 dead). Sparse 40 × 5 lanes × 2 variants (400) + crowded spot
10 × 5 × 2 (100) = 500 tracker runs from existing atom caches (no GPU), 4 workers, OMP=1,
nice-10 (~27 min by the Task-2 measured rate). Hard cap 1.5 h. Bit-identity re-verified first.
