# Pre-registration — WavePrior STAGE-1 integration: learned-init vs v2_konly-init under the phase-fixed tracker

Written and **frozen before any full-scale run** (2026-07-25). Companion result file:
`tmp/wp_integration_2026-07-25/WP_INTEGRATION_2026-07-25.md`. Freezes the endpoint, episode set,
seed protocol, decision rule and success bar so the read-out cannot be chosen after the numbers.

## 0. Question (confirmatory)

The head-to-head (`tmp/wp_crowded_2026-07-25/`) established, PASS, that the learned set-proposal
(cond-diffusion + 80-iter polish) beats the analytic WaveSplat `ws_best=min(init,tracked)` on the
40 crowded finemesh episodes by median −0.105 nRMSE/Hs (−45%, 39/40, p=1.8e-12), and loses the
sparse control 0/12 (clean regime crossover). That was a **nowcast** measurement of the raw
proposal, NOT run through the production tracker.

STAGE-1 asks the engineering question the head-to-head deferred: **when the learned proposal is
wired as `init_method='learned'` and the shipped joint-fit / adaptive-N / phase-fixed sequential
tracker run on top, is the learned advantage preserved in the tracked product?**

## 1. Implementation under test (frozen)

- `init_method='learned'` (additive; default `v2_konly` verified bit-identical BEFORE eval —
  `default_bit_identical.py`, 3/3 avg_nRMSE + init_nrmse match to 1e-10 vs
  `tmp/wp_crowded_2026-07-25/ws_runs.jsonl`).
- The learned proposal runs on the **init window** (`get_init_window(tp_factor=2.0)` — the SAME
  observations the v2 path uses), obs times shifted to min=0 for the beamform map / polish (net's
  training convention; kx/ky/energy are time-origin invariant under the phase-marginalized polish
  likelihood). nsamp=4 ancestral samples, each polished 80 iters, atoms pooled across samples,
  thresholded (energy>0.03·max), de-duplicated (min-sep 0.03 rad/m ≪ analytic MIN_DK=0.15),
  capped at max_n=30.
- Component set installs (kx, ky) and, in the **primary** variant, LS-fit (w, phi) via
  `tnse.fit_amplitudes_phases` on the init obs at ABSOLUTE frame times (convention verified
  identical to `SpectralGaussianModel.forward`). This carries the head-to-head's information
  faithfully. A secondary `learned_konly` variant (w=0.01/random-phi, exactly like v2_konly) is
  reported to isolate the kx/ky contribution from the LS w/phi contribution.
- On top, UNCHANGED: joint fit (same Adam optimizer / loss, budget = 500 iters, the shipped
  final-stage polish budget), `_adaptive_prune_grow` (energy prune + residual-beamform grow),
  `_build_tracking_optimizer` (tracking_mode `w_phi` → kx/ky frozen, w/phi tracked).
- **Both arms** run with `sequential_update.TRACK_PHASE_PREDICT=False` set per worker at runtime
  (fair: same fixed tracker; shipped module default stays True, unedited).

## 2. Primary endpoint (frozen)

- **Comparison unit:** per-episode paired **full-sequence tracked `avg_nRMSE` (/Hs)**.
- **Arms:** `learned` (primary variant, LS w/phi) vs `v2_konly`, identical config otherwise
  (`adaptive_n=True`, device cpu, phase-fix on).
- **Episode set (PRIMARY):** the exact 40 crowded finemesh episodes of the head-to-head —
  `episode_BF{4,5}_{15,30}comp_seed{0..4}_{uniform,slope}.h5` in `data_finemesh_2026-07-23/`.
- **Seed protocol:** 5 paired lanes `(estimator_seed, diffusion_seed)` =
  (42,0),(1,1),(2,2),(3,3),(4,4); **median-of-5** per episode per arm, reported with sd. The
  learned atoms depend on `diffusion_seed`; the WaveSplat pipeline on `estimator_seed`. Single-seed
  per-episode deltas FORBIDDEN.
- **Test:** one-sided Wilcoxon signed-rank on the 40 paired deltas `Δ = learned − v2_konly`,
  H1: learned < v2_konly (Δ<0), **α=0.05**. Effect size = matched-pairs rank-biserial; report
  median paired Δ, win count, p.

### Decision rule (FROZEN)
Primary **PASS** iff BOTH:
- **P1:** one-sided Wilcoxon p < 0.05, AND
- **P2:** median paired Δ < 0.

### Success bar (FROZEN, separate from PASS/FAIL)
The head-to-head init-level advantage was **median paired Δ = −0.1052** (cond vs ws_best). STAGE-1
**retains the advantage** iff the integrated tracked median paired Δ (learned − v2_konly) is
≤ **−0.063** (i.e. ≥ 60% of 0.1052 preserved through joint-fit + tracker). PASS + retention ≥60%
= full STAGE-1 success; PASS + retention <60% = "significant but attenuated, mechanism to report".

## 3. Secondary endpoints (reported, NOT decision-primary)

- **Full 5-metric panel:** nRMSE, Variance_Ratio, Pearson_rho, Skill_Score, Directional_R2
  (sequence-mean per episode, median-of-5), paired learned vs v2_konly on the 40 crowded.
- **By-comp / by-BF / by-arm splits** (15 vs 30comp; BF4 vs BF5; uniform vs slope).
- **Sparse control (12, expected v2_konly-wins or parity):** the head-to-head's fixed 12
  1/2/5comp episodes, same protocol. A learned **loss here is acceptable** and anticipated — the
  product is a regime-switch policy; the niche is crowded seas.
- **Init decomposition:** raw learned-init field nRMSE (LS w/phi, NO joint fit, @ last init
  frame) vs post-pipeline `init_nrmse` vs tracked `avg_nRMSE`, to locate where (if anywhere) the
  learned advantage is gained or lost inside the pipeline.
- **learned_konly ablation** (kx/ky only, v2_konly-style w/phi): isolates kx/ky vs LS-w/phi.

## 4. Deliverable — naive regime-switch arm (measured, small)

Pick per-episode init by a crowding signal available at init time WITHOUT ground truth:
**signal = `n_selected`, the number of components the learned proposal returns after
pool_and_select** (the net's own detected component count). **Rule (FROZEN): n_selected ≥ 8 →
learned, else v2_konly.** Threshold 8 is set from the experimental DESIGN (sparse=1/2/5comp,
crowded=15/30comp), midway above the sparse true-count ceiling (5) and below the crowded floor
(15) — NOT tuned on any nRMSE outcome. Report the switch policy's **52-episode aggregate**
(40 crowded + 12 sparse) tracked avg_nRMSE vs pure-learned and pure-v2_konly, plus an
**oracle switch** (per-episode best arm; NOT deployable) as an upper bound, and the
sparse/crowded separation the signal actually achieves.

## 5. Execution / compute (frozen)

- **GPU (cond proposals):** `precompute_atoms.py` computes raw pooled atoms per
  (episode, diffusion_seed∈{0,1,2,3,4}) for all 52 episodes → `atoms_cache.jsonl`. Read-only ckpt
  `results/direction3/beamform_cond_diffusion.pt`. Never touches `data/`.
- **CPU (tracker):** `run_arms.py`, 4 workers, OMP/OPENBLAS/MKL=1, `nice -n 10`, both arms,
  full panel, `TRACK_PHASE_PREDICT=False` per worker. Data `data_finemesh_2026-07-23/`.
- One other agent shares the box (CPU real-data work). If projected wall-clock > 3h: drop the
  regime-switch arm first, then the sparse control, NEVER the crowded primary.
- No commits, no paper edits, repo tmp only, checkpoints/data untouched.

## 6. Frozen episode lists
Identical to the head-to-head (`tmp/wp_crowded_2026-07-25/PREREG.md §6`): 40 crowded primary +
12 sparse control. The DEGRADED BF4_20comp extra is excluded (not evaluable).

## 7. What a PASS licenses / does not license
Same scope caveats as the head-to-head §7: a regime-specific **in-distribution synthetic**
integration result; does NOT license real-data claims, does NOT reopen the refuted γ/bank-MMSE
axis, does NOT authorize any default flip or paper edit. STAGE-1 measures whether the tracker
preserves the learned init; Stage-2 e2e fine-tuning is inspection-only in this report.
