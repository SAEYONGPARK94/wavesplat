# Pre-registration — WavePrior (learned set-proposal) vs phase-fixed WaveSplat on the CROWDED regime

Written and **frozen before any full-scale run** (2026-07-25). Companion result file:
`tmp/wp_crowded_2026-07-25/WP_CROWDED_HEADTOHEAD_2026-07-25.md`. This freezes the endpoint,
the episode set, the stochasticity protocol, and the decision rule so the read-out cannot be
chosen after seeing the numbers.

## 0. Question and prior (confirmatory framing)

Today's tracker phase-fix (`TRACK_PHASE_PREDICT=False`, `tmp/TRACKER_AMPDECAY_2026-07-25.md`,
100-ep power-gate `tmp/rerun_2026-07-25/PHASEFIX_100EP_2026-07-25.md`) makes the analytic
WaveSplat pipeline near-init-exact on sparse seas (1–5comp nRMSE/Hs 0.016–0.027 at 100ep) but
leaves a surviving wall on the crowded regime (15/30comp ~0.24–0.27). The G1 A1 **pilot**
(`tmp/G1_waveprior_power/a1_ws_finemesh.json`, n=7 finemesh episodes, same /Hs normalizer)
showed the learned set-proposal (beamform-conditioned set-diffusion + 80-iter polish,
`scripts/d3_beamform_cond_diffusion.py`) reaching **0.086–0.100 on 15comp and 0.160–0.233 on
30comp** where the analytic init/tracked sits at **0.23–0.32** — a **~2–3× improvement exactly
where the analytic path hits its wall**.

This test is **confirmatory**: it takes that pilot effect (2–3×, n=7, mixed seeds) and asks
whether it survives at scale on the full pre-enumerated crowded set with a proper paired,
median-of-k, one-sided hypothesis test. The pilot magnitude is the expected effect; we are
not fishing for a new one.

## 1. Normalizer parity (verified before freeze)

Both arms report **nRMSE = RMSE / Hs**:
- WS: `wavesplat/run.py:150` `init_nrmse = nrmse(pred_init, gt_field, Hs_true)`; per-frame
  `metrics.py:312` `metrics['nRMSE'] = nrmse(pred, gt, Hs_true)`; `nrmse = RMSE/max(Hs,1e-6)`
  (`metrics.py:32`).
- cond: `cond_polish_nrmse_Hs = sqrt(mean((bma−gt)^2)) / max(Hs,1e-6)`
  (`d3_beamform_cond_diffusion.py:203`, and the pilot runner `run_a1_cond_finemesh.py:65`).
- Both score against the paper's `gt_field` on the paper's GT grid, obs gathered by the paper's
  `gather_frames`, at the **same 3 nowcast frames** `sel = linspace(nb+3, len-2, 3)`.
Parity confirmed. Nowcast frames are identical between arms by construction (cond's `sel`); the
WS nowcast reconstruction is scored at its init frame (`init_nrmse`) and the WS tracked product
is the sequence mean (`avg_nRMSE`).

## 2. Primary endpoint (frozen)

- **Comparison unit:** per-episode paired nRMSE/Hs.
- **cond arm value:** median-of-5 (see §4) `cond_polish_nrmse_Hs` — best-of-N=4-sample **average**
  (BMA) + polish, nowcast at the 3 frames, mean over frames.
- **WS arm value:** median-of-5 estimator seeds of BOTH `init_nrmse` and `avg_nRMSE`, reported
  separately. The **primary analytic reference is `ws_best = min(median init, median tracked)`**
  per episode — i.e. the analytic path is credited with whichever of init or tracked is lower
  (maximally fair to the analytic side).
- **Episode set (PRIMARY):** ALL crowded finemesh episodes in `data_finemesh_2026-07-23/` —
  every 15comp and 30comp, BF4 and BF5, uniform AND slope, seeds 0–4 = **40 episodes**
  (enumerated in §6). All 40 verified to open cleanly (h5py, 40/40; none of the known-corrupt
  slope files are among them).
- **Test:** one-sided Wilcoxon signed-rank on the 40 paired deltas
  `Δ = cond_median − ws_best`, H1: cond < ws_best (Δ<0, cond better), **α = 0.05**.
- **Effect size:** matched-pairs rank-biserial correlation `r = 1 − 2W_−/(W_+ + W_−)` reported
  alongside the median paired Δ and the win-count (# episodes with Δ<0).

### Decision rule (FROZEN)
Primary **PASS** iff BOTH:
- **P1:** one-sided Wilcoxon p < 0.05 (cond significantly better than `ws_best`), AND
- **P2:** median paired Δ < 0 (cond better on the typical crowded episode).

Otherwise **FAIL**. Reported regardless of verdict: full 40-row paired table, median Δ, win
count, effect size, p-value.

A secondary comparison **cond vs median WS tracked** (the paper's actual shipped product, not the
`min`) is reported for context but is NOT the decision variable.

## 3. Secondary endpoints (reported, NOT decision-primary)

- **By-comp split:** 15comp (n=20) vs 30comp (n=20), each with its own paired median Δ + win rate.
- **By-BF split:** BF4 (n=20) vs BF5 (n=20).
- **By-arm split:** uniform (n=20) vs slope (n=20).
- **Sparse control (expected WS-wins):** a fixed 12-episode subsample of 1/2/5comp finemesh
  episodes (enumerated §6), same paired median-of-5 protocol. The niche claim is regime-specific,
  so a cond **loss** here is *consistent with* the hypothesis (analytic path resolves sparse peaks;
  learned proposal is expected to be worse). This is a control, not a failure of the primary.
- **BF4_20comp extra (FLAGGED DEGRADED):** `data_finemesh_extras_2026-07-24/` was flagged
  DEGRADED. `episode_BF4_20comp_seed7_uniform.h5` is included as a single flagged secondary data
  point (n=1 episode, median-of-5), clearly marked, NOT in the primary set and NOT in any test.

## 4. Stochasticity protocol (frozen)

- **WS arm — STOCHASTIC.** Source: two random-phase init points (`ESTIMATOR_VARIANCE` report);
  bimodal run-to-run spread on calm episodes. Protocol: **median-of-5 estimator seeds
  (42, 1, 2, 3, 4)**, per-episode median reported with sd. Single-seed episode deltas FORBIDDEN.
- **cond arm — STOCHASTIC (ancestral diffusion sampling).** `sample()` draws ancestral noise and
  is only *pinned* by `torch.manual_seed(seed)` (`d3_beamform_cond_diffusion.py:158`); different
  seeds → different posterior atom sets. Verified: seed=0 reproduces the pilot bit-for-bit
  (BF4_15c 0.0858, BF4_30c 0.2332). Because it is genuinely stochastic, the arm is run with
  **median-of-5 sampling seeds (0, 1, 2, 3, 4)**, each seed = a 4-sample BMA + polish; per-episode
  median reported with sd. This mirrors the WS median-of-5. Single-seed cond deltas FORBIDDEN.
- The nsamp=4 samples within one seed are averaged (BMA), NOT best-of-4 (no oracle selection):
  `bma = mean over the 4 polished fields` (`run_a1_cond_finemesh.py:63`). No GT is used to pick
  among samples in either arm.

## 5. Execution (frozen)

- **WS:** `run_estimation(FINE+ep, config={'adaptive_n':True,'seed':s,'device':'cpu'})` with
  `import wavesplat.optimizer.sequential_update as su; su.TRACK_PHASE_PREDICT = False` set in the
  worker process at runtime (the shipped default stays True; **no default edited**). Record
  `init_nrmse` and `avg_nRMSE`. 4 workers max, `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL=1`,
  `nice -n 10`. Data path `data_finemesh_2026-07-23/` (never `data/`).
- **cond:** GPU. Reuse the pilot's `eval_episode` (`run_a1_cond_finemesh.py`) verbatim except the
  sample seed is looped 0–4; same frames, same best-of-N=4 BMA + polish (80 iters), `/Hs`, same
  `data_finemesh_2026-07-23/` path (the checkpoint `results/direction3/beamform_cond_diffusion.pt`
  is read-only, never modified).
- If wall-clock projects beyond ~2.5h, cut the sparse control to 6 episodes and note it; NEVER cut
  the crowded primary set. (Projection at freeze: ~30–35 min total, no cut expected.)

## 6. Frozen episode lists

### PRIMARY — 40 crowded (`data_finemesh_2026-07-23/`)
`episode_BF{4,5}_{15,30}comp_seed{0,1,2,3,4}_{uniform,slope}.h5` — the full 2×2×5×2 = 40 grid.

### Sparse control — 12 (fixed subsample, `data_finemesh_2026-07-23/`)
1. BF2_1comp_seed0_uniform  2. BF2_1comp_seed0_slope  3. BF3_1comp_seed0_uniform
4. BF3_2comp_seed0_uniform  5. BF3_2comp_seed0_slope  6. BF3_5comp_seed0_uniform
7. BF3_5comp_seed1_uniform  8. BF3_5comp_seed0_slope  9. BF4_1comp_seed0_uniform
10. BF4_5comp_seed0_uniform 11. BF4_5comp_seed0_slope 12. BF4_5comp_seed1_uniform

### Flagged extra — 1 (`data_finemesh_extras_2026-07-24/`, DEGRADED set)
`episode_BF4_20comp_seed7_uniform.h5`

## 7. What a PASS licenses / does not license (frozen scope)

- **Licenses:** a *regime-specific, in-distribution synthetic* claim — the learned set-proposal
  beats the analytic WaveSplat (init or tracked, whichever is better) on crowded synthetic seas.
  Crowded synthetic being in-distribution for the cond net (trained on direction3 32-comp JONSWAP)
  IS the niche claim, not a confound.
- **Does NOT license real-data claims.** Real-data transfer is a separate future test; the CAVEAT
  in `d3_beamform_cond_diffusion.py:12` (synthetic-clean nets fail to transfer) stands.
- **Does NOT resurrect the refuted γ-MMSE-prior superiority claims.** hand≈v3≈retrain on the
  gamma/bank-MMSE axis stands (`project_waveprior_promotion_2026-07-24`). This is a different
  object (discrete set-proposal that CAN represent sparse/crowded peaks), a different endpoint
  (crowded-regime nowcast), and a different comparison (vs analytic WaveSplat, not vs hand-prior).
- **Integration (`init_method='learned'` → phase-fixed tracker) is next-step engineering**, not a
  measured result of this test.
