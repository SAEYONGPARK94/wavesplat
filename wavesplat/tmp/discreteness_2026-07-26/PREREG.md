# Pre-registration — SPECTRAL DISCRETENESS as the boundary coordinate (worklist W9)

**Written and frozen 2026-07-26, BEFORE any episode was generated and before any arm was run.**
Companion result file (to be written after the run): `tmp/discreteness_2026-07-26/RESULT.md`.
Raw logs: `tmp/discreteness_2026-07-26/*.jsonl`. Scripts: `gen_cells.py`, `run_ws.py`, `run_cond.py`,
`run_secondary.py`, `analyze.py` — all in this directory. **Nothing in `wavesim/` or `wavesplat/`
is modified**; generation is a standalone script that monkey-patches the component *source*
inside its own process only.

---

## 0. Hypothesis under test (P2, `tmp/persona_debate_2026-07-26/R2_P2_ocean_prof.md` §CRUX F)

The learned set-proposal's crowded-sea advantage (PASS, −45%, 39/40, p=1.8e-12;
`tmp/wp_crowded_2026-07-25/`) exists because synthetic "crowded" is **crowded-AND-DISCRETE**
(15–30 countable cosine atoms), whereas real seas are **crowded-AND-CONTINUOUS**. A network
trained on discrete atom sets, shown a continuum, tiles it with ~24–30 atoms — exactly the
measured real-window n_eff ≈ 24. If true, ONE variable (discreteness) explains the synthetic win,
the real-data accuracy loss (p=0.005, `project_wp_real_2026-07-26`), and the proposal-generation
OOD; and the real-data loss becomes a **prediction** of the regime map instead of an anomaly.

**Directional prediction:** the learned advantage should COLLAPSE as the synthetic spectrum is
made continuum-like at fixed crowding (fixed BF, fixed parent component count, fixed total
variance, fixed total directional spread).

---

## 1. Factor and cells (FROZEN)

Held fixed across all cells: sea state **BF4** (Hs = 1.2 m, Tp = 7.0 s), **15 parent components**,
depth 50 m, duration 30.0 s, dt 0.1 s, n_pts 50, raycast mesh **200 × 167** (dx = 1.2060 m,
dy = 1.2048 m — the `data_finemesh_2026-07-23` mesh), GPU raycast, GT grid 80 × 60 over
x ∈ [10, 250] m, y ∈ [−100, 100] m. Total variance (Hs) and the **total** directional spread of
the sea (30°) are held **exactly** fixed by construction (§2).

The factor is **per-component spreading** — how the sea's 30° of directional spread and its
frequency-band energy are *distributed*: concentrated on 15 discrete atoms, or smeared into a
continuum. Three cells, 10 fresh episodes each (5 seeds × {uniform, slope} raydrop):

| cell | name | M sub-components per parent | σ_sub (per-comp direction) | parent σ | ω jitter | total atoms |
|---|---|---|---|---|---|---|
| **A** | discrete control | 1 | 0° | 30.0000° | none (bin centre) | 15 |
| **B** | mild spreading | 31 | 10° | 28.2843° | none (bin centre) | 465 |
| **C** | strong spreading (continuum-like) | 31 | 28° | 10.7703° | uniform within parent bin | 465 |

Parent σ is set to `sqrt(30² − σ_sub²)` so the **total** directional variance is 30° in every
cell (directions are Gaussian in `jonswap_components`, so the variance decomposition is exact).
This is the control that makes the factor *discreteness* and not *broadness*.

**Seeds: 10, 11, 12, 13, 14** — verified unused by any prior benchmark (`data/`,
`data_finemesh_2026-07-23/`, `data_finemesh_extras_2026-07-24/` contain only seeds 0–4, 7, 42).

Cell C is deliberately an **ordered ladder endpoint** (direction spread AND frequency continuum),
not an orthogonal 2-factor decomposition. Cell B is the intermediate rung and is **reported for
monotonicity only — it is not a decision variable**.

---

## 2. Generation spec (FROZEN)

Standalone script `gen_cells.py`. It calls `wavesim.pipeline.run_episode` **verbatim** (same
raycast, raydrop, noise, confidence, mesh-adequacy, HDF5 writer as
`tmp/regen_run_finemesh.py`), with `wavesim.pipeline.jonswap_components` replaced *in-process*
by a wrapper that:

1. calls the real `jonswap_components(Hs=1.2, Tp=7.0, n_components=15, depth=50,
   direction_deg=0, direction_spread_deg=σ_parent, seed=seed)` — the library function, unmodified;
2. **if M == 1 and σ_sub == 0: returns that list unchanged** (cell A ⇒ bit-identical to the
   shipped generator);
3. otherwise replaces each parent component `c` by M sub-components:
   - amplitude `A_sub = c.A / sqrt(M)` (⇒ Σ A_sub² = c.A² exactly, energy/Hs preserved);
   - direction `θ_sub ~ N(θ_parent, σ_sub)` (i.i.d.);
   - frequency: cell A/B `ω_sub = ω_parent`; cell C `ω_sub ~ U(ω_parent − dω/2, ω_parent + dω/2)`
     with `dω = 2.5·ω_p/15` = the parent JONSWAP bin width, so the band [0.5 ω_p, 3 ω_p] and the
     integrated spectrum are unchanged (this is the standard random-frequency realization of a
     *continuous* spectrum; the discrete cell is its bin-centre degenerate limit);
   - `k_sub = wavenumber_from_omega(ω_sub, depth)` — the library's full dispersion, so every
     sub-atom is exactly on the dispersion shell;
   - phase `φ_sub ~ U(0, 2π)` i.i.d.;
   - `Q` recomputed by the library's own rule on the expanded set
     (`Q = min(0.8/(ωA), 1)·0.5`, then global rescale to ΣQωA = 0.95 if violated). **Q does not
     enter the rendered field** (`gt_generator.evaluate_field` returns only Z and the normals from
     `single_gerstner`, neither of which uses Q), so this is bookkeeping only and introduces no
     nonlinearity confound: all three cells are pure linear cosine superpositions.
4. expansion RNG is `default_rng(900000 + 1000·cell_index + seed)`, independent of the episode's
   sensor/raydrop RNG (`default_rng(seed)` inside `run_episode`).

Output: `tmp/discreteness_2026-07-26/data/cell_{a,b,c}/episode_BF4_15comp_seed{10..14}_{uniform,slope}.h5`.

### Generation gates (FROZEN, checked before any arm is run)
- **G-mesh:** every generated episode must carry `config.attrs['mesh_verdict'] == 'ADEQUATE'`
  (`wavesim/mesh_adequacy.py`, run on the **expanded** component set inside `run_episode`).
  Any non-ADEQUATE episode is a **validity flag** for its cell and is reported, not silently dropped.
- **G-identity:** cell A's stored `wave_components` must be **bit-identical** to
  `jonswap_components(Hs=1.2, Tp=7.0, n_components=15, depth=50, seed=s)` for every seed.
- **G-roundtrip:** one episode per cell is opened with `wavesplat.data_loader.EpisodeDataLoader`;
  `Hs_true == 1.2`, `Tp_true == 7.0`, frame count and grid shape match `data_finemesh`, and the
  realized field `4·std(gt_wave_field)` is within ±10 % of 1.2 m in all three cells.
- **G-spread:** realized circular directional spread (energy-weighted) reported per episode;
  expected ≈ 30° in all three cells. Reported, not gated.

### Addendum (added 2026-07-26 after the generator's dry-run gate check, **before any episode was
generated and before any arm was run**; adds a caveat only — no threshold, cell, metric, arm,
seed or parameter is changed)

The dry run confirms G-identity (cell A bit-identical to the library for all 5 seeds) and exact
Hs preservation (1.2000 in all cells). It also shows a property worth stating up front: the
**realized** energy-weighted circular spread is 17–26° in cell A but 27–31° in cells B/C, even
though the *nominal* total spread is 30° everywhere. Reason: JONSWAP puts ~62 % of the variance on
2 of the 15 parents, so a 15-atom realization's realized spread is dominated by 2–3 draws and is
both low and noisy; smearing each parent over 31 sub-atoms drives the realization toward its
nominal value. This is **intrinsic to discreteness**, not an independent knob — a 15-atom sea
cannot have a continuum's realized fine-scale spread. It is therefore accepted as part of the
factor, and reported per episode (G-spread) rather than compensated. Consequence for
interpretation, stated in advance: a CONFIRM verdict identifies "discrete atom placement", of
which realized-spread concentration is one component; it does not decompose the two.

---

## 3. Arms (FROZEN — verbatim from `tmp/wp_crowded_2026-07-25/PREREG.md`)

- **learned arm ("cond"):** beamform-conditioned set-diffusion proposal
  `results/direction3/beamform_cond_diffusion.pt` (read-only) + 80-iter Gauss–Newton polish,
  n = 4 samples averaged (BMA, **no** GT-based selection), scored at the same 3 nowcast frames
  `sel = linspace(nb+3, len(L)-2, 3)`, `nb = K_TP·Tp/0.1`. **Median of 5 diffusion sampling seeds
  (0, 1, 2, 3, 4).** Code reused from `run_cond_crowded.py::eval_episode` unchanged.
- **analytic arm ("ws"):** `wavesplat.run.run_estimation(..., config={'adaptive_n': True,
  'seed': s, 'device': 'cpu'})` with `sequential_update.TRACK_PHASE_PREDICT = False` set in the
  worker process (shipped default untouched); `init_method` left at the shipped default
  `v2_konly` (dispersion beamforming + coarse-to-fine). Records `init_nrmse` and `avg_nRMSE`.
  **Median of 5 estimator seeds (42, 1, 2, 3, 4).**
- **Scoring, both arms:** nRMSE = RMSE / Hs against the paper's `gt_field` on the paper's GT grid.
  Normalizer parity verified in `wp_crowded/PREREG.md` §1 and unchanged here.
- Per-episode analytic reference **`ws_best = min(median init, median tracked)`** — maximally fair
  to the analytic side, identical to `wp_crowded/analyze.py`.
- **Median-of-k is mandatory. Single-seed episode deltas are FORBIDDEN.**

---

## 4. Primary endpoint and decision cells (FROZEN)

Per episode *i*: `Δ_i = cond_i − ws_best_i` (absolute) and `r_i = Δ_i / ws_best_i` (relative).

- **Cell advantage `R` = median over the cell's 10 episodes of `r_i`.** (This is the "medΔ %"
  referred to in the decision cells. For reference, the same statistic on the
  `wp_crowded` BF4/15comp block is **R = −0.646**; the 40-episode crowded set gives −0.464.)
- **Significance:** one-sided Wilcoxon signed-rank on the 10 paired `Δ_i`, H1: cond < ws_best,
  **α = 0.05**, `zero_method='wilcox'`. (n = 10, one-sided ⇒ minimum attainable p = 0.00098.)
- Mean paired Δ, win count, and rank-biserial effect size reported alongside for every cell.

### Frozen verdict cells — evaluated in this order

0. **INVALID** — if the control cell A fails to replicate (`R_A > −0.20`), the whole experiment is
   INVALID: generation changed too much for the cells to mean anything. **Not evidence in either
   direction.** No other verdict may be issued in this case.
1. **CONFIRM (discreteness is the boundary coordinate)** — `R_A ≤ −0.20` **AND**
   ( `R_C > −0.10` **OR** `p_C ≥ 0.05` ).
2. **REFUTE** — `R_C ≤ −0.20` **AND** `p_C < 0.05`.
3. **AMBIGUOUS** — anything else (e.g. `R_C` between −0.20 and −0.10 with p < 0.05).
   Reported as AMBIGUOUS, **no narrative rescue**.

Cell B is reported (R_B, p_B) as a monotonicity read-out and **cannot change the verdict**.

**No threshold, parameter, arm, seed, M, σ, or metric definition may be changed after any
number is seen.** If the run reveals a defect, the defect is reported and the experiment is
declared INVALID — it is not re-tuned.

---

## 5. Secondary endpoints (reported, NOT decision-primary)

- **`n_selected`** — cardinality of the learned proposal after the shipped pooling:
  `learned_init.pool_and_select(*propose_atoms(get_init_window(2.0), depth, nsamp=4,
  diffusion_seed=d), energy_thresh=0.03, dedup_dk=0.03, max_n=30)`; the same quantity the
  WavePrior UNIFIED router thresholds at 8. Computed per (episode, diffusion seed 0–4);
  per-episode **median-of-5**. Prediction under the hypothesis: n_selected rises toward the
  max_n=30 cap on spread seas (over-proposal / continuum tiling).
- **`bf_peak_ratio`** — `wavesplat.optimizer.initializer.beamform_peak_ratio(L.get_init_window(2.0),
  L.depth)`, the frozen GT-free guard signal (deterministic, one per episode). Per-cell
  distribution compared against the real-window reference
  `tmp/figs_opus_real_2026-07-26/bf_ratio_real.json` (real windows span **0.27–1.00** on the
  guard-relevant range; full real set includes lower outliers). Prediction under the hypothesis:
  spread cells migrate toward the real-window range.
- Realized directional spread, realized Hs, mesh-adequacy attributes, per-episode timings.

These are descriptive. They cannot rescue or overturn the primary verdict.

---

## 6. Scope of what each verdict licenses (FROZEN)

- **CONFIRM licenses:** "on synthetic BF4 seas at fixed crowding and fixed total spread, the
  learned set-proposal's advantage over the analytic init is a function of spectral
  *discreteness*, and disappears on continuum-like spectra" — a *synthetic, init-level, BF4,
  15-parent-component* statement. It makes the measured real-data accuracy loss **consistent with**
  the regime map. It does **NOT** by itself prove the real-data loss is *caused* by continuity
  (real data differ in sensor, noise, aperture, current, and nonlinearity too).
- **REFUTE licenses:** "discreteness is not the boundary coordinate" — P1's objection stands and
  the regime map still mis-predicts real data.
- Neither verdict licenses any new real-data accuracy claim. The
  `docs/HANDOFF_OPUS_2026-07-26.md` §5 do-not-cite rules stand unchanged: no claim of learned
  accuracy advantage on real data (collapse-insurance only), no "more complex ⇒ WavePrior wins"
  outside synthetic, no mean-based real-data victory.
- Nothing here is committed to git; nothing here changes a shipped default.

---

## 7. Compute budget (declared at freeze)

30 episodes generated (~1.5–2 h GPU raycast, 3-way parallel), analytic arm 150 runs
(~25 min, 4 CPU workers), learned arm 150 runs (~20 min GPU), secondary ~15 min.
If the analytic arm over-runs, it is **not** cut — the cond arm's extra diffusion seeds are cut
first and the cut is reported. The episode set is never cut.
