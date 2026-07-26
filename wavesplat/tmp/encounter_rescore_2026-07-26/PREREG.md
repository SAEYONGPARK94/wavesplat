# Pre-registration — LOCAL encounter forecast vs. the GLOBAL-MAP information wall (W9-c)

**Written and frozen 2026-07-26, BEFORE any encounter-scored number was computed.**
Companion result: `tmp/encounter_rescore_2026-07-26/RESULT.md`. Scripts and raw JSONL live in
this directory. **Nothing in `wavesim/` or `wavesplat/` is modified; nothing is committed.**

---

## 0. Hypothesis under test

The W9 discreteness run (`tmp/discreteness_2026-07-26/RESULT.md`, `RESULT_APERTURE.md`) measured
that on **continuum (non-repeating) seas** both arms sit at the no-skill floor for the
**GLOBAL MAP** target: skill 0.055 (learned) / 0.015 (analytic) at base aperture, 0.090 / 0.061
at 1.94× aperture. That is a statement about reconstructing the whole 80×60 GT grid.

**User hypothesis (2026-07-26):** the global-map wall may NOT apply to the **LOCAL, short-horizon
encounter-forecast** task that the thesis actually targets (frozen 2026-07-05: *"given-sensor,
control-compatible real-time encounter forecasting"*) — predict surface elevation η at a point
ahead of the ship, 0–5 s out. If local skill survives where global skill was ≈0.06, the thesis's
original target is the right frame and the W9 wall is **scoped to global maps**.

---

## 0b. USER-DIRECTED DESIGN AMENDMENT (recorded 2026-07-26, before any scored number)

> **Validity statement, explicit as required:** at the moment this amendment was received and
> written into this file, **no encounter/local-skill number of any kind had been computed or
> seen** by the agent. The only measurements taken up to that point were (i) structural
> inspection of the episode HDF5 files, (ii) verification that the stored GT field is
> reproduced by the analytic component superposition to 2e-15, (iii) footprint geometry
> percentiles, (iv) wall-clock timings of `initialize()`, and (v) an arm-identity check on the
> *parent* W9 run (§7). None of these is a scored endpoint of this experiment.

The deliverable is no longer a single local point. It is the **PREDICTABLE-RANGE PROFILE**:
skill as a function of **distance ahead of the observation footprint**, for **per-window
("still-cut") estimation with NO tracking**. Distances, the R\* statistic and the verdict cells
below are the amended, frozen versions.

---

## 1. Cells and episodes (FROZEN — inherited verbatim from W9)

| cell | spectrum | aperture | episodes | D_eff |
|---|---|---|---|---|
| **C** | continuum-like (465 atoms, σ_sub 28°, ω jitter) | base (dt 0.1 s) | seeds 10–14 × {uniform, slope} | 245 m |
| **C-big** | continuum-like | 1.94× (dt 0.4 s) | seeds 20–24 × {uniform, slope} | 475 m |
| **A** | discrete control (15 atoms) | base | seeds 10–14 × {uniform, slope} | 245 m |
| **A-big** | discrete control | 1.94× | seeds 20–24 × {uniform, slope} | 475 m |

Files: `tmp/discreteness_2026-07-26/data/cell_{c,cbig,a,abig}/`. 40 episodes total.
**C / C-big are the decision cells; A / A-big are the discrete reference.**

---

## 2. Ground truth (FROZEN)

GT η is evaluated **analytically** from the episode's own stored `config/wave_components`
(columns `[A, ω, kx, ky, φ, Q]`) as `η(x,y,t) = Σ A_i cos(kx_i x + ky_i y − ω_i t + φ_i)` —
the identical formula `wavesim.wave_generator.single_gerstner` uses to render `gt_wave_field`
(Q enters only the horizontal Gerstner displacement, not Z).

**Verified before freezing:** for cells a / c / cbig, frames 0 and 157, the analytic evaluation
reproduces the stored `gt_wave_field` to **max |Δ| ≤ 2.2e-15 m**. This licenses scoring at
arbitrary (x, y, t), including points outside the stored 80×60 GT grid and times off the frame
raster (needed for 1 s leads at the big cells' dt = 0.4 s).

---

## 3. Geometry (FROZEN)

- **Dominant propagation direction** `ê = +x̂`. This is the **generator's design constant**
  (`jonswap_components(direction_deg=0)`, identical in all four cells by construction); it is
  NOT estimated per episode and NOT read from GT. (Realized energy-weighted mean directions are
  −3.8° / −14.8° / −1.3° for c / a / cbig — reported, not used.)
- **Footprint down-wave edge** `x_edge(i_ref)` = the **95th percentile** of the x-coordinate of
  all valid LiDAR returns in frames `[i_ref−154, i_ref]` (the union window, §4). p95 rather than
  max because the grazing-incidence tail reaches 200+ m on a footprint whose bulk ends near 39 m.
  Measured typical value ≈ **39 m**.
- **Lateral position** `y_c` = median y of the same returns (measured ≈ 0 m).
- **Encounter points** `P_d = (x_edge + d, y_c)` for
  **d ∈ {0, 10, 22, 50, 100, 200} m**. The 22 m point is the user's 2 s @ 22 kt (11.3 m/s)
  control reference and is marked on every figure.
- **Descriptive (not a decision variable):** the fraction of window observations lying beyond
  each d is reported, because the sensor's grazing tail means d = 100/200 m are not strictly
  unobserved regions.

---

## 4. Estimation protocol — per-window, NO tracking (FROZEN)

**Forecast origins** (4 per episode; both arms share them, so both are scored at *identical*
space-time points and against one common zero-predictor floor):

| cells | i_ref frames | t_ref [s] |
|---|---|---|
| A, C (dt 0.1 s) | 157, 188, 219, 250 | 15.7, 18.8, 21.9, 25.0 |
| A-big, C-big (dt 0.4 s) | 157, 200, 243, 287 | 62.8, 80.0, 97.2, 114.8 |

Origins are chosen so that (i) the full 154-frame learned window fits before them and (ii)
`t_ref + 5 s` stays inside the episode duration. Both arms use **only data with t ≤ t_ref**
(strictly causal). **No tracking / no sequential update is run — this is the "still-cut",
per-window estimator the amendment asks for.**

- **Analytic arm (`ws`)** — `SequentialUpdater(cfg).initialize(frames[i_ref−140 : i_ref+1])`
  (141 frames = the shipped `get_init_window(tp_factor=2.0)` length at base, and the aperture
  arm's `tp_factor=8.0` length at the big cells: identical frame count and observation count,
  only shifted so the window ends at the common origin), then
  `predict_field([x_edge+d …], [y_c], t_ref+ℓ)`.
  `cfg = DEFAULT_CONFIG + {adaptive_n: True, device: 'cpu', seed: s, init_method: 'v2_konly'}`,
  with `sequential_update.TRACK_PHASE_PREDICT = False` set in the worker (shipped default
  untouched). **Estimator seeds (42, 1, 2, 3, 4) — median-of-5.**
- **Learned arm (`cond`)** — verbatim `run_cond_crowded.eval_episode` internals:
  `gather_frames(L, i_ref, nb=154)` → `M_OBS` subsample (rng 0) → time shift `t0 = min t` →
  `_bmap_from_obs` → `cd.sample(model, Bmap, n=4, seed=diff_seed)` → `cd.polish_atoms(…, iters=80)`
  → `atoms_to_field` (which refits amplitudes/phases on the window obs) rendered at
  `gx = [x_edge+d …]`, `gy = [y_c]`, `t = t_ref + ℓ − t0` → **BMA = mean of the 4 sample fields**.
  Checkpoint `results/direction3/beamform_cond_diffusion.pt`, read-only.
  **Diffusion seeds (0, 1, 2, 3, 4) — median-of-5.**

**Leads** `ℓ ∈ {0, 1, 2, 5} s` (exact, thanks to §2 analytic GT).

---

## 5. Metric (FROZEN)

For arm ∈ {ws, cond}, cell, distance d, lead ℓ:

1. per (episode, seed): `rms = sqrt( mean over the 4 origins of (pred − gt)² )`
2. per episode: **median over the 5 seeds** of `rms` → `rms_ep` (median-of-k is mandatory;
   single-seed episode deltas are forbidden — project convention)
3. `nRMSE_cell = mean over the 10 episodes of (rms_ep / Hs)`, `Hs = 1.2 m`
4. zero-predictor floor at the **same** points: `rms0_ep = sqrt(mean over origins of gt²)`,
   `floor_cell = mean over episodes of (rms0_ep / Hs)`
5. **`skill(d, ℓ) = 1 − nRMSE_cell / floor_cell`**

This is structurally identical to `tmp/discreteness_2026-07-26/noskill_floor.py`, so the numbers
are directly comparable to the global-map skills 0.055 / 0.015 / 0.090 / 0.061.

**Usable range** `R*(τ)` = the largest d in the grid {0, 10, 22, 50, 100, 200} such that
`min over ℓ ∈ {0, 1, 2} s of skill(d, ℓ) ≥ τ` **and** the same holds at every smaller d in the
grid (contiguous range from the footprint edge outward). If the criterion already fails at
d = 0 m, `R* = NONE` (reported as 0 m in the amendment's wording). Reported for **τ = 0.5**
(primary bar) and **τ = 0.3** (softer bar). Non-contiguous curves, if any, are reported as such.

---

## 6. Frozen verdict cells — evaluated in this order

0. **LOCAL-SKILL-EXISTS** — `R*(0.5, lead ≤ 2 s) ≥ 22 m` on a **continuum** cell (C or C-big)
   for **at least one arm** ⟹ the user's target reframe is validated; the W9 wall is scoped to
   global maps.
1. **NO-LOCAL-SKILL** — `R*(0.3) = NONE` for **every** cell × arm ⟹ the wall is
   task-independent and the reframe fails.
2. **PARTIAL** — anything else. Reported as PARTIAL with the measured `R*` values.
   **No narrative rescue.**

Additionally reported, not verdict-bearing:
- **learned − analytic delta on the encounter metric, per cell** (Δskill and relative nRMSE
  ratio at each d, ℓ): the direct test of whether learning helps LOCALLY on continuum seas.
- **within-run global-map anchor**: the same fitted models scored on the full 80×60 GT grid at
  lead 0, so the local/global contrast is measured inside one run rather than across runs.
- per-episode skill spread and the count of episodes at-or-worse than the zero predictor.

No threshold, distance, lead, arm, seed, cell or metric definition may change after any number
is seen.

---

## 7. Declared protocol deviations from the parent W9 arms

1. **Window placement.** W9's analytic arm used frames `[0, 140]`; here the same 141-frame
   window is slid to end at the shared origin `i_ref`. Frame count, duration and observation
   count are unchanged; only the placement differs. Needed so both arms share one origin and
   one floor.
2. **No tracking.** W9's `ws_best = min(median init, median tracked)`. Here the analytic arm is
   **frozen-init forecast only** — required by the amendment's "still-cut, no tracking".
3. **`init_method` is pinned to `'v2_konly'`.** W9's `run_ws.py` passed
   `config={'adaptive_n', 'seed', 'device'}` and inherited the *shipped* default. That default
   was flipped to `'auto'` in commit `aa8e4c3` at **11:53** on 2026-07-26, and the W9 WS arm ran
   at **14:16–14:42** — i.e. **after** the flip. Live check on `cell_c/seed10_uniform`:
   `init_method='auto'` routes to **learned** (n_selected = 18, bf_peak_ratio = 0.44, no guard
   veto). ⚠ **This means W9's "analytic arm" may not have been analytic.** Corroborating
   pattern: `n_gaussians == max_n == 30` occurs in 0/50 cell-A runs but 16/50 cell-C and
   16/50 cell-C-big runs — mirroring the learned proposal's measured `n_selected` saturation on
   continuum seas. A bit-level identity check against the stored `ws_runs.jsonl` is running and
   its outcome is reported in RESULT.md §Flags. **This experiment therefore pins `v2_konly`
   explicitly so that "analytic" means analytic.** The parent-run flag is reported to the caller
   regardless of how it resolves; it is *not* an endpoint of this experiment.
4. **Analytic GT** (§2) instead of bilinear interpolation of the stored grid — exact, and the
   only way to score off-grid distances and off-raster leads.

## 8. Compute budget (declared at freeze)

- analytic: 4 cells × 10 ep × 4 origins × 5 seeds = **800** `initialize()` calls, ≈ 20–35 s each,
  **4 CPU workers** (hard cap per the GPU-busy instruction) ⟹ ≈ 100–120 min.
- learned: 4 cells × 10 ep × 4 origins × 5 seeds = **800** short GPU evals, ≈ 2–3 s each ⟹
  ≈ 30–40 min, queued so as not to disturb the heldout40 lanes.
- **Pre-declared cut order if the GPU cannot take the load:** cond diffusion seeds 5 → 3
  (0, 1, 2), reported as a deviation. The episode set, the origins and the analytic seeds are
  **never** cut.
