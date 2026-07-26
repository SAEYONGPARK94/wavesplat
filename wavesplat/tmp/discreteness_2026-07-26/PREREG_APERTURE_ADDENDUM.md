# Pre-registration ADDENDUM — aperture arm of the discreteness experiment (W9-b)

**Written and frozen 2026-07-26, BEFORE any new episode was generated and before any arm was run.**
Parent: `tmp/discreteness_2026-07-26/PREREG.md` (+ its result `RESULT.md`, verdict CONFIRM).
Result file to be written: `RESULT_APERTURE.md`. Scripts/logs in this directory.
Nothing in `wavesim/` or `wavesplat/` is modified; no shipped default changes; nothing committed.

---

## 0. Why this run exists

The parent run returned **CONFIRM** (learned advantage collapses on continuum seas:
R_A −29.0 % → R_B −9.6 % → R_C −5.7 %, p_C = 0.080), but with a caveat that blocks the
interpretation: in the continuum cell **both arms fell to the no-skill floor** (mean skill
cond 0.055, analytic 0.015; 5/10 analytic episodes no better than the zero predictor). The paired
ratio therefore cannot separate

- **(i)** "spectral discreteness is what the learned prior exploits", from
- **(ii)** "continuum seas are simply unsolvable at this aperture, for any method".

This addendum repeats the continuum cell at a **larger effective aperture** so that continuum seas
remain solvable, plus an aperture-matched discrete control.

---

## 1. Aperture lever (FROZEN, declared before generation)

**Lever chosen: the observation window `T_obs`, at constant observation count.**

Measured current aperture (`data/cell_c/…seed10_uniform.h5`, init window `tp_factor=2.0`):
`L_x = 168.8 m`, `T_obs = 14.0 s`, `c_g = 5.46 m/s` (Tp = 7 s deep water, c_g = c_p/2 = gT/4π),
`c_g·T_obs = 76.5 m`, **`D_eff = L + c_g·T_obs = 245.3 m`**, `δk = 2π/D_eff = 0.0256 rad/m`
(peak k = 0.0964, i.e. only ~3.8 resolution cells across the peak wavenumber).

New cells are generated with **`dt = 0.4 s`, `duration = 120.0 s` → 301 frames** (unchanged frame
count, unchanged generation cost) and read with **`tp_factor = 8.0`**. Verified frame arithmetic
(computed before freezing, asserted at run time):

| quantity | original cells (A, C) | new cells (A-big, C-big) | ratio |
|---|---|---|---|
| dt / duration | 0.1 s / 30 s | **0.4 s / 120 s** | — |
| frames per episode | 301 | 301 | 1.0 |
| WS init window (`tp_factor`) | 2.0 → **141 frames**, 14.0 s | 8.0 → **141 frames**, 56.0 s | **4.0× time, 1.0× frames** |
| cond window (`nb` frames) | 154 frames, 15.4 s | **154 frames**, 61.6 s | **4.0× time, 1.0× frames** |
| observations in the init window | ~7 070 | ~7 070 | 1.0 |
| `c_g·T_obs` (WS) | 76.5 m | **306.0 m** | 4.0 |
| **`D_eff`** | **245.3 m** | **474.8 m** | **1.94** |
| **`δk = 2π/D_eff`** | 0.0256 rad/m | **0.0132 rad/m** | **0.52** |
| cells across peak k | 3.8 | 7.3 | 1.94 |

This is a **single-lever, data-volume-matched** manipulation: the same number of frames and the
same number of LiDAR returns are spread over a 4× longer baseline, so any change is attributable
to `D_eff`, not to having more data. `L` is untouched — the sensor is unchanged (program
constraint "given sensor"), which is why `D_eff` grows 1.94× rather than 4×.

**Temporal-Nyquist safety (checked before freezing):** the highest component frequency is
ω = 2.69 rad/s (period 2.34 s) → `dt = 0.4 s` gives 5.8 samples/period; the highest component
holding ≥1 % of variance is ω = 1.87 rad/s → 8.4 samples/period. Both comfortably above Nyquist.
`dt = 0.8 s` was rejected for this reason (2.9 samples/period at ω_max). This program has a
mesh-Nyquist artifact history; the temporal analogue is not being repeated.

### Implementation (no library edits)
- Generation: `gen_cells.py --dt 0.4 --duration 120.0`, same monkey-patched component source as
  the parent run (in-process only).
- WS arm: `run.py` hardcodes `get_init_window(tp_factor=2.0)`; for aperture cells the runner
  rebinds `EpisodeDataLoader.get_init_window` / `.get_tracking_frames` **inside the worker
  process** to force `tp_factor = 8.0`. Shipped default untouched, exactly as
  `TRACK_PHASE_PREDICT=False` is handled.
- cond arm: **no change at all.** `nb = round(K_TP·Tp/0.1) = 154` is a frame count, so at
  `dt = 0.4 s` it automatically spans 61.6 s. Asserted at run time, not assumed.

---

## 2. Cells (FROZEN)

Two cells × 10 episodes = 20 episodes. **Fresh seeds 20, 21, 22, 23, 24** (unused anywhere:
prior work used 0–4, 7, 10–14, 42), × {uniform, slope} raydrop.

| cell | recipe | atoms | aperture |
|---|---|---|---|
| **A-big** | identical to parent cell A (M = 1, σ_sub = 0°, σ_parent = 30°) | 15 | `tp_factor` 8.0, dt 0.4 |
| **C-big** | identical to parent cell C (M = 31, σ_sub = 28°, σ_parent = 10.7703°, ω jitter uniform within the parent JONSWAP bin) | 465 | `tp_factor` 8.0, dt 0.4 |

Everything else is unchanged from the parent prereg: BF4 (Hs 1.2, Tp 7.0), 15 parents, depth 50 m,
n_pts 50, mesh 200 × 167, GPU raycast, energy-preserving expansion, total directional spread held
at 30° by `σ_parent = sqrt(30² − σ_sub²)`.

## 3. Arms and endpoints (FROZEN — unchanged from the parent prereg)

Learned = cond-diffusion set-proposal (`beamform_cond_diffusion.pt`, read-only), 5 diffusion seeds
(0–4), n = 4 BMA + 80-iter polish, 3 nowcast frames. Analytic = `run_estimation` with
`TRACK_PHASE_PREDICT=False`, adaptive_n, cpu, 5 estimator seeds (42, 1, 2, 3, 4);
`ws_best = min(median init, median tracked)`. **Median-of-5 mandatory both arms.**
`R` = median over the cell's 10 episodes of `(cond_i − ws_best_i)/ws_best_i`.
`p` = one-sided Wilcoxon signed-rank on the paired absolute deltas, H1 cond < ws_best, α = 0.05.

**Solvability metric (frozen):** `skill = 1 − nRMSE/floor`, where `floor` is the zero-predictor
nRMSE evaluated at the same 3 nowcast frames (`noskill_floor.py`, already written and used in the
parent run). The reported statistic is the **mean analytic (`ws_best`) skill over the cell's 10
episodes**.

---

## 4. Frozen verdict cells — evaluated in this order

0. **APERTURE-LIMITED** — if mean analytic skill in **C-big < 0.15**, the cells are still
   information-limited. Retraining cannot recover skill at this sensor configuration; the
   recommendation becomes observation-window / footprint expansion first. No discreteness verdict
   may be issued from this run.
1. **(1) DISCRETENESS NOT BINDING** — C-big solvable (analytic skill ≥ 0.15) **AND** the learned
   advantage REAPPEARS in C-big: `R_Cbig ≤ −20 %` **AND** `p_Cbig < 0.05`.
   ⇒ discreteness was not the binding constraint at adequate aperture; there is recoverable skill
   on continuum seas for learning to chase ⇒ **GO for the mixed retrain.**
2. **(2) DISCRETENESS IS WHAT LEARNING EXPLOITS** — C-big solvable **AND** the advantage stays
   collapsed (`R_Cbig > −10 %` **OR** `p_Cbig ≥ 0.05`) **AND** the aperture-matched discrete
   control retains a clear advantage (`R_Abig ≤ −20 %` **AND** `p_Abig < 0.05`).
   ⇒ retraining must specifically add continuum competence (mixed curriculum) ⇒ **GO with
   curriculum emphasis.**
3. **AMBIGUOUS** — solvable but neither (1) nor (2) obtains (e.g. C-big collapsed *and* A-big also
   collapsed, which would mean the aperture change itself removed the advantage everywhere; or
   `R_Cbig` between −20 % and −10 % with p < 0.05). Reported as such, **no narrative rescue**.

Cells (1) and (2) differ in mechanism but **both greenlight the retrain**; the report must say
which one obtains.

**No threshold, parameter, seed, arm, σ, M, dt, tp_factor or metric definition may change after
any number is seen.**

## 5. Gates (FROZEN, same as parent)

- **G-mesh:** every new episode must carry `mesh_verdict == 'ADEQUATE'` (checked on the expanded
  component set inside `run_episode`). Non-ADEQUATE ⇒ validity flag, reported not dropped.
- **G-identity:** A-big components bit-identical to `jonswap_components(Hs=1.2, Tp=7.0,
  n_components=15, depth=50, seed=s)` for all 5 seeds.
- **G-roundtrip:** `EpisodeDataLoader` opens one episode per cell; 301 frames, Hs_true 1.2,
  Tp_true 7.0, GT shape (301, 60, 80); **and the asserted window arithmetic: 141 init frames
  spanning 56.0 s at tp_factor 8.0, cond nb = 154 frames spanning 61.6 s.**
- **G-spread:** realized energy-weighted directional spread reported per episode (expect ≈30°).
- Realized-window-Hs scatter is expected again (it is the signature of the manipulation, parent
  §6); `R` is a same-episode ratio and is exactly invariant to it. Reported, not gated.

## 6. Compute budget

20 episodes (same per-episode cost as the parent run: 301 frames), 2 generator processes;
100 analytic runs (**≤ 4 CPU workers**, per coordinator constraint — heldout40 generation is using
CPU); 100 learned runs (GPU); 20 secondary. Projected ≈ 2 h against a ~3 h GPU window.
If it over-runs, the cond arm's diffusion seeds are cut first and the cut is reported; the episode
set is never cut.
