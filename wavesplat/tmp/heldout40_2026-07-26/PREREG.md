# W4 — THE HELD-OUT FORTY · pre-registration (FROZEN before any episode exists)

**Date** 2026-07-26 · **Branch** `learn-boundwave-retrain` · **Author** CPU-phase agent
**Authority** `docs/PERSONA_DEBATE_novelty_phd_value_2026-07-26.md` §6 W4 (the single
non-negotiable condition signed by all five personas) and
`tmp/persona_debate_2026-07-26/R2_P3_methodologist.md` §"CRUX D protocol".

> **Status of this file: FROZEN.** It is written before the 40 episodes exist, before any
> estimator has been run on them, and before any number from them has been seen by anyone.
> **No constant, bar, slice definition, episode list, lane list or analysis rule in this file
> may be changed after any number from these 40 episodes has been seen — by any agent, for
> any reason, including "it was obviously a typo".** A change discovered to be necessary is
> recorded as an *amendment with a timestamp and the reason*, and the run is re-declared
> exploratory. There is no other repair path.

---

## 0. Scope of this document and of this agent's half

This prereg covers the **whole** W4 experiment (all three arms). It is written now, in one
piece, because the bars must be frozen before the data exist — not because the CPU agent will
execute all of it.

| Phase | Contents | Executed by | Status |
|---|---|---|---|
| **CPU (this agent)** | episode generation + mesh-adequacy gate; **PURE-V2 arm only** | this agent, CPU only | in progress |
| **GPU (later)** | learned atom proposals (diffusion), **pure-learned arm**, **auto+guard arm**, oracle, and **every comparison / bar evaluation S1–S7** | a later GPU phase | **PENDING-GPU** |

**Declared explicitly:** the CPU phase computes and reports **per-arm sanity statistics only**
(n, error count, runtime, and per-regime means *of the single v2 arm*). The CPU phase
**does not compute, print, or store any quantity that involves a comparison between arms** —
no deltas, no oracle, no bootstrap, no bar verdict. The scoring is a **single event** that
happens once, in the GPU phase, after all three arms exist.

---

## 1. Frozen constants (the "one sentence" demanded by judge-doc W1(b4))

```
auto_switch_thresh   = 8        # router: route learned iff n_selected >= 8
auto_guard_ratio     = 0.05     # guard: veto learned -> v2_konly iff bf_peak_ratio < 0.05
TRACK_PHASE_PREDICT  = False    # tracker double-phase-advance fix (aa8e4c3)
real detector tau    = 1.5      # not used by W4; recorded so the triple is unambiguous
```

These are the shipped defaults as of `aa8e4c3` (`wavesplat/optimizer/sequential_update.py`).
Judge-doc E4 is hereby resolved in favour of the **code**: the threshold is **8**, not 6.
No sensitivity sweep of these three constants is permitted on the held-out 40 — a sweep is a
second look, and this set is scored once.

**Estimator lanes (frozen).** 5 paired lanes `(est_seed, diff_seed)` =
`(42,0), (1,1), (2,2), (3,3), (4,4)`, **median-of-5 per episode**, identical for all three
arms — verbatim the protocol of `tmp/wp_integration_2026-07-25/run_arms.py` and
`tmp/unified_guard_2026-07-26/PREREG.md`.

**Arms (frozen), all on identical lanes and identical episodes:**
1. `pure_v2` — `init_method='v2_konly'` (dispersion-constrained 2D-MNOMP), CPU;
2. `pure_learned` — `init_method='learned'`, `learned_init_iters=500`, cached atoms; PENDING-GPU;
3. `auto+guard` — `init_method='auto'` with the two constants above; PENDING-GPU;
4. `oracle` — per-episode min over arms 1–2, **recomputed lane-consistently by one script**
   (judge-doc E3 / W1(b3)): the oracle and the switch must be aggregated the same way, or S4
   is not evaluable. PENDING-GPU.

---

## 2. The 40 episodes

### 2.1 Stratification — mirrors the 100-episode grid at exactly 0.4×

The in-sample grid (`tmp/regen_run_finemesh.py:SCENARIOS`, = `configs/default.yaml`
`dataset.scenarios` S01–S10) is 10 scenarios × 5 seeds × 2 raydrop = 100, which splits under
the **scorer's own slice definitions** (`tmp/unified_guard_2026-07-26/score_task1.py:93-95`)
as

```
SPARSE   = n_components in (1, 2)   -> 40 episodes
MOD      = n_components == 5        -> 20 episodes
CROWDED  = n_components in (15, 30) -> 40 episodes
```

The held-out set is the same grid at 0.4×: **10 scenarios × 2 fresh seeds × 2 raydrop = 40**,
giving **16 sparse / 8 moderate / 16 crowded** exactly as demanded.

> **Declared deviation from the task brief's parenthetical.** The brief describes the sparse
> stratum as "BF2–3 × 1–2comp". The in-sample sparse slice is defined *by component count
> alone* and therefore **includes `BF4_1comp`**. Restricting to BF2–3 would (a) make the
> held-out sparse slice a different population from the in-sample one against which bar **S2**
> is stated, and (b) be arithmetically impossible at n=16 while mirroring the grid (3 scenarios
> × 2 raydrop = 6 cells). The slice definition is therefore taken **verbatim from the scorer**
> (`comp(e) in (1,2)`), which is the only definition under which S2/S3 are comparable to their
> in-sample references. This deviation is declared here, before the data exist.

### 2.2 Fresh generator seeds — disjointness evidence

Seeds appearing in **any** episode file anywhere in the repository, enumerated before choosing
(`ls data*/ tmp/*/ tmp/*/*/ | grep -oP 'seed\d+' | sort -u`):

```
data/                              : 0 1 2 3 4 7 42
data_finemesh_2026-07-23/          : 0 1 2 3 4
data_finemesh_extras_2026-07-24/   : 0 7 42
data_finemesh_fixedT_2026-07-23/   : 0 1 2 3 4 5 6 7 8 9
data_fixedT_old_2026-07-11/        : 0 1 2 3 4 5 6 7 8 9
data_overwritten_backup_2026-07-13/: 0 1 2 3 4
-> union used: {0,1,2,3,4,5,6,7,8,9,42}
```

Additionally reserved by the **concurrent** SPREAD-spectrum agent
(`tmp/discreteness_2026-07-26/gen_cells.py:35`, `SEEDS = [10, 11, 12, 13, 14]`):
`{10,11,12,13,14}`. Its directories and files are **not touched** by this work.

**FROZEN CHOICE: generator seeds `777` and `888`.** Both are disjoint from every seed above.
**Reserve ladder for adequacy failures, in this order (frozen): `999, 1001, 1111, 1212,
1313, 1414`** — see §3.3.

### 2.3 The frozen episode list (40)

Filenames follow the repository convention exactly
(`episode_{sea_state}_{n}comp_seed{seed}_{raydrop}.h5`) so that every existing slice regex
(`_(\d+)comp`) and loader works unchanged.

| Stratum | Scenario | seeds | raydrop | n |
|---|---|---|---|---|
| sparse | BF2_1comp | 777, 888 | uniform, slope | 4 |
| sparse | BF3_1comp | 777, 888 | uniform, slope | 4 |
| sparse | BF3_2comp | 777, 888 | uniform, slope | 4 |
| sparse | BF4_1comp | 777, 888 | uniform, slope | 4 |
| **sparse total** | | | | **16** |
| moderate | BF3_5comp | 777, 888 | uniform, slope | 4 |
| moderate | BF4_5comp | 777, 888 | uniform, slope | 4 |
| **moderate total** | | | | **8** |
| crowded | BF4_15comp | 777, 888 | uniform, slope | 4 |
| crowded | BF4_30comp | 777, 888 | uniform, slope | 4 |
| crowded | BF5_15comp | 777, 888 | uniform, slope | 4 |
| crowded | BF5_30comp | 777, 888 | uniform, slope | 4 |
| **crowded total** | | | | **16** |
| **TOTAL** | | | | **40** |

Output directory (new, never written before): **`data_heldout40_2026-07-26/`**.
Nothing in `data/`, `data_finemesh_*`, or `tmp/discreteness_2026-07-26/` is read for writing
or modified.

**Independence note, stated in advance:** as in the in-sample set, `uniform` and `slope`
share a wave realization for a given (scenario, seed), so the 40 episodes carry **20 distinct
sea realizations** — the same 2:1 file-to-realization ratio as the 100-episode set. The
episode bootstrap of **S1** therefore resamples *episodes*, as pre-registered by P3; a
realization-blocked bootstrap is reported alongside as a free robustness check, and if the two
disagree the **realization-blocked** one is the one quoted.

---

## 3. Generation protocol (CPU phase)

### 3.1 Frozen generation constants

Identical to `tmp/regen_run_finemesh.py` defaults, which produced `data_finemesh_2026-07-23`:

```
mesh.nx, mesh.ny = 200, 167      # dx=1.2060 m, dy=1.2048 m
n_pts            = 50
duration         = 30.0 s
dt               = 0.1 s         -> 301 frames
depth            = 50.0 m
direction_deg    = 0.0
peak_locked      = False
hos.enabled      = False         # linear ground truth, as in the in-sample set
raydrop_model    in {uniform, slope}
sea states       from configs default (BF2..BF5)
```

### 3.2 The one deliberate difference from the in-sample generation, and why it is a no-op

The in-sample set was rendered with `use_gpu_raycast=True`. **The GPU is reserved by another
experiment, so this set is rendered on the CPU** with `raycast_surface`
(`wavesim/lidar_simulator.py`), which `wavesim/lidar_gpu.py`'s own docstring and
`wavesim/tests/test_lidar_gpu.py` certify as **hit-for-hit identical** to the GPU path
(intersection coordinates differ by ~1e-13 m from float reassociation; measured here at
3.6e-14 m on a probe frame).

`cfg.raycast.batch_size` is lowered from **4000 to 64**. This is a **memory-chunking
parameter only**: batches do not interact, each batch computes the closest hit for its own
rays over the same candidate set. **Verified bit-identical** before freezing, on a probe frame
(200×167 mesh, 66 068 triangles, 76 000 rays): batch sizes {4000, 1024, 256, 64, 32} all
returned 61 108 identical hits with identical hit points; wall time 25.66 / 18.79 / 9.89 /
5.83 / 6.96 s. **Nothing that decides *which* triangles a ray is tested against is changed**
(spatial hash 20×15, ±1-cell margin, candidate set, acceptance thresholds, closest-hit rule) —
those are the observation operator and are untouched.

Workers: **6 maximum** (`nice -n 10`, `OMP_NUM_THREADS=1`), so the concurrent SPREAD agent is
not starved.

### 3.3 Mesh-adequacy gate (mandatory — the program has a mesh-Nyquist artifact history)

`wavesim.mesh_adequacy.check()` runs inside `run_episode` and its verdict is written into the
file (`config.attrs['mesh_verdict']`). **Frozen rule:**

- An episode is **accepted only if `mesh_verdict == 'ADEQUATE'`** (`var_under_resolved <= 0.05`
  and peak above Nyquist). `DEGRADED` is **not** accepted for this set.
- Any episode that is not ADEQUATE, or that contains NaNs in `gt_wave_field` or in valid
  LiDAR points, or that yields 0 frames, is **regenerated with the next seed from the reserve
  ladder in §2.2**, for that cell only, and **every such regeneration is logged**
  (cell, rejected seed, verdict, replacement seed) in `RESULT_PARTIAL.md` and in
  `data_heldout40_2026-07-26/_heldout40_manifest.json`.
- The reserve ladder is consumed in order and never re-ordered to obtain a nicer episode.
  Seeds are **never** selected by looking at any estimator output — only at the adequacy
  verdict, which is a property of the sea and the mesh alone and is computed **before** any
  estimator sees the file.
- If the ladder is exhausted for any cell, the run **stops** and the failure is reported; the
  set is not silently reduced below 40.

---

## 4. FROZEN BARS

### 4.1 Verbatim from `tmp/persona_debate_2026-07-26/R2_P3_methodologist.md` §"CRUX D protocol (frozen 8/0.05/1.5, one shot, no peeking)"

> Synthetic 40 fresh (16/8/16), three arms, identical lanes, median-of-5. Bars = PAIRED CONTRASTS not absolute levels ("≤0.0875" is a coincidence, not a bar):
> - S1 (primary; the test that never existed): episode-bootstrap 95% BCa CI for mean(auto+guard)−mean(pure_learned) entirely <0; same vs pure_v2. Both required.
> - S2: sparse mean ≤ pure_v2 + 0.005. S3: crowded mean ≤ pure_learned + 0.005. S4: auto−oracle ≤ 0.015 (in-sample 0.0057, allows 2.6x degradation).
> - S5 (converts the tautology into a test): guard fires on 0 crowded episodes AND min(bf_peak_ratio | fresh crowded) > 0.05.
> - Zero-inflation fix (mandatory): report n_discordant per slice; non-inferiority ON THE DISCORDANT SET ONLY; n_discordant<8 → UNDERPOWERED, no direction.
> - Frozen failure handling: S1 fails+S2/S3 pass → "matches the better specialist per regime," no aggregate win. S2 fails → guard didn't generalize, Ch.5 headline cut.

### 4.2 Verbatim from `docs/PERSONA_DEBATE_novelty_phd_value_2026-07-26.md` §6 W4 (judge's reconciled table, including P1's thresholds)

> - **Protocol:** 40 fresh-seed episodes (16 sparse / 8 moderate / 16 crowded), constants **frozen** (`auto_switch_thresh=8`, `auto_guard_ratio=0.05`, real τ=1.5), three arms on identical lanes, median-of-5 per episode, **scored once, no peeking, no re-running.**

| ID | Bar | Source & reconciliation |
|---|---|---|
| **S1 (PRIMARY)** | Episode-bootstrap 95% BCa CI for mean(auto+guard) − mean(pure_learned) entirely **< 0**, **and** the same versus pure_v2. **Both required.** | P3 — **adopted as strictest defensible**: it is draw-invariant, whereas absolute levels depend on the difficulty of the fresh draw. Supersedes P2's and P5's one-sided paired Wilcoxon α=0.05 (weaker) — report the Wilcoxon too, it is free. |
| **S1b (absolute, wording gate)** | mean ≤ **0.095** to retain headline-number wording; mean > **0.1011** ⟹ capstone dead outright. | P1 set ≤0.095, P2 and P5 set ≤0.1011. Both retained, at different jobs: 0.095 gates the *wording*, 0.1011 gates *survival*. Between them ⟹ "matches the better specialist per regime," no aggregate win. |
| **S2** | sparse mean ≤ pure_v2 + **0.005** | P3 and P5 (0.005) adopted over P1 (0.006); P2's absolute ≤0.030 recorded as a cross-check. |
| **S3** | crowded mean ≤ pure_learned + **0.005** | P3 adopted over P1 (0.006). |
| **S4** | auto − oracle ≤ **0.015**, oracle recomputed lane-consistently by the W1b script (in-sample 0.0057, so this allows 2.6× degradation) | P3, with P5's aggregation fix folded in. |
| **S5** | Guard fires on **0** crowded episodes **AND** min(bf_peak_ratio ∣ fresh crowded) > 0.05 | P3 — converts the tautological criterion #2 into a real test. |
| **S6** | ≤ **2** damaging misroutes (in-sample was 5/100; pro-rata on 40) | P3. Failure ⟹ the guard was a patch, not a mechanism. |
| **S7** | The moderate (5-component) cell must not lose more than it wins (in-sample 6 L / 4 W) | P1. |
| **Reporting rule** | Report `n_discordant` per slice; any non-inferiority statement is computed on the discordant set only; `n_discordant < 8` ⟹ **UNDERPOWERED**, no direction claimed. Non-inferiority endpoints otherwise retired. | P3, P5. |

> - **Pre-committed failure branches:** S1 fails while S2/S3 pass ⟹ the claim becomes "matches the better specialist in each regime," and the aggregate win is deleted — **the regime map still passes.** S2 fails ⟹ the guard did not generalize and the Ch.5 headline is cut. Any failure ⟹ delete the unified headline and ship the failure map.

### 4.3 Reference values these bars are read against (in-sample, already published, quoted here so no one re-derives them later)

`pure_v2` sparse in-sample **0.0204**; `pure_learned` sparse in-sample **0.0534**; current-auto
100ep **0.0979**; guard-auto 100ep **0.08747**; oracle **0.0818**; in-sample auto−oracle
**0.0057**; in-sample crowded floor of `bf_peak_ratio` **0.0553**. These are **context**, not
bars. **S1b's absolute gates (0.095 / 0.1011) are the only absolute numbers in the bar set**,
and both were set before the draw.

---

## 5. Rules that bind the scoring event

1. **One shot.** The three arms are run once on these 40 episodes. No re-run "with a fixed
   seed", no "we noticed a bug and re-ran", except as a *declared amendment* that voids the
   held-out status.
2. **No peeking across arms during execution.** The CPU phase reports the v2 arm's own sanity
   statistics only. Cross-arm quantities are computed by one script, once, in the GPU phase.
3. **No constant may change after any number is seen.** Restated because it is the whole point
   of the exercise: the in-sample capstone's defect was that its constants were *estimated
   from the scoring data*. Any adjustment of `auto_switch_thresh`, `auto_guard_ratio`,
   `TRACK_PHASE_PREDICT`, the lane list, the slice definitions, or the bars themselves, made
   after any held-out number exists, converts this run from confirmatory to exploratory and it
   must be reported as such in those words.
4. **The oracle is computed lane-consistently** (judge-doc E3): both the switch and the oracle
   aggregate per-lane-then-median, by the *same* script, or S4 is reported as NOT EVALUABLE.
5. **Failure is a result.** Every pre-committed failure branch in §4.1/§4.2 is already written;
   taking one is compliance, not disappointment.
6. **Evidence map is generated, not typed** (judge-doc C13/W1a): the S1–S7 verdicts must be
   emitted by a script that applies the criteria in code to the arm JSONLs. A hand-typed bar
   verdict is not acceptable evidence.

---

## 6. What the GPU phase must do (declared PENDING-GPU)

Written here so that the GPU phase inherits a frozen instruction and adds nothing:

1. Precompute learned atom proposals for all 40 episodes × 5 diffusion seeds
   `{0,1,2,3,4}` into an atoms cache whose **format is identical to**
   `tmp/wp_integration_2026-07-25/atoms_cache.jsonl`
   (one JSON object per line with at least `episode`, `diff_seed`, `kx`, `ky`, `energy`,
   `n_selected`, `err`).
2. Run `pure_learned` and `auto+guard` on lanes `(42,0),(1,1),(2,2),(3,3),(4,4)`,
   `TRACK_PHASE_PREDICT=False`, median-of-5.
3. Compute `bf_peak_ratio` on all 40 init windows with the frozen helper
   (`wavesplat.optimizer.initializer.beamform_peak_ratio`) — needed by **S5**.
4. Run the single scoring script that emits S1–S7 and the reporting rule, once.

The exact command lines are recorded in `RESULT_PARTIAL.md`.

---

## 7. Compliance record

- Nothing is committed by this phase.
- `data/`, checkpoints, production defaults, and the paper are untouched.
- `tmp/discreteness_2026-07-26/` (the concurrent agent) is read-only-inspected for seed
  disjointness and otherwise untouched; its seeds `{10..14}` are excluded.
- CPU workers ≤ 6, `nice -n 10`; no CUDA context is created by this phase.
