# PREREG — MIXED-CURRICULUM RETRAIN of the beamform-conditioned set-diffusion proposal net

**Frozen 2026-07-26 23:55 KST, BEFORE any training step is taken and BEFORE any gate is run.**
Nothing below may be edited after training starts. No gate may be re-run after its number is seen.

Retrain attempt **#4** of this program. Attempts 1–3 (`prior_net_compdiverse`, `prior_net_fan/fanTF`,
`prior_net_e2eft_A/B`) all targeted **architecture / loss**. That card is spent (three independent
FAILs). **This attempt changes ONLY the training data distribution.** The architecture, the diffusion
schedule, the loss and the optimiser are byte-for-byte the shipped recipe.

---

## 0. Standing caveat, recorded before the fact

`tmp/discreteness_2026-07-26/RESULT_APERTURE.md` frozen recommendation #1 reads:
*"Do not launch the mixed retrain on the strength of this run. Neither GO cell was reached. On
continuum seas there is ≤ 0.09 skill available to either arm at this sensor configuration."*

That recommendation is **accepted and honoured**: the continuum axis (defect D2) therefore carries
**no hard bar** — G3 is report-only. This retrain is launched on the strength of **D1 and D3**, which
are measured, large, and unrelated to the aperture question:

* **D1** — sealed FAIL, `tmp/learned_only_2026-07-26/`: on `BF2_1comp` the net emits the
  `max_n=30` cap of near-degenerate atoms sitting **0.742 kp** from the true swell, while the
  beamform map it is conditioned on puts the peak **0.058 kp** from truth. Two pre-registered
  post-hoc repairs (kp-scaled merge; peak-anchor filter) both FAILED, and the frozen consequence
  text says the fix "is NOT achievable without retraining".
* **D3** — `tmp/wp_e2e_2026-07-25/WP_E2E_2026-07-25.md` L172–174: the single highest-value change
  is to match the training observation distribution to the episode protocol.

---

## 1. Diagnosis being acted on (all MEASURED, with sources)

| # | Defect | Evidence | Fix in this retrain |
|---|--------|----------|---------------------|
| D1a | **cardinality** — the net was *only ever shown 32-line spectra*, so ~24–30 atoms is its only mode | `design_diag_q1.json`: BF2_1comp n_selected = 30 on 8/10 in-sample, 30/19/30/16 on the 4 fresh | stratified `n_components ∈ {1,2,5,15,30}` with an **exact energy-preserving 32-slot target** |
| D1b | **sea-state OOD** — training box `Hs~U[0.8,3.2]`, `Tp~U[5,11]`; deployment envelope is BF2–BF5, and **BF2 = (0.3, 4.0) is outside in BOTH coordinates** | `cd.make_batch` L93 vs `wavesim.pipeline.SEA_STATE` | `(Hs,Tp)` drawn 50 % from a widened box `U[0.25,3.2]×U[3.8,11.5]`, 50 % from jittered BF2–BF5 anchors (±15 %) |
| D2 | **continuum tiling** — never shown a genuine continuum | `tmp/discreteness_2026-07-26/RESULT.md`: cell C n_selected median 22.5, real n_eff ≈ 24 | continuum strata built with the **frozen `gen_cells.expand()` recipe**, observed from all 465 sub-components, supervised by a 32-atom energy-weighted k-means quantisation |
| D3 | **observation geometry** — `make_batch` pairs an (x,y) from a pooled footprint with an **independent** `t~U[0,Tobs]`; deployment pools whole LiDAR frames | `WP_E2E` L172–174. Measured here: the (x,y) **marginals are identical** (r-percentiles 10.7/15.6/21.2/31.5/76.7/144.5 bank vs 10.9/15.5/21.4/32.7/82.4/157.5 gate); the **joint** is not | `GeometryBank` replays real frame-structured (x,y,t) windows: 400 pts over ~130 distinct frame times, 3.1 pts/time — the deployment structure |

---

## 2. Training mixture (FROZEN)

Implemented in `tmp/mixed_retrain_2026-07-26/gen_mixed.py`. Every scene, in every stratum, gets
**sweep-geometry observations** (D3 fix). Verified by `--selftest`.

### 2.1 Stratum proportions — exact, sum = 1.000

| stratum | weight | parents × sub-components | σ_sub | ω-jitter | true comps | distinct target atoms (measured, n=200) |
|---|---|---|---|---|---|---|
| `discrete_crowded` | **0.30** | `nc ∈ {15,30}` × 1 | 0° | no | 15 / 30 | median **15**, range [8, 30] |
| `discrete_sparse` | **0.20** | `nc ∈ {1,2}` × 1 | 0° | no | 1 / 2 | median **1**, range [1, 2] |
| `discrete_moderate` | **0.10** | 5 × 1 | 0° | no | 5 | median **5**, range [3, 5] |
| `continuum_mild` | **0.10** | 15 × 31 | 10° | no | 465 | 32 |
| `continuum_strong` | **0.15** | 15 × 31 | 28° | **yes** | 465 | 32 |
| `intermediate_spread` | **0.15** | `{5,15,30}` × `{3,7,15}` | `U[2,20]°` | p = 0.5 | 15–450 | ≤ 32 |

The crowded weight (0.30) is the largest single stratum by design: G1 is a do-not-forget gate and
crowded is the regime the program's only positive synthetic result lives in.

### 2.2 Per-scene draws (all strata)

`Hs,Tp`: 50 % anchored on BF2–BF5 `(0.3,4.0)(0.6,5.5)(1.2,7.0)(2.0,9.0)` × `U[0.85,1.15]`,
50 % from `Hs~U[0.25,3.2]`, `Tp~U[3.8,11.5]` · `gamma~U[1.5,4.0]` · `direction~U[-50,50]°` ·
total directional spread `~U[max(12,σ_sub+4), 40]°`, split parent/sub by
`σ_parent = sqrt(total² − σ_sub²)` (`gen_cells.parent_spread`) · `Tobs = U[1.9,2.3]·Tp`
(deployment uses `get_init_window(2.0)`, the head-to-head used `K_TP = 2.2`) · obs noise
`N(0, 0.02)` m, linear forward — **both unchanged from the shipped recipe**.

**Declared and deliberate:** anchoring half the scenes on the BF2–BF5 (Hs,Tp) points matches the
*declared deployment envelope* (`reference_agreed_operating_point.md`), which the old training box
did not cover. It is distribution-matching, **not leakage** — every realisation seed is disjoint
(§3.1) and no episode file is ever read for anything but geometry. It does make G1/G2
**in-family, out-of-realisation** tests; **G4 (real stereo) is the out-of-family check** and is the
only gate that can speak to external validity.

### 2.3 32-slot target construction

* `n_true ≤ 32` → `replicate_to_M`: round-robin to exactly 32 slots, energy split
  `A_rep = A/sqrt(count)`. Total energy preserved (verified |ΔE| < 2e-7); the 32 atoms occupy
  exactly `n_true` **distinct** (kx,ky), so `pool_and_select`'s dedup (min-sep 0.03) collapses the
  proposal back to `n_true`. This is `gen_cells.expand()` with σ_sub = 0.
* `n_true > 32` → `quantise_to_M`: energy-weighted k-means (25 Lloyd iters), centroid carries the
  summed cluster energy. Total energy preserved (verified |ΔE| < 3e-8).
* Atoms sorted **|k| ascending** (the canonical order `jonswap_components` already returns).

### 2.4 Representability filter

The atom code cannot hold `|k| > K_LIM = 1.1` and the beamform conditioner only spans `|k| ≤ 0.6`.
Short-Tp / high-nc JONSWAP tails run past that. Those components **stay in the observations** (they
are real waves the sensor sees) but are **excluded from the target** — the honest statement that the
representation is band-limited. Scenes with in-band energy fraction < 0.5 are rejected and redrawn
(≤ 20 tries). All BF2–BF5 gate episodes are fully in-band, so this filter never fires on anything
the gates measure.

---

## 3. Seed and file disjointness (enumerated)

### 3.1 Generator seeds
Training scene seeds are drawn from **`[2_000_000_000, 2_147_483_647)`**. Every generator seed ever
used by any evaluation set is **< 2000**, so disjointness is a checkable interval property:

| evaluation set | seeds used |
|---|---|
| `data_finemesh_2026-07-23/` (G1 secondary) | 0, 1, 2, 3, 4 |
| `data_heldout40_2026-07-26/` (G1 primary, G2) | 777, 888 (+ unconsumed reserve 999, 1001, 1111, 1212, 1313, 1414) |
| `tmp/discreteness_2026-07-26/data/` (G3) | 10–14 (cells a/b/c), 20–24 (aperture cells) |
| `data_finemesh_extras_2026-07-24/` | 0, 7, 42 |
| real stereo (G4) | not synthetic |

Training RNG master seed **20260726**; held-out synthetic validation stream seed **987654321**.

### 3.2 Geometry-donor episodes (14, frozen)
Only `(x, y, t_relative)` is read; `z` is **always** re-rendered from the fresh synthetic scene.

* uniform raydrop (8): `data_finemesh_extras_2026-07-24/episode_BF{6,7}_{1,5,15,30}comp_seed0_uniform.h5`
  — BF6/BF7 are sea states **no gate uses at all**.
* slope raydrop (6): `data_finemesh_2026-07-23/episode_BF{3,4}_5comp_seed{2,3,4}_slope.h5`
  — the 5comp family's seeds 0/1 are the only ones any protocol has ever used; 2/3/4 appear nowhere.

**None of the 14 appears in any of the four gate episode lists.**

---

## 4. Training compute budget (FROZEN)

| item | value |
|---|---|
| steps | **40 000** |
| batch | 32 (unchanged) |
| optimiser | AdamW, lr 2e-4, wd 1e-4, cosine anneal over 40 000 (unchanged) |
| loss | eps-MSE, T = 500 cosine betas (unchanged) |
| init | **from scratch** (not fine-tuned from the shipped ckpt — a data fix must not inherit the OOD mode) |
| measured rate | 140 ms/step ⇒ **≈ 1.6 h**, single overnight run |
| checkpoint | **`results/direction3/beamform_cond_diffusion_mixed_v1.pt`** |
| shipped ckpt | `results/direction3/beamform_cond_diffusion.pt` — **read-only, never written** |

**One run. One checkpoint. No restarts, no checkpoint selection, no early stopping.** The held-out
synthetic validation curve is logged for **reporting only** and may not select anything. If training
diverges (NaN), that is reported as the outcome.

---

## 5. GATES — bars written now

All four use the **new** checkpoint against **already-recorded old-net numbers**; the old-net
columns below are quoted from files that existed before this prereg was written.

### G1 — DO-NOT-FORGET (hard bar)
* **Primary set:** the crowded slice of the fresh-40, **16 episodes**
  `data_heldout40_2026-07-26/episode_BF{4,5}_{15,30}comp_seed{777,888}_{uniform,slope}.h5`.
* **Protocol:** `precompute_atoms_heldout40.py` (5 diffusion seeds) → `run_arms_heldout40.py`
  learned arm, 5 paired lanes (42,0)(1,1)(2,2)(3,3)(4,4), `TRACK_PHASE_PREDICT=False`,
  `adaptive_n=True`, 500 init iters; per-episode **median over 5 lanes**, then **mean over episodes**.
* **Old net:** `pure_learned` crowded mean = **0.16878** (`score_results.json`). `pure_v2` = 0.21845.
* **BAR: new-net mean ≤ 0.16878 + 0.010 = `0.17878`.**
* Secondary, run only if time permits and reported either way: the in-sample
  `tmp/wp_integration_2026-07-25/` 40-crowded protocol, old net **0.18092**, bar **≤ 0.19092**.

### G2 — OOD FIX (hard bar, two parts, BOTH must pass)
* **Set:** the **12 fresh 1-comp episodes**
  `data_heldout40_2026-07-26/episode_BF{2,3,4}_1comp_seed{777,888}_{uniform,slope}.h5`.
* **G2a cardinality — BAR: `max` over the 12 episodes of (per-episode median `n_selected`) ≤ 6.**
  Old net: **max = 30** (BF2_1comp 30/19/30/16; BF3/BF4 1-comp already 1–4, so the pooled median of
  3.5 is *not* the discriminating statistic — the max is).
* **G2b accuracy — BAR: new-net learned mean over the 12 ≤ pure-v2 mean + 0.005 = `0.01855`.**
  Old net learned = **0.03567** (v2 = **0.01355**); the whole gap is `BF2_1comp_seed888_slope` at
  0.3286 vs v2 0.0301.

### G3 — CONTINUUM (report only, **no bar** — see §0)
* **Set:** `tmp/discreteness_2026-07-26/data/cell_c`, 10 episodes (`BF4_15comp_seed{10..14}_{uniform,slope}`).
* **Protocol:** `run_cond.py --ckpt <new>` (already supports `--ckpt`), median-of-5 diffusion seeds.
* **Report:** paired per-episode Δ(new − old) with a two-sided Wilcoxon, and the n_selected
  distribution. Old net: cond mean **0.2330**, median 0.2375, n_selected median 22.5, 20 % at the
  `max_n=30` cap. No verdict is issued from G3 alone.

### G4 — REAL TRANSFER (report only, two axes reported **separately**)
* **Accuracy:** `tmp/wp_real_2026-07-25/` protocol, 30 windows (5 records × 6), decision set = the
  **27 NORMAL** windows, metric = lead-0 RMSE / std(pooled GT). Paired **new-net learned vs
  old-net learned**. Old net: learned **0.805**, hand 0.744, learned wins 6/27, p = 0.0046.
* **Collapse:** `tmp/collapse_controls_2026-07-26/` 18 harvested collapse windows (3 design + 15
  held-out). Old net: repairs **8/15** held-out, **damages 7/15**; 6/18 below nRMSE 1.0 vs the
  analytic escape's 11/18. **Report new-net repaired/damaged counts against the same baseline.**
  The "3/3 collapse windows repaired" phrasing is a known-superseded claim and is **not** used as a
  reference here.

### Gate hygiene
Each gate is run **once**. No gate may be re-run after its number is seen. G1 and G2 are the only
gates that can produce a verdict; G3 and G4 are descriptive.

---

## 6. Pre-committed outcome wording

Exactly one branch is written into `RESULT.md` after all four gates are read:

* **FULL PASS** — G1 ≤ 0.17878 **and** G2a ≤ 6 **and** G2b ≤ 0.01855:
  *"The mixed curriculum fixes the single-swell over-proposal OOD without costing the crowded
  regime — the first retraining attempt in this program to pass its pre-registered primary."*
* **PARTIAL** — G2a passes but G2b fails, **or** G2 passes and G1 misses its bar by ≤ 0.02:
  *"The mixed curriculum fixes the proposal's cardinality on sparse seas but does not convert that
  into a parity-level accuracy (or does so at a measurable crowded cost) — a real but incomplete
  data-distribution fix."*
* **FAIL** — G2a fails, **or** G1 misses its bar by > 0.02:
  *"The mixed curriculum does not repair the single-swell OOD; the fourth retraining attempt fails
  on its pre-registered primary, and the training-distribution axis joins architecture and loss as
  a spent card."*

---

## 7. Standing restrictions

* Do **not** commit. Do **not** modify any file under `wavesim/`, `wavesplat/` or `scripts/`.
* Do **not** overwrite `results/direction3/beamform_cond_diffusion.pt` or any existing checkpoint.
* Existing eval harnesses are reused; where one hardcodes the checkpoint path, a **copy** is made
  in `tmp/mixed_retrain_2026-07-26/` with the path parameterised — the original is untouched.
* A PASS on G1/G2 licenses **no** real-data claim; G4 is descriptive and, per the standing
  do-not-cite list, real-data accuracy superiority may not be claimed for the learned arm.

---

## APPENDIX A — post-freeze arm-purity audit (2026-07-27 00:23, added after the coordinator's
## `init_method='auto'` contamination warning; changes NO bar and NO protocol)

Since commit `aa8e4c3` the shipped **default** `init_method` is `'auto'`, which silently routes to
the learned proposal. Any arm that means "pure v2" must therefore set `init_method='v2_konly'`
EXPLICITLY, and any learned arm `'learned'` explicitly. Every runner used by this retrain was
audited:

| runner | uses `run_estimation`? | `init_method` | `TRACK_PHASE_PREDICT` |
|---|---|---|---|
| `run_arms_new.py` (G1, G2) | yes | **explicit** per arm: `'v2_konly'` / `'learned'` / `'auto'` (L62, L64, L82) | **explicit** `su.TRACK_PHASE_PREDICT = False` (L55), set per worker |
| `precompute_new.py` (G1, G2 atoms) | no — calls `propose_atoms` directly | n/a | n/a |
| `run_cond_new.py` (G3) | no — diffusion + polish + field render only | n/a | n/a |
| `run_secondary_new.py` (G3 cardinality) | no — `propose_atoms` + `pool_and_select` | n/a | n/a |
| `run_wp_real_new.py` (G4 accuracy) | no — `mmse_bank_estimate` / diffusion directly | n/a | n/a |
| `run_collapse_new.py` (G4 collapse) | no — diffusion directly | n/a | n/a |
| `mech_diag.py` (diagnostic) | no — `propose_atoms` directly | n/a | n/a |

The **quoted old-net baselines** were audited at their source: `tmp/heldout40_2026-07-26/v2_arm.py`
sets `P1 = {'init_method': 'v2_konly', ...}` explicitly (L32) and `_su.TRACK_PHASE_PREDICT = False`
(L114), so the `pure_v2` references (crowded 0.21845, one-comp 0.01355) are **not** contaminated.

Independent end-to-end confirmation: feeding the **old** atoms cache through the copied
`run_arms_new.py` reproduced the recorded old-net `pure_learned` on all 28 gate episodes at
**max |Δ| = 0.000e+00** — the copied harness is provably the same arm as the source protocol.

---

# ADDENDUM 1 — ROUND 2 (`_mixed_v2`), frozen 2026-07-27 01:20 KST, before any round-2 training step

Authorised by the coordinator's augmented directive (max 3 rounds; one change per round; gates and
bars unchanged; architecture off-limits). **No bar in Sec 5 is altered.**

## Round-1 outcome (all four gates read once)
G1 **PASS** 0.16052 ≤ 0.17878 (old 0.16878, v2 0.21845) · G2b **PASS** 0.01128 ≤ 0.01855
(old 0.03567, and it beats pure v2 0.01355) · **G2a FAIL** max per-episode median n_selected
**12.0** > 6 (old 30) · G3 0.2291 vs 0.2330 · G4 0.797 vs 0.805. Per Sec 6 the frozen wording is
the **FAIL** branch, because G2a failed.

## Diagnosis (measured, `atoms_cache_mixed_v1.jsonl`, no new compute)

The proposal's **placement is now essentially exact** — distance from truth to the nearest
selected atom is **0.0003–0.019 kp** on all twelve 1-comp episodes (old net: 0.2355 kp median on
BF2, with the beamform peak at 0.055 kp). The BF2 catastrophe is gone. What survives is a **count**
problem, and it is monotone in wavenumber:

| sea state | k_true | new n_selected (median over 4 eps) | median selected-atom spread |
|---|---|---|---|
| BF2_1comp | 0.770 | **3.0** | 0.11 kp ≈ 0.085 rad/m |
| BF3_1comp | 0.407 | **3.5** | 0.28 kp ≈ 0.114 rad/m |
| BF4_1comp | 0.252 | **9.0** | 0.55 kp ≈ 0.139 rad/m |

The proposal cloud has **roughly constant ABSOLUTE width (~0.09–0.14 rad/m)** across sea states.
`pool_and_select` dedups on a **constant ABSOLUTE radius** (`dedup_dk = 0.03`) — so at long
wavelengths the same cloud spans proportionally many more dedup cells. This is the same
scale-dependent-constant family already documented for `min_dk = 0.15`
(`project_scale_constants_2026-07-23.md`, whose declared correct fix is an adaptive kp-scaled
radius). It is **not** something the training mixture can reweight away.

**The direct, data-side cause of the inflated count** is that round 1's `replicate_to_M` target
gives every one of the 32 slots the SAME energy (`A/sqrt(count)`, exactly energy-preserving). For a
1-component sea that means all 4 × 32 = 128 proposed atoms clear `pool_and_select`'s
`energy > 0.03 × max` filter, so **dedup is the only thing that can reduce the count** — and dedup
is defeated by the constant-absolute scatter. Nothing in the energy channel encodes cardinality.

## The single change for round 2

**Energy-graded 32-slot target for the discrete strata** (`TARGET_MODE='graded'`):
for `n_true ≤ 32`, the first `n_true` slots carry the true components at **full** amplitude; the
remaining slots repeat those locations at **`A_fill = 0.10 × A_max`** (energy = 1 % of the scene's
strongest component — comfortably under `pool_and_select`'s 3 % threshold). Continuum scenes
(`n_true > 32`) are **unchanged**: all 32 k-means centroids keep their real summed energy, so the
discrete-vs-continuum contrast the net must learn is sharpened rather than blurred.

Declared consequence: the energy channel stops being an exact energy decomposition for sparse seas
(total inflated by up to 31 % when `n_true = 1`) and becomes a **selection/salience** channel. This
is the quantity `pool_and_select` actually consumes, and deployment re-fits amplitudes by LS
(`ls_fit_wphi`), so no downstream quantity depends on the atom energies being exact.

Everything else is **identical to round 1**: same mixture weights, same geometry bank, same
(Hs,Tp) draws, same representability filter, same 40 000 steps / batch 32 / AdamW 2e-4 / cosine,
same master seed 20260726 (so the *scenes* are the same and the target construction is the only
difference), same architecture. Round-1 behaviour remains reproducible via `TARGET_MODE='replicate'`.

**Expected gate to move: G2a** (max per-episode median n_selected 12.0 → ≤ 6), by removing the
low-energy replicas that currently survive the energy filter.
**Risks accepted and reported either way:** G1 (crowded) may degrade if the graded target costs
the net its fine energy structure; G2b may degrade.

Checkpoint: `results/direction3/beamform_cond_diffusion_mixed_v2.pt` (v1 never overwritten).
Gates: the same four, one shot each.

---

# ADDENDUM 2 — ROUND 3 (`_mixed_v3`), frozen 2026-07-27 03:00 KST, before any round-3 training step

Final authorised round (3 of 3). **No bar in Sec 5 is altered.**

## Round-2 outcome (each gate run once)
G1 **PASS** 0.16379 ≤ 0.17878 · G2b **PASS** 0.01092 (best of the three checkpoints; v2-analytic
0.01355) · **G2a FAIL 8.0** > 6 (round 1: 12.0; old net: 30) · G3 0.2304 · G4 0.800 / collapse 8-7.

## Diagnosis
Round 2's energy grading moved the intended quantity and nothing else broke: the number of raw
proposal atoms clearing `pool_and_select`'s `energy > 0.03 × max` filter on the worst 1-comp
episodes fell from 24/31/13 (round 1) to 18/19/15 (round 2), and G2a fell 12 → 8. The lever works;
it was not turned far enough. With `A_fill = 0.10 × A_max` the filler energy sits at **1 % of peak**,
only 3× below the 3 % selection threshold — so the net's own (noisy) energy predictions still push a
substantial share of fillers over the line.

Secondary observation, recorded but **not** acted on: the residual concentrates on the *slope*-raydrop
episodes at long wavelength (BF3/BF4 slope 4/7/8/6 vs their uniform twins 2/2/3/2). The `slope`
model's specular condition reduces to near-normal incidence, which essentially never fires at this
grazing geometry, so it falls back to uniform Bernoulli with a longer-range footprint — and that
footprint is *already* represented in the geometry bank (6 of 14 donors are slope episodes). With one
round left, the measured, already-demonstrated lever is preferred over this un-confirmed one.

## The single change for round 3
**Increase the energy contrast of the graded target** (`TARGET_MODE='graded2'`):

```
A_fill = min(0.03 * A_max, 0.5 * A_min_real)   floored at the representation floor exp(lo_e/2)
```

* `0.03 × A_max` puts filler energy at **0.09 % of peak — a 33× margin** below the 3 % threshold
  (round 2: 3× margin), so a filler must be mispredicted by more than an order of magnitude in
  energy to survive.
* the `0.5 × A_min_real` cap guarantees a filler is **never stronger than the weakest real
  component**, which protects the crowded selection ordering (G1). Verified across regimes:
  nc=1 → 0.03·A_max binds; nc=30 (BF5) → 0.5·A_min binds at 5e-4, still above the
  representation floor, no clamping.

Everything else identical to rounds 1 and 2 (same mixture, geometry bank, (Hs,Tp) draws, 40 000
steps, batch 32, AdamW 2e-4 cosine, master seed 20260726, same architecture).

**Expected to move: G2a** (8.0 → ≤ 6). **Risk:** G1 ordering if the cap misbehaves; G2b.
Checkpoint `results/direction3/beamform_cond_diffusion_mixed_v3.pt`; v1 and v2 never overwritten.

**If round 3 still fails G2a, no round 4 is launched** — the honest failure analysis is written
instead, and the best checkpoint is selected on the frozen gates alone.
