# RESULT — MIXED-CURRICULUM RETRAIN of the beamform-conditioned set-proposal net

Protocol: `PREREG.md` (frozen 2026-07-26 23:55, sha `b151c710…` at freeze) + `ADDENDUM 1` (round 2,
frozen 01:20) + `ADDENDUM 2` (round 3, frozen 03:00) + `APPENDIX A` (post-freeze arm-purity audit).
Decision trail: `LOOP_LOG.md`. Raw numbers: `gate_results_v{1,2,3}.json`.

Retrain attempt **#4** of the WavePrior program, and the **first aimed at measured
training-distribution defects** rather than at architecture or loss. Attempts 1–3
(`prior_net_compdiverse`, `prior_net_fan/fanTF`, `prior_net_e2eft_A/B`) each changed the model or
the objective and each returned a pre-registered FAIL.

---

## 1. What was changed (data only)

The architecture, the diffusion schedule (T=500 cosine betas), the loss (eps-MSE) and the optimiser
(AdamW 2e-4 / wd 1e-4 / cosine anneal, batch 32) are **byte-for-byte the shipped recipe**. Only the
training scene distribution changed.

| axis | shipped checkpoint | this retrain |
|---|---|---|
| **cardinality (D1a)** | always `jonswap_components(n_components=32)` | stratified `nc ∈ {1,2,5,15,30}` + continuum strata, with an exact 32-slot target |
| **sea-state box (D1b)** | `Hs~U[0.8,3.2] × Tp~U[5,11]` — **misses BF2 = (0.3, 4.0) in both coordinates** | 50 % jittered BF2–BF5 anchors (±15 %), 50 % `U[0.25,3.2] × U[3.8,11.5]` |
| **continuum (D2)** | none — every scene was a 32-line JONSWAP discretisation | `gen_cells.expand()` recipe, 465 sub-components rendered into the obs, supervised by a 32-atom energy-weighted k-means quantisation (25 % of the mixture) |
| **obs geometry (D3)** | pooled LiDAR footprint × **statistically independent** `t~U[0,Tobs]` | `GeometryBank` replays real frame-structured sweep windows: 400 points over ~130 distinct frame times, 3.1 pts/time |
| steps | 15 000 | 40 000, from scratch |

**Mixture (frozen, realised within 0.2 % of declaration):** discrete crowded (15/30 comp) 0.30 ·
discrete sparse (1/2 comp) 0.20 · moderate (5 comp) 0.10 · continuum mild 0.10 · continuum strong
0.15 · intermediate spreading 0.15.

**Disjointness.** Training scene seeds are drawn from `[2e9, 2^31)`; every generator seed used by
any evaluation set is `< 2000`, so disjointness is a checkable interval property. The 14
geometry-donor episodes (8 × BF6/BF7 uniform, 6 × BF3/BF4 5-comp seeds 2/3/4 slope) appear in **none**
of the four gate episode lists, and only `(x, y, t)` is ever read from them — `z` is always
re-rendered from the fresh synthetic scene.

---

## 2. Validity controls (all run before any gate number was read)

| control | result |
|---|---|
| harness identity — old atoms cache through the **copied** `run_arms_new.py` vs the recorded old-net numbers, 28 gate episodes | **max \|Δ\| = 0.000e+00** |
| collapse scorer vs the recorded old-net counts | reproduces **8 repairs / 7 damages** held-out, **6/18** below 1.0, analytic escape **11/18** |
| old-net mechanism signature re-measured | BF2 nearest-atom **0.2355 kp** vs beamform peak **0.0550 kp** — the recorded D1 signature |
| generator energy conservation | `\|ΔE\|` < 2e-7 (replicate), < 3e-8 (k-means) |
| geometry-bank footprint vs gate episodes | r-percentiles 10.7/15.6/21.2/31.5/76.7/144.5 vs 10.9/15.5/21.4/32.7/82.4/157.5 |
| arm purity (post-freeze audit, APPENDIX A) | every runner sets `init_method` explicitly and pins `TRACK_PHASE_PREDICT=False`; the quoted `pure_v2` baselines come from `v2_arm.py`, which sets `'v2_konly'` explicitly. **No `auto`-default contamination.** |
| G3/G4 old-net references reproduced live | G4 old learned **0.805**, hand **0.744** — exactly the recorded values |

---

## 3. Rounds run

Three training rounds, each ~70 min on the RTX 4080 Super, each judged **once** against the same
frozen bars. No checkpoint was ever overwritten. No gate was re-run after its number was seen.

| round | checkpoint | single change | wall | final val |
|---|---|---|---|---|
| 1 | `beamform_cond_diffusion_mixed_v1.pt` | the mixed curriculum itself (D1+D2+D3) | 78.5 min | 0.08619 |
| 2 | `beamform_cond_diffusion_mixed_v2.pt` | energy-**graded** 32-slot target (`A_fill = 0.10 × A_max`) | 69 min | 0.08535 |
| 3 | `beamform_cond_diffusion_mixed_v3.pt` | energy contrast ×11 (`A_fill = min(0.03·A_max, 0.5·A_min)`) | 69 min | 0.08558 |

---

## 4. Per-round gate table

Frozen bars: **G1 ≤ 0.17878** · **G2a ≤ 6** · **G2b ≤ 0.01855** · G3, G4 report-only.

| gate | bar | old net | round 1 | round 2 | round 3 |
|---|---|---|---|---|---|
| **G1** crowded mean nRMSE (n=16) | ≤ 0.17878 | 0.16878 | **0.16052 PASS** | **0.16379 PASS** | **0.17491 PASS** |
| **G2a** max per-ep median n_selected (n=12) | ≤ 6 | 30 | **12.0 FAIL** | **8.0 FAIL** | **6.0 PASS** |
| **G2b** 1-comp mean nRMSE (n=12) | ≤ 0.01855 | 0.03567 | **0.01128 PASS** | **0.01092 PASS** | **0.01047 PASS** |
| G3 continuum cell_c mean (n=10) | report | 0.2330 | 0.2291 | 0.2304 | **0.2287** (8/10, p=0.0098) |
| G4 real accuracy, 27 normal | report | 0.805 (hand 0.744) | 0.797 | 0.800 | **0.783** (15/27, p=0.178) |
| G4 collapse, held-out 15 | report | 8 rep / 7 dam | 7 / 8 | 8 / 7 | 8 / 7; **below-1.0 8/18** (old 6/18) |
| **round verdict** | | | FAIL (G2a) | FAIL (G2a) | **FULL PASS** |

Reference: pure-v2 analytic initializer scores **0.21845** on the crowded 16 and **0.01355** on the
1-comp 12.

---

## 5. What moved, and what did not

### 5.1 The targeted defect (D1) is fixed, and the mechanism is identified

The sealed FAIL was `BF2_1comp`: the old net emitted the `max_n=30` cap of atoms sitting far from
the true swell, and the one catastrophic held-out episode scored **0.3286** where the analytic arm
scored 0.0301. After the retrain:

| | old net | round 1 | round 3 |
|---|---|---|---|
| `BF2_1comp` n_selected (4 fresh episodes) | 30 / 19 / 30 / 16 | 3 / 6 / 3 / 2 | 1 / 1 / 2 / 4 |
| `BF2_1comp_seed888_slope` nRMSE | **0.3286** | **0.0174** | **0.0158** |
| 1-comp mean nRMSE (12 eps) | 0.03567 | 0.01128 | **0.01047** |

The mechanism is the decisive part. Distance from the true swell to the nearest **selected** atom,
in kp units (median of 5 diffusion seeds), against the beamform dominant peak — i.e. against the
net's own conditioning input:

| episodes | old net | round 1 | beamform peak |
|---|---|---|---|
| BF2_1comp (k = 0.770) | **0.2355** | **0.0018** | 0.0550 |
| BF4_1comp (k = 0.252) | ~0.002 | **0.0060** | **0.197 – 0.219** |

The relationship **inverted**. The old net was 4× *worse* than the peak of the map it was
conditioned on; the retrained net is **10–50× better** than that peak on every 1-comp episode. The
proposal genuinely super-resolves its conditioner, which is why G2b now beats the analytic
initializer (0.01047 vs 0.01355) — the first time a learned arm has done so on sparse seas.

### 5.2 The residual was a scale-dependent constant, not a data problem

After round 1 the placement was already exact (nearest atom 0.0003–0.019 kp) but the **count** was
not, and it was monotone in wavenumber: BF2 (k=0.770) → 3 atoms, BF3 (0.407) → 3.5, BF4 (0.252) → 9.
The proposal cloud has roughly **constant absolute width (~0.09–0.14 rad/m)** while
`pool_and_select` dedups on a **constant absolute radius** (`dedup_dk = 0.03`), so at long
wavelengths the same cloud spans proportionally more dedup cells. This is the same family as the
already-documented `min_dk = 0.15` scale-dependence (`project_scale_constants_2026-07-23.md`),
whose declared correct fix is an adaptive kp-scaled radius.

Rounds 2 and 3 worked *around* that constant from the data side, by putting cardinality into the
energy channel `pool_and_select` actually thresholds on. Raw atoms clearing the 3 %-of-max filter on
the three worst 1-comp episodes: **24 / 31 / 13** (r1) → **18 / 19 / 15** (r2) → G2a 12 → 8 → **6**.

### 5.3 The cost, stated plainly

**G1 degrades monotonically with the energy-grading strength.**

| | round 1 | round 2 | round 3 | bar |
|---|---|---|---|---|
| crowded mean | 0.16052 | 0.16379 | 0.17491 | ≤ 0.17878 |
| Δ vs old net | **−0.0088** | −0.0050 | **+0.0061** | |
| episodes better than old | 12/16 | 11/16 | **4/16** | |

Round 3 clears the +0.010 non-inferiority bar with margin +0.00387, and still beats the analytic
initializer by a wide margin (0.17491 vs 0.21845) — but sparse-sea cardinality was **bought with
crowded-sea accuracy**, and round 3 sits closest to the bar of the three.

### 5.4 Mean-vs-median discipline (the project's standing caveat applies)

Exploratory, **not a frozen gate** — the full 40 held-out episodes, pure-learned arm
(`full40_exploratory.json`):

| slice | n | v2_konly | old learned | old auto+guard | **v1** | **v3** |
|---|---|---|---|---|---|---|
| all | 40 | 0.09743 | 0.09074 | 0.08577 | **0.08046** | 0.08664 |
| sparse (1,2c) | 16 | 0.01348 | 0.03547 | 0.03190 | 0.01960 | **0.01896** |
| moderate (5c) | 8 | **0.02328** | 0.04518 | 0.02462 | 0.04207 | 0.04548 |
| crowded (15,30c) | 16 | 0.21845 | 0.16878 | 0.17022 | **0.16052** | 0.17491 |

The mean improvements are large, but the **paired** tests are not:
* v1 pure-learned vs old auto+guard: mean 0.08046 vs 0.08577 but medianΔ −0.00054, 21/40, **p = 0.83**
* v1 vs old pure-learned: medianΔ +0.00020, 18/40, **p = 0.54**
* v3 vs old pure-learned: medianΔ +0.00043, 11/40, **p = 0.029 (worse)**
* v3 vs old auto+guard: medianΔ +0.00352, 14/40, **p = 0.010 (worse)**

**The retrain's mean gains come from removing catastrophic failures, not from a broad accuracy
shift.** On the paired median across all 40 the new nets are not better than the old one. This must
not be reported as "the retrained net is more accurate"; it is "the retrained net stops failing
catastrophically on sparse seas, and is non-inferior elsewhere".

**Moderate (5-comp) is the surviving weakness**: 0.042–0.045 for every learned arm vs 0.023 for the
analytic one. The mixture gave it 10 % and that did not fix it.

### 5.5 An operational consequence the gates did not ask about

The deployed router (`auto_switch_thresh = 8` on `n_selected`, plus the `auto_guard_ratio = 0.05`
beamform-peak veto) was tuned against the **old net's pathology**. Episodes the router would send to
the learned arm:

| net | sparse (16) | crowded (16) | all 40 |
|---|---|---|---|
| old | 4 | 14 | 20 |
| v1 | 3 | 16 | 25 |
| **v3** | **0** | 15 | 15 |

With v3 the sparse routing signal is **degenerate — 0/16** — because n_selected is now correctly
small exactly where the router used it as the "learned is trustworthy" cue. The router would send
every sparse sea to the analytic arm even though the learned arm now **beats** it there
(0.01047 vs 0.01355). The guard's entire justification was the BF2 catastrophe, which no longer
exists.

---

## 6. Verdict

### Pre-committed wording (PREREG Sec 6), FULL PASS branch — triggered by round 3
> *"The mixed curriculum fixes the single-swell over-proposal OOD without costing the crowded
> regime — the first retraining attempt in this program to pass its pre-registered primary."*

**Correction to my own pre-committed sentence, required by the measurement:** the clause *"without
costing the crowded regime"* is stronger than what round 3 shows. Round 3 cost **+0.0061** on the
crowded mean relative to the old net while clearing a **+0.010 non-inferiority** bar. The accurate
statement is:

> **The mixed curriculum fixes the single-swell over-proposal OOD and is non-inferior on the crowded
> regime — the first retraining attempt in this program to pass its pre-registered primary — but the
> fix is bought with a measurable, monotone erosion of crowded accuracy, and its mean-level gains
> come from removing catastrophic failures rather than from a broad accuracy improvement.**

**Best checkpoint by the frozen gates: `beamform_cond_diffusion_mixed_v3.pt`** — the only one of the
three to pass G1, G2a and G2b simultaneously.

**Runner-up worth keeping: `beamform_cond_diffusion_mixed_v1.pt`** — fails G2a (12 > 6) but has the
best crowded accuracy (0.16052, better than the old net by −0.0088, 12/16) and the best exploratory
all-40 pure-learned mean (0.08046). Its only failure is a *cardinality* statistic whose bar was set
against a failure mode (the BF2 catastrophe) that v1 already eliminates.

### Recommendation: **do not auto-deploy — this needs a user decision**, because

1. **The gates license v3, but the estimator evidence prefers v1.** v3 is the only full pass; v1 is
   better on crowded accuracy and on the exploratory all-40 mean, and v3 is *significantly worse
   than the old net on the paired median* (p = 0.029) while v1 is not (p = 0.54).
2. **The router must be re-tuned or retired before either is deployed.** Both thresholds were
   calibrated to the old net's sparse catastrophe; with v3 the sparse routing signal is dead (0/16)
   and would route away from the now-better learned arm. Deploying v3 *behind the existing router*
   would discard most of what the retrain bought.
3. **Nothing here licenses a real-data accuracy claim.** G4 improved (0.805 → 0.783) but the learned
   arm still loses to the analytic one (0.744), and the collapse counts are unchanged (8 repairs /
   7 damages held-out; below-1.0 8/18 vs the analytic escape's 11/18). The standing do-not-cite on
   real-data learned superiority is untouched.

### Suggested next step (not run here)
Re-tune the `auto` router against v1 and v3 on the fresh 40 — the cheap, decisive experiment, since
`atoms_cache_mixed_v{1,3}.jsonl` and the full-40 arms already exist. If a re-tuned router (or no
router at all) reproduces v1's 0.08046, the hybrid machinery built for the old net's pathology can
be retired, which is a larger result than either checkpoint.
