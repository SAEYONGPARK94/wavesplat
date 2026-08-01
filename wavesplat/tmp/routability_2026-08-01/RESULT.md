# RESULT — routability lower-bound audit (CRUX-A re-instrumentation)

**2026-08-01.** Branch `learn-boundwave-retrain`. Zero new estimator compute; every input is JSON
already on disk. **No repo code, config or default was modified. Nothing committed.**

| file | what |
|---|---|
| `PRESTATE.md` | the frozen criteria — written and hashed **before** any number here existed. sha256 `9bff2c169b6eff35051281aeaa2302fffd9c000dcfd60cc985fbf458d02d8234` |
| `compute_routability.py` | the analysis, re-runnable and deterministic (`python3 tmp/routability_2026-08-01/compute_routability.py` from the package root) |
| `routability.json` | every raw number, including the per-episode feature table and per-window support counts |
| `run.log` | console transcript of the run that produced `routability.json` |

> ### ⚠ This is instrument-building, exploratory.
> Its product is **a bound and its confidence interval**, not a verdict. It is pre-committed in
> `PRESTATE.md` §0 and repeated here: **this document does not claim "the boundary is unroutable"
> and does not claim "the boundary is routable".** Every headline below is a measured AUC with an
> interval, plus the support ceiling that bounds what any of it can say about deployment.
> Nothing here is a held-out result: the feature *design* was in-sample (see §D-1).

---

## §A. Pre-stated criteria (frozen before computing; full text in `PRESTATE.md`)

This section restates the frozen protocol. It was written first; `PRESTATE.md` carries the
verbatim text and its hash is recorded above so the ordering is checkable.

**Why re-instrument.** `docs/PERSONA_DEBATE_novelty_2026-08-01.md` §2-2 records P4's objection to
the current form of CRUX-A: the identifiability claim rests on two scalar Spearman |ρ| values
(`bf_peak_ratio` 0.257, `n_selected` 0.372) against the simulator knob `n_components` 0.514.
Spearman |ρ| **cannot separate "unreadable coordinate" from "weakly readable coordinate"**, and
being a monotone-linear rank statistic it is **blind to nonlinear or interaction-carried**
routability. P5's prescription: restate as a **routability lower bound over a declared feature
class** — the best achievable AUC of any classifier on the full runtime-observable vector.

**A-1 Label.** `Δ(ep) = median_of_5_lanes(learned) − median_of_5_lanes(v2)`, lanes
`[(42,0),(1,1),(2,2),(3,3),(4,4)]`, read **verbatim** from
`tmp/obsaxis_2026-07-26/obsaxis_stats.json::synthetic[*].delta` so the label is byte-identical to
the |ρ| version it replaces. `y = 1 if Δ < 0` (learned-init wins). **Primary: dead zone
`|Δ| ≤ 0.02` excluded**; sensitivity at 0.00 and 0.05 run unchanged, all three reported.

**A-2 Feature classes.** Admission rule, applied before any AUC was seen: a feature is admitted to
`OBS` iff computable **from the window's LiDAR observations alone, at initialisation time**, with
no GT field, no simulator knob, no completed estimator run.

- `OBS` (19 declared, 18 after the zero-variance drop): *beamform block* `bf_p2_p1` (=`bf_peak_ratio`),
  `bf_p3_p1`, `bf_p8_p1`, `npk_10/20/30/50`, `n_peaks_found`; *proposal block* (median over
  diffusion seeds 0–4, the same five lanes the label uses) `n_sel` (=`n_selected`), `kp_sel`,
  `dom_frac`, `top2_frac`, `top3_frac`, `nmerge_abs015`, `nmerge_abs0225`, `nmerge_kp15`,
  `nmerge_kp25`, `nmerge_kp40`, `n_sel_range`.
- `NCOMP` — the reference ceiling: `n_components` alone, the simulator knob, run through the
  identical classifiers and CV.
- Extras declared in advance: `OBS+NCOMP`, and `bf_p2_p1` alone / `n_sel` alone (the two |ρ| scalars,
  refitted with the same machinery so AUC and |ρ| sit on one page).
- **Excluded with reasons, pre-committed**: all `avg_*`/`init_nrmse`/`raw_init_nrmse` (GT-dependent
  — admitting any makes the audit circular); `final_n` (GT-free but **post-run**, so it cannot
  inform an init-time route); `sec` (wall-clock, a machine artifact); `regime`/filename (relabels
  of the knob).

**A-3 Classifiers, fixed, not tuned.** `StandardScaler→LogisticRegression(C=1, l2, lbfgs)` and
`GradientBoostingClassifier(n_estimators=200, max_depth=2, lr=0.05)`. sklearn 1.9.0 present, so
the torch-MLP fallback was not needed. **No grid search, no early stopping on the CV metric, no
CV-driven feature selection** — any of those would convert the CV estimate into a selection-biased
optimum.

**A-4 CV and metric.** `RepeatedStratifiedKFold(5 splits × 20 repeats)` = 100 fits per cell (chosen
over LOO, which cannot be stratified and is high-variance at n≈90). Each episode's score = mean
out-of-fold probability over the 20 repeats; **ROC-AUC** computed once on that vector.
**95 % CI = percentile bootstrap, 10 000 episode-level resamples, `default_rng(20260801)`**,
single-class draws discarded. **Gaps use paired bootstrap on the same resample indices.**
Per-feature marginal AUCs use the raw feature as a score (no fitting) with the same bootstrap,
reported with a Holm-corrected p column; individual-feature claims must cite the corrected column.

**A-5 Power, stated before the data.** At n≈90 the SE of an AUC near 0.7 is ≈0.055, so a 95 % CI is
≈±0.11: **AUC 0.60 and AUC 0.50 are not separable at this n**, and the resolvable gap between two
feature sets is ≈0.15.

**A-6 Reading rules, frozen** (`PRESTATE.md` §6, evidential wording only — the matching row gets
written and no other). Reproduced in §E with the row that fired marked.

**A-7 Support-mismatch protocol.** N1 (1-D): `|Δbf_p2_p1| ≤ 0.10`, sensitivity ±0.05/±0.20.
N2 (2-D): standardised Euclidean ≤ 1.0 in `(bf_p2_p1, n_sel)` z-scored **by synthetic mean/sd**,
sensitivity 0.5/2.0. Reported per real window plus median/min/max and zero-neighbour count.

**A-8 Calibration prediction, recorded before computing** (`PRESTATE.md` §8, unedited): I predicted
`AUC(NCOMP)` ≈ 0.78–0.88, `AUC(OBS)` linear ≈ 0.68–0.78 / nonlinear ≈ 0.70–0.82, i.e. observables
**materially below** the knob, with the gap CI **excluding 0**. I also recorded one specific way I
expected to be wrong: *"`n_sel` saturates at the `max_n=30` cap, so if the real windows sit at the
cap, the shipped router's own input may be **constant** on the deployment support."*
**Scored in §D-0: the central prediction was wrong in sign; the recorded failure mode was right.**

---

## §B. MEASURED — the bound

### B-1 Label after exclusion

| dead zone | N | learned wins | v2 wins | sparse (1–2c) | moderate (5c) | crowded (15–30c) |
|---|---|---|---|---|---|---|
| 0.00 | **100** | 65 | 35 | 40 (24 L / 16 v2) | 20 (5 L / 15 v2) | 40 (36 L / 4 v2) |
| **0.02 (primary)** | **60** | **41** | **19** | 19 (6 L / 13 v2) | 9 (3 L / 6 v2) | **32 (32 L / 0 v2)** |
| 0.05 | **30** | 25 | 5 | 6 (1 L / 5 v2) | 1 (1 L / 0) | 23 (23 L / 0 v2) |

`n_peaks_found` was dropped at every dead zone as **zero-variance** (constant 30 — it is the
extraction cap, not a measurement). 18 observables remain.

### B-2 Headline AUCs — primary dead zone 0.02, N=60 **[MEASURED]**

| feature set | # feats | **linear (LogReg)** | **nonlinear (GradBoost)** |
|---|---|---|---|
| **`OBS`** — runtime-observable vector | 18 | **0.8806** [0.7831, 0.9560] | **0.9641** [0.9025, 1.0000] |
| **`NCOMP`** — simulator knob (reference ceiling) | 1 | 0.8331 [0.7145, 0.9358] | **0.9191** [0.8409, 0.9756] |
| `OBS+NCOMP` | 19 | 0.9512 [0.8949, 0.9897] | 0.9756 [0.9332, 1.0000] |
| `bf_p2_p1` alone — *the 0.257 scalar* | 1 | 0.5597 [0.3950, 0.7261] | 0.5353 [0.3347, 0.7372] |
| `n_sel` alone — *the 0.372 scalar, and the shipped router's input* | 1 | 0.6791 [0.4749, 0.8680] | 0.8626 [0.7552, 0.9512] |

**The four deliverable numbers and the gap:**

| quantity | value | 95 % CI |
|---|---|---|
| **AUC_observables (linear)** | **0.881** | [0.783, 0.956] |
| **AUC_observables (nonlinear)** | **0.964** | [0.903, 1.000] |
| **AUC_ncomp reference** (best of the two classifiers) | **0.919** | [0.841, 0.976] |
| **Gap = AUC_ncomp − best AUC_observables** | **−0.045** | **[−0.123, +0.018]** — contains 0 |
| nonlinear − linear, on `OBS` (paired) | +0.083 | [−0.003, +0.176] — just contains 0 |

### B-3 Dead-zone sensitivity **[MEASURED]**

| dead zone | N | `OBS` linear | `OBS` nonlinear | `NCOMP` best | gap (ncomp − best obs) |
|---|---|---|---|---|---|
| 0.00 | 100 | 0.7846 [0.683, 0.877] | **0.8734** [0.793, 0.942] | 0.7530 [0.630, 0.862] | **−0.120 [−0.223, −0.029]** ← excludes 0 |
| **0.02** | 60 | 0.8806 [0.783, 0.956] | **0.9641** [0.903, 1.000] | 0.9191 [0.841, 0.976] | −0.045 [−0.123, +0.018] |
| 0.05 | 30 | 1.0000 [1.000, 1.000] | 1.0000 [1.000, 1.000] | 0.9600 [0.870, 1.000] | −0.040 [−0.130, 0.000] |

nonlinear − linear on `OBS`: **+0.089 [+0.019, +0.165]** at dz 0.00 (**excludes 0**),
+0.083 [−0.003, +0.176] at 0.02, +0.000 at 0.05 (both saturated).

The three dead zones **agree in sign and ordering** (`OBS` ≥ `NCOMP` throughout; nonlinear ≥ linear
throughout). They disagree in *resolution*, and the reason is structural, not statistical — see
**§D-2**, which is the most important caveat in this document: **the dead zone is selection on the
outcome magnitude, and |Δ| is correlated with regime**, so raising it drives the label towards a
relabel of `n_components` (at dz 0.05, 23/23 crowded episodes are learned-wins). dz 0.05 is
degenerate (5 minority cases, AUC pinned at 1.000) and carries no information.

### B-4 Per-feature marginal AUCs, dead zone 0.02 **[MEASURED; Holm-corrected column is the citable one]**

Signed AUC uses the raw feature as the score for "learned wins"; `orient` = `max(AUC, 1−AUC)`.

| feature | AUC | orient | 95 % CI | p_Holm | GT-free? |
|---|---|---|---|---|---|
| **`n_sel_range`** (proposal seed-instability) | 0.8646 | **0.865** | [0.761, 0.948] | **1.1e-04** ✓ | yes |
| *`n_components`* (the knob — reference, not a candidate, excluded from Holm) | 0.8614 | *0.861* | [0.757, 0.950] | — | **no** |
| **`kp_sel`** (\|k\| of dominant selected atom; inverted) | 0.1861 | **0.814** | [0.077, 0.308] | **1.8e-03** ✓ | yes |
| **`bf_p8_p1`** | 0.7882 | **0.788** | [0.654, 0.903] | **5.9e-03** ✓ | yes |
| **`nmerge_kp40`** | 0.7625 | **0.763** | [0.604, 0.897] | **1.7e-02** ✓ | yes |
| `n_sel` — *the 0.372 scalar* | 0.6938 | 0.694 | [0.490, 0.879] | 0.231 | yes |
| `top3_frac` / `dom_frac` / `top2_frac` | 0.306 / 0.308 / 0.309 | 0.694 / 0.692 / 0.691 | ≈[0.12, 0.51] | 0.231 | yes |
| `nmerge_kp25` / `nmerge_kp15` | 0.684 / 0.680 | 0.684 / 0.680 | ≈[0.48, 0.87] | 0.231 / 0.232 | yes |
| `bf_p3_p1` | 0.6329 | 0.633 | [0.471, 0.785] | 0.813 | yes |
| `npk_10` | 0.6162 | 0.616 | [0.466, 0.761] | 0.969 | yes |
| **`bf_p2_p1` — *the 0.257 scalar*** | 0.6072 | **0.607** | [0.435, 0.772] | **1.00** | yes |
| `nmerge_abs015`, `npk_20`, `nmerge_abs0225`, `npk_50`, `npk_30` | 0.46–0.56 | 0.52–0.56 | — | 1.00 | yes |

At dead zone 0.00 the ordering is the same at the top and flatter overall: `n_sel_range` 0.742
[0.637, 0.835] (p_Holm 1.2e-03, **the only observable surviving Holm**), `n_sel` 0.633,
`n_components` 0.614, **`bf_p2_p1` 0.516**.

**The single most consequential line in this table:** the two scalars on which CRUX-A was
built — `bf_peak_ratio` and `n_selected` — are **the weakest and the sixth-weakest members of the
declared observable vector**, while the strongest single runtime observable (`n_sel_range`,
the spread of the proposal's atom count across diffusion seeds) marginally **matches or exceeds
the simulator knob** (0.865 vs 0.861 at dz 0.02; 0.742 vs 0.614 at dz 0.00). [MEASURED]

### B-5 Validity diagnostics **[POST-HOC — not in `PRESTATE.md`; see §D-3]**

*Label-permutation null* (30 permutations, whole CV+scoring pipeline re-run each time). If the
pipeline leaked, permuted AUC would sit above 0.5:

| cell | permuted AUC mean | sd | max over 30 |
|---|---|---|---|
| `OBS` \| logreg | 0.490 | 0.098 | 0.646 |
| `OBS` \| gboost | 0.499 | 0.132 | 0.737 |
| `NCOMP` \| logreg | 0.385 | 0.132 | 0.569 |
| `NCOMP` \| gboost | 0.424 | 0.128 | 0.639 |

`OBS` centres on chance and its worst permutation (0.737) is far below the observed 0.964 ⇒
**the 18-feature/n=60 result is not a leak of the CV machinery.** (`NCOMP` centring *below* 0.5 is
the expected small-sample artifact of a single heavily-tied discrete feature under stratified CV,
where in-fold majority direction anti-correlates out-of-fold; it does not affect the observed
value, which is far above.)

*Training-data provenance* — checked because `n_sel`-derived features come through the learned
proposal net. The net behind these atom caches is `results/direction3/beamform_cond_diffusion.pt`
(2026-07-03), trained by `scripts/d3_beamform_cond_diffusion.py::make_batch` on **freshly drawn
random JONSWAP spectra generated on the fly**, not from any `data/` HDF5. The scored
`data_finemesh_2026-07-23/` episodes **did not exist** when the checkpoint was written. So the
observable features carry **no memorised `n_components`**. Two residual exposures are recorded in
§D-5 rather than waved away.

---

## §C. Support mismatch — what the bound can say about deployment **[MEASURED]**

Reference set = the 60 post-dead-zone synthetic episodes. Real set = the 30 fan-forecast windows
(5 records × 6). Only **2 of the 18** observables exist on both sides (`bf_p2_p1`, `n_sel`); the
other 16 were never computed on real windows (**named gap**, §D-4).

| | synthetic (n=60) | real (n=30) |
|---|---|---|
| `bf_p2_p1` range | 0.024 – 0.955 | 0.108 – 0.998 |
| `bf_p2_p1` mean ± sd | 0.219 ± 0.190 | — (IQR **0.505 – 0.952**) |
| `n_sel` (median of 5 seeds) range | — | 21 – 30 |
| **`n_sel` at the `max_n = 30` cap** | **6 / 60 (10 %)** | **22 / 30 (73 %)** |

**Synthetic episodes inside the real `bf_peak_ratio` IQR [0.505, 0.952]: 4 / 60 = 6.7 %** — an
independent re-derivation of the debate's "6/100" figure on the post-dead-zone set.

Neighborhood occupancy, per real window, as a fraction of the 60 synthetic episodes:

| neighborhood | median | min | max | **real windows with ZERO synthetic neighbours** |
|---|---|---|---|---|
| N1, \|Δbf\| ≤ 0.05 | 1.7 % | 0.0 % | 40.0 % | **2 / 30** |
| **N1, \|Δbf\| ≤ 0.10 (pre-stated primary)** | **3.3 %** | 1.7 % | 65.0 % | 0 / 30 |
| N1, \|Δbf\| ≤ 0.20 | 3.3 % | 3.3 % | 90.0 % | 0 / 30 |
| N2, 2-D, ≤ 0.5 synthetic sd | 0.0 % | 0.0 % | 13.3 % | **24 / 30** |
| **N2, 2-D, ≤ 1.0 synthetic sd (pre-stated primary)** | **0.0 %** | 0.0 % | 20.0 % | **21 / 30** |
| N2, 2-D, ≤ 2.0 synthetic sd | 3.3 % | 3.3 % | 56.7 % | 0 / 30 |

**The median real window has 3.3 % of the synthetic corpus (2 of 60 episodes) within ±0.10 of its
`bf_peak_ratio`, and 0 % within one synthetic standard deviation in the 2-D
`(bf_peak_ratio, n_sel)` space. 21 of 30 real windows have no synthetic neighbour at all at that
radius.** All six `YS17` windows and all six `LJ18g` windows are in the zero-neighbour set, at
2-D distances of 1.8–1.9 sd from the nearest synthetic episode. Because adding dimensions can only
shrink occupancy, **these are upper bounds** on the true overlap in the full 18-D space.

**Deployment-support evaluability [POST-HOC]:** only 4 of the 60 episodes lie inside the real
`bf_peak_ratio` IQR. **No cross-validated, deployment-relevant AUC can be estimated from this
corpus at all** — there is no split of 4 episodes that supports it. The bound in §B is a
bound *on the synthetic grid*, and it is silent about the real windows by construction, not by
oversight.

**The recorded-failure-mode hit:** 73 % of real windows sit at the `n_sel` cap of 30 versus 10 %
of synthetic. `n_selected` — the shipped router's own and only input — is **near-constant on the
deployment support**, which is a property the |ρ| framing could not have surfaced and which
`PRESTATE.md` §8 named in advance. [MEASURED]

### C-1 What the shipped rule actually achieves **[POST-HOC, in-sample, descriptive]**

The shipped route rule is: take `learned` iff `n_selected ≥ 8`, then veto back to `v2` iff
`bf_peak_ratio < 0.05`. A binary rule has no ROC curve, so its single operating point is reported
as sensitivity/specificity; balanced accuracy equals the AUC of that rule used as a 0/1 score.
Evaluated at episode level on the same median-of-5 aggregation as the label.

| rule | dead zone | sens | spec | **bal. acc** | TP / FP / TN / FN |
|---|---|---|---|---|---|
| router only (`n_sel ≥ 8`) | 0.02 | 0.854 | 0.684 | **0.769** | 35 / 6 / 13 / 6 |
| **router + guard (shipped)** | **0.02** | 0.829 | 0.895 | **0.862** | 34 / 2 / 17 / 7 |
| router only | 0.00 | 0.615 | 0.743 | 0.679 | 40 / 9 / 26 / 25 |
| router + guard (shipped) | 0.00 | 0.600 | 0.886 | 0.743 | 39 / 4 / 31 / 26 |

**Both thresholds were fitted on these same episodes, so this is an in-sample, optimistic
operating point** — and it still sits **below** the out-of-fold bound of §B (0.862 vs 0.964 at
dz 0.02; 0.743 vs 0.873 at dz 0.00). The guard is doing real work at both dead zones (specificity
+0.21 / +0.14 for ≈0.02 of sensitivity). **[POST-HOC, in-sample, descriptive — this is not a
held-out evaluation of the shipped router and must not be quoted as one.]** The gap between the
in-sample rule and the out-of-fold bound is **not** a clean "headroom" estimate, because the two
numbers carry opposite biases; it is reported as a bracket, not a delta.

---

## §D. Deviations, surprises, named gaps

**D-0 — The pre-registered calibration prediction was wrong in sign, and this is recorded, not
edited.** `PRESTATE.md` §8 predicted `AUC(OBS)` **materially below** `AUC(NCOMP)` with a gap CI
excluding 0. Measured: `AUC(OBS)` is **above** `AUC(NCOMP)` at all three dead zones, and at dz 0.00
the gap CI excludes 0 *in the opposite direction to the prediction*. The prediction file is left
unedited. The one thing I predicted correctly was the specific failure mode (`n_sel` saturation on
real support), which is now §C's sharpest number.

**D-1 — Feature design was in-sample; CV does not fix that.** `bf_peak_ratio`'s guard threshold
(0.05) and `n_selected`'s router threshold (8) were both chosen by a human looking at these same
100 episodes (`tmp/unified_guard_2026-07-26/explore_beamform.py`, `explore_signals.py`), and the
whole `OBS` vector comes from those two exploration scripts. Cross-validation controls *model*
fitting; it does not control *feature discovery*. **This biases §B upward.** Direction was
pre-stated (`PRESTATE.md` §0) and is repeated wherever a §B number is quoted. Counterweight, also
pre-stated: only two classifier families were tried, which biases §B *downward* — that is the sense
in which §B is a **lower** bound on achievable AUC over the declared class. The two biases are not
assumed to cancel.

**D-2 — ⚠ The primary dead zone conditions the label onto the regime axis. This was not
anticipated in `PRESTATE.md` and is the strongest internal caveat here.** Excluding `|Δ| ≤ 0.02`
removes 40 of 100 episodes, but not uniformly: it removes 21 of 40 sparse and 11 of 20 moderate
episodes versus only 8 of 40 crowded ones, and among the survivors **all 32 crowded episodes are
learned-wins (32/32)**. The dz-0.02 task is therefore close to *"is this episode crowded?"* — which
is why `AUC(NCOMP)` jumps from 0.574 (linear, dz 0.00) to 0.833 (linear, dz 0.02). The dead zone was
chosen for a defensible reason (a coin-flip-magnitude Δ carries no routing decision) but it is
**selection on the outcome**, and it inflates *both* columns. **Consequence for reading: the
dz-0.00 row of §B-3 is the one that answers the CRUX-A question with an uncontaminated label**, and
it is the row where the observables beat the knob with a CI excluding 0. The dz-0.02 headline is
reported as primary because it was pre-stated, not because it is the most informative.

**D-3 — Two post-hoc analyses were added after the first numbers**, and are labelled POST-HOC
everywhere they appear: the label-permutation null (§B-5) and the shipped-rule operating point
(§C-1), plus the deployment-evaluability count (§C). The permutation null was added because an AUC
of 0.964 from 18 features at n=60 obliges a demonstration that the pipeline does not leak; it is a
validity check on the instrument, not a test of any hypothesis. None of them changes a pre-stated
number.

**D-4 — Named gap: 16 of the 18 observables do not exist for the 30 real windows.** Only
`bf_peak_ratio` (`tmp/figs_opus_real_2026-07-26/bf_ratio_real.json`) and `n_selected`
(`tmp/blind_auto_real_2026-07-26/blind_auto_real.json`, per diffusion seed) were ever computed on
real data. Recomputing the other 16 needs the beamform peak-list and the proposal atoms re-run on
each real window — cheap in principle (no GT needed), **out of scope for a zero-compute audit**.
**No proxy was substituted.** The §C numbers are therefore run in the 2-D subspace that exists on
both sides, and are upper bounds on the full-space overlap.

**D-5 — Two residual provenance exposures, recorded rather than dismissed.** (i) The proposal net's
training batches drew LiDAR *(x, y)* positions from six `data/` episodes whose *names* also appear
in the scored set (`scripts/d40_learned_acquisition.py::pool_xy`); only positions were reused, with
times drawn i.i.d. and elevations re-rendered from fresh synthetic spectra, so no sea-state
information transferred. (ii) The checkpoint's *development-time* evaluation used three episodes
that are also in the scored 100 — a **model-selection** exposure on 3/100, not a training-data one.
Neither is gradient leakage; both are stated so a reader can discount §B accordingly.

**D-6 — `n_sel_range` shares its five draws with the label.** The top marginal observable is the
spread of `n_selected` across diffusion seeds 0–4 — the same five draws that produce the
`median_of_5` learned score in Δ. This is **not** GT leakage, and it is deployment-faithful (the
shipped median-of-3 path already runs the proposal several times, so the spread is free at
runtime). But feature and label are functions of one shared set of proposal draws, so a
lucky/unlucky draw moves both. A 3-seed range would be a noisier feature than the 5-seed one
measured here. **[ARGUED]**

**D-7 — dz 0.05 is degenerate** (5 minority cases; `OBS` AUC pinned at exactly 1.000 with a
zero-width bootstrap CI). It is reported for completeness as pre-stated and should not be quoted.

**D-8 — The pre-stated reading table did not anticipate a negative gap.** `PRESTATE.md` §6 row 3
was written as *"partially readable **and measurably worse than** the knob"*, silently assuming the
gap's sign. The measured sign is the opposite. Rather than rewrite the frozen row, the outcome is
reported in §E under the row that matches on *structure* (gap CI excludes/contains 0) with the sign
stated explicitly. Recorded as a defect in the pre-registration, to be fixed in the next one.

---

## §E. What this licenses — the frozen reading rules, with the rows that fired

Applying `PRESTATE.md` §6 mechanically:

- Row 1 (*reference has no dynamic range*) — **did not fire.** `AUC(NCOMP)` CI is
  [0.841, 0.976] at dz 0.02 and [0.630, 0.862] at dz 0.00, both excluding 0.5. The instrument has
  dynamic range and the audit returns a measurement.
- Row 2 (*no evidence observables are readable*) — **did not fire.** Best `AUC(OBS)` CI excludes
  0.5 at every dead zone.
- Row 4 (*bound reached, knob advantage not resolvable*) — **fired at dz 0.02 (primary):**

  > **[MEASURED, in-sample feature design, n=60]** Over the declared 18-feature runtime-observable
  > class and the declared classifier class, the achievable-AUC **lower bound** for predicting
  > which arm wins is **0.964, CI [0.903, 1.000]** (nonlinear) / **0.881, CI [0.783, 0.956]**
  > (linear), and **no advantage of the simulator knob over the observables is resolvable at this
  > n** (gap −0.045, CI [−0.123, +0.018], contains 0).

- Row 3's structural condition (*gap CI excludes 0*) — **fired at dz 0.00, with the opposite sign
  to the one the frozen row assumed (§D-8):**

  > **[MEASURED, in-sample feature design, n=100]** With no outcome-based exclusion, the
  > observable vector reaches AUC **0.873, CI [0.793, 0.942]**, the simulator knob reaches
  > **0.753, CI [0.630, 0.862]**, and the signed gap **`AUC(knob) − AUC(observables)` = −0.120,
  > CI [−0.223, −0.029], which excludes 0 on the negative side** — i.e. the runtime-observable
  > coordinate orders this label *better* than the simulator knob does, not worse.

- Row 5 (*|ρ| was blind to nonlinear routability*) — **fired at dz 0.00:**

  > **[MEASURED]** The nonlinear reader beats the linear reader on the same observables by
  > **+0.089, CI [+0.019, +0.165]** (excludes 0). At dz 0.02 the same difference is +0.083 with a
  > CI that just contains 0 ([−0.003, +0.176]). This is the specific failure P4 predicted for a
  > monotone rank statistic: **part of the routability in these observables is not visible to
  > Spearman |ρ| at all.**

**Every one of the above is bounded by §C**, which is pre-committed to be quoted alongside:

> the bound is measured on a synthetic grid of which only **6.7 %** lies inside the real windows'
> `bf_peak_ratio` IQR; **21 of 30** real windows have **no** synthetic neighbour within one
> standard deviation in the 2-D observable space; and **no deployment-relevant AUC is estimable
> from this corpus at all** (4 in-support episodes). The bound describes the synthetic arena. It
> makes no claim about real deployment.

**Not licensed, explicitly:** that the boundary is routable; that the boundary is unroutable; that
a router built on these observables would work on real data; that `bf_peak_ratio` is a bad signal
*in general* (it is weak **on this label, on this grid**, where its marginal AUC is 0.607 with
p_Holm = 1.00); that any single feature in §B-4 outside the four Holm-surviving rows carries
signal. The word "phase boundary" is not used: this instrument measures separability of an
arm-win label, a weaker object than a phase diagram.

---

## §F. What this does to CRUX-A

**[ARGUED — this section is interpretation, not measurement.]**

The |ρ| version of CRUX-A said: the boundary is sharp on the simulator knob (0.514) and flat on the
runtime observables (0.257, 0.372), therefore *the phase diagram is drawn on a coordinate the
runtime cannot read.* The routability restatement **does not support that inference, and at dz 0.00
it measures the opposite ordering.** Three specific things changed:

1. **The flatness was a property of the two chosen scalars, not of the observable class.** Both are
   near the bottom of the 18-feature ranking (§B-4). The class as a whole reaches 0.87–0.96.
2. **Part of the signal is nonlinear**, so |ρ| was structurally incapable of seeing it (§E row 5) —
   exactly P4's objection, now measured rather than asserted.
3. **The binding constraint moved.** It is no longer "the runtime cannot read the coordinate"; it is
   **"the coordinate has barely been varied where the runtime lives"** (§C). That is the *wide*
   sense of crux F, which `tmp/obsaxis_2026-07-26/RESULT.md` already flagged as unanswered — and it
   is now quantified: median 0 % 2-D neighbour occupancy, 21/30 zero-neighbour windows, `n_selected`
   at its cap for 73 % of real windows versus 10 % of synthetic.

So the restatement **weakens the CRUX-A claim as currently written** — the strongest form ("a
coordinate the runtime cannot read") is not supported by this instrument and is contradicted in
sign at dz 0.00 — while **strengthening the underlying research position**, because the surviving
statement is sharper, harder to attack, and about something the program can actually fix: the
synthetic grid was never varied over the region where the sensor operates. Per the debate's own
prohibition list, **"식별가능성을 |ρ| 2개로 진술" remains forbidden**, and the replacement wording
is not "unroutable" but the §E blocks above with the §C ceiling attached.

Suggested next instrument (**not** run here): recompute the 16 missing observables on the 30 real
windows (GT-free, no new estimator arms needed) so the support check leaves the 2-D subspace; and
generate synthetic episodes *in* the real `bf_peak_ratio` band, which is the only way any of this
becomes a statement about deployment.
