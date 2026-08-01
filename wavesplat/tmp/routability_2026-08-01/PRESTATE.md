# PRE-STATE — routability lower-bound audit (CRUX-A re-instrumentation)

**Frozen 2026-08-01, BEFORE any AUC, N-after-exclusion, or neighborhood count was produced.**
Branch `learn-boundwave-retrain`. Zero new estimator compute: every input is JSON already on
disk. Nothing in this file may be edited after the first number is produced; corrections,
deviations and surprises go in `RESULT.md` §D.

Origin: `docs/PERSONA_DEBATE_novelty_2026-08-01.md` §2-2, P5's prescription —
*"선언된 특징집합에 대한 routability 하한으로 재진술 (관측벡터 전체 × 비선형 분류기의
달성가능 최대 AUC)"*. P4's objection that motivates it: **Spearman |ρ| cannot separate
"unreadable coordinate" from "weakly readable coordinate"**, and |ρ| is a monotone-linear
statistic that is blind to nonlinear or interaction-carried routability.

---

## 0. What this instrument is, and is not

This is **instrument-building, exploratory**. Its output is a *bound with an interval*, not a
verdict. It is pre-committed here that the RESULT.md may **not** contain the sentences
"the boundary is unroutable" or "the boundary is routable" as findings, in any language.
Permitted output shapes are enumerated in §6.

Direction-of-bias, stated in advance so the word "lower bound" is not oversold:

| bias | direction | why |
|---|---|---|
| classifier class is finite (LR + GB only) | **downward** — a better learner may exist ⇒ the measured AUC is a genuine *lower* bound on achievable AUC over the declared features | |
| feature *design* was in-sample | **upward** | `bf_peak_ratio`'s guard threshold and `n_selected`'s router threshold were both chosen by a human looking at these same 100 episodes (`tmp/unified_guard_2026-07-26/explore_beamform.py`, `explore_signals.py`). Cross-validation controls model fitting; it does **not** control feature discovery. |
| n ≈ 90 after dead-zone | **widens CI** | see §5 power note |

Both biases are reported beside every headline number. The net sign is **not** assumed.

---

## 1. Label (frozen)

- Per-episode delta, **identical definition to `tmp/obsaxis_2026-07-26/`**:
  `Δ(ep) = median_of_5_lanes(learned) − median_of_5_lanes(v2)`, lanes
  `[(42,0),(1,1),(2,2),(3,3),(4,4)]`. Negative Δ = learned-init wins.
  Read verbatim from `tmp/obsaxis_2026-07-26/obsaxis_stats.json::synthetic[*].delta`
  (no re-derivation, so the label is byte-identical to the |ρ| version it replaces).
- **Primary label**: `y = 1 if Δ < 0 else 0`, over episodes with `|Δ| > 0.02`.
  Episodes with `|Δ| ≤ 0.02` are **excluded** (dead zone — a coin-flip-magnitude difference
  carries no routing decision worth predicting, and its sign is dominated by lane noise).
  `N` after exclusion is reported; it is **not** known at the time of writing this file.
- **Sensitivity**: the whole pipeline is re-run unchanged at dead zones **0.00** and **0.05**.
  The 0.02 run is the headline; the other two are reported in the same table, and a
  qualitative disagreement between them is a finding to be stated, not a number to be picked.
- Δ is in nRMSE units normalised by `Hs`; 0.02 is ~2 % of a typical Hs-normalised error and
  ~1/5 of the 0.10 sparse-lane bar used elsewhere in the program.

## 2. Feature classes (frozen — no feature is added or removed after the first fit)

### (a) `OBS` — runtime-observable, GT-free at init time

Admission rule, applied before looking at any AUC: a feature is admitted iff it is computable
from **the LiDAR observations of the window alone**, at initialisation time, with no ground-truth
field, no simulator knob, and no completed estimator run.

*Beamform block* — from `tmp/unified_guard_2026-07-26/explore_beamform.json`, produced by
`_beamform_coarse` on the observation point cloud (`wavesplat/optimizer/initializer.py`).
GT-free: the beamform power map is a function of `(x, y, z_obs, t)` only.

| feature | meaning |
|---|---|
| `bf_p2_p1` | 2nd/1st beamform peak power ratio — **this is `bf_peak_ratio`, the 0.257 scalar** |
| `bf_p3_p1` | 3rd/1st peak ratio |
| `bf_p8_p1` | 8th/1st peak ratio |
| `npk_10`, `npk_20`, `npk_30`, `npk_50` | # peaks above 10/20/30/50 % of the dominant peak |
| `n_peaks_found` | # peaks extracted (extraction cap) |

*Proposal block* — from `tmp/unified_guard_2026-07-26/explore_signals.json`, computed from the
learned proposal net's raw atoms (`atoms_cache*.jsonl`) through the repo's `pool_and_select`.
GT-free: the diffusion proposal net consumes the observation-derived beamform power map; no GT
enters. Aggregated to episode level by **median over diffusion seeds 0–4** — the same five
lanes the label uses.

| feature | meaning |
|---|---|
| `n_sel` | # selected atoms — **this is `n_selected`, the 0.372 scalar, and the shipped router's actual input** |
| `kp_sel` | \|k\| of the dominant *selected atom* (proposal-derived; **not** a GT peak wavenumber) |
| `dom_frac`, `top2_frac`, `top3_frac` | energy fraction in the top-1/2/3 selected atoms |
| `nmerge_abs015`, `nmerge_abs0225` | # clusters after single-linkage merge at absolute radius 0.15 / 0.225 |
| `nmerge_kp15`, `nmerge_kp25`, `nmerge_kp40` | same, at radius 0.15/0.25/0.40 × `kp_sel` |
| `n_sel_range` | max−min of `n_sel` across the 5 diffusion seeds (seed-instability of the proposal; GT-free because it only needs the proposal run 5×, which is what the deployed median-of-3/5 path already does) |

Zero-variance features are dropped automatically and the drop is **reported**, not silently
applied.

### Explicitly EXCLUDED from `OBS`, with reasons (pre-committed, so the class cannot be widened post hoc)

| field | in the JSONL | why excluded |
|---|---|---|
| `init_nrmse`, `raw_init_nrmse`, `avg_nRMSE`, `avg_Pearson_rho`, `avg_Skill_Score`, `avg_Variance_Ratio`, `avg_Directional_R2`, `avg_Hs_error` | `arm_runs*.jsonl` | **GT-dependent** — each is an error against the simulated truth field. Admitting any of them makes the audit circular. |
| `final_n` | `arm_runs*.jsonl` | GT-free but **post-run**: only knowable after the learned tracker has finished, so it cannot inform a route decision at init. Recorded here as a deliberate exclusion, not an oversight. |
| `sec` | `arm_runs*.jsonl`, `atoms_cache*.jsonl` | wall-clock; a machine-load artifact, not a physical observable. |
| `regime`, `short`, `episode` | `obsaxis_stats.json` | `regime` is a deterministic relabel of `n_components`; the filename encodes the knob. Both are answer-key leakage. |

### (b) `NCOMP` — the reference ceiling (the answer key)

`n_components ∈ {1,2,5,15,30}`, parsed from the filename. **A simulator knob, unavailable at
runtime.** Given to exactly the same classifiers under exactly the same CV, so the comparison
is like-for-like. This is the "answer key" upper reference against which `OBS` is read.

### (c) declared extras, fitted with the same protocol

- `OBS+NCOMP` — does the knob add anything *on top of* the observables?
- `bf_p2_p1` alone and `n_sel` alone — the two scalars the |ρ| version reported, refitted with
  the same classifier + CV machinery so the AUC and the |ρ| are on one page.

## 3. Classifiers and cross-validation (frozen, no tuning)

- **Linear**: `StandardScaler → LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=5000)`.
- **Nonlinear**: `GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=SEED)`.
  sklearn 1.9.0 is present, so the torch-MLP fallback named in the task is not used.
- **Hyperparameters are fixed here and are NOT tuned.** No grid search, no early stopping on the
  CV metric, no feature selection driven by the CV metric. Any tuning would turn the CV estimate
  into a selection-biased optimum and is therefore forbidden for this audit.
- **CV**: `RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED)` = 100 fits per
  (feature set × classifier). Chosen over LOO because LOO cannot be stratified and gives a
  high-variance AUC at n≈90; 20 repeats average away the fold-partition lottery.
- **Score for scoring**: each episode's score = **mean over the 20 repeats** of its out-of-fold
  predicted probability. AUC is computed once on that averaged out-of-fold score vector.
  The per-repeat AUC spread (sd over 20 repeats) is also reported as a partition-noise diagnostic.
- **SEED = 20260801**, used for the CV splitter, the GB, and the bootstrap.

## 4. Metric and interval (frozen)

- **ROC-AUC** on the averaged out-of-fold scores.
- **95 % CI**: percentile bootstrap, **10 000 resamples**, episode-level resampling with
  replacement, `numpy.random.default_rng(20260801)`. Resamples containing a single class are
  discarded and redrawn (count reported).
- **Gap** = `AUC(NCOMP) − AUC(OBS)`, CI by **paired** bootstrap on the *same* resample indices,
  so the two AUCs share their sampling noise and the gap CI is not inflated.
- **Per-feature marginal AUC**: the raw feature used directly as a score (no fitting, hence no
  CV needed), with the same 10 000-resample bootstrap CI. Reported as the *signed* AUC
  (score = feature value, positive class = learned wins) plus the orientation-free
  `max(AUC, 1−AUC)`. These 18-ish marginals are **multiplicity-uncorrected and descriptive**;
  a Holm-corrected column is reported alongside and any claim about an individual feature must
  cite the corrected column.

## 5. Power note, computed in advance of the data

At n≈90 with roughly balanced classes, the standard error of an AUC near 0.7 is ≈0.055
(Hanley–McNeil), so a 95 % CI is roughly ±0.11. **An AUC of 0.60 and an AUC of 0.50 are not
separable at this n.** This is stated *before* the numbers so that a mid-range AUC cannot be
argued in either direction after the fact. Equivalently: this instrument's minimum detectable
AUC-above-chance is ≈0.11, and the gap it can resolve between two feature sets is ≈0.15.

## 6. How the numbers will be read (frozen wording, evidential not verdictive)

The reading is chosen **before** the numbers; only the row that matches gets written.

| observed | permitted sentence |
|---|---|
| CI of `AUC(NCOMP)` contains 0.5 | *the reference ceiling itself has no dynamic range on this label at this n — the instrument cannot license any statement about the observables, and the audit returns "no measurement", not a result about routability* |
| CI of best `AUC(OBS)` contains 0.5, and `AUC(NCOMP)` CI excludes 0.5 | *over the declared feature class and classifier class, at n=N, there is **no evidence** that the winning arm is predictable from runtime observables, while the simulator knob is; the achievable-AUC lower bound over this class is [lo, hi]* |
| best `AUC(OBS)` CI excludes 0.5 **and** gap CI excludes 0 | *the observables are **partially** readable (lower bound AUC = x, CI [lo,hi]) and measurably worse than the knob (gap g, CI [lo,hi])* |
| best `AUC(OBS)` CI excludes 0.5 **and** gap CI contains 0 | *the observables reach an achievable-AUC lower bound of x, and **no advantage of the knob over the observables is resolvable at this n** (gap CI contains 0)* |
| nonlinear ≫ linear (non-overlapping CIs) | *the |ρ| statistic was blind to routability that a nonlinear reader recovers* — the specific failure P4 predicted |
| nonlinear ≈ linear | *no nonlinear routability beyond the linear reader is detectable at this n* |

In every branch the sentence is prefixed **[MEASURED, in-sample feature design, n=N]** and
followed by the §7 support ceiling. The phrase "phase boundary" is not used at all: this
instrument measures separability of an arm-win label, which is a weaker object.

## 7. Support-mismatch check (frozen)

P4's decisive objection is that the non-readability was measured **outside the deployment
support**. Quantified as follows.

- **Real windows**: the 30 fan-forecast windows, features
  `bf_peak_ratio` (`tmp/figs_opus_real_2026-07-26/bf_ratio_real.json`, cross-checked against
  `tmp/blind_auto_real_2026-07-26/blind_auto_real.json`) and
  `n_sel` = median over diffusion seeds 0–4 of `blind_auto_real.json::rows[*].per_seed[d].n_selected`
  — the same median-over-5-seeds aggregation the synthetic side uses.
- **Neighborhood definitions** (both reported; neither is primary-by-default, both are pre-stated):
  - **N1, 1-D**: `|bf_p2_p1(real) − bf_p2_p1(syn)| ≤ 0.10` (10 % of the [0,1] axis).
    Sensitivity at ±0.05 and ±0.20.
  - **N2, 2-D**: standardised Euclidean distance ≤ 1.0 in the space
    `(bf_p2_p1, n_sel)` z-scored **by the synthetic mean and sd** (so "1 unit" = 1 synthetic sd).
    Sensitivity at radius 0.5 and 2.0.
- **Reported per real window**: the count and fraction of the synthetic *training* episodes
  (post-dead-zone) inside its neighborhood; and the summary — median / min / max fraction,
  and the **number of real windows with zero synthetic neighbours**.
- Also reported, to reproduce the debate's own figure: # synthetic episodes inside the real
  `bf_peak_ratio` IQR band (the "6/100" claim), recomputed from source.
- **Named gap, recorded in advance**: the other ~16 `OBS` features were never computed for the
  30 real windows. Only `bf_p2_p1` and `n_sel` exist on both sides. The support check is
  therefore run in the 2-D subspace that exists on both, and **no proxy is substituted** for the
  missing 16. This means the reported support overlap is an **upper bound** on the true overlap
  in the full 18-D feature space (adding dimensions can only shrink a neighborhood's occupancy).
  That direction is stated in RESULT.md beside the number.

## 8. Prediction recorded before computing (non-binding, for calibration — the `wp_real` template)

Written first, unedited afterwards. I expect `AUC(NCOMP)` ≈ 0.78–0.88 (a 5-level knob that the
delta was designed around, dichotomised); `AUC(OBS)` linear ≈ 0.68–0.78 and nonlinear ≈ 0.70–0.82,
i.e. **materially above chance and materially below the knob** — because the 18-vector contains
`n_sel` and several near-duplicates of it, and 18 features at n≈90 is a regime where GB
over-fits and CV claws most of it back. I expect the gap CI to **exclude 0** but to be wide
(≈0.05–0.25). I expect the support check to be the harshest number in the file: I predict the
**median real window has <10 % of synthetic episodes in its N1 neighborhood** and that at least
a few windows have literally zero neighbours in N2. If `AUC(OBS)` comes back near 0.5 with a CI
covering it, I note in advance that this would be a *stronger* result for P3/P4's reading than
the |ρ| version — and that §0's upward-bias row makes such an outcome harder, not easier, to
obtain, which is why it would be worth something.

I also record the specific way I expect to be wrong: `n_sel` saturates at the `max_n=30` cap,
so if the real windows sit at the cap, the shipped router's own input may be **constant** on the
deployment support — an outcome the |ρ| framing could not have surfaced at all.
