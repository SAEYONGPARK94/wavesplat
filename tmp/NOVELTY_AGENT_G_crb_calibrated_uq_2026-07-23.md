# Agent G — CRB-anchored / inefficiency-calibrated runtime UQ: GO/NO-GO
**2026-07-23.** Axis assigned = the one un-probed synthesis after the 10 eliminated lenses:
*use σ_CRB (sensor-geometry, runtime-computable, no ground truth) as the **anchor** of a **calibrated
absolute** per-location predictive error bar, with an empirically-estimated **inefficiency factor** α
doing the calibration.*
Prereg (frozen before numbers): `tmp/AGENT_G_PREREG_crb_calibrated_uq_2026-07-23.md`.
Code: `wavesplat/tmp/g_uq_collect.py` (raw), `g_uq_analyze.py` (Route 1), `g_uq_verify2.py` (Route 2),
`g_uq_conformal.py` (scoring-rule robustness).
Raw: `results/direction3/g_uq_raw_S.npz`, `g_uq_raw_S_oracle.npz`, `g_uq_raw_H.npz`;
gate: `g_uq_gate_S.json`, `g_uq_gate_H.json`.

---

## 0. TL;DR VERDICT — **NO-GO** for the calibrated-absolute-uncertainty claim (b).
**(a) σ_CRB as a ranking / skill-map signal survives** — but that is the project's existing envelope spine
(E-58/E-59/C8), not new, and my measurements reproduce it — within-scene spread–skill Spearman
**0.90 on synthetic (0.99 under-det / 0.66 well-det) and 1.00 on HOS**. σ_CRB is an excellent *orderer*.
**(b) σ_CRB as a calibrated absolute error bar does NOT clear its pre-registered gate** — decisively on the
powered synthetic arm, and as a *tie* (not a win) on the nonlinear HOS arm. Two independent analysis routes,
one clean mechanism.
1. **[BOTH ARMS — the decisive, metric-free criterion] The single global inefficiency factor does not transfer.** Per-episode α = √⟨e²/σ_CRB²⟩ spans
   **11.5 → 228 (19.9×, CV 0.69)** across 32 synthetic episodes and **6.4 → 30.8 (4.80×, CV 0.44)** across
   12 HOS seas — versus 2.7×/1.7× for a plain constant bar and 3.8×/3.7× for the empirical (ridge-posterior)
   bar. The pre-registered NO-GO threshold was 3×. **Triggered in both arms.**
2. **[Arm S] A σ_CRB-anchored bar is WORSE than a global constant bar.** LOSO paired ΔNLL(const − CRB) =
   **−0.256 nats, 95% CI [−0.364, −0.101], win-fraction 0.16/32**; Route-2 repeated unit split-half
   **−0.386 [−0.617, −0.211], 0/200 reps favour CRB**. G1 fails.
3. **[Arm S] It is also worse than the empirical-UQ bar** (ΔNLL(emp − CRB) = −0.185 [−0.300, −0.037]). G3 fails —
   and `emp` here *is* the already-published construct (Fucile 2018 LEPrE / Rodgers optimal-estimation
   posterior covariance), so the CRB-specific part is not merely un-winning, it is behind the prior art.
4. **Mechanism (MEASURED):** σ_CRB **over-disperses — its log dynamic range is ~6× too wide**. Pooled
   log–log regression of RMS error on the anchor gives slope **+0.158** (ideal 1.0): the anchor sweeps
   **100.7×** while the actual error sweeps only **2.2×**. Across episodes, mean σ_CRB varies **59.5×** while actual RMS error varies **2.70×**, and
   Spearman(per-episode mean σ_CRB, per-episode RMS error) = **+0.177, p = 0.33 (n = 32, n.s.)** — i.e.
   **σ_CRB carries within-episode *shape* information but essentially no transferable *level* information.**
   A multiplicative calibration cannot repair a slope of 0.16.
5. **The obvious rescue is not enough and is not novel.** A 2-parameter power calibration
   σ̂ = α·σ_CRB^β (exploratory, NOT pre-registered) picks β ≈ **0.15–0.35** and then beats the constant bar
   by only **+0.039 nats [+0.024, +0.046]**, while **tying** the empirical bar (+0.057, CI [−0.012,+0.212],
   underpowered). Two global parameters fitted on residuals *is* Levi-2022 variance scaling / normalized
   conformal prediction — textbook, and the gain is negligible.

6. **The pincer (§4.1b).** Re-running with an *oracle* (matched) wavenumber basis, α **does** transfer
   (2.04× spread, CV 0.18) and the CRB bar beats a constant bar 200/200 — **but it then loses to the
   posterior-variance bar** (−0.145 nats, 0/200), because under a matched linear-Gaussian model that bar is
   the *exact* predictive variance and the CRB is merely a bound. **Matched model ⇒ use LEPrE, not the CRB.
   Mis-specified model ⇒ nothing calibrates.** There is no regime where a CRB-anchored bar is the right tool.
7. **Real-data corroboration (§4.3).** Re-reading the project's own E-59/F-5 raw stereo artefact as a
   calibration statement: σ_CRB sweeps **43–216×** across deciles while realised error sweeps only
   **4.7–7.1×** — log–log slope **0.34**. The same over-dispersion, on real data, already in the repo.

8. **HOS (Arm H, 12 nonlinear seas, fitted θ̂, raycast LiDAR operator — the headline-eligible arm) is
   milder and I will not over-read it.** There the CRB bar is not clearly *worse* than a constant bar
   (ΔNLL(const−crb) = −0.079 [−0.212,+0.021], **UNDERPOWERED, no direction label**; Route 2 −0.151, 1 % win
   rate) and it is a **dead-exact tie with the prior-art empirical bar** (−0.004 [−0.029,+0.036]). But the
   pre-registered NO-GO criterion still fires (**α transfer 4.80× > 3×**) and under conformal it is **33 %
   wider** than a constant interval. **Tie, not win.**
9. **One live thread, NOT pre-registered, UNDERPOWERED.** On HOS a *hybrid* — runtime residual level ×
   within-scene-normalised σ_CRB shape — beats a constant bar (+0.067 nats, 200/200 reps) and a
   runtime-level-only bar (+0.021 / +0.049, 99 %). It nevertheless **ties the same hybrid built on the
   empirical/LEPrE anchor** and is still 12.6 % wider under conformal. n=12; SE > effect/2 ⇒ **no direction
   label**. If anyone wants to revive this axis, that hybrid — not the pre-registered one-global-α form — is
   the only thing worth powering up, and it would have to beat LEPrE, which it currently does not.

⟹ **(b) collapses into (a).** What is defensible is exactly what the project already claims: σ_CRB *ranks*
where error will be large, most usefully in the under-determined regime. It does **not** deliver a
shippable calibrated ±bar.

**Scope honesty:** Arm S (32 synthetic episodes) is the powered arm and gives a strong negative; Arm H
(12 HOS seas) is the headline-eligible arm and gives a tie. The overall NO-GO rests on the criterion that is
met in **both** arms and is metric-free: the inefficiency factor does not transfer (19.9× / 4.80×, both > 3×).
([[feedback_validation_nonlinearity]] satisfied: the headline arm is HOS.)

---

## 1. Literature verification (MANDATORY, done before any novelty claim)

### 1.1 The one that actually kills it — analytic prediction-error std for phase-resolved waves
| Ref | URL | What it actually claims | Bearing |
|---|---|---|---|
| **Fucile, Bulian, Lugni, *Ocean Engineering* 163 (2018) 718–736 — "A probabilistic approach for the quantification of prediction error in deterministic phase-resolved wave forecasting"** | https://www.sciencedirect.com/science/article/abs/pii/S0029801818305900 | Semi-analytical **Linear Estimator of Prediction Error (LEPrE)**: closed-form **prediction-error standard deviation in space and time**, given the fitting procedure, the sea spectrum, and the measurement-error level. Wave elevation = Gaussian process; coefficients from linear fitting on measurements. Explicitly framed as "supplementing deterministic predictions with a sound prediction error measure" and defines a **"Predictability Region" in a consistent probabilistic framework**. | **Closest pre-emption; near-fatal for (b).** An analytic, ground-truth-free, per-space-time error *standard deviation* for exactly this problem already exists and is already used to draw predictability regions. The only wedges left are (i) LEPrE is estimator-specific (linear LS) rather than a bound over all unbiased estimators, (ii) it needs the sea spectrum, (iii) it is not (from the abstract/thesis material) *empirically recalibrated* against realised errors under model mismatch. **My gate shows the CRB version is worse than the LEPrE-analogue anyway** (ΔNLL(emp−CRB) < 0), so those wedges buy nothing. |
| Fucile PhD thesis 2017, U. Trieste (open) | https://arts.units.it/retrieve/e2913fdb-8e45-f688-e053-3705fe0a67e0/…F_Fucile_PhD_Thesis_2017.pdf | Full derivation of LEPrE, incl. ensemble variance of the error process and measurement-noise contribution; analytic treatment of the actual spectrum for long- and short-crested seas. (⚠ fetch returned 403 in this session — cited from the ScienceDirect abstract + indexed excerpts, **ARGUED not fully read**.) | Same. |

### 1.2 Adjacent wave/marine UQ — already found by the team, verified here
| Ref | URL | What it actually claims | Bearing |
|---|---|---|---|
| Zhang, Zhao, Jin, Greaves, *Applied Energy* 324 (2022) 119711 | https://www.sciencedirect.com/science/article/pii/S0306261922010042 (OA copy: https://wrap.warwick.ac.uk/id/eprint/168456/) | Variational-Bayes ML phase-resolved wave prediction with **quantified uncertainty (aleatoric + model)**; predicts the **predictable zone without assuming linear sea states**. Probe time series, wave-tank validated. | Pre-empts "phase-resolved prediction + UQ + predictable zone". Estimator-*dependent* posterior. (⚠ I could not machine-read the PDF for coverage/PIT metrics → whether they test *calibration* is **UNVERIFIED**.) |
| Alkarem, Huguenard, Kimball, Grilli, *JMSE* 13 (2025) 1250 | https://doi.org/10.3390/jmse13071250 | TiDE/LSTM phase-resolved forecasting under data sparsity; **τ-trimming** adapts the prediction horizon from **uncertainty thresholds derived from historical forecasts**. | Pre-empts "uncertainty-gated operability horizon". Uncertainty is **empirical/historical**, not analytic. (⚠ MDPI page 403'd; from indexed abstract — **ARGUED**.) |
| Nielsen et al. 2024b (ship-response sea-state estimation) | see e.g. https://www.sciencedirect.com/science/article/pii/S0029801824028816 | UQ for ML wave-spectrum estimation with a **"trust level indicator" that identifies small-error estimates without knowing ground truth**. | Direct pre-emption of the *ranking* function (a). Not spatial, not analytic. |
| Kim et al., arXiv:2412.05475 (AI digital twin of the ocean) | https://arxiv.org/abs/2412.05475 | LSTM + **deep ensemble** UQ for real-time **scalar** wave height; "**uncertainty calibration** … increasing uncertainty quality by over 50 % through simple calibration technique". | Pre-empts "recalibrated UQ for real-time wave prediction" — but scalar, data-driven, no analytic anchor. |
| ⛔ **arXiv 2511.21856** | — | **MISATTRIBUTED in Round 1** — it is a review of phase-averaged/phase-resolving wave *models*, not UQ. | **Do not cite for the UQ gap.** I found no paper that asserts the gap; the gap must be argued from Fucile/Zhao scope, not from a citation. |

### 1.3 Method-level pre-emptions (the "α" idea is textbook in three separate fields)
| Ref | URL | What it claims | Bearing |
|---|---|---|---|
| Levi, Gispan, Giladi, Fetaya (2022) — **variance/σ scaling** recalibration for regression | e.g. https://arxiv.org/html/2508.17761 (review), ENCE metric | Learn **one scalar** s so that s·σ̂(x) is calibrated ("temperature scaling for variances"). | **This is literally the proposed α.** The construction is standard ML-UQ recalibration. |
| Papadopoulos et al. (2008, 2011) — **normalized / locally-adaptive conformal prediction** | https://arxiv.org/pdf/2005.07972 , https://arxiv.org/pdf/2306.04648 | Nonconformity scores normalized by a **difficulty estimator** σ(x); one global quantile then gives *finite-sample valid* intervals. | Same structure (analytic difficulty proxy × one global multiplier) **with a coverage guarantee my version lacks**. If anything, plugging σ_CRB into split-conformal is strictly better engineering — and strictly less novel. |
| X. Rong Li & Zhao — **estimator credibility**; Bar-Shalom & Li — **NEES / ANEES covariance-consistency** | https://scispace.com/pdf/estimator-s-credibility-and-its-measures-474hn1goes.pdf ; https://kalman-filter.com/normalized-estimation-error-squared/ | Formal machinery for "is the *reported* covariance consistent with the *realised* errors", with the CRLB as the reference. | Pre-empts the whole "bound vs realised error, calibrated?" question in tracking/estimation. |
| Geodesy/GNSS — **a posteriori variance factor / Helmert variance-component estimation** | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7038394/ ; https://jgps.springeropen.com/articles/10.1186/s41445-016-0005-5 | Formal least-squares covariance is known to be optimistic; it is **rescaled by an empirically estimated variance factor** to make reported errors match reality. | **This is exactly "inefficiency-calibrated analytic error bars", 50 years old, in a different field.** |
| Rodgers optimal estimation, and its calibration literature | https://www.mdpi.com/2072-4292/11/23/2770 ; https://repository.library.noaa.gov/view/noaa/31068/noaa_31068_DS1.pdf | OE retrievals ship an analytic **a posteriori error covariance** per retrieval; a sizeable literature *tunes* it against validation data (e.g. correcting a ~20 % underestimate). | Pre-empts "analytic Fisher-type per-retrieval error bar, empirically corrected against truth" in remote sensing. |
| GNSS/aviation **protection levels** (RAIM/ARAIM overbounding) | https://doi.org/10.3390/rs12121992 ; https://navi.ion.org/content/72/2/navi.697 | A per-epoch error bound computed from **satellite geometry**, with the overbounding Gaussian variance **multiplied by an inflation factor**, certified by a coverage/integrity-risk requirement. | The single closest *structural* analogue: geometry-derived bound × empirical inflation factor, validated by coverage. Fifty years of prior art on exactly this construction (in another field). |
| NWP **spread–skill / rank-histogram** verification | https://journals.ametsoc.org/view/journals/mwre/144/1/mwr-d-14-00312.1.xml ; https://arxiv.org/html/2512.02160 | Calibration = spread must match RMSE **in magnitude** *and* **track it**; ratio 1 ideal, >1 over-dispersed. | The evaluation framework is standard; using it is not a contribution. **My measured failure is precisely a magnitude/scale-consistency failure.** |

### 1.4 Searched-for and NOT found
No paper was found that names or claims "CRB-anchored calibrated predictive uncertainty", "bound-calibrated
UQ", or "efficiency-calibrated error bars" as a contribution — in ocean/wave sensing, DOA/array processing,
or radar/sonar. DOA/CRB work uses the bound as a **benchmark**, and array "calibration" there means *sensor
gain/phase* calibration, a different sense of the word. So the **exact phrase-level combination is
unoccupied** — but every component is occupied, and (decisively) **the empirical gate says the combination
does not work**, so the vacancy is not worth claiming.

---

## 2. (a) vs (b): does the calibrated claim collapse into the ranking claim?

| | (a) ranking / skill map | (b) calibrated absolute bar |
|---|---|---|
| What it needs | monotone association only | correct **level** *and* correct **dynamic range** |
| Project status before this gate | E-58 §8 PARTIAL (ρ≈0.30 pooled; 0.31 well-det → 0.78 under-det), E-59 real level-ρ 0.89–1.00, C8 cross-estimator (scope-limited) | untested |
| **This gate (MEASURED)** | reproduced: within-episode spread–skill ρ **0.93–0.99** (under-det 0.99) | **fails**: level does not transfer (α 19.9×), dispersion 6× too wide (slope 0.158), loses to a constant bar |

**Answer: yes, (b) collapses into (a).** Note this *refines rather than contradicts* E-59: E-59's
"level-calibration" (level-ρ 0.89–1.00, real error rising 4.7–6.0× across σ_CRB deciles) is a **within-record
monotone-shape** result, and E-59 itself reported that the real dynamic range (4.7–6.0×) is *compressed*
relative to the twin (≈10–40×) — that compression **is** the slope-<1 over-dispersion I measure directly.
The two findings are the same phenomenon seen from two sides.

---

## 3. Gate design (pre-registered decision rule — stated before the numbers; see prereg file)

Bars, all given **identical** calibration machinery (one global scale α = √⟨e²/s²⟩ fit on **training units
only**, evaluated leave-one-unit-out):
`const` (s=1) · **`crb`** (s=σ_CRB) · `emp` (s=ridge posterior std ≈ LEPrE analogue) · `dist` ·
`const_r` (s = runtime CV residual level, per unit, **no GT**) · `crb_r` (runtime level × within-unit-normalised CRB shape) ·
`emp_r` · ORACLE refs `persea`, `crboracle` (α refit on the held-out unit — needs GT).
Primary metric = **Gaussian NLL** (proper scoring rule); secondary coverage@68/95, z-std, PIT-KS,
spread–skill ρ and log–log slope.

**GO required ALL of:** G1 median ΔNLL(const−crb) > 0 with CI excluding 0 and win-frac ≥ 0.70 ·
G2 coverage in [0.62,0.74]@68 and [0.90,0.98]@95 · G3 ΔNLL(emp−crb) ≥ 0 (CI∋0 ⇒ downgrade to CONDITIONAL,
because emp ≡ prior art) · G4 sign positive in **both** regimes · G5 reproduced on HOS.
**NO-GO if** ΔNLL(const−crb) ≤ 0, **or** per-unit α spread > 3×.
**UNDERPOWERED (no label) if** bootstrap SE > |median|/2.

---

## 4. Measured results

### 4.1 Arm S — 32 synthetic episodes (BF3/BF4, 5/15/30-comp, stratified), R=8 phase realizations, 409 600 rows
Oracle-θ CRB, matched linear model class. Per-unit LOSO; median over units.

| bar | NLL ↓ | cov68 | cov95 | z-std | PIT-KS | spread–skill ρ | log-log slope | α |
|---|---|---|---|---|---|---|---|---|
| const | **0.675** | 0.754 | 0.938 | 1.007 | 0.068 | — | — | 0.47 |
| **crb** | 1.009 | 0.887 | 0.991 | 0.645 | 0.131 | 0.903 | 0.592 | 109.7 |
| emp (LEPrE analogue) | 0.695 | 0.869 | 0.976 | 0.795 | 0.110 | 0.970 | 0.522 | 13.9 |
| dist | 2.742 | 0.988 | 0.995 | 0.397 | 0.383 | 0.915 | 0.180 | 0.92 |
| const_r (runtime level, no GT) | 0.718 | 0.788 | 0.956 | 0.889 | 0.086 | 0.381 | 0.414 | 4.63 |
| crb_r (runtime level × CRB shape) | 0.833 | 0.904 | 0.993 | 0.607 | 0.144 | 0.933 | 0.596 | 6.83 |
| emp_r | 0.848 | 0.856 | 0.974 | 0.825 | 0.104 | 0.988 | 0.525 | 6.27 |
| crbpow *(exploratory, 2 params)* | 0.639 | 0.768 | 0.954 | 0.913 | 0.073 | 0.903 | 3.661 | 0.99 |
| persea *(ORACLE)* | 0.675 | 0.737 | 0.938 | 1.000 | 0.050 | — | — | 0.48 |
| crboracle *(ORACLE α)* | 0.520 | 0.736 | 0.943 | 1.000 | 0.035 | 0.903 | 0.592 | 70.8 |

Read the two ORACLE rows together: giving the CRB bar the *held-out episode's own* α buys 0.49 nats
(1.009→0.520) and makes it the best bar of all — **the shape is fine, the transferable level is the whole
problem.**

Paired ΔNLL (positive ⇒ the CRB-side bar wins), median [95 % bootstrap CI] / win-frac:

| comparison | pooled (n=32) | under-det (n=16) | well-det (n=16) |
|---|---|---|---|
| **const − crb (G1)** | **−0.256 [−0.364,−0.101] / 0.16** | +0.076 [−0.099,+0.169] / 0.62 **UNDERPOWERED** | −0.184 [−0.248,−0.082] / 0.19 |
| **emp − crb (G3)** | **−0.185 [−0.300,−0.037] / 0.28** | +0.021 [−0.166,+0.083] **UNDERPOWERED** | −0.074 [−0.325,+0.174] **UNDERPOWERED** |
| const_r − crb_r | −0.110 [−0.166,−0.005] / 0.31 | −0.115 [−0.241,−0.054] | −0.083 [−0.095,−0.019] |
| const_r − crb | −0.247 [−0.330,−0.009] / 0.34 | +0.092 **UNDERPOWERED** | −0.046 **UNDERPOWERED** |
| dist − crb | +1.99 [+1.72,+2.21] / 1.00 | +2.33 | +1.80 |
| const − crbpow *(expl.)* | +0.039 [+0.024,+0.046] / 0.75 | +0.108 [+0.060,+0.115] | +0.006 **UNDERPOWERED** |
| emp − crbpow *(expl.)* | +0.057 [−0.012,+0.212] **UNDERPOWERED** | −0.000 **UNDERPOWERED** | +0.088 **UNDERPOWERED** |
| crboracle − crb *(GT-α gap)* | −0.318 [−0.559,−0.164] / 0.00 | −0.065 | −0.237 |

**α transfer:** pooled 11.5→228 (**19.9×**, CV 0.69); under-det 6.9×; well-det 3.4×. Pre-registered
NO-GO bar (>3×) exceeded **in every slice**. Exploratory β (power calibration) = 0.15 pooled,
0.35 under-det, 0.25 well-det — all far from the 1.0 that a pure-scale calibration assumes.
**Permutation control** (σ_CRB shuffled within unit): ΔNLL(const−crb) degrades further, −0.256 → −0.509 —
so the within-episode spatial signal **is real**; it is simply not worth its dispersion cost.
**Regime reading:** the CRB bar is *not* uniformly useless — in the under-determined regime it draws level
with the constant bar (+0.076, CI∋0, **UNDERPOWERED — no direction label**) and its spread–skill slope is
near-ideal (0.94). It is the well-determined regime, and above all the cross-episode level, that kill it.

### 4.1b ★ The pincer — matched-model control (`--oracle_basis`, same 32 episodes)
Objection to the negative: *"α fails to transfer only because your estimator is mis-specified (non-oracle
k-grid); with a matched model the bound would be tight and α would transfer."* **Tested. The objection is
correct — and it closes the other jaw of the pincer.** Re-running Arm S with the ridge basis set to the
**true wavenumbers** (`results/direction3/g_uq_raw_S_oracle.npz`, Route-2 code):

| | α median | α max/min | CV | ΔNLL vs crb (200 split-halves) |
|---|---|---|---|---|
| crb | 0.441 | **2.04×** | 0.18 | — |
| emp | 1.088 | 2.76× | 0.29 | **−0.145 [−0.194,−0.068], 0/200 favour CRB** |
| const | 0.002 | 25.7× | 1.32 | +1.208 [+0.640,+4.00], 200/200 favour CRB |

So: **matched model ⇒ α transfers (2.0×) and the CRB bar crushes a constant bar — but it still loses to the
posterior-variance bar (= Fucile LEPrE / Rodgers OE), which under a matched linear-Gaussian model is the
*exact* predictive variance while the CRB is only a bound. Mis-specified model (the realistic case) ⇒ α does
not transfer at all (19.9×) and the CRB bar loses to everything.** There is no regime in which the
CRB-anchored bar is the right tool.
⚠ Honest caveat on this control: with an oracle basis the estimator is *given* the wavenumbers, so it solves
an easier problem than the CRB (which treats k as unknown) — hence α = 0.44 < 1, i.e. the realised error is
*below* the "bound". This arm is a matched-model illustration, **not** a measurement of estimator efficiency,
and must not be cited as "the estimator is super-efficient".

### 4.2 Robustness to the scoring rule — normalized split-conformal (rebuts threat T2)
`wavesplat/tmp/g_uq_conformal.py`, 200 unit split-halves, nominal 90 %. Conformal makes coverage valid **by
construction**, so the bars compete only on **interval width** — the metric that cannot be accused of
punishing over-dispersion unfairly.

| anchor | achieved coverage | mean interval half-width (Hs units) |
|---|---|---|
| **const** | 0.901 | **0.767** ← narrowest |
| crb_shape (runtime level × CRB shape) | 0.898 | 0.969 (1.26× wider) |
| emp (LEPrE analogue) | 0.901 | 1.069 (1.39× wider) |
| **crb** | 0.896 | **2.153 (2.81× wider)** |

**The CRB-anchored conformal interval is ~2.8× wider than a constant-width interval at the same coverage.**
The negative is therefore *not* an artefact of the Gaussian NLL.

### 4.3 ★ Third confirmation, on REAL data, from the project's own raw artefact
`results/direction3/f5_level_calibration.json` (E-59 / F-5 real stereo, AA15a·AA14·YS17) stores the
σ_CRB-decile medians. Nobody had read it as a *calibration* statement. Re-reading it that way:

| record | E-59 level-ρ | σ_CRB decile range | actual-error decile range | **log–log slope** |
|---|---|---|---|---|
| AA15a | 1.000 | 144.6× | 5.89× | **0.366** |
| AA14 | 0.915 | 183.7× | 6.00× | **0.314** |
| YS17 | 0.891 | 43.2× | 4.66× | **0.385** |
| pooled | 1.000 | 215.8× | 7.05× | **0.343** |
| TWIN (synthetic, model-class matched — best case) | 0.96–1.00 | 88–300× | 7.7–28.8× | 0.36–0.72 |

**The same over-dispersion appears on real stereo data, at the same magnitude** (slope ≈ 0.34 vs my
synthetic 0.16–0.59). E-59 reported this as the "level-calibration RESCUE"; read as an absolute-bar
statement it is the opposite — an σ_CRB bar scaled to be right in the lowest decile would be wrong by
roughly **25×** in the highest. **E-59's level result is a ranking success and a magnitude failure**, and it
independently corroborates the Arm-S negative on real data. (MEASURED, re-derived here; the underlying
numbers were already MEASURED and independently re-derived by fable in F-5.)

### 4.4 Arm H — HOS nonlinear seas, fitted θ̂ (the headline-eligible arm)
12 HOS seas (Tp 6–9 s × Hs 1.2–4 m, steepness-filtered), 6 observation draws each, raycast LiDAR operator
with raydrop, RELAX-fitted θ̂ (**non-oracle**), truth = the actual nonlinear HOS surface. 201 600 rows,
~73 min GPU. `results/direction3/g_uq_raw_H.npz`, `g_uq_gate_H.json`.

| bar | NLL ↓ | cov68 | cov95 | z-std | PIT-KS | ss ρ | log-log slope | α |
|---|---|---|---|---|---|---|---|---|
| **const** | **−0.074** | 0.743 | 0.949 | 0.950 | 0.057 | — | — | 0.24 |
| emp_r | −0.072 | 0.724 | 0.937 | 1.019 | 0.038 | 0.915 | 1.171 | 1.74 |
| const_r | −0.043 | 0.728 | 0.934 | 1.023 | 0.043 | 0.200 | −0.094 | 1.78 |
| crb_r | −0.033 | 0.712 | 0.953 | 0.963 | 0.031 | 1.000 | 0.525 | 2.00 |
| emp | +0.017 | 0.773 | 0.966 | 0.878 | 0.090 | 0.976 | 1.824 | 38.5 |
| **crb** | **+0.024** | 0.760 | 0.973 | 0.865 | 0.104 | 1.000 | 0.539 | 17.1 |
| dist | +0.764 | 0.937 | 0.981 | 0.830 | 0.224 | 0.964 | 0.140 | 0.15 |
| crbpow *(expl.)* | −0.135 | 0.734 | 0.959 | 0.928 | 0.069 | 1.000 | 1.558 | 0.98 |
| crboracle *(ORACLE α)* | −0.097 | 0.698 | 0.945 | 1.000 | 0.016 | 1.000 | 0.539 | 14.8 |

| comparison | Route 1 (LOSO, n=12) | Route 2 (200–300 unit split-halves) |
|---|---|---|
| **const − crb (G1)** | **−0.079 [−0.212,+0.021] / win 0.33 — UNDERPOWERED** | **−0.151 [−0.408,−0.008], 1 % of reps favour CRB** |
| **emp − crb (G3)** | **−0.004 [−0.029,+0.036] — exact TIE** | **+0.000 [−0.068,+0.035] — exact TIE** |
| const − crb_r | +0.069 [−0.003,+0.142] / 0.67 | **+0.067 [+0.016,+0.122], 100 % of reps** |
| **const_r − crb_r** *(does the CRB **shape** add over a runtime level?)* | **+0.021 [+0.012,+0.110] / 0.92** | **+0.049 [+0.010,+0.083], 99 % of reps** |
| **emp_r − crb_r** *(shape vs prior-art shape)* | **+0.0001 [−0.012,+0.075] — TIE** | +0.021 [−0.017,+0.052], 87 % — TIE/marginal |
| dist − crb | +0.829 [+0.682,+0.887] / 1.00 | — |
| const − crbpow *(expl.)* | +0.050 [+0.009,+0.080] / 0.75 | — |
| crboracle − crb *(GT-α gap)* | −0.144 [−0.327,−0.020] / 0.00 | — |

**α transfer on HOS: 6.42 → 30.78 = 4.80×, CV 0.44** — better than Arm S's 19.9× but still **above the
pre-registered 3× NO-GO threshold** (well-det slice 2.96×, under-det 4.80×).
**Conformal (nominal 90 %, `g_uq_conformal.py`):** const 0.387 (narrowest) · crb_shape 0.436 (+12.6 %) ·
emp 0.437 (+13.1 %) · **crb 0.516 (+33.3 %)**.
**Permutation control:** ΔNLL(const−crb) −0.079 → −0.333 when σ_CRB is shuffled within sea; the shape signal
is real on HOS too (spread–skill ρ = 1.000).

**Arm H verdict — honest and mixed, and I will not over-read it.**
- The **pre-registered form fails**: G1 is not met (median negative, CI∋0 in Route 1; 1 % win-rate in
  Route 2), G3 is a **dead-exact tie with the prior-art bar**, the α-transfer NO-GO criterion is met
  (4.80 × > 3 ×), and under conformal the CRB bar is 33 % less efficient than a constant bar.
- The **one live thread**, dual-verified and clearly labelled **not pre-registered**: a bar built as
  *runtime-CV-residual level × within-scene-normalised σ_CRB shape* (`crb_r`) beats a global constant bar by
  **+0.067 nats (200/200 reps)** and beats a runtime-level-only bar by **+0.021 (Route 1) / +0.049
  (Route 2, 99 %)**. But it **ties the same construction built on the empirical/LEPrE anchor**
  (+0.0001 Route 1; +0.021, 87 % Route 2) and is still **12.6 % wider than a constant interval** under
  conformal. Effect sizes ~0.02–0.05 nats on n=12 seas; **by the SE > effect/2 power gate this thread is
  UNDERPOWERED and receives NO DIRECTION LABEL.**
- Arm H is therefore **milder than Arm S** — on nonlinear seas the CRB bar is *not* clearly worse than a
  constant bar, it is *indistinguishable* from it and from prior art. That is a tie, not a win, and it is
  what the discipline requires me to report rather than the more dramatic synthetic negative.

### 4.5 Why α does not transfer (interpretation — **ARGUED**, consistent with the repo record)
α ≈ 70–110 means the realised error sits **two orders of magnitude above** the bound. That is not
"estimator inefficiency" in the classical sense; it is dominated by **model mis-specification and
non-convex wavenumber search**. The project already measured this directly: in
`tmp/WAVENUMBER_ID_CRB_RESULT_2026-07-22.md`, the true wavenumber set fits the observations to the noise
floor (residual 0.018 vs σ 0.02) while RELAX's recovered set leaves 0.094 — a 5× search failure — and E-52
frames the same quantity as the achievability gap. **Mis-specification error does not scale with the Fisher
geometry**, so multiplying σ_CRB by any single constant cannot track it; the constant is really absorbing a
per-episode search failure, and search failure varies far more (and differently) than information geometry
does. This also predicts, before the fact, that the failure should be **worse** on HOS, where the linear
component model is additionally wrong about the physics.

### 4.6 Dual verification (required)
- **Route 1** = `g_uq_analyze.py` (LOSO, 32 units).
- **Route 2** = `g_uq_verify2.py`, NumPy-only, no import of Route 1, **different resampling** (200 repeated
  random unit split-halves).
  *Arm S:* const NLL 0.670 ± 0.041 < emp 0.807 ± 0.074 < **crb 1.056 ± 0.147**; ΔNLL(const−crb) =
  **−0.386 [−0.617,−0.211], 0/200 reps favour CRB**; α census reproduces 19.94×. **CONFIRM.**
  *Arm H:* ΔNLL(const−crb) = −0.151 [−0.408,−0.008], 1 % of reps favour CRB; ΔNLL(emp−crb) = +0.000
  (exact tie); α census reproduces 4.80× / CV 0.44; the hybrid `crb_shape` beats `const_r` +0.049
  [+0.010,+0.083] (99 %) but only ties `emp_shape` +0.021 [−0.017,+0.052] (87 %). **CONFIRM (both the
  negative and the small live thread).**
- **Scoring-rule route** = `g_uq_conformal.py` (coverage valid by construction, competes on width): const
  narrowest in **both** arms.
- Background numbers I re-derived from the canonical artefacts rather than trusting the memory index:
  `results/direction3/e58_envelope_validation.json` → CRB↔actual **+0.329**, emp **+0.332**, dist **+0.317**
  (n=24) — matches E-58's "emp-UQ ties CRB in every regime". `results/direction3/hos_s8_envelope.json` →
  HOS CRB↔actual **+0.296**, emp **+0.206**, 8/8 seas CRB>emp — matches HOS_S8_RESULT.

---

## 5. Threats to validity (symmetric audit — including threats to the *negative*)
**T1 (against the negative).** Arm S computes σ_CRB at the **oracle θ**, which is the *most favourable*
setting; it still loses. Arm H uses fitted θ̂ (honest). So the negative is not an artefact of a bad θ̂.
**T2 (against the negative) — ADDRESSED.** Gaussian NLL punishes over-dispersion hard, so I re-ran the
comparison as normalized split-conformal (§4.2), where coverage holds *by construction*. The CRB anchor
still loses — by **2.81× in interval width** at matched 90 % coverage. This also settles the "just use
conformal" escape hatch: it works, but it is Papadopoulos et al.'s method, not a contribution, and its
*efficiency* is governed by exactly the dispersion mismatch measured here.
**T2c (against the negative).** Arm S uses a **non-oracle k-grid**, so the estimator is mis-specified.
§4.1b tests the matched-model alternative explicitly and shows it merely relocates the problem (CRB then
loses to the exact posterior variance). Both jaws measured.
**T2b (against the negative — and it makes the negative stronger, not weaker).** My `emp` bar is
√(Φ_q A⁻¹ Φ_qᵀ)·σ — the ridge posterior predictive std from **measurement noise only**. Fucile's LEPrE also
carries the **ensemble variance of the un-modelled part of the spectrum**, which is the dominant term here.
So `emp` is a *weaker* proxy than the actual prior art, and the CRB bar still loses to it. Symmetrically,
this is the theoretical reason the CRB level cannot transfer: g_qᵀF⁻¹g_q is the estimation error of the
*assumed* component set and is structurally blind to un-modelled spectrum content and to search failure —
precisely the terms that dominate realised error.
**T3 (for the negative).** The α-transfer failure (19.9×) is metric-free: it is a direct property of
⟨e²/σ_CRB²⟩ and does not depend on NLL at all.
**T4.** Error ensemble in Arm S = phase realizations; in Arm H = observation draws against one HOS truth per
sea. Coverage is therefore *conditional on the truth realization* in Arm H. n_seas is the effective n.
**T5.** Sandbox operator (synthetic obs = predict + Gaussian noise) in Arm S; Arm H uses the raycast LiDAR
operator with raydrop. Neither uses a real moving LiDAR — the project-wide caveat stands.
**T6.** `crbpow` β was selected on training units per fold (honest), but the β grid and the whole 2-parameter
variant were **not pre-registered** ⇒ EXPLORATORY, must not be cited as a result.
**T7.** Power: 32 episodes (Arm S) / 12 seas (Arm H). Every comparison whose SE exceeds half the effect is
flagged UNDERPOWERED above and carries **no direction label**, per discipline. In particular the entire
Arm-H head-to-head (const vs crb) and the whole `crb_r` live thread are UNDERPOWERED; the only Arm-H
criterion I treat as decided is the α-transfer diagnostic, which is a direct property of the data and not a
hypothesis test.
**T8 (against the negative).** Arm H spans only Tp 6–9 s (kp 0.053–0.110) because the HOS build is ~370 s
per sea; the under-det/well-det split there is consequently weak. A wider Tp range could change the Arm-H
picture — but not the α-transfer number, which is already 4.80× inside this narrow range.
**T9.** NLL and conformal width disagree for the `crb_r` hybrid on HOS (NLL prefers it, width prefers the
constant bar). Both are reported; neither is suppressed. The disagreement is expected — NLL rewards
bulk-variance matching, conformal width is set by the 90 % quantile of the normalised score.

---

## 6. What could and could not be claimed in the thesis (honest paragraph)
**Could be claimed.** That the sensor-geometry Cramér–Rao envelope, evaluated at the fitted model and
requiring no ground truth, *orders* reconstruction error within a scene — strongly where the observation is
under-determined (within-episode spread–skill ρ 0.99 under-det / 0.90 pooled here; ρ 0.78 in E-58's
under-det slice; level-ρ
0.89–1.00 on real stereo in E-59) — and that this ordering is shared by estimators of different model
classes (C8, scope-limited). That is a **skill-map / operability-ranking** claim, and it is the claim the
project already makes. One may additionally claim, as a *negative* contribution with real teaching value,
that **the bound's ordering does not extend to its magnitude**: σ_CRB over-disperses by roughly 6× and its
episode-to-episode level is statistically unrelated to realised error (Spearman +0.18, p=0.33, n=32), so a
single inefficiency factor cannot convert the envelope into a shippable confidence interval — the factor
varies 19.9× across synthetic episodes and 4.8× across HOS seas, against a pre-registered 3× tolerance.
**Could NOT be claimed.** That WaveSplat outputs calibrated per-location predictive uncertainty anchored on the CRB; that
one global inefficiency factor calibrates the bound across sea states; that a CRB-anchored bar improves on
empirical/posterior-variance uncertainty (it is *behind* it here, and the empirical construct is already
published as Fucile 2018 LEPrE and, in remote sensing, as the Rodgers optimal-estimation error covariance);
or that "bound-calibrated UQ" is a novel method (it is variance scaling / normalized conformal prediction /
the geodetic a posteriori variance factor, depending on which field you ask). **Recommendation: do not open
a UQ chapter on this axis.** If a calibrated bar is nevertheless wanted for the operability story, the
cheapest defensible route is σ_CRB-normalized **split-conformal** intervals — free finite-sample coverage,
zero novelty, one paragraph in the methods section, cited to Papadopoulos — and, on this evidence, one
should expect those intervals to be *wider* than a constant-width interval, so even that is a presentation
choice rather than an improvement. The pincer in §4.1b is the compact way to say it in a viva: **where the
model is right, the exact posterior variance (LEPrE) beats the bound; where the model is wrong, nothing with
one global constant calibrates. The CRB's job in this thesis is to say *where*, not *how much*.**
