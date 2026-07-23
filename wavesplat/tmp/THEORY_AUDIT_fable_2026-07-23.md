# Theory audit — is the CRB the right tool, and is it derived for the right regime? (fable, 2026-07-23)

> Scope: this is a **theoretical-foundations** audit, not another empirical result. Everything so far this
> project has asked "did we measure X correctly"; this asks "is X even the right quantity". Read: CRB backbone
> `scripts/d1_crb.py`, partition-CRB `tmp/e51_crb.py`/`tmp/e52_crb_achievability.py`, envelope validation
> `tmp/e58_envelope_validation.py`, `tmp/wavenumber_id_crb.py`, dispersion model `wavesplat/model/dispersion.py`,
> adaptive-N `wavesplat/optimizer/coarse_to_fine.py`, and today's variance findings
> (`tmp/ESTIMATOR_VARIANCE_RESULT_2026-07-23.md`, `tmp/VARIANCE_MECHANISM_HUNT_2026-07-23.md`,
> `tmp/VARIANCE_TRACE_fable_2026-07-23.md`, `tmp/NOVELTY_AGENT_G_crb_calibrated_uq_2026-07-23.md`). One small
> read-only script was run for this audit (`tmp/theory_audit_component_crb_check.py`, an FIM inversion, no
> training — "modest computation" honored).

## TL;DR — ranked findings

| # | Item | Verdict | Severity |
|---|---|---|---|
| 0 | Threshold-effect hypothesis | **Confirmed, but mis-named** — it is a real CRB-vs-ambiguity gap (classical DOA "threshold/outlier" territory), *not* the smooth SNR-threshold ZZB/WWB describe; the discrete trigger is model-order selection (Meir–Routtenberg territory), and it is a **corner case** (energy-contrast-gated), not the general regime | **MATERIAL** |
| 1 | Model-order selection vs the CRB | The deployed CRB is an **oracle-N, oracle-θ** bound; the deployed estimator is **not** fixed-N; nowhere in the CRB code/docs is this gap named; the project's own envelope-validation (E-58/59) tests a *different, non-adaptive* estimator, so the real adaptive-N pipeline has **never** been checked against the envelope | **MATERIAL** |
| 2 | Misspecification (Gerstner/HOS truth vs linear model) | Real, already informally flagged ("Gaussian-근사 CRB", GUARD2), but not connected to the formal misspecified-CRB literature; HOS-arm UQ results (Agent G) are consistent with exactly the predicted degradation | **MATERIAL** |
| 3 | Dispersion hard constraint / current | Real and measured **elsewhere in the project** (5/5 blind current recovery failures, oracle-current full rescue) but **zero acknowledgment inside the CRB derivation itself** — no current term exists in the model at all | **MATERIAL** |
| 4 | Achievability half unpaid (GUARD2) | Already the best-handled item in the whole project (extensive, correctly-labelled caveats); one new piece of evidence (today's bimodal-variance finding) further undercuts the Gaussian/CRB-efficient assumption GUARD2's ceiling relies on | **MATERIAL, but mostly already priced in — closest to COSMETIC** |

**No item is FATAL.** Nothing currently committed to the thesis (`thesis/thesis_main.tex`, which does not yet
contain the CRB-envelope chapter at all — grep for "Cram", "lower bound", "envelope" returns zero hits) makes
an overclaim that these findings break. The risk is entirely in the **draft** (`tmp/PAPER_SKELETON_operability_envelope_2026-07-18.md`)
and in the CRB code's own docstrings, which do overclaim locally ("any unbiased estimator") in a way that
should be fixed **before**, not after, this material reaches the thesis. "The theory is fine as used" is **not**
the verdict — but "the spine has to be abandoned" is equally not the verdict. See §6 for the phrasing that
survives.

---

## 0. The concrete suspicion: is this the threshold region, and is CRB blind to it?

### 0.1 What is actually bimodal (MEASURED, from today's prior work)
`ESTIMATOR_VARIANCE_RESULT` + `VARIANCE_TRACE_fable`: on `BF3_2comp_seed0_uniform` (true components A=0.2088,
97% of energy, and A=0.0372, 3.1% of energy — note: 3.1% of *energy*, and the *weak component's own* recovered
fraction never exceeds 0.08% of *total field* energy, i.e. its own amplitude is essentially never resolved),
run-to-run field-nRMSE spans 0.052–0.168 (CV 53%) across seeds that differ **only** in RNG draws for initial/grown
phase. The forensic trace shows: neither the "good" nor the "bad" run recovers the weak wave; both fit the
strong wave to ~93% amplitude; the only difference is whether **1 or 2** near-degenerate copies of the *strong*
wave survive pruning, and the second copy's spare capacity absorbs the tracking-phase residual from the
un-modelled weak wave. Removing two RNG sites (H4) collapses CV to 0.0% — i.e. **the discrete fork is fully
attributable to two specific random draws, not to intrinsic estimator stochasticity.**

### 0.2 New measurement for this audit — does the oracle CRB already know the weak wave is hard?
If the CRB (evaluated at the **true** θ, **true** N=2, i.e. the most favourable possible setting) already
predicted a huge σ on the weak component's own amplitude/wavenumber, that would mean "CRB correctly signals
non-identifiability, the estimator's failure is expected, no theoretical gap here." **This is not what happens.**
`tmp/theory_audit_component_crb_check.py` (read-only, imports `scripts/d1_crb.py` unmodified, one FIM inversion
per T_obs, on the exact episode/seed pair from the trace):

| T_obs | M (points) | σ_CRB(w), weak comp (true w = 0.0372) | "CRB-SNR" = w/σ_CRB |
|---|---|---|---|
| 0.1 s | 100 | 0.0034 | **10.9** |
| 0.4 s | 274 | 0.0018 | **20.4** |
| 1.0 s | 577 | 0.0012 | **30.0** |
| 3.0 s | 1549 | 0.0007 | **51.9** |
| 19.0 s | 9679 | 0.0003 | **129.9** |

**The oracle local CRB says this weak wave should be trivially identifiable — CRB-SNR ≈ 11 with only 100
points, rising to ≈130 by the end of tracking.** Measured outcome: the best of 8 seeds recovers **0.08% of
total field energy** for this component after the full 900-iteration budget — three orders of magnitude below
even the loosest tested pruning threshold, i.e., **effectively zero**. This is not a modest miss; it is a
40–100× gap between what the local (Fisher) geometry says is achievable and what any run of the actual
non-convex search achieves.

### 0.3 What kind of gap is this — and is the hypothesis's prescription (ZZB/Barankin/WW) right?
**Partially, and for a more specific reason than stated.** The hypothesis says "CRB is a local, small-error,
regular bound; near a threshold it is loose or meaningless; Ziv–Zakai/Barankin/Weiss–Weinstein are the classical
tools." That is correct as a general statement, and the 40–100× CRB-vs-achieved gap measured above is *exactly*
the signature the classical threshold-effect literature describes — but the **mechanism** here is not "SNR is
locally too low" (§0.2 shows the opposite: local SNR is fine) — it is a **global search / ambiguity** failure:
the greedy residual-beamforming step that must *locate* the weak wave's wavenumber cell among many candidates
never lands near it, because the strong wave's imperfectly-fit residual dominates the search landscape at that
point in the schedule. This is precisely the classical DOA "threshold region" / "outlier" phenomenon:

> "The maximum likelihood (ML) estimator exhibits a threshold effect, characterized by rapid deterioration of
> estimation accuracy below a certain SNR or number of snapshots... [the paper] derives approximations to the
> mean square estimation error **and probability of outlier**... for single-source and multisource estimation
> problems." — F. Athley, *Threshold Region Performance of Maximum Likelihood Direction-of-Arrival Estimators*,
> IEEE Trans. Signal Process., 53(4):1359–1373, 2005.
> https://ieeexplore.ieee.org/document/1408188/

and, closer still to this exact setting (multiple sinusoids of very unequal amplitude, needing both detection
and localization of the weak line in the presence of the strong one's leakage):

> Xiao & Slock, *Performance Benchmarks for Line Spectral Estimation: Ordered Ziv–Zakai Characterization and
> Plug-In Amplitude Error Analysis*, arXiv:2604.24007 (2026) — a ZZB-type benchmark for line spectral
> estimation that "captures threshold behavior absent in traditional Fisher-information methods," recovers the
> prior-only bound at low SNR and the marginalized-frequency CRB at high SNR, "moving beyond local performance
> bounds to capture the nonlinear threshold behaviors that existing methods miss."
> https://arxiv.org/abs/2604.24007

General DOA background on the same phenomenon: Bell, Steinberg, Ephraim & Van Trees, *Extended Ziv–Zakai Lower
Bound for Vector Parameter Estimation*, IEEE Trans. Inform. Theory, 1997
(https://static.aminer.org/pdf/PDF/001/104/206/extended_ziv_zakai_lower_bound_for_vector_parameter_estimation.pdf);
recent closed-form DOA ZZBs: arXiv:2209.03700 (https://arxiv.org/pdf/2209.03700), arXiv:2511.08326
(https://arxiv.org/html/2511.08326); survey: Nehorai & Tang, *Bounds on Estimation*
(https://www.ese.wustl.edu/~nehorai/paper/Nehorai_Tang_Bounds_Sec_69-2_2015.pdf) — "because of the nonlinear
parameter-dependence of the signal field, the estimate is subject to ambiguities and the sidelobe contribution
often dominates the estimation error below a threshold SNR."

**But the *specific discrete symptom* measured today (1-copy vs 2-copy of the strong wave; a model-order
decision) is better named by a sibling literature that the hypothesis did not mention: post-model-selection /
selective estimation bounds** (§1). ZZB/WWB are built for ambiguity in a *continuous* parameter (which candidate
angle/wavenumber cell is right) within a *known, fixed* number of sources; the redundant-duplicate mechanism is
a *discrete* order decision (keep 1 or 2 Gaussians) made by a pruning/growth heuristic, which is a different
(related, but formally distinct) failure mode. Both are real here; conflating them would be sloppy.

### 0.4 Corner case or general regime? — **corner case, evidence favours the modest reading**
Three independent pieces of evidence say this is **not** the project's general operating point:
1. **CV is not uniform across episodes**: 53% (BF3 2comp) → 10% (BF3 5comp slope) → 4.8% (BF3 5comp uniform)
   → 2.8% (BF4 15comp), and the *phase-fix* ablation (H4) further cuts the 15-comp CV to 0.9%. The pathology is
   concentrated in **low-component, high-energy-contrast** episodes, not in the crowded (many-component)
   episodes the project's under-determined regime is usually discussed through.
2. **It is removable by fixing two RNG sites**, not merely reduced — a genuine information-theoretic threshold
   effect (one intrinsic to the noise/prior/signal geometry) could not be deleted by de-randomizing
   initialization; a search/implementation artifact riding on top of a real-but-narrow ambiguity region can.
   Both are true simultaneously here: the *underlying* ambiguity (§0.2's 40–100× gap) is real and not removable,
   but *which* of the two "attractor" behaviours (single rigid fit vs. redundant duplicate) a given run lands
   in is an artifact.
3. **§8's own regime axis (kp / aperture-vs-wavelength) already reports the opposite gradient**: envelope
   validity (ρ) is *weaker* in the well-determined, low-kp regime (0.31) and *stronger* under-determined
   (0.78) (`E58_AUDIT_findings_2026-07-18.md`). The energy-contrast corner case identified today is a
   **different axis** from the kp axis the project's envelope is currently organized around — it is ARGUED,
   not measured, that these are independent regime dimensions, but if so, the current envelope map has a
   blind spot along an axis it has never plotted (intra-episode component energy dynamic range), not a
   wholesale failure of its existing axis.

**Verdict: the hypothesis is right that a real threshold/ambiguity phenomenon exists and that CRB is the wrong
tool to describe *it specifically* — but it is confined to a measurable, narrow corner (severe intra-episode
energy contrast, e.g. this project's 2-component low-Hs episodes), it is not what the ZZB/Barankin/WW machinery
alone would fix (the discrete order-selection layer needs the model-selection bound family, §1), and de-randomizing
the estimator removes most — not all — of its symptom. Do not rebuild the thesis spine around "we operate in
the threshold region"; do restate the envelope's scope to exclude/flag this corner, and do not claim CRB
predicts per-episode reliability uniformly.**

---

## 1. Model-order selection vs the CRB

**Finding (MEASURED, code inspection):** `scripts/d1_crb.py::load_ep` builds `theta` directly from the episode
file's ground-truth `wave_components` array — the CRB is always evaluated at the **true θ and true N**. Its own
docstring states the bound holds for "**any unbiased estimator**" (line 21 comment on `field_nrmse_crb`). But
the estimator that ships (`wavesplat/optimizer/coarse_to_fine.py` + `sequential_update.py`) does not know N; it
runs a 3→8→15 schedule with energy-based pruning and residual-triggered growth, and — as this project's own
files just showed — the *order it settles on* (not just the parameters it fits) governs a 3× swing in realized
error on at least one episode class. A bound computed at the true, known order is not a bound on an estimator
that must also *choose* the order; this is not a subtle point, it is the entire subject of a specific
literature:

> Meir & Routtenberg, *Cramér-Rao Bound for Estimation After Model Selection and Its Application to Sparse
> Vector Estimation*, IEEE Trans. Signal Process. 69:2284–2301 (2021), DOI 10.1109/TSP.2021.3068356
> (https://arxiv.org/abs/1904.06837): "the data-based model selection step affects the subsequent estimation…
> **the oracle CRB, which is based on knowledge of the true model, is inappropriate for post-model-selection
> performance analysis and system design outside the asymptotic region**." Their selective CRB is demonstrably
> tighter/more accurate than the oracle CRB and the earlier **SMS-CRB** (Sando, Mitra & Stoica, *On the Cramér–Rao
> Bound for Model-Based Spectral Analysis*, IEEE Signal Process. Lett. 9:68–71, 2002 —
> https://ieeexplore.ieee.org/document/991141/ — the foundational CRB-with-unknown-model-order result, built
> for exactly this project's kind of problem: AR/spectral order chosen by an information criterion).

**Is this acknowledged anywhere in the project's CRB derivation?** No. Grepped `tmp/`, `docs/` for "selective
CRB", "model selection", "post-selection", "SMS-CRB" — zero hits anywhere near the CRB code. The only adjacent
discussion is `tmp/e50_crb_direction_check.md`'s "Gaussian-근사 vs 진짜 random-phase CRB" caveat, which is a
*different* gap (§2) about the noise/generative model, not about model order.

**How much does it matter here, concretely?** Two ways:
1. **The corner case in §0** is a direct instance: the CRB used for envelope/skill claims cannot distinguish
   the 1-copy and 2-copy outcomes of the *same* episode, because it does not condition on what the estimator
   decided — it is the same number regardless. Whatever the CRB's ρ with realized error is (E-58: pooled 0.33,
   under-det 0.78), that correlation is being measured **against realized errors whose own spread (CV up to
   53%) is partly generated by a mechanism the CRB cannot see at all**, in at least the energy-contrast corner.
2. **A structural gap, independent of §0's corner case:** `tmp/e58_envelope_validation.py`'s σ_actual is
   computed from a **k-grid ridge-least-squares forecast** (`build_kgrid`/ridge solve), which is a *different,
   non-adaptive, fixed-basis* estimator from the one that ships in `wavesplat/run.py`. This is already flagged
   for a related reason in `F2_crb_engine_integrity_2026-07-22.md` (two CRB engines, not one) but the
   consequence for *this* audit item has not been stated explicitly: **the deployed adaptive-N WaveSplat
   pipeline — the estimator with the bimodal, order-dependent behaviour this project just discovered — has
   never once been checked against the CRB envelope.** The envelope's empirical support (E-58, E-59, Agent G)
   is entirely about a linear-ridge surrogate. That is a legitimate methodological simplification (it isolates
   the envelope from optimizer idiosyncrasies) but it must be stated as a scope limitation, not left implicit.

**Restate as:** "The reported CRB is a **matched-order** bound: it assumes the number and identity of wave
components are known, and therefore bounds an oracle-order estimator, not the adaptive-N estimator this project
deploys. Where the deployed estimator's own choice of model order is itself uncertain or bimodal (measured in a
low-component, high-energy-contrast corner of the episode set), the reported σ_CRB is not informative about
that discreteness, by construction. The empirical envelope-validation results (§8) were obtained with a
fixed-basis ridge-LS forecast, not the shipped adaptive-N pipeline; extending the validation to the actual
deployed estimator is future work, not measured."

**Severity: MATERIAL.** Real, unpaid, and unlike GUARD2, not currently caveated anywhere near the CRB code. It
does not invalidate the envelope's *ranking* claim (which is about relative σ_CRB across scenes/locations, a
much weaker claim than "bounds this specific estimator's realized error"), but it does mean any future sentence
of the form "the CRB bounds WaveSplat's error" is unsupportable as stated and must be scoped down.

---

## 2. Misspecification: linear model, nonlinear (Gerstner/HOS) truth

**Finding:** the CRB's generative model is `z = Σ wᵢcos(ψᵢ) + ε` — first-order (Airy) superposition. Synthetic
HOS episodes and any real sea are nonlinear (Stokes/Gerstner-type). The project already treats this informally
as a live risk (`e50_crb_direction_check.md`: "1차 = Gaussian-근사 CRB... 진짜 random-phase CRB = 근본 한계
(더 어려움) → 후속"; `F2_crb_engine_integrity`; GUARD2 threads throughout `NOVELTY_REREVIEW`/`SESSION_MASTER`),
but always as "the approximation is optimistic, needs a follow-up," never named via the formal apparatus that
already exists for exactly this situation:

> Richmond & Horowitz, *Parameter Bounds on Estimation Accuracy Under Model Misspecification*, IEEE Trans.
> Signal Process. 63(9):2263–2278 (2015) (https://www.researchgate.net/publication/274404073) — derives the
> pseudo-true parameter (KL-divergence minimizer between assumed and true distributions) and the Misspecified
> CRB (MCRB), a sandwich-form bound (Slepian–Bangs-type) that is the *correct* converse when the assumed model
> is wrong, as opposed to the plain CRB, which is not generally a valid bound on the misspecified estimator's
> error relative to the *true* parameter (only relative to the pseudo-true one). Most recent update: Khatib,
> Harel, Tabrikian & Routtenberg, *Revisiting the Misspecified Cramér–Rao Bound*, arXiv:2605.20739 (2026)
> (https://arxiv.org/html/2605.20739).

**Is the plain CRB, as reported, defensible?** As a bound on the **matched linear model's own parameters**
(i.e., "if the sea really were a sum of Airy waves with these exact θ, no unbiased estimator could beat this"),
yes — that is what it correctly computes, and the gradcheck/GUARD1 audits (`e51_crb.py` independent audit,
F-2) confirm it is implemented correctly for that question. **As a bound on the achievable error against the
actual nonlinear HOS/Gerstner surface, no** — it is, by the Richmond–Horowitz framework's own terms, a
misspecified-CRB situation without a misspecified-CRB computation. This is not a new problem for the project
(GUARD2's whole point is "unpaid"), but the *name* and the *literature that shows how to pay it* were missing.

**Corroborating evidence already in the repo, reread through this lens:** Agent G's Arm H (HOS, 12 nonlinear
seas) shows the CRB-anchored bar *tied* with a constant bar and with the empirical/LEPrE bar, and its
inefficiency factor α spans 4.8× (vs 19.9× on the *linear*-truth but non-oracle-basis Arm S) —
`NOVELTY_AGENT_G_crb_calibrated_uq_2026-07-23.md` §4.4. This is broadly consistent with (not proof of) the
misspecified-CRB prediction that model mismatch degrades the plain CRB's fidelity to real error relative to
the matched case — but note Arm S's own mismatch source (non-oracle k-grid) is a *different* kind of
misspecification (search/basis mismatch, not physics mismatch), so this is a suggestive parallel, not a clean
test of Richmond–Horowitz specifically. **ARGUED, not measured.**

**Restate as:** "The reported CRB is a **matched-model (linear/Airy) bound**. On the nonlinear (HOS, and any
real) sea, this is formally a misspecified-CRB situation (Richmond & Horowitz 2015): the plain CRB bounds
estimation of the linear model's own pseudo-true parameters, not the true nonlinear surface's parameters, and
the two coincide only to the extent the physics mismatch itself is small. A properly derived misspecified CRB
has not been computed for this project; the existing HOS validation (§8, Agent G Arm H) is an *empirical*
substitute for that missing derivation, not equivalent to it."

**Severity: MATERIAL.** Consequential and already suspected by the project, but not FATAL: the project already
declines to claim the plain CRB is valid under nonlinearity (it explicitly calls the linear-model result a
"model-class" bound, not a "ground-truth" bound, in `e52`'s own final framing) — this item mainly supplies the
correct name and citation for a limitation that is otherwise honestly, if informally, conceded.

---

## 3. Dispersion hard constraint and currents

**Finding (MEASURED, code inspection):** `wavesplat/model/dispersion.py` and `scripts/d1_crb.py` implement
`ω = √(g|k|tanh(|k|d))` with **no current term whatsoever** — not a nuisance parameter that gets marginalized,
not a constraint with slack, an *absent* physical effect. The true relation in the presence of an ambient
surface current **U** is the Doppler-shifted dispersion `ω_encounter = √(g|k|tanh(|k|d)) + k·U`
(e.g., Hwang, *Doppler Frequency Shift in Ocean Wave Measurements*, J. Geophys. Res. Oceans 111, C08017, 2006,
https://doi.org/10.1029/2005JC003072: current-induced Doppler shift makes the frequency↔wavenumber map
current-dependent, and "it is impossible to retrieve the ocean surface current speed from Doppler frequency
shift directly" from a single spectral observation — i.e. the shift and a wavenumber reassignment are
observationally close to indistinguishable, exactly the flat-likelihood-ridge finding below).

Because the hard constraint is imposed as an **equality**, not fit, this is a **constrained CRB**
(Stoica & Ng, *On the Cramér–Rao Bound Under Parametric Constraints*, IEEE Signal Process. Lett., 1998 —
https://www.semanticscholar.org/paper/On-the-Cramer-Rao-bound-under-parametric-Stoica-Ng/e9e269c209eed7493197d2e3999e62b8405dba4b)
— valid *only if the constraint is exactly true*. On a sea with nonzero current, the imposed constraint is
false, so this collapses into the same misspecified-bound situation as §2, specialized to the dispersion
relation itself, not merely the wave shape.

**This is not a hypothetical risk for this project — it is already measured, just never connected to the CRB
code.** `docs/SESSION_SYNTHESIS_2026-07-05.md` §⑨–⑫ ("해류 사슬"): blind current recovery failed **5/5** distinct
methods (validation-RMSE selection, spectral-shell concentration, GP-evidence maximization, multi-window,
BMA-over-Uc) at realistic coastal current magnitudes (0.15–0.5 m/s); the record explicitly diagnoses this as a
**flat-likelihood / weak-identifiability** problem ("관측은 (궤적방향 파수, 조우주파수)뿐이라 해류 시프트와
이웃-k 재배정이 구별 불가 — 우도면 평탄"), and shows **full recovery only with oracle current+kp** (track
0.294 ≈ current-free 0.311) — i.e. the damage is not a rounding correction, it is most of the achievable
information in the moving-aperture channel. If Uc had been included as a FIM parameter, this flat-likelihood
description is exactly what a properly computed FIM (near-singular in the Uc direction, strongly coupled with
kx/ky) would show; the project has the empirical symptom of this already but has never computed the
corresponding FIM row.

**Restate as:** "The CRB and the estimator both assume **zero ambient current**. This is not merely a
simplification with a small, bounded penalty: the project's own current-identifiability experiments show that
under realistic coastal currents (0.15–0.5 m/s), the wavenumber/frequency mapping the dispersion constraint
encodes is measurably wrong, current is only weakly identifiable from a single sparse moving trajectory (a flat
likelihood ridge between current-shift and wavenumber reassignment — 5/5 blind-recovery attempts failed), and
the resulting damage recovers almost entirely once current is known by oracle. Any operability envelope
reported under this dispersion model should be read as **current-free-sea envelope**; its numbers are not
expected to transfer, even in ranking, to currented real seas without an explicit current channel — a claim the
CRB derivation does not currently make anywhere in its own documentation."

**Severity: MATERIAL.** The individual facts are already known and well-measured by the project; what's missing
is the explicit statement, inside the CRB derivation itself (not three months' worth of session logs away),
that the reported bound is conditioned on an assumption the project has independently shown to fail
substantially on realistic real-world inputs.

---

## 4. The unpaid achievability half (GUARD2) — honest phrasing

**Status check (MEASURED, from the repo's own record):** this is, by a wide margin, the best-audited item here.
`e50_crb_direction_check.md` → `e52_crb_achievability.py` already: (a) names the gap explicitly ("GUARD2 미납"),
(b) computes an honest *ceiling* (θ̂~N(θ_true,CRB) Monte Carlo) rather than claiming achievability, (c) has an
independent audit correctly downgrading "beats the net" to "is a ceiling, not a demonstrated achiever" (3 named
sources of optimism: correct-family shape, unbiasedness, Gaussian approximation), and (d) the project's own
final framing already states plainly: "CRB의 역할은 *어디서*(where)를 말하는 것이지 *얼마나*(how much)가
아니다" (Agent G §6). This is close to textbook-correct practice already.

**One new, load-bearing piece of evidence from today, not yet connected to GUARD2:** the E-52 ceiling's Monte
Carlo explicitly draws `θ̂ ~ N(θ_true, CRB)` — i.e., it assumes the achievable estimator's error is Gaussian
with the CRB as its covariance (the textbook "efficient, regular" assumption). Today's variance finding shows
this is **directly false** for at least the weak/parasitic-component regime that matters most for the
under-determined story: the actual outcome distribution there is **bimodal** (a "hit" cluster near CRB-implied
precision and a "miss" cluster carrying ~40–100× more error, §0.2), not the unimodal Gaussian the ceiling
assumes. This does not overturn E-52's qualitative conclusion (the ceiling was already flagged as optimistic on
three separate grounds) but it supplies a **fourth, concrete, measured** reason, specific to exactly the
corner where model order is contested, and should be added to E-52's caveat list.

**What is a defensible way to state a converse-only claim?** This is completely standard practice in
information/estimation theory — a bound (necessary condition) does not require a matching achievability result
to be a legitimate, publishable finding; entire subfields (converse-only results in network information theory,
outer bounds in coding theory) report exactly this. The failure mode to avoid is not "reporting only a
converse" — it is **language that implies achievability when none was shown** ("the wall is informational",
"information-limited", "the bound caps recoverable field correlation") without the "necessary, not shown
sufficient" qualifier attached every time.

**Strongest honest phrasing (verbatim, for reuse in the thesis):**
> "We report a **converse only**: under the stated (matched, Gaussian-approximate) model, no unbiased estimator
> can achieve lower field-reconstruction error than σ_CRB, for the class of estimators consistent with the
> assumed component count and generative model. We do not claim, and have not shown, that any realizable
> estimator — including WaveSplat itself — attains this bound; our own ceiling experiment (E-52) suggests the
> gap between σ_CRB and the deployed estimator's realized error is dominated by search/model-order effects, not
> by unrecovered Fisher information, in the regimes tested. Whether σ_CRB is *achievable* in this problem class
> remains open."

**Severity: MATERIAL, but the closest of the five items to COSMETIC** — the project has already done almost
everything right here; the ask is one added caveat (the Gaussian-ceiling-vs-bimodal-reality connection) and
consistent use of the phrasing above wherever the achievability gap is discussed.

---

## 5. What must change, concretely

1. **`scripts/d1_crb.py` docstring** (line ~21, "Field-reconstruction CRB (any unbiased estimator)"): add one
   sentence — *"…conditioned on known component count N and a matched linear/Airy, current-free generative
   model; not a bound on estimators that must also select N (see Meir & Routtenberg 2021), nor, without
   restatement as a misspecified CRB, on nonlinear or currented seas (Richmond & Horowitz 2015)."*
2. **`tmp/e58_envelope_validation.py` and any thesis text drawing on it**: state explicitly that σ_actual is
   from a fixed-basis ridge-LS forecast, not the shipped adaptive-N `wavesplat/run.py` pipeline — this is
   already known internally (F-2) but needs to travel with every citation of the §8 ρ numbers.
3. **Any future "information wall" / "envelope predicts skill" sentence**: attach the current-free and
   matched-order-model scope explicitly; do not let the already-hard-won "PARTIAL, not PASS" discipline erode
   back toward an unscoped claim when this material is finally written into the thesis.
4. **E-52 (GUARD2) caveat list**: add the bimodal-vs-Gaussian-ceiling point from §4 as a fourth source of
   optimism.
5. **Do not** invest in implementing Ziv-Zakai/Barankin/Weiss-Weinstein machinery as the primary fix for §0 —
   the measured mechanism is dominated by model-order selection and a fixable schedule/RNG defect, not by an
   irreducible SNR-threshold property of the physics. If a formal bound is wanted for the corner case, a
   **selective/post-model-selection CRB** (Meir & Routtenberg 2021) or a **Barankin-type detection-estimation
   bound over the discrete order decision** is the correctly-targeted tool — not a general ZZB sweep.

---

## 6. Strongest honest phrasing of the spine's central claim

> "We compute a sensor-and-trajectory-aware Cramér–Rao bound for a **matched, linear, current-free, known-order**
> generative model, and use it as a converse-only, ranking-grade operability signal: within a scene, locations
> and epochs with smaller σ_CRB are, in our own validated data (E-58/E-59, ρ up to 0.78 in the under-determined
> regime), reliably lower-error than locations with larger σ_CRB — but σ_CRB's absolute magnitude does not
> transfer across scenes as a calibrated error bar (Agent G: dispersion over-estimated ~6×, level-Spearman
> ρ=0.18, n.s.), it has not been validated against the deployed adaptive-N estimator (only a fixed-basis
> surrogate), it is not shown to be achievable by any realizable estimator, and it is not a valid bound at all
> on nonlinear or currented real seas without restatement as a misspecified CRB — both of which the project's
> own experiments show carry substantial, not negligible, real-world weight. The envelope's honest job is to
> say *where*, within a scene and within the scope just stated, error will be larger or smaller — not *how
> much*, not *whether it is unavoidable*, and not, without qualification, *for any sea or any estimator*."

This is consistent with — and a slight tightening of — the project's own current best framing
(`NOVELTY_AGENT_G...` §6, `e52`'s "결과 B" framing, the E-58 PARTIAL discipline): nothing here asks for a
retreat from the existing "CONDITIONAL PASS (lower-mid)" verdict; it asks that the scope conditions the thesis
already half-states in scattered session logs be **collected, named with their correct literature, and stated
once, explicitly, next to the CRB derivation itself** rather than three months of tmp/ files away from it.

---

## Labels
- **MEASURED**: the bimodal-variance numbers and forensic trace (§0.1, pre-existing); the oracle-CRB
  component-level check (§0.2, new, this audit, `tmp/theory_audit_component_crb_check.py`); the CRB code's
  oracle-N/oracle-θ/current-free construction (§1, §3, code inspection); the current-recovery 5/5-failure and
  oracle-rescue numbers (§3, pre-existing); GUARD2's existing three-source optimism audit (§4, pre-existing).
- **ARGUED**: that energy-contrast is a genuinely distinct regime axis from kp/aperture (§0.4); that HOS's
  milder α-transfer-vs-Arm-S parallels Richmond–Horowitz's predicted misspecification degradation (§2); that
  a properly computed FIM row for current would show near-singularity (§3, plausible from the flat-likelihood
  description but not itself computed here).
- **Not attempted** (out of scope for "modest computation"): deriving an actual misspecified CRB or a
  current-inclusive FIM; implementing a selective/post-model-selection CRB; a full ZZB/Barankin computation.
  These are the natural next steps if the project wants to *close* rather than merely *scope* items 1–3.
