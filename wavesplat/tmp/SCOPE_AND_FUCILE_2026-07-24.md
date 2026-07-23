# Scope statements, estimator caveat, and the Fucile delta — 2026-07-24

Target: `wavesplat/paper/wavesplat.tex` (Ocean Engineering, elsarticle, natbib `\citep`/`\citet`).
No `.tex` was edited. Everything below is paste-ready.

> ⚠ **Line numbers are volatile.** `wavesplat.tex` was being modified while this was
> written (2143 → 2167 lines, mtime 2026-07-24 00:20, `git status` = ` M`). Every target
> below is therefore given as **(current line number, verbatim anchor string)**. Match on
> the anchor, not the number.

---

## FINDING 0 (read first) — the paper contains no CRB analysis

**MEASURED.** `grep -inE "CRB|Cram|Fisher|information wall|predictable zone|effective aperture"`
over every `.tex` in the repo returns **zero** hits in:

- `wavesplat/paper/wavesplat.tex`
- `wavesplat/thesis/thesis_main.tex`
- `wavesplat/hydrofoil/paper/hydrofoil_feedforward.tex`
- `wavesplat/paper/cover_letter.tex`

The only `.tex` hit anywhere is `wavesplat/paper/encounter_forecast/encounter_forecast.tex`
L39 (keyword `predictable zone`) and L50 (section comment `2.2 predictable zone`) — and that
file is a 121-line **skeleton** whose §2 body has not been pasted in yet.

Where the CRB material actually lives:

| artefact | path | status |
|---|---|---|
| CRB / FIM engine | `wavesplat/scripts/d1_crb.py` (461 lines) | code, runs |
| oracle-N sanity check | `wavesplat/tmp/theory_audit_component_crb_check.py` | code |
| CRB narrative (§2.2 predictable zone, §2.3 limit ladder, §2.4 slope ceiling) | `wavesplat/docs/DRAFT_main_paper_v0.md` L134–L175 | markdown prose |
| destination paper | `wavesplat/paper/encounter_forecast/encounter_forecast.tex` §2 `\label{sec:info}` | skeleton, body not written |

**Consequence for Deliverable 1.** There are no CRB-derived claims in `wavesplat.tex` to
qualify. The deliverable therefore splits, and I have written both halves:

- **1A** — the properly-scoped statement to place *once*, where the bound is introduced.
  Its destination is `encounter_forecast.tex` §2 (and the §2.2 paragraph of
  `DRAFT_main_paper_v0.md` that feeds it). Paste it before the §2 body lands, so the scope
  ships with the derivation rather than being retrofitted.
- **1B** — the **five** limit-flavoured claims that *do* exist in `wavesplat.tex`. These are
  beamforming/initialization resolution claims, not CRB claims, and they carry a *different*
  risk: they are estimator properties phrased as if they were method-independent. They need
  the Deliverable-2 caveat, not the CRB caveat. Targets and wording below.

Count answer: **0 CRB locations in `wavesplat.tex`; 5 limit-claim locations that a reviewer
will read as information-wall claims; 1 place where the CRB scope block belongs, in a
different file.**

---

## DELIVERABLE 1A — the CRB scope statement

### Verification of the three limitations against the code

All three are **MEASURED from `wavesplat/scripts/d1_crb.py`**, not inferred:

1. **Oracle N.** `load_ep()` (L151) builds `theta` directly from the episode's true wave
   components, `N = theta.shape[0]`. `fisher()` (L97) is evaluated at that true `theta`, and
   `per_component_crb` reports per-component variances for exactly those N. No model-order
   term enters the FIM. Confirmed independently by the header of
   `wavesplat/tmp/theory_audit_component_crb_check.py`, which states the failure mode as
   *"CRB's information is thrown away by a model-order selection layer the CRB does not model."*

2. **Misspecified model.** Ground truth is Gerstner: `wavesim/wave_generator.py:73`
   `single_gerstner()` (described in-file as the *rotational exact solution*), driven through
   `wavesim/gt_generator.py:38`. The CRB's forward model, `d1_crb.py:64 predict()`, is a
   linear cosine superposition with `ω = √(g·k·tanh(kd))`. So the differentiated likelihood is
   not the data-generating law.

3. **No current.** `d1_crb.py:52 omega_of_k()` = `sqrt(G*k*tanh(k*depth))`. There is no
   `k·U` term anywhere in the file and `U` is not a column of the Jacobian. Repo evidence
   that this is not a harmless idealisation: `wavesplat/docs/SESSION_SYNTHESIS_2026-07-05.md`
   ("H6 해류 Doppler 불일치 = 유력 원인: 0.15 m/s에 hand gain 전멸") and the
   `gp_spacetime_results` memory entry attributing the sim–field gap to current Doppler.

### Precision point (do not skip)

What `d1_crb.py` computes is the **CRB of the assumed linear model, evaluated at the true
component parameters**. That is *not* the misspecified CRB (MCRB) of Richmond & Horowitz /
Fortunati et al., which is evaluated at the pseudo-true parameter and carries an extra
mismatch term. Claiming "we report the MCRB" would be false. The wording below says only
that the bound is *for the assumed model*, and points the reader at the MCRB literature as
the correct tool if they want the true bound. Both supporting references are Crossref-verified
(see BibTeX at the end).

### Paste-ready — place once, where the bound is introduced

**Target:** `wavesplat/paper/encounter_forecast/encounter_forecast.tex`, immediately after
`\section{Problem setting and information analysis}\label{sec:info}` (currently L48–L54,
anchor: `% markdown §2 (완문): 2.1 observation class / 2.2 predictable zone`), i.e. before the
§2.2 predictable-zone paragraph.
**Mirror into:** `wavesplat/docs/DRAFT_main_paper_v0.md` §2.2, before *"Phase-resolved
forecasts are bounded by group-velocity geometry"* (L134).

```latex
\paragraph{Scope of the bound.}
The Cram\'er--Rao bound (CRB) used throughout this section is evaluated under
three assumptions, and every statement we derive from it is scoped to them.
\emph{(i)~Oracle model order.} The Fisher information is evaluated at the true
parameter vector with the true number of spectral components $N$ known.  The
bound therefore prices the estimation of $\theta$, but not the \emph{selection}
of $N$, which a deployable estimator must also perform from the data.  Model
order selection is a strictly additional error source, and it is one-sided: it
can only move an estimator away from the bound, never below it.  Bounds reported
here are consequently optimistic for any estimator that does not know $N$.
\emph{(ii)~Misspecified forward model.} The ground-truth surface is generated
with Gerstner (rotational, weakly nonlinear) kinematics, whereas the likelihood
differentiated here---in common with every estimator we compare---assumes a
linear free-wave superposition constrained by
$\omega=\sqrt{g|k|\tanh(|k|d)}$.  What we compute is thus the CRB of the
\emph{assumed} model, not the true bound for the data-generating process; it is
the appropriate reference for the class of linear phase-resolved estimators
studied here, and we make no claim about estimators that resolve the nonlinear
kinematics.  Quantifying the penalty of the mismatch itself would require the
misspecified CRB \citep{richmond2015, fortunati2017}, which is evaluated at the
pseudo-true parameter and which we do not compute.
\emph{(iii)~No ocean current.} Neither the bound nor the estimators carry a
current: the dispersion relation contains no Doppler term
$\mathbf{k}\cdot\mathbf{U}$, and $\mathbf{U}$ is not a parameter of the Fisher
information.  A current does not merely add a nuisance parameter that would
inflate the bound---it changes the model, and our own sensitivity checks
indicate that a current of $0.15$~m/s is already enough to matter.  All values
below are bounds for a current-free sea.
Within these three scopes the bound is exact; read outside them it is
indicative, and we mark the distinction where it affects a conclusion.
```

### Short in-place qualifiers (use *instead of* repeating the block)

Each is one clause. Insert only where the bound is being used to license a conclusion.

**(a) At any "the wall is here" / predictable-zone boundary statement**
(`DRAFT_main_paper_v0.md` §2.2, after *"The boundary is where the physics says it should be."*):

```latex
This boundary is a property of the linear, current-free, oracle-order model of
Sec.~\ref{sec:info}; it locates where information runs out under that model, and
does not by itself certify that no estimator outside that class can do better.
```

**(b) At the limit ladder** (`DRAFT_main_paper_v0.md` §2.3, after *"saturation within the
examined class"*):

```latex
``Examined class'' is meant literally: linear free-wave kinematics, no current,
and---for the bound-based rungs---known model order.
```

**(c) At the slope ceiling** (`DRAFT_main_paper_v0.md` §2.4, after the numbers):

```latex
As with the other bound-anchored quantities, this ceiling is computed with the
true component count supplied; an estimator that must also infer it faces a
strictly larger error.
```

**(d) At any figure or table reporting CRB curves** — append to the caption:

```latex
CRB computed with oracle model order, linear free-wave kinematics, and no
current (Sec.~\ref{sec:info}).
```

**(e) In `Limitations`** — one sentence tying it off:

```latex
\emph{Scope of the information analysis}: the bounds of
Sec.~\ref{sec:info} assume known model order, linear free-wave kinematics, and a
current-free sea.  Relaxing any of the three---joint order selection, a
second-order or fully nonlinear likelihood, or a Doppler-shifted dispersion
relation with $\mathbf{U}$ estimated---changes the bound, and we have not
computed it in those settings.
```

**ARGUED, and worth acting on:** the three scopes push the bound in *opposite* directions.
Oracle N and the missing current make it **optimistic**; a misspecified likelihood can make a
bound either optimistic or pessimistic and there is no free-lunch sign argument. Do not write
"the bound is conservative" anywhere.

---

## DELIVERABLE 1B — the five limit claims that exist in `wavesplat.tex`

These are the loci a reviewer will read as information-wall statements. None is CRB-derived;
all are estimator-configuration properties. Each takes the Deliverable-2 caveat by reference.

| # | line (2026-07-24) | verbatim anchor | risk |
|---|---|---|---|
| 1 | 1001 | `($\geq$15 components) where beamforming cannot resolve individual modes.` | reads as method-independent |
| 2 | 1099 | `initialization resolution limit.` | in a per-episode bullet |
| 3 | 1220 | `scenes due to beamforming resolution limits.}` | **figure caption** — travels alone |
| 4 | 1505 | `reflecting the initialization resolution limit as beamforming peaks` | by-Beaufort trend claim |
| 5 | 2013–2015 | `\emph{Initialization resolution}: for $N_{\text{true}}\geq 30$,` … `beamforming cannot resolve all components due to wavenumber-space` | Limitations; the natural host |

Minimal edits (nothing else changes):

- **#1** → `($\geq$15 components) where our beamforming initializer, at its configured
  peak-separation radius (Sec.~\ref{sec:limitations}), cannot resolve individual modes.`
- **#3** (caption) → append: `The saturation reflects the configured beamforming
  peak-separation radius, not a method-independent limit (Sec.~\ref{sec:limitations}).`
- **#5** → replace with the Deliverable-2 block below, which subsumes it.

Leave #2 and #4 as they stand; both already say "initialization", which is honest, and the
Limitations block now defines what that means.

---

## DELIVERABLE 2 — the `min_dk` estimator-scope caveat

### Facts (all MEASURED)

- Constant: `wavesplat/optimizer/initializer.py:28  MIN_DK = 0.15` (rad/m), consumed at
  `initializer.py:403/436/437`, `coarse_to_fine.py:190–191/210`,
  `sequential_update.py:515–516/533`. It is the suppression radius of the greedy beamform
  peak search: two peaks closer than 0.15 rad/m cannot both be accepted, at any SNR.
- Census: 126 / 195 shipped episodes (64.6 %) contain a true component pair closer than
  0.15 rad/m, including **100 %** of every 15-, 20- and 30-component episode.
  (`wavesplat/tmp/SCALE_AUDIT_fable_2026-07-23.md`.)
- Pre-registered sweep, decision rule fixed before the numbers, 5 seeds/cell, re-run on the
  Nyquist-adequate regenerated dataset (`wavesplat/tmp/MIN_DK_RESULT_2026-07-23.md`,
  `tmp/min_dk_sweep_finemesh.py`):

  | episode | 0.15 | 0.05 | paired Δ | t | direction |
  |---|---|---|---|---|---|
  | BF4 15comp (crowded) | 0.2912 | **0.2722** | **−0.0150** (5.2 % of error) | −3.14 | 5/5 improved |
  | BF3 5comp (medium) | 0.1699 | 0.2209 | **+0.0293** | +2.55 | **0/5** improved |
  | BF3 2comp (control, no close pair) | 0.0844 | 0.0876 | +0.0178 | +0.72 | inert |

  0.05 and 0.02 give identical results — the effect saturates. The control being inert is
  what licenses the reading. Both the original and the fine-mesh run agree on magnitude
  (−0.0156 → −0.0150); the fine-mesh run makes **both** arms individually significant.

**Licensed statement:** the wall survives (the crowded regime loses only 5 % of its error),
but empirical "no method does better" phrasing measured through this estimator is optimistic
by a few percent. The correct repair is an aperture-scaled adaptive radius — proposed, **not
tested**. Do not claim it works.

### Paste-ready

**Target:** `wavesplat.tex`, `\subsection{Limitations and future work}` — **replace** the
existing three-line `\emph{Initialization resolution}` entry
(current L2013–2015, anchor `\emph{Initialization resolution}: for $N_{\text{true}}\geq 30$,`).
Placing it here, rather than in Results, keeps one authoritative statement that #1/#3 above
can point at.

```latex
\emph{Initialization resolution and the peak-separation radius}: the greedy
beamforming peak search enforces a fixed minimum separation
$\Delta k_{\min}=0.15$~rad/m between accepted peaks, so component pairs closer
than this in wavenumber space are structurally unresolvable by our initializer
regardless of signal-to-noise ratio.  This is not a corner case: 126 of the 195
simulated episodes---and every 15-, 20- and 30-component episode---contain at
least one true pair separated by less than $0.15$~rad/m.  A pre-registered sweep
of $\Delta k_{\min}$ (5 seeds per cell, decision rule fixed in advance, with a
control episode containing no close pair) shows the constant to be a
regime-dependent trade rather than a universal floor: relaxing it to
$0.05$~rad/m improves the crowded 15-component case on 5/5 seeds by
$0.015$~nRMSE---$5.2\%$ of that regime's error---but degrades the 5-component
case on 0/5 seeds ($+0.029$~nRMSE), because a smaller radius lets the greedy
search stack duplicate peaks on a single true component; the control episode is
unaffected ($t=0.72$).  Two consequences follow.  First, the crowded-regime
error is \emph{not} explained away by this constant: the bulk of it survives
when the constant is removed.  Second, statements in this paper of the form
``these components are not recovered'' are properties of the estimator as
configured, and understate what this estimator class can achieve in crowded seas
by a few percent; they should not be read as method-independent impossibility.
The principled repair is not a smaller fixed radius but an adaptive one, scaled
to the beamforming main-lobe width and hence to the effective aperture; we have
not tested it.
```

**Also applies to** (flagged, no wording supplied here): the 2026-07-23 ANM/alt-methods gate
conclusion *"weak short-λ components are unrecoverable by all methods at all apertures."*
Same pipeline, same constant. The *relative* comparison (SFW ≈ greedy+joint) is unaffected
because both arms ran at 0.15; only the absolute "by all methods" phrasing needs the caveat.

---

## DELIVERABLE 3 — the Fucile delta

### 3.0 Two corrections to the brief, before anything else

**(A) LEPrE is not what the brief says it is.** LEPrE = **Linear Estimator of Prediction
Error**, not "Linear Extended Prediction of the Encountered wave". VERIFIED — the expansion
appears verbatim in the paper's own abstract, recovered independently through three separate
searches. Writing the wrong expansion in a Related Work paragraph, in the same journal, is
exactly the error a reviewer holding the paper would catch first.

**(B) Fucile is not part of the stereo/WASS line.** VERIFIED:

- Authors of the Ocean Engineering paper are **Fucile, Bulian, Lugni** (Crossref + OpenAlex,
  independent agreement). Bergamasco / Benetazzo / Barbariol / Sclavo / Carniel are **not**
  authors.
- Fabio Fucile's complete OpenAlex bibliography (35 records, ocean-related subset filtered by
  hand) contains **no** stereo, WASS, or camera paper. His remote-sensing companions are
  **X-band marine radar**: *"Some challenges in recovering wave features from a wave radar
  system"* (Fucile, Ludeno, Serafino, Bulian, Soldovieri, Lugni, 2016) and Ludeno et al. 2016.
- The 2018 paper's own 38-item reference list (pulled from Crossref) contains **zero**
  WASS/stereo references. It cites radar, wave probes, and a flash-LIDAR camera.

⇒ Do **not** write a Related Work paragraph that positions WaveSplat against "Fucile's stereo
cameras". The stereo/WASS group is a separate body of work; if you also want to cite it, cite
Bergamasco et al. (WASS) separately — that is a different delta and is **not** drafted here.

### 3.1 Verified content of Fucile et al. (2018)

| axis | finding | status |
|---|---|---|
| Bibliographic | Fucile, F., Bulian, G., Lugni, C., *Ocean Engineering* **163** (2018) 718–736, doi `10.1016/j.oceaneng.2018.04.079` | **VERIFIED** — Crossref + OpenAlex agree |
| What it delivers | a semi-analytical methodology for **prediction-error statistics** in deterministic sea wave prediction (DSWP) based on linear phase-resolved wave models | **VERIFIED** (abstract) |
| Statistical model | wave elevation as a **Gaussian stochastic process**; propagation-model coefficients obtained by **linear fitting** on measurements in time and/or space; measurement error explicitly included | **VERIFIED** (abstract) |
| Output | prediction-error **standard deviation in time and space**, given the fitting procedure and the sea spectrum → LEPrE; defines the **"Predictability Region"** probabilistically | **VERIFIED** (abstract) |
| Validation | example applications for **long-crested and short-crested** waves, verified by **Monte Carlo** simulation | **VERIFIED** (abstract) |
| Sensing modality | formulation is **measurement-agnostic** ("available measurements in time and/or space"); the author's companion work is X-band radar; reference list is radar / probe / flash-LIDAR-camera | **VERIFIED** for the formulation; the modality of any worked example is **NOT VERIFIED** |
| Real-sea validation | abstract reports Monte Carlo verification only | **NOT VERIFIED either way** — I could not read the full text (ScienceDirect 403, ResearchGate 403, mirrors down). **Do not assert that it lacks field validation.** |
| CRB / Fisher analysis | no CRB or estimation-theory reference in its 38-item reference list; the quantity is conditioned on a *specified* fitting procedure ("given the fitting procedure"), i.e. the error covariance of one linear estimator | **ARGUED** (strong, but from the abstract + reference list, not the text). **Do not write "Fucile gives no bound."** |

**Standing collision to record:** Fucile's **"Predictability Region"** predates and occupies
the same conceptual ground as our **"predictable zone"**. Any novelty claim on that concept in
the `encounter_forecast` paper is unavailable. This is consistent with the existing repo
ruling that Agent G's CRB-calibrated-UQ direction was NO-GO on Fucile-2018 preemption.

### 3.2 (a) Related Work paragraph — paste-ready

**Target:** `wavesplat.tex`, at the end of `\subsection{Grid-based wave field reconstruction}`,
immediately **after** the WAVEWATCH~III sentence and **before**
`\subsection{Grid-free alternatives: spectral, GPR, and learning-based}`
(currently L340–L344, anchor: `bulk statistics at kilometer scales without resolving individual` / `phases.`).

Rationale for the location: Fucile is DSWP-adjacent, same journal, and belongs with the
deterministic-reconstruction discussion rather than with the grid-free alternatives — it is
not an alternative representation, it is an error model.

Delta is stated on **three verified axes** (what is produced; the estimator class; the
evidence base), and closes by *adopting* Fucile rather than displacing him — which is both
honest and the answer a reviewer wants.

```latex
A complementary line of work asks not how to reconstruct the field but how large
the resulting prediction error is.  \citet{fucile2018}, in this journal, model
the surface elevation as a Gaussian stochastic process and derive a
semi-analytical Linear Estimator of Prediction Error (LEPrE), which returns the
prediction-error standard deviation in time and space for a phase-resolved
linear wave model whose coefficients are obtained by linear fitting on
measurements, and which places the notion of a ``Predictability Region'' on a
probabilistic footing; the formulation is verified against Monte Carlo
simulation for long- and short-crested seas.  Our contribution is orthogonal on
three axes.  First, in \emph{what is produced}: \citet{fucile2018} characterise
the error of a given deterministic prediction procedure and do not propose a
field estimator, whereas we propose one and benchmark its reconstruction
accuracy against ensemble-Kalman and Gaussian-process baselines.  Second, in the
\emph{estimator class}: LEPrE presumes model coefficients determined by linear
fitting, while WaveSplat obtains $(w_i,k_{x,i},k_{y,i},\varphi_i)$ by nonlinear
joint optimisation with the dispersion relation imposed as a hard constraint and
with the model order adapted online---a class outside the linear-fitting
assumption on which the LEPrE derivation rests.  Third, in the \emph{evidence
base}: the LEPrE formulation is agnostic to the measurement device, whereas our
evaluation is specific to the sampling geometry of a scanning shipboard LiDAR,
including sparse, range- and incidence-dependent returns and reflectivity-driven
dropout.  The two are naturally combined rather than opposed: WaveSplat
currently returns point estimates without an error model
(Section~\ref{sec:limitations}), and the LEPrE framework is the natural route to
supplying one for the grid-free spectral representation.
```

**Deliberately absent, and why.** No claim that Fucile is grid-based; no claim that he
predicts at a point rather than over a field (his error model is explicitly "in time and
space"); no claim that he lacks real-sea validation; no claim that he provides no bound. Each
of those is a sentence a reviewer with the PDF could call false, and none is needed — the
three axes above are sufficient and are quoted from the abstract.

### 3.3 (b) One-line delta for the contributions list

**Target:** `wavesplat.tex`, inside the contributions `enumerate`, appended to item **C1**
(currently ends L234, anchor: `established during initialization.`), or as a standalone final
sentence before `\end{enumerate}` (L249).

```latex
Relative to prediction-error frameworks for deterministic wave forecasting such
as LEPrE~\citep{fucile2018}, which quantify the error of a linear-fitting
prediction procedure without proposing an estimator, we contribute the estimator
itself: a nonlinear, dispersion-constrained, grid-free fit with online model
order, evaluated on scanning-LiDAR sampling geometry.
```

Shorter variant if space is tight:

```latex
Where LEPrE~\citep{fucile2018} quantifies the error of a linear-fitting wave
prediction, we contribute the estimator---nonlinear, dispersion-constrained,
grid-free, with online model order---and its LiDAR-specific evaluation.
```

### 3.4 BibTeX

`wavesplat/paper/references.bib` has 36 entries and **no** Fucile / Bergamasco / stereo /
WASS entry (verified). Add:

```bibtex
@article{fucile2018,
  author  = {Fucile, Fabio and Bulian, Gabriele and Lugni, Claudio},
  title   = {A probabilistic approach for the quantification of prediction error
             in deterministic phase-resolved wave forecasting},
  journal = {Ocean Engineering},
  volume  = {163},
  pages   = {718--736},
  year    = {2018},
  doi     = {10.1016/j.oceaneng.2018.04.079}
}
```

Required only if you use the MCRB clause in Deliverable 1A (both Crossref-verified):

```bibtex
@article{richmond2015,
  author  = {Richmond, Christ D. and Horowitz, Larry L.},
  title   = {Parameter Bounds on Estimation Accuracy Under Model Misspecification},
  journal = {IEEE Transactions on Signal Processing},
  volume  = {63},
  number  = {9},
  pages   = {2263--2278},
  year    = {2015},
  doi     = {10.1109/TSP.2015.2411222}
}

@article{fortunati2017,
  author  = {Fortunati, Stefano and Gini, Fulvio and Greco, Maria S. and
             Richmond, Christ D.},
  title   = {Performance Bounds for Parameter Estimation under Misspecified
             Models: Fundamental Findings and Applications},
  journal = {IEEE Signal Processing Magazine},
  volume  = {34},
  number  = {6},
  pages   = {142--157},
  year    = {2017},
  doi     = {10.1109/MSP.2017.2738017}
}
```

Optional supporting cite, **verify the venue before using**: Fucile's PhD thesis,
*"Deterministic sea wave and ship motion forecasting: from remote wave sensing to prediction
error assessment"*, University of Trieste, 2017, `hdl:11368/2908146` (title and handle
verified via OpenAlex; the landing page returned 403, so the defence year 2017 vs 2016 is
reported inconsistently across sources — **check before citing**).

**Do NOT auto-add** the 2016 radar paper *"Some challenges in recovering wave features from a
wave radar system"*: it has no Crossref record and I could not establish its venue. A
fabricated venue in a bib entry is precisely the failure mode commit `aedf5a7`
("인용문헌 fabricated 7건 정정") already had to clean up once.

---

## Sources

- [Fucile, Bulian & Lugni 2018, Ocean Engineering — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0029801818305900)
- [Crossref record, doi:10.1016/j.oceaneng.2018.04.079](https://api.crossref.org/works/10.1016/j.oceaneng.2018.04.079)
- [OpenAlex record and author bibliography](https://api.openalex.org/works?filter=doi:10.1016/j.oceaneng.2018.04.079)
- [WASS project publications (separate stereo group)](https://sites.google.com/unive.it/wass/publications)
- [Fucile PhD thesis, ArTS Trieste, hdl:11368/2908146](https://arts.units.it/handle/11368/2908146)
- [Richmond & Horowitz 2015, IEEE TSP](https://doi.org/10.1109/TSP.2015.2411222)
- [Fortunati et al. 2017, IEEE SPM](https://doi.org/10.1109/MSP.2017.2738017)
