# Methods re-review — was any design, tuning, or mechanism story built on the distorted mesh? (2026-07-23)

Scope per task: re-review the **methods themselves** (WaveSplat, WavePrior, `mmse_bank_estimate`,
RELAX/ETKF/GPR), not results tables — those were already censused today
(`tmp/REVALIDATION_CENSUS_2026-07-23.md`) and three scale-dependent constants were already found
(`tmp/SCALE_AUDIT_fable_2026-07-23.md`: `grow_residual_threshold`, `min_dk`, `lambda_l1`) and the
`min_dk` regime trade-off re-measured on finemesh (`tmp/MIN_DK_RESULT_2026-07-23.md`). None of
that is repeated here. This document found **two new FATAL prose/mechanism errors** (one in each
paper — `paper/wavesplat.tex` and `docs/COMPARE_wavesplat_vs_waveprior_2026-07-09.md`), one new
MATERIAL data-contaminated design choice (the hyperparameter-tuning episode), and confirms several
design choices are **sound as built** (say so plainly, per instructions) — the `mmse_bank_estimate`
/ `prior_net_v3` annulus band and the GPR kernel re-fit are both correctly scale-invariant, and
neither WavePrior's training pipeline nor `mmse_bank_estimate` ever touches the raycast mesh at
all, so the mesh regeneration is a structural no-op for that whole arena (confirming, from a
different angle, `REVALIDATION_CENSUS`'s classification).

---

## Ranked findings

| # | Finding | Method | Class | Severity |
|---|---|---|---|---|
| 1 | `paper/wavesplat.tex:1936` "beamforming detection threshold" at `\|k\|≳0.4 rad/m` is **measured false** — the beamformer has no such threshold; the number is the old mesh's rendering cutoff, mislabeled as an algorithm property | WaveSplat | prose/mechanism | **FATAL** |
| 2 | `docs/COMPARE_wavesplat_vs_waveprior_2026-07-09.md:66` attributes WavePrior's HOS win to learning "compensating for bound waves" — **contradicted by the project's own same-day/next-day results** (bound waves proven unlearnable by Isserlis' theorem; neither arm's bank even models bound waves in this comparison) | WavePrior | prose/mechanism | **FATAL** |
| 3 | WaveSplat's four global hyperparameters (lr, λ₁, τ_prune, τ_grow) were tuned on `BF3_5comp`, an episode measured 77% variance-under-resolved on the shipped mesh; the "generalizes across BF2–5, is itself evidence of low sensitivity" argument (`paper/wavesplat.tex:1580–1598`) was validated entirely inside the missing-short-λ regime | WaveSplat | design/justification | **MATERIAL** |
| 4 | Recurring "wavenumber-space crowding / beamforming resolution limit" attribution for ≥15-comp episodes (4 locations) is mostly fine (min_dk audit already shows the wall survives finemesh at ~95% magnitude) but never names the mesh as a co-contributor (~21–22% var-under-resolved on these episodes) | WaveSplat | prose | MATERIAL/COSMETIC |
| 5 | The paper's single largest headline margin (2-component, ~6×) sits on the two episode classes measured **most** mesh-contaminated (BF2 1comp and BF3 2comp both var_under=1.00) — amplifies, does not duplicate, `REVALIDATION_CENSUS`'s already-top-ranked item 3 | WaveSplat | cross-reference | (see census — already scheduled) |
| 6 | `mmse_bank_estimate` / `prior_net_v3` annulus band (0.35–2.5·kp), bank spacing (kp/10), and val-selected shape hyperparameters are dimensionless/scale-invariant by construction, and the training/eval pipeline never touches the raycast mesh — **sound as designed**, and the mesh regeneration is a structural no-op here | WavePrior / bank | design | sound (no severity) |
| 7 | GPR baseline's `length_scale=20.0` is only an sklearn optimizer seed, re-fit per episode by marginal likelihood — **sound as designed**, not a stale constant | GPR baseline | design | sound (no severity) |
| 8 | ETKF `n_ensemble=50` vs. the repo's own sweep favoring 100 (0.358 vs 0.416) — already flagged as the actual critical-path blocker in `REGEN_SPEC`/census; not re-derived here | ETKF-grid | design | cross-reference only |

---

## 1. WaveSplat — `paper/wavesplat.tex:1936`, the beamforming "detection threshold" is not real (FATAL)

**The claim**, `sec:limitations` ("Tracking phase dependence on initialization"):

> "In moderate sea states where high-wavenumber components (\|k\|≳0.4 rad/m) fall below the
> beamforming detection threshold, tracking nRMSE can be 2–3× higher than the post-initialization
> nRMSE."

This is the *general mechanism* statement in the paper's own Limitations section — it is not a
one-off failure-case aside like the L1553 BF2 case; it is invoked to explain a systematic
degradation pattern.

**MEASURED, mesh-independent, direct test of the claim** (`wavesplat/tmp/beamform_detection_vs_k_probe.py`):
reuses a real episode's actual LiDAR sample geometry (`data/episode_BF4_15comp_seed0_uniform.h5`,
init window = 141 frames, ~50.6 pts/frame) but **replaces the observed z with an analytically
synthesized single plane wave** (`z = A cos(kx·x+ky·y−ωt+φ) + N(0, 0.02²)`), bypassing
`raycast_surface`/the mesh entirely, then calls the *actual* production
`initialize_from_frames_v2` (`wavesplat/optimizer/initializer.py:342`) with `n_gaussians=1`.
Swept `|k| ∈ {0.15, ..., 1.00}` rad/m at fixed A=0.15 m:

| \|k\|_true | recovered dist from true (kx,ky) |
|---|---|
| 0.15 | 0.0065 |
| 0.40 | 0.0041 |
| 0.50 | 0.0035 |
| 0.75 | 0.0036 |
| 1.00 | 0.0066 |

No degradation anywhere near 0.4, or at all, up to the search bound (kx_range=(-4,4) rad/m, which
`SCALE_AUDIT` already confirmed is never binding — max true \|k\| across all 195 shipped episodes
is 1.006).

**MEASURED, companion amplitude sweep** (`wavesplat/tmp/beamform_detection_vs_amplitude_probe.py`),
fixed at \|k\|=0.5 rad/m (inside the claimed failure band), A swept 0.15→0.01 m (below the 0.02 m
per-point noise floor): **every case FOUND**, dist=0.0035 rad/m at every amplitude, A_ratio
0.99–1.09. The beamformer coherently integrates ~7,050 observations matched to the true dispersion
relation, so an isolated component is recovered almost perfectly at any amplitude tested — the
"detection threshold" the paper attributes to \|k\| is not a resolution or SNR wall of the
algorithm in isolation.

**Where the number actually comes from.** \|k\|≳0.4 rad/m ⟺ λ≲15.7 m. This is essentially the
same boundary the mesh-regeneration task itself names (λ≈15 m) as where the *shipped mesh's own*
amplitude transmission collapses — `tmp/REVALIDATION_CENSUS_2026-07-23.md`'s oracle-k probe
measured 4–30% transmission below λ≈15 m on the old mesh vs. 90–99% on the fine mesh. Put together
with the direct test above (which rules out an intrinsic algorithmic cause), the conclusion is that
whoever wrote this sentence observed real detection failures above \|k\|≈0.4 on the shipped
episodes and attributed them to "the beamforming detection threshold" when the actual cause (on
that data) was the mesh suppressing the signal to near-invisibility before the beamformer ever saw
it — **the same class of error as the already-known L1553 case, in a different location, feeding a
different claim** (tracking-phase degradation, not the BF2 failure case).

**What this does and does not affect.** It does not overturn the *existence* of a genuine
crowded-regime wall (separately confirmed by `min_dk` on finemesh data) — it specifically falsifies
the *mechanism* attributed to amplitude/wavenumber-threshold detection failure in isolation. The
"2–3× tracking nRMSE" empirical number itself was not re-measured here (out of scope/budget); only
the causal explanation offered for it.

**Severity: FATAL** for the sentence as written — it is a directly measured-false mechanism claim
in the paper's own limitations section, the section meant to be the most careful about what is and
is not known.

**Corrected statement:**

> "Components with wavelength below ≈15–16 m were rendered by the shipped raycast mesh at 4–30% of
> their true amplitude (data_finemesh_2026-07-23 regeneration, measured directly). A
> mesh-independent synthetic test of the beamforming initializer alone (isolated component,
> realistic sample geometry) shows no detection failure at any \|k\| up to 1.0 rad/m or any
> amplitude down to 0.01 m — the initializer's own resolution/SNR floor is not the mechanism. The
> tracking-phase degradation this section describes should be re-measured on data_finemesh_2026-07-23
> before being attributed to a beamforming property; on the shipped data it is most likely a
> restatement of the mesh's own rendering cutoff."

---

## 2. WavePrior — `docs/COMPARE_wavesplat_vs_waveprior_2026-07-09.md`, the bound-wave mechanism story is contradicted by the project's own later results (FATAL)

**The claim** (§2, "읽는 법", the doc's stated *reason* for WavePrior's reported HOS win, 3/3
seeds):

> "**HOS 비선형에서 뒤집힌다** (WP 3/3승, ...). **속박파·비선형 상호작용이 단봉 자유파 가정을
> 깨뜨리는 만큼 학습 prior가 벌충한다.**"
> ("It flips on HOS nonlinear seas [WP wins 3/3] ... **bound waves and nonlinear interactions
> break the single-free-wave assumption exactly as much as the learned prior compensates for it.**")

This is the paper/companion-doc's headline mechanism for why learning wins — the exact class of
claim the task calls out ("learning compensates where the analytic prior's assumptions break").

**MEASURED (code): neither arm in this comparison models bound waves at all.**
`scripts/d3_compare_gifs.py:233–238` — the actual script behind these numbers — calls:
```python
fq_ws = mmse_bank_estimate(P, z, sc['depth'], Pq, sigma0=s0, kp_hint=sc['kp'], seed=sc['seed'])
fq_wp = net_gamma_mmse(net, P, z, sc['depth'], Pq, sigma0=s0, seed=sc['seed'], kp_hint=sc['kp'])
```
Neither call passes `bound_wave=True`. Both `mmse_bank_estimate` (`scripts/d3_sa_nomp.py:499–502`)
and `net_gamma_mmse` (`scripts/d3_prior_net_eval.py:56–58`) default `bound_wave=False`, meaning
**both** the analytic prior (WaveSplat) and the learned prior (WavePrior) are restricted to the
free-wave-only annulus bank (0.35–2.5·kp) in this exact comparison. Neither can represent bound-wave
harmonics here, by construction.

**MEASURED (code): `prior_net_v3` was never trained on any bound-wave/nonlinear content.**
`scripts/d3_ssl_pretrain.py:4–5`, in the project's own words:
> "Every learned prior we have (**prior_net_v3**, the field nets, wiener2) saw only synthetic seas:
> linear superposition JONSWAP, Gaussian by the central limit theorem."
Confirmed independently by reading `scripts/d3_prior_net_data.py`: samples are generated via
`jonswap_components` + direct analytic superposition (`z = (Am·cos(psi)).sum(1) + noise`), the
docstring stating "HOS fraction... default 0 for the v1 linear batch."

**MEASURED (the project's own formal result, same day, 8 hours later):**
`docs/BOUNDWAVE_LEARNABILITY_2026-07-09.md` (git commit `e146e8e`, **2026-07-09 18:36**, vs.
COMPARE.md's single commit `2b783ee` at **2026-07-09 10:22** — `git log --follow` confirms
COMPARE.md has never been revised since) derives, via Isserlis' theorem, that a 2nd-order bound-wave
functional's correlation with any linear readout of a Gaussian free-wave field is bounded by ρ²,
measures this bound fit at **+0.997**, and concludes "속박파는... 학습 불가다" (bound waves are
unlearnable). Its follow-on, `docs/STORM_WALL_AND_UQ_GATE_2026-07-09.md` (commit `1d41c19`,
**21:37–21:38 the same day**), explicitly retracts the "storm = model-class wall" framing that the
bound-wave/learning-compensation story depends on. `docs/LIMITATIONS_ledger_2026-07-10.md` §B4
records this as an established project limitation: **"B4. 구속파(bound wave)는 학습 불가다."**

**Put together:** COMPARE.md's mechanism sentence was written before the project's own same-day
work showed (a) the comparison as actually coded cannot exercise bound-wave representation in
either arm, (b) the net never saw any bound-wave content in training, and (c) the project's own
formal analysis concludes bound waves are not learnable from this observation class in principle.
The sentence has sat uncorrected in the repo since.

**What this does and does not affect.** The WIN/LOSS ranking (WavePrior wins 3/3 of 3 HOS seeds,
per §1's table) is not challenged here — that is a separate empirical claim on n=3, already flagged
by the doc itself as too small to call significant. What is wrong is the **causal explanation**
offered for that ranking. This is squarely the failure mode the task names in item 4: "if the
assumption-breaking content was not rendered [here: not modeled by either arm, and not
learnable in principle], that mechanism story needs re-examination even where the win/loss ranking
survives."

**Severity: FATAL** for the mechanism sentence — not underpowered, not stale, but actively
contradicted by the project's own higher-confidence, same-day formal result.

**Corrected statement:**

> "WavePrior's apparent advantage on HOS scenes cannot be attributed to representing bound-wave
> harmonics: neither arm's bank models bound waves in this comparison (`bound_wave=False`, both
> calls, `d3_compare_gifs.py:233–238`), `prior_net_v3` was never trained on any bound-wave content
> (`d3_ssl_pretrain.py:4–5`), and the project's own Isserlis'-theorem result
> (`BOUNDWAVE_LEARNABILITY_2026-07-09.md`) shows such content is not learnable in principle from a
> linear readout of this observation class. If the 3/3 ranking survives a larger-n re-run, the
> mechanism is more likely a better-shaped learned prior over the *free-wave* spectrum's
> shape/gamma/directional structure under nonlinear energy redistribution among free-wave modes —
> not bound-wave absorption — and should be described as such, or left as an open mechanism
> question pending a dedicated ablation (e.g., re-run with `bound_wave=True` symmetrically on both
> arms)."

---

## 3. WaveSplat — the hyperparameter-tuning episode was itself the most data-contaminated regime tested (MATERIAL)

**The claim**, `paper/wavesplat.tex:1580–1598` (`sec:hyper`):

> "WaveSplat's four hyperparameters (lr=0.01, λ₁=10⁻³, τ_prune=0.10, τ_grow=0.05) were selected by
> a single coarse sweep on **BF3 5-comp** and held fixed across all episodes... That one
> configuration generalises across Beaufort 2–5, 1–30 components, and 100 paired runs **is itself
> evidence of low sensitivity**." ... perturbation sweep run "on **the full 12-episode subset**...
> No perturbation changed the relative ranking."

These four values are exactly the shipped `DEFAULT_CONFIG` in `wavesplat/run.py:23–39` and
`configs/estimation.yaml` — not illustrative, the literal production defaults.

**MEASURED** (`wavesim/mesh_adequacy.py`, shipped mesh config `nx=40, ny=30`,
`gt_grid.x_range=(10,250), y_range=(-100,100)` from `configs/default.yaml:22–28`), run against
`data/episode_BF3_5comp_seed0_uniform.h5`'s actual `config/wave_components`:

```
lambda_peak=30.2 m, lambda_min=6.25 m, pts_per_lambda_peak=4.38, pts_per_lambda_min=0.91
var_under_resolved=0.773, mesh_verdict=INADEQUATE
```

77% of this episode's true field variance sits in components the shipped mesh cannot resolve
(< 6·dx). Its shortest component, λ=6.25 m, is the exact wavelength
`REVALIDATION_CENSUS`'s oracle-k probe separately measured at **4% amplitude transmission on this
mesh, 83% on the fine mesh** — i.e. on the episode used to fix these four numbers for the entire
project, most of the true short-λ content the estimator was optimizing against was, in practice,
mesh facet noise, not signal.

For context, a spot-check of the same criterion (`wavesplat/tmp` script run inline) across
`data/episode_*_uniform.h5` (n=110, uniform+slope, excludes fixedT variants) finds only 7/110
(6%) ADEQUATE, 53 DEGRADED, 50 INADEQUATE — the tuning episode's contamination is not an outlier
within this dataset, but it is the *specific* episode this section's justification leans on.

**What this does and does not mean.** It does not mean lr=0.01/λ₁=0.001/τ_prune=0.10/τ_grow=0.05
are wrong — `SCALE_AUDIT`/`MIN_DK_RESULT`'s independent re-sweeps of the related `min_dk` and
`grow_residual_threshold` constants on the finemesh data reconfirmed the same qualitative behavior
(same direction, similar magnitude), which is some reassurance that this pipeline's sensitivity to
these particular knobs is not wildly different on richer data. But the **specific stated
justification** — "generalizes across Beaufort 2–5... is itself evidence of low sensitivity" — was
established entirely within a regime that was, at the time, missing most of its own short-λ
content on every episode tested, including the one the sweep was run on. "Generalizes across
Beaufort 2–5" cannot be read as "generalizes to realistically-rendered short-wavelength content,"
because that content was not present when the generalization claim was checked. This has not been
re-swept on `data_finemesh_2026-07-23`.

**Severity: MATERIAL.** Not FATAL — the values are plausibly still reasonable defaults and nothing
here shows they are badly wrong — but the sensitivity/robustness *argument* offered for trusting
them is invalid as written, and no re-sweep exists yet.

**Corrected statement:**

> "WaveSplat's four hyperparameters were fixed by a single coarse sweep on BF3 5-comp, an episode
> later measured to be 77% variance-under-resolved by the shipped raycast mesh
> (`var_under_resolved=0.773`, `wavesim/mesh_adequacy.py`). The reported robustness sweep (±2×
> perturbations on the 12-episode subset) was likewise run entirely within this contaminated
> regime. Independent re-sweeps of two related scale constants (`min_dk`, `grow_residual_threshold`)
> on the corrected `data_finemesh_2026-07-23` set reconfirmed similar qualitative behavior, which is
> partial reassurance, but the specific "low sensitivity, generalizes across Beaufort 2–5" argument
> in this section has not itself been re-run on adequate data and should not be cited as validated
> until it is."

---

## 4. WaveSplat — recurring "wavenumber-space crowding" attribution (MATERIAL/COSMETIC, mostly cross-reference)

Four locations attribute the ≥15-component performance ceiling and directional-R² collapse purely
to beamforming/algorithmic crowding: `paper/wavesplat.tex:987–988` ("spatial Nyquist limit of the
beamforming initializer"), `:1205` (fig. caption, "beamforming resolution limits"), `:1481`
("initialization resolution limit as beamforming peaks crowd"), `:1807–1810` ("wavenumber-space
crowding in beamforming").

**Cross-check against the mesh.** `wavesim/mesh_adequacy.py` on `data/episode_BF4_15comp_seed0_uniform.h5`
and `..._30comp_seed0_uniform.h5`: `var_under_resolved = 0.216` and `0.206` respectively (DEGRADED,
not INADEQUATE) — the ≥15-comp regime is measurably **less** mesh-contaminated (≈21–22% of
variance affected) than the low-component headline episodes in finding 5 below (100%). This is
consistent with, and adds a second independent measurement supporting, the already-existing
`min_dk` sweep's finding (`tmp/MIN_DK_RESULT_2026-07-23.md`) that the crowded-regime wall survives
on finemesh data at ~95% of its original magnitude — i.e., this attribution is largely correct in
direction, unlike finding 1 above.

**What's still imprecise.** `REVALIDATION_CENSUS_2026-07-23.md` §"TOP 3" item 3 already flags that
"spatial Nyquist limit" is ambiguous between the beamformer's own resolution and the simulator
mesh's resolution, and that the paper does not distinguish them; my check here confirms the
attribution is *mostly* about the beamformer (only ~21% mesh-driven), but the text as written still
never names either mechanism explicitly, and a reader cannot currently tell which one is meant.

**Severity: MATERIAL/COSMETIC** — a wording fix (name `min_dk` and/or crowding explicitly, per the
census's own recommendation), not a substantive retraction. Not re-deriving the census's finding;
adding the variance-fraction numbers as corroboration.

---

## 5. WaveSplat — the headline 2-component margin sits on the most contaminated episodes (cross-reference, not new)

`paper/wavesplat.tex:1500–1502` ("the most striking result is the ~6× average advantage on
2-component seas... WaveSplat 0.057 vs. ETKF 0.327") and Table `tab:main`'s largest per-episode
margins are BF3 2-comp (U/S) and BF2 1-comp.

**MEASURED**: `episode_BF2_1comp_seed0_uniform` var_under_resolved=1.00 (already the known L1553
case), and **`episode_BF3_2comp_seed0_uniform` var_under_resolved=1.00 as well** — both of the two
episode classes carrying the paper's single largest quoted margin are, by this metric, 100%
variance-under-resolved on the shipped mesh (not merely "affected," essentially the whole field is
below the mesh's resolving power for these low-component cases, since with only 1–2 components a
larger share of the total is at the field's own peak-adjacent short flank).

**This is not a new independent claim** — `REVALIDATION_CENSUS_2026-07-23.md` §3 item 2 already
identifies the 1–2-comp improvements as needing the still-ungenerated `_pk` (peak-locked) companion
before any mesh-effect magnitude can be assigned, and already recommends against citing them as a
mesh effect until that exists. This entry only adds that BF3 2-comp specifically (previously
examined only as the `min_dk` sweep's "no true pair closer than 0.15" *control* episode, a
different question) is *itself* fully mesh-contaminated by the variance criterion — reinforcing,
not superseding, the census's existing top-priority item.

**Severity:** not independently rated — folds into the census's already-scheduled top item.

---

## 6. `mmse_bank_estimate` / `prior_net_v3` — bank design is sound as built (positive finding)

Checked directly in `scripts/d3_sa_nomp.py:499–537` and `scripts/d3_prior_net_train.py:31–34`:

- **Annulus band** `(0.35, 2.5)` is defined in **units of kp** (the episode's own estimated peak
  wavenumber), not an absolute rad/m constant — scale-invariant by construction, shared identically
  between the analytic bank and the net's training label mask. This is exactly the kind of fix the
  companion `SCALE_AUDIT` recommends for WaveSplat's own absolute-unit constants; here it was
  already done correctly.
- **Bank spacing** `dk_c = kp/10` (docstring: "scale-invariant ~1900 candidates") — same pattern,
  sound.
- **Shape hyperparameters** (`pexp ∈ {0.5,1,2}`, `smult ∈ {1,3,10}`) are **val-selected per episode**
  (held-out RMSE), not fixed global constants baked in from one tuning run — not exposed to the
  "tuned once on old data" failure mode at all.
- **Data provenance**: confirmed by reading `scripts/d3_prior_net_data.py` and
  `scripts/d3_compare_gifs.py` — neither the net's training generator nor its HOS/linear
  evaluation scene ever calls `run_episode`/`LidarSimulator`/`raycast_surface` or opens a `.h5`
  episode file; both synthesize observations directly from `jonswap_components`/`HOSSimulator`.
  **The mesh regeneration is a structural no-op for this entire method** — confirms, from direct
  code inspection rather than the import-based classification, `REVALIDATION_CENSUS`'s placement
  of this arena as mesh-independent.

**One narrower, lower-confidence (ARGUED, not separately measured) scoping note**: for the
shortest-Tp end of the net's own training distribution (Tp~U[5,11]s ⟹ kp up to ~0.16 rad/m), the
band's upper edge 2.5·kp caps out at λ≈15.6 m at Tp=5s — a similar-looking number to the mesh's
λ≈15m cutoff, but for an unrelated, intentional reason (model-class truncation of the bank, not
mesh rendering). This means even the `bound_wave=True` extension (candidates at 2·k) would not
reach bound harmonics of the shortest free-wave tail components (which sit near k≈2.5kp already).
Not independently measured here (would require a dedicated bound-wave-energy-vs-band-coverage
calculation); flagged for completeness, not rated as a finding.

**Severity: sound as designed.** No contamination pathway found; stated per instructions because a
clean result is as reportable as a broken one.

---

## 7. GPR baseline — kernel length-scale is self-calibrating, not stale (positive finding)

`wavesplat/baselines/gpr_baseline.py:31–58`: `length_scale=20.0` is only the **initial value**
passed to `sklearn.gaussian_process.kernels.Matern`; `GaussianProcessRegressor(..., 
n_restarts_optimizer=2)` re-optimizes it by marginal likelihood on every episode's own observations
at fit time (bounds `[1.0, 200.0]` m). This is not a constant "tuned once on old data" — it
re-derives itself per episode regardless of what spatial correlation structure the (old or new)
mesh produces.

**Severity: sound as designed.** No action needed.

---

## 8. ETKF-grid `n_ensemble=50` — already the project's own top blocker, not re-derived

`wavesplat/baselines/etkf_grid.py:28`: default `n_ensemble=50`. `tmp/REVALIDATION_CENSUS_2026-07-23.md`
§3 item 1 already identifies this as unresolved (the repo's own sweep found Ne=100 better, 0.358 vs
0.416) and states no fine-mesh ETKF number may be cited until the mechanism is understood. This is
the single biggest live blocker in the project's own accounting, ahead of anything in this document
— included here only so the "methods in scope" list in the task is not silently missing it, not as
a new derivation.

---

## What was not (re-)audited here, and why

- RELAX / greedy+joint matching-pursuit baseline: spot-checked only far enough to confirm (via
  `REVALIDATION_CENSUS`'s "Alt-methods gate NEGATIVE" entry and the shared `d3_sa_nomp.py` module
  structure) that it lives in the same analytic-sampler arena as `mmse_bank_estimate`, hence is
  structurally unaffected by the mesh regeneration. Its own hyperparameters (`max_comp`, `val_frac`,
  IRLS annealing schedule) were not individually re-derived — no evidence surfaced of a
  mesh-contamination pathway to look for, and the task asked for modest computation.
- The three `SCALE_AUDIT` constants (`grow_residual_threshold`, `min_dk`, `lambda_l1`) and the
  `min_dk` regime trade-off: intentionally not repeated, per the task's explicit instruction.
- ETKF `n_ensemble` mechanism and any fine-mesh ETKF number: intentionally not re-derived; it is
  already the project's own top-ranked open blocker.
- The magnitude of the 1–2-component headline margin's mesh sensitivity: intentionally not
  re-measured (would require the still-ungenerated `_pk` companion and multi-seed runs on
  `data_finemesh_2026-07-23`, both already queued by the census and by the concurrent 100-episode
  regeneration job sharing this machine).

## Files produced

- `wavesplat/tmp/beamform_detection_vs_k_probe.py` — mesh-independent |k|-sweep detection probe
  (finding 1).
- `wavesplat/tmp/beamform_detection_vs_amplitude_probe.py` — companion amplitude sweep at fixed
  |k|=0.5 (finding 1).
- No production file was modified. All `mesh_adequacy` checks in this document were run read-only
  against existing `data/*.h5` files using the shipped `configs/default.yaml` mesh geometry
  (`nx=40, ny=30`, `x_range=(10,250)`, `y_range=(-100,100)`).
