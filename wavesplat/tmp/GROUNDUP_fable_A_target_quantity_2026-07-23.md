# Ground-up re-examination — Reviewer A: THE TARGET QUANTITY

**Date**: 2026-07-23
**Assignment**: question the assumption that the deliverable is "the phase-resolved
surface elevation field, and how accurately we can estimate it." Find the most
valuable question/quantity this asset could produce instead — or report a clean null.
**Hard constraint**: 13 directions are pre-eliminated (better-estimator headline, CRB/
identifiability/envelope, atomic-norm/BLASSO, estimator-family swap, learned priors,
nonlinear/HOS-as-contribution, forecast/streaming/latency, calibrated UQ, contribution-type
pivots (benchmark/simulator/systems/protocol), control-value demos, active sensing,
trajectory/scan-policy design, GS-branding/gridless-ness-as-novelty). See task prompt for
exact wording; every rejection below is cross-checked against this list explicitly.

## 0. What the asset actually is (verified by reading code, not the README's claims)

- `wavesim/config.py` / `wavesplat/wavesplat/model/spectral_gaussian.py`: the estimator
  fits `z(x,y,t) = Σ wᵢ·cos(kxᵢx + kyᵢy − ωᵢt + φᵢ)` — a **linear, vertical-elevation-only**
  model. `spectral_gaussian.py` docstring is explicit: *"This linear model has NO horizontal
  displacement — only vertical elevation... Q-constraint is structurally inapplicable here."*
  There is no orbital-velocity, no sub-surface kinematics, no breaking/whitecap model
  anywhere in the forward model.
- `wavesplat/README.md` "Scope and Limitations": linear only, no breaking, **ship motion
  currently fixed position** (the main 100-episode dataset is a *static* ship — motion/
  encounter dynamics live only in the separate, already-null hydrofoil demo).
- No physical LiDAR and no physical ship exist in this project. The only "real" data are
  third-party **stereo-camera** sea-surface records (`data/real_bergamasco/`: AA_2014,
  AA_2015, BS_2011, BS_2013, LJ_2018, YS_2017 — Guimarães/Bergamasco IFREMER set).
- `wavesplat/evaluation/metrics.py` already computes, per frame: RMSE_z, nRMSE, Hs_error,
  Tp_error, Spectral_R², Directional_R², RMSE_near/far/unobserved, Variance_Ratio, Pearson ρ,
  Skill_Score. I.e. bulk/spectral summary statistics are **already a first-class output**,
  not something that needs to be invented.
- `hydrofoil/disturbance.py` (the already-eliminated, already-null control demo) already
  evaluates surface elevation **and slope (∂η/∂x) and vertical velocity (∂η/∂t)** at the foil
  location — i.e. "derived local kinematic quantities" are already computed and consumed
  inside the exact experiment whose null result is pre-eliminated (see §2, candidate C4).
- `hydrofoil/wavesplat_bridge.py` converts WaveSplat's (w, kx, ky, φ, ω) directly into the
  hydrofoil sim's wave-component format — confirming the parametric output is already
  general enough to drive any linear derived quantity (slope, velocity, group velocity,
  even depth-decayed orbital velocity) essentially for free; nothing here is technically hard.
- The repo's own `docs/strategic_repositioning.md` (12-paper deep read, §31) already scores
  the project against a competitor matrix and explicitly marks **"Breaking, slamming, green
  water" = ❌ for us, ✅ only for VOF-EnKF (Yan et al. 2025)** — i.e. the project has already,
  independently, scoped structural-load-relevant quantities *out* because the linear model
  cannot support them.
- `wavesplat/tmp/` contains **>250** prior probe scripts and the user's MEMORY.md indexes
  ~40 already-closed research arcs (CRB, atomic-norm, control value, active sensing, kp-floor,
  regime axes, real-anchor, cross-estimator, UQ...). This is a mature, heavily-searched idea
  space, not a fresh field.

## 1. Method

For each candidate: (a) state the claim precisely, (b) check it against the 13-item exclusion
list word-for-word, (c) WebSearch the closest prior art, (d) if it survives both filters,
give the cheapest kill experiment.

## 2. Candidates considered

### C1 — Bulk/directional sea-state parameters (Hs, Tp, Tz, mean direction, spread) as the headline deliverable instead of the field
**Claim would be**: "the field is over-ambitious; what actually matters operationally is the
low-dimensional summary, which this sparse-LiDAR asset can deliver faster/cheaper than the
full field."
**Reject — precedented, saturated, and already produced as a byproduct.**
- This is *exactly* the deliverable of the marine-radar wave-sensing industry (WaMoS-type
  X-band radar systems) and of the "wave buoy analogy" ship-response literature
  (Nielsen et al.): "The most important information extracted from a sea state is the
  directional wave spectrum, indicating wave direction, significant wave height, and wave
  spectrum peak period" — [Estimation of sea state parameters by the wave buoy analogy](https://www.sciencedirect.com/science/article/abs/pii/S0029801820307587),
  [The wave buoy analogy — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S014111870800031X),
  [Estimating waves via measured ship responses — Sci Rep 2023](https://www.nature.com/articles/s41598-023-44552-2).
- Also delivered from moving/underway platforms already:
  [Wave measurements using open-source ship-mounted ultrasonic altimeter — one-ocean
  circumnavigation (2023)](https://www.sciencedirect.com/science/article/pii/S0029801823029700),
  [Voluntary Observing Ship program](https://en.wikipedia.org/wiki/Voluntary_observing_ship_program),
  CFOSAT SWIM satellite mean-wave-period product.
- The asset's own `evaluation/metrics.py` already reports Hs_error/Tp_error/Spectral_R² per
  frame — this is not an un-taken quantity, it's an existing secondary output being
  re-proposed as a headline. Doing so would just be C1's own better-estimator claim on a
  coarser target (falls back into eliminated item #1) with a crowded, decades-old competitor
  field on top.
- One genuine wrinkle: Nielsen's wave-buoy-analogy line *needs the vessel to move in
  response to the waves* (it inverts wave spectra from RAO-filtered motion). A foiling/
  motion-suppressed vessel (the hydrofoil in this very repo) structurally breaks that
  method — that's a legitimate reason a *direct-sensing* channel is needed for such hulls.
  But that's an argument for *why LiDAR sensing exists at all* (already implicit in the
  project's stated niche), not a new target-quantity contribution, and it still collapses to
  C1's already-rejected bulk-parameter deliverable.

### C2 — Local wave steepness / crest kinematics ("slamming/green-water risk indicator") as the target quantity, replacing elevation RMSE
**Claim would be**: "elevation nRMSE is the wrong headline metric; downstream structural-risk
consumers need crest steepness/curvature, which may be badly estimated even when nRMSE looks
good (low-N, smoothness-regularized components systematically flatten peaks)."
**Reject — four independent reasons, each sufficient alone:**
1. **Already inside the codebase, already null.** `hydrofoil/disturbance.py` already computes
   wave slope (∂η/∂x) and vertical velocity (∂η/∂t) at the point of interest as part of the
   pre-eliminated, already-null control demo. This isn't an untested derived quantity.
2. **Structurally can't be validated for its stated purpose.** The motivating use
   (slamming/green-water design loads) is a mature, dedicated naval-architecture discipline
   requiring nonlinear/CFD or high-fidelity impact modeling — see
   [Finding Dangerous Waves — Review of Methods to Obtain Wave Impact Design Loads for
   Marine Structures, ASME JOMAE 2023](https://asmedigitalcollection.asme.org/offshoremechanics/article/145/6/060801/1157525/Finding-Dangerous-Waves-Review-of-Methods-to)
   (van Essen & Seyffert): impact loads require extreme-value theory over rare nonlinear
   events, not a linear steepness readout. A linear model's steepness number has no
   established causal link to impact loads without that machinery — the "why it matters"
   story collapses on inspection.
3. **The project already scoped this out, explicitly, in writing.** `docs/strategic_repositioning.md`
   §31.2.3 marks "Breaking, slamming, green water" ❌ for this method and ✅ only for
   VOF-EnKF (Yan, Che, Li 2025) — the project's own 12-paper literature pass already reached
   this conclusion.
4. **Reframed as "add a metric to the evaluation panel," it is the excluded "evaluation
   protocol" contribution-type pivot** (exclusion #9), not a target-quantity pivot, no matter
   how it's dressed up.
Not worth a kill experiment — it's dead by inspection/literature/self-admission before any
code needs to run.

### C3 — Real-time hazard classification / rogue-wave-in-footprint flag (decision rule, not an estimate)
**Claim would be**: "the valuable output is a binary 'is there a dangerous wave in the sensed
footprint right now' flag, not a continuous field — cheaper to calibrate, sufficient for an
operator alarm."
**Reject — collapses into the forecast exclusion, and is separately precedented.**
- Every real instantiation of this idea in the literature is phrased as *early warning*, i.e.
  forecasting: "high-resolution radar or LIDAR sensors watching wave movement... potentially
  providing detection minutes before rogue waves form" —
  [rogue-wave early-detection discussion](https://www.popsci.com/now-ships-can-get-advance-warning-monster-rogue-waves/);
  see also [Early Detection of Rogue Waves Using Compressive Sensing](https://arxiv.org/pdf/1602.00816),
  [Detection of Elusive Rogue Wave with Cross-Track InSAR, Sensors 2025](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12074428/),
  [Machine-guided discovery of a real-world rogue wave model, PNAS 2023](https://www.pnas.org/doi/full/10.1073/pnas.2306275120).
  A *purely-nowcast* (no lead time) version has no operational value — by the time a wave is
  "in the footprint now," a ship traveling toward it has essentially no time to act, so any
  version worth having is a forecast, which is exclusion item #7 (claimed by the companion
  encounter_forecast paper already in this repo: `paper/encounter_forecast/`).
- A "decision rule" framing on top of the field still needs the field to be trustworthy at
  the flagged location, so it also imports the operability-envelope/UQ questions —
  exclusions #2 and #8.

### C4 — Different consumer: subsurface orbital/particle velocity at foil/keel depth, for structural or foil-load monitoring
**Claim would be**: "the model already parametrizes (w, k, ω) per component; linear wave
theory gives orbital velocity at any depth via cosh/sinh(k(z+d)) for free — report *that*,
not surface elevation, since it's the physically relevant driver for a submerged
foil/hull."
**Verified as genuinely unused**: `hydrofoil/disturbance.py` and `foil.py` were grepped for
`cosh`/`sinh`/`orbital` — zero hits. The disturbance model uses surface η, ∂η/∂x, ∂η/∂t only;
true depth-decayed orbital velocity is not computed anywhere in the repo. So, unusually among
these candidates, this one is *not* already tried.
**Reject anyway — it is a reword of an eliminated item, and the rules say reword = failed
answer.** Its entire "why it matters" is: it drives loads/forces on a submerged structure
(the hydrofoil) — i.e., a control/structural-value argument on the exact null demo in
exclusion #10, or a lead-time/forecast argument if framed for anticipatory control (#7). There
is no way to motivate "why does this quantity matter" without walking straight back into one
of the barred items. Also, as in Alkarem's floating-wind-turbine digital-twin work
([WindEurope poster coverage](https://composites.umaine.edu/2024/04/08/forecasting-the-future-rudy-alkarems-award-winning-research-on-predictive-digital-twins-in-renewable-energy/);
underlying method e.g. [Uncertainty-Guided Prediction Horizon of Phase-Resolved Ocean Wave
Forecasting Under Data Sparsity, JMSE 2025](https://doi.org/10.3390/jmse13071250)), the
"reconstruct near-field kinematics to drive a control/digital-twin loop" pattern is itself
already a populated niche — just on a different platform (floating wind, not ships).
Flagging for completeness, not proposing it.

### C5 — Different consumer: feed a moving "virtual buoy" wave/spectrum product into external data assimilation (numerical wave-model nowcast, e.g. WW3/SWAN spin-up)
**Claim would be**: "the right consumer isn't the ship itself — it's an external forecast
system that needs sparse in-situ wave observations along shipping lanes."
**Reject — precedented, and orthogonal to what makes this asset distinctive.**
- This is literally the purpose of the Voluntary Observing Ship program and of underway
  altimeter/IMU wave-measurement systems feeding "wave prediction improvement... through
  assimilation schemes based on measured wave data at specific locations" (search results
  above; also CFOSAT SWIM). The LiDAR-vs-radar-vs-altimeter sensor choice doesn't change the
  target-quantity story (still bulk/spectral summary → DA), so it reduces to C1 with an
  extra external-system dependency this project has no way to validate (no real link to
  WW3/SWAN assimilation exists in the asset, and building one would be a new contribution-type
  pivot — systems/benchmark — exclusion #9).

### C6 — Impossibility claim not phrased as CRB: "phase-resolved field accuracy is ill-posed for a moving platform without externally supplied current/Doppler"
**Checked against the project's own memory first** (required by the "no rediscovery" spirit
of this task): MEMORY.md's "GP Space-Time 결과" entry already documents this exact result —
blind current recovery failed 5/5, and the established fix is "외부 Uc 공급" (external current
must be supplied). This is not a new finding; it is closed, cited, internal knowledge. It is
also, on inspection, an identifiability statement about a specific nuisance parameter — i.e.
CRB/information-wall flavored regardless of the words used, so it would fall under exclusion
#2 even if it were new.

### C7 — Sensor-selection framing: "what sea-state envelope can *this* commercial LiDAR usefully observe, as input to a procurement decision between LiDAR/radar/camera for a new vessel"
**Reject** — this is the operability-envelope question (exclusion #2) wearing a "who buys the
sensor" hat, and the asset has no calibrated radar or camera forward model to compare against
(only the unrelated third-party stereo dataset, which isn't LiDAR-comparable). Dead on
arrival for lack of asset support, redundant with #2 even if it weren't.

### C8 — Categorical/coarse sea-state classification (Beaufort bin) instead of continuous Hs/Tp
Considered briefly: marginal information loss relative to C1, no new "why it matters" story
beyond C1's (voyage/performance-monitoring logging, ISO 15016-style, already served by
existing weather-routing vendors using non-LiDAR inputs). Folds into C1's rejection.

## 3. What was *not* investigated further, and why (asset limits, stated up front per instructions)
- Hull/wake-based relative-motion inference from the point cloud near the vessel: the
  simulator (`wavesim/`) has no hull, no ship geometry, no wake model — there is nothing in
  the asset to ground this in; would require building a new simulation capability, which
  tips into the excluded simulator/systems pivot (#9) even before asking whether it's a good
  idea.
- Anything requiring a second real sensor for cross-calibration/fusion: no real LiDAR, no
  real co-located radar/IMU exists (constraint stated in the task; verified — `real_data.py`
  only ingests the stereo dataset, no multi-sensor real record exists).

## 4. Bottom line

**No candidate survives.** Every reformulation of "what should the output be" that is not
already barred by the 13-item exclusion list turns out to be either (a) a well-established,
saturated deliverable in an adjacent, decades-old sub-field — bulk/directional sea-state
parameters (wave-buoy-analogy, WaMoS-class radar, VOS/altimeter programs, satellite SWIM), or
(b) a re-derivation of something this exact codebase has already computed and already found
null (slope/velocity inside the hydrofoil control demo), or (c) un-motivatable without walking
back into a barred item by the time you ask "why does anyone want this number" (structural
loads → control/forecast; hazard flag → forecast; sensor-selection → envelope; DA feed →
systems pivot), or (d) already closed and documented in this project's own memory (current/
Doppler identifiability). The project's own internal competitor-matrix
(`docs/strategic_repositioning.md` §31) had independently reached the same scoping decisions
years before this review for at least two of these (bulk-parameter competitors catalogued;
slamming/loads explicitly marked out of scope) — this review's conclusion is a confirmation,
not a discovery.

This is consistent with, not a departure from, the project's already-decided PhD spine
(CRB-anchored operability-envelope characterization, per MEMORY.md's resume header): under
the target-quantity lens specifically, there is no rescue pivot available, and the honest
recommendation is to **not** add a target-quantity reframing to the paper. If a metrics-panel
addition is wanted for robustness/reviewer-defense reasons (e.g. reporting a steepness/
gradient decoupling check as a footnote-level limitation, in the same spirit as the existing
Variance_Ratio/Skill_Score "reviewer-defense" metrics), that is legitimate ordinary
methodological hygiene — but it is not a headline, and dressing it up as one would be exactly
the kind of manufactured-to-have-one proposal this task explicitly asked not to produce.
