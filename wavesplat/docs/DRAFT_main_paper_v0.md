# DRAFT v1 — Control-Compatible Real-Time Wave Encounter Forecasting with a Given Forward-Looking LiDAR: Information Limits, Streaming Equivalence, and Self-Diagnosed Reliability

(2026-07-06 v1 — v0 + 감사 정정(oracle 갭≈0, 상한 0.65, 인접성 철회)
+ 서론/결론 완문 + CN-RLS 표 + 탐색/확증 라벨 + 인용 목록. English draft; numbers wired to
`docs/REPRODUCIBILITY_2026-07-06.md`. Gate = "preliminary finding" tier
per pre-registered confirmations #1/#2. Thread sentence of the whole
work: **"Where does the information in a given forward-looking wave
observation end — and can a real-time system reach that boundary, know
it, and specify what lies beyond?"**)

## Abstract (draft)

Wave-preview feedforward control of fast foiling vessels requires a
phase-resolved forecast of the encountered surface elevation a fraction
of a second to a few seconds ahead, computed within a hard real-time
budget. We ask what a *given* commercial forward-looking LiDAR (Seyond
Falcon-class: 120°×25° FOV, 10 Hz, ~48 usable returns per frame over a
20–100 m forward fan, 2 cm ranging noise) can deliver on a *given*
vessel (a Candela C-8-class hydrofoil at 22 kt), and whether the system
can know, online, when its forecast is trustworthy. Three results. (i)
*Information limits*: applying predictable-zone theory in the encounter
frame gives a usable lead ℓ\* ≈ R_max/(U+c_g) ≈ 6 s at the design
point; within the feedforward band (0–2 s) the encounter-forecast
correlation is 0.91 (95% CI [0.88, 0.94], HOS M=3, n=12 seas) and is
robust to pose error, crest shadowing, oblique headings, and sea states
BF2–BF5. A ladder of oracle experiments (true-spectrum prior, 4×
observation budget, 12× learning scale, bank refinement, bound-wave
correction) shows the estimator is saturated within the examined model
class: at realistic return density the estimator matches a
true-spectrum oracle (gap ≈ 0), and the most consequential residual—a
slope-estimation ceiling of ≈0.65—is a property of the observation
class, not the algorithm. (ii) *Streaming equivalence*: the bank-limited minimum
mean-square-error estimator is shown to be exactly a recursive
least-squares filter with a spectral prior; the streaming form matches
the batch solution to machine precision, runs at 18 ms median (p95
37 ms) per 10 Hz frame on a CPU, is stable over 10-minute runs, and
transfers zero-shot to real stereo-reconstructed seas (pooled encounter
correlation 0.70, 95% CI [0.59, 0.79], three sites). (iii) *Self-
diagnosed reliability*: real-sea skill is bimodal across 20 s windows
(half near synthetic-level 0.85–0.95, half at 0.4–0.5); a forensic
elimination of the usual suspects (currents, spread, GT noise,
observation budget) leaves site-specific failure modes, and a site-
adaptive quasi-held-out-residual gate (hyperparameter selection is
held out; the prior shape is not — see Methods)—thresholded against the vessel's own
running statistics—raises conditional skill from 0.49 (ungated) to 0.68
at 50 % availability in exploratory analysis. Two pre-registered
confirmations reproduce the gate's direction and effect size on unseen
data but not its rank significance; we therefore report the gate as a
preliminary finding and release the pre-registrations. We close by
inverting the analysis: the information boundary itself specifies the
sensor requirements for the next capability tier (range for lead; for
slope, no sampling-structure lever exists within the sensor's return
budget—an early positive result on an artifact testbed was retracted
after a matched-conditions audit).

## 1. Introduction

A fast foiling vessel rides through waves that its own sensors can, in
principle, see coming. A wave-preview feedforward (FF) controller needs
the encountered surface elevation—and, ideally, its slope and vertical
velocity—at the foil positions a fraction of a second ahead, computed
within a hard real-time budget; with perfect preview, simulated
feedforward reduces heave acceleration by 11–41 % on a Candela
C-8-class hydrofoil [companion]. The *concept* of forward-looking wave
sensing for vessel control is not new: it is covered by patents
[US10259544; US10538299], realized commercially by radar-based ship
motion forecasting [FutureWaves], demonstrated with LiDAR-based
deterministic wave prediction [OE 2023], and preceded by classical
preview servos for hydrofoil craft [OE 2004]. What none of this prior
art provides is an answer to the question that actually decides a
design: **for a specific, purchasable sensor on a specific vessel,
where does the information in the observation end — and can a
real-time system reach that boundary, detect when it fails, and state
what capability lies beyond it?** That single question threads this
paper.

Radar DSWP studies report encounter/reconstruction correlations in the
0.7–0.9 range [OE 2020; RS 2024 — VERIFY exact per-paper values against
full texts before submission]; CN-RLS [RS 2024] applies recursive least
squares to coherent radar reconstruction; predictable-zone theory [Naaijen et al.] bounds
phase-resolved forecasting by group-velocity geometry. Against this
literature our gaps are concrete: (a) no published *quantified design
verdict* for a given sensor×vehicle pair; (b) no *measured* real-time
latency for a phase-resolved forecaster; (c) no *zero-shot* validation
on real seas (methods are typically tuned on the data they report);
(d) no *online self-diagnosis* of forecast reliability.

We answer the threaded question in three parts, and we state our
negative results with the same prominence as the positive ones.
**(C1) The boundary, mapped.** Applying predictable-zone reasoning in
the encounter frame gives a usable lead ℓ\* ≈ R_max/(U+c_g); at the
design point (22 kt, Beaufort 4) this is ≈6 s, comfortably above the
0.1–0.5 s the FF controller consumes, and the skill-versus-lead curve
measured on nonlinear (HOS M=3) seas tracks the prediction. A ladder
of falsification experiments—true-spectrum oracle, observation-budget,
learning-scale, bank-resolution, bound-wave, and sampling-structure
interventions—shows the estimator is saturated within the examined
class (oracle gap ≈ 0 at realistic return density), with the slope
channel information-limited at ≈0.65. **(C2) The boundary, reached in
real time.** The bank-limited MMSE estimator is exactly a recursive
least-squares filter with a spectral prior; its streaming form matches
the batch solution to machine precision and runs at 18 ms median per
10 Hz frame, drift-free over 10-minute runs, transferring zero-shot to
real stereo seas. **(C3) The boundary, self-detected — honestly
graded.** Real-sea window skill is strongly heterogeneous; a
site-adaptive residual gate raises conditional skill from 0.49 to 0.68
at 50 % availability in exploratory analysis, but two pre-registered
confirmations reproduce only its direction, not its significance, so
we report the gate as a preliminary finding and publish the
pre-registration trail.

## 2. Problem Setting and Information Analysis (C1)

**2.1 The observation class.** The sensor is taken as given: a
Falcon-class forward-looking LiDAR (specifications verified against the
manufacturer's manual, Table 2: 120°×25° FOV, 0.18°×0.24° angular
resolution, 250 m detection range at 10 % reflectivity, 2 cm (1σ)
ranging precision, 10 FPS). Mounted on a foiling vessel and directed at
the sea surface at grazing incidence, its usable product is a forward
fan of returns spanning roughly 20–100 m ahead within ±60° of the
course. The number of surviving returns is not a free parameter but a
physical outcome of reflectivity-driven dropout: our spec-faithful
simulation yields a median of 48 usable returns per frame, spatially
clustered near sensor-facing crests (nearest-neighbour median ≈2 m).
The vessel's trajectory is exogenous — we forecast for a course, we do
not design one — and estimation is performed in earth-fixed
coordinates, which renders the estimator forward-speed invariant by
construction. The computational budget is 100 ms per 10 Hz frame,
set by the flight controller.

**2.2 The encounter-frame predictable zone.** Phase-resolved forecasts
are bounded by group-velocity geometry [Naaijen et al.]: information
observed at range R arrives at the vessel no later than the group
transit time. In the encounter frame of a vessel advancing at U into
head seas, the wave energy that will be met at lead ℓ is currently at
range (U+c_g)ℓ, so the usable lead of a sensor with maximum water
range R_max is ℓ\* ≈ R_max/(U+c_g) — about 6 s at the design point
(22 kt, BF4). Critically, the *demand* scales the same way: the
encounter period shrinks with U, and the feedforward controller's
preview requirement (0.1–0.5 s [companion]) sits an order of magnitude
inside ℓ\*. Fig. `fig_horizon` shows the measured skill-versus-lead
curve on HOS M=3 seas (12 seeds): flat at 0.91 through the FF band, a
decay shoulder near 4.5 s, and collapse beyond the predicted ℓ\*=6.0 s.
The boundary is where the physics says it should be.

2.3 The limit ladder (each rung = one falsifiable "the gap is X"
hypothesis, each rejected):
| rung | intervention | Δ skill (clean testbed, n=2000 comps) |
|---|---|---|
| spectrum prior | oracle (true binned spectrum) | ≈0 (achieved 0.652 vs oracle 0.648, slope) |
| observation budget | 12→48 pts/frame (real data) | ~0 |
| learning scale | amortized prior 4k→50k samples | ~0 (redistributes) |
| bank resolution | dk/2 | ~0 |
| nonlinearity | 2nd-order bound-wave correction | 0 (ka≈0.025) |
| sampling policy | pair-preserving subsample (real pattern) | −0.20 |
| sampling structure | idealized adjacent clusters (same budget) | −0.14 (hurts) |
→ saturation *within the examined class*, now anchored by a **zero
oracle gap at realistic return density**. Audit note: an earlier +0.13
adjacency result was traced to a 200-component testbed artifact
(discrete spectral lines make the sea artificially hard and invert the
oracle); with ≥2000 components the linear and HOS testbeds agree and
the adjacency advantage disappears — coverage dominates any fixed
return budget. Sampling structure is therefore NOT a viable slope
lever; this retraction is documented in the audit appendix.

2.4 The slope ceiling: η 0.94–0.98 / slope 0.65–0.68 (amp 0.6–0.7) /
vertical velocity 0.85–0.88 at the design point, consistent across the
HOS and (clean) linear testbeds; true-spectrum oracle matches the
achieved slope → the quantity FF control consumes is
information-poorest and already at its examined-class ceiling.
(Sets up §6.)

## 3. A Streaming Estimator that Attains the Limit (C2)

**3.1 Bank MMSE without identification.** The field is represented on
a fixed bank of ~1,900 wavenumber cells (annulus 0.35–2.5 k_p, spacing
k_p/10), each carrying a cosine/sine pair whose time dependence is
fixed by the dispersion relation. No component identification is
attempted — the 200-mode identification programme of our earlier work
consistently lost to a prior-weighted linear estimator. The prior
variance per cell is shaped by the beamform power of the observations,
its two shape hyperparameters selected on a 25 % held-out split, and
refined by validation-gated EM. Prediction at any space-time query is
then linear in the observations.

**3.2 Exact streaming equivalence.** Because the dispersion phase is
inside the design row, the bank coefficients are *static* for a
stationary linear sea, and the batch MMSE posterior mean is exactly a
recursive least-squares filter with prior P₀ = diag(γ): process each
frame's returns as a rank-m update. We verify the identity to machine
precision (max deviation 3×10⁻¹⁴; unit-tested in CI). Finite memory
enters as AR(1)-toward-prior dynamics, w←aw, P←a²P+(1−a²)diag(γ) with
a = exp(−Δt/τ) — a proper linear-Gaussian model whose stationary law
is the prior, physically read as a finite wave-group coherence time.
One numerical caveat proved load-bearing: after hundreds of rank-m
downdates the covariance loses symmetry and skill silently collapses;
a per-step symmetrization P←(P+Pᵀ)/2 removes the failure (regression-
tested).

**3.3 Two-timescale architecture.** The per-frame filter never touches
its own prior; a slow loop (every ~10 s) may re-estimate γ from a
trailing buffer and rebuild (w, P) by one batch solve — state transfer
without drift. The per-frame normalized innovation squared (NIS) is
exposed as a health signal, and the estimator's held-out residual is
exposed as the (preliminary) reliability gate of §5.

**3.4 Measured control compatibility.** Under the realistic return
density (48/frame) and with no concurrent load, the filter plus
forecast costs 18 ms median / 31 ms p95 per frame at the design point
and 17.6/37 ms on real-sea data (CPU, numpy) — a factor ≈5 inside the
100 ms budget; a 10-minute continuous run shows no skill or covariance
drift; the slow-loop gate evaluation costs 0.8 s and is asynchronous
by construction.
3.5 Relation to prior recursive wave estimators:

| aspect | CN-RLS [RS 2024] | DSWP (radar) | **this work** |
|---|---|---|---|
| observation | coherent radar image | X-band image seq. | given fwd LiDAR fan (48 pt/frame) |
| recursion | RLS on image basis | batch per window | RLS ≡ batch MMSE (proof + machine-precision test) |
| prior | condition-number reg. | spectrum fit | beamform-power γ + val-select/EM; learned-prior slot |
| latency | not reported | not reported | **18 ms med / 37 ms p95 measured** |
| real-sea validation | same-data tuning | sea trial (0.76) | **zero-shot, 4+ sites** |
| self-diagnosis | — | — | preliminary gate + pre-registrations | 3.6 Learned priors: amortized net as a
slot — parity zero-shot; retraining tracks the oracle gap prediction
(evidence that learning cannot beat the information boundary; AHGP
cited).

## 4. Validation (three legs)

**4.1 Spec-faithful simulation.** The first leg uses episodes generated
by a LiDAR simulator configured to the manufacturer's specification,
including reflectivity-driven raydrop. On these native observations,
2–4 s forecasts cost nothing over the nowcast — the dispersion
relation carries the phase for free at short lead — establishing the
baseline property the whole design relies on.

**4.2 Nonlinear seas.** The second leg replaces the linear sea with
HOS M=3 simulations. At BF2–BF5 steepness (ka ≈ 0.01–0.04) the linear
estimator loses nothing measurable: encounter skill is 0.94–0.96 at
U=3 m/s and 0.91 in the FF band at the design point (Fig. horizon),
and an explicit second-order bound-wave correction changes skill by
exactly zero — the bound-harmonic amplitude (~mm) is far below the
error floor. Nonlinearity is not a binding constraint in the design
envelope.

**4.3 Real seas.** The third leg is zero-shot transfer (config: m=12
returns/frame unless noted): **five**
stereo sites (best 0-10 s band per
site): YS17 0.71, AA15a 0.71, LJ18g 0.62 (storm swell, λ≈300 m ≫ fan —
narrowband temporal coherence compensates spatial under-resolution and
gives the best long-lead skill, 0.44 at 10–20 s), AA15b 0.57, AA14
0.43 (43 % valid pixels, GT-starved); development-pair pooled 0.70
[0.59, 0.79]. Streaming beats the
600-obs batch protocol (0.76–0.78 encounter corr at 15 ms/frame,
m=12 config; at the realistic m=48 density: 0.74–0.75 at 17.6 ms —
all configurations labeled in the reproducibility appendix) because it
has no window cap. Site heterogeneity is reported,
not averaged away. 4.4 Robustness table (Fig. `fig_robust`): FF-band ≥0.90 under pose
0.05–0.3° (white model; frame-correlated left as limitation), crest
shadowing (3 m sensor height), headings 30–60°, BF2/BF5. 4.5 What the
real GT can and cannot support (stereo cross-contamination note).

## 5. Real-Sea Failure Forensics and a Preliminary Reliability Gate (C3)

**5.1 The phenomenon.** Real-sea skill is not a number but a
distribution: across 20 s windows, roughly half sit at synthetic-level
0.85–0.95 and half at 0.4–0.5, with occasional negative windows. Any
operational claim must face this heterogeneity rather than average it
away.

**5.2 Suspect elimination (all exploratory analyses labeled as such).**
The usual explanations fail affirmatively: an ambient-current Doppler
sweep improves nothing (≤+0.03 at the optimum); the directional spread
of the real records (25–35°) already matches the synthetic probes; the
stereo ground truth carries no measurable white-noise floor (the
surfaces are smoothed products); naive nonstationarity metrics
(ΔTp, ΔHs, Δdirection) do not rank window skill; and quadrupling the
observation budget changes nothing on real data — the failure is model
mismatch, not data starvation.

**5.3 A marker that is not a cause.** Directional bimodality of the
observed spectrum ranks window skill strongly on the discovery site
(patch-side −0.84, onboard beamform-side −0.78, p≈0.001–0.003) — but
fails to transfer to a second site (+0.13) and fails causal
reproduction: imposing linear crossing seas of matched geometry does
not collapse the estimator. Failure modes are site-specific; a
universal onboard predictor of them remains open.
5.4 The site-adaptive residual gate: definition (running median/MAD of
the estimator's own quasi-held-out residual — the validation split is
excluded from hyperparameter scoring but the beamform prior shape sees
all observations; leakage is uniform across windows and cannot create
per-window ranking bias); exploratory operating curve
0.49→0.68@50 % (Fig. `fig_gate_roc`). 5.5 **Pre-registered
confirmations** (both released with git-timestamped protocols):
#1 small-patch records — direction and split reproduced (+0.17) but
P1 failed (p=0.23) and the split CI includes zero [−0.22, +0.35];
#2 mid/large-patch (AA15b + time-offset splits) — P1 failed (−0.21,
p=0.084) and P2 failed (+0.084 < 0.10): confirmation failure by the
pre-set bars, with direction consistent across all three independent
evaluations (−0.38/−0.11/−0.21).
Verdict language: *preliminary finding*; what a confirmatory success
would require (larger patches / more sites / field data).

## 6. Control Closure (application demonstration)

**6.1 Protocol.** The forecast is wired into the companion
feedforward controller on the full vessel simulator (paper scenarios:
BF3/BF4 head seas at cruise speed), and scored in the control
currency: heave-acceleration RMS reduction relative to feedback-only.

**6.2–6.3 Results, honestly.** With the original controller settings
the perfect-preview (GT-FF) margin is itself modest (−6 to −16 %), and
the streaming forecast recovers 19–47 % of it. Tuning the controller
(preview 0.5→0.2 s, consistent with the companion paper's 0.1–0.3 s
optimum) doubles the GT margin — the earlier evaluations were run at a
sub-optimal preview — but leaves the streaming fraction unchanged at
≈20 %. De-weighting the slope term does not help either: the loss is
not avoidable by ignoring the badly-estimated channel, because the
channel carries irreplaceable disturbance information. Both
observations point at the same object: the slope information ceiling
of §2.4.

**6.4 Requirement inversion.** Encounter correlation is necessary but
not sufficient for control benefit; the benefit factorizes,
approximately, into (slope information available in the observation
class) × (controller design quality). No sensor-side sampling lever for
slope was found (the adjacency projection was retracted after the
testbed audit); raising the slope tier requires range/noise-class
changes, left as an open requirement. One concrete sensor-side lever exists in
the shipped hardware — the Falcon's configurable ROI mode (40°×4.8° at
0.09°×0.08°, manual Table 2), which concentrates the beam budget into
the look-ahead corridor. We tested the corresponding geometry
(returns concentrated in a ±20° corridor at the same budget): the
achieved slope gain is negligible (+0.01), consistent with the budget
sweep; ROI is therefore not a slope remedy either.

## 7. Limitations and Requirements for the Next Tier

Our raydrop model is specification-derived, not field-measured; the
return-adjacency statistics that would settle the sensor-side slope
question require raw LiDAR point clouds over real seas, which no
public dataset currently provides (a data request trail is included in
the supplementary material). Pose error is validated under a white
model only (frame-correlated attitude error is subsumed under a
decoupled-mount assumption); a constant ranging bias of the
specification's ±5 cm class was tested and has zero effect. There is
no own-vessel field campaign: the validation strategy is deliberately
third-party — open real-sea surface truth plus specification-faithful
simulation — and §4 argues this is sufficient for the claims made,
which are about an observation class. The reliability gate is reported
at preliminary tier by our own pre-registered standard. The design
speed itself (22 kt) cannot be exercised on ≤300 m stereo patches; the
role split is explicit — real data validates the operator, HOS
validates the speed. What the next capability tier requires is now a
specification: longer leads are bought by water range; the slope tier
is not bought by sampling structure, density concentration (ROI), or
learned priors within this sensor's budget — it requires a different
observation class; and gate confirmation requires genuinely new
mid/large-patch records or field data.

## 8. Conclusion

We asked where the information in a given forward-looking wave
observation ends, and whether a real-time system can reach that
boundary, know it, and specify what lies beyond. The boundary is now
mapped: at the design point the encounter forecast is limited by the
predictable-zone horizon ℓ\* ≈ R_max/(U+c_g) in lead, and by an
information ceiling of ≈0.65 in slope—both properties of the
observation class, established by a ladder of falsification
experiments whose strongest rung is a zero oracle gap at realistic
return density. The boundary is reached in real time: a streaming
filter provably identical to the batch estimator runs at 18 ms per
frame and transfers zero-shot to real seas at 0.70 pooled encounter
correlation. The boundary's self-detection remains open at the
standard we set for it: the site-adaptive gate reproduces its
direction on unseen data but not its significance across two
pre-registered confirmations, and we publish that trail rather than
soften the standard. What lies beyond is now a specification, not a
speculation: longer leads are bought by range; the slope tier is not
buyable by sampling structure within this sensor's return budget; and
the failure regimes that the gate must learn to see are site-specific.
For the vessel that motivated this work, the verdict is concrete —
the given sensor already saturates the band its feedforward controller
consumes, in real time, with a documented margin; the remaining
distance to the perfect-preview benefit is a property of physics and
controller design, not of the estimator.

## References (to wire)
- US10259544, US10538299 (predictive sea state mapping for ship motion control)
- Connell et al., FutureWaves™ real-time ship motion forecasting (2016)
- Naaijen et al., predictable zone for phase-resolved wave prediction
- Ocean Eng. 2020, DSWP sea trial (corr 0.76)
- Remote Sens. 2024, CN-RLS coherent radar reconstruction
- Ocean Eng. 2023, real-time phase-resolved directional prediction (LiDAR)
- Ocean Eng. 2004, optimal preview servo for fully-submerged hydrofoil
- Guimarães et al. 2020, Sci. Data — stereo sea-surface dataset (validation data)
- Bergamasco et al. 2017, WASS pipeline
- AHGP (NeurIPS 2020) — amortized spectral priors
- Companion: hydrofoil wave-preview feedforward paper (this project)
- Seyond Falcon LiDAR User Manual (2024-05-28), Table 2 Specifications,
  p.12 — all quoted sensor numbers verified against this source
  (250 m @10 % reflectivity; 120°×25°; 0.18°×0.24°; 2 cm 1σ precision;
  ±5 cm Lambertian accuracy [bias — TESTED: zero effect on skill, the
  DC-free bank plus validation-selected regularization absorb a
  constant offset; the fixed-noise oracle, by contrast, degrades];
  10 FPS; ROI mode 40°×4.8° @0.09°×0.08°)

### Figure plan
- Fig.1 concept: vessel + fan + encounter track (from stream demo GIF still)
- Fig.2 `fig_horizon` (§2.2) — n=12 seas
- Fig.3 limit ladder (table, §2.3)
- Fig.4 `fig_robust` (§4.4)
- Fig.5 real-sea window heterogeneity + `fig_gate_roc` (§5)
- Fig.6 E2E bars (§6)
- Supplementary: GIFs (stream demo, good/bad window), reproducibility ledger

---
### TODO (v2 편집 시)
- [x] 확증 2호 반영 / 서론·결론 완문 / CN-RLS 표 / 인용·그림 계획 (v1)
- [x] 사이트 확대 결과(AA15b/AA14/LJ18g) §4.3 반영 — 5사이트
- [ ] CN-RLS 대비 표 작성 (3.5)
- [ ] 인용 목록: 특허 2, FutureWaves, Naaijen predictable zone, OE
      2020/2023, CN-RLS 2024, preview servo 2004, AHGP, Bergamasco/
      Guimarães 데이터, hydrofoil companion
- [ ] Fig 배치: horizon / robust / gate_roc / stream demo 스틸 + GIF
      부록 링크
- [ ] 탐색/확증 라벨 전수 (§5 표기 방식을 전 섹션에)
