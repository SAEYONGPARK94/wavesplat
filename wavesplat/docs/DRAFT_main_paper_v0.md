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

2.1 Observation class: forward fan ±60°, 20–100 m, 10 FPS, ~48
returns/frame (median of the spec-faithful simulation incl. raydrop),
2 cm noise; exogenous straight course; earth-fixed estimation (forward
speed invariance); latency budget 100 ms.

2.2 Encounter-frame predictable zone: ℓ\* ≈ R_max/(U+c_g); demand-side
scaling (encounter period shrinks with U as preview demand does) —
Fig. `fig_horizon` (flat 0.91 through the FF band; decay shoulder at
~4.5 s; collapse beyond ℓ\*=6.0 s).

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

3.1 Bank MMSE (no identification): beamform-power prior, validation-
selected hyperparameters, val-gated EM. 3.2 Weight-space equivalence:
batch MMSE ≡ RLS with prior (derivation; machine-precision assert;
unit-tested). Finite memory as AR(1)-toward-prior with group-coherence
interpretation. 3.3 Two-timescale architecture: per-frame filter +
slow-loop γ refresh (rebuild); NIS as health signal. 3.4 Measured
compatibility: 18 ms median / 37 ms p95 per frame (solo, 48 pts/frame,
CPU), 10-min drift-free, inline gate evaluation 0.8 s in the slow loop.
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

4.1 Spec-faithful simulation (episodes; native bench: 2–4 s forecast ≡
nowcast). 4.2 HOS M=3 (0.94–0.96 at U=3; design point Fig. horizon;
nonlinearity costs nothing at BF2–BF5 steepness). 4.3 Real seas (config: m=12 returns/frame unless noted): **five**
stereo sites zero-shot (best 0-10 s band per
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

5.1 The window-skill bimodality. 5.2 Suspect elimination (currents via
Doppler sweep ≤+0.03; directional spread already matched; GT white
noise floor ≈0; naive drift metrics null; budget null). 5.3 Directional
bimodality as a *marker* (patch −0.84, onboard beamform −0.78 on the
discovery site) that fails to transfer (+0.13 holdout) and fails causal
reproduction in linear crossing seas → site-specific failure modes.
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

6.1 E2E protocol (C-8 simulator, paper scenarios). 6.2 With the
original controller: GT-FF −6–16 %, stream recovers 19–47 %. 6.3
Controller-side lever: preview 0.5→0.2 s doubles the GT margin
(consistent with the companion paper's 0.1–0.3 s optimum); stream
fraction unchanged → the slope ceiling of §2.4 is the single remaining
bottleneck, and de-weighting slope does not help (the margin is not
recoverable by ignoring the bad channel). 6.4 Requirement inversion:
corr is necessary, not sufficient; the FF benefit is (slope
information) × (controller design). No sensor-side sampling lever for
slope was found (the adjacency projection was retracted after the
testbed audit); raising the slope tier requires range/noise-class
changes, left as an open requirement. One concrete sensor-side lever
does exist in the shipped hardware and is untested here: the Falcon's
configurable ROI mode (40°×4.8° at 0.09°×0.08° angular resolution,
manual Table 2) concentrates the beam budget into the look-ahead
corridor — a density, not adjacency, intervention; our budget sweep
suggests modest slope gains (+0.04 per 4×), so expectations should be
set accordingly.

## 7. Limitations and Requirements for the Next Tier

Raydrop model v1; pose frame-correlation untested (decoupled-mount
assumption); no own-vessel field data — validation strategy is
third-party real seas + spec-faithful simulation (defended in §4.5);
gate at preliminary tier; 22 kt not directly testable on ≤240 m stereo
patches (role split: real data validates the operator, HOS validates
the design speed). Requirement inversion table: lead ↔ range; slope ↔
adjacency; gate confirmation ↔ larger-patch or field records.

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
  ±5 cm Lambertian accuracy [bias, unmodeled — limitation note];
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
