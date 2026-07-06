# DRAFT v0 — Control-Compatible Real-Time Wave Encounter Forecasting with a Given Forward-Looking LiDAR: Information Limits, Streaming Equivalence, and Self-Diagnosed Reliability

(2026-07-06 v0. English draft; numbers wired to
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

**Thread.** [Para 1] Fast foiling vessels ride through waves they can,
in principle, see coming... FF control needs η (and its gradient) at
the foils 0.1–0.5 s ahead [hydrofoil companion paper: 11–41 %
heave-acceleration reduction with perfect preview]. The question is not
whether forward-looking wave sensing can serve control—patents
[US10259544, US10538299], the FutureWaves radar system, LiDAR DSWP
[OE 2023], and classical preview servos [OE 2004] establish the
concept—but **where the information in a specific, purchasable
observation ends**, whether a real-time estimator can reach that end,
and whether it can detect, online, the sea states in which it fails.

[Para 2 — prior art, honest] Radar DSWP sea trials report encounter
correlations of ~0.76 [OE 2020]; CN-RLS [RS 2024] applies recursive
estimation to coherent radar; predictable-zone theory [Naaijen et al.]
bounds phase-resolved forecasting by group-velocity geometry. What is
missing from this literature is (a) a quantified design verdict for a
given sensor×vehicle pair, (b) measured real-time latency for a
phase-resolved forecaster, (c) zero-shot validation on real seas, and
(d) any form of online self-diagnosis.

[Para 3 — contributions C1/C2/C3 as in outline; state the negative/
limit propositions as first-class results.]

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
3.5 Relation to CN-RLS/DSWP (table): what is shared (recursive form),
what is new (equivalence proof, latency measurement, prior slot,
zero-shot real-sea transfer). 3.6 Learned priors: amortized net as a
slot — parity zero-shot; retraining tracks the oracle gap prediction
(evidence that learning cannot beat the information boundary; AHGP
cited).

## 4. Validation (three legs)

4.1 Spec-faithful simulation (episodes; native bench: 2–4 s forecast ≡
nowcast). 4.2 HOS M=3 (0.94–0.96 at U=3; design point Fig. horizon;
nonlinearity costs nothing at BF2–BF5 steepness). 4.3 Real seas: three
stereo sites zero-shot, pooled 0.70 [0.59, 0.79]; streaming beats the
600-obs batch protocol (0.76–0.78 at 15 ms) because it has no window
cap. 4.4 Robustness table (Fig. `fig_robust`): FF-band ≥0.90 under pose
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
changes, left as an open requirement.

## 7. Limitations and Requirements for the Next Tier

Raydrop model v1; pose frame-correlation untested (decoupled-mount
assumption); no own-vessel field data — validation strategy is
third-party real seas + spec-faithful simulation (defended in §4.5);
gate at preliminary tier; 22 kt not directly testable on ≤240 m stereo
patches (role split: real data validates the operator, HOS validates
the design speed). Requirement inversion table: lead ↔ range; slope ↔
adjacency; gate confirmation ↔ larger-patch or field records.

## 8. Conclusion

[Thread closed: boundary mapped (C1), reached in real time (C2), its
detection prototyped and honestly graded (C3), and the next boundary
priced in sensor terms.]

---
### TODO (v1 편집 시)
- [x] 확증 2호 결과 반영 (5.5) — 확증 실패, 방향 일관
- [ ] CN-RLS 대비 표 작성 (3.5)
- [ ] 인용 목록: 특허 2, FutureWaves, Naaijen predictable zone, OE
      2020/2023, CN-RLS 2024, preview servo 2004, AHGP, Bergamasco/
      Guimarães 데이터, hydrofoil companion
- [ ] Fig 배치: horizon / robust / gate_roc / stream demo 스틸 + GIF
      부록 링크
- [ ] 탐색/확증 라벨 전수 (§5 표기 방식을 전 섹션에)
