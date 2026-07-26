# R1 — P2 (senior ocean engineering professor) — opening position

[Full text preserved from agent output]

## The steelman
This is the most disciplined experimental hygiene I have seen from a student in years. Four outcome cells frozen before the real-data run, including the unfavourable one — and the unfavourable one landed and was reported (real accuracy LOSS, 6/27, p=0.005). Three retraining axes closed as honest negatives. A do-not-cite list maintained against the student's own past overclaims. Median-of-k after discovering 53% CV. Most submissions I referee would have shipped the mean-based "real-data win" (+0.063) that this student forbids himself from citing. That instinct is worth more than any of the numbers. And the measured ladder is real: 0.1990 → 0.1145 → 0.0875, oracle 0.0818.

## Q1 — novelty by OE standards
NOT novel: estimator core = textbook DSWP; learned prior occupied (Ehlers FNO, PINO); CRB/predictable-zone spine pre-empted (Fucile LEPrE, Naaijen, 2024-25 tau-trimming).
GENUINELY NOVEL (one item): a phase-resolved reconstruction system that selects its own estimator per window from observations alone, with (i) per-regime non-inferiority, (ii) measured false-alarm rate, (iii) single shipped configuration. Long validity-flag culture exists (X-band quality masks, buoy QC), but no phase-resolved system routing between physics and learned initialization on a GT-free observable with reported operating characteristic. LIT-CHECK-NEEDED: "quality indicator" deterministic sea wave prediction; adaptive estimator selection wave field reconstruction; X-band radar wave inversion quality control mask.
SECOND (underrated): observation-distribution leak = sensor-realism finding — learned wave priors must be trained on the deployment scan geometry, or synthetic gains are sampling-pattern artifacts.
THIRD: tracker bug — own artifact; becomes contribution only if error class (phase advanced in both warm-start and forward operator) is shown as a general pitfall in sequential phase-resolved assimilation. LIT-CHECK-NEEDED: warm start phase error sequential wave assimilation; EnKF-HOS phase drift.

## Real-data story judged hard
MEASURED: 30 windows, 5 records, held-out, zero-shot, prereg. Accuracy LOSS. Repairs 3/3. Detector 2/3 fired, 0/27 false, mean 0.881→0.775.
Objections by severity:
1. **It is not LiDAR.** Virtual forward fan (12 pts/frame, ±60°, cap 600, 2cm white noise) resampling a WASS stereo surface, GT from the SAME reconstruction (4m grid) — common-mode error cancels. No raydrop, incidence dropout, whitecap loss, range bias, pose error. "Real sea with a synthetic sensor" — say exactly that; if the thesis says "real LiDAR validation" I stop the defense. (source: scripts/d3_compare_gifs.py:144 protocol)
2. **Not the design point.** U=1.2–2.4 m/s vs design 11.3 m/s; aperture/encounter/Doppler all change at 22kt; design-point evidence is HOS-only.
3. **THE LEDGER (fatal for the word "insurance").** The two fired windows repair to 1.07 and 0.96 — at/worse than still-water predictor. The one window repaired into usable territory (1.12→0.79) is MISSED. As measured, the guard converts confidently-wrong into harmlessly-useless, never unusable into usable. It is a fault-detection safe-degrade INTERLOCK, not accuracy insurance. Framed as interlock: defensible and valuable. Framed as insurance: I take it apart.
4. **n=2.** Sensitivity 2/3, 95% CI ~ unit interval; specificity 0/27 → rule-of-three ~11% upper bound; threshold designed in-sample (declared). What saves it is the MARGIN: fires at 2.40/2.53 vs normal max 1.20, τ=1.5 in an empty gap. Report the gap, delete significance language.
5. **Windows not independent within record** — effective n nearer 5 than 30. Re-run primary as record-blocked permutation or mixed model.

Committee will demand: (a) one transit of real scanning LiDAR from a moving hull (30 min at 6-10kt) with independent truth (buoy + wave staff); or (b) pier/tower LiDAR + buoy to validate return statistics; (c) CHEAP AND DECISIVE: freeze τ=1.5, run untouched BS_2011/BS_2013 records as genuinely out-of-sample specificity test — on disk already, an afternoon of work.
Integrity item: OE paper ships 0.223/0.199 while branch measures 0.1145 with bug off. Cannot defend citing contaminated numbers. Resolve before writing.

## VERDICTS
Q1: PARTIAL PASS — one genuine OE novelty (observation-only regime router with measured operating characteristic, single config) + one sensor-realism finding (scan-geometry dependence). Estimator/learned prior/CRB = instrumentation, not claims.
Q2: PASS conditionally — as a measurement-systems doctorate about regime boundaries and safe degradation, NOT "learning beats physics". Conditions: HOS re-run of headline ladder; BS held-out detector test; sensor-realism caveat in abstract.

Strongest thesis statement:
> A shipborne phase-resolved wave estimator becomes deployable not by winning everywhere, but by inferring from the observations alone which regime it is in: this thesis measures the physics/learning regime boundary under pre-registration, builds a single self-routing estimator that matches the better specialist in every regime, and shows on five real sea records that the same observation-only signal acts as a fault interlock that turns the analytic path's catastrophic reconstruction failures into detected, safely degraded ones.

## TOP 3 CRUXES
1. **Interlock or fig leaf?** Detector's two catches repair to ~1.0 (useless), the one useful repair is missed: deployable safety contribution or decoration on a null?
2. **Does a linear-GT headline survive reality?** 0.0875 ladder is linear-GT, HOS arm unrun; program already watched +9% evaporate at the observation gap. Physics result or testbed result?
3. **Router: science or plumbing?** Boundary never tested against a third estimator (ETKF, GPR, FNO). Does the boundary survive a fourth arm, or is it a property of this pair?

Files cited: docs/HANDOFF_OPUS_2026-07-26.md, tmp/unified_2026-07-26/fallback_results.json, tmp/wp_real_2026-07-25/PREREG.md, scripts/d3_compare_gifs.py:144.
