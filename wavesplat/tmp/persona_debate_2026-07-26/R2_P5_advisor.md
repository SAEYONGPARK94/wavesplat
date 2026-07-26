# R2 — P5 (pragmatic advisor)
## Concessions
- P3 right, I was wrong: wrote "defensible today" while repeating 3/3 + mean line — the Ch.2 failure mode reproduced by the person proposing Ch.2. This debate goes INTO Ch.2 as a documented instance.
- P2: "insurance" dead → "interlock"; "real sea, synthetic sensor" everywhere incl. abstract.
- P1: concede the ladder figure — **program_journey.png must start at 0.1145, not 0.1990** (claiming credit for fixing our own bug on the headline figure is indefensible).
- P4: router flag moves off "GT-free selection" onto the inversion + pre-solve descriptor.
- NEW (nobody raised): P3's oracle violation (0.17982 vs 0.18053) = aggregation-order inconsistency → "closes 2/3 of oracle gap" is UNSUPPORTED until one script computes both the same way. Fixed inside sprint item 1.
## Revised sprint — 15 working days
1 (d1) Pre-state all bars in writing before any data: held-out agg ≤0.1011; paired Wilcoxon vs both pures α=0.05 two-sided; sparse ≤ pure_v2+0.005; oracle recomputed lane-consistent. Non-inferiority endpoints RETIRED (zero-inflated ritual).
2 (d2-5) Held-out capstone: 40 fresh episodes, constants frozen (8/0.05/1.5), ONE SHOT.
3 (d6-7) Blind auto on 30 real windows vs hand-everywhere. [+P3's one-hour bf_peak_ratio pre-check with written prediction]
4 (d7) Record-blocked re-analysis of real primary (permutation over 5 records) — zero compute.
5 (d8) BS_2011/BS_2013 out-of-sample detector specificity, τ frozen.
6 (d9-10) Observable-coordinate phase diagram: (learned−analytic) delta vs bf_peak_ratio (proposal-independent), colored by n_components, 30 real windows overplotted. Pre-state: if bf_peak_ratio predicts the delta ≥ as well as n_components → not a relabel (tests P3 circularity); if n_components dominates → P3 right, Ch.5 says so.
7 (d11-13) Arm swap, ANALYTIC side only: BLASSO/SFW replaces greedy beamform on 20 crowded + 20 sparse, median-of-3. (The arm we claim fails is the sharp one to swap.)
8 (d14) Integrity sweep: 3/3→2/3, delete mean line, sensor relabel, fix "bank-MMSE" scope mismatch.
9 (d15) 12-episode HOS slice (4/4/4), frozen, one shot, scope check — DROP FIRST if slipping.
OUT with cost/benefit: FNO (2-4wk min, obs-distribution matching prerequisite + geometry retrain already failed once; zero benefit to Paper 1, total to Paper 2 → hard GATE on Paper 2, not sprint item). DETR learned-side swap (in the documented ~40-arch FAIL bucket; failure uninterpretable). ETKF/GPR as boundary arms (field-space assimilators, no init slot; not well-posed; keep as context row). Full 100-ep HOS ladder (1-2wk compute) → replaced by item 9.
## Crux H — sequencing
Paper 1 (corrected WaveSplat) FIRST to Ocean Engineering, drafted in parallel (writing isn't compute-bound). Decisive: no learned arm → no ML reviewer → FNO rejection sentence can't be written against it. Two conditions (P2's): (a) not find-and-replace — ship with a METHODS NOTE describing the double phase advance as a general pitfall in sequential phase-resolved assimilation (embarrassment becomes second contribution; LIT-CHECK: warm-start phase error, EnKF-HOS drift); (b) Paper 1 must NOT claim the boundary (one footnote). Consequence: boundary can't borrow Paper 1's authority → Ch.5 stands entirely on the held-out run.
## Crux F
Yes — days, existing data (explore_beamform.json has all 100; _auto_bf_peak_ratio logged; computable on every real window). Kills "axis unmeasurable outside the simulator"; does NOT kill "axis only varied in a simulator" (real windows cluster in a narrow band — show it honestly; the slide draws its own limit).
## Ch.7 spec (merged with P2)
Title: "Real seas, synthetic sensor: a confirmed prediction, a blind router, and a fault interlock."
Claim 1: prediction confirmed by a loss (medΔ −0.057, 6/27, p=0.005, rank-biserial −0.608; pooled + record-blocked).
Claim 2: router, blind — whatever item 3 returns, with the descriptor-plane figure.
Claim 3 (verbatim discipline): frozen monitor τ=1.5 detects 2/3 collapse windows, 0/27 false alarms design records, [X/N] on untouched BS; on detected windows learned reduces field error 54-67% (2.31→1.07, 2.93→0.96), converting confidently-wrong into detected, safely-degraded. **Does not restore usable accuracy: end-to-end 1/3 below the usability line; the one usable restoration (1.12→0.79) is missed. Evidence = the margin (2.40/2.53 vs normal max 1.20), not a p-value.**
Abstract, not buried: synthetic sensor on real surfaces (virtual fan over WASS, GT same reconstruction, no raydrop/dropout/pose; no real LiDAR observed); off design point; n=3; τ in-sample. Deleted everywhere: "3/3", "0.881→0.775", "insurance".
## Verdicts
Q1 revised: boundary spine right, but WITHDRAW "defensible today" → defensible AFTER the sprint, and only if Ch.5's headline changes from "one estimator beats both pure arms" to **"the regime boundary is decidable from the observations alone, at a measured cost against an oracle"** (survives a held-out miss; the first doesn't).
Revised thesis statement: (as R1, plus) "...and on real sea surfaces the same observation-only signal inverts the usual hybrid: the learned arm becomes the fault interlock that protects the physics arm, not the reverse."
