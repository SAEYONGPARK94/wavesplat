# R1 — P5 (pragmatic advisor) — opening position

Stance: one week from a defensible thesis, one bad framing from "isn't this just a combination?". Assets sufficient; risk is coherence, not evidence.
Factual correction: handoff open-decisions 1&2 already closed in code (aa8e4c3). What remains = PAPER propagation; wavesplat.tex still headlines 0.199 vs ETKF 0.332 (40.1%) and §Limitations L2183-2213 explains weakness with a tracker narrative now known to be a bug. Submitting as-is = largest self-inflicted credibility hazard.

## Q1 — rotate the thesis object from the ESTIMATOR to the BOUNDARY
The estimator is the proof the boundary is real and operational; not the claim. Converts "we tried 13 axes" into "we mapped a frontier".

8-chapter spine (title / claim / artifact):
1. Intro: which estimator, and when can you trust it? — deployable unit = POLICY, not method.
2. The instrument: a benchmark that can decide — mesh-adequate, seed-replicated, pre-registered; building it overturned 3 of our own claims. (mesh regen 20/27→27/27, median-of-k CV53%, min_dk caveat, 17/0 discard audit, 156 tests)
3. Analytic path and its true ceiling — fixed tracker: sparse essentially solved (0.016-0.027), residual error concentrated in crowding (0.24-0.27). (PHASEFIX_100EP, ETKF/LETKF finemesh)
4. Learned path: where it wins and exactly why — cond-diffusion set proposal −45% crowded; ENTIRE gain = learned kx/ky placement (ablation learned_konly≈learned p=0.94). (wp_crowded, wp_integration)
5. Boundary is GT-free detectable — one config 0.0875 on 100 prereg episodes, beats both specialists, 2/3 oracle gap, threshold-insensitive. (unified, unified_guard)
6. What does not transfer — 3 prereg retraining axes fail and LOCALIZE the cause (proposal-generation OOD; e2e obs-distribution leak; fan FAIL). (learned_only_final: 35/40 already at parity 0.0198 vs 0.0193, gap = 5 BF2_1comp episodes)
7. Real seas: prediction, loss, and insurance — prereg prediction of our own defeat CONFIRMED (6/27 p=0.005); learned real contribution = failure-mode insurance (3/3 repair, 0 own collapses, frozen detector 0.881→0.775, 0/27 false alarms). KEYSTONE: "a pre-registered prediction of our own defeat, which then happened, is worth more at a defense than a p<0.05 win."
8. Operating guidance and limits.

WHAT DIES (ruthless):
- CRB-anchored operability-envelope spine → appendix + one equation (D_eff = L + c_g·T_obs). §8 validation PARTIAL (pooled ρ+0.33 vs 0.5 bar); empirical UQ ties it in every regime (0.31≈0.30, 0.74≈0.71, 0.78≈0.78).
- Information-channel 5-element accounting (E-42→E-56) — cut.
- 16s frozen-init forecaster — cut as contribution, one paragraph in Ch.3; 10x margin now ~parity vs fixed tracker; "the symptom that located the bug" is the better story.
- B-2 task-loss (E-60: cluster-t p=0.073, headline_pass:false), all UQ claims, Doppler (one line in negatives ledger; YS17 n=1).
- Weakest surviving = Ch.7; cannot be cut → must be repaired (Q2-①).

## Q2 — three most dangerous committee questions
① "ONE estimator? You use a different switch on real data. Which one is the estimator?" — KILLER, currently NOT answerable. Synthetic: n_selected≥8 + bf guard. Real: amp_ratio>1.5 fallback. Real proposals dense (n_eff≈24) → synthetic router would route every real window to learned = the losing arm. Headline collapses to "one estimator per dataset."
MINIMAL CLOSE (2-3 days, no training): run shipped init_method='auto' blind on 30 real windows; report route distribution honestly; re-specify deployed policy as ONE three-clause GT-free policy — route(crowding) → veto(observability guard) → insure(amplitude anomaly) — scored UNCHANGED on both arenas. One policy, two arenas, one table. Misroutes = a finding, if discovered by us first.
② "Deployed config tuned on the benchmark it is scored on." — half-answerable (prereg declares in-sample; sensitivity reported; but honesty ≠ out-of-sample number).
MINIMAL CLOSE (~1 week): freeze config, 40 fresh-seed episodes across regime grid, score once prereg. Nearly free: 8 of 13 Bergamasco .nc records unused → held-out false-alarm rate for frozen detector. A single held-out number converts Ch.5 from engineering to result.
③ "Textbook model selection — where is the PhD?" — answerable now in difference-first form, never say "combination": (a) boundary LOCATION prereg + reproduced (crowded −45%, sparse 12/12 reverse); (b) MECHANISM attributed (kx/ky placement, p=0.94 ablation); (c) BOUNDARY CONDITIONS falsified (3 axes, proposal-generation OOD at atom level). Scientific, falsifiable claim about where learning helps in dispersion-constrained sparse inverse problems. Lit-check mandatory (deep unfolding / learned matching pursuit threat to Ch.4).
Honorable mention: "your headline was 42% wrong from your own bug — why trust you?" → "The instrument in Ch.2 found it, on our own work; here is the 17/0 discard ledger." Ch.2 earning its place.

## VERDICTS
Q1: boundary narrative defensible TODAY as 8-chapter spine, condition: cut CRB spine to appendix, repair Ch.7 via Q2-①. Do not build a new pillar.
Q2: ① not answerable now (2-3 days closes); ② half (1 week closes); ③ answerable now + lit-check. TOTAL: ≈2 weeks, zero retraining, zero new data. Then it's all writing.

Thesis statement:
> In grid-free, dispersion-constrained spectral estimation of ocean wave fields from sparse moving elevation observations, the value of a learned prior is confined to spectral-component placement under crowding — a boundary that can be pre-registered, mechanistically attributed, and detected without ground truth — so that a single auto-switching estimator outperforms both specialists in simulation while degrading safely to the analytic specialist on real seas, where the learned arm's measured contribution is failure-mode insurance rather than accuracy.

## TOP 3 CRUXES
1. Method or diagnostic? Must the router beat something other than our own two arms to count?
2. Property of the sea or of our simulator? (linear GT, simulated LiDAR, learned loses on the one real dataset)
3. Ledger: asset or admission? And which paper ships in 8 weeks — corrected WaveSplat Paper 1 (vote: FIRST) or boundary Paper 2 (needs ①/②)?

LIT-CHECK-NEEDED: LISTA/deep-unfolding/learned NOMP (biggest threat to Ch.4); diffusion set-proposals in SBI; hybrid regime switching in geophysical inverse problems; runtime OOD/safe fallback for learned components; refresh Ehlers/Desmars/PINO/Fucile deltas vs 2026.
