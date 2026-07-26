# R1 — P4 (literature hawk) — opening position
8 WebSearch + 5 fetches; MDPI/PoF full texts 403'd (snippet-level, confidence downgraded, declared).

## Component-level slaughter (weak novelty, mostly dead)
- **Kachhadiya, arXiv 2605.13068 (May 2026)** "Local Inverse Geometry Can Be Amortized": learned proposal + classical polish (LM/damped LS) with Theorems 4.1-4.2 on WHEN the hybrid succeeds; "classical refinement consistently rescues learned solvers." Domain: inverse kinematics. Preempts: architecture class, hybrid>pure-learned, shape of regime analysis. Does NOT preempt: spectral/wave content, the REVERSED rescue direction (ours: learned rescues classical on real data), deployed GT-free switch.
- **Diffusion-Enhanced Multidim. Variational LSE, Electronics 15(9):1927 (2025)**: diffusion prior + gridless VALSE, motivated by init sensitivity of line-spectral estimators (snippet-level, 403). Near-exact component twin. Plus dense 2023-25 layer: deep-unfolded gridless ANM (RS 15(1):13), DLEANM, off-grid one-bit SBL (2412.10976), off-grid DOA (SCIS 2023). ⟹ "learned prior helps gridless sparse spectral estimation" has ZERO novelty left.
- **Multi-scale FNO phase-resolved wave prediction from radar, Phys. Fluids 37(11):117116 (Nov 2025)** — newer than Ehlers/PINO pair audited 07-24; same camp (grid/operator/radar), class verdict survives but Related Work delta must refresh. Also: LiDAR point-cloud wave characteristics (Ocean Eng. 2025 — statistical EMEP, not phase-resolved, doesn't take our task but takes "LiDAR sea-surface 2025" as unqualified first) + JMSE 12(7):1082 task-driven downsampling.

## THE ONE THAT HURTS
**Katende, arXiv 2606.07153 (2026), "No-Harm Physics-Informed Inverse Learning with Residual-Calibrated Uncertainty"**: "A learned reconstruction is accepted only when its residual-calibrated radius is no worse than the baseline radius … otherwise returns the baseline … a certification-and-selection layer, not another reconstruction architecture." GT-free, a-posteriori bound, "conservative in strongly ill-posed regimes."
Preempts: (a) GT-free learned-vs-baseline selection AS A CONTRIBUTION CLASS, (b) no-harm framing, (c) false-alarm-free framing, (d) qualitative regime statement.
Does NOT preempt — plant the flag here:
- **Pre-solve vs post-solve**: Katende = residual certificate (must run the learned solver); ours = input-side regime descriptor (n_selected + beamform peak ratio) computed BEFORE committing, essentially free. Regime IDENTIFICATION vs solution CERTIFICATION. Nobody routes on a spectral-crowding statistic of the observation.
- No sensor, no field data (Poisson/heat/tomography), no deployment.
- No measured regime map, no oracle gap.
- Single-author preprint, no peer review — attackable on standing, but "I didn't find it" is not a defence.
Theory precedent for crossover phenomenon: Imaizumi-Fukumizu 1905.09195; 2603.23302 (phase transition where deep is suboptimal in sparse regime); + already-flagged null-space line (Schwab 2019, DDL ECCV 2020). ⟹ "learning's value grows with underdetermination" = theory to CITE AND INSTANTIATE, never claim.

## Finding-novelty (strongest ground)
(A) The accuracy cross deployed with prereg switch — no wave-domain equivalent found; instantiation of predicted theory, medium value.
(B) **Collapse-insurance INVERSION = best unpreempted finding.** Every hybrid paper runs the other way (classical rescues learned / classical is the safe fallback). "The learned arm is the insurance policy and the physics arm is the risk" — could not preempt. And it is a real-sensor result, which the entire neural-wave camp lacks (Ehlers: synthetic HOS; PINO: synthetic; msFNO: radar).
(C) Proposal-generation OOD as measured cause of learned-only failure — low citation risk, real dissertation value.

## Internal-consistency wounds
1. Niche sentence says "bank-MMSE" but Round-4 gate measured mmse_bank at 0.20-0.25 vs beamform 0.076-0.107 and shipped artifact is coarse_to_fine greedy — re-scope or reviewer finds self-documented mismatch.
2. **No neural baseline on our bench** — 07-24 ledger called the Ehlers-style FNO arm the existential novelty gate; 07-26 handoff doesn't run it. Against a PoF 2025 FNO reviewer, that is the rejection sentence.

## VERDICT
Niche standing but degraded: "learned set-prior" and "grid-free sparse-spectral" component-dead; "GT-free regime switching" CONTESTED by Katende, survives only on pre-solve/input-side distinction (must state and defend). Genuinely unoccupied: the combination anchored on real ship/stereo sensor data + the collapse-insurance inversion. Reframe headline from "unoccupied niche" to "measured boundary + reversed rescue direction, on real data the neural-wave camp does not have."
Most dangerous citation: Katende 2606.07153. Runner-up: Kachhadiya 2605.13068.

## TOP 3 CRUXES
1. Router: pre-solve regime descriptor vs post-solve residual certificate — difference in kind? Does a two-statistic threshold heuristic survive next to a certified no-harm framework?
2. Unrun FNO/PINO baseline — defensible headline without it, or is every number "beats our own prior"?
3. Which asset is the dissertation: regime map (synthetic-only accuracy) or collapse insurance (real, n=3)? Read: the second is less preempted — but thin.

[Sources listed in original with links: Katende 2606.07153, Kachhadiya 2605.13068, Electronics 15(9):1927, RS 15(1):13, arXiv 2412.10976, PoF 37(11):117116, PoF 37(10):107119, Ehlers 2305.11913, JMSE 12(7):1082, Ocean Eng. LiDAR 2025, Imaizumi-Fukumizu 1905.09195, arXiv 2603.23302, AMSES 2023 NF-safeguards]
