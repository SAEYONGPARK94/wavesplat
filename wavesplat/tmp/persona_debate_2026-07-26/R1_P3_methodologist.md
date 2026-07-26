# R1 — P3 (thesis-committee methodologist) — opening position

Verdict shape: "best pre-registration hygiene I have seen in a single-student computational thesis, and its headline claim is still not tested."

## Steelman (earned credit)
1. Preregs dated, frozen, falsifiable in form; wp_real four-cell interpretation table frozen before numbers, unfavourable cell landed and was published (LOSS 6/27 p=0.0046, rank-biserial -0.608).
2. NO threshold shopping — checked and exonerated: guard grid {0.04,0.05,0.055}: 0.04 gives 0.08725, prereg default 0.05 gives 0.08747 — THEY REPORTED THE PRE-REGISTERED VALUE, NOT THE BETTER ONE. Router 6/8/10: 0.09790/0.09786/0.09819 flat, no cherry-pick. Strongest evidence prereg is load-bearing.
3. Negative controls real: Task2 sparse merge FAIL sealed (0.0546 vs 0.0254 bar; V2 variant failed own no-harm p=0.031). Three retraining axes closed.

## MEASURED
- Tracker bug 0.199→0.1145, 100/100, p=3.9e-18 — most durable finding, and it is a BUG FIX not a method.
- Crowded learned-init win 0.252→0.181, 36/40, ablation → kx/ky placement.
- Real accuracy LOSS, prereg, honest.
- Guard arithmetic reproduces exactly from JSONs.

## ARGUED
- "Auto beats both pure arms": prereg criterion = naked point comparison (switch mean <= min(pure means)) — NO test, NO interval. Capstone claim never subjected to inference. Switch is a re-assembly of the same lanes it beats; slices stratified by the same component counts the router reads via n_selected → "close to reading the answer key"; algebraic consequence of a decomposition already in hand.
- "Collapse insurance" rests on n=3 events.
- "Oracle upper bound" definitional slippage: crowded slice deployed 0.17982 vs oracle 0.18053 — EXCEEDS the bound (oracle = per-episode min-of-medians vs switch per-lane-then-median).

## RITUAL (rigor as costume)
1. Non-inferiority tests structurally near-unfalsifiable: thresh 8 sparse mean_test 0.0509 vs ref 0.0204 (2.5x WORSE) yet median_delta=0.0, p=0.248, noninf_pass=True — zero-inflated delta vector kills power; next-day prereg replaced the bar with mean-with-margin (conservative endpoint drift, but day-1 PASS was ritual).
2. Guard PRIMARY criterion #2 ("0 crowded flips") is a tautology by construction (threshold < crowded floor 0.0553 by design). Only criterion #1 had content.
3. **Criterion relaxation propagating into the headline**: frozen "fixed" = learned < 1.0; scorer output: "AA14 w0: 2.309 -> 1.071 [not fixed]. Result: 2/3." Handoff §3 and MEMORY.md carry bare "3/3"/"전부 수리". Under frozen criterion: 2/3. End-to-end deployed: detector misses AA15a_w3, fires on AA14_w0 which repairs only to 1.071 (still above collapse line) ⟹ **shipped fallback brings 1 of 3 windows below the pre-registered collapse threshold.** Same failure mode the 07-23 self-audit diagnosed, reproduced inside the document written to prevent it.
4. Caveat evaporation: both preregs bold in-sample caveats; handoff TL;DR drops them. TL;DR "0.881→0.775" is a mean driven entirely by 2 collapse windows — median unchanged 0.8091→0.8091 — violating their own do-not-cite rule ("mean 기반 실측 승리 주장 금지"). Wrote the rule, broke it in the summary line.

## Error trail
17 discards audited, 0 unjustified — but 5 mechanism stories wrong in one day. Ranking layer sound; explanation layer not. Corrections generated faster than they propagate (L1553 flagged 2 months). Self-correcting on outcomes, under-correcting on narrative.

## VERDICTS
Q1: Earned: tracker fix, crowded win, real negative. Not earned: "one estimator beats both pures" (untested point comparison, in-sample thresholds, no held-out set) and "collapse insurance" (n=3; frozen criterion 2/3; deployed 1/3). Prereg = ~70% real rigor, ~30% ritual.
Q2: PASS only under framing "where does the learned prior earn its keep, and how do you know at runtime without ground truth?" — regime-boundary characterization with deployable GT-free router; sealed negatives load-bearing. FAILS under "one estimator beats both, 0.0875" — committee finds the in-sample re-assembly in ten minutes. "Sealed negatives are only a virtue when the positive they surround is independently validated; otherwise they are a large denominator advertised as a result."

**LARGEST HOLE: no held-out set anywhere in the capstone chain.** Router thresh (8), guard ratio (0.05), guard signal choice (bf_peak_ratio, third candidate), real detector (1.5) all set by inspecting the same 100 episodes / 30 windows they are scored on. Disclosure ≠ test. Fix: hold out 30/100 episodes, never look, re-run frozen deployed config once. One-day experiment converts capstone ARGUED → MEASURED.

## TOP 3 CRUXES
1. Is n_selected≥8 a discovery or a relabeling of the stratification variable? (circularity)
2. Can a real-data contribution resting on n=3 (1 clean end-to-end repair) survive a defense? If no → synthetic-only thesis, say so in abstract.
3. Does the 07-23 propagation failure invalidate the OE paper as citable? TRACK_PHASE_PREDICT default + 0.199 in paper = live integrity exposure. [NOTE from main session: default WAS flipped later on 07-26, commit aa8e4c3 — P3 read pre-flip state; paper numbers still unpropagated.]
