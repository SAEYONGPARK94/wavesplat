# WaveSplat Novelty Re-Review — ROUND 2 (4 more parallel agents, orthogonal axes) — 2026-07-23

**Prompt (user, Discord):** "다른 각 한 번 더 파보자" (dig one more, different angle), following Round 1's convergence on CRB + gridless atomic-norm.

**Method:** 4 independent agents on axes Round 1 did **not** cover, each **forbidden from re-proposing** the CRB / gridless-atomic-norm answer and required to judge honestly whether its axis is a live novelty source or a dead end; every novelty claim WebSearch-verified.

---

## TL;DR — an ELIMINATION result that strengthens the Round-1 spine

All four new orthogonal axes are **dead ends or collapse back into the CRB/atomic-norm spine.** Two are already spent by the group's *own* prior work; two are externally pre-empted. The only non-dead residue is a **narrow supporting UQ layer**, not a spine. Net: across 8 total orthogonal lenses (Round 1 + Round 2), everything eliminates back to the same survivor → the CRB-anchored spine + gridless-atomic-norm lever is not merely "the best we found," it is **"what's left after eliminating everything else"** — which is itself the honest novelty argument.

| Axis (Round 2) | Verdict | Binding reason |
|---|---|---|
| **A. Contribution-type pivot** (benchmark / simulator / systems / eval-protocol) | **DEAD as primary** | No real shipboard-LiDAR data, no real sensor, no deployment. Benchmark/simulator/systems each *require* that asset. Eval-panel = Nash–Sutcliffe. Residue = reproducibility appendix only. |
| **B. Nonlinear physics-model** | **DEAD** | Triple pre-empted (Desmars JFM 2023; Ehlers OE 2023; Ehlers PoF 2025) **and internally spent** — repo's own nulls: `stokes2_probe` 0.97708→0.97711, `hos_nonlinear_VERDICT` "Nonlinearity is not a learning opportunity", steepness-ceiling non-monotonic & 5–10× smaller than the sampling gap. Rigorous form = misspecified-CRB = Round-1 axis. |
| **C. Temporal / forecast-horizon / streaming-latency** | **DEAD** | All three candidates are the *explicit* claims of the group's OWN companion `encounter_forecast` paper (C1 horizon ℓ\*, C2 streaming≡batch, C3 gate) **and** externally pre-empted (Plackett 1950; Bui–Nguyen–Turner NeurIPS 2017; Qi *Wave Motion* 2018 + ship-motion patents US10,259,544/10,538,299; Alkarem *JMSE* 2025). Residual forecast-MSE bound = CRB-in-time. |
| **D. Calibrated UQ / self-diagnosis** | **Different in method, but MORE pre-empted than CRB; niche only** | Headline nouns each taken: Zhao *Applied Energy* 2022 ("quantified UQ, phase-resolved, first"), Lee 2024 ("calibration, real-time", scalar Hs), Fucile OE 2018 (LEPrE pred-error std), Alkarem 2025 (uncertainty-gated horizon under sparsity), Nielsen OE 2024 (trust indicator). One empty slot only (below). |

---

## The one residue worth bolting on (NOT a spine)

**Calibration-verified, per-location predictive-uncertainty output for the grid-free phase-resolved field from sparse shipboard LiDAR, inside the ≤100 ms budget.** The triple-intersection {phase-resolved *spatial field* × *from LiDAR* × *formal per-location calibration diagnostics*} is genuinely empty (Zhao = probe time-series; Lee = scalar Hs; Fucile = analytical). To actually pass calibration (the repo's prior UQ was overconfident, z-std ≈ 2.4): post-hoc **split-conformal** recalibration + report **spread-skill→1**, **PIT/rank-histogram flatness**, **empirical coverage** — using off-the-shelf conformal/spread-skill tools (cite, don't claim). Frame strictly as a **supporting operability layer extending Alkarem-2025/Nielsen-2024 to the grid-free LiDAR setting; never claim "first UQ."** Impact: modest, engineering-grade.

---

## Two integrity corrections surfaced (act on these regardless of direction)

1. **Round-1's "calibrated UQ is a genuine open gap (arXiv 2511.21856)" rests on a MISATTRIBUTED citation.** 2511.21856 is *"A Comprehensive Review of Phase-Averaged and Phase-Resolving Wave **Models** for Coastal Modeling"* (SWAN/WW3/FUNWAVE numerics), **not** a UQ-calibration-gap review. → Do not use it to motivate a UQ contribution; it would not survive a reviewer opening the PDF. The UQ "open gap" is weaker than Round 1 implied.
2. **Companion `encounter_forecast` paper — internal estimator/theorem mismatch.** It proves "streaming filter ≡ batch MMSE (RLS)", but the shipped WaveSplat tracker (`sequential_update.py`) runs ~10 **Adam** steps/frame on a sliding window, i.e. a *bank-MMSE / gradient* estimator, **not** the RLS filter the theorem describes. A reviewer can catch that the equivalence theorem is about a different estimator than the one deployed. → Reconcile before submitting the companion paper.

**Cleared false alarm:** Agent A flagged a Falcon spec mismatch (32ch/±5°/200m). **Verified false** — that is an unused module-level default in `lidar_simulator.py:18`; the episode pipeline is driven by `wavesim/config.py` (channels=152, vertical −25°→0°, range 250 m) = **matches the Seyond Falcon spec**. No issue.

---

## Bottom line for the user

Digging four more genuinely-different angles did **not** find a new headline; it *eliminated* the alternatives and, in two cases, showed the axis is already covered by the project's own prior work. That is a productive negative: it converts "CRB + gridless atomic-norm is our best idea" into "CRB + gridless atomic-norm is the survivor after systematically ruling out contribution-type, physics-model, temporal, and UQ axes." Recommended posture: **commit to the Round-1 spine (gated by the cheap BLASSO feasibility probe), bolt on the calibrated-UQ layer only as a supporting operability contribution, and fix the two integrity items.** No further from-scratch angle-hunting is likely to pay off — the remaining work is execution + positioning, not more ideation.
