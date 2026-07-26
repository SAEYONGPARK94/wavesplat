# PRE-STATE — W9 observable-coordinate phase diagram (crux A + crux F)

**Frozen 2026-07-26, BEFORE any correlation was computed.** Branch `learn-boundwave-retrain`.
Zero new estimator compute: every number below is re-derived from JSON/JSONL already on disk.
Nothing here may be edited after the first number is produced; corrections go in `RESULT.md`.

## 0. What is being tested

P3's crux A (two conflated x-axes): the phase diagram is drawn on `n_components` — a
**simulator knob**, unavailable at runtime — while the shipped router reads `n_selected`
(a statistic of the learned proposal net). P3: *"cannot claim the boundary is GT-free
detectable on axis A while the detector reads axis B."*

P1's crux F: *is the boundary axis measurable outside the simulator at all?*

The test replaces the knob with a **GT-free observable**: `bf_peak_ratio` — the
beamform dominant-peak scalar (`wavesplat/optimizer/initializer.py:292`,
`beamform_peak_ratio`), computable from the observations alone on both arenas, and
already cached for all 100 synthetic episodes (`tmp/unified_guard_2026-07-26/guard_bf_ratio.json`,
identical to `explore_beamform.json::p2_over_p1`) and all 30 real windows
(`tmp/figs_opus_real_2026-07-26/bf_ratio_real.json`).

## 1. Quantity being correlated (frozen definitions)

- **Per-episode delta** `Δ(ep) = median_of_5_lanes(learned) − median_of_5_lanes(v2)`,
  lanes `[(42,0),(1,1),(2,2),(3,3),(4,4)]`, sources `tmp/wp_integration_2026-07-25/arm_runs.jsonl`
  + `tmp/unified_2026-07-26/arm_runs_extra.jsonl` (arm `learned`, `avg_nRMSE`) and
  `tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl` (`nRMSE`). Negative Δ = learned wins.
  This is the same per-episode aggregation as every shipped number (median-of-5, then compare).
  Secondary (robustness only, not the decision): per-lane Δ then median over lanes.
- **Axis (a)** = `n_components` ∈ {1,2,5,15,30}, parsed from the filename (the simulator knob).
- **Axis (b)** = `bf_peak_ratio` ∈ (0,1], GT-free, observation-only.
- **Statistic** = Spearman |rho| over the n=100 episodes, two-sided p reported for each.

## 2. FROZEN decision rule (P5, R2 sprint item 6; adopted verbatim by P3/P1)

> If `bf_peak_ratio` predicts the delta **at least as well as** `n_components`, the router
> is not a relabel and P3's circularity charge is answered.
> If `n_components` **dominates**, **P3 is right and Ch.5 says so in those words.**

Operationalised, decided now, before the numbers:

| outcome | rule | verdict text that gets written |
|---|---|---|
| **FOR the student** | `|rho_bf| >= |rho_ncomp| − 0.05` | "the boundary is not a relabeling of the answer key: a GT-free observable orders the regime as well as the simulator knob" (crux A resolved FOR the student) |
| **FOR P3** | `|rho_bf| < |rho_ncomp| − 0.05` | "`n_components` dominates; the phase diagram is drawn on a coordinate the runtime cannot read — **P3 is right and Ch.5 says so**" |

The 0.05 slack is a tolerance band on a Spearman rho at n=100 (approximate SE ≈ 0.1
at |rho| ≈ 0.6); it is stated here so that a near-tie cannot be argued either way after
the fact. A near-tie inside the band counts as **FOR the student** by the wording of
P5's rule ("at least as well as").

## 3. What this test cannot license (frozen scope)

- It does **not** validate the shipped router. The shipped router reads `n_selected`, not
  `bf_peak_ratio`; `bf_peak_ratio` enters the shipped path only as the **guard/veto** clause
  (`< 0.05` → veto to v2). A PASS here says the *boundary is observable*, not that the
  *deployed detector is correct*. Crux D (the fresh-40 held-out run) is untouched by this.
- It is **in-sample** on all 100 episodes: `bf_peak_ratio` was inspected during guard design
  (`tmp/unified_guard_2026-07-26/explore_beamform.py`). A correlation measured on the same
  100 episodes is descriptive, not held-out.
- Both correlations are computed on a grid where `n_components` was *chosen* by the designer
  in 5 discrete steps (40/20/40). Spearman on a 5-level ordinal factor with heavy ties is
  not a like-for-like comparison against a continuous variable; the tie structure **favours**
  neither axis systematically, but it is a known asymmetry and is recorded here in advance.

## 4. Honesty clause, pre-committed (P5 crux F, P1 objection 1)

The figure **must** state, whatever the correlations say, that the 30 real windows occupy a
narrow band of the `bf_peak_ratio` axis that the synthetic grid barely populates. This kills
"the axis is unmeasurable outside the simulator"; it does **not** kill "the axis was only
*varied* inside the simulator". Both sentences go on the figure.

Additional pre-committed caveat: the real-window delta uses a different normaliser
(`/gt_std`, `map_nrmse_nowcast`) than the synthetic delta (`/Hs`). The real points are
therefore overplotted for **coordinate placement and sign**, not for magnitude comparison
against the synthetic cloud. Stated on the figure.

## 5. Prediction recorded before computing (non-binding, for calibration)

Written first, as the `wp_real` four-cell freeze template requires: I expect
`|rho_ncomp|` ≈ 0.7–0.85 (the delta was designed around the regime split) and
`|rho_bf|` ≈ 0.6–0.8, i.e. a near-tie that lands inside the tolerance band, because
`bf_peak_ratio` is a monotone-ish proxy for spectral crowding but is also driven by
sea-state and realisation noise. I therefore expect **FOR the student, narrowly** —
and I note in advance that a narrow pass is the weakest form of this result.
