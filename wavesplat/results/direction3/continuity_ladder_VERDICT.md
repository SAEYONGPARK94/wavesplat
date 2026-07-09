# VERDICT — Hardening: CONTINUITY (not nonlinearity) is what kills the learning levers (2026-06-29)

**Goal (user: "단단하게 해보자").** Make the day's two negatives reviewer-proof by isolating the
MECHANISM. d63 found the active-sensing lever wins on discrete-linear (+40%) but not on continuous
HOS (+1%) — confounding two changes: linear→nonlinear AND discrete→continuous. This disentangles
them with a controlled three-rung ladder, same harness, 5 seeds × 2 seas × 3 budgets (30 windows/rung).
`scripts/d65_continuity_ladder.py`.

## Result — the lever collapses at the discrete→continuous step, BEFORE any nonlinearity

| rung | CRB-selection gain over uniform | median | wins/n | sign-test p |
|---|---|---|---|---|
| **discrete-linear** (12 cosines) | **+23.5% ± 6.5%** | +32.6% | 23/30 | **2.6e-3** (significant) |
| continuous-linear (HOS M=1) | +2.3% ± 2.7% | +4.5% | 19/30 | 0.10 (n.s.) |
| continuous-nonlinear (HOS M=3) | +2.7% ± 2.6% | +0.4% | 16/30 | 0.43 (n.s.) |

**ladder: +24% → +2% → +3%.** The drop happens entirely at discrete→continuous; nonlinearity adds
nothing on top (continuous-linear ≈ continuous-nonlinear, both not significantly > 0).

→ **CONTINUITY is the killer.** The active-sensing learning lever is manufactured by the
*discreteness* of the standard benchmark, not by linearity. The moment the spectrum is continuous —
as every real sea is — informative (CRB-driven) selection ceases to beat uniform sampling, because
the error is then dominated by the continuous-representation gap, which no choice of measurement
points can close.

## Parallel confirmation — the predictable-zone tie is also continuity-driven
Predictable-zone test (d62 machinery) re-run on continuous-LINEAR (M=1): FIM AUC 0.644 ≈ GEO 0.628
(Spearman +0.290 vs +0.258) — the FIM and geometric zones coincide WITHOUT nonlinearity too. So the
FIM≈GEO coincidence (d62) is likewise a continuity effect (the FIM's synthetic-aperture term carries
the same c_g transport), independent of nonlinearity. (Discrete-identifiable is the only regime where
they diverge — `predictable_zone_test_VERDICT.md`.)

## Why this hardens the paper
The discrete-benchmark-bias critique sharpens from "discrete benchmarks mislead" to a **precise,
significant, mechanistic claim**: *discreteness specifically manufactures apparent learning levers
(an active-sensing win; a novel FIM predictable-zone) that vanish under spectral continuity, the
defining property of real seas.* This is exactly the kind of controlled, reproducible result that is
hard for a reviewer to attack — it pre-empts the "your nonlinearity was too weak" objection by
showing the effect is already complete at the linear-continuous rung.

## Caveats
- 2 seas × 5 seeds; nowcast; deployable coarse-D-optimal teacher (the d40 learned policy imitates
  it, so it inherits the null). Continuous-linear gain is +2.3% with p=0.10 — i.e. *not* a
  significant win, but not a hard zero either; the honest statement is "collapses from a large,
  significant win to a non-significant ~2%." Nonlinearity rung confirms no further change.
