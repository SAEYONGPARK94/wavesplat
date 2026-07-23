# Integrity corrections raised by Novelty Round 2 — verification & disposition (2026-07-23)

Round 2 (`NOVELTY_REREVIEW_ROUND2_2026-07-23.md` §"Two integrity corrections") flagged two
items to act on "regardless of direction". Both are now checked against the repo. Verdicts
differ: **one is real (but already contained), one is a FALSE ALARM.**

---

## Item 1 — misattributed citation arXiv 2511.21856 → **REAL, and contained (no live use)**

**Claim:** Round 1 motivated the "calibrated UQ is a genuine open gap" argument with
arXiv 2511.21856, which is actually *"A Comprehensive Review of Phase-Averaged and
Phase-Resolving Wave **Models** for Coastal Modeling"* (SWAN / WW3 / FUNWAVE numerics),
not a UQ-calibration-gap review.

**Verification (repo-wide grep over `*.md`, `*.tex`, `*.py`):** the string `2511.21856`
appears in exactly **two** files — `tmp/NOVELTY_REREVIEW_ROUND2_2026-07-23.md` (the flag
itself) and `tmp/NOVELTY_ROUND3_execution_2026-07-23.md` (which references the flag).
It appears in **no** paper, no live skeleton (`PAPER_SKELETON_operability_envelope`),
no thesis framing doc, and no results file.

**Disposition:** nothing to retract in any live document — the bad citation never
propagated past the Round-1 memo. Standing rule recorded here:

> ⛔ **arXiv 2511.21856 must not be cited as evidence of a UQ-calibration gap.** If a UQ
> gap is to be motivated at all, it needs fresh, verified evidence (this is part of the
> Agent-G brief).

Note the substantive consequence, which *does* survive: the UQ "open gap" that Round 1
asserted is **weaker than claimed**, because its only cited support was this
misattribution. Treat the gap as unestablished until Agent G's literature pass lands.

---

## Item 2 — "companion `encounter_forecast` proves RLS≡batch but ships an Adam tracker" → **FALSE ALARM**

**Claim:** the companion paper proves "streaming filter ≡ batch MMSE (RLS)" while the
shipped WaveSplat tracker (`wavesplat/optimizer/sequential_update.py`) runs ~10 Adam
steps per frame on a sliding window — i.e. the equivalence theorem would describe a
different estimator than the one deployed, and a reviewer could catch it.

**Verification — two independent routes, both negative:**

1. **Paper text.** `paper/encounter_forecast/supplementary/docs/DRAFT_main_paper_v0.md`
   §3.1 states the companion's estimator explicitly and *disclaims* component
   identification: a kp-centred coefficient bank, "**No component identification is
   attempted** — the 200-mode identification programme of our earlier work consistently
   lost to a prior-weighted linear estimator." §3.2 then proves the RLS≡batch identity
   **for that bank estimator**. So the paper does not claim the component tracker; it
   deliberately positions itself against it.
2. **Code.** The companion's streaming core is a **separate module** —
   `scripts/d3_stream_seqmmse.py` (`class StreamMMSE`, rank-m updates, with
   `batch_mmse()` as the reference), unit-tested by
   `wavesplat/tests/test_stream_seqmmse.py` (RLS≡batch, rebuild, symmetry, NIS). A
   grep for `Adam` / `sequential_update` / `coarse_to_fine` across
   `encounter_forecast.tex` **and** `DRAFT_main_paper_v0.md` returns **zero hits**.
   `SequentialUpdater` (torch Adam) belongs to the *main* WaveSplat paper's pipeline and
   is never invoked by the companion's claims or its 18 ms latency numbers.

**Disposition:** no reconciliation needed; the two estimators are intentionally distinct
and each paper cites its own. **Round 2's Agent C over-called this one.** The only
residual hygiene point is presentational: when both papers are read together, state once
that the companion deploys the bank-MMSE/RLS estimator and the main paper deploys the
spectral-Gaussian component tracker, so no reader conflates them.

---

## Side finding (relevant to the alt-methods gate, logged here to avoid re-deriving)

The companion's bank geometry is a **kp-centred annulus** `0.35–2.5 kp`, `dk = 0.1 kp`,
with a **beamform-power-shaped prior** (`d3_stream_seqmmse.py:make_bank` /
`gamma_beamform`, == `mmse_bank_estimate`). Any "bank / covariance family" comparison run
with a flat wide-grid bank is testing a **strawman**, not the family the project itself
uses. See `wavesplat/tmp/ALTMETHODS_GATE_RESULT_2026-07-23.md`.
