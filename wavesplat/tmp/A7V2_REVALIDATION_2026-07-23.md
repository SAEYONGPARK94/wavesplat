# A-7 v2 retried on Nyquist-adequate data — **both pre-registered gates PASS** — 2026-07-23

Run: `python3 scripts/d3_a7v2_refine.py --data_dir data_finemesh_fixedT_2026-07-23`
Log `tmp/a7v2_finemesh.log`; results `results/direction3/a7v2_refine_eval.json`.
Gates were frozen in `docs/A7_D20_prereg_2026-07-12.md` long before this retry.

## What A-7 is

A rescue for the one regime where the production estimator collapses: **calm single-swell
seas (BF2, 1 component)**, where the hand path scores field correlation ≈ 0. The diagnostic
that motivated it: feeding the *true* wavenumber gives corr 0.993 — the information is
present, **peak super-resolution is the bottleneck**.

A-7 **v1 FAILED** for two stated reasons: (i) the detect statistic was kp-dependent so it
never fired on BF2 with a BF3–5-tuned threshold; (ii) even force-fired, the beamform argmax
sat on a **sidelobe** in 3/5 seeds (|k| 1.0–1.76 against a true 0.77). v2 replaced
peak-power selection with **refine-then-select on a held-out (20 %) observation residual**,
plus a kp-invariant detector calibrated on BF3/BF4 only (BF2 held out as judgment set).

## Why it was worth retrying

BF2 1comp is λ = 8.2 m rendered on a 6.90 m mesh — **1.2 points per wavelength, below
Nyquist**, and `wavesim/mesh_adequacy.py`'s own docstring cites this very episode as its
canonical violation. Sub-Nyquist rendering **aliases**, placing spurious power at wrong |k|.
The "sidelobes" v1 died on were plausibly aliases.

**Confound control (the reason the July retry didn't count):** that attempt was entangled
with a ~40× observation-count change. Here the fixedT cap is held at the original
**8000 obs/frame**, verified: per-frame returns are **exactly 8000 in both old and new**
files (min = max = mean), so the cap binds identically on both sides.

## Results

**G-A7v2-1 recovery — PASS (5/5, bar was ≥3/5 at corr ≥0.5 AND rho ≥0.5)**

| seed | fires | argmax \|k\| → selected (true 0.77) | hand corr/rho | **hybrid corr/rho** |
|---|---|---|---|---|
| 0 | yes | 0.77 → 0.77 | +0.25 / +0.57 | **+1.00 / +0.95** |
| 1 | yes | 0.78 → 0.77 | +0.54 / +0.68 | **+1.00 / +0.96** |
| 2 | yes | 0.73 → 0.77 | +0.00 / +0.05 | **+1.00 / +0.96** |
| 3 | yes | 0.79 → 0.77 | −0.00 / −0.04 | **+1.00 / +0.90** |
| 4 | yes | 0.77 → 0.77 | +0.90 / +0.91 | **+1.00 / +0.95** |

Detector fire-rate 100 % (v1's failure (i) is gone — the kp-invariant statistic separates
1comp from 5comp with Youden J = 1.00, θ = 0.796).

**G-A7v2-2 harmless — PASS.** Synthetic: fires only on BF2; **Δfield = +0.0000 exactly** on
all five non-BF2 episodes (BF3 5comp, BF4 15comp, BF5 15comp, and both dir180 variants).
Real YS17, 12 windows: **never fires** (ev1 0.11–0.51, all below θ), **Δencounter = +0.0000
on every window**.

## The failure mechanism is confirmed — and it changes what the method is for

On adequate data the argmax sits at **0.73–0.79** against a true 0.77, versus **1.0–1.76**
on the old data. **The sidelobes were sub-Nyquist aliasing artifacts.** v1's stated cause of
death was a rendering artifact.

⚠ **But that undermines A-7's own rationale.** The log reports `dk/k 0.000` on all five
seeds: refine-select never had to *move* the winner off the argmax — it only confirmed it.
The machinery built to rescue a mis-placed peak is solving a problem that no longer exists.
The gain is nonetheless real and large (hand is erratic at −0.00…+0.90; hybrid is 1.00 on
all five), so **the working ingredient is the fine-atom splice around the refined k, not the
sidelobe rejection.** Any revival must re-state the mechanism — this is precisely the
"asserted mechanism" error class that bit this project five times today.

## Scope — deliberately stated narrowly

- **One synthetic corner case.** The detector is designed to fire only on quasi-monochromatic
  seas and does exactly that.
- **Zero demonstrated real-data value.** On all 12 real YS17 windows it never fires and
  contributes Δ = 0.000. That satisfies "harmless" but it also means A-7 would do **nothing**
  on the actual stereo records.
- Synthetic linear episodes only; no HOS arm.
- Single run per seed. The estimator path here is the seeded numpy hand/bank path, not the
  torch tracker whose bimodal CV-53 % variance is documented in
  `tmp/ESTIMATOR_VARIANCE_RESULT_2026-07-23.md` — but a multi-seed repeat would still
  strengthen this.

## Bottom line

A-7 v2 goes from **FAIL to a clean double PASS** once the data can represent the wave it is
trying to find, with the July confound properly controlled. That closes the single genuinely
open item from the discard audit (`tmp/DISCARD_AUDIT_fable_2026-07-23.md`). It is a narrow,
honest rescue of a known failure regime — **not** a headline result, and not something with
demonstrated value on real data.
