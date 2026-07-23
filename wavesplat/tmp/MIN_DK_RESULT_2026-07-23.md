# Is the crowded-regime resolution floor algorithmic? — `min_dk` sweep — 2026-07-23

Script `tmp/min_dk_sweep.py`, log `tmp/min_dk_sweep.log`, raw `tmp/min_dk_sweep_results.json`.
Decision rule pre-registered in the script header **before** the numbers, including a control
episode chosen so that a null result there is required for the reading to be valid.

**Background.** `min_dk = 0.15` rad/m — the suppression radius of the greedy beamform peak
search — was hardcoded identically at three sites. fable's census over all 195 shipped
episodes: 126/195 (64.6 %) contain a true component pair closer than 0.15, **including 100 %
of every 15/20/30-component episode** (`tmp/SCALE_AUDIT_fable_2026-07-23.md`). Hoisted to a
shared constant `initializer.MIN_DK` (default 0.15 preserves behaviour exactly; control run
reproduces 0.2999).

## Result (5 seeds per cell, judged on median and on paired direction)

| episode | min_dk 0.15 | 0.05 | 0.02 | paired Δ(0.05−0.15) | SE | t | direction |
|---|---|---|---|---|---|---|---|
| **BF4 15comp** (crowded) | median 0.2906 | **0.2741** | 0.2741 | **−0.0156** | 0.0076 | −2.06 | **5/5 improved** |
| BF3 5comp (medium) | median 0.1921 | 0.1937 | 0.1937 | **+0.0153** | 0.0093 | +1.65 | **0/5 improved** |
| BF3 2comp (**control**) | median 0.0604 | 0.0617 | 0.0617 | −0.0057 | 0.0290 | −0.20 | 1/5 (noise) |

- **The control behaves as required** — BF3 2comp has no true pair inside 0.15 and does not
  move (t = −0.20). The sweep is measuring what it claims to measure.
- **0.05 and 0.02 give identical results everywhere**: the effect saturates once the radius
  drops below the separations actually present. Nothing is gained by going finer.

## Verdict: a **regime-dependent trade-off**, not a universal floor

1. **In the crowded regime the constant does impose a real algorithmic penalty.** Loosening
   it improves BF4 15comp on **5/5 seeds** (sign test p = 0.031; t = −2.06, borderline).
   MEASURED.
2. **In the medium regime it is doing useful work.** BF3 5comp gets *worse* on **0/5**
   (sign test p = 0.031 the other way) — a smaller radius lets the greedy search stack
   duplicate peaks on one true component instead of finding distinct ones. MEASURED.
3. ⟹ **Lowering `min_dk` is not the fix.** The defensible repair is to make the radius
   **adaptive** — scaled to the beamform main-lobe width, which depends on the aperture — 
   rather than a fixed 0.15 replicated at three sites. Untested.

## What it does to the information-wall argument — **bounds it, does not overturn it**

The worry was that the empirical leg ("no method recovers close / short-λ components") was
an artifact of this constant. The measurement says:

- There **is** an algorithmic component: ~0.016 nRMSE, about **5 %** of the crowded-regime
  error, recoverable by loosening the radius.
- But it is **small relative to the wall itself**: crowded-regime error only falls
  0.2906 → 0.2741. The bulk of the error survives, so the wall is **not** explained away by
  `min_dk`.

**Precise licensed statement:** claims of the form "no method recovers these components",
measured with `min_dk = 0.15`, **understate this estimator class by a few percent** and
should be restated with that caveat — but the crowded-regime wall is not an artifact of this
constant. The CRB theory is estimator-independent and is untouched either way.

⚠ The same caveat attaches to the 2026-07-23 ANM gate's "weak short-λ components are
unrecoverable by all methods at all apertures" — same pipeline, same constant. Its
conclusion (SFW ≈ greedy+joint) is a *relative* comparison between two arms that both used
`min_dk = 0.15`, so the comparison stands; the absolute "unrecoverable by all methods"
phrasing needs the caveat.

## Limits

n = 5 seeds × 3 episodes, synthetic linear, one configuration. Crowded result is borderline
on t (−2.06) and rests on 5/5 direction consistency. No HOS. The adaptive-radius repair is
proposed, not tested.

---

# Re-measured on the regenerated (Nyquist-adequate) dataset → **CONFIRMED, and sharper**

`tmp/min_dk_sweep_finemesh.py`. The census (`tmp/REVALIDATION_CENSUS_2026-07-23.md`) flagged
this result as needing re-measurement, because the original ran on episodes whose affected
components were transmitted at only 7–42 % amplitude. Same sweep, same 5 seeds, same
pre-registered rule, now on `data_finemesh_2026-07-23/`.

| episode | 0.15 (median) | 0.05 | paired Δ | SE | t | direction |
|---|---|---|---|---|---|---|
| **BF4 15comp** (crowded) | 0.2912 | **0.2722** | **−0.0150** | 0.0048 | **−3.14** | **5/5 improved** |
| BF3 5comp (medium) | 0.1699 | 0.2209 | **+0.0293** | 0.0115 | **+2.55** | **0/5 improved** |
| BF3 2comp (control) | 0.0844 | 0.0876 | +0.0178 | 0.0246 | +0.72 | 1/5 (noise) |

**Every conclusion survives, and both sides of the trade-off are now individually
significant** (they were borderline before):

- Crowded gain: t went **−2.06 → −3.14**, magnitude essentially unchanged
  (−0.0156 → −0.0150). As a fraction of crowded-regime error that is
  **0.0150 / 0.2912 = 5.2 %** — the "~5 % algorithmic" bound **holds on faithful data**.
- Medium-regime penalty: t went **+1.65 → +2.55**. Loosening the radius is now *measurably*
  harmful there, strengthening the "do not simply lower it" conclusion.
- Control still inert (t = +0.72), so the reading remains valid.

⟹ **The licensed statement is unchanged**: `min_dk` costs a few percent in the crowded
regime, is beneficial in the medium regime, and does **not** explain the crowded-regime wall.
The repair remains an aperture-scaled adaptive radius, still untested.

**Side observation (unexplained):** run-to-run spread is *larger* on the adequate data —
BF3 2comp sd 0.0491 → 0.0676, BF3 5comp sd 0.0109 → 0.0246. More renderable content appears
to give the non-convex fit more ways to diverge. This makes the median-of-k reporting
requirement **more** pressing on the regenerated dataset, not less.
