# Methodology review for a DOCTORAL positive contribution (2026-06-29)

User: a limits paper alone can't be a PhD thesis; review methodologies for a genuine positive core.

## The constraint our own work established
The bottleneck is **information**, not algorithm (CRB; oracle 0.037 vs deployable ≈ zero-predictor).
A positive doctoral contribution must therefore **get more information** (the only real lever) or
**reformulate** so the limit no longer means failure. Crucially, our CRB analysis already localized
WHERE the information is missing: **direction / cross-range** (the synthetic aperture from *wave*
motion sharpens |k|/ω but is flat in direction — the "30°-flat likelihood"). That points directly at
the fix.

## Candidate methodologies (ranked by doctoral viability)

### ★ 1. Platform-motion synthetic aperture (the lead — our negatives point here)
**Idea.** We have always assumed a (nearly) STATIONARY footprint; the only synthetic aperture was the
wave's c_g·T_obs, which is ALONG propagation and does NOT resolve direction. But the ship MOVES: with
platform velocity v_ship, the footprint translates by v_ship·T_obs, adding a **cross-range** aperture
when the ship moves across (or maneuvers relative to) the wave direction — exactly the missing
dimension. Like SAR: platform motion synthesizes a large aperture.
- **Why doctoral & positive:** turns the whole negative arc into a setup — "naive (stationary)
  estimation is information-limited [our CRB + negatives]; **platform-motion synthetic aperture
  resolves the directional bottleneck** and achieves phase-resolved reconstruction a stationary sensor
  cannot." Theory-driven (CRB is the design tool), novel framing, and the experiments are concrete:
  optimal trajectory/maneuver design to minimize the directional CRB.
- **Theory-grounded confidence: HIGH.** Our own CRB says direction is the gap and cross-range aperture
  is the cure; platform motion supplies it. v_ship·T_obs (e.g. 5 m/s × 20 s = 100 m) is comparable to
  c_g·T_obs → a large effect.
- **First test (cheap, existing code):** extend d1_crb so obs positions move with the ship
  (x_m = x_0 + v_ship·t_m); compute the directional CRB vs v_ship and heading. If cross-track motion
  collapses the directional CRB and the wall → confirmed thesis core.
- **Novelty check needed:** SAR-style platform motion is known in radar; whether it's been framed as
  the *directional-resolution cure* for phase-resolved ship-borne wave estimation needs a lit check.

### 2. Multi-sensor / multi-aperture fusion (LiDAR + X-band radar / stereo / multiple units)
More physical aperture & coverage = more information; the CRB quantifies each sensor's contribution and
the optimal placement. Positive and useful, but heavier (more data/hardware assumptions) and less
theoretically surprising than #1. Strong as a *second* thesis chapter.

### 3. Generative calibrated space-time forecasting (embrace the limit)
The point estimate fails because it hedges to ~0 (info limit). A **generative posterior** that samples
plausible future wave fields with calibrated ensemble spread is the *correct* object for an
unpredictable system — useful for risk-aware control. The information limit becomes the MOTIVATION, not
the failure. Positive *capability* (calibrated ensembles), not accuracy. Builds on the UQ/diffusion
work. Medium confidence; risk = "is a calibrated ensemble a strong enough positive on its own."

### 4. Rogue / extreme-wave short-horizon early warning
Reformulate to a high-value detection task (predict an incoming extreme a few seconds ahead) where even
modest skill matters and extreme events have exploitable spatial structure. Different problem; uncertain
whether the info limit binds favorably. Possible niche chapter.

## Recommendation
Test **#1 (platform-motion synthetic aperture) FIRST** — it is the positive that our own CRB + negative
results logically set up, it is theory-grounded with high confidence, and the first check is cheap
(extend the existing FIM code with a moving footprint). If the directional CRB collapses with
cross-track platform motion, the doctoral thesis writes itself:
> *Phase-resolved ship-LiDAR wave estimation is information-limited for a stationary sensor — and the
> binding gap is directional. Platform-motion synthetic aperture resolves it: by exploiting (and
> optimally designing) the vessel's trajectory, we synthesize the cross-range aperture needed for
> directional identifiability, achieving reconstruction the stationary limit forbids.*
Then #2 (multi-sensor) and #3 (generative UQ) become supporting chapters; the CRB theory (C1) is the
backbone that motivates and designs all of them. This is a positive, doctoral-scope program — the limit
theory becomes the *engine*, not the conclusion.

## Honest caveats
- #1 must be VERIFIED (CRB experiment) and its novelty lit-checked before committing.
- If platform motion does NOT collapse the directional CRB (e.g. the wave field decorrelates faster than
  the platform sweeps), the lever is weaker than hoped — must know early.
- Real-data validation (D1) remains the eventual requirement for any of these.
