# VERDICT — Platform-motion synthetic aperture resolves the directional bottleneck (2026-06-29)

The make-or-break test for the doctoral POSITIVE. Our CRB (d1) showed the directional (cross-range)
identifiability is the binding bottleneck — wave-motion synthetic aperture (c_g·T_obs) sharpens |k|/ω
but is flat in direction. Hypothesis: PLATFORM motion (the ship moving) supplies the missing
cross-range aperture. Tested by feeding moving observation positions (x=x0+vx·t, y=y0+vy·t) into the
existing FIM. `scripts/d72_platform_aperture.py`.

## Result — CONFIRMED (and richer than predicted)
Realistic multidirectional sea (8 components, ±25° spread), footprint 50–60 m, T_obs 20 s, wave
c_g = 5.2 m/s. Median directional CRB vs platform speed:

| platform speed | 0 | 3 | 5 | 7 | 10 m/s |
|---|---|---|---|---|---|
| **cross-track (⊥ wave)** | 9.6° | 7.5° | **3.9°** | 2.7° | **2.2°** |
| along-track (∥ wave) | 9.6° | 23.3° | 23.5° | 27.1° | 19.0° |

- **Cross-track platform motion collapses the directional CRB: 9.6° → 2.2° (×4.4 sharper) at 10 m/s,
  ×2.5 at 5 m/s.** It synthesizes exactly the cross-range aperture the wave motion cannot.
- **Along-track motion HURTS** (9.6° → ~23°): at v ≈ c_g the ship "surfs" the wave → the field looks
  frozen → Doppler/temporal diversity is lost → identifiability degrades. (A bonus physical insight:
  the WORST trajectory is surfing the dominant wave.)
- (Single isolated component has NO directional bottleneck — CRB ~0.3° at any speed — so that sanity
  panel is uninformative; the bottleneck is about SEPARATING nearby components, where the effect is
  strong.)

## Why this is the doctoral positive
It turns the entire negative arc into the SETUP for a positive thesis:
> *Phase-resolved ship-LiDAR wave estimation from a STATIONARY sensor is information-limited, and the
> binding gap is directional (CRB). The ship's own motion is the cure: CROSS-TRACK platform-motion
> synthetic aperture supplies the cross-range aperture needed for directional identifiability
> (CRB ×4.4 sharper), while along-track surfing is worst. We turn the CRB into a TRAJECTORY-DESIGN
> tool: optimal vessel headings/maneuvers that minimize the directional CRB and unlock reconstruction
> the stationary limit forbids.*
The CRB theory (C1) becomes the ENGINE (design tool), not the conclusion. Positive, novel-framed,
doctoral-scope (theory → optimal trajectory design → achievability → validation).

## Novelty (lit-check, honest)
Synthetic aperture from platform motion is well-known in SAR / synthetic-aperture sonar. BUT: for
OCEAN WAVES, SAR measures directional/power SPECTRA (statistical), NOT phase-resolved fields; SAS
images the seafloor, not the surface wave field. No work found framing platform-motion synthetic
aperture as the **directional-identifiability cure for phase-resolved ship-borne wave reconstruction**,
with CRB-guided trajectory optimization. → Novelty is **application + analysis + trajectory design**,
not the synthetic-aperture concept. Defensible doctoral novelty (verify with a deeper lit review).

## Next steps (to make it a thesis, not just a CRB curve)
1. **Achievability:** show a DEPLOYABLE estimator (RELAX / posterior) actually reconstructs better with
   cross-track motion on HOS seas — the CRB says it's possible; demonstrate it end-to-end.
2. **Optimal trajectory design:** minimize the directional CRB over headings/maneuvers (incl. realistic
   ship-speed/turn-rate constraints); compare straight cross-track vs zig-zag vs circular.
3. **HOS + real-data validation** (D1) eventually.

## Caveats
- This is the CRB (a LOWER BOUND / possibility result) on a synthetic multidirectional sea — not yet an
  achieved reconstruction nor HOS. Achievability (step 1) is the immediate must-do before claiming the
  thesis. The mechanism (cross-range aperture) is general and analytically grounded (∂ψ/∂ky = y0+vy·t).
