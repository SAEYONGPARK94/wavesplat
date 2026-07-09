# Review — learning in the platform-motion direction (2026-06-29)

User: consider doing LEARNING in the platform-motion synthetic-aperture direction.

## The key realization: the lever is now PROVEN real (unlike before)
Our earlier active-sensing work was a NEGATIVE: free-aim measurement SELECTION is uniform-optimal, and
the apparent learning lever vanished on continuous seas (d42/d43/d63/d65). The reason: with a fixed
sensor, there was no real degree of freedom that mattered.

**Platform motion changes this.** d72 proves the lever is real and LARGE and DIRECTIONAL: cross-track
motion sharpens the directional CRB ×4.4, along-track (surfing) HURTS. So *which way the ship goes*
matters enormously — and the optimal path is non-obvious (depends on the estimated multidirectional sea,
ship dynamics, and the evolving field). **For the first time, learning has a genuine, winnable thing to
optimize.** This is the honest distinction: before, the lever didn't exist; now it does.

## The learning-central contribution (the doctoral ML core)
**A learned, non-myopic, real-time informative-TRAJECTORY policy** that controls the platform's
heading/maneuver to maximize wave-field information — i.e. **amortized sequential Bayesian
experimental design (BED) for synthetic-aperture wave sensing**, with the CRB (information gain) as the
reward.
- **The NN (policy) IS the contribution** — learned active sensor-PATH control, novel for phase-resolved
  ocean-wave estimation.
- **It is winnable** because the baseline it must beat is hard: greedy/myopic CRB-optimization is
  (a) expensive (re-solve the FIM each step) and (b) SHORT-SIGHTED — it can't plan a maneuver that pays
  off several steps later in an evolving sea under turn-rate constraints. A learned non-myopic policy
  can be both fast and far-sighted. (Precedent: d40's neural acquisition beat greedy D-optimal at 51×
  speed and transferred — but THAT was on a vanished lever; here the lever is real, so the win is
  meaningful.)
- **CRB is the objective, not the contribution** — the limit theory (C1) supplies the reward and the
  achievability bound; the policy is the learned star. Resolves the lab's "learning must be central."

## Secondary learning roles (supporting, not central)
- **Amortized reconstruction from moving-platform data** (fast + calibrated UQ) — useful in the closed
  loop, but estimation is still info-limited so this is speed/UQ, not an accuracy win (as established).
- **End-to-end joint policy + estimator** (differentiable closed loop) — a strong final chapter once the
  policy works.

## Honest caveats / risks (must verify)
1. **Does learning beat greedy?** The win must be shown to be non-myopic optimality and/or real-time
   speed — not just re-deriving greedy. If greedy is already near-optimal AND fast enough, learning's
   value is only speed. Gate experiment needed.
2. **Infrastructure:** this needs a closed-loop sim = evolving sea (HOS/linear) + ship dynamics
   (speed/turn-rate limits) + sequential estimation. More than the static experiments so far.
3. **Achievability first:** even the non-learned version (greedy trajectory control) must be shown to
   improve ACHIEVED reconstruction (not just CRB) on HOS — that is the prerequisite (d72 is CRB only).

## Recommended gate (cheap, decisive — before building the policy)
On an evolving multidirectional sea with a turn-rate-limited ship, compare ACHIEVED reconstruction
(deployable RELAX/posterior) under:
  (i) stationary, (ii) naive straight cross-track, (iii) greedy CRB-optimal heading, (iv) [later] learned.
Two questions: (A) does trajectory control improve achieved reconstruction (achievability of d72)?
(B) is there a gap between greedy and the CRB-optimal / non-myopic that a learned policy could close?
If (A) yes and (B) shows headroom → build the learned non-myopic trajectory policy = the doctoral ML core.

## The resulting thesis (positive, learning-central, theory-grounded)
> *Phase-resolved ship-LiDAR wave estimation is information-limited, and the binding gap is directional
> (CRB). Platform motion is the cure — but WHICH motion matters (cross-track helps ×4.4, surfing hurts),
> and the optimal trajectory in an evolving sea is a hard sequential decision. We learn a real-time
> non-myopic policy that steers the platform to maximize wave information (amortized Bayesian
> experimental design), beating greedy planning, and demonstrate end-to-end that it unlocks
> reconstruction the stationary limit forbids.*
CRB theory = engine; learned trajectory policy = central novel contribution; reconstruction = achievability.
