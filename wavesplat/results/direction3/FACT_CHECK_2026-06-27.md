# Fact-check & paper-direction review (2026-06-27)

Critical pass over the logic/results for anything strange or oversold, + what the
paper logic should be. Honest issues first.

## A. Things to be careful about (not bugs, but must not be conflated)
1. **Oracle/information bound vs deployable achieved — DO NOT conflate.**
   - "Sequential fusion → complex seas **0.02–0.03**" is the *efficient estimator with
     TRUE wavenumbers* = the **information bound** (what the CRB says is achievable).
   - The **deployable** estimator (beamform→RELAX→refine, observations only) achieves
     **0.12–0.20** at 10–15 s — beats single-window methods, but **not** 0.02–0.03.
   - ✅ Correct framing: "the information becomes available with time (oracle 0.02–0.03);
     a deployable estimator realizes a large part of it (0.12–0.20, beating
     single-window) with headroom remaining = component-identification accuracy."

2. **CRB-gated hybrid 0.174 uses an ORACLE gate** (true complexity / FIM from true
   components). Five data-only single-window gate signals failed. So 0.174 is
   "achievable *if* complexity is known" — deployable via onboard sea-state metadata
   (directional spread), not from one LiDAR window. State this every time 0.174 appears.

3. **DPS baseline is a 3 s window (W=30), not strictly "single-window."** The
   deployable RELAX beats DPS only at **T ≥ 10 s** (more observation time); at 3 s DPS
   is better (0.22 vs 0.27–0.34). The deployable advantage is *time accumulation*,
   which DPS does not exploit (DPS stays ~0.27 on 15c even at 6 s). Fair statement:
   "given enough observation time, the deployable sequential estimator overtakes the
   single-window learned-prior method on complex seas."

4. **Single representative frame ⇒ ~±0.03 frame-to-frame variance.** BF4-15c DPS is
   0.235 / 0.257 / 0.270 across different eval frames/settings. **Comparisons at the
   0.01–0.03 level are within noise.** Multi-frame averaging is required before any
   fine-grained ranking claim (already on the to-do).

5. **BMA over-smooths (VR 0.17–0.33).** The DPS BMA's low nRMSE is partly a hedged
   near-mean estimate; report VR/Pearson/skill alongside (done in diagnostics).

6. **Minor:** oracle numbers vary slightly across scripts (0.018–0.027 for BF5-30c)
   due to different random subsampling seeds/caps — cosmetic, not a bug.

## B. Things that check out (sanity-verified)
- Oracle nRMSE **monotonically decreases** with T (information grows) — consistent
  with the CRB; the deployable non-monotonicity is an estimator artifact (verified by
  the warm-start/RELAX fix making Pearson monotone).
- FIM condition separates complexity by ~15 orders (true components) — robust.
- Active-sensing CRB→achieved link holds (−32% on complex 15c) — internally consistent.
- Analytic Jacobian vs autograd 1e-15; dispersion residual <0.3% — solid.
- RELAX model order self-adapts toward the true order (N→13–24 vs 15/30) — sensible.
- No GT leakage in the deployable pipeline (GT used only for scoring).

## C. What the paper logic should be (recommended)
**Spine:** *Limits → why → how to beat them by design → an honest, calibrated,
deployable estimator.* Concretely:
1. **CRB / information wall** (Dir-I) — the theoretical backbone and the most novel
   piece (D_eff = L + c_g·T_obs; T_wall(complexity); predictable-zone = identifiability;
   *not self-diagnosable from one window*). Lead here.
2. **The bottleneck is information, proven** — NN-as-estimator fails, prior scaling is
   inert, post-hoc sharpening fails; all explained by the CRB. (Negative results, used
   as structure, not apology.)
3. **The two levers** that beat the wall, each tied to the CRB:
   (a) **active sensing** (D-optimal/cross-range scan design) — lowers the wall where
   information exists; (b) **sequential time fusion** — supplies the missing
   information for complex seas.
4. **Deployable realization:** the RELAX recursive sequential estimator (observations
   only) realizes the time-fusion lever — monotonic, Pearson up to 0.89, self-adapting
   order, **beating single-window methods on complex multidirectional seas** — plus the
   **CRB-gated hybrid** (−15%, gated by sea-state metadata) and **calibrated UQ**.
5. **Honest scope:** oracle = bound, deployable = realization-with-headroom; sim-only;
   directional generalization → simulation + future ocean basin; tank validates the
   1-D/temporal/aperture levers.

**Framing to avoid:** "new SOTA single-window estimator" (it is not — DPS 0.220 >
ST-DF-GP 0.205, and BMA over-smooths). **Framing that wins:** "we characterize the
fundamental limits and show *information-gathering by design* (sensing + time) is the
way past them, with a deployable estimator and calibrated UQ."

## D. Immediate to-dos to harden claims
1. **Multi-frame averaging** for every comparison (remove ±0.03 single-frame noise).
2. **RELAX × multi-seed × all sea-states** for the deployable-vs-single-window claim.
3. **Give DPS/ST-DF-GP the same time budget** in head-to-heads (fairness).
4. Always report **VR/Pearson/skill**, never nRMSE alone.
5. State **oracle vs deployable** and **gate = metadata** explicitly wherever cited.
