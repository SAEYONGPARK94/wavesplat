# Validation methodology — standing requirements

## R1. Validation datasets MUST always include nonlinearity (HOS) — REQUIRED
*(set 2026-06-28, user directive)*

Every validation/evaluation set must include **nonlinear (HOS, M≥2) wave fields**, not only
the linear sum-of-cosines synthetic data. Rationale:
- The linear benchmark (sum of cosines) is exactly the regime where the classical linear
  estimator (RELAX) is near-optimal — so it cannot expose where a method (esp. learning)
  helps or fails on realistic waves.
- Real ocean waves are weakly-to-moderately nonlinear (bound harmonics, nonlinear
  evolution). A method must be validated on nonlinear GT to claim deployability.
- Nonlinearity is also the regime where the linear model is *wrong*, i.e., the most
  promising place for a learned (nonlinear-capable) method to genuinely beat RELAX.

**How:** generate HOS episodes via `HOSSimulator(HOSConfig(M=3))` /
`wavesim.pipeline.run_episode(cfg.hos.enabled=True)` for every sea-state used in evaluation;
report results on BOTH linear and HOS GT. Headline claims (RELAX −37%, any learning result)
must be reproduced on HOS data before they go in the paper.

Tracking: `scripts/d44_hos_nonlinear.py`, `results/direction3/hos_nonlinear.json`.

---
(Add further standing validation requirements below as they are established.)
