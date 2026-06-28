# Learned detection (unrolled-RELAX front-end) — verdict (2026-06-28)

**Idea (user's):** learn *why RELAX works* — a CNN that deconvolves the dispersion-constrained
beamform map (suppress sidelobes / split merged peaks) so detection recovers closely-spaced
components greedy RELAX misses. `d37_learned_detection.py` (CNN), `d38_cnn_relax.py` (integration).

## Results
**Synthetic detection peak-error (mean nearest-true, rad/m):**
| training obs | raw beamform | CNN-cleaned |
|---|---|---|
| uniform random | 0.253 | 0.021 (12× — but in-distribution + partly circular) |
| **real LiDAR geometry** | **0.047** | **0.024 (2×)** |

Note the key fact: on the *real* ship-LiDAR geometry the raw beamform is already accurate
(0.047, vs 0.25 for uniform obs) — the deployment scan has a large effective aperture, so
**detection is not the bottleneck.**

**Integration on the benchmark (CNN-RELAX vs RELAX, complex seas, real-geom CNN):**
| | CNN-RELAX | RELAX | oracle |
|---|---|---|---|
| overall (6 complex ep) | **0.240** | **0.154** | 0.026 |

CNN-RELAX is **worse** than RELAX everywhere, even though the CNN improves raw detection 2×.

## Why it fails
- Detection is not the weak point: raw beamform detection on the real geometry is already
  good (0.047). A 2× detection gain does not translate to reconstruction.
- RELAX's greedy add→refine→subtract loop is already near-optimal; replacing detection with
  a multi-peak CNN proposal gives a noisier candidate set that joint refinement handles
  *worse* than the greedy sequence.
- The oracle gap (0.15 → 0.027) is therefore **not** a detection-miss problem an NN can close
  — it is the fundamental difficulty of separating closely-spaced components from finite
  information (the CRB), which iterative refinement already addresses about as well as
  possible.

## Net — a robust negative (5 independent learning attempts)
Classical iterative off-grid identification (RELAX) is hard to beat with learning on this
CRB-limited problem:
1. learned prior (DPS) — ties at equal budget, over-smooths (d30/d33/d36);
2. off-grid feedforward — collapses (precision barrier, d34);
3. learned warm-start hybrid — neutral (d35);
4. learned detection, uniform-trained — fails (domain gap, d38);
5. learned detection, real-geometry-trained — still fails (detection isn't the bottleneck).

**Learning's genuine, honest contributions are NOT headline accuracy but:** (a) a graceful
floor in the observation-scarce regime (`scarcity_crossover_VERDICT.md`), and (b) calibrated
posterior UQ matching the CRB predictable zone. The publishable NN story is *why learning
fails here (rigorously, tied to the CRB) and the narrow regime where it is the right tool* —
stronger and more honest than a forced accuracy win.
