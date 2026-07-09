# VERDICT — Constrained active sensing on HOS: the lever is a linear-benchmark artifact (2026-06-29)

**Question (#3 revival):** does informative, CRB-driven selection from the realistic
(raydrop-constrained) feasible return set still beat uniform sampling on NONLINEAR HOS seas? The
−16% (learned, d40) / −25% (coarse D-optimal, d39) wins were only ever shown on the LINEAR
benchmark. `scripts/d63_constrained_active_sensing_hos.py`.

## Result — CRB selection helps massively on linear, ~not at all on HOS

Deployable coarse-D-optimal selection vs uniform-random, same iterative RELAX, nowcast nRMSE:

| budget K | LINEAR random→coarseD (gain) | HOS random→coarseD (gain) |
|---|---|---|
| 80  | 0.216 → 0.207 (**+4%**) | 0.287 → 0.290 (−1%) |
| 150 | 0.194 → 0.055 (**+72%**) | 0.294 → 0.293 (+1%) |
| 250 | 0.186 → 0.081 (**+57%**) | 0.324 → 0.288 (+11%*) |
| **mean** | **+44%** | **+4%** |

\* the only non-trivial HOS number (K=250) is driven by one window where uniform-random was
unstable (0.421); it is partly variance reduction, not informativeness.

**Expanded statistics (3 seas BF3-5c/BF4-15c/BF5-30c × 4 seeds) CONFIRM and tighten:** mean
CRB-selection gain **HOS +1%** (K80 −0%, K150 +3%, K250 −1%) vs **linear +40%** (K150 +61%,
K250 +47%, 4 seeds). The earlier +11% (K=250) washed out with more windows → the HOS lever is
definitively absent; the linear win is large and reproducible.

- **Harness validated:** the known linear active-sensing win reproduces strongly (+44%, up to
  +72%) — so the HOS null is real, not a coding failure.
- **On realistic HOS seas the constrained active-sensing lever is WEAK/ABSENT (+4% ≈ wash).**

## Why (principled)
On a LINEAR sum-of-cosines the model is exactly right, so reconstruction error IS parameter
estimation error, which D-optimal (max Fisher information) directly reduces — selection helps a
lot. On HOS the error is dominated by the continuous/nonlinear **representation gap** (RELAX's
finite discrete-linear model cannot represent the continuous nonlinear field, ~0.25–0.30 floor),
which is invariant to WHICH feasible points you select. So maximizing parameter information buys
almost nothing once the model is mismatched. The lever requires the linear model to be correct.

## Implication
- The constrained active-sensing thread does **NOT** revive as a learning win on realistic seas;
  the −16…−25% gains were a **linear-benchmark artifact**.
- This is the same pattern as the predictable-zone test (`predictable_zone_test_VERDICT.md`, same
  day): a quantity the discrete/linear idealization makes look like a winnable lever evaporates on
  HOS. **Two independent experiments → the discrete-benchmark-bias critique is now the most
  repeatedly-supported, strongest novelty in the project.**
- Honest active-sensing claim stays narrow (and now HOS-qualified): CRB/learned selection is a
  compute-/bandwidth-efficient subsampling of a fixed scan **on near-linear seas only**; under
  realistic nonlinearity it offers no accuracy lever.

## Caveats
- 2 HOS seas × 2 seeds, nowcast only, deployable coarse-D-optimal (the d40 learned policy imitates
  this teacher — since the teacher has no HOS lever, the learned policy cannot either). Uniform-
  random shows high budget-to-budget variance on HOS (small n); the central finding (+4% mean,
  no monotone budget trend) is robust to that.
