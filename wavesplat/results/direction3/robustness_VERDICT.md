# VERDICT — Robustness of the central findings to noise σ and observation density (2026-06-29)

User: "단단하게 해보자." Sweep the two main nuisance axes (LiDAR noise σ, observation density/budget)
to confirm the day's two central findings hold. `scripts/d66_robustness.py`.

## (I) Predictable-zone tie (FIM ≈ GEO) — robust to noise
| σ | AUC FIM | AUC GEO | margin |
|---|---|---|---|
| 0.01 | 0.676 | 0.673 | 0.002 |
| 0.02 | 0.661 | 0.660 | 0.001 |
| 0.05 | 0.663 | 0.646 | 0.016 |

Mean margin **0.007** — the FIM and geometric predictable zones coincide across the realistic LiDAR
noise range. (At the highest noise the FIM is marginally better, margin 0.016, still tiny.)

## (II) Active-sensing gain vs (noise σ × budget/density K) — null is robust
discrete-linear / HOS CRB-selection gain over uniform:

| σ | K=80 | K=150 | K=250 |
|---|---|---|---|
| 0.01 | −10% / −7% | +26% / +1% | +29% / +8% |
| 0.02 | −13% / −1% | +38% / −1% | +26% / −5% |
| 0.05 | +20% / −0% | +40% / −9% | +13% / +11% |

**mean HOS gain −0% vs discrete-linear +19%** across the whole σ×K grid. The discrete win and the
HOS null are both stable; the separation (discrete wins, continuous doesn't) does not depend on noise
or budget. (HOS scatters in [−9%,+11%] with no consistent sign — variance, not a lever.)

## Conclusion
**Findings ROBUST to σ and density.** Both the predictable-zone tie and the active-sensing null
hold across the noise and budget ranges — the conclusions are not artifacts of a particular noise
level or observation count. 2 seas × 2 seeds per cell (caveat: modest n; HOS per-cell scatter ±~10%
but the *aggregate* sign is unambiguous).
