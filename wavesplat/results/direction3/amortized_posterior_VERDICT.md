# Fast amortized neural posterior with UQ — the deployment-ideal (2026-06-28)

The whole 28-experiment investigation converges here. Real-time feed-forward control needs
SHORT window (low latency, stationarity-safe) + ms INFERENCE + calibrated UQ — a package no
existing method gives. A conditional neural field maps obs -> per-query (mean, sigma) in ONE
forward pass, trained by Gaussian NLL (mean → info-limit accuracy; sigma → calibrated UQ).
`d56_amortized_posterior.py`, eval on cached HOS continuous windows (short 2.2·Tp window).

## Result (short deployable window, continuous HOS)
| method | nRMSE | inference | UQ | real-time deployable |
|---|---|---|---|---|
| RELAX (classical discrete) | 0.279 | 1054 ms | no | weak |
| DPS (learned diffusion) | 0.249 | 9782 ms | YES | NO (too slow) |
| F1 (continuous, non-learned) | 0.235 | 404 ms | no | ok (no UQ) |
| **amortized posterior (this)** | **0.219** | **8.3 ms** | **YES (calibrated)** | **YES** |

UQ calibration: ±1σ coverage 0.68 (ideal .68), ±2σ 0.95 (ideal .95) after a single
temperature s=1.07 (≈ no recalibration needed); corr(log σ, log |err|) = 0.40 (the predicted
uncertainty tracks the actual error spatially).

## Findings
- **It uniquely satisfies all three real-time requirements:** fastest (8.3 ms — 50× faster
  than F1, 1180× faster than DPS), most accurate at the short window (0.219, ≤ F1 0.235),
  and the only fast method with **calibrated UQ**.
- This is exactly the gap classical methods structurally cannot fill: RELAX/F1 cannot provide
  a posterior (no UQ); DPS provides UQ but is 9.8 s (non-deployable). Only an amortized
  learned posterior gives speed + accuracy + UQ together.

## Honest caveats
- Accuracy (0.219) is on 6 HOS windows (3 conditions × 2 seeds) — the *order* vs F1 (0.235)
  is within possible noise; the *robust* wins are **inference speed and calibrated UQ**.
  Confirm accuracy with more windows.
- It does NOT beat the information limit (consistent with the whole thesis) — it REACHES it
  fast, with UQ. Long-window RELAX (d54) is still more accurate but non-deployable.

## Paper — the complete, honest, learning-central + deployment-grounded story
1. **CRB information-wall theory** (core, novel): the fundamental limits.
2. **Exhaustive analysis** (rigorous): the problem is information-limited; classical adaptive
   estimation exploits the levers (aperture, density) best on *accuracy*; the field's discrete
   benchmark masked this (continuous representation needed for realistic spectra).
3. **Deployment reframing**: real-time control forbids long windows and slow inference.
4. **Contribution** (learning-central, deployable): a **fast amortized neural posterior with
   calibrated UQ** — the deployment-optimal for real-time phase-resolved nowcasting, providing
   the speed + short-window accuracy + uncertainty package that no classical method can.
Not "learning beats the information limit" (it can't) — but "learning delivers the *deployable
package* (ms + accuracy + UQ) that real-time control needs and classical methods cannot."
