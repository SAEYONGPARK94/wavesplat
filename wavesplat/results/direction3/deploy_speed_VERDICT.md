# Deployment tradeoff (speed × accuracy × UQ) — the honest learning-central role (2026-06-28)

User point: real-time feed-forward control can't use unbounded windows, and INFERENCE SPEED
matters. So d54's "RELAX wins with long windows" is non-deployable; the deployable regime is
SHORT window + fast inference. `d55_deploy_speed.py`, cached HOS windows, 2.2·Tp window.

## Result
| method | nRMSE | time/window | UQ | deployable for real-time control? |
|---|---|---|---|---|
| **F1 (continuous, non-learned)** | **0.235** | **404 ms** (optimisable →~ms) | no | yes (fast + accurate) |
| RELAX (discrete, classical) | 0.279 | 1054 ms | no | weak (slower AND less accurate at short window) |
| DPS (learned diffusion) | 0.249 | **9782 ms (9.8 s)** | YES | NO (too slow for real-time) |
| F4 (neural operator, amortised) | 0.250 | ~5 ms | no | fast but inaccurate |

## Findings
1. **At the deployable short window, the classical DISCRETE estimator (RELAX) is doubly bad:
   slower (1 s) AND less accurate (0.279) than the continuous methods.** Its accuracy edge
   only appears at non-deployable long windows (d54).
2. **DPS (the learned method that provides UQ) is unusably slow (9.8 s/window)** — the 500-step
   iterative diffusion makes it non-deployable for real-time control, despite its accuracy+UQ.
3. **No single method gives all three (fast + short-window-accurate + calibrated UQ):** F1 has
   fast+accuracy but no UQ; DPS has accuracy+UQ but is slow; F4 is fast but inaccurate.

## The honest learning-central DEPLOYMENT contribution
The genuine gap is a **fast AMORTISED estimator** (one forward pass, ~ms — like F4, NOT the
iterative DPS) that is **accurate at short windows (≈F1, the info-limit)** AND provides
**calibrated UQ** for real-time risk-aware control. Such a model would uniquely satisfy the
real-time control requirements that F1 (no UQ), RELAX (slow + long-window-dependent), and DPS
(slow) each miss. This is learning-central, deployment-motivated, and honest:
- not "learning beats accuracy" (it can't — info-limit), but
- "for real-time control on realistic continuous seas, the deployable optimum is a fast
  amortised learned posterior (speed of F4 + accuracy of F1 + UQ of DPS)" — a packaging only
  learning can deliver, because classical methods cannot give UQ, and the accurate classical
  options are too slow or long-window-dependent.

→ Concrete next step: amortise the DPS posterior into a fast conditional generative model
(distill DPS → one-pass, or train a conditional flow), targeting ms-latency + UQ at the
info-limit accuracy. This is the deployable learning-central method the whole investigation
points to.
