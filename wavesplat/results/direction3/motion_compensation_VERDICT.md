# VERDICT — D2 option 1: feedforward motion compensation — clean honest NEGATIVE + key reframing (2026-06-29)

Goal: does the phase-resolved forecast cancel wave-induced platform motion (feedforward) better than
trivial baselines? Motion m(t)≈η(x_op,t); actuator latency τ needs a τ-ahead forecast; residual
motion = m − command. `scripts/d71_motion_compensation.py`. 3 seas × 4 seeds, test split.

## Result — the deployable forecast cancels ~0% of motion
| actuator latency τ | persistence FF cancels | forecast FF cancels |
|---|---|---|
| 0.5 s | **+49%** | −0% |
| 1.0 s | +3% | −0% |
| 1.5 s | −35% (HURTS) | −0% |
| 2.0 s | −61% (HURTS) | −0% |
| 3.0 s | −77% (HURTS) | −0% |

(ceiling: a PERFECT forecast — oracle nRMSE 0.037 — would cancel **85%** of the motion.)

- **The deployable posterior forecast cancels essentially NO motion at any latency** (residual = full
  motion). It predicts ≈ flat water, which cancels nothing.
- **Trivial persistence** ("command = currently-measured motion") cancels ~half at 0.5 s (water barely
  changes in half a second), but **HURTS at ≥1.5 s** (the wave has reversed → it actuates in the wrong
  phase, amplifying motion).
- The script's auto-line "forecast beats persistence" is misleading: the forecast only wins *on
  average* by doing NOTHING while persistence actively hurts at long latency. **At the short latency
  where compensation actually works (0.5 s), persistence (49%) crushes the forecast (0%).**

## ⭐ Key reframing this exposes (important for the whole paper)
**nRMSE ≈ 0.25 IS the zero-predictor** (predict flat water): for a Gaussian sea σ_surface = Hs/4, so
RMS(η)/Hs = 0.25 exactly. Therefore:
- a forecast at nRMSE 0.25 cancels 0% of motion; at 0.22 → ~12%; at 0.119 → ~52%; oracle 0.037 → 85%.
- The **deployable short-window** methods (posterior/F1/short RELAX, 0.22–0.25) sit AT the
  zero-predictor → almost no usable phase-resolved skill for control. Only **high-observation /
  long-window** results (RELAX 0.119) carry real skill — but those are non-deployable in real time
  (compute-cycle Pareto, d69).
- Lesson: report SKILL vs the 0.25 zero-predictor, not raw nRMSE — many "good ≈0.22" numbers across
  the field (and this project) are barely-above-chance for control.

## Honest conclusion (consistent across BOTH D2 framings)
Both operability (d70) and motion compensation (d71) give the same verdict: **the deployable
phase-resolved forecast provides NO control value over trivial baselines**, because at the real-time
(short-window) operating point it is pinned at the information wall (≈ zero-predictor). Persistence
(use the current measurement) is as good or better at the short latency that matters; the forecast
cannot extend useful compensation to longer latency because the waves there are unpredictable.

→ **D2 is an honest NEGATIVE.** The calibrated UQ's only surviving value is *risk awareness* (its σ
correctly predicts the residual / that compensation won't work — coverage 71% ≈ ideal 68%), not
better control. This is the strongest, most tangible operational evidence yet that phase-resolved
ship-LiDAR estimation is information-limited at the deployable operating point.

## Caveats
- Motion = η proxy (heave RAO≈1); the specific posterior here trained to nowcast 0.25 (zero-predictor
  level — somewhat under-trained; a 0.22 posterior would cancel ~12%, still < persistence's 49%). The
  qualitative conclusion (deployable forecast ≪ persistence for cancellation; both ≪ oracle) is robust
  and follows directly from the zero-predictor reframing. 3 seas × 4 seeds.
