# #1 Beam-steering active sensing — critical honest finding (2026-06-28)

**Intent.** Strengthen the active-sensing contribution from "select among already-measured
returns" (informative subsampling) to TRUE active sensing: choose where to measure (free
aim over the footprint × window), measure there, close the loop. `d42_beam_steering.py`,
`d43_forecast_sensing.py`.

## Result — free-aim active sensing does NOT beat random uniform (any objective)
Budget sweep, free-aim dense pool, complex seas:
| budget | 30 | 65 | 100 | 135 | 170 | 205 |
|---|---|---|---|---|---|---|
| random  | 0.271 | 0.247 | 0.212 | 0.232 | 0.144 | **0.079** |
| learned | 0.291 | 0.266 | 0.230 | 0.217 | 0.153 | 0.131 |
| greedy D-opt | 0.312 | 0.310 | 0.271 | 0.191 | 0.194 | 0.132 |

Nowcast vs forecast (free-aim, K=150): random best for BOTH —
nowcast rand 0.142 / learned 0.206 / greedy 0.170; forecast(+2 s) rand 0.145 / 0.209 / 0.170.

**Random uniform coverage beats D-optimal/learned concentration at every budget, for
nowcast, forecast, and (forecast-implied) spectrum.** Greedy D-optimal also loses to random
→ this is NOT a learning bug; it is an objective mismatch: D-optimal maximises the linearised
Fisher information (concentrates), but the achieved RELAX reconstruction (and component-ID)
needs spatial DIVERSITY, which uniform coverage provides.

## Implication (honest, reshapes the contribution)
- The earlier active-sensing "win" (−13…−34%, d39/d40) holds ONLY when selecting from a
  CONSTRAINED, already-non-uniform set (real ship-LiDAR returns) under a processing budget —
  i.e., it is **informative return SELECTION / compute-efficient subsampling**, NOT active
  measurement control. You are picking the best from a fixed bad set.
- When measurements can be placed freely (idealised beam steering), uniform coverage is
  near-optimal, so there is no sensing lever for learning to exploit.
- Therefore the "neural adaptive sensing moves the information wall" framing is NOT
  supported. The honest learned-policy claim is narrower: "learned informative return
  selection for compute-/bandwidth-limited processing of a fixed scan."

## Where learning might still genuinely win (pivot)
The synthetic benchmark is LINEAR (sum of cosines) — exactly the regime where the classical
linear estimator (RELAX) is near-optimal, so learning cannot beat it. Learning should win
where the MODEL IS WRONG: **nonlinear (HOS) waves** (bound harmonics, nonlinear evolution)
that RELAX's linear superposition cannot represent. The #2 HOS validation is thus not just a
robustness check but the most promising path to a genuine learning win — a learned model
capturing the nonlinear structure the linear estimator misses.
