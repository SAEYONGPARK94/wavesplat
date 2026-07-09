# VERDICT — "Did learning win because of the compute/real-time CYCLE?" → YES, confirmed (2026-06-29)

User fact-check: isn't the amortized posterior's win an artifact of the real-time cycle constraint
(short window + ms inference) rather than genuine superiority? Tested with an accuracy–compute/latency
Pareto on HOS nowcast. `scripts/d69_compute_pareto.py`.

## Result
| method | window | latency (wait) | inference | nRMSE | UQ |
|---|---|---|---|---|---|
| **amortized posterior** | 2.2 Tp | ~18 s | **16.9 ms** | **0.250** | **YES** |
| F1 ridge (crude) | 2.2 Tp | ~18 s | 85 ms | 0.300 | no |
| RELAX | 2.2 Tp | ~18 s | 1303 ms | 0.263 | no |
| **RELAX** | **3.5 Tp** | ~28 s | 918 ms | **0.240** | no |
| RELAX | 5.0 Tp | ~40 s | 927 ms | 0.245 | no |

## Interpretation — the user is right
- **At a FIXED short window (2.2 Tp):** the posterior is actually best on all three axes — most
  accurate (0.250 vs RELAX 0.263; classical suffers the continuous-representation gap at short
  window), ~60–80× faster (17 ms vs ~1300 ms), and the ONLY one with calibrated UQ.
- **But classical RELAX becomes MORE accurate with a LONGER window** (0.240 at 3.5 Tp, synthetic
  aperture) — at the cost of ~28 s window latency + ~0.9 s inference.
- **Therefore the posterior's overall advantage is CONDITIONAL on the real-time cycle**: it wins
  *because* the regime is constrained to a short window + ms inference. Relax that cycle (afford
  ≥28 s latency + slower compute) and longer-window RELAX is more accurate. **"Learning won because
  of the cycle" = TRUE.**

## What this means for the paper (important framing correction)
- Do NOT claim the posterior is "the most accurate" or "beats classical" in general — it is **best
  at the real-time operating point**, not globally. State the conditionality explicitly.
- The honest, defensible claim: *"At the real-time operating point (short window + ms inference + need
  for UQ), the amortized posterior dominates — fastest, calibrated UQ, and accuracy ≥ classical there;
  classical surpasses it on accuracy only by extending the window (≥3.5 Tp), incurring latency/compute
  incompatible with feedforward control."*
- This makes the **justification of the real-time operating point** load-bearing → reinforces **D2
  (downstream control demonstration)**: the cycle constraint must come from a real control loop, or the
  framing is reverse-engineerable (a reviewer can say we picked the regime where the NN fits). With a
  genuine control use case, the constraint is principled and the posterior's scoped win is meaningful.

## Caveats
- The inline F1 (19×19 ridge) is cruder than d51's F1 (0.234) — here it underperforms (0.300); treat
  F1 as indicative only. The clean comparison is posterior vs the RELAX window sweep. RELAX inference
  ~constant ~1 s (obs capped at 2500); the accuracy gain is from window/aperture, the cost is latency.
- 2 seas × 2 seeds, nowcast; posterior nRMSE cited from d68 (4-seed) with forward-pass timing here.
