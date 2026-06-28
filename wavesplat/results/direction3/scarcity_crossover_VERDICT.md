# Observation-budget crossover — learning's genuine niche (2026-06-28)

**Question:** Given that learning does not improve the *headline* (dense-data) accuracy, is
there a regime where the learned generative prior genuinely beats the classical off-grid
estimator? Probe: sweep observations-per-window m and compare DPS (learned prior) vs RELAX
(off-grid) on complex seas. (`d33_dps_density.py --caps 50 100 150 200 400`,
`scarcity_sweep.json`, fig `fig_scarcity_crossover.png`.)

## Result — a clean monotone crossover at m* ≈ 200 (8 complex episodes, BF4/BF5 15c+30c)
| m (obs/window) | DPS (learned) | RELAX (off-grid) | winner |
|---|---|---|---|
| 50  | **0.221** | 0.283 | learning −22% |
| 100 | **0.216** | 0.274 | learning −21% |
| 150 | **0.209** | 0.237 | learning −12% |
| 200 | 0.217 | 0.219 | tie (crossover) |
| 400 | 0.216 | **0.196** | classical −9% |
| 700–3000 | (cap) | **0.201→0.156** | classical dominates |

## Mechanism
- The learned generative prior is **flat in m** (~0.21 from 50 to 400 obs): it supplies
  structure from the prior, so it degrades gracefully when data is scarce.
- RELAX **degrades steeply as m falls** (0.196 → 0.283 from 400 → 50): matching-pursuit
  needs enough observations to detect/separate components; data-starved, it fails.
- Below ~150–200 returns/window the prior's structural bias beats data-starved
  identification; above it, explicit off-grid identification converts density into
  accuracy that the over-smoothing prior cannot.

## Why this is a satisfying, deployable contribution
LiDAR return counts vary by orders of magnitude with range, sea-state, reflectivity and
weather (rain/fog/spray → heavy dropout). The crossover gives an operational rule:
**when returns are scarce (harsh sensing) use the learned prior; when plentiful use the
off-grid estimator.** The two are complementary, and the CRB explains both — the prior is
the *mandatory bias* exactly in the information-starved regime the CRB identifies, and is
redundant once observations supply the information. This reconciles the whole arc:
- learned prior NOT needed for the dense headline (d30, M3) — but
- learned prior IS the right tool under scarcity (this), a regime the deployable system
  will actually meet.

## CORRECTION after scrutiny (d36_dps_diagnose.py, dps_diagnose.json)
A reviewer (user) flagged "learning gets worse with more data" as suspicious — it is, so we
diagnosed it. Hypothesis: the DPS guidance is normalised per step (x += g·grad/‖grad‖), so
data strength is discarded. **Tested and REJECTED:** DPS is flat in m at *every* guidance
(0.1/0.3/0.6), higher guidance makes it *worse* (0.20→0.26), and per-sample-BEST is flat
too (not a BMA artifact). So the flat-with-m is NOT a guidance bug.

**Corrected interpretation:** the learned prior has a hard, m-independent **floor (~0.20)**
= the prior-typical reconstruction. It cannot go below it regardless of data or guidance,
because the prior cannot represent the precise components (the same precision barrier as
d34/d30). Therefore the crossover means:
- learned prior = a **robust floor / graceful degradation** (substitutes for missing data,
  capped at prior quality), NOT superior accuracy;
- RELAX **collapses below that floor when data is scarce** (m<200) and **breaks through it
  when data is sufficient** (m>200).
"Learning wins under scarcity" precisely = "the prior floor (0.20) beats a data-starved
classical estimator (0.28)", not "learning is good" (0.20 ≫ oracle 0.027). This is the
honest, defensible framing — the user's scrutiny sharpened it.

## Net story (all experiments)
1. CRB sets the wall; information (time + density) is the first-order limit.
2. Dense regime: classical off-grid (RELAX) is best, −37% vs GP at equal budget; learning
   ties/over-smooths and is not amortizable to the needed precision (d34/d35).
3. Scarce regime: the learned generative prior wins by up to −22% (this).
→ A complete, honest map: *characterize the limit, give the best dense estimator, and
identify exactly when learning helps.*
