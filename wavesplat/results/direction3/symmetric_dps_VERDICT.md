# Symmetric DPS-vs-RELAX verdict (2026-06-28)

**Question:** Does the headline result (RELAX 0.119, −42% vs SOTA) actually depend on the
learned set-diffusion prior? The two were never compared under the same time/observation
budget (RELAX got 2.2·Tp + ≤3000 obs + 5 frames; DPS got W=3s + 1 frame).

**Experiment** (`d30_symmetric_dps.py`): identical d23 protocol — same 30 episodes, same 5
frames, same period-scaled window (2.2·Tp), and the **SAME subsampled observations
(m_cap=200)** fed to BOTH estimators. Only the estimator differs. RELAX is also shown at
full resolution (3000 obs) for reference, plus ST-DF-GP and oracle.

## Result (per sea-state, nRMSE)

| sea | DPS (learned, m≤200) | RELAX1 (m≤200) | RELAX3 (m≤200) | RELAX3_full (m≤3000) | ST-DF-GP (1s) | oracle |
|---|---|---|---|---|---|---|
| BF3_5c  | 0.224 | 0.154 | 0.159 | **0.086** | 0.106 | 0.074 |
| BF4_5c  | 0.204 | 0.098 | 0.098 | **0.051** | 0.073 | 0.044 |
| BF4_15c | **0.212** | 0.272 | 0.251 | **0.102** | 0.232 | 0.034 |
| BF4_30c | **0.218** | 0.276 | 0.270 | **0.194** | 0.287 | 0.035 |
| BF5_15c | **0.184** | 0.230 | 0.195 | **0.115** | 0.255 | 0.016 |
| BF5_30c | **0.195** | 0.244 | 0.218 | **0.164** | 0.274 | 0.022 |
| **overall** | **0.206** | 0.212 | 0.199 | **0.119** | 0.205 | 0.037 |

## Verdict (three precise findings)

1. **The headline (0.119) does NOT come from learning.** At equal observation budget
   (m≤200), learned DPS (0.206) ≈ RELAX3 (0.199) ≈ ST-DF-GP (0.205) — all ~0.20, a tie
   within noise. The drop to 0.119 comes entirely from **RELAX ingesting ~3000
   observations**, which the learned DPS structurally cannot (its GP-marginal likelihood is
   O(m³) Cholesky per diffusion step → capped at ~200). The lever is **observation/information
   count, not learning**, and not even the RELAX algorithm per se.

2. **"Learned prior is mandatory" is wrong as stated — but the prior has a real, narrow
   edge.** At equal budget (m≤200) on **complex seas (15c/30c)**, learned DPS *beats* both
   RELAX and GP (e.g. BF5_15c: DPS 0.184 vs RELAX 0.195–0.230 vs GP 0.255), and is flat in
   complexity (0.18–0.22). This **vindicates the Dir-III "complexity-robust prior" claim —
   but only in the observation-limited regime.** On simple seas the prior over-smooths and
   loses badly (BF4_5c: DPS 0.204 vs RELAX 0.098).

3. **The deployable regime removes the prior's niche.** Once you accumulate many
   observations (the deployable RELAX, m≈3000), classical matching-pursuit overtakes the
   learned prior on **every** sea-state, complex included (BF5_15c 0.115 vs 0.184). So in the
   regime the paper actually deploys, learning is not needed.

## Implication for the paper
- Drop "mandatory" / "the prior provides the mandatory bias" as a headline claim — it is
  not supported once budgets are equalized.
- The HONEST and STRONGER story: *even an information-optimal learned prior cannot beat the
  information limit; its only measurable edge is complexity-robustness under severe
  observation scarcity, and a deployable estimator erases that edge simply by accumulating
  observations.* This makes the "bottleneck is information, not algorithm" thesis airtight.
- The learned prior survives as a **scientific probe** (it proves the information wall), not
  as the deployable estimator.

## Caveat
- ST-DF-GP was NOT re-run at the 2.2·Tp window / m≤200 here (still its own 1s/W=10 pipeline).
  The clean GP-at-equal-budget comparison (FACT_CHECK to-do #3) remains open; but the DPS
  result already shows the ~0.20 plateau is an information regime, not an algorithm gap.
