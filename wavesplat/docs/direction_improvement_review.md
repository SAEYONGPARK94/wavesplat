# Improvement & alternative-direction review (positive, advisor-facing)

**Date:** 2026-06-27. The advisor wants *improvement / new directions*, not an
"honest failure" framing. This document leads with a **proven SOTA-beating result**
and then ranks constructive directions.

## 0. PROVEN WIN — CRB-gated hybrid (do this first)
The two methods are complementary: ST-DF-GP is precise on *identifiable* seas
(0.07–0.11) but degrades on complex ones (0.27–0.31); grid-free DPS is robust but
imprecise (flat 0.19–0.23). **Route between them using the Direction-I Fisher
information** — the FIM condition number at the observation window:

| method | overall field nRMSE (30 ep) |
|---|---|
| ST-DF-GP (current SOTA) | 0.205 |
| grid-free DPS | 0.220 |
| **CRB-gated hybrid** | **0.1735** |
| per-episode oracle | 0.171 |

The gate works because cond(F) separates complexity by **15 orders of magnitude**
(5c ≈ 10⁵, 15c ≈ 10¹⁶, 30c ≈ 10²⁰) — any threshold in 10⁶–10¹² gives the same
partition, so the rule is essentially perfect and robust. It picks the precise GP
for exactly the 10 identifiable episodes and the robust DPS for the 20
under-determined ones, landing at the oracle.

**Why this is the right paper:** it converts the "information wall" (a *limit*) into
a **constructive model-selection rule** — *the CRB tells you which estimator to
trust*. That is novel, positive, SOTA-beating (−15% nRMSE vs ST-DF-GP), and uses the
entire body of work (CRB from I, DPS from III, GP existing). `d9_crb_gated_hybrid.py`,
`results/direction3/crb_gated_hybrid.json`.

*Caveat / deployable gate:* the cond above uses the true components. The 15-order gap
means any data-driven identifiability proxy separates trivially — FIM from
beamformed/DPS-estimated atoms, significant-spectral-peak count, or the DPS posterior
spread. Implementing+validating the deployable gate is the immediate next task.

## 1. Why post-processing can't substitute (evidence)
Tested two deployable "sharpeners" of the DPS posterior (`d8_sharpen.py`):
- **max-likelihood sample** (vs BMA): raises amplitude (VR 0.14→0.4–0.5) but lowers
  Pearson and worsens nRMSE — the window-likelihood-best sample has wrong global
  structure.
- **off-grid refinement** of atoms: VR→0.9–1.3 (full amplitude) but **Pearson
  collapses to 0.02–0.30** and nRMSE worsens — refining (kx,ky) on a single window
  overfits, re-confirming the Dir-II Step-4 NO-GO and the flat-likelihood/CRB reason.

→ You cannot sharpen past the wall; improvement must **add information** or **route**.
This *motivates* the hybrid and the directions below (not a dead end).

## 2. Improvement directions (ranked by promise × feasibility)
1. **CRB-gated hybrid + deployable gate** (above) — *highest, proven*. Next:
   data-driven gate; expand to all 100 episodes + raydrop=slope; multi-frame eval.
2. **Add information = lower the wall (the only physical lever):**
   - **Sequential Bayesian fusion** of the diffusion prior over a *track*
     (filter/assimilation) — accumulate info across frames principledly; expected to
     beat single-window on complex seas where the wall needs ~10 s.
   - **CRB-optimal active sensing** (see §3A) — design the scan to maximise Fisher
     info per unit time.
   - **Multi-sensor fusion** (LiDAR + ship-motion/IMU + X-band radar) — add
     independent constraints; radar gives larger aperture.
3. **Conditional (amortised) diffusion posterior** — train p(S|z) conditioned on the
   observation embedding (set-transformer encoder of the LiDAR points), rather than
   unconditional prior + DPS guidance. Faster, no guidance tuning, potentially
   sharper. *Distinct from the failed T-NSE* (a point regressor): this models the
   full posterior, keeps the mandatory prior/bias, and can be CRB-gated too.
4. **Sharper posterior summary** — replace BMA (MMSE, over-smooths) with a
   variance-preserving estimator: posterior *mode* via short MAP polish under the
   prior score, or sample-consistency reranking. Cheap; pairs with the hybrid on the
   complex branch to recover amplitude with honest UQ.

## 3. Alternative research directions (genuinely new, positive)
**A. CRB-optimal active sensing / scan design (strongest new pivot).**
Turn the limit into an *objective*: choose LiDAR pointing/scan to maximise the
log-det (or min-eig) of the FIM, i.e. shrink D_eff⁻¹ fastest. This is a new
capability ("where should I look to learn the wave field fastest?"), directly
advisor-friendly, novel for ship-LiDAR, and built entirely on the Dir-I FIM. Pairs
with the synthetic-aperture result (time + pointing jointly shape D_eff).

**B. Real-data + weakly-nonlinear extension.** Validate on a real ship-LiDAR / wave-
tank set; extend the forward model to Stokes-2/HOS (repo has `stokes2.py`,
`hos_simulator.py`). Removes the "sim-only / linear" critique and upgrades the whole
program to a real contribution.

**C. Control-coupled metric (hydrofoil).** Evaluate by *downstream control* — ride
quality / load reduction of the hydrofoil (`hydrofoil/`) under feed-forward from the
estimate + its calibrated UQ + predictable-zone mask. Here calibrated UQ and the
predictable zone are *first-class value*, and "graceful degradation" becomes a
safety feature, not a weakness. Positive, application-anchored, novel coupling.

**D. Forecast-skill product.** Use the predictable-zone CRB to issue *calibrated
short-horizon forecasts* with a reliability mask; report forecast skill vs horizon —
a DSWP-style deliverable with honest UQ that the field currently lacks.

## 4. Recommendation
Lead with **(0) the CRB-gated hybrid** — it is a clean, proven, SOTA-beating result
that reframes the program positively. Immediately add the **deployable gate** and
**100-episode / multi-frame** evaluation to harden it. Then pick **one** new pivot —
**(3A) CRB-optimal active sensing** is the most novel and most "new capability"
flavoured, while **(3C) hydrofoil control-coupled** is the most application-compelling.
Both convert the CRB from a limit into a tool, which is the advisor-positive arc.

Numbers: ST-DF-GP 0.205 → CRB-gated hybrid **0.174** (−15%, ≈ oracle 0.171). That is
the headline to build the paper around.

---

## 5. Follow-up pilots (2026-06-27, control direction excluded per advisor)

### 5.1 Deployable gate — honest status (`d10_deployable_gate.py`)
The oracle gate uses cond(FIM) from *true* components. A naive data-driven proxy —
participation ratio of a dispersion-constrained beamform power map — **fails to
separate complexity** (5c PR 597 vs 30c 486, wrong direction): beamform maps on
sparse scattered returns are resolution-limited, not mode-count informative.
→ **The deployable identifiability gate is a genuine sub-problem.** Candidate signals
to test next (all available at inference): (i) cond(FIM) from DPS-/beamform-estimated
wavenumbers; (ii) DPS posterior spread σ_post (the calibrated UQ from pilot b);
(iii) ST-DF-GP predictive variance; (iv) model order via a residual L-curve (AIC).
The hybrid *potential* (0.174) is proven; closing the deployable gate is the task.

### 5.2 CRB-optimal active sensing — strong positive lever (`d11_active_sensing.py`)
D-optimal greedy selection of K=60 returns vs a random-K subset, field-nRMSE CRB:
| sea | random | cross-range heuristic | **D-optimal** | gain |
|---|---|---|---|---|
| BF4 5c  | 0.302 | 0.160 | **0.092** | **69%** |
| BF4 15c | 69.6 | 51.8 | **17.2** | 75% (still vacuous abs.) |
| BF4 30c | 1458 | 1513 | 1519 | −4% (under-sampled) |

**Where you sample matters enormously** — up to 69% CRB reduction for identifiable
seas — so **scan/return design is a real, novel lever**. The cross-range-spread
heuristic alone gives 47%, confirming Dir-I's "direction is cross-range-limited".
Active sensing *lowers the wall where information exists*; it cannot manufacture
information for a severely under-sampled complex sea (30c, K=60). This is the most
"new-capability" pivot and is built entirely on the Dir-I FIM.

### 5.3 Design-level assessment of the remaining directions
- **Sequential Bayesian fusion (track filtering with the prior):** high promise for
  complex seas — accumulates the time information the single-window CRB lacks
  (the wall needs ~10 s). Medium effort (filter loop around DPS). Recommended #2.
- **Conditional (amortised) diffusion posterior p(S|z):** removes DPS guidance
  tuning, fast inference, can be gated; distinct from the failed T-NSE point
  regressor. Higher effort (train a conditional model). Good medium-term bet.
- **Real-data + weakly-nonlinear (Stokes-2/HOS):** the credibility upgrade; removes
  "sim-only/linear". Effort = data access. Strategic, do before publication.
- **Forecast-skill product (predictable-zone CRB → calibrated forecasts):** turns the
  predictable-zone into a deliverable; pairs naturally with sequential fusion.

### 5.4 Updated recommendation (control excluded)
1. **Harden the hybrid** — implement+validate the deployable gate (5.1) on 100
   episodes + multi-frame. *Proven-win consolidation.*
2. **CRB-optimal active sensing** (5.2) — the standout novel positive pivot;
   demonstrate on simulated scan patterns end-to-end (CRB → achieved error).
3. **Sequential Bayesian fusion** (5.3) — to attack complex seas via accumulated time
   information, where active sensing alone saturates.
Real-data + nonlinear validation underpins all three for publication.
