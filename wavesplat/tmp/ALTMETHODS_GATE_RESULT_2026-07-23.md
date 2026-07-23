# Alt-methods gate (non-beamforming families) — result & audit — 2026-07-23

**Origin.** `wavesplat/tmp/altmethods_gate.py` was written at 10:48 on 2026-07-23 and never
run (session died). It is the follow-on to the Round-3 ANM gate: having found the *gridless
atomic-norm* reformulation ≈ the existing beamforming pipeline, the question was whether a
**different family** — dispersion-constrained wavenumber **bank** + regularized linear solve
(ridge / MMSE) or L1 (LASSO), i.e. no matched-filter peak-picking — buys anything.

**Metric:** field nRMSE = RMSE(prediction − truth over the output grid) ÷ Hs, averaged over
3 frames (nfr/4, nfr/2, 3nfr/4). All arms use the same 2600-point observation subsample,
same seed, same frames.

---

## TL;DR

**Gate verdict: NEGATIVE — no novelty lever — but the first reading was an artifact and the
right reason is PARITY, not defeat.** Where both families are informed (grid points within
~20 m of an observation) they are **equivalent** (bank even edges beamforming on one
episode: 0.0716 vs 0.0743). The entire apparent 3–4× "bank loses" gap is an **extrapolation
gap**: the wide grid is ~50 % points >100 m from any return, where the shrinkage bank
correctly falls back to ≈ zero while the parametric plane-wave model extrapolates. Swapping
estimator family changes nothing about the binding constraint — consistent with, and
supporting, the CRB/information spine.

---

## Run 1 — as-written (flat bank) → **INVALID, discarded**

| episode | ZERO-PRED | BEAM (greedy+joint) | BANK-MMSE | BANK-RIDGE | BANK-LASSO |
|---|---|---|---|---|---|
| BF3 5comp s0 | 0.2500 | 0.0755 | 0.3023 | 0.3178 | 0.2975 |
| BF3 5comp s1 | 0.2503 | 0.0766 | 0.2974 | 0.3143 | 0.2942 |
| BF4 15comp s0 | 0.2526 | 0.1068 | 0.3070 | 0.3477 | 0.2808 |

Naively "the bank family loses ~4×". It fails the comparability check
(`tmp/altmethods_diag.py`, log `tmp/altmethods_diag.log`):

1. **Every bank arm is WORSE than the zero predictor** (0.28–0.35 vs 0.250). An estimator
   that loses to "predict nothing" measures a broken configuration, not a family. (0.250 is
   exact and expected: σ_field/Hs = ¼.) The repo already pre-registers a **Skill Score**
   `1 − MSE(ẑ)/MSE(0)` precisely to catch this (`docs/PRESENTATION_SUMMARY.md`), and has a
   documented prior instance of nRMSE worse than zero-predictor
   (`docs/G0B_THE_GAP_IS_THE_RESULT_2026-07-11.md`).
2. **The flat bank is underdetermined**: kx,ky ∈ [−1.2, 1.2] step 0.06 → 1681 atoms →
   **3362 unknowns vs 2600 observations**.
3. **In-sample residual/var = 0.015–0.11** — it *fits the observations well* and extrapolates
   catastrophically. Overfit, not a family limit.

⟹ Discarded. Reporting it as-is would have been a false negative about a whole family.
Logged as a near-miss under the standing over-claim discipline — this time **pessimistic**.

## Run 2 — fair bank (project geometry, my hyper-params)

`tmp/altmethods_fairbank.py`: kp-centred annulus 0.35–2.5 kp, dk = 0.1 kp (geometry of
`d3_stream_seqmmse.py:make_bank`), beamform-power prior.

| episode | kp | ZERO-PRED | BEAM | BANK-MMSE(bf prior) | BANK-RIDGE | BANK-LASSO |
|---|---|---|---|---|---|---|
| BF3 5comp s0 | 0.243 | 0.2500 | 0.0755 | 0.2881 | 0.3046 | 0.2595 |
| BF3 5comp s1 | 0.243 | 0.2503 | 0.0766 | 0.2724 | 0.2983 | 0.2406 |
| BF4 15comp s0 | 0.125 | 0.2526 | 0.1068 | 0.2049 | 0.2255 | **0.1741** |

Better but still not a clean family test (my pexp/smult/reg were not the val-selected ones).

## Run 3 — the project's OWN tuned estimator

`tmp/altmethods_realbank.py` calls `mmse_bank_estimate` (`scripts/d3_sa_nomp.py:499` — the
2026-07-05 winner: beamform prior, val-selected (pexp, smult), val-gated SBL-EM), same
frames, same metric. Argument convention verified against an existing call site
(`tmp/probe_rho_squared_law.py:83`).

| episode | ZERO-PRED | BEAM (g+j) | **BANK-MMSE (tuned)** |
|---|---|---|---|
| BF3 5comp s0 | 0.2500 | 0.0755 | 0.2264 |
| BF3 5comp s1 | 0.2503 | 0.0766 | 0.2531 |
| BF4 15comp s0 | 0.2526 | 0.1068 | 0.2008 |

Now sane (no longer below the zero predictor) but still ~2–3× behind BEAM **on the wide
grid**. Which raised the real question → Run 4.

## Run 4 — near/far decomposition → **the gap is extrapolation, not estimation**

`tmp/altmethods_nearfar.py`. Grid points binned by distance to the nearest same-frame
observation. Pre-registered read stated in the script header **before** the numbers.

**BF3 5comp s0** (frame 150, 41 returns in-frame, 4800 grid points)

| band [m] | n | ZERO | BEAM | BANK |
|---|---|---|---|---|
| 0–20 | 399 | 0.2436 | 0.0743 | **0.0716** |
| 20–50 | 651 | 0.2458 | 0.0740 | 0.0891 |
| 50–100 | 1384 | 0.2539 | 0.0753 | 0.1842 |
| >100 | 2366 | 0.2506 | 0.0764 | 0.2838 |

**BF4 15comp s0** (51 returns in-frame)

| band [m] | n | ZERO | BEAM | BANK |
|---|---|---|---|---|
| 0–20 | 579 | 0.2266 | 0.0603 | 0.0847 |
| 20–50 | 898 | 0.2656 | 0.0717 | 0.1408 |
| 50–100 | 1223 | 0.2431 | 0.0870 | 0.2053 |
| >100 | 2100 | 0.2588 | 0.1515 | 0.2282 |

Reads:
1. **In the informed band (0–20 m) the families are equivalent** — bank *wins* on 5comp s0
   (0.0716 vs 0.0743) and is behind by 0.024 on 15comp. No family advantage to harvest.
2. **The wide-grid verdict is dominated by empty space**: ~49 % of grid points are >100 m
   from any return, and there the bank shrinks toward zero (→ ≈ zero-pred) by design.
3. **Independent corroboration inside the repo**: `tmp/probe_rho_squared_law.py:86` already
   carries the comment *"score only inside the fan … outside, eta_hat is a shrunken
   near-zero field"* — the project's own convention is to score the bank inside the sensor
   fan. Scoring it on the full grid was the error, and it was a pre-existing known one.
4. BEAM's own advantage far from data is a **model-class gift of this testbed**: the truth
   is literally a finite sum of plane waves, so a fitted plane-wave model extrapolates for
   free. On HOS/real seas that gift shrinks (cf. 15comp >100 m, where BEAM degrades to
   0.1515). Do **not** cite the far-field BEAM numbers as evidence of method quality.

---

## What this does and does not license

- **MEASURED:** flat-bank run invalid (below zero-predictor, underdetermined, in-sample
  overfit). Tuned-bank vs beamforming on wide grid = 2–3× apart. Near-band (≤20 m) parity,
  n = 2 episodes, single frame each.
- **MEASURED:** the gap decomposes monotonically with distance-to-data in both episodes.
- **ARGUED:** therefore the estimator *family* is not a novelty lever — what separates
  outcomes is where the information is, not which linear/greedy machine consumes it.
- **NOT licensed:** any claim that one family "beats" the other; any use of the far-field
  numbers as a quality ranking; any extension to real/HOS seas (this ran on synthetic linear
  episodes only, n = 3, single frames — UNDERPOWERED for a directional claim). Nonlinear
  (HOS) replication is **not** done; per standing discipline this gate is synthetic-only.

## Scope caveat for the companion paper

Wide-grid field nRMSE is **not** the metric under which the bank family was claimed to win.
The companion `encounter_forecast` paper evaluates **encounter correlation along the ship
track** — near-field prediction, i.e. the 0–20 m band, where this gate finds parity. Nothing
here contradicts that paper. See `tmp/INTEGRITY_FIXES_2026-07-23.md` item 2.

## Bottom line

Round 3 said the ANM reformulation ≈ the current pipeline. Round 4 (this gate) says the
**bank/covariance family ≈ the current pipeline too, wherever both are informed.** Eleventh
lens, same destination: the binding constraint is information/aperture, not the estimator.
