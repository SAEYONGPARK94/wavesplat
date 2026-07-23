# Alt-methods gate (non-beamforming families) — result & audit — 2026-07-23

**Origin.** `wavesplat/tmp/altmethods_gate.py` was written at 10:48 on 2026-07-23 and never
run (session died). It is the follow-on to the Round-3 ANM gate: having found the *gridless
atomic-norm* reformulation ≈ the existing beamforming pipeline, the question was whether a
**different family** — dispersion-constrained wavenumber **bank** + regularized linear solve
(ridge / MMSE) or L1 (LASSO), i.e. no matched-filter peak-picking — buys anything.

**Metric:** field nRMSE = RMSE(prediction − truth over the output grid) ÷ Hs, averaged over
3 frames (nfr/4, nfr/2, 3nfr/4). All arms use the same 2600-point observation subsample,
same seed, same frames — **with one documented exception**: inside
`blasso_feasibility.greedy()`, the per-atom `refine_single`/`fit_amp` steps use the **full
~15 000-point episode**, not the subsample (only the coarse peak search is
subsample-restricted). A fair variant restricting every step to the subsample was built and
run by the red-team (`tmp/r4_fulldata_advantage_probe.py`): the difference is **≤ 0.4 %
relative** and slightly *favours* the fair variant (BF4 near-band 0.0601 vs 0.0603), because
`joint_refine`'s final L-BFGS pass — correctly subsample-restricted in both — washes out the
initialization difference. Empirically inert, but the unqualified "same inputs" phrasing was
inaccurate and is corrected here.

---

## TL;DR

> 🛑 **2026-07-23 정정 배너 — fable 적대감사 결과 반영.** 이 문서의 초판 TL;DR은
> "informed 밴드에서 두 계열이 **동등**(BANK 0.0716 < BEAM 0.0743)"이라고 했고, 그것은
> **9-프레임 복제로 반증되었다.** 권위 = `tmp/R4_REDTEAM_fable_2026-07-23.md`.
> ⛔ **인용 금지: "0.0716 vs 0.0743 = 동등/bank 근소승", "광역 격차는 *전부* 외삽 아티팩트".**
> 아래 본문은 정정판이며, 초판 수치는 §Run 4에 그대로 보존하되 대표성 없음으로 표시한다.

**Gate verdict: NEGATIVE — no novelty lever.** The bank/covariance family never beats the
beamforming component family anywhere tested. The wide-grid 3–4× gap decomposes into two
unequal parts:

- **Dominant part = extrapolation** (UPHELD, 18 frame-level replicates). ~49 % of grid
  points are >100 m from any return; there the shrinkage bank correctly falls back to ≈ zero
  while the parametric plane-wave model extrapolates for free — a **model-class gift** of a
  testbed whose truth *is* a finite plane-wave sum.
- **Smaller but real part = a genuine near-band estimation-efficiency gap** favouring the
  matched parametric family. 9-frame replication: BF3 5comp Δ(BANK−BEAM) = **+0.0117**,
  SE 0.0048, t = 2.46; BF4 15comp Δ = **+0.0227**, SE 0.0022, t = 10.36 (BEAM ahead on
  **every** frame). The single frame originally reported (nfr/2 = 150) was, for BF3, the
  **most BANK-favourable of 9** — the "parity" reading came from that outlier.

So: the gap is **predominantly, not entirely,** extrapolation. The gate conclusion is
unchanged and if anything cleaner — swapping estimator family buys nothing — but it rests on
"the bank never wins", not on "the families are equivalent".

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

`tmp/altmethods_nearfar.py`. Grid points binned by distance to the nearest observation.
Pre-registered read stated in the script header **before** the numbers. **Only 2 of the 3
episodes were run here** (BF3 5comp s1 dropped for runtime, without comment in the first
draft — noted now).

⚠ **Window-labelling bug (found by red-team, reproduced):** the script's primary in-frame
threshold selects only 18 of 2600 points, so the `< 50` fallback branch **silently fires**
every time; the printed `n_obs_in_frame = 41 / 51` are counts within a **~7-frame / 0.6 s
swath** (≈ 6.8 m of ship travel at the 11.3 m/s design speed), *not* a single ping. The
distance bands are therefore "distance to a short along-track swath", not "to one return".
This does not change which data fits which estimator (both arms always see the full 2600
subsample) and the same definition was used consistently across all replicate frames, so it
does not drive the correction below — but the physical reading of "within 20 m" must be
stated with the swath in mind.

⛔ **The single-frame table below is retained for the record but is NOT representative** —
see the 9-frame replication that follows it.

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

## Run 5 (red-team replication) — 9 frames per episode → **near-band "parity" REFUTED**

`tmp/r4_nearfar_power_probe.py` (fable red-team). Same fitted BEAM and BANK, same subsample
and seed, queried at 9 frames (30, 60, … 270) instead of one. Frame 150 reproduces the
published numbers exactly, confirming probe fidelity.

| episode | BEAM near (mean±sd) | BANK near (mean±sd) | Δ(BANK−BEAM) | SE | t | n |
|---|---|---|---|---|---|---|
| BF3 5comp s0 | 0.0742 ± 0.0005 | 0.0859 ± 0.0142 | **+0.0117** | 0.0048 | 2.46 | 9 |
| BF4 15comp s0 | 0.0608 ± 0.0130 | 0.0835 ± 0.0177 | **+0.0227** | 0.0022 | 10.36 | 9 |

For BF3, frame 150's Δ = −0.0027 was the **best of 9 frames for BANK**; 6/9 frames show BANK
behind by +0.006 to +0.036. For BF4, BANK is behind on **all 9**. Far-band behaviour
replicates cleanly across all 18 frame-level replicates (BF3 BEAM_far flat 0.0763–0.0774 vs
BANK_far 0.271–0.296; BF4 BEAM_far 0.108–0.178 vs BANK_far 0.205–0.251), so the
extrapolation mechanism is **UPHELD**.

**Power labels:** BF4 near-band gap = MEASURED, well powered (SE ≪ effect). BF3 near-band
gap = MEASURED but borderline (t = 2.46) — enough to reject the specific published headline
as representative, not enough for a strong standalone directional claim on that episode.
Cross-episode generalization ("a near-band gap is real") = ARGUED, n = 2.

Reads (corrected):
1. **In the informed band the bank family is consistently behind, not equivalent** — the
   published "bank edges beamforming" was a single-frame outlier. There is a real, smaller
   near-band estimation-efficiency gap in favour of the matched parametric model.
2. **The wide-grid verdict is still dominated by empty space**: ~49 % of grid points are
   >100 m from any return, and there the bank shrinks toward zero (→ ≈ zero-pred) by design.
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
  overfit). Tuned-bank vs beamforming on wide grid = 2–3× apart.
- **MEASURED (18 replicates):** the gap decomposes monotonically with distance-to-data, and
  the far-band component is by far the larger one.
- **MEASURED (9 frames/episode):** a smaller **near-band gap in favour of BEAM** also
  exists — BF4 well powered (t = 10.36), BF3 borderline (t = 2.46).
- **MEASURED:** the oracle atom count `n_true + 1` does **not** confound the comparison
  (sweep: BF3 flat over 3–20 atoms; BF4 only *under*-counting hurts, over-provisioning is
  free). The deployed pipeline uses `adaptive_n`, no oracle count.
- **ARGUED:** the estimator *family* is not a novelty lever — the bank never wins anywhere
  tested, and the bulk of its deficit is the testbed's plane-wave model-class gift rather
  than an estimation difference.
- **NOT licensed:** "the families are equivalent"/"bank edges beamforming" (**refuted**);
  "the wide-grid gap is *entirely* extrapolation" (revise to *predominantly*); any use of
  the far-field numbers as a quality ranking; any extension to real/HOS seas (synthetic
  linear episodes only, n = 2–3). Nonlinear (HOS) replication is **not** done; per standing
  discipline this gate is synthetic-only.

## Scope caveat for the companion paper

Wide-grid field nRMSE is **not** the metric under which the bank family was claimed to win.
The companion `encounter_forecast` paper evaluates **encounter correlation along the ship
track** — near-field prediction, i.e. the 0–20 m band, where this gate finds parity. Nothing
here contradicts that paper. See `tmp/INTEGRITY_FIXES_2026-07-23.md` item 2.

## Bottom line

Round 3 said the ANM reformulation ≈ the current pipeline. Round 4 (this gate) says the
**bank/covariance family never beats the current pipeline**, and that most of its apparent
deficit is an artifact of demanding extrapolation into empty space. Eleventh lens, same
destination: no estimator-family swap is a novelty lever.

**Two self-corrections logged** (both in the same session, opposite directions — the reason
the standing symmetric-audit discipline exists):
1. Run 1's "bank loses 4×" was **pessimistic** over-claim (invalid configuration below the
   zero predictor). Caught in-house before reporting.
2. Run 4's "near-band parity" was **optimistic** over-claim (n = 1 frame, and that frame the
   most favourable of 9). Caught by the fable red-team, not in-house — the single-frame
   design was mine and the pre-registered alternative reading was the correct one.
