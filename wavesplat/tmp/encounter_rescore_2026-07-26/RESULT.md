# RESULT — LOCAL encounter forecast vs. the GLOBAL-MAP information wall (W9-c)

**Run 2026-07-26 → 27.** Design frozen in `PREREG.md` before any encounter number was computed.
Raw: `ws_enc.jsonl` (800 runs, **0 err**), `cond_enc.jsonl` (800 runs, **0 err**).
Analysis `analyze.py` → `results.json`. Figure `results/gifs/encounter_rescore.png`.
Design checks: `DESIGN_CHECKS.md`. **Nothing committed; no shipped default changed.**

---

## 1. PRE-REGISTERED VERDICT — **LOCAL-SKILL-EXISTS**

Frozen cells, evaluated in the frozen order:

| # | frozen condition | measured | fires |
|---|---|---|---|
| **0 LOCAL-SKILL-EXISTS** | `R*(0.5, lead ≤ 2 s) ≥ 22 m` on a **continuum** cell (C or C-big) for **at least one arm** | **cell C, learned arm: R\* = 22 m** | **YES** |
| 1 NO-LOCAL-SKILL | `R*(0.3) = NONE` for every cell × arm | 6 of 8 cell×arm combinations reach ≥ 10 m | no |
| 2 PARTIAL | otherwise | — | not reached |

> **Verdict: LOCAL-SKILL-EXISTS.** The user's reframe is validated by the frozen rule: on a
> continuum sea where the **global map** is at the no-skill floor, a **local encounter forecast
> 0–2 s ahead at ≤ 22 m** carries skill 0.56–0.65.

**Read it with the three qualifiers, which are part of the result:**
1. It fires **exactly at the bar** (22 m ≥ 22 m), not comfortably past it.
2. It fires for the **learned arm only**. The true analytic arm reaches `R*(0.5) = NONE` on
   both continuum cells.
3. **C-big does not replicate** — at 1.94× aperture the learned arm's `R*(0.5)` drops to 0 m.

---

## 2. The substantive finding: a **RANGE** wall, not a **HORIZON** wall

This is the part that does not depend on where the 0.5 bar was drawn.

**Skill is flat in lead and collapses in distance.** Cell C, learned arm, at the 22 m control point:

| lead | 0 s | 1 s | 2 s | 5 s |
|---|---|---|---|---|
| skill | +0.558 | +0.584 | +0.648 | +0.630 |

Five seconds of forecast costs **nothing**. The same arm, at lead 0 s, across distance:

| distance | 0 m | 10 m | 22 m | 50 m | 100 m | 200 m |
|---|---|---|---|---|---|---|
| skill | +0.667 | +0.563 | +0.558 | +0.332 | −0.156 | −0.409 |

The identical pattern holds in all four cells and both arms (see the figure): the curves are
near-horizontal top-to-bottom (lead) and monotonically falling left-to-right (distance).

⟹ **What the W9 global-map wall measures is a spatial-extrapolation limit, not a
predictability-horizon limit.** The global map is scored over x ∈ [10, 250] m — i.e. it is
dominated by cells 100–200 m beyond the footprint, exactly where every arm has zero skill. A
control-relevant target sits at 22 m, where the same fitted model is at 0.56–0.65. **Averaging
the good near field with the dead far field is what produced "0.06".** This is the
`direction1_crb.md` *predictable zone* concept, measured directly for the first time.

---

## 3. Local vs global, measured on the *same fitted models*

Global-map skill = the same per-window fits scored on the full 80×60 GT grid at lead 0.

| cell | arm | **global-map skill** | **local skill @ 22 m, lead ≤ 2 s** | ratio |
|---|---|---|---|---|
| **C** continuum | analytic | **−0.124** | +0.279 … +0.432 | — (global < 0) |
| **C** continuum | learned | **+0.063** | **+0.558 … +0.648** | **≈ 10×** |
| **C-big** continuum | analytic | **−0.094** | −0.076 … +0.256 | — (global < 0) |
| **C-big** continuum | learned | **+0.097** | +0.192 … +0.465 | ≈ 4× |
| A discrete | analytic | +0.065 | +0.561 … +0.638 | ≈ 9× |
| A discrete | learned | +0.663 | +0.811 … +0.833 | 1.2× |
| A-big discrete | analytic | +0.258 | +0.312 … +0.421 | 1.4× |
| A-big discrete | learned | +0.667 | +0.746 … +0.807 | 1.2× |

The local/global gap is **largest exactly where W9 found the wall** (continuum cells) and
smallest where the problem was already solvable (discrete cells). The wall is where the
local–global distinction matters most.

---

## 4. Learned − analytic on the encounter metric: **learning helps LOCALLY on continuum seas**

Δskill = learned − analytic. **Positive at every distance, every lead, every cell — 96/96 cells.**

| cell | Δskill @ 0 m | @ 10 m | @ 22 m | @ 50 m | @ 100 m | @ 200 m | (range over leads) |
|---|---|---|---|---|---|---|---|
| **C** | +0.07…+0.15 | +0.14…+0.21 | +0.15…+0.28 | +0.27…+0.41 | +0.13…+0.32 | +0.07…+0.34 | all > 0 |
| **C-big** | +0.22…+0.32 | +0.25…+0.35 | +0.19…+0.34 | +0.23…+0.33 | +0.16…+0.49 | +0.14…+0.39 | all > 0 |
| A | +0.13…+0.26 | +0.20…+0.27 | +0.17…+0.27 | +0.32…+0.47 | +0.57…+0.82 | +0.61…+0.84 | all > 0 |
| A-big | +0.29…+0.46 | +0.34…+0.49 | +0.38…+0.43 | +0.35…+0.43 | +0.40…+0.52 | +0.26…+0.52 | all > 0 |

**Descriptive** (not a pre-registered endpoint) paired test on per-episode min-skill over
lead ≤ 2 s at the 22 m control point, n = 10 episodes, one-sided Wilcoxon:

| cell | median Δ | learned wins | p |
|---|---|---|---|
| **C** continuum | **+0.307** | 9/10 | 0.0049 |
| **C-big** continuum | **+0.244** | 10/10 | 0.0010 |
| A discrete | +0.395 | 9/10 | 0.0049 |
| A-big discrete | +0.594 | 10/10 | 0.0010 |

**This is the direct answer to "can learning matter on real-like seas at the right target":
yes — on continuum seas, at the control-relevant point and horizon, the learned proposal beats
the analytic initializer by ~0.25–0.31 skill, 9–10 episodes out of 10.** This is a *synthetic*
statement (see §8); it is not a real-data claim.

---

## 5. Full skill tables

Floors, nRMSE and per-episode spreads are in `results.json`.

### cell C — continuum, D_eff 245 m *(decision cell)*
| arm | lead | 0 m | 10 m | 22 m | 50 m | 100 m | 200 m |
|---|---|---|---|---|---|---|---|
| analytic | 0 s | +0.553 | +0.354 | +0.279 | −0.006 | −0.322 | −0.481 |
| analytic | 1 s | +0.454 | +0.545 | +0.432 | +0.076 | −0.227 | −0.755 |
| analytic | 2 s | +0.579 | +0.445 | +0.399 | +0.035 | −0.204 | −0.438 |
| analytic | 5 s | +0.429 | +0.344 | +0.407 | +0.122 | −0.058 | −0.381 |
| **learned** | 0 s | +0.667 | +0.563 | +0.558 | +0.332 | −0.156 | −0.409 |
| **learned** | 1 s | +0.599 | +0.681 | +0.584 | +0.346 | −0.036 | −0.416 |
| **learned** | 2 s | +0.647 | +0.651 | +0.648 | +0.447 | +0.118 | −0.156 |
| **learned** | 5 s | +0.542 | +0.498 | +0.630 | +0.393 | +0.068 | −0.251 |

`R*(0.5)`: analytic **NONE**, learned **22 m** · `R*(0.3)`: analytic **10 m**, learned **50 m**

### cell C-big — continuum, D_eff 475 m *(decision cell)*
| arm | lead | 0 m | 10 m | 22 m | 50 m | 100 m | 200 m |
|---|---|---|---|---|---|---|---|
| analytic | 0 s | +0.286 | +0.280 | +0.141 | −0.061 | −0.223 | −0.165 |
| analytic | 1 s | +0.202 | +0.103 | +0.256 | −0.023 | −0.386 | −0.257 |
| analytic | 2 s | +0.300 | +0.213 | −0.076 | −0.150 | −0.170 | −0.352 |
| analytic | 5 s | +0.279 | +0.153 | +0.071 | −0.281 | −0.100 | −0.185 |
| **learned** | 0 s | +0.586 | +0.530 | +0.465 | +0.269 | +0.031 | +0.032 |
| **learned** | 1 s | +0.520 | +0.403 | +0.451 | +0.281 | +0.104 | −0.115 |
| **learned** | 2 s | +0.623 | +0.482 | +0.192 | +0.080 | +0.043 | +0.038 |
| **learned** | 5 s | +0.495 | +0.499 | +0.406 | +0.037 | +0.057 | +0.069 |

`R*(0.5)`: analytic **NONE**, learned **0 m** · `R*(0.3)`: analytic **NONE**, learned **10 m**

### cell A — discrete reference, D_eff 245 m
| arm | lead | 0 m | 10 m | 22 m | 50 m | 100 m | 200 m |
|---|---|---|---|---|---|---|---|
| analytic | 0 s | +0.567 | +0.600 | +0.601 | +0.291 | −0.217 | −0.382 |
| analytic | 1 s | +0.699 | +0.595 | +0.561 | +0.396 | −0.194 | −0.110 |
| analytic | 2 s | +0.667 | +0.583 | +0.638 | +0.394 | +0.113 | −0.176 |
| analytic | 5 s | +0.711 | +0.650 | +0.642 | +0.423 | +0.128 | −0.138 |
| **learned** | 0 s | +0.824 | +0.813 | +0.811 | +0.758 | +0.600 | +0.459 |
| **learned** | 1 s | +0.825 | +0.833 | +0.833 | +0.711 | +0.564 | +0.503 |
| **learned** | 2 s | +0.863 | +0.848 | +0.811 | +0.795 | +0.684 | +0.552 |
| **learned** | 5 s | +0.857 | +0.849 | +0.843 | +0.854 | +0.693 | +0.577 |

`R*(0.5)`: analytic **22 m**, learned **100 m** · `R*(0.3)`: analytic **22 m**, learned **200 m**

### cell A-big — discrete reference, D_eff 475 m
| arm | lead | 0 m | 10 m | 22 m | 50 m | 100 m | 200 m |
|---|---|---|---|---|---|---|---|
| analytic | 0 s | +0.460 | +0.429 | +0.382 | +0.351 | +0.201 | +0.096 |
| analytic | 1 s | +0.345 | +0.310 | +0.421 | +0.281 | +0.203 | +0.292 |
| analytic | 2 s | +0.449 | +0.395 | +0.312 | +0.377 | +0.274 | +0.324 |
| analytic | 5 s | +0.309 | +0.265 | +0.377 | +0.326 | +0.302 | +0.339 |
| **learned** | 0 s | +0.749 | +0.770 | +0.796 | +0.765 | +0.698 | +0.611 |
| **learned** | 1 s | +0.801 | +0.760 | +0.807 | +0.714 | +0.719 | +0.651 |
| **learned** | 2 s | +0.758 | +0.761 | +0.746 | +0.722 | +0.737 | +0.584 |
| **learned** | 5 s | +0.747 | +0.754 | +0.757 | +0.758 | +0.704 | +0.649 |

`R*(0.5)`: analytic **NONE**, learned **200 m** · `R*(0.3)`: analytic **22 m**, learned **200 m**

---

## 6. ⚠⚠ INCIDENTAL FINDING — the parent W9 "analytic arm" was **not analytic**

Found while pinning `init_method` for this run; **triple-verified**; details and the bit-identity
table in `DESIGN_CHECKS.md` §5.

`tmp/discreteness_2026-07-26/run_ws.py` and `run_ws_aperture.py` inherit the **shipped**
`init_method`. That default was flipped `v2_konly → 'auto'` in commit `aa8e4c3` at
**2026-07-26 11:53**; the W9 WS arms ran at **14:16–14:42** and **18:43**. On these cells the
`auto` router selects **learned** (n_selected 18, bf_peak_ratio 0.44 — no guard veto).

Three independent confirmations:

1. **Bit-identity.** Re-running the exact W9 config reproduces the stored `ws_runs.jsonl`
   *only* under `auto`: cell a/seed42 init 0.148491 (stored) = 0.148491 (auto) ≠ 0.234347
   (v2_konly); cell c/seed42 0.252451 = 0.252451 (auto) ≠ 0.284619 (v2_konly).
2. **Independent-protocol replication.** This run's lead-0 full-grid anchor reproduces W9's
   **cond** arm to ±0.01 while the "ws" arm is far off — i.e. the machinery agrees, the arm
   identity does not:

   | cell | this run, ws = **true** `v2_konly` | this run, cond | W9 stored "ws" (= auto) | W9 stored cond |
   |---|---|---|---|---|
   | C | **−0.124** | +0.063 | +0.015 | +0.055 |
   | C-big | **−0.094** | +0.097 | +0.061 | +0.090 |
   | A | **+0.065** | +0.663 | +0.543 | +0.660 |
   | A-big | **+0.258** | +0.667 | +0.780 | +0.673 |
3. **Structural fingerprint.** With `v2_konly` pinned, `n_gaussians == max_n == 30` occurs in
   **1/800** runs here. In W9 it occurs in **32/250** — and only on the continuum cells
   (0/50 cell A, 16/50 cell C, 16/50 cell C-big), mirroring the learned proposal's measured
   `n_selected` saturation on continuum seas.

**What this changes in the W9 write-ups** (flagged for the caller; re-deriving W9's own
endpoints is out of scope here — the protocols differ):

- `RESULT.md` §3's *"in the continuum cell neither method has meaningful skill … analytic at
  1.5 % skill"* is wrong in both parts. The **true** analytic arm is **below the zero
  predictor** (−0.124 / −0.094), and the learned arm is above it (+0.063 / +0.097). "Both arms
  hit the no-skill floor" should read "**the learned arm sits just above it and the analytic arm
  sits below it**".
- `RESULT.md` §2's *"control-cell caveat"* — R_A = −29.0 % vs the −64.6 % reference, explained
  there as "the analytic arm did much better on the fresh seeds" — is **explained by the
  contamination instead**. With the true analytic arm, this run's cell-A nowcast ratio is
  (0.0846 − 0.2344)/0.2344 = **−63.9 %**, i.e. back on top of the −64.6 % reference.
- `RESULT_APERTURE.md` §4's flagged *"the learned advantage INVERTS at larger aperture
  (+43.8 %, analytic 9/10)"* rests on the same contaminated comparator: here A-big's true
  analytic global-map skill is **+0.258**, not the +0.780 W9 recorded. The inversion flag —
  and the "the −45 % crowded headline needs an *at sub-wavelength aperture* qualifier"
  consequence in `docs/HANDOFF_OPUS_2026-07-26.md` — **should be treated as unsupported until
  the W9 arms are re-run with `init_method` pinned.**
- The *direction* of W9's CONFIRM (learned advantage shrinks from discrete to continuum) is
  **not** overturned by anything here — at 22 m the learned edge is +0.395 on cell A vs +0.307
  on cell C, and the global-map ratio goes −63.9 % (A) → −16.6 % (C). The **magnitudes** and
  the "no-skill floor for both" framing are what need re-deriving.

---

## 7. Robustness, caveats and protocol deviations

**Robustness of the verdict to the floor estimator (post-hoc sensitivity, clearly labelled).**
The frozen floor is estimated from only 40 point samples per (cell, distance, lead), so it
carries sampling noise (measured range 0.145–0.278 against an expected ≈ 0.25). Recomputing
skill against a **robust per-cell floor** (grid-RMS of the realized GT, 0.2476 / 0.2478 /
0.2508 / 0.2493) moves every `R*` **up or leaves it unchanged** — the frozen metric is the
conservative one:

| cell | arm | `R*(0.5)` frozen → robust | `R*(0.3)` frozen → robust |
|---|---|---|---|
| C | analytic | NONE → 10 m | 10 m → 22 m |
| **C** | **learned** | **22 m → 22 m** | 50 m → 50 m |
| C-big | analytic | NONE → NONE | NONE → NONE |
| C-big | learned | 0 m → 10 m | 10 m → 50 m |
| A | analytic | 22 m → 22 m | 22 m → 50 m |
| A | learned | 100 m → 200 m | 200 m → 200 m |
| A-big | analytic | NONE → NONE | 22 m → 50 m |
| A-big | learned | 200 m → 200 m | 200 m → 200 m |

**The verdict is unchanged under both floors.**

**Caveats (measured, in `DESIGN_CHECKS.md`):**
- **Observation support.** `x_edge` = p95 of window returns ≈ 39 m; ~1 % of returns lie beyond
  the 22 m point, 0.04 % beyond 100 m, 0 % beyond 200 m. So d ≤ 22 m is at the **edge of the
  observed footprint**, not deep in unobserved water: the near-field skill is partly
  reconstruction. **The lead axis is the clean forecasting axis, and it is flat.**
- **Fixed +x encounter axis.** Well aligned on the decision cells (off-track ≤ 12 m at 200 m,
  ≤ 0.2 λ_p); on the *discrete reference* cells the realized direction wanders ±15°, so their
  far-field numbers are a lower bound. The alternative (a per-episode GT-derived axis) would
  leak ground truth into the geometry.
- **C-big < C is measured, its cause is not.** The larger-aperture continuum cell is worse at
  the local task at every distance. Hypothesis (**flagged, not measured**): the aperture lever
  is `dt` 0.1 → 0.4 s, so the window spans 61.6 s instead of 15.4 s, and a continuum sea
  decoheres across it — a longer dwell buys global-map aperture and costs local currency.
  This deserves its own test before it is repeated as an explanation.
- **No spatial envelope in either arm** (both are global plane-wave sums), so the far-field
  collapse is physical, not representational.

**Declared deviations from the parent W9 arms** (all in `PREREG.md` §7, frozen in advance):
window slid to end at the shared origin (same 141-frame count/duration); **no tracking**
(frozen-init forecast only, per the amendment's "still-cut"); `init_method` pinned to
`v2_konly`; analytic GT instead of grid interpolation. **No budget cut was taken** — both arms
ran the full 800 runs at 5 seeds, 0 errors.

---

## 8. What this licenses

**Licensed:** on synthetic BF4 seas — including continuum-like ones where the global-map
reconstruction is at or below the no-skill floor — a per-window, no-tracking encounter forecast
at a point ≤ 22 m down-wave of the LiDAR footprint carries **skill 0.56–0.65 at leads 0–5 s**,
about **10× the same fitted model's global-map skill**; skill is **flat in lead and falls with
distance**; and the **learned** set-proposal beats the analytic initializer on this local metric
on continuum seas by **+0.24 to +0.31 median skill, 9–10 of 10 episodes** (descriptive
p = 0.005 / 0.001).

**NOT licensed:**
- No real-data claim of any kind. `docs/HANDOFF_OPUS_2026-07-26.md` §5 stands unchanged: no
  learned **accuracy** advantage on real data; the real-data contribution remains collapse
  insurance, and even that now reads as no measured contribution on either axis.
- Not "the wall is gone". On continuum seas the analytic arm still fails the 0.5 bar at every
  distance, C-big fails it for both arms, and **beyond ~50–100 m both arms are at or below the
  zero predictor in every cell**. The wall is real — it is a **range** wall.
- Not a re-derivation of the W9 discreteness endpoints. §6 flags them; it does not replace them.
- The 22 m result is at the **edge of the observed footprint**; do not restate it as
  "forecasting 22 m into unobserved water".
- Single-seed episode deltas remain forbidden; every number here is median-of-5.

## 9. Follow-ups this run earns

1. **Re-run the W9 arms with `init_method` pinned** (both the base and aperture cells). Until
   then, W9's `R_A`, `R_C`, the "both arms at the no-skill floor" reading, and the A-big
   inversion flag in the HANDOFF are all resting on a contaminated comparator.
2. **Audit every other result produced after `aa8e4c3` (2026-07-26 11:53)** whose "analytic"
   arm did not pin `init_method`. The same one-line omission would silently convert any
   analytic baseline into the auto router.
3. **Score the thesis's own target, not the map.** The 80×60 grid over x ∈ [10, 250] m
   averages a skilful near field with a dead far field. If the frozen 2026-07-05 target is
   encounter forecasting, the headline metric should be the encounter metric — this run gives
   the protocol.
4. **Test the dwell-vs-currency trade** (C-big < C) directly: platform motion, which was
   `RESULT_APERTURE.md`'s recommended lever, buys aperture *without* lengthening the window,
   and is the one lever predicted to help both the global and the local target.
