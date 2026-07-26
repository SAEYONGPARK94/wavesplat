# RESULT — aperture arm of the discreteness experiment (W9-b)

**Run 2026-07-26.** Design frozen in `PREREG_APERTURE_ADDENDUM.md` before any episode was
generated. Raw: `ws_runs_aperture.jsonl` (100 runs, 0 err), `cond_runs_aperture.jsonl`
(100 runs, 0 err), `secondary_aperture.jsonl` (20 rows, 0 err). Analysis:
`analyze_aperture.py` → `results_aperture.json`, `noskill_floor_aperture.json`.
Nothing committed; no shipped default changed.

---

## 1. PRE-REGISTERED VERDICT — **APERTURE-LIMITED**

Frozen gate 0: *mean analytic skill in C-big < 0.15 ⇒ APERTURE-LIMITED.*

> **C-big mean analytic skill = 0.061 < 0.15 → APERTURE-LIMITED.**

Doubling the effective aperture did **not** make continuum seas solvable. Per the frozen rule,
**no discreteness verdict may be issued from this run**, and the recommendation becomes
observation-window / footprint expansion before retraining.

---

## 2. The lever, and proof it was actually applied

**Lever: observation window `T_obs`, at constant observation count.** `dt` 0.1 → **0.4 s**,
duration 30 → **120 s** (301 frames either way), read at `tp_factor` 2.0 → **8.0**.

| quantity | original cells (A, C) | aperture cells (A-big, C-big) | ratio |
|---|---|---|---|
| WS init window | 141 frames, 14.0 s | **141 frames, 56.0 s** (asserted per run) | 4.0× time, **1.0× frames** |
| observations in init window | ~7 066 | **6 930 – 6 962** (measured) | ~1.0 |
| cond window | 154 frames, 15.4 s | **154 frames, 61.6 s** (asserted) | 4.0× time, 1.0× frames |
| `c_g·T_obs` | 76.5 m | 306.0 m | 4.0 |
| **`D_eff = L + c_g·T_obs`** | **245.3 m** | **474.8 m** | **1.94** |
| `δk = 2π/D_eff` | 0.0256 rad/m | 0.0132 rad/m | 0.52 |

The runners *assert* the window (they do not assume it): the WS worker checks 141 frames / 56.000 s,
the cond runner checks `nb = 154` / 61.6 s. Both passed on every run. Data volume is matched to
within 2 %, so any change is attributable to `D_eff`, not to more data.

**The lever demonstrably works** — on discrete seas the analytic arm's skill rose from 0.543 to
**0.780** and its nRMSE halved (0.1141 → 0.0554). It simply is not enough for the continuum.

---

## 3. Solvability (frozen metric: skill = 1 − nRMSE/zero-predictor floor)

| cell | floor | ws_best | cond | **analytic skill** | learned skill | analytic episodes at-or-worse than zero pred. |
|---|---|---|---|---|---|---|
| **A-big** discrete @ D_eff 475 m | 0.2511 | 0.0554 | 0.0821 | **0.780** | 0.673 | 0/10 |
| **C-big** continuum @ D_eff 475 m | 0.2479 | 0.2310 | 0.2244 | **0.061** | 0.090 | **2/10** |
| *(parent) cell A @ D_eff 245 m* | 0.2501 | 0.1141 | 0.0851 | *0.543* | *0.660* | *0/10* |
| *(parent) cell C @ D_eff 245 m* | 0.2466 | 0.2426 | 0.2330 | *0.015* | *0.055* | *5/10* |

Nearly doubling `D_eff` bought **+0.237 skill on discrete seas but only +0.046 on continuum seas**
— a 5:1 asymmetry. The continuum cell is still essentially at the no-skill floor for both arms.

**How much aperture would continuum seas need?** Two-point log fit over the two apertures
(0.015 @ 245 m, 0.061 @ 475 m) gives 0.070 skill per e-fold of `D_eff`, so skill 0.15 needs
`D_eff ≈ 1 700 m` — **3.6× more than this run**. As dwell time alone that is `T_obs ≈ 281 s`
(~4.7 min), which is not physical for a stationary sensor (the sea is not stationary that long).
**With the program's design ship speed U = 11.3 m/s** the platform sweeps its own baseline:
`D_eff ≈ L + (c_g+U)·T_obs` = **1 107 m already at T_obs = 56 s**, and reaches ~1 700 m at
`T_obs ≈ 92 s`. *(Two-point extrapolation — indicative only, not a measured result.)*
⇒ **the promising lever is platform motion (synthetic aperture), not longer dwell.**

---

## 4. Advantage numbers (median-of-5 both arms, n = 10 per cell)

| cell | cond mean | ws_best mean | med Δ (abs) | **R (median rel)** | cond wins | Wilcoxon p |
|---|---|---|---|---|---|---|
| **A-big** discrete | 0.0821 | 0.0554 | **+0.0226** | **+43.8 %** | **1/10** | 0.998 |
| **C-big** continuum | 0.2244 | 0.2310 | −0.0110 | **−4.8 %** | 6/10 | 0.161 |

C-big: the advantage stays collapsed (−4.8 %, n.s.), the same picture as the parent cell C
(−5.7 %, n.s.) — but per the frozen gate this cannot be read as a discreteness result, because the
cell is not solvable.

### ⚠ A-big: the learned advantage **inverts** at the larger aperture
On *discrete* crowded seas — the exact regime where WavePrior's headline −45 % lives — the learned
proposal is **worse than the analytic init by +43.8 %, winning 1 of 10 episodes**. The analytic
init at `D_eff` 475 m reaches nRMSE 0.055, better than the learned arm's 0.082.

`R_Abig`/`p_Abig` were pre-registered quantities (they appear in frozen cell 2), so the numbers are
not post-hoc. The *causal reading* is, and it is confounded — state both:

- **Confound:** A-big uses fresh seeds 20–24, whose realized energy-weighted directional spread is
  **20.9°**, versus 28.7° for the parent cell A (seeds 10–14). A narrower realized spread is easier
  for the beamformer independently of aperture. **This run cannot separate the aperture gain from
  the seed draw**, so "aperture inverts the advantage" is a hypothesis, not a measured result.
- What *is* measured: at `D_eff` 475 m with these seeds, the analytic init beats the learned
  proposal decisively on discrete crowded seas.
- A seed-matched A-vs-A-big rerun (same seeds 10–14, both apertures) would settle it cheaply and is
  the single highest-value follow-up from this arm.

---

## 5. Secondary

| cell | n_selected (median, range) | bf_peak_ratio (median, IQR) |
|---|---|---|
| A-big discrete | 18.5, [11, 26] | 0.358, [0.200, 0.574] |
| C-big continuum | **30.0**, [21, 30] | 0.169, [0.155, 0.193] |
| *(parent) A / C @ 245 m* | *13.0 / 22.5* | *0.128 / 0.140* |
| REAL windows | — (real n_eff ≈ 24) | **0.879, [0.505, 0.952]** |

- **n_selected saturates the cap on continuum seas.** C-big's median is **30.0 = the `max_n=30`
  ceiling**, up from 22.5 at the smaller aperture. The proposal tiles a continuum with as many
  atoms as it is allowed; the cap is now binding, so `max_n` is itself a hidden parameter of the
  learned arm's continuum behaviour.
- **bf_peak_ratio tracks aperture, not discreteness.** A-big rose to 0.358 (into the real band)
  while the continuum cell stayed at 0.169. Combined with the parent run's non-monotonicity, the
  guard signal now confounds *aperture* with *sea structure* — it is not a usable regime
  coordinate. (Parent-run do-not-cite item, reinforced.)

## 6. Gates and validity

| cell | ok | mesh | atoms | max var_under | frames | realized Hs | realized spread |
|---|---|---|---|---|---|---|---|
| A-big | 10/10 | ADEQUATE | 15 | 0.0000 | 301 | 1.199 ± 0.001 | 20.9° |
| C-big | 10/10 | ADEQUATE | 465 | 0.0000 | 301 | 1.166 ± 0.033 | 29.5° |

- **G-mesh PASS** (20/20 ADEQUATE, var_under 0.0000) · **G-identity PASS** (A-big bit-identical to
  `jonswap_components` for all 5 seeds) · **G-roundtrip PASS** (301 frames, Hs 1.2, Tp 7.0, and the
  asserted 141-frame/56.0 s and 154-frame/61.6 s windows) · **G-spread** reported above.
- 100/100 + 100/100 arm runs, 0 errors, full 5+5 seeds on every episode. No cut taken.
- **`analyze_aperture.py` reports zero validity flags.**
- Only caveat carried forward: the A-vs-A-big cross-aperture comparison is **not seed-matched**
  (§4). Within-aperture comparisons (A-big vs C-big) are clean.

---

## 7. Answer to the question this run was commissioned to settle

The commission was to separate "learning exploits discreteness" from "continuum seas are
unsolvable at this aperture". **The frozen answer is the third cell: at 1.94× aperture the
continuum is still unsolvable (skill 0.061), so the separation could not be made.** What the run
adds is that this is not a marginal miss — the required aperture is ~3.6× further out, and the
gap between discrete and continuum *widens* rather than closes as aperture grows (+0.237 vs +0.046
skill).

**Recommendation (per the frozen APERTURE-LIMITED cell):**
1. **Do not launch the mixed retrain on the strength of this run.** Neither GO cell was reached.
   On continuum seas there is ≤ 0.09 skill available to either arm at this sensor configuration —
   a retrain would be chasing a target that barely exists.
2. **Expand the aperture first, and do it with platform motion**, not dwell: `U = 11.3 m/s` reaches
   `D_eff ≈ 1 100 m` at the same 56 s window (vs 475 m stationary). Re-run C-big there; if analytic
   skill clears 0.15, the discreteness question becomes answerable and the retrain decision can be
   made on evidence.
3. **Seed-matched A/A-big rerun** to test whether the learned advantage on *discrete* crowded seas
   really inverts with aperture. If it does, the WavePrior headline (−45 % crowded) is
   aperture-conditional and that qualifier belongs in the thesis claim.
4. Continue to treat `bf_peak_ratio` as unusable for regime routing (§5).
