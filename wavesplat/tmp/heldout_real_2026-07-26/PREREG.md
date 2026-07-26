# PREREG — Held-out real-data detector evaluation (W6 + W7 + W8)

**Frozen: 2026-07-26, BEFORE any held-out score was computed.**
Source of bars: `docs/PERSONA_DEBATE_novelty_phd_value_2026-07-26.md` §6 (W6, W7, W8) and
`tmp/persona_debate_2026-07-26/R2_P3_methodologist.md` §CRUX D "Real 8 records" +
`tmp/persona_debate_2026-07-26/R2_P2_ocean_prof.md` §CRUX D (13-record collapse harvest).

Nothing in this file may be edited after the first held-out score is produced.
CPU ONLY. The learned arm is **not** run (GPU reserved). No file under `scripts/`,
`wavesplat/`, `wavesim/`, `data/` is modified. Nothing is committed.

---

## 0. What is already known (in-sample; NOT re-decided here)

From `tmp/unified_2026-07-26/hand_amp.json` + `fallback_results.json` (design set, 30 windows):

- Design records (IN-SAMPLE, used to choose τ): `YS17, AA15a, AA14, AA15b, LJ18g`.
- Collapse definition (frozen): `hand_now >= 1.0`. On the design set this selects exactly
  `{AA14_w0 (2.309), AA15a_w3 (1.120), LJ18g_w3 (2.925)}` — identical to the hard-coded
  `COLLAPSE` set, confirming the definition is not a separate free parameter.
- Detector (frozen): **fires iff `amp_ratio_grid > 1.5`**, where
  `amp_ratio_grid = rms(hand nowcast field over the query grid) / rms(observed z)`.
  Both terms are metres ⟹ the statistic is dimensionless and scale-free, so it is
  comparable across records with different Hs. Definition taken verbatim from
  `tmp/unified_2026-07-26/rescore_hand_amp.py`.
- In-sample behaviour: fired on 2/3 collapses (2.402, 2.534), 0/27 false fires,
  max normal `amp_ratio_grid` = 1.196 (`LJ18g_w4`).

## 1. Record inventory and held-out declaration

13 `.nc` files on disk under `data/real_bergamasco/` (verified by
`tmp/heldout_real_2026-07-26/probe_geometry.py`; results in `geometry.json`).

**DESIGN (in-sample, 5):**

| id | file | patch Ly×Lx [m] | dx [m] | dt [s] |
|---|---|---|---|---|
| AA14 | AA_2014/Surfaces_20140327_091000.nc | 86×70 | 0.20 | 0.083 |
| AA15a | AA_2015/Surfaces_20150305_103500.nc | 85×65 | 0.20 | 0.083 |
| AA15b | AA_2015/Surfaces_20150515_090000.nc | 80×55 | 0.20 | 0.083 |
| LJ18g | LJ_2018/…_Grid2_short.nc | 300×250 | 0.50 | 0.100 |
| YS17 | YS_2017/Surfaces_20170513_050000.nc | 240×220 | 1.00 | 0.100 |

**HELD-OUT (8):**

| id | file | patch Ly×Lx [m] | Lmin [m] |
|---|---|---|---|
| BS11a | BS_2011/Surfaces_20111004_113800_short.nc | 13.5×14.0 | 13.5 |
| BS11b | BS_2011/Surfaces_20111001_161800_short.nc | 13.0×13.5 | 13.0 |
| BS11c | BS_2011/Surfaces_20111004_130700_short.nc | 13.5×13.5 | 13.5 |
| BS11d | BS_2011/Surfaces_20111004_153000_short.nc | 14.5×19.0 | 14.5 |
| BS13a | BS_2013/Surfaces_20130925_121501_short.nc | 19.0×19.0 | 19.0 |
| BS13b | BS_2013/Surfaces_20130930_102001_short.nc | 17.0×17.0 | 17.0 |
| BS13c | BS_2013/Surfaces_20130922_130001_short.nc | 19.0×19.0 | 19.0 |
| LJ18n | LJ_2018/…_Grid1_short.nc | 40.0×40.0 | 40.0 |

This is exactly the "8 unused Bergamasco records" the debate demanded in W7.

### 1.1 How the held-out records enter the existing `scene_real` path

`d3_compare_gifs.scene_real` reads `d3_falcon_real_operator.RECORDS`, which contains only
the 5 design records ("only records whose patch can host a (possibly scaled) forward fan").
The 8 held-out entries are **injected into the live dict at runtime** by
`tmp/heldout_real_2026-07-26/heldout_cfg.py` (additive `.update`; no source file edited,
no loader logic touched). Required fields and how they are fixed — **frozen rule, derived
from the design records only, never from any score**:

- `depth`, `stride`: taken verbatim from the pre-existing committed config
  `scripts/d3_sa_multirecord.py::RECORDS` (BS site depth 30.0 m, LJ 45.0 m; BS stride 1,
  LJ stride 2). **No new physical constant is invented by this analysis.**
- `scale = Lmin / 55.0`, where 55.0 m = AA15b, the *smallest design patch* that hosts the
  design "0.4×-scaled" fan.
- `r_fan = (8·scale, 40·scale)` m, `U = 1.2·scale` m/s — geometric similarity to the AA
  design recipe; because both the fan and the ship speed scale with `Lmin`, the
  ship-traverse-to-patch ratio is held at its design value by construction.
- `lead_max = 14.0` s (AA design-family value, unscaled). The endpoint of this study is the
  **lead-0 nowcast**, so `lead_max` affects only the reserved frame count and the trajectory
  centring.

**Feasibility (probed BEFORE freezing, load-only, no scoring —
`tmp/heldout_real_2026-07-26/probe_load.json`): all 8 held-out records build successfully
through the unmodified `scene_real` at win 0 and win 5. Zero records are excluded on
loading grounds.**

### 1.2 Declared threat to validity, stated before scoring

`scene_real` fixes the query-grid spacing at an absolute 4.0 m (`sq = round(4.0/dx)`).
On the small BS patches this yields very few query points, whereas the design records yield
hundreds to thousands:

| arena | records | query grid | n query px |
|---|---|---|---|
| design | AA×3 | ~18×22 | ~400 |
| design | YS17, LJ18g | ~56×61, ~63×63 | 3400–4000 |
| held-out | BS11a/b/c | 4×4 | 16 |
| held-out | BS11d | 4×5 | 20 |
| held-out | BS13a/b/c | 5×5 | 25 |
| held-out | LJ18n | 11×11 | 121 |

nRMSE on 16 points is a far noisier statistic than on 400+. Changing the 4.0 m constant
would be a loader modification and is therefore **forbidden**. Instead:

- **PRIMARY analysis** = all 8 held-out records (48 windows).
- **Pre-registered SECONDARY (grid-adequacy) subset** = held-out records with
  **≥ 25 query points**: `BS13a, BS13b, BS13c, LJ18n` (n = 4 records, 24 windows).
  Both are reported; neither may be selected after the fact.

Also stated in advance: on the BS patches and on LJ18n the spatial peak wavelength returned
by `directional_spread` is **patch-limited** (probe: λ_p ≈ 13–19 m on 13–19 m patches;
λ_p ≈ 40 m on the 40 m LJ Grid1 patch whose true swell is λ ≈ 300 m). This is the artifact
already documented for LJ Grid1 in `d3_sa_multirecord.py`. `kp` enters the hand arm only as
`kp_hint`, so the consequence is a *harder, less well-posed* problem on the held-out set,
i.e. the specificity test is **conservative** (biased toward producing false fires), not
optimistic.

## 2. HARVEST protocol (W8)

- Windowing: **unchanged** from `wp_real`. `scene_real` hard-codes
  `i0 = linspace(nfw+1, T_all-n_lead-2, 6)[win]` ⟹ exactly **6 windows per record**, evenly
  spaced over the usable span. No record supports "more windows" without editing
  `scene_real`, so 6/record is used everywhere. Total = 13 × 6 = **78 windows**.
- Hand (analytic) arm = `d3_sa_nomp.mmse_bank_estimate(sc['P'], sc['z'], sc['depth'],
  sc['Pq'], sigma0=0.05*Hs, kp_hint=sc['kp'], seed=sc['seed'])` — the exact call used by
  `run_wp_real.py` and `rescore_hand_amp.py`.
- Per window record: `hand_now` (lead-0 nowcast nRMSE), `amp_ratio_grid`, `amp_ratio_obs`,
  `z_rms`, `pred_grid_rms`, `Hs`, `kp`, `n_obs`, `npx`.
- **The 30 design windows are NOT recomputed**: they are read verbatim from
  `tmp/unified_2026-07-26/hand_amp.json`, which already passed a bit-identity control
  against `wp_real.json` (`hand_now_maxdiff < 1e-9`). Only the 48 held-out windows are run.
- The learned arm is **not** run anywhere in this study.

## 3. FROZEN BARS

### R1 — held-out specificity (PRIMARY), τ = 1.5 frozen
Population: held-out windows with `hand_now < 1.0` ("normal"). Statistic: number of windows
with `amp_ratio_grid > 1.5` ("false fires").

- **0 false fires ⟹ PASS**
- **1 false fire ⟹ PARTIAL** (report cluster-robust exact-binomial CI at the record unit)
- **≥ 2 false fires ⟹ FAIL**
- **Additional FAIL condition:** any single record contributing more than one false fire.
- Analysis unit for any inference = **RECORD** (n = 8 primary; n = 4 in the adequacy subset).
  The cluster-robust CI is the exact binomial CI on (# records with ≥1 false fire)/(# records).

### R2 — separation margin (replaces all sensitivity tables)
- `margin = min(amp_ratio_grid | collapse windows that FIRED) − max(amp_ratio_grid | normal)`
- Reported **pooled (design + held-out)** and **per arena** (design alone, held-out alone).
- **Separation is declared BROKEN if any normal window exceeds 1.5**, regardless of the
  margin's sign or of the false-fire count. (Note, stated in advance: this condition is
  arithmetically identical to "R1 recorded ≥1 false fire"; it is retained verbatim as
  written by P3 so that the margin is reported as a magnitude, not only as a count.)
- In-sample reference to be beaten/matched: design margin = 2.402 − 1.196 = **+1.206**.

### R3 — collapse harvest (W8), sensitivity
- Count **new** collapse events (`hand_now >= 1.0`) outside the 3 known design collapses.
- For each new event report whether the detector fires (`amp_ratio_grid > 1.5`).
- Report **detection count / total new events**, and the pooled detection rate over all
  collapse events (design + held-out).
- **< 3 harvested collapse events (new) ⟹ sensitivity UNDERPOWERED; specificity only.**
- **Two counts, reported separately and NEVER merged** (the C1/R3 structural fix):
  **(a) large error reduction Δ ≥ 0.5** and **(b) below the line, learned < 1.0**.
  Both require the learned arm ⟹ both are reported as **PENDING-GPU** and are not
  estimated, imputed, or substituted by any proxy in this study.
- The cheap non-learned escape controls (multi-restart greedy, second-peak init, relaxed
  `min_dk`) demanded by P2/P1 are **out of scope here** and are recorded as NOT RUN.

### W6 — record-blocked re-analysis of the original 27-window primary
Zero compute; input = `tmp/wp_real_2026-07-25/wp_real.json`.

- Data: the 27 normal design windows, `delta = hand_now − learned_now`
  (negative ⟹ learned worse; the published primary was two-sided Wilcoxon on windows,
  W = 74, **p = 0.004613**, mean Δ = −0.0611).
- Record-blocked test: statistic = unweighted mean of the 5 **record-level mean deltas**.
  Null = record-level sign exchangeability. **Exhaustive enumeration of all 2⁵ = 32 sign
  vectors**; two-sided p = #{|stat_perm| ≥ |stat_obs|} / 32.
- Secondary at the same unit: Wilcoxon signed-rank on the 5 record means; paired t-test on
  the 5 record means; per-record direction table. All-30 windows repeated as a secondary.
- **Stated in advance (this is a ceiling, not a result):** with 5 records the smallest
  attainable two-sided permutation p is **2/32 = 0.0625**, and the smallest attainable
  two-sided signed-rank p at n = 5 is also 0.0625. **Therefore p = 0.005 cannot survive
  record-blocking under any data configuration.** The bar is consequently restated in the
  only form it can honestly take:
  - **PASS-in-direction**: all 5 record-level means point the same way (learned worse) AND
    the permutation p attains its floor 0.0625.
  - **FAIL-in-direction**: the record-level means are mixed in sign ⟹ the window-level
    p = 0.005 was driven by within-record pseudo-replication and must be reported at the
    record unit, with the record-level effect size and direction stated instead.
- Whatever the outcome, the correct reporting unit is the record and the RESULT.md must say
  so in those words.

## 4. Execution order (frozen)

1. This file is written and frozen. ✔
2. `harvest.py` runs the hand arm on the 48 held-out windows (CPU pool). No scores read.
3. `analyze.py` merges the 30 design rows and applies §3 mechanically.
4. `RESULT.md` reports the verdicts **whatever they are**. No bar is revisited.

## 5. Pre-committed failure branches

- **R1 FAIL (≥2 false fires):** the interlock's 0/27 false-alarm rate is an in-sample fit,
  not a specificity. The claim "0 false alarms" is deleted from all documents and replaced
  by the held-out rate with its record-clustered CI.
- **R1 PARTIAL:** the claim becomes "1 false fire in N held-out normal windows across 8
  unseen records", with the CI, and never "no false alarms".
- **R2 broken:** τ = 1.5 is reported as un-transferable and the detector is reported as
  requiring per-arena calibration.
- **R3 < 3 new collapse events:** sensitivity is declared UNDERPOWERED and only specificity
  is claimed — the "3/3 collapse windows repaired" line stays deleted (W1 b1).
- **W6 FAIL-in-direction:** Ch.7's keystone ("the pre-registered prediction of the
  program's own defeat, confirmed") is restated at the record unit and loses its p-value.
