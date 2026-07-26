# RESULT — SPECTRAL DISCRETENESS as the boundary coordinate (worklist W9)

**Run 2026-07-26.** Design frozen in `tmp/discreteness_2026-07-26/PREREG.md` before any episode
was generated. Raw: `ws_runs.jsonl` (150 runs, 0 err), `cond_runs.jsonl` (150 runs, 0 err),
`secondary.jsonl` (30 rows, 0 err). Analysis: `analyze.py` → `results.json`;
`noskill_floor.py` → `noskill_floor.json`. Nothing committed; no library default changed.

---

## 1. PRE-REGISTERED VERDICT — **CONFIRM**

Applying the frozen cells verbatim, in the frozen order:

| gate | frozen condition | measured | met |
|---|---|---|---|
| 0 INVALID | `R_A > −20%` | `R_A = −29.0%` | no → not invalid |
| 1 CONFIRM | `R_A ≤ −20%` AND (`R_C > −10%` OR `p_C ≥ 0.05`) | `R_A = −29.0%`; `R_C = −5.7%`; `p_C = 0.080` | **YES — both disjuncts hold** |
| 2 REFUTE | `R_C ≤ −20%` AND `p_C < 0.05` | −5.7%, p = 0.080 | no |

**Verdict: CONFIRM.** The learned set-proposal's advantage over the analytic init collapses as the
synthetic spectrum is made continuum-like at fixed crowding, fixed Hs and fixed total directional
spread. The ladder is monotone:

> **R_A = −29.0% → R_B = −9.6% → R_C = −5.7%**

**⚠ The verdict must be read with §3. The collapse is real, but it is not "learned loses its edge
while physics keeps working" — in the continuum cell BOTH arms fall to the no-skill floor.**

---

## 2. The three cells (median-of-5 per episode, both arms; n = 10 episodes per cell)

nRMSE = RMSE/Hs, `ws_best = min(median WS init, median WS tracked)`, Δ = cond − ws_best.

| cell | spec | cond mean | cond median | ws_best mean | ws_best median | med Δ (abs) | **R (median rel)** | cond wins | Wilcoxon p (1-sided) |
|---|---|---|---|---|---|---|---|---|---|
| **A** | discrete control, 15 atoms | 0.0851 | 0.0804 | 0.1141 | 0.0977 | −0.0298 | **−29.0 %** | 7/10 | 0.0186 |
| **B** | mild spread, 465 atoms, σ_sub 10° | 0.1806 | 0.1877 | 0.1989 | 0.2101 | −0.0221 | **−9.6 %** | 10/10 | 0.00098 |
| **C** | strong spread, 465 atoms, σ_sub 28° + ω jitter | 0.2330 | 0.2375 | 0.2426 | 0.2449 | −0.0133 | **−5.7 %** | 6/10 | 0.0801 |

Per-episode spread (mean of the per-episode sd over the 5 seeds): cond 0.008 / 0.006 / 0.005 for
A/B/C; WS init 0.041 / 0.015 / 0.012. The analytic arm is the noisy one (the known bimodal
random-phase-init variance), which is why median-of-5 was mandatory.

Both arms degrade monotonically as the sea becomes continuum-like — the *learned* arm degrades
faster (0.085 → 0.181 → 0.233, ×2.7) than the analytic arm (0.114 → 0.199 → 0.243, ×2.1). That
asymmetry is the whole effect.

### Control-cell caveat (reported, does not change the verdict)
Cell A replicates the known advantage in direction and clears the frozen −20 % bar, but at
**less than half the reference magnitude**: R_A = −29.0 % here vs **−64.6 %** on the
`wp_crowded_2026-07-25` BF4/15comp block (seeds 0–4). The cond arm is comparable
(0.085 here vs 0.124 there); the difference is almost entirely that the **analytic arm did much
better on the fresh seeds** (ws_best 0.114 vs 0.216). Seeds 10–14 happen to include several
narrow-realized-spread draws (realized directional spread 28.7 ± 11.7°, i.e. some seeds near 17°),
which the analytic beamformer resolves well. Consequence: the experiment's dynamic range is
compressed, so the *magnitude* of the collapse (−29 % → −5.7 %) is less impressive than it would
have been against a −65 % baseline. The *monotone ordering* is unaffected.

---

## 3. ⚠ WHAT THE COLLAPSE ACTUALLY IS — both arms hit the no-skill floor

The paired ratio R cannot tell "the learned arm lost its specific edge" from "the problem became
unsolvable for everything". Measured directly (`noskill_floor.py`; zero-predictor nRMSE evaluated
at the same 3 nowcast frames both arms are scored on, skill = 1 − nRMSE/floor):

| cell | zero-predictor floor | cond | ws_best | **skill (cond)** | **skill (ws)** | skill gap | episodes at-or-worse than zero predictor (of 10) |
|---|---|---|---|---|---|---|---|
| A | 0.2501 | 0.0851 | 0.1141 | **0.660** | **0.543** | 0.117 | cond 0, ws 0 |
| B | 0.2369 | 0.1806 | 0.1989 | **0.233** | **0.155** | 0.077 | cond 0, ws 2 |
| C | 0.2466 | 0.2330 | 0.2426 | **0.055** | **0.015** | 0.040 | cond 2, **ws 5** |

In the continuum cell **neither method has meaningful skill**: the analytic arm is at 1.5 % skill
with half its episodes no better than predicting zero, and the learned arm at 5.5 %. So:

- **Honest reading:** continuum-ization does not merely delete the learned prior's advantage; it
  moves the whole problem past the information wall for *both* estimators at this aperture. The
  learned advantage shrinks partly because there is almost nothing left to be better at.
- The skill *gap* also shrinks monotonically (0.117 → 0.077 → 0.040), so the learned arm's
  relative edge is genuinely eroding, not only being compressed by the floor. Both statements are
  true; neither alone is the result.
- This experiment therefore **cannot separate** "discreteness is what the learned prior exploits"
  from "continuum seas are unsolvable here with either method". Both are consistent with CONFIRM.
  Separating them needs a cell that is continuum-like *and* still solvable (e.g. larger aperture /
  longer window, or continuum-ization applied only in direction at fixed narrow band).
- Note the consistency with the real-data record: real reconstructions also sit at/near no-skill
  (P2, CRUX E). A continuum sea being a near-no-skill regime for both arms is exactly what the
  real-window behaviour looks like — which is the hypothesis's prediction, arrived at from the
  other side.

---

## 4. Secondary — n_selected: **the predicted mechanism is measured**

Proposal cardinality after the shipped pooling (`pool_and_select`, energy_thresh 0.03,
dedup_dk 0.03, max_n 30), median-of-5 diffusion seeds per episode:

| cell | median | mean | range | fraction at the max_n = 30 cap | fraction ≥ router threshold 8 |
|---|---|---|---|---|---|
| A discrete | **13.0** | 12.8 | [10, 16] | 0.00 | 0.96 |
| B mild | **17.5** | 18.3 | [10, 25] | 0.00 | 0.92 |
| C strong | **22.5** | 22.3 | [14, 30] | 0.20 | 0.94 |

Mann–Whitney A vs C: **p = 4.75 × 10⁻⁴**. Monotone rise, and cell C's median (22.5) lands on the
**measured real-window n_eff ≈ 24**. This is the specific mechanism P2 predicted — a net trained
on discrete atom sets, shown a continuum, tiles it with ~22–30 atoms — reproduced in a controlled
synthetic cell. It is the strongest single piece of evidence in this run, and it is independent of
the primary endpoint (GT-free, computed from the proposal alone).

## 5. Secondary — bf_peak_ratio: **the migration prediction FAILS**

| source | n | median | IQR | range | inside the real band [0.27, 1.00] |
|---|---|---|---|---|---|
| **REAL windows** (`figs_opus_real_2026-07-26/bf_ratio_real.json`) | 30 | **0.879** | [0.505, 0.952] | [0.108, 0.998] | 87 % |
| known-advantage synthetic (`data_finemesh` BF4/15comp, seeds 0–4) | 10 | 0.186 | [0.132, 0.315] | [0.096, 0.649] | — |
| cell A discrete | 10 | 0.128 | [0.122, 0.167] | [0.100, 0.214] | 0 % |
| cell B mild | 10 | 0.269 | [0.096, 0.325] | [0.069, 0.346] | 50 % |
| cell C strong | 10 | **0.140** | [0.115, 0.161] | [0.102, 0.440] | 10 % |

**Not monotone, and cell C does not migrate toward the real range.** Making the sea continuum-like
leaves the beamform second/first-peak ratio essentially where the discrete sea had it (0.128 →
0.140), far below the real median 0.879. Consequences, stated plainly:

- `bf_peak_ratio` **does not measure discreteness**. It measures how many *comparably powerful,
  MIN_DK-separated* beamform peaks the init window shows — and smearing each atom into an arc
  keeps one dominant lobe, so the ratio barely moves.
- Therefore the shipped UNIFIED routing guard would **not** veto a continuum sea. Whatever else the
  guard does, it cannot be the mechanism that keeps real (continuum) seas on the analytic path.
- The gap between real (0.879) and *all three* synthetic cells (0.13–0.27) is unexplained by
  discreteness, so **at least one further coordinate separates real windows from synthetic ones**
  on the guard axis. P2's "replot the regime map on two axes" is not yet satisfiable with these
  two proxies.

---

## 6. Generation provenance and validity flags

All 30 episodes generated with the untouched `wavesim.pipeline.run_episode` at the
`data_finemesh_2026-07-23` mesh (200 × 167, dx 1.2060 m, dy 1.2048 m), GPU raycast, 301 frames.

| cell | ok | mesh verdict | atoms | max var_under | realized window Hs | realized dir. spread |
|---|---|---|---|---|---|---|
| A | 10/10 | ADEQUATE | 15 | 0.0000 | 1.201 ± 0.004 | 28.7 ± 11.7° |
| B | 10/10 | ADEQUATE | 465 | 0.0000 | 1.145 ± 0.134 | 29.0 ± 9.9° |
| C | 10/10 | ADEQUATE | 465 | 0.0000 | 1.181 ± 0.039 | 29.8 ± 1.7° |

- **G-mesh: PASS.** 30/30 ADEQUATE, `var_under_resolved = 0.0000` everywhere — the same standard
  `data_finemesh_2026-07-23` meets. The mesh-Nyquist artifact history is not in play here.
- **G-identity: PASS.** Cell A's components are **bit-identical** to
  `jonswap_components(Hs=1.2, Tp=7.0, n_components=15, depth=50, seed=s)` for all 5 seeds — the
  standalone generator is a verified passthrough when M = 1.
- **G-roundtrip: FLAGGED (one clause).** Schema/loader round-trip is clean — `EpisodeDataLoader`
  reads 465-component episodes with identical `n_frames` (301), `Hs_true` (1.2), `Tp_true` (7.0),
  GT shape (301, 60, 80) and init-window length (141) as the reference finemesh episode; nothing
  caps the component count. The flagged clause is the ±10 % realized-Hs band: cell B spans
  [1.022, 1.394]. **This is not a generation defect** — component-level Hs is exact
  (1.2000 ± 0.00000) in every cell. It is finite-window realization scatter, which cell A does not
  have because a 15-atom discrete sea has *deterministic* component amplitudes, whereas the spread
  cells are properly Gaussian (cell B worst because its M = 31 near-co-directional, same-ω
  sub-atoms sum to Rayleigh-amplitude quasi-components with beat lengths longer than the 240 m
  window). It is a *signature* of the manipulation, not a bug.
  **Handling:** the decision statistic R is a same-episode ratio of the two arms and is therefore
  **exactly invariant** to any per-episode common rescaling. Confirmed by re-running with both arms
  renormalized by realized window Hs: R unchanged to the digit (−29.0 / −9.6 / −5.7 %) and the
  Wilcoxon p unchanged (0.0186 / 0.00098 / 0.0801). Absolute nRMSE levels shift by ≤ 6 %.
- **G-spread: PASS.** Realized energy-weighted spread ≈ 30° in all three cells (28.7 / 29.0 /
  29.8), so "broadness" is controlled and the factor is discreteness, as designed. Pre-registered
  in the addendum: cell A's realized spread is intrinsically noisier (± 11.7°) because 2 of its 15
  parents carry ~62 % of the variance.
- **Arm integrity:** 150/150 WS runs and 150/150 cond runs completed, 0 errors; every episode has
  the full 5 + 5 seeds. No cut was taken; the cond arm kept all 5 diffusion seeds.
- One incident, no data impact: the generator's *manifest bookkeeping* raised a duplicate-kwarg
  `TypeError` after each HDF5 had already been written and read back cleanly. Files are unaffected;
  the manifest was rebuilt from disk with the same gate checks (`gen_cells.py --reindex`).

---

## 7. What this licenses (per PREREG §6, unchanged)

**Licensed:** on synthetic BF4 seas at fixed crowding, fixed Hs and fixed total directional
spread, the learned set-proposal's advantage over the analytic init is a monotone function of
spectral **discreteness** and is gone (−5.7 %, n.s.) on continuum-like spectra; and the proposal's
cardinality rises monotonically with continuity to the measured real n_eff ≈ 24. This makes the
real-data accuracy loss **consistent with** the regime map rather than an anomaly.

**NOT licensed:**
- Not a demonstration that continuity *causes* the real-data loss — real data also differ in
  sensor, noise, aperture, current and nonlinearity. This is a synthetic, init-level, BF4,
  15-parent-component statement.
- Not "the analytic method survives continuum seas". It does not: 1.5 % skill, 5/10 episodes no
  better than the zero predictor.
- The `docs/HANDOFF_OPUS_2026-07-26.md` §5 do-not-cite rules stand: no learned accuracy advantage
  on real data (collapse-insurance only), no "more complex ⇒ WavePrior wins" outside synthetic,
  no mean-based real-data victory.
- `bf_peak_ratio` may **not** be cited as a discreteness proxy — measured here as non-monotone and
  non-migrating (§5).

## 8. Follow-ups this run earns

1. **A solvable continuum cell.** Cell C is past the information wall for both arms, which
   confounds the primary reading. Repeat cell C at a larger effective aperture (longer init window
   or higher ship speed) so continuum seas are still solvable; only then can "discreteness is what
   learning exploits" be separated from "continuum is unsolvable".
2. **The guard-axis gap is open.** Real windows sit at bf_peak_ratio 0.879 and every synthetic
   cell at 0.13–0.27, with discreteness ruled out as the explanation. Find the coordinate that
   does separate them before the regime map is plotted on two axes.
3. **n_selected as the routing signal.** It moved monotonically and lands on the real n_eff; it is
   a better continuity proxy than bf_peak_ratio. Worth testing as a guard input on real windows.
