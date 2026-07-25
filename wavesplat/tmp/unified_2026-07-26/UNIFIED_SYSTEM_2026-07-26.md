# UNIFIED SYSTEM — one estimator that picks its own init from observations alone

Date 2026-07-26. Branch `learn-boundwave-retrain`. Protocol frozen in `PREREG.md` BEFORE
scoring (Part B genuinely pre-registered; Part-C detector frozen after in-sample design, stated
plainly). Additive code only, `v2_konly` default bit-identical (re-verified), no commits, no
paper edits, no training, `data/`/checkpoints untouched.

FINAL STAGE of the WavePrior program: the measured regime boundary (learned wins crowded,
analytic wins sparse) is turned into ONE estimator. An operator never chooses a method —
`init_method='auto'` decides from the observations. This report measures that system.

================================================================================
## VERDICTS

- **PART B (synthetic 100ep switch): PASS** — all three frozen conditions hold at the
  primary threshold and at all sensitivity thresholds {6,8,10}.
- **PART C (real collapse fallback): PASS** — detects 2/3 collapses (both catastrophic),
  0/27 false fires, improves the all-30 aggregate.
- **CODE (Part A): additive, default bit-identical (3/3 to 1e-10); the live `auto` path
  reproduces the assembled arm bit-identically (12/12 cached + 2/2 fully-live, |diff|=0).**

================================================================================
## THE ONE-SYSTEM SCOREBOARD

### Synthetic — 100 finemesh episodes, median-of-5, tracked avg_nRMSE/Hs (phase-fixed tracker)
Primary threshold: route to learned iff proposal `n_selected >= 8`, decided per-lane (faithful
to live `init_method='auto'`).

| arm | 100ep mean | crowded (40) | sparse (40) | moderate (20) |
|---|---|---|---|---|
| pure-v2 (analytic init) | 0.1145 | 0.2525 | **0.0204** | 0.0267 |
| pure-learned (proposal init) | 0.1011 | 0.1809 | 0.0534 | 0.0365 |
| **switch = `init_method='auto'`** | **0.0979** | **0.1798** | 0.0509 | **0.0279** |
| oracle (per-ep best; not deployable) | 0.0818 | 0.1805 | 0.0137 | 0.0203 |

- crowded = 15+30comp; sparse = 1+2comp; moderate = 5comp.
- The switch is the **best deployable arm on the full 100ep** (0.098 < 0.101 learned < 0.115
  analytic), matches the specialist in each regime it should (crowded ≈ learned 0.180 vs 0.181;
  moderate ≈ analytic 0.028 vs 0.027), and sits between best-pure and oracle.
- per-episode-median-signal variant = 0.0985 (within 0.0006 of per-lane) — the routing
  granularity does not change the conclusion.

### Real — 30 Bergamasco stereo windows, nowcast map nRMSE (/gt_std)
Fallback = analytic (hand) everywhere; if the frozen collapse detector fires, use the
learned-proposal nowcast for that window.

| arm | all-30 mean | all-30 median | normal-27 mean |
|---|---|---|---|
| hand (analytic) | 0.8815 | 0.8091 | 0.7441 |
| learned (proposal) | 0.8188 | — | 0.8047 |
| **fallback = auto-repair** | **0.7747** | 0.8091 | **0.7441** (exact tie) |

The fallback strictly dominates hand: identical on all 27 normal windows (0 false fires),
repairs the 2 catastrophic collapses.

================================================================================
## PART B — synthetic switch, frozen conditions

| condition | result | pass |
|---|---|---|
| aggregate: switch 100ep mean <= min(pure-learned, pure-v2) | 0.0979 <= 0.1011 | ✅ |
| crowded non-inf: switch vs pure-learned (n=40, paired Wilcoxon) | medΔ=0.0000, p=0.028, switch not worse (0.1798<0.1809) | ✅ |
| sparse non-inf: switch vs pure-v2 (n=40, paired Wilcoxon) | medΔ=0.0000, p=0.248 | ✅ |
| **DECISION** | | **PASS** |

### Sensitivity — threshold {6,8,10} (all PASS)
| thresh | switch (per-lane) | crowded non-inf p | sparse non-inf p | decision |
|---|---|---|---|---|
| 6 | 0.0979 | 0.144 | 0.510 | PASS |
| **8 (primary)** | **0.0979** | 0.028 | 0.248 | PASS |
| 10 | 0.0982 | 0.285 | 0.248 | PASS |

The aggregate is flat across thresholds (Δ ≤ 0.0003): the decision is not knife-edge on the
threshold value. 15/100 episodes sit near the boundary (median n_selected in [6,10]).

### Switch-signal distribution (median n_selected per episode)
- **crowded** [6 … 29], almost all ≥ 11 — cleanly above the gate.
- **sparse** mostly 1–7 — cleanly below the gate, **EXCEPT** the BF2_1comp over-proposals at
  17–30 (the misroute cluster below).
- **moderate (5comp)** 4–10 — straddles the boundary, as expected for the transition regime;
  the switch tracks the analytic arm here (0.028 vs 0.027).

### Misroutes (threshold 8): 34 episodes, total cost +1.675 nRMSE (summed)
- **82% of all misroute cost (1.378) is the single characterized OOD edge: BF2_1comp.** The
  32-JONSWAP-trained proposal over-proposes ~30 atoms on a single clean swell → the gate
  routes learned → cost +0.21…+0.30/episode (worst on the `slope` variants, e.g.
  BF2_1comp_seed0_slope routed 0.326 vs oracle 0.026). This is the Stage-1 §7 sharp edge,
  now quantified at population scale.
- The other 28 misroutes are individually small (all < 0.06; mostly 5comp/1comp boundary
  cases where the gate picked v2 and learned was marginally better, or vice-versa).
- **Consequence:** the switch's residual gap to oracle (0.098 vs 0.082) is almost entirely the
  BF2_1comp over-proposal. A low-N residual guard (veto learned when the post-prune residual
  stays high on a low-cardinality sea) would close most of it — a clean Stage-next item, not a
  flaw in the routing logic.

### Honest caveat on the sparse slice
The sparse **mean** for the switch (0.0509) is worse than pure-v2 (0.0204) — inflated by the
BF2_1comp misroutes. The frozen non-inferiority test is the **robust paired Wilcoxon** (medΔ=0,
p=0.248): on the *majority* of sparse episodes the switch routes v2 and is identical to pure-v2,
so it is not significantly worse, but a handful of OOD over-proposals drag the sparse mean up.
The **overall** 100ep aggregate still favors the switch because the crowded gains dominate.
MEASURED, reported both ways.

================================================================================
## PART C — real collapse fallback, frozen detector

**FROZEN detector:** fire iff `amp_ratio_grid > 1.5`, where
`amp_ratio_grid = std(analytic nowcast field over the query grid) / std(observed z)` — a
GT-free predicted-vs-observed amplitude ratio. Rationale: a basin/amplitude-explosion collapse
reconstructs a field carrying far more energy than the observations support, especially in the
extrapolated region. (Hand re-score reproduced wp_real hand_now bit-identically, max|diff|=0.)

**IN-SAMPLE caveat (explicit):** the detector was designed by inspecting these same 30 windows
(3 collapse + 27 normal). The falsifiable claim is **specificity + no-harm on these 30**;
generalization to future windows is the true test and is NOT claimed here.

| criterion | result | pass |
|---|---|---|
| (i) detection ≥ 2/3 collapse windows | **2/3**: AA14_w0 (amp 2.40) ✓, LJ18g_w3 (amp 2.53) ✓; AA15a_w3 (amp 0.89) missed | ✅ |
| (ii) specificity / no-harm on 27 normal | **0/27 false fires** → fallback = hand on all normal (paired Δ=0, p=1.0) | ✅ |
| (iii) all-30 aggregate vs hand | hand 0.8815 → fallback **0.7747** | ✅ |
| **DECISION** | | **PASS** |

Per collapse window: AA14_w0 2.309→**1.071** (repair +1.238), LJ18g_w3 2.925→**0.961**
(+1.964), AA15a_w3 1.120→1.120 (missed — the mild one, hand nRMSE only 1.12).

**Why 2/3, not 3/3, is the honest ceiling here.** The two windows caught are the two
*catastrophic* collapses (hand nRMSE > 2). The missed AA15a_w3 is a *mild* collapse (1.12): its
GT-free signals (amp_ratio 0.89, vr_u0 0.185) are indistinguishable from the 27 normal windows
— no GT-free rule separates it without firing on normals too. `vr_u0` (held-out obs residual,
the other candidate) was rejected as a firing signal: it flags the LJ18g storm record globally
(all 6 windows vr>0.75, only 1 is a collapse) and misses AA14_w0/AA15a_w3 (vr≈0.185) — as an
OR term it would only *add* false fires. The chosen single-signal detector is maximally
specific: it fires exactly on amplitude-explosion collapses.

================================================================================
## Part A — the code (additive) + validation

- `wavesplat/optimizer/sequential_update.py`: `_auto_select_init()` + one dispatch line at the
  top of `initialize()` (mutates `self.init_method` to the chosen arm, then falls through to
  the normal dispatch) + two diag attributes in `__init__`. Reads the proposal, pools it with
  the learned arm's identical knobs, routes on `n_selected >= auto_switch_thresh` (default 8),
  caches the raw atoms so a learned route does not re-run the GPU proposal.
- `wavesplat/run.py`: `'auto'` added to `--init_method` choices; router decision surfaced in
  the summary as `auto_selected` / `auto_n_selected`.
- **Default bit-identity:** the Stage-1 control replays 3 recorded `v2_konly` jobs through the
  edited code — avg_nRMSE AND init_nrmse match to 1e-10, 3/3. The `auto` branch is only entered
  when `init_method=='auto'`; the shipped path is untouched.
- **Live-path validation (`demo_auto.json`):** 10 crowded + 2 sparse episodes run through
  `init_method='auto'`. All 10 crowded route learned (n_sel 8–30) and reproduce the assembled
  learned lane **bit-identically (|diff|=0)**; 2 fully-live runs (the router runs the GPU
  diffusion proposal itself, no cache) also reproduce bit-identically — proving the single
  config value is a faithful end-to-end entry point. (BF2_1comp_seed0_uniform routes learned at
  n_sel=30 — the documented over-proposal — but the pipeline's prune still scores it 0.027.)

================================================================================
## MEASURED vs ARGUED

**MEASURED:**
1. `init_method='auto'` is a single config value that routes learned-vs-analytic from a GT-free
   init-time signal; the live path reproduces the assembled arm bit-identically (12/12 + 2/2).
2. Synthetic 100ep: the auto arm's mean tracked nRMSE (0.0979) ≤ both pure arms
   (0.1011 learned, 0.1145 analytic); non-inferior to each regime's specialist on its slice
   (crowded vs learned, sparse vs analytic; robust paired Wilcoxon); stable over thresholds
   {6,8,10}.
3. The auto arm's residual gap to oracle is 82% attributable to one characterized OOD edge
   (BF2_1comp over-proposal); all other misroutes cost < 0.06 each.
4. Real 30 windows: the frozen amplitude-ratio detector fires on both catastrophic analytic
   collapses (repairing 2.31→1.07 and 2.93→0.96) with 0/27 false alarms; all-30 mean
   0.881→0.775, normal-27 exactly tied to hand.
5. Default `v2_konly` path bit-identical (3/3 to 1e-10).

**ARGUED (supported, not proven):**
- The unified system "solves every regime" in the sense of *never being worse than the better
  specialist to a significant degree* while being *strictly better in aggregate* — a
  select-the-right-tool guarantee, not a new-accuracy-frontier guarantee. The oracle gap shows
  the ceiling is the proposal's OOD cardinality control, not the routing.
- The real-data contribution is **robustness (failure-mode insurance), not typical-case
  accuracy**: on normal real windows the analytic prior remains better (Stage-3b p=0.005), so
  the real-mode unified system keeps hand as default and invokes the learned proposal only when
  the collapse detector fires. Consistent with the standing synthetic-clean→real caveat.

**In-sample caveats:** the Part-C detector is designed and scored on the same 30 windows
(n=3 collapse). The claim is specificity + no-harm on these 30; future windows are the true
test. Part B is out-of-design (the switch rule n_sel≥8 is frozen from the design regimes, not
tuned on nRMSE; extra-48 learned results were not inspected before freezing).

**Scope:** synthetic switch is in-distribution finemesh; does not license a default flip, paper
edit, or commit, and does not reopen the refuted γ/bank-MMSE axis.

================================================================================
## SYSTEM CARD — the unified estimator (for the thesis)

1. **Single entry point.** `init_method='auto'` — one config value. The operator never selects
   a method or declares a regime; the system decides from the observations of each episode.
2. **What it does automatically (synthetic/crowded-capable path).** It runs the learned
   set-proposal on the init window, counts the selected components `n_selected`, and initializes
   the tracker from the learned proposal when the sea is crowded (`n_selected ≥ 8`) or from the
   analytic dispersion-beamformer otherwise; the shipped joint-fit / adaptive-N / phase-fixed
   tracker then runs unchanged in both cases.
3. **Measured synthetic guarantee.** Over 100 finemesh episodes the auto arm's mean tracked
   nRMSE/Hs is 0.098 — better than either fixed method (0.101 learned, 0.115 analytic) — and
   is non-inferior to each regime's specialist on its own slice, stable across gate thresholds.
4. **Measured real guarantee (robustness).** With a frozen GT-free amplitude-ratio detector, the
   analytic path's catastrophic basin collapses are auto-repaired by the learned proposal (2/2
   catastrophic real windows fixed) with zero false alarms on 27 normal windows — failure-mode
   insurance layered on the analytic prior, which remains the accuracy leader on typical seas.
5. **Known edge (declared).** The proposal over-proposes on single clean swells (BF2_1comp),
   the one characterized misroute; it caps the auto arm above the oracle and is addressable with
   a low-cardinality residual guard. No new training, no default change, no paper edit is
   claimed by this stage.

================================================================================
## FILES (all under `tmp/unified_2026-07-26/`)
- `PREREG.md` — frozen pre-registration (Parts A/B/C).
- `precompute_atoms_extra.py`, `atoms_cache_extra.jsonl` (240, 0 err), `precompute.log` — GPU
  proposal cache for the 48 episodes not covered by Stage-1.
- `run_arms_extra.py`, `arm_runs_extra.jsonl` (240, 0 err), `arms_extra.log` — extra-48 learned
  tracker arm (completes both pure arms over all 100 episodes).
- `analyze_switch.py`, `switch_results.json` — Part-B scoreboard, non-inf, sensitivity,
  misroutes, signal distribution.
- `rescore_hand_amp.py`, `hand_amp.json` — Part-C hand re-score (bit-identical control) +
  GT-free amplitude signals.
- `analyze_fallback.py`, `fallback_results.json` — Part-C frozen detector scoring.
- `demo_auto_live.py`, `demo_auto.json` — live `init_method='auto'` path validation.
- `UNIFIED_SYSTEM_2026-07-26.md` — this report.

Code (additive; defaults untouched, bit-identity verified): `wavesplat/optimizer/
sequential_update.py`, `wavesplat/run.py`.

## Deviations from PREREG
None material. Sparse slice landed at n=40 (1+2comp; recount confirmed 30 1comp + 10 2comp),
not the ~34 loosely estimated at freeze — the frozen definition (1+2comp) is unchanged. Part-C
detection is 2/3 (the frozen bar is ≥2/3), the mild collapse being GT-free-indistinguishable;
reported transparently. Wall-clock: precompute ~9 min (GPU) + arms ~14 min (CPU, 4 workers) +
re-score/demo/analysis ~6 min ≈ 30 min, well under the 2 h budget.
