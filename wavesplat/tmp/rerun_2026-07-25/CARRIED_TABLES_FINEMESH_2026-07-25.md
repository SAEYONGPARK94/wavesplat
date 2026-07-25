# Carried-over tables — finemesh refresh (2026-07-25)

Clears the "carried over with a provenance note" debt from the 2026-07-24 paper re-run
(`tmp/SESSION_2026-07-24_REPORT.md` §3.11): the 5-method stronger-baseline table, the paired
Wilcoxon table, the spectral-accuracy table, and the RMSE_unobs column of the five-metric panel.

## Bottom line
| task | verdict | conclusion survives? |
|---|---|---|
| (1) tab:strong_baseline | **CONFIRMED** (rank order intact, margin widens) | yes |
| (2) Wilcoxon paired | **CONFIRMED** (both arms strengthen) | yes |
| (3) tab:spectral_accuracy | **CONFIRMED** (≤5-comp strengthens, crowded wall persists) | yes |
| (4) RMSE_unobs | **CONFIRMED** (WS lowest; structural ordering holds) | yes |

No rerun contradicted a stated *conclusion*. Two sub-claims flipped **in WaveSplat's favour** and
were rewritten transparently (not silently): the WS aggregate SS CI now excludes zero, and ≤5-comp
spectral detection is 100% (the old 50-60% was a coarse-mesh Nyquist artifact). Both flagged below.

## Discipline
- **WaveSplat arm = per-episode median over 5 estimator seeds** (mandatory: WS run-to-run variance
  is bimodal, CV≈53%; this materially matters — e.g. RMSE_unobs shifts up to ±0.085/episode between
  a single seed and the median). Baselines = their own established single-filter-seed protocol from
  the paper generators.
- All accuracy metrics on `data_finemesh_2026-07-23/`.
- **Per-frame timing (ms) carried over** from the original measurement — it is mesh-independent
  (identical operations; only LiDAR `z`-values differ) and my batch timing used a different
  threading/environment setup (OMP=1, `nice`, shared box) that is ~5× faster and not comparable.
  Substituting it would inject a measurement-protocol change disguised as a mesh refresh.
- **No code defaults changed.** The obs-mask pass for RMSE_unobs reuses the repo's existing
  `wavesplat/evaluation/metrics.py::rmse_unobserved` (min_dist=10 m), which `run_estimation` already
  calls per frame (run.py:181) — the 07-24 runner simply didn't record the field. Nothing additive
  to shared code was even required.

## Provenance of the OLD numbers (verified before touching anything)
| paper element | generator (old) | reproduced exactly from repo JSON |
|---|---|---|
| tab:strong_baseline 100-ep | `run_100ep_letkf_and_ws_metrics.py` → `results/100ep_ws_letkf.json` | WS 0.223 / LETKF 0.228 ✓ |
| tab:strong_baseline 12-ep | `run_12ep_full_analysis.py` → `results/12ep_full_analysis.json` | all 5 methods exact ✓ |
| Wilcoxon | `run_multiseed_extended.py` → `results/multiseed_extended.json` | W=194 p=1.1e-15 d=1.16; W=1508 p=4.7e-4 d=0.42 ✓ |
| tab:fivemetric RMSE_unobs | `run_b1_full_comparison.py` → `results/b1_full_comparison.json` | WS 0.262 / ETKF 0.348 / GPR-1 0.294 ✓ |
| tab:spectral_accuracy | `run_three_contributions.py` → `results/three_contributions.json` | all rows exact ✓ |

Stock `ETKFGrid` defaults (`obs_noise=0.05`, `max_obs=40`) are identical to the `etkf_100ep_panel`
FastETKF config, so one stock-class code path reproduces both the fivemetric ETKF (0.332) and the
strong-baseline ETKF; ETKF nRMSE cross-check confirms (0.3321 vs panel 0.332).

---

## TABLE 1 — Stronger baselines (tab:strong_baseline) — CONFIRMED

**Reused:** WS panel (100-ep + 12-ep subset) from `ep100_runs.jsonl` median-of-5.
**Reran (batchA, stock classes, seed 42):** LETKF (100-ep), ETKF/GPR-1 (100-ep → 12-ep subset),
GPR-disp (12-ep).

### Full 100-episode (old → new)
| method | nRMSE | VR | rho | SS | Dir R² | Hs err |
|---|---|---|---|---|---|---|
| WaveSplat | 0.223→**0.199** | 0.661→0.833 | 0.446→0.551 | +0.034→**+0.171** | −0.253→−0.025 | 0.277→0.186 |
| LETKF | 0.228→**0.228** | 0.111→0.127 | 0.359→0.357 | +0.142→+0.144 | +0.087→+0.094 | 0.692→0.667 |

LETKF nRMSE mesh-invariant to 4 dp (0.2276→0.228, matches §4.5 Δ=−0.0006). WS improves every accuracy
column; **retains best avg nRMSE (0.199<0.228), margin widens, and now also leads SS** (+0.171>+0.144).

### 12-episode subset — identical 5-way rank order
WS 0.211→**0.182** < LETKF 0.238→0.237 < GPR-disp 0.267→0.268 < GPR-1 0.273→0.274 < ETKF 0.358→0.377.
WS 0.182 and ETKF 0.377 match tab:main (ETKF slightly worse on finemesh = the inverse-crime artifact
of §3.4, not information loss).

### Regime / CI updates (MEASURED, `ss_regime_stats.py`)
- Head-to-head vs LETKF: WS wins **58/100** nRMSE and **58/100** SS (was 51/49, "draw" → majority).
- ≤5-comp: WS SS +0.37→**+0.58** [+0.50,+0.65]; LETKF +0.11 [+0.09,+0.13].
- ≥15-comp: WS SS −0.46→−0.44 [−0.52,−0.35]; LETKF +0.20 [+0.18,+0.22] (complementarity holds).
- Dir R² 100-ep −0.025 (was −0.25); 30-comp −1.66 (was −2.13); ≤5-comp +0.68 (was +0.41).
- LETKF variance deficit persists: VR 0.127 (was 0.111).

**★ FLAGGED strengthening #1:** WS **aggregate SS bootstrap CI now EXCLUDES zero, [+0.06,+0.28]**
(old coarse-mesh [−0.08,+0.15] included zero). The paper's honesty caveat "aggregate CI includes
zero" is contradicted *in WS's favour*. Rewritten transparently (prose states both old and new CI;
abstract updated) rather than silently.

---

## TABLE 2 — Paired Wilcoxon — CONFIRMED (both arms strengthen)

**Reused:** WS (ep100 median-of-5), ETKF (batchA seed 42). **Reran (batchA2):** GPR-3 sliding
(window=3, noise=0.05, seed 42 — genuinely new: `gpr_100ep` is window=1/noise=0.02). 100 paired episodes.

| | old | new (finemesh) |
|---|---|---|
| means | WS 0.219 / ETKF 0.321 / GPR3 0.253 | WS 0.199 / ETKF 0.332 / GPR3 0.255 |
| WS vs ETKF | W=194, p=1.1e-15, d=1.16, CI[0.085,0.119] | **W=141, p=2.5e-16, d=1.30, CI[0.114,0.154]** |
| WS vs GPR-3 | W=1508, p=4.7e-4, d=0.42, CI[0.018,0.050] | **W=1002, p=1.6e-7, d=0.65, CI[0.040,0.073]** |
| Holm-Bonferroni | p1=2.2e-15, p2=4.7e-4 | p1=4.9e-16, p2=1.6e-7 |
| 20-config-mean check | "p<0.001" | WS-ETKF p=3.8e-6; WS-GPR3 p=0.011 |

Both strengthen; GPR gap upgrades small-to-medium → medium (d 0.42→0.65). **Protocol deviation
stated in prose:** original paired a *single* estimator seed per (distinct) episode; the WS arm now
uses median-of-5 per the binding convention (ETKF/GPR3 single seed 42). The config-mean check for
GPR-3 is p=0.011 (not <0.001) — stated precisely rather than left under the blanket "p<0.001".

---

## TABLE 3 — Spectral accuracy — CONFIRMED (median-of-5, 8 spectral episodes, batchB)
| N_comp | det old→new | energy-det old→new | \|k\| err old→new | dir err old→new |
|---|---|---|---|---|
| 1 | 100→100% | 100→100% | 0.005→0.000 | 0.1→0.0 |
| 2 | **50→100%** | 96.9→100% | 0.001→0.001 | 0.2→0.0 |
| 5 | **60→100%** | 93.7→100% | 0.001→0.001 | 0.1→0.1 |
| 15 | 50→60% | 66.7→74.7% | 0.595→0.553 | 26.2→30.7 |
| 30 | 38→33% | 49.1→42.1% | 0.714→0.102 | 26.1→11.4 |

**★ FLAGGED strengthening #2:** ≤5-comp detection 50-60% → **100%**. The old low raw-detection was a
coarse-mesh artifact (shorter components below the reference mesh's spatial Nyquist were unmatchable);
the adequate mesh recovers them. The paper's "high energy-detection despite low raw-detection" reading
for ≤5-comp is retracted in prose (rewritten transparently). Crowded-regime wall persists
(≥15-comp detection 60-33%). The 30-comp |k|/dir drop (0.714→0.102, 26.1→11.4) is a **median-of-5
effect** (per-seed |k| ranges [0.08,1.1]; old single-seed 0.714 was an unfavourable draw), not only
mesh. Amplitude error 0.17-8.2× → **0.24-0.75** (the 8.2× extreme was a single-seed/coarse-mesh
outlier; median-of-5 is bounded). Conclusion (accurate when resolvable, degraded when crowded) holds.

---

## TABLE 4 — RMSE_unobs (fivemetric col 5) — CONFIRMED (WS lowest; ordering holds)
Obs-mask pass = `metrics.py::rmse_unobserved` (≥10 m from any observation, matched exactly). WS =
median-of-5 per episode then mean over 100 (matches the refreshed nRMSE aggregation); ETKF/GPR seed 42.
batchB: 500/500 rows, 0 errors, 100/100 episodes × 5 seeds; WS nRMSE xcheck = 0.199 (re-confirms headline).
- WaveSplat 0.262 → **0.250** (median-of-5)
- ETKF 0.348 → **0.362** (nRMSE xcheck 0.3321 ✓ = fivemetric 0.332)
- GPR-1 0.294 → **0.297** (nRMSE xcheck 0.2634 ✓ = fivemetric 0.263)

Ordering WS 0.250 < GPR-1 0.297 < ETKF 0.362 = same as old (0.262 < 0.294 < 0.348). Single-seed would
have biased WS ~+0.012 high (seed-42 aggregate 0.2525 vs median-of-5 0.2496; per-episode
median−seed42 ranged −0.085…+0.026) — **median-of-5 was necessary, not optional.**

---

## Paper edits applied (file: `paper/wavesplat.tex`)
- **tab:strong_baseline** table body (100-ep + 12-ep rows) + bold reassignment (WS now bold on SS & Hs); caption SS CI + ms-carry-over note.
- Strong-baseline prose: "Regime-dependent advantage" (58/58; ≤5c +0.58; ≥15c −0.44); "LETKF variance deficit" (VR 0.127; ETKF VR 1.048; ρ/SS); "Directional R² gap" (−0.025/−1.66/+0.68); "Statistical significance" (WS aggregate CI now excludes zero — documents old vs new; regime CIs).
- **Provenance note** (start of §strongbaselines): rewritten — all three tables now recomputed; only the localization sweep + BF6-7 probe remain single-seed (LETKF mesh-invariant, unaffected).
- **fivemetric** §: ≤5-comp SS +0.37→+0.58, CI, VR 0.53→0.78.
- **Wilcoxon** §subsec:stats: means, W/p/d/CI both arms, Holm-Bonferroni, config-mean check, VR 0.11→0.10, GPR-disp 21%→32%; footnote simplified.
- **tab:spectral_accuracy** table body + caption (median-of-5, regenerated) + "geometric vs energetic" prose + amplitude 0.17-8.2×→0.24-0.75.
- **tab:fivemetric** RMSE_unobs column (WS 0.262→0.250, ETKF 0.348→0.362, GPR 0.294→0.297) + caption ("carried from the original evaluation... not re-derived" → "re-measured on the regenerated data; WaveSplat retains the lowest value").
- **sec:dirdeficit** consistency fix: WS Dir R² −0.25→−0.03 vs ETKF +0.07→+0.06 (flows from the tab:strong_baseline Dir R² refresh).
- **Abstract**: SS-CI hedge rewritten (aggregate now excludes zero); Wilcoxon p 1.1e-15→2.5e-16, 4.7e-4→1.6e-7. **Headline 0.199 / 0.332 / 0.228 / 0.182 UNTOUCHED.**
- **Highlights** + **Conclusion**: Wilcoxon p/d updated.

## Reused vs rerun
- **Reused (no compute):** WS 100-ep + 12-ep panels (ep100_runs.jsonl median-of-5); WS-vs-ETKF Wilcoxon pairing.
- **Reran (batchA, ~13 min, 4 workers):** ETKF/LETKF/GPR-1 100-ep + GPR-disp 12-ep (full panel + RMSE_unobs).
- **Reran (batchA2, ~15 min):** GPR-3 sliding 100-ep (Wilcoxon GPR arm).
- **Reran (batchB, ~2 h):** WS 100-ep × 5 seeds (RMSE_unobs + spectral). Fields weren't cached, so a WS re-run was unavoidable; the pass also re-confirmed the 0.199 headline.

## MEASURED vs ARGUED
Every number above is MEASURED. The only ARGUED step is the ms carry-over rationale (timing is
mesh-independent) — an argument for *not* substituting a differently-measured number, not a claim.

## Compute / environment notes
- `nice -n 10`, ≤4 workers per pool, pools serialized (batchA → batchA2 → batchB) to never exceed 4 heavy processes while 3 other agents shared the box.
- Baselines were **not** 10× slower than the 07-24 logs (that was GPU-chain contention, now gone): ETKF ~7 s/ep, LETKF ~13 s/ep, GPR-1 ~13 s/ep, GPR-disp ~57 s/ep, WS ~55 s/run under load.
- **Concurrency observed:** another agent edited *other* sections of `wavesplat.tex` concurrently (raydrop ~L1516/1550, ~L1779). Non-overlapping with my 4-table regions; my edits verified intact. Not committed (per instruction).

## Artifacts (all under `tmp/rerun_2026-07-25/`)
`batchA_baselines.py`/`.jsonl`, `batchA2_gpr3.py`/`.jsonl`, `batchB_ws.py`/`.jsonl`, `assemble.py`,
`ss_regime_stats.py`, `time_probe.py`, `run_chain.sh`, logs.
