# Timing Provenance Ledger — 2026-08-05

Read-only audit. No experiments re-run; no production code or `paper/` touched.
Scripts: `recompute.py`, `samerun.py` (this directory). Probe artifacts inherited from
`tmp/latency_bug_2026-08-05/`.

**MEASURED / ARGUED separation is marked inline. Every number below labelled MEASURED was
recomputed from raw JSON/JSONL in this session.**

---

## 0. Clock forensics — re-measured today, and one given fact is REFUTED

Two independent probes on the live box (75 s and 240 s, idle 2 ms sampling loop,
`time.monotonic()` as reference):

| quantity | 75 s probe | 240 s probe | status |
|---|---|---|---|
| backward step magnitude | 1243.7, 1250.6 ms | mean **1255.5 ms**, sd 7.7, n=8 | MEASURED |
| step period | 32.50 s (n=1 gap) | **32.496 s**, sd 0.023 s, n=7 gaps | MEASURED |
| rate error *between* steps | −0.0000 % | **−0.00 ppm ± 0.01 ppm** | MEASURED |
| apparent aggregate rate | −3.326 % | **−3.8635 %** | MEASURED |

### REFUTED given fact: "the guest clock runs ~3.9% fast"

It does not. Measured **between** consecutive steps, `CLOCK_REALTIME` tracks
`CLOCK_MONOTONIC` to **−0.00 ± 0.01 ppm** — i.e. the rate is correct to one part in 10^8.
The entire apparent −3.86 % is the periodic backward steps amortised over wall time:
`1255.5 ms / 32.496 s = 3.863 %`. That is where "≈3.9 %" came from. It is the *same defect
counted twice*, and its sign in the summary is inverted — durations are not systematically
**over**stated by 3.9 %; mean-aggregated durations are **under**stated by ≈3.86 %.

### The constant-relative-bias result (MEASURED mechanism + ARGUED algebra)

A step corrupts a frame only if it lands inside the timed region. Over one episode the
expected number of corrupted frames is `(N · mean_ms / 1000) / PERIOD_S`, and each costs
`STEP_MS / N` of the episode mean. So

```
E[bias_ms] = (N·mean_ms/1000 / 32.496) · (1255.5 / N) = mean_ms · (1255.5 / 32496)
E[relative bias] = −STEP_MS / (PERIOD_S · 1000) = −3.86 %
```

**N cancels. mean_ms cancels. Method cancels.** Every mean-aggregated per-frame timing
produced on this box carries the same −3.86 % expected bias, regardless of algorithm,
frame count, or duty cycle. Verified against four independent files:

| file | measured relative bias |
|---|---|
| `tmp/rerun_2026-07-24/ep100_runs.jsonl` | −2.83 % |
| `tmp/rerun_2026-07-25/batchB_ws.jsonl` | −4.25 % |
| `tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl` | −3.94 % |
| `tmp/rerun_2026-07-24/ws_runs.jsonl` (P1) | −5.13 % |
| **predicted** | **−3.86 %** |

**The median is immune.** On `ep100_runs.jsonl`, raw mean 11.008 → clean mean 11.329
(−2.83 %), but raw median 10.062 → clean median 10.081 (**−0.19 %**). The project's own
median-of-k convention would have prevented this entire class of error.

---

## 1. Spot-checks of the given facts

| given fact | verdict | evidence |
|---|---|---|
| `sequential_update.py:840` / `:875` use `time.time()` | **CONFIRMED** | `t0 = time.time()` L840; `elapsed_ms = (time.time() - t0) * 1000` L875 |
| `run.py:274` aggregates with mean | **CONFIRMED** | `avg_ms = np.mean([r['elapsed_ms'] for r in frame_results])` |
| `eval_compute_cost.py:90` hardcodes 13.4 | **CONFIRMED** | `'inference_ms': 13.4,  # from existing results (13.4ms/frame)`; also L91 in `total_30frame_s`. Propagates verbatim into all 5 per-episode entries and the average of `results/compute_cost_comparison.json` |
| `ep100_runs.jsonl` raw mean 11.008, clean 11.329, bias −2.8 % | **CONFIRMED exactly** | 11.008 / 11.329 / −2.83 %, 30 of 500 records dropped at FRAC=0.5 |
| ~49 negative timing values repo-wide | **CONFIRMED (51 by my key set)** | see §5 |
| WS-arm contamination 2.8–6.5 % | **CONFIRMED** | 2.8 % (batchC), 3.8 % (batchB), 6.0 % (ep100) |
| clock runs ~3.9 % fast | **REFUTED** | see §0 |

**Additional given fact not in the brief, discovered here and load-bearing:** all four
baselines (`etkf_grid.py:125/197`, `letkf_grid.py:144/261`, `gpr_baseline.py:72/104`,
`gpr_dispersion.py:139/187`) also time with `time.time()`. The defect is repo-wide, not
WaveSplat-specific — which is *why* it cancels in ratios.

---

## 2. The ledger

Exclusion rule used throughout, stated once: group records by episode (5 estimator seeds);
take the group median as clean reference; drop any record sitting `≥ 0.5 · (1255.5 ms / N_frames)`
below it, with `N_frames` = 220/190/160/120 for BF2/3/4/5. FRAC=0.75 changes nothing material
(ep100: 30→24 dropped, clean mean 11.329→11.369).

### 2a. Paper claims

| # | claim | location | traced source | class | corrected |
|---|---|---|---|---|---|
| 1 | "11.0 vs. 132.6 ms/frame … single CPU thread, Intel i7-12700H", ~10× | `wavesplat.tex:52-53` | 11.0 ← `tmp/rerun_2026-07-24/ep100_runs.jsonl` raw mean 11.008; 132.6 ← `results/b1_full_comparison.json` ETKF mean 132.639 | **CROSS-RUN + MEASURED-CONTAMINATED + wrong CPU** | see §3 |
| 2 | "$8.1\times$ faster" (vs LETKF) | `:67` | `results/100ep_ws_letkf.json`, LETKF 118.300 / WS 14.546 = **8.133** — *both arms same file, same run* | **MEASURED-CONTAMINATED (but same-run)** | ratio survives: 8.1× |
| 3 | "~10× faster (11.0 vs 132.6)" | `:104` | as #1 | **CROSS-RUN** | — |
| 4 | "13.4 ms/frame on CPU (single-threaded, Intel i7-12700H)" | `:753-754` | `results/b1_full_comparison.json` WaveSplat mean **13.376** | **MEASURED-CONTAMINATED** (orig. measurement); **HARDCODED** where re-emitted by `eval_compute_cost.py:90` | 13.376 → **13.91** de-biased |
| 5 | ETKF "132.6 ms/frame" at $N_e{=}100$, $G{=}40\times30$ | `:761` | `results/b1_full_comparison.json` ETKF mean 132.639, median 110.089 | **MEASURED-CONTAMINATED** | 132.64 → **137.9** de-biased |
| 6 | GPR "exceeds 200 ms" | `:767` | `results/b1_full_comparison.json` GPR-1 mean 284.5 (min 182.3); `results/12ep_full_analysis.json` 208.8 | **MEASURED-CONTAMINATED** | holds for that run; **contradicted** by same-run 2026-07-25 (32.9 ms) |
| 7 | tab:complexity 13.4 / 132.6 / >200 ms | `:782-784` | as #4/#5/#6 | **CROSS-RUN within table** (13.4 & 132.6 same file — OK; ">200" from 12ep file) | — |
| 8 | tab:main WS ms/f column `7.9 12.6 20.2 9.2 20.5 18.3 15.8 18.7 21.4 18.9 19.5 11.7`, Avg 16.2 | `:1340-1355` | `tmp/rerun_2026-07-24/ws_runs.jsonl`, `cfg_name='P1'`, **median over 5 seeds** — **12/12 exact match** | **MEASURED-CLEAN (median-aggregated)** | no correction needed; median absorbs the defect (one seed logged **−0.8 ms**) |
| 9 | tab:main ETKF ms/f column `154 154 635 120 …`, Avg 166 | `:1340-1355` | `results/b1_full_comparison.json` ETKF `avg_inference_ms` — exact (154.32, 154.35, 634.96 …, mean 166.297) | **MEASURED-CONTAMINATED + CROSS-RUN** | 166.30 → **172.9** de-biased |
| 10 | "11.0 ms/frame on the full 100-episode dataset (16 ms on … 12-episode subset)" | `:1369-1371` | 11.0 ← ep100_runs raw mean; 16 ← tab:main avg 16.2 (different run, different aggregator) | **CROSS-RUN (WS vs WS!)** | 11.008 → **11.329** clean mean; **10.081** clean median |
| 11 | ETKF "median 110 ms" | `:1371-1372` | `results/b1_full_comparison.json` ETKF median **110.089** | **MEASURED-CONTAMINATED** | median ≈ unbiased → **110.1** |
| 12 | "well within the 100 ms real-time budget for 10 Hz LiDAR" | `:1373-1374` | derived from #10 + 10 Hz spec | **ARGUED** | survives every correction |
| 13 | "$n_{ens}$=50, 31.7 ms", "~4× higher per-frame latency" | `:1376-1378` | 31.7 ← `results/p1_comparison.json` ETKF mean **31.713** (12 episodes); paired against 132.6 (100 episodes, different file) | **CROSS-RUN + different episode set** | ratio 132.6/31.7 = 4.18 not comparable |
| 14 | fig:speed "~13 ms/frame … ETKF 90 to over 600 ms (median 110, mean 132.6)" | `:1385-1388` | `results/b1_full_comparison.json`: WS mean 13.376; ETKF min **81.15**, max **634.97**, median 110.089, mean 132.639 | **MEASURED-CONTAMINATED**; "90 to" understates the range (true min 81.2) | — |
| 15 | "computational footprint (13.4 ms/frame, single CPU thread)" | `:2113` | as #4 | **MEASURED-CONTAMINATED** | — |
| 16 | tab:strong_baseline ms column: WS 14.5, LETKF 118.3 (100 ep); WS 13.2, LETKF 115.7, ETKF 103.9, GPR-1 208.8, GPR-disp 473.6 (12 ep) | `:2192-2205` | **all seven traced exactly**: `results/100ep_ws_letkf.json` (14.546 / 118.300); `results/12ep_full_analysis.json` (13.224 / 115.693 / 103.920 / 208.768 / 473.612) | **MEASURED-CONTAMINATED**; caption already flags "carried from the original measurement" | de-bias all by ÷0.9614; **ratios unchanged** |
| 17 | "10× faster (11.0 vs 132.6 ms/frame)" | `:2684` | as #1 | **CROSS-RUN** | — |
| 18 | 10 Hz LiDAR rate, 1 Hz buoy, 1–2 s/rev radar, Δt = 0.1 s, 301 frames = 30 s | `:179, :315-318, :716, :807, :835` | sensor spec / simulation parameter / literature (`belmont2008`) | **SPEC — not a measurement** | n/a |
| 19 | "at 10 Hz, the window spans 0.3 s" ($K_w{=}3$) | `:740-741` | derived from #18 | **ARGUED (correct)** | n/a |
| 20 | $T_{init} = 2T_p$, "~16 s at BF4" | `:2609` | **data** latency, not compute | **SPEC** | see §6 gap |

### 2b. Class counts (paper)

| class | n | claims |
|---|---|---|
| CROSS-RUN | 6 | #1, #3, #7(partial), #9, #10, #13, #17 → 6 distinct numbers/pairings |
| MEASURED-CONTAMINATED | 9 | #2, #4, #5, #6, #9, #11, #14, #15, #16 |
| MEASURED-CLEAN | 1 | #8 (median-aggregated) |
| HARDCODED | 1 | #4 as re-emitted by `eval_compute_cost.py:90-91` |
| UNTRACEABLE | **0** | — every paper timing number resolved to raw data |
| SPEC / ARGUED (not measurements) | 4 | #12, #18, #19, #20 |

**No UNTRACEABLE paper timing claims.** That is the good news. The bad news is §3.

---

## 3. The two findings that matter more than the clock bug

### 3a. `11.0 vs 132.6` is not one comparison — it is three mismatches stacked

| | 11.0 ms (WaveSplat) | 132.6 ms (ETKF) |
|---|---|---|
| source | `tmp/rerun_2026-07-24/ep100_runs.jsonl` | `results/b1_full_comparison.json` |
| date | 2026-07-24 | 2026-04/06 (committed `e57528c`, 2026-04-09) |
| dataset | `data_finemesh_2026-07-23/` | `data/` |
| BLAS threads | **unset** (multi-threaded) | unset |
| concurrent workers | **5** (`WS_WORKERS=5`, `multiprocessing.Pool`) | serial |
| aggregator | mean over 500 runs | mean over 100 episodes |

The paper labels both "**single CPU thread, Intel i7-12700H**". Neither claim is verifiable
and one is measurably wrong:

- **The box is an AMD Ryzen 7 7800X3D (16 threads), not an Intel i7-12700H.** The
  2026-07-24 and 2026-07-25 numbers were produced here; the paper attributes them to a
  machine this repo does not run on.
- `ep100jobs.py:67` sets `n_workers = 5` and sets **no** `OMP_NUM_THREADS`. "Single CPU
  thread" is false for the 11.0 ms measurement in both senses.

Arithmetic consequence, MEASURED: swapping 13.4 → 11.0 at commit `77b8123` while leaving
132.6 untouched moved the true implied ratio from 132.6/13.4 = **9.90×** to
132.6/11.0 = **12.05×**, but the prose kept saying "~10×". The headline ratio is now
inconsistent with its own two operands.

### 3b. The same-run measurement says the speed advantage is 2.6×, not 10×

`tmp/rerun_2026-07-25/` ran WaveSplat (batchC) and all four baselines (batchA) in **one
chain, one box, one dataset, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, 4 workers**,
with the baselines configured *exactly* as the paper specifies
(`ETKFGrid(40×30, n_ensemble=100, mult_inflation=1.0, add_inflation=0.005)`,
`LETKFGrid(..., loc_radius=50.0)`). MEASURED:

| arm | raw mean (ms) | de-biased (ms) | median (ms) | ratio to WS |
|---|---|---|---|---|
| **WaveSplat** (100 ep, 5 seeds) | 6.73 | 7.00 clean | 7.06 | 1.00× |
| ETKF (tuned, 100 ep) | 18.46 | 19.20 | 18.62 | **2.64×** |
| LETKF (loc50, 100 ep) | 29.51 | 30.69 | 29.98 | **4.21×** |
| GPR-1 (100 ep) | 32.86 | 34.18 | 32.84 | **4.69×** |
| GPR-disp (12 ep) | 102.11 | 106.18 | 100.81 | **14.6×** |

vs. the paper's 10× (ETKF) and 8.1× (LETKF). This independently reproduces the 2.6×
figure from the 2026-08-04 blank-slate debate, from raw data, by a different route.

**Why the old numbers were 7–9× larger — MEASURED evidence, ARGUED attribution:**
`tmp/rerun_2026-07-24/chain_cpu.sh` carries an in-repo comment stating
*"ETKF/GPR became 100x slower under load-34 contention"*. The 2026-07-24/25 chains set
`OMP_NUM_THREADS=1`; `run_b1_full_comparison.py` sets nothing. ARGUED: the 132.6 / 208.8 /
473.6 figures are dominated by machine identity, BLAS threading and CPU contention rather
than by algorithm cost. This is consistent with the arm-selective gap (ETKF 7.2×,
GPR 8.6×, but WaveSplat only ~1.2×) — numpy/scipy baselines respond to BLAS threading,
the torch tracking loop much less. **MEASURED fact: the gap is arm-selective, so it cannot
be pure wall-clock contention affecting everything equally.**

---

## 4. Does the 3.86 % cancel in ratios? — the explicit argument

**Yes, and more cleanly than the "rate error" framing suggested.**

1. There is **no rate error** (§0): between steps the clock is exact to 0.01 ppm. So a
   *single* short duration measured on this box is unbiased. Nothing to cancel there.
2. What exists is a **discrete corruption**: with probability `p = mean_ms/32496` per frame,
   the recorded value is `mean_ms − 1255.5` instead of `mean_ms`.
3. For a mean over N frames, `E[recorded mean] = mean_ms · (1 − 1255.5/32496) = 0.9614 · mean_ms`.
   The multiplier `0.9614` contains no method-, N-, or duty-cycle-dependent term.
4. Therefore for two arms A and B measured **on the same box**:
   `E[recorded_A]/E[recorded_B] = (0.9614·μ_A)/(0.9614·μ_B) = μ_A/μ_B` — **exact cancellation
   in expectation**.

### What this does and does not affect

| affected? | claims |
|---|---|
| **NOT affected** — ratio claims where both arms were measured on the same box | #2 (8.1×), the 2.64× / 4.21× / 4.69× of §3b, tab:strong_baseline column ordering, tab:main WS-vs-ETKF ranking |
| **AFFECTED** — absolute latency values | #1, #4, #5, #6, #9, #11, #14, #15, #16 — every absolute ms figure is ~3.9 % low and should be divided by 0.9614 |
| **AFFECTED** — the 100 ms budget claim (#12), in the *safe* direction | true latency is ~4 % higher than reported; 11.3 ms vs 100 ms budget is unaffected in substance |
| **NOT fixed by cancellation** | #1, #3, #10, #13, #17 — cross-run/cross-machine pairings. Cancellation requires *same box*; these span an Intel laptop and a Ryzen desktop, and differ in BLAS threading and worker count. **This is the dominant error term, ~7× vs the clock bug's 4 %.** |

**Variance caveat (ARGUED):** cancellation is in *expectation*. Per episode the number of
caught steps is a small integer, so single-episode timings have high residual variance —
e.g. `ws_runs.jsonl` P1 contains a **−0.785 ms** episode average. Aggregate over ≥100
episodes, or use medians.

---

## 5. `results/*.json` and repo-wide census

MEASURED, 51 negative timing values across all repo JSON/JSONL (keys `elapsed_ms`,
`avg_inference_ms`, `inference_ms`, `ms`, `latency_ms`, `time_ms`, `elapsed_s`, `init_time_s`):

| n | range (ms) | file |
|---|---|---|
| 14 | −6.72 … −0.71 | `tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl` |
| 11 | −5.49 … −0.05 | `tmp/rerun_2026-07-25/batchB_ws.jsonl` |
| 5 | −3.04 … −0.66 | `tmp/relax_extctrl_2026-08-05/progress.jsonl` ⚠ **today** |
| 5 | −3.04 … −0.66 | `tmp/relax_extctrl_2026-08-05/relax_extctrl.json` ⚠ **today** |
| 5 | −5.12 … −1.84 | `tmp/rerun_2026-07-24/ep100_runs.jsonl` |
| 2 | −0.79 … −0.70 | `tmp/rerun_2026-07-24/ws_runs.jsonl` |
| 2 | −0.55 … −0.01 | `tmp/rerun_2026-08-03/p1_comparison_rerun.json` (+ progress.jsonl) |
| 2 | −0.08 … −0.00 | `tmp/heldout40_2026-07-26/v2_arm.jsonl` |
| 1 | −1.04 | `tmp/spread_xdesign_2026-08-05/xdesign.json` (+ progress.jsonl) ⚠ **today** |
| 1 | −0.07 | `tmp/etkf_diag_2026-07-24.json` |

**`results/*.json` contains zero negatives** — not because it is clean, but because every
entry there is already an episode-level *mean*, which buries the corrupted frame. Absence
of negatives in `results/` is not evidence of absence of contamination; the expected
−3.86 % applies to all of them.

⚠ **The bug is live.** Two directories dated **2026-08-05** (`relax_extctrl`,
`spread_xdesign`) carry fresh negative timings. Anything measured today is affected.

---

## 6. Bottom line

### Strongest defensible latency statement today

> On a single-threaded CPU run (`OMP_NUM_THREADS=1`, AMD Ryzen 7 7800X3D), measured in one
> chain over 100 Nyquist-adequate episodes with all arms in identical configuration,
> WaveSplat's per-frame tracking cost has a median of **7.1 ms** (clean mean 7.0 ms,
> ≈7.3 ms after correcting the −3.9 % clock bias), against **18.6 ms** for the tuned
> ETKF-grid ($N_e{=}100$, $40\times30$) and **30.0 ms** for LETKF (loc50) — a **2.6×**
> advantage over ETKF and **4.2×** over LETKF. All four arms sit **well inside the 100 ms
> budget** for a 10 Hz LiDAR.

Everything in that sentence is same-box, same-run, same-config, and robust to the clock
defect (medians) or corrected for it.

**What must be retired until re-measured:**
- "~10× faster", "8.1× faster", "11.0 vs 132.6 ms" — cross-machine pairings.
- "Intel i7-12700H" — this repo runs on a Ryzen 7 7800X3D.
- "single CPU thread" for the 11.0 ms figure — it was 5 workers, unpinned BLAS.
- "GPR exceeds 200 ms" — same-run measurement gives 32.9 ms.
- Any *absolute* ms figure quoted to 3 significant figures.

**What survives unchanged:** the real-time-budget claim (#12), the *ordering* of the arms,
and tab:main's WaveSplat ms/f column (#8) — the only timing column in the paper that was
median-aggregated, and consequently the only one that needs no correction.

### Re-measurement plan (with cost)

Prerequisite, one-line fix in three places — `sequential_update.py` (5 call sites: L431/505,
L542/576, L722/739, L789/808, L840/875), the four baselines, and `eval_compute_cost.py`:
replace `time.time()` with `time.perf_counter()` (monotonic, unaffected by the step) and
change `run.py:274` from `np.mean` to reporting **both** median and mean. Delete the
`13.4` literal at `eval_compute_cost.py:90-91`.

| step | what | cost |
|---|---|---|
| 0 | `perf_counter` swap + median reporting + delete hardcode | ~30 min editing; **0 compute**. Bit-identical estimates (timing is not fed back into the optimizer) — verify with one episode |
| 1 | Re-run the 2026-07-25 chain design: WS 100 ep × 5 seeds + ETKF/LETKF/GPR-1 100 ep + GPR-disp 12 ep, `OMP_NUM_THREADS=1`, **1 worker** (no contention) | WS ≈ 500 × 32 s ≈ 4.5 h; baselines ≈ 312 × 60 s ≈ 5 h → **~10 h serial**, or ~3 h at 4 workers (but 4 workers reintroduces the contention confound — **run serial**) |
| 2 | A dedicated timing-only pass: 12 representative episodes × 3 repeats × 5 arms, serial, machine otherwise idle, report median + IQR | ~45 min. This is the number the paper should quote |
| 3 | Record and report the actual CPU (`/proc/cpuinfo`), thread env, worker count, and Python/numpy/torch versions alongside every timing table | 0 compute |

Recommended minimum before any latency claim is re-asserted: **steps 0 + 2 (~1.5 h)**.
Step 1 is only needed if per-episode timing tables are kept.

---

## 7. What I could not determine

1. **Whether 132.6 ms was ever measured on an Intel i7-12700H.** `results/b1_full_comparison.json`
   was committed 2026-04-09 with no environment record. The i7-12700H attribution is
   unverifiable from the repo; I can only show the current box is not one.
2. **How much of the 7.2× ETKF gap (132.6 → 18.5) is machine, how much is BLAS threading,
   and how much is contention.** Separating these requires re-running `run_b1_full_comparison.py`
   under controlled threading — a re-measurement, which was out of scope here. The
   arm-selectivity (ETKF 7.2×, GPR 8.6×, WS 1.2×) is MEASURED; the attribution is ARGUED.
3. **Whether the clock step existed on 2026-04/06.** If the defect is a recent WSL2/host
   condition, `results/b1_full_comparison.json` may be clean and my −3.86 % de-biasing of
   #4/#5/#9/#11/#14/#16 would be an over-correction. The de-biased values are therefore
   upper bounds on those arms. I found no dated clock record in the repo.
4. **Initialization latency is entirely absent from the paper.**
   `results/compute_cost_comparison.json` records WaveSplat `init_s` = **77.4 s average**
   (65.6–95.6 s) — ~5800× the per-frame tracking cost, and never reported. Whether that
   number still describes the shipped configuration I could not determine without re-running
   (project memory notes a later bank-MMSE port taking init to 0.7–1.3 s, but that is the
   real-domain gate, not the arm behind these tables). **This is the largest un-audited
   latency quantity in the project**, and it bears directly on the "real-time" framing.
5. **`tab:complexity`'s ">200 ms" for GPR** — I could not determine which of the two
   conflicting measurements (284.5 / 208.8 ms old, 32.9 ms same-run) the paper intends,
   because the two differ by 8.6× and no environment record distinguishes them.
