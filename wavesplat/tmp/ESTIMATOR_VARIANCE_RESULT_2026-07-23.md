# WaveSplat run-to-run variance on a fixed episode — 2026-07-23

Script `tmp/estimator_variance.py`, log `tmp/estimator_variance.log`, raw
`tmp/estimator_variance_results.json`. Pre-registered read stated in the script header
**before** the numbers (including the specific instruction to watch for **bimodality**
rather than a smooth spread).

**Design.** One fixed episode file, one fixed config, **only the estimator's `seed` varies**
(initialization / optimizer stochasticity). Nothing about the data changes. 4 episodes × 8
seeds = 32 runs.

## Result

| episode | mean | sd | **CV** | min | max | range/mean | seed 42's rank |
|---|---|---|---|---|---|---|---|
| BF3 2comp uniform | 0.0930 | 0.0492 | **53.0 %** | 0.0524 | 0.1682 | **124.6 %** | **1st of 8** |
| BF3 5comp uniform | 0.1879 | 0.0091 | 4.8 % | 0.1675 | 0.1925 | 13.3 % | **1st of 8** |
| BF3 5comp slope | 0.1799 | 0.0182 | 10.1 % | 0.1491 | 0.1901 | 22.8 % | **2nd of 8** |
| BF4 15comp uniform | 0.2895 | 0.0080 | 2.8 % | 0.2753 | 0.2999 | 8.5 % | 8th of 8 |

### 1. Bimodality confirmed — MEASURED
Three of four episodes split into two clusters rather than spreading smoothly:
- **BF3 2comp**: {0.0524, 0.0588, 0.0590, 0.0604, 0.0705} vs {0.1066, 0.1676, 0.1682} — 5 good / 3 bad basins.
- **BF3 5comp slope**: {0.1491, 0.1517} vs {0.1893, 0.1895, 0.1895, 0.1897, 0.1901, 0.1901} — 2 good / 6 bad.
- **BF3 5comp uniform**: seed 42 alone at 0.1675, the other seven at 0.18–0.19.

This is the signature of a **non-convex fit with a bad convergence basin** (consistent with
greedy init + adaptive-N + Adam refinement), not of measurement noise.

### 2. The variance is largest where the problem is *easiest* — MEASURED
CV **53 %** on the 2-component episode versus **2.8 %** on the 15-component one. Counter to
intuition, and it means the low-component episodes — the ones that moved most in the mesh
pilot — are exactly the ones whose single-run numbers are least trustworthy.

### 3. The reported seed sits at the favourable end — MEASURED, but UNDERPOWERED as a claim
`seed=42` (the config default used throughout the repo) ranks 1st, 1st, 2nd, 8th across the
four episodes. Under chance, P(≥2 first-places in 4 draws of 8) ≈ 0.08 — **suggestive, not
significant at n=4**. Seed 42 is a conventional default, not a tuned choice, so this is most
likely luck. ⚠ **The consequence does not depend on intent**: with a bimodal estimator, the
per-episode numbers this project reports are one draw, and on this sample the draw landed
favourably in 3 of 4 cases.

## What this does to today's mesh pilot

The mesh pilot's per-episode deltas were **single-seed**. With per-episode run sd of
0.008–0.049, a paired coarse→fine difference carries a noise sd of roughly
√2 × sd ≈ 0.011–0.070 **before any mesh effect at all**. Over the 12-episode set that is an
aggregate SE of ~0.012 from estimator noise alone, against the **observed** SE of 0.0184.

⟹ **Most of the spread in the mesh pilot is estimator noise, not mesh effect.**
1. The pilot's **aggregate** conclusion (no headline movement; WS Δ t = −0.44) **stands and
   is reinforced** — noise can only widen the interval, and it was already centred on zero.
2. The pilot's **per-episode** numbers are **not interpretable**. In particular the
   `BF3_5comp` "+85.4 % vs −1.9 %" contrast that motivated this whole check is largely a
   basin draw: the coarse value 0.1675 was that episode's **best of 8** seeds (mean 0.1879).
   The fine value 0.3104 is still far above even the worst coarse seed (0.1925), so a real
   degradation there is plausible — but it must be re-measured multi-seed before being cited.
3. ⛔ **Withdrawn**: my earlier reading that "component count drives the direction of the mesh
   effect (≤2 comps −29.7 %, ≥5 comps +13.3 %)". That contrast is built from single-seed
   per-episode values whose noise is of the same order. It may still be true; it is not shown.

## What it does to the paper

The paper reports **one number per episode** (12-episode table, 100-episode averages). Those
per-episode entries inherit a CV of up to 53 %. The 100-episode *average* is far more stable
(noise averages down by ~10×), so the headline is not threatened — but any statement about an
individual episode, any ablation delta measured on a handful of episodes, and any "worst
case" claim is currently a point estimate with no error bar.

## Recommended fix (concrete, and it is also a genuine method improvement)

Multi-restart with **selection on the observation-fit residual, not on ground truth** — run
k restarts, keep the one with the lowest data misfit. This is legitimate at deployment time
(no truth needed), removes the bad basin, and would tighten every number in the paper. The
current `run_estimation` exposes no restart option and returns no residual in its summary; a
small change to `wavesplat/run.py` plus reporting median-of-k with a spread would fix both
the instability and the missing error bars.

**Priority note:** this outranks the dataset regeneration and any retraining. Until the
per-episode spread is either reduced or reported, re-validation on a regenerated dataset
would produce numbers that are equally uninterpretable.

## Limits

n = 4 episodes, 8 seeds, synthetic linear episodes, one estimator configuration
(`init_method: v2_konly`, `adaptive_n: True`, `tracking_window: 3`). Not tested: HOS episodes,
other configs, whether the bad basin is stable across episode content, and whether
residual-based selection actually picks the good basin (that is the next experiment).
