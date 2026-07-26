# RESULT — W5: the shipped `auto` policy on the 30 real windows

**2026-07-26. Protocol frozen in `PREREG.md` (sha256 `bbe759386fca36c0…`) before this
session executed any routing decision on a real window. Nothing committed. No file under
`scripts/`, `wavesplat/` or `data/` modified.**

Artifacts: `PREREG.md`; `run_blind_auto.py` → `blind_auto_real.json` (30 windows × 6
diffusion seeds through the shipped `_auto_select_init`, 769 s, RTX 4080 Super);
`analyze.py` → `results.json`, `analysis.txt`.

---

## 0. Honesty declaration, reproduced from `PREREG.md` §0 (binding)

This run is **not blind, and this document does not pretend it is.**

- The routing outcome was **already MEASURED** before it ran (cell C,
  `tmp/figs_opus_real_2026-07-26/`), and **both arms' per-window scores already existed on
  disk** (`wp_real.json`). The composed policy number is a deterministic function of
  already-measured data. It may **not** be described anywhere as a "blind test", a
  "held-out result", or "confirmation of a prediction".
- **What was genuinely new:** the shipped router's own statistic `n_selected`
  (`learned_init.pool_and_select` over the pooled proposal) had **never been computed on a
  real window** — every prior real number used `n_eff`, a different statistic. That, and
  the composed record itself, are this run's contribution.
- Status: an **official, frozen, auditable record (W5)** with the decision wording
  pre-committed.

## 1. Headline

**All three outcome bars go against the unified claim. On real seas `auto` is not a router
at all — it degenerates to "learned everywhere", and the learned arm is the measured losing
arm.**

| bar | condition | measured | verdict |
|---|---|---|---|
| **B1** (P1, primary) | median(policy) ≤ median(hand) + 0.010 over 30 | **+0.0270** | **FAIL** |
| **B2** (P1, scoping) | routed-learned ≥ 20/27 normals AND median paired Δ < −0.03 | **27/27**, **−0.0571** | **FIRES** |
| **B3** (P3, wording) | median(policy) − median(hand) > +0.02 | **+0.0270** | **FIRES** |
| **B4** (P3, lint) | one callable, shipped constants, no arena branch | verified | **PASS** |

## 2. Route distribution — the guard never fires

Primary diffusion seed **42** (`wavesplat/run.py:DEFAULT_CONFIG['seed']`, which
`_auto_select_init` uses as `learned_diffusion_seed`).

| | count |
|---|---|
| routed **learned** | **30/30** (27/27 of the normal windows) |
| routed `v2_konly` | **0/30** |
| guard vetoes | **0/30** |

- `n_selected` (router clause 1, threshold 8): **min 11, median 30, max 30.** Every window
  clears the threshold; 21/30 sit at the `max_n = 30` cap.
- `bf_peak_ratio` (guard, veto below 0.05): **min 0.1082, median 0.8789, max 0.9982** —
  every window is **2 to 20× above** the veto threshold.
- **Independent confirmation of cell C through the shipped callable:** the live guard values
  agree with `tmp/figs_opus_real_2026-07-26/bf_ratio_real.json` to **max |Δ| = 4.6e-08**,
  despite that pre-check being a hand-transposed replica taking `z − mean(z)` and this run
  calling `initializer.beamform_peak_ratio` on `FrameData` with `z` as-is. The cell-C
  measurement is reproduced by the deployed code path.
- 🛑 **Summary-layer correction:** `docs/HANDOFF_OPUS_2026-07-26.md` §2 states
  "bf_peak_ratio on all 30 real windows reads **0.27–1.00** (normal min 0.27)". The measured
  minimum is **0.1082** (AA15a_w0); 0.265 is AA14_w0, a *collapse* window. The verdict is
  unaffected (0.108 ≫ 0.05), but the handoff line is wrong and must be fixed.

### 2.1 NEW: the router's statistic is seed-unstable on real data

The 6-seed diagnostic (declared secondary; cannot change any verdict) shows `n_selected` is
**not** the stable ≈24 the record implied:

| window | `n_selected` across seeds (42,0,1,2,3,4) | routes |
|---|---|---|
| AA15b_w3 | 30, 30, **7**, 9, 22, 25 | **learned ×5, v2_konly ×1** |
| AA15b_w2 | 11, 15, 30, 16, 27, 30 | learned ×6 |
| AA15b_w0 | 30, 30, 30, **9**, 30, 30 | learned ×6 |
| AA14_w5 | 30, 30, 30, **11**, 30, 30 | learned ×6 |

- **29/30 windows are unanimous across all six seeds; 1/30 (AA15b_w3) flips**, i.e. 1 of 180
  seed-window decisions crosses the threshold.
- The prior real-data figure "n_eff ≈ 22–26" is a *per-posterior-sample* count. The
  statistic the router actually reads spans **7 to 30 on the same window** depending only
  on the diffusion seed. Any future claim that the router "reads crowding" on real data
  must carry this: on real seas its input has a run-to-run range comparable to its own
  decision threshold.

## 3. Per-window policy vs hand (Δ = hand − policy; negative = policy worse)

Because the route is learned on all 30 windows, **policy ≡ learned arm** everywhere.

| window | route | n_sel | bf_ratio | hand | learned | **policy** | Δ |
|---|---|---|---|---|---|---|---|
| YS17_w0 | learned | 30 | 0.9515 | 0.7407 | 0.8429 | 0.8429 | −0.102 |
| YS17_w1 | learned | 30 | 0.9863 | 0.7328 | 0.8361 | 0.8361 | −0.103 |
| YS17_w2 | learned | 30 | 0.9109 | 0.9422 | 1.0829 | 1.0829 | −0.141 |
| YS17_w3 | learned | 30 | 0.9982 | 0.8382 | 1.0049 | 1.0049 | −0.167 |
| YS17_w4 | learned | 30 | 0.8843 | 0.8066 | 1.0425 | 1.0425 | −0.236 |
| YS17_w5 | learned | 14 | 0.8729 | 0.9522 | 0.9947 | 0.9947 | −0.043 |
| AA15a_w0 | learned | 30 | 0.1082 | 0.6301 | 0.9419 | 0.9419 | −0.312 |
| AA15a_w1 | learned | 30 | 0.9081 | 0.8117 | 0.8362 | 0.8362 | −0.025 |
| AA15a_w2 | learned | 16 | 0.4125 | 0.6333 | 0.5329 | 0.5329 | +0.100 |
| AA15a_w3 ⚑ | learned | 19 | 0.9885 | 1.1200 | 0.7913 | 0.7913 | +0.329 |
| AA15a_w4 | learned | 23 | 0.9523 | 0.6645 | 0.7057 | 0.7057 | −0.041 |
| AA15a_w5 | learned | 26 | 0.4384 | 0.4610 | 0.4281 | 0.4281 | +0.033 |
| AA14_w0 ⚑ | learned | 30 | 0.2650 | 2.3091 | 1.0709 | 1.0709 | +1.238 |
| AA14_w1 | learned | 30 | 0.5482 | 0.8487 | 0.8203 | 0.8203 | +0.028 |
| AA14_w2 | learned | 26 | 0.2468 | 0.5528 | 0.6644 | 0.6644 | −0.112 |
| AA14_w3 | learned | 30 | 0.2231 | 0.8269 | 0.8450 | 0.8450 | −0.018 |
| AA14_w4 | learned | 28 | 0.4459 | 0.6625 | 0.5440 | 0.5440 | +0.119 |
| AA14_w5 | learned | 30 | 0.9893 | 0.8133 | 0.9612 | 0.9612 | −0.148 |
| AA15b_w0 | learned | 30 | 0.6762 | 0.6667 | 0.7443 | 0.7443 | −0.078 |
| AA15b_w1 | learned | 30 | 0.4901 | 0.4757 | 0.6224 | 0.6224 | −0.147 |
| AA15b_w2 | learned | 11 | 0.6498 | 0.6875 | 0.7759 | 0.7759 | −0.088 |
| AA15b_w3 | learned | 30 | 0.7452 | 0.6489 | 0.7060 | 0.7060 | −0.057 |
| AA15b_w4 | learned | 30 | 0.8734 | 0.6532 | 0.6944 | 0.6944 | −0.041 |
| AA15b_w5 | learned | 24 | 0.7183 | 0.6197 | 0.6254 | 0.6254 | −0.006 |
| LJ18g_w0 | learned | 30 | 0.9423 | 0.9814 | 0.9324 | 0.9324 | +0.049 |
| LJ18g_w1 | learned | 30 | 0.9588 | 0.8529 | 0.8935 | 0.8935 | −0.041 |
| LJ18g_w2 | learned | 30 | 0.9028 | 0.9471 | 0.8132 | 0.8132 | +0.134 |
| LJ18g_w3 ⚑ | learned | 30 | 0.9502 | 2.9254 | 0.9615 | 0.9615 | +1.964 |
| LJ18g_w4 | learned | 30 | 0.9952 | 0.8225 | 0.9263 | 0.9263 | −0.104 |
| LJ18g_w5 | learned | 30 | 0.9880 | 0.8168 | 0.9215 | 0.9215 | −0.105 |

⚑ = frozen collapse window.

## 4. Medians and the frozen bars

| set | n | median hand | median policy | policy − hand | median paired Δ |
|---|---|---|---|---|---|
| **PRIMARY: all 30** | 30 | **0.8091** | **0.8361** | **+0.0270** | −0.0418 |
| SECONDARY: 27 normals | 27 | 0.7407 | 0.8361 | **+0.0954** | −0.0571 |

**B1 — FAIL.** `+0.0270 > +0.0100` allowance. On the 27 non-collapse windows the gap is
almost four times the allowance (`+0.0954`).

**B2 — FIRES.** Routed learned on **27/27** normal windows (≥ 20 required) with median
paired Δ **−0.0571** (worse than −0.03 required). Pre-committed consequence, in force:

> The loss is **reportable**, and the thesis abstract must scope the unified
> ("one estimator") claim to **synthetic** data.

**B3 — FIRES.** `median(policy) − median(hand) = +0.0270 > +0.02`. Pre-committed wording,
verbatim and now mandatory:

> The thesis must report **"two policies"**.

**B4 — PASS.** Router = the shipped `SequentialUpdater._auto_select_init`, called directly;
`auto_switch_thresh=8`, `auto_guard=True`, `auto_guard_ratio=0.05` verified in source;
`DEFAULT_CONFIG['init_method']='auto'`, `DEFAULT_CONFIG['seed']=42` = the primary diffusion
seed; no arena-conditional branch anywhere in the run.

### Declared secondary views (never a verdict)

- Record-unit medians (n = 5): hand 0.8224 / 0.6489 / 0.8201 / 0.6511 / 0.9000 vs policy
  0.9998 / 0.7485 / 0.8327 / 0.7002 / 0.9239 — **the policy is worse on 5/5 records**
  (median-of-record-medians +0.0126).
- Window win/loss: **policy worse than hand on 21/30 windows (21/27 normals)**.
- Windows left unusable (nRMSE ≥ 1.0): policy **4/30** vs hand 3/30 — the policy repairs two of the
  three analytic collapses below the line (AA15a_w3 1.120→0.791, LJ18g_w3 2.925→0.962;
  AA14_w0 2.309→1.071 stays above) and *creates* three new over-line windows on YS17
  (w2 1.083, w3 1.005, w4 1.043).
- Means are not reported as a win/loss statistic (standing do-not-cite rule).

## 5. Scope finding, reported not buried (`PREREG.md` §1.1)

The shipped router emits `'learned'` / `'v2_konly'`. On synthetic data those label two
initializers feeding one tracker; on real records the deployed analytic estimator is
`d3_sa_nomp.mmse_bank_estimate`, and the `wavesplat.run` default path **has never been run
on a real record**. This run therefore composed: route `learned` → the frozen `wp_real`
learned arm; route `v2_konly` → the frozen `wp_real` analytic arm.

**Consequence, independent of how the numbers came out:** what transfers between arenas is
the **router** — a GT-free decision function over observations — and two *arm labels*, not
one literal end-to-end estimator binary. **"One estimator, one config, both arenas" is not
literally executable today.** As it happens the point is moot: the router never takes its
second branch on real data, so on real seas the "unified estimator" is indistinguishable
from running the learned arm unconditionally.

**Cache reuse (declared, `PREREG.md` §1.2):** per-window arm scores were reused from
`wp_real.json`. Licensed by the same-session bit-identity control in
`tmp/collapse_controls_2026-07-26/RESULT_GPU.md` §1, which recomputed the learned arm live
on the three design collapse windows and reproduced `wp_real.json` at **max |Δ| =
0.000e+00**. The router itself was computed live on every window.

## 6. The sentence the thesis must now carry

> Run with its shipped defaults (`init_method='auto'`, `auto_switch_thresh=8`,
> `auto_guard_ratio=0.05`) on the 30 real stereo windows, the unified estimator routes to
> the learned arm on **30/30** windows and its guard never fires — real seas read
> `bf_peak_ratio` 0.11–1.00, two to twenty times above the 0.05 veto — so on real data
> `auto` degenerates to the learned arm everywhere. Its median per-window nowcast nRMSE is
> **0.836 against 0.809** for the analytic estimator everywhere (**+0.027** over all 30
> windows, **+0.095** over the 27 non-collapse windows), worse on 21/30 windows and on 5/5
> records. **The single-configuration claim is therefore scoped to synthetic data, and the
> deployed system is two policies**: `auto` + guard on simulated seas, and analytic
> everywhere with a triggered fallback on real records.

## 7. Consequences (pre-committed; no new interpretation added)

1. **`docs/HANDOFF_OPUS_2026-07-26.md` §0's opening sentence must be rewritten.** "One
   estimator, one config" is now measured to be a synthetic-arena statement. The handoff's
   own §2 already anticipated this ("real deployment = hand + amp-fallback (two policies —
   report in those words)"); this run is the executed, frozen record that fixes the wording
   as mandatory rather than anticipated.
2. **Do not run `wavesplat.run` defaults on real records** — reaffirmed with the number
   attached: doing so costs +0.027 median (+0.095 on normal windows) versus the analytic
   arm, and turns 3 unusable windows into 4.
3. **The guard is uninformative on real seas** — not "not yet validated", **measured
   uninformative**: 0 fires with a 2–20× margin to the threshold on every window. The
   quadrant it was fitted to (dense proposal + single dominant swell) does not exist in
   this real record set.
4. **New open item:** the router's own input statistic is seed-unstable on real data
   (7–30 on one window, one route flip in 180 seed-window decisions). Any future GT-free
   routing claim on real data needs `n_selected` reported as a distribution over diffusion
   seeds, not a point value.
