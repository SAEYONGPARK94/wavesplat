# PREREG — W5: the shipped `auto` policy, run end-to-end on the 30 real windows

**Frozen 2026-07-26, BEFORE this session executed any routing decision on a real window and
before any policy score existed.** Nothing is committed. No file under `scripts/`,
`wavesplat/` or `data/` is modified.

Trigger: persona debate R2 — P1 ("Blind policy on 30 real windows, REQUIRED: route
distribution predicted in writing first") and P3 (CRUX B: "one policy iff one callable,
same constants, no arena-conditional branch … failure branch pre-committed: policy loses
real by > +0.02 → report *two policies*, in the thesis, in that wording").

---

## 0. ⚠ HONESTY DECLARATION — what is genuinely new here, and what is not

This run is **not blind, and this document does not pretend it is.** Stated before running:

1. **The routing outcome is already MEASURED, not predicted.** Cell C
   (`tmp/figs_opus_real_2026-07-26/PREREG_bf_ratio_real.md` → `bf_ratio_real.json`,
   pre-registered with its prediction written first, verdict **cell C**) measured
   `bf_peak_ratio` on all 30 real windows: range **0.27–1.00**, normal-window minimum
   0.27 — every window is an order of magnitude above the 0.05 veto threshold, so
   **the guard cannot fire anywhere.** And `wp_real.json` already records `n_eff ≈ 22–24`
   per window, far above `auto_switch_thresh = 8`. **Predicted route distribution,
   written here before execution: learned 30/30, v2 0/30, guard fires 0/30.**
2. **Both arms' per-window scores already exist on disk** (`wp_real.json`: `hand_now`,
   `learned_now`), produced under the frozen `tmp/wp_real_2026-07-25/PREREG.md`. The
   composed policy number is therefore a **deterministic function of data already
   measured**. Reporting it as a "test the policy passed or failed" would be
   criterion-theatre.
3. **What has genuinely never been computed:** the shipped router's own statistic
   `n_selected` — `learned_init.pool_and_select(...)` over the pooled proposal atoms — on
   a real window. Every previous real-data number used `n_eff` (a *per-posterior-sample*
   atom count from `run_wp_real.py`), which is a **different statistic** from the one the
   deployed router reads. Grep confirms `pool_and_select` has never been called on real
   observations. This run executes the shipped `_auto_select_init` verbatim.
4. **Therefore the status of this document is:** an *official, frozen, auditable record*
   of the composed policy (W5), with the decision wording pre-committed — **not** a
   discovery instrument. Its verdict may not be described anywhere as a "blind test",
   a "held-out result", or "confirmation of a prediction". The one legitimate confirmatory
   element is item 3.

## 1. The policy — one callable, one config, no arena-conditional branch

**Constants, verbatim shipped defaults, unchanged for this run:**
`init_method='auto'`, `auto_switch_thresh = 8`, `auto_guard = True`,
`auto_guard_ratio = 0.05`, `learned_energy_thresh = 0.03`, `learned_dedup_dk = 0.03`,
`max_n = 30`, `learned_nsamp = 4`.

**Router = the shipped code, not a replica.** For each window:

1. `sc = d3_compare_gifs.scene_real(record, win=win)` — unmodified, same seeds as
   `wp_real.json`.
2. Build `FrameData` objects from `sc['P']`, `sc['z']` grouped by unique observation time
   (`x_obs, y_obs, z_obs` verbatim, `confidence = ones`, `t` = the frame time). This is a
   **format adapter only** — no value is altered, nothing is filtered, no mean removed.
3. `SequentialUpdater({'init_method':'auto', 'depth': sc['depth'], 'seed': 42, …shipped
   defaults})._auto_select_init(frames, verbose=True)` is called **directly**. It performs,
   in shipped code: `propose_atoms` (cond-diffusion, nsamp=4, `diffusion_seed` =
   `cfg['seed']` per `_auto_select_init`'s own default) → `pool_and_select` →
   `n_selected` → `n_selected >= 8 ? learned : v2_konly` → if learned,
   `beamform_peak_ratio(frames, depth) < 0.05 ? veto to v2_konly : learned`.
4. Recorded per window: `n_selected`, `bf_peak_ratio`, `guard_fired`, route.

**PRIMARY diffusion seed = 42**, because `wavesplat/run.py:DEFAULT_CONFIG['seed'] = 42` and
`_auto_select_init` defaults `learned_diffusion_seed` to `self.seed`. This is the literal
shipped configuration; it is fixed here before running and is not chosen from any score.

**Robustness diagnostic (declared secondary, cannot change any verdict):** the router is
additionally evaluated at `diffusion_seed ∈ {0,1,2,3,4}` — the five seeds the `wp_real`
learned arm itself draws — and the route agreement across all six seeds is reported.

### 1.1 Declared scope statement — the arm mapping on real records (state, do not bury)

The shipped router emits the label `'v2_konly'` or `'learned'`. On the **synthetic**
benchmark those labels name two initializers feeding one tracker. On **real stereo
records** the deployed analytic estimator is `d3_sa_nomp.mmse_bank_estimate` (a linear
MMSE over a continuous bank on scattered observations), **not** the `v2_konly` initializer
+ tracker — and per `docs/HANDOFF_OPUS_2026-07-26.md` §2 the `wavesplat.run` default path
has never been run on a real record. This run therefore maps:

- route `'learned'` → the frozen learned arm of `wp_real.json` (5 diffusion seeds × n = 4
  BMA, 80-iter polish, lead-0 pooled-GT nRMSE, median-of-5);
- route `'v2_konly'` → the frozen analytic arm of `wp_real.json` (`mmse_bank_estimate`,
  same metric).

**Consequence that must be reported, not hidden:** what transfers across arenas is the
**router** (a GT-free decision function on observations) and the two *arm labels* — not one
literal end-to-end estimator binary. "One estimator, one config, both arenas" is therefore
**not literally executable today**, independently of how the numbers come out. That is a
finding of this run and belongs in the result.

### 1.2 Declared cache reuse

Per-window arm scores are **reused from `wp_real.json`** rather than recomputed. Licensed
by the same-session bit-identity control in
`tmp/collapse_controls_2026-07-26/PREREG_GPU_ADDENDUM.md` §5, which recomputes the learned
arm live on the three design collapse windows and requires reproduction of `wp_real.json`
to `1e-9`. **If that control FAILS, this run's policy scores are invalid** and must be
recomputed from scratch before any verdict is reported. The router itself (§1, item 3) is
always computed live and is never cached.

## 2. Endpoints and analysis sets (frozen)

**Metric:** the same lead-0 (nowcast) map nRMSE as `wp_real.json` —
`sqrt(mean((pred−gt)²))/std(gt_pooled)`, finite mask ≥ 8 px. Nothing else is an endpoint.

- **PRIMARY set: all 30 windows** (5 design records × 6 windows) — the set P1 named.
- **SECONDARY set: the 27 normal windows** (excluding the 3 frozen collapse windows
  AA14_w0, AA15a_w3, LJ18g_w3) — the set P1's ≥20/27 clause is defined on.
- **Reported as a declared robustness view only (never a verdict): record-unit medians
  (n = 5) and means.** Means may not carry a win claim (do-not-cite list, §5 of the
  handoff).

Sign convention, fixed here: **`Δ = hand − policy`**; `Δ < 0` means the policy is **worse**
than hand-everywhere.

## 3. Frozen bars — pre-committed, verbatim from the debate

**B1 — PRIMARY (P1, R2):**
> PASS iff `median(policy nRMSE over 30) ≤ median(hand-everywhere nRMSE over 30) + 0.010`.

**B2 — P1's scoping clause (R2, "learned-routed loss ≥20/27 & medΔ < −0.03 → reportable,
abstract says synthetic-scoped"):**
> If the policy routes **learned on ≥ 20 of the 27 normal windows** AND the median paired
> `Δ = hand − policy` over those routed-learned normal windows is **worse than −0.03**,
> then the loss is **reportable**, and the thesis abstract must scope the unified
> ("one estimator") claim to **synthetic** data. P1 wrote this loss direction as
> reportable-but-synthetic-scoped; it is frozen here in exactly that form.

**B3 — P3's pre-committed failure wording (R2, CRUX B):**
> If `median(policy) − median(hand)` on the primary set **exceeds +0.02**, the thesis must
> report the result with the words **"two policies"** — i.e. the deployed real-data policy
> (analytic everywhere + triggered fallback) and the deployed synthetic policy
> (`auto` + guard) are **two policies, not one**. This wording is committed now and is used
> verbatim in `RESULT.md` if the condition fires.

**B4 — one-policy lint (P3, "one callable, same constants, no arena-conditional branch"):**
> The router call must be the shipped `_auto_select_init` with the shipped constants and no
> branch on record/arena. The runner asserts the four constants at runtime and records
> them in the output JSON. Any deviation invalidates the run.

Bars are evaluated **independently**; B1 passing does not suppress B2/B3, and B3 firing
does not suppress B1's reported value.

## 4. Deliverable (frozen)

`RESULT.md` containing:

1. the **route distribution** (learned / v2 / guard-fired counts) with `n_selected` and
   `bf_peak_ratio` per window, plus the 5-seed route-agreement diagnostic;
2. the **per-window policy nRMSE** table alongside hand and learned;
3. **median policy vs median hand** on the primary set and on the 27-normal secondary set,
   with the paired-Δ distribution;
4. the **B1 / B2 / B3 / B4 verdicts**, each with its pre-committed sentence, whatever they
   say;
5. the §0 honesty declaration reproduced, and the §1.1 scope statement reported as a
   finding;
6. **the exact sentence the thesis must now carry**, derived only from the frozen wording
   above.

**Invalidation conditions:** any constant changed from §1; any arena-conditional branch;
any window added or dropped; the Task-A bit-identity control failing; any bar altered after
this file was written; any oracle or GT-dependent quantity entering the router.
