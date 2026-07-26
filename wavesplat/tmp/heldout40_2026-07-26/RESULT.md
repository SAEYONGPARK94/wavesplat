# W4 — THE HELD-OUT FORTY · RESULT (single scoring event, 2026-07-26)

**Frozen protocol** `tmp/heldout40_2026-07-26/PREREG.md` · **recovered S6/S7 formulas**
`tmp/heldout40_2026-07-26/SCORING_DEFINITIONS.md` · **generated evidence**
`tmp/heldout40_2026-07-26/score_results.json` (`score_heldout40.py`, run once) ·
**stdout** `score.log`.

> **VERDICT — the pre-committed branch that the data selected:**
> **S2 FAILED ⟹ "the guard did not generalize and the Ch.5 headline is cut."**
> Compounded by **S1 FAILED ⟹ the aggregate win is deleted**, and by the prereg's catch-all
> **"Any failure ⟹ delete the unified headline and ship the failure map."**
> **5 of 8 bars pass (S1b, S3, S4, S5, S6); S1, S2, S7 fail.**

---

## 0. Compliance record

| Rule (PREREG §5) | Status |
|---|---|
| One shot, arms run once | **Yes.** Each arm produced exactly 200 lane-runs, 0 errors, 0 re-runs. |
| No constant changed after any number was seen | **Yes.** `auto_switch_thresh=8`, `auto_guard_ratio=0.05`, `TRACK_PHASE_PREDICT=False` (verified `flag=False` on all 200 learned rows), lanes `(42,0),(1,1),(2,2),(3,3),(4,4)`, median-of-5. Nothing edited after scoring. |
| Oracle computed lane-consistently by the same script | **Yes.** `median_lanes(min(v2_lane, learned_lane))`; `min-of-medians` reported alongside. |
| Evidence map generated, not typed | **Yes.** Every verdict in this file is copied from `score_results.json`. |
| Episodes | 40/40 `ADEQUATE` on the first two frozen seeds (777, 888); **reserve ladder never consumed** (`seed_ladder_log.json`). |
| Nothing committed | **Yes.** |

**Integrity control (handoff Step 3) — PASSES EXACTLY.** The CPU phase's batchC-protocol
pure-v2 lanes and the run_arms-protocol `v2_konly` lanes agree to **max |Δ| = 0.0 over all 200
lanes** (bit-identical, not merely 1e-10). The two arms are the same estimator; the contrast
is valid.

**Live-router validation (handoff Step 5) — PASSES EXACTLY.** `init_method='auto'` executed
live on all 200 lanes reproduces the re-assembled auto+guard arm with **max per-episode
|Δ| = 0.0, 0/40 episodes differing**, mean 0.08577 both ways. The re-assembly is the deployed
policy, not a paper construction.

### Declared deviation / disclosure (volunteered, nothing was changed by it)

Two disclosures, both made because concealing them would be worse than the facts:

1. **S6 and S7 were stated in the prereg by reference to an in-sample number, not by
   formula.** Their formulas were recovered from **in-sample data only** and written to
   `SCORING_DEFINITIONS.md` **before** any held-out number was computed: S6 = `guard −
   oracle_lane > 0.03` (the only threshold reproducing the quoted 5/100 under *both* oracle
   conventions); S7 = moderate `L ≤ W` against pure_v2 (reproduces the quoted 4W/6L exactly).
2. **A dry run of the scoring script was executed on *fabricated* learned values** (scratchpad,
   random multiples of the real v2 lanes) to validate code paths before the real arms landed.
   It necessarily displayed three **arm-independent** quantities early: the pure-v2 slice means,
   the routing census (`n_selected`), and the `bf_peak_ratio` distribution — hence the inputs to
   **S5**, which does not depend on the learned arm at all. No constant, bar, slice, lane or
   definition was altered afterwards, and no cross-arm quantity involving the real learned arm
   was seen before the single scoring event.

---

## 1. Per-arm means (median-of-5 per episode, then mean over episodes)

| slice | n | pure_v2 | pure_learned | auto (no guard) | **auto+guard** | oracle (lane) | oracle (min-of-med) |
|---|---|---|---|---|---|---|---|
| **all** | 40 | 0.09743 | 0.09074 | 0.08539 | **0.08577** | 0.07553 | 0.07618 |
| sparse | 16 | **0.01348** | 0.03547 | 0.03096 | **0.03190** | 0.01070 | 0.01123 |
| moderate | 8 | **0.02328** | 0.04518 | 0.02462 | **0.02462** | 0.02025 | 0.02138 |
| crowded | 16 | 0.21845 | **0.16878** | 0.17022 | **0.17022** | 0.16800 | 0.16854 |

The regime crossing itself **replicates cleanly out of sample**: pure_v2 wins sparse
(0.0135 vs 0.0355) and moderate (0.0233 vs 0.0452); pure_learned wins crowded
(0.1688 vs 0.2184, a 23% reduction). The **map** transfers. What fails is the claim that the
router+guard captures it.

---

## 2. The frozen bars — five pass, three fail

| ID | Bar | Measured | Verdict |
|---|---|---|---|
| **S1 (PRIMARY)** | BCa 95% CI of mean(auto+guard) − mean(arm) entirely < 0, **both** contrasts | vs **pure_learned**: mean −0.00497, CI **[−0.01011, −0.00119]** ✔ <br> vs **pure_v2**: mean −0.01166, CI **[−0.02502, +0.01691]** ✘ (straddles 0) | **FAIL** |
| **S1b** | mean ≤ 0.095 (wording) / > 0.1011 (dead) | **0.08577** | **PASS** (wording gate; see §5) |
| **S2** | sparse ≤ pure_v2 + 0.005 | **0.03190** vs bar **0.01848** — overshoot **+0.01341** | **FAIL** |
| **S3** | crowded ≤ pure_learned + 0.005 | **0.17022** vs bar **0.17378** | **PASS** |
| **S4** | (deployed − oracle) ≤ 0.015, lane-consistent | **+0.01024** (min-of-medians +0.00959) | **PASS** |
| **S5** | guard fires on 0 crowded **and** min(bf_peak_ratio ∣ crowded) > 0.05 | **0 crowded flips**; min bfr ∣ crowded = **0.0746** | **PASS** |
| **S6** | ≤ 2 damaging misroutes (`cost > 0.03`) | **1** (same count under both oracle conventions) | **PASS** |
| **S7** | moderate must not lose more than it wins | **W=1, L=3, T=4** | **FAIL** |

Free paired Wilcoxons (one-sided, "deployed < ref"), all 40: vs pure_v2 **p = 0.00225**
(medΔ = 0.0000, 22 non-tied); vs pure_learned **p = 0.0944** (medΔ = 0.0000, 24 non-tied).
Realization-blocked BCa (20 realizations) agrees in direction with the episode bootstrap for
both contrasts, so the episode bootstrap is quoted, per PREREG §2.3.

### Reporting rule — discordance per slice (all regime slices are UNDERPOWERED)

| slice | comparator | n_slice | **n_discordant** | verdict |
|---|---|---|---|---|
| sparse | pure_v2 | 16 | **2** | **UNDERPOWERED** — no direction claimed |
| moderate | pure_v2 | 8 | **4** | **UNDERPOWERED** — no direction claimed |
| crowded | pure_learned | 16 | **2** | **UNDERPOWERED** — no direction claimed |
| all | pure_v2 | 40 | 22 | medΔ −0.02158, W/L 17/5, p = 0.00162 |
| all | pure_learned | 40 | 24 | medΔ +0.00025, W/L 11/13, p = 0.0987 |

**Every per-regime non-inferiority endpoint is underpowered on the fresh draw** (n_discordant
2, 4, 2 — all < 8). Per the frozen rule, no per-regime direction may be claimed from these.
The router leaves 14/16 sparse and 14/16 crowded episodes with an answer *identical* to the
respective specialist; the zero-inflation P3 warned about is worse out of sample than in it.

---

## 3. Why S2 failed — the mechanism, from the frozen scoring event

The entire sparse failure is **one episode**, and the guard's behaviour on the four fresh
`BF2_1comp` episodes is the finding of this experiment:

| episode | bf_peak_ratio | n_sel | pure_v2 | pure_learned | guard fired? | deployed | cost vs oracle |
|---|---|---|---|---|---|---|---|
| BF2_1comp_seed777_slope | **0.0216** | 30 | 0.0278 | **0.0211** | **yes (5/5 lanes)** | 0.0278 | +0.0067 |
| BF2_1comp_seed777_uniform | **0.0187** | 19 | 0.0306 | **0.0216** | **yes (4/5 lanes)** | 0.0306 | +0.0097 |
| BF2_1comp_seed888_slope | **0.0652** | 30 | **0.0301** | 0.3286 | **no** | **0.3286** | **+0.2985** |
| BF2_1comp_seed888_uniform | **0.0625** | 16 | 0.0226 | **0.0187** | no | 0.0187 | 0.0000 |

**On the fresh draw the guard's signal is anti-correlated with the failure it was built to
catch.** It fired on exactly the two episodes where the learned arm was *right* (costing
+0.0067 and +0.0097), and stayed silent on the one episode where the learned arm exploded
(+0.2985). In-sample, the damaging `BF2_1comp` misroutes sat at `bf_peak_ratio` 0.024–0.030 and
the constant 0.05 was placed just above them; the fresh catastrophic misroute sits at **0.0652
— above the threshold**. The constant was fitted to the in-sample cluster's *location*, and the
location did not transfer.

Consequences, all measured in the same event:

- **The guard is net-negative on held-out data.** all: 0.08539 (no guard) → **0.08577** (guard).
  sparse: 0.03096 → **0.03190**. In-sample it moved the aggregate 0.0979 → 0.0875 and closed
  60% of the oracle gap; **on 40 fresh episodes it moves the aggregate the wrong way by
  +0.00038.** The 0.0875 headline does not reproduce as a *mechanism*, only as a level.
- **S6 passes at 1 ≤ 2 for the wrong reason.** The bar counts damaging misroutes, and the fresh
  draw contained only one qualifying episode; the guard did not prevent it. P3's failure clause
  for S6 ("the guard was a patch, not a mechanism") is what the *forensics* show even though the
  *count* passes.
- **S5 passes and means less than it was meant to.** 0 crowded vetoes with the crowded
  `bf_peak_ratio` floor at 0.0746 — the crowded-side safety **did** generalize (the threshold
  sits well below the fresh crowded floor). Only the sparse side, which the guard exists for,
  did not.
- **S2's arithmetic is one episode.** That episode alone adds **+0.01866** to the sparse mean;
  the S2 overshoot is **+0.01341**. *(Exploratory, not a bar, reported only to locate the
  failure:* excluding it, sparse would be 0.01211 vs pure_v2 0.01237 and the aggregate 0.07954
  vs pure_v2 0.09915. **This does not rescue anything** — the prereg forbids removing an episode
  because it is inconvenient, and the removed episode is precisely the failure mode the guard
  was introduced to fix.*)*

**S7 (moderate, W=1/L=3).** Losses are all `BF4_5comp`, where median `n_selected` sits at 7–10,
straddling the threshold 8: 1–4 lanes of 5 route learned, and the learned arm is 2–4× worse
there (e.g. seed777_uniform v2 0.0147 → deployed 0.0274). The transition regime is where the
router's per-lane disagreement converts into a real cost, exactly as in-sample (in-sample 4W/6L,
fresh 1W/3L — the bar was already failing in-sample and continues to fail).

**S4 passes on a technicality worth stating.** The deployed−oracle gap is 0.01024 ≤ 0.015, but
the oracle itself is 0.07553 and the *no-guard* router would have been 0.08539 — i.e. the gap is
small mainly because the fresh oracle is close to the fresh router, not because the guard closed
anything.

---

## 4. Routing census (GT-free signals, fresh draw)

| slice | lanes routed learned | episodes majority-learned | guard vetoes | median n_selected | bf_peak_ratio range |
|---|---|---|---|---|---|
| all | 87/200 (96 before the guard) | 18/40 | **9 lanes, all sparse, 2 episodes** | 7.5 | 0.0096 – 0.9001 |
| sparse | 10/80 (19 before) | 2/16 | 9 | 2.0 | 0.0096 – 0.1678 |
| moderate | 10/40 | 2/8 | 0 | 6.0 | 0.0673 – 0.2119 |
| crowded | 67/80 | 14/16 | 0 | 11.5 | 0.0746 – 0.9001 |

The router's coarse behaviour transfers: sparse median `n_selected` = 2, crowded = 11.5, with
moderate straddling at 6. The **two conflated axes** P3 identified are visible again — sparse
episodes with `n_selected` = 16–30 exist on fresh seeds (both `BF2_1comp_seed888`), and the
second constant introduced to patch that decoupling caught the wrong two of them.

---

## 5. The sentence the thesis must carry

Taking the pre-committed wording, unedited, for the branch the data selected (**S2 fails**):

> **The guard did not generalize: on 40 fresh held-out episodes the auto+guard sparse mean is
> 0.0319 against pure-v2's 0.0135 — an overshoot of +0.0134 on a +0.005 bar — so the Chapter-5
> unified headline is cut and the result ships as a failure map.**

Attached, because both are also pre-committed and both are true:

- **S1 failed ⟹ the aggregate win is deleted.** The deployed arm beats pure_learned
  (CI [−0.0101, −0.0012]) but **not** pure_v2 (CI [−0.0250, +0.0169]). The permitted fallback
  wording — *"matches the better specialist in each regime"* — **is not available**, because
  that branch required S2 **and** S3 to pass and S2 failed. Only the crowded half of the regime
  claim survives (S3 PASS).
- **S1b's absolute gates are met (0.0857 ≤ 0.095)** — the level is respectable — but under the
  frozen framing absolute levels are "a coincidence, not a bar." The number may be quoted as a
  level; it may **not** be quoted as a win.

**What survives out of sample (MEASURED, quotable):** the *regime map* — analytic wins sparse
and moderate, learned wins crowded by 23% — reproduces cleanly on fresh seeds; the crowded-side
routing and its safety guard generalize (S3, S5); the deployed policy is exactly the live code
path (0.0 discrepancy on 200 lanes) and the two v2 protocols are bit-identical.

**What does not survive:** the unified estimator as an *aggregate* claim, and the beamform guard
as a *mechanism*. On fresh data it fired twice on episodes where the learned arm was correct,
missed the single catastrophic misroute (whose `bf_peak_ratio` = 0.0652 sits above the frozen
0.05), and left the aggregate 0.0004 worse than no guard at all.

---

## 6. Artifacts

```
data_heldout40_2026-07-26/           40 episodes, all ADEQUATE, manifest + seed ladder log
tmp/heldout40_2026-07-26/
  PREREG.md                          frozen protocol (unmodified)
  SCORING_DEFINITIONS.md             S6/S7 formulas, frozen from in-sample before scoring
  atoms_cache.jsonl                  200 rows, 0 err (40 ep x 5 diffusion seeds)
  bf_ratio.json                      40 bf_peak_ratio (frozen helper)
  v2_arm.jsonl                       pure_v2, 200 lanes (CPU phase, batchC protocol)
  arm_runs.jsonl                     pure_learned, 200 lanes, 0 err, flag=False
  arm_runs_v2_control.jsonl          v2 integrity control, 200 lanes, max|Δ| = 0.0
  arm_runs_auto_live.jsonl           live init_method='auto', 200 lanes, max|Δ| = 0.0
  score_heldout40.py / score.log / score_results.json    the single scoring event
  RESULT.md                          this file
```

Runtime: proposals 200 × ~2.2 s (GPU); learned arm 760 s; v2 control 1749 s; auto live ~1200 s;
6 CPU workers, `nice -n 10`. Nothing committed.
