# Can GT-free restart selection rescue the per-episode numbers? — **No** — 2026-07-23

Script `tmp/restart_selection.py`, log `tmp/restart_selection.log`, raw
`tmp/restart_selection_results.json`. Decision rule was **pre-registered in the script
header before any numbers were produced**.

**Context.** `tmp/ESTIMATOR_VARIANCE_RESULT_2026-07-23.md` found bimodal run-to-run variance
(CV up to 53 % on a fixed episode). The proposed fix was multi-restart selecting on the
**observation-fit loss** — the only quality signal available without ground truth, therefore
the only one a deployed system could use. To expose it, `wavesplat/run.py` now reports
`obs_loss` per frame and `avg_obs_loss` in the summary (pure addition, no behaviour change).

## Result

| episode | ρ(obs_loss, nRMSE) | p | **selected** (GT-free) | median | seed 42 | oracle best | worst |
|---|---|---|---|---|---|---|---|
| BF3 2comp uniform | +0.452 | 0.260 | **0.1066** | 0.0655 | 0.0524 | 0.0524 | 0.1682 |
| BF3 5comp uniform | +0.690 | 0.058 | 0.1817 | 0.1924 | 0.1675 | 0.1675 | 0.1925 |
| BF3 5comp slope | +0.810 | 0.015 | 0.1517 | 0.1895 | 0.1517 | 0.1491 | 0.1901 |
| BF4 15comp uniform | +0.357 | 0.385 | 0.2875 | 0.2904 | 0.2999 | 0.2753 | 0.2999 |
| **pooled mean** | — | — | **0.1819** | 0.1845 | 0.1679 | 0.1611 | — |

**Selection beats the median in 3/4 episodes, but beats the seed-42 default in only 1/4.**

## Verdict against the pre-registered rule: **FAIL**

The rule required the argmin-`obs_loss` run to beat *both* the median run **and** the seed-42
default in most episodes. It beats the median (barely) and loses to seed 42.

1. **The correlation is real but too weak to select on.** ρ is positive in all four episodes
   (+0.36 … +0.81), so `obs_loss` does carry signal — but only one episode reaches p < 0.05
   at n = 8. A moderate rank correlation is not enough to reliably identify the good basin.
2. **It captures almost none of the available headroom.** The deployable range is
   median 0.1845 → oracle 0.1611; the selector lands at 0.1819, i.e. it recovers **~11 %**
   of what perfect selection would give.
3. **On the worst episode it actively hurts.** BF3 2comp (the CV = 53 % case, where the fix
   was most needed): selection picked a **0.1066** run when the median was 0.0655 and the
   best was 0.0524. The bad basin fits the observations *better* there — exactly the failure
   mode the pre-registration named.
4. ⚠ **Losing to seed 42 is partly an artifact of seed 42 being a lucky draw** (it ranked
   1st, 1st, 2nd, 8th — documented *before* this experiment, so this is not a post-hoc
   excuse). But the pre-registered bar was set with that knowledge and is not being moved.

## Consequence: the fix is **error bars, not best-of-k**

This was one of the two pre-registered outcomes and it is the one that occurred. The
project should report **median-of-k with spread**, not a selected best run.

⚠ **This makes the reported numbers slightly worse, and that is the honest direction.** On
these four episodes, moving from the current single `seed=42` value to the median-of-8 raises
mean nRMSE **0.1679 → 0.1845 (+9.9 %)**. The current numbers benefit from a favourable draw;
proper reporting gives that back.

## What may / may not be cited

- **May:** `obs_loss` correlates positively with nRMSE across restarts (ρ +0.36…+0.81,
  MEASURED, n = 8 seeds × 4 episodes; only 1/4 individually significant → the *pooled*
  direction is ARGUED, the per-episode magnitudes are UNDERPOWERED).
- **May:** GT-free restart selection does not close the basin gap (FAIL against a
  pre-registered bar).
- **May NOT:** the `oracle best` column as a result — it selects on ground truth and is not
  deployable. It is an upper bound only.
- **May NOT:** any claim that multi-restart "fixes" the instability.

## Open dependency

This entire analysis assumes the spread is genuinely a **seed** effect. An independent
verification is in progress (`tmp/VARIANCE_VERIFY_fable_2026-07-23.md`) checking, among other
things, whether re-running the *same* seed reproduces exactly — if it does not, the spread is
plain nondeterminism and both this document and the variance document need reframing.

## Limits

4 synthetic linear episodes, 8 seeds, one estimator configuration. No HOS. `obs_loss` is the
mean per-frame training loss, which includes whatever regularization the updater applies —
a purer held-out observation residual might correlate better and has not been tried.

---

## ⚠ Correction (2026-07-23, from `tmp/VARIANCE_TRACE_fable_2026-07-23.md`)

This document explained the selection failure as *"the bad basin fits the observations
better"* — a signal inversion. **A forensic trace shows that is not the general mechanism.**

- For the canonical good/bad pair (seeds 42 vs 1) there is **no inversion at all**:
  `obs_loss` tracks nRMSE tightly (within-run correlation **+0.95**) and is correctly worse
  for the bad seed throughout.
- The inversion cited above belongs to **seed 3 specifically**, and is a **third, distinct
  failure mode**: the prune's 30 % safety check rolls back entirely, leaving all 15
  components free to **locally overfit each 3-frame tracking window**. That is a
  **generalisation gap**, not a corrupted quality signal.

**This reopens the selection idea.** `avg_obs_loss` is the *training* misfit on the very
frames being fitted, so an overfitting run is rewarded. Selecting on a **held-out
observation residual** (points or frames withheld from the fit) should not have that
weakness — and it was already flagged as untried in the Limits section above. That is now
the highest-value follow-up on this line, and it could convert this FAIL into a PASS.
