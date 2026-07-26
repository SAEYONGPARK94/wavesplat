# Scoring definitions frozen by the GPU phase BEFORE any held-out number was computed

Written 2026-07-26, after reading `PREREG.md` and `GPU_HANDOFF.md`, and **before**
`bf_ratio.json`, `atoms_cache.jsonl` verdicts, or any arm mean from the held-out 40 was
inspected. Two bars in PREREG §4.2 (S6, S7) are stated by reference to an in-sample number
rather than by formula. The formula is recovered here **from in-sample data only**
(`data_finemesh_2026-07-23` / `tmp/unified_guard_2026-07-26` / `tmp/obsaxis_2026-07-26`),
by requiring that it reproduce the quoted in-sample reference exactly. No held-out quantity
was used to choose either definition.

## S6 — "damaging misroute"

> PREREG §4.2: "≤ **2** damaging misroutes (in-sample was 5/100; pro-rata on 40)".

Recovered definition: **an episode whose deployed (auto+guard) median-of-5 value exceeds the
lane-consistent oracle by more than 0.03 nRMSE**, i.e. `guard − oracle_lane > 0.03`.

In-sample calibration (100 episodes, guard_ratio=0.05, thresh=8):

| cost threshold | count vs `oracle_lane` | count vs `oracle_old` (min-of-medians) |
|---|---|---|
| >0.02 | 9 | 8 |
| **>0.03** | **5** | **5** |
| >0.05 | 3 | 2 |
| >0.06 | 1 | 1 |

`>0.03` is the only threshold that reproduces **5/100** under *both* oracle conventions, so
the definition is not sensitive to the E3 aggregation choice. Pro-rata 5/100 → 2/40, exactly
the bar P3 wrote. Both conventions are reported at scoring time.

## S7 — "the moderate cell must not lose more than it wins (in-sample 6 L / 4 W)"

Recovered definition: on the moderate slice (`comp == 5`), per episode compare the deployed
arm (auto+guard, median-of-5) against **pure_v2** (median-of-5);
`W = #{guard < pure_v2}`, `L = #{guard > pure_v2}`, ties otherwise. **PASS iff L ≤ W.**

In-sample check on the 20 moderate episodes: **W=4, L=6, T=10** — reproduces P1's "6 L / 4 W"
exactly. (Cross-checks that do *not* reproduce it, and are therefore rejected: guard vs
pure_learned = 13W/7L; guard vs oracle = 0W/13L.) Note the bar **fails in-sample**; it is
carried to the fresh draw as written.

## Other definitions taken verbatim, not recovered

- Slices: `SPARSE = comp in (1,2)`, `MOD = comp == 5`, `CROWDED = comp in (15,30)`
  (`tmp/unified_guard_2026-07-26/score_task1.py:93-95`).
- Lane-consistent oracle: `median_lanes( min(v2_lane, learned_lane) )`
  (`tmp/obsaxis_2026-07-26/oracle_consistent.py:40`). `oracle_old = min(median v2, median learned)`
  reported alongside.
- auto+guard re-assembly: per lane route learned iff `n_selected >= 8`, veto to v2 iff
  `bf_peak_ratio < 0.05`, then median over the 5 lanes
  (`score_task1.py:assemble`, identical to the live `_auto_select_init`).
- `n_discordant` per slice: episodes on which the deployed arm's per-episode value differs
  from the comparator's (i.e. the routing actually changed the answer), `|Δ| > 1e-12`.
  `n_discordant < 8` ⟹ UNDERPOWERED, no direction claimed (PREREG §4.2 reporting rule).
- S1 BCa bootstrap: 20 000 resamples of **episodes** (primary, per P3), plus a
  realization-blocked resample over the 20 (scenario, seed) realizations as the free
  robustness check; if they disagree the realization-blocked one is quoted (PREREG §2.3).
