# RESULT — three zero-compute items from the persona-debate worklist (2026-07-26)

Protocol frozen in `PRESTATE.md` (written before any number was produced). No estimator
was run: every value is re-derived from JSON/JSONL already on disk. Nothing committed.

---

## 1. W9 — observable-coordinate phase diagram (crux A + crux F) → `results/gifs/diag_observable_axis.png`

**The pre-stated rule returns FOR P3.**

| axis | what it is | Spearman ρ vs per-episode Δ | \|ρ\| | p (n=100) |
|---|---|---|---|---|
| (a) `n_components` | the simulator knob — **not readable at runtime** | −0.5145 | **0.514** | 4.4e-08 |
| (b) `bf_peak_ratio` | GT-free observable (`initializer.py:292`) | −0.2569 | **0.257** | 9.9e-03 |

`|ρ_bf| = 0.257 < |ρ_ncomp| − 0.05 = 0.464` ⇒ **`n_components` dominates clearly.** Under the
rule frozen in `PRESTATE.md` the verdict text is fixed and is written on the figure:

> the phase diagram is drawn on a coordinate the runtime cannot read — **P3 is right, and
> Ch.5 says so in those words.**

Robustness (same direction, decided in advance as secondary): with Δ formed per lane and then
medianed, `|ρ_ncomp| = 0.551` vs `|ρ_bf| = 0.257`. The two axes are themselves correlated
(ρ = +0.573), so this is not two independent orderings — the observable is a **partial** proxy
for the knob, and a weak one.

**Post-hoc (NOT part of the frozen rule, reported because P3's two-axes charge is about it):**
the signal the shipped router actually reads, `n_selected`, scores `|ρ| = 0.372` (p=1.4e-04) —
better than `bf_peak_ratio`, still well below the knob. So both runtime-readable coordinates
order the boundary worse than the simulator knob does; the charge is not answered by swapping
one observable for the other.

Calibration note, recorded because `PRESTATE.md` §5 asked for a prediction first: I predicted a
near-tie inside the tolerance band. That prediction was **wrong**, and by a wide margin
(0.257 vs the predicted 0.6–0.8). The prediction is left in the pre-state file unedited.

**Honesty clause, discharged on the figure (pre-committed in §4):** the 30 real windows cluster
at `bf_peak_ratio` 0.51–0.95 (IQR) — a band where only **6 of 100** synthetic episodes live and
**none** of them is sparse. Inside that band 4 of 6 synthetic episodes favour the learned init
while 20 of 29 real windows favour the analytic one. Real `bf_peak_ratio` spans 0.11–1.00 and
**0/30 fall below the 0.05 guard**, so the synthetic guard never fires on real data (independent
re-derivation of the `PREREG_bf_ratio_real.md` cell-C finding). Crux F is therefore answered in
the narrow sense — *the axis is measurable outside the simulator* — and not in the wide sense:
*it was only varied inside it.* Both sentences are on the figure. Also on the figure: the real Δ
uses the `/gt_std` nowcast-map normaliser and the synthetic Δ uses `/Hs`, so the real markers are
placed for coordinate and sign, never for magnitude comparison against the synthetic cloud.

---

## 2. E3 — lane-consistent oracle → `oracle_consistent.json`

One script now computes the deployed number and the oracle with the **same aggregation**, so the
sentence is quotable for the first time.

| slice | n | pure v2 | pure learned | auto | **deployed (auto+guard)** | oracle OLD (min-of-medians) | **oracle LANE-CONSISTENT** |
|---|---|---|---|---|---|---|---|
| all | 100 | 0.11448 | 0.10105 | 0.09786 | **0.08747** | 0.08177 | **0.08061** |
| sparse | 40 | 0.02037 | 0.05345 | 0.05087 | 0.02490 | 0.01374 | 0.01261 |
| moderate | 20 | 0.02671 | 0.03652 | 0.02790 | 0.02790 | 0.02030 | 0.01997 |
| crowded | 40 | 0.25246 | 0.18092 | 0.17982 | **0.17982** | 0.18053 ← *paradox* | **0.17893** |

**The paragraph.** The deployed single-estimator configuration measures **0.0875** mean nRMSE
over the 100-episode finemesh benchmark. Its oracle upper bound, computed with the *identical*
aggregation the deployed number uses — route per lane, median over the five lanes, mean over
episodes — is **0.0806**, not the 0.0818 previously quoted. The 0.0818 figure was a min *of*
medians, which is not an upper bound on a per-lane policy, and that is exactly why the crowded
slice of the deployed configuration (0.17982) appeared to sit *below* its own oracle (0.18053).
Under the consistent definition the crowded oracle is 0.17893, i.e. below the deployed value as
it must be, and the per-episode violations vanish: **6 episodes → 0**, in every slice. With one
definition throughout, the guard moves the aggregate 0.09786 → 0.08747, which is **60.2 %** of
the 0.01725 router-to-oracle gap — not the 64.6 % the inconsistent pairing implied. The residual
0.00686 is concentrated: 38 % of it in `episode_BF2_1comp_seed3_slope` (a single sparse episode
whose `bf_peak_ratio` of 0.19 puts it out of the guard's reach) and 77 % in eight episodes.

**Is "fraction of gap closed" meaningful?** Marginally, and only with three qualifiers stated
beside it: (i) it is a ratio of two aggregate differences with **no uncertainty interval** — one
episode moves it by several points; (ii) it is entirely **in-sample** — the guard ratio 0.05 was
estimated from this same scoring data (P3's "fitting, not shopping"), so the oracle gap it closes
is the gap it was fitted to close; (iii) the denominator is an oracle that peeks at the answer per
lane, which makes it a *tighter*, more honest bound than the old one and therefore a *smaller*
headline. Recommended wording: "with a single aggregation throughout, the guard closes 60 % of the
router-to-oracle gap on the same 100 episodes it was tuned on." The old "closes ~2/3 of the oracle
gap" should be replaced, not merely caveated.

Follow-on for the packet (**b3** in the debate's correction list is now dischargeable):
`HANDOFF` §1 and `MEMORY.md` still carry the unsupported "2/3 of the oracle gap"; both should be
restated to 60 % with the lane-consistent oracle, or deleted.

---

## 3. W1a — machine-generated evidence map → `gen_evidence_map.py` → `docs/EVIDENCE_MAP_GENERATED_2026-07-26.md`

Deterministic (verified: two consecutive runs byte-identical), no compute, re-runnable:

```
python3 tmp/obsaxis_2026-07-26/gen_evidence_map.py > docs/EVIDENCE_MAP_GENERATED_2026-07-26.md
```

Ten claims, each with the number **read** from its result file and the verdict **computed** from
its frozen criterion in code. The two errors the debate caught are now structurally impossible:

- **E1** — the collapse count is produced by `learned < 1.0` applied in code and prints
  `2/3, NOT 3/3`, alongside the two counts P3's R3 rule demands, kept separate:
  end-to-end (detector ∘ fallback) `1/3` and large-reduction (Δ ≥ 0.5) `2/3`. The per-window
  line is auto-annotated `AA14_w0 2.31->1.07 [NOT FIXED]`.
- **E2** — any mean-based line whose median is unchanged is auto-caveated: the deployed real
  aggregate row prints `MEAN 0.8815 -> 0.7747; MEDIAN 0.8091 -> 0.8091`, verdict
  **NOT A GENERAL WIN**, plus "the entire mean delta comes from the 2 window(s) the detector
  touched."
- **E3** — the oracle row reads `oracle_consistent.json` and prints both the lane-consistent
  60.2 % and what the old pairing would have said, with the paradox count `6 → 0`.
- **W9** — row 10 emits today's negative automatically, including the verdict sentence P3 is owed.

Router threshold is read as `8` throughout (`E4`; `switch_results.json::by_threshold["8"]`), so
the doc-vs-code ambiguity cannot be reintroduced by hand-typing.

The script is 197 lines rather than the ~100 requested; the excess is entirely the frozen-criterion
strings and the auto-caveat texts, which are the part that carries the value. It has no
project imports, so it keeps working if the analysis scripts move.

---

## Files

| file | what |
|---|---|
| `tmp/obsaxis_2026-07-26/PRESTATE.md` | frozen rule + prediction, written before computing |
| `tmp/obsaxis_2026-07-26/compute_obsaxis.py` → `obsaxis_stats.json` | the two Spearman ρ + per-episode/per-window table |
| `tmp/obsaxis_2026-07-26/make_obsaxis_fig.py` → `results/gifs/diag_observable_axis.png` | the figure |
| `tmp/obsaxis_2026-07-26/oracle_consistent.py` → `oracle_consistent.json` | E3 fix |
| `tmp/obsaxis_2026-07-26/gen_evidence_map.py` → `docs/EVIDENCE_MAP_GENERATED_2026-07-26.md` | W1a |
| `tmp/obsaxis_2026-07-26/RESULT.md` | this file |
