# GPU-phase handoff — W4 held-out forty (learned lanes + the single scoring event)

Frozen protocol: `tmp/heldout40_2026-07-26/PREREG.md`. **Read it before running anything.**
Everything below is already implied by that file; this page only gives the literal commands.

**Do not change any constant.** `auto_switch_thresh=8`, `auto_guard_ratio=0.05`,
`TRACK_PHASE_PREDICT=False`, lanes `(42,0),(1,1),(2,2),(3,3),(4,4)`, median-of-5.
The 40 episodes are already generated and adequacy-gated; do not regenerate them.

---

## Step 1 — learned atom proposals (GPU, ~40 × 5 proposals)

```bash
cd /home/vexit/wavesplat/wavesplat
nohup python3 tmp/heldout40_2026-07-26/precompute_atoms_heldout40.py \
  > tmp/heldout40_2026-07-26/precompute.log 2>&1 &
```

Writes `tmp/heldout40_2026-07-26/atoms_cache.jsonl`.
**Format is identical to `tmp/wp_integration_2026-07-25/atoms_cache.jsonl`** — one JSON object
per line:

```json
{"episode": "...h5", "diff_seed": 0, "kx": [...], "ky": [...], "energy": [...],
 "n_selected": 30, "sec": 12.3, "err": null}
```

`propose_atoms(frames, depth, nsamp=4, diffusion_seed=ds)` then
`pool_and_select(kx, ky, en, max_n=30)`; the **raw pooled** atoms are cached (not the
selected subset), exactly as in the Stage-1 cache, so the tracker workers need no GPU.

Gate before proceeding: 200 rows, 0 with `err != null`.

## Step 2 — pure-learned arm (CPU workers, atoms from cache)

```bash
nice -n 10 python3 tmp/heldout40_2026-07-26/run_arms_heldout40.py \
  --arms learned --workers 6 --out tmp/heldout40_2026-07-26/arm_runs.jsonl
```

## Step 3 — integrity control (recommended, ~15 min)

```bash
nice -n 10 python3 tmp/heldout40_2026-07-26/run_arms_heldout40.py \
  --arms v2_konly --workers 6 --out tmp/heldout40_2026-07-26/arm_runs_v2_control.jsonl
```

The CPU phase produced the pure-v2 lanes with the **batchC** protocol
(`tmp/rerun_2026-07-25/batchC_ws_phasefix.py`, the same script that produced the in-sample
`pure_v2` values consumed by `tmp/unified_guard_2026-07-26/score_task1.py`); this control
reproduces them through the **run_arms** protocol. The program previously verified these two
agree on the 52-episode Stage-1 overlap. They must agree here to ~1e-10. **If they do not,
stop: the two arms are not the same estimator and the held-out contrast is invalid.**

## Step 4 — `bf_peak_ratio` on the 40 held-out episodes (needed by S5)

Frozen helper, same convention as the guard:

```python
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.initializer import beamform_peak_ratio
ratio = beamform_peak_ratio(EpisodeDataLoader(path).get_init_window(2.0), depth)
```

Mirror `tmp/unified_guard_2026-07-26/score_task1.py:bf_ratios()`; cache to
`tmp/heldout40_2026-07-26/bf_ratio.json`.
**The CPU phase deliberately did not compute this** — it feeds bar S5, and S5 is part of the
single scoring event.

## Step 5 — live validation of the deployed policy (optional but cheap)

```bash
nice -n 10 python3 tmp/heldout40_2026-07-26/run_arms_heldout40.py \
  --arms auto --workers 6 --out tmp/heldout40_2026-07-26/arm_runs_auto_live.jsonl
```

`init_method='auto'` with the frozen defaults; `auto_selected` / `auto_n_selected` are
recorded per lane. Use it to confirm that the re-assembled auto+guard score reproduces the
live router (the same validation `tmp/unified_guard_2026-07-26` ran in-sample).

## Step 6 — THE SINGLE SCORING EVENT

One script, run once, emitting **S1–S7 and the reporting rule in code** (judge-doc C13:
evidence maps are generated, never hand-typed). Inputs:

| Arm | Source |
|---|---|
| `pure_v2` | `tmp/heldout40_2026-07-26/v2_arm.jsonl` (field `nRMSE`, key `(episode, seed)`) |
| `pure_learned` | `tmp/heldout40_2026-07-26/arm_runs.jsonl` (field `avg_nRMSE`, arm `learned`) |
| `auto+guard` | re-assembly: per lane route `learned` iff `n_selected >= 8`, then veto to `v2` iff `bf_peak_ratio < 0.05`; take the routed arm's lane value; median-of-5 |
| `oracle` | per-episode min over the two pure arms, **aggregated lane-consistently with the switch by the same script** (judge-doc E3) |

Slices, verbatim from `tmp/unified_guard_2026-07-26/score_task1.py:93-95`:
`SPARSE = comp in (1,2)` (16), `MOD = comp == 5` (8), `CROWDED = comp in (15,30)` (16).

Required outputs: S1 (BCa episode bootstrap, both contrasts), S1b, S2, S3, S4, S5, S6, S7,
`n_discordant` per slice, and the free paired Wilcoxons. Then take the pre-committed branch
that the result selects — **the branches are already written in PREREG §4**.
