# PREREG ADDENDUM (GPU) — the learned arm's readings on the 18 collapse windows

**Frozen 2026-07-26, BEFORE the learned arm was run on any held-out collapse window and
before any held-out learned score existed.** This file only *fills the PENDING-GPU slot*
already defined in `PREREG.md` §3 and `RESULT.md` §5. It adds **no new endpoint, no new
window, no new criterion**. Everything decision-relevant (target set, metric, the two
counts, the comparator `C_best`, the 1.5× bar) was frozen earlier and is restated here
verbatim so the run can be executed and judged without reopening those files.

Nothing is committed. No file under `scripts/`, `wavesplat/` or `data/` is modified;
held-out record entries are injected at runtime by the already-frozen
`tmp/heldout_real_2026-07-26/heldout_cfg.py` (additive only).

---

## 1. The learned arm — protocol frozen verbatim from `tmp/wp_real_2026-07-25/run_wp_real.py`

The learned arm is **not re-specified here**. It is the *same callable sequence*, in the
same order, with the same constants, as Phase B of `run_wp_real.py` (the run that produced
`wp_real.json`, the source of the three design learned numbers everyone has been quoting).
Restated for the record:

1. `sc = d3_compare_gifs.scene_real(record, win=win)` — unmodified, same seeds.
2. Observations: `P = sc['P'].astype(float32)`, `z = sc['z'].astype(float32)`; if
   `len(z) > cd.M_OBS (=400)`, subsample with `np.random.default_rng(0).choice(len(z),
   400, replace=False)`. Times shifted so `t0 = P[:,2].min()` → `Ps[:,2] -= t0`.
3. `Bmap = cd._bmap_from_obs(Ps, z)`; model = `CondSetDenoiser` with
   `results/direction3/beamform_cond_diffusion.pt`, stats from `cd.PRIOR_CKPT`,
   `betas = cd.cosine_betas(cd.T)`, `abar = cumprod(1-betas)`.
4. **Diffusion seeds `SEEDS = (0,1,2,3,4)`**; for each seed `sd`:
   `X = cd.sample(model, Bmap, abar, betas, n=4, seed=int(sd))` — **n = 4 posterior
   samples**; each sample `b` is polished by
   `xb = cd.polish_atoms(X[b], stats, Pt, zt, depth, iters=80)` — **80 Gauss-Newton
   iterations**.
5. **Primary field per seed = the BMA over the 4 polished samples of
   `cd.atoms_to_field(xb, stats, Ps, z, depth, target)`**, with
   `target = {'gx': sc['gxq'], 'gy': sc['gyq'], 't': lead_ts[0], 'gt': sc['maps_gt'][0],
   'Hs': sc['Hs']}` and `lead_ts[0] = float(sc['Pq'][0,2]) + 0.0 - t0`.
   Samples returning `None` are dropped from the BMA (verbatim behaviour).
6. **Score (lead-0 nowcast, pooled-GT normalisation — identical to the hand arm and to the
   escape controls):** `sg = std(maps_gt[isfinite(maps_gt)])` over **all leads pooled**;
   mask `m0 = isfinite(gt[0]) & isfinite(bma)`;
   `nrmse = sqrt(mean((bma[m0]-gt[0][m0])**2)) / sg`. Finite-mask ≥ 8 px.
7. **Cited value per window = `median` of the 5 seed nRMSEs** (`learned_now` in
   `wp_real.json`). Per-seed values and the seed sd are recorded but are **not** endpoints.

No knob of this arm is tuned, swept, or selected for this run. `n_eff`, `bf_peak`, and the
per-seed spread are recorded as diagnostics only and cannot satisfy or fail any bar.

## 2. Target set — the same 18 windows, unchanged

15 held-out (`arena='heldout'`) + 3 design (`arena='design'`), exactly the list frozen in
`PREREG.md` §1 and executed in `escapes.json`:

BS11b_w4, BS11c_w3, BS11c_w4, BS13a_w0, BS13a_w1, BS13a_w2, BS13a_w4, BS13a_w5,
BS13c_w2, BS13c_w3, LJ18n_w0, LJ18n_w1, LJ18n_w3, LJ18n_w4, LJ18n_w5,
AA14_w0, AA15a_w3, LJ18g_w3.

**Baseline hand nRMSE** per window is taken from `escapes.json:rows[*].baseline_recheck`,
which already passed the bit-identity control against `harvest_heldout.json` /
`hand_amp.json` at `0.000e+00`. No baseline is recomputed here.

Declared property of the target set (stated now, not after): the held-out BS windows have
very small query grids (`npx` 16–25 finite GT pixels; LJ18n 121; design 374–4788). This is
a property of the frozen harvest that the hand baselines and all six escapes were already
scored on; the learned arm is scored on **the identical mask**, so the comparison is
paired. It does mean per-window nRMSE on BS windows is a low-count statistic — a caveat on
the *windows*, applied equally to both arms, not a selection rule.

## 3. The two counts — frozen, **never merged**

For a learned reading `l` on a window with baseline `h` (`PREREG.md` §1.1, verbatim):

- **(a) LARGE REDUCTION**: `h - l ≥ 0.5`.
- **(b) BELOW-LINE**: `l < 1.0`.

Reported as two independent counts out of 18. **They are not summed, averaged,
intersected, or collapsed into a single number.** (This is the structural fix for the
historical 3/3-vs-2/3 error.)

## 4. Decision rule — P2's bar, already numerically specified, applied per criterion

`C_best(criterion)` = max count over the **six GT-free single-config escape readings**
{E1-median, E2-peak2, E2-half, E2-double, E3-half, E3-double}, **measured**:
`C_best(a) = 6/18`, `C_best(b) = 11/18` (both attained by E2-double).

| criterion | C_best | learned must reach (`≥ 1.5 × C_best`) |
|---|---|---|
| (a) Δ ≥ 0.5 | 6/18 | **≥ 9/18** |
| (b) < 1.0 | 11/18 | **≥ 17/18** |

- Bar met on a criterion ⟹ that criterion supports "the learned repair is not merely
  re-initialization."
- Bar missed on a criterion ⟹ for that criterion, **Ch.7's interlock is a paragraph about
  initialization robustness**, in those words.
- The max-over-six comparator is **deliberately generous to the control** (the control may
  be chosen with hindsight, the learned arm may not). That asymmetry must be stated
  wherever the comparison is reported.
- Oracle-flavored readings (E1-best, `*-bestvar`, ANY-bestvar) may **not** satisfy or fail
  the bar in either direction.
- **Expectation written before running** (from the design-3 evidence in `RESULT.md` §3,
  where E2-double is numerically better than the learned arm on all three windows):
  the bar is expected to **FAIL on both criteria**. Whatever comes out is reported.

## 5. Validity control — design-3 must reproduce `wp_real.json` bit-identically

The three design collapse windows are **recomputed live**, not read from disk, and must
reproduce the stored `wp_real.json` `learned_now` values:

| window | stored `learned_now` |
|---|---|
| AA14_w0 | 1.0708955847548527 |
| AA15a_w3 | 0.7913120579697095 |
| LJ18g_w3 | 0.9614859266044530 |

**Tolerance: `|Δ| ≤ 1e-9` on `learned_now`** (the whole path is deterministic: fixed
`rng(0)` subsample, fixed diffusion seeds, fixed polish iteration count). The per-seed
vectors `learned_now_seeds` are compared at the same tolerance.

- **Any mismatch > 1e-9 invalidates the held-out readings of this run** — they would have
  been produced by a different arm than the one whose numbers are in circulation, and the
  run must be re-executed or the discrepancy explained before any count is reported.
- A PASS additionally licenses the **cache reuse** declared in
  `tmp/blind_auto_real_2026-07-26/PREREG.md` (Task B): it establishes that this session's
  learned-arm code path reproduces `wp_real.json` exactly, so `wp_real.json` learned fields
  may be reused there instead of recomputed.

## 6. Reporting requirements (frozen)

`RESULT.md` (appended section, or a sibling `RESULT_GPU.md`) must contain:

1. the bit-identity outcome (PASS/FAIL, max |Δ|);
2. a per-window table: `baseline hand | learned (median-of-5) | E2-double`, all 18 rows,
   held-out and design separated by arena, plus the learned per-seed sd;
3. the two counts (a) and (b) for the learned arm, over 18, **never merged**;
4. the P2-bar verdict per criterion, with the pre-committed consequence sentence spelled
   out if a bar is missed;
5. a head-to-head table restricted to the **5 windows the frozen τ = 1.5 detector fires on**
   (AA14_w0, LJ18g_w3, LJ18n_w0, LJ18n_w1, LJ18n_w4) — the only windows where a fallback
   would ever run — learned vs E2-double, explicitly labelled *operational subset,
   post-hoc-defined-by-deployment* as in `RESULT.md` §4;
6. no oracle-flavored number used in any verdict.

**Invalidation conditions:** design-3 bit-identity mismatch > 1e-9; any change to the
learned-arm protocol of §1; any window added or dropped; any criterion or bar altered after
this file was written; any per-window learned value substituted from cache instead of
computed (all 18 are computed live here).
