# REGEN_SPEC — WaveSim dataset regeneration at Nyquist-adequate mesh resolution

**Date**: 2026-07-23. **Author**: design/audit agent (fable). **Status**: design complete,
ETKF-baseline mechanism partially (not fully) resolved — see Sec 4. **Do not execute the
full 100-episode regen from this document alone until Sec 4's open item is read.**

**Authority chain**: `wavesplat/tmp/GROUNDUP_fable_C_premises_2026-07-23.md` (premise audit
that triggered this work) → this document (design/spec) → execution agent runs it.

Every claim below cites the script/command that produced it. Scripts referenced as
`tmp/regen_*.py` were written during this audit and left in `wavesplat/tmp/` for
inspection; none of them touched `data/`.

---

## 0. Scope

**In scope for regeneration** (the "primary paired-replacement set"): the 100 files
matching `configs/default.yaml`'s `dataset.scenarios` grid —

    {BF2:1, BF3:1, BF3:2, BF3:5, BF4:1, BF4:5, BF4:15, BF4:30, BF5:15, BF5:30}
    × seeds {0,1,2,3,4} × raydrop {uniform, slope}  =  10 × 5 × 2 = 100 files

This is exactly the set the premise audit's "0 ADEQUATE / 52 DEGRADED / 50 INADEQUATE"
finding covers modulo two stray extra files (below), and exactly what CLAUDE.md's paper
pipeline and `scripts/run_p1_comparison.py` draw from.

**Explicitly OUT of scope** — do not touch, but tracked here so nothing is silently
dropped:
- `fixedT` raydrop episodes (85 files) — a third, separate raydrop model (F2/H3-OD
  research track), not one of the two models this task's confound ruling covers, and not
  the paper's raydrop-sensitivity comparison. Regenerating it would double the compute
  budget for no benefit named in the task brief. **Recommendation**: if the broader PhD
  CRB/envelope work depends on `fixedT` files at the coarse mesh, that is a materially
  identical problem to this one and should get its own, later regen pass reusing this same
  spec's mesh choice — flagged, not actioned here.
- `_dir180` files (30) and `BF6`/`BF7` files (8) — both already explicitly out of the
  paper's declared scope per the premise audit (§4, §1a).
- `episode_BF4_20comp_seed7_uniform.h5` and `episode_BF4_30comp_seed42_uniform.h5` — two
  stray ad hoc files not part of the `default.yaml` scenario grid (confirmed: no matching
  scenario/seed combination). Leave as-is; not part of any headline table found via
  `grep -rn "20comp\|seed42" scripts/*.py`.

**Output location**: `wavesplat/data_finemesh_2026-07-23/` (primary set) and
`wavesplat/data_finemesh_2026-07-23_pk/` (peak-locked companion, Sec 3d). Both are new
sibling directories to `data/` — `data/` is never opened for writing by the driver script
below (it refuses by path-prefix check).

---

## 1. Generator audit — defect table

Full wavesim generator chain read: `pipeline.py`, `wave_generator.py`, `gt_generator.py`,
`hos_simulator.py`, `lidar_simulator.py`, `lidar_gpu.py`, `raydrop_uniform.py`,
`raydrop_slope.py`, `raydrop_threshold.py`, `noise_model.py`, `config.py`,
`data_writer.py`, `mesh_adequacy.py`, `generate_dataset.py`, plus the full
`wavesim/tests/` suite (75 tests, all read; all pass — `pytest wavesim/tests/ -q` →
`75 passed in 40.72s`).

| # | Finding | Severity | Check performed | Regen decision |
|---|---|---|---|---|
| G1 | Raycast mesh sub-Nyquist for all 102 in-scope BF2–5 episodes at shipped dx=6.15m/dy=6.90m | **the reason for this regen** | `mesh_adequacy.check()` run directly against the real `config/wave_components` of every shipped in-scope file (`tmp/regen_mesh_search.py`); matches premise audit | **FIX** — Sec 2 |
| G2 | `noise_model.add_noise()`: the variable used to scale angular-noise lateral displacement by "range" is actually `norm(hit_points - noisy)` computed *after* range-noise was already applied — i.e. it equals the range-noise perturbation magnitude (~0.02 m) plus a constant `+1.0`, **not** the true sensor-to-point range (which spans 1.5–250 m). Reproduced directly (see below): `dist` ≈ 1.00–1.06 m at every range. Intended-vs-actual: at 250 m the angular contribution is ~0.0009 m instead of the ~0.22 m it would be if `dist` were the real range — a ~250× under-simulation of angular noise at max range, present at **every** range past a few meters. `hit_dist` is available in `pipeline.py` at the call site but never passed in. | MATERIAL, but **orthogonal to mesh** | Executed the exact 5 lines with synthetic points at range 10–250 m; `dist` came back 1.00–1.06 regardless of true range (script in this session, reproduced below) | **DO NOT FIX** in this regen — would confound "why did the numbers change" (Sec 3). Track separately. Note: `docs/WaveSplat_paper_review_2026-04-22.md` §5.5 already assumes "~5mm" angular-noise contribution when deriving τ_R=0.05m — the *same order of magnitude* as the buggy behavior (~1.2mm combined for the two lateral axes), not the ~110mm the correct range-scaled behavior would give at d_ref=125m. This suggests the bug (or an equivalent assumption) may already be baked into an existing paper derivation; fixing it later needs its own isolated ablation, not a silent change here. |
| G3 | `generate_dataset.py`'s worker (`_run_one`) calls `run_episode()` **without** `use_gpu_raycast` → defaults to `False` → CPU raycast path | MATERIAL (operational) | Read `_run_one()`; CPU raycast measured at 13.9 s/frame for a 180×150 mesh per `lidar_gpu.py`'s own docstring, vs GPU-measured 0.974–1.235 s/frame (this session, below) — ~14× slower if used naively | **MUST NOT use `generate_dataset.py` as-is.** Spec ships a corrected driver (Sec 5) that passes `use_gpu_raycast=True`. |
| G4 | `interpolate_normals()` (used for incidence angle → raydrop probability → confidence) sources normals from the **GT grid** (analytic, nx=80/ny=60 default), not from the raycast-mesh facet the ray actually hit | informational / mitigating | Read `pipeline.py:172-180`: `hit_normals = interpolate_normals(hit_pts[hit_idx], X_gt, Y_gt, NX_gt, NY_gt, NZ_gt)` | No action. Means refining the raycast mesh mainly corrects hit **position** (x,y,z); the raydrop/confidence **decision** was already running on a materially better-resolved analytic field. This decouples defects (b)/(c) from the mesh fix more than a naive reading suggests — see Sec 3. |
| G5 | GT grid (default 80×60, dx≈3.04m/dy≈3.39m — itself sub-Nyquist for BF2 1-comp by a small margin) does **not** bias nRMSE/RMSE scoring | **CLEARED** | Traced `gt_generator.evaluate_field()` → `single_gerstner()`: analytic sum, no mesh interpolation involved. Traced `wavesplat/run.py`, `wavesplat/evaluate.py`: prediction is evaluated at `loader.gt_grid_x/gt_grid_y` — the *same* discrete points the GT field is stored at. Both sides of every metric sit on the identical grid; no facet-type artifact possible. | No action. Residual: `Directional_R2`'s 2D-FFT-based spectral comparison could in principle alias short components at the GT grid's own Nyquist limit, but since pred and GT are sampled identically this does not bias the comparison if the estimator is honest — COSMETIC, not actioned. |
| G6 | RNG desync: wave-component generation is reproducible per-seed (own internally-seeded RNG in `jonswap_components`, independent of mesh/raydrop), but the **per-frame LiDAR sampling** (raydrop Bernoulli draws + sensor noise) consumes a *different* number of random draws depending on how many rays hit — which differs between mesh resolutions and between raydrop models. So "same seed" only guarantees identical wave components, not identical per-frame point realizations, beyond frame 0, between old/new files or between uniform/slope variants of the same seed. | expected behavior, not a bug — but must be stated | Traced `pipeline.py:70` (`rng = np.random.default_rng(seed)`, consumed per-frame by `apply_uniform`/`apply_slope`/`add_noise`) vs `wave_generator.py:158` (jonswap's own internal seeded rng). Empirically verified: |k| (hence λ) identical across every seed/model combination sharing (BF,ncomp) — `tmp/regen_mesh_calc.py` output. | Document only (Sec 5's verification checklist asserts component-array equality, not point-level equality). |
| G7 | ETKF baseline's own analysis grid (`etkf_gx/gy`, hardcoded 40×30 in `run_p1_comparison.py` and the mesh pilot) is decoupled from `cfg.mesh` — refining the raycast mesh does not refine ETKF's internal state grid | MATERIAL | Read `scripts/run_p1_comparison.py:82-87`, `tmp/mesh_headline_subset.py:71-73` | See Sec 4 (baseline retuning ruling — added mid-task per coordinator request) |
| G8 | `run_p1_comparison.py`/the mesh pilot use ETKF `n_ensemble=50`, but the repo's own tuning sweep (`results/etkf_tuning_sweep.json`, 24 configs × 12 episodes) found `n_ensemble=100` (same grid/inflation) materially better: mean nRMSE 0.358 vs 0.416 — a **pre-existing staleness bug, unrelated to mesh** | MATERIAL, pre-existing | Read `results/etkf_tuning_sweep.json` directly, computed matched-pair deltas (this session) | Flag; folded into the Sec 4 retuning recommendation. |
| G9 | Dispersion relation, Q-constraint enforcement, HOS bound-harmonic accounting, seeding determinism | **CLEARED** | Full `wavesim/tests/` suite passes (75/75), including `test_pipeline_tank_depth_dispersion`, `test_hos_harmonics_also_count_against_the_variance_budget`, `test_hos_order_accounts_for_bound_harmonics` | No action. |
| G10 | `n_components=1` places the component at 1.75·ω_p (a spectral-flank wave), not at the peak — task brief's defect (d) | MATERIAL, already known/instrumented | Generator's own runtime warning; `test_peak_locked_puts_the_lone_component_at_the_peak` regression test exists; confirmed shipped files carry no `_pk` tag (`ls data/*.h5 | grep -c _pk` → 0) | See Sec 3(d) confound ruling. |

**G2 reproduction** (run this session):
```python
rng = np.random.default_rng(0)
hit_points = np.array([[10,0,0.1],[250,0,0.1],[125,0,0.1]], dtype=float)
ray_dirs = np.array([[1,0,-0.1]]*3); ray_dirs /= np.linalg.norm(ray_dirs,axis=1,keepdims=True)
delta_range = rng.normal(0.0, 0.02, 3)
noisy = hit_points + delta_range[:,None]*ray_dirs
dist = np.linalg.norm(hit_points - noisy, axis=1) + 1.0
# dist = [1.0025, 1.0026, 1.0128]  <- should have been [10, 250, 125]
```

---

## 2. Mesh-resolution computation

**Method**: read `config/wave_components` directly from the shipped `.h5` files for all
10 in-scope `(BF, n_components)` families (verified λ depends only on `(BF, n_components)`
— identical across seed and raydrop model, since only phase/direction are randomized, not
|k|). Ran `wavesim.mesh_adequacy.check()` — the actual repo function, not a hand
re-derivation — across a sweep of candidate mesh spacings, for every family
(`tmp/regen_mesh_search.py`, `tmp/regen_mesh_table.py`).

No HOS involved: confirmed shipped files' `config` group carries no HOS-related attrs and
`configs/default.yaml`/`server.yaml` never set `hos.enabled`; `hos_order=None` throughout.

**Result** — verdict flips from INADEQUATE (worst case: BF2, 1-comp, λ=8.157m) to
ADEQUATE-for-all exactly at mesh spacing d ≤ λ_min/6 = 8.157/6 = **1.3595 m**. This is a
sharp threshold because a 1-component sea has a binary variance-under fraction (100% or
0%) — the ADEQUATE bar for a single-component episode is exactly "6 pts/λ at the peak."

| BF | ncomp | λ_peak (m) | λ_min (m) | **40×30 (shipped)** | pts/λ | var_under | **180×150** | **200×167** |
|---|---|---|---|---|---|---|---|---|
| BF2 | 1 | 8.16 | 8.16 | INADEQUATE | 1.18 | 100% | ADEQUATE | ADEQUATE |
| BF3 | 1 | 15.42 | 15.42 | INADEQUATE | 2.24 | 100% | ADEQUATE | ADEQUATE |
| BF3 | 2 | 37.32 | 8.37 | INADEQUATE | 1.21 | 100% | ADEQUATE | ADEQUATE |
| BF3 | 5 | 30.23 | 6.25 | INADEQUATE | 0.91 | 77.3% | ADEQUATE | ADEQUATE |
| BF4 | 1 | 24.98 | 24.98 | INADEQUATE | 3.62 | 100% | ADEQUATE | ADEQUATE |
| BF4 | 5 | 48.96 | 10.12 | DEGRADED | 1.47 | 21.2% | ADEQUATE | ADEQUATE |
| BF4 | 15 | 65.18 | 17.63 | DEGRADED | 2.56 | 21.6% | ADEQUATE | ADEQUATE |
| BF4 | 30 | 70.49 | 21.76 | DEGRADED | 3.16 | 20.6% | ADEQUATE | ADEQUATE |
| BF5 | 15 | 107.15 | 29.14 | DEGRADED | 4.22 | 9.3% | ADEQUATE | ADEQUATE |
| BF5 | 30 | 115.54 | 35.97 | DEGRADED | 5.22 | 7.2% | ADEQUATE | ADEQUATE |

**Two candidate resolutions, both give ADEQUATE-for-all-10-families**:

| Mesh | dx | dy | max(dx,dy) | margin below 1.3595m required | GPU cost (measured) | Full 100-episode wall time |
|---|---|---|---|---|---|---|
| 180×150 | 1.3408 | 1.3423 | 1.3423 | **1.3%** (thin) | 0.974 s/frame | ~8.1 h |
| **200×167** | 1.2060 | 1.2048 | 1.2060 | **11.3%** | 1.235 s/frame | ~10.3 h |

**Recommendation: 200×167.** The 180×150 choice (already in use by the in-flight pilot,
`test_fine_mesh_resolves_bf2_1comp`, and independently by this audit's own search) is
exactly correct at the bare-minimum threshold but leaves only ~1.3% margin on a hard
binary criterion computed from one specific random draw (BF2 1-comp, seed 0) — reasonable
for a quick pilot, less reassuring as the number the whole regenerated dataset stands on.
The extra cost for 200×167 is small: GPU-raycast wall time is overhead-dominated, not
triangle-count-dominated (measured 40×30 GPU: 0.769 s/frame vs 180×150 GPU: 0.974 s/frame
— only 1.27× despite 22× more triangles — because `evaluate_field`/`interpolate_normals`
per-frame cost dominates, not the raycast itself), so going from 180×150 to 200×167 only
costs ~27% more (2.2 extra hours) for a much safer margin. **If wall-clock is the binding
constraint, 180×150 remains a defensible fallback** — it is independently validated three
ways (this audit's sweep, `test_fine_mesh_resolves_bf2_1comp`, and the pilot's own choice)
and every in-scope family clears it, just not by much.

Neither choice leaves any in-scope episode DEGRADED or INADEQUATE — "best achievable tier"
discussion is moot here; full ADEQUATE is achieved and affordable.

---

## 3. Confound ruling — what changes, what doesn't, and in what order

**(a) Raycast mesh coarseness → CHANGE.** This is the entire point of the regeneration
(Sec 2).

**(b) `slope` raydrop specular-degenerate branch → HOLD FIXED (do not patch).**
Regenerate `*_slope.h5` files with the same (unrepaired) `raydrop_slope.py`. Reasoning:
mesh refinement cannot change *whether* the specular branch fires — `specular_alignment()`
consumes ray direction and the GT-grid-interpolated normal (G4), neither of which depends
on `cfg.mesh.nx/ny`. The branch will still be 0-of-N regardless of mesh resolution, so
there is no scientific reason to touch it in this pass, and touching it simultaneously
with the mesh would make it impossible to attribute post-regen deltas to mesh vs.
raydrop-physics. A dedicated future experiment should evaluate a corrected specular model
against both the old and new mesh, separately.

**(c) `uniform` raydrop's `R_fresnel(θ)·cos(θ)/d²` category error → HOLD FIXED.** Same
attribution logic. Additionally mitigated by G4: the incidence angle this formula consumes
comes from the well-resolved GT-grid interpolation, not the raycast facet, so mesh
refinement's effect on this defect's *behavior* is small (only via slightly more accurate
`hit_dist`) — the two defects are largely orthogonal, which is itself a useful thing to be
able to say once regenerated numbers are compared.

**(d) 1-component spectral-flank placement → HOLD FIXED for the primary set, but generate
an explicit companion.** Holding it fixed (not passing `peak_locked=True`) is required for
"paired, same seed ⇒ same wave components" comparability with the shipped set — flipping
it now would confound mesh-effect with placement-effect for the exact episodes (all the
1-comp scenarios) that are also the mesh's worst case, which is precisely the confound the
task brief warns about. **Separately**, because the task brief is right that this makes
BF2/BF3/BF4 1-comp the *least* representative "worst case" already in the paper, generate
a second, clearly-tagged, additional set: the three 1-comp scenarios × 5 seeds ×
{uniform,slope} = 30 files with `peak_locked=True` (files get pipeline.py's own `_pk`
suffix automatically). This is **additive**, not a replacement, and lets a downstream
analysis do a clean three-way split: shipped-flank vs. regen-flank (isolates mesh effect)
vs. regen-peak-locked (isolates placement effect, at the new mesh).

**Run ordering**:

1. *(already in flight, do not restart)* 12-episode coarse/fine pilot
   (`tmp/mesh_headline_subset.py`) — 4/12 done as of this writing
   (BF2_1c_s0_uniform, BF3_1c_s0_uniform, BF3_2c_s0_uniform, BF3_2c_s0_slope). Let it
   finish; it is the cheapest possible go/no-go signal and already surfaced the Sec 4
   anomaly.
2. **Baseline retuning diagnostic** (Sec 4) — must run before any ETKF number from
   fine-mesh data is trusted for a headline claim. Partially done this session; one item
   left open (Sec 4).
3. Primary regen — 100 files at 200×167 (or 180×150 if wall-clock-constrained), `tmp/regen_run_finemesh.py`.
4. Peak-locked companion — 30 files, same script with `--peak_locked`.
5. Verification pass (Sec 6) against every file from steps 3–4.
6. Full comparison re-run using the Sec 4-validated baseline configuration (Sec 7).

---

## 4. Baseline re-tuning ruling (added mid-task at coordinator's request)

**Trigger**: pilot episode `BF3_1c_s0_uniform` showed ETKF getting **worse** on the
refined mesh (0.352→0.481, +37%) while WaveSplat improved (0.217→0.095, −56%) — a
direction flip that, if the baseline's hyperparameters were simply stale/mistuned for the
new data, would make any resulting margin claim indefensible.

### 4.1 Must baselines be re-tuned before the final comparison? — **Yes.**

Two independent lines of evidence, both found by reading code + running small checks, not
assumed:

- **Pre-existing staleness, unrelated to mesh (G8)**: the repo's own 24-config tuning
  sweep (`results/etkf_tuning_sweep.json`) already shows the *currently used* ETKF
  configuration (`grid=40×30, Ne=50, mi=1.0, ai=0.005` — what `run_p1_comparison.py` and
  the mesh pilot both hardcode) is **not** that sweep's own winner. At matched grid/
  inflation, `Ne=100` beats `Ne=50` by a wide margin on the *old* data (mean nRMSE 0.358 vs
  0.416 across the 12-episode set). The "tuned" baseline the paper cites was never fully
  applied to the comparison scripts.
- **New evidence this session**: re-running a reduced version of the same sweep
  (`grid`×`Ne`×`mult_inflation` ∈ {40×30, 80×60, 20×15}×{50,100}×{1.0,1.02}, 6 configs) on
  the two already-generated fine-mesh pilot pairs (`tmp/etkf_retune_probe.py`,
  results in `tmp/etkf_retune_probe_results.json`) shows the ranking **is** sensitive to
  mesh resolution: for BF2_1c, `20×15` (coarser than the shipped default) gives the best
  fine-mesh result (0.355, a 36% improvement over coarse), edging out the old winner
  `40×30/Ne100` (0.340). Reusing old hyperparameters unexamined is not safe.

**Concrete protocol** (reuses the existing procedure, does not invent a new one): extend
`scripts/run_etkf_tuning_sweep.py`'s `SWEEP` dict to also include `obs_noise` (currently
**not swept at all** — a gap in the existing tuning procedure itself, see 4.2), point it at
a batch of regenerated fine-mesh episodes (the 12-episode pilot set once complete is
sufficient), and use *that* sweep's winner for the full 100-episode headline comparison —
exactly the same way the original sweep selected the "medium" config for the old data.
GPR needs a lighter touch: `GPRBaseline`/`GPRDispersion` re-fit their kernel
hyperparameters via sklearn's own MLE optimizer every frame (`n_restarts_optimizer=2`,
confirmed by reading `gpr_baseline.py`), so the `length_scale=20.0` constructor argument is
only an initial guess, not a frozen tuned value — lower risk, but still worth one spot-run
on regenerated data before citing.

### 4.2 Is there a plausible mechanism? — **Two found, but neither fully explains it; this is the open item.**

**Mechanism A (confirmed, generic, but insufficient alone): analysis-grid resolution vs.
fixed ensemble/observation budget.** State dimension is `2·nx·ny` (η and ∂η/∂t); with
`Ne=50–100` members and `max_obs=40` (thinned), an 80×60 grid (9600-dim state) is more
poorly conditioned than 40×30 (2400-dim state) for the same information budget — a
standard EnKF property, confirmed generic in the *old* sweep data (40×30 beats 80×60 at
**every** matched `(Ne, mi, ai)` pair, 6/6 comparisons, even before any mesh change).
**But this does not explain BF3_1c**: I tested `{40×30, 80×60, 20×15}` × `{Ne=50,100}` ×
`{mi=1.0,1.02}` (6 configs) on BF3_1c's coarse/fine pair and **every single one** still
regressed on the fine mesh, from +23% to +44%. Grid/Ne/inflation retuning alone does not
rescue this scenario.

**Mechanism B (confirmed real and substantial via a follow-up probe, not yet fully
resolving): fixed scalar observation noise `obs_noise=0.05` (R=5cm) is not adapted to the
actual noise in the observation stream, and is doing real work unrelated to the sensor's
literal spec.** `ETKFGrid.update()` never reads `lidar_confidence`/incidence angle — every
point gets the same fixed R regardless of geometry (`etkf_grid.py`, confirmed by reading
`_observation_operator`/`update()`). I swept `obs_noise` on BF3_1c's coarse/fine pair
(`tmp/etkf_obsnoise_probe.py`, not part of the repo's existing sweep — **this parameter is
not swept by `run_etkf_tuning_sweep.py` at all**, a gap in the existing tuning procedure):

| obs_noise | coarse nRMSE | fine nRMSE | Δ |
|---|---|---|---|
| 0.02 (literal sensor spec) | 0.441 | 0.604 | **+36.9%** (worse) |
| 0.05 (shipped default) | 0.347 | 0.447 | +29.0% |
| 0.10 | 0.292 | 0.356 | +21.7% |
| 0.20 | 0.270 | 0.299 | **+10.9%** (best tested) |

Monotonic and substantial: quadrupling R from the shipped 0.05 to 0.20 cuts the regression
from +29% to +11%, while moving R *toward* the sensor's advertised 0.02m noise floor makes
it worse. This is consistent with `obs_noise=0.05` doing legitimate "representativeness
error" work (a single LiDAR point representing a ~6–7m grid cell carries error beyond its
own sensor noise) that has nothing to do with the sensor spec — but I did not sweep past
0.20, so I do not know whether a larger R fully closes the gap or the regression asymptotes
to a real residual. `max_obs=74` (disabling thinning, since BF3_1c averages ~74 pts/frame)
made no material difference (+26.9%, close to shipped).

**Net assessment**: the BF3_1c regression is **not** simply "we forgot to retune" — a
reasonably thorough retune sweep (6 grid/Ne/inflation configs, all failing) plus a
follow-up obs_noise sweep (real, dose-dependent, but not fully closing the gap even at 4×
the shipped R) leaves a residual, unexplained regression. **BF2_1c shows no such
pathology** — fine helps at the shipped config and stays at least neutral across every
config tried. Two structurally similar 1-component "flank wave" episodes (BF2 T=2.29s vs
BF3 T=3.14s) behaving oppositely under identical treatment is the single most interesting
open thread from this audit and is **not resolved here**.

### 4.3 Should the pilot be re-run with retuned baselines before drawing conclusions? — **Yes, for anything you plan to cite; the existing 2-episode evidence stands as a narrower claim.**

Do not report a "coarse→fine ETKF margin" number for BF3_1c-type episodes using the
shipped hyperparameters — it is now demonstrated (not merely suspected) that hyperparameter
choice materially moves that number and that no configuration tried in a real search
recovers parity with the coarse-mesh baseline. What the existing (un-retuned) 2-episode
data *does* support, narrowly: "the ranking (WaveSplat < ETKF) is not an artifact of the
mesh fix for BF2_1c under any tested ETKF configuration; for BF3_1c the picture is
murkier and requires the obs_noise-inclusive sweep to resolve before citing." Recommend
inserting the extended sweep (Sec 4.1's protocol, obs_noise included) as a required step
before the full 100-episode ETKF headline is computed — this does **not** block the mesh
regeneration itself (WaveSplat's own numbers, and the "paper's absolute numbers are
mesh-artifact-inflated" finding, do not depend on resolving this).

**A lighter-touch note on WaveSplat's own hyperparameters** (`configs/estimation.yaml`):
these were also implicitly set against the old coarse-mesh data and have not been
independently re-validated either. The available evidence (improved on both tested pilot
pairs, no direction flips observed) is reassuring but not a full retune. Given the
asymmetric risk (a method that only improves needs less urgent re-justification than one
that flips direction), I recommend a lighter sanity pass — confirm P0/P1 both still improve
across the full 12-episode pilot once it finishes — rather than a full re-tune, unless that
pilot itself turns up a WaveSplat regression somewhere.

---

## 5. Executable regeneration spec

**Driver script** (written this session, not yet run at scale):
`wavesplat/tmp/regen_run_finemesh.py`. Resumable (JSON manifest,
`<output_dir>/_regen_manifest.json`, skips anything already marked `ok`), uses
`use_gpu_raycast=True` (fixes G3), sequential (single GPU is the shared bottleneck —
running `generate_dataset.py`'s multiprocessing `Pool` with GPU raycast is untested and
risks CUDA-context contention for no benefit, since the GPU work cannot be parallelized
across processes anyway), catches exceptions per-episode so one failure does not kill the
run, refuses by path-prefix check to write into `data/`.

**Parameters** (all confirmed against `configs/default.yaml`, unchanged from shipped):
`n_pts=50`, `duration=30.0s`, `dt=0.1s` (301 frames), `depth=50.0m`, `seeds=[0,1,2,3,4]`,
`raydrop_models=[uniform, slope]`, scenario grid as listed in Sec 0. Mesh:
`nx=200, ny=167` (default in the script; pass `--nx 180 --ny 150` for the thinner-margin
alternative).

```bash
# Step 3 — primary set (~10.3h at 200x167; ~8.1h at 180x150). Safe to Ctrl-C and re-run.
python3 tmp/regen_run_finemesh.py --output_dir data_finemesh_2026-07-23

# Step 4 — peak-locked companion (30 files, ~40min), run after Step 3 completes
python3 tmp/regen_run_finemesh.py --peak_locked --output_dir data_finemesh_2026-07-23_pk

# Dry run first to confirm the 100-job plan before committing:
python3 tmp/regen_run_finemesh.py --output_dir data_finemesh_2026-07-23 --dry-run
```

**Disk budget**: shipped in-scope (100) files total **1.14 GB** (measured:
`du`/file-size sum over exactly this file set — not the 52GB figure, which includes
`fixedT`/`dir180`/BF6-7). Regenerated files should be nearly identical in size (LiDAR
point arrays are capped at ~`n_pts`; GT field grid is unchanged at 80×60 — mesh resolution
only affects internal raycasting, not what gets written). **Budget impact: negligible**
against 852 GB free (~0.15%), regardless of which mesh choice is used.

**Runtime budget**: 200×167 ≈ 10.3h primary + ~40min companion ≈ **11h total**, single
GPU, single process, unattended-safe (resumable). 180×150 ≈ 8.1h + ~30min ≈ **8.6h**.
Recommend starting the 200×167 run overnight/unattended; if it needs to fit a shorter
window, fall back to 180×150 (Sec 2 already establishes this is not scientifically
inferior, just thinner-margin).

---

## 6. Verification checklist (per regenerated file)

Run after generation completes (a follow-up script should assert all of these; not yet
written — this is the checklist, the execution agent should implement it as the last step
of Sec 5's run, or as a separate pass):

1. **File opens, non-empty**: `h5py.File(fp)['frames/lidar_points'].shape[0] > 0`.
2. **Mesh verdict is ADEQUATE**: `f['config'].attrs['mesh_verdict'] == 'ADEQUATE'` for
   every file (should be true by construction at 200×167/180×150 per Sec 2 — a violation
   here means the mesh choice or the scenario grid has drifted from what was verified).
3. **Component match against the shipped file**: for the *primary* (non-peak-locked) set,
   `np.allclose(new['config/wave_components'][:], shipped['config/wave_components'][:])`
   for the same `(BF, ncomp, seed)` — must be bit-close (rtol=1e-10 or better; G6 predicts
   *exact* equality since `jonswap_components` uses its own independently-seeded RNG,
   untouched by mesh/raydrop choice — treat any mismatch as a real regression, not
   expected noise, and stop to investigate before proceeding).
4. **Point counts sane**: mean points/frame within the historical range seen in the
   shipped file for the same scenario (`n_returns_raw` and `lidar_valid_mask.sum(axis=1)`,
   should track close to `n_pts=50` on average, per-frame variance expected — see G6).
5. **No NaNs where valid**: `np.isnan(pts[valid_mask]).any() == False`,
   `np.isnan(gt_wave_field).any() == False` (GT field should never be NaN; it's an
   analytic evaluation).
6. **Frame count**: exactly 301 frames (`duration=30.0, dt=0.1`) unless a frame was
   legitimately skipped for zero hits (`pipeline.py` skips + prints — should not happen at
   any in-scope sea state/component count; flag if it does).
7. **mesh_dx/mesh_dy attrs present**: `config.attrs` should carry the full `mesh_info`
   dict (`mesh_dx, mesh_dy, lambda_peak, lambda_min, pts_per_lambda_peak, pts_per_lambda,
   var_under_resolved, mesh_verdict, mesh_adequate, mesh_above_nyquist`) — the shipped
   files lack this (predate the feature); regenerated files having it is itself a useful,
   checkable provenance improvement flagged by the premise audit.
8. **Peak-locked companion only**: assert the single component's period equals `Tp` (not
   `Tp/1.75`) via `2*np.pi/wave_components[0,1]`.

The driver script (Sec 5) already performs a cheap subset of this (NaN check, frame count,
mesh verdict) inline per-episode and records it in the manifest; items 3 and 8 require a
separate pass comparing against `data/`.

---

## 7. Ranked re-run list — what's threatened, what survives

Ranked by how directly the mesh-Nyquist finding threatens the conclusion, reasoning given
for each:

1. **[HIGH — re-run required before citing]** All absolute nRMSE/Hs-error/etc. numbers in
   `paper/wavesplat.tex` (Tables `tab:fivemetric`, `tab:bybf`, `tab:raydrop`, the abstract's
   "30.1%", the line-1555 worst-case claim). Already measured directly inflated by 28.5%
   (WS) / 25.7% (ETKF) on the one episode fully re-derived. **Full 100-episode re-run is
   the point of this regen.**

2. **[HIGH — re-run required, mechanism partially open]** `scripts/run_etkf_tuning_sweep.py`
   and any ETKF number derived from it. Cannot be trusted as-is on new data (Sec 4);
   requires the obs_noise-extended sweep before any fine-mesh ETKF number is cited.

3. **[HIGH, and specifically flagged by the task brief — genuinely at risk, not just
   inflated]**: **"weak short-λ components are unrecoverable by all methods at all
   apertures" (the information-wall evidence)**. If this finding's short-wavelength
   regime was generated at a raycast mesh that could not geometrically render those
   wavelengths in the first place (which G1's table confirms for essentially every
   short-λ scenario tested: BF3 5-comp λ_min=6.25m at 77% var_under, BF4 5/15/30-comp
   all DEGRADED with real unresolved variance), then "unrecoverable" may currently be
   conflating **facet noise masquerading as an information limit** with a genuine
   information limit. This is circular exactly as the task brief warns: if short-λ was
   never renderable, its unrecoverability is not evidence of an information wall, it's a
   simulator artifact. **This is the single most important documented conclusion to
   re-derive** — locate whichever `docs/`/`tmp/` file states this finding (likely
   `project_kp_floor_audit_2026-07-14` or an E-series short-λ ablation; not identified by
   filename in this audit's scope, flagged as the next agent's first lookup) and re-run
   its exact short-λ episodes at the new mesh before citing it again anywhere, thesis or
   paper.

4. **[MEDIUM — likely survives, verify with a small sample]**: relative rankings/orderings
   (WaveSplat < ETKF < GPR-1; raydrop-model sensitivity's *direction*, if not magnitude).
   The one fully-tested episode preserved ordering (margin widened, 34.8%→37.3%); the
   4/12-complete pilot shows the same pattern for 3 of 4 episodes so far, with BF3_1c's
   ETKF anomaly (Sec 4) as the one exception found. Comparisons made **entirely within**
   the new mesh (e.g., WS-P0 vs WS-P1 ablations, or raydrop-model sensitivity once both
   uniform/slope are regenerated at the same new mesh) are self-consistent and least at
   risk.

5. **[LOW — orthogonal to mesh, not re-run for this reason]**: anything hinging on
   platform motion/moving-ship claims (finding #5 in the premise audit — mesh-independent,
   already disclosed in the paper), the stereo "real data" substitution (#6 — doesn't touch
   `wavesplat.tex` at all), directional/spectral-shape narrowness (#4 — a sampling-scope
   issue, not a rendering one). None of these change because the mesh changed.

6. **[LOW, deferred]**: G2 (angular-noise range-scaling bug) and G8 (ETKF `Ne` staleness)
   are real but were deliberately **not** fixed in this pass (Sec 1, Sec 4) to keep
   attribution clean — they are follow-up items, not things the mesh regen's own
   comparisons depend on resolving first.

---

## Appendix: scripts produced during this audit (all in `wavesplat/tmp/`, none touched `data/`)

- `regen_mesh_calc.py` — loads real `wave_components` from all in-scope shipped files,
  confirms λ depends only on `(BF, ncomp)`.
- `regen_mesh_search.py` — sweeps candidate mesh spacings through the real
  `mesh_adequacy.check()`, finds the 1.36→1.35m ADEQUATE-for-all transition.
- `regen_mesh_table.py` — produces the Sec 2 per-scenario table at 40×30/180×150/200×167.
- `regen_run_finemesh.py` — the Step 3/4 driver (not yet run at scale; smoke-test with
  `--only BF2 --output_dir tmp/regen_smoketest` before the full run if any doubt remains).
- `etkf_retune_probe.py` / `etkf_retune_probe_results.json` — Sec 4.1/4.2 grid/Ne/inflation
  sweep on the two available fine-mesh pilot pairs.
- `etkf_obsnoise_probe.py` / `etkf_obsnoise_probe_results.json` — Sec 4.2 obs_noise sweep,
  the dose-response evidence for Mechanism B.
- Timing probes (inline, not saved as standalone files) establishing 0.769/0.974/1.235
  s/frame at 40×30/180×150/200×167 with GPU raycast.
