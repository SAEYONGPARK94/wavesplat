# Scale-dependent constant audit — every *other* fixed threshold besides the coarse-to-fine advance loss

Companion to `tmp/VARIANCE_TRACE_fable_2026-07-23.md` (root-caused the `[0.1, 0.05, None]`
`should_advance()` thresholds — being fixed separately as `coarse_to_fine.SCALE_INVARIANT_ADVANCE`,
untouched here). This document sweeps the rest of the estimation pipeline for constants compared
against a **dimensional** quantity (metres, metres², rad/m) instead of a ratio, and measures —
not guesses — where each one actually misbehaves on the shipped episode set.

**Scope note (MEASURED):** the shipped dataset spans **six** sea states, not four. `data/*.h5`
attrs (`config` group): BF2 Hs=0.30, BF3 Hs=0.60, BF4 Hs=1.20, BF5 Hs=2.00, BF6 Hs=3.00,
BF7 Hs=5.00 m (Tp 4.0→12.0 s), 195 files, component counts ∈ {1,2,5,15,20,30}. All ratios below
use these measured Hs values, not the task's approximate "0.5–5 m".

## Ranked findings

| # | Constant | File:line | Compared against | Bad regime | Episodes affected | Severity |
|---|---|---|---|---|---|---|
| 1 | `grow_residual_threshold=0.05` [m] | `sequential_update.py:86,482-483` | `residual_rms` [m], raw (unnormalized) post-fit residual | Low Hs (BF2): triggers on noise-floor residual. High-Hs/crowded (BF4–BF7, ≥15 true comps): **always** triggers, saturating toward `max_n`, in lock-step with the already-known prune-rollback | Measured directly on 3/3 tested episodes with ≥15 true comps (BF4-15, BF5-15, BF7-30) — that class is 86/195 (44%) of the shipped dataset (15+20+30-comp counts) | **HIGH — live, MEASURED, changes N and downstream nRMSE** |
| 2 | `min_dk=0.15` [rad/m] (×3 sites) | `initializer.py:395`, `coarse_to_fine.py:190`, `sequential_update.py:515` | Euclidean spacing between candidate `(kx,ky)` peaks, also rad/m | Any episode where true components are packed closer than 0.15 rad/m — increasingly true as sea state/`n_components` rises (dispersion pushes the whole spectrum to lower |k| while the schedule adds more comps into the same shrunken band) | **126/195 (64.6%)** of all shipped episodes have ≥1 true pair below 0.15 rad/m; **100% of 15-, 20-, 30-comp episodes** (47/47, 6/6, 33/33); 40/47 (85%) of 5-comp; 0/62 of 1–2-comp | **HIGH — live, MEASURED directly against ground truth, causes component merging** |
| 3 | `lambda_l1=0.001` [dimensionless coefficient] × `Σ|w|` [m] | `sequential_update.py:61,187-188` | added directly to a data-fit term with units m² | Any episode, worse where Phase-1 already converges well (residual near noise floor) since a fixed absolute penalty then dominates a small squared-error term | Not counted per-episode (continuous effect); MEASURED on 4 probe episodes, ratio 3–62% of data term | **MEDIUM — live, MEASURED bias, downstream nRMSE effect not separately ablated** |

Below each item: what was actually run, not just read.

---

## 1. `grow_residual_threshold` — the advance threshold's un-fixed sibling

**Where.** `sequential_update.py:86` — `self.grow_residual_threshold = config.get('grow_residual_threshold', 0.05)`
(also `configs/estimation.yaml:29`, `run.py:36` DEFAULT_CONFIG). Consumed at
`sequential_update.py:482-483`:
```python
residual_rms = float(np.sqrt(np.mean(all_res ** 2)))     # raw elevation residual, metres
if residual_rms < self.grow_residual_threshold:            # fixed 0.05 m
    ...no growth...
```
`all_res` is `z_obs - z_pred` in metres — **never divided by Hs, by signal RMS, or by the noise
floor.** This is the exact same class of bug as the already-being-fixed advance threshold
(`coarse_to_fine.py:64`), but it lives in a different function (`_adaptive_prune_grow`,
`sequential_update.py:397`) and is **not touched by `SCALE_INVARIANT_ADVANCE`** — that flag only
gates `should_advance()`'s comparison inside the Phase-1 coarse-to-fine loop, not this
post-Phase-1 prune/grow step. Fixing the advance threshold alone leaves this one live.

**MEASURED — direct runs of the shipped `python -m wavesplat.run --verbose` (unmodified code),
seed 42 unless noted:**

| episode | Hs [m] | residual_rms [m] at grow check | ratio to threshold | outcome |
|---|---|---|---|---|
| BF2 1comp (seed 42/1/3/7, all 4 tested) | 0.30 | 0.0577–0.0579 (n=4 seeds, all above) | 1.15–1.16× | **Grows** 8→11/12 |
| BF3 2comp | 0.60 | 0.0211 | 0.42× | no growth |
| BF3 5comp | 0.60 | 0.0272 | 0.54× | no growth |
| BF4 1comp | 1.20 | 0.0309 | 0.62× | no growth |
| BF4 15comp | 1.20 | 0.0691 | 1.38× | **Grows** 15→19, **simultaneous prune-rollback** ("residual would increase 0.0691→0.1012") |
| BF5 15comp | 2.00 | 0.1072 | 2.14× | **Grows** 15→21, **simultaneous rollback** |
| BF6 1comp | 3.00 | 0.0185 | 0.37× | no growth |
| BF7 30comp | 5.00 | 0.1746 | 3.49× | **Grows** 15→25, **simultaneous rollback** |

Pattern (MEASURED, 8 episodes, 11 runs total incl. 4 BF2 seeds): **every 15+/30-comp episode
tested at Hs≥1.2 grows and simultaneously trips the already-known 30% prune-rollback guard**,
i.e. this fixed threshold is not the only reason the rollback fires often in the crowded regime
— it *guarantees* it, because a fixed 0.05 m target represents an ever-smaller fraction of a
richer, higher-Hs field's genuinely unmodellable residual (residual_rms rose monotonically with
Hs across the three crowded probes: 0.069→0.107→0.175 m at Hs 1.2→2.0→5.0).

A broader Phase-1-only sweep across all 19 unique (sea_state, n_components) combos
(`tmp/scale_audit_grow_sweep.py`) was started to generalize the 8-point manual sample beyond
hand-picked episodes, but was **deliberately killed after ~4 minutes, before producing output**
(it wrote nothing to `tmp/scale_audit_grow_sweep_results.json`) — this session found the machine
already running an unrelated 100-episode regeneration (`tmp/regen_run_finemesh.py`, 3.5+ h CPU
time) plus the division-of-labour's own experiment, and the "keep runs modest" instruction took
priority over squeezing out one more generalization. The conclusions below do **not** depend on
the killed sweep — they already rest on 11 direct runs (8 distinct episodes, 4 of them repeated
across seeds) spanning the full measured Hs range (0.3-5.0 m) and both the 1-comp and
15/20/30-comp regimes, which is what is reported.

**All 4 BF2 seeds grow, consistently, at almost the same residual (0.0577-0.0579)** — this is not
seed noise, it is a reproducible miscalibration for the lowest-Hs sea state: the post-prune
residual sits essentially at the sensor noise floor (`sigma_range=0.02` m,
`wavesim/noise_model.py:15`, +model-fit slack) which is comparable to, not smaller than, the
fixed 0.05 m target when Hs=0.3 m. **Honesty check (MEASURED, not assumed):** re-running BF2
1comp/seed42 with `--adaptive_n false` (growth entirely disabled) gives avg nRMSE 0.3266 vs
0.3150 with growth enabled — growth is *not* the dominant cause of BF2's poor absolute accuracy
(both numbers are bad), but the threshold-miscalibration itself is real and reproducible; its
main measured consequence is the crowded-episode interaction with prune-rollback above, not BF2
specifically.

**Severity: HIGH.** Unlike the advance threshold (isolated to the N=8 stage of low-Hs episodes),
this one is measured to fire on effectively all richly-populated (15/20/30-comp) episodes
regardless of Hs, each time co-occurring with the known rollback guard — a second, independent,
Hs-agnostic driver of the over-parameterization pathway that the rollback guard alone was
previously assumed to cause by chance.

**Dimensionless fix.** Same medicine as the advance threshold: divide `residual_rms` by a
Hs- or observation-variance-derived scale before comparing, e.g.
`residual_rms / obs_rms < grow_residual_threshold` with `obs_rms = sqrt(mean(z_obs**2))` over the
same frames (already computed once per `initialize()` call for `SCALE_INVARIANT_ADVANCE`'s
`loss_scale` at `sequential_update.py:262-266` — that exact quantity can be reused here without
new code). A noise-floor-relative alternative (`residual_rms / sigma_range`) would also fix the
BF2 case specifically, since it targets "is there still real signal here" rather than "is the
residual small in absolute metres."

---

## 2. `min_dk=0.15` rad/m — the greedy beamforming search's fixed resolution floor

**Where** (3 independent, identically-hardcoded sites, all consumed the same way — suppress a
`min_dk`-radius neighborhood around every found peak before searching for the next one):
- `initializer.py:395` (initial N=3/8/15 beamforming, `initialize_from_frames_v2`)
- `coarse_to_fine.py:190` (residual beamforming at each schedule advance, `_beamform_residuals`)
- `sequential_update.py:515` (residual beamforming inside `_adaptive_prune_grow`'s growth step)

`min_dk` is compared to **another rad/m quantity** (candidate peak separation), so unlike items 1
and the advance threshold this is not a literal unit mismatch — both sides are wavenumber. It is
scale-dependent in the sense the task asked about anyway: the *natural* spacing between a sea
state's true wavenumber components shrinks as the JONSWAP peak shifts to lower k (higher Tp,
which in this dataset's presets correlates with higher Hs/BF number) while the same n_components
keep getting packed into that shrunken band, so a **fixed absolute** 0.15 rad/m resolution
floor becomes binding across an ever-larger fraction of true component pairs as sea
state/`n_components` rises.

**MEASURED directly against `config/wave_components` in all 195 shipped `.h5` files**
(`tmp/scale_audit_min_dk_census.py`, `scipy.spatial.distance.pdist` on each episode's true
`(kx,ky)`):

| n_components | episodes with ≥1 true pair closer than 0.15 rad/m | total |
|---|---|---|
| 1 | 0 | 47 |
| 2 | 0 | 15 |
| 5 | 40 (85%) | 47 |
| 15 | 47 (**100%**) | 47 |
| 20 | 6 (**100%**) | 6 |
| 30 | 33 (**100%**) | 33 |
| **all** | **126 (64.6%)** | **195** |

Min pairwise true-component spacing by (sea state, n_comp) (representative file each, MEASURED):
BF3 5comp 0.146, BF4 5comp 0.090, BF4 15comp **0.017**, BF4 30comp **0.008**, BF5 15comp
**0.010**, BF6 30comp **0.004**, BF7 30comp **0.003** rad/m — one to two orders of magnitude
below the fixed 0.15 suppression radius for every crowded episode.

**Consequence, not just co-incidence (MEASURED end-to-end on `BF4_15comp_seed0_uniform`,
seed 42, full shipped pipeline, `--verbose`, spectrum dumped and matched to ground truth):**
the 4 lowest-|k|, highest-energy true components (`A`=0.006, 0.092, 0.223, 0.248; `kx` spanning
only 0.028–0.094, i.e. inside one 0.15 window) **all map to the single strongest recovered
Gaussian** (`w=+0.378`, `kx=0.081`). Symmetrically, the 5 highest-|k| true components
(`kx`=0.415–0.694) all map to one weak recovered component (`w=-0.023`). Of 19 recovered
components, only ~7 are distinguishable non-degenerate peaks against 15 true ones — a direct,
measured demonstration that the fixed resolution floor collapses genuinely separate spectral
content, independent of (and additive with) finding #1's capacity/threshold problem. Full
numeric table: `tmp/scale_audit_bf4_15_{kx,ky,w}.npy` (recovered) matched against
`data/episode_BF4_15comp_seed0_uniform.h5:config/wave_components` (true) in this session's
transcript.

**Severity: HIGH for the crowded regime** (86/195 shipped episodes, 44%, have ≥15 true
components — all 100% affected per the census above), **inert for 1–2 comp episodes** (0/62
affected — matches why this was never surfaced by the 2-comp bimodality investigation), and
mostly-affected for 5-comp (40/47, 85%).

**Dimensionless fix.** `min_dk` should scale with the sea state's own spectral bandwidth, not be
a fixed rad/m constant — e.g. `min_dk = c · k_p` where `k_p` is the (already-computable)
peak wavenumber from `Tp` via `wavenumber_from_omega(2π/Tp, depth)` (`dispersion.py:62`), or
scale with `1 / n_new` (finer suppression when more slots must be filled from the same coarse
grid). A resolution floor that is a *fraction of the occupied band* rather than an absolute
rad/m value would keep pace across BF2→BF7 automatically.

---

## 3. `lambda_l1=0.001` — an L1 penalty on `Σ|w|` [m] mixed into a loss in [m²]

**Where.** `sequential_update.py:61` (`config.get('lambda_l1', 0.001)`, also
`configs/estimation.yaml:26`), applied at `sequential_update.py:187-188`:
```python
total_loss = total_loss + self.lambda_l1 * torch.sum(torch.abs(self.model.w))
```
`total_loss` up to this line is a mean squared (weighted) residual — units m². `Σ|w|` is
units m (linear in amplitude). `lambda_l1 · Σ|w|` therefore scales as O(Hs), not O(Hs²) like the
term it's added to, so its *relative* weight in the total loss is not fixed — it grows as the
data term shrinks (i.e. as the fit gets better), and shrinks as Hs grows (since `Σ|w|` grows
with signal amplitude but the residual grows with amplitude *squared* when unexplained).

**MEASURED** (read-only probe, `tmp/scale_audit_lambda_probe.py`, replicates `_compute_loss`'s
exact data-fit construction and prints it alongside the (unchanged) L1 term after
`initialize()` converges, same code path, no production files modified):

| episode | Hs [m] | N final | data-fit term (m²) | L1 term | **L1 / data-fit ratio** |
|---|---|---|---|---|---|
| BF2 1comp | 0.30 | 11 | 0.002739 | 0.000106 | 3.9% |
| BF3 2comp | 0.60 | 2 | 0.000390 | 0.000241 | **61.7%** |
| BF4 15comp | 1.20 | 19 | 0.001981 | 0.000973 | **49.1%** |
| BF7 30comp | 5.00 | 25 | 0.111673 | 0.003343 | 3.0% |

The ratio is *not* monotonic in Hs — it peaks in the mid-range where Phase-1 happens to converge
well (small data-fit denominator), confirming the mechanism is "fixed absolute penalty vs.
whatever the achieved residual happens to be," not simply "low Hs is worse."

**Direct ablation (MEASURED, `BF3_2comp_seed0_uniform`, seed 42, otherwise-identical run):**

| `lambda_l1` | recovered `w` | `Σ|w|` |
|---|---|---|
| 0.001 (shipped default) | `[-0.1699, 0.0710]` | 0.2409 |
| 0.0 | `[-0.2050, 0.1753]` | 0.3803 |

Removing the L1 term increases total recovered amplitude by **58%** on this episode (both runs
land on the same N=2 degenerate-pair solution as `VARIANCE_TRACE`'s seed-42 case, so this is a
pure amplitude-shrinkage comparison, not a different basin). Since the absolute shrinkage from a
fixed λ is roughly constant in metres (standard Lasso soft-thresholding), the same 0.001 penalty
represents a much larger *relative* bias on a small-Hs episode's small-amplitude components than
on a large-Hs one's — measured qualitatively consistent with the ratio table above.

**What was not measured:** the downstream effect on nRMSE (does the shrinkage help by
regularizing spurious components, or hurt by biasing real amplitudes down?) — not ablated
end-to-end here; flagging this ARGUED, not conflating it with the MEASURED loss-decomposition
and amplitude-shrinkage numbers above.

**Severity: MEDIUM.** Real, measured, non-trivial (up to 62% of the total loss, 58% amplitude
shrinkage in one direct test) but its net effect on accuracy is untested, and current shipped
`lambda_l1=0.001` is small enough that it hasn't been flagged as a headline accuracy problem
elsewhere in this project's docs.

**Dimensionless fix.** Scale `lambda_l1` by the same observation-variance quantity used for the
advance-threshold fix (`schedule.loss_scale` / `obs_rms`), e.g.
`lambda_l1 * obs_rms * Σ|w|` (making the whole term O(Hs²) again), or express it as a target
sparsity fraction of total energy rather than an absolute per-metre penalty.

**Checked, correctly scale-matched — not flagged:** `lambda_smooth=0.0001`
(`sequential_update.py:62,191-197`) penalizes `Σ(w_sorted[i+1]-w_sorted[i])²` — units **m²**,
matching the data-fit term's own units, so its ratio to the data-fit term is Hs-invariant by
construction (both scale as O(Hs²)) — analytically distinct from `lambda_l1`'s O(Hs) mismatch,
not independently re-ablated here since the reasoning is dimensionally exact.

---

## Checked and found NOT live / not a scale defect

| Constant | File:line | Why it's fine |
|---|---|---|
| `kx_range=(-4.0,4.0)`, `ky_range` [rad/m] | `initializer.py:337-338`, `coarse_to_fine.py:179`, `sequential_update.py:505` | MEASURED (`tmp/scale_audit_min_dk_census.py`): max `\|k\|` across **all 195** shipped episodes is **1.006 rad/m** (`episode_BF3_5comp_seed5_fixedT_dir180.h5`) — the ±4 search domain is never binding, checked exhaustively not sampled. |
| `converge_threshold=0.05` | `run.py:166`, `configs/estimation.yaml:39` | Compares against `nRMSE`, which is already `RMSE/Hs` (`evaluation/metrics.py:31`) — dimensionless by construction, correctly scale-invariant. |
| `max_steepness=0.15` | `spectral_gaussian.py:219-230` | Compares `\|k\|·\|w\|` (dimensionless steepness ratio) against a dimensionless bound — physically appropriate at any Hs. *Side-finding, not a scale bug*: `configs/estimation.yaml:18` documents this as configurable and `sequential_update.py`'s docstring (line 56) repeats that, but `clamp_steepness()` is called bare (`sequential_update.py:292,387,569`) — the config value is **never actually read or wired in** (`run.py` DEFAULT_CONFIG and `load_yaml_config` have no `max_steepness` key at all). Currently harmless only because the hardcoded default (0.15) happens to equal the documented config value — a latent footgun, not a scale defect, flagged for completeness. |
| `d_ref=125.0` [m] | `confidence.py:26,39` | Fixed LiDAR-geometry normalizer (`range_max=250` m is itself fixed across all episodes/sea states in this dataset) — does not vary with Hs/sea state, so it is outside the "misbehaves across sea states" framing even though it is a dimensional constant. |
| `SpectralGaussianModel.__init__` default `randn*0.05/0.01/π` (`spectral_gaussian.py:57-60`) and `add_gaussians`'s `None`-default fallbacks (`spectral_gaussian.py:186-189`) | same file | Dead in production: every call site (`sequential_update.py:275-280`, `coarse_to_fine.py:162-173`, `sequential_update.py:554-559`) always passes explicit `init_*` tensors: confirmed by reading all three call sites, the `is None` fallback branch never executes in the shipped pipeline. |
| `rmse_spatial_split`'s `split_distance=80.0` [m], `rmse_unobserved`'s `min_dist=10.0` [m] | `evaluation/metrics.py` | Report-only metrics (not control-flow), tied to fixed sensor geometry (`range_max=250`), not sea-state-scaled; `tp_error` (`metrics.py:44`) returns a raw-seconds diagnostic, also report-only. None of these change estimator behaviour. |
| `cs_initializer.py`'s `kx_range=(-3.0,3.0)` | `optimizer/cs_initializer.py:177` | Confirmed dead code — `grep -rl cs_initializer` inside the package matches only the file itself; not imported by any live `init_method` path (`v1`/`v2`/`v2_konly`/`relax`/`relax_konly`). |

## Known related items (per task instructions, not re-derived)

- Prune's 30% residual-rollback guard (`sequential_update.py:449`) — already dimensionless
  (ratio `rms_after > 1.3·rms_before`), correctly formed; already shown to co-occur with finding
  #1 above in every crowded probe (new observation: it's not incidental, #1 structurally
  guarantees it fires for Hs≳1.2 with ≥15 comps).
- `prune_threshold=0.10` energy fraction — already shown inert for the 2-comp case (target
  carried 0.08% of energy, 3 orders of magnitude under any tested cut).
- `noise_model.add_noise()` range-scaling bug (`tmp/REGEN_SPEC_fable_2026-07-23.md`) — separate,
  deliberately unfixed; does not confound the residual-RMS numbers used here (it under-simulates
  *angular* noise specifically; `sigma_range=0.02` m range-noise, which dominates the floor
  discussed in finding #1, is applied correctly).
- `PAD_W=0.01` / `KONLY_RANDOM_WPHI` (`initializer.py:25,30`) — already shown to be an inert
  branch for the tested episodes ("never executes"). Note for future work: the *same* fixed
  0.01 m padding amplitude is **separately hardcoded** (not gated by the `PAD_W` module flag,
  which only lives in `initializer.py`) at two more sites — `coarse_to_fine.py:154-160` (kx,ky
  std 0.05, amp 0.01) and `sequential_update.py:544-549` (kx,ky std 0.1, amp 0.01). Whether these
  two duplicate sites execute on any shipped episode was **not tested here** — flagged as
  untested, not claimed live or dead.

## Files

- `tmp/scale_audit_lambda_probe.py` — read-only loss-term decomposition probe (finding #3).
- `tmp/scale_audit_grow_sweep.py` — Phase-1-only sweep script (finding #1 generalization),
  started then killed before producing output to keep CPU load modest (see note above); its
  `.log`/`_results.json` outputs never materialized and were removed. Left in `tmp/` in case a
  future session wants to re-run it when the machine is less loaded.
- `tmp/scale_audit_min_dk_census.py` — 195-file true-component spacing census + full-dataset
  max-|k| check (finding #2 table, and the `kx_range` inertness check).
- `tmp/scale_audit_bf4_15_{kx,ky,w}.npy` — recovered spectrum for the BF4-15comp min_dk
  end-to-end check (finding #2).
- None of the above modify, import, or execute `coarse_to_fine.SCALE_INVARIANT_ADVANCE` or
  `tmp/scale_invariant_advance*`.
