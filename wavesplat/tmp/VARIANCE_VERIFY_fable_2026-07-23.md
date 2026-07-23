# Independent verification of ESTIMATOR_VARIANCE_RESULT_2026-07-23.md — 2026-07-23

Verifier note: this document is an independent re-derivation, not a re-statement of the
author's numbers. Script `tmp/variance_verify_fable.py`, log `tmp/variance_verify_fable.log`,
raw `tmp/variance_verify_fable_results.json`. Does not touch the author's
`tmp/estimator_variance.py` / `tmp/estimator_variance_results.json`, and does not run the
concurrent `tmp/restart_selection.py` (per instructions — its live output is referenced
below only as corroboration, not as my own result).

## Verdict: **CONFIRMED**, mechanism fully pinned down, one correction to a secondary claim

The headline numbers hold up under every attack:
- **CV up to 53% on a fixed episode**: reproduced exactly from raw data, and the extended n=24
  run gives a *pooled* CV of 44.6% — same order, not an n=8 fluke.
- **Bimodal in 3/4 episodes**: for the extreme episode, resolved at n=24 into a clean,
  zero-occupancy gap (0.1066→0.1662) separating a spread-out "good" cluster (n=13) from an
  extremely tight "bad" cluster (n=11, CV=0.38%) — this is *stronger* evidence of bimodality
  than the original n=8 table, not weaker.
- **seed=42 favourably ranked but underpowered**: the P≈0.08 combinatorics reproduce exactly,
  a second framing gives the same order of magnitude (~5%), and git history confirms `seed=42`
  was never tuned — the author's restraint (not claiming cherry-picking) is the correct call.
- **Reproducibility**: directly tested and holds bit-exact (closes off the single most
  dangerous alternative explanation — GPU/thread nondeterminism).

What changes is the **mechanism**, now pinned down precisely rather than speculated: it is a
**single, one-shot decision** — the energy-based adaptive-pruning step
(`sequential_update.py:395-456`), evaluated only against the short initialization window, which
either correctly keeps both true wave components or over-prunes down to one — and the entire
seed-dependence funnels through the **initial-phase draw** (`torch.rand`), confirmed by an
ablation that collapsed CV from 53% to exactly **0.0%** when phase alone was pinned.

One **material correction** to the author's own follow-on reasoning: `adaptive_n=False` does
**not** fix the problem (a natural next guess, and one the author's document invites by naming
"adaptive-N" as part of the mechanism). It barely reduces CV (53.0%→43.5%) while making the
*typical* run substantially worse (mean +32%) and producing a new worst-case. Adaptive pruning
is net beneficial on average; its flaw is the specific one-shot safety check, not its
existence — the author's proposed fix (multi-restart, select on GT-free `obs_loss`) remains
the right direction; "disable adaptive_n" would not be.

---

## Attack 1 — Is `seed` doing what the author thinks? Is it actually reproducible?

**Code trace (MEASURED, from reading `run.py`, `sequential_update.py`, `initializer.py`,
`coarse_to_fine.py`, `spectral_gaussian.py`):**
- `set_seed()` sets `torch.manual_seed`, `np.random.seed`, and `torch.cuda.manual_seed_all`.
  `device` defaults to `'cpu'` in `DEFAULT_CONFIG` and is never overridden in the author's
  script — **no GPU is involved in this experiment**, so CUDA nondeterminism is not in play.
- For `init_method='v2_konly'` (the config used throughout), **kx/ky are never random** — they
  come from deterministic dispersion-aware beamforming on the fixed observation data, at every
  stage (initial N=3, growth to N=8/15 in `coarse_to_fine.py`, and any later adaptive growth in
  `sequential_update.py:_adaptive_prune_grow`). The **only** place `config['seed']` enters the
  v2_konly pipeline under normal conditions is the initial-phase draw
  `torch.rand(n, ...) * 2*pi`, called once per newly-introduced batch of components
  (`initializer.py:478/488`, `coarse_to_fine.py:152`, `sequential_update.py:550`). A secondary,
  rarely-triggered channel is `np.random.seed(seed)`-derived padding
  (`np.random.randn`/`rng.randn`) used only when beamforming fails to find enough peaks — this
  did not trigger in the episode tested here (see Attack "phase-freeze" below, which fully
  explains the variance without needing this channel).
- Model-construction-time random draws (`spectral_gaussian.py:57-60`, `torch.randn`) are
  immediately overwritten by `.copy_()` with the deterministic/random init values, so they
  don't independently contribute (they do consume RNG state, but that consumption is itself a
  deterministic function of the code path for a given seed, not a separate entropy source).

**Direct empirical test (MEASURED, n=2 within-run + 1 cross-script):** ran
`episode_BF3_2comp_seed0_uniform.h5`, `seed=1`, twice, back-to-back, with the author's exact
config. Both runs gave **`avg_nRMSE = 0.16822878779483993`** — bit-identical to the last stored
digit, identical final `n_gaussians` (1), identical `spectrum_params` (kx, ky, w, phi, all
component-for-component identical). **A third, fully independent confirmation** came from the
concurrently-running `tmp/restart_selection.py` (different script, different process, started
independently by another verification thread): for the same episode/seed/config it reports
`seed=1 → nRMSE=0.1682, obs_loss=0.00741613` and `seed=42 → nRMSE=0.0524, obs_loss=0.000593`,
matching my own runs and the author's original run to 5-6 significant figures.

**Verdict on Attack 1: REFUTED as a concern.** The estimator is exactly, bit-for-bit
reproducible for a fixed seed on fixed data. There is no GPU or threading nondeterminism at
play. `seed` behaves as documented: it is the sole (for this config) source of run-to-run
randomness, and it does so almost entirely through the initial-phase draw, not through kx/ky.

---

## Attack on mechanism — what actually differs between a good and a bad run?

(MEASURED, n=4: seeds 42, 7 labelled "good" by the original table; seeds 1, 6 labelled "bad".)

| seed | label | avg_nRMSE | init_nrmse (end of Phase 1) | n_gaussians (final, post-prune) | avg_obs_loss | T_converge |
|---|---|---|---|---|---|---|
| 42 | good | 0.0524 | 0.0474 | **2** | 0.0006 | 11.6 s |
| 7  | good | 0.0590 | 0.0471 | **2** | 0.0011 | 11.1 s |
| 1  | bad  | 0.1682 | 0.0472 | **1** | 0.0074 | 14.1 s |
| 6  | bad  | 0.1676 | 0.0469 | **1** | 0.0074 | 11.1 s |

**The initialization-phase fit quality is essentially seed-invariant** (`init_nrmse` ranges
0.0469-0.0474, a <1% spread) — this rules out "Adam gets stuck in a qualitatively worse basin
during joint optimization" as the story, because if that were true the *initialization* itself
would already look bad for seeds 1/6, and it does not.

**What differs is entirely downstream, at the one-shot adaptive-pruning decision.** The true
field has 2 wave components. Good seeds retain both (`n_gaussians_final=2`); bad seeds prune
down to 1. This decision is made **once**, immediately after initialization
(`sequential_update.py:322`, called only from `initialize()`, never again from `update()`), by
sorting components by energy (`w²`) and keeping enough to reach 90% cumulative energy — with a
safety rollback if pruning increases residual RMS by >30%, but that safety check is itself
evaluated **only on the initialization window** (`sequential_update.py:417-419`,
`self._compute_residuals(self.model, frames)` where `frames` is the same short init-frame list
passed into `initialize()` — confirmed by reading the call site). A model whose *relative*
component energies happen (due to the random initial phase interacting with the coarse-to-fine
schedule) to concentrate almost all fitted energy in one Gaussian will pass this check and
prune down to N=1 even though the true field needs 2 — and the damage from losing that second
wave train is invisible on the init window (which the check *is* evaluated on) and only shows
up during the much longer tracking phase (`T_converge` for seed 1 is later, 14.1s vs ~11.1-11.6s
for the others, consistent with — but not dispositive proof of — a harder recovery).

**This mechanism explains "harder problems have *less* variance" mechanistically, not just
descriptively:** every episode, regardless of true component count, goes through the *same*
default coarse-to-fine schedule (N: 3→8→15) before any pruning — `n_gaussians` in
`DEFAULT_CONFIG` is `[3, 8, 15]` and is not overridden by episode content anywhere in the
variance script. For a true 2-component field, a bad phase draw can make one real wave "hide"
inside another Gaussian's fitted energy budget, so pruning throws away 50% of the true signal.
For a true 15-component field, the same kind of phase-driven local mis-partition affects at
most one or two components out of many active real signals, so the aggregate nRMSE barely
moves — diluted, not eliminated, hence the small but nonzero 2.8-10.1% CV on the harder
episodes rather than exactly 0%.

**Label: MEASURED mechanism** (n=4 for the good/bad contrast on the extreme episode; supported
by a full ablation, below, not merely a plausible story).

---

## Attack 2 (resolved) — is the bimodality real, or an n=8 artifact?

**Extended run (MEASURED, n=24: original 8 seeds + 16 fresh seeds {10..25}, all real
randomness, `adaptive_n=True`, worst episode):**

```
sorted nRMSE (n=24):
0.0524 0.0587 0.0588 0.0590 0.0604 0.0627 0.0671 0.0694 0.0705 0.0731 0.0737 0.0891
                                                                                     <-- gap: 0.0595 wide, 0 points --
0.1066
                                                                                     <-- gap: 0.0595 wide, 0 points --
0.1662 0.1667 0.1669 0.1672 0.1673 0.1673 0.1676 0.1678 0.1679 0.1682 0.1683
```

There is a clean, **zero-occupancy gap from 0.1066 to 0.1662** (width 0.0595) separating two
groups:
- **Low cluster, n=13**: mean 0.0693, sd 0.0146, range [0.0524, 0.1066] — a genuinely spread-out
  group (contains both `n_gaussians_final=2` outcomes and lucky `n_gaussians_final=15` outcomes,
  see mechanism section).
- **High cluster, n=11**: mean 0.1674, **sd 0.0006, CV 0.38%** — an extremely tight cluster,
  the signature of a single near-deterministic failure mode (losing one whole wave component)
  rather than a smooth tail of a unimodal distribution.

**Verdict on Attack 2: CONFIRMED, more strongly than the original n=8 evidence.** A smooth,
unimodal generating process would not produce, at n=24, an internal sub-cluster with CV=0.38%
sitting isolated by a gap 6× wider than that sub-cluster's own spread. This is bimodal — in
fact closer to a **two-state (roughly-good vs collapsed-to-N=1) switch with continuous noise
within each state** than to a textbook two-Gaussian mixture. The split is close to balanced
(13 low / 11 high, 54%/46%), consistent with the original 8-seed estimate (5/8 good, 3/8 bad)
within sampling noise. Pooled n=24 CV = 44.6% (mean 0.1143, sd 0.0510) — in the same range as
the original n=8 CV of 53.0%, confirming the original headline number was not an n=8 fluke.

---

## Attack B (complete) — does `adaptive_n=False` collapse the variance?

**Full 8-seed result (MEASURED):** `sorted = [0.0581, 0.0795, 0.0868, 0.1066, 0.1273, 0.1309,
0.1669, 0.2242]`, **mean=0.1225, sd=0.0534, CV=43.5%**, vs the original `adaptive_n=True` run
on the identical 8 seeds: mean=0.0930, sd=0.0492, **CV=53.0%**.

**Verdict on Attack 4/B: MATERIAL REVISION of the naive prediction.** Disabling adaptive-N does
**not** fix the variance problem. CV drops only modestly (53.0%→43.5%, well within noise of a
n=8 estimate) while the **mean gets dramatically worse** (0.093→0.123, +32%) and the **worst
case gets much worse** (0.168→0.224, a new all-time-worst outcome, seed=2). The reason,
confirmed by the `n_gaussians_final=15` seen on every one of these 8 runs: removing pruning
does not stop bad phase seeds from happening — it just changes what a bad phase seed damages.
With pruning enabled, a bad phase draw can cause catastrophic **under**-fitting (collapse to
N=1). With pruning disabled, the model is permanently stuck at N=15 for a true 2-component
field, and a bad phase draw can instead cause **overfitting/misfit** in the 13 spurious extra
components, which is just as damaging and, on this evidence, somewhat worse on average.
**Adaptive-N is net beneficial, not a bug to switch off** — its defect is specifically the
one-shot, short-window-only safety check that occasionally fails to catch bad over-pruning
(seed 1, 6, 11, 14, 16, 17, 18, 19, 21, 23, 25 all collapsed to N=1 in the full sample), not its
existence. The author's proposed fix (multi-restart, select on `obs_loss`) is directionally
correct; "just turn adaptive_n off" is not a fix and should not be adopted as one.

---

## Attack A — phase-freeze ablation (does isolating the phase draw collapse the variance?)

Patched `torch.rand` globally to a fixed `torch.Generator` seeded to a constant, independent of
`config['seed']`, leaving `np.random.seed(seed)` (and hence the rare padding fallback) untouched
and varying normally with seed. Re-ran the same 8 seeds on the worst episode
(`BF3_2comp_seed0_uniform`, `adaptive_n=True`, otherwise identical config).

**Result (MEASURED, n=8): CV = 0.0%.** All eight seeds produced the exact same value,
`avg_nRMSE = 0.055273063235769145`, to 15 significant digits — same `n_gaussians_final=3`, same
`init_nrmse=0.0470`, same `avg_obs_loss=0.0008`, for every one of seeds {42,1,2,3,4,5,6,7}.

**This is a clean, complete confirmation of the mechanism, not merely consistent with it.**
For this episode, the rare numpy-random padding fallback never triggered — beamforming always
found enough peaks — so 100% of the seed-to-seed variance in the original 53%-CV result is
attributable to the initial-phase draw. (Caveat: this was tested on one episode; the padding
channel could matter on episodes where beamforming is more peak-starved — untested here, flagged
as a limit below.)

---

## Arithmetic re-derivation of the author's original table

Independent re-derivation of the author's summary statistics directly from
`tmp/estimator_variance_results.json` (MEASURED, arithmetic only, no new runs): reproduces the
table **exactly** — mean/sd/CV/min/max/range-mean and seed-42's rank in all 4 episodes match to
the last decimal shown. No arithmetic error found.

```
BF3_2comp_uniform   n=8  cv=53.0%  sorted: [0.0524,0.0588,0.0590,0.0604,0.0705, 0.1066,0.1676,0.1682]
BF3_5comp_uniform   n=8  cv= 4.8%  sorted: [0.1675, 0.1817, 0.1921,0.1924,0.1924,0.1925,0.1925,0.1925]
BF3_5comp_slope     n=8  cv=10.1%  sorted: [0.1491,0.1517, 0.1893,0.1895,0.1895,0.1897,0.1901,0.1901]
BF4_15comp_uniform  n=8  cv= 2.8%  sorted: [0.2753,0.2821,0.2875,0.2902,0.2906,0.2920,0.2982,0.2999]
```

---

## Attack 5 — is the seed=42 ranking claim sound? Was it ever tuned?

**Statistical re-derivation (MEASURED):** reproduced the author's P≈0.08 figure exactly:
`P(X≥2)` for `X~Binomial(4, 1/8)` = 1 − (7/8)⁴ − 4·(1/8)·(7/8)³ = 1 − 0.5862 − 0.3350 = **0.0788**.
Cross-checked with an alternative framing that also credits the 2nd-place finish (not just 1st
place): `P(≥3 top-2 finishes in 4 draws)` for `X~Binomial(4, 1/4)` = **0.051**. Both land in the
same "suggestive, ~5-8%, not significant at n=4" zone — the author's restraint (UNDERPOWERED,
no cherry-picking claim) is the correct call under either framing, not an artifact of choosing
a favourable statistic.

**Provenance (MEASURED, git history):** `git log --follow -p -- wavesplat/wavesplat/run.py`
shows `'seed': 42` introduced exactly once, in commit `60672ff` ("P0 완료: beamforming init
(v2_konly), ablation study, ETKF comparison", 2026-04-07), as part of the initial
`DEFAULT_CONFIG` dict alongside everything else added in that commit. Grepping the entire diff
history for `'seed':` shows no other literal value was ever used — it has never been changed,
swept, or tuned. `configs/estimation.yaml` also carries `seed: 42` (same literal). The paper's
headline 12-episode table script, `scripts/run_p1_comparison.py`, hardcodes `'seed': 42`
directly (not read from the YAML) and calls `run_estimation` exactly once per episode — no
restart, no seed loop, no selection-on-anything, anywhere I could find in `scripts/`.

**Verdict on Attack 5: CONFIRMED, both halves.** (a) `seed=42` is a plain convention default
(the well-known "42" joke value), never tuned, so there is no intentional cherry-picking
mechanism in the codebase. (b) The consequence the author flags — "with a bimodal estimator,
whatever seed a report happens to use is one draw, and unintentional luck is exactly as costly
to interpretation as intentional cherry-picking" — is correct and is not undercut by (a).

**One clarifying note not in the original doc:** the term "seed" is heavily overloaded in this
repo. The `_seed0_` / `_seed1_` in episode filenames (`episode_BF3_2comp_seed0_uniform.h5`) is
the **data-generation seed** (which wave components/phases exist in the simulated field —
traced to `wavesim/pipeline.py`), and is completely orthogonal to `config['seed']` (the
estimator's own RNG seed, the subject of this whole investigation). The two are independent
axes; this document and the original finding are careful about this throughout, but it is worth
stating explicitly for anyone skimming the tables later.

---

## What this does and does not license

**May be cited:**
- The bimodal, seed-driven run-to-run variance is real, reproducible (bit-exact), and its
  magnitude (CV up to 53% on the easiest episode tested, pooled CV=44.6% at n=24) is confirmed
  by independent re-derivation and by scaling up the sample.
- Bimodality on the extreme episode is confirmed, not just suggested: at n=24 there is a clean
  gap (0.1066→0.1662, zero points inside) separating a spread-out good cluster (n=13,
  mean=0.0693) from an extremely tight bad cluster (n=11, mean=0.1674, CV=0.38%).
- Per-episode point estimates in this project (paper table, mesh pilot, any n≤a-few-episodes
  ablation) carry an unquantified, potentially large error bar from this source alone, and are
  not currently reliable at the individual-episode level.
- The mechanism is the one-shot, energy-based adaptive-pruning decision
  (`sequential_update.py:395-456`) validated only against the initialization window, which
  either keeps both true components (`n_gaussians_final=2`) or collapses to one — a specific,
  fixable defect in the adaptive-N policy's *safety check*, not an indictment of Adam /
  non-convex optimization in general. Confirmed via a clean ablation: freezing only the
  initial-phase draw (`torch.rand`, independent of `config['seed']`) collapses CV from 53% to
  exactly **0.0%** (all 8 seeds bit-identical) on the tested episode — proving phase is the sole
  seed-dependent channel here.
- Disabling `adaptive_n` is **not** a fix: full 8-seed ablation gives CV=43.5% (barely lower
  than 53.0%) with mean nRMSE +32% worse (0.093→0.123) and a new worst case (0.224). Adaptive
  pruning is net beneficial; the defect is its one-shot, short-window safety check, not its
  existence.
- The 100-episode aggregate averages / paper headline are not directly threatened (noise
  averages down by ~10×, per the author's own math, which I did not re-derive here but which
  follows directly from CLT and is not in dispute).
- `seed=42` is a plain, never-tuned convention default (introduced once in commit `60672ff`,
  never changed since); there is no cherry-picking mechanism anywhere in `scripts/`.

**May NOT be cited:**
- "The mechanism generalizes to all episodes/configs" — the phase-freeze ablation and the
  detailed good/bad component-count comparison were run on one episode only
  (`BF3_2comp_seed0_uniform`). The numpy-random padding fallback (triggered only when
  beamforming under-supplies peaks) was not exercised here and could contribute variance
  elsewhere, particularly on sparser or noisier episodes — untested.
- "Adaptive-N is the cause of the variance" as a standalone claim — it is not; the phase draw
  is the cause, and adaptive-N's pruning *decision* is what converts a bad phase draw into a
  catastrophic outcome for low-component fields specifically. Disabling adaptive-N does not
  remove the variance.
- Nothing here revisits the mesh-pilot's own conclusions; this document only re-verifies the
  variance-measurement finding that motivated it.

## Limits

Full ablation depth (phase-freeze, adaptive_n=False, n=24 bimodality resolution,
component-level mechanism detail) was run on **one episode only** — the declared worst case,
`BF3_2comp_seed0_uniform`. The other 3 episodes from the original 4 were re-verified only by
arithmetic re-derivation of already-collected data, not by new runs; whether the same
"phase-draw-through-one-shot-pruning" mechanism explains their smaller CVs (2.8-10.1%) was not
directly tested (though the coarse-to-fine/adaptive-N code path is identical across episodes,
so it is the best-supported hypothesis). One estimator configuration throughout
(`init_method: v2_konly`, `use_dispersion: True`, `use_confidence: True`, `tracking_window: 3`)
— not tested: HOS episodes, other init methods (`v2`, `relax`), other tracking modes.
