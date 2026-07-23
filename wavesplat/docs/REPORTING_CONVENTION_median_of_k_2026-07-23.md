# Reporting convention: median-of-k with spread (2026-07-23)

**Status: binding for all per-episode numbers from this date forward.**

## The problem this fixes

On a **fixed episode with a fixed config**, changing only the estimator's random seed moves
`avg_nRMSE` by up to **CV 53 %**, and the spread is **bimodal** — runs land in one of two
well-separated convergence basins rather than scattering smoothly.

| episode | mean | sd | CV | range |
|---|---|---|---|---|
| BF3 2comp uniform | 0.0930 | 0.0492 | **53.0 %** | 0.0524 – 0.1682 |
| BF3 5comp uniform | 0.1879 | 0.0091 | 4.8 % | 0.1675 – 0.1925 |
| BF3 5comp slope | 0.1799 | 0.0182 | 10.1 % | 0.1491 – 0.1901 |
| BF4 15comp uniform | 0.2895 | 0.0080 | 2.8 % | 0.2753 – 0.2999 |

Source: `tmp/ESTIMATOR_VARIANCE_RESULT_2026-07-23.md`; independently confirmed in
`tmp/VARIANCE_VERIFY_fable_2026-07-23.md` (same seed reproduces bit-identically, so this is a
genuine seed effect, not nondeterminism).

Three consequences:

1. **A single-seed number is one draw from a wide two-cluster distribution**, not an estimate
   of the method's performance. The repo's default `seed=42` happened to rank 1st, 1st, 2nd,
   8th of 8 across four episodes — favourable in three of four (P ≈ 0.08 under chance; a
   conventional default, not a tuned choice, but the reported numbers inherit the luck).
2. **The variance is *larger* on the Nyquist-adequate regenerated dataset**, not smaller
   (BF3 2comp sd 0.0491 → 0.0676; 5comp 0.0109 → 0.0246;
   `tmp/MIN_DK_RESULT_2026-07-23.md`). Fixing the data made this more pressing.
3. **The variance is largest where the problem is easiest** (2 components: 53 %; 15
   components: 2.8 %) — so the calm, sparse episodes have the least trustworthy numbers.

## Why not best-of-k

Selecting the best run needs a quality signal available **without ground truth**. The only
candidate is the observation-fit loss, and a pre-registered test of it **FAILED**
(`tmp/RESTART_SELECTION_RESULT_2026-07-23.md`): ρ(obs_loss, nRMSE) is positive but weak
(+0.36 … +0.81, 1/4 significant), selection recovers only ~11 % of the available headroom,
and on the worst episode it actively *hurts*. Selecting on true nRMSE is not deployable and
must never be reported as a result.

## The convention

For **any per-episode number**, run **k ≥ 5 seeds** and report the **median together with the
spread** (sd or min–max). A median quoted without its spread is **not citable**.

```bash
python3 -m wavesplat.run --episode <ep.h5> --n_seeds 5
```
```python
from wavesplat.run import run_estimation_multi
res = run_estimation_multi('data_finemesh_2026-07-23/episode_BF3_2comp_seed0_uniform.h5',
                           config={...}, seeds=(42, 1, 2, 3, 4))
res['nRMSE_median'], res['nRMSE_sd'], res['nRMSE_cv_pct']
```

**Aggregates over many episodes are exempt in practice** — averaging ~100 episodes reduces
seed noise by ~10×, which is why the 12-episode mesh pilot's headline was stable
(`tmp/MESH_PILOT_RESULT_2026-07-23.md`). The requirement binds on: individual episode entries,
ablation deltas measured on a handful of episodes, and any "worst case" or "failure case"
claim.

## What is NOT fixed by this

The underlying cause is random phase initialisation at **two** sites — `initializer.py`'s
`v2_konly` branch (which discards the least-squares `(w, φ)` it just fitted) and
`coarse_to_fine.py`'s growth path. Disabling both makes the estimator deterministic
(CV 53 % → **0.0 %**) *and* more accurate on crowded episodes (t = −5.28); flags
`KONLY_RANDOM_WPHI` and `GROW_RANDOM_PHI` exist for this, defaulting to historical behaviour
(`tmp/VARIANCE_MECHANISM_HUNT_2026-07-23.md`).

That is **not** adopted as the default here, for two reasons: it does not deliver the *good*
basin (it lands between the clusters, which would make some published per-episode numbers
worse), and it has been tested on only 2 episodes. Determinism should be offered as
**reproducibility**, not as accuracy, and only after a full sweep on the regenerated dataset.

Deeper still: at the pathological operating point the fit is effectively **ill-posed** — the
weak component is recovered by *no* run, and "good" runs merely retain a redundant copy of
the strong wave whose spare DOF absorbs the interference. Fixing the schedule's
scale-dependent advance threshold only **re-deals** which seeds win (`H5`, same document).
So the spread is a property to report, not purely a bug to remove.
