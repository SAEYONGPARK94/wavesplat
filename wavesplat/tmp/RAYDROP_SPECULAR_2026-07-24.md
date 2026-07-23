# The `slope` raydrop arm is not a specular model (2026-07-24)

**Status: MEASURED, three independent confirmations. Paper prose corrected in this commit.**

## Claim under test

`paper/wavesplat.tex` describes the two observation models as *diffuse Fresnel* (`uniform`)
vs *specular backscatter* (`slope`), and twice explains the measured raydrop sensitivity by a
mechanism — "specular backscatter preferentially preserves wave-crest points" (`:1384`,
`:1589`).

## Finding

**The specular branch of `wavesim/raydrop_slope.py` cannot fire under this sensor geometry.**
`apply_dropout` therefore always takes its fallback at `raydrop_slope.py:115-120`, which is an
**unweighted uniform Bernoulli** subsample over all ray hits.

### Route A — closed form

`specular_alignment` computes `reflected = d - 2(d·n)n` and `alignment = reflected·(-d)`.
Expanding:

```
alignment = -(d·d) + 2(d·n)(n·d) = -1 + 2(d·n)^2
```

so the gate `alignment >= cos(alpha)` is exactly

```
|d·n| >= sqrt((1 + cos alpha)/2) = cos(alpha/2)
```

With the shipped `alpha_threshold_deg = 5.0`, a specular return requires the ray to strike
**within 2.50 deg of normal incidence**. The Falcon fan is 78-90 deg from vertical (grazing),
and the surface normal departs from vertical only by the local wave-slope angle
(3.4 deg at BF4, 6.8 deg at BF7). The two can never meet.

### Route B — numerical, using the shipped function

`tmp/rerun_2026-07-24/verify_specular_branch.py`, 200,000 rays per case, elevation drawn
uniformly over the real 78-90 deg fan:

| case | max wave slope | specular hits | max alignment |
|---|---|---|---|
| BF2 (eps~0.02) | 1.15 deg | **0** | -0.8975 |
| BF4 (eps~0.06) | 3.43 deg | **0** | -0.8591 |
| BF7 (eps~0.12) | 6.84 deg | **0** | -0.7917 |
| unphysically steep | 21.80 deg | **0** | -0.3898 |

Gate threshold is `+0.99619`. The margin is not narrow — it is a sign error away, i.e. the
reflected ray always goes *away* from the sensor, which is what grazing incidence means.

### Route C — the repo already knew

`scripts/d3_s2b_true_f.py:4-7` states independently: "raydrop_slope.py wants the facet normal
along the ray: required slope tau = r/h = 3..40 against a wave-slope sd of 0.08, so
P(specular) = 0 and the code always takes its constant-P fallback at raydrop_slope.py:115".
That self-correction was never carried into the paper — the same stalled-correction path that
left the `:1553` BF2 misdiagnosis in the text for two months.

## What this does and does not change

**Dies**: the mechanism. There is no crest preference in an unweighted subsample, so
"specular backscatter preferentially preserves wave-crest points, providing richer spatial
structure" is not a description of anything the simulator did.

**Survives**: the ranking. The two arms *are* two genuinely different sampling patterns, so
"WaveSplat wins 10/10 under both" and the measured Delta% values stand as reported.

**Correct restatement**: the contrast actually tested is
**geometry-weighted sampling (`R_Fresnel(theta) cos(theta)/d^2`, labelled "uniform")
vs unweighted random sampling (labelled "slope")** — a test of sensitivity to *how return
density is distributed over range and incidence angle*, not of diffuse vs specular optics.
That is still a worthwhile robustness axis, and arguably a more honest one to claim, since
the unweighted arm is the adversarial case for any method that relies on confidence weighting.

**Pre-existing, separate, unfixed**: the `uniform` arm's formula
`R_Fresnel(theta) cos(theta)/d^2` uses a *specular* Fresnel reflectance as a *diffuse* albedo,
which a previous repo audit classified as a category error. Point *counts* are hidden by
calibration to `target_n_pts`; point *patterns* are not. Not addressed here.

## Sites corrected in `paper/wavesplat.tex`

`:70` (abstract), `:99` (highlights), `:737-740` (methods), `:1336-1339` (results intro),
`:1382-1389` (mechanism), `:1584-1596` (discussion), `:2083` (conclusion).
