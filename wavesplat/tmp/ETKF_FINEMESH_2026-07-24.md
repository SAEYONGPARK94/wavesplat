# ETKF on the fine-mesh dataset — diagnosis + honest re-tune (2026-07-24)

Pre-registration: `tmp/ETKF_FINEMESH_PREREG_2026-07-24.md` (written before any diagnostic
number). Consolidated numbers: `tmp/ETKF_FINEMESH_2026-07-24_numbers.json`. Raw:
`tmp/etkf_diag_2026-07-24.json`, `tmp/etkf_seedspread_2026-07-24.json`,
`tmp/etkf_beta_2026-07-24.json`, `tmp/etkf_ceiling_2026-07-24.json`,
`tmp/letkf_finemesh_2026-07-24.json`, `tmp/etkf_tune_stage{A1,A2,A3,C}_2026-07-24.json`.
Harness: `tmp/etkf_lib.py`. All CPU/numpy; no GPU used.

**TL;DR.** (1) Diagnosis: the coarse→fine ETKF swing is an **artifact of a stale baseline
configuration** (small analysis grid + additive inflation, over data that committed an
inverse crime against that exact operator), **not** a loss of information and **not** a
systematic regression (aggregate t = +0.91, n.s.). A *fairly*-tuned ETKF is mesh-invariant
(fine 0.261 vs old 0.260). (2) The headline margin **survives and widens**: WS-P1 0.182 vs
a paper-tuned ETKF 0.374 on fine-mesh data = **+51 %, paired t = 6.9, 12/12** — provided the
baseline is reported on the full metric panel, because pure-nRMSE tuning degenerates the
ETKF into a zero predictor (ρ → 0.015).

---

## 0. One-paragraph answer

The premise handed to me — *"ETKF systematically degrades on the fine-mesh data"* — is
**not supported**, and the coordinator's independent 12-episode re-run said the same. With
median-of-7 filter seeds the aggregate is 0.4155 → 0.4324 nRMSE, paired-by-seed
mean Δ = **+0.0123, t(11) = +0.91, sign test 9/12 p = 0.15 — not significant**. What *is*
real is two individually-resolvable episode-level swings *in opposite directions*
(BF2 1comp **−22 %**, BF3 1comp **+29 %**), and those are fully explained, quantitatively,
by a mechanism that is **an artifact of a stale baseline configuration, not a loss of
information**: the shipped dataset's raycast mesh was `40×30` on exactly the domain the
ETKF's analysis grid covers at its `'medium'` setting, so the simulator's low-pass filter
and the ETKF's own bilinear observation operator were the *same operator*
(|β_obs,old − β_H| = 0.024 mean over 12 episodes). Removing the renderer's low-pass breaks
that coincidence and changes how much variance the filter injects. Since the ETKF's field
correlation with truth is ρ ≈ 0.00–0.10 at **every** configuration tested on **both**
datasets, its nRMSE is a measure of injected variance, not of reconstruction skill, and the
coarse→fine change moves that variance rather than any information content.

---

## 1. Two premises in the brief were wrong; correcting them changes the question

**MEASURED.**

| brief said | actually |
|---|---|
| "The GT grid changed from 40×30 to 200×167" | The **GT grid is 80×60 in both datasets** and `frames/gt_wave_field` is **bit-identical** (max abs diff exactly 0.0 on every episode checked). 200×167 is the *raycast* mesh. |
| "ETKF gets WORSE" (systematic) | Aggregate change is +0.0123 nRMSE, t(11)=+0.91 — **inside noise**. |
| `Hs_true` might differ | Identical (0.30 / 0.60 / 1.20 / 2.00 by sea state, both datasets). |

⟹ **The only thing that changed for any estimator is the `z` value of the LiDAR returns.**
Point counts, point geometry, the GT field and the metric normalisation are unchanged.
This rules out H-NORM outright and makes "interpolation to a changed GT grid" a non-issue.

How much did `z` change? Regressing the observed `z` on the analytic truth at the same
`(x, y, t)` gives the transmission β and the correlation r of the observation channel:

| episode | λ_peak | λ_min | β_old | β_new | β_H(40×30) |
|---|---|---|---|---|---|
| BF2 1comp U | 8.16 | 8.16 | **0.066** | 0.924 | **0.084** |
| BF3 1comp U | 15.42 | 15.42 | **0.551** | 0.978 | **0.572** |
| BF3 2comp U | 37.32 | 8.37 | 0.919 | 0.995 | 0.883 |
| BF3 2comp S | 37.32 | 8.37 | 0.917 | 0.996 | 0.884 |
| BF3 5comp U | 30.23 | 6.25 | 0.856 | 0.990 | 0.795 |
| BF3 5comp S | 30.23 | 6.25 | 0.851 | 0.991 | 0.796 |
| BF4 1comp U | 24.98 | 24.98 | 0.803 | 0.991 | 0.816 |
| BF4 5comp U | 48.96 | 10.12 | 0.925 | 0.996 | 0.899 |
| BF4 15comp U | 65.18 | 8.99 | 0.926 | 0.996 | 0.919 |
| BF4 30comp U | 70.49 | 8.74 | 0.942 | 0.997 | 0.929 |
| BF5 15comp U | 107.15 | 14.87 | 0.972 | 0.999 | 0.971 |
| BF5 30comp U | 115.54 | 14.45 | 0.974 | 0.999 | 0.969 |

(observation-truth correlation r rises from 0.10–1.00 (old) to ≥ 0.996 (new)).

`β_H` is the attenuation of the **ETKF's own observation operator** — analytic truth sampled
at the ETKF's 40×30 analysis-grid nodes, then bilinearly interpolated to the observation
points. It is a property of the baseline, not of the data.

---

## 2. The mechanism (MEASURED, then one ARGUED step)

### 2.1 The shipped dataset committed an inverse crime against this baseline

**MEASURED.** `configs/default.yaml` renders the raycast surface at `mesh.nx=40, mesh.ny=30`
over `x∈[10,250], y∈[−100,100]` — **numerically the same grid** as
`run_p1_comparison.run_etkf`'s `res_map['medium'] = (40,30)` over `gt_grid_x[0..-1]`,
`gt_grid_y[0..-1]`. Both apply piecewise-linear interpolation on that grid. Consequently:

    mean |β_obs,old − β_H(40×30)| = 0.024   (max 0.061, n=12)
    mean |β_obs,new − β_H(40×30)| = 0.195

On the shipped data the ETKF's forward operator **was** the data-generating operator. On the
fine-mesh data it is not. This is a structural defect of the old dataset+baseline pair that
is worth disclosing in the paper's data section regardless of what it does to the numbers.

**Independent re-derivation** (dual-verification discipline — second route, different
statistic). Correlate the *actually observed* `z` against the ETKF's own forward operator
applied to the analytic truth, `H_{40×30}[truth]`, versus against the truth itself:

| episode | OLD: corr(z, H[truth]) | OLD: corr(z, truth) | NEW: corr(z, H[truth]) | NEW: corr(z, truth) |
|---|---|---|---|---|
| BF2 1comp | **0.995** | 0.078 | 0.127 | 0.996 |
| BF3 1comp | **0.997** | 0.890 | 0.910 | 0.999 |
| BF3 5comp | **0.990** | 0.946 | 0.954 | 0.999 |
| BF4 15comp | **0.998** | 0.985 | 0.988 | 1.000 |
| BF5 30comp | **0.999** | 0.998 | 0.998 | 1.000 |

The shipped observations reproduce the ETKF's own observation operator to within 1 %, while
in the worst case bearing correlation 0.078 with physical truth. Textbook inverse crime.

### 2.2 With ρ ≈ 0, nRMSE is a variance meter, and the swing is a variance change

**MEASURED.** At the as-published config, ρ(pred, GT) and the variance ratio VR = σ_pred/σ_GT,
median of 7 filter seeds:

| episode | ρ_old | ρ_new | VR_old | VR_new | Δ nRMSE (median-of-7) |
|---|---|---|---|---|---|
| BF2 1comp U | 0.001 | 0.000 | 0.98 | **0.63** | **−0.1094** |
| BF3 1comp U | 0.022 | 0.030 | 0.88 | **1.45** | **+0.1025** |
| BF3 2comp U | 0.046 | 0.038 | 1.29 | 1.39 | +0.0295 |
| BF3 2comp S | 0.079 | 0.078 | 1.00 | 1.04 | +0.0144 |
| BF3 5comp U | 0.052 | 0.046 | 1.24 | 1.34 | +0.0117 |
| BF3 5comp S | 0.075 | 0.075 | 0.95 | 1.05 | +0.0198 |
| BF4 1comp U | 0.040 | 0.043 | 1.17 | 1.41 | +0.0550 |
| BF4 5comp U | 0.080 | 0.077 | 1.23 | 1.41 | +0.0356 |
| BF4 15comp U | 0.055 | 0.052 | 1.52 | 1.55 | +0.0148 |
| BF4 30comp U | 0.082 | 0.103 | 1.41 | 1.47 | +0.0069 |
| BF5 15comp U | 0.098 | 0.078 | 1.35 | 1.29 | +0.0155 |
| BF5 30comp U | 0.072 | 0.078 | 1.40 | 1.40 | +0.0067 |

ρ does not improve on the fine mesh anywhere (it moves ±0.02, no sign structure). So:

    nRMSE ≈ (σ_GT/H_s) · sqrt(1 − 2ρ·VR + VR²)   with ρ ≈ 0
          ≈ 0.25 · sqrt(1 + VR²)

**ARGUED (one step), then MEASURED.** If the filter's grid can represent the wave, matching
observations that are no longer attenuated forces the grid amplitude up by exactly the factor
β_new/β_old, so VR should scale by that factor. Test:

    corr( predicted VR ratio = β_new/β_old , measured VR_new/VR_old ) = +0.975   (n = 11)

and the ΔnRMSE the model predicts tracks the measured one episode by episode
(BF3 1comp: predicted +0.132 vs measured +0.103; BF4 1comp: +0.055 vs +0.055;
BF3 2comp S: +0.015 vs +0.014; BF5 30comp: +0.009 vs +0.007).

**The one excluded episode is the one that proves the mechanism.** BF2 1comp
(λ = 8.16 m) sits **below** the 40×30 analysis grid's Nyquist limit (2·dy = 13.8 m), so the
"amplify until H matches the observations" channel does not exist — the grid has no mode to
put that wave in. Its VR therefore *falls* (0.98 → 0.63) and its nRMSE *improves* by 22 %,
i.e. it moves toward the trivial zero predictor. On the shipped data the same wave had been
**aliased by the renderer into the grid-representable band**, so the filter chased the alias
and injected variance. That is why the two 1-component episodes move in opposite directions.

---

## 3. Verdict: artifact or genuine?

**Artifact — of the baseline's configuration, not of the data; and smaller than the
paper's framing implies.**

- The regression is **not** a loss of information: the fine-mesh observations are strictly
  better (r 0.10–1.00 → ≥ 0.996) and the ETKF's skill ρ does not fall.
- It is **not** a systematic regression at all at the aggregate level (t = +0.91, n.s.).
- The two resolvable per-episode swings are a **representation mismatch** between the
  baseline's 40×30 analysis grid and a GT field it cannot represent — exactly the
  handicap the brief hypothesised, plus the additional finding that the *old* number was
  flattered by an inverse crime (the renderer and the baseline shared an operator).
- **But the correction is small in practice**, because this baseline has ρ ≈ 0 at every
  configuration tested. Its nRMSE was never measuring reconstruction skill, on either
  dataset. The paper already says as much for the old data
  (`wavesplat.tex` L965/L984: ETKF VR 0.898, ρ 0.109, SS −0.70, "variance-correct but
  shape-incoherent"); the fine-mesh data does not change that characterisation.

### 3.1 Why ρ ≈ 0 — a structural property of this baseline, not of either dataset

**MEASURED.** The ETKF ensemble is initialised as `np.random.randn(Ne, n_state) * 0.1`
(white), the additive inflation adds white noise every cycle, the analysis only re-mixes
existing members (`X_pert_new = W @ X_pert`), and the spectral forecast is a per-mode
rotation that preserves the spectrum. So the background-error covariance **stays white
forever**. After 301 assimilation cycles on fine-mesh data, the spatial autocorrelation of
the ensemble perturbations at one grid spacing (6.15 m):

| lag (cells, ≈6.1 m) | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| BF4 15comp — ETKF ensemble | 1.000 | **0.083** | 0.049 | 0.040 | 0.027 | 0.025 |
| BF4 15comp — true GT field | 1.000 | **0.720** | 0.303 | −0.089 | −0.394 | −0.554 |
| BF3 1comp — ETKF ensemble | 1.000 | **0.079** | 0.029 | 0.029 | 0.025 | 0.014 |
| BF3 1comp — true GT field | 1.000 | **−0.783** | 0.225 | 0.431 | −0.899 | 0.976 |

With a delta-correlated **B**, each observation can only correct its own cell. There are
~40 assimilated observations per frame against 1200 grid cells, so at most ~3 % of the field
is ever touched, and never coherently. **That, not the mesh, is why ρ ≈ 0 on both datasets.**

### 3.2 Did we handicap it? — ceiling probe

**MEASURED** (`tmp/etkf_ceiling_2026-07-24.py`, fine-mesh data, seed 0). Best-case
combinations well beyond the paper's search space: fine analysis grid, **JONSWAP-shaped
background covariance instead of white noise**, additive inflation off, `Ne` up to 200,
observation thinning disabled, `R` down to the sensor spec.

| config | BF3 1comp | BF3 5comp | BF4 15comp |
|---|---|---|---|
| | nRMSE / ρ | nRMSE / ρ | nRMSE / ρ |
| white 40×30 (as-published) | 0.4548 / **0.056** | 0.4181 / 0.032 | 0.4711 / 0.060 |
| white 160×120, ai=0, Ne=200 | **0.2824** / 0.020 | **0.2816** / 0.016 | **0.2926** / 0.032 |
| prior-B 40×30, ai=0, Ne=100, no thinning | 0.3383 / 0.041 | 0.3584 / 0.019 | 0.4837 / 0.016 |
| prior-B 80×60, ai=0, Ne=200, no thinning | 0.3377 / **0.067** | 0.3891 / **0.041** | 0.4707 / 0.038 |
| prior-B 160×120, ai=0, Ne=200, no thinning | 0.3139 / 0.059 | 0.3232 / 0.019 | 0.4017 / **0.043** |
| prior-B 80×60, Ne=200, R=0.02 | 0.3475 / 0.065 | 0.4103 / 0.040 | 0.4937 / 0.036 |
| *zero predictor* | *0.2500* / *0* | *0.2500* / *0* | *0.2511* / *0* |

**No configuration reaches ρ > 0.07.** The best nRMSE any of them achieves (0.282–0.293) is
the zero-predictor value. Giving the filter a physically-correct background covariance —
which is a *change to the baseline*, not a tuning knob — buys ρ ≈ 0.02 → 0.06 and no more.

⟹ **We are not handicapping the baseline. It is structurally incapable of reconstructing
this field**: ~50 returns per frame over a 240 × 200 m domain is a mean point spacing of
~31 m against wavelengths of 8–115 m, and a sequential grid filter with a delta-correlated
**B** has no mechanism to fill the gaps. This is the paper's own thesis
(`etkf_grid.py` docstring: *"demonstrates the grid interpolation limitation that WaveSplat
overcomes"*) — the fine-mesh data does not weaken it, and the retune below does not rescue
the baseline.

### 3.3 Representation floor of the analysis grid

**MEASURED.** Best achievable nRMSE for a *perfect* estimator restricted to an (nx,ny)
analysis grid (analytic GT sampled at the nodes, bilinearly interpolated to the 80×60 GT
grid). Identical for both datasets because the GT is identical. Zero-predictor = 0.25–0.29.

| episode | 20×15 | 40×30 | 60×45 | 80×60\* | 120×90 | 160×120 | zero-pred |
|---|---|---|---|---|---|---|---|
| BF2 1comp | 0.2762 | **0.3026** | 0.1782 | 0.0000 | 0.0519 | 0.0298 | 0.2500 |
| BF3 1comp | 0.3233 | 0.1238 | 0.0586 | 0.0000 | 0.0151 | 0.0085 | 0.2500 |
| BF3 2comp | 0.1015 | 0.0569 | 0.0318 | 0.0000 | 0.0090 | 0.0052 | 0.2500 |
| BF3 5comp | 0.1547 | 0.0825 | 0.0480 | 0.0000 | 0.0154 | 0.0090 | 0.2500 |
| BF4 5comp | 0.1075 | 0.0436 | 0.0226 | 0.0000 | 0.0062 | 0.0036 | 0.2499 |
| BF4 15comp | 0.0869 | 0.0365 | 0.0184 | 0.0000 | 0.0049 | 0.0028 | 0.2511 |
| BF4 30comp | 0.0846 | 0.0352 | 0.0179 | 0.0000 | 0.0049 | 0.0028 | 0.2636 |
| BF5 30comp | 0.0490 | 0.0157 | 0.0073 | 0.0000 | 0.0019 | 0.0011 | 0.2883 |

\*80×60 coincides with the GT grid, so its floor is exactly 0 by construction — a sanity
check on the measurement, not a recommendation.

Two readings. (a) Grid representation is **not** the binding constraint anywhere except
BF2 1comp, where the 40×30 floor (0.3026) is **above the zero predictor** — at that grid no
honest estimator can beat predicting zero. (b) The ETKF's measured 0.35–0.52 sits an order
of magnitude above its own floor everywhere else, i.e. the grid is not what is stopping it.

---

## 4. Re-tune protocol (honest hold-out)

### 4.0 Protocol, fixed before the sweep

| | |
|---|---|
| **Tuning episodes** | the 12 paper scenarios at **seed 1**: `episode_{BF2_1comp,BF3_1comp,BF3_2comp,BF3_5comp,BF4_1comp,BF4_5comp,BF4_15comp,BF4_30comp,BF5_15comp,BF5_30comp}_seed1_uniform.h5` + `BF3_2comp_seed1_slope.h5` + `BF3_5comp_seed1_slope.h5`, all from `data_finemesh_2026-07-23/` |
| **Reporting episodes** | the same 12 scenarios at **seed 0** — never used for any selection decision |
| **Selection metric** | mean over the 12 tuning episodes of the per-episode **median over 3 filter seeds** of frame-mean nRMSE |
| **Reported numbers** | per-episode **median over 7 filter seeds** {0..6}, with min–max spread |
| **Search** | staged coordinate search: A1 grid × additive inflation → A2 obs_noise × N_e × multiplicative inflation → A3 max_obs |

This fixes a defect independent of the mesh: the paper's own sweep (`tab:etkf_sweep`) was
scored on the same 12 episodes it reports.

### 4.1 Stage A1 — analysis grid × additive inflation

MEASURED, tuning set, mean nRMSE (median of 3 filter seeds), `Ne=50, mi=1.01, R=0.05,
max_obs=40`:

| rank | grid | add_infl | mean nRMSE | mean ρ | ms/frame |
|---|---|---|---|---|---|
| 1 | 160×120 | 0.0 | **0.2603** | 0.017 | 174.5 |
| 2 | 120×90 | 0.0 | 0.2656 | 0.023 | 87.2 |
| 3 | 60×45 | 0.0 | 0.2834 | 0.034 | 10.7 |
| 4 | 40×30 | 0.0 | 0.2994 | 0.048 | 3.7 |
| 5 | 80×60 | 0.0 | 0.3071 | 0.020 | 25.6 |
| 6 | 160×120 | 0.005 | 0.3163 | 0.017 | 214.3 |
| 9 | **40×30 | 0.005** (as-published grid/inflation) | 0.3978 | 0.056 | 6.0 |
| 11–15 | any grid | 0.02 | 0.650–0.978 | 0.011–0.061 | — |

Two things to read here, and they point in opposite directions:

1. **Additive inflation is the single biggest knob and the shipped value is too large.**
   Setting `add_inflation = 0` improves the shipped grid from 0.3978 to 0.2994 (−25 %);
   `0.02` is catastrophic (0.65–0.98), which independently reproduces the worst rows of the
   paper's own `tab:etkf_sweep` (0.986/1.066/1.187 at 80×60, σ=0.02).
2. ⚠ **The selection criterion is degenerate in this regime.** Every configuration that
   lowers nRMSE also lowers ρ (0.056 → 0.017): with the filter at ρ ≈ 0, `nRMSE ≈
   0.25·sqrt(1+VR²)` is minimised by making the field *smaller*, not better. The "best"
   ETKF configuration is the one that most closely approximates predicting zero. This is
   exactly the pathology the repo's own metric-panel docstring warns about
   (`wavesplat/evaluation/metrics.py`, "a trivial zero predictor … achieves nRMSE ≈ 0.25").

### 4.2 Stages A2 (obs_noise × N_e × mi) and A3 (max_obs)

MEASURED, tuning set. A2 top rows all sit at 160×120, ai=0, mi=1.0; larger `obs_noise`
helps slightly (R=0.20 best, 0.2515) and `Ne=100` does **not** beat `Ne=50` here. A3 picks
`max_obs=20` (0.2503) over 40/100 by a hair. Every step lowers nRMSE by lowering ρ — the
degeneracy of §4.1 persists through the whole search.

### 4.3 Final recommended fine-mesh ETKF configuration

    grid 160×120,  Ne=50,  mult_inflation=1.0,  add_inflation=0.0,
    obs_noise=0.20,  max_obs=20
    (tuning-set mean nRMSE 0.2503; ~148 ms/frame)

### 4.4 Report set (seed 0, held out from tuning) — median of 7 filter seeds

| episode | as-published (fine) | paper-tuned* (fine) | **re-tuned (fine)** | ρ re-tuned | re-tuned (old) | paper-tuned* (old) |
|---|---|---|---|---|---|---|
| BF2 1comp U | 0.3792 | 0.3516 | **0.2550** [.2543,.2566] | 0.011 | 0.2525 | 0.4250 |
| BF3 1comp U | 0.4548 | 0.4002 | **0.2559** [.2553,.2571] | 0.010 | 0.2517 | 0.3261 |
| BF3 2comp U | 0.4462 | 0.3710 | **0.2562** | 0.012 | 0.2554 | 0.3627 |
| BF3 2comp S | 0.3583 | 0.3256 | **0.2541** | 0.016 | 0.2537 | 0.3172 |
| BF3 5comp U | 0.4186 | 0.3719 | **0.2559** | 0.010 | 0.2555 | 0.3604 |
| BF3 5comp S | 0.3597 | 0.3378 | **0.2539** | 0.013 | 0.2538 | 0.3193 |
| BF4 1comp U | 0.4442 | 0.3787 | **0.2561** | 0.003 | 0.2537 | 0.3412 |
| BF4 5comp U | 0.4380 | 0.3694 | **0.2551** | 0.026 | 0.2543 | 0.3488 |
| BF4 15comp U | 0.4751 | 0.3898 | **0.2572** | 0.024 | 0.2576 | 0.3767 |
| BF4 30comp U | 0.4579 | 0.3908 | **0.2721** | 0.019 | 0.2711 | 0.3859 |
| BF5 15comp U | 0.4574 | 0.3702 | **0.2658** | 0.014 | 0.2665 | 0.3727 |
| BF5 30comp U | 0.4994 | 0.4274 | **0.2957** | 0.021 | 0.2951 | 0.4181 |
| **mean** | **0.4324** | **0.3737** | **0.2611** | 0.015 | **0.2601** | **0.3628** |

\*"paper-tuned" = `tab:etkf_sweep` winner (40×30, Ne=100, mi=1.00, ai=0.005), which is
**not** the config `run_p1_comparison.py` actually runs (that is "as-published", Ne=50).

Two results, one of them a caution:

- **The re-tuned ETKF is mesh-invariant** (fine 0.2611 vs old 0.2601, Δ = +0.0010). Tuning
  on the fine mesh and re-scoring on the old mesh gives the same number. So a *fairly*-tuned
  ETKF has no coarse→fine regression at all — the swing in §2 was entirely an artifact of a
  configuration (small analysis grid + additive inflation) that let the innovation amplify
  the now-unattenuated observations. This is the clean confirmation the coordinator asked
  for: the per-episode swing sits inside what re-tuning removes.
- ⚠ **But the "re-tuned" number is a near-zero predictor and must be reported as such.**
  Its ρ collapses to 0.015 (vs 0.098 for the paper-tuned config on old data). Its nRMSE 0.261
  is the zero-predictor value 0.25–0.29; the tuner reached it by shrinking the field to
  nothing, exactly the GPR pathology the paper already calls out. **Reporting the re-tuned
  0.261 as "ETKF got better" would be misleading.** The honest statement is:
  *when the ETKF baseline is tuned purely on nRMSE it degenerates to a zero predictor;
  a fair comparison must therefore be made on the full metric panel (ρ, SS, VR), on which
  this configuration scores as badly as GPR.*

---

## 4.5 Supplementary — LETKF(loc50) is completely mesh-invariant

Not requested, but decisive for the margin question and cheap, so it was measured
(`tmp/letkf_finemesh_2026-07-24.py`; the repo's own `LETKFGrid` at the exact configuration
`scripts/run_strong_baseline_comparison.py` uses — 40×30, Ne=100, mi=1.0, ai=0.005,
`loc_radius=50` — median of 5 filter seeds).

| episode | OLD nRMSE / ρ | FINE nRMSE / ρ | Δ |
|---|---|---|---|
| BF2 1comp U | 0.2557 / 0.007 | 0.2544 / 0.001 | −0.0013 |
| BF3 1comp U | 0.2441 / 0.212 | 0.2442 / 0.219 | +0.0001 |
| BF3 2comp U | 0.2354 / 0.328 | 0.2359 / 0.324 | +0.0005 |
| BF3 2comp S | 0.2190 / 0.487 | 0.2190 / 0.479 | +0.0001 |
| BF3 5comp U | 0.2369 / 0.314 | 0.2370 / 0.313 | +0.0001 |
| BF3 5comp S | 0.2254 / 0.436 | 0.2254 / 0.429 | +0.0001 |
| BF4 1comp U | 0.2379 / 0.305 | 0.2377 / 0.300 | −0.0001 |
| BF4 5comp U | 0.2338 / 0.349 | 0.2337 / 0.350 | −0.0001 |
| BF4 15comp U | 0.2292 / 0.406 | 0.2288 / 0.410 | −0.0005 |
| BF4 30comp U | 0.2361 / 0.443 | 0.2319 / 0.474 | −0.0042 |
| BF5 15comp U | 0.2313 / 0.449 | 0.2295 / 0.465 | −0.0018 |
| BF5 30comp U | 0.2636 / 0.405 | 0.2637 / 0.404 | +0.0001 |
| **mean** | **0.2374** | **0.2368** | **−0.0006** (SE 0.0004, t = −1.56) |

Three consequences.

1. **The strongest baseline in the paper does not move on the fine-mesh data at all**
   (Δ = −0.0006, four decimal places of agreement on 8/12 episodes). Whatever the paper says
   about LETKF stands unchanged under regeneration.
2. It **confirms the §2 mechanism from the opposite side**: LETKF differs from ETKF here
   almost only by the Gaspari–Cohn taper on the gain, and the taper is exactly what stops
   the unlocalised amplification of the innovation. Remove the amplification channel and the
   mesh sensitivity disappears.
3. LETKF carries **real** field correlation (ρ 0.21–0.49) where the vanilla ETKF does not
   (ρ 0.02–0.10) — so the paper's ETKF row should not be read as "the state of the art in
   grid assimilation cannot do this"; the localised version can, it is just heavily shrunk
   (its nRMSE ≈ 0.237 sits just below the zero predictor because VR is small).

---

## 5. Does the headline margin over ETKF survive?

**Yes — it widens against any honestly-configured ETKF, and holds even against the
degenerate nRMSE-minimising one.** Using the parallel WaveSplat-P1 fine-mesh run
(`tmp/rerun_2026-07-24/ws_runs.jsonl`, median of 5 estimator seeds; **provenance: another
workstream, not produced here — treat as provisional until that run is finalised**), all on
the 12 report episodes, fine-mesh data:

| comparison | ETKF mean | WS-P1 mean | margin | paired t(11) | WS lower in |
|---|---|---|---|---|---|
| WS vs ETKF **as-published** (fine) | 0.4324 | 0.1821 | **+57.9 %** | — | 12/12 |
| WS vs ETKF **paper-tuned** (fine) | 0.3737 | 0.1821 | **+51.3 %** | **+6.90** | 12/12 |
| WS vs ETKF **re-tuned, nRMSE-degenerate** (fine) | 0.2611 | 0.1821 | **+30.2 %** | +2.82 | 8/12 |
| WS vs **LETKF(loc50)** (fine) | 0.2368 | 0.1821 | +23.1 % | — | — |

Reading:

- Against the ETKF the paper actually cites (its `tab:etkf_sweep` winner), the margin on
  fine-mesh data is **+51 %, paired t = 6.9, 12/12** — larger than the paper's headline
  30.1 %, because the fine mesh helps WaveSplat on the sparse episodes while leaving the
  properly-tuned ETKF flat.
- The **only** way to shrink the margin to ~30 % is to compare against the re-tuned ETKF —
  but that configuration has ρ = 0.015, i.e. it bought its nRMSE by becoming a zero
  predictor. On the metric panel the paper already uses (SS, ρ, VR) it is not a
  reconstruction at all, and WaveSplat's advantage there is categorical, not 30 %.
- So the paper's claim survives regeneration in the form: *WaveSplat beats every ETKF
  configuration that actually reconstructs a field, and the coarse-mesh headline was, if
  anything, conservative.* The one thing that must change is the **framing**: quote the
  margin against a **named, honestly-configured** ETKF and defend it on the full metric
  panel, because a pure-nRMSE tuning of the baseline drives it to a zero predictor and any
  single-number nRMSE margin is therefore soft.

**Recommended for the paper.** Do not headline the re-tuned 0.261 as "ETKF" without its ρ.
The defensible fine-mesh baseline number is the **paper-tuned config re-scored on the fine
mesh, 0.374** (mesh-invariant to ±0.001; it is 0.363 on old data), reported **with its
ρ = 0.10 and SS**. That keeps the comparison honest on both axes and preserves a ~51 %
nRMSE margin with a decisive skill-score gap.


---

## 6. Limits, and what would close what is still open

**Harness validity (checked, three ways).** `tmp/etkf_lib.py::FastETKF` is a
batched-observation-operator rewrite of `ETKFGrid`.
(i) It reproduces the stock class to 2.9e-15 nRMSE on a full episode.
(ii) It reproduces the coordinator's independent as-published run **exactly** at seed 42
(BF3 1comp old 0.3519 / fine 0.4436; BF2 1comp old 0.4831 / fine 0.3956).
(iii) It reproduces the **paper's own published ETKF sweep** on the old data at seed 42:
best row 40×30/Ne100/mi1.00/ai0.005 → **0.3584** (paper `tab:etkf_sweep`: 0.358);
worst row 80×60/Ne50/mi1.05/ai0.02 → **1.1871** (paper: 1.187); ρ 0.098 (paper
`tab:fivemetric`: 0.109).

**Limits.**
1. Synthetic linear (Gerstner) episodes only, `seed0` for reporting and `seed1` for tuning,
   one episode per scenario. No HOS/nonlinear replication.
2. The tuning search is a **staged coordinate search**, not a full factorial; a joint optimum
   outside the visited path could exist. Stage A1 fixes `obs_noise`/`Ne`/`mi` at shipped
   values while choosing grid and additive inflation, which is where an interaction would
   most plausibly hide.
3. `run_etkf`'s nRMSE is the frame-mean of per-frame nRMSE including the filter's spin-up
   transient, matching `run_p1_comparison.py`. No warm-up was discarded, for comparability.
4. The variance-ratio model in §2.2 is a scalar-gain argument; it explains 11/12 episodes
   (r = 0.975) and explicitly does not apply to the sub-Nyquist episode, which is handled
   separately and consistently.
5. ρ here is the whole-field Pearson correlation on the 80×60 GT grid, frame-averaged — the
   same definition the paper's metric panel uses.

**Still open / what would close it.**
- **The ETKF baseline is structurally incapable of field reconstruction in this setting**
  (§3.1: white background covariance, ~40 obs against 1200+ cells). If a reviewer asks
  "is the baseline a straw man?", the honest answer is that it is the baseline the paper
  ported from \citet{desmars2020} with a white-noise ensemble initialisation. Closing this
  properly requires a background-error covariance drawn from the wave prior *and* spatial
  localisation, i.e. changing the baseline, not tuning it. §3.2's ceiling probe measures how
  much that would buy; anything beyond that is a new baseline and should be labelled as one.
- **The paper's own ETKF tuning was performed on the episodes it reports** (`tab:etkf_sweep`:
  "mean nRMSE over the 12 sweep episodes" = the 12 reported episodes). That is an
  independent defect from the mesh question and it is not fixed by regeneration — it is
  fixed by the hold-out split used here.
- **Margin claims** use fine-mesh WaveSplat-P1 numbers (median of 5 estimator seeds) from a
  separate workstream (`tmp/rerun_2026-07-24/ws_runs.jsonl`); all 12 report episodes were
  present when §5 was computed, but they are that run's numbers, not re-derived here — treat
  as provisional until that run is finalised and re-confirm §5 against its final table.
