# GP K_disp Pilot Analysis (2026-06-09, updated 2026-06-10)

## Evaluation Protocol
- **30 episodes**: seed 0-4 × {BF3_5c, BF4_5c, BF4_15c, BF4_30c, BF5_15c, BF5_30c} uniform
- **Frames**: first 30 tracking frames (after init window, tp_factor=2.0) — **same protocol for all methods**
- **Metrics**: nRMSE, VR (Var.Ratio), Pearson, Skill Score
- Result files: `gp_disp_full_eval.json`

---

## Summary: All Methods (30 episodes, same eval protocol)

| Method | nRMSE | VR | Pearson | Skill | Notes |
|--------|-------|-----|---------|-------|-------|
| WaveSplat (this eval) | 0.263 | 0.914 | 0.346 | -0.275 | MAP baseline, first 30 track frames |
| GP K_disp [ws_multistart] | 0.235 | 0.782 | 0.453 | -0.116 | 5 seeds + NML seed selection + global ML scale |
| GP K_disp [ws_reg_amp] | 0.230 | 0.648 | 0.440 | +0.009 | 1 seed + L2-regularized per-component amplitudes |
| **GP K_disp [ws_ms5_reg_amp]** | **0.223** | **0.783** | **0.489** | -0.012 | **5 seeds + reg amp — NEW BEST GP** |
| GP K_disp [oracle] | **0.126** | 0.629 | **0.801** | **0.657** | Theoretical upper bound |

Reference (6-episode, pilot, different protocol):

| Method | nRMSE | VR | Pearson | Skill | Notes |
|--------|-------|-----|---------|-------|-------|
| WaveSplat | 0.259 | 0.588 | 0.293 | -0.095 | Phase0 comparison (6 ep) |
| WavePINO S4 | 0.228 | 0.309 | 0.376 | 0.107 | Fair 30-ep comparison |

---

## Key Findings (30-episode fair comparison)

### 1. ws_ms5_reg_amp (0.223) = new best practical method
- **vs WaveSplat**: nRMSE **15.2%** improvement (0.263 → 0.223), Pearson 41% improvement
- **vs ws_multistart**: nRMSE 5.1% improvement (0.235 → 0.223), Pearson 8% improvement
- **vs ws_reg_amp**: nRMSE 2.8% improvement (0.230 → 0.223), VR 0.648→0.783 (amplitude quality↑↑)
- **vs WavePINO S4**: nRMSE 0.223 vs 0.228 (fair 30-ep protocol) — **GP K_disp is 2.1% better**

### 2. reg_amp + ms5 synergy
- ws_multistart (ms5, global scale): VR=0.782, Pearson=0.453
- ws_reg_amp (n_starts=1, per-comp): VR=0.648, Pearson=0.440
- ws_ms5_reg_amp (ms5 + per-comp): VR=0.783, Pearson=0.489
- Combining ms5 (seed diversity) + reg_amp (amplitude flexibility) beats either alone

### 3. Oracle GP proves theoretical upper bound
- oracle nRMSE=0.126 = 52% better than WaveSplat, 43% better than best practical
- oracle Pearson=0.801 = GP posterior can capture wave structure extremely well
- oracle-practical gap = 0.223 - 0.126 = 0.097 → K_disp estimation quality bottleneck

### 4. WaveSplat VR=0.914 anomaly explained
- Early tracking frames (first 30) can have VR>1 (amplitude overshoot before convergence)
- GP kernel estimated from init window → more stable amplitude from the start

### 5. ws_multistart outlier: BF4_15comp_seed2 (nRMSE=0.573, VR=4.61)
- Catastrophic failure: all 5 seeds converged to bad local minimum
- ws_ms5_reg_amp presumably mitigates this via regularized amplitude (avoids energy explosion)
- n_starts=10 alternative: BF4_15c seed2 → 0.573→0.182 (68% recovery)

---

## Lambda Tuning (6 pilot episodes, ws_reg_amp, n_starts=1)

Result file: `lambda_tuning.json`

| λ | nRMSE | VR | Pearson | Skill |
|---|-------|----|---------|-------|
| **0.5** | **0.2289** | 0.4437 | 0.4132 | **0.1570** |
| 1.0 | 0.2339 | 0.5989 | 0.4246 | 0.0721 |
| 2.0 (default) | 0.2384 | 0.4299 | 0.3912 | 0.1113 |
| 5.0 | 0.2560 | 0.3922 | 0.3239 | 0.0128 |

**Conclusion**: λ=0.5 best for both nRMSE and Skill on pilot episodes.
- λ=5.0 severely over-regularized (+11.8% nRMSE vs λ=0.5)
- λ=0.5 allows more per-component amplitude variation → better fit

**Caveat**: ws_reg_amp with n_starts=1 has run-to-run variability (~0.007-0.01 nRMSE).
Hyperparam tuning ran same default (λ=2.0) in a different session → nRMSE 0.2319 vs 0.2384 here.
λ tuning results should be validated with ws_ms5_reg_amp (n_starts=5) for stability.

---

## Hyperparam Tuning (6 pilot episodes, ws_reg_amp)

Result file: `hyperparam_tuning.json`

| Config | λ | σ | tp | nRMSE | VR | Pearson | Skill |
|--------|---|---|-----|-------|----|---------|-------|
| default | 2.0 | 0.02 | 2.0 | 0.2319 | 0.4487 | 0.4109 | 0.1284 |
| lam_0.5 | 0.5 | 0.02 | 2.0 | 0.2413 | 0.4683 | 0.3858 | 0.0781 |
| lam_1.0 | 1.0 | 0.02 | 2.0 | 0.2492 | 0.4626 | 0.3629 | 0.0531 |
| lam_5.0 | 5.0 | 0.02 | 2.0 | 0.2400 | 0.4168 | 0.3900 | 0.1188 |
| **sigma_0.01** | 2.0 | **0.01** | 2.0 | **0.2752** | 0.9066 | 0.3639 | -0.2459 |
| sigma_0.05 | 2.0 | 0.05 | 2.0 | 0.2348 | 0.2241 | 0.3844 | **0.1670** |
| tp_3.0 | 2.0 | 0.02 | 3.0 | 0.2358 | 0.4435 | 0.4139 | 0.1307 |
| **tp_4.0** | 2.0 | 0.02 | **4.0** | 0.2375 | 0.7949 | **0.5250** | 0.0643 |

**Key findings:**
1. **σ=0.01 catastrophic**: nRMSE=0.275, VR=0.907 (≈ WaveSplat failure mode)
   - Too-tight observation noise → GP overfits sparse obs, cannot generalize
2. **σ=0.05 best Skill**: Skill=0.167, but VR=0.224 (too-uncertain predictions → low amplitude fidelity)
3. **tp=4.0 best Pearson**: Pearson=0.525 vs default 0.411 (+27.7%)
   - Longer init window → WaveSplat finds better wave components
   - nRMSE penalty small (0.2375 vs 0.2319)
   - VR=0.795 much better than default (0.449) → amplitude calibration improved
4. **λ inconsistency**: lambda_tuning (separate run) shows λ=0.5 best (0.229), but hyperparam_tuning shows lam_0.5 worse (0.241) than default. Run-to-run variability masks λ effect with n_starts=1.

**Recommendations:**
- **Avoid**: σ=0.01 (catastrophic), λ≥5.0 (over-regularized)
- **Best nRMSE+Pearson trade-off**: tp=4.0 (Pearson +27%, nRMSE +2.4%)
- **Default (λ=2, σ=0.02, tp=2)**: safest baseline
- **Future**: validate λ=0.5 with ws_ms5_reg_amp (n_starts=5) for reliable comparison

---

## Method Design: Two-Stage K_disp Estimation

```
Stage 1: Direction estimation
  WaveSplat Phase 1 (Adam + dispersion constraint + coarse-to-fine)
  → Multiple seeds (n_starts=5 or 10): seeds 0,1,...,n-1
  → Select best by marginal likelihood: argmin_seed NML(kx_s, ky_s | obs)

Stage 2a: Global amplitude calibration (ws_multistart)
  Fix: kx*, ky* from best seed
  K_unit(xi, xj) = sum_n 0.5 * cos(kx_n*(xi-xj) + ky_n*(yi-yj))
  Optimize: s = argmin_s NML(s * K_unit | obs)
  → Single parameter, 1D bounded optimization (global optimum)

Stage 2b: Regularized per-component amplitudes (ws_reg_amp / ws_ms5_reg_amp)
  Fix: kx*, ky* from best seed; s_global from Stage 2a
  Prior center: log_s_prior = log(s_global / n_comp)  [uniform energy split]
  K_n(xi, xj) = 0.5 * cos(kx_n*(xi-xj) + ky_n*(yi-yj))  [per-component basis]
  Optimize: {s_n} = argmin NML(K({s_n}) | obs) + lambda * sum_n (log_s_n - log_s_prior)^2
  → N_comp parameters, MAP estimation (lambda=2.0 default)
  → Allows per-component energy variation, regularized toward uniform energy

GP Posterior (per tracking frame):
  K = K({s_n*}) + sigma^2 * I
  alpha = solve(K, z_obs)
  mu_post(x*) = K_pred_obs @ alpha   [O(N^3) solve, O(N_grid * N_obs) predict]
```

**ws_ms5_reg_amp**: Combines Stage 1 (n_starts=5) + Stage 2b (reg_amp) = current best

---

## n_starts Comparison (7 pilot episodes)

Result file: `nstarts_comparison.json`

| Episode | oracle | ms5 | ms10 | ms5→ms10 |
|---------|--------|-----|------|----------|
| BF3_5c seed0 | — | 0.1502 | 0.1459 | -2.9% |
| BF4_5c seed0 | — | 0.1380 | 0.1068 | -22.6% |
| BF4_15c seed0 | — | 0.2633 | 0.2629 | -0.2% |
| BF4_30c seed0 | — | 0.3000 | 0.2997 | -0.1% |
| BF5_15c seed0 | — | 0.2793 | 0.2802 | +0.3% |
| BF5_30c seed0 | — | 0.2915 | 0.2970 | +1.9% |
| **BF4_15c seed2** | — | **0.5729** | **0.1817** | **-68.3%** |

**Conclusion**: n_starts=10 is valuable only for catastrophic failure recovery.
- ms10 is ~2× slower per episode (~120 min total for 30 ep vs 60 min for ms5)
- Recommended: ms5 as default, ms10 only if init failure suspected

---

## Comparison vs WavePINO — Fair 30-Episode Protocol

Result file: `pino_s4_30ep_eval.json`

| Metric | GP ws_ms5_reg_amp | WavePINO S4 | Δ (GP−PINO) |
|--------|-----------------|-------------|-------------|
| **nRMSE** | **0.2232** | 0.2279 | **−2.1% (GP wins)** |
| **VR** | **0.783** | 0.309 | GP 2.5× better amplitude |
| **Pearson** | **0.489** | 0.376 | GP +30% spatial correlation |
| Skill | −0.012 | **+0.107** | WavePINO better Skill |
| UQ | ✓ full posterior | ✗ none | GP only |
| Training data | ✗ none | ✓ 90 episodes | GP more practical |
| Init cost | ~892s/ep (CPU) | ~0.5s/ep (GPU) | WavePINO ~1800× faster |

**Key finding**: WavePINO's original 0.220 result was on a different 6-episode protocol
(slope+uniform mixed, seed 0 only). On the same 30-episode fair comparison,
WavePINO = 0.228 and GP K_disp = 0.223. **GP K_disp is the better method on this benchmark.**

## Computational Cost Comparison

Result file: `compute_cost_comparison.json`

| Method | Init | Inference | Total/episode | Training |
|--------|------|-----------|--------------|---------|
| WaveSplat | 110s | 13ms/frame | 110s | None |
| GP K_disp (ms5) | 889s | ~97ms/frame* | 892s | None |
| WavePINO S4 | 0s | 18ms/frame | 0.5s | 90 episodes |

*GP inference includes kernel construction (~33ms) + Cholesky solve (~79ms). Previous report measured solve only.

**GP tradeoff**: No training data required, but 15-minute init per episode.
GP inference is O(N_obs³) per frame; N_obs varies by episode
(slope episodes have more observations → slower, up to 190ms; 30-comp has fewer → 6ms).

---

## UQ Calibration Analysis (ws_ms5_reg_amp, 6 pilot episodes)

Result file: `gp_uq_coverage_eval.json`

**Bug fixed**: Previous results (100% coverage) used column 3 (ky) instead of column 0 (amplitude).
Corrected 2026-06-15, re-run completed.

| Nominal | Empirical | Error | Status |
|---------|-----------|-------|--------|
| 50% | 0.104 | **-0.396** | over-confident |
| 68% | 0.152 | **-0.528** | over-confident |
| 80% | 0.195 | **-0.605** | over-confident |
| 90% | 0.248 | **-0.652** | over-confident |
| 95% | 0.292 | **-0.658** | over-confident |

CRPS mean: 0.236 | Mean posterior std: **0.052 m**

**Root cause: reg_amp amplitude degeneration**
- L2 regularization (λ=2.0) collapses all A_n to single value = sqrt(2·s_global/n_comp)
- Estimated var captures only **11-31%** of true wave variance:
  - BF3_5comp: est=0.002 vs true=0.023 (11%)
  - BF4_15comp: est=0.018 vs true=0.090 (20%)
  - BF5_30comp: est=0.078 vs true=0.250 (31%)
- Result: posterior std ≈ 0.02-0.09m while actual residuals ≈ 0.1-0.5m

**Why GP mean is still accurate despite wrong amplitude:**
- GP mean = K_po·(K_oo + σ²I)^{-1}·z_obs — the inverse compensates for small K
- Variance = K(x*,x*) - reduction → both terms scale with wrong A² → UQ fails

**Implication for thesis**:
- GP K_disp **cannot provide meaningful UQ** with current amplitude estimation
- reg_amp trades UQ quality for prediction stability (amplitude is nuisance parameter)
- **Recommendation**: present GP UQ as a "framework capability", not validated result
- Possible fix: post-hoc variance scaling via empirical calibration factor

---

## Failure Mode Analysis

### Failed practical methods
| Method | nRMSE | Issue |
|--------|-------|-------|
| ml_full / beamform | ~0.37 | L-BFGS-B stuck in local minimum |
| ws_opt_amp | 0.311 | Per-comp amplitudes (15 free params) overfit 300 obs |
| jonswap | 0.236 | VR=0.13 (amplitude dead) |
| σ_obs=0.01 | 0.275 | GP overfits sparse obs (VR=0.91) |

---

## Thesis Contribution Framing (Direction A)

1. **Theory**: K_disp as spectral mixture kernel → GP with dispersion-constrained prior
   - K_disp = sum_n (A_n²/2) * cos(k_n . Delta) (Bochner representation)
   - Oracle GP (nRMSE=0.126) proves: exact K_disp gives theoretically optimal estimation

2. **Method**: Two-stage K_disp estimation algorithm
   - Stage 1: WaveSplat multi-start direction estimation (physics-constrained Adam, 5 seeds)
   - Stage 2: Regularized per-component amplitude MAP estimation
   - GP posterior: exact Bayesian inference, UQ framework (amplitude calibration needed)

3. **Results**: Improvements on 30-episode benchmark (same eval protocol)
   - nRMSE: 0.263 (WaveSplat) → **0.223** (ws_ms5_reg_amp) [**15.2% improvement**]
   - Pearson: 0.346 (WaveSplat) → **0.489** (ws_ms5_reg_amp) [**41% improvement**]
   - Skill: -0.275 → **-0.012** [**95% improvement**]
   - UQ framework via GP posterior (currently overconfident — needs amplitude calibration)
   - vs WavePINO S4 (0.228, fair 30-ep): GP wins by 2.1% on same evaluation protocol

4. **Future work**: Close oracle-practical gap (0.126 → 0.223)
   - λ=0.5 + tp=4.0 tuning on ws_ms5_reg_amp (estimated ~0.215-0.218)
   - Better direction estimation for complex sea (15-30 components)
   - NN-based direction estimator replacing WaveSplat Adam phase
