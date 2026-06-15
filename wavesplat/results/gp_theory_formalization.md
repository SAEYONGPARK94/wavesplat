# GP K_disp: Theory Formalization
# (WaveSplat as MAP Approximation of GP Posterior)
# Updated: 2026-06-10

---

## 1. Problem Statement

**Given**: sparse LiDAR observations y = {z_i} at positions X_obs = {(x_i, y_i)} (N_obs ~ 50)
**Goal**: reconstruct full 2D wave field f(x, y) on dense grid (80×60 = 4800 points)
**Physics constraint**: dispersion relation ω_n² = g|k_n|tanh(|k_n|h) for all wave components n

Formal Bayesian problem:
```
p(f | y, dispersion) ∝ p(y | f) · p(f | dispersion)
```
where
- p(y | f) = N(y; Hf, σ_obs² I)   [H = observation operator, linear]
- p(f | dispersion) = ?             [physics prior]

---

## 2. The Physics Prior: K_disp as Spectral Mixture Kernel

### 2.1 Linear wave field model

A linear random wave field at fixed time t is:
```
f(x) = sum_n A_n · cos(kx_n · x + ky_n · y - ω_n · t + φ_n)
```
where {φ_n} are independent uniform phases and ω_n² = g|k_n|tanh(|k_n|h).

If {φ_n} are random (stochastic sea), f(x) is a zero-mean Gaussian process with covariance:

```
K_disp(x, x') = E_φ[f(x) f(x')]
              = sum_n (A_n²/2) · cos(kx_n·(x_1-x'_1) + ky_n·(x_2-x'_2))
```

**This is the dispersion-constrained kernel**.

### 2.2 Bochner representation

By Bochner's theorem, K_disp is a valid stationary positive-definite kernel because it is a
non-negative mixture of cosine (Fourier) basis functions. Explicitly:

```
K_disp(Δx) = ∫_{ω²=g|k|tanh(|k|h)} e^{ik·Δx} dμ(k)
```
where μ(k) = sum_n (A_n²/2) δ(k - k_n) is the discrete spectral measure.

This places K_disp in the class of **Spectral Mixture (SM) kernels** (Wilson & Adams, 2013),
but restricted to the dispersion manifold.

**Key property**: frequencies NOT on the dispersion manifold have zero prior probability.
This is the physics constraint encoded in the prior.

---

## 3. GP Posterior under K_disp Prior

Given f ~ GP(0, K_disp), with linear observations y = Hf + ε, ε ~ N(0, σ²I):

**Posterior is Gaussian**:
```
p(f | y, K_disp) = N(μ_post, Σ_post)

μ_post(x*) = K_*(X_obs)^T · (K_oo + σ²I)^{-1} · y
Σ_post(x*, x**) = K_disp(x*, x**) - K_*(X_obs)^T · (K_oo + σ²I)^{-1} · K_*(X_obs)
```

where:
- K_oo = K_disp(X_obs, X_obs)  [N_obs × N_obs]
- K_*(X_obs) = K_disp(x*, X_obs)  [N_obs × 1 for each prediction point x*]

**Computation**:
- Solve: α = (K_oo + σ²I)^{-1} y   [O(N_obs³), N_obs~50 → ~125k ops, < 1ms]
- Predict: μ_post(x*) = K_*(X_obs)^T α   [O(N_grid × N_obs) = O(4800×50), < 1ms]

**Critical property**: μ_post is the GLOBALLY OPTIMAL linear predictor (no local optima).
For any observation set, the GP posterior mean is the unique solution.

---

## 4. WaveSplat as Approximate MAP of GP Posterior

### 4.1 WaveSplat's objective

WaveSplat Phase 1 minimizes (implicitly):
```
L_WS(θ) = sum_t sum_i (f_θ(x_i, t) - z_i(t))²  +  λ·R(θ)
subject to: ω_n² = g|k_n|tanh(|k_n|h)  ∀n
```
where f_θ(x) = sum_n w_n · cos(k_n·x - ω_n·t + φ_n) and R(θ) is regularization.

### 4.2 Connection to GP MAP

**Proposition**: Under the Gaussian likelihood and K_disp prior, the MAP estimate of f is:
```
f_MAP = argmax_f log p(f | y, K_disp)
      = argmin_f [ ||y - Hf||² / σ_obs²  +  f^T K_disp^{-1} f ]
```

For the GP, MAP = mean (since posterior is Gaussian): f_MAP = μ_post.

WaveSplat's objective is equivalent to the MAP problem when:
1. The parametric model f_θ is expressive enough to represent the true field
2. The regularization R(θ) matches the GP prior K_disp^{-1}

**WaveSplat does NOT compute K_disp explicitly**. Instead:
- It uses beamforming to implicitly estimate {k_n} (implicit K_disp estimation)
- It runs gradient-based optimization (implicit MAP search in parameter space)
- It approximately finds μ_post when the model is well-specified

### 4.3 Why WaveSplat fails for complex sea

For complex sea (N_comp = 15-30):
- The optimization landscape of L_WS has many local minima
- Gradient descent (Adam) converges to different local optima depending on initialization
- Result: WaveSplat finds a local MAP, not the global MAP

**GP posterior does not have this problem** because:
- K_disp kernel is fixed before prediction
- The linear system (K_oo + σ²I)α = y has a UNIQUE solution (positive definite system)
- μ_post is always the global optimum, regardless of complexity

The failure mode is therefore in **K_disp estimation** (Stage 1 of our method), not in the posterior computation (Stage 2).

---

## 5. Two-Stage Algorithm: Formal Justification

### 5.1 Stage 1: Direction Estimation

**Goal**: find {k_n} = {(kx_n, ky_n)} that maximize marginal likelihood:
```
{k_n*} = argmax_{k_n} log p(y | K_disp({k_n}))
        = argmax_{k_n} log N(y; 0, K_disp({k_n}) + σ²I)
```

**Why not direct ML?** The marginal likelihood landscape over {k_n} is multi-modal in k-space.
Joint optimization over 15 direction pairs (30 parameters) → L-BFGS-B fails.

**Two-stage solution**:
1. WaveSplat Phase 1 with dispersion constraint (physics-constrained Adam):
   - Dispersion hard constraint reduces search space from R^30 to S^15 (unit circles × scales)
   - Multiple seeds explore different basins of attraction
2. Select best seed by NML: argmin_seed NML(K_disp({k_n}^seed) | y)
   - This is Bayesian model selection: choose the model with highest evidence

**Formal guarantee**: Among the 5 candidate direction sets, we select the one that
maximizes p(y | K_disp), providing a principled criterion for model selection.

### 5.2 Stage 2: Amplitude Calibration

**Goal**: given fixed {k_n}, find amplitude scale s:
```
s* = argmax_s log p(y | s · K_unit)
   = argmin_s NML(s · K_unit | y)
```
where K_unit = sum_n 0.5 · cos(k_n · Δx) (unit-amplitude kernel).

**Why single scale?** This reduces the amplitude estimation from N_comp free parameters
(ws_opt_amp: 15 DoF >> 300 obs → overfit) to 1 parameter (convex in log s → global optimum).

**Extended: Regularized per-component scales (ws_reg_amp)**:
```
{s_n*} = argmin_{s_n} NML(K({s_n}) | y) + λ · sum_n (log s_n - log(s*/N))²
```
Prior center: s*/N (= global scale uniformly distributed over N components)
This interpolates between global scale (λ→∞) and per-component ML (λ→0).

---

## 6. Oracle-Practical Gap: Two-Source Decomposition

From 30-episode evaluation (tracking frames, seed 0-4 × 6 sea states):

| Source | Sea state | Evidence |
|--------|-----------|---------|
| **Amplitude miscalibration** | 5 comp | oracle VR~0.8, ws_ms5 VR~0.4-0.6 (undershoot) → single scale wrong |
| **Direction estimation failure** | 30 comp | oracle Pearson=0.60, ws_ms5 Pearson=0.20 (vs oracle 0.61) |
| **Catastrophic failure** | 15 comp | 1/5 seeds: all 5 Adam restarts in same bad local min |

### 6.1 Sea state breakdown (nRMSE)

| Sea state | N_comp | Oracle | ws_ms5 | WaveSplat | Gap(abs) | WS→GP |
|-----------|--------|--------|--------|-----------|----------|-------|
| BF3_5c    | 5      | 0.083  | 0.175  | 0.213     | 0.092    | +18%  |
| BF4_5c    | 5      | 0.050  | 0.130  | 0.220     | 0.081    | +41%  |
| BF4_15c*  | 15     | 0.115  | 0.316* | 0.284     | 0.201*   | -11%* |
| BF4_15c†  | 15     | 0.117  | 0.251  | 0.296     | 0.134    | +15%  |
| BF4_30c   | 30     | 0.181  | 0.265  | 0.284     | 0.084    | +7%   |
| BF5_15c   | 15     | 0.147  | 0.245  | 0.303     | 0.098    | +19%  |
| BF5_30c   | 30     | 0.180  | 0.278  | 0.276     | 0.098    | -1%   |

*with catastrophic outlier (seed2: nRMSE=0.573, VR=4.61)
†excluding outlier (4 episodes)

### 6.2 VR (amplitude accuracy) by sea state

| Sea state | Oracle VR | ws_ms5 VR | WaveSplat VR | ws_ms5 amplitude bias |
|-----------|-----------|-----------|--------------|----------------------|
| BF3_5c    | 0.686     | 0.339     | 0.742        | -50% (large underestimate) |
| BF4_5c    | 0.842     | 0.574     | 0.893        | -31% |
| BF4_30c   | 0.403     | 0.575     | 0.788        | +43% (overestimate!) |
| BF5_30c   | 0.508     | 1.220     | 0.961        | +140% (severe overestimate) |

### 6.3 Interpretation: what ws_reg_amp should fix

For 5 comp sea: VR=0.34-0.57 → amplitude systematically underestimated by single global scale.
The single scale s* underestimates variance because low-energy components pull the global scale down.
ws_reg_amp allows per-component variation with regularization → should improve VR toward 1.0.

For 30 comp sea: VR overestimates (1.22 for BF5_30c) → scale overfit to noise.
Regularization should prevent this overcorrection.

---

## 7. Novel Contributions (Formal)

### Contribution 1: K_disp as Bayesian prior (Theory)
- **Theorem**: K_disp(Δx) = sum_n (A_n²/2)cos(k_n·Δx) is the exact covariance of a
  linear wave field with components {A_n, k_n} and random phases.
- **Consequence**: GP(0, K_disp) is the correct Bayesian prior for wave field estimation
  under linear wave theory.
- **Prior work gap**: Generic GP methods (RBF, Matern) do not encode dispersion physics;
  WaveSplat does not formalize the prior.

### Contribution 2: Two-stage K_disp estimation (Algorithm)
- Stage 1: Multi-start physics-constrained Adam → NML-based seed selection
- Stage 2: 1D ML amplitude calibration (global) or regularized per-component (ws_reg_amp)
- **Key advantage over joint ML**: Stage 1 hard constraint (ω²=g|k|tanh) makes search
  tractable; Stage 2 is convex in log-scale.

### Contribution 3: Closed-form GP posterior (Prediction)
- Replace iterative WaveSplat tracking with single matrix solve per frame
- Prediction is globally optimal (no local optima) once K_disp is fixed
- Full posterior variance available for uncertainty quantification

### Connection to PhD thesis:
- WaveSplat Paper 1: WaveSplat MAP approach (established baseline)
- GP K_disp Paper 2: Bayesian formalization, principled prior, improved prediction
- "Oracle-practical gap" = the research question for future work

---

## 8. Key References to Distinguish

| Reference | Method | Difference from our work |
|-----------|--------|--------------------------|
| Wilson & Adams 2013 (SM kernel) | Spectral mixture kernel (general) | Our K_disp is restricted to dispersion manifold |
| Rasmussen & Williams 2006 (GP) | General GP regression | We define the specific physics prior |
| 2311.05205, 2111.12035 | GP for acoustic/elastic waves (3D PDE) | Hyperbolic PDE in (x,y,z,t); different physics |
| ETKF (Wang 2025) | Ensemble Kalman filter | Sequential filter; our GP is single-frame Bayesian |
| WavePINO / ConvCNP | Neural wave estimation | Learned prior; our prior is physics-derived |
