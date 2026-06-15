# Phase 1: Theory Framework — Dispersion-Constrained Posterior

## The Correct Probabilistic Problem

We want: p(f | y, dispersion)
- f: full 2D wave field on grid [Nx, Ny]
- y: sparse LiDAR observations at ~1% spatial coverage
- dispersion: omega^2 = g|k|tanh(|k|h) (hard physical constraint)

Bayes factorization:
  p(f | y, disp) ∝ p(y | f) * p(f | disp)

Components:
  p(y | f) = N(y; Hf, sigma_obs^2 * I)   [Gaussian noise model]
  p(f | disp) = ?                          [physics prior — how to define?]

---

## Existing Methods as Approximations

| Method | Actually computes | Missing | Failure mode |
|--------|------------------|---------|--------------|
| WaveSplat | argmax_f p(f|y, disp) [MAP] | Posterior distribution | Complex sea: local optima |
| WavePINO | E[f|y] [posterior mean] | Physics prior (disp weak) | Amplitude hedging (MSE theorem) |
| WaveDiffusion | sample from p_theta(f|y) | Hard physics prior | Unconstrained, amplitude hedging in ensemble |
| DPPS | Heuristic steering of WaveDiffusion toward disp | Convergence guarantee | DPS gradient unreliable at sparse obs |
| GPR-1 | E[f|y] under RBF prior | Dispersion kernel | Generic spatial smoothness, not wave physics |

### Amplitude hedging theorem (MSE → posterior mean)
MSE optimal prediction: f_hat = argmin_theta E[||f - f_theta(y)||^2] = E[f|y]
For multimodal p(f|y): E[f|y] averages over modes → amplitude < actual peaks
WaveSplat avoids this by finding MAP (one mode), but pays with local optima in complex sea.

---

## Defining p(f | disp) — The Physics Prior

### Option 1: Spectral Gaussian (parametric, like WaveSplat)
f(x) = sum_n A_n * cos(k_n . x - omega_n * t + phi_n)
  where omega_n^2 = g|k_n|tanh(|k_n|h)  [dispersion enforced]
This is WaveSplat's model. Prior is implicit: any superposition of dispersive sinusoids.
Limitation: fixed N components, optimization required.

### Option 2: GP with Dispersion Kernel (non-parametric)
f ~ GP(0, K_disp)
K_disp(x, x') = int_{omega^2 = g|k|tanh(|k|h)} e^{ik.(x-x')} dmu(k)
= integral over dispersion manifold of plane waves

Explicit form (assuming isotropic spectrum S(|k|)):
K_disp(r) = int_0^infty S(k) J_0(k*r) k dk
  where S(k) = E[A(k)^2] is the wave energy spectrum
  and J_0 is the zeroth-order Bessel function

For JONSWAP spectrum: S(k) = Hs^2/k^4 * exp(-5/4 * (k_p/k)^2) * gamma^R
This kernel encodes: wave scale, directional spread, spectral shape.

Key property: Waves with omega NOT on dispersion manifold have zero prior probability.

### Option 3: Score-based (implicit, learned)
Train score network s_theta(f, t) = -nabla_f log p_t(f | disp)
DPPS approximates this at inference. True solution: train with physics-consistent data.
(Already done: WaveDiffusion is trained on simulated dispersive waves!)

---

## Connection: WaveSplat = MAP of the GP posterior

Given:
  f ~ GP(0, K_disp)   [physics prior]
  y = H f + eps       [observation]

Posterior is also Gaussian:
  p(f|y, disp) = N(mu_post, Sigma_post)
  mu_post = K_disp H^T (H K_disp H^T + sigma^2 I)^{-1} y
  Sigma_post = K_disp - K_disp H^T (H K_disp H^T + sigma^2 I)^{-1} H K_disp

WaveSplat does not compute K_disp explicitly. Instead:
  - Initializes spectral components via beamforming (implicit K_disp estimate)
  - Runs gradient-based optimization (implicit MAP search)
  - Is approximately finding mu_post, but without the full covariance

This means:
  - WaveSplat ≈ mu_post (posterior mean) for simple sea (few components)
  - WaveSplat ≠ mu_post for complex sea (many local optima)
  - No uncertainty estimate from WaveSplat

---

## What Amplitude Hedging Means in the GP Framework

For Gaussian posterior, mu_post IS the MAP (mode = mean for Gaussian).
So GP posterior mean = MAP for the linear observation model.
Amplitude hedging in WavePINO arises because:
  1. WavePINO uses a learned (non-parametric) prior, not the true p(f|disp)
  2. The learned prior is implicitly p_theta(f) ~ marginal of training data
  3. With sparse obs, the posterior mean is dominated by the prior mean (→ 0)
  4. Result: WavePINO regresses toward zero mean → amplitude underestimation

Fix: Use the correct physics prior p(f|disp) which has non-trivial covariance K_disp.
With K_disp, the posterior mean at unobserved locations is:
  mu_post(x*) = k_*(H K_disp H^T + sigma^2 I)^{-1} y
  where k_* = K_disp(x*, x_obs)
  This does NOT regress to zero if K_disp encodes wave structure!

---

## Summary: The Key Insight

| Method | Prior | Posterior access | Amplitude |
|--------|-------|-----------------|-----------|
| WaveSplat | Implicit dispersion (parametric) | MAP only | OK (simple), FAIL (complex) |
| WavePINO | Learned (marginal data) | Mean only | Hedged (VR=0.29) |
| WaveDiffusion | Learned implicit | Samples | Hedged in ensemble |
| **GP + K_disp** | Explicit dispersion | Full posterior (mean + variance + samples) | Correct (if K_disp is right) |

The GP framework is the unifying theory. The question for Phase 2 is:
- Can we build K_disp that captures real wave physics?
- Is this computationally feasible for 80x60 grids?
- Is this sufficiently novel vs existing GPR baseline?
