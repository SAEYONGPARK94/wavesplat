# Phase 2: Implementation Direction Review — Novelty & Contribution

## Prior Art Summary (검색 결과 기반)

| Paper | Date | What | Relevance |
|-------|------|------|-----------|
| Ehlers 2025 (2508.03315) | Aug 2025 | PINO for nonlinear wave reconstruction, buoy+radar, no GT | PINO competitor |
| 2602.09708 | Feb 2026 | Physics-informed diffusion in spectral space, Adam-DPS | DPPS competitor |
| 2603.22319 (PICSB) | Mar 2026 | Schrödinger bridge for sparse data assimilation with PDE | DPPS/diffusion competitor |
| 2302.09580 | Feb 2023 | Non-separable GP kernels with dispersion functions (general PDE) | GP-kernel general |
| Wang 2025 (ETKF) | 2025 | Ship motion-based ETKF wave estimation | ETKF competitor |
| **2311.05205** | Nov 2023 | GP covariance for 3D free space wave equation + inverse problems | **Dir A threat** |
| **2111.12035** | Nov 2021 | GP under linear differential constraints, 3D wave equation | **Dir A theory basis** |
| **2502.04276** | Feb 2025 | GP regression for inverse problems in linear PDEs (2D wave eq.) | **Dir A threat** |

**No paper found**: Ship LiDAR → 2D wave field with OCEAN DISPERSION-CONSTRAINED GP posterior.

### Critical Distinction: Wave Equation GP vs Ocean Dispersion GP

The threatening papers (2311.05205, 2111.12035, 2502.04276) all deal with:
- **Acoustic wave equation**: ∂²u/∂t² = c²∇²u (hyperbolic PDE in space+time)
- **Purpose**: reconstruct initial conditions or source location from sensor observations
- **Kernel**: derived from PDE constraint in (x,y,z,t) space

Our Direction A uses:
- **Ocean dispersion relation**: ω²=g|k|tanh(|k|h) (algebraic constraint in k-ω space)
- **Purpose**: spatial field estimation at FIXED TIME from sparse 2D LiDAR
- **Kernel**: derived from spectral density concentrated on dispersion manifold in (kx,ky) space

These are mathematically different problems:
1. Wave equation GP: spatiotemporal propagation from point sources
2. Ocean dispersion GP: spatial covariance matching JONSWAP spectral shape

2502.04276 explicitly notes "2D wave equation is of practical interest in oceanography" but doesn't apply it.
Key: they haven't done the ocean wave estimation problem with ship LiDAR.

**Residual novelty of Direction A**: HIGH
- Specific application: ship LiDAR 2D spatial field estimation
- Specific kernel: K_disp from ocean wave spectrum (JONSWAP + directional spreading)
- Connection: K_disp derived from spectral mixture (Bochner theorem), not PDE constraint
- Practical contribution: first GP with physically correct spatial covariance for wave estimation

---

## Direction A: GP with Dispersion Kernel

### What it is
- Replace GPR-1's RBF kernel with K_disp: covariance derived from dispersion manifold
- K_disp(x,x') = integral over omega^2=g|k|tanh(|k|h) of exp(ik.(x-x')) * S(k) dk
  where S(k) is the wave energy spectrum (JONSWAP or similar)
- GP posterior given sparse LiDAR obs = closed-form: mu_post, Sigma_post
- Amplitude: GP posterior mean does NOT regress to zero if K_disp has non-trivial covariance

### Technical challenge
- K_disp depends on wave spectrum parameters (Hs, Tp, direction, spreading)
- These are unknown → need to estimate from obs
- Two-step approach:
  1. Estimate spectrum params from obs (beamforming, like WaveSplat init)
  2. Build K_disp from estimated params
  3. Compute GP posterior
- OR: jointly optimize GP hyperparameters + posterior (like WaveSplat, but non-parametric)

### Novelty assessment

Strengths:
- No paper has applied dispersion-constrained GP specifically to ship LiDAR 2D wave estimation
- Theoretically principled: exact Bayesian posterior, provable amplitude preservation
- Unifies WaveSplat (parametric MAP) and GPR-1 (RBF posterior) in one framework
- Explicit uncertainty quantification (Sigma_post)
- 2302.09580 shows dispersion-like kernels for general PDEs but NOT ocean wave specific + NOT ship LiDAR

Weaknesses:
- GPR-1 already exists as baseline → reviewers may say "just kernel engineering"
- K_disp estimation is complex (nested optimization or two-stage)
- O(N_obs^3) per frame — but N_obs ~ 1% * 4800 = 48 → very fast!
- Need to show meaningful improvement over GPR-1 (currently 0.261 on 6-ep)

Reviewer concerns:
1. "How is this different from GPR-1 you already have?"
   Answer: GPR-1 uses RBF kernel (smooth interpolation), K_disp encodes wave physics (oscillatory, anisotropic, multi-scale). Different prior → different posterior.
2. "This is just a kernel choice in a GP — incremental"
   Answer: The kernel choice IS the physics. Dispersion relation is the fundamental physical law. Encoding it as a covariance structure is a principled contribution, not just engineering.
3. "How do you estimate K_disp parameters?"
   Need: clear algorithm for spectrum estimation from sparse LiDAR.

PhD fit: HIGH (theoretical depth, connects to WaveSplat, unifying framework)
OE journal fit: HIGH (ocean-specific physics, practical application)
Implementation effort: MEDIUM (GP is straightforward, K_disp derivation is the hard part)

Score: 8/10

---

## Direction B: WavePINO + Amplitude Fix

### What it is
- Keep WavePINO S4 architecture
- Fix amplitude hedging (VR = 0.29 → ~1.0)
- Options:
  B1. Add VR loss: ||VR(pred) - 1||^2 during training — ad-hoc but simple
  B2. Energy normalization at inference: rescale pred by Hs_estimate / Hs_pred
  B3. Frequency-domain output: predict in k-space constrained to dispersion manifold
  B4. MAP inference: use learned score to find mode instead of mean

### Novelty assessment

Strengths:
- WavePINO already achieves nRMSE 0.222 (best), complex sea advantage (Pearson 0.498 vs 0.293)
- Fixing amplitude would make it Pareto-dominant over WaveSplat on ALL metrics
- Ehlers 2025: different problem (no GT, buoy+radar, nonlinear physics). WavePINO is supervised, ship LiDAR specific.

Weaknesses:
- Ehlers 2025 significantly overlaps in methodology (PINO for wave reconstruction)
- Amplitude hedging is well-known problem in neural networks — fix options are not novel
- B1 (VR loss): ad-hoc, not theoretically principled
- B2 (Hs normalization): engineering hack, not a contribution
- B3 (k-space output): interesting but changes architecture fundamentally
- Alone, this is a system improvement paper, not a principled contribution

Reviewer concerns:
1. "Ehlers 2025 already does physics-informed neural operators for waves"
   Answer: different sensor (ship LiDAR vs buoy/radar), different training (supervised vs self-supervised), different architecture
2. "Amplitude fix is just loss engineering"
   Answer: Hard to defend B1/B2. B3/B4 more defensible but harder to implement.
3. "WavePINO is just ConvCNP + FNO + dispersion loss — derivative work"
   Answer: Need stronger theoretical contribution or clear systematic analysis

PhD fit: LOW (incremental, lacks theoretical depth)
OE journal fit: MEDIUM (practical improvement, but not groundbreaking)
Implementation effort: LOW-MEDIUM

Score: 5/10

---

## Direction C: DPPS Formalization / Physics-Constrained Diffusion

### What it is
- Formalize DPPS as principled physics-constrained posterior sampling
- Show DPPS approximates Langevin dynamics on p(f|y, disp)
- Option C1: Just theory — prove DPPS convergence, show why DPS fails alone
- Option C2: Improved DPPS — better observation likelihood, convergence-preserving
- Option C3: New diffusion model with physics-informed score (retrain with K_disp prior)

### Novelty assessment

Strengths:
- DPPS shows 14.7% improvement over diffusion baseline (ablation is clean)
- DPS failure analysis for sparse observations is a genuine insight
- "Training-free physics enforcement" is a useful engineering contribution

Weaknesses:
- 2602.09708 (Feb 2026): Physics-informed diffusion in spectral space, uses DPS-style Adam updates
  → Very similar to DPPS methodology. Key overlap: inference-time physics constraint on diffusion.
- 2603.22319 (PICSB, Mar 2026): Schrödinger bridge with PDE constraints for sparse data assimilation
  → Directly competes with our physics-constrained posterior sampling story
- DPPS is heuristic — DPS gradient unreliable for sparse obs, dispersion projection ad-hoc
- DPPS does NOT beat WavePINO (0.272 vs 0.222), and amplitude is poor (VR~0.30)
- Cannot extend to v5b (v-prediction model already good, DPS adds nothing)

Reviewer concerns:
1. "2602.09708 already does physics-constrained diffusion inference"
   The specific combination of DPS + dispersion projection for ocean wave LiDAR may still be novel, but the competitive landscape is crowded.
2. "DPPS is worse than WavePINO on all important metrics"
   Hard to sell as main contribution when your method has worse amplitude AND worse nRMSE.
3. "DPS failure analysis is incremental — well known that DPS degrades for sparse obs"

PhD fit: MEDIUM (some theory, interesting ablation, but crowded field and weak numbers)
OE journal fit: LOW-MEDIUM (results don't support it as main contribution)
Implementation effort: LOW (already done)

Score: 6/10

---

## Comparison Summary

| Criterion | Dir A (GP+K_disp) | Dir B (PINO+amp) | Dir C (DPPS theory) |
|-----------|-----------------|-----------------|-------------------|
| Novelty | HIGH | LOW-MEDIUM | MEDIUM (crowded) |
| Theoretical depth | HIGH | LOW | MEDIUM |
| Prior art conflict | LOW | MEDIUM (Ehlers) | HIGH (2602/2603) |
| Current results support | MEDIUM (GPR-1=0.26) | HIGH (0.222) | LOW (0.272) |
| Amplitude problem solved | YES (if K_disp correct) | Partially | NO (VR=0.30) |
| Complex sea advantage | YES (if K_disp correct) | YES (0.498 Pearson) | UNKNOWN |
| PhD thesis depth | HIGH | LOW | MEDIUM |
| Implementation effort | MEDIUM | LOW | LOW |
| OE journal fit | HIGH | MEDIUM | LOW |

---

## Recommended Direction: A (GP + Dispersion Kernel)

### Rationale
1. Most novel: no competing paper on ship LiDAR 2D wave GP with K_disp
2. Theoretically deepest: unifies WaveSplat, GPR, WavePINO under one Bayesian framework
3. Amplitude problem: solved by correct physics prior (not ad-hoc fix)
4. Uncertainty quantification: free bonus from GP framework
5. Connection to existing work: GPR-1 baseline already shows GP works (~0.26), K_disp improves it

### What success looks like
- K_disp-based GP posterior: nRMSE < 0.22 (beat WavePINO)
- VR closer to 1.0 (amplitude preserved)
- Pearson > 0.5 on complex sea (structure captured)
- Uncertainty estimates meaningful (coverage ~0.90)

### How to combine with existing work
- Chapter 1 (WaveSplat paper): MAP optimization baseline, establishes problem
- Chapter 2 (GP framework): theory + K_disp derivation + comparison with WaveSplat/WavePINO
- Chapter 3 (optional): DPPS as approximate posterior sampling, ablation analysis

### Risk
- K_disp estimation from sparse LiDAR is technically hard
- GP may not capture amplitude correctly if spectrum estimate is poor
- Need pilot experiment to verify before committing

---

## Next Step (before committing to Direction A)

1. Pilot: Build K_disp with known spectrum params (oracle test)
   - If GP posterior with true K_disp gives VR~1.0 and nRMSE < 0.22 → proceed
   - If not → K_disp approach is flawed, need to reconsider
2. Then: develop spectrum estimation algorithm from sparse LiDAR
3. Then: full evaluation
