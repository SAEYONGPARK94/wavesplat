# Phase 0: Comprehensive Method Comparison

## Evaluation Protocol
- 6 test episodes: BF3_5c, BF4_5c, BF4_15c, BF4_30c, BF5_15c, BF5_30c
- 30 frames / episode
- Metrics: nRMSE (lower=better), VR (Variance Ratio, 1.0=ideal), Pearson_rho, Skill_Score, Hs_error, Directional_R2

## Per-Episode Results (nRMSE)

| Episode | WaveSplat | WavePINO | Diffusion v3 (5s) | DPPS best (1s) |
|---------|-----------|----------|---------------------|----------------|
| BF3_5c  | 0.152     | 0.177    | 0.249               | 0.268          |
| BF4_5c  | 0.200     | 0.203    | 0.253               | 0.267          |
| BF4_15c | 0.302     | **0.226** | 0.248              | **0.262**      |
| BF4_30c | 0.314     | **0.244** | 0.262              | **0.267**      |
| BF5_15c | 0.297     | **0.226** | 0.259              | **0.274**      |
| BF5_30c | 0.290     | **0.254** | 0.289              | **0.296**      |
| **AVG** | **0.259** | **0.222** | **0.260**          | **0.272**      |

*Bold = wins vs WaveSplat. Diffusion/DPPS use same 6 episodes but different eval protocol (grid-based).*

## Full Metric Comparison (Averages, 6 episodes)

| Method | nRMSE | VR (1.0=ideal) | Pearson | Skill | Hs_err |
|--------|-------|----------------|---------|-------|--------|
| WaveSplat | 0.259 | 0.588 | 0.293 | -0.095 | 0.276 |
| **WavePINO S4** | **0.222** | 0.292 | **0.498** | **0.264** | 0.455 |
| WaveDiff v3 (5s,gs3) | 0.260 | ~0.270 | n/a | n/a | n/a |
| DPPS (z2,d1.0,1s) | 0.272 | ~0.296 | n/a | n/a | n/a |
| ConvCNP v2 | 0.244 | ~0.40 | n/a | n/a | n/a |

## Key Findings

### Finding 1: WaveSplat fails on complex sea
- BF4_15c+: WaveSplat Pearson = 0.04-0.16 (effectively random), Skill = -0.33 to -0.60
- WavePINO: Pearson = 0.39-0.45, Skill = +0.14 to +0.21 on same episodes
- WaveSplat wins only on BF3_5c (simple sea, 5 components): 0.152 vs 0.177

### Finding 2: Amplitude hedging is severe in WavePINO
- Average VR = 0.292 (produces only 29% of actual amplitude)
- WaveSplat VR = 0.588 (better, but also hedged on complex sea)
- Hs_error: WavePINO 0.455 vs WaveSplat 0.276 (much worse!)
- Physics explanation: MSE training = posterior mean = amplitude underestimation

### Finding 3: Diffusion/DPPS do NOT beat WaveSplat or WavePINO
- WaveDiffusion v3 (5s ensemble) = 0.260: same as WaveSplat, no complex sea advantage evident
- DPPS best (1s) = 0.272: worse than WavePINO 0.222
- Key: diffusion results lack Pearson/Skill data — need to evaluate fully

### Finding 4: Trade-off landscape
| Method | Complex sea | Amplitude | nRMSE | Speed |
|--------|-------------|-----------|-------|-------|
| WaveSplat | FAIL | Medium | 0.259 | Fast |
| WavePINO | Good | BAD (VR=0.29) | 0.222 | Fast |
| WaveDiffusion | Unknown | Hedged (5s) | 0.260 | Slow |
| DPPS | Unknown | Better (1s) | 0.272 | Very slow |

## What the data tells us about the right direction

No single method is universally best:
- nRMSE winner: WavePINO (0.222)
- Amplitude winner: WaveSplat on simple sea; DPPS on single-sample (amp=0.30)
- Structure winner: WavePINO (Pearson 0.498 vs 0.293)
- Hs winner: WaveSplat (0.276 vs 0.455)

The ideal method would combine:
- WavePINO's complex sea structure recovery
- WaveSplat's amplitude fidelity (VR near 1.0)
- Physics consistency (dispersion relation)
