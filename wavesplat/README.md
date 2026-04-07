# WaveSplat

**Grid-free Wave Field Estimation from Sparse Ship-mounted LiDAR via Spectral Gaussian Estimation**

---

## Overview

WaveSplat addresses the problem of reconstructing a continuous wave height field `z(x,y,t)` from the extremely sparse surface point clouds captured by a forward-looking LiDAR mounted on a ship.

Unlike grid-based approaches (e.g., ETKF), WaveSplat represents the wave field as a weighted sum of spectral Gaussian components with physics-based constraints (dispersion relation, Fresnel confidence weighting), enabling grid-free estimation that adapts model complexity to wave scene complexity.

### Key Results (P1)

| Method | Avg nRMSE | Speed (ms/frame) | Win Rate |
|--------|-----------|-------------------|----------|
| **WaveSplat (ours)** | **0.211** | **13.4** | **12/12** |
| ETKF-grid | 0.436 | 31.7 | 0/12 |

- **51.5% lower nRMSE** than ETKF across BF2–BF5 sea states
- **2.4× faster** inference
- Robust to both uniform and slope raydrop observation models (20/20 wins)

### Key Features

- Grid-free spectral Gaussian representation: `z(x,y,t) = Σ wᵢ·cos(kxᵢ·x + kyᵢ·y - ωᵢ·t + φᵢ)`
- Dispersion constraint: `ω = √(g·|k|·tanh(|k|·d))` — hard constraint, no deep-water approximation
- Fresnel confidence weighting: `cⱼ = cos(θⱼ) / (dⱼ/d_ref)`
- Adaptive N: energy-based pruning + residual-triggered growth
- Two-phase estimation: Init (beamforming + coarse-to-fine) → Tracking (phase prediction + gradient update)
- Physics-based wave simulation (Gerstner waves + JONSWAP spectrum)
- Two LiDAR return models: uniform Fresnel-based and specular slope-dependent
- 39 unit tests, all passing

---

## Research Context

| | |
|---|---|
| **Problem** | Ship-mounted forward-looking LiDAR produces only tens to hundreds of surface returns per frame |
| **Goal** | Recover continuous wave field `z(x,y,t)` from irregular sparse observations |
| **Method** | Grid-free Spectral Gaussian Estimation with dispersion-constrained parameterization |
| **Baseline** | ETKF-grid (Ensemble Transform Kalman Filter with spectral FFT propagation) |

---

## Repository Structure

```
wavesplat/
├── configs/                    # YAML configuration files
│   ├── default.yaml            # Local dev (76K rays, 1 worker)
│   └── server.yaml             # Server (304K rays, 4 workers, full resolution)
├── wavesim/                    # Simulation package (complete)
│   ├── config.py               # Config loader + dataclasses
│   ├── wave_generator.py       # Gerstner wave + JONSWAP spectrum
│   ├── lidar_simulator.py      # Ray generation + ray-mesh intersection
│   ├── raydrop_uniform.py      # Model A: Fresnel/distance-based dropout
│   ├── raydrop_slope.py        # Model B: Specular backscatter dropout
│   ├── gt_generator.py         # Ground truth wave field evaluation
│   ├── noise_model.py          # Sensor noise + confidence weighting
│   ├── data_writer.py          # HDF5 save/load
│   ├── pipeline.py             # End-to-end episode pipeline
│   └── tests/                  # 39 unit tests
├── wavesplat/                  # Estimation algorithm (complete)
│   ├── model/
│   │   ├── spectral_gaussian.py  # N spectral Gaussian components
│   │   ├── dispersion.py         # ω = √(g|k|tanh(|k|d))
│   │   └── confidence.py         # Fresnel-zone confidence weights
│   ├── optimizer/
│   │   ├── initializer.py        # Beamforming init (v2_konly)
│   │   ├── coarse_to_fine.py     # N=3→8→15 schedule
│   │   └── sequential_update.py  # Two-phase estimator + adaptive N
│   ├── baselines/
│   │   └── etkf_grid.py          # ETKF baseline
│   ├── evaluation/
│   │   └── metrics.py            # nRMSE, Hs error, spectral metrics
│   ├── data_loader.py            # HDF5 episode loader
│   └── run.py                    # Main inference entry point
├── scripts/                    # Experiment & paper generation scripts
│   ├── run_p1_comparison.py      # P0 vs P1 vs ETKF (12 episodes)
│   ├── run_raydrop_comparison.py # Uniform vs slope (10 paired episodes)
│   ├── generate_paper_tables.py  # LaTeX tables (Table 1-4)
│   └── generate_paper_figures.py # PDF figures (Fig 1-5)
├── paper/                      # IEEE conference paper
│   ├── wavesplat.tex             # Main LaTeX source
│   └── figures/                  # Generated PDF figures
├── results/                    # JSON experiment results
│   ├── p1_comparison.json        # Main comparison (authoritative)
│   ├── raydrop_comparison.json   # Raydrop model comparison
│   ├── ablation_seeded.json      # Ablation study (3-seed)
│   └── etkf_comparison.json      # ETKF comparison (historical)
├── data/                       # Generated HDF5 episodes (100 total)
├── generate_dataset.py         # Batch dataset generation (parallel)
└── docs/                       # Development documentation
```

---

## Installation

```bash
pip install numpy scipy matplotlib h5py pyyaml pytest torch
```

---

## Quick Start

### Run estimation on an episode

```bash
cd wavesplat
python3 -m wavesplat.run data/episode_BF4_5comp_seed0_uniform.h5 --adaptive_n
```

### Run full P1 comparison

```bash
python3 scripts/run_p1_comparison.py
```

### Generate paper tables and figures

```bash
python3 scripts/generate_paper_tables.py
python3 scripts/generate_paper_figures.py
```

### Run tests

```bash
python3 -m pytest wavesim/tests/ -v
```

---

## Experiment Results

### Main Comparison (12 episodes, BF2–BF5)

- WaveSplat avg nRMSE: **0.211** (ETKF: 0.436, **51.5% better**)
- Speed: **13.4 ms/frame** (ETKF: 31.7ms, **2.4× faster**)
- Best case: BF3_2comp nRMSE=0.052 (8.5× better than ETKF)

### Adaptive N

- BF4_1comp: 15→1, BF3_2comp: 15→2, BF5_15comp: 15→21
- Energy-based pruning with safety rollback (>30% residual increase)

### Raydrop Model Comparison (10 paired episodes)

- WaveSplat wins 10/10 under both uniform and slope raydrop
- Slope improves both methods: WS -9.1%, ETKF -16.2%

### Ablation Study (BF3, 3-seed average)

- Full model: 0.188
- w/o Dispersion: 0.158 (acts as regularizer, important for long-term tracking)
- w/o Confidence: 0.215 (+14.5%, essential for heterogeneous observations)

---

## Physics

### Wave Generation

Gerstner wave superposition with JONSWAP spectrum sampling.

```
z(x,y,t) = Σ Aᵢ · cos(kᵢ·r - ωᵢ·t + φᵢ)
```

Full dispersion: `ω = √(g·k·tanh(k·d))`, Q-constraint: `Σ Qᵢ·ωᵢ·Aᵢ ≤ 1`

### LiDAR Return Models

**Model A — Uniform (Fresnel/distance)**: `P_return = η · R_fresnel(θᵢ) · cos(θᵢ) / (d/d_ref)²`

**Model B — Slope-dependent (Specular backscatter)**: Return only when reflected ray aligns with sensor within `α_threshold = 5°`

---

## Dataset

100 episodes: BF2–BF5 × 1/2/5/15/30 components × 5 seeds × 2 raydrop models

Format: HDF5 with ground truth wave fields, LiDAR observations, and metadata.

---

## Scope and Limitations

- Valid for sea states BF2–BF5, non-breaking waves
- Linear wave superposition (Stokes 2nd-order: future work)
- Ship motion: currently fixed position
- High complexity (N≥30): beamforming init resolution limits nRMSE to ~0.30
