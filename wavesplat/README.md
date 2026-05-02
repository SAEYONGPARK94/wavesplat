# WaveSplat

**Grid-Free Spectral Wave Field Estimation from Sparse Shipboard LiDAR**

---

## Overview

WaveSplat addresses the problem of reconstructing a continuous wave height field `z(x,y,t)` from the extremely sparse surface point clouds captured by a forward-looking LiDAR mounted on a ship.

Unlike grid-based approaches (e.g., ETKF), WaveSplat represents the wave field as a weighted sum of spectral components with physics-based constraints (dispersion relation, geometry-aware confidence weighting), enabling grid-free estimation that adapts model complexity to wave scene complexity.

### Key Results (100-episode, pre-specified 6-metric panel)

| Method | Avg nRMSE | Skill Score | Speed (ms/frame) | Complexity |
|--------|-----------|-------------|-------------------|------------|
| **WaveSplat (ours)** | **0.223** | **+0.034** | **13.4** | O(M·N) |
| GPR (Matérn 5/2) | 0.261 | −0.140 | >200 | O(M³) |
| ETKF (tuned) | 0.319 | −0.702 | 132.6 | O(Nₑ·G²·logG²) |

- **30.1% lower nRMSE** than tuned ETKF, **14.6% better** than GPR
- **~10× faster** than ETKF, **15× faster** than GPR
- Wilcoxon signed-rank test: p = 1.1×10⁻¹⁵, Cohen's d = 1.16 (large effect)
- Only method with positive Skill Score (zero-predictor baseline)
- Robust to both uniform and slope raydrop observation models (20/20 wins)

### Key Features

- Grid-free spectral representation: `z(x,y,t) = Σ wᵢ·cos(kxᵢ·x + kyᵢ·y - ωᵢ·t + φᵢ)`
- Dispersion constraint: `ω = √(g·|k|·tanh(|k|·d))` — hard constraint, no deep-water approximation
- Geometry-aware confidence weighting: `cⱼ = cos(θⱼ) / (dⱼ/d_ref)` (Lambertian projection × inverse range)
- Adaptive N: energy-based pruning + residual-triggered growth
- Two-phase estimation: Init (beamforming + coarse-to-fine) → Tracking (phase prediction + gradient update)
- Physics-based wave simulation (Gerstner waves + JONSWAP spectrum)
- Two LiDAR return models: uniform Fresnel-based (raydrop) and specular slope-dependent
- 57 unit tests, all passing (39 wavesim + 18 wavesplat)

---

## Research Context

| | |
|---|---|
| **Problem** | Ship-mounted forward-looking LiDAR produces only tens to hundreds of surface returns per frame |
| **Goal** | Recover continuous wave field `z(x,y,t)` from irregular sparse observations |
| **Method** | Grid-free spectral estimation with dispersion-constrained parameterization |
| **Baselines** | ETKF (tuned, 24-config sweep), GPR (Matérn 5/2), LETKF (Gaspari-Cohn) |
| **Target** | Ocean Engineering (Elsevier) |

---

## Repository Structure

```
wavesplat/
├── configs/                    # YAML configuration files
│   ├── estimation.yaml         # Main estimation config (physics, adaptive, tracking)
│   ├── default.yaml            # Local dev (76K rays, 1 worker)
│   └── server.yaml             # Server (304K rays, 4 workers, full resolution)
├── wavesim/                    # Simulation package (39 tests)
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
├── wavesplat/                  # Estimation algorithm (18 tests)
│   ├── model/
│   │   ├── spectral_gaussian.py  # N spectral Gaussian components
│   │   ├── dispersion.py         # ω = √(g|k|tanh(|k|d))
│   │   └── confidence.py         # Geometry-aware confidence weights
│   ├── optimizer/
│   │   ├── initializer.py        # Beamforming init (v1/v2/v2_konly)
│   │   ├── coarse_to_fine.py     # N=3→8→15 schedule
│   │   └── sequential_update.py  # Two-phase estimator + adaptive N + tracking
│   ├── baselines/
│   │   └── etkf_grid.py          # ETKF baseline
│   ├── evaluation/
│   │   └── metrics.py            # nRMSE, Hs error, spectral metrics
│   ├── data_loader.py            # HDF5 episode loader
│   └── run.py                    # Main inference entry point
├── scripts/                    # Experiment & paper generation scripts
│   ├── run_p1_comparison.py      # P1 vs ETKF (12 episodes)
│   ├── run_gpr_comparison.py     # WS vs ETKF vs GPR (10 episodes)
│   ├── run_raydrop_comparison.py # Uniform vs slope (10 paired episodes)
│   ├── run_ablation_seeded.py    # Ablation study (3-seed)
│   ├── run_multiseed_comparison.py # Multi-seed statistical validation
│   ├── run_extended_analysis.py  # Sparsity, BF trend, convergence analysis
│   ├── run_etkf_comparison.py    # ETKF historical comparison
│   ├── generate_paper_tables.py  # LaTeX tables (Table 1-6)
│   ├── generate_paper_figures.py # PDF figures (Fig 1-7)
│   ├── generate_qualitative_figure.py  # Fig 8: GT vs WS vs ETKF vs GPR heatmap
│   └── generate_tracking_animation.py  # Tracking animation (MP4)
├── paper/                      # Ocean Engineering journal paper
│   ├── wavesplat.tex             # Main LaTeX source (~1500 lines)
│   └── figures/                  # 9 PDF figures (fig1-fig8 + png)
├── results/                    # JSON experiment results (7 files)
│   ├── p1_comparison.json        # Main P1 comparison
│   ├── gpr_comparison.json       # GPR 3-way comparison
│   ├── raydrop_comparison.json   # Raydrop model comparison
│   ├── ablation_seeded.json      # Ablation study (3-seed)
│   ├── multiseed_comparison.json # Multi-seed statistics
│   ├── extended_analysis.json    # Extended analysis
│   └── etkf_comparison.json     # ETKF comparison (historical)
├── data/                       # Generated HDF5 episodes (100 total)
├── generate_dataset.py         # Batch dataset generation (parallel)
├── docs/
│   └── PRESENTATION_SUMMARY.md   # Detailed presentation summary (~600 lines)
└── requirements.txt            # Python dependencies
```

---

## Installation

```bash
pip install -r requirements.txt
# Or manually:
pip install numpy torch scipy h5py matplotlib pyyaml pytest scikit-learn
```

---

## Quick Start

### Run estimation on an episode

```bash
cd wavesplat
python3 -m wavesplat.run data/episode_BF4_5comp_seed0_uniform.h5 --adaptive_n
```

### Run full comparison experiments

```bash
python3 scripts/run_p1_comparison.py       # P1 vs ETKF (12 episodes)
python3 scripts/run_gpr_comparison.py      # WS vs ETKF vs GPR (10 episodes)
python3 scripts/run_raydrop_comparison.py  # Raydrop model comparison
python3 scripts/run_ablation_seeded.py     # Ablation study
python3 scripts/run_multiseed_comparison.py # Multi-seed statistics
```

### Generate paper tables and figures

```bash
python3 scripts/generate_paper_tables.py
python3 scripts/generate_paper_figures.py
python3 scripts/generate_qualitative_figure.py
```

### Run tests

```bash
python3 -m pytest wavesim/tests/ wavesplat/tests/ -v
```

---

## Experiment Results

### Full 100-episode Comparison (WS vs ETKF vs GPR)

| Method | Avg nRMSE | Skill Score | Speed | Improvement over ETKF |
|--------|-----------|-------------|-------|-----------------------|
| **WaveSplat** | **0.223** | **+0.034** | **13.4 ms** | **30.1%** |
| GPR | 0.261 | −0.140 | >200 ms | 18.2% |
| ETKF (tuned) | 0.319 | −0.702 | 132.6 ms | — |

### 12-episode Detailed Comparison (BF2-BF5)

- WaveSplat avg nRMSE: **0.211** (ETKF tuned: 0.358, **41.1% better**)
- Speed: **13.4 ms/frame** (ETKF: 132.6ms, **~10x faster**)
- Best case: BF3_2comp nRMSE=0.052 (8.5x better than ETKF)

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

### Failure Case

- BF2_1comp: WS nRMSE 0.31 — single low-frequency component with minimal spatial variation, insufficient gradient signal

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

100 episodes: BF2-BF5 × 1/2/5/15/30 components × 5 seeds × 2 raydrop models

Format: HDF5 with ground truth wave fields, LiDAR observations, and metadata.

---

## Configuration

Main config: `configs/estimation.yaml`
- `tracking_mode: w_phi` — kx/ky fixed during tracking
- `init_method: v2_konly` — beamforming for wavenumber, optimizer for amplitude/phase
- `adaptive_n: true` — energy-based pruning + residual-triggered growth
- `physics.use_dispersion: true` — hard dispersion constraint

---

## Scope and Limitations

- Valid for sea states BF2-BF5, non-breaking waves
- Linear wave superposition (Stokes 2nd-order: future work)
- Ship motion: currently fixed position
- High complexity (N>=30): beamforming init resolution limits nRMSE to ~0.30
