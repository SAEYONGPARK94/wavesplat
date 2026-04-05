# WaveSplat

**Grid-free Wave Field Estimation from Sparse Ship-mounted LiDAR via Spectral Gaussian Estimation**

---

## Overview

WaveSplat addresses the problem of reconstructing a continuous wave height field `z(x,y,t)` from the extremely sparse surface point clouds captured by a forward-looking LiDAR mounted on a ship.

This repository contains **WaveSim**, the physics-based LiDAR simulation environment used to generate training and evaluation data. The estimation algorithm (WaveSplat) will be added in Phase 3.

### Key Features

- Physics-based wave simulation (Gerstner waves + JONSWAP spectrum)
- Two LiDAR return models: uniform Fresnel-based and specular slope-dependent
- Full dispersion relation `ω = √(g·k·tanh(k·d))` — no deep-water approximation
- Q-parameter constraint enforcement to prevent particle crossing
- HDF5 dataset output with ground truth wave fields
- 39 unit tests, all passing

---

## Research Context

| | |
|---|---|
| **Problem** | Ship-mounted forward-looking LiDAR produces only tens to hundreds of surface returns per frame |
| **Goal** | Recover continuous wave field `z(x,y,t)` from irregular sparse observations |
| **Method** | Grid-free Spectral Gaussian Estimation with dispersion-constrained parameterization |
| **Downstream** | Hydrofoil predictive control (Paper 2) |

---

## Repository Structure

```
wavesplat/
├── configs/                    # YAML configuration files
│   ├── default.yaml            # Local dev (76K rays, 1 worker)
│   └── server.yaml             # Server (304K rays, 4 workers, full resolution)
├── wavesim/                    # Simulation package (Phase 1-B, complete)
│   ├── config.py               # Config loader + dataclasses
│   ├── wave_generator.py       # Gerstner wave + JONSWAP spectrum
│   ├── lidar_simulator.py      # Ray generation + spatial-filtered ray-mesh intersection
│   ├── raydrop_uniform.py      # Model A: Fresnel/distance-based dropout
│   ├── raydrop_slope.py        # Model B: Specular backscatter dropout
│   ├── gt_generator.py         # Ground truth wave field evaluation
│   ├── noise_model.py          # Sensor noise + confidence weighting
│   ├── data_writer.py          # HDF5 save/load
│   ├── pipeline.py             # End-to-end episode pipeline
│   └── tests/                  # 39 unit tests
│       ├── test_wave_generator.py
│       ├── test_raycast.py
│       ├── test_raydrop.py
│       ├── test_gt.py
│       └── test_pipeline.py
├── generate_dataset.py         # Batch dataset generation (parallel)
├── wavesplat/                  # Estimation algorithm (Phase 3, TBD)
├── experiments/
├── data/                       # Generated HDF5 episodes
└── docs/
```

---

## Installation

```bash
pip install numpy scipy matplotlib h5py pyyaml pytest
```

Optional (for faster raycasting):
```bash
pip install open3d
```

---

## Quick Start

### Run tests

```bash
cd wavesplat
python3 -m pytest wavesim/tests/ -v
```

Expected: **39 passed**

### Generate a single episode

```bash
# With config file (recommended)
python3 -m wavesim.pipeline --config configs/default.yaml \
    --sea_state BF4 --n_components 30 --seed 42

# Override specific parameters via CLI
python3 -m wavesim.pipeline --config configs/default.yaml \
    --sea_state BF4 --n_components 30 --seed 42 \
    --raydrop slope --duration 10.0 --n_pts 100
```

Sea states: `BF2` (Hs=0.3m), `BF3` (Hs=0.6m), `BF4` (Hs=1.2m), `BF5` (Hs=2.0m)

Raydrop models: `uniform` (Model A), `slope` (Model B)

### Generate full dataset

```bash
# 1. Preview the generation plan
python3 generate_dataset.py --config configs/default.yaml --dry-run

# 2. Run (sequential, local dev)
python3 generate_dataset.py --config configs/default.yaml --skip-existing

# 3. Run on server (parallel, full resolution)
python3 generate_dataset.py --config configs/server.yaml --skip-existing
```

The dataset matrix is defined in the config YAML (10 scenarios × 5 seeds × 2 raydrop models = 100 episodes).

---

## Configuration

All simulation parameters are managed via YAML config files in `configs/`.

| Config | LiDAR Rays | Mesh | Workers | Use Case |
|--------|-----------|------|---------|----------|
| `default.yaml` | 76K/frame | 40×30 | 1 | Local development, testing |
| `server.yaml` | 304K/frame | 80×60 | 4 | Paper-quality dataset (32GB+ RAM, i9) |

### Key config sections

```yaml
lidar:          # Seyond Falcon K1 sensor parameters
raycast:        # Batch size, spatial grid resolution
mesh:           # Raycast mesh resolution
gt_grid:        # Ground truth evaluation grid
episode:        # Duration, timestep, depth, target points
dataset:        # Scenario matrix, seeds, workers
```

CLI arguments (`--sea_state`, `--n_pts`, `--duration`, etc.) override config values.

---

## Physics

### Wave Generation

Gerstner wave superposition with JONSWAP spectrum sampling.

```
x(u,v,t) = u - Σ Qᵢ·Aᵢ·kxᵢ/kᵢ · sin(kᵢ·r - ωᵢ·t + φᵢ)
y(u,v,t) = v - Σ Qᵢ·Aᵢ·kyᵢ/kᵢ · sin(kᵢ·r - ωᵢ·t + φᵢ)
z(u,v,t) =     Σ Aᵢ           · cos(kᵢ·r - ωᵢ·t + φᵢ)
```

Particle crossing prevention: `Σ Qᵢ·ωᵢ·Aᵢ ≤ 1` enforced at all times.

### LiDAR Return Models

**Model A — Uniform (Fresnel/distance)**
```
P_return = η · R_fresnel(θᵢ) · cos(θᵢ) / (d/d_ref)²
```

**Model B — Slope-dependent (Specular backscatter)**
- Return only when reflected ray direction aligns with sensor within `α_threshold = 5°`
- Physically motivated by forward-looking geometry at grazing angles

### Trust Chain

| Component | Physical Law | Validated Against |
|-----------|-------------|-------------------|
| Wave mesh | Gerstner exact solution | Analytic RMSE < 1e-10 m |
| Dispersion | ω = √(gk·tanh(kd)) | Analytic deep/shallow water limits |
| Ray intersection | Möller-Trumbore (1997) | Known surface geometry |
| Fresnel reflectance | Fresnel equations (n=1.321 @1550nm) | Textbook values at reference angles |
| Q constraint | Jacobian > 0 condition | Monitored every frame |

---

## Dataset Format (HDF5)

```
episode_BF4_30comp_seed42_uniform.h5
├── /config/           sea_state, Hs, Tp, n_components, wave_components, lidar_config
├── /frames/           timestamps, lidar_points, lidar_valid_mask,
│                      lidar_incidence_angles, lidar_confidence,
│                      gt_wave_field, gt_grid_x, gt_grid_y
├── /raydrop_model/    model_type, parameters
└── /gt_params/        Hs_true, Tp_true, spectrum_1d
```

---

## Roadmap

- [x] Phase 1-B: WaveSim — physics-based simulation (complete)
- [x] Phase 2-A: Config system + batch generation pipeline (complete)
- [ ] Phase 2-B: Dataset generation — 100 episodes (10 scenarios × 5 seeds × 2 models)
- [ ] Phase 3: WaveSplat estimation algorithm
  - `wavesplat/model/spectral_gaussian.py`
  - `wavesplat/optimizer/sequential_update.py`
  - `wavesplat/baselines/` — GPR, ETKF-grid
- [ ] Phase 4: Experiments + paper submission

---

## Scope and Limitations

- Valid for sea states BF3–BF5, non-breaking waves
- Linear wave superposition (Stokes 2nd-order correction: future work)
- Ship motion: currently fixed position (underway scenarios in Phase 2)
- No wave–wave interaction (bound harmonics)
