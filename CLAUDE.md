# WaveSplat Project — Claude Context

## Project Overview
Grid-free Spectral Gaussian wave field estimation from sparse ship-mounted LiDAR.
Target: OCEANS 2026 conference paper submission.

## Repository Layout
```
wavesplat/                   # Root package directory
├── wavesim/                 # Wave simulation + LiDAR (39 tests)
├── wavesplat/               # Estimation algorithm (18 tests, 57 total)
│   ├── model/               # spectral_gaussian, dispersion, confidence
│   ├── optimizer/            # initializer, coarse_to_fine, sequential_update
│   ├── baselines/etkf_grid.py
│   ├── evaluation/metrics.py
│   └── run.py               # Main entry point
├── scripts/                 # 11 experiment/paper scripts
├── paper/                   # Ocean Engineering LaTeX (wavesplat.tex) + 9 figures
├── results/                 # 7 JSON result files
├── configs/                 # YAML configs (estimation.yaml, server.yaml, default.yaml)
├── data/                    # HDF5 episodes (100 total)
├── docs/PRESENTATION_SUMMARY.md  # 발표 요약 (~600줄)
└── requirements.txt
```

## Key Commands
```bash
# Run estimation
python3 -m wavesplat.run data/episode_BF4_5comp_seed0_uniform.h5 --adaptive_n

# Run tests
python3 -m pytest wavesim/tests/ wavesplat/tests/ -v

# Generate paper figures/tables
python3 scripts/generate_paper_tables.py
python3 scripts/generate_paper_figures.py

# Run comparisons
python3 scripts/run_p1_comparison.py       # Main 12-episode comparison
python3 scripts/run_gpr_comparison.py      # GPR 3-way comparison
python3 scripts/run_raydrop_comparison.py  # Raydrop model comparison
```

## Current State (2026-04-07)
- **All phases complete**: P0 (init+ablation+ETKF), P1 (adaptive N+tracking), P2 (paper)
- **57 tests all passing** (39 wavesim + 18 wavesplat)
- **Paper**: Ocean Engineering format ~1500 lines, 12 tables, 9 figures
- **Key result**: WS avg nRMSE 0.223 vs ETKF 0.319 vs GPR 0.276
- **Speed**: WS 13.4ms, ETKF 132.6ms (tuned, Ne=100), GPR >200ms

## Important Design Decisions
- `tracking_mode: w_phi` — kx/ky fixed during tracking, only w+φ updated
- `init_method: v2_konly` — beamforming for kx/ky, optimizer handles w/φ
- Adaptive N: energy-based pruning + residual-triggered growth with safety rollback
- Dispersion hard constraint: `ω = √(g·|k|·tanh(|k|·d))`

## Language
User prefers Korean for discussion, English for code/technical terms.
