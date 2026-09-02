# WaveSplat

**Grid-Free Spectral Wave Field Estimation from Sparse Shipboard LiDAR**

---

## Overview

WaveSplat addresses the problem of reconstructing a continuous wave height field `z(x,y,t)` from the extremely sparse surface point clouds captured by a forward-looking LiDAR mounted on a ship.

Unlike grid-based approaches (e.g., ETKF), WaveSplat represents the wave field as a weighted sum of spectral components with physics-based constraints (dispersion relation, geometry-aware confidence weighting), enabling grid-free estimation that adapts model complexity to wave scene complexity.

### Key Results

> ⛔**2026-08-13 감사 — 이 절의 헤드라인 표는 철회됨 (금지 유지 · 사유는 3라운드에서 교체됨).**
> `0.223 vs 0.319 (= 30.1%)` · `13.4 ms` · `132.6 ms` · `~10× faster` · `84/100` 인용 금지.
>
> ⚠**1라운드가 적은 "결과 파일이 저장소에 존재하지 않는다" 는 거짓으로 판명됐다** — 그 감사관은
> 스스로 적었듯 `results/` 를 **한 파일도 열지 않았다**(= 범위 없는 부재 주장). 네 수치는
> `results/b1_full_comparison.json`(**git 추적**, 커밋 `e57528c`, 100 에피소드, `etkf_config`
> 내장)에서 **전부 재도출된다**: `WaveSplat.avg_nRMSE` **0.2229** · `ETKF.avg_nRMSE` **0.3189** ·
> `WaveSplat.avg_inference_ms` **13.3760** · `ETKF.avg_inference_ms` **132.6387**.
>
> **금지가 유지되는 진짜 사유는 셋이다:**
> 1. **프레임 불일치** — 같은 파일에서 `WaveSplat.n_frames` 는 **120–220**, `ETKF`·`GPR-1` 은
>    **301** 이고 **100/100 에피소드 전부에서 WS 가 더 적다** ⟹ 두 팔이 **같은 프레임에서 채점된
>    적이 없다**. ⟹ ⛔그 파일의 **`84/100` 승패비도 인용 금지**.
>    (참고: S13 배치는 이 점검을 통과했다 — 불일치 0/128.)
> 2. **기준선 선택** — 그 ETKF 는 `grid 40×30 · ne=100 · mult_inflation 1.0` 구성이다. **같은
>    `WaveSplat` 열**을 튜닝된 LETKF 상대로 재도출하면 `results/100ep_ws_letkf.json` 에서
>    **0.2229 vs 0.2278, 51/100, p = 0.9204 = 널**이다.
> 3. **지연은 프로세스 간 비교** — 동일 프로세스·1스레드 정직값은 **1.5–2.6×**
>    (CPU = **AMD Ryzen 7 7800X3D**).
>
> 근거: `reference_citation_bans.md`, `tmp/AUDIT_recheck_2026-08-13/`

**검증된 재도출** (`results/100ep_ws_letkf.json`, 100 에피소드):

| Method | Avg nRMSE | Verdict |
|--------|-----------|---------|
| WaveSplat | 0.2229 | — |
| LETKF | 0.2278 | **51/49 wins, p = 0.9204 → NULL (no difference)** |

- ⛔**"30% lower nRMSE" 는 근거가 없다.** 이 표본에서 정확도 차이는 **통계적으로 널**이다.
- Latency: **1.5–2.6×** in the same process, single thread (CPU = **AMD Ryzen 7 7800X3D**).
  ⛔RETRACTED — `13.4 ms` / `132.6 ms` / `~10×` 는 **프로세스 간 비교**의 산물이다
  (스레드 미고정 포함). 값 자체는 `b1_full_comparison.json` 에 있으나 그렇게 잰 값이다.
- ⛔RETRACTED (2026-08-13 감사) — Skill Score (`+0.034` / `−0.140` / `−0.702`), GPR `0.261`,
  Wilcoxon `p`, Cohen's `d`, raydrop `20/20`. **사유는 두 갈래다**: Skill Score 열과 GPR `0.261`
  은 **위 헤드라인과 같은 `b1_full_comparison.json` 산출**이라 **프레임 불일치·기준선 선택 결함을
  그대로 물려받고**, Wilcoxon `p`·Cohen's `d`·raydrop `20/20` 은 그 파일에서 **그대로 재현되지
  않는**(구버전 12-에피소드 계열) 값이다. 근거: `reference_citation_bans.md`
- Asymptotic cost (analytic, unaffected): WaveSplat O(M·N) · GPR O(M³) · ETKF O(Nₑ·G²·logG²)

### Key Features

- Grid-free spectral representation: `z(x,y,t) = Σ wᵢ·cos(kxᵢ·x + kyᵢ·y - ωᵢ·t + φᵢ)`
- Dispersion constraint: `ω = √(g·|k|·tanh(|k|·d))` — hard constraint, no deep-water approximation
- Geometry-aware confidence weighting: `cⱼ = cos(θⱼ) / (dⱼ/d_ref)` (Lambertian projection × inverse range)
- Adaptive N: energy-based pruning + residual-triggered growth
- Two-phase estimation: Init (beamforming + coarse-to-fine) → Tracking (phase prediction + gradient update)
- Physics-based wave simulation (Gerstner waves + JONSWAP spectrum)
- Two LiDAR return models: uniform Fresnel-based (raydrop) and specular slope-dependent
- 204 unit tests, all passing (85 wavesim + 119 wavesplat) — 2026-08-13 실측
  `204 passed, 0 failed, 0 skipped` (기존 196 + 신규 8 =
  `wavesplat/tests/test_shipping_defaults.py` 배포 기본값 회귀 테스트)

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
├── wavesim/                    # Simulation package (85 tests)
│   ├── config.py               # Config loader + dataclasses
│   ├── wave_generator.py       # Gerstner wave + JONSWAP spectrum
│   ├── lidar_simulator.py      # Ray generation + ray-mesh intersection
│   ├── raydrop_uniform.py      # Model A: Fresnel/distance-based dropout
│   ├── raydrop_slope.py        # Model B: Specular backscatter dropout
│   ├── gt_generator.py         # Ground truth wave field evaluation
│   ├── noise_model.py          # Sensor noise + confidence weighting
│   ├── data_writer.py          # HDF5 save/load
│   ├── pipeline.py             # End-to-end episode pipeline
│   └── tests/                  # 85 unit tests
├── wavesplat/                  # Estimation algorithm (119 tests)
│   ├── model/
│   │   ├── spectral_gaussian.py  # N spectral Gaussian components
│   │   ├── dispersion.py         # ω = √(g|k|tanh(|k|d))
│   │   └── confidence.py         # Geometry-aware confidence weights
│   ├── optimizer/
│   │   ├── initializer.py        # Init router `auto` (shipping default) + v1/v2/v2_konly
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
│   ├── waveprior.tex             # Main LaTeX source (2026-09-02 개명; 구명 wavesplat.tex)
│   └── figures/                  # 9 PDF figures (fig1-fig8 + png)
├── results/                    # JSON experiment results (96 files, + direction1/2/3)
│   ├── p1_comparison.json        # Main P1 comparison
│   ├── gpr_comparison.json       # GPR 3-way comparison
│   ├── raydrop_comparison.json   # Raydrop model comparison
│   ├── ablation_seeded.json      # Ablation study (3-seed)
│   ├── multiseed_comparison.json # Multi-seed statistics
│   ├── extended_analysis.json    # Extended analysis
│   └── etkf_comparison.json     # ETKF comparison (historical)
├── data/                       # Generated HDF5 episodes (110 total, 2026-08-13 실측)
├── generate_dataset.py         # Batch dataset generation (parallel)
├── docs/
│   └── PRESENTATION_SUMMARY.md   # Detailed presentation summary (683 lines, 2026-08-13 실측)
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
python3 -m wavesplat.run --episode data/episode_BF4_5comp_seed0_uniform.h5 --adaptive_n true
```

> ⚠**2026-08-13 정정** — 이전에 적혀 있던
> `python3 -m wavesplat.run data/episode_... --adaptive_n` 은 **실행되지 않는다**(실측 EXIT=2).
> `run.py` 에 위치인자가 없어 에피소드는 `--episode` 로 줘야 하고, `--adaptive_n` 은
> `{true,false}` 값을 받는 플래그다.

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

### Full 100-episode Comparison

> ⛔**2026-08-13 감사 — 아래에 있던 3-way 표는 철회됨.** 근거: `reference_citation_bans.md`

| Method | Avg nRMSE | n | Source | Verdict |
|--------|-----------|---|--------|---------|
| WaveSplat | 0.2229 | 100 | `results/100ep_ws_letkf.json` | — |
| LETKF | 0.2278 | 100 | `results/100ep_ws_letkf.json` | 51/49 wins, p = 0.9204 (양측 이항 부호검정) → **NULL** |

- ⚠**GPR 은 이 표에서 뺐다 (2026-08-13 3라운드 감사)** — 재도출로 확인된 유일한 GPR 값
  `0.276` 은 `results/gpr_comparison.json` 의 **10 조건** 산출(평균 0.2756)이고, 같은 파일의
  동반 팔은 `WS-P1 0.2325` · `ETKF 0.4499` 다. **다른 코퍼스·다른 기준선**이므로 100-에피소드
  표에 라벨 없이 섞으면 안 된다(`docs/STUDY_GUIDE.md` 는 이 3수치를 "구버전"이라 부른다).
  ⟹ 인용할 때는 반드시 **"GPR 0.276, n=10, `gpr_comparison.json`"** 형태로 쓸 것.
  근거: `tmp/AUDIT_recheck_2026-08-13/R3_regression.md §M-5`
- ⛔RETRACTED (2026-08-13 감사, 3라운드에서 **사유 교체**) — `0.223 / 0.319 / 30.1%`,
  Skill Score 열, `13.4 ms`, `132.6 ms`, GPR `0.261`, `84/100` 및 "18.2% improvement".
  ⚠**"어느 결과 파일에서도 재현되지 않는다" 는 틀렸다** — 이 값들은
  `results/b1_full_comparison.json`(커밋 `e57528c`, 100 ep)에서 재도출된다
  (`0.2229` / `0.3189` / `13.3760 ms` / `132.6387 ms`). 금지 사유는 ①**프레임 불일치**
  (`WaveSplat.n_frames` 120–220 vs `ETKF`·`GPR-1` 301, **100/100 에피소드에서 WS 가 더 적음**
  ⟹ 같은 프레임에서 채점된 적 없음 ⟹ `84/100` 도 금지) ②**기준선 선택**(그 ETKF 는
  `grid 40×30 · ne=100 · mult_inflation 1.0`; 같은 WS 열을 튜닝 LETKF 와 붙이면 위 표대로 **널**)
  ③**지연은 프로세스 간 비교**다. 근거: `reference_citation_bans.md`

### 12-episode Detailed Comparison (BF2-BF5)

- ⛔RETRACTED (2026-08-13 감사) — `0.211 vs 0.358 (41.1% better)` · `best case 8.5x` 는
  **12-에피소드 구버전 실행**의 산물이다(사유 유지).
- ⛔RETRACTED (2026-08-13 감사, 3라운드에서 **사유 교체**) — `13.4 ms/frame` ·
  `ETKF 132.6 ms` · `~10x faster`. 이 두 지연값은 `results/b1_full_comparison.json` 에
  **실재한다**(`13.3760` / `132.6387`). 금지 사유는 "재현 불가"가 아니라 **프로세스 간
  비교**(스레드 미고정 포함)이고, 정확도 쌍은 위 §Full 100-episode 의 **프레임 불일치·기준선
  선택** 사유를 함께 받는다. 근거: `reference_citation_bans.md`
- 허용되는 지연 서술: 동일 프로세스·1스레드에서 **1.5–2.6×** (CPU = AMD Ryzen 7 7800X3D).

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

레거시 코퍼스 100 에피소드 (`data/` 전체는 2026-08-13 실측 **110 파일**):
BF2-BF5 × 1/2/5/15/30 components × 5 seeds × 2 raydrop models

> ⚠**2026-08-13 감사 — 이 코퍼스의 독립성은 에피소드 수보다 훨씬 낮다.** `data/` 의 100 에피소드는
> **레거시 생성기 산출물**이다(HDF5 `config` 에 `random_amplitudes`/`random_frequencies`/
> `decorrelate_stream` 키가 **부재**). 독립 draw 를 세면 진실장 **50** · 진폭 **10** · omega **10** ·
> **위상/방향 5**(시드 0–4 뿐) ⟹ 클러스터 단위 바닥 p = 2⁻⁵ = **0.031**(단측).
> **이 코퍼스에서 나온 p 값을 에피소드 단위 n=100 으로 인용하지 말 것.**
> 2026-08-07 이후 생성된 데이터(`data_s7_*`·`data_s8_*`·`data_s9_*`·`data_indep*`)는 수정된
> 생성기로 만들어졌고 세 플래그가 전부 ON 이다.

Format: HDF5 with ground truth wave fields, LiDAR observations, and metadata.

---

## Configuration

Main config: `configs/estimation.yaml`
- `tracking_mode: w_phi` — kx/ky fixed during tracking
- `init_method: auto` — GT-free init router (shipping default since the 2026-07-26 flip;
  see `wavesplat/run.py:43` and `configs/estimation.yaml:24`).
  ⛔`v2_konly` 는 더 이상 출하 기본값이 아니다 — 명시 지정 시에만 쓰인다.
- `adaptive_n: true` — energy-based pruning + residual-triggered growth
- `physics.use_dispersion: true` — hard dispersion constraint

---

## Scope and Limitations

- Valid for sea states BF2-BF5, non-breaking waves
- Linear wave superposition (Stokes 2nd-order: future work)
- Ship motion: currently fixed position
- High complexity (N>=30): beamforming init resolution limits nRMSE to ~0.30
