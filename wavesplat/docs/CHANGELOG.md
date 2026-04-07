# WaveSplat Changelog

프로젝트 개발 이력을 시간순으로 기록합니다.

---

## [0.5.0] — 2026-04-07

### Added — Raydrop 모델 비교 실험
- `scripts/run_raydrop_comparison.py` — Uniform vs Slope 10 paired episodes 비교
- `results/raydrop_comparison.json` — 비교 결과 데이터
- `paper/figures/fig5_raydrop.pdf` — Grouped bar chart (4-way comparison)
- `paper/wavesplat.tex` — Table 4 (raydrop), Discussion 단락, Conclusion 업데이트
- `scripts/generate_paper_tables.py` — `table4_raydrop()` 추가
- `scripts/generate_paper_figures.py` — `fig5_raydrop()` 추가

### Results
- WaveSplat wins 10/10 under both raydrop models
- Slope improves both: WS -9.1%, ETKF -16.2%
- ETKF benefits more from specular observations (grid methods leverage spatial structure)

---

## [0.4.0] — 2026-04-07

### Added — P2: 논문 초안 작성
- `paper/wavesplat.tex` — IEEE conference format 논문
  - Sections: Abstract, Introduction, Related Work, Method, Experiments, Results, Discussion, Conclusion
  - Table 1: Main comparison (12 episodes)
  - Table 2: Ablation study (3-seed)
  - Table 3: Adaptive N behavior
- `scripts/generate_paper_tables.py` — LaTeX 테이블 생성 (Table 1-3)
- `scripts/generate_paper_figures.py` — PDF 그림 생성 (Fig 1-4)
- `paper/figures/` — fig1_comparison, fig2_adaptive_n, fig3_speed, fig4_ablation

**Commit**: `dc00878`

---

## [0.3.0] — 2026-04-07

### Added — P1: 성능 튜닝 (Adaptive N + Multi-frame Tracking)
- `wavesplat/optimizer/sequential_update.py` — Adaptive N (energy-based pruning + residual growth)
  - `prune_threshold=0.10` (keep 90% energy)
  - `grow_residual_threshold=0.05` (trigger beamforming-based growth)
  - Safety check: rollback if residual > 1.3× after pruning
  - `max_n=30`
- `wavesplat/optimizer/sequential_update.py` — Multi-frame tracking window (k=3)
- `wavesplat/run.py` — `--adaptive_n`, `--max_n`, `--tracking_window` CLI args
- `scripts/run_p1_comparison.py` — P0 vs P1 vs ETKF (12 episodes)
- `results/p1_comparison.json` — Authoritative P1 results

### Results (P1)
- P1 avg nRMSE: **0.211** (P0: 0.226, +6.4% | ETKF: 0.436, +51.5%)
- 12/12 episodes WaveSplat wins
- Speed: 13.4 ms/frame (ETKF 31.7ms, 2.4× faster)
- Adaptive N: BF4_1comp 15→1, BF3_2comp 15→2, BF5_15comp 15→21

**Commit**: `1564ec8`

---

## [0.2.1] — 2026-04-06

### Added — P0: 실험 기반 확보
- **P0-1: Init 개선** — Beamforming v2_konly, 평균 18% tracking 개선
- **P0-2: Ablation study** — 3-seed, A1(no disp) competitive, A2(no conf) slope 필수
- **P0-3: ETKF comparison** — 12 episodes × 3 methods
- `results/ablation_seeded.json` — Ablation 결과
- `results/etkf_comparison.json` — ETKF 비교 결과

**Commit**: `2dfc932`

---

## [0.2.0] — 2026-04-05

### Added
- `wavesim/visualize.py` — `save_video()` 함수 추가
- `wavesplat/` — Estimation algorithm (Phase 3)
  - `model/spectral_gaussian.py` — N spectral Gaussian components
  - `model/dispersion.py` — Dispersion relation
  - `model/confidence.py` — Fresnel confidence weights
  - `optimizer/initializer.py` — Beamforming init
  - `optimizer/coarse_to_fine.py` — N=3→8→15 schedule
  - `optimizer/sequential_update.py` — Two-phase estimator
  - `baselines/etkf_grid.py` — ETKF baseline
  - `evaluation/metrics.py` — nRMSE, spectral metrics
  - `data_loader.py` — HDF5 episode loader
  - `run.py` — Main inference entry point

### Generated
- `data/` — 100 episodes (BF2-BF5 × 1/2/5/15/30comp × seed0-4 × uniform/slope)

---

## [0.1.1] — 2026-03-30

### Added
- `README.md` — 프로젝트 개요, 물리 문서, 설치/실행 가이드, 로드맵

**Commit**: `cf6e5d0`

---

## [0.1.0] — 2026-03-30

### Added — WaveSim 시뮬레이션 패키지 (Phase 1-B 완료)

#### 핵심 모듈
- `wavesim/wave_generator.py` — Gerstner wave + JONSWAP 스펙트럼
- `wavesim/lidar_simulator.py` — 전방 조사 LiDAR 시뮬레이션
- `wavesim/raydrop_uniform.py` — Model A: Fresnel 균일 dropout
- `wavesim/raydrop_slope.py` — Model B: Specular backscatter dropout
- `wavesim/gt_generator.py` — Ground truth 파 필드
- `wavesim/noise_model.py` — 센서 노이즈 + 신뢰도
- `wavesim/data_writer.py` — HDF5 저장/로드
- `wavesim/pipeline.py` — 엔드투엔드 파이프라인

#### 테스트
- `wavesim/tests/` — 39개 단위/통합 테스트 전체 통과

**Commit**: `3070038`

---

## Roadmap

- [x] **Phase 1-B**: WaveSim 물리 기반 시뮬레이션
- [x] **Phase 2**: 데이터셋 생성 — 100 에피소드
- [x] **Phase 3**: WaveSplat 추정 알고리즘
- [x] **P0**: 실험 기반 확보 (Init, Ablation, ETKF comparison)
- [x] **P1**: 성능 튜닝 (Adaptive N, Multi-frame tracking)
- [x] **P2**: 논문 초안 작성 (IEEE conference format)
- [x] **Raydrop 비교**: Uniform vs Slope 분석
- [ ] (Optional) 추가 seed 통계 검증, real LiDAR data
