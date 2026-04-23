# WaveSplat Changelog

프로젝트 개발 이력을 시간순으로 기록합니다.

---

## [0.8.0] — 2026-04-23

### Added — 투고 준비 완료 (Ocean Engineering)

#### 논문 최종 검토 반영 (v0.7.1)
- Algorithm Table Step 3.5: fine refinement (2-stage beamforming) 추가
- Dir. R²=-0.25 물리 원인 Discussion 섹션 추가 (§ Directional estimation deficit)
  - Forward LiDAR ±θ 방향 모호성 설명, yaw variation 완화 제안
- Grown components C2F 없음 명시 (재현성)
- 13.4 vs 14.2ms 불일치: Table caption에 median(13.3ms) 추가로 해명
- 컴파일 에러 수정: L1393 `$\sim$120^\circ$` → `${\sim}120^\circ$`

#### 투고 패키지 완성
- `paper/cover_letter.tex` — Ocean Engineering 커버레터 (컴파일 완료)
  - 4개 핵심 기여 설명, 주요 결과 수치, reproducibility statement
  - 리뷰어 3명 추천 포함
- `paper/submission/graphical_abstract.pdf` — 3-panel 요약 그림
  - (a) Sparse LiDAR observations, (b) WaveSplat pipeline, (c) 100-episode results

#### 실험 스크립트 추가
- `scripts/run_tau_sensitivity.py` — τ_R ∈ {0.02, 0.05, 0.08, 0.10, 0.15}m 민감도 분석
- `scripts/run_gpr_confidence_ablation.py` — GPR + confidence 4-way ablation
  - WS+conf / WS-conf / GPR / GPR+conf 비교
  - C3 novelty가 WaveSplat 고유인지 general technique인지 검증

#### 기타
- Thesis 컴파일 (pdflatex) 에러 없음 확인
- Memory 파일 업데이트 (project_status, ocean_eng_conversion)

### 논문 투고 준비 상태
- Paper 컴파일: ✅ (pdflatex + bibtex 에러 없음)
- Cover letter: ✅
- Graphical abstract: ✅
- CRediT, Data availability, Declaration: ✅ (이전 세션)

---

## [0.7.0] — 2026-04-23

### Fixed — Paper/Thesis 재현성 및 정확성 수정 (4-persona 검토 기반)

#### 문제 1: Fine Refinement 누락 (재현성 필수)
- **paper/wavesplat.tex** §3 빔포밍 섹션에 2단계 빔포밍 기술 추가
  - Coarse 81×81 (Δk=0.10 rad/m) + Fine 41×41 (Δk=0.01 rad/m, ±0.2 rad/m)
  - "Without this refinement, ~17% |k| accuracy limit at BF4" 재현성 경고 추가
- **thesis/thesis_main.tex** 에 동일 내용 한국어 동기화
  - Table Step 3.5 추가
  - 본문 단락 추가 (greedy 추출 뒤)
  - Algorithm 1 Step 2.5 pseudocode 추가

#### 문제 2: Confidence Normal 가정 — sim/real 구분 명시
- **paper/wavesplat.tex** L419: "local surface normal" → sim/real 구분 설명으로 교체
  - Sim: Gerstner true normal, Real: flat 가정 (n≈ẑ), 오차 <2% (θ<30°) ~ 9% (θ~60°)
- **thesis/thesis_main.tex** L1098: 동일 내용 한국어로 추가

#### 문제 3: A1 Ablation 재해석 — bound harmonics 정량 설명 추가
- **paper/wavesplat.tex** L1120: A1 해석 단락에 "bound harmonics <0.15% at BF4" 정량 근거 추가
  - R̄=0.51 >> 물리적 기대값 → over-fitting artifact 논리 강화
  - `holthuijsen2007` 인용 추가

#### 문제 4: τ_R = 0.05m 유도 보강
- **paper/wavesplat.tex** L544: 값만 제시 → 이론 하한 (0.022m) × 2.3× safety margin 유도 삽입

### Added — Limitations 보강
- **paper/wavesplat.tex** Limitations 섹션에 2개 항목 추가:
  - Surface normal 실측 근사 한계 (grazing ~9% bias)
  - Fresnel-aware 명칭 정직화 (Lambertian projection × inverse distance 공식 명시)
- Current estimation 항목에 "0.5m/s → ~5% BF4 bias" 정량 수치 추가

### Added — Bibliography
- **paper/references.bib**: `born1999` (Born & Wolf, Principles of Optics) 추가

**Commits**: 이번 세션 수정

---

## [0.6.0] — 2026-04-22

### Fixed — 빔포밍 그리드 설명 버그 수정 (코드↔문서 불일치)

#### 핵심 버그
- `wave_components` 컬럼 인덱스 오류 발견 및 수정
  - 기존: `wc[:,1]`(omega)를 kx로, `wc[:,2]`(kx)를 ky로 잘못 참조
  - 수정: `wc[:,2]`=kx, `wc[:,3]`=ky (컬럼 순서: [A, omega, kx, ky, phi, Q])
  - 영향: beamforming 시각화에서 GT 마커 위치가 전부 틀렸음

#### 논문/코드 불일치 수정
- 빔포밍 격자: polar 64×128, |k|∈[0.02,0.50] rad/m, 15° 억제
  → **Cartesian 81×81, kx,ky∈[-4.0,4.0] rad/m, 0.15 rad/m 억제**
- 수정 대상:
  - `paper/wavesplat.tex` — 본문, Algorithm 표
  - `thesis/thesis_main.tex` — 본문, 의사코드, 파라미터 표 (7곳)
  - `docs/WaveSplat_Algorithm_Guide.html`
  - `docs/WaveSplat_Algorithm_Guide.pptx` — Slide 46/49/54
  - `docs/labmeeting_wavesplat.pptx` — Slide 6

### Changed — C1 Novelty 재프레이밍 (tracking 역할 명확화)

#### 변경 이유
- 페르소나 토론: tracking이 단순 sea state(BF3/4 5comp)에서 C2F 대비 nRMSE 2~3× 악화
- 원인: w_phi 모드가 고주파 성분(|k|≥0.4 rad/m) 미탐지를 고착시킴

#### C1 수정 (paper + thesis)
- 기존: "frame-to-frame temporal tracking via dispersion-based phase propagation" (accuracy 함의)
- 수정: tracking = **계산 효율성** (~10 steps vs ~200 steps) + 연속 동작
- "Estimation accuracy depends on initialization quality" 명시

#### Limitations 추가 (paper + thesis)
- Tracking phase dependence on initialization 항목 신설
- |k|≥0.4 rad/m 미탐지 시 tracking nRMSE 2~3× 악화 조건 기술

### Added — 빔포밍 시각화 스크립트

- `scripts/plot_beamforming_3d.py` — 4 에피소드 × 3단계 비교 그림 생성
  - Raw 파워맵 → Greedy 추출 → C2F 최적화 후
  - GT 마커 정확한 컬럼 인덱스 사용

### Analysis — 알고리즘 검증 결과 (수치)

```
빔포밍 GT 탐지율 (P/Pmax):
  BF3 GT[0~2]: 0.168~1.000 → 정확 탐지
  BF3 GT[3,4]: 0.000       → 탐지 불가 (|k|=0.67, 1.01 rad/m)
  BF4 GT[3,4]: 0.001~0.006 → 탐지 불가 (|k|=0.42, 0.62 rad/m)

C2F 최적화 후 nRMSE:
  BF2 1comp:  0.264 (단일 성분이지만 SNR 낮아 과파라미터화)
  BF3 5comp:  0.075 (저주파 성분 정확, 고주파 미탐지)
  BF4 5comp:  0.067 (동일)
  BF4 15comp: 0.217 (6/15 정확, 고주파 계통적 누락)
```

**Commits**: `92f98d1` → `9eb6b29` → `89c0f7a` → `bd01f21` → `617b7e9`

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
