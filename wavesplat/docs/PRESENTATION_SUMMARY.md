# WaveSplat: 교수님 발표 요약

## 1. 연구 개요

**제목**: WaveSplat — Grid-Free Spectral Gaussian Wave Field Estimation from Sparse LiDAR

**핵심 아이디어**: 선박 탑재 LiDAR의 sparse point cloud로부터 실시간 파도장(wave field)을 추정하되,
기존 grid 기반 방식(ETKF) 대신 **spectral Gaussian** 표현을 사용하여 grid-free로 추정.

**모델**: `z(x,y,t) = Σᵢ wᵢ · cos(kxᵢ·x + kyᵢ·y − ωᵢ·t + φᵢ)`
- 각 component = 하나의 plane wave (진폭, 파수, 위상)
- 분산관계 제약: `ω = √(g|k|tanh(|k|d))` — 물리적 일관성 보장
- Adaptive N: 에너지 기반 pruning + residual-triggered growth

---

## 2. 방법론 요약

### Phase 1: Initialization (2Tp window)
1. Beamforming으로 dominant wavenumber (kx, ky) 추정
2. Coarse-to-fine: N=[3, 8, 15] 단계적 증가
3. Projected gradient descent — 분산관계 constraint
4. Fresnel confidence weighting: 거리·입사각 기반 관측 가중

### Phase 2: Tracking (10Hz, 실시간)
1. Phase prediction: `φᵢ += ωᵢ·Δt`
2. w_phi mode: 진폭·위상만 업데이트, 파수는 freeze
3. Sliding window (k=3) multi-frame tracking

### Adaptive N
- Pruning: 에너지 상위 90% component만 유지
- Growth: 잔차 > 0.05m이면 beamforming으로 새 component 추가
- Safety rollback: pruning 후 잔차 30% 이상 증가 시 원복
- N_max = 30

---

## 3. 실험 결과

### 3.1 Main Comparison (12 episodes, seed0)

| 지표 | WaveSplat | ETKF | 개선율 |
|------|-----------|------|--------|
| **avg nRMSE** | **0.211** | 0.436 | **51.5%** |
| **추론 속도** | **13.4 ms/frame** | 31.7 ms | **2.4× 빠름** |
| 승률 | **12/12** | 0/12 | — |

### 3.2 Multi-Seed 통계 검증 (10 episodes × 5 seeds = 50 paired runs)

| 지표 | WaveSplat | ETKF |
|------|-----------|------|
| Mean±Std | **0.230 ± 0.084** | 0.395 ± 0.059 |
| Wilcoxon p-value | **p < 0.001** | — |
| 승률 | **50/50** | 0/50 |

### 3.3 Sea State별 개선율

| Beaufort | WS nRMSE | ETKF nRMSE | 개선율 |
|----------|----------|------------|--------|
| BF3 | 0.130 | 0.393 | **66.9%** |
| BF4 | 0.247 | 0.456 | **45.8%** |
| BF5 | 0.293 | 0.481 | **39.1%** |

### 3.4 Sparsity Robustness
- 관측점 20%로 줄여도 BF4-5에서 **8% 미만 성능 저하**
- Spectral 표현의 강점: 적은 관측으로도 주파수 정보 보존

### 3.5 Raydrop Model Comparison (10 paired episodes)
- Uniform vs Slope raydrop → WaveSplat **20/20 승리**
- Slope이 양쪽 다 개선하나, ETKF가 더 많이 개선됨 (16.2% vs 9.1%)

### 3.6 Ablation Study (3-seed)
- **Dispersion 제거 (A1)**: BF3에서 오히려 개선 → regularizer 역할
- **Confidence 제거 (A2)**: 14.5% 성능 저하 → slope raydrop에서 필수

---

## 4. 계산 복잡도 비교

| 방법 | Per-frame | 실측 | 비고 |
|------|-----------|------|------|
| **WaveSplat** | O(M·N) | **13.4 ms** | M=50, N≤30 |
| ETKF | O(Nₑ·G²·log G²) | 31.7 ms | Nₑ=50, G=40×30 |
| GPR | O(M³) | >200 ms | Kernel optimization |

---

## 5. 기여 (Contributions)

1. **Grid-free spectral 표현**: 파수 공간에서 직접 최적화, grid aliasing 없음
2. **Physics-constrained optimization**: 분산관계 hard constraint
3. **Adaptive complexity**: scene에 맞는 N 자동 선택
4. **LiDAR-specific confidence weighting**: Fresnel 반사 물리 반영
5. **포괄적 실험**: 100 episodes, 2 raydrop 모델, 5 seeds 통계 검증

---

## 6. 타겟 학회/저널

### 1순위: IEEE/MTS OCEANS Conference
- **적합성**: 높음 — 해양 계측, LiDAR, 실시간 시스템
- **형식**: Conference paper (6-8 pages)
- **일정**: OCEANS 2026 (Singapore, Sep 또는 Washington DC, Oct)
- **장점**: 해양공학 + 신호처리 커뮤니티, 실용적 시스템 발표에 적합

### 2순위: ISOPE (International Offshore and Polar Engineering)
- **적합성**: 높음 — 파랑 계측/예측 세션 있음
- **형식**: Conference paper
- **일정**: ISOPE 2026 (보통 6월)

### 3순위: Ocean Engineering (Elsevier) — 저널
- **적합성**: 매우 높음 — wave estimation 핵심 저널
- **Impact Factor**: ~4.6
- **형식**: Full paper (확장 필요: real data, 추가 baseline)
- **장점**: 시뮬레이션만으로도 게재 가능 (cf. Desmars 2020, Huhn 2022)

### 4순위: Applied Ocean Research (Elsevier) — 저널
- **적합성**: 높음
- **Impact Factor**: ~4.3
- **형식**: Full paper

### 추천 전략
1. **단기**: OCEANS 2026 conference paper → 빠른 발표 실적
2. **중기**: Ocean Engineering 저널 → real LiDAR 데이터 추가 후 확장 논문

---

## 7. 현재 코드 상태

### 완성된 모듈
- `wavesplat/` — 핵심 추정 알고리즘 (model, optimizer, evaluation)
- `wavesim/` — 데이터 생성 시뮬레이터 (wave + LiDAR)
- `baselines/` — ETKF, GPR baseline
- `scripts/` — 실험 스크립트 10+개
- `paper/` — IEEE format LaTeX 초안 + figures + references.bib
- `configs/` — YAML 설정 (estimation, data generation)

### 데이터셋
- 100 에피소드 (BF2-5 × 1/2/5/15/30comp × seed0-4 × uniform/slope)
- HDF5 형식, GT + LiDAR observations + metadata

### 재현성
- `requirements.txt` 제공
- `configs/estimation.yaml` — 모든 하이퍼파라미터 문서화
- Unit tests: `wavesplat/tests/test_model.py`, `test_integration.py`

---

## 8. 다음 단계 (향후 계획)

### 논문 1 (WaveSplat) 완성
- [x] LaTeX 초안 완성
- [x] 모든 실험 테이블/피규어 생성
- [x] Multi-seed 통계 검증
- [x] Sparsity/BF trend/Convergence 분석
- [ ] GPR 3-way 비교 실행 (스크립트 준비 완료)
- [ ] Qualitative figure 생성 (스크립트 준비 완료)
- [ ] (Optional) Real LiDAR 데이터 테스트

### 논문 2 (Hydrofoil MPC) — 별도 진행
- WaveSplat 출력 → wave encounter prediction → hydrofoil MPC
- 현재 hydrofoil 모델 작성 중
