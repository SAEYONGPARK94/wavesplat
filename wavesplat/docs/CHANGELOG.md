# WaveSplat Changelog

프로젝트 개발 이력을 시간순으로 기록합니다.

---

## [0.2.0] — 2026-04-05 (미커밋)

### Added
- `wavesim/visualize.py` — `save_video()` 함수 추가
  - 4패널 MP4 동영상 출력 (matplotlib + cv2 기반)
  - 패널: 2D 파고맵 + LiDAR pts, GT vs 관측 산점도, 시간별 밀도, 정보 텍스트
  - `--save_video` CLI 옵션으로 실행 가능

### Generated
- `wavesim.mp4` — 기본 시뮬레이션 동영상
- `wavesim_BF4_5s.mp4` — BF4 해상 5초 시뮬레이션 동영상
- `data/episode_BF4_20comp_seed7_uniform.h5` — 추가 에피소드 (20성분, seed=7)

---

## [0.1.1] — 2026-03-30

### Added
- `README.md` — 프로젝트 개요, 물리 문서, 설치/실행 가이드, 로드맵

**Commit**: `cf6e5d0` — Add README with project overview and physics documentation

---

## [0.1.0] — 2026-03-30

### Added — WaveSim 시뮬레이션 패키지 (Phase 1-B 완료)

#### 핵심 모듈
- `wavesim/wave_generator.py` — Gerstner wave 중첩 + JONSWAP 스펙트럼 샘플링
  - 완전 분산 관계: `ω = √(g·k·tanh(k·d))`
  - Q-parameter 제약 (`Σ Qᵢ·ωᵢ·Aᵢ ≤ 1`) 자동 검증/스케일링
  - Newton-Raphson 역분산 roundtrip 정확도 < 1e-6 rad/s
- `wavesim/lidar_simulator.py` — 전방 조사 LiDAR 시뮬레이션
  - Ray 생성 (32ch × 360 수평빔), Möller-Trumbore ray-mesh intersection
  - 입사각/grazing angle 계산, 쌍선형 법선 보간
- `wavesim/raydrop_uniform.py` — Model A: Fresnel/거리 기반 균일 dropout
  - `P_return = η · R_fresnel(θ_i) · cos(θ_i) / (d/d_ref)²`
- `wavesim/raydrop_slope.py` — Model B: Specular backscatter 경사 기반 dropout
  - α_threshold = 5° 반사 정렬 판정, diffuse fallback 포함
- `wavesim/gt_generator.py` — Ground truth 파 필드 (정규 그리드 80×60)
- `wavesim/noise_model.py` — 센서 노이즈 (range σ=2cm, angle σ=0.05°) + 신뢰도 계산
- `wavesim/data_writer.py` — HDF5 저장/로드 (gzip 압축)
- `wavesim/pipeline.py` — 엔드투엔드 파이프라인 (CLI 지원)
  - 해상 상태: BF2(0.3m) / BF3(0.6m) / BF4(1.2m) / BF5(2.0m)

#### 시각화
- `wavesim/visualize.py` — 4패널 인터랙티브 시각화
  - 3D 파 표면 + LiDAR, 2D 파고맵, 시간별 밀도, GT vs 관측 산점도
  - HTML (인터랙티브) / PNG / 애니메이션 HTML 출력

#### 테스트
- `wavesim/tests/` — 39개 단위/통합 테스트 전체 통과
  - `test_wave_generator.py` (10), `test_raycast.py` (8), `test_raydrop.py` (9)
  - `test_gt.py` (8), `test_pipeline.py` (4)

#### 문서
- `docs/WaveSplat_개발가이드.md` — 연구 설계, 방법론 결정 과정, 실험 계획
- `docs/WaveSim_코드문서.md` — 모듈별 API, 물리 검증 Trust Chain, 테스트 현황

#### 생성 데이터
- `data/episode_BF4_30comp_seed42_uniform.h5` — BF4 해상, 30성분, seed=42
- `wavesim_vis.html`, `wavesim_vis.png` — 정적 시각화 (uniform 모델)
- `wavesim_vis_slope.html`, `wavesim_vis_slope.png` — 정적 시각화 (slope 모델)

**Commit**: `3070038` — Initial commit: WaveSplat LiDAR wave simulation project

---

## Roadmap

- [x] **Phase 1-B**: WaveSim 물리 기반 시뮬레이션 — 완료
- [ ] **Phase 2**: 데이터셋 생성 — 15개 시나리오 × 5 seed = 75 에피소드
- [ ] **Phase 3**: WaveSplat 추정 알고리즘 (Spectral Gaussian Estimation)
- [ ] **Phase 4**: 실험 + 논문 제출