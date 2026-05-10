"""
fill_labmeeting_template.py

Build full 25-content-slide presentation matching labmeeting_2026_05_13.pptx
structure, using user template's layout/master/fonts.

Strategy:
- Open user template (has master + layouts + 폰트 + 4:3)
- Preserve: slide 1 (title), slide 14 (감사합니다), slides 15-19 (appendix)
- Preserve: slide 4 user-filled content for §1 첫 페이지
- Replace all other slides with new structured content matching
  labmeeting_2026_05_13 (25 main slides) using user's tone style
- Reorder XML so structure is correct

User tone (slide 4 분석):
- Plain text, level 0/1/2 indent
- 짧은 명사구 (full sentence 아님)
- 한국어 + 영어 기술 용어 inline
- "...이지만/하나..." 비교 패턴
- Markdown bold 없음, emoji 없음
"""
import os
import shutil
import copy
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'docs', 'labmeeting_20260513_박세용.pptx')
DST = os.path.join(ROOT, 'docs', 'labmeeting_20260513_박세용_filled.pptx')
DOCS_DIR = os.path.join(ROOT, 'docs')
PAPER_FIG_DIR = os.path.join(ROOT, 'paper', 'figures')


# ───────────────────────────────────────────────────────────
# Slide content specs — list of (level, text) tuples per slide
# ───────────────────────────────────────────────────────────

# §1 Introduction & Motivation (4 content slides)
# slide A: 파 추정의 한계 (USER's existing — preserve verbatim)
S1_A_USER = [
    (0, '파 추정의 한계'),
    (1, '통계적 추정으로는 유의파고, 주기 정도만 제공'),
    (2, '평균적 파 환경은 알 수 있지만, 언제 어디서 파가 오는지 미지수'),
    (2, '선박 설계에는 충분하나 실시간 제어 목적으로는 부족'),
    (1, '따라서 본 연구에서는 하기와 같은 연구를 수행'),
    (2, 'Phase resolved estimation: η(x, y, t) – 각 파도의 위치·시점을 추정'),
    (2, 'BF4 sea state가 아닌 특정 위치 특정 시간 후 어떤 파와 같은 구체적 정보 추정'),
    (1, '연구 결과를 이용한 응용'),
    (2, 'Hydrofoil ride control, 자율운항 안전판단, 선박 motion 예측 등'),
]

# slide B: 본 task의 어려움 (3 challenges) — 자세하고 쉬운 설명
S1_B_CHALLENGES = [
    (0, '본 task 의 핵심 도전 과제 — 왜 어려운가'),
    (1, '도전 1 — 희소하고 비균일한 측정 (sparse, non-uniform)'),
    (2, '한 frame 의 LiDAR 유효 점이 약 50개 정도 (raydrop, shadowing 후)'),
    (2, '점밀도 비균일 — 가까운 거리는 dense, 먼 거리는 sparse'),
    (2, 'Wave gauge 나 dense radar 와 다른 setup → 기존 방법 직접 적용 어려움'),
    (1, '도전 2 — 실시간 제약 (frame 당 100 ms 이내)'),
    (2, '선박 제어 루프 약 10 Hz → 한 frame 처리 100 ms 이내 필요'),
    (2, '선행 방법별 처리 시간 비교:'),
    (2, '   • EnKF + HOS 비선형 (Wang & Pan): ensemble × HOS forecast 로 수백 ms 이상'),
    (2, '   • EnKF + linear (Feng Wang 2025, Ne=300): 632 ms — 100 ms 초과로 부적합'),
    (2, '   • PFT-PINN (Ehlers 2025): instance 마다 약 30 분 학습 (RTX 3090)'),
    (2, '   • FNO supervised (Ehlers 2023): 학습 후 inference 1.9 ms — 단 3120 sample 학습 필요'),
    (1, '도전 3 — 2D 방향성 sea state'),
    (2, '실해 파도는 한 방향 아닌 여러 방향 혼재 (wind sea + 다방향 swell)'),
    (2, 'Wavenumber 도 단순 scalar 아닌 vector (k_x, k_y) 추정 필요'),
    (2, '선행 연구 다수는 1D 가정으로 단순화 (Klein 2020, Desmars 2023, FNO …)'),
    (2, '본 연구는 2D directional 그대로 처리 — 차별점 중 하나'),
]

# slide C: spoiler — 결과 미리 (먼저 결과부터 보여주는 hook)
S1_C_SPOILER = [
    (0, '본 발표의 결과 미리보기 — Spoiler'),
    (1, '왜 결과부터 보여주는가'),
    (2, '배경 설명 전에 "왜 이 연구를 계속할 가치가 있는가" 를 숫자로 먼저 제시'),
    (1, '핵심 정량 비교 (12-episode benchmark, BF2~BF5 sweep)'),
    (2, 'WaveSplat (본 연구): 평균 nRMSE 0.211, 처리 13.4 ms / frame'),
    (2, 'Wang2025 baseline (Feng Wang et al. 2025): 0.253, 632 ms / frame'),
    (2, '→ 정확도 16% 우위 + 속도 47배 빠름 (CPU only, GPU 불필요)'),
    (2, 'WaveSplat 13.4 ms = 초당 약 75 frame → 10 Hz 제어 루프 충분히 적합'),
    (1, '핵심 메커니즘 발견'),
    (2, '분산 관계 (ω↔k) = 자유도 감소 + 물리 정규화 — 5N 변수 → 4N (20% 감소)'),
    (2, 'Multi-frame temporal coherence — wave 자체의 시간 진화로 phase 정보 추출'),
    (2, '두 메커니즘 결합 시 spatial Nyquist 0.4 rad/m 한계 우회 가능'),
    (1, '응용 사례 — Hydrofoil ride control 시험 적용'),
    (2, 'Heave aRMS: PD-only 0.090 → PD+FF (WaveSplat preview) 0.047 m/s²'),
    (2, '약 48% 감소 — preliminary test 단계, 응용 가능성 확인'),
]

# §2 Related Work (4 content slides)
# slide A: 5 카테고리
S2_A_CATEGORIES = [
    (0, '선행 연구 5가지 접근법'),
    (1, '① 격자 기반 최적화'),
    (2, 'Desmars 2018, 2020, 2023'),
    (2, 'Pre-defined wavenumber grid 위 variational 최적화'),
    (2, '한계: on-grid (격자 사이 wavenumber 표현 불가), 대부분 1D'),
    (1, '② 앙상블 칼만 필터'),
    (2, 'Wang & Pan 2021/22 — EnKF + HOS 비선형 wave'),
    (2, 'Feng Wang 2025 — ETKF + fuzzy R + Delaunay'),
    (2, 'Yan 2025 — VOF-EnKF + Navier-Stokes (wave breaking 처리)'),
    (2, '한계: 50–300 ensemble × 수백 ms+ → real-time 불가, 대부분 1D'),
    (1, '③ 지도학습 신경망'),
    (2, 'Ehlers 2023 — Fourier Neural Operator (FNO)'),
    (2, 'Inference 1.9 ms (학습 후), HOS 합성 데이터 3120 sample 학습 필요'),
    (2, '한계: distribution shift, 1D, black-box'),
    (1, '④ 물리 기반 비지도 NN'),
    (2, 'Ehlers 2025 — PFT-PINN'),
    (2, '학습 데이터 0개로 비선형 처리, hard-constraint trick'),
    (2, '한계: instance 마다 ~30 분 학습, 1D + t, buoy only'),
    (1, '⑤ 연속 파라미터 추정 (Ours) ⭐'),
    (2, 'WaveSplat — 격자 없이 N개 plane wave 파라미터 직접 추정'),
    (2, '< 100 ms inference + 2D directional + LiDAR raydrop sensor model'),
    (2, '한계: linear superposition (LWT), ε < 0.07 영역에서만 신뢰'),
    (1, '※ 자세한 paper 정보는 Appendix A 참조'),
]

# slide B: EnKF prior art detail
S2_B_ENKF = [
    (0, 'EnKF 기반 wave estimation — 두 독립 group'),
    (1, 'Wang & Pan group (Yulin Pan, U.Michigan)'),
    (2, '2021: EnKF + HOS 첫 결합'),
    (2, '2022: IEnKF-HOS-C — current 동시 추정'),
    (2, 'Sensor: X-band radar, Data: synthetic + 실측 shipborne radar'),
    (1, 'Feng Wang et al. (Harbin Engineering Univ., China)'),
    (2, '2025: ETKF + fuzzy R + Delaunay 보간'),
    (2, 'Ne=300, 80×60 grid, dx≈3 m'),
    (2, 'Sensor: Stereo camera (WASS), Data: 실측 (Acqua Alta 해양 플랫폼 + 자체 2024)'),
    (2, '본 baseline 의 직접 prior art (GitHub 공개)'),
    (1, '본 연구의 baseline 재현 (공정 비교를 위한 통일)'),
    (2, '알고리즘만 재현 — Ne=300, 80×60 격자, fuzzy R, Delaunay 보간 동일'),
    (2, '입력 sensor 는 LiDAR sim 으로 통일 (원논문은 stereo camera)'),
    (2, 'Forecast model 도 HOS 대신 linear dispersion 으로 통일'),
    (1, '공통 한계'),
    (2, '격자 해상도 한계 — 격자 밖 wavenumber 표현 불가 (~0.25 nRMSE floor)'),
    (2, 'Ensemble cost 632 ms/frame — real-time control loop 불가'),
]

# slide C: PFT-PINN detail (Ehlers 2025) + FNO (Ehlers 2023)
S2_C_PINN = [
    (0, '학습 기반 접근 — Ehlers group (TUHH + Imperial)'),
    (1, 'Ehlers 2023 — FNO supervised'),
    (2, 'Architecture: Fourier Neural Operator (Fourier 공간에서 global mapping)'),
    (2, 'Training data: 3120 sample (HOS sim + geometric radar model)'),
    (2, 'Inference 1.9 ms, U-Net 보다 wave 패턴에 적합'),
    (2, '한계: distribution shift, 1D unidirectional, black-box'),
    (1, 'Ehlers 2025 — PFT-PINN'),
    (2, 'Architecture: 두 NN 병렬 — η-net (wave 표면) + Φ-net (velocity potential)'),
    (2, 'PDE residual 강제 (Laplace + 표면 BC + 바닥 BC) via auto-diff'),
    (2, 'Hard-constraint trick: 관측치 NN architecture에 직접 hard-encode'),
    (2, '검증: Wave flume 실측 (MARATHON, T_p=1.2 s), SSP 0.049 (assim) / 0.084 (predict)'),
    (2, '학습 데이터 0개로 비선형 wave 처리'),
    (2, '한계: ~30 분/instance (RTX 3090), 1D + t, wave gauge buoy only'),
    (1, '본 연구와의 핵심 차이'),
    (2, '속도 차이가 결정적: WaveSplat 13 ms vs PFT-PINN ~30 분 (수만 배 이상 차이)'),
    (2, '응용 영역 분리: WaveSplat = real-time control, PINN = design-time analysis'),
    (2, '향후 위협: 같은 group 이 1-3년 안에 2D + real-time PINN 발표 가능 — niche 좁아질 risk'),
]

# slide D: 응용 매트릭스 + niche
S2_D_NICHE = [
    (0, '응용 매트릭스 — 본 연구의 위치'),
    (1, '비교 항목 (가로) — 6가지 응용 요구사항'),
    (2, '실시간 제어 가능 (<100 ms / frame)'),
    (2, '2D 방향성 추정 가능'),
    (2, '파도 파쇄 (wave breaking) 처리'),
    (2, '유속장 (velocity field) 전체 추정'),
    (2, '학습 데이터 불필요 (unsupervised)'),
    (2, '희소 LiDAR 센서 모델 지원'),
    (1, '본 연구만의 고유 영역 — 4가지 동시 만족'),
    (2, 'Real-time + 2D directional + sparse LiDAR + 학습 데이터 0개'),
    (2, '이 4가지 모두 만족하는 선행 연구는 없음'),
    (1, '응용 영역 분리 — 본 연구는 real-time control'),
    (2, 'Real-time control 영역: WaveSplat, FNO (학습 후 추론)'),
    (2, 'Design-time analysis 영역: VOF-EnKF, PFT-PINN (정밀 분석)'),
    (2, '두 영역은 경쟁이 아닌 보완 — 응용 목적 자체가 다름'),
    (1, '정직한 한계'),
    (2, 'Wave breaking 미처리 — VOF-EnKF 만 가능한 영역'),
    (2, 'Velocity field 미추정 — PINN/EnKF 영역'),
]

# §3 Method (8 content slides — WaveSim 2 + Dataset 1 + WaveSplat 5)
# slide A: WaveSim physics
S3_A_WAVESIM = [
    (0, 'WaveSim — 합성 파도장 + LiDAR 시뮬레이터'),
    (1, '파도 모델 — Gerstner family'),
    (2, '회전 정확해 — 입자 운동까지 모사 (Gerstner 1802, Tessendorf 2001)'),
    (2, 'Q ∈ [0, 1] 가파름 — Q=0 이면 LWT (선형), Q>0 이면 약비선형'),
    (2, 'x = x₀ − Q·A·(k_x/|k|)·sin(Φ), z = A·cos(Φ), Φ = k·x − ω·t + φ'),
    (2, 'Q 제약: Σ Q_n·ω_n·A_n ≤ 1 (입자 궤적 자체 교차 방지)'),
    (1, '스펙트럼 — JONSWAP (Hasselmann 1973)'),
    (2, 'S(ω) = α·g²·ω⁻⁵·exp(−5/4·(ω_p/ω)⁴) · γ^r (γ=3.3)'),
    (2, '입력 H_s, T_p → N개 성분의 (A_n, ω_n) sampling'),
    (2, '방향 spreading: cos²ˢ 함수로 (k_xn, k_yn) 분배'),
    (2, 'ITTC, IACS 채택 표준 wave spectrum'),
    (1, '분산 관계 — Airy 1845'),
    (2, 'ω = √(g·|k|·tanh(|k|·d)) — 100 년 이상 검증된 표준'),
    (2, '깊은 물 한계 (kd >> 1): ω ≈ √(g·|k|)'),
]

# slide B: WaveSim LiDAR + raydrop
S3_B_LIDAR = [
    (0, 'LiDAR 모사 — Falcon K1'),
    (1, 'Sensor 사양'),
    (2, '152 채널 × 500 horizontal beam = 76,000 ray / scan'),
    (2, 'Range 10–250 m, 파장 1550 nm, vessel 위 5 m mount'),
    (2, 'Horizontal FOV 120° (선수 ±60°), Vertical FOV −25° ~ 0°'),
    (1, 'Ray tracing — Möller-Trumbore 1997'),
    (2, '각 ray 가 wave 표면 mesh 와 만나는 first-hit 점 검출'),
    (2, 'Graphics 분야 표준 ray-triangle intersection 알고리즘'),
    (1, 'Raydrop 모델 — Fresnel + geometry'),
    (2, '반사된 빛이 LiDAR receiver 방향으로 돌아와야 함'),
    (2, 'Snell 굴절: sin θ_t = sin θ_i / n (n=1.321, water at 1550 nm)'),
    (2, 'Fresnel s/p polarization 평균 → 반사율 R(θ_i)'),
    (2, 'Lambert + Fresnel 결합: P = η · R · cos(θ_i) / (d/d_ref)²'),
    (1, 'Vessel 정지 가정 (V=0 kts)'),
    (2, '모든 frame 동일 위치에서 측정 — 합성 dataset, 코드 모두 정지 가정'),
    (2, 'Multi-frame 효과는 wave 자체의 시간 진화에서 옴'),
    (2, 'Vessel motion 기반 synthetic aperture (SAR-style) 는 향후 연구'),
]

# slide C: Dataset
S3_C_DATASET = [
    (0, 'Dataset — 110 episode 합성'),
    (1, '구성'),
    (2, 'Sea state: BF2 (Hs=0.3, Tp=4.0) ~ BF7 (Hs=5.0, Tp=12.0)'),
    (2, 'N components: 1, 2, 5, 15, 30 (단순 ~ 복잡)'),
    (2, 'Seeds: 0–4 (재현 가능)'),
    (2, 'Raydrop: uniform / slope (각 episode)'),
    (1, '에피소드별 정량'),
    (2, 'Duration 30 s, dt = 0.1 s (10 Hz) → frame 301개'),
    (2, 'GT η field: 80×60 격자, 240 m × 200 m 영역'),
    (2, 'LiDAR scan: 76,000 ray → raydrop 후 valid 점 약 50개/frame'),
    (2, 'Patch coverage: 약 50–150 m (실측 확인)'),
    (1, '용량'),
    (2, '110 episodes × 약 12 MB = 1.2 GB 합성 dataset'),
    (2, 'HDF5 format (timestamps, lidar_points, gt_wave_field, lidar_confidence …)'),
    (1, 'BF3-BF6 wavenumber 범위 (실측)'),
    (2, 'BF3 (Tp=5.5s): k = 0.04–1.16 rad/m, λ_peak ≈ 47 m'),
    (2, 'BF4 (Tp=7.0s): k = 0.03–0.72 rad/m, λ_peak ≈ 77 m'),
    (2, 'BF5 (Tp=9.0s): k = 0.02–0.43 rad/m, λ_peak ≈ 126 m'),
    (2, 'BF6 (Tp=10s): k = 0.02–0.35 rad/m, λ_peak ≈ 156 m (전부 Nyquist 이내)'),
]

# slide D: WaveSplat 알고리즘 개요
S3_D_OVERVIEW = [
    (0, 'WaveSplat — 알고리즘 개요'),
    (1, '발상 — 격자 대신 파라미터로 직접 표현'),
    (2, '실제 파도 = 5–30개 sine wave 의 합 (JONSWAP 분해)'),
    (2, '각 성분: (A, k_x, k_y, ω, φ) 5개 파라미터로 완전 결정'),
    (2, 'N=10 이면 변수 50개 — LiDAR 50점으로 1:1 추정 가능'),
    (1, 'Wave field 표현'),
    (2, 'η(x, y, t; θ) = Σ_n A_n · cos(k_xn·x + k_yn·y − ω_n·t + φ_n)'),
    (2, '파라미터 θ = {A_n, k_xn, k_yn, φ_n} (ω_n 은 dispersion 으로 자동)'),
    (1, '분산 관계 hard constraint'),
    (2, 'ω(k) = √(g·|k|·tanh(|k|·d)) 강제 → 자유도 5N → 4N (20% 감소)'),
    (2, '물리적으로 불가능한 (k, ω) 조합 자동 배제'),
    (1, '2단계 흐름'),
    (2, 'Init (1회, 1초 이내): beamforming + coarse-to-fine + Adam refinement'),
    (2, 'Track (매 frame, 13 ms): k 고정, (A, φ) 만 update'),
    (2, '무거운 작업은 처음 한 번만, 가벼운 작업은 매 frame 실시간'),
    (1, 'Adaptive N — 성분 수 동적 조정'),
    (2, 'Pruning + Growth + Safety rollback'),
]

# slide E: Init (1/3) — Nyquist + multi-frame motivation
S3_E_INIT1 = [
    (0, 'Init (1/3) — 왜 multi-frame 이 필요한가'),
    (1, 'Spatial Nyquist 한계 (Shannon-Nyquist 1949)'),
    (2, "'신호 추정에는 wavelength 의 절반 간격 sampling 필요'"),
    (2, '점 간격 Δy ≈ 5–8 m → k_max = π/Δy ≈ 0.4–0.6 rad/m'),
    (2, 'Single-frame LiDAR 로 추정 가능한 wavenumber 한계: 약 0.4 rad/m'),
    (1, '필요한 wavenumber 범위'),
    (2, 'BF3 의 짧은 wave 성분은 Nyquist 한계의 약 3배까지 도달'),
    (2, 'BF5-BF6 dominant wave 는 대부분 Nyquist 이내 (longer wave)'),
    (2, '결과: BF3 short component 만 single-frame Nyquist 초과 — Paradox'),
    (1, '해결 — temporal coherence'),
    (2, 'Vessel 정지에도 wave 자체의 시간 진화 활용 가능'),
    (2, 'η(x, t) = A·cos(k·x − ω·t + φ) 의 t 진행이 phase 변화 제공'),
    (2, '여러 frame 의 (x_i, y_i, t_i, η_i) 4D 데이터 → wavenumber 모호성 해소'),
    (1, '본 case 정량'),
    (2, 'Init window 길이: 2·T_p (sea state 별 약 10–20 초)'),
    (2, '여러 frame × 점 수십개 → 수천 개 (x, y, t, η) 4D 점 누적'),
    (2, '시간 정보 + dispersion 의 ω↔k 매핑 → spatial Nyquist 우회'),
    (1, 'Vessel motion 기반 SA (synthetic aperture) 는 현재 미적용 — 향후 연구'),
]

# slide F: Init (2/3) — Beamforming
S3_F_INIT2 = [
    (0, 'Init (2/3) — Beamforming + Coarse-to-fine + Adam'),
    (1, 'Beamforming score (matched filter)'),
    (2, 'S(k) = | Σ_i η_obs(x_i, t_i) · exp(−j(k·x_i − ω(k)·t_i)) |²'),
    (2, 'i: init window 의 모든 점 (frame × 점 수 = 수천 개)'),
    (2, 'ω(k) 는 dispersion 으로 자동 — 2D search (k_x, k_y) 만'),
    (2, 'Search range: k_x, k_y ∈ [-4, 4], 81×81 grid (수천 개 후보)'),
    (1, 'Coarse-to-fine schedule'),
    (2, '3 stage: N = 3 → 8 → 15 (점진적 component 증가)'),
    (2, 'Loss threshold 충족 시 다음 stage 로 advance'),
    (2, '빠른 수렴 → 다음 단계 즉시 진행 (전체 시간 단축)'),
    (1, 'Adam refinement (multi-frame fitting)'),
    (2, '파라미터 (k_x, k_y, A, φ) 동시 update'),
    (2, '(k_x, k_y) 는 lr 의 절반으로 천천히 update (k 변화 신중)'),
    (2, 'Loss = data fit (averaged + confidence-weighted) + L1 sparsity + spectral smoothness'),
    (1, '주요 hyperparameter'),
    (2, 'L1 sparsity 가중치 λ_L1 = 0.001'),
    (2, 'Spectral smoothness 가중치 λ_smooth = 0.0001'),
    (2, 'Steepness clamp ε_max = 0.15 (Q 제약 보호)'),
]

# slide G: Init (3/3) — Score 식 유도
S3_G_INIT3 = [
    (0, 'Init (3/3) — Beamforming Score 식 유도'),
    (1, '핵심 직관 — matched filter 의 코러스 효과'),
    (2, '실제 wave (k, ω) 면 measurement η_obs · exp(−j(k·x − ω·t)) 가 phase 정렬'),
    (2, '맞는 (k, ω): 모든 점에서 phase 일치 → 합 절댓값 N 배 증가'),
    (2, '틀린 (k, ω): random phase → 합 √N 으로만 증가'),
    (2, '→ SNR gain = N²/N = N (점 수에 비례)'),
    (1, 'Score 식의 기여'),
    (2, '맞는 (k, ω) 일 때: S = (NA/2)² ∝ N² (coherent stack)'),
    (2, '틀린 (k, ω) 일 때: S ∝ N (random phase, incoherent)'),
    (2, 'Init window 의 수천 4D 점으로 충분한 SNR gain 확보 (~10·log10(N) dB)'),
    (1, '왜 이 형태가 표준인가'),
    (2, 'Delay-and-sum beamforming (Kraus 1988, Van Trees 2002)'),
    (2, 'Matched filter 정리 (Turin 1960) — Gaussian noise 에서 SNR 최대화'),
    (2, '단순·robust — covariance matrix 추정 불필요 (sparse 적합)'),
    (1, 'Dispersion 으로 차원 축소'),
    (2, 'Dispersion 없이 3D search (k_x, k_y, ω) → 후보 수십만 개 (계산 비현실적)'),
    (2, 'Dispersion 강제로 2D search (k_x, k_y) → 후보 수천 개로 축소'),
    (1, '다른 spectral 방법 비교'),
    (2, 'MUSIC, ESPRIT (subspace): 더 정확하나 covariance 필요 → sparse 부적합'),
    (2, 'Capon/MVDR (adaptive): 더 좋은 resolution 이나 ill-conditioning 위험'),
]

# slide H: Track + Adaptive N
S3_H_TRACK = [
    (0, 'Track + Adaptive N — 매 frame 13 ms'),
    (1, 'w_phi tracking mode (configs default)'),
    (2, '(k_x, k_y): Init 결과 고정 — 변경 안 함'),
    (2, 'ω: dispersion 으로 자동 (k 에서 계산)'),
    (2, '(A, φ): Adam 으로 매 frame update — 자유도 2N'),
    (1, 'Adam optimizer update rule (Kingma & Ba 2015, arXiv:1412.6980)'),
    (2, 'm_t = β₁·m_{t-1} + (1−β₁)·∇L (β₁=0.9)'),
    (2, 'v_t = β₂·v_{t-1} + (1−β₂)·(∇L)² (β₂=0.999)'),
    (2, 'θ ← θ − lr · m̂_t / (√v̂_t + ε)'),
    (2, 'tracking_lr = 0.005, tracking_steps = 10/frame → 약 13 ms'),
    (1, 'Phase prediction (kinematic preview)'),
    (2, '매 update 전: φ_i ← φ_i + ω_i · Δt'),
    (2, 'Tracking window = 3 frames (recent buffer)'),
    (1, 'Adaptive N — component 수 동적 조정'),
    (2, 'Pruning: cumulative energy 90% 유지 (prune_threshold=0.10)'),
    (2, 'Growth: residual RMS > 0.05 m 시 새 (k_x,k_y) 추가 (max_n=30)'),
    (2, 'Safety rollback: 변경 전후 RMS_after > 1.3 × RMS_before → 즉시 복구'),
    (1, '왜 k 고정?'),
    (2, 'Single frame 에서 k 의 FIM eigenvalue 가 (A,φ) 보다 10⁻³ 작음'),
    (2, 'k 자유 시 50점에 over-fit → 비물리적 발산'),
]

# §4 Experiments & Mechanism (5 content slides)
# slide A: 12-episode 비교 결과
S4_A_RESULTS = [
    (0, '12-episode benchmark — WaveSplat vs Wang2025'),
    (1, '평균 결과'),
    (2, 'WaveSplat: nRMSE 0.211, 13.4 ms / frame'),
    (2, 'Wang2025 (Feng Wang et al., Ne=300): 0.253, 632 ms / frame'),
    (2, '정확도 16% 우위 + 47배 빠름 (CPU only)'),
    (1, 'Real-time 의미'),
    (2, 'WaveSplat 13.4 ms → 초당 75 frame 추정 가능 (10 Hz 제어 적합)'),
    (2, 'Wang2025 632 ms → 초당 1.6 frame (실시간 제어 부적합)'),
    (2, '참고: Ehlers 2023 FNO 1.9 ms (단 supervised 학습 필요)'),
    (1, '비교 baseline 의 충실 재현'),
    (2, 'Ne=300, 80×60 격자, dx≈3 m, fuzzy R, Delaunay 보간 — 원논문 그대로'),
    (2, '단 forecast model 은 HOS 대신 linear dispersion 으로 통일 (공정 비교)'),
    (2, 'Sensor modality 도 LiDAR 로 통일 (원논문은 stereo camera)'),
    (1, '비교 episode 구성'),
    (2, 'BF2_1comp ~ BF5_30comp, seed=0 uniform/slope 혼합'),
    (2, '전체 12 case (BF2~BF5 sweep × N components)'),
]

# slide B: Episode-level scatter
S4_B_SCATTER = [
    (0, 'Episode-level 패턴 — 강점과 한계'),
    (1, '단순 파도 (1–5 components) — WaveSplat 압도'),
    (2, 'BF3_2comp: 0.052 vs Wang2025 0.250 — 거의 완벽 추정'),
    (2, '소성분이면 파라미터 적어 50개 관측으로 직접 추정 가능'),
    (2, '격자 방법은 성분 수와 무관하게 격자 해상도 한계'),
    (1, '복잡 파도 (15–30 components) — Wang2025 일부 우세'),
    (2, 'BF4_15comp: WaveSplat 0.302 vs Wang2025 0.249 (역전)'),
    (2, '격자는 모든 성분을 한꺼번에 담을 수 있음'),
    (2, 'WaveSplat 은 성분을 하나씩 찾아야 — beamforming peak 분리 어려움'),
    (1, 'Reversal point — 약 N=15 component'),
    (2, '단순 파도: 파라미터 표현이 효율'),
    (2, '복잡 파도: 격자 표현이 표현력 우위'),
    (1, '대응 — hybrid init (향후 연구)'),
    (2, '격자로 대략 분포 파악 → 파라미터로 정밀화'),
    (2, '두 표현의 장점 결합'),
    (2, 'Learned init (NN 기반) 도 대안'),
]

# slide C: Ablation
S4_C_ABLATION = [
    (0, 'Ablation — 두 메커니즘의 시너지'),
    (1, '4 가지 변형 비교 (12 episodes)'),
    (2, 'A (spatial-only): single-frame, no dispersion → nRMSE 0.213'),
    (2, 'B (multi-frame, no dispersion): → 0.208 (A 보다 개선)'),
    (2, 'C (disp-coupled, single-frame): → 0.226 (오히려 악화)'),
    (2, 'D = WaveSplat (full): → 0.211 (가장 안정)'),
    (1, 'C 가 왜 악화되는가'),
    (2, 'Single frame 에서 k 추정 부정확'),
    (2, 'Dispersion 으로 잘못된 k → 잘못된 ω 전파'),
    (2, '오류가 amplify 됨 (cascading failure)'),
    (1, 'D 가 어떻게 안정화 되는가'),
    (2, 'Multi-frame 으로 k 정확 추정'),
    (2, 'Dispersion 으로 ω 자동 결정 (자유도 감소)'),
    (2, 'Multi-frame + dispersion 결합 시 시너지'),
    (1, '핵심 메시지'),
    (2, '물리 법칙 (dispersion) 은 마법 아닌 *정규화* 역할'),
    (2, 'Multi-frame 과 함께 써야 효과 발휘'),
    (2, '두 메커니즘 단독은 불완전'),
]

# slide D: Main framing — dispersion = 정규화
S4_D_MECHANISM = [
    (0, '핵심 메커니즘 — 분산 관계 = 물리적 정규화 장치'),
    (1, '메커니즘 1 — 자유도 감소'),
    (2, '분산 관계: ω = √(g·|k|·tanh(|k|·d))'),
    (2, '주파수 ω 가 파수 k 로부터 자동 결정'),
    (2, '적용 전: 5N 변수 (A, k_x, k_y, ω, φ)'),
    (2, '적용 후: 4N 변수 (A, k_x, k_y, φ) — 20% 감소'),
    (2, '추가 효과: 물리적으로 불가능한 (k, ω) 자동 배제'),
    (1, '메커니즘 2 — Multi-frame temporal coherence'),
    (2, 'η(x, t) = A·cos(k·x − ω·t + φ) 의 t 진행이 phase 변화 제공'),
    (2, 'Vessel 정지여도 wave 자체의 시간 진화 → multi-frame 효과'),
    (2, '여러 frame 의 (x, y, t, η) 4D 정보 → k 모호성 해소'),
    (2, 'Note: vessel motion 기반 SA 는 현재 미적용 → 향후 연구'),
    (1, '두 메커니즘의 시너지 (Ablation 결과)'),
    (2, '단독으로는 불완전: dispersion 단독 → 0.226 (악화)'),
    (2, 'Multi-frame 단독 → 0.208 (개선되나 불안정)'),
    (2, '결합 (D=WaveSplat) → 0.211 (가장 안정)'),
    (1, '직관 비유'),
    (2, "Dispersion = '눈가리개' (물리 법칙으로 답 좁히기)"),
    (2, "Multi-frame = '여러 사진 으로 모호성 해소'"),
    (2, '둘 다 있어야 sparse LiDAR 로 phase-resolved 추정 가능'),
]

# slide E: 2-stage 설계 (Identifiability)
S4_E_STAGES = [
    (0, '왜 2단계 설계가 자연스러운가'),
    (1, '핵심 관찰 — 변수마다 필요한 정보가 다름'),
    (2, '파수 k = (k_x, k_y): 여러 frame 시간 정보 필요 (공간 패턴)'),
    (2, '주파수 ω, 진폭 A, 위상 φ: 한 frame 만으로도 추정 가능'),
    (1, '2단계로 자연 분리'),
    (2, 'Init: 여러 frame 모아 k 정확히 결정 — 한 번만 (< 1 s)'),
    (2, 'Track: k 고정, ω·A·φ 만 매 frame update — 실시간 (< 13 ms)'),
    (2, '무거운 작업은 한 번, 가벼운 작업은 매 frame'),
    (1, 'Identifiability 증거'),
    (2, 'Single frame 에서 k 의 FIM eigenvalue 가 (A,φ) 보다 10⁻³ 작음'),
    (2, 'k 자유 시 50점에 over-fit → 비물리적 발산'),
    (2, 'k 고정 → 자유도 5N → 2N → 안정적 추정'),
    (1, '전체 흐름 정리'),
    (2, '1. Multi-frame temporal coherence → 파수 추정 (Init)'),
    (2, '2. 분산 관계로 ω↔k 결합 → 자유도 감소 + 물리적 제약'),
    (2, '3. 매 frame Adam → 실시간 (A, φ) 추적'),
    (2, '4. Adaptive N → 성분 수 자동 조정'),
]

# §5 Limitations & Research plan (5 content slides)
# slide A: Limitations
S5_A_LIMITS = [
    (0, '한계 — 4 가지 정직한 제약'),
    (1, '한계 1 — 선형 근사 (Linear Wave Theory)'),
    (2, '파도를 사인파의 합으로 표현'),
    (2, '실제 파도는 꼭대기 뾰족·골 넓음 (비선형)'),
    (2, 'Klein 2020 비교: HOSM > NLSE2 > NLSE > LWT 순서'),
    (2, 'ε < 0.07 신뢰 → BF6 까지 적용 가능'),
    (2, '개선 방향: Stokes 2차 보정 → 적용 범위 확장'),
    (1, '한계 2 — 복잡 파도 (≥15 components) 정확도 저하'),
    (2, 'BF4_15comp: WaveSplat 0.302 vs Wang2025 0.249 (역전)'),
    (2, '원인: beamforming 이 다수 peak 분리 어려움'),
    (2, '개선 방향: hybrid init (격자 + 파라미터 정밀화)'),
    (1, '한계 3 — Vessel 정지 가정 (V=0 kts)'),
    (2, '본 dataset 은 vessel 정지 상태로 합성 — synthetic aperture (SA) 미적용'),
    (2, 'Multi-frame 효과는 wave 자체 시간 진화에 의존 (vessel 운동 효과 X)'),
    (2, '영향: 매우 긴 swell (λ > 250 m) 의 direction 해상도 저하 가능'),
    (2, '개선 방향: vessel motion 도입 → V·T 만큼 synthetic aperture 효과'),
    (1, '한계 4 — Wave breaking + 유속장 미지원'),
    (2, 'VOF-EnKF, PFT-PINN 영역 (offline only)'),
    (2, '본 연구는 real-time 제어 영역 — 응용 자체가 다름'),
    (2, '필요 시 응용 분야 분리하여 향후 협업 연구 가능'),
]

# slide B: Hydrofoil intro (응용)
S5_B_HYDROFOIL = [
    (0, '응용 사례 — Hydrofoil ride control'),
    (1, '왜 hydrofoil 인가'),
    (2, 'Hydrofoil 은 파도 만나면 lift 변동 → heave/pitch 흔들림 → 승차감 저하'),
    (2, '앞쪽 파도 preview 있으면 fin angle 미리 조절 가능'),
    (1, '제어 구조 — PD vs PD + FF (Feed-Forward)'),
    (2, '기존 PD: heave error 만 보고 fin 조절 (feedback only)'),
    (2, '제안 PD + FF: WaveSplat 추정 → 미래 wave disturbance 예측 → fin 보상'),
    (2, '최종 fin angle = PD(heave error) + FF(predicted wave disturbance)'),
    (1, 'Test setup'),
    (2, 'WaveSim 파도 + hydrofoil dynamics simulator'),
    (2, 'Preview window: 0.3 s lookahead (V=11.3 m/s 에서 약 3 m 앞)'),
    (1, '현재 단계 — preliminary test (응용 가능성 확인)'),
    (2, '체계적 controller tuning + robustness 평가는 후속 연구'),
]

# slide C: PD vs PD+FF results — animation focused, brief text
S5_C_HYDROFOIL_RESULTS = [
    (0, 'PD vs PD + FF — heave acceleration 감소 확인'),
    (1, 'aRMS: PD 0.090 m/s² → PD+FF 0.047 m/s² (약 48% 감소)'),
    (1, 'ISO 2631 ride comfort 기준에서 의미 있는 수준의 감소'),
    (1, '같은 파도 시퀀스, 두 controller 직접 비교 (위 영상)'),
]

# slide D: Future work
S5_D_FUTURE = [
    (0, '향후 연구 — 4가지 방향'),
    (1, '방향 1 — 고복잡도 파도 대응 (초기화 개선)'),
    (2, '현재: ≥15 component 에서 정확도 저하 (Wang2025 가 일부 우세)'),
    (2, '계획: hybrid init (격자 + 파라미터 정밀화)'),
    (2, '대안: learned init (NN 으로 초기 component 추정)'),
    (2, '기대: 고복잡도에서도 격자 방법 이상 정확도'),
    (1, '방향 2 — 비선형 파도 확장'),
    (2, '현재: LWT (선형) → BF6 까지 신뢰 (ε < 0.07)'),
    (2, '계획: Stokes 2차 보정 추가'),
    (2, '장기: Klein 2020 의 NLSE 수준 도달'),
    (1, '방향 3 — Vessel motion 기반 synthetic aperture (SAR-style)'),
    (2, '현재: 코드/dataset 모두 vessel 정지 (V=0 kts) — SA 미적용'),
    (2, '계획: V > 0 시 V·T 만큼 추가 spatial coverage 효과 검증'),
    (2, 'Reference: Wiley 1954 SAR 원리'),
    (2, 'Ship-mounted LiDAR 에서 SA 적용 — 첫 시도가 될 가능성'),
    (1, '방향 4 — 실측 데이터 검증'),
    (2, '현재: 합성 dataset (110 episode) 만 검증'),
    (2, '계획: 실제 선박 LiDAR 데이터'),
    (2, 'Ground-truth: wave buoy 또는 X-band radar 병행 측정'),
    (1, '최종 목표 — 선박 실시간 제어용 phase-resolved wave 추정 시스템'),
]


# ───────────────────────────────────────────────────────────
# Slide spec list — order = final presentation order
# Each spec: ('content', section_title, body_items) or ('keep', source_idx, ...)
# ───────────────────────────────────────────────────────────

# ───────────────────────────────────────────────────────────
# Presenter notes (발표자 프롬프트) — 각 슬라이드별 발표 스크립트
# ───────────────────────────────────────────────────────────

NOTES = {
    'S1_A': (
        "먼저 지금까지 바다 파도를 어떻게 측정하고 있고 무엇이 부족한지부터 말씀드릴게요.\n\n"
        "현재 산업계에서 표준으로 쓰는 방법은 *통계적 추정* 이에요. 평균 파고가 얼마고 "
        "평균 주기가 얼마인지 — 그런 *평균적인 바다 상태* 만 알려주는 방식이죠. 선박이나 "
        "해양 플랫폼을 *설계* 할 때는 이 정도면 충분합니다. '이 정도 환경에서는 버티게 "
        "만들자' 식으로 설계 기준을 잡을 수 있으니까요.\n\n"
        "그런데 *실시간으로 제어* 해야 할 때는 평균 정보로 부족합니다. 예를 들어 "
        "hydrofoil — 수중 날개로 부양하는 고속선인데요 — 이게 파도를 만나서 흔들리면 "
        "승객이 멀미가 나요. 이걸 막으려면 *앞쪽에 어떤 파도가 오는지 정확히 알아야* "
        "미리 대응할 수 있는데, 평균 정보로는 어림도 없거든요.\n\n"
        "그래서 이 연구의 목표는 phase-resolved estimation — 어렵게 들리지만 *각 파도가 "
        "언제 어디서 어떤 모양으로 오는지* 직접 추정하는 거예요. 응용 가능한 분야는 "
        "hydrofoil ride control, 자율운항 안전 판단, 선박 motion 예측 등 다양합니다."
    ),
    'S1_B': (
        "이 연구가 왜 어려운지 세 가지로 풀어서 말씀드릴게요.\n\n"
        "*첫 번째 어려움 — 측정 데이터 자체가 부족합니다.* LiDAR 는 빛을 쏴서 부딪힌 점들을 "
        "측정하는 센서인데, 한 번 스캔하면 *유효한 점이 50개 정도밖에* 안 됩니다. 그것도 "
        "거리가 가까우면 점이 빽빽하고 멀면 듬성듬성하고요. 게다가 파도 표면이 매끄러우면 "
        "빛이 카메라 쪽으로 안 돌아와서 점이 빠지고 (raydrop), 앞쪽 큰 파도가 뒤쪽을 "
        "가려서 또 빠지기도 해요. 그래서 기존에 '점이 촘촘하고 균일하다' 고 가정한 방법들은 "
        "그대로 못 씁니다.\n\n"
        "*두 번째 어려움 — 시간이 너무 부족해요.* 선박 제어 루프가 보통 10 Hz 로 돌아요. "
        "한 frame 처리에 0.1 초 안에 끝나야 하는 거죠. 그런데 기존 방법들은 다 이걸 "
        "못 만족합니다. Wang & Pan 그룹의 EnKF 방식은 비선형 wave 시뮬레이션을 매번 돌려야 "
        "해서 수백 ms 이상 걸리고요, Feng Wang 의 더 가벼운 ETKF + linear 방식도 시뮬레이션 "
        "300 개를 동시에 돌려야 해서 632 ms 가 걸려요. PINN 같은 신경망 방법은 instance "
        "마다 30 분씩 새로 학습해야 하고, FNO 같은 supervised ML 만 1.9 ms 로 빠른데 *학습 "
        "데이터 3120개* 를 미리 만들어 둬야 합니다.\n\n"
        "*세 번째 어려움 — 진짜 바다는 한 방향이 아니에요.* 실해는 바람으로 생긴 짧은 파 "
        "(wind sea) 와 먼 곳에서 온 긴 파 (swell) 가 *여러 방향에서* 섞여 옵니다. 그러니까 "
        "단순히 '파도가 어떤 길이냐' 가 아니라 '어디 방향으로 오느냐' 까지 알아야 해요. "
        "수학적으로는 단순한 숫자 k 가 아니라 방향까지 포함된 벡터 (k_x, k_y) 로 추정하는 "
        "거죠. 선행 연구들은 단순화한다고 1D 가정 (한 방향만) 으로 처리하는 경우가 많은데, "
        "우리는 2D 방향까지 그대로 처리합니다."
    ),
    'S1_C': (
        "배경 설명에 들어가기 전에 *결과부터 먼저* 보여드릴게요. '왜 이 연구를 계속할 가치가 "
        "있는지' 를 숫자로 보여드리는 게 더 직관적이거든요.\n\n"
        "12 개의 다양한 파도 시나리오 (BF2~BF5 까지) 에서 비교했을 때, 우리 *WaveSplat* 은 "
        "오차 0.211, 한 frame 처리에 13.4 밀리초가 걸립니다. 같은 데이터에 *Feng Wang 의 "
        "2025 년 baseline* 을 돌리면 오차 0.253, 632 밀리초가 걸리고요. 즉 *정확도는 16% "
        "더 좋고 속도는 47배 빠릅니다*. 그것도 GPU 없이 CPU 만으로요. 13 밀리초면 *초당 "
        "75 번* 추정 가능하니까 10 Hz 제어 루프에 충분히 들어갑니다.\n\n"
        "이게 어떻게 가능한지 핵심 메커니즘은 두 가지예요. 첫째, *분산 관계* 라는 물리 "
        "법칙을 강제로 끼워 넣었어요. 파도의 길이와 속도 관계를 수식으로 묶어버리면, 우리가 "
        "추정해야 할 변수가 5N 개에서 4N 개로 *20% 줄어들고*, 물리적으로 말이 안 되는 답은 "
        "자동으로 배제됩니다. 둘째, *여러 frame 의 시간 정보* 를 활용해요. 파도가 시간에 "
        "따라 움직이는 걸 누적하면, 한 순간 점만 봐서는 못 풀던 문제를 풀 수 있어요. 두 "
        "가지가 결합되어야 *공간 해상도 한계* 를 우회할 수 있습니다.\n\n"
        "응용 사례도 보여드릴 거예요. hydrofoil 의 흔들림을 제어해보니 *heave 가속도가 약 "
        "48% 감소* 했습니다. 이제부터 이 결과가 어떻게 나왔는지, 선행 연구와 어떻게 다른지 "
        "차례로 설명드릴게요."
    ),
    'S2_A': (
        "지난 6개월간 정독한 13 편의 선행 연구를 *5 가지 접근법* 으로 분류했어요. 하나씩 "
        "어떻게 접근하는지 풀어 말씀드릴게요.\n\n"
        "*① 격자 기반 최적화* — Desmars 그룹의 방법이에요. 바다를 격자로 나눠서 각 격자점의 "
        "파고를 변수로 두고, 측정값과 모델이 잘 맞도록 최적화하는 방식이죠. 수치해석에서 "
        "잘 정립된 framework 라 안정적입니다. 단점은 격자 사이의 짧은 파장은 표현이 불가능 "
        "(Nyquist 한계라고 해요) 하고, 대부분 한 방향 (1D) 만 다룬다는 거예요.\n\n"
        "*② 앙상블 칼만 필터* — Wang & Pan 그룹, Feng Wang 그룹, Yan 그룹이 있어요. 일종의 "
        "*시뮬레이션 다발* 방법인데, 50~300 개 wave simulation 을 동시에 돌리면서 측정이 "
        "들어올 때마다 통계적으로 갱신해요. Wang & Pan 은 비선형 HOS 모델을 쓰고, Feng Wang "
        "은 linear + fuzzy R 으로 가볍게, Yan 은 Navier-Stokes 까지 풀어서 wave breaking 도 "
        "처리합니다. 강점은 물리 정확도이고, 한계는 시뮬레이션 다발이라 *너무 느리다* 는 "
        "거예요.\n\n"
        "*③ 지도학습 신경망* — Ehlers 의 FNO (Fourier Neural Operator) 가 대표예요. 학습 "
        "후에는 1.9 ms 로 *엄청 빠릅니다*. 그런데 HOS 합성 데이터 3120 개를 미리 만들어 "
        "학습시켜야 하고, 학습한 분포 밖에서는 정확도가 떨어지고 (distribution shift), 1D "
        "전용에 black-box 라 해석이 어려운 한계가 있어요.\n\n"
        "*④ 물리 기반 비지도 신경망* — Ehlers 의 PFT-PINN 이에요. 신경망에 *물리 법칙* 을 "
        "직접 강제하는 방식이라 학습 데이터가 필요 없어요. 비선형 파도까지 처리할 수 있고요. "
        "단점은 instance 마다 *30분씩 새로 학습* 해야 해서 real-time 은 불가능합니다.\n\n"
        "*⑤ 우리 WaveSplat* — 격자를 안 쓰고 *N 개의 sine wave 파라미터를 직접 추정* 해요. "
        "100 ms 이내 처리 + 2D 방향 + LiDAR raydrop 모사까지 *모두 만족하는 첫 사례* 입니다. "
        "한계는 선형 가정이라 매우 가파른 파도는 못 다룬다 (ε < 0.07 영역) 는 거예요. "
        "자세한 paper 정보는 Appendix A 표 참조."
    ),
    'S2_B': (
        "이 슬라이드는 EnKF 기반 wave estimation 의 *두 독립 그룹* 을 자세히 보여드려요. "
        "이름이 비슷해서 헷갈리는데 *완전 다른 사람들* 입니다.\n\n"
        "*Wang & Pan 그룹* — Yulin Pan 교수가 이끄는 미국 미시건대 그룹이에요. 2021 년 "
        "EnKF 와 HOS (비선형 wave 모델) 를 처음 결합했고, 2022 년에는 IEnKF-HOS-C 로 *해류* "
        "까지 같이 추정하는 데 성공했어요. 센서는 X-band radar 를 쓰고 합성 데이터 + 실측 "
        "shipborne radar 데이터 둘 다 검증했습니다.\n\n"
        "*Feng Wang 그룹* — 중국 Harbin Engineering University 의 별개 그룹이에요. 2025 년 "
        "에 ETKF (EnKF 변형) 에 fuzzy R 이라는 노이즈 자동 적응 기법과 Delaunay 보간을 "
        "추가했어요. *우리 baseline 의 직접 prior art* 입니다. 시뮬레이션 300 개를 80×60 "
        "격자에서 돌리는데, 흥미로운 점은 센서가 *스테레오 카메라* 라는 거예요. Acqua Alta "
        "라는 베니스 인근 해양 플랫폼에서 실측 데이터를 받아 검증했고 GitHub 에 코드도 "
        "공개되어 있어요.\n\n"
        "우리 연구는 Feng Wang 의 *알고리즘만 재현* 했고, 공정한 비교를 위해 입력 sensor 는 "
        "LiDAR sim 으로 통일하고 forecast model 도 HOS 대신 linear 로 통일했어요. 알고리즘 "
        "차이만 깨끗하게 비교하기 위해서요.\n\n"
        "EnKF 류의 공통 한계는 *격자 해상도* (오차가 0.25 아래로 잘 안 내려가요) 와 "
        "*시뮬레이션 다발 비용* (632 ms / frame, real-time 안 됩니다) 입니다."
    ),
    'S2_C': (
        "*신경망을 쓰는* 두 가지 접근법을 보여드려요. 같은 Ehlers 그룹 (독일 함부르크 공대 "
        "+ 임페리얼 칼리지) 이 두 방향으로 발전시킨 사례입니다.\n\n"
        "*Ehlers 2023 — FNO* 는 *지도학습* 방식이에요. Fourier Neural Operator 라는 신경망 "
        "구조인데, 이름이 어렵지만 *주파수 공간에서 패턴을 학습* 한다는 뜻이에요. 일반 CNN "
        "보다 파도 같은 주기적 패턴에 잘 맞아요. 학습 후 추론은 1.9 ms 로 *가장 빠릅니다*. "
        "단점은 — 학습 데이터를 *3120 개나 미리 만들어야* 하고, 학습 분포 밖에서는 정확도가 "
        "떨어지고요 (distribution shift), 1D 전용에 *black-box* 라 왜 그렇게 추정하는지 "
        "설명이 어려워요.\n\n"
        "*Ehlers 2025 — PFT-PINN* 은 *학습 데이터가 0개* 라는 게 충격적인 부분이에요. *물리 "
        "법칙 자체를 신경망의 손실 함수* 로 강제해서 학습합니다. 두 신경망을 동시에 학습 "
        "시키는데, 하나는 파도 표면 (η) 을, 다른 하나는 *유속 potential* (Φ) 을 출력해요. "
        "비선형 파도까지 처리할 수 있고, 실험실 wave flume (MARATHON) 에서 실측 데이터로 "
        "검증했어요. 한계는 *instance 마다 30 분씩 새로 학습* 해야 한다는 거예요 "
        "(RTX 3090 GPU 기준). 1D + t 만 다루고 wave gauge 같은 점 센서만 받습니다.\n\n"
        "이 PFT-PINN 이 사실 *우리에게 가장 위협적인* 선행 연구예요. 이유는 *근본 철학이 "
        "같기 때문* — 학습 데이터 없이 물리 법칙으로 답을 좁히는 방식이거든요. 차이는 *속도* "
        "에서 결정적입니다. 우리는 13 ms, PINN 은 30 분 — *수만 배 차이* 가 나요. 응용 영역도 "
        "달라요. 우리는 real-time 제어, PINN 은 design-time 정밀 분석.\n\n"
        "솔직히 말하면 같은 그룹이 1-3 년 안에 *2D + real-time PINN* 을 발표할 가능성도 "
        "있어서 우리 niche 가 좁아질 위험은 있습니다."
    ),
    'S2_D': (
        "이 표가 *우리 연구가 어떤 빈 영역을 채우는지* 한눈에 보여줘요. 가로축은 6 가지 "
        "응용 요구사항, 세로축은 우리와 5 개 baseline 입니다. 노란색이 우리 WaveSplat 이고요.\n\n"
        "핵심은 — *어떤 방법도 모든 ✓ 를 받지 못합니다*. 각자 강점이 다르고 적합한 응용도 "
        "다르거든요.\n\n"
        "그런데 우리만 *4 가지를 동시에 만족* 해요. real-time 가능 + 2D 방향 추정 + 희소한 "
        "LiDAR 처리 + 학습 데이터 불필요 — 이 조합을 만족하는 선행 연구는 없습니다. 이게 "
        "우리의 *고유 영역* 이에요.\n\n"
        "응용 영역을 둘로 나눠 보면 더 명확해집니다. *real-time control 영역* 에는 우리 "
        "WaveSplat 과 FNO 가 들어가요 (즉석 결정 필요). *design-time 정밀 분석 영역* 에는 "
        "VOF-EnKF 와 PFT-PINN 이 들어갑니다 (시간 충분, 정확도 추구). 두 영역은 *경쟁이 "
        "아닌 보완* 관계예요. 우리가 PINN 과 직접 싸우는 게 아니라 서로 다른 영역을 담당 "
        "하는 거죠.\n\n"
        "물론 정직한 한계도 표에 보여요. 파도 파쇄 처리는 VOF-EnKF 만 가능하고, 유속장 "
        "추정은 PINN 영역이에요. 우리는 응용 영역 자체가 달라서 이런 건 처음부터 안 다룹 "
        "니다. paper 의 limitation 섹션에 정직히 명시할 거예요."
    ),
    'S3_A': (
        "여기서부터는 *우리가 만든 WaveSim 시뮬레이터* 를 보여드릴게요. 파도와 LiDAR 측정을 "
        "물리적으로 모사하는 도구입니다. 4 가지 구성 요소가 있어요.\n\n"
        "*첫째, 파도 모델* — Gerstner 라는 모델을 썼어요. 단순 sin 파도 (LWT) 는 *수면이 "
        "올라갔다 내려갔다* 만 표현하는데, Gerstner 는 *물 입자 자체가 원이나 타원으로 "
        "회전* 하면서 표면을 만든다고 봅니다. 더 사실적이에요. Q 라는 가파름 척도가 있는데 "
        "Q=0 이면 단순 sin 파, Q>0 이면 마루는 뾰족하고 골은 평평한 약비선형 파도가 됩니다. "
        "Q 가 너무 크면 입자 궤적이 자기 자신을 교차해버려서 비물리적이라 *Σ Q·ω·A ≤ 1* 라는 "
        "제약을 강제해요 (Tessendorf 2001 의 표준 방식).\n\n"
        "*둘째, 스펙트럼* — JONSWAP 이라는 표준 파도 분포 모델을 썼어요. 1973 년 Hasselmann "
        "이 북해 측정으로 정립한 거고요, 유의파고 H_s 와 첨두주기 T_p 만 입력하면 N 개의 "
        "파도 성분이 자동으로 sampling 됩니다. 방향은 cos² 함수로 *흩뿌려서* 여러 방향을 "
        "만들어요. 국제 학회 ITTC, IACS 가 모두 채택한 표준이라 신뢰성은 확실합니다.\n\n"
        "*셋째, 분산 관계* — 1845 년 Airy 가 정립한 *파도 길이와 속도의 관계식* 이에요. "
        "100 년 이상 검증된 물리 법칙이고, 깊은 물에서는 단순한 형태로 떨어집니다 "
        "(ω ≈ √(g·k))."
    ),
    'S3_B': (
        "이제 LiDAR 센서를 어떻게 모사했는지 말씀드릴게요. *실제 marine LiDAR 인 Falcon K1 "
        "스펙* 을 그대로 본떴습니다.\n\n"
        "*센서 사양* — 152 개 채널 × 500 개 horizontal beam = 한 번 스캔에 *76,000 개 빔* 을 "
        "쏩니다. range 는 10 m 부터 250 m 까지, 파장은 1550 nm (eye-safe 적외선). 선박 위 "
        "5 m 에 마운트되어 있다고 가정했어요. 시야는 *전방 ±60°*, 수평선 *아래로 25° 까지* "
        "내려다보는 구조예요.\n\n"
        "*Ray tracing* 은 그래픽스 분야의 표준 알고리즘 (Möller-Trumbore 1997) 을 썼어요. "
        "각 빔이 파도 표면 mesh 와 처음 만나는 점을 찾는 거죠.\n\n"
        "*Raydrop 모델* 이 좀 복잡한데요, 이게 LiDAR 만의 특징이에요. 빛을 쐈다고 무조건 "
        "점이 찍히는 게 아니라, *받기 위해서는 충분한 비율의 빛이 카메라 방향으로 반사돼야* "
        "해요. 그래서 Snell 굴절 + Fresnel 반사율 + Lambert 산란을 다 결합해서 backscatter "
        "확률을 계산했어요. 1550 nm 파장에서 물의 굴절률은 1.321 입니다. 결과적으로 76,000 "
        "개 빔 중에 *50 개 정도만 valid 한 점* 으로 남게 되는 거예요. 그래서 sparse 한 거고요.\n\n"
        "*중요한 한 가지* — 이 시뮬레이션에서 *선박은 정지* 가정 (V=0 kts) 입니다. 모든 frame "
        "이 같은 위치에서 측정되고, 그래서 multi-frame 효과는 *파도 자체의 시간 진화* 에서 "
        "옵니다. 선박이 움직이면서 생기는 synthetic aperture (SAR 처럼) 는 *현재 미적용* "
        "이고 향후 연구로 분리했어요."
    ),
    'S3_C': (
        "*합성 dataset 의 구성* 을 말씀드릴게요.\n\n"
        "총 *110 개 episode* 를 만들었어요. 다양한 sea state (BF2 잔잔한 바다부터 BF7 거친 "
        "폭풍까지 6 단계), 다양한 wave 성분 수 (1 개 단순한 파부터 30 개 복잡한 파까지 5 "
        "단계), seed 5 개로 재현 가능하게, 그리고 raydrop 모델도 2 가지를 섞어서요.\n\n"
        "각 episode 는 — *30 초 동안 0.1 초 간격* (10 Hz) 으로 시뮬레이션해서 frame 301 개. "
        "Ground-truth 파도장은 80×60 격자에 240 × 200 m 영역으로 만들고요, LiDAR scan 은 "
        "76,000 빔 쏴서 raydrop 후 *valid 점 50 개 정도* 가 남습니다. 실제 보면 patch "
        "coverage 가 50~150 m 정도고요.\n\n"
        "전체 용량은 *1.2 GB* 정도예요. 참고로 BF3~BF6 의 dominant 파도 길이 (λ_peak) 를 "
        "JONSWAP 으로 계산해보면 BF3 가 47 m, BF4 가 77 m, BF5 가 126 m, BF6 가 156 m 입니다. "
        "patch 가 50~150 m 니까 *1~3 wavelength 가 들어와* 방향 추정에 충분해요. 특히 BF6 는 "
        "wavenumber 가 전부 Nyquist 한계 이내라 spatial sampling 으로도 잡을 수 있습니다."
    ),
    'S3_D': (
        "*WaveSplat 알고리즘의 근본 발상* 을 말씀드릴게요. 발상 전환이 핵심입니다.\n\n"
        "기존 격자 방법은 *80×60 = 4800 개 격자점* 의 파고를 추정해야 했어요. LiDAR 측정은 "
        "50 점밖에 없는데 4800 개를 추정하려니까 정보가 절대 부족하잖아요? 그래서 부드러움 "
        "가정 같은 추가 제약이 필요했어요.\n\n"
        "그런데 *발상을 전환했어요*. JONSWAP 으로 실해 파도를 분해해보면 *보통 5~30 개 sine "
        "wave 의 합* 으로 충분히 표현됩니다. 각 성분은 5 개 파라미터 — 진폭 A, 파수 k_x, k_y, "
        "주파수 ω, 위상 φ 로 완전 결정되고요. *N=10 개* 성분이면 변수 50 개라 LiDAR 50 점 "
        "으로 *1:1 추정* 가능해집니다. 4800 개를 풀려는 게 아니라 50 개만 풀면 되는 거예요.\n\n"
        "수식으로 쓰면 η = Σ A_n · cos(k_n·x − ω_n·t + φ_n) 입니다. 그리고 여기서 *분산 "
        "관계를 강제* 해요 — ω 가 k 에서 자동 결정되니까 ω 를 따로 변수로 안 둬도 됩니다. "
        "결과: 변수 5N 에서 4N 으로 *20% 감소* + 물리적으로 불가능한 (k, ω) 조합 자동 배제.\n\n"
        "그리고 추정을 *2 단계로 나눴어요*. *Init 단계* 는 처음 1 초 이내에 한 번만 실행 "
        "되고 (k_x, k_y) 를 정확히 결정합니다. *Track 단계* 는 매 frame 13 ms 안에 끝나고 "
        "(A, φ) 만 update 해요. *무거운 작업은 처음 한 번만, 가벼운 작업은 매 frame 실시간* "
        "하는 게 핵심 디자인입니다."
    ),
    'S3_E': (
        "Init 단계의 핵심 도전을 풀어 말씀드릴게요. 왜 *여러 frame 을 모아야* 하는지가 "
        "이 슬라이드의 주제예요.\n\n"
        "*Spatial Nyquist 한계* — 신호처리에 있는 유명한 정리예요 (Shannon 1949). 짧게 "
        "설명하면 *어떤 wavelength 의 신호를 측정하려면 그 wavelength 의 절반 간격으로 "
        "sampling* 해야 한다는 거예요. 우리 LiDAR 의 점 간격이 5~8 m 정도니까, 한 순간만 "
        "보면 *최대 0.4 rad/m 까지의 wavenumber* 만 추정 가능합니다.\n\n"
        "그런데 우리가 잡아야 할 wavenumber 범위는 더 넓어요. 특히 BF3 처럼 짧은 파도가 "
        "섞인 sea state 에서는 *Nyquist 한계의 3 배까지* 도달할 수 있어요. Paradox 인 거죠. "
        "다행히 BF5-BF6 처럼 긴 파도가 dominant 인 경우는 대부분 Nyquist 이내고요. BF3 의 "
        "short tail 만 문제예요.\n\n"
        "*해결책은 — 시간을 활용하는 거예요.* 선박이 정지해 있어도 *파도 자체가 시간에 따라 "
        "이동* 하잖아요. η(x, t) = A·cos(k·x − ω·t + φ) 라는 식에서 시간 t 가 진행하면 phase "
        "가 변하니까, 여러 frame 을 누적하면 같은 위치의 점이라도 *다른 phase 정보* 를 "
        "얻습니다. 거기에 분산 관계로 ω 와 k 를 연결하면 spatial Nyquist 한계를 *우회* 할 "
        "수 있어요.\n\n"
        "구체적으로 우리가 쓰는 init window 길이는 sea state 별 약 *10~20 초* (파도 주기의 "
        "2 배) 이고, 그 동안 *수천 개의 4D 점* (x, y, t, η) 이 누적되어 wavenumber 모호성을 "
        "해소합니다. 참고로 vessel 이 움직이면 추가로 synthetic aperture 효과까지 얻을 수 "
        "있는데 이건 향후 연구로 분리했습니다."
    ),
    'S3_F': (
        "이제 Init 단계가 *어떻게 작동* 하는지 풀어 말씀드릴게요. 세 가지 sub-step 이 있어요.\n\n"
        "*첫째, Beamforming score* — *어떤 wavenumber 가 측정 데이터와 가장 잘 맞는가* 를 "
        "점수로 평가하는 방식이에요. 신호처리에서 흔히 쓰는 matched filter 와 같은 원리고요. "
        "각 후보 (k_x, k_y) 에 대해 데이터를 phase 정렬해보고 *얼마나 강하게 합산되는지* 를 "
        "score 로 매깁니다. 분산 관계 덕분에 ω 는 자동이라 2D 검색 (k_x, k_y) 만 하면 돼요. "
        "검색 범위는 [-4, 4] rad/m, 81×81 격자에서 *수천 개 후보* 를 평가합니다.\n\n"
        "*둘째, Coarse-to-fine schedule* — 처음부터 N 개 성분을 다 찾으려고 하면 어려워서 "
        "*점진적으로 늘려가요*. 처음엔 N=3 으로 큰 성분만 찾고, 어느 정도 수렴하면 N=8 로 "
        "늘리고, 마지막에 N=15 까지. 빠르게 수렴하는 단계는 바로 다음으로 넘어가서 전체 "
        "시간을 단축해요.\n\n"
        "*셋째, Adam refinement* — Beamforming 결과는 *초기값* 일 뿐이에요. 그 후 Adam "
        "optimizer 로 init window 전체 데이터에 다시 fitting 합니다. (k_x, k_y, A, φ) 모두 "
        "update 하지만, k 는 변화에 신중해야 해서 *학습률을 절반으로* 줘요. Loss 는 세 항 — "
        "data 가 잘 맞는지 + 성분이 너무 많지 않은지 (L1 sparsity) + spectrum 이 부드러운지 "
        "(smoothness) 입니다.\n\n"
        "주요 hyperparameter 들은 — L1 가중치 0.001, smoothness 가중치 0.0001, 가파름 한계 "
        "0.15 정도예요."
    ),
    'S3_G': (
        "이 슬라이드는 Beamforming score 식이 *왜* 이런 형태인지 직관적으로 풀어드려요.\n\n"
        "*핵심 직관 — '코러스 효과' 라고 생각하시면 돼요.* 합창단이 같은 음을 정확히 부르면 "
        "소리가 *N 배로 커지지만*, 음이 다 다르면 서로 상쇄돼서 √N 정도만 커져요. 우리도 "
        "비슷합니다. 후보 (k, ω) 가 *맞으면* 모든 LiDAR 점의 phase 가 정렬돼서 합이 N 배로 "
        "강해지고요, *틀리면* random phase 로 합산돼서 약 √N 정도만 강해져요. 그러니까 SNR "
        "(신호 대 잡음비) 이득이 *점 개수에 비례* 합니다. Init window 의 수천 점 정도면 "
        "충분한 dB 의 SNR 향상이 나와요.\n\n"
        "*왜 이 형태가 표준인가* — 이건 공돌이라면 알고 있어야 할 *delay-and-sum "
        "beamforming* 이라는 신호처리 표준이에요 (Kraus 1988, Van Trees 2002). 1960 년 Turin "
        "의 *matched filter 정리* 가 Gaussian noise 에서 SNR 을 최대화하는 *optimal filter* "
        "임을 증명했고요. 게다가 *단순하고 robust*, covariance matrix 같은 것 안 추정해도 "
        "되는 게 sparse data 에 적합해요.\n\n"
        "*분산 관계의 차원 축소 효과* — 사실 dispersion 없이 풀면 (k_x, k_y, ω) 3 차원 검색 "
        "이라서 후보가 *수십만 개* 나오고 계산이 안 돼요. dispersion 으로 ω 를 자동 결정하게 "
        "강제하면 2 차원 검색이라 수천 개로 줄어요.\n\n"
        "참고로 더 정교한 spectral 방법 (MUSIC, ESPRIT, Capon/MVDR) 도 있는데, 이런 건 "
        "covariance matrix 가 필요하거나 sparse 점에서 불안정해서 *우리 상황에 부적합* "
        "합니다. delay-and-sum 의 단순함이 오히려 강점이에요."
    ),
    'S3_H': (
        "*Track 단계* 가 매 frame 13 ms 안에 끝나는 비결을 말씀드릴게요. *Adaptive N* 도 "
        "이 단계에서 작동합니다.\n\n"
        "*핵심 — 파수 k 는 Init 에서 결정한 값으로 고정해요*. 이게 WaveSplat 의 가장 중요한 "
        "설계 결정이에요. 왜냐하면 한 순간의 50 개 점만 보면 k 추정이 *극도로 불안정* 하거든요. "
        "수학적으로 표현하면 *Fisher 정보 (k 변화에 데이터가 반응하는 정도) 가 (A, φ) 보다 "
        "1000 배 작아요*. k 를 자유롭게 두면 50 점에 over-fit 해서 비물리적인 답으로 발산해 "
        "버립니다. 그래서 k 는 init 결과로 고정하고 *진폭 A 와 위상 φ 만 매 frame update* "
        "하는 거죠. ω 는 분산 관계로 자동.\n\n"
        "*Adam optimizer* — 딥러닝에서 가장 많이 쓰이는 최적화 방법이에요. gradient 의 "
        "1 차 모멘트와 2 차 모멘트를 추적하면서 *적응적 학습률* 로 update 합니다. 학습률 "
        "0.005, frame 당 10 step 정도 돌리면 약 *13 ms* 가 걸려요. 매 update 전에 *phase "
        "예측* (φ ← φ + ω·Δt) 도 해줘요 — kinematic preview 라고 부릅니다. 그리고 최근 "
        "3 개 frame 을 buffer 로 모아서 gradient 가 안정적으로 계산되도록 했어요.\n\n"
        "*Adaptive N* 은 성분 수를 *동적으로 조정* 해요. 세 가지 메커니즘이 있습니다.\n"
        "(1) *Pruning (가지치기)* — 진폭이 작아진 성분은 제거. 누적 에너지 90% 만 유지.\n"
        "(2) *Growth (성장)* — 잔여 오차가 5 cm 넘으면 새 (k_x, k_y) 를 추가 (최대 30 개).\n"
        "(3) *Safety rollback (안전 복구)* — 변경 후 오차가 1.3 배 이상 늘어나면 즉시 이전 "
        "상태로 되돌림. 잘못된 prune/grow 가 cascade failure 로 번지는 걸 방지하는 거죠. 매 "
        "frame 에서 이 세 가지가 자동 작동하면서 N 을 적정 수준으로 유지합니다."
    ),
    'S4_A': (
        "이 슬라이드가 *핵심 비교 결과* 입니다. 12 개 다양한 파도 시나리오에서 측정한 평균 "
        "값이에요.\n\n"
        "*WaveSplat* 평균 오차 0.211, 13.4 ms / frame. *Wang2025* (Feng Wang 의 baseline 을 "
        "충실히 재현한 것) 는 0.253, 632 ms. 즉 *정확도는 16% 더 좋고 속도는 47 배 빠른* "
        "거예요. 그것도 GPU 없이 *CPU 만으로* 입니다. 13 ms 면 초당 75 frame 처리 가능하니까 "
        "10 Hz 제어 루프에 충분히 들어갑니다. Wang2025 의 632 ms 는 초당 1.6 frame 이라 "
        "real-time 제어에는 못 써요.\n\n"
        "참고로 다른 baseline 들도 표에 넣었어요. *Ehlers FNO* 는 1.9 ms 로 가장 빠르지만 "
        "학습 데이터 3120 개를 미리 만들어야 하고요. *Ehlers PFT-PINN* 은 학습 데이터가 0 개 "
        "지만 instance 마다 30 분씩 새로 학습해야 합니다. *Yan VOF-EnKF* 는 wave breaking 까지 "
        "처리할 수 있지만 *offline 만 가능* 해요.\n\n"
        "WaveSplat 의 우위는 *모든 항목을 동시에 만족* 한다는 점이에요. 정확도 좋고, 속도 "
        "빠르고, 학습 데이터 필요 없고, 2D 방향 처리하고, 희소 LiDAR 도 받고. 다만 다음 "
        "슬라이드에서 *episode 별로* 자세히 보면 *강점도 한계도* 분명히 드러납니다."
    ),
    'S4_B': (
        "이 그래프가 *우리 연구의 강점과 한계를 정직하게* 보여줘요.\n\n"
        "*단순한 파도 (1~5 개 성분) 에서는 WaveSplat 이 압도적입니다.* 예를 들어 BF3 의 "
        "2 성분 case 에서는 우리 오차가 0.052 인데 Wang2025 는 0.250 이에요. *거의 완벽한 "
        "추정* 입니다. 왜냐하면 2 개 성분이면 파라미터 10 개만 추정하면 되고, LiDAR 50 점 "
        "이면 충분히 잘 맞을 수 있거든요. 격자 방법은 성분이 적어도 *격자 해상도 한계* 가 "
        "있어서 이 정확도에 못 도달해요.\n\n"
        "*그런데 복잡한 파도 (15~30 개 성분) 에서는 일부 역전* 이 일어나요. BF4 의 15 성분 "
        "case 에서 WaveSplat 은 0.302 인데 Wang2025 는 0.249 로 *오히려 낫습니다*. 이유는 — "
        "격자 방법은 모든 성분을 격자에 *한꺼번에 담을 수 있지만*, 우리 WaveSplat 은 "
        "beamforming 으로 peak 를 *하나하나 분리해야* 해요. 30 개 peak 를 정확히 분리하기는 "
        "쉽지 않거든요.\n\n"
        "대략 *N=15 부근에서 역전* 이 일어나요. 단순한 파도는 파라미터 표현이 효율, 복잡한 "
        "파도는 격자가 표현력 우위입니다.\n\n"
        "대응 방향은 *hybrid init* — 격자로 대략 분포를 잡은 다음 파라미터로 정밀화하는 "
        "2 단계 초기화예요. 두 표현의 장점을 결합하는 거죠. 신경망으로 초기 성분을 학습 "
        "하는 *learned init* 도 대안이고요. 이건 향후 연구로 두었습니다."
    ),
    'S4_C': (
        "이 슬라이드가 *각 기술이 실제로 도움이 되는지* 분리해서 검증한 ablation 입니다. "
        "Ablation 이란 의학에서 온 용어인데, *기술을 하나씩 빼보면서 빠진 게 도움이 되는지* "
        "확인하는 실험이에요.\n\n"
        "4 가지 변형을 비교했어요.\n"
        "  *A (기본)* — 한 frame 만 보고 dispersion 도 안 쓴 가장 단순한 버전. 오차 0.213.\n"
        "  *B* — A 에 multi-frame 만 추가. 0.208 로 *조금 개선*.\n"
        "  *C* — A 에 dispersion 만 추가 (한 frame 그대로). 0.226 으로 *오히려 악화* 됐어요!\n"
        "  *D = WaveSplat* — multi-frame + dispersion 둘 다. 0.211 로 *가장 안정*.\n\n"
        "*가장 흥미로운 건 C 가 오히려 악화된다는 거예요.* 이유는 — 한 frame 에서 k 가 부정확 "
        "하게 추정되면, dispersion 으로 ω 가 *그 부정확함을 그대로 따라가서* 오류가 증폭 "
        "됩니다. 즉 dispersion 은 *데이터로 k 가 잘 잡혀야만* 효과를 발휘하는 거예요.\n\n"
        "*D 가 안정적인 이유는* multi-frame 이 k 를 정확히 잡아주고, 그 위에서 dispersion 이 "
        "ω 를 자동 결정하면서 자유도를 줄여주기 때문이에요. 두 가지가 *시너지* 를 내는 거죠.\n\n"
        "*핵심 메시지* — 물리 법칙 (dispersion) 은 마법이 아니라 *정규화 역할* 입니다. 데이터 "
        "로 잘 잡힌 k 위에서 작동해야 의미가 있어요. 그래서 우리는 multi-frame 과 dispersion 을 "
        "*함께* 써야 한다는 게 핵심 발견입니다."
    ),
    'S4_D': (
        "이 슬라이드가 *우리 연구의 핵심 메시지* 예요. 분산 관계가 *정규화 역할* 을 한다는 "
        "발견이 paper 의 main framing 입니다.\n\n"
        "*메커니즘 1 — 자유도 감소* — 분산 관계 ω = √(g·|k|·tanh(|k|·d)) 라는 물리 법칙을 "
        "강제로 끼워 넣으면, ω 가 k 에서 자동 결정됩니다. 그러면 우리가 추정해야 할 변수가 "
        "5N 개에서 4N 개로 *20% 줄어들어요*. 그리고 *물리적으로 불가능한 (k, ω) 조합은 자동 "
        "배제* 됩니다. 마치 답을 미리 좁혀놓는 정규화 효과예요.\n\n"
        "*메커니즘 2 — 여러 frame 시간 정보 활용* — η(x, t) 가 시간에 따라 propagate 하는 걸 "
        "이용해요. *선박이 정지해 있어도* 파도 자체가 움직이니까 multi-frame 효과가 자연스 "
        "럽게 옵니다. 여러 frame 의 (x, y, t, η) 4D 정보를 모으면 wavenumber 의 모호성을 "
        "해소할 수 있어요. 참고로 *선박이 움직이면* SAR (synthetic aperture radar) 처럼 "
        "추가로 분해능이 향상되는 효과까지 기대할 수 있는데 이건 향후 연구로 분리했습니다.\n\n"
        "*두 메커니즘의 시너지가 핵심* — 앞 슬라이드 ablation 에서 봤듯이, dispersion 만 "
        "쓰면 *오히려 악화* (0.226), multi-frame 만 쓰면 *개선되지만 불안정* (0.208), 둘 다 "
        "쓰면 *가장 안정* (0.211) 입니다.\n\n"
        "*비유로 정리하면* — Dispersion 은 *눈가리개* (물리 법칙으로 답 좁히기), multi-frame "
        "은 *여러 시점 사진* (시간 차이로 모호성 해소) 이라고 할 수 있어요. 두 가지가 다 "
        "있어야 sparse LiDAR 로 phase-resolved 추정이 가능합니다."
    ),
    'S4_E': (
        "*왜 2 단계 (Init/Track) 설계가 자연스러운지* 풀어 말씀드릴게요.\n\n"
        "*핵심 관찰부터* — 변수마다 *필요한 정보가 다르다* 는 거예요. 파수 k 는 파도의 "
        "*공간적 패턴* 이라 한 순간 50 점 LiDAR 만으로는 정확히 파악이 어려워요. 여러 frame "
        "의 시간 정보가 필요합니다. 반면에 진폭 A 와 위상 φ 는 파도의 *크기와 현재 위치* "
        "라서 한 순간 데이터로도 충분히 추정 가능해요.\n\n"
        "*수학적 증거도 있어요* — Fisher information matrix 라는 걸로 분석해보면, k 의 "
        "고유값이 (A, φ) 보다 *약 1000 배 작아요*. 이게 무슨 뜻이냐면 *k 가 변해도 데이터가 "
        "거의 반응 안 한다* 는 거예요. 그러면 k 를 자유롭게 두면 50 개 점에 *멋대로 "
        "over-fit 해서 비물리적인 답으로 발산* 합니다. k 를 고정하면 자유도가 5N 에서 2N 으로 "
        "줄어 *훨씬 안정적인 추정* 이 가능해져요.\n\n"
        "*그래서 전체 흐름은 자연스럽게 4 단계로 정리됩니다*:\n"
        "  (1) 여러 frame 시간 정보로 파수 k 추정 (Init 단계, 한 번)\n"
        "  (2) 분산 관계로 ω 자동 결정 → 자유도 감소 + 물리 제약\n"
        "  (3) 매 frame Adam 으로 (A, φ) 빠르게 추적 (Track 단계)\n"
        "  (4) Adaptive N 으로 성분 수 자동 조정\n\n"
        "이 4 단계가 합쳐져서 *13 ms / frame 의 real-time* + *정확도 0.211* 을 동시에 "
        "달성하는 게 우리 알고리즘의 디자인입니다."
    ),
    'S5_A': (
        "*한계를 정직하게 공유* 하는 시간입니다. 연구 신뢰도를 높이려면 잘 되는 점뿐 아니라 "
        "한계도 먼저 인정해야 해요. 4 가지 정직한 제약을 말씀드릴게요.\n\n"
        "*한계 1 — 선형 근사를 썼다는 점* — 우리는 파도를 *sin 파의 합* 으로 표현해요. "
        "그런데 실제 파도는 *비선형* 입니다. 마루는 뾰족하고 골은 평평하죠. Klein 2020 의 "
        "wave model 비교 연구를 보면 정확도 순서가 *HOSM > NLSE2 > NLSE > LWT* 인데, 우리 "
        "LWT 는 가장 단순한 family 예요. 가파름 ε < 0.07 영역에서만 신뢰할 수 있고 BF6 "
        "까지가 적용 한계입니다. 향후 Stokes 2 차 보정 추가하면 확장 가능해요.\n\n"
        "*한계 2 — 복잡한 파도에서 정확도 저하* — 앞에서 본 reversal 이에요. 15 개 이상 "
        "성분에서는 Wang2025 가 더 나은 경우도 있어요. BF4_15comp 에서 우리 0.302 vs "
        "Wang2025 0.249. 원인은 beamforming 이 *다수 peak 를 정확히 분리* 하기 어렵다는 거 "
        "고요. 향후 hybrid init (격자 → 파라미터 정밀화) 로 해결할 예정입니다.\n\n"
        "*한계 3 — 선박이 정지 상태라고 가정한 점* — 사실 우리 dataset 은 *vessel 정지 "
        "(V=0 kts) 로 합성* 되어 있어요. 그래서 SA (synthetic aperture) 효과를 활용하지 "
        "못합니다. 영향은 *매우 긴 swell (λ > 250 m)* 의 방향 해상도가 저하될 수 있다는 "
        "거예요. 향후 vessel motion 도입하면 V·T 만큼 추가 공간 정보를 얻을 수 있습니다.\n\n"
        "*한계 4 — Wave breaking 과 유속장 추정은 미지원* — 이런 건 VOF-EnKF, PFT-PINN 같은 "
        "비선형 시뮬레이션 방법만 가능해요 (그것도 offline). 우리 연구는 *real-time 제어* "
        "영역이라 응용 목적 자체가 다릅니다. 필요하면 그 분야 연구자들과 협업 가능."
    ),
    'S5_B': (
        "WaveSplat 을 실제 응용에 *시험 적용* 한 사례를 보여드릴게요. *Hydrofoil ride control* "
        "에 적용해본 결과예요.\n\n"
        "*왜 hydrofoil 인가* — Hydrofoil 은 수중 날개로 lift 를 만들어서 *선체를 물 위로 "
        "들어올리는* 고속선이에요. 그런데 *파도를 만나면 lift 가 변동* 해서 선체가 heave/"
        "pitch 로 흔들립니다. 승차감이 떨어지는 거죠. 만약 *앞쪽에 어떤 파도가 오는지 미리 "
        "알면* fin angle 을 미리 조절해서 disturbance 를 상쇄할 수 있어요. *WaveSplat 의 "
        "real-time 추정이 정확히 이 preview 를 제공* 하기 때문에 좋은 응용입니다.\n\n"
        "*제어 구조 비교* — 기존 *PD controller* 는 *현재 heave error 만* 보고 fin 을 조절 "
        "해요 (feedback only). 우리가 제안하는 *PD + Feed-Forward* 는 WaveSplat 으로 미래 "
        "wave disturbance 를 예측해서 fin angle 에 *보상값을 미리 더해줍니다*. 수식으로 "
        "쓰면 `최종 fin angle = PD(heave error) + FF(predicted wave)` 입니다.\n\n"
        "*Test setup* — WaveSim 합성 파도와 hydrofoil dynamics simulator 를 결합했어요. "
        "Preview window 는 0.3 초 lookahead 로 잡았고요 (V=11.3 m/s 에서 약 3 m 앞). 같은 "
        "파도 시퀀스에 두 controller 를 적용해 직접 비교했습니다.\n\n"
        "*강조드리고 싶은 건* — 이건 *preliminary test* 입니다. *응용 가능성을 확인* 하는 "
        "정도예요. 체계적 controller tuning, robustness 평가, 실제 deployment 변수들은 "
        "후속 연구로 두었습니다."
    ),
    'S5_C': (
        "*위 영상이 PD vs PD+FF 의 직접 비교* 입니다. 잠시 영상에 집중해 주세요.\n\n"
        "영상은 두 부분으로 구성되어 있어요. *위쪽* 은 시간에 따른 파도와 vessel 위치를 "
        "보여주고요, *아래쪽* 은 heave acceleration 의 time series 입니다. *파란선이 PD-only, "
        "빨간선이 PD+FF (WaveSplat 사용)* 입니다. 보시면 빨간선의 *진동이 명확히 작아요*.\n\n"
        "*정량 결과로 보면* — aRMS (root mean square 가속도) 가 PD-only 0.090 m/s² 에서 "
        "PD+FF 0.047 m/s² 로 *약 48% 감소* 했어요. ISO 2631 ride comfort 기준에서 *의미 있는 "
        "수준의 감소* 라 승차감 향상으로 직결됩니다.\n\n"
        "*다시 강조드리지만 이건 시험 적용 결과* 예요. WaveSplat 정보가 제어에 *실제로 "
        "도움이 된다는 proof-of-concept* 입니다. 실제 deployment 에는 controller robustness, "
        "sensor noise, vessel dynamics 의 uncertainty 등 추가 고려가 필요하고요, 이건 후속 "
        "연구나 학위논문에서 이어가겠습니다.\n\n"
        "*발표 tip* — 영상을 잠시 정지하고 청자가 *두 line 의 차이를 직접 눈으로* 보게 하면 "
        "효과적이에요. *눈으로 봐도 차이* 가 가장 직관적인 evidence 거든요."
    ),
    'S5_D': (
        "마지막으로 *향후 연구 방향 4 가지* 를 말씀드리겠습니다.\n\n"
        "*방향 1 — 고복잡도 파도 대응 (초기화 개선)* — 앞에서 봤듯이 15 개 이상 성분에서 "
        "정확도가 떨어져요. 두 가지 해결 방향이 있어요. *Hybrid init* 은 격자 방법으로 먼저 "
        "대략적인 분포를 잡고 그 결과를 파라미터 초기값으로 정밀화하는 거고, *Learned init* "
        "은 신경망을 활용해서 LiDAR 점으로부터 초기 성분을 직접 추정하는 거예요. 기대 — "
        "복잡 파도에서도 격자 방법 이상의 정확도.\n\n"
        "*방향 2 — 비선형 파도 확장* — 지금은 LWT 라서 BF6 까지만 신뢰할 수 있어요. *Stokes "
        "2 차 보정* 을 추가하면 파도의 비대칭성 (마루 뾰족, 골 평평) 을 일부 반영할 수 있고, "
        "Klein 2020 에 따르면 적용 범위가 크게 넓어집니다. 장기적으로는 NLSE 수준까지 도달 "
        "이 목표예요.\n\n"
        "*방향 3 — Vessel motion 기반 synthetic aperture* — 이게 흥미로운 방향이에요. 지금은 "
        "vessel 정지 가정이지만, V > 0 으로 움직이면 *V·T 만큼 추가 spatial coverage* 가 "
        "생깁니다. Wiley 1954 의 SAR (synthetic aperture radar) 원리를 ocean LiDAR 에 적용 "
        "하는 거죠. 우리가 *ship-mounted LiDAR 에서 SA 를 적용한 첫 사례* 가 될 가능성도 "
        "있어요.\n\n"
        "*방향 4 — 실측 데이터 검증* — 지금은 모두 합성 시뮬레이션이에요. *실제 선박에 "
        "LiDAR 를 달고* 측정한 데이터로 검증해야 합니다. 가장 큰 도전은 *ground-truth* 를 "
        "어떻게 확보할지인데, wave buoy 나 X-band radar 로 병행 측정하는 방법을 고려하고 "
        "있어요.\n\n"
        "*최종 목표* — 선박 LiDAR 로 주변 파도를 real-time 으로 추정하고, 그 정보를 "
        "hydrofoil ride control 이나 자율운항 같은 *선박 제어에 직접 활용할 수 있는 시스템* "
        "을 만드는 것입니다.\n\n"
        "이상으로 발표를 마치겠습니다. *질문이나 피드백 환영합니다.* 감사합니다."
    ),
}


CONTENT_SLIDES = [
    # Each spec: dict with title, body, optional fig (path relative to ROOT), notes
    # §1 Introduction & Motivation (3 content slides)
    {'title': '1. Introduction & Motivation', 'body': S1_A_USER, 'fig': None,
     'notes': NOTES['S1_A']},
    {'title': '1. Introduction & Motivation', 'body': S1_B_CHALLENGES,
     'fig': 'docs/sea_trial_lidar.png', 'notes': NOTES['S1_B']},
    {'title': '1. Introduction & Motivation', 'body': S1_C_SPOILER,
     'fig': 'docs/fig_3way_bar.png', 'notes': NOTES['S1_C']},

    # §2 Related work (4 slides — text-heavy, no figures)
    {'title': '2. Related work', 'body': S2_A_CATEGORIES, 'fig': None,
     'notes': NOTES['S2_A']},
    {'title': '2. Related work', 'body': S2_B_ENKF, 'fig': None,
     'notes': NOTES['S2_B']},
    {'title': '2. Related work', 'body': S2_C_PINN, 'fig': None,
     'notes': NOTES['S2_C']},
    {'title': '2. Related work',
     'body': [
         (0, '응용 매트릭스 — 본 연구의 위치'),
         (1, '6가지 응용 요구사항으로 본 연구와 5개 baseline 비교'),
     ],
     'fig': None,
     'table': {
         'headers': ['요구사항', 'Ours', 'PFT-PINN', 'VOF-EnKF', 'Wang2025', 'FNO', 'Desmars'],
         'rows': [
             ['실시간 제어 (<100 ms)', '✓', '✗', '✗', '✗', '✓', '✗'],
             ['2D 방향성 추정', '✓', '✗', '✗', '✗', '✗', '✗'],
             ['파도 파쇄 처리', '✗', '✗', '✓', '✗', '✗', '✗'],
             ['유속장 추정', '✗', '✓', '✓', '✗', '✗', '일부'],
             ['학습 데이터 불필요', '✓', '✓', '✓', '✓', '✗', '✓'],
             ['희소 LiDAR 모델', '✓', '✗', '✗', '✗', '✗', '일부'],
         ],
         'top': 2.5, 'left': 0.35, 'width': 9.3, 'height': 3.0,
         'highlight_col': 1, 'font_size': 9, 'body_height': 1.4,
     },
     'notes': NOTES['S2_D']},

    # §3 Method (8 slides)
    {'title': '3. Method', 'body': S3_A_WAVESIM,
     'fig': 'docs/wavesim_BF4_5c_wave_lidar.png', 'notes': NOTES['S3_A']},
    {'title': '3. Method', 'body': S3_B_LIDAR,
     'fig': 'docs/wavesim_BF3_1c_wave_lidar.png', 'notes': NOTES['S3_B']},
    {'title': '3. Method', 'body': S3_C_DATASET,
     'fig': 'docs/fig_demo_simple_2d3d.png', 'notes': NOTES['S3_C']},
    {'title': '3. Method', 'body': S3_D_OVERVIEW,
     'fig': 'paper/figures/fig_pipeline.png', 'notes': NOTES['S3_D']},
    {'title': '3. Method', 'body': S3_E_INIT1, 'fig': None,
     'notes': NOTES['S3_E']},
    {'title': '3. Method', 'body': S3_F_INIT2,
     'fig': 'paper/figures/fig_beamforming_2stage.png', 'notes': NOTES['S3_F']},
    {'title': '3. Method', 'body': S3_G_INIT3, 'fig': None,
     'notes': NOTES['S3_G']},
    {'title': '3. Method', 'body': S3_H_TRACK, 'fig': None,
     'notes': NOTES['S3_H']},

    # §4 Experiments & Mechanism (5 slides)
    {'title': '4. Experiments & Mechanism',
     'body': [
         (0, '12-episode benchmark 결과'),
         (1, 'WaveSplat 정확도 16% 우위 + 47배 빠름 (CPU only)'),
         (1, '단순 파도 (1–5 comp): 압도적 / 복잡 파도 (≥15): 일부 역전'),
     ],
     'fig': None,
     'table': {
         'headers': ['방법', 'nRMSE', '속도', '특징'],
         'rows': [
             ['WaveSplat (ours)', '0.211', '13 ms', '연속 파라미터 추정, real-time'],
             ['Wang2025 ETKF (Ne=300)', '0.253', '632 ms', '논문 충실 재현 (격자 기반)'],
             ['Ehlers 2023 FNO (참고)', '—', '1.9 ms', '학습 데이터 3120개 필요'],
             ['Ehlers 2025 PFT-PINN (참고)', '—', '~30 분', '학습 데이터 0개, instance 학습'],
             ['Yan 2025 VOF-EnKF (참고)', '—', '오프라인', 'wave breaking 처리'],
         ],
         'top': 3.2, 'left': 0.35, 'width': 9.3, 'height': 3.5,
         'highlight_col': 0, 'font_size': 10, 'body_height': 2.0,
     },
     'notes': NOTES['S4_A']},
    {'title': '4. Experiments & Mechanism', 'body': S4_B_SCATTER,
     'fig': 'docs/fig_complexity_scatter.png', 'notes': NOTES['S4_B']},
    {'title': '4. Experiments & Mechanism', 'body': S4_C_ABLATION,
     'fig': 'docs/fig_ablation_bar.png', 'notes': NOTES['S4_C']},
    {'title': '4. Experiments & Mechanism', 'body': S4_D_MECHANISM,
     'fig': 'docs/3dgs_analogy.png', 'notes': NOTES['S4_D']},
    {'title': '4. Experiments & Mechanism', 'body': S4_E_STAGES,
     'fig': 'paper/figures/fig_beamforming_2stage.png', 'notes': NOTES['S4_E']},

    # §5 Limitations & Research plan (4 slides)
    {'title': '5. Limitations & Research plan', 'body': S5_A_LIMITS, 'fig': None,
     'notes': NOTES['S5_A']},
    {'title': '5. Limitations & Research plan', 'body': S5_B_HYDROFOIL,
     'fig': 'docs/hydrofoil_controller_flow.png', 'notes': NOTES['S5_B']},
    {'title': '5. Limitations & Research plan', 'body': S5_C_HYDROFOIL_RESULTS,
     'fig': 'docs/hydrofoil_wave_animation.gif',
     'fig_layout': 'top_full',   # animation 풀-슬라이드 + 짧은 caption 아래
     'notes': NOTES['S5_C']},
    {'title': '5. Limitations & Research plan', 'body': S5_D_FUTURE, 'fig': None,
     'notes': NOTES['S5_D']},
]


# Section divider TOC content (5 dividers, 한 번만 만들고 재사용)
TOC_TEXT = [
    (0, 'Contents'),
    (0, '  1.\tIntroduction & Motivation'),
    (0, '  2.\tRelated work'),
    (0, '  3.\tMethod'),
    (0, '  4.\tExperiments & Mechanism'),
    (0, ' 5.\tLimitations & Research plan'),
]

SECTION_DIVIDER_INDICES = {  # before which content slide should divider go (0-indexed in CONTENT_SLIDES)
    0: '1',   # before §1
    3: '2',   # before §2 (after 3 §1 slides)
    7: '3',   # before §3 (after 4 §2 slides)
    15: '4',  # before §4 (after 8 §3 slides)
    20: '5',  # before §5 (after 5 §4 slides)
}


# ───────────────────────────────────────────────────────────
# Main builder
# ───────────────────────────────────────────────────────────

def get_layouts(prs):
    """Return (title_layout, content_layout)."""
    return prs.slide_layouts[0], prs.slide_layouts[1]


def find_body_placeholder(slide):
    """Largest text frame in lower half = body content."""
    candidates = []
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        l = shape.left / 914400 if shape.left else 0
        t = shape.top / 914400 if shape.top else 0
        h = shape.height / 914400 if shape.height else 0
        if t > 0.5 and h > 4:
            candidates.append((shape, h))
    if candidates:
        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0]
    return None


def find_title_placeholder(slide):
    """Top placeholder with height < 1.5 inch."""
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        t = shape.top / 914400 if shape.top else 0
        h = shape.height / 914400 if shape.height else 0
        w = shape.width / 914400 if shape.width else 0
        if t < 0.5 and h < 1.2 and w > 5:
            return shape
    return None


def find_date_placeholder(slide):
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        t = shape.top / 914400 if shape.top else 0
        if t > 7.0:
            return shape
    return None


# ── Compact font sizes per level (for tight layout) ──
LEVEL_SIZES = {0: 13, 1: 11, 2: 10}        # Pt
LEVEL_SPACE_AFTER = {0: 3, 1: 1, 2: 0}     # Pt (after each paragraph)


def fill_body_text(text_frame, items, compact=True):
    """Fill text frame with (level, text) paragraphs at compact sizes."""
    text_frame.word_wrap = True
    p0 = text_frame.paragraphs[0]
    p0.text = ''
    while len(text_frame.paragraphs) > 1:
        text_frame._txBody.remove(text_frame.paragraphs[-1]._p)
    for i, (level, text) in enumerate(items):
        if i == 0:
            p = p0
        else:
            p = text_frame.add_paragraph()
        p.text = text
        p.level = level
        if compact:
            sz = LEVEL_SIZES.get(level, 10)
            sp = LEVEL_SPACE_AFTER.get(level, 0)
            p.space_after = Pt(sp)
            for r in p.runs:
                r.font.size = Pt(sz)
                if level == 0:
                    r.font.bold = True


def resize_body_placeholder(slide, new_width_in=4.5, new_left_in=0.35):
    """Shrink the body placeholder to make room for a side figure.
    Must set ALL 4 dimensions (left/top/width/height) explicitly because
    python-pptx creates a fresh xfrm element on first modification, dropping
    inherited values for the ones not explicitly set."""
    body_ph = find_body_placeholder(slide)
    if body_ph is None:
        return
    cur_top = body_ph.top
    cur_height = body_ph.height
    body_ph.left = Inches(new_left_in)
    body_ph.top = cur_top
    body_ph.width = Inches(new_width_in)
    body_ph.height = cur_height


def add_side_figure(slide, fig_path, left_in=5.0, top_in=1.2, width_in=4.7, height_in=5.4):
    """Add figure on right side of slide. Auto fits within bounds preserving aspect."""
    if not os.path.isabs(fig_path):
        fig_path = os.path.join(ROOT, fig_path)
    if not os.path.exists(fig_path):
        print(f'    [warn] figure not found: {fig_path}')
        return False
    # Determine native aspect, scale to fit box
    from PIL import Image
    try:
        with Image.open(fig_path) as im:
            iw, ih = im.size
        ratio = iw / ih
        box_ratio = width_in / height_in
        if ratio > box_ratio:
            # image wider than box — fit width
            w = width_in
            h = w / ratio
        else:
            h = height_in
            w = h * ratio
        # center within box
        cx = left_in + (width_in - w) / 2
        cy = top_in + (height_in - h) / 2
        slide.shapes.add_picture(fig_path, Inches(cx), Inches(cy),
                                 width=Inches(w), height=Inches(h))
        return True
    except Exception as e:
        print(f'    [warn] PIL error: {e}, falling back to fixed size')
        slide.shapes.add_picture(fig_path, Inches(left_in), Inches(top_in),
                                 width=Inches(width_in))
        return True


def add_inline_table(slide, headers, rows, left_in, top_in, width_in, height_in,
                     header_color=None, font_size=9, highlight_col=None):
    """Add a table to slide at given position. Compact font for body."""
    if header_color is None:
        header_color = RGBColor(0x00, 0x7A, 0xCC)
    n_rows = len(rows) + 1
    n_cols = len(headers)
    table_shape = slide.shapes.add_table(
        n_rows, n_cols,
        Inches(left_in), Inches(top_in),
        Inches(width_in), Inches(height_in)
    )
    table = table_shape.table
    for c, h in enumerate(headers):
        cell = table.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_color
        tf = cell.text_frame
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = str(h)
        run.font.size = Pt(font_size + 1)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = table.cell(r + 1, c)
            cell.fill.solid()
            if highlight_col is not None and c == highlight_col:
                cell.fill.fore_color.rgb = RGBColor(0xFF, 0xF2, 0xCC)
            else:
                cell.fill.fore_color.rgb = RGBColor(0xF8, 0xF8, 0xF8) if r % 2 == 0 \
                                          else RGBColor(0xFF, 0xFF, 0xFF)
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT
            run = p.add_run()
            run.text = str(val)
            run.font.size = Pt(font_size)
    return table


def add_presenter_notes(slide, text):
    """Add presenter notes (발표자 노트 / 프롬프트) to a slide."""
    if not text:
        return
    notes_slide = slide.notes_slide
    tf = notes_slide.notes_text_frame
    tf.text = text


def fill_content_slide(slide, section_title, body_items, fig=None, table=None,
                       fig_layout='split', notes=None,
                       date_text='May 13, 2026'):
    """spec: section_title, body bullets, optional fig (path) or table (dict).

    fig_layout:
      - 'split' (default): body left ~4.5", figure right ~4.7"
      - 'top_full': figure top ~5", body brief at bottom (for animation/large fig)
    """
    title_ph = find_title_placeholder(slide)
    if title_ph is not None:
        title_ph.text_frame.paragraphs[0].text = section_title

    if table is not None:
        # Layout: short body bullets at top, table below
        body_ph = find_body_placeholder(slide)
        if body_ph is not None:
            # shrink to top portion only
            cur_left = body_ph.left
            cur_width = body_ph.width
            body_ph.left = cur_left
            body_ph.top = Inches(1.07)
            body_ph.width = cur_width
            body_ph.height = Inches(table.get('body_height', 1.5))
            fill_body_text(body_ph.text_frame, body_items, compact=True)
        add_inline_table(
            slide,
            headers=table['headers'],
            rows=table['rows'],
            left_in=table.get('left', 0.35),
            top_in=table.get('top', 2.7),
            width_in=table.get('width', 9.3),
            height_in=table.get('height', 4.0),
            font_size=table.get('font_size', 9),
            highlight_col=table.get('highlight_col'),
        )
    elif fig:
        if fig_layout == 'top_full':
            # Figure occupies most of slide (top), brief text at bottom
            body_ph = find_body_placeholder(slide)
            if body_ph is not None:
                cur_left = body_ph.left
                cur_width = body_ph.width
                body_ph.left = cur_left
                body_ph.top = Inches(5.7)
                body_ph.width = cur_width
                body_ph.height = Inches(1.4)
                fill_body_text(body_ph.text_frame, body_items, compact=True)
            # Large figure on top — preserve aspect, fit width
            add_side_figure(slide, fig, left_in=0.5, top_in=1.0,
                            width_in=9.0, height_in=4.5)
        else:
            # Split layout: body left half, figure right half
            resize_body_placeholder(slide, new_width_in=4.5, new_left_in=0.35)
            body_ph = find_body_placeholder(slide)
            if body_ph is not None:
                fill_body_text(body_ph.text_frame, body_items, compact=True)
            add_side_figure(slide, fig, left_in=5.0, top_in=1.2,
                            width_in=4.7, height_in=5.4)
    else:
        body_ph = find_body_placeholder(slide)
        if body_ph is not None:
            fill_body_text(body_ph.text_frame, body_items, compact=True)

    date_ph = find_date_placeholder(slide)
    if date_ph is not None:
        for p in date_ph.text_frame.paragraphs:
            for r in p.runs:
                if r.text and ('April' in r.text or '2023' in r.text or '2026' in r.text):
                    r.text = date_text

    if notes:
        add_presenter_notes(slide, notes)


def fill_divider_slide(slide, section_num, date_text='May 13, 2026'):
    """Section divider slide — TOC with current section highlighted."""
    body_ph = find_body_placeholder(slide)
    if body_ph is not None:
        fill_body_text(body_ph.text_frame, TOC_TEXT)
    date_ph = find_date_placeholder(slide)
    if date_ph is not None:
        for p in date_ph.text_frame.paragraphs:
            for r in p.runs:
                if r.text and 'April' in r.text:
                    r.text = date_text


def delete_slide(prs, slide_idx):
    """Remove slide at given index."""
    xml_slides = prs.slides._sldIdLst
    slides_list = list(xml_slides)
    if slide_idx >= len(slides_list):
        return
    sldId = slides_list[slide_idx]
    rId = sldId.rId
    try:
        prs.part.drop_rel(rId)
    except Exception:
        pass
    xml_slides.remove(sldId)


def move_slide(prs, src_idx, dst_idx):
    xml_slides = prs.slides._sldIdLst
    slides_list = list(xml_slides)
    sldId = slides_list[src_idx]
    xml_slides.remove(sldId)
    xml_slides.insert(dst_idx, sldId)


def main():
    # Strategy: linear build to avoid reorder/duplicate issues
    #   1. Open template, delete content slides 2-12 → 8 slides remain
    #      (title, TOC, 감사, app×5 in that order)
    #   2. Move 감사 + appendix to END (after deletion they're at idx 2-7)
    #      We'll do this by deleting them FIRST, saving content,
    #      and re-adding them at end after new slides.
    #   3. Add new content slides linearly — they'll naturally be in middle.

    shutil.copy(SRC, DST)
    prs = Presentation(DST)
    title_layout, content_layout = get_layouts(prs)

    # Step 1: Delete all slides EXCEPT title (idx 0) and TOC (idx 1)
    # This includes 감사 and appendix — we'll recreate them from saved content.
    print('[1] Saving 감사 + appendix content for later re-creation...')
    # 감사 = slide 13 (0-indexed)
    # appendix A,B = slides 14, 15 (with tables)
    # appendix C×3 = slides 16, 17, 18 (with pictures)
    thanks_text = []
    for shape in prs.slides[13].shapes:
        if shape.has_text_frame and shape.text_frame.text.strip():
            thanks_text.append(shape.text_frame.text)

    print('[2] Deleting all template content slides (indices 2-18)...')
    # Delete indices 2..18 (17 slides)
    for _ in range(17):
        delete_slide(prs, 2)
    # Now: slides 0=title, 1=TOC

    # Step 3: Add all new content + divider slides linearly
    print('[3] Adding new content slides...')
    for content_idx, spec in enumerate(CONTENT_SLIDES):
        if content_idx in SECTION_DIVIDER_INDICES:
            section_num = SECTION_DIVIDER_INDICES[content_idx]
            divider_slide = prs.slides.add_slide(content_layout)
            fill_divider_slide(divider_slide, section_num)

        section_title = spec['title']
        body_items = spec['body']
        fig_path = spec.get('fig')
        table_spec = spec.get('table')
        fig_layout = spec.get('fig_layout', 'split')
        notes = spec.get('notes')
        new_slide = prs.slides.add_slide(content_layout)
        fill_content_slide(new_slide, section_title, body_items,
                           fig=fig_path, table=table_spec,
                           fig_layout=fig_layout, notes=notes)
        marker = ''
        if fig_path:
            marker = f'  📷 {os.path.basename(fig_path)}'
        if table_spec:
            marker = f'  ▦ {len(table_spec["rows"])}-row table'
        print(f'    + {section_title}: {body_items[0][1][:36]}{marker}')

    # Step 4: Re-add 감사합니다 slide (use title layout)
    print('[4] Adding 감사합니다 slide...')
    thanks_slide = prs.slides.add_slide(title_layout)
    # Title layout has CENTER_TITLE placeholder
    if thanks_slide.placeholders:
        thanks_slide.placeholders[0].text = '감사합니다.'

    # Step 5: Re-add appendix slides
    print('[5] Adding Appendix slides...')
    add_appendix_a_slide(prs, content_layout)
    add_appendix_b_slide(prs, content_layout)
    add_appendix_c_slides(prs, content_layout)

    # Step 6: update dates on all slides
    print('[6] Updating dates...')
    for slide in prs.slides:
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for p in shape.text_frame.paragraphs:
                for r in p.runs:
                    if r.text and 'April xx' in r.text:
                        r.text = 'May 13, 2026'

    prs.save(DST)
    print(f'\nSaved: {DST}')
    print(f'Total slides: {len(prs.slides)}')


# ── Appendix slide builders ──

APPENDIX_A_HEADERS = ['#', 'Paper (Year)', 'Sensor / Data', '핵심 방법', '한계 / 우리 차별']
APPENDIX_A_ROWS = [
    ['1', 'Desmars 2018 (OMAE)', 'LIDAR (proposed)\nSim only (HOS synth)',
     'LIDAR + ICWM, on-grid variational', 'LIDAR concept, 1D'],
    ['2', 'Desmars 2020 (Ocean Eng 212)', 'Wave gauge array\n실측 (wave tank)',
     'Wave gauge + dispersion correction', 'Wave gauge ≠ LiDAR, 1D'],
    ['3', 'Desmars 2023 (JFM 975)', 'X-band radar (modeled)\nSim only + wave tank',
     'HOS-OCM, vessel optical, 3-stage', '1D, on-grid'],
    ['4', 'Wang & Pan 2021 (JFM 918)', 'X-band radar\nSim + 실측 radar',
     'EnKF + HOS 첫 결합', 'Dense obs, 1D'],
    ['5', 'Wang & Pan 2022 (JFM 949)', 'Shipborne radar\n실측 + sim',
     'EnKF-HOS-C (current 동시)', '1D, dense obs'],
    ['6', 'Feng Wang 2025\n(Coastal Eng 197)', 'Stereo camera (WASS)\n실측 (Acqua Alta)',
     'ETKF + Ne=300 + fuzzy R', '격자 한계, 632 ms'],
    ['7', 'Hunt 2007 (Physica D 230)', '기상학\nSim (atmospheric)',
     'LETKF', 'Wave-specific 아님'],
    ['8', 'Klein 2020 (Fluids 5)', 'Model 비교\nSim + semi-exp',
     'LWT/NLSE/HOSM 정량 비교', '새 알고리즘 아님'],
    ['9', 'Ehlers 2023 (OE 288)', 'X-band radar\nSim only (HOS+geom radar)',
     'FNO supervised, 1.9 ms', '3120 sample 학습 필요, 1D'],
    ['10', 'Ehlers 2025 (arXiv:2501.08430)', 'Wave gauge\n실측 (wave flume)',
     'PFT-PINN unsupervised', '30 분/instance, 1D + t'],
    ['11', 'Yan 2025 (arXiv:2510.14356)', 'Synthetic\nSim only (VOF NWT)',
     'NS + EnKF + POD', 'Offline only, 1D xz'],
    ['12', 'Tang 2013 (IEEE TIT 59)', '이론\n— (theoretical)',
     'Atomic norm off-grid CS', '1D, SDP 비용'],
    ['13', 'Möller-Trumbore 1997 (JGT)', 'Graphics\n— (algorithm only)',
     'Ray-triangle intersection', '비물리적 (geometry only)'],
]

APPENDIX_B_HEADERS = ['Sea state', 'ε (steepness)', 'N_components', 'Seeds', 'Episode 수']
APPENDIX_B_ROWS = [
    ['BF2 (Hs=0.3, Tp=4.0)', '0.012', '1', '0–4', '10'],
    ['BF3 (Hs=0.6, Tp=5.5)', '0.029', '1, 2, 5', '0–4', '30'],
    ['BF4 (Hs=1.2, Tp=7.0)', '0.045', '1, 5, 15, 30', '0–4', '40'],
    ['BF5 (Hs=2.0, Tp=9.0)', '0.062', '15, 30', '0–4', '20'],
    ['BF6 (Hs=3.0, Tp=10.0)', '0.079', '1, 5, 15, 30', '0', '4'],
    ['BF7 (Hs=5.0, Tp=12.0)', '0.094', '1, 5, 15, 30', '0', '4'],
    ['합계', '—', '—', '—', '110 episodes (1.2 GB)'],
]


def add_appendix_a_slide(prs, content_layout):
    slide = prs.slides.add_slide(content_layout)
    title_ph = find_title_placeholder(slide)
    if title_ph:
        title_ph.text_frame.paragraphs[0].text = 'Appendix A. Related work table'
    # Add table
    rows = len(APPENDIX_A_ROWS) + 1
    cols = len(APPENDIX_A_HEADERS)
    table_shape = slide.shapes.add_table(
        rows, cols,
        Inches(0.32 * (10/13.333)), Inches(1.05),
        Inches(9.13 * (10/13.333)), Inches(5.85)
    )
    table = table_shape.table
    for c, h in enumerate(APPENDIX_A_HEADERS):
        cell = table.cell(0, c)
        cell.text_frame.paragraphs[0].text = h
        cell.text_frame.paragraphs[0].runs[0].font.bold = True
    for r, row in enumerate(APPENDIX_A_ROWS):
        for c, val in enumerate(row):
            cell = table.cell(r + 1, c)
            tf = cell.text_frame
            tf.word_wrap = True
            lines = val.split('\n')
            tf.paragraphs[0].text = lines[0]
            for line in lines[1:]:
                p = tf.add_paragraph()
                p.text = line


def add_appendix_b_slide(prs, content_layout):
    slide = prs.slides.add_slide(content_layout)
    title_ph = find_title_placeholder(slide)
    if title_ph:
        title_ph.text_frame.paragraphs[0].text = 'Appendix B. Dataset table'
    rows = len(APPENDIX_B_ROWS) + 1
    cols = len(APPENDIX_B_HEADERS)
    table_shape = slide.shapes.add_table(
        rows, cols,
        Inches(0.33 * (10/13.333)), Inches(2.32),
        Inches(9.0 * (10/13.333)), Inches(3.0)
    )
    table = table_shape.table
    for c, h in enumerate(APPENDIX_B_HEADERS):
        cell = table.cell(0, c)
        cell.text_frame.paragraphs[0].text = h
        cell.text_frame.paragraphs[0].runs[0].font.bold = True
    for r, row in enumerate(APPENDIX_B_ROWS):
        for c, val in enumerate(row):
            cell = table.cell(r + 1, c)
            cell.text_frame.paragraphs[0].text = val


def add_appendix_c_slides(prs, content_layout):
    fig_files = [
        ('fig_wang_compare_bf3_n2.png', 'Appendix C-1. BF3 / N=2'),
        ('fig_wang_compare_bf3_n5.png', 'Appendix C-2. BF3 / N=5'),
        ('fig_wang_compare_bf4_n5.png', 'Appendix C-3. BF4 / N=5'),
    ]
    for fig_name, title in fig_files:
        fig_path = os.path.join(DOCS_DIR, fig_name)
        slide = prs.slides.add_slide(content_layout)
        title_ph = find_title_placeholder(slide)
        if title_ph:
            title_ph.text_frame.paragraphs[0].text = title
        if os.path.exists(fig_path):
            slide.shapes.add_picture(
                fig_path,
                Inches(0.5), Inches(1.2),
                width=Inches(9.0), height=Inches(5.5)
            )




if __name__ == '__main__':
    main()
