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
        "파 추정의 한계부터 짚어드립니다. 현재 표준은 통계적 추정 — 유의파고, 첨두주기, "
        "JONSWAP 스펙트럼 — 으로 평균적 sea state 만 알 수 있습니다. 선박 설계나 해양 "
        "구조물 설계에는 충분하지만, 실시간 제어처럼 *언제 어디서 큰 파도가 오는가* 를 "
        "정확히 알아야 하는 응용에는 부족합니다.\n\n"
        "그래서 본 연구는 phase-resolved estimation — 파도장 η(x, y, t) 를 직접 추정 — "
        "을 목표로 합니다. 평균 정보가 아니라 특정 위치, 특정 시점에 어떤 파도가 오는지 "
        "구체적으로 답하는 것이죠. 응용 영역은 hydrofoil ride control, 자율운항 안전 "
        "판단, 선박 motion 예측 등 다양합니다."
    ),
    'S1_B': (
        "본 task 가 어려운 이유 세 가지로 정리합니다.\n\n"
        "첫째 *희소·비균일 측정* — LiDAR 는 frame 당 valid 점이 약 50 개 정도로 매우 "
        "희소하고, raydrop·shadowing 으로 점이 누락되며 거리에 따라 점밀도가 달라집니다. "
        "기존 wave gauge·dense radar 가정과 다른 setup 이라 직접 적용이 어렵습니다.\n\n"
        "둘째 *실시간 제약* — 제어 루프 10 Hz 이므로 frame 당 100 ms 이내 처리가 필요한데, "
        "선행 방법들은 모두 이를 초과합니다. Wang & Pan 의 EnKF + HOS 는 비선형 wave model "
        "이라 수백 ms 이상, Feng Wang 의 ETKF + linear 도 Ne=300 으로 632 ms, PFT-PINN 은 "
        "아예 instance 마다 30 분 학습이 필요합니다. FNO 만 1.9 ms 로 빠르지만 학습 데이터 "
        "3120 개를 미리 만들어야 합니다.\n\n"
        "셋째 *2D directional* — 실해 파도는 한 방향이 아니라 wind sea 와 다방향 swell 이 "
        "섞여 있어서 wavenumber 도 vector (k_x, k_y) 추정이 필요합니다. 선행 연구 다수가 "
        "1D 로 단순화하지만 본 연구는 2D 그대로 처리합니다."
    ),
    'S1_C': (
        "배경 설명에 들어가기 전에 결과부터 보여드립니다 — 왜 이 연구를 계속할 가치가 있는가 "
        "를 숫자로 먼저 제시.\n\n"
        "12 episode benchmark (BF2~BF5 sweep) 평균 결과: WaveSplat 0.211 nRMSE / 13.4 ms, "
        "Feng Wang 2025 baseline (Ne=300, fuzzy R) 0.253 / 632 ms. 정확도 16% 우위 + 속도 "
        "47 배 빠름, CPU only 입니다. 13 ms 는 초당 약 75 frame 이라 10 Hz 제어 루프에 "
        "충분히 적합합니다.\n\n"
        "이게 가능한 핵심 메커니즘은 두 가지 — 분산 관계 hard constraint 가 자유도를 5N → "
        "4N 으로 20% 줄이면서 물리적으로 불가능한 답을 자동 배제하는 정규화 역할을 하고, "
        "multi-frame temporal coherence 가 wave 자체의 시간 진화로 phase 정보를 추출합니다. "
        "두 메커니즘이 결합되어야 spatial Nyquist 0.4 rad/m 한계를 우회할 수 있습니다.\n\n"
        "응용 사례 — hydrofoil ride control 시험에서 heave aRMS 0.090 → 0.047 m/s² (약 "
        "48% 감소) 까지 확인했습니다. 이제부터 이 결과가 어떻게 나왔는지, 선행 연구와 "
        "어떻게 다른지를 차례로 설명드립니다."
    ),
    'S2_A': (
        "13 편 prior art 를 정독해 5 가지 접근법으로 분류했습니다.\n\n"
        "① 격자 기반 최적화 (Desmars family) — pre-defined wavenumber grid 위 비용함수 "
        "최소화. 수치해석 분야에서 framework 가 잘 정립돼 안정적이지만, 격자 사이 "
        "wavenumber 표현 불가 (Nyquist 한계) + 대부분 1D 가정.\n\n"
        "② 앙상블 칼만 필터 (Wang & Pan, Feng Wang, Yan VOF) — 50–300 개 wave simulation "
        "을 동시 진행하며 측정값으로 베이지안 update. Wang-Pan 은 HOS 비선형, Feng Wang 은 "
        "linear + fuzzy R, Yan 은 Navier-Stokes 까지 써서 wave breaking 처리. 강점은 물리 "
        "충실도, 한계는 ensemble 비용으로 real-time 불가.\n\n"
        "③ 지도학습 (Ehlers FNO) — 학습 후 inference 1.9 ms 매우 빠름. 단 HOS 합성 데이터 "
        "3120 개 학습 필요 + distribution shift + 1D + black-box.\n\n"
        "④ 비지도 PINN (Ehlers PFT-PINN) — 학습 데이터 0 개로 비선형 처리. 단 instance "
        "마다 30 분 학습이라 real-time 불가.\n\n"
        "⑤ 본 연구 WaveSplat — 격자 없이 N 개 plane wave 파라미터 직접 추정. 100 ms 이내 "
        "+ 2D directional + LiDAR sensor model 까지 갖춘 첫 사례. 한계는 LWT linear "
        "superposition 가정, ε < 0.07 영역에서만 신뢰. 자세한 paper 정보는 Appendix A 참조."
    ),
    'S2_B': (
        "EnKF 기반 wave estimation 의 두 독립 그룹을 자세히 소개합니다.\n\n"
        "*Wang & Pan group* (Yulin Pan, U.Michigan) 은 2021 년 EnKF + HOS 첫 결합으로 "
        "phase-resolved 의 시작을 열었고, 2022 년 IEnKF-HOS-C 로 wave + current 동시 추정 "
        "까지 확장했습니다. Sensor 는 X-band radar, 데이터는 합성 + 실측 shipborne radar.\n\n"
        "*Feng Wang et al.* (Harbin Engineering Univ., 중국) 은 2025 년 ETKF + fuzzy R + "
        "Delaunay 보간 — 우리 baseline 의 직접 prior art 입니다. Ne=300, 80×60 격자, "
        "dx ≈ 3 m. Sensor 는 stereo camera (WASS), 데이터는 Acqua Alta 해양 플랫폼 + 자체 "
        "2024 년 vessel keeping 측정. GitHub 까지 공개되어 있습니다.\n\n"
        "본 연구는 Feng Wang 의 *알고리즘만* 재현하고 입력 sensor 와 forecast model 을 "
        "공정 비교를 위해 통일했습니다 — 입력은 LiDAR sim, forecast 는 HOS 대신 linear "
        "dispersion. EnKF 류의 공통 한계는 격자 해상도 (≈ 0.25 nRMSE floor) 와 ensemble "
        "비용 (632 ms / frame, real-time 불가) 입니다."
    ),
    'S2_C': (
        "학습 기반 접근의 두 paper — Ehlers group 의 supervised 와 unsupervised 진화 사례.\n\n"
        "*Ehlers 2023* 의 FNO 는 ML supervised SOTA 입니다. Fourier Neural Operator 가 "
        "Fourier 공간에서 global mapping 을 학습해 U-Net 같은 지역 CNN 보다 wave 패턴에 "
        "잘 맞습니다. 학습 후 inference 1.9 ms — 가장 빠름. 단 HOS 합성 데이터 3120 개 "
        "학습이 필요하고 distribution shift, 1D, black-box 한계가 있습니다.\n\n"
        "*Ehlers 2025* 의 PFT-PINN 은 우리에게 *가장 위협적인* 선행 연구입니다. 이유는 "
        "근본 철학이 같기 때문 — 학습 데이터 없이 physics constraint 를 직접 강제. 두 NN "
        "병렬 (η-net + Φ-net) + Laplace + 표면 BC + 바닥 BC 의 PDE residual 강제 + "
        "hard-constraint trick. MARATHON wave flume 실측 검증, SSP 0.049–0.084. 한계는 "
        "instance 마다 30 분 학습 (RTX 3090) → real-time 불가, 1D + t, buoy only.\n\n"
        "본 연구와의 차이는 속도 — 13 ms vs 30 분 (수만 배 차이). 응용 영역도 다름 — "
        "WaveSplat = real-time control, PINN = design-time analysis. 다만 같은 group 이 "
        "1-3 년 안에 2D + real-time PINN 발표 가능 — niche 좁아질 risk 도 있습니다."
    ),
    'S2_D': (
        "응용 매트릭스로 본 연구의 정확한 위치를 확정합니다.\n\n"
        "행은 6 가지 응용 요구사항, 열은 본 연구와 5 개 baseline. 핵심 관찰은 — *어떤 "
        "방법도 모든 항목에 ✓ 를 받지는 못한다*. 각 방법마다 고유한 강점이 있고 적합한 "
        "응용 영역이 다릅니다.\n\n"
        "본 연구의 고유 위치는 *real-time + 2D directional + sparse LiDAR + 학습 데이터 "
        "0개* 의 4 가지 동시 만족 — 이 조합을 만족하는 선행 연구는 없습니다.\n\n"
        "응용 영역을 둘로 나누면, *real-time control 영역* 에는 WaveSplat, FNO 가 들어가고 "
        "(즉시 결정), *design-time analysis 영역* 에는 VOF-EnKF, PFT-PINN 이 들어갑니다 "
        "(시간 충분, 정확도 추구). 두 영역은 경쟁이 아닌 보완 관계.\n\n"
        "정직한 한계도 표에 보입니다 — 파도 파쇄는 VOF-EnKF 만, 유속장은 PINN/EnKF 영역. "
        "본 연구는 응용 영역 자체가 다릅니다."
    ),
    'S3_A': (
        "WaveSim 시뮬레이터의 4 가지 구성 — 파도 모델, 스펙트럼, 분산 관계, LiDAR 센서.\n\n"
        "*파도 모델* 은 Gerstner family. 단순 sinusoidal LWT 는 수면 변위만 표현하지만, "
        "Gerstner 는 입자 운동까지 모사 — 입자가 원/타원 궤적으로 회전하며 표면을 형성합니다. "
        "Q ∈ [0, 1] 가파름 척도로 Q=0 이면 LWT, Q>0 이면 약비선형 (crest sharp, trough flat). "
        "Q 제약 ΣQ·ω·A ≤ 1 은 입자 궤적 자체 교차 방지 (Tessendorf 2001).\n\n"
        "*스펙트럼* 은 JONSWAP — Hasselmann 1973 이 북해 측정으로 정립한 표준. 입력 H_s, T_p "
        "에서 N 개 성분의 (A, ω) sampling. 방향 spreading 으로 (k_x, k_y) 분배. ITTC, IACS "
        "모두 채택한 표준이라 합당성은 검증됨.\n\n"
        "*분산 관계* 는 Airy 1845 — ω = √(g·|k|·tanh(|k|·d)). 100 년 이상 검증된 표준. "
        "깊은 물에서 ω ≈ √(g·|k|) 로 단순화."
    ),
    'S3_B': (
        "LiDAR 센서 모사 — Falcon K1 spec 기준입니다.\n\n"
        "*Sensor 사양* — 152 채널 × 500 horizontal beam = 76,000 ray / scan, range 10–250 m, "
        "파장 1550 nm, vessel 위 5 m mount. HFOV 120° 는 선수 ±60° 반구 (marine LiDAR 의 "
        "일반 setup), VFOV −25° ~ 0° 로 surface 측정용 down-looking.\n\n"
        "*Ray tracing* 은 Möller-Trumbore 1997 의 graphics 표준 알고리즘 — 각 ray 가 wave "
        "표면 mesh 와 만나는 first-hit 점 검출.\n\n"
        "*Raydrop 모델* 은 Fresnel + geometry 결합. 충분한 비율의 빛이 LiDAR receiver 방향 "
        "으로 돌아와야 점이 찍힙니다. Snell 굴절 (n=1.321 water at 1550 nm) → Fresnel s/p "
        "polarization 평균 → Lambert + Fresnel 결합으로 backscatter probability 계산.\n\n"
        "중요한 정정 — 본 dataset 은 vessel 정지 (V=0 kts) 로 합성. 모든 frame 동일 위치 "
        "에서 측정이고, multi-frame 효과는 wave 자체의 시간 진화에서 옵니다. Vessel motion "
        "기반 synthetic aperture (SAR-style) 는 향후 연구."
    ),
    'S3_C': (
        "합성 dataset 의 구성과 정량입니다.\n\n"
        "BF2 (Hs=0.3, Tp=4) 부터 BF7 (Hs=5.0, Tp=12) 까지 6 개 sea state, N components 1 ~ "
        "30 (단순 ~ 복잡), seed 0–4 재현 가능, raydrop uniform/slope 두 가지. 총 110 episode.\n\n"
        "Per episode — 30 초 시뮬레이션, dt 0.1 s (10 Hz) 로 frame 301 개. GT η field 는 "
        "80×60 격자, 240 m × 200 m 영역. LiDAR scan 은 76,000 ray 발사 후 raydrop 적용해 "
        "valid 점 약 50 개 / frame. 실측해보면 patch coverage 50–150 m 정도.\n\n"
        "전체 용량 1.2 GB. JONSWAP 으로 BF3-BF6 의 wavenumber 분포를 보면 λ_peak 이 BF3 "
        "47 m, BF4 77 m, BF5 126 m, BF6 156 m — 대부분 patch 안에 1–3 wavelength 가 들어와 "
        "direction 추정에 충분합니다. BF6 는 전부 Nyquist 이내."
    ),
    'S3_D': (
        "WaveSplat 알고리즘의 근본 발상 — 격자 대신 파라미터로 직접 표현.\n\n"
        "JONSWAP 분해를 보면 실해 파도는 보통 5–30 개 sine wave 의 합으로 충분히 표현됩니다. "
        "각 성분은 5 개 파라미터 (A, k_x, k_y, ω, φ) 로 완전 결정. N=10 이면 변수 50 개 — "
        "LiDAR 50 점으로 1:1 추정 가능합니다. 이게 격자 4,800 개와의 큰 차이.\n\n"
        "Wave field 표현은 η = Σ A_n · cos(k_n·x − ω_n·t + φ_n). 분산 관계 hard constraint "
        "강제 — ω(k) = √(g·|k|·tanh(|k|·d)) 로 ω 가 k 에서 자동 결정. 자유도 5N → 4N "
        "(20% 감소) + 물리적으로 불가능한 (k, ω) 조합 자동 배제.\n\n"
        "2 단계 흐름 — Init (1 회, 1 초 이내) 은 beamforming + coarse-to-fine + Adam 정밀화로 "
        "(k_x, k_y) 결정. Track (매 frame, 13 ms) 은 k 고정하고 (A, φ) 만 update. 무거운 "
        "작업은 처음 한 번만, 가벼운 작업은 매 frame 실시간."
    ),
    'S3_E': (
        "Init 의 핵심 도전 — 왜 multi-frame 이 필요한가.\n\n"
        "Spatial Nyquist 한계 (Shannon 1949) — 점 간격 Δy 면 추정 가능한 최대 wavenumber 가 "
        "π/Δy. 우리 LiDAR 의 점 간격이 5–8 m 이라 single-frame 으로 추정 가능한 k 는 "
        "약 0.4 rad/m 까지.\n\n"
        "그런데 추정해야 할 wavenumber 범위를 보면 — BF3 의 짧은 wave 성분이 Nyquist 한계의 "
        "약 3 배까지 도달. Paradox 입니다. 단 BF5-BF6 dominant wave 는 longer wave 라 대부분 "
        "Nyquist 이내 — BF3 short tail 만 문제.\n\n"
        "해결 — *temporal coherence*. Vessel 이 정지해 있어도 wave 자체가 시간에 따라 "
        "propagate 하므로 multi-frame 효과가 자연스럽게 옵니다. η(x, t) = A·cos(k·x − ω·t "
        "+ φ) 의 t 진행이 phase 변화를 만들고, 여러 frame 의 (x, y, t, η) 4D 데이터 + "
        "dispersion 의 ω↔k 매핑이 결합되면 spatial Nyquist 우회 가능.\n\n"
        "Init window 길이는 sea state 별 약 10–20 초 (2·T_p), 수천 개 4D 점이 누적됩니다."
    ),
    'S3_F': (
        "Init 의 세 단계 — beamforming + coarse-to-fine + Adam refinement.\n\n"
        "*Beamforming score* 는 matched filter 형태 — S(k) = | Σ η_obs · exp(−j(k·x − ω(k)·t)) |². "
        "ω(k) 는 dispersion 으로 자동이라 2D search (k_x, k_y) 만 필요. Search range "
        "k_x, k_y ∈ [-4, 4], 81×81 격자에서 수천 개 후보 평가.\n\n"
        "*Coarse-to-fine schedule* — 3 stage 의 N 점진 증가 (3 → 8 → 15). Loss threshold 충족 "
        "시 다음 stage advance. 빠르게 수렴하는 stage 는 즉시 진행해서 전체 시간 단축.\n\n"
        "*Adam refinement* — beamforming 결과는 *초기값* 일 뿐. 그 후 multi-frame init "
        "window 전체에 fitting 합니다. (k_x, k_y, A, φ) 동시 update, 단 (k_x, k_y) 는 lr 의 "
        "절반 (0.005) 으로 천천히 — k 변화에 신중. Loss 는 data fit + L1 sparsity + spectral "
        "smoothness 3 항.\n\n"
        "주요 hyperparameter — λ_L1 = 0.001, λ_smooth = 0.0001, max_steepness = 0.15."
    ),
    'S3_G': (
        "Beamforming score 식이 *왜* 이 형태인지 유도.\n\n"
        "*핵심 직관* — matched filter 의 코러스 효과. 실제 wave 가 (k, ω) 로 진행하면 측정 "
        "η_obs 에 phase factor exp(−j(k·x − ω·t)) 곱했을 때 모든 점에서 phase 가 정렬됩니다. "
        "맞는 (k, ω): 합 절댓값 N 배 증가 → power N². 틀린 (k, ω): random phase 라 √N 으로만 "
        "증가 → power N. 결과: SNR gain = N²/N = N (점 수에 비례). Init window 의 수천 4D 점 "
        "이면 충분한 dB gain (약 10·log10(N)) 확보.\n\n"
        "*왜 표준 형태인가* — Delay-and-sum beamforming (Kraus 1988, Van Trees 2002) 이 "
        "신호처리의 standard, Matched filter 정리 (Turin 1960) 가 Gaussian noise 에서 SNR "
        "최대화하는 optimal filter 임을 보장. 단순 robust + covariance 추정 불필요라 sparse "
        "에 적합.\n\n"
        "*Dispersion 의 차원 축소 효과* — 없으면 (k_x, k_y, ω) 3D search → 후보 수십만 개 "
        "(계산 비현실적). 강제하면 2D search → 수천 개로 축소. 80 배 정도 절약.\n\n"
        "다른 spectral 방법 (MUSIC, ESPRIT, Capon/MVDR) 도 있지만 covariance matrix 필요 "
        "또는 ill-conditioning 위험으로 sparse 점에는 부적합."
    ),
    'S3_H': (
        "Track 단계와 Adaptive N — 매 frame 13 ms 의 핵심.\n\n"
        "*w_phi tracking mode* — k 는 Init 결과로 *고정* 합니다. 이게 WaveSplat 의 가장 "
        "중요한 설계 결정. Single frame 에서 k 의 FIM eigenvalue 가 (A, φ) 보다 ~10⁻³ 작아 "
        "k 자유 시 50 점에 over-fit 해서 비물리적 발산. ω 는 dispersion 으로 자동, (A, φ) "
        "만 매 frame Adam update.\n\n"
        "*Adam optimizer* update rule — m_t (1차 모멘트, β₁=0.9) + v_t (2차 모멘트, "
        "β₂=0.999) + bias correction. tracking_lr = 0.005, tracking_steps = 10/frame 로 "
        "약 13 ms / frame. 매 update 전 phase prediction (φ ← φ + ω·Δt) 로 kinematic "
        "preview, tracking window 3 frames 의 buffer 로 stable gradient.\n\n"
        "*Adaptive N* — 성분 수 동적 조정. (1) Pruning: cumulative energy 90% 유지 (작은 "
        "성분 제거). (2) Growth: residual RMS > 0.05 m 이면 새 (k_x, k_y) 추가 (max_n=30). "
        "(3) Safety rollback: 변경 후 RMS 가 1.3 배 초과 증가하면 즉시 이전 상태 복구. "
        "이게 잘못된 prune/grow 의 cascade failure 방지."
    ),
    'S4_A': (
        "12 episode benchmark 결과 — 이 표가 *핵심 비교* 입니다.\n\n"
        "WaveSplat 평균 0.211 nRMSE / 13.4 ms 대비 Wang2025 (Feng Wang 충실 재현, Ne=300) "
        "0.253 / 632 ms. 정확도 16% 우위, 속도 47 배 빠름, CPU only 입니다. 13 ms = 75 fps "
        "이라 10 Hz 제어 충분. 632 ms 는 1.6 fps 라 real-time 부적합.\n\n"
        "참고 데이터 — Ehlers FNO 1.9 ms 매우 빠르지만 학습 데이터 3120 개 필요. Ehlers "
        "PFT-PINN 은 학습 데이터 0 이지만 instance 마다 30 분 학습. Yan VOF-EnKF 는 wave "
        "breaking 처리 가능하지만 offline only.\n\n"
        "WaveSplat 의 우위는 *모든 항목 동시 만족* — 정확도 + 속도 + 학습 데이터 0 + 2D "
        "directional + sparse LiDAR. 다만 다음 슬라이드에서 Episode-level 패턴 보면 강점과 "
        "한계가 보입니다."
    ),
    'S4_B': (
        "Episode-level 패턴 — 강점과 한계를 정직하게 보여드립니다.\n\n"
        "*단순 파도 (1–5 components) 에서는 WaveSplat 압도* — BF3_2comp 에서 0.052 vs "
        "Wang2025 0.250. 거의 완벽한 추정이죠. 왜? 2 개 성분이면 파라미터 10 개, LiDAR "
        "50 점이면 충분히 잘 맞습니다. 격자 방법은 성분이 적어도 격자 해상도 한계가 있어 "
        "이 정확도에 못 도달.\n\n"
        "*복잡 파도 (15–30 components) 에서는 일부 역전* — BF4_15comp 에서 WaveSplat 0.302 "
        "vs Wang2025 0.249. 격자는 모든 성분을 한꺼번에 담을 수 있지만, WaveSplat 은 "
        "beamforming 으로 peak 를 하나하나 분리해야 해서 30 개를 정확히 찾기 어렵기 때문.\n\n"
        "Reversal point 는 약 N=15. 단순 파도는 파라미터 표현이 효율, 복잡 파도는 격자가 "
        "표현력 우위.\n\n"
        "대응은 hybrid init — 격자로 대략 분포 파악 → 파라미터로 정밀화. 두 표현의 장점 "
        "결합. Learned init (NN) 도 대안. 향후 연구."
    ),
    'S4_C': (
        "Ablation — 두 메커니즘 (multi-frame + dispersion) 의 시너지 검증.\n\n"
        "4 가지 변형 비교 (12 episodes):\n"
        "  A (spatial-only, single-frame, no dispersion): 0.213\n"
        "  B (+multi-frame, no dispersion): 0.208 (개선)\n"
        "  C (+dispersion, single-frame): 0.226 (오히려 악화!)\n"
        "  D (= WaveSplat full): 0.211 (가장 안정)\n\n"
        "*C 가 왜 악화되는가* — single frame 에서 k 추정이 부정확하면 dispersion 으로 "
        "잘못된 ω 가 결정되고, 그 오류가 amplify 됩니다 (cascading failure). 즉 dispersion 은 "
        "multi-frame 과 함께 써야 효과 발휘.\n\n"
        "*D 의 안정성* — multi-frame 으로 k 정확 추정 → dispersion 으로 ω 자동 → 시너지. "
        "두 메커니즘 단독은 불완전, 결합 시 안정.\n\n"
        "핵심 메시지 — 물리 법칙 (dispersion) 은 마법이 아닌 *정규화* 역할. 데이터로 "
        "잘 잡힌 k 위에서 작동해야 합니다."
    ),
    'S4_D': (
        "본 연구의 핵심 메커니즘 — 분산 관계가 *정규화 장치* 라는 발견.\n\n"
        "*메커니즘 1: 자유도 감소* — ω = √(g·|k|·tanh(|k|·d)) 로 ω 가 k 에서 자동 결정. "
        "변수 5N → 4N (20% 감소) + 물리적으로 불가능한 (k, ω) 조합 자동 배제. 정규화 효과.\n\n"
        "*메커니즘 2: Multi-frame temporal coherence* — η(x, t) 의 t 진행이 매 frame phase "
        "변화 제공. Vessel 정지 상태에서도 wave 자체의 시간 진화로 multi-frame 효과 작동. "
        "여러 frame 의 (x, y, t, η) 4D 정보로 wavenumber 모호성 해소. Vessel motion 기반 "
        "synthetic aperture (SAR-style) 는 *향후 연구* — 현재는 wave 시간 진화만 활용.\n\n"
        "*두 메커니즘의 시너지* — Ablation 결과: dispersion 단독 0.226 (악화), multi-frame "
        "단독 0.208 (개선되나 불안정), 결합 D=WaveSplat 0.211 (가장 안정).\n\n"
        "비유로 정리하면 — Dispersion = 눈가리개 (물리 법칙으로 답 좁히기), Multi-frame = "
        "여러 사진 (시점 차이로 모호성 해소). 둘 다 있어야 sparse LiDAR 로 phase-resolved "
        "추정 가능."
    ),
    'S4_E': (
        "왜 2 단계 (Init/Track) 설계가 자연스러운가 — Identifiability 분석.\n\n"
        "*핵심 관찰* — 변수마다 필요한 정보가 다릅니다. 파수 k 는 *공간적 패턴* 이라 한 "
        "순간의 LiDAR 50 점만으로는 정확히 파악 어려움 — 여러 frame 의 시간 정보 필요. "
        "반면 진폭 A 와 위상 φ 는 *크기와 현재 위치* 라 한 순간 데이터로도 충분.\n\n"
        "*Identifiability 증거* — single frame 에서 k 의 FIM eigenvalue 가 (A, φ) 보다 "
        "약 10⁻³ 작음. 즉 k 변화에 데이터가 거의 반응 안 함. k 자유 시 50 점에 over-fit "
        "해서 비물리적 발산. k 고정하면 자유도 5N → 2N 으로 안정적 추정.\n\n"
        "*전체 흐름* — (1) Multi-frame temporal coherence 로 파수 추정 (Init), (2) "
        "Dispersion 으로 ω↔k 결합 → 자유도 감소 + 물리 제약, (3) 매 frame Adam 으로 "
        "real-time (A, φ) 추적, (4) Adaptive N 으로 성분 수 자동 조정.\n\n"
        "이 4 단계가 합쳐져 13 ms / frame 의 real-time + 정확도 0.211 을 동시 달성합니다."
    ),
    'S5_A': (
        "한계를 정직하게 공유합니다 — 신뢰도를 높이려면 잘 되는 점뿐 아니라 한계도 먼저 "
        "인정해야 합니다.\n\n"
        "*한계 1 — 선형 근사 (LWT)* — 파도를 사인파의 합으로 표현. 실제 파도는 비선형 "
        "(꼭대기 뾰족, 골 넓음). Klein 2020 의 wave model 비교: HOSM > NLSE2 > NLSE > LWT. "
        "ε < 0.07 신뢰 영역, BF6 까지 적용 가능. 향후 Stokes 2차 보정으로 확장 가능.\n\n"
        "*한계 2 — 복잡 파도 정확도 저하* — 15 components 이상에서 격자 방법이 일부 우세. "
        "BF4_15comp 0.302 vs 0.249 (역전). 원인은 beamforming 이 다수 peak 분리 어려움. "
        "Hybrid init (격자 + 파라미터 정밀화) 로 향후 해결 예정.\n\n"
        "*한계 3 — Vessel 정지 가정 (V=0 kts)* — 본 dataset 은 vessel 정지로 합성. SA "
        "(synthetic aperture) 효과 미적용. 영향: 매우 긴 swell (λ > 250 m) direction 해상도 "
        "저하 가능. 향후 vessel motion 도입 시 V·T 만큼 추가 spatial coverage.\n\n"
        "*한계 4 — Wave breaking + 유속장 미지원* — VOF-EnKF, PFT-PINN 영역 (offline). 본 "
        "연구는 real-time 제어 영역 — 응용 자체가 다름. 협업 연구로 보완 가능."
    ),
    'S5_B': (
        "WaveSplat 의 응용 사례 — hydrofoil ride control 에 시험 적용.\n\n"
        "*왜 hydrofoil 인가* — 수중 날개로 lift 생성해 vessel 부양하는 구조인데, 파도 "
        "만나면 lift 변동으로 heave/pitch 흔들림 → 승차감 저하. 앞쪽 파도 preview 가 있으면 "
        "fin angle 미리 조절해서 disturbance 상쇄 가능. WaveSplat 의 real-time phase-resolved "
        "추정이 정확히 이 preview 를 제공.\n\n"
        "*제어 구조 비교* — 기존 PD: heave error 만 보고 fin 조절 (feedback only). 제안 "
        "PD + Feed-Forward: WaveSplat 추정 → 미래 wave disturbance 예측 → fin angle 보상 "
        "추가. 최종 fin angle = PD(heave error) + FF(predicted wave).\n\n"
        "*Test setup* — WaveSim 합성 파도 + hydrofoil dynamics simulator. Preview window "
        "0.3 s lookahead (V=11.3 m/s 에서 약 3 m 앞). 같은 파도 시퀀스에 두 controller 비교.\n\n"
        "*현재 단계* — preliminary test, 응용 가능성 확인 정도. 체계적 controller tuning + "
        "robustness 평가는 후속 연구."
    ),
    'S5_C': (
        "Hydrofoil 결과 — 위 영상이 PD vs PD+FF 의 직접 비교입니다.\n\n"
        "위쪽은 시간에 따른 wave + vessel 위치, 아래쪽은 heave acceleration time series. "
        "파란선 PD-only, 빨간선 PD+FF (WaveSplat 사용). PD+FF 가 명확히 진동이 작습니다.\n\n"
        "정량 — aRMS PD 0.090 m/s² → PD+FF 0.047 m/s² (약 48% 감소). ISO 2631 ride comfort "
        "기준에서 의미 있는 수준 — 승차감 향상으로 직결됩니다.\n\n"
        "이건 *시험 적용* 결과 — WaveSplat 정보가 제어에 실제로 도움이 된다는 *proof-of-"
        "concept* 입니다. 실제 deployment 에는 controller robustness, sensor noise, vessel "
        "dynamics uncertainty 등 추가 고려가 필요. 후속 연구로 이어가겠습니다.\n\n"
        "발표 tip — 잠시 영상 정지하고 청자가 두 line 의 차이를 직접 보게 하면 효과적. "
        "*눈으로 봐도 차이* 가 가장 직관적인 evidence."
    ),
    'S5_D': (
        "마지막 — 향후 연구 4 가지 방향.\n\n"
        "*방향 1: 고복잡도 파도 대응 (초기화 개선)* — 현재 15 component 이상에서 정확도 "
        "저하. 계획: hybrid init (격자 → 파라미터 정밀화) 또는 learned init (NN 으로 초기 "
        "성분 추정). 기대: Wang2025 수준 이상.\n\n"
        "*방향 2: 비선형 파도 확장* — 현재 LWT (BF6 까지 신뢰). 계획: Stokes 2 차 보정 "
        "추가. 장기: NLSE 수준 도달.\n\n"
        "*방향 3: Vessel motion 기반 synthetic aperture (SAR-style)* — 현재 코드/dataset "
        "모두 vessel 정지 가정. 계획: V > 0 시 V·T 만큼 추가 spatial coverage 검증. "
        "Wiley 1954 의 SAR 원리, ship-mounted LiDAR 첫 시도가 될 가능성.\n\n"
        "*방향 4: 실측 데이터 검증* — 현재 합성 dataset 만 검증. 계획: 실제 선박 LiDAR + "
        "wave buoy 또는 X-band radar 병행 측정으로 ground-truth 확보.\n\n"
        "최종 목표 — 선박 LiDAR 로 주변 파도를 real-time 추정하고 hydrofoil ride control, "
        "자율운항 같은 선박 제어에 직접 활용 가능한 시스템.\n\n"
        "이상으로 발표 마치겠습니다. 질문 환영합니다."
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
