#!/usr/bin/env python3
"""
Restructure filled PPTX for 20-minute lab meeting (v2).

Changes from v1:
- Hydrofoil control video (S30) restored to main slides
- Fact errors fixed (init frame count, beamforming grid)
- Speaker notes rewritten for 공학출신 비전문가 persona
- All Q&A made friendlier and more detailed
"""

from pptx import Presentation

INPUT = 'wavesplat/docs/labmeeting_20260513_박세용_filled.pptx'
OUTPUT = 'wavesplat/docs/labmeeting_20260513_박세용_20min.pptx'

prs = Presentation(INPUT)
n_slides = len(prs.slides)
assert n_slides == 37, f"Expected 37 slides, got {n_slides}"
print(f"Input: {n_slides} slides")

# ============================================================
# Slide order definition
# ============================================================

# Main (17 slides, ~20 min)
# 배경 압축: Related Work(7) → Appendix, Dataset(14) → Appendix
# 한계 → VIDEO → 향후연구 → 감사합니다
MAIN = [
    0,   # S1  Title
    1,   # S2  Contents (0초, 설명 없이 넘김)
    3,   # S4  Motivation — 왜 필요한가
    4,   # S5  Challenge — 왜 어려운가
    10,  # S11 Application Matrix (Related Work 통합)
    12,  # S13 WaveSim + Dataset 통합
    15,  # S16 WaveSplat overview
    16,  # S17 Init - why multi-frame
    17,  # S18 Init - beamforming + coarse-to-fine
    19,  # S20 Track + Adaptive N
    21,  # S22 12-episode Results
    22,  # S23 Episode-level patterns
    23,  # S24 Ablation
    27,  # S28 한계
    29,  # S30 Hydrofoil PD vs PD+FF (VIDEO)
    30,  # S31 향후 연구
    31,  # S32 감사합니다
]

# Original appendix
APPENDIX_ORIG = [32, 33, 34, 35, 36]

# Moved to appendix
APPENDIX_MOVED = [
    (7,  "Appendix D. Related Work 5 approaches 상세"),
    (8,  "Appendix E. EnKF 기반 상세 비교"),
    (9,  "Appendix F. 학습 기반 접근 상세"),
    (13, "Appendix G. LiDAR 센서 상세"),
    (14, "Appendix H. Dataset 110 episodes 상세"),
    (18, "Appendix I. Beamforming Score 유도"),
    (24, "Appendix J. 핵심 메커니즘 — 분산관계 정규화"),
    (25, "Appendix K. 왜 2단계 설계인가 — Fisher 분석"),
    (28, "Appendix L. Hydrofoil ride control 개요"),
]

# Deleted entirely
DELETED = {2, 5, 6, 11, 20, 26}  # S3, S6, S7, S12, S21, S27

# Verify
all_used = set(MAIN) | set(APPENDIX_ORIG) | set(i for i, _ in APPENDIX_MOVED) | DELETED
assert all_used == set(range(37)), f"Missing: {set(range(37)) - all_used}"

NEW_ORDER = MAIN + APPENDIX_ORIG + [i for i, _ in APPENDIX_MOVED]

print(f"Main: {len(MAIN)} slides, Appendix: {len(APPENDIX_ORIG) + len(APPENDIX_MOVED)}, Deleted: {len(DELETED)}")

# ============================================================
# Helper functions
# ============================================================

def get_notes(slide):
    if slide.has_notes_slide:
        return slide.notes_slide.notes_text_frame.text.strip()
    return ""

def set_notes(slide, text):
    if not slide.has_notes_slide:
        slide.notes_slide
    notes_tf = slide.notes_slide.notes_text_frame
    for i in range(len(notes_tf.paragraphs) - 1, 0, -1):
        p = notes_tf.paragraphs[i]._p
        p.getparent().remove(p)
    notes_tf.paragraphs[0].text = ""
    notes_tf.text = text

def append_notes(slide, extra):
    existing = get_notes(slide)
    set_notes(slide, (existing + "\n\n" + extra) if existing else extra)

slides = list(prs.slides)

# ============================================================
# Fix slide title text (Init 1/3, 2/3 → 1/2, 2/2)
# ============================================================

def fix_slide_text(slide, old, new):
    """Replace text in slide shapes (title etc.)."""
    for shape in slide.shapes:
        if shape.has_text_frame:
            for para in shape.text_frame.paragraphs:
                for run in para.runs:
                    if old in run.text:
                        run.text = run.text.replace(old, new)
                        print(f"  Fixed: '{old}' → '{new}'")

# Init (3/3) moved to Appendix, so main slides become (1/2), (2/2)
fix_slide_text(slides[16], "Init (1/3)", "Init (1/2)")
fix_slide_text(slides[17], "Init (2/3)", "Init (2/2)")

# ============================================================
# Speaker notes updates (friendly persona + fact-checked)
# ============================================================

# --- S28 (한계): 한계만. 향후연구는 S31에서 별도 발표 ---
append_notes(slides[27], """
--- [발표 흐름 가이드] ---
한계 4가지를 짚고, 전환 멘트:
"이런 한계가 있지만, 그럼에도 실제 응용에서 어떤 가능성이 있는지 영상으로 보여드리겠습니다."
→ Hydrofoil VIDEO로 전환.
""")

# --- S31 (향후 연구): VIDEO 뒤, 감사합니다 직전 ---
append_notes(slides[30], """
--- [발표 흐름 가이드] ---
VIDEO 여운이 남아있을 때, 짧게 (~1분) 향후 방향만 언급하고 감사합니다로 마무리.

"영상에서 보신 것처럼 WaveSplat의 wave preview가 실제 제어에 도움이 됩니다.
 앞으로 네 가지 방향으로 발전시킬 계획입니다."

1. 고복잡도 파도 — hybrid init (격자로 대략 잡고 → WaveSplat으로 정밀화)
2. 비선형 확장 — Stokes 2차 보정으로 가파른 파도까지
3. Vessel motion — 선박 이동 시 synthetic aperture 효과로 해상도 향상
4. 실해역 검증 — 합성 → 실제 LiDAR 데이터

"감사합니다. 질문 받겠습니다."
""")

# --- S22 (Results): Wang2025 재현 상세 ---
append_notes(slides[21], """
--- [Wang2025 재현 방법 — 질문 나올 때 대비] ---

"혹시 baseline이 공정하냐는 질문이 나오면 이렇게 답하세요:"

"Wang2025 baseline은 원논문을 최대한 충실하게 재현했습니다.
 - 앙상블 크기 Ne=300 (— 앙상블: 여러 개의 시뮬레이션을 동시에 돌려 통계적으로 최적 해를 찾는 방식. Ne=300이면 300개의 서로 다른 파도 시뮬레이션을 병렬로 운용)
 - 80×60 격자, 격자 간격 dx=3m (논문 Table 2 기준)
 - Fuzzy R (— 관측 노이즈 공분산 R을 퍼지 논리로 자동 조절하는 기법. 측정값의 산포가 크면 R을 키우고, 작으면 R을 줄여 필터가 데이터를 얼마나 신뢰할지 적응적으로 결정)
 - Delaunay 보간 (— 불규칙하게 흩어진 점들을 삼각형으로 연결한 뒤, 각 삼각형 내에서 선형 보간하여 균일 격자 값으로 변환하는 표준 기법)
 - Sub-state ETKF (— ETKF: Ensemble Transform Kalman Filter, 앙상블 칼만 필터의 효율적 변형. Sub-state는 4800개 격자를 영역별로 나눠 각 영역을 독립적으로 업데이트하여 계산량을 줄이는 방법)

같은 110개 에피소드, 같은 LiDAR 입력을 사용했으니 동등한 조건입니다."
""")

# --- S30 (PD vs PD+FF video): Hydrofoil context + Q&A ---
# S30 is now the finale (한계 다음, 감사합니다 직전). 마지막 인상 = 영상.
append_notes(slides[29], """
--- [Hydrofoil 배경 설명 — 한계를 말씀드린 후, 전환] ---

"한계를 솔직히 말씀드렸지만, 그럼에도 이 기술이 실제로 어떻게 쓰일 수 있는지 영상으로 보여드리겠습니다."

"이 영상을 보시기 전에 배경을 간단히 말씀드릴게요.

Hydrofoil은 수중 날개(foil)로 양력(lift)을 만들어 선체를 물 위로 들어올리는 고속선이에요.
문제는 파도를 만나면 foil의 submergence depth가 변하면서 양력이 변동 → 선체가 heave/pitch로 흔들린다는 겁니다.
이걸 줄이려면 '앞에 어떤 파도가 오는지' 미리 알아야 fin angle을 선제적으로 조절할 수 있어요.

WaveSplat이 바로 이 '미래 파도 미리보기'를 real-time (13ms)으로 제공합니다.

영상 설명:
- 위쪽: 시간에 따른 파도와 선박 위치
- 아래쪽: heave acceleration 시계열
  파란선 = PD 제어만 (현재 흔들림만 보고 반응하는 feedback)
  빨간선 = PD + Feed-Forward (WaveSplat으로 미래 파도도 반영)

결과: aRMS (— Root Mean Square acceleration, 가속도의 제곱 평균 제곱근. 진동의 '평균적 세기'를 나타내는 표준 지표) 가 0.090 → 0.047 m/s²로 약 48% 감소했습니다.
ISO 2631 (— 인체 전신 진동에 대한 국제 표준. 가속도 크기별로 'comfortable', 'slightly uncomfortable' 등 등급을 정의) 승차감 기준으로 의미 있는 수준이에요.

다만 이건 proof-of-concept이에요. Perfect estimation, 2자유도 단순 모델, actuator delay 미반영 등의 가정이 있습니다.
중요한 건 수치보다 'wave preview가 제어에 실제로 도움이 된다'는 정성적 결론입니다."

--- [예상 질문] ---

Q: Hydrofoil 48% 감소가 실제로도 재현될까요?
A: [비유] 풍동 실험에서 항력이 줄었다고 실도로에서 똑같이 줄진 않지만, 개선 방향은 맞는 것과 같아요.
[정확한 한계]
 시뮬레이션 가정: (1) WaveSplat 추정이 완벽(실제로는 nRMSE ~0.21 → 추정 오차가 FF 성능을 떨어뜨림), (2) 2DOF 선형 운동 모델(실제 6DOF + 비선형 유체력), (3) actuator delay = 0 (실제 유압 시스템 지연 ~50ms), (4) foil 양력이 정상류 근사.
 그래도 의미 있는 이유: WaveSplat + noisy estimation (nRMSE 40%)에서도 약 20% 감소 달성 — 추정이 완벽하지 않아도 효과가 남음. 핵심은 wave preview 정보가 제어에 유용하다는 정성적 결론.

Q: PD+FF에서 FF가 정확히 뭔가요?
A: [비유] 날씨 예보 보고 우산 챙기는 것이 FF, 비 맞은 후 우산 펴는 것이 PD.
[정확한 제어 구조]
 - PD (Proportional-Derivative) controller: u_PD = Kp·e(t) + Kd·(de/dt) — 현재 오차 e(t)와 오차 변화율만으로 fin angle 결정. 순수 feedback.
 - Feed-Forward term: u_FF = -Kff · η_pred(x_foil, t + Δt) — WaveSplat이 제공하는 foil 위치에서의 미래 파고 예측값에 비례하여 fin angle을 미리 조절. Δt는 선박 속도와 foil-LiDAR 간 거리로 결정.
 - 합산: u_total = u_PD + u_FF. PD가 baseline 안정성 보장, FF가 disturbance 선제 보상 → 시너지.
 - FF의 key insight: 파도가 foil에 도달하기 전에 반응하므로, PD의 고유 phase lag를 보상할 수 있음.

Q: 왜 2자유도(heave+pitch)만 했나요?
A: [정확한 이유]
 (1) 승차감 관점: ISO 2631에서 heave acceleration이 ride comfort의 지배적 factor. Pitch도 두 번째로 중요 (전후 가속도 유발). Roll, sway, surge, yaw는 foil 제어 대상이 아님.
 (2) Hydrofoil 역학: Foil의 fin angle이 직접 조절하는 것은 heave force와 pitch moment. 커플링은 있지만 2DOF로도 핵심 동역학을 포착.
 (3) Proof-of-concept: 6DOF는 추가 파라미터 (hull hydrodynamics, moment of inertia 등)가 필요하고, 이 연구의 목적은 "WaveSplat이 제어에 유용한가?"의 증명이지 정밀 제어기 설계가 아님.
""")

# --- S11 (Matrix): Related Work 5 approaches 내용 통합 ---
append_notes(slides[10], """
--- [선행연구 요약 — 매트릭스 보면서 한 번에 설명] ---

"이 매트릭스를 보시면 기존 접근법 5가지와 WaveSplat을 한눈에 비교할 수 있습니다.
 잠깐 각 접근법을 한 줄로 소개하면:
 ① 격자 기반 최적화 (Desmars): 균일 격자 위에서 파고를 추정 — dense 센서 전제
 ② 앙상블 칼만 필터 (Wang): 4,800개 격자점의 상태를 300개 앙상블로 추적 — 632ms
 ③ 학습 기반 FNO (Ehlers): 신경망으로 1D 파고 예측 — 1.9ms지만 학습 데이터 3,120개 필요
 ④ 학습 기반 PFT-PINN (Ehlers): 물리 손실 학습 — 데이터 불필요하지만 30분/instance
 ⑤ GPR (Gaussian Process): 비모수 보간 — 방향 추정 불가, 200ms+

 각 방법의 상세 설명은 Appendix D~F에 정리해두었습니다.

 핵심은 이 표의 마지막 행: WaveSplat만이 real-time + 2D + sparse + 학습불필요를 동시에 만족합니다."
""")

# --- S13 (WaveSim): Dataset 내용 통합 ---
append_notes(slides[12], """
--- [Dataset 요약 — WaveSim 설명 직후에 이어서] ---

"이 시뮬레이터로 만든 데이터셋을 간단히 말씀드리면:
 - 총 110개 에피소드, 각 300 frame (30초, 10Hz)
 - 4가지 sea state: BF2 (Hs=0.3m, 잔잔) ~ BF5 (Hs=3.5m, 거친 바다)
 - 각 sea state별 complexity: 1, 2, 5, 15, 20, 30 성분 조합
 - 3가지 방향 분포: uniform, narrow, bimodal
 - 비교 실험에는 대표 12개 에피소드 사용, 전체 결과는 Appendix H에 있습니다.

 데이터셋 상세 구성표는 Appendix H를 참조해주세요."
""")

# --- Q&A for each main slide (friendly persona) ---

qa_s4 = """
--- [예상 질문] ---

Q: 실해역 데이터가 없는 건 한계 아닌가요?
A: 맞습니다. 현재는 합성 데이터로 알고리즘 자체의 성능을 검증한 단계입니다.
[비유] 비행기 시뮬레이터로 조종 연습을 먼저 하고, 실제 비행은 그다음에 하는 것과 같아요.
[정확한 의미] WaveSim은 5단계 파이프라인으로 현실적 조건을 모사합니다:
 JONSWAP 파 생성 (— JONSWAP: Joint North Sea Wave Project에서 만든 해양파 에너지 스펙트럼 모델. 주파수별 에너지 분포를 수식으로 정의하며, 실해역 파도의 표준 모델로 사용됨)
 → GT field 계산 (— Ground Truth, 정답 파도장)
 → ray casting (— LiDAR 광선을 가상으로 쏘아 파도 표면과의 교차점을 계산하는 과정)
 → Fresnel 기반 raydrop (— Fresnel 방정식: 빛이 물 표면에 부딪힐 때 반사/투과 비율을 입사각의 함수로 정의. 입사각이 커지면 반사율이 급감하여 LiDAR 신호가 돌아오지 않음 = raydrop)
 → range noise σ~2cm.
 특히 raydrop은 입사각이 커지면 반사율이 급격히 떨어지는 Fresnel 방정식을 따르고, 거리 감쇠도 반영합니다. 이를 통해 "sparse + non-uniform + noisy"라는 실제 LiDAR의 세 가지 핵심 특성을 재현합니다. 물론 바람에 의한 spray, 파도 쇄파(breaking), multi-path reflection 등은 아직 미반영이고, 이 gap이 실해역 적용 시 정확도에 영향을 줄 수 있습니다.

Q: Hydrofoil 말고 다른 응용이 있나요?
A: 네, phase-resolved wave estimation이 가치를 발휘하는 분야가 여러 개 있습니다.
[구체 사례]
 - DP(Dynamic Positioning): 파도 disturbance를 미리 알면 thruster 제어의 energy efficiency 향상
 - 해상 크레인: 파도에 의한 relative motion을 예측해서 lifting window를 정확히 잡음 (Desmars 2020이 이 문제를 다룸)
 - 파력 발전: WEC(Wave Energy Converter)의 최적 제어에 미래 파도 정보가 필수 — 이론적으로 energy capture 30% 이상 향상 가능
 - 선박 경로 최적화: 파도장을 알면 실시간으로 최적 경로를 수정할 수 있음
[핵심] '미래 파도의 공간 분포를 실시간으로 아는 것'이 가치 있는 모든 해양 제어 응용에 적용 가능합니다.
"""
append_notes(slides[3], qa_s4)

qa_s5 = """
--- [예상 질문] ---

Q: LiDAR 점이 50개밖에 안 되는데 정말 추정이 가능한가요?
A: 한 순간 50점만으로는 부족합니다.
[비유] 한 장의 흐릿한 사진으로는 글자를 못 읽지만, 조금씩 다른 각도에서 찍은 사진 100장을 모으면 읽을 수 있는 것과 같아요.
[정확한 로직] Init 단계에서 init_window = 2 × Tp (— Tp: peak period, 에너지가 가장 집중된 파도의 주기) 만큼의 frame을 모읍니다. BF4(Tp=7s) 기준 14초 × 10Hz = 약 140 frame, 각 frame 50점이면 총 약 7,000개의 (x, y, t, η) 4D 측정값이 됩니다. 추정 대상은 N=10 성분 × 4 파라미터 (kx, ky: 파수 = 파도의 공간 주파수, A: 진폭, φ: 위상) = 40개 변수. ω(— 각주파수, 파도가 시간적으로 얼마나 빨리 진동하는지) 는 분산관계 (— dispersion relation: ω = sqrt(g|k|tanh(|k|d)), 파도의 공간 주파수 k와 시간 주파수 ω 사이의 물리적 관계식. 깊은 바다에서는 파장이 길수록 빠르게 전파됨) 로 자동 결정. 즉 측정 7,000개 vs 변수 40개 → 175배의 over-determined system (— 미지수보다 방정식이 훨씬 많아 안정적으로 풀리는 연립방정식) 이라 충분히 풀립니다.
[추가] 분산관계를 강제하면 5N → 4N으로 자유도가 20% 줄어, 같은 데이터로 더 안정적인 추정이 가능합니다.

Q: 다른 LiDAR를 쓰면 어떤가요?
A: 점이 더 많은 센서를 쓰면 당연히 유리합니다.
[비유] 저해상도 카메라로도 인식 가능하면, 고해상도에서는 당연히 더 잘 되는 것과 같아요.
[정확한 로직] WaveSplat의 알고리즘은 점의 개수나 분포에 대한 가정이 없습니다. 입력은 단순히 (x_i, y_i, z_i, t) 좌표 리스트이고, 어떤 센서에서 왔는지 무관합니다. 점이 많아지면 (1) beamforming score의 SNR이 올라가 peak 분리가 쉬워지고, (2) Adam fitting의 conditioning이 좋아져 수렴이 빨라집니다. Falcon K1(50점)은 scanning LiDAR 중에서도 sparse한 편이라, 더 dense한 센서 (Livox 수백점, stereo camera 수천점)에서는 성능 향상이 기대됩니다.
"""
append_notes(slides[4], qa_s5)

qa_s8 = """
--- [예상 질문] ---

Q: 기존 방법을 그대로 LiDAR에 적용하면 안 되나요?
A: [비유] 촘촘한 그물로 잡던 방식을 듬성듬성한 그물에 그대로 쓸 수 없는 것과 같아요.
[정확한 로직] 기존 격자 기반 방법(Desmars, Wang 등)은 Nyquist 조건 (— Nyquist: 신호를 정확히 복원하려면 최소 2배 이상의 샘플링 밀도가 필요하다는 정리. 격자 간격 dx면 파장 2dx 이상만 정확히 표현 가능) 을 전제합니다. ETKF (— Ensemble Transform Kalman Filter: 앙상블 칼만 필터의 효율적 변형으로, 여러 시뮬레이션을 병렬 운용하여 관측과 모델 예측을 통계적으로 합성하는 데이터 동화 기법) (Wang2025)의 경우 80×60 = 4,800개 격자점의 파고를 상태 변수로 두는데, LiDAR 50점으로 4,800차원 상태를 관측하면 관측 rank가 극도로 낮아 앙상블 축소 (— ensemble collapse: 앙상블 멤버들이 서로 비슷해져서 다양성을 잃고 필터가 발산하는 현상. 관측 정보가 부족할 때 발생) 가 일어나기 쉽습니다. 실제로 Wang2025 재현 시 Ne=300에서도 632ms가 걸리며, 격자 해상도 한계(dx=3m → k_max ≈ 1.05 rad/m)로 인해 높은 wavenumber 성분을 표현하지 못합니다.
[반면] WaveSplat은 격자가 아닌 N개(5~15) 연속 파라미터를 직접 추정하므로, 50점으로도 충분히 over-determined됩니다.

Q: PFT-PINN이 학습 데이터 0개면 좋아 보이는데 왜 안 쓰나요?
A: [비유] 시험 때마다 교과서를 처음부터 다시 읽어야 하는 것과 같아요 — 결국 시간이 오래 걸립니다.
[정확한 로직] PFT-PINN (Ehlers 2025)은 두 신경망(flow field NN + free-surface NN)을 물리 손실함수 (— loss function: 모델 예측과 실제 값의 차이를 수치로 표현한 함수. 이걸 최소화하는 방향으로 파라미터를 조정) (Navier-Stokes PDE residual + 경계조건)로 동시에 학습합니다. 학습 데이터는 불필요하지만, 각 wave scenario마다 수만 iteration의 역전파 (— backpropagation: 출력 오차를 입력 방향으로 전파하며 각 파라미터의 기여도를 계산하는 신경망 학습 핵심 알고리즘) 가 필요해서 약 30분/instance가 걸립니다. 이건 offline 분석(실험 후 정밀 복원)에는 매우 강력하지만, 10Hz(100ms/frame) 제어 루프에는 원리적으로 사용 불가능합니다. 반면 WaveSplat은 Init 1회(~1초) 후 매 frame Adam 10 step = 13ms로 real-time을 만족합니다.
"""
append_notes(slides[7], qa_s8)

qa_s11 = """
--- [예상 질문] ---

Q: 매트릭스의 기준이 주관적이지 않나요?
A: 각 기준은 실제 시스템 요구사항에서 직접 나온 것입니다.
[정확한 근거]
 - Real-time (<100ms): 선박 제어 루프가 통상 10Hz로 동작 → 1 frame 처리에 100ms 이내 필수. WaveSplat 13ms, Wang2025 632ms(불가), PFT-PINN 30분(불가)
 - 2D 방향 추정: 파도의 propagation direction을 모르면 feed-forward 제어 시 phase 맞춤 불가능. 1D 방법(FNO, Desmars)은 방향을 미리 알아야 하는 추가 가정 필요
 - Sparse 처리: Falcon K1 scanning LiDAR의 물리적 제약 — 76,000 ray 중 유효 water hit은 ~50점. 기존 stereo camera(수천점)나 radar(수백점) 전제 방법은 직접 적용 불가
 - 학습 데이터 불필요: 새 해역에 배치할 때마다 3,120개 HOS 시뮬레이션을 돌릴 수 없음 (FNO의 실용적 장벽)
[핵심] 어떤 방법도 6개 기준을 모두 만족하지 못하고, WaveSplat만 4개를 동시에 만족합니다. 이것이 우리의 고유 영역입니다.

Q: 이름이 3D Gaussian Splatting과 비슷한데 관련이 있나요?
A: 영감을 받았습니다.
[공통점] 둘 다 연속 파라미터의 집합으로 대상을 표현합니다 (grid-free parametric representation). 3DGS: {μ, Σ, color, opacity} × 수만 개 → radiance field. WaveSplat: {kx, ky, A, φ} × 5~30개 → wave field.
[차이점]
 - 도메인: 3DGS는 spatial domain (xyz 공간의 radiance), WaveSplat은 spectral domain (kx-ky 공간의 wave energy)
 - 물리 제약: 3DGS는 물리 제약 없음 (photometric loss만), WaveSplat은 분산관계 ω=f(k) 강제
 - 규모: 3DGS는 수만~수십만 Gaussian, WaveSplat은 5~30개 (해양파의 physical sparsity 활용)
 - 최적화: 3DGS는 differentiable rendering + densification, WaveSplat은 beamforming init + Adam tracking
"""
append_notes(slides[10], qa_s11)

qa_s13 = """
--- [예상 질문] ---

Q: WaveSim의 물리적 타당성은 어느 정도인가요?
A: [5단계 파이프라인의 각 단계 근거]
 1) 파도 생성: JONSWAP 스펙트럼 (Hasselmann et al. 1973)에서 성분 추출 → Gerstner wave 중첩. 선형 중첩 가정이므로 BF5(Hs~3.5m)까지 유효
 2) GT field: 200×200 격자에서 해석적으로 계산 — 수치 오차 없음
 3) Ray casting: Moller-Trumbore 알고리즘으로 152채널 × 500수평빔 = 76,000 ray/frame. 삼각 메시 기반으로 정확한 교차점 계산
 4) Raydrop: Fresnel 방정식 (n_water = 1.321 @1550nm) + 거리 감쇠 (ref distance 125m). 입사각이 커질수록 반사율 급감 → 원거리+저입사각에서 점이 빠짐
 5) Noise: range σ ~ 2cm (Falcon K1 사양 기반)
[한계] 바람 spray, 쇄파(breaking), 조류(current), multi-path reflection은 미반영. 이 gap이 sim-to-real transfer의 핵심 과제입니다.

Q: 테스트는 어떻게 하나요?
A: [정확한 구조]
 - WaveSim 39개: 파도 생성기 (dispersion 정합, energy 보존, 위상 속도), ray cast (교차점 정밀도, edge case), raydrop (Fresnel 경계값, 거리 의존성), noise model (분포 검증)
 - WaveSplat 18개: 분산관계 (심해/천해 극한), spectral Gaussian forward model, beamforming score, coarse-to-fine 수렴, adaptive N (pruning/growth), tracking stability
 - 전체 57개 테스트가 pytest로 자동화되어 있고, CI에서 매 커밋마다 실행됩니다.
"""
append_notes(slides[12], qa_s13)

qa_s16 = """
--- [예상 질문] ---

Q: 왜 하필 Gaussian인가요? 다른 basis function은 안 되나요?
A: [비유] 음악을 표현할 때 "도레미파" 음표를 쓰는 것과 같아요 — 파도도 기본 "음표"(sine wave)의 합으로 정확히 표현됩니다.
[정확한 로직] Linear Wave Theory에 의해 해양파는 η(x,t) = Σ A_n cos(k_n · x - ω_n t + φ_n)으로 정확히 분해됩니다. 각 성분은 (A_n, kx_n, ky_n, ω_n, φ_n) 5개 파라미터로 완전 결정되고, 분산관계 ω = sqrt(g|k|tanh(|k|d))를 강제하면 4개로 줄어듭니다. 'Spectral Gaussian'이란 이름은 — JONSWAP 스펙트럼에서 에너지가 peak 주변에 Gaussian-like하게 분포하는 것에서 유래합니다. 다른 basis(wavelet, Fourier grid 등)도 가능하지만, sine basis는 (1) 해양파의 물리적 mode와 1:1 대응, (2) gradient가 해석적으로 계산 가능 (Adam 최적화에 유리), (3) 파라미터의 물리적 해석이 직접적 (A=파고, k=파향+파장, ω=주기)이라는 장점이 있습니다.

Q: 성분 수 N은 어떻게 정하나요?
A: [비유] 오케스트라에서 연주자가 필요한 만큼 자동으로 늘고 줄어드는 것과 같아요.
[정확한 로직] 두 단계로 작동합니다.
 (1) Init: coarse-to-fine schedule [3, 8, 15] — 처음 3개로 dominant peak만 찾고, 각 단계에서 loss 수렴 기준(Δloss < threshold)을 만족하면 다음 단계로. 큰 성분부터 잡아야 작은 성분의 init이 안정적 (greedy approach)
 (2) Track (매 frame): 에너지 기반 pruning (A_n² < threshold → 제거) + residual-triggered growth (잔차가 크면 beamforming으로 새 성분 탐색 → 추가). Safety rollback: 새 성분 추가 후 loss가 오히려 증가하면 이전 상태로 복구.
 결과적으로 N은 sea state의 실제 복잡도에 자동 적응합니다. BF2(단순)면 N≈3-5, BF4(복잡)면 N≈10-15.
"""
append_notes(slides[15], qa_s16)

qa_s17 = """
--- [예상 질문] ---

Q: Multi-frame이면 시간 지연이 생기지 않나요?
A: [비유] 자동차 네비가 처음 GPS 위성을 잡는 데 몇 초 걸리지만, 한번 잡으면 실시간으로 위치가 갱신되는 것과 같아요.
[정확한 로직] Init은 시스템 시작 시 1회만 실행됩니다. init_window = 2 × Tp 시간(BF4 기준 14초)의 데이터를 모아 beamforming → coarse-to-fine → Adam refinement를 순차 수행하고, 이후 Track 모드로 전환합니다. Track에서는 매 frame의 50점만으로 Adam 10 step = 13ms 안에 (A, φ)를 갱신합니다. 즉 Init에서 결정한 k(파도의 공간 주파수)는 고정하고, 빠르게 변하는 A(진폭)과 φ(위상)만 추적하는 것입니다. Init delay 14초는 선박 운항 시작 시 1회뿐이므로 실용적으로 문제가 없습니다.

Q: Nyquist 한계를 넘는 파도 성분은 어떻게 잡나요?
A: [비유] 한 장의 사진으로는 보이지 않던 세밀한 패턴이 동영상으로 보면 보이는 것과 같아요.
[정확한 로직] Single-frame spatial Nyquist: LiDAR 점 간격 Δy ≈ 5-8m → k_max = π/Δy ≈ 0.4-0.6 rad/m. 이 이상의 wavenumber는 한 순간의 공간 정보만으로는 aliasing 없이 추정 불가능합니다.
해결: 파도 η(x,t) = A cos(k·x - ωt + φ)에서 시간 t에 따라 위상이 -ωt만큼 변합니다. 분산관계로 ω가 k에서 결정되므로, 여러 frame에서의 위상 변화 패턴이 각 k를 고유하게 식별합니다. Beamforming score에서 시간 위상 정렬 term exp(iω(k)t)이 이 역할을 합니다.
[한계] BF3의 short tail(k > 1.0 rad/m, 최대 Nyquist의 ~2.9배)까지 모두 정확하게 잡기는 어렵고, 에너지가 집중된 peak 근처에서 효과적입니다.
"""
append_notes(slides[16], qa_s17)

qa_s18 = """
--- [예상 질문] ---

Q: Beamforming에서 61×61 grid면 그것도 결국 grid 아닌가요?
A: [비유] 지도에서 대략적 위치를 찾고(검색 grid), GPS로 정밀 좌표를 잡는(Adam refinement) 것과 같아요.
[정확한 로직] 두 단계가 구분됩니다:
 (1) Beamforming (— 여러 센서의 신호를 위상 맞춰 합산해 특정 방향의 신호를 강화하는 기법. 원래 안테나 배열에서 유래) (discrete search): kx, ky ∈ [-4, 4] rad/m 범위에서 61×61 = 3,721개 후보점의 score를 계산합니다. Score = |Σ_i z_i · exp(-i(kx·xi + ky·yi - ω(k)·ti))|² — 이건 matched filter (— 찾고자 하는 신호 패턴과 데이터를 곱해서 합산하는 탐지 기법. 패턴이 맞으면 값이 커지고, 안 맞으면 상쇄) 로, 각 후보 (kx,ky)에 대해 데이터의 위상 정렬도를 평가합니다. Peak를 찾아 초기 k 후보를 선정합니다.
 (2) Adam refinement (— Adam: Adaptive Moment Estimation, gradient descent의 개선 버전. 각 파라미터별로 학습률을 자동 조절하여 빠르고 안정적으로 수렴하는 최적화 알고리즘) (continuous optimization): 선정된 k 후보를 초기값으로, 전체 multi-frame 데이터에 대해 MSE loss를 minimize합니다. kx, ky가 연속 실수값으로 미세조정됩니다. 이 단계에서 격자 해상도 Δk ≈ 0.13 rad/m의 한계를 극복합니다.
[핵심] 최종 출력은 격자점이 아닌 연속 파라미터 (kx_n, ky_n, A_n, φ_n)이고, 이 점에서 grid-based 방법과 근본적으로 다릅니다.

Q: Coarse-to-fine schedule (3→8→15)은 어떻게 정했나요?
A: [비유] 퍼즐을 풀 때 큰 조각부터 맞추고 작은 조각을 채우는 것과 같아요.
[정확한 로직] 비선형 최적화에서 변수가 많을수록 loss landscape의 local minima (— 전체 최적이 아닌 국소적 골짜기. 여기 빠지면 더 좋은 해가 있어도 탈출하기 어려움) 가 기하급수적으로 늘어납니다. Stage별 동작:
 - N=3: 에너지 상위 3개 dominant mode만 탐색. Loss landscape (— 파라미터 조합에 따른 오차(loss) 값을 지형처럼 시각화한 것. 골짜기=좋은 해, 봉우리=나쁜 해) 이 비교적 smooth해서 global optimum (— 전체 탐색 공간에서 가장 좋은 해) 에 수렴하기 쉬움
 - N=8: 이전 3개를 고정(warm start)하고 5개를 추가 탐색. 이전 결과가 gradient landscape을 안정화
 - N=15: 마찬가지로 기존 8개 위에 7개 추가
 각 단계에서 loss 변화율 |Δloss/loss| < convergence_threshold를 만족하면 다음 단계로 자동 진행합니다. Schedule [3,8,15]는 config에서 조정 가능하며 (default: wavesplat/run.py), 대부분의 BF2~BF5 sea state에서 안정적으로 작동합니다.
"""
append_notes(slides[17], qa_s18)

qa_s20 = """
--- [예상 질문] ---

Q: k를 고정하면 파도 조건이 바뀔 때 대응이 안 되지 않나요?
A: [비유] 한번 설정한 라디오 주파수를 고정하고 볼륨만 조절하는 것과 같아요. 방송국이 바뀌면 다시 튜닝이 필요합니다.
[정확한 로직] w_phi tracking mode에서는 k = (kx, ky)를 Init 결과로 고정하고, 매 frame (A, φ)만 Adam으로 갱신합니다. 자유도가 5N → 2N으로 줄어 50점으로도 안정적 추정이 가능합니다. Sea state 변화에 대한 대응:
 - 느린 변화 (수 분~수십 분): 해양 sea state는 통상 20-30분 단위로 변하므로, 수 분 간격 periodic re-init으로 대응 가능
 - Adaptive N의 growth: 기존 성분의 k는 유지하되, 잔차에서 새 peak가 감지되면 beamforming으로 새 성분을 추가. 즉 "기존 k 유지 + 새 k 추가"는 가능
 - 한계: 기존 성분의 k 자체가 drift하는 경우(예: 조류 변화로 도플러 효과) → re-init 필요. 이 부분이 향후 과제

Q: Fisher information 분석이 뭔가요? 쉽게 설명해주세요.
A: [비유] 체중계가 몸무게 변화에는 민감하게 반응하지만, 키 변화에는 거의 반응하지 않는 것과 같아요. 반응 안 하는 양(키=k)을 자유롭게 두면 아무 값이나 맞아버립니다.
[정확한 로직] Fisher Information Matrix (FIM)의 정의: F_ij = E[∂log p(z|θ)/∂θ_i · ∂log p(z|θ)/∂θ_j]
 - 우리 모델 η = Σ A_n cos(k_n·x - ω_n·t + φ_n)에서 FIM을 계산하면:
 - ∂η/∂A = cos(k·x - ωt + φ) → 크기 O(1)
 - ∂η/∂φ = -A sin(k·x - ωt + φ) → 크기 O(A) ≈ O(1)
 - ∂η/∂k = -A·x·sin(k·x - ωt + φ) → 크기 O(A·x)이지만, 50개 sparse점에서 x 범위가 좁아 실질 민감도가 낮음
 - 수치적으로: k의 FIM 고유값이 (A,φ)보다 약 10³배 작음 → k를 자유롭게 두면 gradient가 noise에 지배되어 overfitting
 - 결론: k는 multi-frame Init에서 충분한 정보로 결정하고, Track에서는 고정하는 것이 최적 설계
 자세한 수식은 Appendix K 참조.
"""
append_notes(slides[19], qa_s20)

qa_s22 = """
--- [예상 질문] ---

Q: GPU 쓰면 더 빨라지나요?
A: [현재 구조] CPU-only, PyTorch autograd (— 자동 미분: 수식의 gradient를 사람이 유도하지 않고 컴퓨터가 자동으로 계산하는 기능) 기반 Adam optimizer (— Adam: gradient descent의 개선 버전으로 학습률을 파라미터별로 자동 조절). 매 frame: forward model evaluation (N개 cos 합산, O(N·M)) + backward (자동 미분) + Adam update × 10 step = 13ms.
[GPU 가능성] Forward/backward의 cos 계산이 embarrassingly parallel (N×M 독립 연산)이므로 GPU batch 처리로 가속 가능합니다. 다만 N=10, M=50이면 연산 자체가 작아 GPU kernel launch overhead가 이점을 상쇄할 수 있습니다. 13ms로 이미 10Hz를 만족하므로 현재 우선순위가 낮습니다.

Q: FNO가 1.9ms로 훨씬 빠른데 왜 안 쓰나요?
A: [비유] FNO는 문제마다 맞춤 사전을 만들어야 하고, WaveSplat은 문법 규칙(물리법칙)만 알면 어떤 문장이든 해석 가능해요.
[정확한 비교]
 - FNO (— Fourier Neural Operator: Fourier 변환 공간에서 패턴을 학습하는 신경망. 입력→FFT→학습된 필터 적용→역FFT→출력 구조) (Ehlers 2023, Section 4.3.3, RTX 3050 Ti 기준 1.9ms). 장점은 inference 속도. 한계: (1) HOS 시뮬레이션으로 3,120개 학습 데이터를 합성해야 함 — 새 해역이면 다시 만들어야 함, (2) 학습 분포 밖(다른 Hs, Tp, 다른 센서)에서 정확도 보장 불가(generalization 한계), (3) 1D cross-section만 처리 — 2D 방향 정보 없음
 - WaveSplat: 물리 모델(분산관계) 기반 → 학습 데이터 0개, 어떤 sea state에서든 작동, 2D (kx, ky) 방향 추정 가능. 13ms면 실용적으로 충분
[핵심] 1.9ms vs 13ms의 7배 차이보다, "학습 데이터 의존성 없음 + 2D + generalization"이 실용적으로 더 중요합니다.

Q: 왜 12개 에피소드만으로 비교하나요?
A: [정확한 구성] 12개 = 4 sea states (BF2, BF3, BF4, BF5) × 3 complexity levels (1comp, 5comp, 15 or 30comp)의 대표 조합. 각 에피소드는 300 frame (30초)으로 구성되어 총 3,600 frame에서 비교합니다.
[통계적 의미] 전체 110개 에피소드 결과는 Appendix B의 표에 있고, 12개 대표 에피소드의 평균 nRMSE (— normalized Root Mean Square Error: 예측과 정답의 차이를 정답 파고의 표준편차로 나눈 정확도 지표. 0이면 완벽, 1이면 정답만큼 틀림) 와 전체 평균의 차이는 <2%입니다. 12개는 각 sea state × complexity의 교차점을 커버하는 최소 대표 집합입니다.
"""
append_notes(slides[21], qa_s22)

qa_s23 = """
--- [예상 질문] ---

Q: 15-30 성분에서 지는데, 실해역은 성분이 훨씬 많지 않나요?
A: [비유] 오케스트라에서 주요 악기 5-15개의 소리만 들어도 곡을 충분히 파악할 수 있지만, 30개 악기의 미세한 차이까지 모두 구분하긴 어려운 것과 같아요.
[정확한 로직] 실해역 파도의 에너지 스펙트럼은 JONSWAP으로 근사되며, 에너지가 peak frequency 주변에 집중됩니다. Spectral bandwidth로 분석하면:
 - 단봉 스펙트럼 (unimodal): peak ± Δf 범위에 에너지 ~95% 집중 → 5-10개 성분으로 충분히 representation
 - 이봉 스펙트럼 (bimodal, swell+wind sea): 두 peak 각각에 5-10개 → 총 10-20개
 - 30성분은 에너지가 매우 넓은 directional spreading (σ_θ > 30°)을 가정한 극단적 시나리오
[한계의 원인] 30개 peak를 61×61 beamforming grid에서 분리해야 하는데, peak 간 간격이 Δk ≈ 0.13 rad/m 이하로 가까워지면 분리 불가 (spectral leakage — 유한한 데이터 윈도우로 인해 한 주파수 성분의 에너지가 인접 주파수로 '누출'되어 peak 분리가 어려워지는 현상). Wang2025는 격자 전체를 한번에 추정하므로 이 문제가 없음 → 고복잡도에서 역전.
[개선 방향] Hybrid init (격자 방법으로 대략적 스펙트럼 추정 → peak 위치를 WaveSplat 초기값으로 전달)으로 해결 계획.

Q: 왜 단순 파도에서 격자보다 훨씬 잘 하나요?
A: [비유] "서울에 사람이 3명 있다"고 알려주는 것이 "서울 지도의 1만개 격자 각각에 몇 명 있는지" 맞추는 것보다 훨씬 쉬운 것과 같아요.
[정확한 로직 — 자유도 관점]
 - Grid (Wang2025): 80×60 = 4,800개 격자점의 파고 η를 상태변수로 추정. LiDAR 50점 → 관측 rank = 50. 즉 4,800차원 상태를 50차원 관측으로 추정하는 극도의 under-determined problem. 앙상블(Ne=300)과 시간 전파로 보상하지만, 격자 해상도 한계(dx=3m → k_max ≈ 1.05 rad/m)는 극복 불가
 - WaveSplat (N=2): 2 성분 × 4 파라미터 = 8개 변수. 50점 관측으로 8개를 추정 → 6배 over-determined. nRMSE 0.052 달성 (BF3_2comp)
 - WaveSplat (N=10): 40개 변수. 여전히 over-determined
[핵심] Parametric representation은 "실제 파도의 자유도"에 맞게 변수 수를 설정하므로, 측정이 sparse할수록 grid 대비 유리합니다.
"""
append_notes(slides[22], qa_s23)

qa_s24 = """
--- [예상 질문] ---

Q: Ablation (— 모델의 각 구성요소를 하나씩 제거/변경하며 기여도를 측정하는 실험 방법론) 에서 C(dispersion만 추가)가 오히려 나빠지는 게 직관에 반하는데요?
A: [비유] GPS가 부정확할 때 네비를 켜면 '잘못된 길로 확신 있게 안내'해서 오히려 더 헤매는 것과 같아요.
[정확한 메커니즘]
 Ablation 4가지 변형:
  A (기본): v1 init (spatial FFT, single-frame) + free ω → nRMSE 0.213
  B: v2 init (beamform, multi-frame) + free ω → 0.208 (시간정보 도움)
  C: v2 init (beamform, single-frame) + hard dispersion → 0.226 (악화!)
  D = WaveSplat: v2 init (beamform, multi-frame) + hard dispersion → 0.211 (최적)

 C가 악화되는 정확한 이유:
  - Single-frame beamforming에서 k 추정 오차가 큼 (Nyquist 한계 근처에서 peak 위치 부정확)
  - Dispersion 강제: ω = sqrt(g|k|tanh(|k|d)) → 잘못된 k에서 ω가 자동 결정됨
  - Adam tracking 시 ω가 잘못 고정되어 위상 φ가 data에 맞지 않는 방향으로 수렴
  - 즉 "error propagation through physics constraint" — constraint가 오히려 오류 증폭 경로가 됨

 D가 최적인 이유:
  - Multi-frame에서 k를 충분히 정확하게 추정한 후 dispersion 적용 → ω도 정확 → φ 추정 안정
  - 두 메커니즘의 시너지: temporal info (k 정확화) × physics constraint (자유도 축소) = 상승효과

Q: 분산관계가 '정규화'라는 표현이 과장 아닌가요?
A: [정확한 논리] Regularization의 수학적 정의와 비교합니다:
 (1) 자유도 감소: ω를 자유변수에서 제거 → 5N → 4N (20% 축소). Ridge regression의 λ||θ||²가 solution space를 좁히는 것과 동일 효과
 (2) Solution space 제약: (k, ω) 쌍이 분산관계 곡면 위로 제한됨 → 물리적으로 불가능한 해 (예: 느린 파도가 짧은 파장) 자동 배제
 (3) Bias-variance trade-off (— 모델이 너무 유연하면 noise에 흔들리고(high variance), 너무 제약하면 체계적 오차가 생김(high bias). 둘의 최적 균형점을 찾는 것이 핵심): Constraint가 약간의 bias를 도입 (실제 파도가 정확히 linear가 아닐 수 있음) 하지만 variance를 크게 줄임 → 50점 sparse 관측에서 net benefit
 (4) Conditional effectiveness: Ablation C vs D에서 보듯, 단독으로는 악화되고 충분한 정보(multi-frame)와 결합돼야 효과 발현 → regularization의 전형적 패턴

 따라서 "물리적 정규화 장치"라는 표현은 수학적으로도 정당화됩니다.

[핵심 메커니즘 상세: Appendix J / 2단계 설계 Fisher 분석: Appendix K 참조]
"""
append_notes(slides[23], qa_s24)

qa_s28 = """
--- [예상 질문] ---

Q: 선형 근사의 적용 한계는 어디까지인가요?
A: [비유] 작은 파도는 sine wave로 정확하지만, 큰 파도는 마루가 뾰족하고 골이 평평해져서 단순 sine으로 안 맞는 것과 같아요.
[정확한 기준] Wave steepness ε = kA (— 파수 × 진폭. 파도의 '가파른 정도'를 무차원으로 표현. 값이 클수록 비선형 효과가 강해짐) 로 판단합니다:
 - ε < 0.05: LWT 정확 (대부분 BF2~BF4)
 - 0.05 < ε < 0.1: LWT 근사적 유효 (BF5)
 - ε > 0.1: 비선형 효과 무시 불가 (BF6+, crest-trough 비대칭, wave-wave interaction)
 Klein 2020 (Fluids)에 따르면 Stokes 2차 보정 η₂ = (kA²/4)·cos(2(kx-ωt))를 추가하면 ε ~ 0.15까지 커버 가능 → BF7(Hs~6m)까지 확장. 장기적으로는 NLSE (비선형 슈뢰딩거) 수준을 목표로 합니다.

Q: Vessel stationary (V=0) 가정이 비현실적이지 않나요?
A: [비유] 카메라를 고정하고 찍어도 충분히 좋은 사진이 나오면, 움직이면서 찍으면 (panorama 효과로) 더 좋아질 가능성이 높은 것과 같아요.
[정확한 로직] V=0은 가장 보수적인 조건입니다:
 - V=0: LiDAR coverage가 고정 → spatial sampling 제한적. 파도 자체의 이동(phase velocity)만이 temporal diversity 제공
 - V>0: 선박 이동에 의해 LiDAR scan 영역이 변함 → Synthetic Aperture 효과 (Wiley 1954). 실효 spatial bandwidth가 V/f_scan만큼 증가. 즉 공간 해상도가 향상되어 beamforming 분해능이 개선됨
 - 결론: V=0에서 nRMSE 0.211이면, V>0에서는 더 좋아질 것으로 기대. 향후 연구 방향 3번에서 이를 활용할 계획.

Q: 논문 제출 일정은 어떻게 되나요?
A: OCEANS 2026 conference를 1차 목표로 합니다. Conference 버전에서는 핵심 알고리즘 + 12-episode 비교 결과를 다루고, 확장 journal 버전 (Ocean Engineering)에서는 비선형 확장, 실해역 검증, vessel motion SAR 활용을 추가할 계획입니다.

Q: 실해역 검증은 언제쯤 가능한가요?
A: [선결 조건] 회사(Avikus)에서 LiDAR 탑재 선박의 해상 운항 데이터가 필요합니다. 데이터에 포함되어야 하는 것: (1) LiDAR point cloud (timestamped), (2) vessel IMU (motion reference), (3) 가능하면 wave buoy 또는 radar 기반 GT.
[기술적 readiness] WaveSplat의 입력은 (x, y, z, t) 좌표 리스트이므로, 실제 LiDAR 데이터에서 water surface hit을 필터링하면 바로 적용 가능합니다. 주요 gap은 sim-to-real의 raydrop 분포 차이와 multi-path reflection인데, WaveSim에서 Fresnel 기반 raydrop을 반영했으므로 기본적인 robustness는 확보했다고 봅니다.
"""
append_notes(slides[27], qa_s28)

# --- Appendix labels ---
for orig_idx, label in APPENDIX_MOVED:
    slide = slides[orig_idx]
    existing = get_notes(slide)
    prefix = f"[{label}]\n\n"
    if existing and not existing.startswith("[Appendix"):
        set_notes(slide, prefix + existing)

print("Speaker notes updated with Q&A and fact-checked content.")

# ============================================================
# Reorder slides
# ============================================================

sldIdLst = prs.slides._sldIdLst
original_elements = list(sldIdLst)

for elem in list(sldIdLst):
    sldIdLst.remove(elem)

for idx in NEW_ORDER:
    sldIdLst.append(original_elements[idx])

print(f"Reordered: {len(NEW_ORDER)} slides")

# ============================================================
# Save
# ============================================================

prs.save(OUTPUT)
print(f"Saved to: {OUTPUT}")

# Verify
prs2 = Presentation(OUTPUT)
print(f"\nFinal: {len(prs2.slides)} slides")
for i, slide in enumerate(prs2.slides):
    texts = []
    for shape in slide.shapes:
        if shape.has_text_frame:
            t = shape.text_frame.text.strip()
            if t:
                texts.append(t[:50])
    title = texts[0] if texts else '(empty)'
    section = "MAIN" if i < len(MAIN) else "APPENDIX"
    marker = " ★VIDEO" if i == 14 else ""  # S30 is at index 14 (0-based)
    print(f"  {i+1:2d}. [{section:7s}] {title[:55]}{marker}")
