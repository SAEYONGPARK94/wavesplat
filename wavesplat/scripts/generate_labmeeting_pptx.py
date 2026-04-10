"""
generate_labmeeting_pptx.py
Lab meeting presentation: WaveSplat overview (~15 slides)
Structure:
  1. Why this research
  2. Novelty
  3. Methodology
  4. Results
  5. Plans (paper, PhD)

Output: docs/labmeeting_wavesplat.pptx
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pptx import Presentation
from pptx.util import Inches, Pt, Emu, Cm
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import copy

# ── paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, '..')
FIG_DIR = os.path.join(ROOT, 'paper', 'figures')
SUB_DIR = os.path.join(ROOT, 'paper', 'submission')
OUT_PATH = os.path.join(ROOT, 'docs', 'labmeeting_wavesplat.pptx')

# ── colors ──
DARK = RGBColor(0x1A, 0x1A, 0x2E)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ACCENT = RGBColor(0x00, 0x7A, 0xCC)   # blue
ACCENT2 = RGBColor(0xE8, 0x55, 0x3A)  # orange-red
LIGHT_BG = RGBColor(0xF0, 0xF4, 0xF8)
GRAY = RGBColor(0x66, 0x66, 0x66)
GREEN = RGBColor(0x2C, 0xA0, 0x2C)
RED = RGBColor(0xCC, 0x33, 0x33)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# ── helpers ──
def add_slide(bg_color=WHITE):
    layout = prs.slide_layouts[6]  # blank
    slide = prs.slides.add_slide(layout)
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = bg_color
    return slide

def add_textbox(slide, left, top, width, height, text, font_size=18,
                bold=False, color=DARK, alignment=PP_ALIGN.LEFT,
                font_name='Calibri'):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top),
                                     Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = alignment
    return tf

def add_bullet_list(slide, left, top, width, height, items, font_size=16,
                    color=DARK, bold_prefix=True):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top),
                                     Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.space_after = Pt(6)
        p.level = 0

        # Handle bold prefix with "**text**" syntax
        if '**' in item and bold_prefix:
            parts = item.split('**')
            for j, part in enumerate(parts):
                if not part:
                    continue
                run = p.add_run()
                run.text = part
                run.font.size = Pt(font_size)
                run.font.color.rgb = color
                run.font.name = 'Calibri'
                run.font.bold = (j % 2 == 1)
        else:
            run = p.add_run()
            run.text = f"• {item}"
            run.font.size = Pt(font_size)
            run.font.color.rgb = color
            run.font.name = 'Calibri'
    return tf

def add_section_header(slide, number, title, subtitle=""):
    """Add a colored section bar at top"""
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                    Inches(0), Inches(0),
                                    Inches(13.333), Inches(1.2))
    shape.fill.solid()
    shape.fill.fore_color.rgb = DARK
    shape.line.fill.background()

    add_textbox(slide, 0.5, 0.15, 12, 0.5,
                f"  {number}", 14, color=ACCENT, alignment=PP_ALIGN.LEFT)
    add_textbox(slide, 0.5, 0.35, 12, 0.7,
                f"  {title}", 28, bold=True, color=WHITE)
    if subtitle:
        add_textbox(slide, 0.5, 0.75, 12, 0.4,
                    f"  {subtitle}", 14, color=RGBColor(0xAA, 0xBB, 0xCC))

def add_page_number(slide, num, total):
    add_textbox(slide, 12.0, 7.1, 1.2, 0.3,
                f"{num}/{total}", 10, color=GRAY, alignment=PP_ALIGN.RIGHT)

def try_add_image(slide, fig_name, left, top, width, height=None):
    """Try to add image, return True if successful"""
    for ext in ['.png', '.pdf']:
        path = os.path.join(FIG_DIR, fig_name + ext)
        if os.path.exists(path) and ext == '.png':
            if height:
                slide.shapes.add_picture(path, Inches(left), Inches(top),
                                        Inches(width), Inches(height))
            else:
                slide.shapes.add_picture(path, Inches(left), Inches(top),
                                        Inches(width))
            return True
    # try submission dir
    for ext in ['.png', '.pdf']:
        path = os.path.join(SUB_DIR, fig_name + ext)
        if os.path.exists(path) and ext == '.png':
            if height:
                slide.shapes.add_picture(path, Inches(left), Inches(top),
                                        Inches(width), Inches(height))
            else:
                slide.shapes.add_picture(path, Inches(left), Inches(top),
                                        Inches(width))
            return True
    return False

def add_table(slide, left, top, width, height, rows, cols, data,
              header_color=DARK, header_font_color=WHITE):
    """Add a formatted table"""
    table_shape = slide.shapes.add_table(rows, cols,
                                          Inches(left), Inches(top),
                                          Inches(width), Inches(height))
    table = table_shape.table

    for i in range(rows):
        for j in range(cols):
            cell = table.cell(i, j)
            cell.text = str(data[i][j])
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE

            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(13)
                paragraph.font.name = 'Calibri'
                paragraph.alignment = PP_ALIGN.CENTER
                if i == 0:
                    paragraph.font.bold = True
                    paragraph.font.color.rgb = header_font_color
                else:
                    paragraph.font.color.rgb = DARK

            if i == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = header_color
            elif i % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT_BG
    return table

TOTAL_SLIDES = 16

# ══════════════════════════════════════════════════════════════
# SLIDE 1: Title
# ══════════════════════════════════════════════════════════════
s = add_slide(DARK)
add_textbox(s, 1.5, 1.8, 10.5, 1.5,
            "WaveSplat", 54, bold=True, color=WHITE,
            alignment=PP_ALIGN.CENTER)
add_textbox(s, 1.5, 3.0, 10.5, 1.0,
            "Grid-Free Spectral Gaussian Wave Field Estimation\nfrom Sparse Shipboard LiDAR",
            22, color=RGBColor(0xAA, 0xCC, 0xEE), alignment=PP_ALIGN.CENTER)

add_textbox(s, 1.5, 4.8, 10.5, 0.5,
            "박세용 (Saeyong Park)", 20, color=WHITE,
            alignment=PP_ALIGN.CENTER)
add_textbox(s, 1.5, 5.3, 10.5, 0.5,
            "Avikus · HD현대  |  Lab Meeting 2026", 16,
            color=GRAY, alignment=PP_ALIGN.CENTER)
add_textbox(s, 1.5, 6.2, 10.5, 0.4,
            "Target: Ocean Engineering (Elsevier)", 14,
            color=ACCENT, alignment=PP_ALIGN.CENTER)
add_page_number(s, 1, TOTAL_SLIDES)

# ══════════════════════════════════════════════════════════════
# SLIDE 2: Why — Problem & Motivation
# ══════════════════════════════════════════════════════════════
s = add_slide()
add_section_header(s, "01", "왜 이 연구가 필요한가?", "Problem & Motivation")

add_textbox(s, 0.8, 1.5, 6, 0.5,
            "선박 전방 파랑 예측이 필요한 이유", 22, bold=True, color=DARK)
add_bullet_list(s, 0.8, 2.1, 6, 3.5, [
    "• 자율운항 선박의 안전 · 승차감 · 에너지 효율",
    "• 하이드로포일 선박: 파랑 preview → feedforward 제어",
    "• 해상 크레인: 파고 예측 → 보상 제어",
    "• 해양 구조물: 파랑 하중 예측 → 구조 안전",
], font_size=16, bold_prefix=False)

add_textbox(s, 0.8, 4.3, 6, 0.5,
            "LiDAR 관측의 본질적 어려움", 22, bold=True, color=ACCENT2)
add_bullet_list(s, 0.8, 4.9, 6, 2.5, [
    "• Sparse: 200×150 m 영역에서 프레임당 ~50점",
    "• Heterogeneous: 근거리 조밀 ↔ 원거리 희박",
    "• Noisy: beam divergence + specular dropout",
    "• 10 Hz 실시간 처리 필요 (< 100 ms/frame)",
], font_size=16, bold_prefix=False)

# Right side: graphical abstract panel 1 area
try_add_image(s, 'graphical_abstract', 7.5, 1.5, 5.3, 3.0)

add_textbox(s, 7.5, 4.8, 5, 2.0,
            "목표: 희소 LiDAR 관측으로부터\n"
            "연속적이고 물리적으로 타당한\n"
            "파도장을 실시간 복원",
            18, bold=True, color=ACCENT, alignment=PP_ALIGN.CENTER)
add_page_number(s, 2, TOTAL_SLIDES)

# ══════════════════════════════════════════════════════════════
# SLIDE 3: Prior Art Limitations
# ══════════════════════════════════════════════════════════════
s = add_slide()
add_section_header(s, "01", "기존 방법의 한계", "Prior Art")

data = [
    ["Method", "Approach", "Key Limitation"],
    ["ETKF-grid\n(Desmars 2020)", "Ensemble Kalman Filter\non spatial grid", "격자 보간 → aliasing\nO(Nₑ·G²logG²), 하이퍼파라미터 민감"],
    ["GPR\n(RBF/Matérn)", "Gaussian Process\nRegression", "O(M³), 시간 모델 없음\n단순 공간 보간 → zero-predictor"],
    ["PINN / FNO\n(Ehlers 2024)", "Neural Network\n기반 추정", "학습 분포 의존\nLiDAR 불규칙 샘플링과 궁합 나쁨"],
]
add_table(s, 0.8, 1.5, 11.7, 3.5, 4, 3, data)

add_textbox(s, 0.8, 5.5, 11.7, 0.5,
            "공통 문제: 격자(grid)에 의존 → 희소 관측에서 aliasing & 불필요한 계산",
            18, bold=True, color=ACCENT2, alignment=PP_ALIGN.CENTER)

add_textbox(s, 0.8, 6.2, 11.7, 0.5,
            "→ Grid-free representation이 필요하다!",
            20, bold=True, color=ACCENT, alignment=PP_ALIGN.CENTER)
add_page_number(s, 3, TOTAL_SLIDES)

# ══════════════════════════════════════════════════════════════
# SLIDE 4: Novelty — Our Approach
# ══════════════════════════════════════════════════════════════
s = add_slide()
add_section_header(s, "02", "WaveSplat의 Novelty", "Why this is new")

add_textbox(s, 0.8, 1.5, 12, 0.6,
            "핵심 아이디어: 파도장을 N개의 spectral Gaussian (plane wave)으로 직접 파라미터화",
            18, bold=True, color=DARK)

add_textbox(s, 1.5, 2.2, 10, 0.6,
            "z(x, y, t)  =  Σᵢ  wᵢ · cos(kx,ᵢ · x  +  ky,ᵢ · y  −  ωᵢ · t  +  φᵢ)",
            22, bold=True, color=ACCENT, alignment=PP_ALIGN.CENTER,
            font_name='Consolas')

novelties = [
    "• ❶ Grid-free spectral representation — 3D Gaussian Splatting의 해양파 버전",
    "• ❷ Dispersion hard constraint: ωᵢ = √(g·|kᵢ|·tanh(|kᵢ|·d))  →  5N → 4N 파라미터",
    "• ❸ Adaptive model order: energy pruning + residual growth + safety rollback",
    "• ❹ Fresnel confidence weighting: cⱼ = cos(θⱼ)/(dⱼ/d_ref)  — LiDAR 관측 품질 모델링",
    "• ❺ Pre-registered 6-metric evaluation panel — zero-predictor 구분 가능",
    "• ❻ 24-config ETKF hyperparameter sweep — 공정한 tuned baseline 제공",
]
add_bullet_list(s, 0.8, 3.0, 11.5, 4.0, novelties, font_size=16, bold_prefix=False)
add_page_number(s, 4, TOTAL_SLIDES)

# ══════════════════════════════════════════════════════════════
# SLIDE 5: Methodology — Forward Model & Observation
# ══════════════════════════════════════════════════════════════
s = add_slide()
add_section_header(s, "03", "방법론 (1): Forward & Observation Model", "Methodology")

# Left: Forward model
add_textbox(s, 0.8, 1.5, 5.5, 0.5,
            "Forward Model", 22, bold=True, color=ACCENT)

add_textbox(s, 0.8, 2.1, 5.5, 2.5,
            "z(x, y, t) = Σᵢ wᵢ cos(kx,ᵢ x + ky,ᵢ y − ωᵢ t + φᵢ)\n\n"
            "ωᵢ = √(g |kᵢ| tanh(|kᵢ| d))     ← hard constraint\n\n"
            "파라미터: {wᵢ, kx,ᵢ, ky,ᵢ, φᵢ}  → 4N개\n"
            "(ω는 k로부터 결정되므로 자유변수 아님)",
            15, font_name='Consolas')

# Right: Observation model
add_textbox(s, 7.0, 1.5, 5.5, 0.5,
            "Observation Model (Fresnel Confidence)", 22, bold=True, color=ACCENT)

add_textbox(s, 7.0, 2.1, 5.5, 2.5,
            "cⱼ = cos(θⱼ) / (dⱼ / d_ref)\n\n"
            "L = (1/M) Σⱼ cⱼ (zⱼ − ẑⱼ)²\n"
            "    + λ₁ Σ|wᵢ|             (sparsity)\n"
            "    + λs Σ(Δwᵢ)²          (smoothness)\n\n"
            "θⱼ: beam incidence angle\n"
            "dⱼ: range,  d_ref = 125 m",
            15, font_name='Consolas')

# Bottom: why this matters
shape = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                            Inches(1.5), Inches(5.0),
                            Inches(10.3), Inches(1.5))
shape.fill.solid()
shape.fill.fore_color.rgb = LIGHT_BG
shape.line.color.rgb = ACCENT

tf = shape.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = ("물리적 의미: dispersion 제약은 '이 파도가 실제 중력파'임을 보장.\n"
          "ω를 자유롭게 놓으면 nRMSE는 낮아지지만, 1초 뒤 예측이 발산 (A1 ablation에서 확인)")
p.font.size = Pt(15)
p.font.color.rgb = DARK
p.font.name = 'Calibri'
add_page_number(s, 5, TOTAL_SLIDES)

# ══════════════════════════════════════════════════════════════
# SLIDE 6: Methodology — Pipeline (Init + Track)
# ══════════════════════════════════════════════════════════════
s = add_slide()
add_section_header(s, "03", "방법론 (2): 2-Phase Pipeline", "Init → Track")

# Phase 1
add_textbox(s, 0.8, 1.5, 5.8, 0.4,
            "Phase 1: Initialization  (T_init = 2Tₚ)", 20, bold=True, color=ACCENT)

init_steps = [
    "• ① Dispersion-aware beamforming (v2_konly)",
    "•    → polar k-grid (64 radial × 128 azimuthal)",
    "• ② Greedy peak extraction (15° suppression)",
    "• ③ Coarse-to-fine Adam: N=3 → 8 → 15",
    "•    (200 iters/stage, lr=5e-3)",
    "• ④ Energy-based pruning (90% threshold)",
    "•    + RMSE 130% safety rollback",
    "• ⑤ Residual growth: RMS > 0.05m → beamform",
    "•    성분 추가 (Nmax = 30)",
]
add_bullet_list(s, 0.8, 2.0, 5.8, 4.5, init_steps, font_size=14, bold_prefix=False)

# Phase 2
add_textbox(s, 7.0, 1.5, 5.8, 0.4,
            "Phase 2: Tracking  (w_phi mode, 10 Hz)", 20, bold=True, color=ACCENT2)

track_steps = [
    "• Predict:  φᵢ ← φᵢ + ωᵢ · Δt",
    "•   (dispersion-exact phase propagation)",
    "",
    "• Update:  K=10 Adam steps (lr=1e-3)",
    "•   on Kw=3 sliding-frame window",
    "•   → only (wᵢ, φᵢ) updated",
    "•   → kx, ky are FROZEN",
    "",
    "• Why freeze k?",
    "•   단일 프레임에서 kx/ky는 ill-conditioned",
    "•   Multi-frame init이 충분히 평균화",
    "•   φ warm start → 200 → 10 gradient steps",
]
add_bullet_list(s, 7.0, 2.0, 5.8, 4.5, track_steps, font_size=14, bold_prefix=False)

# Bottom: adaptive N
shape = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                            Inches(1.5), Inches(5.8),
                            Inches(10.3), Inches(1.0))
shape.fill.solid()
shape.fill.fore_color.rgb = LIGHT_BG
shape.line.color.rgb = GREEN

tf = shape.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Adaptive N:  N=1 (단일 swell) ~ N=21 (complex sea)  |  13.4 ms/frame 평균"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = DARK
p.alignment = PP_ALIGN.CENTER
add_page_number(s, 6, TOTAL_SLIDES)

# ══════════════════════════════════════════════════════════════
# SLIDE 7: WaveSim — LiDAR Simulation Physics
# ══════════════════════════════════════════════════════════════
s = add_slide()
add_section_header(s, "03", "시뮬레이션 데이터의 물리적 신뢰성", "WaveSim: Why We Can Trust It")

# Left column: Wave Generation
add_textbox(s, 0.8, 1.5, 5.8, 0.4,
            "파도 생성: JONSWAP + Gerstner", 20, bold=True, color=ACCENT)
wave_physics = [
    "• JONSWAP 스펙트럼 (Hasselmann et al. 1973)",
    "•   → Hs, Tₚ로 파라미터화, 해양공학 표준",
    "• Gerstner wave (1802): 2차 비선형 정확해",
    "•   → 수평 orbital 운동 포함 (trochoid)",
    "•   → Q-constraint: ΣQᵢωᵢAᵢ ≤ 1 (입자 교차 방지)",
    "• 분산관계: ω = √(g|k|tanh(|k|d))",
    "•   → Newton-Raphson 역변환, 오차 < 10⁻⁶ rad/s",
    "• 단일 성분 해석해 대비 RMSE < 10⁻¹⁰ m",
]
add_bullet_list(s, 0.8, 2.0, 5.8, 3.5, wave_physics, font_size=13, bold_prefix=False)

# Right column: LiDAR Model
add_textbox(s, 7.0, 1.5, 5.8, 0.4,
            "LiDAR 모델: Seyond Falcon K1 기반", 20, bold=True, color=ACCENT)
lidar_physics = [
    "• 실제 스펙: 1550 nm, 152 ch, 120°×25° FOV",
    "• Ray-surface 교차: Möller-Trumbore 알고리즘",
    "•   → 삼각형 메시 위 정확한 교차점 계산",
    "• 프레임당 ~76,000 ray → raydrop → ~50점 반환",
    "",
    "• Fresnel 반사 (unpolarized, n=1.321 @1550nm):",
    "•   R(0°) ≈ 2%, R(89°) > 80%  — 수면 광학 정확",
    "• 거리 noise: σ = 0.02 m (2 cm, 1550nm 표준)",
    "• 각도 noise: σ = 0.05° (beam divergence)",
]
add_bullet_list(s, 7.0, 2.0, 5.8, 3.5, lidar_physics, font_size=13, bold_prefix=False)

# Bottom: Raydrop models + validation chain
add_textbox(s, 0.8, 5.2, 6.0, 0.4,
            "두 가지 Raydrop 모델", 18, bold=True, color=ACCENT2)
add_textbox(s, 0.8, 5.6, 6.0, 1.5,
            "Uniform (diffuse):  P = η · R_Fresnel(θ) · cos(θ) / (d/d_ref)²\n"
            "  → 확산 반사 + 역제곱 감쇠 + Lambertian 투영\n\n"
            "Slope (specular):  반사각 r = d − 2(d·n)n 에서\n"
            "  retroreflection 조건 r·(−d) ≥ cos(5°) 일 때만 반환\n"
            "  → 파면 기울기에 따라 반환점 분포가 달라짐 (물리적)",
            12, font_name='Consolas')

# Validation trust chain box
shape = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                            Inches(7.0), Inches(5.2),
                            Inches(5.8), Inches(1.8))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(0xE8, 0xF5, 0xE9)
shape.line.color.rgb = GREEN

tf = shape.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "검증 체인 (39 unit tests)"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = GREEN

items = [
    "✓ 분산관계 deep/shallow 극한 해석해 일치",
    "✓ Gerstner 단일파 RMSE < 10⁻¹⁰ m",
    "✓ Fresnel R(0°) = ((n−1)/(n+1))² 정확",
    "✓ Q-constraint 위반 없음 (JONSWAP 전체)",
    "✓ 파면 법선 항상 상향 (nz > 0)",
    "✓ Raydrop 확률 ∈ [0,1], 단조 증가",
]
for item in items:
    p2 = tf.add_paragraph()
    p2.text = item
    p2.font.size = Pt(11)
    p2.font.color.rgb = DARK
    p2.space_after = Pt(1)

add_page_number(s, 7, TOTAL_SLIDES)

# ══════════════════════════════════════════════════════════════
# SLIDE 8: Experimental Setup
# ══════════════════════════════════════════════════════════════
s = add_slide()
add_section_header(s, "04", "실험 설계", "Dataset & Baselines")

# Dataset table
data_ds = [
    ["Parameter", "Value"],
    ["Total episodes", "100  (5 seeds × 10 configs × 2 raydrop)"],
    ["Sea state", "BF2~BF5  (Hs 0.3~2.0 m)"],
    ["Components", "1, 2, 5, 15, 30"],
    ["Raydrop", "Uniform (diffuse)  +  Slope (specular)"],
    ["Duration", "301 frames (30 s @ 10 Hz)"],
    ["LiDAR", "Seyond Falcon K1 spec"],
]
add_table(s, 0.8, 1.5, 5.5, 3.0, 7, 2, data_ds)

# Baseline table
data_bl = [
    ["Method", "Config", "Note"],
    ["ETKF (tuned)", "24-config sweep\nbest: g=40×30, nₑ=100", "nRMSE 0.319\n(sweep range: 3.3×)"],
    ["GPR-1", "Matérn ν=5/2\nℓ=20m, σ=0.02m", "시간 모델 없음"],
    ["GPR-sliding", "3-frame window", "강한 GPR 버전"],
]
add_table(s, 7.0, 1.5, 5.5, 2.5, 4, 3, data_bl)

# 6-metric panel
add_textbox(s, 0.8, 5.0, 12, 0.5,
            "Pre-registered 6-metric evaluation panel", 20, bold=True, color=ACCENT)
add_textbox(s, 0.8, 5.5, 12, 1.5,
            "nRMSE ↓  |  Variance Ratio (VR) → 1  |  Pearson ρ ↑  |  "
            "Skill Score (SS) ↑  |  Directional R² ↑  |  Hs error ↓\n\n"
            "핵심: SS = 1 − MSE(ẑ)/MSE(0)  →  양수면 zero-predictor(ẑ=0)보다 나은 것",
            15, color=DARK)
add_page_number(s, 8, TOTAL_SLIDES)

# ══════════════════════════════════════════════════════════════
# SLIDE 9: Main Results — 6-metric table (★ most important)
# ══════════════════════════════════════════════════════════════
s = add_slide()
add_section_header(s, "04", "핵심 결과: 6-Metric Evaluation (100 episodes)",
                   "★ Primary Result")

data_main = [
    ["Method", "nRMSE ↓", "Hs err ↓", "VR → 1", "ρ ↑", "SS ↑", "Dir R² ↑"],
    ["WaveSplat", "0.223", "0.277", "0.661", "0.446", "+0.034", "−0.253"],
    ["ETKF (tuned)", "0.319", "0.258", "0.898", "0.109", "−0.702", "+0.065"],
    ["GPR-1", "0.261", "0.780", "0.065", "0.193", "−0.140", "+0.028"],
]
add_table(s, 1.0, 1.5, 11.3, 2.5, 4, 7, data_main)

# Key observations
observations = [
    "• ✅ Skill Score: WaveSplat만 양수 (+0.034) — zero-predictor보다 나은 유일한 방법",
    "• ⚠️ GPR은 거의 zero predictor: VR=0.065 → 진폭의 6.5%만 재현",
    "• ⚠️ ETKF는 variance-correct but shape-wrong: VR=0.898 but ρ=0.109",
    "• ⚡ WaveSplat nRMSE 30.1% lower  &  ~10× faster (13.4 vs 132.6 ms)",
    "",
    "• ❗ WaveSplat 한계도 노출:",
    "•    VR=0.66 → 진폭 ~34% 과소추정 (pruning의 대가)",
    "•    Dir R²=−0.25 → 방향 성분은 chance 미만",
]
add_bullet_list(s, 1.0, 4.2, 11.3, 3.0, observations, font_size=15, bold_prefix=False)
add_page_number(s, 9, TOTAL_SLIDES)

# ══════════════════════════════════════════════════════════════
# SLIDE 10: Per-episode & Speed
# ══════════════════════════════════════════════════════════════
s = add_slide()
add_section_header(s, "04", "Per-episode 결과 & 속도", "12 Representative Episodes")

# Left: fig1 comparison
added = try_add_image(s, 'fig1_comparison', 0.5, 1.5, 6.0, 4.5)
if not added:
    add_textbox(s, 0.5, 2.0, 6.0, 4.0,
                "[fig1_comparison.png]\n\n"
                "12 에피소드 전체 WaveSplat 승\n"
                "Best case: BF3 2c — 8.5× (0.052 vs 0.444)",
                16, color=GRAY, alignment=PP_ALIGN.CENTER)

# Right: speed comparison
added2 = try_add_image(s, 'fig3_speed', 6.8, 1.5, 6.0, 4.5)
if not added2:
    add_textbox(s, 6.8, 1.5, 6.0, 4.5,
                "[fig3_speed.png]\n\n"
                "WaveSplat: 13.4 ms/frame\n"
                "ETKF (tuned): 132.6 ms (median 110)\n"
                "GPR: > 200 ms\n\n"
                "→ ~10× speed advantage",
                16, color=GRAY, alignment=PP_ALIGN.CENTER)

add_textbox(s, 0.5, 6.3, 12, 0.6,
            "WaveSplat: 12/12 승  |  Best 8.5×  |  13.4 ms vs 132.6 ms (~10× faster)",
            18, bold=True, color=ACCENT, alignment=PP_ALIGN.CENTER)
add_page_number(s, 10, TOTAL_SLIDES)

# ══════════════════════════════════════════════════════════════
# SLIDE 11: Statistical Validation
# ══════════════════════════════════════════════════════════════
s = add_slide()
add_section_header(s, "04", "통계 검증 (100 Paired Runs)", "Statistical Validation")

data_stat = [
    ["Comparison", "mean ± std", "Wilcoxon p", "Cohen's d", "95% CI"],
    ["WaveSplat", "0.219 ± 0.079", "—", "—", "—"],
    ["vs ETKF", "0.321 ± 0.041", "1.1 × 10⁻¹⁵", "1.16 (large)", "[0.085, 0.119]"],
    ["vs GPR-sliding", "0.253 ± 0.020", "4.7 × 10⁻⁴", "0.42 (sm-med)", "[0.018, 0.050]"],
]
add_table(s, 1.5, 1.5, 10.3, 2.5, 4, 5, data_stat)

# fig6 multiseed
added = try_add_image(s, 'fig6_multiseed', 1.0, 4.2, 5.5, 2.8)

explanations = [
    "• 100 paired = 10 configs × 5 seeds × 2 raydrop",
    "• 같은 조건에서 WS vs baseline → paired test",
    "",
    "• Cohen's d = 1.16 → large effect (> 0.8)",
    "•   = WS vs ETKF 차이가 매우 유의미",
    "• Cohen's d = 0.42 → small-medium",
    "•   = WS vs GPR도 통계적으로 robust",
    "• 두 CI 모두 0을 포함하지 않음",
]
add_bullet_list(s, 7.0, 4.2, 5.5, 3.0, explanations, font_size=14, bold_prefix=False)
add_page_number(s, 11, TOTAL_SLIDES)

# ══════════════════════════════════════════════════════════════
# SLIDE 12: Ablation — Dispersion Drift
# ══════════════════════════════════════════════════════════════
s = add_slide()
add_section_header(s, "04", "Ablation: Dispersion 제약의 가치", "A1 vs Full")

# Ablation table
data_abl = [
    ["Config", "avg nRMSE", "vs Full", "R̄ (dispersion drift)"],
    ["Full (dispersion ON)", "0.188", "baseline", "≡ 0 (by construction)"],
    ["A1 (no dispersion)", "0.158", "−16% nRMSE", "0.51 ± 0.12"],
    ["A2 (no confidence)", "0.215", "+14.5%", "≡ 0"],
]
add_table(s, 1.0, 1.5, 11.3, 2.2, 4, 4, data_abl)

# Key insight box
shape = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                            Inches(1.0), Inches(4.0),
                            Inches(11.3), Inches(2.5))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(0xFF, 0xF3, 0xE0)
shape.line.color.rgb = ACCENT2

tf = shape.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "A1이 nRMSE는 16% 낮다. 그런데 왜 dispersion이 필요한가?"
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = ACCENT2

p2 = tf.add_paragraph()
p2.space_before = Pt(10)
p2.text = ("R̄ = Σ|wᵢ|·|ωᵢ − ω_disp|/(ωᵢ·Σ|wⱼ|)\n\n"
           "A1: R̄ = 0.51 ± 0.12  →  성분의 약 50%가 dispersion manifold에서 이탈\n"
           "→ 절대값 ~0.10 Hz off  →  1초 뒤 phase prediction이 de-cohere\n"
           "→ 시간 추적, 파고 예측, downstream 제어에 사용 불가\n\n"
           "결론: A1의 16%는 '물리적 유효성을 포기하고 얻은 nRMSE'")
p2.font.size = Pt(14)
p2.font.color.rgb = DARK
add_page_number(s, 12, TOTAL_SLIDES)

# ══════════════════════════════════════════════════════════════
# SLIDE 13: Robustness
# ══════════════════════════════════════════════════════════════
s = add_slide()
add_section_header(s, "04", "Robustness: Raydrop · Sparsity · Noise", "")

# Raydrop
add_textbox(s, 0.8, 1.5, 3.8, 0.4,
            "Raydrop Sensitivity", 20, bold=True, color=ACCENT)
raydrop = [
    "• 10 paired episodes (U vs S)",
    "• WaveSplat: 20/20 wins",
    "• Slope: WS −9.1%, ETKF −16.2%",
    "• 두 raydrop 모두에서 안정",
]
add_bullet_list(s, 0.8, 2.0, 3.8, 2.5, raydrop, font_size=14, bold_prefix=False)

# Sparsity
add_textbox(s, 5.0, 1.5, 3.8, 0.4,
            "Sparsity Robustness", 20, bold=True, color=ACCENT)
sparsity = [
    "• 관측 밀도 20~100% 실험",
    "• BF4~5: 20% (~10점)에서도",
    "  < 8% 성능 저하",
    "• 예외: BF3 2c (N=2 해상 한계)",
]
add_bullet_list(s, 5.0, 2.0, 3.8, 2.5, sparsity, font_size=14, bold_prefix=False)

# Noise
add_textbox(s, 9.2, 1.5, 3.8, 0.4,
            "Noise Robustness", 20, bold=True, color=ACCENT)
noise = [
    "• σ ∈ {0, 0.02, 0.05, 0.10} m",
    "• σ=0.10 (5× baseline)에서도",
    "  nRMSE 변화 ±13% 이내",
    "• Adaptive N이 노이즈 흡수",
    "  → pruning이 제거",
]
add_bullet_list(s, 9.2, 2.0, 3.8, 2.5, noise, font_size=14, bold_prefix=False)

# Hyperparameter
add_textbox(s, 0.8, 4.8, 12, 0.4,
            "Hyperparameter Sensitivity", 20, bold=True, color=ACCENT)
add_textbox(s, 0.8, 5.3, 12, 1.5,
            "lr=0.01, λ₁=1e-3, τ_prune=0.10, τ_grow=0.05  — BF3 5c 한 번의 sweep으로 결정\n"
            "각 파라미터 ±2× 변경해도 nRMSE ±5% 이내  |  100 runs 전체가 동일 설정으로 재현",
            15, color=DARK)
add_page_number(s, 13, TOTAL_SLIDES)

# ══════════════════════════════════════════════════════════════
# SLIDE 14: Qualitative Result
# ══════════════════════════════════════════════════════════════
s = add_slide()
add_section_header(s, "04", "Qualitative 비교", "GT vs WaveSplat vs ETKF vs GPR")

added = try_add_image(s, 'fig8_qualitative', 0.5, 1.5, 12.3, 5.5)
if not added:
    add_textbox(s, 1.0, 3.0, 11.3, 2.0,
                "[fig8_qualitative.png]\n"
                "BF3 / BF4 / BF5  ×  GT / WaveSplat / ETKF / GPR\n"
                "heatmap comparison",
                18, color=GRAY, alignment=PP_ALIGN.CENTER)
add_page_number(s, 14, TOTAL_SLIDES)

# ══════════════════════════════════════════════════════════════
# SLIDE 15: Honest Limitations
# ══════════════════════════════════════════════════════════════
s = add_slide()
add_section_header(s, "04", "한계와 향후 과제", "Honest Limitations")

limits = [
    "• VR = 0.66 → 진폭 34% 과소추정 (energy pruning의 대가)",
    "• SS = +0.034 → 양수이지만 marginal (robust하다고 보기엔 작음)",
    "• Dir R² = −0.25 → 방향 성분이 chance 미만 (가장 큰 한계)",
    "•   → hydrofoil BF4-head 시나리오에서 feedforward gain 반전 관찰",
    "• 시뮬레이션 전용 — real LiDAR 검증 필요",
    "• 선형 모델 → Stokes-2+ nonlinearity 미반영",
    "• Nₜᵣᵤₑ ≥ 30 이상 복잡 해상에서 성분 수 한계",
    "• 해류(current) 미추정",
]
add_bullet_list(s, 0.8, 1.8, 6.0, 4.5, limits, font_size=15, bold_prefix=False)

# Right: contributions summary
add_textbox(s, 7.5, 1.5, 5.3, 0.4,
            "Contributions 요약", 20, bold=True, color=ACCENT)
contributions = [
    "• ❶ Grid-free spectral Gaussian 파면 추정",
    "• ❷ Dispersion hard constraint (5N→4N)",
    "• ❸ Adaptive N (pruning + growth)",
    "• ❹ Fresnel confidence weighting",
    "• ❺ Pre-registered 6-metric panel",
    "• ❻ 24-config ETKF sweep (fair baseline)",
    "• ❼ Dispersion drift 정량화 (R̄ metric)",
]
add_bullet_list(s, 7.5, 2.0, 5.3, 4.0, contributions, font_size=14, bold_prefix=False)

# Bottom box with key numbers
shape = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                            Inches(0.8), Inches(5.8),
                            Inches(11.7), Inches(1.2))
shape.fill.solid()
shape.fill.fore_color.rgb = LIGHT_BG
shape.line.color.rgb = ACCENT

tf = shape.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = ("Summary:  nRMSE 30.1% lower  |  ~10× faster  |  "
          "SS only positive  |  Cohen's d=1.16 (large)  |  "
          "100 paired runs, p=1.1×10⁻¹⁵")
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = DARK
p.alignment = PP_ALIGN.CENTER
add_page_number(s, 15, TOTAL_SLIDES)

# ══════════════════════════════════════════════════════════════
# SLIDE 16: Plans — Paper & PhD
# ══════════════════════════════════════════════════════════════
s = add_slide()
add_section_header(s, "05", "계획: 논문 투고 & 박사 학위", "Plans")

# Left: paper plan
add_textbox(s, 0.8, 1.5, 5.8, 0.4,
            "논문 투고 계획", 22, bold=True, color=ACCENT)
paper_plan = [
    "• Paper 1 (WaveSplat)",
    "•   Target: Ocean Engineering (Elsevier)",
    "•   Status: 40p 원고 + cover letter + graphical abstract 완료",
    "•   Next: 공동저자 sign-off → 투고",
    "",
    "• Paper 2 (Hydrofoil Feedforward)",
    "•   파면 예측 → 하이드로포일 2DOF feedforward 제어",
    "•   WaveSplat의 downstream application",
    "•   Status: 실험 완료, 초안 작성 완료",
    "•   Target 저널: 미정 (Ocean Engineering / JSR 등)",
]
add_bullet_list(s, 0.8, 2.0, 5.8, 4.5, paper_plan, font_size=14, bold_prefix=False)

# Right: PhD roadmap
add_textbox(s, 7.0, 1.5, 5.8, 0.4,
            "박사 학위 로드맵", 22, bold=True, color=ACCENT2)
phd_plan = [
    "• Core thesis:",
    "•   '선박 탑재 LiDAR 기반 실시간 파랑장 추정과",
    "•    그 제어 응용'",
    "",
    "• Paper 1: Wave estimation (WaveSplat)",
    "• Paper 2: Wave-preview control (Hydrofoil)",
    "• Paper 3+: 확장 연구",
    "•   → MPC 기반 6-DOF 제어",
    "•   → Real LiDAR 검증 (SNU 조파수조 → 실선)",
    "•   → Stokes-2 nonlinear model",
    "•   → 해류(current) 추정 통합",
]
add_bullet_list(s, 7.0, 2.0, 5.8, 4.5, phd_plan, font_size=14, bold_prefix=False)

# Bottom
add_textbox(s, 1.5, 6.5, 10.3, 0.5,
            "감사합니다. 질문 부탁드립니다.",
            22, bold=True, color=DARK, alignment=PP_ALIGN.CENTER)
add_page_number(s, 16, TOTAL_SLIDES)

# ── Save ──
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
prs.save(OUT_PATH)
print(f"Saved: {OUT_PATH}")
print(f"Slides: {TOTAL_SLIDES}")
