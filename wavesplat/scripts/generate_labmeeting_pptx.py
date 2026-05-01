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

HF_FIG_DIR = os.path.join(ROOT, 'hydrofoil', 'figures')

def try_add_hf_image(slide, fig_name, left, top, width, height=None):
    """Try to add hydrofoil figure image"""
    for ext in ['.png', '.pdf']:
        path = os.path.join(HF_FIG_DIR, fig_name + ext)
        if os.path.exists(path) and ext == '.png':
            if height:
                slide.shapes.add_picture(path, Inches(left), Inches(top),
                                        Inches(width), Inches(height))
            else:
                slide.shapes.add_picture(path, Inches(left), Inches(top),
                                        Inches(width))
            return True
    return False

TOTAL_SLIDES = 23

# ══════════════════════════════════════════════════════════════
# SLIDE 1: Title
# ══════════════════════════════════════════════════════════════
s = add_slide(DARK)
add_textbox(s, 1.5, 1.8, 10.5, 1.5,
            "WaveSplat", 54, bold=True, color=WHITE,
            alignment=PP_ALIGN.CENTER)
add_textbox(s, 1.5, 3.0, 10.5, 1.0,
            "Grid-Free Spectral Wave Field Estimation\nfrom Sparse Shipboard LiDAR",
            22, color=RGBColor(0xAA, 0xCC, 0xEE), alignment=PP_ALIGN.CENTER)

add_textbox(s, 1.5, 4.8, 10.5, 0.5,
            "박세용 (Saeyong Park)", 20, color=WHITE,
            alignment=PP_ALIGN.CENTER)
add_textbox(s, 1.5, 5.3, 10.5, 0.5,
            "Avikus · HD현대  |  서울대학교 조선해양공학과  |  Lab Meeting 2026", 16,
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
            "핵심 아이디어: 파도장을 N개의 spectral component (plane wave)로 직접 파라미터화",
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
    "• ❺ Pre-specified 6-metric evaluation panel — zero-predictor 구분 가능",
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

# Left column: Wave Generation (simplified)
add_textbox(s, 0.8, 1.5, 5.8, 0.4,
            "파도 생성: JONSWAP + Gerstner", 20, bold=True, color=ACCENT)
wave_physics = [
    "• JONSWAP 스펙트럼 → BF2~BF5 (Hs 0.3~2.0m)",
    "• Gerstner wave: 2차 비선형 정확해 (trochoid)",
    "• 분산관계: ω = √(g|k|tanh(|k|d)),  오차 < 10⁻⁶",
    "• 단일 성분 해석해 대비 RMSE < 10⁻¹⁰ m",
]
add_bullet_list(s, 0.8, 2.0, 5.8, 2.5, wave_physics, font_size=15, bold_prefix=False)

# Right column: LiDAR Model (simplified)
add_textbox(s, 7.0, 1.5, 5.8, 0.4,
            "LiDAR 모델: Seyond Falcon K1 기반", 20, bold=True, color=ACCENT)
lidar_physics = [
    "• 1550 nm, 152 ch, 120°×25° FOV",
    "• Ray-surface: Möller-Trumbore → ~50점/frame",
    "• Fresnel 반사 (n=1.321): R(0°)≈2%, R(89°)>80%",
    "• Noise: σ_range=0.02m, σ_angle=0.05°",
]
add_bullet_list(s, 7.0, 2.0, 5.8, 2.5, lidar_physics, font_size=15, bold_prefix=False)

# Bottom left: Raydrop (simplified)
add_textbox(s, 0.8, 4.8, 5.8, 0.4,
            "2가지 Raydrop 모델", 18, bold=True, color=ACCENT2)
raydrop_items = [
    "• Uniform (diffuse): Fresnel × cos(θ) / d²",
    "• Slope (specular): retroreflection 조건 기반",
]
add_bullet_list(s, 0.8, 5.3, 5.8, 1.2, raydrop_items, font_size=14, bold_prefix=False)

# Bottom right: Validation trust chain box
shape = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                            Inches(7.0), Inches(4.8),
                            Inches(5.8), Inches(2.0))
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
    "✓ 분산관계 deep/shallow 해석해 일치",
    "✓ Gerstner RMSE < 10⁻¹⁰ m",
    "✓ Fresnel R(0°) = ((n-1)/(n+1))² 정확",
    "✓ Q-constraint 위반 없음 (BF2~BF5 전체)",
    "✓ Raydrop 확률 ∈ [0,1], 물리적 일관성",
]
for item in items:
    p2 = tf.add_paragraph()
    p2.text = item
    p2.font.size = Pt(12)
    p2.font.color.rgb = DARK
    p2.space_after = Pt(2)

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
            "Pre-specified 6-metric evaluation panel", 20, bold=True, color=ACCENT)
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
    "• WaveSplat 한계도 노출:",
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
# SLIDE 15: 3 Additional Contributions
# ══════════════════════════════════════════════════════════════
s = add_slide()
add_section_header(s, "04", "추가 기여: 3가지 고유 능력 검증", "Spectral Accuracy · Sparse Robustness · Forecast")

# Left: Spectral Accuracy
add_textbox(s, 0.8, 1.5, 3.8, 0.4,
            "Spectral Accuracy", 20, bold=True, color=ACCENT)
spec_items = [
    "• 5comp 이하: energy det >93%",
    "• |k| error < 0.5%",
    "• direction error < 0.2 deg",
    "• ETKF/GPR에 없는 고유 능력",
    "  (grid-free spectral → 주파수 직접 추출)",
]
add_bullet_list(s, 0.8, 2.0, 3.8, 2.5, spec_items, font_size=14, bold_prefix=False)

# Center: Sparse Robustness
add_textbox(s, 5.0, 1.5, 3.8, 0.4,
            "Sparse Robustness (WS vs LETKF)", 18, bold=True, color=ACCENT)
sparse_items = [
    "• 관측 밀도 10~50점 비교",
    "• 10pts: WS 0.191 vs LETKF 0.239",
    "• 희소 관측에서 더 강건",
    "• Grid-free → aliasing 없음",
]
add_bullet_list(s, 5.0, 2.0, 3.8, 2.5, sparse_items, font_size=14, bold_prefix=False)

# Right: Forecast
add_textbox(s, 9.2, 1.5, 3.8, 0.4,
            "Wave Forecast", 20, bold=True, color=ACCENT)
forecast_items = [
    "• 10초 후 nRMSE 0.219",
    "• Skill Score > 0 유지",
    "• BF3 2c: SS = +0.953 @t+10s",
    "• Dispersion constraint 덕분",
    "  (A1 없으면 예측 발산)",
]
add_bullet_list(s, 9.2, 2.0, 3.8, 2.5, forecast_items, font_size=14, bold_prefix=False)

# Bottom: significance
shape = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                            Inches(1.0), Inches(5.0),
                            Inches(11.3), Inches(1.8))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(0xE8, 0xF5, 0xE9)
shape.line.color.rgb = GREEN

tf = shape.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "이 3가지 능력은 WaveSplat만의 고유한 장점 (ETKF/GPR 불가능)"
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = GREEN
p2 = tf.add_paragraph()
p2.text = ("Spectral accuracy: 파면 스펙트럼을 직접 파라미터화하므로 가능\n"
           "Sparse robustness: grid-free → 관측 밀도 무관\n"
           "Forecast: dispersion constraint가 시간적 일관성 보장 → feedforward 제어의 물리적 근거")
p2.font.size = Pt(14)
p2.font.color.rgb = DARK
add_page_number(s, 15, TOTAL_SLIDES)

# ══════════════════════════════════════════════════════════════
# SLIDE 16: Hydrofoil — Problem & Architecture
# ══════════════════════════════════════════════════════════════
s = add_slide()
add_section_header(s, "05", "논문 2: Hydrofoil Feedforward Control",
                   "Wave-Preview Control for Hydrofoil Vessels")

# Left: Why hydrofoil?
add_textbox(s, 0.8, 1.5, 5.8, 0.4,
            "왜 하이드로포일 제어에 WaveSplat?", 20, bold=True, color=ACCENT)
hf_motivation = [
    "• Hydrofoil: 수면 위 비행 → 파도 교란에 극도로 민감",
    "• Foil 잠수 깊이 0.5~1.0m → AoA 직접 변동",
    "• PD feedback만으로는 encounter freq 대응 한계",
    "  (50~100ms loop delay = 파도 주기의 5~10%)",
    "",
    "• 핵심 아이디어: WaveSplat의 spectral model로",
    "  foil 전방 파도를 미리 예측 → feedforward 보상",
    "• 자동차 preview suspension의 해양 버전",
]
add_bullet_list(s, 0.8, 2.0, 5.8, 3.5, hf_motivation, font_size=14, bold_prefix=False)

# Right: Architecture
add_textbox(s, 7.0, 1.5, 5.8, 0.4,
            "제어 구조: PD + Wave-Preview FF", 20, bold=True, color=ACCENT2)
hf_arch = [
    "• 2DOF 모델: heave + pitch",
    "• 선박: 1700kg, 22kts, tandem foil",
    "• NACA 63-series foil sections",
    "",
    "• PD feedback: collective-differential",
    "  δ_f = δ_c - δ_d,  δ_r = δ_c + δ_d",
    "",
    "• Feedforward:",
    "  x_preview = x_foil + V · Δt_prev",
    "  α_wave = -∂η/∂x - w/V  (at preview pt)",
    "  δ_ff = -K_ff · α_wave  (K_ff=0.8)",
    "",
    "• IIR LP filter (α=0.05, fc≈0.8Hz)",
    "  → 추정 노이즈 억제, 파도 신호 통과",
]
add_bullet_list(s, 7.0, 2.0, 5.8, 4.5, hf_arch, font_size=13, bold_prefix=False)
add_page_number(s, 16, TOTAL_SLIDES)

# ══════════════════════════════════════════════════════════════
# SLIDE 17: Hydrofoil — Phase 2 Results (FF Comparison)
# ══════════════════════════════════════════════════════════════
s = add_slide()
add_section_header(s, "05", "Hydrofoil 결과 (1): Feedforward 비교",
                   "PD-only vs PD+GT_FF vs PD+WS_FF vs PD+Noisy_FF")

# Results table
data_hf = [
    ["Scenario", "PD-only\naRMS [m/s²]", "GT FF\n↓%", "WS FF\n↓%", "Noisy FF\n↓%"],
    ["BF3 Head", "1.631", "+11.0%", "+11.4%", "+11.7%"],
    ["BF4 Head", "2.162", "+12.3%", "+12.2%", "+11.8%"],
    ["BF3 Beam", "0.402", "+40.7%", "+37.7%", "+27.5%"],
    ["BF3 Quarter", "1.178", "+17.6%", "+17.7%", "+17.0%"],
]
add_table(s, 1.0, 1.5, 11.3, 3.0, 5, 5, data_hf)

# Key insight
add_textbox(s, 1.0, 4.8, 11.3, 0.5,
            "WaveSplat FF → GT FF의 93~103% 성능 달성!", 22,
            bold=True, color=GREEN, alignment=PP_ALIGN.CENTER)

hf_obs = [
    "• WS 20% nRMSE로도 GT 대비 93~103% feedforward 효과 회복",
    "• Beam sea에서 가장 큰 개선 (+37.7%) — 파면이 평행하여 추정 유리",
    "• Noisy FF (인위적 노이즈 주입)도 LP filter 덕분에 효과적",
    "• BF3 Beam: ISO 2631 'Slight discomfort' → 'Comfortable' 달성 (aRMS < 0.315)",
]
add_bullet_list(s, 1.0, 5.4, 11.3, 2.0, hf_obs, font_size=14, bold_prefix=False)

# Add phase 2 comparison figure
try_add_hf_image(s, 'fig_phase2_comparison', 9.5, 4.8, 3.3, 2.2)
add_page_number(s, 17, TOTAL_SLIDES)

# ══════════════════════════════════════════════════════════════
# SLIDE 18: Hydrofoil — Sensitivity Analysis
# ══════════════════════════════════════════════════════════════
s = add_slide()
add_section_header(s, "05", "Hydrofoil 결과 (2): Sensitivity Analysis",
                   "nRMSE Tolerance & Preview Horizon")

# Left: nRMSE sensitivity
add_textbox(s, 0.8, 1.5, 5.8, 0.4,
            "추정 오차 허용 범위", 20, bold=True, color=ACCENT)
nrmse_sens = [
    "• nRMSE 0~50% 전 구간에서 FF 유효",
    "• 50% nRMSE에서도 FF 효과 존재",
    "  → threshold 없음 (graceful degradation)",
    "• LP filter가 고주파 추정 노이즈 억제",
    "  → 부정확한 추정도 유용하게 활용",
    "",
    "• 실용적 의미:",
    "  WaveSplat의 20% nRMSE는 충분",
    "  더 정확할 필요 없이 제어에 활용 가능",
]
add_bullet_list(s, 0.8, 2.0, 5.8, 3.5, nrmse_sens, font_size=14, bold_prefix=False)
try_add_hf_image(s, 'fig_nrmse_sensitivity', 0.8, 5.0, 5.5, 2.2)

# Right: Preview distance
add_textbox(s, 7.0, 1.5, 5.8, 0.4,
            "최적 Preview Horizon", 20, bold=True, color=ACCENT)
preview_sens = [
    "• Head sea: 최적 0.1s (~1.1m ahead)",
    "• Beam sea: 최적 0.3s (~3.4m ahead)",
    "• 물리적 의미: 파도-foil 상호작용 시간 스케일",
    "",
    "• 너무 긴 preview → 위상 불일치",
    "  너무 짧은 preview → reaction 부족",
    "• Encounter freq에 따라 최적점 변화",
    "",
    "• 실용: 0.1~0.3s면 충분",
    "  (cruise speed에서 1~3m 전방)",
]
add_bullet_list(s, 7.0, 2.0, 5.8, 3.5, preview_sens, font_size=14, bold_prefix=False)
try_add_hf_image(s, 'fig_preview_sweep', 7.0, 5.0, 5.5, 2.2)
add_page_number(s, 18, TOTAL_SLIDES)

# ══════════════════════════════════════════════════════════════
# SLIDE 19: Hydrofoil — E2E Demo
# ══════════════════════════════════════════════════════════════
s = add_slide()
add_section_header(s, "05", "Hydrofoil 결과 (3): End-to-End Demo",
                   "LiDAR → WaveSplat → Feedforward → Ride Control")

# E2E results
add_textbox(s, 0.8, 1.5, 6.0, 0.4,
            "End-to-End Pipeline 결과", 20, bold=True, color=ACCENT)

e2e_data = [
    ["Scenario", "WS/GT ratio", "평가"],
    ["BF3 2-comp", "96%", "near-perfect (simple sea)"],
    ["BF3 beam", "92%", "excellent"],
    ["BF4 5-comp", "37%", "limited (direction error)"],
]
add_table(s, 0.8, 2.0, 5.5, 2.2, 4, 3, e2e_data)

e2e_obs = [
    "• BF3: WaveSplat E2E → GT의 92~96% 달성",
    "• BF4: 37% — directional misestimation 영향",
    "  → Dir R² = -0.25 문제가 제어에 전파",
    "",
    "• 핵심 발견:",
    "  amplitude 정확도보다 direction 정확도가",
    "  feedforward 성능에 더 중요",
]
add_bullet_list(s, 0.8, 4.5, 5.5, 2.5, e2e_obs, font_size=14, bold_prefix=False)

# Right: E2E timeseries figure
try_add_hf_image(s, 'fig_e2e_timeseries', 6.8, 1.5, 6.0, 5.5)

add_page_number(s, 19, TOTAL_SLIDES)

# ══════════════════════════════════════════════════════════════
# SLIDE 20: Hydrofoil Animation Demo
# ══════════════════════════════════════════════════════════════
s = add_slide()
add_section_header(s, "05", "Hydrofoil 시뮬레이션 데모",
                   "PD-only vs PD+Feedforward — BF3 Head Sea Animation")

# Try to embed the GIF as a static frame; note actual GIF playback needs manual
ANIM_PATH = os.path.join(ROOT, 'docs', 'hydrofoil_wave_animation.gif')
if os.path.exists(ANIM_PATH):
    try:
        from PIL import Image
        img = Image.open(ANIM_PATH)
        # Save first frame as PNG for embedding
        first_frame_path = os.path.join(ROOT, 'docs', '_anim_first_frame.png')
        img.save(first_frame_path)
        s.shapes.add_picture(first_frame_path, Inches(1.0), Inches(1.5),
                             Inches(11.3), Inches(5.0))
    except Exception:
        add_textbox(s, 1.0, 2.5, 11.3, 3.0,
                    "[Animation: docs/hydrofoil_wave_animation.gif]\n\n"
                    "파도 위를 달리는 하이드로포일 선박\n"
                    "빨강 = PD-only  /  초록 = PD+Feedforward\n"
                    "하단: heave acceleration 실시간 비교\n\n"
                    "→ Feedforward로 가속도 ~48% 감소",
                    18, color=DARK, alignment=PP_ALIGN.CENTER)
else:
    add_textbox(s, 1.0, 2.5, 11.3, 3.0,
                "[Animation: docs/hydrofoil_wave_animation.gif]\n\n"
                "파도 위를 달리는 하이드로포일 선박\n"
                "빨강 = PD-only  /  초록 = PD+Feedforward\n"
                "하단: heave acceleration 실시간 비교\n\n"
                "→ Feedforward로 가속도 ~48% 감소",
                18, color=DARK, alignment=PP_ALIGN.CENTER)

add_textbox(s, 1.0, 6.5, 11.3, 0.5,
            "※ 실제 발표 시 GIF 파일(docs/hydrofoil_wave_animation.gif)을 삽입하여 재생",
            12, color=GRAY, alignment=PP_ALIGN.CENTER)
add_page_number(s, 20, TOTAL_SLIDES)

# ══════════════════════════════════════════════════════════════
# SLIDE 21: Honest Limitations
# ══════════════════════════════════════════════════════════════
s = add_slide()
add_section_header(s, "06", "한계와 향후 과제", "Honest Limitations")

limits_ws = [
    "• WaveSplat 추정 한계:",
    "•   VR = 0.66 → 진폭 34% 과소추정 (pruning 대가)",
    "•   Dir R² = −0.25 → 방향 추정이 가장 큰 약점",
    "•     → 개선: multi-frame 방향 voting / 2D k-space 정규화",
    "•   진폭 감쇠 (Adam momentum 부호 진동)",
    "•   Nₜᵣᵤₑ ≥ 30 이상 복잡 해상 → 성분 수 한계",
    "",
    "• Hydrofoil 제어 한계:",
    "•   BF4 E2E → 37% (direction error 전파)",
    "•   2DOF (heave+pitch): roll은 별도 roll foil 담당 구조",
    "•   선형 foil model (ventilation 간소화)",
    "",
    "• 공통 한계:",
    "•   시뮬레이션 전용 → 실측 검증 필요",
    "•   선형 파도 모델 → Stokes-2+ 미반영",
    "•   해류/바람 변화에 따른 k 천이 미추정",
]
add_bullet_list(s, 0.8, 1.8, 6.0, 5.0, limits_ws, font_size=13, bold_prefix=False)

# Right: contributions summary (both papers)
add_textbox(s, 7.5, 1.5, 5.3, 0.4,
            "전체 Contributions (논문 1+2)", 20, bold=True, color=ACCENT)
contributions = [
    "• [논문1] ❶ Grid-free spectral 파면 추정",
    "• [논문1] ❷ Dispersion hard constraint (5N→4N)",
    "• [논문1] ❸ Adaptive N (pruning + growth)",
    "• [논문1] ❹ Fresnel confidence weighting",
    "• [논문1] ❺ Pre-specified 6-metric panel",
    "• [논문1] ❻ 24-config ETKF sweep (fair baseline)",
    "• [논문1] ❼ Spectral accuracy + Sparse robustness + Forecast",
    "",
    "• [논문2] ❽ Dual-actuator wave-preview FF architecture",
    "• [논문2] ❾ nRMSE tolerance + preview horizon 설계 지침",
    "• [논문2] ❿ E2E LiDAR→추정→제어 파이프라인 실증",
]
add_bullet_list(s, 7.5, 2.0, 5.3, 4.5, contributions, font_size=12, bold_prefix=False)

# Bottom box with key numbers
shape = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                            Inches(0.8), Inches(6.0),
                            Inches(11.7), Inches(1.0))
shape.fill.solid()
shape.fill.fore_color.rgb = LIGHT_BG
shape.line.color.rgb = ACCENT

tf = shape.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = ("WaveSplat: nRMSE 30.1%↓, ~10× faster, SS only+  |  "
          "Hydrofoil: aRMS 11~41%↓, BF3 E2E 92~96%")
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = DARK
p.alignment = PP_ALIGN.CENTER
add_page_number(s, 21, TOTAL_SLIDES)

# ══════════════════════════════════════════════════════════════
# SLIDE 22: Thesis Structure
# ══════════════════════════════════════════════════════════════
s = add_slide()
add_section_header(s, "07", "학위논문 구조",
                   "선박 탑재 LiDAR 기반 실시간 파랑장 추정과 그 제어 응용")

# Left: thesis chapters
add_textbox(s, 0.8, 1.5, 5.8, 0.4,
            "학위논문 목차", 22, bold=True, color=ACCENT)
thesis_ch = [
    "• Ch.1 서론: 연구 배경, 목적, 범위",
    "• Ch.2 문헌 조사: 파랑 추정 + 제어 선행연구",
    "",
    "• Ch.3 WaveSplat (논문 1)",
    "•   시뮬레이션, 알고리즘, 실험, 분석",
    "•   + 3 추가 기여 (spectral/sparse/forecast)",
    "",
    "• Ch.4 Hydrofoil FF Control (논문 2)",
    "•   동역학 모델, 제어 설계, sensitivity, E2E",
    "",
    "• Ch.5 결론 및 향후 연구",
    "• 부록: 파라미터 표, 추가 그래프",
]
add_bullet_list(s, 0.8, 2.0, 5.8, 4.5, thesis_ch, font_size=14, bold_prefix=False)

# Right: current status
add_textbox(s, 7.0, 1.5, 5.8, 0.4,
            "현재 진행 상태", 22, bold=True, color=ACCENT2)

status_items = [
    "• 논문 1 (WaveSplat)",
    "•   ✅ 코드/실험 완료 (57 tests pass)",
    "•   ✅ 논문 작성 완료 (Ocean Eng. format)",
    "•   ✅ 적대적 페르소나 10라운드 검토 완료",
    "•   ⏳ 공동저자 sign-off → 투고",
    "",
    "• 논문 2 (Hydrofoil)",
    "•   ✅ 시뮬레이션 완료 (Phase 1~3)",
    "•   ✅ 초안 작성 완료",
    "•   ⏳ 공동저자 검토 → 투고",
    "",
    "• 학위논문",
    "•   ✅ 초안 완성 (203K lines, PDF 생성)",
    "•   ⏳ 지도교수 검토",
]
add_bullet_list(s, 7.0, 2.0, 5.8, 4.5, status_items, font_size=13, bold_prefix=False)
add_page_number(s, 22, TOTAL_SLIDES)

# ══════════════════════════════════════════════════════════════
# SLIDE 23: Plans & Future Work
# ══════════════════════════════════════════════════════════════
s = add_slide()
add_section_header(s, "07", "향후 계획", "Plans & Future Work")

# Left: immediate plans
add_textbox(s, 0.8, 1.5, 5.8, 0.4,
            "단기 계획 (2026)", 22, bold=True, color=ACCENT)
near_plan = [
    "• 논문 1 투고 (Ocean Engineering)",
    "• 논문 2 투고 (Ocean Engineering / JSR)",
    "• 수조 실험 검증 (SNU 조파수조, 2026 하반기)",
    "•   → 규칙파/불규칙파 3~5 조건",
    "•   → 용량식 wave gauge GT + Seyond Falcon K1",
    "•   → Sim-to-Real gap 정량화 (raydrop, noise)",
]
add_bullet_list(s, 0.8, 2.0, 5.8, 3.0, near_plan, font_size=15, bold_prefix=False)

# Right: long-term
add_textbox(s, 7.0, 1.5, 5.8, 0.4,
            "확장 연구 방향", 22, bold=True, color=ACCENT2)
future_plan = [
    "• MPC 기반 6-DOF 통합 제어",
    "•   → preview information의 최적 활용",
    "• Stokes-2 nonlinear wave model",
    "•   → 급경사파, 쇄파 환경 대응",
    "• 해류(current) 동시 추정",
    "• 실선 탑재 → 자율운항 통합",
]
add_bullet_list(s, 7.0, 2.0, 5.8, 3.0, future_plan, font_size=15, bold_prefix=False)

# Bottom: PhD thesis title
shape = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                            Inches(1.5), Inches(5.2),
                            Inches(10.3), Inches(1.5))
shape.fill.solid()
shape.fill.fore_color.rgb = DARK
shape.line.fill.background()

tf = shape.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "박사학위 논문 주제"
p.font.size = Pt(14)
p.font.color.rgb = ACCENT
p.alignment = PP_ALIGN.CENTER
p2 = tf.add_paragraph()
p2.text = "선박 탑재 LiDAR 기반 실시간 파랑장 추정과 그 제어 응용"
p2.font.size = Pt(22)
p2.font.bold = True
p2.font.color.rgb = WHITE
p2.alignment = PP_ALIGN.CENTER
p3 = tf.add_paragraph()
p3.text = "\n감사합니다. 질문 부탁드립니다."
p3.font.size = Pt(18)
p3.font.color.rgb = RGBColor(0xAA, 0xCC, 0xEE)
p3.alignment = PP_ALIGN.CENTER
add_page_number(s, 23, TOTAL_SLIDES)

# ── Save ──
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
prs.save(OUT_PATH)
print(f"Saved: {OUT_PATH}")
print(f"Slides: {TOTAL_SLIDES}")
