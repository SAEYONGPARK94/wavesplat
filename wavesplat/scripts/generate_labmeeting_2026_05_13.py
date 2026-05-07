"""
generate_labmeeting_2026_05_13.py

Lab meeting presentation 2026-05-13 — WaveSplat exploration phase + new framing
Target: 외부 청자 20분 발표 (1 slide ≈ 1분)

Structure:
  Part 1 (S1-S10): Motivation + Prior art (5-category + 5 detailed papers + matrix + niche)
  Part 2 (S11-S12): WaveSim
  Part 3 (S13-S14): WaveSplat exploration phase
  Part 4 (S15-S17): New framing — Dispersion-Coupled Spatio-Temporal Inversion
  Closing (S18-S19): Limitations + timeline + Q&A
  Backup (A1-A7): supporting prior art papers

Output: docs/labmeeting_2026_05_13.pptx

Source content: docs/labmeeting_2026_05_13_slides.md
"""
import os
import sys

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, '..')
DOC_DIR = os.path.join(ROOT, 'docs')
PAPER_FIG_DIR = os.path.join(ROOT, 'paper', 'figures')
OUT_PATH = os.path.join(DOC_DIR, 'labmeeting_2026_05_13.pptx')

# ── colors ──
DARK = RGBColor(0x1A, 0x1A, 0x2E)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ACCENT = RGBColor(0x00, 0x7A, 0xCC)
ACCENT2 = RGBColor(0xE8, 0x55, 0x3A)
LIGHT_BG = RGBColor(0xF0, 0xF4, 0xF8)
GRAY = RGBColor(0x66, 0x66, 0x66)
GREEN = RGBColor(0x2C, 0xA0, 0x2C)
RED = RGBColor(0xCC, 0x33, 0x33)
GOLD = RGBColor(0xD4, 0xA0, 0x17)

# ── presentation ──
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

TOTAL_MAIN = 19  # main slides (excluding backup)


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def add_blank(bg_color=WHITE):
    layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(layout)
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = bg_color
    return slide


def add_text(slide, left, top, width, height, text, font_size=18,
             bold=False, color=DARK, alignment=PP_ALIGN.LEFT,
             italic=False, font_name='Calibri'):
    box = slide.shapes.add_textbox(Inches(left), Inches(top),
                                   Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.italic = italic
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = alignment
    return tf


def add_bullets(slide, left, top, width, height, items, font_size=15,
                color=DARK, bullet_char='•', line_space_after=4):
    """items: list of strings. '**...**' = bold."""
    box = slide.shapes.add_textbox(Inches(left), Inches(top),
                                   Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.space_after = Pt(line_space_after)
        # Add bullet prefix
        prefix = f"{bullet_char} " if not item.startswith('  ') else "  - "
        item_clean = item.lstrip()
        # Handle '**bold**'
        if '**' in item_clean:
            parts = item_clean.split('**')
            run0 = p.add_run()
            run0.text = prefix
            run0.font.size = Pt(font_size)
            run0.font.color.rgb = color
            run0.font.name = 'Calibri'
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
            run.text = prefix + item_clean
            run.font.size = Pt(font_size)
            run.font.color.rgb = color
            run.font.name = 'Calibri'
    return tf


def add_page_number(slide, num, total=TOTAL_MAIN):
    add_text(slide, 12.5, 7.05, 0.7, 0.35,
             f"{num} / {total}", font_size=10, color=GRAY,
             alignment=PP_ALIGN.RIGHT)


def add_section_tag(slide, tag, color=ACCENT):
    """Top-left small section indicator."""
    add_text(slide, 0.5, 0.2, 4, 0.3, tag,
             font_size=11, bold=True, color=color)


def add_title(slide, title, y=0.5, color=DARK, size=26):
    add_text(slide, 0.5, y, 12.3, 0.8, title,
             font_size=size, bold=True, color=color)


def add_subtitle(slide, subtitle, y=1.25, color=GRAY, size=14, italic=True):
    add_text(slide, 0.5, y, 12.3, 0.5, subtitle,
             font_size=size, color=color, italic=italic)


def add_image_or_placeholder(slide, fig_path, left, top, width, height=None,
                             placeholder_text='[FIGURE]'):
    if fig_path and os.path.exists(fig_path):
        if height is None:
            slide.shapes.add_picture(fig_path, Inches(left), Inches(top),
                                     width=Inches(width))
        else:
            slide.shapes.add_picture(fig_path, Inches(left), Inches(top),
                                     width=Inches(width), height=Inches(height))
        return True
    else:
        # placeholder rectangle
        from pptx.enum.shapes import MSO_SHAPE
        h = height if height else width * 0.6
        shape = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            Inches(left), Inches(top), Inches(width), Inches(h))
        shape.fill.solid()
        shape.fill.fore_color.rgb = LIGHT_BG
        shape.line.color.rgb = GRAY
        tf = shape.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = placeholder_text
        run.font.size = Pt(11)
        run.font.color.rgb = GRAY
        run.font.italic = True
        return False


def add_notes(slide, text):
    """Speaker notes (presenter notes)."""
    notes_slide = slide.notes_slide
    tf = notes_slide.notes_text_frame
    tf.text = text


def add_table_simple(slide, left, top, width, height, headers, rows,
                     header_color=ACCENT, font_size=11):
    cols = len(headers)
    n_rows = len(rows) + 1  # +1 for header
    table = slide.shapes.add_table(n_rows, cols,
                                   Inches(left), Inches(top),
                                   Inches(width), Inches(height)).table
    # header row
    for c, h in enumerate(headers):
        cell = table.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_color
        tf = cell.text_frame
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = str(h)
        run.font.size = Pt(font_size)
        run.font.bold = True
        run.font.color.rgb = WHITE
    # data rows
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = table.cell(r + 1, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT_BG if r % 2 == 0 else WHITE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            run = p.add_run()
            run.text = str(val)
            run.font.size = Pt(font_size)
            run.font.color.rgb = DARK
    return table


# ─────────────────────────────────────────
# Slide builders
# ─────────────────────────────────────────

# S1 — Title
def build_s1():
    s = add_blank(LIGHT_BG)
    add_text(s, 0.5, 2.0, 12.3, 1.0,
             "Phase-Resolved Ocean Wave Estimation",
             font_size=36, bold=True, color=DARK,
             alignment=PP_ALIGN.CENTER)
    add_text(s, 0.5, 3.0, 12.3, 0.7,
             "from Sparse Shipboard LiDAR",
             font_size=30, bold=True, color=ACCENT,
             alignment=PP_ALIGN.CENTER)
    add_text(s, 0.5, 4.0, 12.3, 0.5,
             "Exploration phase results & a path forward",
             font_size=18, italic=True, color=GRAY,
             alignment=PP_ALIGN.CENTER)
    add_text(s, 0.5, 5.5, 12.3, 0.4,
             "Saeyong Park",
             font_size=18, bold=True, color=DARK,
             alignment=PP_ALIGN.CENTER)
    add_text(s, 0.5, 6.0, 12.3, 0.4,
             "Lab Meeting · 2026-05-13",
             font_size=14, color=GRAY,
             alignment=PP_ALIGN.CENTER)
    add_notes(s, (
        "오늘 발표는 두 부분으로 나뉩니다. 앞 절반은 우리가 6개월 동안 진행한 "
        "exploration phase 의 결과 — 어떤 prior art 가 있고, 우리가 만든 "
        "simulator (WaveSim) 와 estimator (WaveSplat) 이 어떻게 작동하는지. "
        "뒤 절반은 그 과정에서 발견한 본질적 mechanism — vessel motion 과 "
        "dispersion 의 결합이 어떻게 spatial Nyquist 한계를 우회하는지, "
        "그리고 이를 main contribution 으로 격상시킬 publication plan. "
        "외부 청자라 해서 도메인 background 부터 시작하겠습니다."
    ))


# S2 — Motivation
def build_s2():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 1 · Motivation")
    add_title(s, "Phase-resolved waves enable look-ahead control")
    add_subtitle(s, "Statistical descriptions are not enough — we need where each crest is, when")
    bullets = [
        "**Statistical (H_s, T_p, JONSWAP):** good for design, insufficient for real-time control",
        "**Phase-resolved η(x, y, t):** deterministic, enables look-ahead window (0–30 s)",
        "",
        "**Direct applications:**",
        "  Hydrofoil ride control — wave preview → feed-forward → ride RMS ↓",
        "  Autonomous vessel — hazardous wave avoidance, course planning",
        "  Offshore operations — lifting / heli landing window decisions",
        "  Wave-energy converter — optimal damping tuning",
        "",
        "**Common requirement:** real-time + look-ahead window > horizon of control loop",
    ]
    add_bullets(s, 0.6, 1.9, 12.0, 5.0, bullets, font_size=15)
    fig_hf = os.path.join(DOC_DIR, 'hydrofoil_controller_flow.png')
    add_image_or_placeholder(s, fig_hf, 8.5, 4.2, 4.3,
                             placeholder_text='[hydrofoil_controller_flow.png]')
    add_page_number(s, 2)
    add_notes(s, (
        "파도 추정 분야에는 두 layer 가 있습니다. 첫째 statistical estimation — "
        "H_s, T_p, JONSWAP spectrum 같은 통계량. 둘째 phase-resolved estimation — "
        "η(x,y,t) 자체를 시공간상에서 결정론적으로 추정. 통계는 design 에 필수지만 "
        "실시간 control 에는 부족합니다. 어디에 큰 파도가 오는지 알아야 ride control, "
        "hydrofoil, autonomous navigation 같은 응용이 가능. 응용으로는 hydrofoil "
        "ride control 이 가장 직접적입니다."
    ))


# S3 — Problem
def build_s3():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 1 · Problem")
    add_title(s, "Three compounding challenges of shipboard LiDAR")
    add_subtitle(s, "Sparse · Real-time · 2D directional — three conditions, simultaneously")
    bullets = [
        "**Challenge 1 — Sparse, non-uniform measurements**",
        "  ~50 valid points / frame after raydrop, shadowing, Fresnel filtering",
        "  Distance-dependent density: 근거리 dense, 원거리 sparse",
        "  Range-limited ~250 m vs ocean wavelength 50–300 m",
        "",
        "**Challenge 2 — Real-time constraint**",
        "  Control loop horizon: ~10 Hz update",
        "  Per-frame budget: < 100 ms",
        "  Offline CFD/PINN/EnKF 들은 분~시간 단위",
        "",
        "**Challenge 3 — 2D directional sea state**",
        "  Real ocean: short-crested, multi-directional",
        "  1D unidirectional 가정은 부족 (Klein 2020, Desmars 2023 대부분 1D)",
        "  2D (k_x, k_y) 추정 필요",
    ]
    add_bullets(s, 0.6, 1.9, 8.0, 5.0, bullets, font_size=14)
    fig_lidar = os.path.join(DOC_DIR, 'wavesim_BF4_5c_wave_lidar.png')
    add_image_or_placeholder(s, fig_lidar, 8.8, 2.0, 4.2,
                             placeholder_text='[sparse LiDAR scan]')
    add_page_number(s, 3)
    add_notes(s, (
        "이 3가지가 동시에 만족되어야 하는 setup 은 prior art 에 거의 없습니다. "
        "sparse non-uniform — LiDAR 가 wave gauge 나 dense radar 와 다르게 frame "
        "당 50점 정도, 거리 따라 density 다름. real-time — control loop 에 "
        "들어가야 하므로 < 100 ms. 2D directional — 실제 바다는 directional "
        "spectrum. 다음 슬라이드부터 prior art 12편을 검토한 결과를 보여드립니다."
    ))


# S4 — 5-category landscape
def build_s4():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 1 · Prior art landscape")
    add_title(s, "Phase-resolved wave estimation — 5 categories of prior art")
    add_subtitle(s, "12 paper review — we belong to the 5th: parametric off-grid")
    headers = ["#", "Category", "Examples", "Strength", "Limitation"]
    rows = [
        ["1", "Variational / grid-based",
         "Desmars 2018-23, Aragh 2008",
         "Mature theory, on-grid", "predefined k-grid, 1D"],
        ["2", "Ensemble Kalman filter",
         "Wang-Pan 2021/22, Yan 2025",
         "Full nonlinear physics", "Offline, ensemble cost"],
        ["3", "ML supervised",
         "Ehlers 2023 FNO, Duan 2020",
         "Fast inference (1.9 ms)", "Training data, generalization"],
        ["4", "Physics-informed unsupervised",
         "Ehlers 2025 PFT-PINN",
         "No training data, full nonlinear", "~30 min/instance, 1D"],
        ["5", "Parametric off-grid (ours) ⭐",
         "WaveSplat",
         "Real-time + 2D + LiDAR", "Linear approximation"],
    ]
    add_table_simple(s, 0.5, 1.9, 12.3, 4.5, headers, rows,
                     header_color=ACCENT, font_size=11)
    add_text(s, 0.5, 6.6, 12.3, 0.4,
             "Axis: fidelity (linear → fully nonlinear) × inference speed (offline → real-time). We sit at top-left + add 2D + LiDAR.",
             font_size=11, italic=True, color=GRAY)
    add_page_number(s, 4)
    add_notes(s, (
        "12 paper 정독 결과 5 카테고리로 분류 가능합니다. 첫 4개는 기존 paradigm — "
        "variational on-grid, EnKF ensemble, ML supervised, physics-informed "
        "unsupervised. 우리는 5번째 카테고리, parametric off-grid 입니다. "
        "다른 카테고리들은 grid 또는 NN 으로 wave field 를 표현하지만, 우리는 "
        "spectral parameter 자체를 직접 추정. 이게 가능한 이유는 dispersion 제약을 "
        "hard constraint 로 강제하기 때문 — 다음 5 슬라이드에서 가장 가까운 "
        "prior art 5 편을 detail 로 보여드립니다."
    ))


# Helper: standardized prior-art slide
def build_prior_art(num, title_full, key_msg, limits, contrast,
                    fig_path=None, notes='', star=False):
    s = add_blank(WHITE)
    add_section_tag(s, "Part 1 · Prior art detail")
    if star:
        add_title(s, title_full + "  ⭐")
    else:
        add_title(s, title_full)
    # Key message
    add_text(s, 0.5, 1.3, 12.3, 0.4, "Key message",
             font_size=12, bold=True, color=ACCENT)
    add_bullets(s, 0.6, 1.7, 7.5, 1.7, key_msg, font_size=13)
    # Figure placeholder
    if fig_path:
        add_image_or_placeholder(s, fig_path, 8.4, 1.7, 4.5,
                                 placeholder_text='[paper figure placeholder]')
    else:
        add_image_or_placeholder(s, None, 8.4, 1.7, 4.5,
                                 placeholder_text='[paper figure — to insert]')
    # Limitations
    add_text(s, 0.5, 4.1, 12.3, 0.4, "Limitations (our perspective)",
             font_size=12, bold=True, color=RED)
    add_bullets(s, 0.6, 4.5, 12.0, 1.3, limits, font_size=13, color=DARK)
    # Contrast
    add_text(s, 0.5, 5.9, 12.3, 0.4, "vs WaveSplat",
             font_size=12, bold=True, color=GREEN)
    add_bullets(s, 0.6, 6.3, 12.0, 0.9, contrast, font_size=13, color=DARK)
    add_page_number(s, num)
    add_notes(s, notes)


def build_s5():
    build_prior_art(
        num=5,
        title_full="Desmars et al. 2023 (JFM 975, A8) — HOS-OCM",
        key_msg=[
            "**First vessel-mounted optical** phase-resolved reconstruction + prediction with HOS",
            "Setup: shipborne LiDAR/camera, 1D longitudinal, predefined k-grid",
            "Variational DA → cost function inversion → HOS predict",
        ],
        limits=[
            "**On-grid k-band assumption** — wavenumber 가 predefined grid 에 묶임",
            "**1D unidirectional** — 2D directional sea state 처리 안 함",
            "**Optical sensor model** 명시 없음 — raydrop/shadowing/Fresnel 가정 미상",
        ],
        contrast=[
            "Ours: **off-grid continuous (k_x, k_y)** + **2D directional** + **full LiDAR sensor model**",
        ],
        fig_path=None,
        notes=(
            "12 paper 중 우리와 가장 가까운 prior art. Setup 자체가 거의 동일 — "
            "선박 mounted optical sensor 로 phase-resolved wave 를 추정하고 HOS 로 "
            "predict. 차이는 1D vs 우리 2D, on-grid vs 우리 off-grid, sensor model "
            "detail. 우리 paper 의 main reference 가 됩니다."
        ),
        star=True,
    )


def build_s6():
    build_prior_art(
        num=6,
        title_full="Wang & Pan 2021 (JFM 918) + 2022 (JFM 949) — EnKF-HOS",
        key_msg=[
            "**First EnKF + HOS integration** — ensemble-based DA on phase-resolved waves",
            "2022 extension: EnKF-HOS-C with **current estimation**",
            "GPU implementation, near-real-time (단 dense observation 가정)",
        ],
        limits=[
            "**Dense observation 가정** — sparse LiDAR setup 직접 적용 어려움",
            "**1D wave** — 2D directional 처리 안 함",
            "**Ensemble cost** — 50–100 members × HOS time integration",
        ],
        contrast=[
            "Ours: **sparse LiDAR input** + **2D directional** + **<100 ms inference**",
            "Note: Wang 2025 (Coastal Eng 197) 정독 → paper 작성 단계 (Case A/B/C 결정)",
        ],
        fig_path=None,
        notes=(
            "EnKF-HOS family 는 우리 ETKF baseline 의 직접적 prior art. Wang & Pan 2021 "
            "이 처음으로 HOS 와 EnKF 를 결합했고, 2022 에서 current estimation 으로 "
            "확장. 한계는 ensemble 비용으로 real-time 어렵고, dense obs 가정이라 "
            "sparse LiDAR setup 에 직접 적용 안 됨. 우리는 이 baseline 을 sparse-input "
            "으로 tune 해서 비교."
        ),
        star=True,
    )


def build_s7():
    build_prior_art(
        num=7,
        title_full="Ehlers et al. 2023 (Ocean Eng) — Fourier Neural Operator",
        key_msg=[
            "**First ML-based phase-resolved reconstruction** from sparse X-band radar",
            "FNO architecture (Fourier global mapping) outperforms U-Net (CNN local)",
            "**Inference 1.9 ms** / instance — 빠름. 단 supervised pre-training 필요",
        ],
        limits=[
            "**Supervised** — 3120 HOS samples 학습, distribution shift risk",
            "**1D unidirectional**, no 2D directional",
            "**Black-box** — physical interpretability 부족",
        ],
        contrast=[
            "Ours: **unsupervised** (no training), **interpretable parametric**, **2D directional**",
            "Speed comparable: ~13 ms vs 1.9 ms — 둘 다 real-time 영역",
        ],
        fig_path=None,
        notes=(
            "ML supervised 카테고리의 SOTA. 1.9 ms 의 inference 속도가 인상적이지만 "
            "3120 HOS sample 학습이 필요하고 out-of-distribution sea state 에 약함. "
            "우리는 training 자체가 없어 generalization 문제 없음. 1D 만 처리, "
            "우리는 2D directional. 둘 다 real-time 영역인 점에서 paper 1 의 main competitor."
        ),
        star=True,
    )


def build_s8():
    build_prior_art(
        num=8,
        title_full="Ehlers et al. 2025 (Phys. Rev. Fluids 10, 094901) — PFT-PINN",
        key_msg=[
            "**Two parallel NNs** (η-net + Φ-net) + Laplace + 3 BCs via auto-diff",
            "**Hard-constraint trick**: η̃ = M(x,t) + R(x,t)·N_η̃ — measurements hard-encoded",
            "Validated on MARATHON wave flume — SSP=0.049 (assimilation), 0.084 (prediction)",
        ],
        limits=[
            "**~30 min training / instance** (RTX 3090, 40K+ epochs)",
            "**1D + t**, future work: 2D + t extension explicitly stated",
            "Buoy time-series only — sparse spatial LiDAR setup 처리 안 함",
        ],
        contrast=[
            "Speed: 13 ms (ours) vs ~30 min (PFT-PINN) — **130,000× difference**",
            "Ours: **2D + t already**, **LiDAR raydrop sensor model**",
            "**Threat**: same group (TUHH+Imperial) likely to publish 2D + real-time PINN in 1-3 yr",
        ],
        fig_path=None,
        notes=(
            "가장 위협적인 prior art. Physics-informed unsupervised 라는 점에서 우리와 근본 "
            "철학 같습니다 — training data 없이 dispersion + BCs 를 enforce. 다만 1D + t "
            "만 다루고, 30분 training 필요. Future work 가 명시적으로 2D extension + "
            "real-time speedup 인데, 같은 그룹이 1-3년 안에 이걸 발표하면 우리 niche 가 "
            "좁아짐. 따라서 빨리 publish + 응용 영역 차별화 (PINN = design-time, "
            "우리 = real-time control) 으로 대응."
        ),
        star=True,
    )


def build_s9():
    build_prior_art(
        num=9,
        title_full="Yan, Che, Li 2025 (arXiv 2510.14356) — VOF-EnKF",
        key_msg=[
            "**First VOF-NS + EnKF integration** — full Navier-Stokes + ensemble DA",
            "POD reduction: M=4.8e4 cells → ~30 modes (level-set 4 modes)",
            "**Plunging breaker reconstruction** — assimilate upstream linear, CFD evolves breaking",
        ],
        limits=[
            "**Offline** — 50 members × full OpenFOAM = orders of magnitude slower",
            "**1D xz-plane only** — 2D directional 처리 안 함",
            "Inlet boundary handling 어려움 — near-inlet RMS 0.228 H_s vs downstream 0.090",
        ],
        contrast=[
            "Speed: 13 ms (ours) vs offline (VOF-EnKF) — **응용 영역 분리**",
            "Ours: **2D directional**, VOF-EnKF: 1D xz",
            "Their strength: **wave breaking** (we acknowledge this limitation as future work)",
        ],
        fig_path=None,
        notes=(
            "다른 극단의 prior art — full physics fidelity 추구. Plunging breaker 처리 "
            "가능한 유일한 method 지만 50 ensemble × OpenFOAM 으로 fundamentally offline. "
            "우리와 응용 영역 다름 — 우리는 real-time control loop, VOF-EnKF 는 "
            "design-time digital twin. 정직하게 인정할 점은 우리가 wave breaking "
            "못 처리한다는 점, future work 로."
        ),
        star=True,
    )


# S10 — Application matrix + niche
def build_s10():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 1 · Niche")
    add_title(s, "Where does WaveSplat fit? — Application matrix")
    add_subtitle(s, "Real-time control loop vs Design-time analysis — two distinct application zones")
    headers = ["Application requirement", "WaveSplat", "PFT-PINN", "VOF-EnKF",
               "EnKF-HOS", "FNO", "Desmars"]
    rows = [
        ["Real-time control loop", "✓", "✗", "✗", "✗", "✓", "✗"],
        ["2D directional phase-resolved", "✓", "✗", "✗", "✗ (1D)", "✗", "✗"],
        ["Wave breaking, plunging", "✗", "✗", "✓", "✗", "✗", "✗"],
        ["Wave-structure interaction", "✗", "✗", "✓", "✗", "✗", "✗"],
        ["Velocity field full inference", "✗", "✓", "✓", "✓", "✗", "partial"],
        ["Bathymetry inversion", "✗", "✓ ext.", "✗", "✗", "✗", "✗"],
        ["Training-free unsupervised", "✓", "✓", "✓", "✓", "✗", "✓"],
        ["Sparse LiDAR sensor model", "✓", "✗", "✗", "✗", "✗", "partial"],
    ]
    add_table_simple(s, 0.4, 1.9, 12.5, 4.0, headers, rows,
                     header_color=ACCENT, font_size=10)
    add_text(s, 0.5, 6.0, 12.3, 0.4, "Our niche:",
             font_size=14, bold=True, color=ACCENT)
    add_bullets(s, 0.6, 6.4, 12.0, 0.8, [
        "**Real-time + 2D directional + sparse LiDAR + unsupervised** = unique combination",
        "Application: 선박 ride control, hydrofoil FF preview, autonomous navigation",
    ], font_size=12)
    add_page_number(s, 10)
    add_notes(s, (
        "이 매트릭스가 우리 paper 의 positioning 핵심. 모든 항목을 만족하는 method 는 "
        "없습니다 — 각자의 specialty 가 있어요. 우리 specialty 는 real-time + "
        "2D directional + sparse LiDAR 의 unique 조합. 응용 영역으로 보면 "
        "PFT-PINN / VOF-EnKF 와 경쟁 안 함 — 그들은 design-time analysis tool, "
        "우리는 real-time control component. 한계도 명시 — wave breaking, velocity "
        "inference, full nonlinear 는 우리 영역 밖. 다음부터는 우리가 만든 것을 "
        "보여드립니다."
    ))


# S11 — WaveSim physics + raydrop
def build_s11():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 2 · WaveSim — what we built", color=ACCENT2)
    add_title(s, "WaveSim — physics + LiDAR sensor model")
    add_subtitle(s, "Linear superposition + dispersion + JONSWAP, plus full LiDAR raydrop")
    bullets = [
        "**Physics**: η(x,y,t) = Σ_n A_n cos(k_xn·x + k_yn·y − ω_n·t + φ_n)",
        "**Dispersion (deep water)**: ω_n = √(g·|k_n|·tanh(|k_n|·d))",
        "**Spectrum**: JONSWAP-2D (Hasselmann + spreading), N_components 가변",
        "",
        "**LiDAR sensor model** (이 부분이 우리 unique contribution):",
        "  Ray-triangle intersection (Möller-Trumbore 1997)",
        "  Raydrop probability — Fresnel reflectance (specular vs diffuse)",
        "  Shadowing — closer wave occludes far",
        "  Vessel motion + ego-velocity correction",
        "",
        "**Validation**: 39 unit tests passing, 100 episodes generated",
    ]
    add_bullets(s, 0.6, 1.9, 8.0, 5.2, bullets, font_size=14)
    fig_field = os.path.join(DOC_DIR, 'wavesim_BF4_5c_wave_field.png')
    add_image_or_placeholder(s, fig_field, 8.7, 2.0, 4.3,
                             placeholder_text='[wavesim wave field]')
    add_page_number(s, 11)
    add_notes(s, (
        "WaveSim 은 우리가 만든 simulator. Physics 는 linear superposition 으로 시작 — "
        "Klein 2020 가 정량 비교한 LWT/NLSE/HOSM 중 가장 단순한 LWT family. 한계는 "
        "이후 슬라이드에서 정직 disclosure. 하지만 sensor model 은 매우 정확하게 "
        "구현 — Möller-Trumbore ray-triangle intersection, Fresnel reflectance 기반 "
        "raydrop, geometric shadowing, vessel motion correction 까지. 12 paper 중 "
        "이 정도 sensor model 을 simulate 한 prior art 는 없습니다."
    ))


# S12 — WaveSim 100 episodes
def build_s12():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 2 · WaveSim — dataset", color=ACCENT2)
    add_title(s, "100 episodes — controlled benchmark")
    add_subtitle(s, "Reproducible, varying complexity, ground-truth η + LiDAR points")
    bullets = [
        "**Episode parameters**:",
        "  Sea state: BF3 (mild), BF4 (moderate), BF6 (rough)",
        "  N_components: 1, 5, 15 (low / mid / high complexity)",
        "  Spreading: uniform / unimodal / bimodal directional",
        "",
        "**Per episode**: T = 30 s, Δt = 0.1 s (300 frames)",
        "  Ground-truth η field on 50×50 grid",
        "  LiDAR scan (~50 valid pts/frame after raydrop)",
        "  HDF5 format — 100 files, ~5 GB total",
        "",
        "**Tests**: 39 unit tests passing (wavesim/) + 18 (wavesplat/) = 57 total",
    ]
    add_bullets(s, 0.6, 1.9, 8.0, 5.2, bullets, font_size=14)
    fig_lidar = os.path.join(DOC_DIR, 'wavesim_BF4_5c_wave_lidar.png')
    add_image_or_placeholder(s, fig_lidar, 8.7, 2.0, 4.3,
                             placeholder_text='[wavesim BF4 LiDAR]')
    add_page_number(s, 12)
    add_notes(s, (
        "100 episodes 의 controlled benchmark 를 구축했습니다. Sea state × N_components × "
        "directional spreading 의 grid. Reproducibility 보장 — seed 까지 저장. "
        "39 + 18 = 57 unit tests 가 simulator + estimator 의 정확성 보증. "
        "이 인프라 자체가 paper 의 contribution 일부 — Ehlers 2023 도 비슷한 "
        "synthetic benchmark 만들었지만 1D 였음."
    ))


# Override: section tag with custom color
def add_section_tag(slide, tag, color=ACCENT):  # noqa: F811
    add_text(slide, 0.5, 0.2, 4, 0.3, tag,
             font_size=11, bold=True, color=color)


# S13 — WaveSplat algorithm (exploration phase)
def build_s13():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 3 · WaveSplat — exploration phase", color=GOLD)
    add_title(s, "WaveSplat algorithm — parametric off-grid")
    add_subtitle(s, "Continuous spectral parameters (k_x, k_y, ω, A, φ) with dispersion as hard constraint")
    bullets = [
        "**State**: N components × {A_n, k_xn, k_yn, ω_n, φ_n}",
        "  Reduced to 4N free parameters via dispersion: ω_n = √(g·|k_n|·tanh(|k_n|·d))",
        "",
        "**Two-stage estimation**:",
        "  **Init** — multi-frame: 2-stage beamforming (k-only init), all components together",
        "  **Track** — per-frame: freeze (k_x, k_y), update (ω, A, φ) via Adam",
        "",
        "**Adaptive N**:",
        "  Energy-based pruning + residual-triggered growth",
        "  Safety rollback when RMSE > 130%",
        "",
        "**Loss**: ‖η_LiDAR − η_model(θ)‖² + λ_disp · dispersion penalty (hard)",
    ]
    add_bullets(s, 0.6, 1.9, 8.0, 5.2, bullets, font_size=14)
    fig_pipe = os.path.join(PAPER_FIG_DIR, 'fig_pipeline.png')
    add_image_or_placeholder(s, fig_pipe, 8.7, 2.0, 4.3,
                             placeholder_text='[WaveSplat pipeline]')
    add_page_number(s, 13)
    add_notes(s, (
        "WaveSplat 의 exploration phase 알고리즘. 핵심은 N component 의 spectral "
        "parameter 를 직접 추정 — 5 parameter × N. 단 dispersion 으로 4 자유도. "
        "Two-stage 가 핵심 — single frame 에서 k 가 ill-conditioned 라서 multi-frame "
        "init 으로 한 번에 잡고, 이후 track 단계에서 per-frame ω, A, φ 만 update. "
        "이게 다음에 보일 main framing 의 motivation 임."
    ))


# S14 — Prior result (P1 12-ep + real-time)
def build_s14():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 3 · WaveSplat — exploration phase", color=GOLD)
    add_title(s, "Exploration phase results — 12 episode benchmark")
    add_subtitle(s, "WaveSplat ≈ best avg nRMSE; ETKF/GPR for context (tuned)")
    headers = ["Method", "Avg nRMSE", "Per-frame time", "Setup"]
    rows = [
        ["WaveSplat (ours)", "0.223", "13.4 ms", "parametric off-grid + dispersion hard"],
        ["GPR (single + window)", "0.276", ">200 ms", "Gaussian process, dense kernel"],
        ["ETKF-grid (tuned, Ne=100)", "0.319", "132.6 ms", "Hunt 2007 LETKF, grid-spatial"],
        ["Oracle (lower bound)", "0.087", "—", "ground-truth components, fit only A,φ"],
    ]
    add_table_simple(s, 0.5, 1.9, 12.3, 2.5, headers, rows,
                     header_color=ACCENT, font_size=12)
    add_bullets(s, 0.6, 4.6, 12.0, 2.5, [
        "**Speed advantage**: 13 ms vs 132 ms (ETKF) vs >200 ms (GPR) — **10× faster than ETKF**",
        "**Accuracy advantage**: 0.223 vs 0.276 vs 0.319 — **best average among unsupervised methods**",
        "Oracle 0.087 = irreducible noise floor (sensor + numerical)",
        "These results are exploration phase — main framing comes from analyzing how WaveSplat achieves this",
    ], font_size=13)
    add_page_number(s, 14)
    add_notes(s, (
        "12 episode 비교 결과. WaveSplat 이 unsupervised methods 중 best average — "
        "ETKF, GPR 모두 능가. 속도는 더 명확 — 10× 차이. 단 이게 *exploration phase* "
        "결과라는 점에 주목 — 우리가 처음에 만든 알고리즘이 *왜 이렇게 잘 작동하는가* "
        "를 분석하면서 본질적 mechanism 을 발견. 다음 슬라이드부터 그 mechanism 의 "
        "본질을 보여드립니다."
    ))


# S15 — Spatial Nyquist 한계 (the key insight)
def build_s15():
    s = add_blank(LIGHT_BG)
    add_section_tag(s, "Part 4 · The essential insight", color=ACCENT2)
    add_title(s, "The key question — why does parametric work so well?")
    add_subtitle(s, "Per-frame spatial Nyquist is way too small for ocean waves")
    bullets = [
        "**Per-frame LiDAR effective sampling**:",
        "  ~50 valid pts / 240 m FOV → effective Δy ≈ 5–8 m",
        "  Spatial Nyquist: k_Nyq = π/Δy ≈ **0.4 rad/m**",
        "",
        "**Required wavenumber range** (BF3-BF6 sea states):",
        "  λ ∈ 5–25 m → k ∈ **0.5–3 rad/m**",
        "",
        "**🚨 The paradox**: required k is 1.25–7.5× **beyond per-frame Nyquist!**",
        "",
        "**Yet, WaveSplat recovers them. How?**",
        "  Hint: we use multi-frame data and dispersion physics",
        "  → Next slide: the mechanism",
    ]
    add_bullets(s, 0.8, 1.9, 12.0, 5.5, bullets, font_size=16,
                line_space_after=6)
    add_page_number(s, 15)
    add_notes(s, (
        "여기서 가장 중요한 발견. Single-frame LiDAR 의 spatial Nyquist 가 0.4 rad/m "
        "정도. 그런데 ocean wave 의 typical wavenumber 는 0.5–3 rad/m. 즉 *per-frame "
        "에서는 추정 불가능한 영역*. 그런데 WaveSplat 은 이걸 추정합니다. 어떻게? "
        "이게 우리 main framing 의 출발점 — vessel motion 과 dispersion 의 결합이 "
        "spatial Nyquist 한계를 우회하는 mechanism 입니다."
    ))


# S16 — Main framing
def build_s16():
    s = add_blank(LIGHT_BG)
    add_section_tag(s, "Part 4 · Main framing ⭐", color=ACCENT2)
    add_title(s, "Dispersion-Coupled Spatio-Temporal Inversion",
              size=24, color=ACCENT)
    add_text(s, 0.5, 1.5, 12.3, 1.6,
             ('"Vessel motion (synthetic aperture) and dispersion physics jointly '
              'resolve ocean wavenumbers 3–7× beyond the per-frame spatial Nyquist '
              'of shipboard LiDAR."'),
             font_size=18, italic=True, color=DARK,
             alignment=PP_ALIGN.CENTER)
    bullets = [
        "**Synthetic aperture (SA)**: Δx_SA = V · T_init = 11.3 m/s × 1 s = **11.3 m**",
        "  (vs single-frame Δy ≈ 5–8 m → 1.5–2× larger effective aperture)",
        "",
        "**Dispersion 1-to-1 mapping**: ω = √(g·|k|·tanh(|k|·d))",
        "  Each ω uniquely determines |k| → temporal info → spatial wavenumber",
        "",
        "**Joint mechanism**:",
        "  Multi-frame stacking → multi-temporal coherence → ω resolution",
        "  ω → |k| via dispersion → spatial wavenumber even beyond Nyquist",
        "",
        "**Quantitative claim**: super-resolution factor **3–7×** beyond per-frame Nyquist",
        "  Validated by 4-way ablation on 100-episode benchmark (Phase 1 plan)",
    ]
    add_bullets(s, 0.6, 3.3, 12.0, 4.0, bullets, font_size=13,
                line_space_after=4)
    add_page_number(s, 16)
    add_notes(s, (
        "이게 우리 paper 1 의 main contribution claim. Per-frame spatial Nyquist 가 "
        "0.4 rad/m 인데 우리가 추정하는 wavenumber 가 0.5–3 rad/m, 즉 3–7× 우회. "
        "Mechanism 은 두 가지 — vessel motion 으로 synthetic aperture (SA) 형성 + "
        "dispersion 의 ω → |k| 1-to-1 매핑 으로 temporal info 가 spatial 로 전환. "
        "이 두 mechanism 이 함께 작용해야 작동. 단일 mechanism 만으로는 부족 — "
        "Phase 1 ablation 으로 정량 증명할 계획."
    ))


# S17 — Mechanism + Identifiability
def build_s17():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 4 · Mechanism", color=ACCENT2)
    add_title(s, "Two complementary insights — why init/track separation works")
    add_subtitle(s, "Identifiability decomposition + dispersion-coupled super-resolution")
    bullets = [
        "**Identifiability (per-parameter Fisher Information)**:",
        "  k = (k_x, k_y) — multi-frame identifiable (needs aperture)",
        "  (ω, φ, A) — per-frame identifiable (single snapshot enough)",
        "",
        "**→ Justifies our two-stage design**:",
        "  Init (multi-frame) — solve k from temporal coherence",
        "  Track (per-frame) — update (ω, A, φ) cheaply",
        "",
        "**Combined story**:",
        "  Init phase: SA + dispersion → resolve k beyond per-frame Nyquist",
        "  Track phase: per-frame Adam keeps inference < 100 ms",
        "  Adaptive N + safety rollback: robust to changing component count",
    ]
    add_bullets(s, 0.6, 1.9, 8.0, 5.2, bullets, font_size=14)
    fig_bf = os.path.join(PAPER_FIG_DIR, 'fig_beamforming_2stage.png')
    add_image_or_placeholder(s, fig_bf, 8.7, 2.0, 4.3,
                             placeholder_text='[2-stage beamforming]')
    add_page_number(s, 17)
    add_notes(s, (
        "두 supporting insight. 첫째 identifiability — k 는 multi-frame 에서 "
        "identifiable, ω·φ·A 는 per-frame identifiable. 이 분해가 우리 init/track "
        "two-stage 설계의 정당화. Fisher Information 정량 분석은 Phase 1 추가 작업. "
        "둘째 — combined story 로 main framing 과 연결. Init 에서 SA+dispersion 으로 "
        "k 잡고, track 에서 per-frame 으로 빠르게 update. Adaptive N + safety "
        "rollback 으로 robustness 확보."
    ))


# S18 — Limitations + future + threat response
def build_s18():
    s = add_blank(WHITE)
    add_section_tag(s, "Closing · Limitations & response", color=GRAY)
    add_title(s, "Honest limitations & response strategy")
    add_subtitle(s, "Linear approximation, no breaking, threat from PFT-PINN scale-up")
    bullets = [
        "**Limitation 1 — Linear approximation**:",
        "  Klein 2020 정량: HOSM > NLSE₂ > LWT (우리 family)",
        "  Operational range: ε < 0.07 (valid for control loop sea states)",
        "  Future: Stokes ε² correction (Phase 2, JFM upgrade)",
        "",
        "**Limitation 2 — No breaking, no velocity inference**:",
        "  VOF-EnKF, PFT-PINN handle these — different application zones",
        "  Future: hybrid linear-realtime + on-demand CFD refinement",
        "",
        "**Threat — PFT-PINN extension**:",
        "  Same group (TUHH+Imperial) likely 2D + real-time PINN in 1-3 yr",
        "  Response 1: **publish fast** (OE-ready by 7월, JFM upgrade option)",
        "  Response 2: **application separation** (PINN = design-time, ours = real-time control)",
    ]
    add_bullets(s, 0.6, 1.9, 12.0, 5.5, bullets, font_size=13,
                line_space_after=4)
    add_page_number(s, 18)
    add_notes(s, (
        "정직한 disclosure. Linear approximation 의 한계 — Klein 2020 가 정량 비교. "
        "Operational range 명시 + Stokes correction 을 Phase 2 에. Wave breaking 은 "
        "응용 영역 분리로 정당화. Threat — Ehlers 그룹이 2D+realtime PINN 으로 "
        "올 가능성. 대응: 빨리 publish + 응용 차별화."
    ))


# S19 — Timeline + venue + Q&A
def build_s19():
    s = add_blank(LIGHT_BG)
    add_section_tag(s, "Closing · Plan", color=ACCENT)
    add_title(s, "Publication plan & timeline", color=DARK)
    add_subtitle(s, "Phase 1 (2 mo) → Decision gate → Phase 2 (3 mo) → Submit")
    bullets = [
        "**Phase 1 — OE-ready (5월–7월)**:",
        "  4-way ablation (single / multi-spatial / multi-disp-post / multi-disp-first)",
        "  Theoretical bound (1-page derivation)",
        "  Paper rewrite: Dispersion-Coupled Spatio-Temporal Inversion framing",
        "",
        "**Decision gate (7월 말)**:",
        "  Super-res factor 3–7× confirmed → JFM/PRF 도전 (Phase 2)",
        "  Else → OE submit immediately",
        "",
        "**Phase 2 — JFM/PRF upgrade (8월–10월)**:",
        "  Sample complexity bound (atomic-norm-inspired)",
        "  Stokes ε² correction (linear → 2nd order)",
        "  Identifiability FIM 정량 분석",
        "",
        "**Submit**: 11월–12월 (OE 또는 JFM/PRF)",
    ]
    add_bullets(s, 0.6, 1.9, 12.0, 5.0, bullets, font_size=13,
                line_space_after=4)
    add_text(s, 0.5, 6.85, 12.3, 0.4,
             "Questions & discussion — feedback most welcomed",
             font_size=14, italic=True, color=ACCENT,
             alignment=PP_ALIGN.CENTER)
    add_page_number(s, 19)
    add_notes(s, (
        "Publication plan. Phase 1 (2개월) 으로 OE-ready, 7월 말 decision gate 에서 "
        "super-res factor 검증 후 JFM 도전 vs OE submit 결정. Phase 2 (3개월) 는 "
        "JFM upgrade — sample complexity bound, Stokes correction, FIM 정량. "
        "최종 submit 11월-12월. 이게 5일 후 lab meeting 끝나고 즉시 시작할 plan. "
        "질문 받겠습니다."
    ))


# ─────────────────────────────────────────
# Backup slides A1–A7
# ─────────────────────────────────────────
def build_backup(num_letter, title, key_points, notes=''):
    s = add_blank(WHITE)
    add_section_tag(s, f"Backup · {num_letter}", color=GRAY)
    add_title(s, title, size=22)
    add_bullets(s, 0.6, 1.7, 12.0, 5.2, key_points, font_size=14)
    add_text(s, 12.5, 7.05, 0.7, 0.35,
             num_letter, font_size=10, color=GRAY,
             alignment=PP_ALIGN.RIGHT)
    add_notes(s, notes)


def build_backups():
    build_backup("A1",
                 "Desmars 2018 (OMAE) — LIDAR + ICWM concept",
                 [
                     "**Setup**: scanning LIDAR + Improved Choppy Wave Model (ICWM)",
                     "Predefined k-grid, variational inversion",
                     "**Limitation**: 1D, on-grid, simulator only",
                     "**Relation**: precursor to Desmars 2023 JFM (HOS-OCM)",
                 ])
    build_backup("A2",
                 "Desmars 2020 (Ocean Eng 212) — wave gauge experiment",
                 [
                     "**First experimental validation** of phase-resolved DA in wave tank",
                     "Wave gauge array, dispersion correction post-processing",
                     "**Limitation**: 1D longitudinal only, gauge ≠ LiDAR",
                     "**Relation**: empirical anchor for the family",
                 ])
    build_backup("A3",
                 "Klein et al. 2020 (Fluids 5(1), 9) — wave model comparison",
                 [
                     "**Quantitative**: HOSM > NLSE₂ > NLSE > LWT (decreasing fidelity)",
                     "Wave tank validation, SSP metric standardization",
                     "**Implication for us**: linear (LWT) operational range ε < 0.07",
                     "**Relation**: source of Stokes ε² correction motivation",
                 ])
    build_backup("A4",
                 "Hunt, Kostelich, Szunyogh 2007 (Tellus A) — LETKF",
                 [
                     "**Foundational**: Local Ensemble Transform Kalman Filter",
                     "Gauge of all EnKF wave reconstruction work",
                     "**Our use**: ETKF baseline implements this with grid-spatial obs",
                     "**Relation**: Wang-Pan 2021/22 + Yan 2025 derive from this",
                 ])
    build_backup("A5",
                 "Tang, Bhaskar, Recht 2013 (IEEE TIT) — atomic norm CS",
                 [
                     "**Off-grid compressed sensing** with continuous parameter space",
                     "SDP formulation, theoretical recovery bounds",
                     "**Inspiration for us**: parametric off-grid spectral inversion",
                     "**Future direction**: sample complexity bound (Phase 2)",
                 ])
    build_backup("A6",
                 "Möller-Trumbore 1997 (JGT) — Ray-triangle intersection",
                 [
                     "**Standard ray-triangle algorithm** for graphics + simulation",
                     "Robust, fast, no preprocessing",
                     "**Our use**: WaveSim LiDAR ray-mesh intersection (verified against this paper)",
                     "**Relation**: numerical correctness of sensor model",
                 ])
    build_backup("A7",
                 "Nouguier, Grilli, Guérin 2014 (IEEE TGRS) — flash LIDAR foundational",
                 [
                     "**Historical first**: flash LIDAR for ocean wave reconstruction",
                     "Paywalled, not fully read — cited as foundational reference",
                     "**Implication**: LIDAR for waves predates Desmars by ~5 years",
                     "**Relation**: bibliographic anchor in our prior art",
                 ])


# ─────────────────────────────────────────
# Build all
# ─────────────────────────────────────────
def main():
    builders = [
        build_s1, build_s2, build_s3, build_s4, build_s5,
        build_s6, build_s7, build_s8, build_s9, build_s10,
        build_s11, build_s12, build_s13, build_s14, build_s15,
        build_s16, build_s17, build_s18, build_s19,
    ]
    for b in builders:
        b()
    build_backups()

    os.makedirs(DOC_DIR, exist_ok=True)
    prs.save(OUT_PATH)
    print(f"✅ saved: {OUT_PATH}")
    print(f"   slides: {len(prs.slides)}")


if __name__ == '__main__':
    main()
