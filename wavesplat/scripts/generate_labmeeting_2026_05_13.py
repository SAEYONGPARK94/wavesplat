"""
generate_labmeeting_2026_05_13.py

Lab meeting presentation 2026-05-13 — WaveSplat exploration phase + new framing.
외부 청자 (비전문가) 대상 20분 발표 (1 slide ≈ 1분).

Style: 친절한 설명 — jargon 옆에 부연, motivation 강조, presenter notes 상세.

Structure:
  Part 1 (S1-S10): Motivation + 5-category prior art + 5 detailed papers + niche
  Part 2 (S11-S12): WaveSim simulator
  Part 3 (S13-S14): WaveSplat exploration phase
  Part 4 (S15-S17): Main framing — Dispersion-Coupled Spatio-Temporal Inversion
  Closing (S18-S19): Limitations + publication plan
  Backup (A1-A7): Supporting prior art

Output: docs/labmeeting_2026_05_13.pptx
Source content: docs/labmeeting_2026_05_13_slides.md
"""
import os

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

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

TOTAL_MAIN = 19


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


def add_bullets(slide, left, top, width, height, items, font_size=14,
                color=DARK, bullet_char='•', line_space_after=4,
                indent_str='  '):
    """items: list of strings. '**...**' = bold. '  ' prefix = indented sub-bullet."""
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

        # detect indent
        if item.startswith(indent_str):
            prefix = '   – '
            item_clean = item.lstrip()
        elif item == '':
            # empty paragraph (spacer)
            run = p.add_run()
            run.text = ' '
            run.font.size = Pt(font_size // 2)
            continue
        else:
            prefix = f"{bullet_char} "
            item_clean = item

        # parse bold inside
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
    add_text(slide, 0.5, 0.2, 5, 0.3, tag,
             font_size=11, bold=True, color=color)


def add_title(slide, title, y=0.55, color=DARK, size=24):
    add_text(slide, 0.5, y, 12.3, 0.7, title,
             font_size=size, bold=True, color=color)


def add_subtitle(slide, subtitle, y=1.25, color=GRAY, size=13, italic=True):
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
    notes_slide = slide.notes_slide
    tf = notes_slide.notes_text_frame
    tf.text = text


def add_table_simple(slide, left, top, width, height, headers, rows,
                     header_color=ACCENT, font_size=11, highlight_col=None):
    cols = len(headers)
    n_rows = len(rows) + 1
    table = slide.shapes.add_table(n_rows, cols,
                                   Inches(left), Inches(top),
                                   Inches(width), Inches(height)).table
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
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = table.cell(r + 1, c)
            cell.fill.solid()
            if highlight_col is not None and c == highlight_col:
                cell.fill.fore_color.rgb = RGBColor(0xFF, 0xF2, 0xCC)
            else:
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
# Slide builders — 친절한 외부 청자 버전
# ─────────────────────────────────────────

# S1 — Title
def build_s1():
    s = add_blank(LIGHT_BG)
    add_text(s, 0.5, 1.7, 12.3, 1.0,
             "Phase-Resolved Ocean Wave Estimation",
             font_size=36, bold=True, color=DARK,
             alignment=PP_ALIGN.CENTER)
    add_text(s, 0.5, 2.7, 12.3, 0.7,
             "from Sparse Shipboard LiDAR",
             font_size=30, bold=True, color=ACCENT,
             alignment=PP_ALIGN.CENTER)
    add_text(s, 0.5, 3.8, 12.3, 0.5,
             "선박 LiDAR 로 파도의 형태를 실시간 추정하는 연구 — 탐색 단계 결과와 향후 방향",
             font_size=15, italic=True, color=GRAY,
             alignment=PP_ALIGN.CENTER)
    add_text(s, 0.5, 4.5, 12.3, 0.4,
             "Exploration phase results & a path forward",
             font_size=14, italic=True, color=GRAY,
             alignment=PP_ALIGN.CENTER)
    add_text(s, 0.5, 5.7, 12.3, 0.4,
             "Saeyong Park",
             font_size=18, bold=True, color=DARK,
             alignment=PP_ALIGN.CENTER)
    add_text(s, 0.5, 6.2, 12.3, 0.4,
             "Lab Meeting · 2026-05-13",
             font_size=14, color=GRAY,
             alignment=PP_ALIGN.CENTER)
    add_notes(s, (
        "안녕하세요. 오늘 발표는 두 부분으로 나뉩니다.\n\n"
        "[앞 절반] 우리가 지난 6개월 동안 진행한 *exploration phase*의 결과를 정리합니다. "
        "구체적으로 (1) 이 분야 prior art 12편을 정독한 결과, (2) 우리가 만든 simulator(WaveSim)와 "
        "estimator(WaveSplat)이 어떻게 작동하는지를 설명드립니다.\n\n"
        "[뒤 절반] 그 과정에서 발견한 본질적 mechanism을 보여드립니다. "
        "선박이 움직이면서 발생하는 효과(synthetic aperture)와 파도의 분산 관계(dispersion physics)가 "
        "어떻게 결합해서 LiDAR의 공간 해상도 한계를 우회하는지 — 이것을 우리 paper의 main contribution으로 "
        "격상시키려는 plan입니다.\n\n"
        "외부 청자분들이 계셔서 도메인 background부터 천천히 시작하겠습니다. "
        "전공 용어가 처음 나오면 옆에 짧게 설명을 붙였으니 부담 없이 들으시면 됩니다."
    ))


# S2 — Motivation
def build_s2():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 1 · Motivation — 왜 이 연구가 필요한가")
    add_title(s, "위상까지 추정하는 파도 정보가 있어야 미리 대응할 수 있습니다")
    add_subtitle(s, "Statistical (평균적 통계) 만으로는 부족 — 'when & where' 의 결정론적 정보가 필요")
    bullets = [
        "**Statistical estimation** (현재 표준): 유의파고 H_s, 첨두주기 T_p, JONSWAP 스펙트럼",
        "  → 평균적 파도 환경은 알 수 있지만, *언제 어디서 큰 파도가 올지* 는 알 수 없음",
        "  → 선박 설계에는 충분, 실시간 제어에는 부족",
        "",
        "**Phase-resolved estimation** (이 연구): η(x, y, t) — 각 파도의 위치·시점을 결정론적으로",
        "  → 'BF4 sea state' 가 아닌 'x = 120 m 위치에 5초 후 H = 2 m 파도' 같은 구체적 정보",
        "  → look-ahead window (앞으로 0–30 초) 확보 가능",
        "",
        "**직접 응용 분야**:",
        "  Hydrofoil ride control — 미리 본 파도로 feed-forward 제어 → 승차감 향상",
        "  Autonomous vessel — 위험 파도 회피, 항로 계획",
        "  Offshore operation — 인양·헬기 착륙 안전 window 결정",
        "  Wave-energy converter — 최적 damping 튜닝",
    ]
    add_bullets(s, 0.6, 1.85, 12.0, 5.2, bullets, font_size=13)
    fig_hf = os.path.join(DOC_DIR, 'hydrofoil_controller_flow.png')
    add_image_or_placeholder(s, fig_hf, 9.0, 4.4, 4.0,
                             placeholder_text='[hydrofoil control flow]')
    add_page_number(s, 2)
    add_notes(s, (
        "파도 추정 분야에는 두 단계가 있습니다.\n\n"
        "첫째 statistical estimation — 우리가 흔히 듣는 H_s (significant wave height, "
        "유의파고), T_p (peak period, 첨두주기), JONSWAP 스펙트럼 같은 '평균적 통계'. "
        "선박 디자인이나 해양 구조물 설계에는 이걸로 충분합니다.\n\n"
        "둘째 phase-resolved estimation — 각 파도의 위치와 시점을 결정론적으로 추정. "
        "예를 들어 'BF4 sea state' 라는 통계가 아니라 'x = 120 m 위치에 5초 후 H = 2 m 파도가 도착한다' "
        "같은 구체적 정보입니다.\n\n"
        "왜 phase-resolved가 필요한가? 실시간 제어에서는 '평균적으로 파도가 강하다' 는 정보로는 "
        "특정 시점의 제어 결정을 못 합니다. 어디에 큰 파도가 오는지 정확히 알아야 ride control, "
        "hydrofoil, autonomous navigation 같은 응용이 가능합니다.\n\n"
        "응용 중 가장 직접적인 것은 hydrofoil ride control입니다 — 우리 연구실에서 "
        "진행 중인 hydrofoil 연구의 motivation이기도 합니다. 파도를 미리 보고 hydrofoil의 angle을 "
        "조절하면 승차감(ride RMS)을 크게 줄일 수 있습니다."
    ))


# S3 — Problem
def build_s3():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 1 · Problem — 무엇이 어려운가")
    add_title(s, "선박 LiDAR 기반 phase-resolved 추정의 세 가지 어려움")
    add_subtitle(s, "이 세 조건을 동시에 만족하는 prior art는 존재하지 않습니다")
    bullets = [
        "**Challenge 1 — Sparse, non-uniform measurements (희소·비균일 측정)**",
        "  LiDAR 한 frame당 유효 점 ~50개 (전체 ray 중 raydrop·shadowing 후 남은 점)",
        "  거리에 따라 점밀도 다름: 근거리 dense, 원거리 sparse",
        "  탐지 거리 ~250 m 제한 (vs 파도 파장 50–300 m, 한 파장도 못 보는 경우)",
        "",
        "**Challenge 2 — Real-time constraint (실시간 제약)**",
        "  제어 루프 주기: ~10 Hz update (즉 0.1 s 간격으로 추정 결과 필요)",
        "  Per-frame 예산: < 100 ms (offline CFD/PINN/EnKF는 분~시간 단위)",
        "",
        "**Challenge 3 — 2D directional sea state (2차원 방향성 해양)**",
        "  실제 바다는 short-crested, multi-directional (한 방향 파도가 아님)",
        "  1D unidirectional 가정은 부족 (Klein 2020, Desmars 2023 대부분 1D)",
        "  2D wavenumber vector (k_x, k_y) 추정 필요",
    ]
    add_bullets(s, 0.6, 1.85, 8.0, 5.2, bullets, font_size=13)
    fig_lidar = os.path.join(DOC_DIR, 'wavesim_BF4_5c_wave_lidar.png')
    add_image_or_placeholder(s, fig_lidar, 8.8, 2.0, 4.2,
                             placeholder_text='[sparse LiDAR scan]')
    add_page_number(s, 3)
    add_notes(s, (
        "이 세 조건을 동시에 만족하는 setup은 prior art에 거의 없습니다. 하나씩 보겠습니다.\n\n"
        "(1) Sparse non-uniform — LiDAR는 흔히 쓰이는 wave gauge나 dense radar와 매우 다릅니다. "
        "한 frame에 점 50개 정도로 매우 희소하고, 거리에 따라 점밀도가 변합니다. "
        "또 raydrop(파도가 LiDAR 빔을 반사하지 못해 점 누락), shadowing(앞쪽 파도가 뒤쪽을 가림) 같은 "
        "물리 효과로 점이 더 줄어듭니다. 기존 wave estimation 방법론은 dense·균일 측정 가정으로 "
        "설계되어서 LiDAR setup에 직접 적용하기 어렵습니다.\n\n"
        "(2) Real-time — 제어 루프에 들어가야 하므로 100 ms 안에 끝나야 합니다. "
        "EnKF 같은 ensemble 방법론, PINN 같은 NN 학습 방법론은 분 단위에서 시간 단위까지 걸려서 "
        "이 제약을 만족 못 합니다.\n\n"
        "(3) 2D directional — 실제 바다는 한 방향 파도가 아니라 여러 방향 파도가 섞여 있습니다. "
        "Wavenumber도 단순한 k가 아니라 vector (k_x, k_y)를 추정해야 합니다. "
        "Klein 2020, Desmars 2023 같은 prior art는 대부분 1D 가정으로 단순화하지만, 우리는 2D를 그대로 다룹니다.\n\n"
        "다음 슬라이드부터 prior art 12편을 검토한 결과를 보여드리고, 이 niche가 실제로 비어 있다는 것을 보이겠습니다."
    ))


# S4 — 5-category landscape
def build_s4():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 1 · Prior art landscape — 12편 정독 후 분류")
    add_title(s, "기존 연구를 5 가지 카테고리로 분류")
    add_subtitle(s, "12편 정독 결과 — 우리 위치는 5번째 카테고리 (parametric off-grid)")
    headers = ["#", "Category (접근법)", "대표 paper", "Strength (강점)", "Limitation (한계)"]
    rows = [
        ["1", "Variational / grid-based\n(격자 위 비용함수 최소화)",
         "Desmars 2018-23\nAragh 2008",
         "성숙한 이론, on-grid", "Predefined k-grid, 1D 위주"],
        ["2", "Ensemble Kalman filter\n(앙상블 베이지안 필터)",
         "Wang-Pan 2021/22\nYan 2025",
         "Full nonlinear physics", "Offline (앙상블 비용)"],
        ["3", "ML supervised\n(지도학습 NN)",
         "Ehlers 2023 (FNO)\nDuan 2020",
         "Fast inference (1.9 ms)", "학습 데이터 필요, generalization 한계"],
        ["4", "Physics-informed unsupervised\n(물리 기반 NN, 비지도)",
         "Ehlers 2025 PFT-PINN",
         "학습 데이터 불필요, 비선형 가능", "~30 분/instance, 1D 위주"],
        ["5", "Parametric off-grid (Ours) ⭐\n(연속 spectral 파라미터)",
         "WaveSplat",
         "Real-time + 2D + LiDAR sensor model", "Linear approximation"],
    ]
    add_table_simple(s, 0.4, 1.85, 12.5, 4.8, headers, rows,
                     header_color=ACCENT, font_size=10)
    add_text(s, 0.5, 6.8, 12.3, 0.3,
             "Axis: x=fidelity (linear → fully nonlinear), y=inference speed. 우리는 top-left + 추가 차원 (2D directional + LiDAR sensor)",
             font_size=10, italic=True, color=GRAY)
    add_page_number(s, 4)
    add_notes(s, (
        "12편을 정독해서 5개 카테고리로 분류했습니다.\n\n"
        "(1) Variational/grid-based — 가장 전통적인 방법입니다. 공간을 격자로 나누고, "
        "관측치와 모델의 차이를 비용함수로 정의 후 최소화. Desmars 2018-2023이 대표적이고 "
        "우리와 가장 가까운 setup입니다.\n\n"
        "(2) Ensemble Kalman filter — 50-100개의 시뮬레이션을 동시에 돌리고 관측치로 보정. "
        "Wang-Pan 2021/22가 대표적, 가장 최근 Yan 2025는 VOF (full Navier-Stokes)에 EnKF를 결합. "
        "물리 충실도는 높지만 ensemble 비용으로 offline.\n\n"
        "(3) ML supervised — pre-trained NN으로 빠르게 추정. Ehlers 2023의 FNO가 1.9 ms로 가장 빠름. "
        "단점은 학습 데이터 필요와 distribution shift risk.\n\n"
        "(4) Physics-informed unsupervised — PINN을 써서 학습 데이터 없이 물리 손실로 학습. "
        "Ehlers 2025 PFT-PINN이 대표적. 비선형 처리 가능하지만 instance마다 30분 학습 필요.\n\n"
        "(5) Parametric off-grid — 우리 위치. Spectral 파라미터 (k_x, k_y, ω, A, φ)를 직접 추정. "
        "다른 카테고리들은 grid나 NN으로 wave field를 표현하지만, 우리는 파라미터 자체를 추정. "
        "이게 가능한 이유는 dispersion 제약을 hard constraint로 강제하기 때문입니다.\n\n"
        "다음 5 슬라이드에서 가장 가까운 prior art 5편을 detail로 보여드립니다."
    ))


# Helper: standardized prior-art slide
def build_prior_art(num, title_full, key_msg, limits, contrast,
                    fig_path=None, notes='', star=False):
    s = add_blank(WHITE)
    add_section_tag(s, f"Part 1 · Prior art detail ({num - 4}/5)")
    if star:
        add_title(s, title_full + "  ⭐", size=22)
    else:
        add_title(s, title_full, size=22)
    add_text(s, 0.5, 1.35, 8.0, 0.35, "핵심 메시지",
             font_size=12, bold=True, color=ACCENT)
    add_bullets(s, 0.6, 1.7, 7.7, 1.9, key_msg, font_size=12,
                line_space_after=3)
    if fig_path:
        add_image_or_placeholder(s, fig_path, 8.5, 1.4, 4.4,
                                 placeholder_text='[paper figure]')
    else:
        add_image_or_placeholder(s, None, 8.5, 1.4, 4.4,
                                 placeholder_text='[paper figure — to insert]')
    add_text(s, 0.5, 4.2, 8.0, 0.35, "한계 (우리 관점)",
             font_size=12, bold=True, color=RED)
    add_bullets(s, 0.6, 4.55, 12.0, 1.6, limits, font_size=12, color=DARK,
                line_space_after=3)
    add_text(s, 0.5, 6.05, 8.0, 0.35, "vs WaveSplat (우리 차별점)",
             font_size=12, bold=True, color=GREEN)
    add_bullets(s, 0.6, 6.4, 12.0, 0.95, contrast, font_size=12, color=DARK,
                line_space_after=3)
    add_page_number(s, num)
    add_notes(s, notes)


def build_s5():
    build_prior_art(
        num=5,
        title_full="Desmars et al. 2023 (JFM 975, A8) — HOS-OCM",
        key_msg=[
            "**선박 mounted optical sensor**로 phase-resolved wave 추정 + 예측 (HOS 기반)",
            "  HOS = High-Order Spectral method, 비선형 wave model",
            "Setup: shipborne LiDAR/camera, 1D longitudinal (선박 진행 방향만), 사전정의 격자",
            "방법: 관측치-모델 mismatch를 비용함수로 정의 → 최소화 → HOS로 미래 prediction",
        ],
        limits=[
            "**On-grid k-band assumption** — wavenumber가 미리 정의된 격자에 묶임 (off-grid 아님)",
            "**1D unidirectional** — 한 방향 파도만 가정, 2D directional 처리 안 함",
            "**Sensor model 명시 없음** — raydrop/shadowing/Fresnel 가정 미상",
        ],
        contrast=[
            "Ours: **연속 (k_x, k_y)** (격자 없음) + **2D directional** + **full LiDAR sensor model 포함**",
        ],
        notes=(
            "12편 중 우리와 setup이 가장 비슷한 prior art입니다. 선박에 광학 센서(LiDAR나 카메라)를 "
            "달고 phase-resolved wave를 추정하고 HOS로 predict — 거의 동일한 problem statement입니다.\n\n"
            "차이점은 세 가지:\n"
            "  - 1D vs 우리 2D: 한 방향 파도 가정이라 directional 정보 손실\n"
            "  - On-grid vs 우리 off-grid: 격자에 묶여서 격자 사이 wavenumber는 표현 불가\n"
            "  - Sensor model: raydrop, Fresnel, shadowing 같은 LiDAR 물리 효과를 명시하지 않음\n\n"
            "이 paper가 우리 paper의 main reference가 됩니다 — 'closest prior art는 무엇이고, "
            "우리가 무엇을 더 했는가'를 명확히 보여주는 비교 대상."
        ),
        star=True,
    )


def build_s6():
    build_prior_art(
        num=6,
        title_full="Wang & Pan 2021 (JFM 918, A19) + 2022 (JFM 949, A31) — EnKF-HOS",
        key_msg=[
            "**EnKF + HOS 첫 결합** — Ensemble Kalman Filter로 phase-resolved wave를 데이터 동화",
            "  EnKF = 50-100개 시뮬레이션을 평행하게 돌리고 관측치로 보정하는 베이지안 필터",
            "2022년 확장: EnKF-HOS-C — 해류(current) 도 동시 추정",
            "GPU 구현, near-real-time (단, 관측치가 dense 격자라는 가정)",
        ],
        limits=[
            "**Dense observation 가정** — sparse LiDAR setup에 직접 적용 어려움",
            "**1D wave** — 2D directional 미처리",
            "**Ensemble cost** — 50–100 members × HOS 시간 통합 (GPU에도 ~수백 ms)",
        ],
        contrast=[
            "Ours: **sparse LiDAR input** + **2D directional** + **<100 ms (GPU 없이)**",
            "Note: Wang 2025 (Coastal Eng 197, ETKF 변형)은 paper 작성 단계에서 정독 예정 — Case A/B/C 결정",
        ],
        notes=(
            "EnKF-HOS family는 우리 ETKF baseline의 직접적 prior art입니다. EnKF가 무엇인가 — "
            "'시뮬레이션 50개를 동시에 돌리고, 관측치를 받을 때마다 평균과 분산을 베이지안으로 갱신' "
            "하는 필터입니다. 기상학에서 출발해서 wave estimation에도 적용된 방법론.\n\n"
            "Wang & Pan 2021이 처음으로 HOS와 EnKF를 결합했고, 2022에서 current estimation으로 확장. "
            "한계는 두 가지:\n"
            "  - Ensemble 비용으로 real-time이 어려움 (GPU 써도 수백 ms 단위)\n"
            "  - Dense obs 가정이라 sparse LiDAR setup에 직접 적용 안 됨 (격자 위 모든 점에 측정값 필요)\n\n"
            "우리는 이 baseline을 sparse-input으로 tune해서 비교했습니다. 자세한 결과는 Slide 14에서 보여드립니다.\n\n"
            "또한 Wang 2025 (Coastal Eng 197)이라는 더 최근 변형이 있는데, 이건 paper 작성 단계에서 "
            "정독해서 'Case A/B/C 결정' (우리 baseline이 그들과 같은 family인지 다른지)을 할 예정입니다."
        ),
        star=True,
    )


def build_s7():
    build_prior_art(
        num=7,
        title_full="Ehlers et al. 2023 (Ocean Eng) — Fourier Neural Operator",
        key_msg=[
            "**ML 기반 phase-resolved 첫 사례** — sparse X-band radar로 wave field 학습",
            "**FNO architecture** (Fourier 공간에서 전역 mapping) > U-Net (CNN 지역 처리)",
            "**Inference 1.9 ms** / instance — 매우 빠름. 단 supervised pre-training 필요 (3120 sample)",
        ],
        limits=[
            "**Supervised** — 3120개 HOS 합성 데이터로 학습. 학습 분포 밖에서는 약함",
            "**1D unidirectional** — 2D directional 미처리",
            "**Black-box** — physical interpretability 부족 (왜 그렇게 추정하는지 이해 어려움)",
        ],
        contrast=[
            "Ours: **unsupervised** (학습 데이터 0개), **interpretable parametric model**, **2D directional**",
            "속도 비교: Ours ~13 ms vs FNO 1.9 ms — 둘 다 real-time 영역, 같은 분류",
        ],
        notes=(
            "ML supervised 카테고리의 SOTA입니다. 1.9 ms의 inference 속도가 가장 빠르지만 trade-off가 있습니다.\n\n"
            "장점은 분명합니다 — 1.9 ms는 real-time 제어에 매우 충분한 속도. "
            "FNO (Fourier Neural Operator)는 Fourier 공간에서 global mapping을 학습하는 구조라 "
            "U-Net 같은 지역 CNN보다 wave 패턴에 더 잘 맞습니다.\n\n"
            "단점:\n"
            "  - 3120개의 HOS 합성 sample로 학습해야 함 — 학습 데이터 자체를 만드는 비용\n"
            "  - Distribution shift — 학습 안 된 sea state나 sensor 조건에서는 성능 저하\n"
            "  - Black-box — 왜 그렇게 추정했는지 해석 어려움\n\n"
            "우리 차별점:\n"
            "  - Unsupervised (학습 데이터 0개) — 새로운 sea state에서도 즉시 작동\n"
            "  - Interpretable — (k_x, k_y, ω, A, φ) 5개 파라미터의 의미가 명확\n"
            "  - 2D directional — FNO는 1D만\n\n"
            "둘 다 real-time 영역이라 paper 1의 main competitor 중 하나입니다."
        ),
        star=True,
    )


def build_s8():
    build_prior_art(
        num=8,
        title_full="Ehlers et al. 2025 (Phys. Rev. Fluids 10, 094901) — PFT-PINN",
        key_msg=[
            "**Physics-Informed NN + Potential Flow Theory** — 두 NN 병렬 (η-net + Φ-net)",
            "  η = wave 표면, Φ = velocity potential (속도 함수). 둘 다 NN으로 표현",
            "**Hard-constraint trick**: 관측치를 NN architecture에 직접 hard-encode (data loss 회피)",
            "**Wave flume 검증** (MARATHON, T_p=1.2 s): SSP=0.049 (assimilation), 0.084 (prediction)",
        ],
        limits=[
            "**~30 분 학습 / instance** (RTX 3090, 40K+ epochs) — instance마다 처음부터",
            "**1D + t**, future work에서 명시: 2D + t extension",
            "**Buoy time-series only** — sparse spatial LiDAR setup 처리 안 함",
        ],
        contrast=[
            "**속도 차이가 결정적**: Ours 13 ms vs PFT-PINN ~30 min = **약 130,000× 차이**",
            "Ours: **2D + t 이미 가능**, **LiDAR raydrop sensor model 포함**",
            "**위협**: 같은 그룹 (TUHH+Imperial)이 1-3년 안에 2D + real-time PINN 발표 가능 — 우리 niche 좁아질 risk",
        ],
        notes=(
            "가장 위협적인 prior art입니다. 이유는 우리와 *근본 철학이 같기* 때문입니다 — "
            "training data 없이 physics constraint (dispersion + boundary conditions)을 직접 강제. "
            "다만 구현 방식이 다릅니다.\n\n"
            "PFT-PINN의 방식:\n"
            "  - 두 NN을 동시에 학습: η-net (wave 표면) + Φ-net (velocity potential)\n"
            "  - 손실함수: Laplace 방정식 + 표면 boundary condition + 바닥 condition\n"
            "  - Auto-differentiation으로 PDE residual 계산 (NN의 강점)\n"
            "  - Hard-constraint trick — 측정치를 NN architecture에 직접 hard-encode (loss balancing 문제 회피)\n\n"
            "이게 1D wave flume에서는 잘 작동합니다. 30분 학습 후 SSP < 0.05 정도.\n\n"
            "한계는:\n"
            "  - Instance마다 30분 학습 — design-time analysis tool로는 OK, real-time 제어로는 불가\n"
            "  - 1D + t만 다룸 — 2D directional은 future work\n"
            "  - Buoy 시계열만 다룸 — sparse spatial LiDAR setup 처리 안 함\n\n"
            "그러나 결정적인 위협은 *future work*입니다. 같은 그룹이 'real-time speedup via parallelization, "
            "pre-training, multi-GPU' 와 '2D + t extension'을 명시했어요. 1-3년 안에 우리 niche를 흡수할 수 있습니다.\n\n"
            "대응 전략 두 가지:\n"
            "  1. 우리가 빨리 publish — 시간 우위 확보\n"
            "  2. 응용 영역 차별화 — PINN은 design-time 분석 도구, 우리는 real-time 제어 component"
        ),
        star=True,
    )


def build_s9():
    build_prior_art(
        num=9,
        title_full="Yan, Che, Li 2025 (arXiv 2510.14356) — VOF-EnKF",
        key_msg=[
            "**VOF + Navier-Stokes + EnKF 첫 결합** — full CFD 시뮬레이션을 데이터 동화",
            "  VOF = Volume Of Fluid, 자유 표면 (water-air 경계) 추적 방법",
            "**POD 차원 축소**: 격자 4.8e4 cells → 30 modes (level-set 4 modes)",
            "  POD = Proper Orthogonal Decomposition, 주성분 분석과 유사",
            "**Plunging breaker 재구성** — 부서지는 파도까지 처리. 단, breaking 이전 upstream에서만 동화",
        ],
        limits=[
            "**Offline** — 50 ensemble × full OpenFOAM (Navier-Stokes solver) = 매우 느림",
            "**1D xz-plane only** — 2D directional 미처리 (xz는 진행 방향과 수직 단면만)",
            "**Inlet boundary 처리 어려움** — 입구 근처 RMS 0.228 H_s vs 하류 0.090",
        ],
        contrast=[
            "속도: Ours 13 ms vs VOF-EnKF (offline) — **응용 영역이 본질적으로 다름**",
            "Ours: **2D directional**, VOF-EnKF: 1D xz",
            "그들의 강점: **wave breaking** (우리는 이걸 limitation으로 인정하고 future work로 둠)",
        ],
        notes=(
            "다른 극단의 prior art입니다 — full physics fidelity (가장 정확한 물리)를 추구.\n\n"
            "방법론:\n"
            "  - VOF (Volume Of Fluid) - 자유 표면을 fluid volume fraction α로 표현. α=1 (물), 0 (공기), 0.5 (계면)\n"
            "  - Navier-Stokes 풀이 (OpenFOAM) - 가장 정확한 fluid dynamics\n"
            "  - EnKF로 데이터 동화 - 50 ensemble을 동시에 propagate\n"
            "  - POD 차원 축소 - 격자 cell 4.8만 개를 30 modes로 압축 (계산 가능하게)\n\n"
            "강점:\n"
            "  - Plunging breaker (부서지는 파도)까지 처리 — wave physics의 가장 어려운 영역\n"
            "  - Wave-structure interaction까지 확장 가능 (선박/구조물 포함)\n"
            "  - 물리적으로 가장 정확\n\n"
            "한계:\n"
            "  - Offline 외에 다른 게 없음 — 50 OpenFOAM 시뮬레이션을 매 step마다 돌림\n"
            "  - 1D xz-plane만 — 2D directional sea state 처리 안 함\n"
            "  - Inlet boundary 근처 정확도 떨어짐 (boundary가 state에 포함 안 됨)\n\n"
            "응용 영역이 본질적으로 다릅니다:\n"
            "  - VOF-EnKF: design-time digital twin (사후 분석, 매우 정확하게)\n"
            "  - 우리: real-time control component (즉시 결정, 충분히 정확하게)\n\n"
            "정직한 인정 — 우리는 wave breaking을 처리 못 합니다. 이건 paper의 limitation 섹션에 명시하고 future work로 둡니다."
        ),
        star=True,
    )


# S10 — Application matrix + niche
def build_s10():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 1 · Niche — 응용 매트릭스로 우리 위치 확정")
    add_title(s, "WaveSplat의 응용 영역 — 어디에 적합한가?")
    add_subtitle(s, "Real-time control loop vs Design-time analysis — 두 응용 영역으로 분리")
    headers = ["응용 요구사항", "Ours", "PFT-PINN", "VOF-EnKF",
               "EnKF-HOS", "FNO", "Desmars"]
    rows = [
        ["Real-time 제어 루프 (선박/hydrofoil)", "✓", "✗", "✗", "✗", "✓", "✗"],
        ["2D directional phase-resolved", "✓", "✗", "✗", "✗ (1D)", "✗", "✗"],
        ["Wave breaking, plunging (부서지는 파도)", "✗", "✗", "✓", "✗", "✗", "✗"],
        ["Wave-structure interaction (구조물 하중)", "✗", "✗", "✓", "✗", "✗", "✗"],
        ["Velocity field 전체 추정", "✗", "✓", "✓", "✓", "✗", "partial"],
        ["Bathymetry inversion (수심 추정)", "✗", "✓ ext.", "✗", "✗", "✗", "✗"],
        ["Training-free unsupervised", "✓", "✓", "✓", "✓", "✗", "✓"],
        ["Sparse LiDAR sensor model", "✓", "✗", "✗", "✗", "✗", "partial"],
    ]
    add_table_simple(s, 0.4, 1.85, 12.5, 4.2, headers, rows,
                     header_color=ACCENT, font_size=10, highlight_col=1)
    add_text(s, 0.5, 6.15, 12.3, 0.35, "→ 우리 niche:",
             font_size=14, bold=True, color=ACCENT)
    add_bullets(s, 0.6, 6.5, 12.0, 0.85, [
        "**Real-time + 2D directional + sparse LiDAR + unsupervised** — 12편 prior art 중 0개가 모두 만족",
        "응용: 선박 ride control, hydrofoil FF preview, autonomous navigation (real-time 제어 영역)",
    ], font_size=12)
    add_page_number(s, 10)
    add_notes(s, (
        "이 매트릭스가 우리 paper의 positioning 핵심입니다. 8개 응용 요구사항 × 7개 method를 비교했고, "
        "어떤 method도 모든 요구사항을 만족하지 못합니다 — 각자의 specialty가 있습니다.\n\n"
        "우리 specialty (highlighted column): real-time + 2D directional + sparse LiDAR + unsupervised. "
        "이 4개 조건의 조합이 12편 prior art 중 0개에서만 만족. 이게 우리 niche의 본질입니다.\n\n"
        "응용 영역으로 바라보면 두 갈래로 나뉩니다:\n"
        "  - 'Design-time analysis' 영역: VOF-EnKF, PFT-PINN — 사후 분석, 가장 정확하게, 시간 무관\n"
        "  - 'Real-time control' 영역: 우리, FNO — 즉시 결정, 충분히 정확하게\n\n"
        "우리는 PFT-PINN이나 VOF-EnKF와 *경쟁하지 않습니다* — 그들은 다른 응용 영역. "
        "직접 경쟁자는 FNO (1D supervised) — 우리는 2D unsupervised로 차별화.\n\n"
        "정직한 한계:\n"
        "  - Wave breaking 못함 — VOF-EnKF만 가능\n"
        "  - Velocity field 추정 못함 — PINN/EnKF 가능\n"
        "  - Full nonlinear 못함 — Klein 2020가 정량 비교한 HOSM > NLSE > LWT 중 우리는 LWT family\n\n"
        "이런 한계는 paper의 limitation 섹션에 정직히 명시. 다음부터는 *우리가 만든 것*을 보여드립니다."
    ))


# S11 — WaveSim physics + raydrop
def build_s11():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 2 · WaveSim — 우리가 만든 simulator", color=ACCENT2)
    add_title(s, "WaveSim — physics + LiDAR sensor model")
    add_subtitle(s, "선형 중첩 + 분산 + JONSWAP 스펙트럼, 그리고 LiDAR raydrop 물리까지")
    bullets = [
        "**Physics (수면 모델)**:",
        "  η(x, y, t) = Σ_n A_n cos(k_xn·x + k_yn·y − ω_n·t + φ_n)",
        "  N개 plane wave의 선형 중첩 (Linear Wave Theory, LWT)",
        "",
        "**Dispersion relation (분산 관계, 깊은 물)**:",
        "  ω_n = √(g·|k_n|·tanh(|k_n|·d))   — 주파수와 wavenumber의 1-to-1 매핑",
        "",
        "**Spectrum**: JONSWAP-2D (Hasselmann 1973 + 방향 spreading), N_components 가변",
        "",
        "**LiDAR sensor model** (이 부분이 우리의 unique contribution):",
        "  Ray-triangle intersection (Möller-Trumbore 1997, 검증된 알고리즘)",
        "  Raydrop probability — Fresnel reflectance (specular vs diffuse 선택)",
        "  Geometric shadowing — 가까운 파도가 먼 파도를 가림",
        "  Vessel motion + ego-velocity correction",
        "",
        "**검증**: 39 unit test 통과, 100 episode 합성 dataset 생성",
    ]
    add_bullets(s, 0.6, 1.85, 8.0, 5.3, bullets, font_size=12,
                line_space_after=2)
    fig_field = os.path.join(DOC_DIR, 'wavesim_BF4_5c_wave_field.png')
    add_image_or_placeholder(s, fig_field, 8.7, 2.0, 4.3,
                             placeholder_text='[wavesim wave field]')
    add_page_number(s, 11)
    add_notes(s, (
        "WaveSim은 우리가 만든 simulator입니다. 두 부분으로 나뉩니다.\n\n"
        "(1) 수면 model (physics):\n"
        "  - Linear Wave Theory — N개 plane wave의 선형 중첩으로 수면 표현\n"
        "  - Dispersion relation — frequency ω와 wavenumber k의 관계 (깊은 물에서 ω² = g·k·tanh(kd))\n"
        "  - JONSWAP-2D 스펙트럼 — 실제 바다의 통계적 특성 모방\n\n"
        "한계는 정직하게 인정합니다 — Klein 2020이 정량 비교한 HOSM > NLSE > LWT 중 우리는 LWT family로 가장 단순. "
        "ε (steepness) < 0.07 영역에서만 신뢰 가능.\n\n"
        "(2) LiDAR sensor model — 이게 우리의 unique contribution:\n"
        "  - Ray-triangle intersection: LiDAR ray가 mesh와 만나는 점을 계산. Möller-Trumbore 1997 알고리즘 사용 (그래픽 분야 표준)\n"
        "  - Raydrop: 파도 표면에서 ray가 반사되어 LiDAR에 못 돌아오는 확률. Fresnel reflectance 사용 (specular = 거울 반사, diffuse = 산란)\n"
        "  - Shadowing: 앞쪽 파도가 뒤쪽 파도를 가림\n"
        "  - Vessel motion: 선박이 V=11.3 m/s로 움직이므로 매 frame 다른 위치에서 관측\n\n"
        "12편 prior art 중 이 정도 sensor model을 simulate한 것은 없습니다. "
        "39개 unit test로 simulator 정확성 검증, 100 episode를 합성 dataset으로 생성했습니다."
    ))


# S12 — WaveSim 100 episodes
def build_s12():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 2 · WaveSim — 합성 dataset", color=ACCENT2)
    add_title(s, "100 episode — 통제된 benchmark")
    add_subtitle(s, "재현 가능, 다양한 복잡도, ground-truth η + LiDAR 점 모두 보존")
    bullets = [
        "**Episode 변수**:",
        "  Sea state: BF3 (잔잔), BF4 (보통), BF6 (거친) — Beaufort scale",
        "  N_components: 1, 5, 15 (저/중/고 복잡도)",
        "  Spreading (방향성): uniform / unimodal / bimodal",
        "",
        "**Episode당 데이터**:",
        "  T = 30 s, Δt = 0.1 s → 300 frames",
        "  Ground-truth η field: 50×50 격자 (참값으로 평가용)",
        "  LiDAR scan: ~50 valid pts/frame (raydrop 후)",
        "  HDF5 format, 100 files, 총 ~5 GB",
        "",
        "**Test coverage**:",
        "  39 unit test (wavesim/) + 18 (wavesplat/) = **57 total, 모두 통과**",
        "  → simulator + estimator의 정확성 보증",
    ]
    add_bullets(s, 0.6, 1.85, 8.0, 5.3, bullets, font_size=13,
                line_space_after=3)
    fig_lidar = os.path.join(DOC_DIR, 'wavesim_BF4_5c_wave_lidar.png')
    add_image_or_placeholder(s, fig_lidar, 8.7, 2.0, 4.3,
                             placeholder_text='[BF4 LiDAR scan]')
    add_page_number(s, 12)
    add_notes(s, (
        "100 episode의 통제된 benchmark를 구축했습니다.\n\n"
        "Episode 설계:\n"
        "  - Sea state: BF3 (잔잔, ε~0.03), BF4 (보통, ε~0.05), BF6 (거친, ε~0.08) — 모두 우리 LWT 신뢰 영역\n"
        "  - N_components: 1 (단순), 5 (중간), 15 (복잡) — 알고리즘이 component 수에 robust한지 검증\n"
        "  - Directional spreading: uniform (전 방향 균등), unimodal (한 방향 집중), bimodal (두 방향 swell+wind)\n\n"
        "Episode당 데이터:\n"
        "  - 30초 시뮬레이션, 0.1초 간격 → 300 frame\n"
        "  - Ground-truth η field 50×50 격자 — 평가 시 참값\n"
        "  - LiDAR scan은 raydrop/shadowing 적용 후 frame당 ~50점\n\n"
        "전체 100 episode를 HDF5 파일로 저장 (~5 GB).\n\n"
        "Reproducibility 보장:\n"
        "  - Seed 저장 — 동일 seed로 재실행 시 동일 결과\n"
        "  - 39 + 18 = 57개 unit test — simulator와 estimator 모두 검증\n\n"
        "이 dataset 자체가 paper의 contribution 일부입니다. Ehlers 2023도 비슷한 합성 benchmark를 만들었지만 1D였고, "
        "우리는 2D directional + LiDAR sensor model까지 포함."
    ))


# S13 — WaveSplat algorithm
def build_s13():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 3 · WaveSplat — 우리 estimator (탐색 단계)", color=GOLD)
    add_title(s, "WaveSplat — parametric off-grid estimator")
    add_subtitle(s, "5N 파라미터 (k_x, k_y, ω, A, φ) × N개 component를 직접 추정")
    bullets = [
        "**State 표현**: N개 component × {A_n, k_xn, k_yn, ω_n, φ_n}",
        "  Dispersion 적용 후: ω_n = √(g·|k_n|·tanh(|k_n|·d))",
        "  → 자유도 감소: 5N → 4N (ω가 k에 의해 결정됨)",
        "",
        "**Two-stage estimation** (이게 후반부 main framing의 motivation):",
        "  **Init** — 다중 frame: 2-stage beamforming으로 (k_x, k_y)만 추정",
        "  **Track** — 매 frame: (k_x, k_y) 고정, (ω, A, φ)만 Adam으로 update",
        "",
        "**Adaptive N (component 수 동적 조정)**:",
        "  Energy-based pruning — 작은 amplitude component 제거",
        "  Residual-triggered growth — 잔여 오차 크면 새 component 추가",
        "  Safety rollback — RMSE > 130%면 이전 상태로 복귀",
        "",
        "**Loss**: ‖η_LiDAR − η_model(θ)‖² + λ·(dispersion penalty)",
        "  Dispersion은 hard constraint로 강제 (LWT family 가정)",
    ]
    add_bullets(s, 0.6, 1.85, 8.0, 5.3, bullets, font_size=12,
                line_space_after=2)
    fig_pipe = os.path.join(PAPER_FIG_DIR, 'fig_pipeline.png')
    add_image_or_placeholder(s, fig_pipe, 8.7, 2.0, 4.3,
                             placeholder_text='[WaveSplat pipeline]')
    add_page_number(s, 13)
    add_notes(s, (
        "WaveSplat의 탐색 단계 알고리즘입니다. 핵심 아이디어:\n\n"
        "(1) State 표현 — N개 component × 5 파라미터 = 5N 자유도. "
        "Dispersion 관계로 ω가 k에 의해 결정되므로 실제 자유도는 4N. 매우 compact.\n\n"
        "(2) Two-stage 추정 — 단일 frame에서 (k_x, k_y)는 ill-conditioned (정보 부족). "
        "그래서 init 단계는 multi-frame으로 (k_x, k_y)만 정확히 추정하고, "
        "track 단계는 (k_x, k_y) 고정 후 매 frame (ω, A, φ)만 update. "
        "이게 후반부 main framing 'identifiability decomposition'의 motivation입니다.\n\n"
        "(3) Adaptive N — 처음에는 N을 모르고, episode마다 N이 다를 수 있음. "
        "  - Pruning: amplitude 작은 component 제거\n"
        "  - Growth: residual 크면 새 component 추가\n"
        "  - Safety rollback: 새 component가 잘못되어 RMSE 폭증하면 원상복구\n\n"
        "(4) Loss — 측정치-모델 차이 squared error + dispersion penalty. "
        "Dispersion은 항상 만족하도록 hard constraint.\n\n"
        "이 알고리즘이 *왜 작동하는지*에 대한 분석이 다음 슬라이드부터의 main framing입니다."
    ))


# S14 — Prior result (P1 12-ep + real-time)
def build_s14():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 3 · WaveSplat — 탐색 단계 결과", color=GOLD)
    add_title(s, "탐색 단계 결과 — 12 episode 비교")
    add_subtitle(s, "WaveSplat이 best avg nRMSE; ETKF/GPR보다 정확하면서 10× 빠름")
    headers = ["Method", "Avg nRMSE", "Per-frame time", "Setup"]
    rows = [
        ["WaveSplat (ours)", "0.223", "13.4 ms", "parametric off-grid + dispersion hard"],
        ["GPR (single + window)", "0.276", ">200 ms", "Gaussian Process, dense kernel"],
        ["ETKF-grid (tuned, Ne=100)", "0.319", "132.6 ms", "Hunt 2007 LETKF, grid-spatial"],
        ["Oracle (lower bound)", "0.087", "—", "ground-truth components, fit only A,φ"],
    ]
    add_table_simple(s, 0.5, 1.85, 12.3, 2.4, headers, rows,
                     header_color=ACCENT, font_size=12, highlight_col=0)
    add_bullets(s, 0.6, 4.4, 12.0, 2.7, [
        "**속도 우위**: 13 ms vs 132 ms (ETKF) vs >200 ms (GPR) — **ETKF보다 10× 빠름**",
        "**정확도 우위**: 0.223 vs 0.276 vs 0.319 — **unsupervised method 중 best average**",
        "Oracle 0.087 = 환원 불가능한 noise floor (sensor + 수치 오차) — 이론적 하한",
        "**Important**: 이 결과는 *탐색 단계* (현재 알고리즘) — main framing은 *왜 잘 되는가*의 분석에서 나옴",
    ], font_size=13, line_space_after=3)
    add_page_number(s, 14)
    add_notes(s, (
        "12 episode 비교 결과입니다. 표를 보시면:\n\n"
        "  - WaveSplat (우리): nRMSE 0.223, 13.4 ms — 가장 정확하면서 가장 빠름\n"
        "  - GPR: nRMSE 0.276, >200 ms — Gaussian Process baseline, 정확도 떨어지고 속도도 느림\n"
        "  - ETKF (tuned): nRMSE 0.319, 132.6 ms — Wang-Pan family의 simplified implementation, 100개 ensemble로 tune\n"
        "  - Oracle: 0.087 — ground-truth components 알고 있을 때의 lower bound (sensor noise floor)\n\n"
        "결과 해석:\n"
        "  - 정확도: WaveSplat이 unsupervised method 중 best (0.223). Oracle 0.087과의 gap은 알고리즘 추정 오차.\n"
        "  - 속도: 13.4 ms는 10 Hz control loop에 충분. ETKF의 132 ms는 빡빡하고, GPR 200 ms는 불가능.\n\n"
        "여기서 '잠깐'하고 멈춰야 합니다 — 이 결과 자체가 main contribution이 아닙니다. "
        "이건 *탐색 단계* 결과 — 우리가 알고리즘을 만들고 돌려본 결과. "
        "Main contribution은 다음에 보일 *왜 이렇게 잘 되는가*에 대한 본질적 분석입니다.\n\n"
        "그 분석에서 흥미로운 발견이 있었습니다. 우리 알고리즘이 *불가능해 보이는 일을 하고 있다*는 것."
    ))


# S15 — Spatial Nyquist 한계
def build_s15():
    s = add_blank(LIGHT_BG)
    add_section_tag(s, "Part 4 · The essential insight — 본질적 발견", color=ACCENT2)
    add_title(s, "핵심 질문 — 우리 parametric 모델이 왜 이렇게 잘 작동하는가?")
    add_subtitle(s, "Per-frame spatial Nyquist는 ocean wave 추정에 너무 작은데도, WaveSplat은 추정에 성공")
    bullets = [
        "**Per-frame LiDAR 유효 sampling**:",
        "  ~50 valid 점 / 240 m FOV → 유효 점 간격 Δy ≈ 5–8 m",
        "  Spatial Nyquist 한계: k_Nyq = π/Δy ≈ **0.4 rad/m**",
        "  → 이론적으로 이 이상 큰 wavenumber는 single frame으로 추정 불가능",
        "",
        "**필요한 wavenumber 범위 (BF3-BF6 sea state)**:",
        "  파장 λ ∈ 5–25 m → wavenumber k ∈ **0.5–3 rad/m**",
        "  → Nyquist 한계의 1.25×–7.5× 범위",
        "",
        "**🚨 Paradox**: 우리가 추정하려는 k는 per-frame Nyquist를 1.25–7.5× 초과!",
        "",
        "**그런데 WaveSplat은 이걸 추정합니다. 어떻게?**",
        "  Hint: 우리는 multi-frame (시계열) data와 dispersion physics를 사용",
        "  → 다음 슬라이드: mechanism의 본질",
    ]
    add_bullets(s, 0.7, 1.85, 12.0, 5.5, bullets, font_size=15,
                line_space_after=5)
    add_page_number(s, 15)
    add_notes(s, (
        "여기서 가장 중요한 발견이 나옵니다. 알고리즘을 분석하면서 우리가 깨달은 것 — "
        "*우리가 불가능해 보이는 일을 하고 있다*.\n\n"
        "이론적 한계를 봅시다:\n\n"
        "(1) Per-frame LiDAR가 유효 점 50개를 240 m FOV에 가지고 있으면 유효 점 간격은 5-8 m. "
        "Nyquist-Shannon 정리에 따르면 single frame으로 추정 가능한 spatial frequency 한계는 "
        "k_Nyq = π/Δy ≈ 0.4 rad/m. 즉 이 이상 큰 wavenumber는 single frame으로 못 추정합니다.\n\n"
        "(2) 그런데 ocean wave의 typical wavenumber는 0.5-3 rad/m. "
        "BF3 (잔잔)에서도 k ≈ 0.5-1, BF6 (거친)에서는 k ≈ 1-3. "
        "이건 per-frame Nyquist의 1.25-7.5배.\n\n"
        "(3) 그런데 WaveSplat은 이걸 추정합니다. 12 episode에서 nRMSE 0.223. 어떻게 가능한가?\n\n"
        "여기에 우리 main framing의 핵심이 있습니다 — 두 가지 mechanism이 결합해서 spatial Nyquist 한계를 우회합니다:\n"
        "  1. 선박이 움직이면서 발생하는 synthetic aperture 효과\n"
        "  2. 파도의 dispersion 관계 (ω → |k|의 1-to-1 매핑)\n\n"
        "이 두 mechanism의 결합이 다음 슬라이드의 main contribution claim입니다."
    ))


# S16 — Main framing
def build_s16():
    s = add_blank(LIGHT_BG)
    add_section_tag(s, "Part 4 · Main framing ⭐⭐", color=ACCENT2)
    add_title(s, "Dispersion-Coupled Spatio-Temporal Inversion",
              size=24, color=ACCENT)
    add_text(s, 0.5, 1.4, 12.3, 1.5,
             ('"Vessel motion (synthetic aperture) and dispersion physics jointly '
              'resolve ocean wavenumbers 3–7× beyond the per-frame spatial Nyquist '
              'of shipboard LiDAR."'),
             font_size=17, italic=True, color=DARK,
             alignment=PP_ALIGN.CENTER)
    bullets = [
        "**Mechanism 1 — Synthetic aperture (선박 움직임을 활용)**",
        "  선박이 V=11.3 m/s로 1초 동안 11.3 m 이동 → 유효 spatial coverage 확장",
        "  Δx_SA = V · T_init = 11.3 m   (vs single-frame Δy ≈ 5–8 m)",
        "  → SAR (synthetic aperture radar)와 같은 원리, ocean LiDAR에 첫 적용",
        "",
        "**Mechanism 2 — Dispersion 1-to-1 매핑 (물리법칙 활용)**",
        "  ω = √(g·|k|·tanh(|k|·d))   — 깊은 물 dispersion 관계",
        "  각 ω는 유일한 |k|를 결정 → temporal info를 spatial wavenumber로 전환",
        "",
        "**Joint mechanism (두 mechanism의 결합)**:",
        "  Multi-frame stacking → temporal coherence → ω resolution",
        "  ω → |k| via dispersion → spatial wavenumber (Nyquist 우회)",
        "  Direction sweep → (k_x, k_y) 결정",
        "",
        "**Quantitative claim**: super-resolution factor **3–7×** beyond per-frame Nyquist",
        "  Phase 1에서 4-way ablation으로 정량 증명 예정 (100 episode)",
    ]
    add_bullets(s, 0.6, 3.05, 12.0, 4.3, bullets, font_size=12,
                line_space_after=3)
    add_page_number(s, 16)
    add_notes(s, (
        "이게 우리 paper 1의 main contribution claim입니다. Per-frame spatial Nyquist 한계를 우회하는 mechanism 둘:\n\n"
        "(1) Synthetic aperture — SAR (synthetic aperture radar) 분야에서 유명한 원리. "
        "위성/항공기가 움직이면서 매 시점 다른 위치에서 관측하면, 짧은 antenna로도 큰 antenna 효과를 얻습니다. "
        "우리는 이걸 ocean LiDAR에 적용 — 선박이 11.3 m/s로 움직이며 1초간 11.3 m 이동. "
        "이게 sigle-frame Δy 5-8 m보다 1.5-2배 큰 effective aperture.\n\n"
        "(2) Dispersion 1-to-1 매핑 — 깊은 물에서 ω² = g·|k|·tanh(|k|·d). "
        "이게 무엇을 의미하나? *각 frequency ω는 유일한 wavenumber |k|를 결정*. "
        "Time-series에서 ω는 frame rate (10 Hz, Nyquist 5 Hz)로 추정 가능 — 매우 큼. "
        "그 ω를 dispersion에 넣으면 |k|가 결정. 이게 temporal info를 spatial로 전환하는 기적.\n\n"
        "(3) Joint mechanism — 두 mechanism이 함께 작용해야 작동. 단독으로는 부족:\n"
        "  - Synthetic aperture만: aperture는 1.5-2x 증가, 충분치 않음 (Nyquist 우회 불가)\n"
        "  - Dispersion만: |k|는 알아도 (k_x, k_y) 방향은 별도 정보 필요\n"
        "  - 결합 시: ω → |k| + multi-frame coherence → (k_x, k_y) 방향 sweep으로 모두 결정\n\n"
        "Quantitative claim:\n"
        "  - Per-frame Nyquist 0.4 rad/m\n"
        "  - 우리가 추정하는 k 범위 0.5-3 rad/m\n"
        "  - Super-res factor = 3 / 0.4 = 7.5x (worst), 0.5 / 0.4 = 1.25x (best)\n\n"
        "Phase 1 작업으로 4-way ablation (single-frame / multi-frame-spatial-only / multi-frame-disp-postproc / multi-frame-disp-first) "
        "을 100 episode에서 돌려서 이 mechanism을 정량 증명할 예정입니다."
    ))


# S17 — Mechanism + Identifiability
def build_s17():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 4 · Mechanism — supporting analysis", color=ACCENT2)
    add_title(s, "추가 통찰 — 왜 init/track 분리가 자연스러운가")
    add_subtitle(s, "Identifiability decomposition + dispersion-coupled super-resolution")
    bullets = [
        "**Identifiability analysis (Fisher Information 관점)**:",
        "  k = (k_x, k_y) — multi-frame 에서만 identifiable (aperture 필요)",
        "  (ω, φ, A) — single frame에서도 identifiable (snapshot 하나로 충분)",
        "",
        "**→ 이 분해가 우리 two-stage 설계의 정당화**:",
        "  Init phase (multi-frame, 무거움): synthetic aperture + dispersion으로 k 한 번에 결정",
        "  Track phase (per-frame, 가벼움): k 고정 후 (ω, A, φ)만 Adam으로 cheap update",
        "  → Init은 < 1초 (오프라인 가능), Track은 < 100 ms (real-time 보장)",
        "",
        "**전체 story 정리**:",
        "  Init: SA + dispersion → k_max 3-7× 우회 (super-resolution)",
        "  Track: per-frame Adam으로 inference < 100 ms 유지",
        "  Adaptive N + safety rollback: component 수 변화에 robust",
        "",
        "**Phase 1 추가 작업**: FIM 정량 분석으로 identifiability 정량 증명",
    ]
    add_bullets(s, 0.6, 1.85, 8.0, 5.3, bullets, font_size=12,
                line_space_after=3)
    fig_bf = os.path.join(PAPER_FIG_DIR, 'fig_beamforming_2stage.png')
    add_image_or_placeholder(s, fig_bf, 8.7, 2.0, 4.3,
                             placeholder_text='[2-stage beamforming]')
    add_page_number(s, 17)
    add_notes(s, (
        "두 가지 supporting insight를 더합니다.\n\n"
        "(1) Identifiability decomposition — Fisher Information Matrix (FIM) 분석으로:\n"
        "  - k = (k_x, k_y)는 single frame에서 ill-conditioned (FIM eigenvalue 작음)\n"
        "    → multi-frame coherence가 있어야 identifiable\n"
        "  - (ω, φ, A)는 single frame에서도 well-conditioned\n"
        "    → snapshot 하나로 충분히 추정 가능\n\n"
        "이게 우리 init/track two-stage 설계의 *theoretical justification*입니다. "
        "현재는 직관적 설계 — Phase 1에서 FIM 정량 분석으로 증명할 예정.\n\n"
        "(2) Combined story — main framing과 연결:\n"
        "  - Init phase는 multi-frame이라 무겁지만 < 1초. Synthetic aperture + dispersion으로 k 추정.\n"
        "  - Track phase는 per-frame이라 가벼움 < 100 ms. k 고정 후 (ω, A, φ)만 Adam으로 update.\n"
        "  - Adaptive N + safety rollback이 component 수 변화에 robust성 보장.\n\n"
        "이 정도가 우리 paper 1의 main contribution입니다 — '우리가 한 일이 이거다'를 명확히 하고, "
        "'왜 이렇게 잘 작동하는지'를 mechanism으로 설명, '이게 prior art와 어떻게 다른지'를 매트릭스로 정리.\n\n"
        "Phase 1 작업이 끝나면:\n"
        "  - 4-way ablation으로 super-resolution factor 정량 증명\n"
        "  - FIM 분석으로 identifiability 정량 증명\n"
        "  - Theoretical bound (1-page derivation)\n\n"
        "이 셋이 paper 1을 OE(Ocean Engineering) 수준으로 격상시키는 작업입니다."
    ))


# S18 — Limitations + future + threat
def build_s18():
    s = add_blank(WHITE)
    add_section_tag(s, "Closing · 한계와 대응 전략", color=GRAY)
    add_title(s, "정직한 한계 + 위협 대응")
    add_subtitle(s, "Linear approximation, no breaking, PFT-PINN의 잠재적 위협")
    bullets = [
        "**Limitation 1 — Linear approximation**:",
        "  Klein 2020 정량 비교: HOSM > NLSE₂ > LWT (우리 family는 가장 단순)",
        "  Operational range: ε < 0.07 (BF6까지는 신뢰 가능, 그 이상은 위험)",
        "  Future: Stokes ε² correction 추가 (Phase 2, JFM upgrade)",
        "",
        "**Limitation 2 — No wave breaking, no velocity inference**:",
        "  VOF-EnKF, PFT-PINN은 이걸 처리 — 응용 영역이 본질적으로 다름",
        "  Future: hybrid linear-realtime + on-demand CFD refinement",
        "",
        "**Threat — PFT-PINN의 2D + real-time 확장**:",
        "  Same group (TUHH+Imperial)이 1-3년 안에 발표 가능 (future work에 명시됨)",
        "  Response 1: **빨리 publish** — OE-ready by 7월, JFM upgrade option",
        "  Response 2: **응용 영역 차별화** — PINN = design-time, 우리 = real-time control component",
    ]
    add_bullets(s, 0.6, 1.85, 12.0, 5.5, bullets, font_size=13,
                line_space_after=4)
    add_page_number(s, 18)
    add_notes(s, (
        "정직한 disclosure 시간입니다.\n\n"
        "(1) Linear approximation — 가장 큰 한계입니다. Klein 2020이 정량 비교했어요:\n"
        "  - HOSM (가장 정확): ε > 0.1까지 처리\n"
        "  - NLSE2 (중간): ε ~ 0.05-0.07\n"
        "  - LWT (우리, 가장 단순): ε < 0.07에서만 신뢰\n"
        "Operational range 명시 — BF6 (ε~0.08)까지는 신뢰 가능. 그 이상의 거친 sea state는 risk. "
        "Phase 2 작업으로 Stokes ε² correction 추가 — 두 번째 차수까지 보정해서 ε ~ 0.10까지 확장.\n\n"
        "(2) Wave breaking, velocity field 추정 — 우리 영역 밖. "
        "VOF-EnKF가 wave breaking 처리, PFT-PINN/EnKF가 velocity field 추정. "
        "Future work: hybrid linear-realtime estimator + 필요 시 CFD refinement.\n\n"
        "(3) Threat — Ehlers 2025 PFT-PINN 그룹의 future work에 명시:\n"
        "  - 'Improving training speed via parallelization, multi-GPU, pre-training'\n"
        "  - '2D + t extension'\n"
        "둘 다 만족하는 PFT-PINN-2D-realtime이 1-3년 안에 발표 가능. "
        "그러면 우리 niche (real-time + 2D)가 흡수될 수 있습니다.\n\n"
        "대응 전략:\n"
        "  1. 빨리 publish — 7월 OE-ready, 12월 submit. 시간 우위 확보.\n"
        "  2. 응용 영역 차별화 — PINN은 본질적으로 30분 학습이라 design-time analysis tool 영역. "
        "     우리는 real-time control component 영역. 두 영역의 응용 demand가 분리되어 있다는 것을 paper에 명시."
    ))


# S19 — Timeline + venue + Q&A
def build_s19():
    s = add_blank(LIGHT_BG)
    add_section_tag(s, "Closing · Plan — publish 일정", color=ACCENT)
    add_title(s, "Publication plan — Phase 1 → Decision gate → Phase 2", color=DARK)
    add_subtitle(s, "5월–12월 timeline, OE/AOR submit (JFM upgrade option)")
    bullets = [
        "**Phase 1 — OE-ready (5월–7월, 2 개월)**:",
        "  4-way ablation 코드 (single / multi-spatial / multi-disp-post / multi-disp-first)",
        "  100 episode에서 super-res factor 정량 측정",
        "  Theoretical bound derivation (1-page math)",
        "  Paper 본문 재작성 — Dispersion-Coupled Spatio-Temporal Inversion framing",
        "",
        "**Decision gate (7월 말)**:",
        "  Super-res factor 3–7× 정량 증명 성공 → JFM/PRF 도전 (Phase 2)",
        "  결과 약하면 → OE 빠르게 submit",
        "",
        "**Phase 2 — JFM/PRF upgrade (8월–10월, 3 개월, 선택)**:",
        "  Sample complexity bound (atomic-norm 기반)",
        "  Stokes ε² correction (linear → 2nd order)",
        "  Identifiability FIM 정량 분석",
        "",
        "**Submit**: 11월–12월 (OE/AOR 또는 JFM/PRF)",
    ]
    add_bullets(s, 0.6, 1.85, 12.0, 4.9, bullets, font_size=12,
                line_space_after=3)
    add_text(s, 0.5, 6.85, 12.3, 0.4,
             "Questions & discussion — feedback most welcomed",
             font_size=14, italic=True, color=ACCENT,
             alignment=PP_ALIGN.CENTER)
    add_page_number(s, 19)
    add_notes(s, (
        "Publication plan 정리:\n\n"
        "Phase 1 (5월-7월, 2개월): OE-ready 만들기. 핵심 작업:\n"
        "  - 4-way ablation script — single-frame / multi-frame-spatial-only / multi-frame-dispersion-postproc / multi-frame-dispersion-first\n"
        "    이 4개 비교로 *어느 mechanism이 super-res에 기여하는지* 정량 분리\n"
        "  - 100 episode 실험\n"
        "  - Theoretical bound 1-page 수학 — 'V, T_init이 주어지면 recoverable k_max는?'\n"
        "  - Paper 본문 재작성 — abstract/intro/method/result/discussion 모두 새 framing으로\n\n"
        "Decision gate (7월 말):\n"
        "  - Super-res factor 3-7x 정량 확인 → JFM 도전\n"
        "  - 약하면 (예: 2-3x만) → OE 빨리 submit (안전한 venue)\n\n"
        "Phase 2 (8월-10월, 3개월, 선택):\n"
        "  - Sample complexity bound — atomic-norm-inspired (Tang 2013 기반)\n"
        "  - Stokes ε² correction — 2차 비선형 보정으로 LWT 한계 극복\n"
        "  - FIM identifiability 정량 분석\n\n"
        "Submit (11월-12월):\n"
        "  - Phase 1만 끝낸 경우: OE/AOR (Ocean Engineering, Applied Ocean Research)\n"
        "  - Phase 2까지: JFM/PRF (Journal of Fluid Mechanics, Physical Review Fluids)\n\n"
        "이게 5/13 lab meeting 끝나고 즉시 시작할 plan입니다. 발표는 여기서 마치고, "
        "질문 받겠습니다 — 특히 mechanism이나 framing에 대한 feedback 환영합니다."
    ))


# ─────────────────────────────────────────
# Backup slides A1–A7 (간략)
# ─────────────────────────────────────────
def build_backup(letter, title, key_points, notes=''):
    s = add_blank(WHITE)
    add_section_tag(s, f"Backup · {letter} (Q&A 시 참조)", color=GRAY)
    add_title(s, title, size=22)
    add_bullets(s, 0.6, 1.75, 12.0, 5.3, key_points, font_size=13,
                line_space_after=4)
    add_text(s, 12.5, 7.05, 0.7, 0.35,
             letter, font_size=10, color=GRAY,
             alignment=PP_ALIGN.RIGHT)
    add_notes(s, notes)


def build_backups():
    build_backup("A1",
                 "Desmars 2018 (OMAE) — LIDAR + ICWM concept",
                 [
                     "**Setup**: scanning LIDAR + Improved Choppy Wave Model (ICWM)",
                     "  ICWM = LWT의 nonlinear 1차 보정",
                     "**Method**: predefined k-grid + variational inversion",
                     "**Limitation**: 1D, on-grid, simulator only (실측 없음)",
                     "**Relation**: Desmars 2023 JFM의 precursor — concept 제안",
                 ],
                 "Desmars 2018은 Desmars 2023 JFM의 precursor입니다. ICWM라는 LWT의 1차 비선형 보정을 사용했고, "
                 "scanning LIDAR setup을 처음 제안. 격자 기반 + 1D + simulator 검증만이라 한계 있음.")
    build_backup("A2",
                 "Desmars 2020 (Ocean Eng 212) — Wave gauge experiment",
                 [
                     "**첫 wave tank 실험 검증** of phase-resolved DA (이전 simulator 검증과 차별)",
                     "**Setup**: wave gauge array (선박 mounted 아님)",
                     "**Method**: dispersion correction post-processing",
                     "**Limitation**: 1D longitudinal, gauge ≠ LiDAR (다른 sensor)",
                     "**Relation**: 우리 family의 empirical anchor",
                 ],
                 "Desmars 2020은 Desmars 2018/2023과 같은 group의 wave tank 실험 paper. "
                 "Wave gauge array (선박 mounted 아님)로 dispersion correction 후처리 사용. "
                 "Sensor 종류는 다르지만 family 전체의 empirical anchor 역할.")
    build_backup("A3",
                 "Klein 2020 (Fluids 5(1), 9) — Wave model comparison",
                 [
                     "**Quantitative comparison**: HOSM > NLSE₂ > NLSE > LWT (정확도 순)",
                     "**Wave tank validation**: 다양한 sea state에서 4 model 비교",
                     "**SSP metric standardization**: spatial similarity parameter",
                     "**Implication for us**: linear (LWT) operational range ε < 0.07",
                     "**Relation**: 우리 Stokes ε² correction motivation의 출처",
                 ],
                 "Klein 2020 Fluids는 wave model 4개 (LWT, NLSE, NLSE2, HOSM)를 정량 비교한 paper. "
                 "SSP metric을 표준화. 우리는 LWT family로 가장 단순한 model — Klein 결과로 ε < 0.07 operational range를 정당화하고, "
                 "Phase 2의 Stokes ε² correction motivation 제공.")
    build_backup("A4",
                 "Hunt-Kostelich-Szunyogh 2007 (Tellus A) — LETKF",
                 [
                     "**Foundational paper**: Local Ensemble Transform Kalman Filter",
                     "**Mechanism**: ensemble을 local subspace에 transform 후 update",
                     "**Wave estimation foundation**: Wang-Pan EnKF-HOS, Yan VOF-EnKF 모두 derive",
                     "**Our use**: ETKF baseline implementation 기반",
                 ],
                 "Hunt 2007은 LETKF (Local ETKF)의 foundational paper. 기상학에서 출발했지만 "
                 "wave reconstruction의 EnKF 계열 (Wang-Pan, Yan 2025) 모두가 이 paper에서 derive. "
                 "우리 ETKF baseline도 이 paper 기반.")
    build_backup("A5",
                 "Tang-Bhaskar-Recht 2013 (IEEE TIT) — Atomic norm CS",
                 [
                     "**Off-grid compressed sensing**: 연속 parameter space에서 sparse recovery",
                     "**SDP formulation**: semidefinite programming으로 atomic norm 최소화",
                     "**Theoretical bound**: M = O(K log K) measurements suffice",
                     "**Inspiration for us**: parametric off-grid spectral inversion",
                     "**Future direction**: Phase 2의 sample complexity bound 기반",
                 ],
                 "Tang 2013은 off-grid compressed sensing의 foundational paper. "
                 "기존 CS는 격자 위 sparse signal을 추정하는데 이 paper는 연속 parameter (e.g. 주파수)에서 sparse recovery. "
                 "우리 parametric off-grid 방식의 spirit이 이 paper와 같음. "
                 "Phase 2의 sample complexity bound 도출 시 기반으로 사용.")
    build_backup("A6",
                 "Möller-Trumbore 1997 (JGT) — Ray-triangle intersection",
                 [
                     "**Standard ray-triangle algorithm**: 그래픽 분야 표준",
                     "**Properties**: robust, fast, no preprocessing required",
                     "**Our use**: WaveSim LiDAR ray-mesh intersection 구현",
                     "**Verification**: 39 unit test 중 일부가 이 알고리즘 정확성 검증",
                 ],
                 "Möller-Trumbore 1997은 ray-triangle intersection의 standard algorithm. "
                 "그래픽/시뮬레이션 분야에서 거의 모든 ray tracing이 사용. "
                 "우리 WaveSim의 LiDAR ray-mesh intersection도 이 알고리즘으로 구현, 39 unit test로 정확성 검증.")
    build_backup("A7",
                 "Nouguier-Grilli-Guérin 2014 (IEEE TGRS) — Flash LIDAR foundational",
                 [
                     "**Historical first**: flash LIDAR for ocean wave reconstruction",
                     "**Status**: paywall로 본문 정독 못 함, abstract만 검토",
                     "**Implication**: LIDAR for ocean waves는 Desmars (2018)보다 5년 빠름",
                     "**Our use**: bibliographic anchor — 'flash LIDAR 첫 적용' 인용",
                 ],
                 "Nouguier 2014는 flash LIDAR를 ocean wave 재구성에 처음 적용한 historical paper. "
                 "Paywall로 본문 정독 못 했지만 abstract로 핵심 확인. Desmars (2018)보다 5년 앞선 work. "
                 "Bibliographic reference로 우리 paper에 'LIDAR for ocean wave estimation 첫 사례'로 인용.")


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
    print(f"saved: {OUT_PATH}")
    print(f"slides: {len(prs.slides)} (19 main + 7 backup)")


if __name__ == '__main__':
    main()
