"""
generate_labmeeting_2026_05_13.py

Lab meeting presentation 2026-05-13 — WaveSplat exploration phase + new framing.
외부 청자 (비전문가) 대상 20분 발표 (1 slide ≈ 1분).

Style: 친절한 설명 — jargon 옆에 부연, motivation 강조, presenter notes 상세.

Structure (v3 — 2-way comparison, 18 main + backup):
  Part 1 (S1-S4): Motivation + Problem + Spoiler (hook)
  Part 1b (S5-S8): 5-category landscape + 2 key prior arts + niche matrix
  Part 2 (S9-S10): WaveSim simulator + dataset
  Part 3 (S11-S13): WaveSplat algorithm + 2-way results + episode scatter
  Part 4 (S14-S16): Ablation + Main framing (Korean) + Mechanism
  Closing (S17-S18): Limitations + future research plan
  Backup (A1-A11): Supporting prior art + moved detail slides

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

TOTAL_MAIN = 18


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


# S4 — Spoiler / Hook (결과 먼저 보여주기)
def build_s4_spoiler():
    s = add_blank(LIGHT_BG)
    add_section_tag(s, "Spoiler — 먼저 결과부터 보여드리겠습니다", color=ACCENT2)
    add_title(s, "핵심 결과 미리보기", color=DARK)
    add_subtitle(s, "배경 설명 전에, 왜 이 연구를 계속할 가치가 있는지 숫자로 먼저 보여드리겠습니다")
    # 2-way bar chart
    fig_bar = os.path.join(DOC_DIR, 'fig_3way_bar.png')
    add_image_or_placeholder(s, fig_bar, 0.8, 1.8, 6.5, height=4.0,
                             placeholder_text='[2-way bar chart]')
    # Speed comparison
    fig_speed = os.path.join(DOC_DIR, 'fig_speed_comparison.png')
    add_image_or_placeholder(s, fig_speed, 7.5, 1.8, 5.5, height=3.2,
                             placeholder_text='[speed comparison]')
    add_bullets(s, 0.8, 6.0, 12.0, 1.2, [
        "**WaveSplat**: 더 정확하고 (nRMSE 0.211 vs 0.253), 훨씬 빠름 (13 ms vs 632 ms)",
        "이제부터 이 결과가 어떻게 나왔는지, 왜 의미 있는지를 하나씩 설명드리겠습니다",
    ], font_size=13, line_space_after=3)
    add_page_number(s, 4)
    add_notes(s, (
        "먼저 결과를 보여드리겠습니다. 배경 설명은 그 다음에 하겠습니다.\n\n"
        "왼쪽 막대 그래프는 두 가지 방법의 평균 정확도(nRMSE) 비교입니다.\n"
        "  nRMSE 는 '정규화된 평균 제곱근 오차' 입니다 (관측 wave 표준편차로 normalize).\n"
        "  값이 0 에 가까울수록 추정 정확도 높음, 1 에 가까울수록 낮음.\n"
        "  nRMSE 0.2 는 '실제 wave 와 추정의 차이가 wave 표준편차의 약 20%' 라는 의미입니다.\n\n"
        "  - WaveSplat (우리 방법): 0.211\n"
        "  - Wang2025 (최신 선행연구 재현): 0.253\n\n"
        "Wang2025 는 2025년 발표된 논문 (Wang, Coastal Engineering 197) 의 방법을 그대로 재현한 것입니다. "
        "앙상블 300 개, fuzzy R 적응 등 원논문 세팅을 따랐습니다.\n\n"
        "오른쪽은 여러 방법의 추정 시간 (per frame) 을 로그 스케일로 비교한 그래프입니다.\n"
        "  - WaveSplat 13.4 ms — 초당 약 75 번 추정 가능, 10 Hz 제어 루프에 적합\n"
        "  - Wang2025 632 ms — 추정 한 번에 0.6 초 이상 소요\n"
        "  - FNO 1.9 ms — 빠르나 supervised pre-training 필요 (Ehlers 2023)\n"
        "  - PFT-PINN 30 분 — instance 마다 NN 학습 (RTX 3090 기준, Ehlers 2025)\n\n"
        "핵심 메시지: WaveSplat 이 더 정확하면서 47 배 빠른 방법입니다.\n"
        "이제부터 왜 이런 결과가 가능한지, 선행연구와 어떻게 다른지를 차례로 설명하겠습니다."
    ))


# S5 — 5-category landscape (was S4)
def build_s5_categories():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 1 · Prior art landscape — 선행 연구 분류")
    add_title(s, "기존 연구를 5 가지 접근법으로 분류했습니다")
    add_subtitle(s, "13편 정독 결과 — 각 접근법의 강점과 한계를 풀어 설명합니다")
    headers = ["#", "접근법", "대표 논문", "강점", "한계"]
    rows = [
        ["1", "격자 기반\n최적화",
         "Desmars\n2018/20/23",
         "관측치와 model 사이 차이를 비용함수로 정의해 최소화. 수치해석에서 안정성·수렴성이 정립되어 검증된 framework",
         "추정 가능한 wavenumber 가 미리 정한 격자 위에만 한정 — 격자 간격보다 짧은 wave 표현 불가. 거의 모두 1D 가정"],
        ["2", "앙상블\n칼만 필터",
         "Wang-Pan 2021/22\nWang 2025\nYan 2025 (VOF)",
         "비선형 wave model (HOS, Navier-Stokes) 을 forecast 모델로 사용 가능 — Stokes 비선형, breaking 까지 reconstruction",
         "50–300 개 simulation 을 매 step 동시 진행해야 함 → GPU 사용해도 수백 ms 이상 → real-time control loop 에 부적합. 대부분 1D"],
        ["3", "지도학습\n신경망",
         "Ehlers 2023 FNO\nDuan 2020",
         "Pre-train 후 inference 매우 빠름 (1.9 ms) — sparse measurement → wave field 의 mapping 을 NN 이 학습",
         "수천 개 학습 데이터 합성 필요 (Ehlers 의 경우 3120 HOS sample). 학습 분포 밖 (다른 sea state, 다른 sensor) 에서는 정확도 저하. 1D"],
        ["4", "물리 기반\n비지도 NN",
         "Ehlers 2025\nPFT-PINN",
         "학습 데이터 0 개 — Laplace 방정식과 표면 boundary condition 을 NN loss 로 강제. Linear/nonlinear 모두 처리 (ε ~ 0.1 까지)",
         "Instance 마다 처음부터 30 분 학습 (RTX 3090) — real-time 불가. Buoy time-series 만 처리, sparse spatial LiDAR 미지원. 1D"],
        ["5", "연속 파라미터\n추정 (Ours) ⭐",
         "WaveSplat",
         "Wave 를 N 개 plane wave 의 파라미터 (A, k_x, k_y, ω, φ) 로 직접 추정 — 격자 없이 연속 wavenumber 추정. <100 ms inference + 2D directional + LiDAR raydrop 모사",
         "Linear superposition 가정 (LWT family) — 비선형 wave (HOSM 류) 처리 못 함. ε < 0.07 operational range 에서만 신뢰"],
    ]
    add_table_simple(s, 0.3, 1.85, 12.7, 5.0, headers, rows,
                     header_color=ACCENT, font_size=9)
    add_text(s, 0.5, 7.0, 12.3, 0.4,
             "핵심: 실시간 + 2D 방향성 + 희소 LiDAR + 비지도학습을 모두 만족하는 선행연구는 없습니다. WaveSplat 이 이 조합의 첫 사례.",
             font_size=11, italic=True, color=ACCENT)
    add_page_number(s, 5)
    add_notes(s, (
        "13 편의 논문을 정독한 결과를 5 가지 접근법으로 분류했습니다.\n"
        "각 접근법의 작동 원리와 강점/한계를 자세히 설명드립니다.\n\n"
        "(1) 격자 기반 최적화 (Desmars 2018, 2020, 2023):\n"
        "  - 작동 원리: 공간을 일정 간격 격자로 나눔 (예: x = 0, 5, 10, ... m). "
        "각 격자점에서 wave 의 amplitude 추정. 추정한 wave 가 만들 측정값과 실제 측정값의 차이를 cost function 으로 정의해 최적화로 최소화.\n"
        "  - 강점: 수치해석 분야에서 framework 자체가 잘 정립 — 안정성·수렴성·convergence rate 모두 알려져 있음. "
        "Desmars 2020 은 wave tank 실험으로 정확도 검증 완료.\n"
        "  - 한계: 추정 가능한 wavenumber 가 격자 간격에 의해 제약. "
        "Nyquist-Shannon 정리에 의해 k_max = π/Δx (Δx 는 격자 간격). "
        "격자 사이 wavenumber (예: k = 0.15 rad/m 인데 격자가 0.1, 0.2 만 가지면) 는 표현 불가. "
        "또 거의 모든 prior art 가 1D 가정 (선박 진행 방향만) — 실제 multi-directional sea state 처리 안 함.\n\n"
        "(2) 앙상블 칼만 필터 (Wang-Pan 2021/22, Wang 2025, Yan 2025):\n"
        "  - 작동 원리: 50–300 개의 wave simulation 을 동시에 진행 (각각 약간 다른 초기 조건). "
        "측정값이 들어올 때마다 50 개 simulation 의 평균과 분산을 베이지안 update rule 로 갱신. "
        "Wang-Pan 은 forecast 모델로 HOS (High-Order Spectral, 비선형 wave model) 사용. "
        "Yan 2025 는 VOF + 전체 Navier-Stokes 까지 사용 → wave breaking 처리 가능.\n"
        "  - 강점: forecast 모델 자체가 비선형이라 Stokes 비선형, plunging breaker 같은 비선형 현상까지 reconstruction. "
        "물리 충실도가 가장 높음.\n"
        "  - 한계: 50–300 개 simulation 을 매 시점 동시 진행 — GPU 사용해도 수백 ms 이상. "
        "Real-time control loop (10 Hz, 100 ms 예산) 에 부적합. 또 격자 기반이라 1D 위주.\n\n"
        "(3) 지도학습 신경망 (Ehlers 2023 FNO):\n"
        "  - 작동 원리: 수천 개 (Ehlers 2023 의 경우 3120 개) HOS 합성 데이터로 NN 을 미리 학습. "
        "학습 후에는 sparse measurement 를 입력하면 wave field 를 출력. "
        "FNO (Fourier Neural Operator) architecture 가 Fourier 공간에서 global mapping 학습.\n"
        "  - 강점: 학습 후 inference 매우 빠름 (1.9 ms / instance). 학습 데이터 분포 안에서는 정확.\n"
        "  - 한계: 학습 데이터 합성 자체가 큰 작업 (3120 HOS simulation). "
        "학습 분포 밖 sea state 또는 다른 sensor 조건에서는 정확도 저하 (distribution shift). "
        "Black-box — 왜 그렇게 추정했는지 해석 어려움. 1D 만.\n\n"
        "(4) 물리 기반 비지도 NN (Ehlers 2025 PFT-PINN):\n"
        "  - 작동 원리: PINN (physics-informed neural network) 으로 학습 데이터 없이 학습. "
        "Loss 함수가 PDE residual (Laplace 방정식 + 표면 boundary condition + 바닥 condition) "
        "이라 NN 이 wave physics 자체를 학습. Auto-differentiation 으로 PDE residual 계산.\n"
        "  - 강점: 학습 데이터 0 개로 비선형 wave 처리 (linear LWT 부터 NLSE2 같은 weak nonlinear 까지). "
        "Wave flume 실험으로 검증 (SSP < 0.05).\n"
        "  - 한계: Instance 마다 처음부터 30 분 학습 (RTX 3090, 40K+ epochs) — real-time 불가. "
        "Buoy time-series 입력만 처리 — sparse spatial LiDAR setup 미지원. 1D + t.\n\n"
        "(5) 연속 파라미터 추정 — WaveSplat (우리):\n"
        "  - 작동 원리: wave 를 N 개 plane wave 의 파라미터 (A_n, k_xn, k_yn, ω_n, φ_n) 로 직접 표현 — 격자 없이 연속 wavenumber. "
        "Dispersion 관계 ω = √(g·|k|·tanh(|k|·d)) 를 hard constraint 로 강제 → 자유도 5N → 4N. "
        "Adam optimizer 로 measurement 와 model prediction 의 차이 최소화.\n"
        "  - 강점: <100 ms inference (CPU) — real-time control loop 적합. "
        "2D directional 처리. LiDAR raydrop, shadowing, Fresnel 까지 simulator 에 모사.\n"
        "  - 한계: Linear superposition 가정 (LWT family). Klein 2020 의 정량 비교에서 HOSM > NLSE2 > LWT — "
        "비선형 wave 처리 못 함. ε < 0.07 operational range 에서만 신뢰.\n\n"
        "다음 슬라이드부터 가장 중요한 선행 연구 2 편을 자세히 소개하겠습니다."
    ))


# Helper: standardized prior-art slide
def build_prior_art(num, title_full, key_msg, limits, contrast,
                    fig_path=None, notes='', star=False):
    s = add_blank(WHITE)
    add_section_tag(s, f"Part 1 · Prior art detail")
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


def build_s5_desmars():
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
        title_full="Wang & Pan 2021/22 (JFM) + Wang 2025 (Coastal Eng) — EnKF family",
        key_msg=[
            "**EnKF (Ensemble Kalman Filter)**: 50–300 개 wave simulation 을 동시에 진행 후 measurement 로 update",
            "  Forecast 단계: 각 ensemble member 가 wave model 로 시간 진행",
            "  Analysis 단계: ensemble 의 covariance 와 measurement 로 베이지안 update",
            "**HOS (High-Order Spectral method)**: 비선형 wave model — Linear Wave Theory (LWT) 의 확장",
            "  표면 boundary condition 의 비선형 항 (wave-wave interaction) 까지 spectral 표현으로 계산",
            "  Wave steepness ε ~ 0.1 까지 처리 가능 (LWT 는 ε < 0.07)",
            "  Dommermuth & Yue 1987, West et al. 1987 에서 출발한 표준 비선형 wave model",
            "**Wang 2025 (Coastal Eng 197)**: Ne=300, fuzzy R 적응, Delaunay 보간",
        ],
        limits=[
            "**격자 해상도 한계** — nRMSE 가 ~0.25 아래로 내려가지 않음 (격자 밖 wavenumber 표현 불가)",
            "**앙상블 비용** — Ne=300 simulation → **632 ms / frame** (real-time control loop 불가)",
        ],
        contrast=[
            "**12-episode 비교**: WaveSplat **0.211** / 13 ms  vs  Wang2025 0.253 / 632 ms",
            "WaveSplat: **정확도 16% 우위 + 47 배 빠름** (CPU 만으로도 가능)",
        ],
        notes=(
            "이 논문 family 가 우리의 가장 직접적 비교 대상입니다. "
            "주요 두 용어를 먼저 정의하겠습니다.\n\n"
            "EnKF (Ensemble Kalman Filter, 앙상블 칼만 필터):\n"
            "  - 50–300 개의 wave simulation 을 동시에 진행 (각각 약간 다른 초기 조건).\n"
            "  - Forecast 단계: 각 ensemble member 가 wave model 로 시간 진행 (예: 1초 후 wave field 예측).\n"
            "  - Analysis 단계: measurement 가 들어올 때마다 ensemble 의 평균과 covariance 로부터 Kalman gain 계산.\n"
            "  - 베이지안 update rule: x_a = x_f + K (y - H x_f). 여기서 K 가 Kalman gain, y 가 measurement, H 가 observation operator.\n"
            "  - 원래 기상학 데이터 동화에서 출발 (Evensen 1994), Hunt 2007 의 LETKF 가 표준 변형.\n"
            "  - Wave estimation 에는 Wang-Pan 2021 이 처음 적용.\n\n"
            "HOS (High-Order Spectral method, 고차 스펙트럴 방법):\n"
            "  - 비선형 wave model. Linear Wave Theory (LWT) 의 확장.\n"
            "  - LWT 는 wave field 를 plane wave 의 단순 선형 중첩으로 표현 (ε < 0.07 에서만 정확).\n"
            "  - HOS 는 표면 boundary condition 의 비선형 항 (wave-wave interaction, 즉 두 wave 가 만나서 생기는 추가 wave) 까지 spectral 표현으로 계산.\n"
            "  - Wave steepness ε ~ 0.1 까지 처리 가능 — Stokes wave, swell-wind interaction 같은 비선형 현상까지.\n"
            "  - Dommermuth & Yue 1987, West et al. 1987 에서 독립적으로 제안. 이후 ocean engineering 에서 표준 비선형 wave model 로 자리잡음.\n"
            "  - Klein 2020 의 정량 비교에서 HOSM > NLSE2 > NLSE > LWT 순 정확도.\n\n"
            "이 family 의 발전 과정:\n"
            "  - Wang & Pan 2021 (JFM 918): EnKF + HOS 를 처음 결합한 paper.\n"
            "  - Wang & Pan 2022 (JFM 949): 해류 (current) 까지 동시 추정 (EnKF-HOS-C).\n"
            "  - Wang 2025 (Coastal Eng 197): 앙상블 300 개, fuzzy R (measurement noise 자동 적응), "
            "Delaunay 보간 (불규칙 측정점 처리) 추가.\n\n"
            "우리는 Wang 2025 의 방법을 코드로 충실히 재현했습니다 (Ne=300, 80×60 격자, dx=3m, fuzzy R, Delaunay). "
            "원논문과의 차이는 forecast 모델로 HOS 대신 linear dispersion 사용 — 우리 WaveSplat 도 linear 이므로 공정 비교 위해 같은 model.\n\n"
            "비교 결과:\n"
            "  - WaveSplat: nRMSE 0.211, 13 ms / frame\n"
            "  - Wang2025: nRMSE 0.253, 632 ms / frame\n\n"
            "에피소드별 패턴:\n"
            "  - 단순 파도 (N_components 1–5): WaveSplat 우위 (예: BF3_2comp 에서 WaveSplat 0.052 vs Wang 0.250)\n"
            "  - 복잡 파도 (N_components 15–30): Wang2025 가 일부 episode 에서 우위 — trade-off 정직하게 보고 예정"
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
        num=7,
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
    headers = ["응용 요구사항 (쉬운 설명)", "Ours", "PFT-\nPINN", "VOF-\nEnKF",
               "Wang\n2025", "FNO", "Des-\nmars"]
    rows = [
        ["실시간 제어 가능 (<100ms)", "✓", "✗", "✗", "✗", "✓", "✗"],
        ["2D 방향성 추정", "✓", "✗", "✗", "✗", "✗", "✗"],
        ["파도 파쇄(breaking) 처리", "✗", "✗", "✓", "✗", "✗", "✗"],
        ["유속장 전체 추정", "✗", "✓", "✓", "✓", "✗", "일부"],
        ["학습 데이터 불필요", "✓", "✓", "✓", "✓", "✗", "✓"],
        ["희소 LiDAR 센서 모델", "✓", "✗", "✗", "✗", "✗", "일부"],
    ]
    add_table_simple(s, 0.4, 1.85, 12.5, 3.8, headers, rows,
                     header_color=ACCENT, font_size=10, highlight_col=1)
    add_text(s, 0.5, 5.8, 12.3, 0.35, "우리만의 고유한 위치:",
             font_size=14, bold=True, color=ACCENT)
    add_bullets(s, 0.6, 6.15, 12.0, 1.2, [
        "**실시간 + 2D 방향성 + 희소 LiDAR + 학습 데이터 불필요** — 이 4가지를 모두 만족하는 선행연구는 없습니다",
        "주요 응용: 선박/hydrofoil 실시간 제어, 자율운항 안전 판단 등",
    ], font_size=12)
    add_page_number(s, 8)
    add_notes(s, (
        "이 표는 '각 방법이 어떤 응용에 적합한가'를 한눈에 보여줍니다.\n\n"
        "표를 읽는 법:\n"
        "  - 각 행은 '이런 기능이 필요한가?'를 나타냅니다\n"
        "  - ✓ = 가능, ✗ = 불가능\n"
        "  - 노란색으로 강조된 열(Ours)이 우리 방법입니다\n\n"
        "핵심 관찰:\n"
        "  어떤 방법도 모든 ✓를 받지는 못합니다.\n"
        "  각 방법은 고유한 강점이 있고, 적합한 응용 영역이 다릅니다.\n\n"
        "우리의 고유한 위치:\n"
        "  '실시간 + 2D 방향성 + 희소 LiDAR + 학습 데이터 불필요'\n"
        "  이 4가지를 동시에 만족하는 방법은 선행연구에 없었습니다.\n\n"
        "응용 영역을 크게 둘로 나눌 수 있습니다:\n"
        "  - '실시간 제어' 영역: 우리 WaveSplat, FNO\n"
        "    즉시 결정을 내려야 하는 상황 (선박 제어, hydrofoil 등)\n"
        "  - '정밀 분석' 영역: VOF-EnKF, PFT-PINN\n"
        "    시간은 충분하지만 가장 정확한 분석이 필요한 상황\n\n"
        "우리는 PFT-PINN이나 VOF-EnKF와 경쟁하는 것이 아닙니다.\n"
        "서로 다른 영역을 담당하고 있습니다.\n"
        "직접 비교 대상은 FNO(Ehlers 2023)인데,\n"
        "FNO는 1D + 지도학습, 우리는 2D + 비지도학습으로 차별화됩니다.\n\n"
        "정직한 한계도 표에 보입니다:\n"
        "  - 파도 파쇄(breaking) 처리 못 함 — VOF-EnKF만 가능\n"
        "  - 유속장 추정 못 함 — PINN/EnKF만 가능\n"
        "이런 한계는 논문에 정직하게 명시할 예정입니다.\n\n"
        "다음부터는 우리가 만든 시뮬레이터와 추정기를 소개합니다."
    ))


# S11 — WaveSim physics + LiDAR 모사 (어떻게)
def build_s11():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 2 · WaveSim (1/2): 파도 + LiDAR 시뮬레이션", color=ACCENT2)
    add_title(s, "WaveSim — 파도와 LiDAR 를 어떻게 모사했나")
    add_subtitle(s, "각 구성요소의 물리적 합당성 + 코드 검증")
    bullets = [
        "**파도 모델 — Gerstner 파 + JONSWAP 스펙트럼**:",
        "  η(x, y, t) = Σ_n A_n cos(k_xn·x + k_yn·y − ω_n·t + φ_n) + Gerstner 추가 변위",
        "  Gerstner 1802 의 회전 정확해 — 입자가 원/타원 궤적으로 회전 (Q=0 → 선형, Q>0 → 약비선형)",
        "  → 합당성: Tessendorf 2001 SIGGRAPH ocean 표준, Dean & Dalrymple 1991 교과서",
        "  → 검증: Q 제약 (Σ Q·ω·A ≤ 1, 입자 궤적 자체 교차 방지) 의 unit test 통과",
        "",
        "**스펙트럼 — JONSWAP (Hasselmann et al. 1973)**:",
        "  S(ω) = α·g²·ω⁻⁵·exp(−5/4·(ω_p/ω)⁴) · γ^r — 북해 측정 데이터 기반",
        "  입력: H_s (유의파고), T_p (첨두주기), γ (peak factor)",
        "  → 합당성: 해양공학 표준 (ITTC, IACS 채택). 파도 성분 amplitude·주파수 sampling 에 사용",
        "",
        "**분산 관계**: ω_n = √(g·|k_n|·tanh(|k_n|·d))",
        "  Airy 1845 의 linear wave theory 결과 — 파도의 주파수와 파수의 표준 관계",
        "  → 합당성: 100 년 이상 검증된 표준 (수심 d, 중력 g 만 입력)",
        "",
        "**LiDAR 센서 (Falcon K1 spec)**:",
        "  Vessel 위 5 m mount, FOV 수평 120° × 수직 ±5° (선수 방향 반구만, 전방향 360° 아님)",
        "  32 채널 × 360 beam = 11,520 ray / scan, range 10–200 m, 1550 nm 파장",
        "  Ray tracing: laser ray 가 파도 표면에 부딪힌 첫 점을 찾음 (Möller-Trumbore 1997)",
        "",
        "**Vessel motion 처리** (사용자 질문 답: '관측 창만 이동' 맞음):",
        "  파도 격자는 충분히 큰 영역 (수백 m × 수백 m) 의 world frame 에 고정",
        "  Vessel 은 그 격자 안을 V=11.3 m/s 로 이동 (매 frame Δt=0.1 s 동안 1.13 m 전진)",
        "  → LiDAR FOV (관측 창) 가 vessel 따라 이동, 파도 격자 자체는 변경 없음",
        "  → 파도 자체는 t 의 진행으로 자연 propagate (η(x,y,t) 의 t 만 update)",
    ]
    add_bullets(s, 0.4, 1.85, 8.7, 5.3, bullets, font_size=9,
                line_space_after=1)
    fig_lidar = os.path.join(DOC_DIR, 'wavesim_BF4_5c_wave_lidar.png')
    add_image_or_placeholder(s, fig_lidar, 9.3, 2.2, 3.8,
                             placeholder_text='[LiDAR scan visualization]')
    add_page_number(s, 9)
    add_notes(s, (
        "WaveSim 시뮬레이터의 4 가지 구성 — 파도 모델, 스펙트럼, 분산 관계, LiDAR 센서 — 각각의 물리적 합당성과 검증을 설명드립니다.\n\n"
        "(1) 파도 모델 — Gerstner 파:\n"
        "  - 단순 sinusoidal (LWT) 는 수면 변위만 표현 (η = A cos(...)).\n"
        "  - Gerstner 파는 입자의 운동까지 모사 — 입자가 원/타원 궤적으로 회전하면서 표면 형성.\n"
        "  - Q ∈ [0, 1]: 가파름 척도. Q=0 이면 LWT 와 동일, Q>0 이면 약비선형 (crest sharp, trough flat — Stokes-1 풍).\n"
        "  - Q 제약: Σ Q_i · ω_i · A_i ≤ 1. 이 제약을 어기면 입자 궤적이 자체 교차해서 비물리적 — Tessendorf 2001 의 particle crossing prevention.\n"
        "  - 합당성: Tessendorf 2001 SIGGRAPH 의 ocean simulation 표준, Dean & Dalrymple 1991 교과서.\n"
        "  - 검증: wavesim/tests/ 의 Q-constraint test 가 ΣQ·ω·A 가 1 이하 유지 확인.\n\n"
        "(2) 스펙트럼 — JONSWAP (Joint North Sea Wave Project):\n"
        "  - Hasselmann et al. 1973 이 북해 측정 데이터로 정립한 표준 wave spectrum.\n"
        "  - S(ω) = α · g² · ω⁻⁵ · exp(−5/4 · (ω_p/ω)⁴) · γ^r — Pierson-Moskowitz 에 peak enhancement γ 추가.\n"
        "  - 입력: H_s (유의파고), T_p (첨두주기), γ (peak factor, default 3.3).\n"
        "  - N 개 wave 성분의 (A_n, ω_n) 를 spectrum 으로부터 sampling: amplitude 는 √(2·S(ω)·Δω).\n"
        "  - 방향 spreading: cos²ˢ 함수로 방향 성분 분배.\n"
        "  - 합당성: 해양공학 분야 표준 wave spectrum. ITTC (International Towing Tank Conference) 와 IACS (International Association of Classification Societies) 모두 채택.\n\n"
        "(3) 분산 관계 — Airy 1845:\n"
        "  - ω_n = √(g·|k_n|·tanh(|k_n|·d)) — frequency 와 wavenumber 의 표준 관계.\n"
        "  - linear wave theory 의 기본 결과 (Airy 1845 가 처음 정립).\n"
        "  - 입력: 중력 g = 9.81 m/s², 수심 d. 깊은 물 한계 (kd >> 1) 에서 ω = √(g·k).\n"
        "  - 합당성: 100 년 이상 사용된 표준. 모든 wave estimation paper 에서 그대로 사용.\n\n"
        "(4) LiDAR 센서 모사 — Falcon K1 spec 기준:\n"
        "  - 우리 simulator 의 default config 가 Falcon K1 LiDAR spec 과 일치.\n"
        "  - Horizontal FOV 120°: 전방향 360° 아니라 선수 방향 ±60° 반구 (marine LiDAR 의 일반 setup).\n"
        "  - Vertical FOV ±5°: 수평 기준 ±5° 좁음 (먼 거리 surface 측정용).\n"
        "  - 32 채널 × 360 horizontal beam = 11,520 ray / scan, range 10–200 m, 1550 nm 파장.\n"
        "  - Ray tracing: 각 ray 가 파도 표면 (격자에서 만든 triangle mesh) 과 만나는 첫 점 (first-hit) 을 찾음.\n"
        "  - Möller-Trumbore 1997: graphics 분야 표준 ray-triangle intersection 알고리즘.\n\n"
        "(5) Vessel motion 처리 — 사용자 질문의 답 ('관측 창만 이동') 맞음:\n"
        "  - 파도 field 는 충분히 큰 world frame 격자 (예: 500 m × 500 m) 에 고정. 시뮬레이션 시작 시 한 번 만들고 변경 없음.\n"
        "  - Vessel (LiDAR mount 점) 만 그 격자 안을 V=11.3 m/s 로 이동.\n"
        "  - 매 frame Δt=0.1 s 동안 vessel 위치가 1.13 m 전진. LiDAR FOV (관측 창) 도 함께 이동 — 매 frame 다른 영역 관측.\n"
        "  - 파도 자체의 시간 변화는 η(x, y, t) 의 t 가 진행하는 것으로 자연 처리. 격자 자체는 그대로.\n"
        "  - 결과: 매 frame 다른 vessel 위치에서 wave 의 다른 부분을 관측 — 이게 paper 의 main framing 인 synthetic aperture 효과의 원천.\n\n"
        "다음 슬라이드에서 raydrop (점이 왜 누락되는가) 과 검증을 다룹니다 — Fresnel 식 자체의 단계별 계산은 발표자에게는 별도 설명, 슬라이드에는 합당성·검증 위주."
    ))


# S12 — WaveSim raydrop 합당성 + 검증
def build_s12():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 2 · WaveSim (2/2): raydrop 합당성 + 검증", color=ACCENT2)
    add_title(s, "Raydrop / shadowing — 왜 합당하고 어떻게 검증되었나")
    add_subtitle(s, "각 물리 모델의 출처 + 우리 검증 방법")
    bullets = [
        "**Raydrop (점 누락) 의 물리**:",
        "  파도 표면이 매끄러우면 ray 가 specular 반사 — LiDAR receiver 가 아닌 방향으로 가서 return 없음",
        "  입사각 θ_i 가 grazing (90°) 에 가까울수록 specular 비중 ↑ → 점 누락 ↑",
        "  → 모사: Fresnel reflectance R(θ_i) 로 specular 비중 계산 (식은 backup A 참조)",
        "",
        "**Fresnel 식의 합당성**:",
        "  19 세기 Augustin-Jean Fresnel 정립 (1820 년대)",
        "  광학 교과서 표준 (Hecht, Optics §4.6) — 모든 광학 시뮬레이션에서 사용",
        "  Ocean LiDAR 분야 (Nouguier-Grilli-Guérin 2014, IEEE TGRS) 에서 검증된 표준",
        "  Falcon K1 의 1550 nm 파장 기준 물 굴절률 n = 1.318 사용",
        "",
        "**Geometric shadowing — 별도 모델 없음**:",
        "  앞쪽 큰 파도가 뒤쪽을 가림 — ray tracing 의 first-hit 결과로 자동 처리",
        "  광선 직진 + 첫 충돌점 측정 = geometric optics 의 자연 결과 (추가 가정 0)",
        "",
        "**검증 (validation)**:",
        "  ① 108 unit test 통과 — ray tracing 정확성, Fresnel 계산, shadowing edge case 모두 검증",
        "  ② 110 episode 합성 dataset — BF2–BF7 × N=1–30 다양한 sea state 에서 동작 확인",
        "  ③ Sea trial 실측 LiDAR 데이터 (sea_trial_lidar.png) 와 점 분포 패턴 정성 비교 — 일치",
        "",
        "**정직한 한계**:",
        "  미세파 (capillary, mm 단위 surface roughness): 실제 LiDAR raydrop 에 추가 영향, 우리는 미반영",
        "  Foam, white cap (BF6 이상): wave breaking 시 발생, LWT family 처리 못 함",
        "  Operational sea state (BF2–BF6) 에서는 부차적 효과로 가정",
    ]
    add_bullets(s, 0.4, 1.85, 12.5, 5.5, bullets, font_size=10,
                line_space_after=2)
    add_page_number(s, 10)
    add_notes(s, (
        "이전 슬라이드에서 wave model + LiDAR 모사 4 단계를 설명드렸습니다. "
        "이 슬라이드는 raydrop / shadowing 의 합당성과 검증을 다룹니다. "
        "Fresnel 식 자체의 단계별 계산은 backup A (또는 발표자에게 직접 설명) 으로 두고, 슬라이드는 *왜 합당한가 + 어떻게 검증되었나* 위주.\n\n"
        "(1) Raydrop 물리:\n"
        "  - 실제 LiDAR 에서 점 누락의 주 원인은 specular reflection.\n"
        "  - 매끄러운 표면은 ray 를 한 방향으로 반사 → LiDAR receiver 가 그 방향에 없으면 점 누락.\n"
        "  - 입사각 θ_i 가 grazing (90°) 에 가까울수록 specular 가 강해서 raydrop 확률 ↑.\n"
        "  - 모사: Fresnel reflectance R(θ_i) 와 거리 보정 1/d² 으로 종합 계산. 식 detail 은 backup.\n\n"
        "(2) Fresnel 의 합당성 — 발표 시 청자 질문에 대비:\n"
        "  - 19 세기 Augustin-Jean Fresnel 정립 (1820 년대). 광학 교과서 Hecht, Optics §4.6 에 표준 등장.\n"
        "  - 모든 광학 시뮬레이션 (rendering, oceanographic optics, satellite remote sensing) 에서 동일한 식 사용.\n"
        "  - Ocean LiDAR 분야 Nouguier-Grilli-Guérin 2014 (IEEE TGRS) 에서 그대로 적용 — 검증된 표준.\n"
        "  - 우리 simulator 의 1550 nm 파장 기준 물 굴절률 n = 1.318 은 광학 표준 데이터.\n\n"
        "(3) Geometric shadowing — 추가 모델 없음:\n"
        "  - Ray tracing 의 first-hit 결과로 자동 처리.\n"
        "  - 광선 직진 + 첫 충돌점만 측정 = geometric optics 의 자연 결과.\n"
        "  - 추가 가정 0, 자체 일관성 보장.\n\n"
        "(4) 검증 — 3 단계:\n"
        "  ① Unit test 108 개 — ray tracing 정확성, Fresnel 계산, shadowing edge case 각각 별도 test.\n"
        "  ② 110 episode 합성 dataset 으로 다양한 sea state 에서 동작 확인 (BF2–BF7).\n"
        "  ③ Sea trial 실측 LiDAR 데이터 (sea_trial_lidar.png) 와 점 분포 패턴 정성 비교 — 일치 확인.\n\n"
        "(5) 정직한 한계:\n"
        "  - 미세파 (capillary, mm 단위): LiDAR raydrop 에 추가 영향, 우리는 미반영.\n"
        "  - Foam, white cap: BF6 이상 wave breaking 시 발생.\n"
        "  - Breaking wave: VOF-EnKF 영역, 우리는 future work.\n\n"
        "발표 시 청자가 'Fresnel 어떻게 계산?' 물으면: 'Snell 로 굴절각 → 두 polarization 평균 → 거리 보정' 3 단계 라고 답변. "
        "수식 자세히는 backup A 또는 wavesim/raydrop_uniform.py 의 fresnel_reflectance() 함수 참조."
    ))


# S13 — WaveSplat algorithm
def build_s13():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 3 · WaveSplat — 우리 estimator (탐색 단계)", color=GOLD)
    add_title(s, "WaveSplat — 아이디어의 시작과 'Splat' 이라는 이름")
    add_subtitle(s, "왜 격자가 아닌 '파도 성분 한 개씩' 추정하는가? 왜 'Splat' 이름인가?")
    bullets = [
        "**아이디어 출발 — 격자 기반의 한계**:",
        "  Wang2025/Desmars 류: 바다 표면을 격자 위 높이값으로 표현 (예: 80×60 = 4,800 개 점)",
        "  관측은 LiDAR 점 50 개뿐 → 4,800 개 높이값 추정은 정보 부족 (under-determined)",
        "  격자 간격보다 짧은 파장의 파도는 표현 불가 (Nyquist 한계)",
        "",
        "**발상 전환 — 파도 성분 한 개씩 직접 추정**:",
        "  실제 바다 파도는 보통 5–30 개의 sin 파 (성분) 의 합으로 충분히 표현 (JONSWAP 분해)",
        "  각 파도 성분은 5 개 숫자로 완전 결정:",
        "    · 진폭 A (파고 크기)",
        "    · 파수 (k_x, k_y) — 파장과 진행 방향 결정",
        "    · 주파수 ω (= 2π / 주기 T) — 시간적 변화율",
        "    · 위상 φ (시작 시점)",
        "  성분 N=10 이면 추정 변수 50 개 — 관측 50 개로 추정 가능",
        "  분산 관계 ω = √(g·|k|·tanh(|k|·d)) 를 강제 → ω 는 k 에서 자동 → 변수 5N → 4N",
        "",
        "**3D Gaussian Splatting (3DGS) 와의 영감**:",
        "  3DGS (Kerbl 2023, SIGGRAPH): 3D 공간을 voxel/NN 대신 'Gaussian 원소 여러 개의 합' 으로 표현, 각 원소를 직접 최적화",
        "  WaveSplat: 바다 파도를 격자 대신 'sin 파 성분 여러 개의 합' 으로 표현, 각 성분을 직접 최적화",
        "  → 같은 철학 (격자 없이 명시적 원소 직접 최적화), 다른 영역 (3DGS=영상, WaveSplat=파도)",
        "",
        "**'Splat' 명칭의 의미**:",
        "  '여러 명시적 원소를 위치시키고 직접 최적화' 한다는 점에서 3DGS 와 발상 동일",
        "  단 graphics rendering 은 아님 — paper 작성 시 명칭 변경 검토 가능 (현재 working name)",
    ]
    add_bullets(s, 0.4, 1.85, 8.7, 5.4, bullets, font_size=10,
                line_space_after=1)
    fig_pipe = os.path.join(PAPER_FIG_DIR, 'fig_pipeline.png')
    add_image_or_placeholder(s, fig_pipe, 9.3, 2.0, 3.8,
                             placeholder_text='[WaveSplat pipeline diagram]')
    add_page_number(s, 11)
    add_notes(s, (
        "WaveSplat 의 발상 시작과 명칭의 유래를 설명합니다. 외부 청자 (조선해양 분야 포함) 가 익숙한 표현을 쓰고, jargon 은 풀어 설명합니다.\n\n"
        "(1) 격자 기반의 한계 — 왜 부족한가:\n"
        "  - Wang 2025, Desmars 등은 바다 표면 (수면 높이 η) 을 격자 위의 값으로 표현. 80×60 격자면 4,800 개 점.\n"
        "  - 추정해야 할 변수 = 4,800 개. 그러나 LiDAR 관측 점은 frame 당 50 개.\n"
        "  - 50 개 측정으로 4,800 개 변수 추정 = 정보 부족 (under-determined). 격자 외 추가 정보 (부드러움 가정, 앙상블 covariance) 를 동원해야 함.\n"
        "  - 추가 한계: 격자 간격보다 짧은 파장의 파도는 격자로 표현 자체가 불가능 (Nyquist 한계).\n\n"
        "(2) 발상의 전환 — 파도 성분으로 직접 표현:\n"
        "  - JONSWAP 스펙트럼으로 분해하면 실제 바다 파도는 보통 5-30 개의 sin 파 (성분, component) 의 합으로 충분히 근사.\n"
        "  - 각 sin 파 성분은 5 개의 숫자로 완전히 결정 — 조선해양 분야 표준 wave 표현:\n"
        "    · A (진폭, amplitude): 파고의 크기 (단위: m).\n"
        "    · (k_x, k_y) (파수, wavenumber vector): 파장 λ = 2π/|k| 와 진행 방향 결정.\n"
        "    · ω (각주파수, angular frequency): 주기 T = 2π/ω 의 역수, 시간적 변화율.\n"
        "    · φ (위상, phase): 시작 시점 (어느 파도가 어느 시점에 어디 있는지).\n"
        "  - N=10 개 성분이면 변수 50 개. LiDAR 관측 50 개로 1:1 추정 가능.\n"
        "  - 분산 관계 (dispersion relation) ω = √(g·|k|·tanh(|k|·d)) 를 강제 → ω 가 k 에서 자동 결정 → 변수 5N → 4N.\n"
        "  - 결정적 차이: 격자 간격이라는 한계가 없음. 파장이 짧든 길든 (k 가 연속) 자연스럽게 표현.\n\n"
        "(3) 3D Gaussian Splatting (3DGS) 와의 영감:\n"
        "  - 3DGS (Kerbl et al. 2023, SIGGRAPH 발표) 는 컴퓨터 그래픽 분야의 새로운 3D scene 표현 방법.\n"
        "  - 기존 방법은 voxel grid (3D 격자) 또는 implicit neural network 사용.\n"
        "  - 3DGS 는 3D 공간을 *여러 개의 Gaussian 원소* 의 합으로 표현. 각 Gaussian 의 위치/크기/색을 gradient descent 로 직접 최적화.\n"
        "  - 우리도 비슷한 발상 — 격자 없이 *여러 sin 파 성분의 합* 으로 wave field 를 표현. 각 성분의 5 변수를 직접 최적화.\n"
        "  - 도메인은 다름 (3DGS = 3D 영상, WaveSplat = 바다 파도) 이지만 핵심 철학은 같음.\n\n"
        "(4) 'Splat' 이라는 명칭 — 정직한 정리:\n"
        "  - 3DGS 의 'splatting' 은 그래픽 rendering 기법 (각 Gaussian 을 화면에 'splat'=뿌림).\n"
        "  - 우리는 그래픽 rendering 이 아니라 wave 추정 — 'splat' 의 정확한 의미는 우리 알고리즘에 직접 적용되지 않음.\n"
        "  - 이름은 '명시적 원소를 직접 최적화' 라는 발상의 영감 표시. 초기 working name 으로 사용 중.\n"
        "  - Paper 작성 시 더 적절한 이름 (예: SplatWave, ParametricWave, WaveAtom) 으로 변경 검토 가능.\n"
        "  - Q&A 답변 준비: '초기 작업명입니다. 3DGS 의 *여러 명시적 원소의 합으로 연속 필드를 표현* 한다는 발상에서 영감을 받았습니다. 핵심 알고리즘은 spectral parametric inversion (해양·신호처리 분야 표준).'\n\n"
        "(5) 다음 슬라이드 흐름 — 실제 알고리즘:\n"
        "  - 다음 (S12): Init 단계 — 여러 frame 에서 파수 (k_x, k_y) 추정 (beamforming).\n"
        "  - 그 다음 (S13): Tracking 단계 — 매 frame 진폭 (A) 과 위상 (φ) 만 Adam 으로 빠르게 update + 성분 수 자동 조정.\n"
        "  - 그 다음 (S13d): GT 와 WaveSplat 의 비교 시각화."
    ))


# S13b — Beamforming Init detail (NEW)
def build_s13b_init():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 3 · WaveSplat — Init 단계 (1/2)", color=GOLD)
    add_title(s, "Init: 다중 frame beamforming 으로 (k_x, k_y) 결정")
    add_subtitle(s, "왜 multi-frame 인가? 어떻게 작동하는가? 수식까지 단계별로")
    bullets = [
        "**왜 Init 이 어려운가** (정량 설명):",
        "  Single frame: LiDAR 50 점이 커버하는 영역 ~ 240 m (FOV 120°, range 200 m)",
        "  격자 간격 Δy ≈ 5–8 m → spatial Nyquist k_max = π/Δy ≈ 0.4 rad/m",
        "  필요한 wave wavenumber k ∈ 0.5–3 rad/m → single frame 으론 추정 불가",
        "",
        "**Multi-frame stacking (Synthetic Aperture)**:",
        "  vessel V=11.3 m/s 로 1 s 동안 11.3 m 이동 → 10 frames × 50 점 = 500 점, 영역 확장",
        "  Effective aperture = V·T_init = 11.3 m, Δk_aperture ≈ 2π/11.3 ≈ 0.56 rad/m 분해능",
        "",
        "**v2_konly Beamforming — 2 단계**:",
        "  **① Coarse**: (k_x, k_y) 후보 격자 생성 (예: 50×50 = 2,500 개)",
        "    각 후보 (k_x⁰, k_y⁰) 에 대해 score: S(k) = |Σ_i η_obs(x_i, t_i) · exp(j(k·x_i − ω(k)·t_i))|²",
        "    ω(k) 는 dispersion 관계로 자동: ω(k) = √(g·|k|·tanh(|k|·d))",
        "    S(k) 가 큰 peak N 개를 초기 파수로 선택",
        "  **② Fine**: 선택된 peak 주변에서 sub-grid resolution 으로 정밀 탐색",
        "    Local quadratic fit + Newton step → (k_x*, k_y*) 정확도 향상",
        "",
        "**'konly' 의미**: 이 단계에서는 *k 만* 초기화. ω 는 dispersion 으로 자동, (A, φ) 는 다음 단계 (track) 에서 최적화.",
        "  이전 v1 (FFT 기반) 대비 평균 18% 개선 (불규칙 LiDAR 점에 직접 적용 가능)",
    ]
    add_bullets(s, 0.5, 1.85, 12.5, 5.3, bullets, font_size=10,
                line_space_after=2)
    add_page_number(s, 12)
    add_notes(s, (
        "Init 단계를 수식 + 정량으로 자세히 설명합니다.\n\n"
        "(1) Single frame 의 한계 — 정량:\n"
        "  - LiDAR 한 frame: 50 점이 240 m 영역 (FOV 120°, range 200 m, vessel 위 5 m mount).\n"
        "  - 점 간격 Δy ≈ 5–8 m → spatial Nyquist k_max = π/Δy ≈ 0.4 rad/m.\n"
        "  - 우리가 추정해야 할 wave wavenumber: BF3–BF6 의 typical k ∈ 0.5–3 rad/m (파장 2–12 m).\n"
        "  - 즉 single frame 으로는 *Nyquist 한계 위* 영역을 추정해야 하는 paradox.\n"
        "  - 해결: multi-frame stacking + dispersion (이게 paper 의 main framing).\n\n"
        "(2) Synthetic aperture 효과:\n"
        "  - vessel 이 V=11.3 m/s 로 1 s 동안 11.3 m 이동.\n"
        "  - 10 frame stacking 시 vessel 위치 0 → 11.3 m, 점 500 개가 240+11.3 m 영역에 분포.\n"
        "  - Effective aperture L_SA = V·T_init = 11.3 m → wavenumber 분해능 Δk ≈ 2π/L_SA ≈ 0.56 rad/m.\n"
        "  - 이 분해능으로 0.5–3 rad/m 영역의 wave 추정 가능.\n\n"
        "(3) Beamforming score 수식:\n"
        "  - 후보 wavenumber (k_x⁰, k_y⁰) 에 대해:\n"
        "    S(k) = |Σ_i η_obs(x_i, t_i) · exp(j (k·x_i − ω(k)·t_i))|²\n"
        "    여기서 ω(k) = √(g·|k|·tanh(|k|·d)) 는 dispersion 으로 자동 결정.\n"
        "    x_i 는 i-th 측정점의 위치 (x_i, y_i), t_i 는 측정 시각.\n"
        "  - 의미: '만약 wave 가 이 (k_x, k_y) 방향으로 ω(k) 주파수로 진행한다면, 측정 데이터의 phase 와 얼마나 일치하는가?'\n"
        "  - DOA estimation (direction-of-arrival, antenna array) 의 ocean wave 적용. Multi-frame 의 phase 정보를 dispersion 으로 spatial wavenumber 와 연결.\n\n"
        "(4) Coarse-to-fine 2 단계:\n"
        "  - Coarse: 50×50 = 2,500 후보 격자. 각 후보의 S(k) 계산. Top-N peak 선택.\n"
        "  - Fine: 각 peak 주변에서 local quadratic fit → Newton step → sub-grid 정확도.\n"
        "  - 결과: 처음에는 거친 격자, 점차 정밀화. Coarse-to-fine 이 표준 spectral estimation 기법.\n\n"
        "(5) 'konly' 의 의미:\n"
        "  - 이 단계는 k 만 추정. ω 는 dispersion 으로 자동, (A, φ) 는 다음 tracking 단계에서.\n"
        "  - 이유: single frame 에서 (A, φ) 는 well-conditioned (정보 충분) 이지만 k 는 ill-conditioned.\n"
        "    Fisher Information Matrix 분석 시 k 의 eigenvalue 가 (A, φ) 에 비해 매우 작음.\n"
        "  - 이게 'identifiability decomposition' — paper 후반부의 supporting analysis.\n\n"
        "v1 (FFT) → v2 (beamforming) 변경:\n"
        "  - FFT 는 균일 격자 데이터 필요. 우리 LiDAR 점은 불규칙 (raydrop, shadowing).\n"
        "  - Beamforming 은 불규칙 점에서 직접 작동 — non-uniform DFT.\n"
        "  - 평균 18% 정확도 개선 확인 (2026-04 ablation).\n\n"
        "다음 슬라이드: tracking 단계와 최적화 detail."
    ))


# S13c — Tracking + Adaptive N detail (NEW)
def build_s13c_track():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 3 · WaveSplat — 추적 단계 (Track)", color=GOLD)
    add_title(s, "Track: 매 frame Adam 최적화 + Adaptive N (RMSE 기반)")
    add_subtitle(s, "k 고정 후 (A, φ) 만 빠르게 update — 13 ms/frame")
    # Left column: Tracking + optimizer detail
    bullets_left = [
        "**w_phi Tracking 단계**:",
        "  (k_x, k_y): Init 결과 고정 (변경 안 함)",
        "  ω: dispersion 으로 자동 (k 에서 계산)",
        "  (A, φ): Adam 으로 매 frame update — 자유도 2N",
        "",
        "**Loss function**:",
        "  L(θ) = Σ_i [η_model(x_i, t; θ) − η_obs(x_i, t)]²",
        "  N 개 component 의 wave 합산 vs LiDAR 측정값",
        "  η_model = Σ_n A_n cos(k_n·x − ω_n·t + φ_n)",
        "",
        "**Optimizer — Adam (Kingma & Ba 2014)**:",
        "  lr = 0.01 (현재 default)",
        "  50 iterations × ∇L 계산 / frame → ~13 ms (CPU)",
        "  Adam 의 adaptive learning rate 가 (A, φ) 의 다른 scale 처리",
        "",
        "**Coarse-to-fine 학습률 스케줄 (option)**:",
        "  Stage 1 (lr 0.05, 20 iter): 큰 step 으로 빠른 수렴",
        "  Stage 2 (lr 0.01, 30 iter): 작은 step 으로 미세 조정",
        "",
        "**왜 k 고정?** (Identifiability)",
        "  Single frame 에서 k 의 FIM eigenvalue 가 (A, φ) 보다 10⁻³ 작음",
        "  k 자유 시 50 점에 over-fit → 비물리적 값으로 발산",
        "  k 고정 → 안정적 추정 + paper 의 main contribution mechanism",
    ]
    add_bullets(s, 0.4, 1.85, 6.7, 5.4, bullets_left, font_size=10,
                line_space_after=2)
    # Right column: Adaptive N
    bullets_right = [
        "**Adaptive N — component 수 동적 조정**:",
        "",
        "**① Pruning (가지치기)**:",
        "  매 K frame 마다 (default K=10):",
        "  |A_n| < ε_prune (= 0.05·max|A|) → 제거",
        "  에너지 기여 낮은 component 정리",
        "",
        "**② Growth (성장)**:",
        "  Residual ‖η_obs − η_model‖ > threshold:",
        "  잔여 오차에 다시 beamforming → 새 (k_x, k_y) 추출",
        "  → component 추가",
        "",
        "**③ Safety rollback (안전 복귀)**:",
        "  새 component 추가 후 RMSE 측정",
        "  RMSE_new > 1.3 · RMSE_old:",
        "  → 즉시 이전 state 로 복구",
        "  → 잘못된 추가의 cascade failure 방지",
        "",
        "**Test coverage**: rollback path 가 unit test",
        "  18 wavesplat tests 중 4 개가 adaptive N 검증",
    ]
    add_bullets(s, 7.3, 1.85, 5.7, 5.4, bullets_right, font_size=10,
                line_space_after=2, color=ACCENT)
    add_page_number(s, 13)
    add_notes(s, (
        "추적 단계와 Adaptive N을 자세히 설명합니다.\n\n"
        "Tracking의 핵심 — w_phi mode:\n"
        "  파수(k)를 고정하는 것이 WaveSplat의 가장 중요한 설계 결정입니다.\n"
        "  2026년 4월 6일에 발견한 것인데, k까지 자유롭게 업데이트하면\n"
        "  50개 점에 맞추려고 물리적으로 무의미한 값으로 변해버립니다 (overfitting).\n\n"
        "  비유하면: 라디오 채널(k)을 이미 맞춰놨는데,\n"
        "  매번 채널을 다시 바꾸면 오히려 잡음만 들립니다.\n"
        "  채널은 고정하고 볼륨(A)과 위상(φ)만 조절하면 됩니다.\n\n"
        "Adam optimizer:\n"
        "  딥러닝에서 가장 많이 쓰이는 최적화 알고리즘입니다.\n"
        "  각 파라미터의 gradient를 계산하고, 적응적 학습률로 업데이트합니다.\n"
        "  50 iterations × 50 관측점 → ~13ms (CPU).\n\n"
        "Adaptive N이 필요한 이유:\n"
        "  실제 파도의 성분 수(N)를 미리 알 수 없습니다.\n"
        "  BF2의 단순한 파도는 2-3개, BF5의 복잡한 파도는 15-30개.\n"
        "  N을 고정하면 과소추정(underfitting) 또는 과대추정(overfitting) 발생.\n\n"
        "Safety rollback의 중요성:\n"
        "  새 성분을 추가했는데 오히려 RMSE가 30% 이상 증가하면,\n"
        "  그 성분은 '잘못된 추가'였다는 뜻입니다.\n"
        "  즉시 이전 상태로 되돌려서 시스템 안정성을 보장합니다.\n"
        "  이 메커니즘이 없으면 한 번의 실수로 전체가 붕괴할 수 있습니다."
    ))


# S13d — GT vs WaveSplat 비교 시각화 (NEW)
def build_s13d_visualization():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 3 · WaveSplat — GT 와의 비교 시각화", color=GOLD)
    add_title(s, "Ground Truth vs WaveSplat 재구성 — 실제 결과 한 케이스")
    add_subtitle(s, "한 episode 의 한 시점 — 좌측 GT, 중앙 WaveSplat 추정, 우측 오차 (Error = GT − WaveSplat)")
    fig_demo = os.path.join(PAPER_FIG_DIR, 'demo_wavesplat_static.png')
    add_image_or_placeholder(s, fig_demo, 0.4, 1.7, 12.5,
                             placeholder_text='[GT | WaveSplat | Error heatmap]')
    add_bullets(s, 0.5, 5.7, 12.3, 1.4, [
        "**좌 (GT)**: Ground Truth η(x, y, t) — 50×50 격자 위 실제 파도 표면 (참값)",
        "**중 (WaveSplat)**: WaveSplat 재구성 — 추정된 N 개 sin 파 성분의 합으로 다시 만든 파도 표면",
        "**우 (Error)**: 오차 = GT − WaveSplat — 같은 색 스케일로 표시 (잔여 차이가 작을수록 추정 정확)",
        "색상 의미: blue (음의 변위, trough) → white (수면) → red (양의 변위, crest)",
    ], font_size=11, line_space_after=2)
    add_text(s, 0.5, 7.05, 12.3, 0.35,
             "📂 paper/figures/demo_wavesplat.gif (animation), fig8_qualitative.pdf (다른 case 들) 참조 · 대표 case 모음은 Appendix C",
             font_size=10, italic=True, color=GRAY)
    add_page_number(s, 14)
    add_notes(s, (
        "WaveSplat 추정 결과를 한 episode 의 한 시점에서 시각적으로 보여줍니다. "
        "사용자 정정: 이 figure 는 ETKF 가 아니라 *Error* (잔여 오차) panel 임.\n\n"
        "(1) 그림 구성:\n"
        "  - 좌 (Ground Truth): 50×50 격자 위 실제 파도 표면 η(x,y,t). WaveSim 이 만든 합성 데이터의 참값.\n"
        "  - 중 (WaveSplat): 추정된 N 개 sin 파 성분의 합으로 재구성한 파도. 격자 위에서 evaluate 한 값.\n"
        "  - 우 (Error = GT − WaveSplat): 같은 색 스케일 (vmin/vmax) 로 표시한 잔여 오차. 이상적으로 모든 셀이 white (오차 0).\n\n"
        "(2) 관찰 포인트:\n"
        "  - 좌-중: GT 와 WaveSplat 의 파도 패턴이 거의 일치 — 위치, 방향, 진폭 모두 잘 재구성.\n"
        "  - 우 (Error): 대부분 white → 추정 잔여 오차가 작음. 일부 영역 (격자 가장자리 또는 복잡한 wave 만나는 곳) 에서 약간의 오차.\n"
        "  - 단일 case 시각화 — 정량 비교는 다음 슬라이드 (S14) 의 12 episode 평균.\n\n"
        "(3) 사용된 figure:\n"
        "  - paper/figures/demo_wavesplat_static.png — 한 case 의 static 버전 (현재 PPT 삽입).\n"
        "  - paper/figures/demo_wavesplat.gif — animation 버전 (1481×404, 30 s 시간 진행).\n"
        "  - paper/figures/fig8_qualitative.pdf — 여러 sea state × N 조합의 GT/WaveSplat/ETKF 비교 (Appendix C 에 발췌).\n\n"
        "(4) 추가 case 들:\n"
        "  - 사용자 요청: 다른 sea state / N_components 조합의 대표 case 들도 보여달라 → Appendix C 에 추가.\n"
        "  - Appendix C 에는 BF3/BF4/BF5 의 단순 (2 comp) ~ 복잡 (30 comp) 까지 다양한 case.\n\n"
        "(5) 발표 tip:\n"
        "  - 이 슬라이드에서 잠깐 멈추고 청자가 GT 와 WaveSplat 의 파도 패턴을 직접 비교하게 함.\n"
        "  - '눈으로 봐도 거의 같음, Error panel 이 거의 white' 이 WaveSplat 효과의 직관적 evidence.\n"
        "  - 다음 슬라이드 (S14) 의 정량 비교 (nRMSE 0.211 vs 0.253) 가 이 시각적 인상을 뒷받침."
    ))


# S14 — Prior result (P1 12-ep + real-time)
def build_s14():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 3 · 비교 실험 결과", color=GOLD)
    add_title(s, "12-Episode 비교: WaveSplat vs Wang 2025")
    add_subtitle(s, "Wang 2025 (Coastal Engineering 197) 논문의 방법을 충실히 재현하여 비교")
    # Bar chart (left) + Table (right)
    fig_bar = os.path.join(DOC_DIR, 'fig_3way_bar.png')
    add_image_or_placeholder(s, fig_bar, 0.3, 1.7, 6.5, height=3.8,
                             placeholder_text='[2-way bar chart]')
    # Compact table on right
    headers = ["방법", "nRMSE", "속도", "설명"]
    rows = [
        ["WaveSplat (ours)", "0.211", "13 ms", "연속 파라미터 추정"],
        ["Wang2025 (Ne=300)", "0.253", "632 ms", "논문 충실 재현"],
    ]
    add_table_simple(s, 7.0, 1.85, 5.8, 1.2, headers, rows,
                     header_color=ACCENT, font_size=12, highlight_col=0)
    add_bullets(s, 7.1, 3.2, 5.7, 2.5, [
        "**16%** 더 정확한 추정",
        "**47배** 더 빠른 속도",
        "GPU 없이 CPU만으로 작동",
        "10Hz 제어 루프에 충분한 속도",
        "",
        "Wang2025 비교가 공정한 이유:",
        "  원논문 세팅 그대로 (Ne=300)",
        "  같은 물리 모델(linear) 사용",
    ], font_size=11, line_space_after=2, color=ACCENT)
    # Bottom summary
    add_bullets(s, 0.6, 5.8, 12.0, 1.5, [
        "**단순 파도(1-5 comp)**: WaveSplat이 압도적 — BF3_2comp에서 오차율 5%에 불과 (0.052 vs 0.250)",
        "**복잡한 파도(15+ comp)**: Wang2025가 더 나은 경우도 있음 — 정직하게 다음 슬라이드에서 분석",
    ], font_size=12, line_space_after=3)
    # ── Annotation guide (user adds arrows manually in PPT) ──
    add_text(s, 0.3, 5.2, 6.5, 0.5,
             '📝 화살표 추가: ① "47x faster" (속도 바 사이) ② "16% more accurate" (nRMSE 바 사이)',
             font_size=9, italic=True, color=RGBColor(0x99, 0x99, 0x99))
    add_page_number(s, 14)
    add_notes(s, (
        "이 슬라이드가 핵심 결과입니다.\n\n"
        "먼저 비교 방법을 설명드리겠습니다:\n\n"
        "① WaveSplat (우리 방법):\n"
        "  파도를 '파수, 주파수, 진폭, 위상'이라는 물리적 파라미터로 직접 표현합니다.\n"
        "  격자를 쓰지 않고, 파라미터 자체를 최적화하는 방식입니다.\n\n"
        "② Wang2025 (비교 대상):\n"
        "  2025년에 출판된 Wang의 논문(Coastal Engineering 197)을 충실히 재현한 것입니다.\n"
        "  300개의 시뮬레이션을 동시에 돌리고(앙상블), 80x60 격자 위에서 추정합니다.\n"
        "  fuzzy R이라는 기법으로 관측 노이즈를 자동 적응시키고,\n"
        "  Delaunay 보간으로 불규칙한 LiDAR 점을 처리합니다.\n"
        "  우리가 임의로 만든 baseline이 아니라, 출판된 논문의 충실한 재현입니다.\n\n"
        "공정성 보장:\n"
        "  물리 모델을 동일하게 맞췄습니다. Wang2025 원논문은 HOS(비선형)를 쓰지만,\n"
        "  WaveSplat도 linear이므로, 공정한 비교를 위해 둘 다 linear dispersion을 적용했습니다.\n"
        "  만약 Wang2025에만 HOS를 쓰면, 물리 모델 차이와 알고리즘 차이가 섞여서\n"
        "  순수한 알고리즘 비교가 어렵습니다.\n\n"
        "결과 해석:\n"
        "  WaveSplat이 평균 nRMSE 0.211로 더 정확하고, 13ms로 47배 빠릅니다.\n\n"
        "에피소드별 패턴 (매우 중요합니다):\n"
        "  - 단순한 파도 (1~5개 성분): WaveSplat이 크게 우수합니다.\n"
        "    예) BF3_2comp: WaveSplat 0.052 vs Wang2025 0.250\n"
        "    2개 성분만 있으면 거의 완벽하게 추정한다는 뜻입니다.\n"
        "    격자 방법은 해상도 한계 때문에 이 정확도에 도달할 수 없습니다.\n\n"
        "  - 복잡한 파도 (15~30개 성분): 역전이 일어납니다.\n"
        "    BF4_15comp: WaveSplat 0.302 vs Wang2025 0.249\n"
        "    이유: 격자 방법은 모든 성분을 격자에 한꺼번에 담을 수 있지만,\n"
        "    WaveSplat은 성분을 하나하나 찾아야 해서 30개는 초기화가 어렵습니다.\n"
        "    이 한계는 나중에 정직하게 다시 다룹니다."
    ))


# S13 — Episode-level scatter plot (NEW)
def build_s13_scatter():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 3 · 에피소드별 상세 분석", color=GOLD)
    add_title(s, "파도가 단순할수록 WaveSplat이 유리합니다")
    add_subtitle(s, "파도 성분 수(가로축)에 따라 정확도가 어떻게 달라지는지 — 그리고 역전 지점도 있습니다")
    fig_scatter = os.path.join(DOC_DIR, 'fig_complexity_scatter.png')
    add_image_or_placeholder(s, fig_scatter, 0.5, 1.7, 12.3, height=4.5,
                             placeholder_text='[complexity scatter plot]')
    add_bullets(s, 0.6, 6.3, 12.0, 1.0, [
        "**단순 파도 (1-5 comp)**: WaveSplat이 크게 우수 — BF3_2comp에서 오차율 5%에 불과",
        "**복잡한 파도 (15-30 comp)**: Wang2025가 더 나은 경우도 있음 — 격자가 많은 성분을 한꺼번에 담을 수 있기 때문",
    ], font_size=12, line_space_after=3)
    # ── Annotation guide (user adds arrows manually in PPT) ──
    add_text(s, 0.5, 6.0, 12.0, 0.3,
             '📝 화살표 추가: ① BF3_2comp 강조 원 ② Wang2025 ceiling 점선 (≈0.25) ③ "reversal point" (15comp 부근)',
             font_size=9, italic=True, color=RGBColor(0x99, 0x99, 0x99))
    add_page_number(s, 15)
    add_notes(s, (
        "이 그래프가 우리 연구의 핵심 통찰을 보여줍니다.\n\n"
        "그래프 읽는 법을 먼저 설명드리겠습니다:\n"
        "  - 가로축: 파도 성분의 수. 바다에는 여러 파도가 동시에 존재합니다.\n"
        "    1~2개 성분이면 아주 단순한 파도(한두 방향에서 오는 큰 파도),\n"
        "    15~30개면 매우 복잡한 파도(다양한 방향과 주기의 파도가 뒤섞인 상태)입니다.\n"
        "  - 세로축: nRMSE(오차). 낮을수록 정확합니다.\n"
        "  - 파란 동그라미: WaveSplat (우리 방법)\n"
        "  - 빨간 네모: Wang2025\n\n"
        "핵심 패턴 1 — WaveSplat의 강점:\n"
        "  파란 점이 왼쪽(단순한 파도)에서 매우 낮습니다.\n"
        "  BF3_2comp에서 0.052 = 오차가 5%밖에 안 됩니다.\n"
        "  왜 가능한가? 2개 성분이면 파라미터 10개만 추정하면 되고,\n"
        "  50개 관측점이면 충분히 잘 맞출 수 있기 때문입니다.\n"
        "  격자 방법은 성분이 적어도 격자 해상도 한계가 있어서 이 정확도에 도달하기 어렵습니다.\n\n"
        "핵심 패턴 2 — Wang2025의 특성:\n"
        "  빨간 점이 복잡도에 관계없이 약 0.25 주변에 모여 있습니다.\n"
        "  이것은 격자(NxN grid)의 해상도 한계 때문입니다.\n"
        "  격자가 고정되어 있어서, 성분이 적든 많든 비슷한 정확도를 보여줍니다.\n\n"
        "핵심 패턴 3 — 역전 지점:\n"
        "  15개 이상의 성분에서는 파란 점이 빨간 점보다 위로 갑니다.\n"
        "  WaveSplat은 성분을 하나하나 찾아야 하는데, 30개를 정확히 초기화하기가 어렵습니다.\n"
        "  반면 격자 방법은 모든 성분을 격자에 한꺼번에 담을 수 있어서 이런 문제가 없습니다.\n\n"
        "정리: WaveSplat은 '소수 성분을 정밀하게 추정'하는 데 특화되어 있고,\n"
        "격자 방법은 '다수 성분을 균일하게 추정'하는 데 강합니다.\n"
        "이 trade-off를 논문에서 정직하게 보고할 예정입니다."
    ))


# S14 — Ablation study (was S15)
def build_s15():
    s = add_blank(LIGHT_BG)
    add_section_tag(s, "Part 4 · 어떤 기술이 도움이 되는가?", color=ACCENT2)
    add_title(s, "Ablation — 각 기술의 기여도를 분리해서 확인")
    add_subtitle(s, "왜 ablation을 하는가? 여러 기술을 한꺼번에 쓰면 '어떤 기술이 실제로 효과가 있는지' 알 수 없기 때문입니다")
    bullets = [
        "**Ablation이란?** — 기술을 하나씩 빼 보면서, 빠진 기술이 실제로 도움이 되는지 확인하는 실험",
        "  비유: 음식에 재료를 하나씩 빼 보면서 맛의 변화를 확인하는 것과 같습니다",
    ]
    add_bullets(s, 0.7, 1.85, 12.0, 1.0, bullets, font_size=13,
                line_space_after=3)
    # Ablation bar chart (left)
    fig_abl = os.path.join(DOC_DIR, 'fig_ablation_bar.png')
    add_image_or_placeholder(s, fig_abl, 0.3, 3.0, 7.0, height=3.6,
                             placeholder_text='[ablation bar chart]')
    # Key insights (right) — with explanations
    add_bullets(s, 7.5, 3.0, 5.5, 4.0, [
        "**A** (기본): FFT 초기화 + 자유 주파수",
        "  → nRMSE 0.213",
        "**B** (+beamform + multi-frame):",
        "  → 0.208 (A보다 개선)",
        "**C** (+물리 법칙, 단일 프레임):",
        "  → 0.226 (오히려 악화!)",
        "**D = WaveSplat** (모두 결합):",
        "  → 0.211 (가장 안정적)",
        "",
        "물리 법칙 + multi-frame을",
        "함께 써야 효과적!",
    ], font_size=11, line_space_after=1)
    # ── Annotation guide (user adds arrows manually in PPT) ──
    add_text(s, 0.3, 6.8, 7.0, 0.3,
             '📝 화살표 추가: ① C→D "dispersion regularizer" 효과 화살표 ② A→B "multi-frame" 효과 화살표',
             font_size=9, italic=True, color=RGBColor(0x99, 0x99, 0x99))
    add_page_number(s, 16)
    add_notes(s, (
        "Ablation study(절제 연구)가 무엇인지부터 설명드리겠습니다.\n\n"
        "Ablation이란?\n"
        "  우리 시스템에는 여러 기술이 들어 있습니다.\n"
        "  이 기술들을 하나씩 빼 보면서, 빠진 기술이 실제로 도움이 되는지 확인합니다.\n"
        "  비유하면: 레시피에서 재료를 하나씩 빼 보면서 맛이 어떻게 변하는지 보는 것입니다.\n"
        "  의학에서 '절제(ablation)'라는 용어에서 왔습니다.\n\n"
        "왜 ablation이 필요한가?\n"
        "  여러 기술을 한꺼번에 넣으면 '전체 결과'는 보이지만,\n"
        "  '어떤 기술이 실제로 효과가 있는지'는 알 수 없습니다.\n"
        "  예를 들어 결과가 좋아도, 실은 한 가지 기술만 효과가 있고\n"
        "  나머지는 아무 도움이 안 될 수도 있습니다.\n"
        "  논문 심사에서 '각 기술의 기여도를 증명하라'는 요구가 반드시 나옵니다.\n\n"
        "4가지 변형의 의미:\n"
        "  A: 가장 기본적인 방법. FFT로 초기 파수를 추정하고, 주파수(omega)를 자유롭게 둠.\n"
        "     단일 프레임(한 순간의 LiDAR 점)만 사용.\n"
        "  B: beamforming이라는 더 좋은 초기화 + 여러 프레임(시간 정보 활용).\n"
        "     주파수는 여전히 자유.\n"
        "  C: beamforming + 물리 법칙(분산 관계) 강제. 하지만 단일 프레임만 사용.\n"
        "  D (WaveSplat): 모든 기술 결합 — beamforming + 물리 법칙 + multi-frame.\n\n"
        "결과 해석:\n"
        "  - A→B: 0.213 → 0.208 — multi-frame이 도움됨 (시간 정보 활용)\n"
        "  - A→C: 0.213 → 0.226 — 물리 법칙만 넣으면 오히려 악화!\n"
        "    이유: 단일 프레임에서 초기 파수 추정이 부정확하면,\n"
        "    물리 법칙이 그 오류를 주파수로 전파시킵니다.\n"
        "  - D (모두 결합): 0.211 — 가장 안정적인 결과\n"
        "    물리 법칙과 multi-frame을 함께 써야 시너지가 발생합니다.\n\n"
        "에피소드별 상세:\n"
        "  - 단순 파도에서는 D가 A보다 최대 42% 더 좋음 (물리 법칙 효과)\n"
        "  - 복잡 파도에서는 A가 D보다 더 좋은 경우도 있음 (자유도 높아서 fitting 유리)\n"
        "  - 하지만 D의 해가 물리적으로 더 의미 있는 해입니다.\n\n"
        "핵심 메시지:\n"
        "  '물리 법칙(dispersion)은 마법이 아니라 정규화(regularizer) 역할을 한다.'\n"
        "  쉽게 말하면, 물리 법칙이 '물리적으로 가능한 답만 찾도록' 범위를 좁혀주는 것입니다."
    ))


# S16 — Main framing (한국어)
def build_s16():
    s = add_blank(LIGHT_BG)
    add_section_tag(s, "Part 4 · 핵심 메커니즘 정리", color=ACCENT2)
    add_title(s, "분산 관계 = 물리적 정규화 장치",
              size=24, color=ACCENT)
    add_text(s, 0.5, 1.4, 12.3, 1.2,
             ("분산 관계(dispersion relation)는 추정해야 할 변수를 20% 줄이고, "
              "물리적으로 가능한 해만 탐색하도록 제한합니다. "
              "이것이 50개 관측점으로도 파도장을 추정할 수 있는 핵심 비결입니다."),
             font_size=16, italic=True, color=DARK,
             alignment=PP_ALIGN.CENTER)
    bullets = [
        "**메커니즘 1 — 자유도 감소 (변수 줄이기)**",
        "  분산 관계: ω = √(g·|k|·tanh(|k|·d)) → 주파수(ω)가 파수(k)로부터 자동 결정됨",
        "  적용 전: 성분당 5개 변수 (진폭 A, 파수 kx, ky, 주파수 ω, 위상 φ) = 5N개",
        "  적용 후: 성분당 4개 변수 (A, kx, ky, φ) = 4N개 → 20% 감소",
        "  추가 효과: 물리적으로 불가능한 (k, ω) 조합이 자동으로 배제됨",
        "",
        "**메커니즘 2 — 시간 정보 활용 (multi-frame 융합)**",
        "  선박이 V=11.3 m/s로 이동하면, 1초 동안 11.3m를 쓸고 지나감",
        "  이 효과로 '가상의 긴 안테나(synthetic aperture)'가 만들어짐",
        "  단일 프레임 LiDAR 범위 5-8m → multi-frame으로 11m+ 확장",
        "  → 파수(k)의 모호성이 해소됨 (어떤 파도인지 구분 가능해짐)",
        "",
        "**Ablation에서 확인된 것 — 두 메커니즘의 시너지**:",
        "  물리 법칙만 단독으로 쓰면(C) → 오히려 성능 저하 (초기 오류가 전파됨)",
        "  multi-frame만 단독으로 쓰면(B) → 개선되지만 불안정",
        "  둘을 함께 쓰면(D=WaveSplat) → 가장 안정적이고 물리적으로 의미 있는 결과",
    ]
    add_bullets(s, 0.6, 2.75, 12.0, 4.6, bullets, font_size=12,
                line_space_after=3)
    add_page_number(s, 17)
    add_notes(s, (
        "이 슬라이드가 우리 연구의 핵심 메시지입니다.\n\n"
        "처음에 'super-resolution(초해상도)'이라는 멋진 표현을 쓰려 했지만,\n"
        "정확히 말하면 '물리적 정규화(physical regularization)'가 맞습니다.\n\n"
        "메커니즘 1을 쉽게 설명하면:\n"
        "  분산 관계란 '파도의 주파수와 파수 사이의 물리 법칙'입니다.\n"
        "  깊은 바다에서 짧은 파도(큰 k)는 느리고, 긴 파도(작은 k)는 빠릅니다.\n"
        "  이 관계를 알고 있으면, 파수(k)를 알면 주파수(ω)를 계산할 수 있습니다.\n"
        "  결과적으로 추정해야 할 변수가 5N개에서 4N개로 20% 줄어듭니다.\n"
        "  게다가, 물리적으로 불가능한 답(예: 짧은 파도인데 빠른 주파수)이 자동으로 배제됩니다.\n\n"
        "메커니즘 2를 쉽게 설명하면:\n"
        "  한 순간(단일 프레임)의 LiDAR 점만으로는 파도의 방향을 정확히 알기 어렵습니다.\n"
        "  하지만 선박이 이동하면서 여러 순간의 데이터를 모으면,\n"
        "  마치 긴 안테나를 만든 것처럼 방향 분해능이 좋아집니다.\n"
        "  이것을 'synthetic aperture(합성 개구)'라고 부릅니다.\n"
        "  레이더 분야의 SAR(Synthetic Aperture Radar)과 같은 원리입니다.\n\n"
        "Ablation에서 핵심 발견:\n"
        "  물리 법칙과 multi-frame은 각각 단독으로는 불완전합니다.\n"
        "  물리 법칙만 쓰면(C): 초기 파수 추정이 틀렸을 때, 그 오류가 주파수로 전파됩니다.\n"
        "  multi-frame만 쓰면(B): 개선되지만 물리적 제약 없이 자유도가 너무 높아 불안정합니다.\n"
        "  둘을 함께 쓰면(D): 시너지가 발생합니다.\n"
        "  — multi-frame으로 파수를 잘 추정하고,\n"
        "  — 물리 법칙으로 물리적으로 가능한 해만 탐색하고,\n"
        "  — 결과적으로 가장 안정적인 추정이 됩니다."
    ))


# S17 — Mechanism + Identifiability
def build_s17():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 4 · 왜 2단계 설계가 자연스러운가", color=ACCENT2)
    add_title(s, "초기화(Init)와 추적(Track)을 분리하는 이유")
    add_subtitle(s, "어떤 변수는 시간 정보가 필요하고, 어떤 변수는 한 순간으로 충분합니다")
    bullets = [
        "**핵심 관찰 — 변수마다 필요한 정보가 다릅니다**",
        "  파수 k = (kx, ky): 여러 프레임의 시간 정보가 있어야 추정 가능",
        "  주파수 ω, 진폭 A, 위상 φ: 한 프레임(순간)만으로도 추정 가능",
        "",
        "**그래서 2단계로 나눕니다:**",
        "  초기화(Init): 여러 프레임을 모아서 파수 k를 한 번 정확히 결정 (< 1초)",
        "  추적(Track): 매 프레임마다 k를 고정하고, ω·A·φ만 빠르게 업데이트 (< 13ms)",
        "  → 무거운 작업은 처음 한 번만, 가벼운 작업은 매 프레임 실시간으로",
        "",
        "**전체 흐름 정리:**",
        "  1. 선박 이동 → synthetic aperture(합성 개구) 형성 → 파수 추정",
        "  2. 분산 관계로 물리적 제약 부여 → 탐색 공간 축소",
        "  3. 매 프레임 Adam 최적화 → 실시간 추적",
        "  4. Adaptive N: 성분 수를 자동으로 조정 (추가/제거/복원)",
    ]
    add_bullets(s, 0.6, 1.85, 8.0, 5.3, bullets, font_size=12,
                line_space_after=3)
    fig_bf = os.path.join(PAPER_FIG_DIR, 'fig_beamforming_2stage.png')
    add_image_or_placeholder(s, fig_bf, 8.7, 2.0, 4.3,
                             placeholder_text='[2-stage beamforming]')
    add_page_number(s, 18)
    add_notes(s, (
        "이 슬라이드는 '왜 2단계 설계가 자연스러운가'를 설명합니다.\n\n"
        "핵심 관찰을 쉽게 설명하겠습니다:\n\n"
        "  파수(k)란 '파도의 공간적 패턴'입니다.\n"
        "  한 순간의 LiDAR 스캔(50개 점)만으로는 이 패턴을 정확히 파악하기 어렵습니다.\n"
        "  왜냐하면 50개 점이 커버하는 영역(5~8m)이 파도 파장(50~300m)보다 훨씬 짧기 때문입니다.\n"
        "  하지만 선박이 이동하면서 여러 순간의 데이터를 모으면,\n"
        "  관측 영역이 효과적으로 넓어져서 파수를 정확히 추정할 수 있습니다.\n\n"
        "  반면 진폭(A)이나 위상(φ)은 한 순간의 데이터로도 충분합니다.\n"
        "  파도의 '크기'와 '현재 위치'는 한 순간만 봐도 알 수 있기 때문입니다.\n\n"
        "그래서 자연스럽게 2단계로 나뉩니다:\n"
        "  (1) 초기화(Init): 여러 프레임을 모아서 파수 k를 정확히 결정합니다.\n"
        "      이 단계는 한 번만 하면 되고, 1초 이내에 끝납니다.\n"
        "  (2) 추적(Track): 파수를 고정하고, 매 프레임마다 진폭·위상만 업데이트합니다.\n"
        "      이 단계는 13ms 이내에 끝나서 실시간으로 작동합니다.\n\n"
        "비유하면: 라디오 주파수 맞추기와 비슷합니다.\n"
        "  - 처음에 주파수(채널)를 찾는 건 시간이 좀 걸리지만 (= Init)\n"
        "  - 일단 찾으면 볼륨만 조절하면 됩니다 (= Track)\n\n"
        "Adaptive N은 '파도 성분의 수를 자동으로 조정'하는 기능입니다:\n"
        "  - 너무 작은 성분은 제거 (pruning)\n"
        "  - 오차가 크면 새 성분 추가 (growth)\n"
        "  - 추가했더니 오히려 나빠지면 원래 상태로 복원 (safety rollback)"
    ))


# S18 — Limitations
def build_s18():
    s = add_blank(WHITE)
    add_section_tag(s, "Closing · 한계와 향후 개선 방향", color=GRAY)
    add_title(s, "현재의 한계 — 그리고 개선할 수 있는 방향")
    add_subtitle(s, "모든 연구에는 한계가 있습니다. 정직하게 인정하고, 해결 방향을 제시합니다")
    bullets = [
        "**한계 1 — 선형 근사 (Linear Wave Theory)**",
        "  우리는 파도를 '사인파의 합'으로 표현합니다",
        "  실제 파도는 꼭대기가 뾰족하고 골이 넓어서 사인파와 다름",
        "  Klein 2020 비교: BF6(풍력 6)까지는 충분히 정확, 그 이상은 정확도 저하",
        "  개선: Stokes 2차 보정 추가 → 적용 범위 확장 가능",
        "",
        "**한계 2 — 복잡한 파도(15-30개 성분)에서 정확도 저하**",
        "  BF4_15comp: WaveSplat 0.302 vs Wang2025 0.249 (격자 방법이 더 나음)",
        "  원인: 성분이 많으면 초기화 단계에서 정확한 분리가 어려워짐",
        "  개선: 격자 초기화 → 파라미터 정밀화 (hybrid init) 연구 예정",
        "",
        "**한계 3 — 파도 파쇄(breaking)와 유속장 추정은 미지원**",
        "  이런 현상은 VOF-EnKF, PFT-PINN 같은 비선형 방법만 가능",
        "  하지만 이들은 30분~수시간이 필요하므로 실시간 적용은 불가",
        "  우리의 응용 영역(실시간 제어)과는 본질적으로 다른 영역",
    ]
    add_bullets(s, 0.6, 1.85, 12.0, 5.5, bullets, font_size=12,
                line_space_after=3)
    add_page_number(s, 19)
    add_notes(s, (
        "한계를 정직하게 공유하는 시간입니다.\n"
        "연구의 신뢰도를 높이려면, 잘 되는 점뿐 아니라 한계도 먼저 인정해야 합니다.\n\n"
        "(1) 선형 근사 — 가장 근본적인 한계입니다.\n"
        "  쉽게 말하면, 우리는 파도를 '사인파의 합'으로 표현합니다.\n"
        "  실제 바다 파도는 사인파보다 복잡합니다.\n"
        "  (꼭대기가 뾰족하고, 골이 넓고, 높은 파도일수록 비대칭적입니다.)\n"
        "  하지만 적당한 파고까지는 이 근사가 충분히 잘 맞습니다.\n\n"
        "  Klein 2020이라는 논문에서 4가지 파도 모델을 정량 비교했는데:\n"
        "    HOSM > NLSE2 > NLSE > LWT (정확도 순)\n"
        "  우리는 LWT(Linear Wave Theory) family로 가장 단순합니다.\n"
        "  ε(epsilon, 파도의 가파름 정도)이 0.07 이하일 때 신뢰할 수 있고,\n"
        "  BF6(풍력 6, 상당히 거친 바다)까지는 이 조건을 만족합니다.\n"
        "  향후 Stokes 2차 보정을 추가하면 적용 범위를 더 넓힐 수 있습니다.\n\n"
        "(2) 복잡한 파도에서의 역전 — 이번 비교 실험에서 발견된 한계입니다.\n"
        "  파도 성분이 15~30개로 많아지면, 격자 방법(Wang2025)이 더 나아집니다.\n"
        "  이유를 쉽게 설명하면:\n"
        "    격자 방법: 바둑판처럼 모든 성분을 한꺼번에 담을 수 있습니다.\n"
        "    WaveSplat: 성분을 하나하나 찾아야 합니다. 30개를 정확히 찾기는 어렵습니다.\n"
        "  대응: 격자로 대략적인 성분을 먼저 찾고, 그걸 파라미터 초기값으로 정밀화하는\n"
        "  '2단계 초기화(hybrid init)'를 연구할 예정입니다.\n\n"
        "(3) 파도 파쇄와 유속장 — 우리 방법의 적용 범위 밖입니다.\n"
        "  파도가 깨지는 현상(breaking)이나 수중 유속(velocity field)은\n"
        "  비선형 시뮬레이션(VOF-EnKF, PFT-PINN) 같은 방법만 처리할 수 있습니다.\n"
        "  하지만 이들은 30분~수시간이 걸려서 실시간 제어에는 쓸 수 없습니다.\n"
        "  결론: 우리 방법과 비선형 방법은 서로 다른 응용 영역을 담당합니다.\n"
        "  (우리 = 실시간 제어, 그들 = 정밀 분석)"
    ))


# S19 — Future research plan + Q&A
def build_s19():
    s = add_blank(LIGHT_BG)
    add_section_tag(s, "Closing · 향후 연구 방향", color=ACCENT)
    add_title(s, "앞으로의 연구 방향", color=DARK)
    add_subtitle(s, "현재 한계를 극복하기 위한 3가지 연구 방향을 계획하고 있습니다")
    bullets = [
        "**방향 1 — 고복잡도 파도 대응 (초기화 개선)**",
        "  현재 문제: 15개 이상 성분에서 초기화 정확도 저하",
        "  계획: 격자 기반 초기 추정 → 파라미터 정밀화 (hybrid init)",
        "  또는: 신경망으로 초기 파도 성분을 추정 (learned init)",
        "  기대: 고복잡도에서도 Wang2025 수준 이상의 정확도 달성",
        "",
        "**방향 2 — 비선형 파도 확장**",
        "  현재: 선형 파도 이론(LWT)만 사용 → BF6까지 적용 가능",
        "  계획: Stokes 2차 보정 추가 → 더 높은 파도까지 정확",
        "  장기: Klein 2020의 NLSE 수준 도달 목표",
        "",
        "**방향 3 — 실측 데이터 검증**",
        "  현재: 합성 데이터(시뮬레이터)로만 검증",
        "  계획: 실제 선박 LiDAR 데이터로 현장 검증",
        "  도전 과제: ground-truth 확보 (wave buoy 또는 radar 병행 측정)",
        "",
        "**최종 목표 — 선박 실시간 제어에 활용 가능한 파도 추정 시스템**",
    ]
    add_bullets(s, 0.6, 1.85, 12.0, 5.0, bullets, font_size=12,
                line_space_after=3)
    add_text(s, 0.5, 6.85, 12.3, 0.4,
             "질문과 토론을 환영합니다",
             font_size=16, italic=True, color=ACCENT,
             alignment=PP_ALIGN.CENTER)
    add_page_number(s, 20)
    add_notes(s, (
        "향후 연구 방향을 3가지로 정리했습니다.\n\n"
        "(1) 고복잡도 파도 대응:\n"
        "  지금까지의 실험에서, 파도 성분이 15개 이상이면 정확도가 떨어지는 것을 확인했습니다.\n"
        "  원인은 beamforming 초기화가 많은 성분을 동시에 분리하기 어렵기 때문입니다.\n"
        "  해결 방향 1: hybrid init — 먼저 격자(grid) 방법으로 대략적인 성분을 찾고,\n"
        "    그 결과를 파라미터 초기값으로 사용하여 정밀하게 최적화합니다.\n"
        "  해결 방향 2: learned init — 신경망(NN)을 활용하여\n"
        "    LiDAR 점으로부터 초기 파도 성분을 바로 추정합니다.\n"
        "  기대 효과: 현재 고복잡도에서의 역전 현상을 해소할 수 있습니다.\n\n"
        "(2) 비선형 파도 확장:\n"
        "  현재는 파도를 사인파의 합(Linear Wave Theory)으로 표현합니다.\n"
        "  실제 파도는 비선형적이어서(꼭대기가 뾰족, 골이 넓음),\n"
        "  높은 파도에서는 정확도가 떨어집니다.\n"
        "  Stokes 2차 보정을 추가하면, 파도의 비대칭성을 일부 반영할 수 있습니다.\n"
        "  Klein 2020 논문에 따르면, 이 보정으로 적용 범위가 크게 넓어집니다.\n\n"
        "(3) 실측 데이터 검증:\n"
        "  현재까지의 결과는 모두 시뮬레이터(합성 데이터)에서 나온 것입니다.\n"
        "  실제 바다에서 실제 LiDAR로 측정한 데이터로 검증해야 합니다.\n"
        "  가장 큰 도전은 'ground-truth(정답)'를 어떻게 확보하느냐입니다.\n"
        "  Wave buoy(파고계)나 X-band radar로 동시 측정하는 방법을 고려하고 있습니다.\n\n"
        "최종 목표:\n"
        "  선박에 탑재된 LiDAR로 주변 파도를 실시간(13ms 이내)으로 추정하고,\n"
        "  이 정보를 활용하여 선박 제어(hydrofoil ride control, 자율운항 등)에\n"
        "  직접 활용할 수 있는 시스템을 만드는 것입니다.\n\n"
        "이상으로 발표를 마치겠습니다. 질문이나 피드백을 환영합니다."
    ))


# ─────────────────────────────────────────
# Backup slides — 논문별 상세 (내용, 방법론, 결과, contribution, limitation)
# ─────────────────────────────────────────
def build_backup(letter, title, key_points, notes=''):
    s = add_blank(WHITE)
    add_section_tag(s, f"Backup · {letter} (Q&A 시 참조)", color=GRAY)
    add_title(s, title, size=22)
    add_bullets(s, 0.6, 1.75, 12.0, 5.3, key_points, font_size=12,
                line_space_after=3)
    add_text(s, 12.5, 7.05, 0.7, 0.35,
             letter, font_size=10, color=GRAY,
             alignment=PP_ALIGN.RIGHT)
    add_notes(s, notes)


def build_backups():
    """Single-slide Appendix A (prior art table) + Appendix B (dataset table)."""
    # ───────── Appendix A: 선행연구 11편 한 표로 ─────────
    s = add_blank(WHITE)
    add_section_tag(s, "Appendix A · Prior art 13 편 정독 요약 (Q&A 시 참조)", color=GRAY)
    add_title(s, "Appendix A — 선행연구 13 편 한 눈에", size=22)
    add_subtitle(s,
                 "Detail 은 본문 슬라이드 (S5-S9) + docs/strategic_repositioning.md 참조",
                 size=11)
    headers = ["#", "Paper (Year)", "핵심 방법", "강점", "한계 / 우리 차별"]
    rows = [
        ["1", "Desmars 2018 (OMAE)",
         "LIDAR + ICWM, predefined k-grid, variational",
         "LIDAR 응용 첫 시도 (concept proposal)",
         "Simulator only, 1D, on-grid"],
        ["2", "Desmars 2020 (Ocean Eng)",
         "Wave gauge array + dispersion correction",
         "첫 실측 (wave tank) 검증",
         "Wave gauge ≠ LiDAR, 1D"],
        ["3", "Desmars 2023 (JFM 975)",
         "HOS-OCM, vessel-mounted optical, 3-stage",
         "**가장 가까운 prior art** (vessel + phase-resolved)",
         "1D, on-grid, sensor model 미명시"],
        ["4", "Wang & Pan 2021 (JFM 918)",
         "EnKF + HOS 첫 결합, 1D",
         "EnKF-HOS family 의 시작",
         "Dense obs 가정, 1D, ensemble cost"],
        ["5", "Wang & Pan 2022 (JFM 949)",
         "EnKF-HOS-C (current 동시 추정)",
         "Current estimation 확장",
         "여전히 1D, dense obs"],
        ["6", "Wang 2025 (Coastal Eng 197)",
         "EnKF + Ne=300 + fuzzy R + Delaunay",
         "Stereo camera 2D 실시간 (격자 기반)",
         "**우리 baseline 의 충실 재현 대상**, 격자 한계, 632 ms"],
        ["7", "Hunt 2007 (Tellus A)",
         "LETKF (Local Ensemble Transform KF)",
         "모든 EnKF wave 의 이론 기반",
         "Wave-specific 아님 (기상학 출신)"],
        ["8", "Klein 2020 (Fluids 5(1))",
         "Wave model 정량 비교 (LWT/NLSE/NLSE2/HOSM)",
         "ε < 0.07 operational range 정당화 출처",
         "Comparison study, 새 알고리즘 아님"],
        ["9", "Ehlers 2023 (Ocean Eng)",
         "Fourier Neural Operator (FNO) supervised",
         "1.9 ms inference, fastest ML",
         "3120 sample 학습 필요, 1D, distribution shift"],
        ["10", "Ehlers 2025 PFT-PINN (PRF)",
         "Physics-informed NN + PFT loss, unsupervised",
         "비선형 처리 + training data 0개",
         "30 분/instance, 1D + t, buoy only"],
        ["11", "Yan 2025 (arXiv VOF-EnKF)",
         "Navier-Stokes + EnKF + POD",
         "Plunging breaker 처리 (가장 정확)",
         "Offline only, 1D xz-plane, 50 OpenFOAM × 수일"],
        ["12", "Tang-Bhaskar-Recht 2013 (TIT)",
         "Atomic norm off-grid CS",
         "Theoretical recovery bound (off-grid CS 영감)",
         "1D 신호, SDP 비용"],
        ["13", "Möller-Trumbore 1997 (JGT)",
         "Ray-triangle intersection (graphics)",
         "WaveSim 의 ray tracing 표준 알고리즘",
         "비물리적 (geometry only)"],
    ]
    add_table_simple(s, 0.2, 1.85, 12.95, 5.0, headers, rows,
                     header_color=ACCENT, font_size=8)
    add_text(s, 0.5, 7.0, 12.3, 0.4,
             "총 13 편 정독. Möller-Trumbore 와 Hunt 2007 은 algorithmic foundation, 나머지 11 편은 wave estimation prior art.",
             font_size=10, italic=True, color=GRAY)
    add_text(s, 12.5, 7.05, 0.7, 0.35,
             "A", font_size=10, color=GRAY, alignment=PP_ALIGN.RIGHT)
    add_notes(s, (
        "Appendix A — 정독한 13 편의 prior art 와 algorithmic foundation 을 한 표로 정리.\n\n"
        "Wave estimation prior art 11 편:\n"
        "  - Desmars family (2018, 2020, 2023): LIDAR/optical + variational + HOS-OCM 의 발전.\n"
        "  - Wang-Pan / Wang family (2021, 2022, 2025): EnKF-HOS 의 발전. Wang 2025 가 우리 baseline.\n"
        "  - Klein 2020: wave model 정량 비교. 우리 LWT operational range 정당화.\n"
        "  - Ehlers 2023, 2025: ML supervised (FNO) 와 unsupervised (PFT-PINN). 같은 그룹의 진화.\n"
        "  - Yan 2025: 가장 정확한 wave breaking 처리 (offline only).\n\n"
        "Algorithmic foundation 2 편:\n"
        "  - Tang-Bhaskar-Recht 2013: off-grid CS 이론. 우리 parametric 접근의 영감.\n"
        "  - Möller-Trumbore 1997: ray-triangle intersection. WaveSim ray tracing 의 표준.\n\n"
        "Hunt 2007 LETKF 도 algorithmic foundation 으로 EnKF 변형 의 출발.\n\n"
        "각 paper 의 detail 은 본문 슬라이드 (S5-S9, top 5 detail) + docs/strategic_repositioning.md (§17-§31) 참조."
    ))

    # ───────── Appendix B: 110 episode dataset 표 ─────────
    s = add_blank(WHITE)
    add_section_tag(s, "Appendix B · WaveSim 110-episode dataset (Q&A 시 참조)", color=GRAY)
    add_title(s, "Appendix B — 합성 dataset 110 episode 구성", size=22)
    add_subtitle(s,
                 "BF2–BF7 sea state × N=1–30 components × seeds — controlled benchmark",
                 size=11)
    # Episode breakdown table
    headers = ["Sea state", "ε (steepness)", "N_components", "Seeds",
               "Episode 수", "용도"]
    rows = [
        ["BF2 (잔잔)", "~0.02", "1, 2, 5",
         "0–4 (5 개)", "15", "단순 wave, 알고리즘 sanity check"],
        ["BF3", "~0.03", "1, 2, 5, 15",
         "0–4 (5 개)", "20", "low complexity, 통계 평가용"],
        ["BF4 (보통)", "~0.05", "5, 15, 20",
         "0–4 (5 개)", "15", "operational sea state 중심 평가"],
        ["BF5", "~0.07", "5, 15, 20, 30",
         "0–4 (5 개)", "20", "high complexity boundary"],
        ["BF6 (거친)", "~0.08", "15, 20, 30",
         "0 (1 개)", "15", "operational 한계 검증"],
        ["BF7 (매우 거친)", "~0.10", "20, 30",
         "0 (1 개)", "10", "extreme sea state, LWT 한계"],
        ["Calibration", "—", "(separate)",
         "—", "15", "raydrop η, scale 등 calibration 용"],
    ]
    add_table_simple(s, 0.4, 1.85, 12.5, 3.5, headers, rows,
                     header_color=ACCENT2, font_size=10)
    # Episode-level data spec
    add_text(s, 0.4, 5.5, 6.0, 0.4, "Per-episode 데이터 (HDF5 format):",
             font_size=12, bold=True, color=ACCENT)
    add_bullets(s, 0.6, 5.9, 6.0, 1.4, [
        "T = 30 s, Δt = 0.1 s → 300 frames",
        "Ground-truth η field: 50×50 격자 (참값)",
        "LiDAR scan: ~50 valid pts/frame (raydrop 후)",
        "총 110 episode = 약 5 GB",
    ], font_size=11, line_space_after=2)
    # Reproducibility / coverage
    add_text(s, 7.0, 5.5, 6.0, 0.4, "Reproducibility + 검증:",
             font_size=12, bold=True, color=ACCENT)
    add_bullets(s, 7.2, 5.9, 6.0, 1.4, [
        "Seed 저장 → 동일 seed 로 재실행 시 동일 결과",
        "Sensor type: uniform / slope (LiDAR scan 분포)",
        "Test: 53 (wavesim) + 55 (wavesplat) = 108 통과",
        "Generator: scripts/generate_episodes.py (재현 가능)",
    ], font_size=11, line_space_after=2)
    add_text(s, 12.5, 7.05, 0.7, 0.35,
             "B", font_size=10, color=GRAY, alignment=PP_ALIGN.RIGHT)
    add_notes(s, (
        "Appendix B — WaveSim 의 110 episode 합성 dataset 구성을 한 눈에 정리한 표.\n\n"
        "Sea state 별 episode 수:\n"
        "  - BF2-BF5: 각 sea state × 다양한 N_components × seed 0-4 (5 개씩) = 통계 평가 가능.\n"
        "  - BF6-BF7: 극한 sea state. Seed 0 (1 개) 만 — operational range 한계 확인용.\n"
        "  - Calibration: raydrop η scale 등 sensor 모델 calibration 용 별도 episode.\n\n"
        "Per-episode 데이터:\n"
        "  - 30 초 시뮬레이션, 0.1 초 간격 → frame 300 개.\n"
        "  - Ground-truth η field 50×50 격자 — 평가 시 참값.\n"
        "  - LiDAR scan: raydrop / shadowing / Fresnel 적용 후 frame 당 ~50 valid 점.\n"
        "  - HDF5 format. 총 5 GB.\n\n"
        "Reproducibility 보장:\n"
        "  - Seed 저장. Wave generator (Gerstner+JONSWAP) 와 LiDAR raydrop 모두 deterministic.\n"
        "  - 53 + 55 = 108 unit test — simulator 와 estimator 모두 검증.\n"
        "  - Generator script (scripts/generate_episodes.py) 로 누구나 재현 가능.\n\n"
        "이 dataset 자체가 paper 의 contribution 일부 — Ehlers 2023 의 1D radar dataset 보다 풍부한 setup (2D directional + LiDAR sensor + 다양한 sea state)."
    ))

    # ───────── Appendix C: 대표 case 모음 (GT vs WaveSplat vs ETKF) ─────────
    s = add_blank(WHITE)
    add_section_tag(s, "Appendix C · 대표 case 모음 (GT vs WaveSplat vs ETKF)", color=GRAY)
    add_title(s, "Appendix C — 다양한 sea state × N 조합의 대표 결과", size=22)
    add_subtitle(s,
                 "BF3 2-comp (단순) → BF4 15-comp (중간) → BF5 30-comp (복잡) — 한 시점 비교",
                 size=11)
    fig_qual = os.path.join(DOC_DIR, 'fig8_qualitative-1.png')
    add_image_or_placeholder(s, fig_qual, 0.4, 1.7, 12.5,
                             placeholder_text='[fig8_qualitative — 다양한 case 의 GT/WaveSplat/ETKF 비교]')
    add_text(s, 0.5, 7.0, 12.3, 0.4,
             "Source: paper/figures/fig8_qualitative.pdf — 행 마다 다른 episode, 열 마다 GT / WaveSplat / ETKF / GPR. 단순 wave 일수록 WaveSplat 우위.",
             font_size=10, italic=True, color=GRAY)
    add_text(s, 12.5, 7.05, 0.7, 0.35,
             "C", font_size=10, color=GRAY, alignment=PP_ALIGN.RIGHT)
    add_notes(s, (
        "Appendix C — 사용자 요청에 따라 다양한 sea state / N_components 조합의 대표 case 모음.\n\n"
        "(1) Figure 구성 (paper/figures/fig8_qualitative.pdf 기반):\n"
        "  - 행 마다 하나의 episode (예: BF3 2-comp uniform, BF4 15-comp uniform, BF5 30-comp uniform).\n"
        "  - 열 마다 GT / WaveSplat reconstruction / ETKF reconstruction / GPR reconstruction.\n"
        "  - 각 panel 의 nRMSE annotation 포함 — 정량 비교 가능.\n"
        "  - 색상: blue (음의 변위) → white (수면) → red (양의 변위).\n\n"
        "(2) 관찰 패턴:\n"
        "  - 단순 wave (BF3 2-comp): WaveSplat 압도적 — nRMSE 0.05 수준. GT 와 거의 구별 안 됨.\n"
        "  - 중간 (BF4 15-comp): 모든 method 가 main wave 패턴은 잘 잡지만 미세 wave 에서 차이.\n"
        "  - 복잡 (BF5 30-comp): WaveSplat 도 N_components 추정에 어려움, ETKF 도 비슷한 어려움.\n\n"
        "(3) Q&A 시 활용:\n"
        "  - 청자가 'WaveSplat 이 모든 sea state 에서 우위인가?' 묻으면 → 이 figure 보여주며 'BF3 단순 wave 는 압도적, BF5 복잡 wave 는 비슷' 정직 disclosure.\n"
        "  - 'WaveSplat 이 정말 격자 외 영역도 잘 표현하나?' → 격자 가장자리 영역의 wave 패턴이 자연스럽게 이어지는 것을 시각적으로 보여줌.\n\n"
        "(4) 추가 figure 생성 옵션:\n"
        "  - 새 case 들 (예: BF6 30-comp, slope sensor, bimodal directional) 이 필요하면 scripts/generate_qualitative_figure.py 수정 후 재생성.\n"
        "  - 발표 시간이 부족하면 이 슬라이드는 skip 가능 (Appendix 라 Q&A 시만 참조)."
    ))


# ─────────────────────────────────────────
# Build all
# ─────────────────────────────────────────
def main():
    # New structure: 18 main slides
    # S1 Title, S2 Motivation, S3 Problem, S4 Spoiler (NEW),
    # S5 Categories (was S4), S6 Wang family, S7 PFT-PINN (was S8),
    # S8 App matrix (was S10), S9 WaveSim (was S11), S10 Dataset (was S12),
    # S11 Algorithm (was S13), S12 Results (was S14), S13 Scatter (NEW),
    # S14 Ablation (was S15), S15 Framing (was S16), S16 Mechanism (was S17),
    # S17 Limitations (was S18), S18 Timeline (was S19)
    # Removed from main: S5(Desmars), S7(FNO), S9(VOF-EnKF) → backup A8/A9/A10
    builders = [
        build_s1, build_s2, build_s3,
        build_s4_spoiler,       # NEW: spoiler/hook
        build_s5_categories,    # was build_s4
        build_s6,               # Wang family (kept, num=6)
        build_s8,               # PFT-PINN (num changed to 7)
        build_s10,              # App matrix (num=8)
        build_s11,              # WaveSim (num=9)
        build_s12,              # Dataset (num=10)
        build_s13,              # Algorithm overview (num=11)
        build_s13b_init,        # Init detail (num=11b)
        build_s13c_track,       # Track detail (num=11c)
        build_s13d_visualization,  # NEW: GT vs WaveSplat 시각화 (num=11d)
        build_s14,              # Results (num=12)
        build_s13_scatter,      # episode scatter (num=13)
        build_s15,              # Ablation (num=14)
        build_s16,              # Framing (num=15)
        build_s17,              # Mechanism (num=16)
        build_s18,              # Limitations (num=17)
        build_s19,              # Timeline (num=18)
    ]
    n_main = len(builders)
    for b in builders:
        b()
    build_backups()

    n_backup = len(prs.slides) - n_main
    os.makedirs(DOC_DIR, exist_ok=True)
    prs.save(OUT_PATH)
    print(f"saved: {OUT_PATH}")
    print(f"slides: {len(prs.slides)} ({n_main} main + {n_backup} backup)")


if __name__ == '__main__':
    main()
