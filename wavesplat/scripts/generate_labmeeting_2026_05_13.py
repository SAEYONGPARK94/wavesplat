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
    add_section_tag(s, "Part 2 · WaveSim — LiDAR 모사 (1/2: 어떻게)", color=ACCENT2)
    add_title(s, "WaveSim — LiDAR 센서를 어떻게 시뮬레이션했나")
    add_subtitle(s, "Wave 수면 model + ray-mesh intersection + vessel motion 보정")
    bullets = [
        "**수면 model (간략)**:",
        "  η(x, y, t) = Σ_n A_n cos(k_xn·x + k_yn·y − ω_n·t + φ_n)",
        "  N 개 plane wave 의 선형 중첩 (Linear Wave Theory, LWT)",
        "  Dispersion: ω_n = √(g·|k_n|·tanh(|k_n|·d))",
        "",
        "**LiDAR 센서 모사 — 단계별 처리**:",
        "",
        "**1) Ray emit (광선 발사)**:",
        "  vessel 위치에서 모든 (azimuth × elevation) 방향으로 laser ray 발사",
        "  실제 LiDAR (예: Velodyne, Ouster) 의 angular grid 와 동일한 패턴",
        "",
        "**2) Wave mesh 생성**:",
        "  η(x, y, t) 를 매 frame 격자 (50×50) 로 sampling 후 triangulation",
        "  → 시간 t 에서의 wave 표면 = triangle 들",
        "",
        "**3) Ray-triangle intersection** (Möller-Trumbore 1997):",
        "  각 ray 가 모든 triangle 과 교차하는지 계산 → first-hit point 반환",
        "  Möller-Trumbore: graphics 분야 표준 알고리즘, 수치적 안정 + 빠름",
        "  → first-hit point 까지 거리 = LiDAR 측정값",
        "",
        "**4) Vessel motion 보정**:",
        "  선박 V=11.3 m/s 이동 (매 frame Δt=0.1 s 동안 1.13 m 전진)",
        "  vessel-frame 측정 → world-frame 변환 (좌표계 변환)",
    ]
    add_bullets(s, 0.5, 1.85, 8.5, 5.3, bullets, font_size=11,
                line_space_after=2)
    fig_lidar = os.path.join(DOC_DIR, 'wavesim_BF4_5c_wave_lidar.png')
    add_image_or_placeholder(s, fig_lidar, 9.2, 2.2, 3.9,
                             placeholder_text='[LiDAR scan visualization]')
    add_page_number(s, 9)
    add_notes(s, (
        "WaveSim 의 LiDAR 센서 모사 부분을 단계별로 설명드립니다. "
        "다음 슬라이드에서 raydrop 과 shadowing 의 합당성을 다루겠습니다.\n\n"
        "(1) Ray emit — vessel 위치 (선박의 LiDAR 장착 점) 에서 laser ray 발사. "
        "실제 LiDAR (Velodyne VLP-16, Ouster OS1-64 등) 는 azimuth (수평 방위각) × elevation (수직 각도) 의 fixed angular grid 로 ray 를 발사. "
        "예: 수평 0–360°, 수직 −15° ~ +15°, 1024 azimuth × 16 elevation = 16,384 ray / scan. "
        "우리는 이 angular grid 를 시뮬레이션 입력으로 받음 — 실제 sensor spec 그대로 적용 가능.\n\n"
        "(2) Wave mesh 생성 — wave 표면 η(x, y, t) 를 격자 (50×50 또는 더 dense) 로 sampling 후 두 triangle/cell 로 triangulation. "
        "결과: 시간 t 에서의 wave 표면 = 약 4900 개 triangle 의 mesh.\n\n"
        "(3) Ray-triangle intersection — 각 ray 가 발사된 후 모든 triangle 과 교차 검사. "
        "Möller & Trumbore 1997 알고리즘 사용 — 1997 년 Journal of Graphics Tools 에 발표된 표준 ray-triangle intersection 알고리즘. "
        "특징: barycentric 좌표 직접 계산, 수치적 안정, preprocessing 불필요, 매우 빠름. "
        "이 알고리즘은 그래픽 (ray tracing renderer), 시뮬레이션, 물리 엔진 분야에서 30 년 동안 표준으로 사용. "
        "각 ray 의 first-hit point (가장 가까운 intersection) 가 LiDAR 의 측정값.\n\n"
        "(4) Vessel motion 보정 — 선박이 frame 사이 (Δt = 0.1 s) 에 1.13 m 전진. "
        "각 ray 의 emit 위치는 vessel 의 frame 별 위치 (world frame). "
        "측정된 hit point 도 world frame 으로 표현. "
        "실제 선박에서는 IMU + GPS 로 vessel pose (위치 + orientation) 측정 → 좌표계 변환의 표준 절차.\n\n"
        "이 4 단계가 LiDAR scan 한 번의 시뮬레이션 절차입니다. "
        "다음 슬라이드에서 raydrop (점이 누락되는 현상) 과 shadowing (가려짐) 의 모사가 왜 합당한지 설명드립니다."
    ))


# S12 — WaveSim LiDAR 모사 (합당성 + 검증)
def build_s12():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 2 · WaveSim — LiDAR 모사 (2/2: 합당성)", color=ACCENT2)
    add_title(s, "Raydrop, shadowing — 합당성과 검증")
    add_subtitle(s, "Fresnel 반사 + 기하 차폐 — 광학·기하학의 표준; 108 unit test + 110 episodes 로 검증")
    bullets = [
        "**Raydrop — 점이 왜 누락되는가**:",
        "  실제 LiDAR: 파도 표면이 specular (반사가 한 방향으로 집중) 이면 ray 가 LiDAR 가 아닌 다른 방향으로 반사 → return signal 없음",
        "  입사각이 표면 normal 에서 멀어질수록 specular 비중 증가 → raydrop 확률 ↑",
        "  **모사**: Fresnel reflectance R(θ_i) = ((n₁cos θ_i − n₂cos θ_t) / (n₁cos θ_i + n₂cos θ_t))² 로 specular 확률 계산",
        "  **합당성**: Fresnel 식은 air-water 광학 경계의 표준 physics — 광학 교과서 표준, ocean LiDAR (Nouguier 2014) 에서 검증",
        "",
        "**Geometric shadowing**:",
        "  실제 LiDAR: 앞쪽 큰 파도 (높은 crest) 가 뒤쪽 파도를 가림 → 가려진 영역 점 누락",
        "  **모사**: ray-triangle intersection 의 first-hit 결과로 자동 처리",
        "  **합당성**: ray tracing 의 geometric optics 결과, 추가 가정 없음 (광선 직진 + 첫 충돌점만 측정)",
        "",
        "**검증 (validation)**:",
        "  108 unit test — ray-triangle 정확성, Fresnel 계산, shadowing edge case, vessel motion 모두 검증 통과",
        "  110 episodes 합성 dataset — BF2–BF7 × N=1–30 × directional spreading 으로 다양성 확보",
        "  Sea trial 실측 LiDAR 데이터와 시각적 비교 (sea_trial_lidar.png — 분포 패턴 일치)",
        "",
        "**한계 (정직 인정)**:",
        "  미세파 (capillary, mm 단위), foam, breaking wave 영향 미반영 — operational sea state (BF2–BF6) 에서는 부차적",
    ]
    add_bullets(s, 0.4, 1.85, 12.5, 5.5, bullets, font_size=11,
                line_space_after=2)
    add_page_number(s, 10)
    add_notes(s, (
        "이전 슬라이드에서 LiDAR 모사 단계 (ray emit → mesh → intersection → vessel motion 보정) 를 설명드렸습니다. "
        "이 슬라이드는 두 가지 핵심 효과 — raydrop 과 shadowing — 의 모사가 왜 합당한지 자세히 설명드립니다.\n\n"
        "(1) Raydrop 의 합당성:\n"
        "  실제 LiDAR 에서 점이 누락되는 가장 큰 원인은 specular reflection (일방향 반사) 입니다. "
        "물 표면은 매끄러우면 specular, 거칠면 diffuse (다방향 산란) 가 됩니다. "
        "Specular 비중이 높을 때, ray 가 LiDAR receiver 가 아닌 다른 방향으로 모두 반사되면 return signal 이 없어 점이 누락됩니다. "
        "  - 입사각 (incidence angle, θ_i) 이 표면 normal 에 가까우면 일부 ray 가 LiDAR 로 돌아옴 (점 검출).\n"
        "  - θ_i 가 표면 normal 에서 멀어질수록 specular 가 강해지고 LiDAR 로 안 돌아옴 (점 누락).\n"
        "이 specular 비중을 Fresnel reflectance 식 R(θ_i) 로 계산. "
        "Fresnel 식 자체는 air-water 광학 경계의 표준 physics — 대학 광학 교과서 (예: Hecht, Optics) 에 표준으로 등장. "
        "Air refractive index n₁ ≈ 1, water n₂ ≈ 1.33 가 우리 simulator 의 default 값. "
        "이 식은 ocean LiDAR 분야 (Nouguier-Grilli-Guérin 2014, IEEE TGRS) 에서도 동일하게 사용 — 검증된 표준.\n\n"
        "(2) Geometric shadowing 의 합당성:\n"
        "  실제 LiDAR 에서 앞쪽 큰 파도가 뒤쪽 파도를 가리는 현상 (shadowing). "
        "우리 simulator 는 이걸 별도 modeling 안 함 — ray-triangle intersection 의 first-hit (가장 가까운 intersection) 만 반환. "
        "결과적으로 앞쪽 triangle 이 뒤쪽 triangle 의 ray path 를 자연스럽게 차단. "
        "이건 geometric optics (광선 직진 + 첫 충돌점 측정) 의 자연스러운 결과 — 추가 가정 0. "
        "Möller-Trumbore 알고리즘이 first-hit 만 반환하므로 코드 수준에서도 자동으로 처리됨.\n\n"
        "(3) 검증 (validation):\n"
        "  - 108 unit test 통과 — ray-triangle intersection, Fresnel 계산, shadowing edge case (예: 거의 수직 입사), vessel motion 모두 별도 검증.\n"
        "  - 110 episode 합성 dataset — BF2 (아주 잔잔) ~ BF7 (매우 거친), N_components 1–30, directional spreading 다양화. "
        "각 episode 30 초 / 0.1 초 간격 / 50×50 ground-truth grid + LiDAR scan ~50 점/frame. HDF5 format.\n"
        "  - Sea trial 실측 LiDAR 데이터 (sea_trial_lidar.png 참조) 와 시각적 비교. "
        "Synthetic LiDAR scan 의 점 분포 (raydrop 패턴, 거리에 따른 density 변화) 가 실측과 정성적으로 일치.\n\n"
        "(4) 정직한 한계:\n"
        "  - 미세파 (capillary wave, mm 단위 surface roughness): 실제 LiDAR raydrop 에 추가 영향. 우리는 미반영. "
        "Operational sea state (BF2–BF6) 에서는 dominant 영향 아님.\n"
        "  - Foam, white cap (BF6 이상): wave breaking 시 발생, 우리 LWT family 가 처리 못 하는 비선형 현상.\n"
        "  - Breaking wave: VOF-EnKF 영역, 우리는 future work 로.\n\n"
        "이 정도의 sensor model 을 simulate 한 prior art 는 12 편 정독 결과 0 편입니다. "
        "Sensor 물리까지 포함된 simulator + 110 episode 합성 dataset 자체가 paper 의 contribution 일부입니다."
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
    add_page_number(s, 11)
    add_notes(s, (
        "WaveSplat의 전체 구조를 한 장에 보여주는 슬라이드입니다.\n\n"
        "핵심 아이디어를 먼저 설명합니다:\n\n"
        "기존 방법(Wang2025, Desmars 등)은 파도를 NxN 격자 위의 높이값으로 표현합니다.\n"
        "격자점이 4800개(80×60)면 파라미터도 4800개 — 관측점 50개로 추정하기엔 너무 많습니다.\n\n"
        "WaveSplat은 완전히 다른 접근:\n"
        "  파도 = '소수의 사인파 합' → 각 성분을 5개 물리량으로 표현\n"
        "  N=5개 성분이면 파라미터 25개 → 관측점 50개로 충분히 추정 가능\n\n"
        "분산 관계(ω = √(g|k|tanh(|k|d)))를 hard constraint로 적용하면\n"
        "ω를 k에서 자동 계산하므로 자유도가 5N → 4N으로 줄어듭니다.\n\n"
        "다음 두 슬라이드에서 Init(초기화)과 Track(추적) 각각을 자세히 설명합니다."
    ))


# S13b — Beamforming Init detail (NEW)
def build_s13b_init():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 3 · WaveSplat — 초기화 단계 (Init)", color=GOLD)
    add_title(s, "Beamforming 초기화: 파수(k)를 정확히 결정하는 과정")
    add_subtitle(s, "여러 프레임의 LiDAR 점군을 모아서 파도의 '공간 패턴'을 추정합니다")
    bullets = [
        "**왜 초기화가 중요한가?**",
        "  한 프레임의 LiDAR 50점 → 커버 영역 ~5-8m → 파장 50-300m 파도 패턴 파악 불가",
        "  여러 프레임을 모으면 (선박 이동) → 관측 영역이 수십m로 확장",
        "",
        "**v2_konly Beamforming (2단계 파수 추정)**:",
        "  ① Coarse: 후보 (kx, ky) 격자 생성 → 각 후보의 관측 상관 계산",
        "     수천 개 후보 중 에너지가 높은 peak N개를 초기 파수로 선택",
        "  ② Fine: 선택된 peak 주변에서 정밀 탐색 (sub-grid resolution)",
        "     최종 (kx, ky) 결정 → 분산 관계로 ω 자동 계산",
        "",
        "**'konly'의 의미**: k만 초기화하고 ω, A, φ는 최적화에 맡김",
        "  이전 v1(FFT 기반) 대비 **평균 18% 개선** (특히 방향 추정 정확도↑)",
        "",
        "**Synthetic Aperture 효과**:",
        "  선박 속도 V = 11.3 m/s → 1초간 11.3m 이동",
        "  10 프레임 누적 시 유효 개구(aperture) = 11.3m",
        "  파수 분해능 Δk ≈ 2π / aperture ≈ 0.56 rad/m",
    ]
    add_bullets(s, 0.6, 1.85, 12.0, 5.3, bullets, font_size=12,
                line_space_after=2)
    add_page_number(s, 12)
    add_notes(s, (
        "초기화 단계를 상세히 설명합니다.\n\n"
        "왜 초기화가 어려운가?\n"
        "  LiDAR 한 프레임에 50개 점이 있지만, 이 점들이 커버하는 영역은 약 5-8m입니다.\n"
        "  그런데 우리가 찾으려는 파도의 파장은 50-300m입니다.\n"
        "  비유하면: 코끼리의 다리 일부만 만져보고 코끼리 전체 모양을 맞추는 것과 같습니다.\n\n"
        "해결책: Synthetic Aperture (합성 개구)\n"
        "  선박이 이동하면서 여러 시점의 데이터를 모읍니다.\n"
        "  각 시점의 50개 점이 모이면 수백 개의 점이 넓은 영역에 분포합니다.\n"
        "  이제 파도 패턴을 파악할 수 있는 충분한 정보가 됩니다.\n\n"
        "Beamforming 작동 원리:\n"
        "  '만약 파도가 이 방향(kx, ky)으로 온다면, 관측 데이터와 얼마나 잘 맞는가?'\n"
        "  모든 후보 방향에 대해 이 질문을 던지고, 가장 잘 맞는 방향을 선택합니다.\n"
        "  레이더/음향 분야에서 신호 방향을 찾는 것과 같은 원리입니다.\n\n"
        "v1(FFT) → v2(beamforming) 변경 이유:\n"
        "  FFT는 균일 격자 위의 데이터가 필요하지만, LiDAR 점은 불규칙합니다.\n"
        "  Beamforming은 불규칙 점에서도 작동합니다.\n"
        "  이 변경으로 평균 18% 정확도 개선 (2026-04 확인)."
    ))


# S13c — Tracking + Adaptive N detail (NEW)
def build_s13c_track():
    s = add_blank(WHITE)
    add_section_tag(s, "Part 3 · WaveSplat — 추적 단계 (Track)", color=GOLD)
    add_title(s, "실시간 추적: 매 프레임 13ms로 파도 상태 업데이트")
    add_subtitle(s, "파수(k)는 고정하고, 진폭·위상만 빠르게 최적화합니다")
    # Left column: Tracking
    bullets_left = [
        "**w_phi Tracking Mode**:",
        "  k_x, k_y → 고정 (Init에서 결정된 값 유지)",
        "  ω → k에서 자동 계산 (분산 관계)",
        "  A, φ → Adam optimizer로 매 프레임 update",
        "",
        "**왜 k를 고정하는가?**",
        "  50점으로 5N개 변수를 모두 최적화하면 과적합(overfitting)",
        "  k 고정 시 자유도 = 2N (A와 φ만) → 안정적 추정",
        "  이 설계가 tracking 성능을 결정적으로 개선 (04-06 발견)",
        "",
        "**Loss Function**:",
        "  L = Σ (η_model(xi, yi, t) − η_obs(xi, yi, t))²",
        "  관측 50점에서의 모델-관측 높이 차이 최소화",
        "  Adam: lr=0.01, 50 iterations/frame → ~13ms",
    ]
    add_bullets(s, 0.5, 1.85, 6.5, 5.3, bullets_left, font_size=11,
                line_space_after=2)
    # Right column: Adaptive N
    bullets_right = [
        "**Adaptive N (동적 성분 수)**:",
        "",
        "① Pruning (가지치기):",
        "  |A_n| < threshold → 제거",
        "  에너지 기여도 낮은 성분 정리",
        "",
        "② Growth (성장):",
        "  residual > growth_threshold",
        "  → 잔여 오차에서 새 성분 추출",
        "  → beamforming으로 (kx, ky) 결정",
        "",
        "③ Safety Rollback (안전 복귀):",
        "  RMSE_new > 1.3 × RMSE_old",
        "  → 이전 상태로 즉시 복구",
        "  → 잘못된 성분 추가 방지",
    ]
    add_bullets(s, 7.2, 1.85, 5.8, 5.3, bullets_right, font_size=11,
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
    # ── A1: Desmars 2018 ──
    build_backup("A1",
                 "Desmars 2018 (OMAE) — LIDAR + ICWM",
                 [
                     "**내용**: 선박 탑재 scanning LIDAR로 파도면 측정 → phase-resolved 파도장 추정",
                     "**방법론**: Improved Choppy Wave Model (ICWM = LWT + 1차 nonlinear 보정)",
                     "  predefined k-grid 위에서 variational cost function 최소화",
                     "  관측-모델 mismatch를 최소화하는 spectral coefficients 탐색",
                     "**결과**: 합성 데이터 (HOS simulator) 에서 개념 검증 성공",
                     "**Contribution**: 선박 탑재 LIDAR + variational wave inversion 개념을 최초 제안",
                     "  이후 Desmars 2020, 2023으로 이어지는 연구 시리즈의 출발점",
                     "**Limitation**: 1D 종방향만 처리, 격자 기반 (해상도 한계), simulator 검증만 (실측 없음)",
                 ],
                 "Desmars 2018 OMAE 상세:\n\n"
                 "이 논문은 선박에 scanning LIDAR를 장착하여 파도를 측정하고, "
                 "그 데이터를 역변환(inversion)하여 phase-resolved 파도장을 재구성하는 개념을 처음 제안했습니다.\n\n"
                 "ICWM (Improved Choppy Wave Model)은 선형파 이론(LWT)에 1차 비선형 보정을 추가한 모델로, "
                 "Lagrangian 좌표계에서 파도면 변위를 기술합니다. 기존 LWT보다 가파른 파도를 더 잘 표현합니다.\n\n"
                 "방법론: k-space를 미리 정의된 격자로 이산화하고, 각 격자점의 spectral coefficient를 "
                 "variational optimization으로 추정합니다. 관측된 파도면 높이와 모델 예측의 차이(mismatch)를 "
                 "cost function으로 정의하고 이를 최소화합니다.\n\n"
                 "한계: (1) 1D만 처리 (종방향 단면), (2) 격자 해상도가 정확도의 상한을 결정, "
                 "(3) 실험 검증 없이 HOS synthetic data로만 검증.")
    # ── A2: Desmars 2020 ──
    build_backup("A2",
                 "Desmars 2020 (Ocean Eng 212) — Wave tank 실험 검증",
                 [
                     "**내용**: phase-resolved 파도 추정의 첫 실험실 검증 (wave tank)",
                     "**방법론**: wave gauge array (7개 gauge, 간격 ~0.5m) 로 표면 높이 측정",
                     "  spectral decomposition + dispersion correction 후처리",
                     "  gauge 간 위상차를 이용하여 파수(k) 추정 → 분산관계로 보정",
                     "**결과**: SSP ≈ 0.05~0.15 (sea state 의존), gauge 수 증가 → 정확도 향상",
                     "**Contribution**: simulator가 아닌 실제 wave tank에서 검증한 최초 사례",
                     "  dispersion correction이 accuracy 개선에 효과적임을 실증",
                     "**Limitation**: 1D 종방향, wave gauge ≠ LiDAR (다른 sensor modality)",
                     "  고정 설치 sensor (선박 탑재 아님), 실해역 검증 아님 (wave tank 제한)",
                 ],
                 "Desmars 2020 Ocean Engineering 상세:\n\n"
                 "이전 Desmars 2018이 simulator에서만 검증했다면, 이 논문은 실제 wave tank에서 "
                 "phase-resolved 파도 추정의 정확도를 검증한 첫 논문입니다.\n\n"
                 "7개의 wave gauge를 일렬로 배치하여 해수면 높이를 동시 측정합니다. "
                 "gauge 간의 위상차를 분석하여 각 주파수 성분의 파수(k)를 추정하고, "
                 "dispersion relation을 사용하여 보정합니다.\n\n"
                 "dispersion correction: 측정된 k와 이론적 k(ω)의 차이를 보정하는 후처리. "
                 "이를 통해 propagation 예측의 정확도가 크게 개선됩니다.\n\n"
                 "실험 결과 SSP(Surface Similarity Parameter) 기준 0.05~0.15 정도의 정확도를 달성했으며, "
                 "gauge 수가 많을수록 정확도가 향상됨을 보였습니다.")
    # ── A3: Klein 2020 ──
    build_backup("A3",
                 "Klein 2020 (Fluids 5(1), 9) — 파도 모델 4종 정량 비교",
                 [
                     "**내용**: 4개 wave model의 정확도를 wave tank에서 정량 비교",
                     "  LWT (선형파), NLSE (비선형 Schrödinger), NLSE₂ (2차 NLSE), HOSM (High-Order Spectral)",
                     "**방법론**: 초기 조건 고정 → 각 모델로 propagation → wave gauge 관측과 비교",
                     "  SSP (Surface Similarity Parameter) 를 표준 정량 지표로 사용",
                     "**결과**: 정확도 순서 HOSM > NLSE₂ > NLSE > LWT",
                     "  LWT는 ε < 0.07 (wave steepness) 에서만 SSP < 0.1 달성",
                     "  HOSM은 ε = 0.10 이상에서도 높은 정확도 유지",
                     "**Contribution**: SSP metric을 wave reconstruction 분야에서 표준화",
                     "  각 모델의 유효 범위(operational envelope)를 정량적으로 규명",
                     "**Limitation**: 1D wave tank 실험만, 2D 방향성 파도 미고려",
                     "  reconstruction 자체가 아닌 propagation model 비교에 국한",
                 ],
                 "Klein 2020 Fluids 상세:\n\n"
                 "이 논문은 파도 물리 모델 4종을 동일 조건에서 비교한 benchmark study입니다.\n\n"
                 "4가지 모델:\n"
                 "1. LWT (Linear Wave Theory): 가장 단순, 선형 중첩\n"
                 "2. NLSE (Nonlinear Schrödinger Equation): 좁은 bandwidth 비선형 효과\n"
                 "3. NLSE₂: NLSE의 2차 확장\n"
                 "4. HOSM (High-Order Spectral Method): 가장 정확, M차 비선형\n\n"
                 "SSP (Surface Similarity Parameter) 정의:\n"
                 "  두 파도면의 Fourier 스펙트럼 차이를 정규화한 것. SSP=0이면 완벽 일치, SSP=1이면 완전 불일치.\n"
                 "  위상, 진폭, 주파수 오류를 모두 포함하는 종합 지표.\n\n"
                 "핵심 발견: LWT는 wave steepness ε < 0.07에서만 SSP < 0.1을 달성합니다.\n"
                 "이는 선형파 이론의 operational envelope을 정량적으로 규명한 중요한 결과입니다.")
    # ── A4: Hunt et al. 2007 ──
    build_backup("A4",
                 "Hunt et al. 2007 (Tellus A 59) — LETKF",
                 [
                     "**내용**: Local Ensemble Transform Kalman Filter 제안 — 대규모 DA의 핵심 알고리즘",
                     "**방법론**: 앙상블 멤버를 local subspace로 변환(transform) 후 update",
                     "  관측 주변 local domain에서만 분석 → 고차원 상태공간 처리 가능",
                     "  배경 오차 공분산을 앙상블로부터 추정 (명시적 공분산 행렬 불필요)",
                     "**결과**: 기상학 Lorenz-96 모델에서 기존 EnKF 대비 안정적 성능",
                     "  localization으로 spurious correlation 제거 → 수렴 안정성 확보",
                     "**Contribution**: EnKF의 localization 문제를 체계적으로 해결",
                     "  이후 기상, 해양, CFD 등 다양한 DA 분야의 기반 알고리즘이 됨",
                     "**Limitation**: 앙상블 크기(Ne)에 따라 계산 비용 선형 증가",
                     "  비선형성이 강한 시스템에서는 Gaussian 가정이 깨질 수 있음",
                 ],
                 "Hunt et al. 2007 Tellus A 상세:\n\n"
                 "LETKF는 Data Assimilation의 핵심 알고리즘 중 하나입니다.\n\n"
                 "기본 아이디어:\n"
                 "1. Ne개의 앙상블 멤버로 상태공간의 불확실성을 표현\n"
                 "2. 관측이 들어오면, 관측 주변의 local domain에서만 분석(analysis) 수행\n"
                 "3. 앙상블 멤버를 관측 공간으로 변환(transform)하여 update weights 계산\n"
                 "4. 새로운 앙상블 = 배경 앙상블 + weighted correction\n\n"
                 "localization이 중요한 이유:\n"
                 "  앙상블 크기가 유한하면, 멀리 떨어진 격자점 간에 물리적으로 무의미한 상관관계(spurious correlation)가 "
                 "  나타납니다. local domain으로 제한하면 이를 방지할 수 있습니다.\n\n"
                 "이 알고리즘이 파도 추정에 적용된 사례: Wang & Pan 2022/2025 (EnKF-HOS), "
                 "Yan et al. 2025 (VOF-EnKF) 등.")
    # ── A5: Tang et al. 2013 ──
    build_backup("A5",
                 "Tang et al. 2013 (IEEE TIT) — Atomic Norm Compressed Sensing",
                 [
                     "**내용**: 격자 없이(off-grid) 연속 파라미터 공간에서 sparse signal recovery",
                     "**방법론**: atomic norm minimization via Semidefinite Programming (SDP)",
                     "  atoms = continuous sinusoids, atomic norm = ℓ₁의 연속 확장",
                     "  Vandermonde decomposition으로 SDP를 효율적으로 풀 수 있음",
                     "**결과**: M = O(K log K) 관측으로 K개 주파수 성분의 exact recovery 보장",
                     "  격자 기반 CS 대비 basis mismatch 문제 완전 해결",
                     "**Contribution**: off-grid compressed sensing의 이론적 토대 확립",
                     "  spectral super-resolution의 수학적 보장 최초 제공",
                     "**Limitation**: 1D 주파수 추정에 한정, 2D 확장은 NP-hard에 가까움",
                     "  SDP solver 계산 비용이 커서 real-time 적용 어려움",
                 ],
                 "Tang et al. 2013 IEEE TIT 상세:\n\n"
                 "기존 compressed sensing (CS)의 한계:\n"
                 "  CS는 신호가 '어떤 basis 위에서 sparse'하다고 가정합니다.\n"
                 "  하지만 실제 주파수는 격자점 위에 정확히 놓이지 않습니다 (basis mismatch).\n"
                 "  격자를 세밀하게 만들면 계산 비용이 폭증하고, coherence 조건이 악화됩니다.\n\n"
                 "atomic norm의 아이디어:\n"
                 "  ℓ₁ norm은 '이산 벡터의 sparsity'를 측정합니다.\n"
                 "  atomic norm은 '연속 원자(atom) 집합 위의 sparsity'를 측정합니다.\n"
                 "  파도 문제에서 atom = 특정 주파수의 sinusoid입니다.\n\n"
                 "이론적 보장: K개의 주파수 성분이 최소 간격 조건을 만족하면,\n"
                 "M = O(K log K) 개의 무작위 관측만으로 exact recovery가 가능합니다.\n\n"
                 "한계: 2D 이상으로 확장하면 SDP 크기가 급격히 커지며, "
                 "real-time 적용에는 속도가 부족합니다.")
    # ── A6: Möller-Trumbore 1997 ──
    build_backup("A6",
                 "Möller-Trumbore 1997 (JGT) — Ray-Triangle Intersection",
                 [
                     "**내용**: 3D ray와 triangle의 교차점을 계산하는 고속 알고리즘",
                     "**방법론**: barycentric coordinates를 직접 계산 (Cramér's rule)",
                     "  ray: O + tD,  triangle: V₀, V₁, V₂",
                     "  교차 조건: O + tD = (1-u-v)V₀ + uV₁ + vV₂  →  [−D, E₁, E₂][t,u,v] = T",
                     "  t > 0, u ≥ 0, v ≥ 0, u+v ≤ 1 이면 교차",
                     "**결과**: 기존 방법 대비 ~20% 빠른 연산 (전처리 불필요)",
                     "**Contribution**: 컴퓨터 그래픽스의 de facto standard ray-triangle 알고리즘",
                     "  GPU ray tracing, physics simulation 등 광범위하게 사용됨",
                     "**Limitation**: 단일 삼각형 기준이므로, 대규모 mesh에는 BVH 등 가속 구조 필요",
                 ],
                 "Möller-Trumbore 1997 상세:\n\n"
                 "ray-triangle intersection은 ray tracing의 가장 기본적인 연산입니다.\n\n"
                 "알고리즘 핵심:\n"
                 "1. ray를 O + tD (원점 + t × 방향)로 표현\n"
                 "2. triangle을 barycentric coordinates (u, v)로 표현\n"
                 "3. 연립방정식을 Cramér's rule로 풀어 t, u, v를 한번에 계산\n"
                 "4. early rejection: determinant이 0에 가까우면 평행, u나 v가 범위 밖이면 빗나감\n\n"
                 "이 알고리즘이 중요한 이유:\n"
                 "  전처리(pre-stored edge 등)가 불필요하여 메모리 효율적이고,\n"
                 "  branch 조건이 적어 SIMD/GPU에서도 효율적입니다.\n"
                 "  1997년 이후 거의 모든 ray tracing 구현에서 사용됩니다.")
    # ── A7: PFT-PINN (Ehlers 2025) ──
    build_backup("A7",
                 "Ehlers 2025 (Ocean Eng) — PFT-PINN",
                 [
                     "**내용**: Physics-Informed NN으로 radar 데이터에서 파도면 + velocity potential 동시 추정",
                     "**방법론**: 2개 병렬 NN — η-net(표면 높이), Φ-net(속도 포텐셜)",
                     "  Loss = data fidelity + Laplace(∇²Φ=0) + KFSBC + DFSBC + bottom BC",
                     "  ReLoBRaLo: 각 loss term의 gradient magnitude에 따라 가중치 자동 조절",
                     "  학습 = 관측 데이터 1개에 대해 수만 iteration Adam optimization",
                     "**결과**: nL2 ≈ 0.08~0.15 (1D, 합성 radar 데이터 기준)",
                     "  U-Net/FNO 대비 shadowed area 재구성 quality 우수",
                     "  **30분 소요** (RTX 3090 기준) — supervised ML의 ms 단위와 대조적",
                     "**Contribution**: PINN으로 wave reconstruction을 푼 최초 시도",
                     "  unsupervised (학습 데이터 불필요) + 물리 법칙 hard constraint",
                     "  파도 물리 5개 경계조건을 loss로 직접 반영",
                     "**Limitation**: 1D만 처리, ~30분 소요 (real-time 불가)",
                     "  각 관측마다 처음부터 학습해야 함 (transfer 불가)",
                 ],
                 "Ehlers 2025 PFT-PINN 상세:\n\n"
                 "PINN (Physics-Informed Neural Network):\n"
                 "  일반적인 NN은 데이터로부터 학습하지만, PINN은 물리 법칙(PDE)을 loss function에 포함합니다.\n"
                 "  따라서 학습 데이터가 적거나 없어도 물리적으로 일관된 해를 찾을 수 있습니다.\n\n"
                 "2개의 병렬 네트워크:\n"
                 "  η-net: (x, t) → η(x,t) 표면 높이\n"
                 "  Φ-net: (x, z, t) → Φ(x,z,t) 속도 포텐셜\n\n"
                 "5개의 물리 loss term:\n"
                 "  1. Laplace: ∇²Φ = 0 (비압축성 유체)\n"
                 "  2. KFSBC (Kinematic Free Surface BC): ∂η/∂t = ∂Φ/∂z - ∂η/∂x · ∂Φ/∂x at z=η\n"
                 "  3. DFSBC (Dynamic Free Surface BC): ∂Φ/∂t + gη + ½|∇Φ|² = 0 at z=η\n"
                 "  4. Bottom BC: ∂Φ/∂z = 0 at z=-d\n"
                 "  5. Data fidelity: radar 관측과 예측의 차이\n\n"
                 "ReLoBRaLo (Relative Loss Balancing with Random Lookback):\n"
                 "  각 loss term의 gradient magnitude를 추적하여 가중치를 동적으로 조절합니다.\n"
                 "  gradient가 작은 term에 더 큰 가중치를 줘서 균형을 맞춥니다.\n\n"
                 "핵심 한계: 30분 소요는 근본적으로 PINN의 per-instance optimization 때문입니다.\n"
                 "매 관측마다 수만 번의 gradient descent를 수행해야 합니다.")
    # ── A8: Desmars 2023 ──
    build_backup("A8",
                 "Desmars 2023 (JFM 975) — HOS-OCM",
                 [
                     "**내용**: 선박 탑재 센서(LIDAR/카메라)로 1D 파도장 추정 + HOS 기반 예측",
                     "**방법론**: Observation-Conditioned Model (OCM) — 3단계 파이프라인",
                     "  ① spectral decomposition: 관측을 predefined k-grid 위의 계수로 분해",
                     "  ② assimilation window: 여러 시간 프레임의 관측을 cost function으로 결합",
                     "  ③ HOS propagation: 추정된 초기 조건으로 비선형 파도 예측 수행",
                     "  wave model: M=1 (linear) ~ M=3 (3차 비선형) 까지 비교",
                     "**결과**: M=3 HOS에서 SSP ≈ 0.03~0.10 (합성 데이터, sea state 의존)",
                     "  관측 영역 밖 수백m까지 예측 가능 (HOS propagation 덕분)",
                     "**Contribution**: 선박 기반 센서 + phase-resolved 추정 + 비선형 예측을 일관된 프레임워크로 통합",
                     "  assimilation window 길이의 trade-off 분석 (길면 정확 but 초기화 지연)",
                     "**Limitation**: 1D 종방향만, predefined k-grid (해상도 한계)",
                     "  sensor model 없음 (raydrop, shadowing, 빔 확산 등 미고려)",
                     "  실해역 검증 없음 (HOS synthetic data로만 검증)",
                 ],
                 "Desmars 2023 JFM 975 상세:\n\n"
                 "OCM (Observation-Conditioned Model) 3단계:\n\n"
                 "1단계 — spectral decomposition:\n"
                 "  관측된 파도면 높이를 미리 정의된 k-grid 위의 Fourier 계수로 분해합니다.\n"
                 "  k-grid의 범위와 해상도가 추정 가능한 파도 성분의 범위를 결정합니다.\n\n"
                 "2단계 — assimilation window:\n"
                 "  단일 시점이 아닌 여러 시점(T_a 동안)의 관측을 모아서 cost function을 구성합니다.\n"
                 "  시간 정보를 활용하여 주파수 분해능을 높이는 효과가 있습니다.\n"
                 "  T_a가 길면 더 정확하지만, 초기화 지연이 커집니다.\n\n"
                 "3단계 — HOS propagation:\n"
                 "  추정된 초기 스펙트럼으로 HOS (High-Order Spectral Method)를 돌려서\n"
                 "  관측 영역 바깥의 미래 파도를 예측합니다.\n"
                 "  M=3이면 3차까지의 비선형 파도-파도 상호작용을 포함합니다.\n\n"
                 "wave model 비교: M=1(선형), M=2(2차), M=3(3차)을 비교하여\n"
                 "높은 차수가 실질적으로 정확도를 개선하는 조건을 분석했습니다.")
    # ── A9: Ehlers 2023 FNO ──
    build_backup("A9",
                 "Ehlers 2023 (Ocean Eng) — FNO for Wave Reconstruction",
                 [
                     "**내용**: Fourier Neural Operator로 X-band radar 이미지에서 파도면 재구성",
                     "**방법론**: supervised learning — HOS synthetic radar → wave surface 매핑 학습",
                     "  FNO 구조: Fourier layer 3개 (spectral domain에서 global convolution)",
                     "  U-Net과 비교: local (CNN) vs global (Fourier) inductive bias",
                     "  입력: n_s개 연속 radar snapshot (n_r=512 grid points × n_s channels)",
                     "  학습: N_train=2496, N_test=624, Adam optimizer, 800 epochs, RTX 3050 Ti",
                     "**결과**: FNO M_{F,9} (9 snapshots) — nL2=0.153, SSP=0.077",
                     "  U-Net M_{U,10} (10 snapshots) — nL2=0.123, SSP=0.061",
                     "  **Inference: FNO 1.9ms vs U-Net 37ms** (20배 빠름)",
                     "  FNO가 shadowed area 재구성에서 더 균일한 품질 (nL2_shad/nL2_vis 비율 낮음)",
                     "**Contribution**: FNO의 global inductive bias가 radar shadowing 처리에 유리함을 입증",
                     "  1.9ms inference로 real-time wave reconstruction 가능성 제시",
                     "**Limitation**: supervised (3120 sample 학습 필요), 1D만 처리",
                     "  학습 분포 밖의 sea state에서 distribution shift 위험",
                     "  black-box — 추정 결과의 물리적 해석 불가",
                 ],
                 "Ehlers 2023 Ocean Engineering 상세:\n\n"
                 "FNO (Fourier Neural Operator)의 핵심 아이디어:\n"
                 "  CNN은 공간 영역에서 local convolution을 수행하지만,\n"
                 "  FNO는 Fourier 영역에서 global convolution을 수행합니다.\n"
                 "  각 Fourier layer: v_{j+1} = σ(F⁻¹(R_j · F(v_j)) + W_j · v_j)\n"
                 "  R_j는 Fourier 계수를 스케일링하는 학습 가능한 행렬입니다.\n\n"
                 "U-Net vs FNO 비교:\n"
                 "  U-Net: kernel size s_k=5로 local 정보를 처리 → shadowed area에서 정보 부족\n"
                 "  FNO: Fourier 변환으로 전체 domain의 global 정보를 활용 → shadowed area도 보간 가능\n\n"
                 "radar shadowing 문제:\n"
                 "  X-band radar는 높은 파도가 뒤의 낮은 영역을 가리는 shadowing이 발생합니다.\n"
                 "  ε=0.01이면 ~9%, ε=0.10이면 ~70%가 shadowed area입니다.\n"
                 "  FNO는 이 shadowed area를 U-Net보다 더 균일하게 재구성합니다.\n\n"
                 "속도: FNO 1.9ms (custom implementation) vs U-Net 37ms (PyTorch standard).\n"
                 "RTX 3050 Ti Laptop GPU 기준. FNO가 20배 빠른 이유는 Fourier layer의 병렬성 때문입니다.\n\n"
                 "정확도: 전체 평균 nL2 기준으로는 U-Net(0.123)이 FNO(0.153)보다 약간 우수하지만,\n"
                 "shadowed/visible area 간 균일성에서는 FNO가 우수합니다.")
    # ── A10: Yan et al. 2025 VOF-EnKF ──
    build_backup("A10",
                 "Yan et al. 2025 (arXiv) — VOF-EnKF",
                 [
                     "**내용**: Navier-Stokes(VOF) + EnKF로 wave breaking까지 포함한 파도장 동화",
                     "**방법론**: Volume of Fluid (VOF) — OpenFOAM 기반 2상 유체 시뮬레이션",
                     "  50개 앙상블 멤버 각각이 독립적 OpenFOAM CFD 시뮬레이션 실행",
                     "  관측이 들어오면 EnKF로 50개 멤버의 VOF field + velocity field 동시 update",
                     "  POD (Proper Orthogonal Decomposition) 로 고차원 상태를 저차원으로 축소",
                     "**결과**: plunging breaker (쇄파) 재구성 성공 — 물리 최고 난이도",
                     "  파도가 부서지는 과정의 공간 분포를 정확하게 추정",
                     "  비선형 효과가 강한 영역에서 기존 방법(LWT 기반) 대비 큰 우위",
                     "**Contribution**: wave breaking이라는 극한 비선형 현상의 DA를 최초 시도",
                     "  VOF + EnKF 조합으로 자유 표면의 topology 변화(breaking)까지 처리 가능함을 증명",
                     "**Limitation**: offline only — 50개 OpenFOAM × 수백 time step = 수시간~수일 소요",
                     "  1D xz-plane만 처리 (2D 수평면 불가), real-time 적용 근본적으로 불가능",
                     "  POD mode 수 선택에 따라 결과 민감도 존재",
                 ],
                 "Yan et al. 2025 arXiv 상세:\n\n"
                 "VOF (Volume of Fluid):\n"
                 "  각 격자 셀에 'fluid fraction' α ∈ [0,1]을 저장합니다.\n"
                 "  α=1이면 물, α=0이면 공기, 0<α<1이면 자유 표면입니다.\n"
                 "  Navier-Stokes 방정식을 풀면서 α의 시간 변화를 추적합니다.\n"
                 "  이 방법은 파도가 부서지거나 합쳐지는 등 topology 변화를 자연스럽게 처리합니다.\n\n"
                 "EnKF 적용 방법:\n"
                 "  50개 앙상블 멤버 = 50개의 독립적인 OpenFOAM 시뮬레이션.\n"
                 "  각 멤버는 약간 다른 초기/경계 조건으로 시작합니다.\n"
                 "  관측(wave gauge 등)이 들어오면 EnKF가 50개 멤버의 상태(α, u, v, p)를 update합니다.\n\n"
                 "POD (Proper Orthogonal Decomposition):\n"
                 "  VOF field는 수만 개의 격자 셀로 구성되어 직접 EnKF를 적용하면 차원이 너무 높습니다.\n"
                 "  POD로 주요 모드(패턴) 몇 개만 추출하여 저차원 공간에서 update합니다.\n\n"
                 "핵심 성과: plunging breaker — 파도의 꼭대기가 앞으로 넘어가며 부서지는 현상.\n"
                 "이를 DA로 재구성한 것은 이 논문이 최초입니다.\n\n"
                 "근본적 한계: 50개 CFD 시뮬레이션을 병렬로 돌려야 하므로\n"
                 "HPC 자원이 필수적이고, real-time은 원리적으로 불가능합니다.")
    # ── A11: Wang 2025 ──
    build_backup("A11",
                 "Wang & Pan 2025 (Coastal Eng 197) — EnKF-HOS",
                 [
                     "**내용**: stereo camera 관측 기반 2D 파도장 실시간 추정 + HOS 예측",
                     "**방법론**: Ensemble Kalman Filter + HOS (High-Order Spectral) propagation",
                     "  상태 변수: NxN spectral grid 위의 η(x,y) + ∂η/∂t (sub-state 분리)",
                     "  Ne=300 앙상블 멤버, spectral propagation으로 forecast step 수행",
                     "  **Fuzzy R**: 관측 오차 공분산을 앙상블 spread 기반으로 자동 적응",
                     "  **Delaunay interpolation**: 불규칙 관측점을 격자로 매핑",
                     "  **Sub-state localization**: η와 ∂η/∂t를 분리 update하여 안정성 확보",
                     "**결과**: stereo camera 실험에서 SSP ≈ 0.05~0.15 달성",
                     "  3분 prediction horizon까지 유효한 파도 예측",
                     "**Contribution**: 2D 방향성 파도의 실시간 추정을 stereo camera로 구현",
                     "  fuzzy R + sub-state ETKF 조합으로 EnKF의 filter divergence 문제 해결",
                     "**Limitation**: Ne=300 → 계산 비용 높음 (격자 크기에 의존)",
                     "  격자 해상도가 추정 정확도의 상한을 결정 (sub-grid scale 불가)",
                     "  stereo camera 전제 (수천 관측점) — sparse sensor에서는 미검증",
                 ],
                 "Wang & Pan 2025 Coastal Engineering 197 상세:\n\n"
                 "핵심 알고리즘 구조:\n"
                 "1. Forecast: 현재 앙상블의 spectral coefficients를 HOS로 다음 시간까지 propagation\n"
                 "2. Analysis: 새 관측이 들어오면 ETKF로 앙상블 update\n\n"
                 "Fuzzy R (관측 오차 적응):\n"
                 "  기존 EnKF는 관측 오차 R을 고정값으로 설정합니다.\n"
                 "  Wang은 앙상블의 spread(분산)에 따라 R을 동적으로 조절합니다.\n"
                 "  앙상블이 관측과 크게 어긋나면 R을 키워서(관측을 덜 신뢰) filter divergence를 방지합니다.\n\n"
                 "Sub-state ETKF:\n"
                 "  전체 상태 [η, ∂η/∂t]를 한꺼번에 update하면 cross-correlation이 noisy합니다.\n"
                 "  η만 먼저 update하고, ∂η/∂t는 물리 관계를 통해 간접 update합니다.\n\n"
                 "Delaunay interpolation:\n"
                 "  stereo camera의 관측점은 불규칙하게 분포합니다.\n"
                 "  Delaunay 삼각분할 → 격자점으로 보간하여 ETKF에 입력합니다.\n\n"
                 "계산 비용: Ne=300, 80x60 grid일 때 각 time step에서\n"
                 "300개 FFT propagation + ETKF analysis = ~632ms/frame (CPU).")
    # ── A12: WaveSim 110 episodes dataset detail ──
    build_backup("A12",
                 "WaveSim — 110 episodes 합성 dataset",
                 [
                     "**Episode 변수**:",
                     "  Sea state: BF2–BF7 (6 단계 Beaufort scale, 잔잔 → 매우 거친)",
                     "  N_components: 1, 2, 5, 15, 20, 30 (저/중/고 복잡도)",
                     "  Sensor type: uniform / slope (LiDAR scan 분포)",
                     "  Seeds: BF2–BF5 = seed 0–4 (5 개씩), BF6–BF7 = seed 0",
                     "**Episode 당 데이터**:",
                     "  T = 30 s, Δt = 0.1 s → 300 frames",
                     "  Ground-truth η field: 50×50 격자 (참값으로 평가용)",
                     "  LiDAR scan: ~50 valid pts/frame (raydrop 후)",
                     "  HDF5 format, 110 files",
                     "**Reproducibility**: seed 저장 — 동일 seed 로 재실행 시 동일 결과",
                     "**Test coverage**: 53 (wavesim) + 55 (wavesplat) = 108 unit test 모두 통과",
                 ],
                 "WaveSim 합성 dataset 의 detail 입니다.\n\n"
                 "Episode 설계:\n"
                 "  - Sea state: BF2 (잔잔) ~ BF7 (매우 거친) 6 단계. ε (steepness) 는 BF2 ~0.02, BF7 ~0.10.\n"
                 "  - N_components: 1 ~ 30 — 알고리즘이 component 수에 robust 한지 검증.\n"
                 "  - Sensor type 두 가지: uniform (균등 분포), slope (거리 비례 분포 — 실제 LiDAR 의 elevation grid).\n"
                 "  - Seeds: BF2-BF5 는 seed 0-4 (5 개씩, 통계 평가용), BF6-BF7 은 seed 0 (1 개씩, 극한 sea state).\n\n"
                 "Episode 당 데이터 구성:\n"
                 "  - 30 초 시뮬레이션, 0.1 초 간격 → 300 frame.\n"
                 "  - Ground-truth η field 50×50 격자 — 평가 시 참값.\n"
                 "  - LiDAR scan: raydrop / shadowing 적용 후 frame 당 ~50 점.\n\n"
                 "Reproducibility 보장:\n"
                 "  - Seed 저장 — 동일 seed 로 재실행 시 동일 결과.\n"
                 "  - 53 + 55 = 108 unit test — simulator 와 estimator 모두 검증.\n\n"
                 "이 dataset 자체가 paper 의 contribution 일부입니다. "
                 "Ehlers 2023 도 비슷한 합성 benchmark 만들었지만 1D + radar 였고, "
                 "우리는 2D directional + LiDAR sensor model 까지 포함.")


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
        build_s14,              # Results (num=12)
        build_s13_scatter,      # NEW: episode scatter (num=13)
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
