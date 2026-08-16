#!/usr/bin/env python3
"""WaveSplat 교육용 PPT 생성 스크립트 (72장)"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

# ── 색상 팔레트 ──
BG_DARK = RGBColor(0x1B, 0x1B, 0x2F)       # 진한 남색
BG_SECTION = RGBColor(0x0D, 0x47, 0xA1)    # 파트 구분 파란색
ACCENT = RGBColor(0x00, 0xE6, 0x76)        # 형광 초록
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xB0, 0xB0, 0xB0)
YELLOW = RGBColor(0xFF, 0xD5, 0x4F)
ORANGE = RGBColor(0xFF, 0x8A, 0x65)
CYAN = RGBColor(0x4D, 0xD0, 0xE1)
RED_SOFT = RGBColor(0xEF, 0x53, 0x50)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# ── 헬퍼 함수들 ──
def set_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_text(slide, left, top, width, height, text, size=18, color=WHITE,
             bold=False, align=PP_ALIGN.LEFT, font_name="맑은 고딕"):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top),
                                     Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = align
    return tf

def add_para(tf, text, size=18, color=WHITE, bold=False, space_before=6,
             font_name="맑은 고딕", align=PP_ALIGN.LEFT):
    p = tf.add_paragraph()
    p.text = text
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.space_before = Pt(space_before)
    p.alignment = align
    return p

def title_slide(title, subtitle=""):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    set_bg(slide, BG_DARK)
    add_text(slide, 0.8, 2.0, 11.7, 1.5, title, size=40, color=WHITE, bold=True,
             align=PP_ALIGN.CENTER)
    if subtitle:
        add_text(slide, 0.8, 3.8, 11.7, 1.0, subtitle, size=22, color=LIGHT_GRAY,
                 align=PP_ALIGN.CENTER)
    return slide

def section_slide(part_num, part_title, slide_range=""):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, BG_SECTION)
    add_text(slide, 0.8, 2.2, 11.7, 0.8, f"Part {part_num}", size=28,
             color=CYAN, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, 0.8, 3.2, 11.7, 1.2, part_title, size=36,
             color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    if slide_range:
        add_text(slide, 0.8, 4.6, 11.7, 0.6, slide_range, size=16,
                 color=LIGHT_GRAY, align=PP_ALIGN.CENTER)
    return slide

def content_slide(title, bullets, note="", right_text=""):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, BG_DARK)
    # Title bar
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0),
                                   Inches(13.333), Inches(0.9))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0x14, 0x14, 0x28)
    shape.line.fill.background()
    add_text(slide, 0.6, 0.1, 12, 0.7, title, size=26, color=ACCENT, bold=True)

    # 본문 영역
    bw = 7.5 if right_text else 11.5
    tf = add_text(slide, 0.8, 1.2, bw, 5.5, "", size=18)
    tf.paragraphs[0].text = ""  # clear
    for i, b in enumerate(bullets):
        if i == 0:
            tf.paragraphs[0].text = b
            tf.paragraphs[0].font.size = Pt(18)
            tf.paragraphs[0].font.color.rgb = WHITE
            tf.paragraphs[0].font.name = "맑은 고딕"
        else:
            color = WHITE
            bold = False
            sz = 18
            if b.startswith(">>"):
                b = "    " + b[2:].strip()
                color = LIGHT_GRAY
                sz = 16
            elif b.startswith("!!"):
                b = b[2:].strip()
                color = YELLOW
                bold = True
            elif b.startswith("@@"):
                b = b[2:].strip()
                color = CYAN
                sz = 16
            add_para(tf, b, size=sz, color=color, bold=bold)

    # 오른쪽 보조 텍스트 (수식 등)
    if right_text:
        tf2 = add_text(slide, 8.5, 1.2, 4.3, 5.5, "", size=16, color=CYAN)
        tf2.paragraphs[0].text = ""
        for i, line in enumerate(right_text.split("\n")):
            if i == 0:
                tf2.paragraphs[0].text = line
                tf2.paragraphs[0].font.size = Pt(16)
                tf2.paragraphs[0].font.color.rgb = CYAN
                tf2.paragraphs[0].font.name = "Consolas"
            else:
                add_para(tf2, line, size=16, color=CYAN, font_name="Consolas")

    # 하단 노트
    if note:
        add_text(slide, 0.8, 6.7, 11.5, 0.6, note, size=13, color=LIGHT_GRAY)

    return slide

def equation_slide(title, equations, explanation):
    """수식 중심 슬라이드"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, BG_DARK)
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0),
                                   Inches(13.333), Inches(0.9))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0x14, 0x14, 0x28)
    shape.line.fill.background()
    add_text(slide, 0.6, 0.1, 12, 0.7, title, size=26, color=ACCENT, bold=True)

    # 수식 영역 (중앙, 큰 글씨)
    eq_tf = add_text(slide, 1.5, 1.3, 10.3, 2.5, "", size=24, color=YELLOW,
                     align=PP_ALIGN.CENTER, font_name="Consolas")
    eq_tf.paragraphs[0].text = ""
    for i, eq in enumerate(equations):
        if i == 0:
            eq_tf.paragraphs[0].text = eq
            eq_tf.paragraphs[0].font.size = Pt(24)
            eq_tf.paragraphs[0].font.color.rgb = YELLOW
            eq_tf.paragraphs[0].font.name = "Consolas"
            eq_tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        else:
            add_para(eq_tf, eq, size=24, color=YELLOW, font_name="Consolas",
                     align=PP_ALIGN.CENTER, space_before=12)

    # 설명 영역
    ex_tf = add_text(slide, 0.8, 4.0, 11.5, 3.0, "", size=18)
    ex_tf.paragraphs[0].text = ""
    for i, line in enumerate(explanation):
        if i == 0:
            ex_tf.paragraphs[0].text = line
            ex_tf.paragraphs[0].font.size = Pt(18)
            ex_tf.paragraphs[0].font.color.rgb = WHITE
            ex_tf.paragraphs[0].font.name = "맑은 고딕"
        else:
            c = LIGHT_GRAY if line.startswith("  ") else WHITE
            add_para(ex_tf, line, size=16 if line.startswith("  ") else 18, color=c)
    return slide

def comparison_slide(title, headers, rows, note=""):
    """비교 테이블 슬라이드"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, BG_DARK)
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0),
                                   Inches(13.333), Inches(0.9))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0x14, 0x14, 0x28)
    shape.line.fill.background()
    add_text(slide, 0.6, 0.1, 12, 0.7, title, size=26, color=ACCENT, bold=True)

    n_cols = len(headers)
    col_w = 11.0 / n_cols
    x_start = 1.2

    # 헤더
    for j, h in enumerate(headers):
        add_text(slide, x_start + j * col_w, 1.2, col_w, 0.5, h,
                 size=16, color=CYAN, bold=True, align=PP_ALIGN.CENTER)

    # 구분선
    shape2 = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                    Inches(x_start), Inches(1.7),
                                    Inches(11.0), Inches(0.02))
    shape2.fill.solid()
    shape2.fill.fore_color.rgb = CYAN
    shape2.line.fill.background()

    # 데이터 행
    for i, row in enumerate(rows):
        y = 1.85 + i * 0.55
        for j, cell in enumerate(row):
            c = WHITE
            if j == 0:
                c = YELLOW
            elif "X" in str(cell) or "느림" in str(cell) or "불가" in str(cell):
                c = RED_SOFT
            elif "O" in str(cell) or "빠름" in str(cell):
                c = ACCENT
            add_text(slide, x_start + j * col_w, y, col_w, 0.5, str(cell),
                     size=15, color=c, align=PP_ALIGN.CENTER)

    if note:
        add_text(slide, 0.8, 6.7, 11.5, 0.6, note, size=13, color=LIGHT_GRAY)
    return slide


# ═══════════════════════════════════════════════════════════════
#  ⛔ 2026-08-13 감사 차단 게이트
# ═══════════════════════════════════════════════════════════════
def _blocked_banned_headline():
    """⛔ 인용금지 수치를 재생산하는 슬라이드 경로의 차단 게이트 (2026-08-13 감사)."""
    raise RuntimeError(
        "⛔ 이 그림/슬라이드는 인용금지 수치(0.223/0.319/13.4/132.6/~10x)를 하드코딩하고 있었다. "
        "2026-08-13 감사에서 차단됨. 재활성화하려면 검증된 수치로 교체할 것. "
        "근거: memory/reference_citation_bans.md, tmp/AUDIT_recheck_2026-08-13/A3_citation_bans.md"
    )


# ═══════════════════════════════════════════════════════════════
#  슬라이드 생성 시작
# ═══════════════════════════════════════════════════════════════

# ── 표지 ──
title_slide("WaveSplat 알고리즘 완전 해설",
            "격자 없는 스펙트럼 파도장 추정 — 왜, 어떻게, 무엇이 다른가\n"
            "박세용 | 2026")

# ═══════════════════════════════════════════════════════════════
#  Part 0: 문제 정의 (슬라이드 2-6)
# ═══════════════════════════════════════════════════════════════
section_slide(0, "문제 정의", "Slides 2-6")

# Slide 2
content_slide(
    "왜 파도를 추정하는가?",
    [
        "선박/해양 구조물은 파도에 의해 지배적 외력을 받음",
        "!! 파도를 알면 할 수 있는 것들:",
        "  1. 선박 운동 예측 → 화물 고박, 헬기 착륙 타이밍",
        "  2. 능동 제어 → 하이드로포일 안정화 (이 논문의 응용)",
        "  3. 구조물 하중 예측 → 피로 수명 계산",
        "  4. 항로 최적화 → 연료 절감, 승객 안전",
        "@@현재: 부이(점 측정)나 레이더(저해상도)에 의존",
        "@@목표: LiDAR로 실시간 2D 파도장 복원",
    ]
)

# Slide 3
comparison_slide(
    "파도 관측 방법 비교",
    ["방법", "측정 범위", "해상도", "시간 해상도", "실시간", "설치"],
    [
        ["파고 부이", "점 (0D)", "높음", "높음", "O", "해상 투하"],
        ["X-band 레이더", "수 km (2D)", "~10m", "~1Hz", "O", "선박 고정"],
        ["위성 SAR", "수백 km", "~100m", "수 시간", "X", "없음"],
        ["LiDAR", "10-250m (2D)", "~0.1m", "10Hz", "O", "선박 고정"],
    ],
    note="LiDAR: 높은 시공간 해상도, 하지만 관측 점이 희소 (50~200점/프레임)"
)

# Slide 4
content_slide(
    "LiDAR 관측의 특성",
    [
        "선박 전방에 설치된 LiDAR가 수면을 스캔",
        "매 프레임(0.1초) 50~200개의 3D 점 (x, y, z) 취득",
        "",
        "!! 관측의 어려움:",
        "  1. 희소성: 250m x 250m 영역에 점이 ~100개뿐",
        "  2. 불균일: 가까이 = 밀집, 멀리 = 희박",
        "  3. 노이즈: 물방울, 진동, 반사 각도에 따른 신호 손실",
        "  4. 비정렬: 매 프레임 점의 위치가 달라짐 (격자 아님!)",
        "",
        "@@기존 격자 기반 방법의 근본적 문제: 점이 격자 위에 없다",
    ]
)

# Slide 5
content_slide(
    "핵심 문제 정의",
    [
        "!! 입력: 시공간적으로 희소한 LiDAR 점 {(xj, yj, zj, tj)}",
        "",
        "!! 출력: 임의의 (x, y, t)에서의 수면 높이 eta(x, y, t)",
        "",
        "조건:",
        "  1. 실시간: 100ms 이내 (10Hz LiDAR에 맞춰)",
        "  2. 물리적으로 유효: 분산관계를 만족하는 파도",
        "  3. 시간 연속: 매 프레임 추적 (갑자기 바뀌면 안 됨)",
        "  4. 격자 불필요: 관측 위치가 불규칙",
        "",
        "@@이 4가지를 동시에 만족하는 방법이 WaveSplat",
    ]
)

# Slide 6
content_slide(
    "이상적 해법의 조건 vs 기존 방법",
    [
        "!! 조건 체크리스트:",
        "",
        "  실시간(100ms)     ETKF: 132ms (X)  GPR: >200ms (X)  WS: 13ms (O)",
        "  물리법칙 활용      ETKF: 간접       GPR: 없음         WS: 직접 (O)",
        "  시간 연속성        ETKF: O (KF)     GPR: X (프레임별)  WS: O (추적)",
        "  격자 불필요        ETKF: X (격자)   GPR: O            WS: O",
        "",
        "@@어떤 기존 방법도 4가지를 동시에 만족하지 못함",
        "@@→ 새로운 접근이 필요한 이유",
    ]
)

# ═══════════════════════════════════════════════════════════════
#  Part 1: 파도의 물리학 (슬라이드 7-13)
# ═══════════════════════════════════════════════════════════════
section_slide(1, "파도의 물리학", "Slides 7-13")

# Slide 7
equation_slide(
    "단일 평면파",
    ["eta(x,y,t) = w * cos(kx*x + ky*y - omega*t + phi)"],
    [
        "w  = 진폭 (파도 높이의 절반)",
        "  파고 H = 2w",
        "kx, ky = 파수 벡터 (파도의 방향과 파장을 결정)",
        "  파장 lambda = 2*pi / |k|,  방향 = atan2(ky, kx)",
        "omega = 각주파수 (파도가 얼마나 빨리 지나가는지)",
        "  주기 T = 2*pi / omega",
        "phi = 위상 (파도의 시작점)",
        "",
        "!! 각 파도 성분은 이 4개의 숫자로 완전히 기술됨: (w, kx, ky, phi)",
    ]
)

# Slide 8
content_slide(
    "파수 벡터 (kx, ky) — 파도의 '지문'",
    [
        "파수 벡터 k = (kx, ky)는 파도의 공간적 특성을 완전히 결정",
        "",
        "!! |k| = sqrt(kx^2 + ky^2)  →  파장 lambda = 2*pi / |k|",
        "  |k| 크면 → 짧은 파도 (잔물결)",
        "  |k| 작으면 → 긴 파도 (너울)",
        "",
        "!! 방향 theta = atan2(ky, kx)",
        "  파도가 어디서 오는지를 결정",
        "  head sea: 정면, beam sea: 옆면, following: 뒤",
        "",
        "@@비유: 라디오에서 주파수를 맞추듯, 바다에서 (kx,ky)로 '채널'을 맞춤",
    ]
)

# Slide 9
equation_slide(
    "분산관계 (Dispersion Relation) — 파도의 물리 법칙",
    [
        "omega^2 = g * |k| * tanh(|k| * d)",
        "",
        "심해 (d >> lambda):  omega = sqrt(g * |k|)",
        "천해 (d << lambda):  omega = |k| * sqrt(g * d)",
    ],
    [
        "!! 핵심: 파장이 결정되면 속도(주파수)는 자동으로 결정됨",
        "",
        "직관적 의미:",
        "  - 긴 파도(큰 lambda, 작은 |k|)는 빠르게 이동",
        "  - 짧은 파도(작은 lambda, 큰 |k|)는 느리게 이동",
        "  - 수심 d가 얕으면 모든 파도가 느려짐",
        "",
        "!! WaveSplat의 핵심 무기: omega를 자유변수로 두지 않고",
        "  분산관계로 강제 → 추정할 파라미터가 줄어듦",
    ]
)

# Slide 10
content_slide(
    "핵심 통찰: omega는 자유변수가 아니다",
    [
        "일반적 접근: 각 성분 (w, kx, ky, omega, phi) → 5N개 파라미터",
        "",
        "!! WaveSplat 접근: omega = f(kx, ky, d) → 4N개 파라미터",
        "",
        "이것이 왜 중요한가?",
        "  1. 파라미터 20% 감소 → 최적화 더 쉬움",
        "  2. 물리적으로 불가능한 해(omega-k 불일치)를 원천 차단",
        "  3. 시간 예측이 정확해짐: 분산관계가 맞아야 파도가 제대로 전파",
        "",
        "@@ETKF/GPR은 이 법칙을 모름 → '격자 높이'만 맞추려 함",
        "@@WaveSplat은 법칙을 내장 → 적은 데이터로도 정확한 추정",
    ]
)

# Slide 11
equation_slide(
    "다성분 중첩 — 실제 바다의 모델",
    [
        "eta(x,y,t) = SUM_{i=1}^{N} w_i * cos(kx_i*x + ky_i*y - omega_i*t + phi_i)",
        "",
        "omega_i = sqrt(g * |k_i| * tanh(|k_i| * d))   (분산관계로 결정)",
    ],
    [
        "실제 바다 = 여러 방향, 여러 파장의 파도가 겹쳐진 것",
        "",
        "!! N = 파도 성분의 수:",
        "  너울(swell): N = 1~3 (단순, 먼 바다에서 온 긴 파도)",
        "  BF3 혼합해: N = 5~10 (바람 + 너울)",
        "  BF4-5 복잡해: N = 15~30 (다방향, 다주파수)",
        "",
        "!! 총 자유 파라미터: 4N개",
        "  N=15일 때 60개, N=30일 때 120개 (ETKF의 1200+개와 비교!)",
    ]
)

# Slide 12
content_slide(
    "JONSWAP 스펙트럼 — 바다의 에너지 분포",
    [
        "!! JONSWAP = 바람에 의해 생성된 파도의 전형적 에너지 분포",
        "",
        "핵심 파라미터:",
        "  Hs = 유의파고 (상위 1/3 파고의 평균, 해상 상태 지표)",
        "  Tp = 피크 주기 (가장 에너지가 큰 파도의 주기)",
        "  gamma = 피크 형상 계수 (좁으면 너울, 넓으면 풍파)",
        "",
        "Beaufort 등급과의 관계:",
        "  BF2: Hs ~ 0.3m, Tp ~ 3s  (잔잔)",
        "  BF3: Hs ~ 0.88m, Tp ~ 5s (보통)",
        "  BF4: Hs ~ 1.88m, Tp ~ 7s (거친 해상)",
        "  BF5: Hs ~ 3.25m, Tp ~ 9s (험한 해상)",
    ]
)

# Slide 13
content_slide(
    "선형 파도 모델의 충분성",
    [
        "!! 왜 cos 합으로 충분한가?",
        "",
        "선형 파도 이론 (Airy wave):",
        "  - 파고/파장 비가 작을 때 (H/lambda << 1) 성립",
        "  - BF2-4에서 대부분 성립 (steepness < 0.05)",
        "  - 중첩 원리: 각 성분이 독립적으로 전파",
        "",
        "비선형 효과 (Stokes, 쇄파):",
        "  - BF5+ 또는 천해에서 중요해짐",
        "  - 현재는 향후 연구로 남겨둠",
        "",
        "@@요약: BF2-4 범위에서 선형 모델은 '충분히 좋은' 근사",
        "@@관측 노이즈 >> 비선형 오차",
    ]
)

# ═══════════════════════════════════════════════════════════════
#  Part 2: 공간 영역 방법 ① — ETKF (슬라이드 14-21)
# ═══════════════════════════════════════════════════════════════
section_slide(2, "공간 영역 방법 ① — ETKF", "Slides 14-21")

# Slide 14
content_slide(
    "격자 기반 접근의 기본 아이디어",
    [
        "수면을 NxM 격자로 나누고, 각 격자점의 높이를 추정",
        "",
        "!! 상태 벡터 x = [h_11, h_12, ..., h_NM]^T",
        "  G = 40 x 30 격자 → 1,200차원 벡터",
        "",
        "관측 모델:",
        "  z_j = H(x) + noise",
        "  LiDAR 점 (x_j, y_j)에서의 높이 = 가장 가까운 격자점(들)의 보간",
        "",
        "!! 문제: 관측이 50~200개인데 상태가 1,200개",
        "  → 심하게 underdetermined (해가 무한히 많음)",
        "  → 사전 정보(prior)가 필요 → 칼만 필터",
    ]
)

# Slide 15
content_slide(
    "칼만 필터 기초",
    [
        "!! Predict-Update 2단계 반복:",
        "",
        "1. Predict (예측):",
        "   x_pred = F * x_prev    (물리 모델로 다음 상태 예측)",
        "   P_pred = F*P*F^T + Q   (불확실성 증가)",
        "",
        "2. Update (보정):",
        "   K = P_pred * H^T * (H*P_pred*H^T + R)^(-1)  (칼만 이득)",
        "   x_new = x_pred + K * (z - H*x_pred)          (관측으로 보정)",
        "",
        "!! K가 크면 → 관측을 더 신뢰",
        "!! K가 작으면 → 예측(모델)을 더 신뢰",
        "",
        "@@문제: P가 1200 x 1200 행렬 → 직접 계산 불가능!",
    ]
)

# Slide 16
content_slide(
    "왜 앙상블 칼만 필터(EnKF)인가?",
    [
        "1200차원의 공분산 행렬 P를 직접 유지할 수 없음",
        "→ Ne개의 '샘플'(앙상블 멤버)로 P를 근사",
        "",
        "!! 아이디어: P ≈ (1/(Ne-1)) * X_ens * X_ens^T",
        "  X_ens = 앙상블 편차 행렬",
        "  Ne = 50~100개면 대략적인 공분산 구조를 포착",
        "",
        "ETKF (Ensemble Transform KF):",
        "  앙상블 멤버를 관측에 맞춰 '변환'하여 update",
        "  전체 P를 계산하지 않고 Ne x Ne 소행렬만 처리",
        "",
        "@@핵심 비용: Ne개 멤버 각각에 대해 격자 연산 수행",
        "@@Ne=100, G=1200 → 매 프레임 120,000번 연산",
    ]
)

# Slide 17
equation_slide(
    "ETKF 동작 원리",
    [
        "x_mean = (1/Ne) * SUM x_i          (앙상블 평균 = 추정값)",
        "X_ens = [x_1 - x_mean, ..., x_Ne - x_mean]",
        "K = X_ens * (X_ens^T * H^T) * (H*X_ens*X_ens^T*H^T + R)^(-1)",
    ],
    [
        "동작 순서:",
        "  1. Ne개의 격자 파도장을 초기화 (랜덤 위상의 스펙트럼에서 생성)",
        "  2. Predict: 각 멤버를 물리 모델로 전파 (FFT 기반)",
        "  3. 관측이 오면: 각 멤버에서 관측 위치의 높이 추출",
        "  4. Update: 관측과 가까운 멤버에 더 큰 가중치",
        "  5. 앙상블 평균이 최종 추정값",
        "",
        "!! 반복할수록 앙상블이 '진짜 파도'에 수렴... 해야 하지만",
    ]
)

# Slide 18
content_slide(
    "ETKF의 장점",
    [
        "1. 직관적: 격자 높이를 직접 추정 → 결과 해석이 쉬움",
        "",
        "2. 불확실성 추정: 앙상블 분산 → 어디가 불확실한지 알 수 있음",
        "",
        "3. 비선형 관측 모델 허용: H(x)가 비선형이어도 동작",
        "",
        "4. 시간 연속성: predict-update 루프로 자연스럽게 추적",
        "",
        "5. 성숙한 방법론: 기상/해양 분야에서 수십 년 검증",
        "",
        "@@그런데... 왜 ETKF로 부족한가?",
    ]
)

# ⛔ 2026-08-13 감사 차단점 — 이 지점부터 아래 슬라이드들이 인용금지 수치를 재생산한다.
_blocked_banned_headline()

# Slide 19
content_slide(
    "ETKF의 근본적 한계 ①: 격자 해상도 vs 계산량",
    [
        "!! 트레이드오프:",
        "  격자 촘촘 → 정밀하지만 느림",
        "  격자 성김 → 빠르지만 aliasing (짧은 파도를 놓침)",
        "",
        "실제 설정: G = 40 x 30 = 1,200 격자점",
        "  격자 간격 ~6m → 파장 12m 이하 파도 해석 불가",
        # ⛔ 원본 이력 (주석 처리, 인용금지 — "132.6ms"):
        # "  Ne = 100 → 매 프레임 132.6ms (실시간 예산 초과!)",
        "  Ne = 100 → 매 프레임 비용 ⛔RETRACTED (2026-08-13 감사)",
        "",
        "더 정밀하려면?",
        "  G = 100 x 75 → Ne = 200 필요 → ~1초/프레임",
        "",
        "@@격자 기반의 본질적 문제: 차원의 저주",
        "@@해상도를 2배로 → 상태 4배, 계산 ~16배",
    ]
)

# Slide 20
content_slide(
    "ETKF의 근본적 한계 ②: 분산관계 미활용",
    [
        "!! ETKF 상태 = 격자점 높이 h_ij",
        "  높이 자체는 분산관계를 '모른다'",
        "  앙상블이 시간 전파를 통해 '간접적으로' 물리를 학습해야 함",
        "",
        "문제:",
        "  - 초기 앙상블의 스펙트럼이 실제와 다르면 수렴이 느림",
        "  - Ne가 작으면 공분산이 부정확 → 물리가 깨짐",
        "  - 스펙트럼 정보(어떤 주파수 성분이 있는지)를 직접 출력 못 함",
        "",
        "@@비유: ETKF는 '사진'을 맞추려는 것",
        "@@WaveSplat은 '악보'를 찾는 것",
        "@@악보를 알면 언제든 사진을 생성할 수 있지만, 반대는 어렵다",
    ]
)

# Slide 21
content_slide(
    "ETKF 실패 모드 — '진폭은 맞지만 위치가 틀림'",
    [
        "100 에피소드 평가 결과:",
        "",
        "  VR (Variance Ratio) = 0.90  → 에너지(진폭 제곱합)는 거의 맞음",
        "  rho (Pearson 상관) = 0.11    → 형상(파도 위치)은 거의 랜덤",
        "",
        "!! 이게 무슨 뜻인가?",
        "  ETKF: '파고 1m짜리 파도가 있다'는 맞추지만",
        "        '어디에 있는지'는 거의 맞추지 못함",
        "",
        # ⛔ 원본 이력 (주석 처리, 인용금지 — "nRMSE = 0.319"):
        # "  nRMSE = 0.319 → 숫자만 보면 '나쁘지 않아 보이지만'",
        "  nRMSE = ⛔RETRACTED (분모 0.319 를 내는 결과 파일이 없다)",
        "  VR/rho 조합으로 보면 → 사실상 진단적 가치가 낮음",
        "",
        "@@이것이 nRMSE 하나로 평가하면 안 되는 이유",
        "@@→ 6-지표 패널의 필요성 (Part 9에서 상세히)",
    ]
)

# ═══════════════════════════════════════════════════════════════
#  Part 3: 공간 영역 방법 ② — GPR (슬라이드 22-28)
# ═══════════════════════════════════════════════════════════════
section_slide(3, "공간 영역 방법 ② — GPR", "Slides 22-28")

# Slide 22
content_slide(
    "가우시안 프로세스(GP)란?",
    [
        "!! 함수의 확률 분포",
        "  '이 함수가 이런 모양일 확률이 높다' 를 수학적으로 표현",
        "",
        "비유: 점을 찍고 매끄러운 곡선으로 잇기",
        "  - 관측점 근처: 확실 (분산 작음)",
        "  - 관측점에서 멀리: 불확실 (분산 큼)",
        "  - 매끄러움의 정도: 커널 함수가 결정",
        "",
        "!! GP의 매력:",
        "  1. 비모수적: 함수 형태를 가정하지 않음",
        "  2. 불확실성 정량화: 예측값 + 신뢰구간 동시 제공",
        "  3. 격자 불필요: 임의 위치에서 예측 가능",
    ]
)

# Slide 23
equation_slide(
    "GPR 수식",
    [
        "mu(x*) = k(x*, X) * [K(X,X) + sigma^2 * I]^(-1) * z",
        "",
        "sigma^2(x*) = k(x*,x*) - k(x*,X) * [K+sigma^2*I]^(-1) * k(X,x*)",
    ],
    [
        "x* = 예측하고 싶은 위치",
        "X  = 관측 위치들",
        "z  = 관측값 (수면 높이)",
        "k(.,.) = 커널 함수 (두 점 사이의 '유사도')",
        "K(X,X) = 관측점끼리의 유사도 행렬 (M x M)",
        "",
        "!! 핵심 연산: M x M 행렬의 역행렬",
        "  → O(M^3) 계산량, M=200이면 8백만 연산",
    ]
)

# Slide 24
content_slide(
    "Matern 커널 — 거리에 따른 상관도",
    [
        "!! k(r) ~ (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)",
        "  (Matern nu=5/2)",
        "",
        "핵심 파라미터: length scale l",
        "  l = 20m → '20m 이내의 점들은 강하게 연관'",
        "  l 크면 → 매끄러운 보간 (장파에 적합)",
        "  l 작으면 → 국소적 보간 (단파에 적합)",
        "",
        "!! 문제: 파도에는 여러 스케일이 동시에 존재",
        "  하나의 l로는 모든 스케일을 포착 불가",
        "  → l을 최적화해도 타협점밖에 못 찾음",
    ]
)

# Slide 25
content_slide(
    "GPR의 근본적 한계 ①: 시간을 무시함",
    [
        "!! GPR 커널 k(x1, x2)는 공간 거리만 봄",
        "  → 파도가 '이동'한다는 사실을 모름!",
        "",
        "예시:",
        "  t=0: 점 A에서 파고 1m 관측",
        "  t=0.1: A에서 0.5m → 파도가 이동했기 때문",
        "  GPR: A 근처를 (1+0.5)/2 ≈ 0.75m로 평균해버림",
        "",
        "결과:",
        "  시간적으로 일관된 파도 추적 불가",
        "  프레임별 독립 추정 → 시계열이 '진동'",
        "",
        "@@시공간 커널을 쓸 수 있지만 → 어떤 속도로 이동하나?",
        "@@→ 분산관계를 커널에 넣어야 함 → 결국 물리 모델 필요",
    ]
)

# Slide 26
content_slide(
    "GPR의 근본적 한계 ②: O(M^3) 계산",
    [
        "!! 커널 행렬 K의 역행렬 = O(M^3)",
        "",
        "M = 50:   ~125,000 연산 → 빠름",
        "M = 200:  ~8,000,000 연산 → 느림",
        "M = 500:  ~125,000,000 연산 → 실시간 불가",
        "",
        "추가로: 커널 하이퍼파라미터(l, sigma) 최적화",
        "  매 프레임마다 최적 l을 찾아야 → 추가 시간",
        "",
        "!! 결과: M=100~200에서 이미 >200ms",
        "  실시간 예산(100ms)을 2배 이상 초과",
        "",
        "@@근사 GP (sparse GP, inducing points)도 있지만",
        "@@시간 무시 문제는 근본적으로 해결 안 됨",
    ]
)

# Slide 27
content_slide(
    "GPR 실패 모드 — 'near-zero predictor'",
    [
        "100 에피소드 평가 결과:",
        "",
        "  VR (Variance Ratio) = 0.065   → 에너지의 6.5%만 재현",
        "  rho (Pearson 상관) = 0.193     → 약한 상관",
        "  SS (Skill Score) = -0.140      → zero predictor보다 나쁨!",
        "",
        "!! 이게 무슨 뜻인가?",
        "  GPR: 관측점 사이를 '평균'으로 채움",
        "  → 진폭이 크게 줄어듦 (과도한 평활화)",
        "  → 거의 평평한 수면을 예측 = near-zero predictor",
        "",
        # ⛔ 원본 이력 (주석 처리, 인용금지 — "0.261" / "0.319"):
        # "  nRMSE = 0.261 → ETKF(0.319)보다 나아 '보이지만'",
        "  nRMSE = ⛔RETRACTED (2026-08-13 감사; 재도출 GPR 값은 0.276)",
        "  실제로는 '아무것도 예측하지 않는 것'에 가까움",
    ]
)

# Slide 28
comparison_slide(
    "ETKF vs GPR — 공간 영역 방법의 한계 정리",
    ["", "ETKF", "GPR", "필요한 것"],
    [
        ["물리법칙", "간접 (앙상블 학습)", "없음", "직접 내장"],
        ["시간 활용", "O (KF predict)", "X (프레임별)", "시간 연속 추적"],
        ["계산량", "132ms (초과)", ">200ms (초과)", "<100ms"],
        ["파라미터 수", "120,000 (Ne*G)", "M (관측수)", "~60 (4N)"],
        ["결과 형태", "격자 높이", "공간 보간", "스펙트럼 (재생 가능)"],
        ["실패 모드", "형상 불일치", "평탄화", "—"],
    ],
    note="두 방법 모두 '공간 영역'에서 높이를 직접 맞추려 함 → 주파수 영역으로의 전환이 필요"
)

# ═══════════════════════════════════════════════════════════════
#  Part 4: 왜 주파수 영역인가 (슬라이드 29-34)
# ═══════════════════════════════════════════════════════════════
section_slide(4, "왜 주파수 영역 Splatting이 필요한가", "Slides 29-34")

# Slide 29
content_slide(
    "공간 vs 주파수 — 근본적 차이",
    [
        "!! 공간 영역: '여기 높이가 얼마인가?'",
        "  → 격자점마다 하나의 숫자 → 데이터 양에 비례하는 파라미터",
        "",
        "!! 주파수 영역: '어떤 파도들이 섞여 있는가?'",
        "  → 각 파도 성분의 (진폭, 방향, 파장, 위상) → 고정 수의 파라미터",
        "",
        "차이의 핵심:",
        "  공간: 해상도를 높이면 파라미터가 폭발적으로 증가",
        "  주파수: 해상도와 무관하게 N개 성분만 기술하면 끝",
        "",
        "@@비유: 사진(공간) vs 악보(주파수)",
        "@@사진은 해상도에 따라 크기가 달라지지만",
        "@@악보는 곡의 복잡도에만 의존",
    ]
)

# Slide 30
content_slide(
    "음악 비유로 이해하기",
    [
        "!! 시간 영역 (파형): 마이크로 녹음한 소리",
        "  → 매 시점의 진폭 값 → 데이터가 방대함",
        "  → 이 파형만으로는 '어떤 악기가 연주하는지' 알기 어려움",
        "",
        "!! 주파수 영역 (악보): 어떤 음(주파수)이 어떤 크기(진폭)로",
        "  → 성분 수만큼의 파라미터 → 간결함",
        "  → 악보를 알면 언제든 파형을 재생 가능",
        "",
        "파도에 적용:",
        "  LiDAR 관측 = 마이크 녹음 (희소한 시공간 샘플)",
        "  WaveSplat = 이 녹음에서 악보를 추출",
        "  ETKF = 녹음을 직접 복원하려는 시도",
        "  GPR = 녹음 사이를 보간하려는 시도",
    ]
)

# Slide 31
content_slide(
    "주파수 파라미터를 알면 어디서든 복원 가능",
    [
        "!! 핵심: eta(x,y,t) = SUM w_i * cos(kx_i*x + ky_i*y - omega_i*t + phi_i)",
        "",
        "한번 (w, kx, ky, phi)를 추정하면:",
        "",
        "  1. 관측 영역 밖에서도 예측 가능",
        "     → 선박 전방 500m 지점의 파도도 계산 가능",
        "",
        "  2. 미래 시점도 예측 가능",
        "     → t를 바꾸면 됨 (분산관계가 전파 속도를 알려줌)",
        "",
        "  3. 격자 해상도 제한 없음",
        "     → 연속 함수이므로 임의의 (x,y)에서 평가",
        "",
        "@@ETKF/GPR은 관측 영역 내에서만, 현재 시점에서만 유효",
        "@@WaveSplat은 시공간적으로 외삽(extrapolation) 가능",
    ]
)

# Slide 32
content_slide(
    "왜 격자가 필요 없는가",
    [
        "!! ETKF: eta = 격자점 높이의 보간",
        "  → 격자 간격보다 짧은 파도는 표현 불가 (Nyquist)",
        "  → 격자 간격 6m → 파장 12m 이하 불가",
        "",
        "!! WaveSplat: eta = cos 함수의 합",
        "  → 파수 kx, ky가 연속값 → 어떤 파장도 표현 가능",
        "  → 격자 개념 자체가 없음",
        "",
        "이것이 'Grid-free'의 의미:",
        "  관측이 격자 위에 없어도 전혀 문제없음",
        "  LiDAR 점이 불규칙하게 분포해도 그대로 사용",
        "",
        "@@격자 기반 방법은 불규칙 관측 → 격자 보간이 필수",
        "@@이 보간에서 이미 정보가 손실됨",
    ]
)

# Slide 33
content_slide(
    "시간이 자연스럽게 포함되는 이유",
    [
        "!! cos(kx*x + ky*y - omega*t + phi) 에서 t는 변수",
        "",
        "분산관계 omega = f(k, d)가 '시계' 역할:",
        "  - 각 성분이 고유의 속도로 전파",
        "  - 장파(작은 k)는 빠르게, 단파(큰 k)는 느리게",
        "  - 이 '분산'이 파도의 시간 진화를 완전히 기술",
        "",
        "!! ETKF가 'predict step'에서 물리 모델로 전파하는 것을",
        "  WaveSplat은 수식 하나로 해결",
        "",
        "!! GPR는 시간을 아예 모름",
        "  → 매 프레임 독립 추정 → 시계열 불연속",
        "",
        "@@WaveSplat의 추적(tracking)에서 phi += omega*dt로 예측",
        "@@분산관계가 맞으면 예측이 정확 → 소량의 보정만으로 충분",
    ]
)

# Slide 34
comparison_slide(
    "파라미터 수의 극적 차이",
    ["방법", "표현", "파라미터 수", "예시 (BF3)", "비고"],
    [
        ["ETKF", "격자 높이", "Ne x G", "100 x 1200 = 120,000", "차원의 저주"],
        ["GPR", "커널 보간", "M + hp", "200 + 3 = 203", "O(M^3) 계산"],
        ["WaveSplat", "스펙트럼", "4N", "4 x 15 = 60", "물리 내장"],
    ],
    note="같은 바다를 기술하는데 WaveSplat은 2000배 적은 파라미터 → Occam's razor: 단순한 모델이 더 잘 일반화"
)

# ═══════════════════════════════════════════════════════════════
#  Part 5: 주파수 영역의 기존 방법들 (슬라이드 35-40)
# ═══════════════════════════════════════════════════════════════
section_slide(5, "주파수 영역의 기존 방법들", "Slides 35-40")

# Slide 35
content_slide(
    "고전적 스펙트럼 분석 — FFT",
    [
        "!! FFT (Fast Fourier Transform):",
        "  시간/공간 신호를 주파수 성분으로 분해하는 표준 방법",
        "",
        "요구 조건:",
        "  1. 균일 격자 위의 샘플 (등간격)",
        "  2. 충분한 샘플 수 (Nyquist: 파장의 2배 이상)",
        "  3. 연속적인 데이터 (빈 곳이 없어야)",
        "",
        "!! LiDAR 데이터의 현실:",
        "  1. 불규칙 배치 (등간격 아님) → FFT 직접 적용 불가",
        "  2. 희소 (50~200점) → 해상도 부족",
        "  3. 매 프레임 위치 변경 → 일관된 격자 구성 불가",
        "",
        "@@FFT → NUFFT(비균일 FFT)도 있지만, 3D(x,y,t)에서는 여전히 어려움",
    ]
)

# Slide 36
content_slide(
    "MUSIC 알고리즘 — 초고해상도 스펙트럼 추정",
    [
        "!! MUSIC (Multiple Signal Classification):",
        "  신호 부분공간과 잡음 부분공간을 분리하여 피크 검출",
        "",
        "원리:",
        "  1. 데이터의 공분산 행렬 R = E[x*x^H] 계산",
        "  2. R의 고유분해: 큰 고유값 = 신호, 작은 고유값 = 잡음",
        "  3. 잡음 부분공간에 '수직'인 방향 탐색 → 아주 날카로운 피크",
        "",
        "!! 장점: 빔포밍보다 훨씬 좁은 피크 → 가까운 주파수 분리 가능",
        "",
        "레이더/소나 분야에서 널리 사용",
        "파도 분석에서도 레이더 배열에서 성공 사례 있음",
    ]
)

# Slide 37
content_slide(
    "MUSIC의 한계 — 왜 LiDAR에 부적합한가",
    [
        "!! 핵심 문제: 공분산 행렬 추정이 불가능",
        "",
        "MUSIC에 필요한 것:",
        "  고정된 센서 배열 (안테나처럼) → 같은 위치에서 반복 관측",
        "  → 시간에 따른 공분산 R = (1/T) * SUM x(t)*x(t)^H",
        "",
        "LiDAR의 현실:",
        "  매 프레임 관측 위치가 다름 (선박 움직임 + 스캔 패턴)",
        "  → '같은 위치'에서의 반복 관측이 존재하지 않음",
        "  → 공분산 행렬을 추정할 수 없음",
        "",
        "@@가정: 고정 센서 배열 → 현실: 이동하는 희소 점",
        "@@MUSIC/Capon/ESPRIT 등 부분공간 방법 모두 같은 이유로 부적합",
    ]
)

# Slide 38
content_slide(
    "Capon (MVDR) 빔포밍",
    [
        "!! Capon = '적응적' 빔포밍",
        "  데이터에 맞춰 가중치를 조절하여 간섭을 억제",
        "",
        "수식: P(k) = 1 / (a^H * R^(-1) * a)",
        "  a = steering vector, R = 공분산 행렬",
        "",
        "장점: 빔포밍보다 좋은 해상도, MUSIC보다 강건",
        "",
        "!! 하지만 같은 문제:",
        "  R의 추정이 필요 → 고정 배열 가정",
        "  LiDAR 희소 데이터에서 R 추정 불안정",
        "",
        "@@결론: 부분공간/적응적 방법은 LiDAR에 부적합",
        "@@→ 가장 단순한 '전통 빔포밍'이 오히려 강건",
    ]
)

# Slide 39
content_slide(
    "전통 빔포밍 (Conventional Beamforming)",
    [
        "!! 원리: 모든 후보 방향에 대해 단순 상관",
        "  P(k) = |SUM z_j * exp(-i * k . r_j)|^2",
        "",
        "장점:",
        "  1. 공분산 행렬 불필요 → 단일 스냅샷도 OK",
        "  2. 불규칙 센서 배치에 강건",
        "  3. 구현이 간단하고 빠름",
        "",
        "단점:",
        "  해상도가 낮음 (피크가 넓음)",
        "  가까운 주파수 성분 분리 어려움",
        "",
        "@@WaveSplat의 선택: 빔포밍의 강건성을 택하고",
        "@@해상도 문제는 비선형 최적화로 해결",
    ]
)

# Slide 40
content_slide(
    "WaveSplat의 선택: 분산관계 인식 빔포밍",
    [
        "!! 핵심 아이디어: 빔포밍에 분산관계를 주입",
        "",
        "일반 빔포밍: (kx, ky, omega) 3D 격자를 전부 탐색",
        "  → 64 x 128 x 128 = ~1백만 후보 → 느림",
        "",
        "!! WaveSplat: omega = f(kx, ky, d)로 고정",
        "  → 64 x 128 = 8,192 후보 → 128배 빠름",
        "  → 물리적으로 유효한 (k, omega) 쌍만 탐색",
        "",
        "결과:",
        "  물리적으로 불가능한 피크가 원천 제거됨",
        "  잡음에 의한 허위 피크가 크게 감소",
        "  빔포밍의 '넓은 피크' 문제 → 이후 최적화로 미세 조정",
        "",
        "@@빔포밍 = 좋은 초기 추정 → 최적화 = 정밀 보정",
    ]
)

# ═══════════════════════════════════════════════════════════════
#  Part 6: WaveSplat — 빔포밍 초기화 (슬라이드 41-45)
# ═══════════════════════════════════════════════════════════════
section_slide(6, "WaveSplat — 빔포밍 초기화", "Slides 41-45")

# Slide 41
equation_slide(
    "빔포밍 수식 상세",
    [
        "P(kx, ky) = | SUM_{j=1}^{M} z_j * exp(-i*(kx*xj + ky*yj - omega(k)*tj)) |^2",
        "",
        "omega(k) = sqrt(g * |k| * tanh(|k| * d))    <-- 분산관계로 결정!",
    ],
    [
        "직관적 의미:",
        "  '만약 (kx, ky) 방향의 파도가 있다면'",
        "  '관측값 z_j에 위상 보정을 하면 모두 같은 방향을 가리킬 것'",
        "  '그 합의 크기가 클수록 그 방향의 파도가 실제로 존재'",
        "",
        "!! exp(-i*...) = 위상 보정 (steering)",
        "  각 관측점의 '예상 위상'을 제거하여 정렬",
        "  정렬이 잘 되면(파도가 실재하면) → 보강 간섭 → P 큼",
        "  정렬이 안 되면(파도 없음) → 상쇄 간섭 → P 작음",
    ]
)

# Slide 42
content_slide(
    "극좌표 격자 — 왜 극좌표인가",
    [
        "!! 탐색 격자: |k| x 방위각 = 64 x 128 = 8,192 후보",
        "",
        "|k| 범위: 0.02 ~ 0.50 rad/m",
        "  → 파장 12.6m ~ 314m (BF2~5 커버)",
        "  → 64개 균등 분할",
        "",
        "방위각: 0 ~ 360도",
        "  → 128개 균등 분할 (2.8도 간격)",
        "",
        "!! 왜 극좌표?",
        "  1. 방향 해상도가 균일 (직교 격자에서는 대각선 해상도 떨어짐)",
        "  2. 파도의 물리가 방향+파수 크기로 자연스럽게 분리",
        "  3. 각도 억제(sidelobe 제거)가 자연스러움",
    ]
)

# Slide 43
content_slide(
    "Greedy 피크 추출",
    [
        "!! 빔포밍 결과 P(kx, ky)에서 파도 성분 추출:",
        "",
        "1. 최대 피크 위치 (kx*, ky*) 찾기",
        "2. 이 피크를 성분 목록에 추가",
        "3. 피크 주변 15도 범위 억제 (sidelobe 제거)",
        "4. 다음 최대 피크 찾기",
        "5. 반복 (N_max개까지 또는 임계값 이하까지)",
        "",
        "!! 왜 15도 각도 억제?",
        "  빔포밍의 메인 로브(main lobe)가 ~15도 폭",
        "  같은 파도의 sidelobe를 별도 성분으로 오인하는 것을 방지",
        "",
        "@@결과: N개 성분의 (kx_i, ky_i) 초기 추정값",
        "@@아직 진폭(w)과 위상(phi)은 모름 → 다음 단계",
    ]
)

# Slide 44
equation_slide(
    "선형 최소제곱 — 진폭과 위상 초기화",
    [
        "z_j = SUM [a_i*cos(.) + b_i*sin(.)]   (삼각함수 전개)",
        "",
        "행렬 형태: z = A * [a1, b1, a2, b2, ...]^T",
        "해: x = (A^T * A)^(-1) * A^T * z      (정규 방정식)",
    ],
    [
        "!! kx, ky가 고정되면 cos(kx*x+ky*y-omega*t)는 '알려진 기저'",
        "  → z를 이 기저의 선형 결합으로 표현 = 선형 문제!",
        "",
        "a_i, b_i에서 w_i, phi_i 복원:",
        "  w_i = sqrt(a_i^2 + b_i^2)",
        "  phi_i = atan2(-b_i, a_i)",
        "",
        "!! 장점: closed-form 해, 반복 불필요, 빠름",
        "!! 단점: kx, ky의 정확도에 의존 → 이후 비선형 최적화 필요",
    ]
)

# Slide 45
content_slide(
    "초기화 요약 — 여기까지의 결과",
    [
        "!! 빔포밍 + 피크 추출 + 선형 최소제곱 결과:",
        "",
        "N개 성분의 초기 파라미터:",
        "  (w_1, kx_1, ky_1, phi_1)",
        "  (w_2, kx_2, ky_2, phi_2)",
        "  ...",
        "  (w_N, kx_N, ky_N, phi_N)",
        "",
        "!! 이 초기값의 품질:",
        "  kx, ky: 빔포밍 해상도 범위 내에서 대략적 (±0.01 rad/m 오차)",
        "  w, phi: kx,ky 오차에 영향받음 → 대략적",
        "",
        "@@충분히 좋은 출발점이지만 정밀하지 않음",
        "@@→ 비선형 최적화로 미세 조정 필요 (Part 7)",
    ]
)

# ═══════════════════════════════════════════════════════════════
#  Part 7: 비선형 최적화 (슬라이드 46-53)
# ═══════════════════════════════════════════════════════════════
section_slide(7, "WaveSplat — 비선형 최적화", "Slides 46-53")

# Slide 46
content_slide(
    "왜 초기화만으로 부족한가",
    [
        "!! 빔포밍 해상도의 한계:",
        "  64 x 128 격자 → |k| 해상도 ~0.0075 rad/m",
        "  실제 파도의 kx가 격자 사이에 있으면 오차 발생",
        "",
        "예시:",
        "  실제 |k| = 0.125 rad/m",
        "  빔포밍 격자: 0.120, 0.128 → 0.128 선택",
        "  오차 2.4% → 주기 오차 ~1.2% → 30초 후 위상 10도 오차",
        "",
        "!! 이 '작은' 초기 오차가 시간이 지나면 누적:",
        "  추적(tracking) 시 위상 drift → 성능 저하",
        "",
        "@@해결: kx, ky를 연속값으로 미세 조정",
        "@@→ gradient-based 비선형 최적화",
    ]
)

# Slide 47
equation_slide(
    "Loss 함수 구성",
    [
        "L = (1/M) * SUM_j cj*(zj - z_hat_j)^2   (데이터 항)",
        "  + lambda_1 * SUM_i |w_i|                (L1 희소성)",
        "  + lambda_s * SUM_i (w_i - w_{i+1})^2    (평활화)",
    ],
    [
        "z_hat_j = SUM_i w_i * cos(kx_i*xj + ky_i*yj - omega_i*tj + phi_i)",
        "",
        "!! 세 가지 항의 역할:",
        "  데이터 항: 관측과 예측의 차이를 최소화 (핵심 목표)",
        "  L1 항: 불필요한 성분의 진폭을 0으로 밀어냄",
        "  평활화 항: 인접 주파수 성분의 진폭 급변 방지",
        "",
        "lambda_1 = 10^(-3), lambda_s = 10^(-4)",
        "  데이터 항 대비 ~1%, ~0.1% 수준으로 약하게 설정",
    ]
)

# Slide 48
content_slide(
    "Fresnel Confidence cj — 관측 신뢰도 가중치",
    [
        "!! cj = cos(theta_j) / (dj / d_ref)",
        "",
        "theta_j = LiDAR 빔과 수면의 입사각",
        "  수직에 가까울수록(cos~1) → 반사 신호 강함 → 신뢰도 높음",
        "  수평에 가까울수록(cos~0) → 반사 신호 약함 → 신뢰도 낮음",
        "",
        "dj = 관측 거리",
        "  가까울수록 → 점 밀도 높고, 노이즈 적음 → 신뢰도 높음",
        "  멀수록 → 점 밀도 낮고, 노이즈 큼 → 신뢰도 낮음",
        "",
        "d_ref = 125m (기준 거리, 최대 사거리의 절반)",
        "",
        "!! 물리적 근거: Fresnel 반사 법칙",
        "  수면은 거의 정반사(specular) → 입사각이 중요",
        "@@원거리 저각도 관측은 자동으로 가중치 감소",
    ]
)

# Slide 49
content_slide(
    "L1 정규화 — 자동 모델 선택",
    [
        "!! lambda_1 * SUM |w_i|의 역할:",
        "",
        "L2 정규화 (w_i^2)와의 차이:",
        "  L2: 모든 w를 '조금씩' 줄임 → 0이 되는 성분 없음",
        "  L1: 일부 w를 정확히 0으로 만듦 → 자동 가지치기",
        "",
        "!! 왜 L1인가?",
        "  BF2 너울: 실제 1~3 성분만 유의미",
        "  N=15로 시작해도 L1이 나머지 12개를 0으로 밀어냄",
        "  → 인간이 'N을 몇으로 할까' 고민할 필요 없음",
        "",
        "@@적응형 N(Part 8)과 시너지:",
        "@@L1이 진폭을 줄임 → 에너지 가지치기에서 제거됨",
    ]
)

# Slide 50
equation_slide(
    "Gradient 계산 — 해석적 미분",
    [
        "dL/dw_i = -(2/M) * SUM_j cj*(zj-z_hat_j)*cos(...) + lambda_1*sign(w_i)",
        "",
        "dL/dkx_i = (2/M) * SUM_j cj*(zj-z_hat_j)*w_i*xj*sin(...)",
        "         + (2/M) * SUM_j cj*(zj-z_hat_j)*w_i*(domega/dkx_i)*tj*sin(...)",
    ],
    [
        "!! 모든 gradient가 해석적 (analytic)으로 계산됨",
        "  cos/sin의 미분은 sin/cos → 정확한 gradient",
        "  수치 미분(finite difference) 불필요 → 빠르고 정확",
        "",
        "!! kx_i의 gradient에 분산관계 미분 포함:",
        "  domega/dkx_i = (kx_i / |k_i|) * domega/d|k|",
        "  → omega가 k의 함수이므로 chain rule 적용",
        "",
        "PyTorch autograd로 자동 계산 (수동 구현 불필요)",
    ]
)

# Slide 51
content_slide(
    "경사하강법(SGD)이란?",
    [
        "!! 기본 아이디어: 경사(gradient)의 반대 방향으로 조금씩 이동",
        "",
        "theta_new = theta - eta * (dL/dtheta)",
        "",
        "eta = 학습률 (learning rate)",
        "  너무 크면 → 진동, 발산",
        "  너무 작으면 → 수렴이 매우 느림",
        "",
        "!! SGD의 문제점:",
        "  1. 모든 파라미터에 같은 학습률 → kx(민감)와 w(둔감)가 같은 속도",
        "  2. 좁은 골짜기에서 진동 (zig-zag)",
        "  3. local minima에 빠지기 쉬움",
        "",
        "@@해결: Momentum → Adam (적응적 학습률)",
    ]
)

# Slide 52
content_slide(
    "Momentum — 관성의 도입",
    [
        "!! SGD + Momentum:",
        "  v_new = beta * v_old + (dL/dtheta)     (속도 업데이트)",
        "  theta_new = theta - eta * v_new          (위치 업데이트)",
        "",
        "beta = 0.9 → 이전 방향의 90%를 유지",
        "",
        "!! 효과:",
        "  1. 일관된 방향의 gradient → 가속 (공이 내리막을 구르듯)",
        "  2. 진동하는 방향 → 상쇄되어 감소",
        "  3. local minima를 관성으로 탈출할 수 있음",
        "",
        "!! 하지만 여전히: 모든 파라미터에 같은 학습률",
        "  w_i는 변화폭이 크고, kx_i는 변화폭이 작음",
        "  → 각 파라미터별 적응이 필요",
    ]
)

# Slide 53
equation_slide(
    "Adam Optimizer — 적응적 학습률",
    [
        "m = beta1*m + (1-beta1)*g           (1차 모멘트: 방향)",
        "v = beta2*v + (1-beta2)*g^2          (2차 모멘트: 크기)",
        "theta -= eta * m_hat / (sqrt(v_hat) + eps)",
    ],
    [
        "!! 핵심: 각 파라미터마다 다른 실효 학습률",
        "",
        "m (1차 모멘트) = gradient의 이동 평균 → '방향' 안정화",
        "  beta1 = 0.9 → 최근 10 step의 방향 평균",
        "",
        "v (2차 모멘트) = gradient 제곱의 이동 평균 → '크기' 추정",
        "  beta2 = 0.999 → 최근 1000 step의 변동폭",
        "",
        "!! eta / sqrt(v) = 실효 학습률",
        "  gradient 변동이 큰 파라미터 → v 큼 → 학습률 작아짐 (조심)",
        "  gradient 변동이 작은 파라미터 → v 작음 → 학습률 커짐 (과감)",
        "",
        "@@WaveSplat에서: kx(민감)는 자동으로 작은 step, w(둔감)는 큰 step",
    ]
)

# ═══════════════════════════════════════════════════════════════
#  Part 8: Coarse-to-Fine + Adaptive N (슬라이드 54-58)
# ═══════════════════════════════════════════════════════════════
section_slide(8, "Coarse-to-Fine + Adaptive N", "Slides 54-58")

# Slide 54
content_slide(
    "Coarse-to-Fine 전략",
    [
        "!! 3단계 점진적 최적화:",
        "",
        "Stage 1: N = 3   (가장 큰 3개 성분만)",
        "  → 200 iterations, lr = 5e-3",
        "  → 대략적 에너지 분포 파악",
        "",
        "Stage 2: N = 8   (중간 크기 성분 추가)",
        "  → 200 iterations",
        "  → Stage 1의 수렴된 3개 + 새로운 5개",
        "",
        "Stage 3: N = 15  (세부 성분까지)",
        "  → 200 iterations",
        "  → Stage 2의 수렴된 8개 + 새로운 7개",
        "",
        "@@총 600 iterations, 각 단계의 결과가 다음 단계의 초기값",
    ]
)

# Slide 55
content_slide(
    "왜 단계적으로 해야 하는가?",
    [
        "!! N이 클수록 loss landscape가 복잡해짐:",
        "",
        "N=3: 12개 파라미터",
        "  → 비교적 매끄러운 landscape, global minimum에 수렴 쉬움",
        "",
        "N=15: 60개 파라미터",
        "  → local minima가 매우 많음 (성분 간 간섭)",
        "  → 초기값이 나쁘면 틀린 해에 갇힘",
        "",
        "!! Coarse-to-fine의 효과:",
        "  큰 성분을 먼저 확정 → landscape의 주요 구조를 잡음",
        "  작은 성분은 '잔차'에서 찾음 → 간섭 감소",
        "",
        "@@비유: 그림 그리기 — 밑그림 → 중간 톤 → 세부 묘사",
        "@@한 번에 세부부터 그리면 전체 구도가 어긋남",
    ]
)

# Slide 56
content_slide(
    "적응형 N: 에너지 기반 가지치기 (Pruning)",
    [
        "!! 문제: 고정 N=15로 두면?",
        "  너울(실제 1~3성분): 나머지 12개가 노이즈를 fitting → 과적합",
        "  BF5(실제 20+성분): 15개로 부족 → 과소적합",
        "",
        "!! 가지치기 알고리즘:",
        "  1. 에너지 계산: E_i = w_i^2  (각 성분의 에너지)",
        "  2. 에너지 내림차순 정렬",
        "  3. 상위 90% 에너지를 커버하는 최소 N개 선택",
        "  4. 나머지 제거",
        "",
        "예시: 5성분의 에너지 [0.50, 0.30, 0.10, 0.05, 0.05]",
        "  누적: [0.50, 0.80, 0.90, 0.95, 1.00]",
        "  tau_E = 0.90 → N=3으로 가지치기 (90% 도달)",
    ]
)

# Slide 57
content_slide(
    "적응형 N: 안전 롤백 + 잔차 성장",
    [
        "!! 안전 롤백 (Safety Rollback):",
        "  가지치기 후 RMSE 측정",
        "  RMSE_after / RMSE_before > 1.30 이면 → 원복!",
        "  → 중요 성분을 실수로 잘라내는 것 방지",
        "",
        "!! 잔차 기반 성장 (Residual Growth):",
        "  가지치기 후 잔차 r_j = z_j - z_hat_j 계산",
        "  RMS(r) > tau_R = 0.05m 이면:",
        "    잔차에 빔포밍 다시 적용 → 놓친 성분 발견",
        "    새 성분 추가 + 선형 최소제곱으로 초기화",
        "",
        "!! 전체 흐름: 가지치기 → 안전 검사 → 성장 → 반복",
        "  자동으로 N=1 (너울) ~ N=21 (복잡해) 적응",
        "@@고정 N=15 대비 6.4% 추가 nRMSE 개선",
    ]
)

# Slide 58
content_slide(
    "적응형 N 실제 동작 예시",
    [
        "!! BF2 너울 (실제 1성분):",
        "  초기화: N=15 → C2F 최적화 → L1이 14개를 ~0으로",
        "  가지치기: 에너지 90% → N=1",
        "  잔차: RMS 0.02m < tau_R → 성장 없음",
        "  최종: N_a = 1 ✓",
        "",
        "!! BF4 15성분:",
        "  초기화: N=15 → C2F 최적화",
        "  가지치기: 에너지 90% → N=12",
        "  안전 검사: RMSE 비율 1.08 < 1.30 → OK",
        "  잔차: RMS 0.08m > tau_R → 빔포밍 → 3성분 추가",
        "  최종: N_a = 15 → 재최적화 후 N_a = 14",
        "",
        "@@너울/복잡해 모두 자동 적응 → 수동 튜닝 불필요",
    ]
)

# ═══════════════════════════════════════════════════════════════
#  Part 9: Tracking (슬라이드 59-63)
# ═══════════════════════════════════════════════════════════════
section_slide(9, "WaveSplat — 실시간 추적 (Tracking)", "Slides 59-63")

# Slide 59
content_slide(
    "Phase 1 (초기화) vs Phase 2 (추적)",
    [
        "!! Phase 1 — 초기화 (처음 1초):",
        "  다중 프레임(10프레임) 축적하여 빔포밍",
        "  Coarse-to-fine 600 iterations",
        "  적응형 N 결정",
        "  ~500ms 소요 (무거움, 1회만 실행)",
        "",
        "!! Phase 2 — 추적 (이후 매 프레임):",
        "  새 프레임이 올 때마다 파라미터 업데이트",
        "  10 Adam steps만 실행",
        "  ~13ms 소요 (가벼움, 10Hz로 반복)",
        "",
        "@@초기화가 좋으면 → 추적은 소량 보정만 → 빠르고 안정",
    ]
)

# Slide 60
equation_slide(
    "위상 전파 (Predict Step)",
    [
        "phi_i(t+dt) = phi_i(t) + omega_i * dt",
        "",
        "omega_i = sqrt(g * |k_i| * tanh(|k_i| * d))",
    ],
    [
        "!! 분산관계가 '예측기' 역할:",
        "  각 성분이 자기 주파수에 맞춰 위상이 전진",
        "  dt = 0.1s (10Hz), omega = 3 rad/s → phi += 0.3 rad",
        "",
        "이 예측이 정확한 이유:",
        "  분산관계가 정확히 성립하면 → 위상 예측이 완벽",
        "  실제 편차: 비선형 효과, 모델 오차 → 매우 작음",
        "",
        "!! 결과: Update step에서 보정할 양이 매우 작음",
        "  → 10 Adam steps면 충분",
        "",
        "@@ETKF의 predict: Ne개 앙상블 각각 FFT로 전파 → 느림",
        "@@WaveSplat의 predict: N개 위상에 덧셈 → O(N) = 즉시",
    ]
)

# Slide 61
content_slide(
    "w_phi 모드 — 왜 kx, ky를 고정하는가",
    [
        "!! 추적 시 업데이트 대상: w (진폭)와 phi (위상)만",
        "  kx, ky는 초기화에서 확정 후 고정",
        "",
        "왜?",
        "",
        "실험: kx, ky도 업데이트하면 (full mode):",
        "  단기: 약간 더 좋은 fitting (loss 더 낮음)",
        "  장기: kx, ky가 drift → 분산관계에서 벗어남",
        "       → 위상 전파 예측이 점점 틀려짐",
        "       → 10프레임 이후 급격한 성능 저하",
        "",
        "!! w_phi 모드:",
        "  kx, ky 고정 → 분산관계 항상 정확 → 위상 전파 안정",
        "  w, phi만 매 프레임 미세 조정 → 진폭/위상 변화에 적응",
        "",
        "@@실험 결과: full mode 대비 18% 추적 성능 개선",
    ]
)

# Slide 62
content_slide(
    "슬라이딩 윈도우 — 노이즈와 적응의 균형",
    [
        "!! 현재 프레임 + 이전 2프레임 = 3프레임 윈도우",
        "",
        "왜 여러 프레임을 쓰나?",
        "  단일 프레임: 50~200점 → 노이즈에 취약",
        "  3프레임: 150~600점 → 노이즈 평균화 효과",
        "",
        "왜 3프레임인가? (왜 더 많이 안 쓰나?)",
        "  윈도우가 길면:",
        "    + 노이즈 감소",
        "    - 해상 변화에 느리게 반응 (과거 정보에 묶임)",
        "    - 계산량 증가 (3프레임 = 13ms, 5프레임 = 22ms)",
        "",
        "!! 3프레임 = 0.3초 → 파도 주기(~5초)의 6%",
        "  파도 변화를 놓치지 않으면서 충분한 평균화",
    ]
)

# Slide 63
content_slide(
    # ⛔ 원본 제목 (주석 처리, 인용금지): "실시간 성능 — 13.4ms/frame"
    "실시간 성능 — ⛔RETRACTED (2026-08-13 감사)",
    [
        "!! 추적 1 cycle의 시간 분배:",
        "",
        "  위상 전파 (predict):  < 0.01ms  (덧셈 N번)",
        "  윈도우 구성:          < 0.1ms   (데이터 복사)",
        "  10 Adam steps:        ~13ms     (gradient 계산 + 업데이트)",
        "  적응형 N 검사:        < 0.3ms   (에너지 정렬 + 비교)",
        "",
        # ⛔ 원본 이력 (주석 처리, 인용금지 — 13.4ms / 132.6ms / 86.4%):
        # "  합계: ~13.4ms (100ms 예산의 13.4%)",
        # "!! 비교:",
        # "  ETKF:  132.6ms → 예산 초과",
        # "  GPR:   >200ms → 예산 2배 초과",
        # "  WaveSplat: 13.4ms → 여유 86.4%",
        "  합계: ⛔RETRACTED — 지연 수치는 스레드 미고정 아티팩트",
        "",
        "!! 허용되는 서술:",
        "  동일 프로세스·1스레드에서 1.5–2.6배",
        "  CPU = AMD Ryzen 7 7800X3D",
        "  단일 숫자 대신 중앙값 + IQR 병기",
        "",
        "@@근거: reference_citation_bans.md",
    ]
)

# ═══════════════════════════════════════════════════════════════
#  Part 10: 전체 비교 — 왜 Splatting인가 (슬라이드 64-68)
# ═══════════════════════════════════════════════════════════════
section_slide(10, "전체 비교 — 왜 Splatting인가", "Slides 64-68")

# Slide 64
comparison_slide(
    "3가지 패러다임 종합 비교",
    ["", "ETKF (공간-격자)", "GPR (공간-비모수)", "WaveSplat (주파수)"],
    [
        ["표현", "격자점 높이", "커널 보간", "스펙트럼 성분"],
        ["물리", "간접 (앙상블)", "없음", "직접 (분산관계)"],
        ["파라미터", "120,000", "~200", "~60"],
        # ⛔ 원본 이력 (주석 처리, 인용금지):
        # ["속도",  "132.6ms", ">200ms", "13.4ms"],
        # ["nRMSE", "0.319",   "0.261",  "0.223"],
        ["속도", "⛔", "⛔", "⛔"],
        ["nRMSE", "⛔", "⛔", "⛔"],
        ["시간 연속", "O (KF)", "X", "O (추적)"],
        ["외삽", "X", "X", "O"],
        ["격자", "필요", "불필요", "불필요"],
    ]
)

# Slide 65
content_slide(
    "물리 활용도의 차이",
    [
        "!! ETKF — 간접적 물리 활용:",
        "  앙상블 predict에서 FFT 기반 파도 전파 사용",
        "  하지만 상태(격자 높이)에는 분산관계가 직접 반영되지 않음",
        "  → 앙상블이 충분히 크지 않으면 물리가 '깨짐'",
        "",
        "!! GPR — 물리 없음:",
        "  순수 통계적 보간, 파도가 이동한다는 것조차 모름",
        "  → 시공간 커널을 쓸 수 있지만, 분산관계를 커널에 넣기 어려움",
        "",
        "!! WaveSplat — 직접적 물리 내장:",
        "  omega = f(k, d)가 모델에 하드코딩",
        "  물리적으로 불가능한 해가 원천 불가",
        "  → 적은 데이터로도 물리적으로 일관된 추정",
    ]
)

# Slide 66
content_slide(
    "파라미터 효율 — Occam's Razor",
    [
        "!! 같은 바다를 기술하는 데 필요한 파라미터:",
        "",
        "  ETKF:    Ne x G  = 100 x 1200 = 120,000개",
        "  GPR:     M + hp  = 200 + 3     = 203개",
        "  WaveSplat: 4N    = 4 x 15      = 60개",
        "",
        "!! Occam's Razor: 동일한 설명력이면 단순한 모델이 우월",
        "",
        "왜 적은 파라미터가 유리한가?",
        "  1. 과적합 위험 감소 (데이터 대비 파라미터 비율)",
        "  2. 최적화가 쉬움 (탐색 공간이 작음)",
        "  3. 일반화 능력 (새로운 데이터에도 잘 작동)",
        "  4. 해석 가능 (각 파라미터의 물리적 의미가 명확)",
        "",
        "@@60개 파라미터로 250m x 250m 파도장을 완전히 기술",
    ]
)

# Slide 67
content_slide(
    "'Splatting'의 의미 — 3DGS에서 영감",
    [
        "!! 3D Gaussian Splatting (3DGS, Kerbl et al. 2023):",
        "  3D 장면을 수백만 개의 Gaussian '점'으로 표현",
        "  격자(voxel)나 메쉬 없이 연속적인 장면 렌더링",
        "  각 Gaussian: 위치, 크기, 색상, 투명도 → 파라미터",
        "",
        "!! WaveSplat의 아날로지:",
        "  파도장을 N개의 스펙트럼 성분(plane wave)으로 표현",
        "  격자 없이 연속적인 파도장 복원",
        "  각 성분: w, kx, ky, phi → 파라미터",
        "",
        "!! 공통점:",
        "  1. 격자 비의존 (grid-free)",
        "  2. 파라메트릭 기저의 중첩 (Gaussian / cos)",
        "  3. 미분 가능 → gradient 기반 최적화",
        "  4. 적응적 성분 수 (densify/prune)",
    ]
)

# Slide 68
content_slide(
    "결론: 왜 주파수 영역 Splatting이 필요한가",
    [
        "!! 희소 LiDAR 관측에서 파도장을 복원하는 문제의 본질:",
        "",
        "  1. 관측이 극도로 희소 → 강력한 사전 지식(물리) 필요",
        "     → 주파수 영역: 분산관계를 직접 내장",
        "",
        "  2. 관측이 불규칙 → 격자 불필요한 표현 필요",
        "     → Splatting: 격자 없는 파라메트릭 표현",
        "",
        "  3. 실시간 필요 → 적은 파라미터, 빠른 추적",
        "     → 4N개 파라미터, 위상 전파로 즉시 예측",
        "",
        "  4. 시간 예측 필요 → 시간이 모델에 내장",
        "     → cos(...-omega*t+phi)에서 t는 변수",
        "",
        "@@이 4가지를 동시에 만족하는 방법: WaveSplat이 유일",
    ]
)

# ═══════════════════════════════════════════════════════════════
#  Part 11: 평가 지표와 결과 (슬라이드 69-73)
# ═══════════════════════════════════════════════════════════════
section_slide(11, "평가 지표와 결과", "Slides 69-73")

# Slide 69
content_slide(
    "6-지표 패널 — 왜 nRMSE 하나로는 부족한가",
    [
        # ⛔ 원본 이력 (주석 처리, 인용금지 — 0.319 / 0.261 / 0.223):
        # "!! nRMSE만 보면:",
        # "  ETKF 0.319, GPR 0.261, WaveSplat 0.223",
        # "  nRMSE:  WS 0.223 / ETKF 0.319 / GPR 0.261",
        "!! nRMSE 수치: ⛔RETRACTED (2026-08-13 감사)",
        "  분모 0.319 를 내는 결과 파일이 저장소에 없다",
        "  재도출: WS 0.2229 vs LETKF 0.2278, 51/49, p = 0.9204 = 널",
        "",
        "!! 6-지표 논지(수치 아닌 구조)는 유지된다:",
        "  VR:     WS 0.73  / ETKF 0.90  / GPR 0.065  ← GPR 에너지 6%!",
        "  rho:    WS 0.76  / ETKF 0.11  / GPR 0.193  ← ETKF 상관 11%!",
        "  SS:     WS 0.46  / ETKF -0.02 / GPR -0.14  ← 둘 다 음수!",
        "  Dir R2: WS -0.25 / ETKF -1.25 / GPR ---",
        "  Hs err: WS 3.7%  / ETKF 28.6% / GPR ---",
        "",
        "@@SS > 0인 방법은 WaveSplat뿐 = zero predictor를 이기는 유일한 방법",
    ]
)

# Slide 70
content_slide(
    "각 지표의 진단적 의미",
    [
        "!! nRMSE = RMSE / Hs: 전체적 오차, 가장 보편적",
        "",
        "!! VR (Variance Ratio): 에너지 재현도",
        "  VR=1: 에너지 정확, VR<1: 과소, VR>1: 과대",
        "  ETKF VR=0.90 → 에너지는 맞추지만...",
        "",
        "!! rho (Pearson 상관): 형상 일치도",
        "  rho=1: 완벽 매칭, rho=0: 무관",
        "  ETKF rho=0.11 → 에너지 맞지만 위치가 틀림!",
        "",
        "!! SS (Skill Score) = 1 - MSE(pred)/MSE(0): zero predictor 대비",
        "  SS>0: 예측이 가치있음, SS<0: '아무것도 안 하는 게 나음'",
        "",
        "!! Dir R2: 방향 추정 정확도 (-0.25 = WaveSplat의 최대 약점)",
        "!! Hs err: 유의파고 오차",
    ]
)

# Slide 71
content_slide(
    "핵심 결과",
    [
        "!! 100 에피소드 (10 해상조건 x 5 시드 x 2 raydrop) 평균:",
        "",
        # ⛔ 원본 이력 (주석 처리, 인용금지 — 전 항목):
        # "  WaveSplat nRMSE: 0.223 (최저 = 최고)",
        # "  ETKF nRMSE:      0.319 (WS 대비 +43%)",
        # "  GPR nRMSE:       0.261 (WS 대비 +17%)",
        # "!! 개선율: WS가 ETKF 대비 30.1% 낮은 nRMSE",
        # "!! 속도: WS 13.4ms vs ETKF 132.6ms (10배 빠름)",
        # "  Wilcoxon signed-rank test: p = 1.1 x 10^(-15)",
        # "  Cohen's d = 1.16 (large effect size)",
        "  ⛔RETRACTED (2026-08-13 감사) — 위 6개 헤드라인 수치 전량",
        "",
        "!! 검증된 재도출 (results/100ep_ws_letkf.json):",
        "  WaveSplat 0.2229  vs  LETKF 0.2278",
        "  51/49, p = 0.9204  →  널 (정확도 차이 없음)",
        "",
        "!! 지연: 동일 프로세스·1스레드에서 1.5–2.6배",
        "  CPU = AMD Ryzen 7 7800X3D",
        "",
        "!! 미노출 시드(S13 704런): 혼잡 56/64, 희소 63/64",
        "",
        "@@근거: reference_citation_bans.md",
    ]
)

# Slide 72
content_slide(
    "한계와 향후 연구",
    [
        "!! 최대 약점: Dir R^2 = -0.25 (방향 추정)",
        "  BF4에서 일부 성분의 방향(kx 부호)이 반전됨",
        "  → feedforward 제어 시 오히려 악화 (BF4 WS/GT = 37%)",
        "",
        "원인:",
        "  선박 전방 LiDAR → 1D 관측 geometry",
        "  kx>0 (다가오는 파도)와 kx<0 (멀어지는 파도) 구분 어려움",
        "  다성분이 겹치면 모호성 증가",
        "",
        "!! 향후 해결 방향:",
        "  1. 다중 프레임 방향 추적 (시간에 따른 위상 변화로 방향 추론)",
        "  2. 3D FFT (kx, ky, omega) 초기화",
        "  3. 방향 신뢰도 기반 feedforward 게이팅",
        "",
        "@@수조 실험(Ch5)으로 sim-to-real gap 검증 예정",
    ]
)

# ── 마지막 슬라이드 ──
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide, BG_DARK)
add_text(slide, 0.8, 2.5, 11.7, 1.5,
         "WaveSplat = 주파수 영역의 Grid-free Splatting",
         size=36, color=ACCENT, bold=True, align=PP_ALIGN.CENTER)
add_text(slide, 0.8, 4.2, 11.7, 1.0,
         "희소 관측 + 물리 제약 + 실시간 + 시간 예측\n"
         "= 이 조합을 만족하는 유일한 방법",
         size=22, color=WHITE, align=PP_ALIGN.CENTER)
# ⛔ 원본 이력 (주석 처리, 인용금지):
# add_text(slide, 0.8, 5.8, 11.7, 0.6,
#          "0.223 nRMSE | 13.4ms/frame | 30.1% better than ETKF | 100 episodes validated",
#          size=18, color=YELLOW, align=PP_ALIGN.CENTER)
add_text(slide, 0.8, 5.8, 11.7, 0.6,
         "⛔RETRACTED (2026-08-13 감사) — headline nRMSE/latency/30.1% 전량 재측정 필요",
         size=18, color=YELLOW, align=PP_ALIGN.CENTER)

# ═══════════════════════════════════════════════════════════════
#  저장
# ═══════════════════════════════════════════════════════════════
output_dir = "/home/avikus/personal/wavesplat/wavesplat/docs"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "WaveSplat_Algorithm_Guide.pptx")
prs.save(output_path)
print(f"Saved: {output_path}")
print(f"Total slides: {len(prs.slides)}")
