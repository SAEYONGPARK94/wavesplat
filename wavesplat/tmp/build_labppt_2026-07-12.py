"""
Build docs/labmeeting_2026-07-12.pptx from the slide plan.
16:9, Korean-friendly (Malgun Gothic), embeds real PNGs from results/direction3/.
No commit. Run: python3 tmp/build_labppt_2026-07-12.py
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "direction3")
OUT = os.path.join(ROOT, "docs", "labmeeting_2026-07-12.pptx")

NAVY = RGBColor(0x1F, 0x2D, 0x5A)
TEAL = RGBColor(0x0F, 0x9C, 0x7A)
GREY = RGBColor(0x44, 0x44, 0x44)
LGREY = RGBColor(0x88, 0x88, 0x88)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Malgun Gothic"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]


def _set_font(run, size, color=GREY, bold=False, italic=False):
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = FONT


def add_title_slide():
    s = prs.slides.add_slide(BLANK)
    # navy band
    band = s.shapes.add_shape(1, 0, Inches(2.2), SW, Inches(2.7))
    band.fill.solid(); band.fill.fore_color.rgb = NAVY
    band.line.fill.background()
    tf = band.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.7); tf.margin_right = Inches(0.7)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
    r = p.add_run(); r.text = "주어진 상용 LiDAR 사양으로\n실해역 파랑장 추정·조우예보"
    _set_font(r, 30, WHITE, bold=True)
    p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.LEFT
    r2 = p2.add_run()
    r2.text = "정보 회계로 한계를 확정하고, 학습이 이기는 자리를 창-수준 인과로 분리"
    _set_font(r2, 15, RGBColor(0xCF, 0xE0, 0xF0), italic=True)
    # sub
    tb = s.shapes.add_textbox(Inches(0.7), Inches(5.3), Inches(12), Inches(1.6))
    tf = tb.text_frame; tf.word_wrap = True
    for i, (txt, sz, col, bd) in enumerate([
        ("박세용 · 서울대학교 조선해양공학과 · 2026-07-12", 15, GREY, True),
        ("연구실 발표 — 방향 승인용 (지도교수·연구실)", 14, LGREY, False),
        ("요청: 연구 방향 승인 · 수조 최소 실증 1캠페인 승인 · 집필 일정 합의", 14, TEAL, True),
    ]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = txt; _set_font(r, sz, col, bold=bd)


def add_content(title, bullets, img=None, notes="", accent=NAVY, section=None):
    s = prs.slides.add_slide(BLANK)
    # top accent bar
    bar = s.shapes.add_shape(1, 0, 0, SW, Inches(0.16))
    bar.fill.solid(); bar.fill.fore_color.rgb = accent; bar.line.fill.background()
    # section tag
    top = Inches(0.32)
    if section:
        tb = s.shapes.add_textbox(Inches(0.55), top, Inches(12), Inches(0.35))
        p = tb.text_frame.paragraphs[0]; r = p.add_run(); r.text = section
        _set_font(r, 12, TEAL, bold=True)
        top = Inches(0.66)
    # title
    tb = s.shapes.add_textbox(Inches(0.55), top, Inches(12.3), Inches(0.9))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; r = p.add_run(); r.text = title
    _set_font(r, 23, NAVY, bold=True)
    # body geometry
    body_top = Inches(1.7)
    if img and os.path.exists(img):
        text_w = Inches(6.0)
        img_left = Inches(6.85)
        img_w = Inches(6.0)
    else:
        text_w = Inches(12.2)
        img_left = None
    # bullets
    tb = s.shapes.add_textbox(Inches(0.55), body_top, text_w, Inches(5.3))
    tf = tb.text_frame; tf.word_wrap = True
    for i, b in enumerate(bullets):
        # b may be tuple (level, text, bold) or str
        if isinstance(b, tuple):
            level, txt, bold = b
        else:
            level, txt, bold = 0, b, False
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = level
        p.space_after = Pt(6)
        bullet = ("• " if level == 0 else "   – ")
        r = p.add_run(); r.text = bullet + txt
        _set_font(r, 15 if level == 0 else 13.5,
                  NAVY if bold else GREY, bold=bold)
    # image
    if img and os.path.exists(img):
        from PIL import Image
        try:
            iw, ih = Image.open(img).size
            ar = ih / iw
        except Exception:
            ar = 0.62
        w = img_w
        h = Emu(int(w * ar))
        maxh = Inches(5.0)
        if h > maxh:
            h = maxh; w = Emu(int(h / ar))
        pic_top = body_top + Inches(0.1)
        s.shapes.add_picture(img, img_left, pic_top, width=w, height=h)
    elif img:
        # placeholder box
        ph = s.shapes.add_shape(1, Inches(6.85), Inches(2.0), Inches(6.0), Inches(3.6))
        ph.fill.solid(); ph.fill.fore_color.rgb = RGBColor(0xEE, 0xEE, 0xEE)
        ph.line.color.rgb = LGREY
        tf = ph.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = "[그림 자리표시 — 제작 중]\n" + os.path.basename(img)
        _set_font(r, 12, LGREY, italic=True)
    # notes
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def add_table_slide(title, header, rows, notes="", section=None):
    s = prs.slides.add_slide(BLANK)
    bar = s.shapes.add_shape(1, 0, 0, SW, Inches(0.16))
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
    top = Inches(0.32)
    if section:
        tb = s.shapes.add_textbox(Inches(0.55), top, Inches(12), Inches(0.35))
        r = tb.text_frame.paragraphs[0].add_run(); r.text = section
        _set_font(r, 12, TEAL, bold=True); top = Inches(0.66)
    tb = s.shapes.add_textbox(Inches(0.55), top, Inches(12.3), Inches(0.8))
    r = tb.text_frame.paragraphs[0].add_run(); r.text = title
    _set_font(r, 23, NAVY, bold=True)
    nrows = len(rows) + 1; ncols = len(header)
    gt = s.shapes.add_table(nrows, ncols, Inches(0.55), Inches(1.6),
                            Inches(12.2), Inches(0.4 * nrows)).table
    for j, htxt in enumerate(header):
        c = gt.cell(0, j); c.text = htxt
        for para in c.text_frame.paragraphs:
            for run in para.runs:
                _set_font(run, 12, WHITE, bold=True)
        c.fill.solid(); c.fill.fore_color.rgb = NAVY
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            c = gt.cell(i, j); c.text = val
            for para in c.text_frame.paragraphs:
                for run in para.runs:
                    _set_font(run, 11, GREY)
            c.fill.solid()
            c.fill.fore_color.rgb = RGBColor(0xF2, 0xF5, 0xFA) if i % 2 else WHITE
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


# ─────────────────────────────────────────────────────────────
# SLIDES
# ─────────────────────────────────────────────────────────────
add_title_slide()

add_content(
    "문제·동기 (1): 통계 추정 vs 위상분해 추정",
    [
        (0, "산업 표준 = 통계 추정(H_s, T_p, 스펙트럼) — 설계엔 충분, 실시간 제어엔 부족", True),
        (0, "우리 목표 = 위상분해 파랑장 η(x,y,t) 직접 추정", False),
        (1, "\"지금 이 배 앞 어디에 어떤 파도가 오는가\"", False),
        (0, "응용 = 조우예보 기반 피드포워드 제어 (hydrofoil 등)", True),
        (1, "액추에이터가 파도를 만나기 전에 명령을 준비", False),
        (0, "핵심 제약(고정): 계산 지연 ≤ 100 ms, 지구-고정 좌표(선속 불변)", False),
    ],
    section="1. 문제·동기",
    notes="파도의 평균 통계가 아니라 위상이 붙은 파도장 자체를 실시간으로 추정하는 것이 목표. "
          "이유는 제어 — 수중익선이 파도를 만나기 전에 제어 명령을 준비하려면 앞바다 실제 파형이 필요. "
          "그래서 100 ms 예산과 실시간이 본질적 제약.")

add_content(
    "문제·동기 (2): 왜 어려운가 — \"주어진 센서\"의 제약",
    [
        (0, "센서는 주어짐: Seyond Falcon LiDAR 공표 사양(매뉴얼 p.12)", True),
        (1, "FOV 120°×25°, 각분해능 0.18°×0.24°, 250 m @10%, 2 cm 1σ, 10 FPS", False),
        (0, "관측 3중 난제", True),
        (1, "① 희소 전방-부채꼴 ② 실시간·짧은 창 ③ 2-D 방향(short-crested)", False),
        (0, "스냅샷 1장으론 방향 180° 모호 → 1–3 s 시간 버퍼(버퍼 ≠ 지연)", False),
        (0, "합의 작동점: 설계 선속 11.3 m/s (Candela C-8), 해상상태 BF2–BF5", True),
    ],
    section="1. 문제·동기",
    notes="센서를 우리가 설계하는 게 아니라 상용 사양이 주어진 상태에서 무엇이 가능한지를 역산하는 프레임. "
          "관측은 희소·실시간·2차원 방향까지 풀어야 해서 어렵다. 작동점은 11.3 m/s, BF2–BF5.")

add_content(
    "문제·동기 (3): 이 분야가 비어 있는 자리",
    [
        (0, "이 분야의 문제는 \"조금 된다\"가 아니라", True),
        (1, "\"얼마나 되는지 아무도 수치로 확정하지 않은 채 데모만 쌓인다\"", True),
        (0, "반복 함정: 수조 시연 → 실해역 재현 실패 · 평가 규약 의존 · 선택 편향", False),
        (0, "우리의 각도 = 재도출 가능한 수치로 답한다", True),
        (1, "검출 법칙 · 작동영역 · 조우예보 상한·도달 · 학습의 추가 몫과 그 경계", False),
        (0, "방법 novelty는 주장 안 함 (AHGP·DSWP·Ehlers·DFL 선점 인정·인용)", False),
    ],
    section="1. 문제·동기",
    notes="문헌엔 데모는 많은데 '정확히 얼마나 되는가'를 확정한 연구가 드물다. 성능이 규약에 출렁이고 "
          "실해역 재현이 안 된다. 방법이 새롭다고 주장하지 않고 원리 선점은 전부 인용. 자리는 "
          "'얼마나·왜·어디서 이기는가'를 수치로 확정하는 것.")

# Thesis
s = add_content(
    "Thesis Statement",
    [
        (0, "주어진 상용 LiDAR 사양으로 제어-호환(≤100 ms) 실시간 조우예보가", True),
        (0, "어디까지 가능한지를 정보 회계로 확정하고,", True),
        (0, "그 한계 안에서 학습이 이기는 자리(오지정 보정)와", True),
        (0, "못 이기는 자리(정보 한계)를 창-수준 인과로 분리해,", True),
        (0, "재현 가능한 배포 레시피와 인증-게이트 방법론으로 실용화한다.", True),
        (0, " ", False),
        (0, "세 층(회계·학습지도·레시피)이 서로의 검증 장치 — 프레임: small, certified learning", False),
        (0, "\"학습은 물리가 모르는 것을 채울 때만 이긴다\"", False),
    ],
    accent=TEAL, section="2. Thesis Statement",
    notes="오늘 승인받고 싶은 핵심 문장. 셋으로 쪼개면 각각 약함(회계=CRB, 학습자리=Ehlers 선점). "
          "강한 것은 연결 — 세 층이 서로를 검증하는 구조이고, 그 구조가 하루 만에 가짜 양성 7건을 "
          "사전등록 게이트로 걸러냈다.")

add_content(
    "방법 (1): 파이프라인 — 학습은 \"슬롯\"에만 들어간다",
    [
        (0, "관측(희소 점군) → beamform 파워맵 → γ-prior CNN → 분산관계 은행 → MMSE 사후평균", True),
        (1, "분산 하드제약: ω = √(g·|k|·tanh(|k|·d))", False),
        (0, "학습이 꽂히는 자리는 두 슬롯뿐", True),
        (1, "γ (진폭 스펙트럼 prior) — nowcast 담당", False),
        (1, "σ (잡음바닥 배율 smult) — forecast 담당", False),
        (0, "CNN은 장(場)을 직접 내지 않는다 — 파는 MMSE 사후평균(mean-collapse 회피)", True),
        (0, "CNN = 이 인터페이스의 최소 충분 인스턴스(아키텍처-불가지론적, 111× 가속)", False),
    ],
    section="3. 방법 개요",
    notes="뼈대는 물리 — 분산관계로 구속된 파수 은행 위 MMSE 사후평균이 파도장을 낸다. 학습은 딱 두 슬롯, "
          "진폭 prior γ와 잡음바닥 σ만 예측. 망이 장을 직접 그리면 평균 붕괴하므로 물리 사상만 망에 맡김. "
          "CNN은 최소 충분 구현.")

add_table_slide(
    "방법 (2): 검증 3무대 + \"실측 검증\" 정의 고정",
    ["무대", "참값", "관측", "역할"],
    [
        ["합성(선형)", "시뮬", "사양-모사", "회계 천장·오지정 없음"],
        ["HOS(비선형)", "고차 스펙트럼법", "사양-모사", "속박파 오지정 스트레스"],
        ["실측 스테레오", "IFREMER 스테레오 표면", "사양-모사 zero-shot", "분포-외 강건성"],
    ],
    section="3. 방법 개요",
    notes="세 무대 — 깨끗한 선형, HOS 비선형, 실측 스테레오에 사양-모사 연산자 zero-shot. 중요한 규율: "
          "'실측 검증'을 실물 선박 LiDAR가 아니라 이 정의로만 쓴다. 규약 반전 발견으로 벤치마크를 "
          "고정-임계(fixedT)로 재구성. 이 정직한 경계가 뒤 한계 슬라이드로 이어짐.")

# Evidence
add_content(
    "핵심 증거 (1): 3무대 오지정 기울기",
    [
        (0, "학습 이득(learned − hand)이 오지정 커질수록 단조 증가 (전장 corr)", True),
        (1, "깨끗한 선형(35 ep): −0.041 (학습 손해 — 정리대로 사후평균 최적)", False),
        (1, "HOS 비선형(8 셀): +0.070 (7/8, Hs1.2 +0.075 / Hs3.0 +0.064)", False),
        (1, "실측 스테레오(60창): +0.033", False),
        (0, "해석: bw-오지정으로 훈련된 net이 HOS 속박파 오지정을 정확히 채움", True),
        (1, "훈련 표적과 정합", False),
    ],
    img=os.path.join(FIG, "fig_gradient_2026-07-12.png"),
    section="4. 핵심 증거",
    notes="중심 명제의 정량 형태. 완벽한 선형에선 학습이 살짝 손해 — 정리(사후평균 최적)라 이길 수 없음. "
          "비선형·실측으로 오지정 커질수록 학습 이득 단조 증가. '물리가 모르는 것을 채울 때만 이긴다'가 그래프.")

add_content(
    "핵심 증거 (2): 실측 조우 트랙 판정 (군집-보정 생존)",
    [
        (0, "3 사이트 조우점 파고 예보:", True),
        (1, "La Jument(폭풍) ρ 0.97 · 황해 ρ 0.94 · Acqua Alta ρ 0.93", False),
        (0, "학습 vs 물리 (창-짝지음, 60창 = 5rec×12win)", True),
        (1, "전창 +0.088 (t_win 3.14, t_rec 4.59, 5/5)", False),
        (1, "물리 실패창(hand<0.5, n=25) +0.227 (t_win 4.75, t_rec 5.01, 5/5)", False),
        (0, "레코드-군집 부트스트랩 6/6 생존 — 창-수준 t 과대평가 공격 방어", True),
    ],
    img=os.path.join(FIG, "fig_ext1_track_match_2026-07-12.png"),
    section="4. 핵심 증거",
    notes="실측 세 사이트에서 배가 만날 미래 파형 예보 — 좋은 창은 상관 0.93–0.97. '60창은 사실 5레코드다' "
          "통계 공격 대비 군집 부트스트랩 6/6 생존. 레코드-수준 t가 창-수준과 동급 이상 = 효과 레코드-일관.")

add_content(
    "핵심 증거 (3): 전장(全場) 판정 — 본체는 트랙이 아니라 장",
    [
        (0, "문제의 본체 = 전장 파랑 추정. 조우 트랙은 그 장에 대한 하나의 질의", True),
        (0, "채점 부채꼴 전역(60창): learned 0.583 vs hand 0.550", True),
        (1, "Δ +0.033 (t_win 3.22, t_rec 3.83, 5/5, CI[0.018,0.048])", False),
        (0, "손해 없이 소폭 우위 (CI 하한 +0.018 > 0), 전 리드 우위", False),
        (0, "진폭비 learned 0.69× vs hand 0.89×", True),
        (1, "→ \"겉보기\" 논쟁의 실체 (다음 슬라이드)", False),
    ],
    img=os.path.join(FIG, "fig_field_verdict_2026-07-12.png"),
    section="4. 핵심 증거",
    notes="본체는 전장 추정, 트랙은 질의. 채점 부채꼴 전체에서도 학습이 손해 없이 소폭 우위, 신뢰구간 하한이 "
          "0을 넘음. 학습 팔의 진폭 수축(0.69배)이 다음 슬라이드 '왜 hand가 더 세세해 보이나' 논쟁으로.")

add_content(
    "핵심 증거 (4): \"자신있게 틀림 vs 겸손하게 맞음\"",
    [
        (0, "실패창 사례(AA14 w10): 이 창에선 hand가 장 corr +0.72 > learned +0.66", True),
        (1, "진폭도 hand 0.73× 더 뚜렷 → \"더 세세해 보임\"", False),
        (0, "그러나 배가 실제 만나는 조우점:", True),
        (1, "hand ρ −0.43 (역위상) vs learned ρ +0.32", False),
        (0, "기제: 물리는 비구속 성분까지 자신 있게 그려 위상이 틀림", False),
        (1, "학습은 장을 흐리는 대신 구속 구조의 위상을 산다", False),
        (0, "\"장-겉보기 ≠ 트랙-정확도\" — 겉보기 선호는 진폭 수축의 착시", True),
    ],
    img=os.path.join(FIG, "fig_ext2_field_vs_track_2026-07-12.png"),
    section="4. 핵심 증거",
    notes="지도교수 강조 지점. 어떤 창에선 물리가 눈으로 더 그럴듯하고 진폭도 뚜렷. 하지만 배가 만나는 점에서 "
          "물리는 위상이 반대(자신있게 틀림). 학습은 장을 조금 흐리는 대가로 그 점 위상을 맞춤(겸손하게 맞음). "
          "겉보기 선호는 진폭 수축 착시.")

add_content(
    "핵심 증거 (5): 기여 분해 — 가법·과제-의존",
    [
        (0, "4팔 창-짝지음(60창): 물리 0.53 → +바닥 0.60 → +γ-net 0.54 → +둘 다 0.62", True),
        (0, "바닥(σ) 단독 = forecast 승자: 전체 +0.073(t2.9) / 실패창 +0.159", False),
        (0, "γ-net 단독 = nowcast 승자(t4.42); forecast 단독 기여는 ns", False),
        (0, "상호작용 ≈ 0 (가법): 0.145 + 0.159 + 0.071 ≈ 0.375", True),
        (0, "정밀 결론: 승리는 과제-의존 — γ는 nowcast, 바닥은 forecast", True),
    ],
    img=os.path.join(FIG, "fig_attribution_2026-07-12.png"),
    section="4. 핵심 증거",
    notes="두 학습 요소 분해. 진폭 prior γ는 nowcast를, 잡음바닥 σ는 forecast를 각각 유의하게 이기고 결합은 "
          "거의 정확히 가법. 상호작용 0 = 두 슬롯이 다른 결함을 독립적으로 메움. headfix 이득을 통째로 γ에 "
          "귀속하지 않고 과제별로 정직하게 분배.")

add_content(
    "핵심 증거 (6): 방향·속도 불변",
    [
        (0, "선박 이동은 본질이 아님 — 전장 추정이 목적, 트랙은 질의 (本末 규율)", True),
        (0, "방향 불변: 지구-고정 좌표 추정 → 침로에 종속되지 않음", False),
        (0, "속도 불변: 등방(isotropic) 파수 은행 → 선속·조우주파수 편향 없음", False),
        (0, "실증: dir180 head-sea 30 에피소드", True),
        (1, "전장 hand 0.915 / learned 0.869, 트랙 0.962 / 0.949 → dir0 구조 유지", False),
        (0, "캐비엇: 트랙 규약은 외부 구성(h5에 궤적 미저장) — 공시", False),
    ],
    section="4. 핵심 증거",
    notes="사용자 재확인 — '선박 이동이 본질이 아니다, 전장 추정이 목적, 방향·속도 불변'. 지구-고정 좌표와 "
          "등방 은행이 불변성을 보장. dir180 정면파 30 에피소드에서 dir0 구조 재현. 트랙 좌표 규약은 h5에 "
          "궤적이 없어 외부 구성했음을 정직 공시.")

add_content(
    "핵심 증거 (7): 추정 이득의 제어 전달 (D-20)",
    [
        (0, "실측 스테레오 표면 위 가상 주행 폐루프 리플레이(3rec, τ∈{0.5,1.0}s)", True),
        (0, "G-D20-1 PASS: 스트리밍 예보-FF가 persistence를 이김(2/3 rec, 양 τ)", False),
        (0, "G-D20-2 PASS(헤드라인): learned-FF 잔여 ≤ hand-FF", True),
        (1, "창-짝 −0.086 (t −4.02, n=72), 3/3 레코드, 71% 창 승", False),
        (1, "주역 = LJ18g 실패-창 구조(−0.195, hand-FF 파국 치유)", False),
        (0, "G-D20-3 PASS(예산): learned GPU p50 5.2 ms (<100 ms), 등가 corr 1.0000", True),
    ],
    img=os.path.join(FIG, "fig_d20_control_2026-07-12.png"),
    section="4. 핵심 증거",
    notes="추정 이득이 실제 제어 성능으로 전달되는지 실측 리플레이로 확인. 학습 예보 FF가 물리보다 잔여 히브를 "
          "유의하게 줄이고 세 레코드 모두, 창의 71%에서 승. 주역은 폭풍 레코드 실패창 치유. 계산도 GPU 5 ms로 "
          "예산 통과. 스코프는 개구가 파장을 감당하는 레코드 한정 공시.")

add_content(
    "핵심 증거 (8): 배포 레시피 · 벤치마크 v2 · 부록 갤러리",
    [
        (0, "무학습 배포 레시피: 고정-T + 범위-층화 2,000점 + 잡음바닥", True),
        (1, "조우 rho 8/8 바다 0.90–0.99, HOS M=3 재현 0.92/0.91, 100 ms 내", False),
        (0, "벤치마크 v2: 85 에피소드(dir0 55 + head-sea 30) — 공개 가능 평가 자산", True),
        (0, "부록 갤러리: 합성/실측 레코드별 전장·트랙 시트(불일치 감사 0)", False),
        (0, "학습은 이 레시피 위에 얹는 추가 몫", False),
    ],
    img=os.path.join(FIG, "figs_appendix", "appendix_YS17_v2.png"),
    section="4. 핵심 증거",
    notes="학습 없이도 도는 무학습 배포 레시피 — 고정 임계, 범위 층화 2천 점, 잡음바닥이면 8개 바다 전부 조우 "
          "상관 0.90 이상, HOS도 재현. 학습은 이 위에 얹는 추가 몫. 벤치마크 v2와 부록 갤러리는 공개 가능 "
          "평가 자산.")

# Limitations
add_content(
    "한계·정직 공개 (1): 실물 선박 LiDAR 부재",
    [
        (0, "L1(1번 한계): 실물 선박 탑재 LiDAR 실측 0바이트", True),
        (1, "공개 스테레오 12–13 레코드(4 사이트)가 전량 (IFREMER 전수조사 확정)", False),
        (1, "직접 포함 안 됨: 반사율·입사각 결측, 자세 잡음, 물보라/강우", False),
        (0, "방어", True),
        (1, "① 골격 = \"주어진 사양\"(센서 실물 아닌 사양이 연구 대상)", False),
        (1, "② 사양 연산자 실측 표면 zero-shot 전이 8/8 양성", False),
        (1, "③ corr 절대치 용도 한정(조우-단기·운용지표·UQ로 이미 좁힘)", False),
        (0, "용어 규율: 본문 전체 \"실측 검증\" = 사양-모사, 제목/초록 오독 방지", True),
    ],
    accent=RGBColor(0xB0, 0x30, 0x30),
    section="5. 한계·정직 공개",
    notes="가장 아픈 한계를 먼저. 실물 선박 LiDAR 없음, 공개 스테레오가 전량. 방어 세 개: 연구 대상이 센서 "
          "실물이 아닌 사양, 그 연산자 실측 표면 zero-shot 8/8 양성, 절대 상관값 용도 이미 좁힘. 제목·초록 "
          "오독 방지 용어 고정, 수조로 외연 확장.")

add_content(
    "한계·정직 공개 (2): 경계의 지도 (부정 결과 = 경계 증명)",
    [
        (0, "BF2 하한: 1comp fixedT 0.19 — 원인 = 저SNR 단일선 초분해(은행 양자화 아님)", True),
        (1, "오라클 참-k 0.993으로 정보는 존재", False),
        (0, "표현 축 폐쇄(M3): point-native가 CNN에 미달(+0.472 vs +0.620)", True),
        (1, "갭이 합성 도메인부터 존재 → 병목은 표현이 아니라 정보", False),
        (0, "n=5 레코드 계열: 창-수준 주장 전부 군집-보정 병기", False),
        (0, "학습 못 이기는 자리(정리): 선형-가우시안·올바른-은행·MSE 사후평균 불패", True),
        (1, "폐쇄 10칸을 게이트·원인·상금으로 수치화", False),
    ],
    accent=RGBColor(0xB0, 0x30, 0x30),
    section="5. 한계·정직 공개",
    notes="부정 결과를 독립 실패 목록이 아니라 승리의 경계 증명으로 배치. BF2 하한은 은행이 아니라 저SNR "
          "초분해, 최신 아키텍처로 바꿔도 CNN 못 넘음 = 병목이 표현 아닌 정보. '학습이 못 이기는 자리'의 "
          "증명이 중심 명제의 나머지 절반.")

# Roadmap
add_content(
    "로드맵 (1): 기여 지도 · 집필 일정",
    [
        (0, "양성 산출물 5종", True),
        (1, "① 무학습 배포 레시피 ② 학습 추정기(net+σ-head) ③ 벤치마크 v2", False),
        (1, "④ 111× amortizer ⑤ 검출 법칙·작동영역 정량화", False),
        (0, "경계 증명 9~10칸 + 사전등록 게이트 방법론(하루 가짜 4건 차단 실적)", False),
        (0, "집필 2편 구도", True),
        (1, "1편: 추정기 + 배포 레시피 + 실측 전이 (OCEANS 2026 / Ocean Engineering급)", False),
        (1, "2편: 학습의 자리 지도 + 게이트 방법론 (JOE급, 통계 보강 후)", False),
        (0, "즉시 작업 = Ch.A/B 수치 표 채우기 — 잔여 GO 실험 셀은 M3 하나", True),
    ],
    accent=TEAL, section="6. 로드맵·요청",
    notes="다섯 양성 산출물, 경계 증명들, 사전등록 게이트 방법론. 집필은 두 편 — 추정기+레시피+실측 전이가 "
          "1편, 학습 지도+게이트 방법론이 2편. 실험 캠페인은 사실상 결판, 남은 GO 셀은 표현 축 하나. 지금은 "
          "집필 전환이 권고.")

add_content(
    "로드맵 (2): ★수조 최소 실증 1캠페인 (승인 요청 핵심)",
    [
        (0, "목적 = 관측 연산자의 실물 검증 (성능 헤드라인 아님)", True),
        (0, "장소: 서울대 예인수조. 선박급 solid-state LiDAR(σ≈2 cm, ≥10 Hz) 갠트리", False),
        (1, "참값 = 파고계 배열, 공통 트리거 동기화, 조건별 ≥5회 반복", False),
        (0, "최소 4실험", True),
        (1, "E1 규칙파 센서 정합(최우선) · E3 불규칙 1-D nowcast", False),
        (1, "E7 예보/예측가능 영역 · E9 UQ 보정", False),
        (0, "정직한 범위: 예인수조 단방향 → 다방향은 시뮬 + 향후 해양공학수조 분리", True),
        (1, "Froude 축척(H 0.05–0.3 m, T 0.6–2 s) 모형·실선 등가 병기", False),
    ],
    accent=TEAL, section="6. 로드맵·요청",
    notes="오늘 가장 중요한 승인 요청. L1 한계 직접 대응으로 서울대 예인수조 최소 1캠페인. 목적은 성능 경신이 "
          "아니라 관측 연산자를 실물 센서로 검증 — E1 규칙파 센서 정합이 본체, 1-D nowcast·예보·UQ 부착. "
          "예인수조 단방향이라 다방향은 시뮬·향후 해양공학수조로 정직 분리.")

add_content(
    "요청사항 종합 (승인 3건)",
    [
        (0, "1. 연구 방향 승인: \"small, certified learning\"", True),
        (1, "정보 회계 × 학습 지도 × 레시피의 연결을 학위 척추로", False),
        (0, "2. 수조 최소 실증 1캠페인 승인: E1/E3/E7/E9 (관측 연산자 실물 검증)", True),
        (0, "3. 집필 일정 합의: 1편(추정기+레시피+실측 전이) 우선, 2편 통계 보강 후", True),
        (0, "집필 전 필수 행동항목(페르소나 판정)", False),
        (1, "#1 군집 부트스트랩 완료(6/6 생존) · #2 용어 규율 반영 · #3 수조 로드맵 작성", False),
    ],
    accent=TEAL, section="6. 로드맵·요청",
    notes="세 가지 승인 — 방향, 수조 캠페인, 집필 일정. 심사에서 나올 통계·용어·실물 데이터 공격의 세 "
          "행동항목은 이미 이행: 군집 부트스트랩 6/6 생존, 용어 규율·수조 로드맵 문서 준비 완료.")

# Backup
add_content(
    "[예비] Q: novelty가 있나?",
    [
        (0, "인정: 방법 수준 novelty 없음 — AHGP·DSWP/Ehlers·DFL·FutureWaves 선점·인용", True),
        (0, "살아남는 자리 4개", True),
        (1, "① 규약 반전 진단(top-k가 과제 사망 제조 → fixedT 재구성)", False),
        (1, "② 학습의 자리 지도(창-수준 인과 분해)", False),
        (1, "③ 실패-창 법칙(과제-보정 고정 바닥 ≈3σ₀)", False),
        (1, "④ 요구-역산 설계(구속 축=개구; B1 위반 0/100)", False),
        (0, "주장 형식 = \"조합 + 보장 + 도메인\", '최초'류 전면 금지", True),
    ],
    accent=LGREY, section="예비 슬라이드",
    notes="원리 선점은 전부 인정·인용. 남는 자리는 방법이 아니라 규약 진단+학습 지도+실패창 법칙+요구역산의 "
          "조합·보장·도메인. '최초'는 쓰지 않음.")

add_content(
    "[예비] Q: n=5인데 60창은 통계 부풀리기 아닌가",
    [
        (0, "세 겹 방어", True),
        (1, "① 레코드-수준 부호검정 8/8, p=0.004 (비모수, 클러스터 가정 없음)", False),
        (1, "② 창-짝지음은 레코드-내 비교 → 레코드 간 상관 차분에서 소거", False),
        (1, "③ 레코드-군집 부트스트랩 6/6 생존(γ CI[0.087,0.206], t_rec 4.32)", False),
        (0, "작동점 주의: 실측 창 U 1.2–2.4 m/s = 판정 프로토콜용 가상 속도", True),
        (1, "설계 선속 11.3 m/s 재판정은 합성 스윕(1–12 m/s) 담당, 불일치 한계 공개", False),
    ],
    accent=LGREY, section="예비 슬라이드",
    notes="창-수준 t만으로 우기지 않음 — 레코드-수준 부호검정 8/8 p=0.004, 군집 부트스트랩 6/6 생존. 실측 "
          "창 속도와 설계 선속이 다른 점도 한계로 명시.")

add_content(
    "[예비] Q: 배에 단 LiDAR 실측 0바이트면 학위감인가",
    [
        (0, "정면 인정: 실물 선박 LiDAR 부재 = Limitations 1번(공개 스테레오가 전량)", True),
        (0, "방어", True),
        (1, "① 연구 대상 = 사양(요구-역산) ② 사양 연산자 실측 표면 zero-shot 8/8", False),
        (1, "③ 과잉 주장 이미 내부 게이트로 제거(07-05 정직 판정)", False),
        (0, "행동: 제목/초록 \"LiDAR 실측 검증 완료\"류 금지 → 사양-모사 한정", True),
        (0, "반론 프레임: 진짜 공백 = \"정확히 얼마나 되는가\" 부재를 수치로 남긴다", True),
    ],
    accent=LGREY, section="예비 슬라이드",
    notes="가장 아픈 산업 공격. 정면 인정 — 실물 선박 LiDAR 없음. 대신 연구 대상이 사양이고 그 연산자를 "
          "실측 표면에 zero-shot 얹어 검증. 제목·초록 오독 방지 용어 고정, 수조로 외연 확장.")

add_content(
    "[예비] Q: CNN은 구식 아닌가",
    [
        (0, "novelty는 아키텍처-독립(어떤 예측기든 γ·σ만 내면 됨)", True),
        (0, "성능 축 3중 반박", True),
        (1, "① 인코더 용량 13층→1,600만 파라미터 불변(+0.508 vs +0.527)", False),
        (1, "② wiener2 대각 구조가 자유 CNN 잔차 앞섬", False),
        (1, "③ 속박파 고정 물리(0.195) > 자유 CNN(0.111)", False),
        (0, "직접 검정(M3): point-native가 CNN 미달(Δ−0.148), 갭이 합성 도메인부터", True),
        (1, "→ 물리-정규화 귀납 편향(beamform 맵+kp 정규화)이 기여", False),
    ],
    accent=LGREY, section="예비 슬라이드",
    notes="주장은 아키텍처와 독립. 용량을 키워도, 최신 point-native로 바꿔도 CNN 못 넘고 그 격차가 합성 "
          "도메인부터 존재 = 물리-정규화 귀납 편향이 실제 기여. 민감도 표는 승패 무관 심사 방어 자산.")

add_content(
    "[예비] Q: 부정 결과(9연패)가 너무 많지 않나",
    [
        (0, "신규 판정 전패 = 전부 사전등록 게이트로 수치화", True),
        (1, "비평가 v1/v2·폭풍·M8·M2×M7 v1/v2·소프트결합·L4-v1·레코드게인·M3 point", False),
        (0, "프레임: 폐쇄들은 양성 산출물 5종의 경계 증명", True),
        (1, "\"왜 더 안 되는가\"를 게이트·원인·상금으로 확정", False),
        (0, "집필 규칙: 부정 결과는 경계 서술로만(독립 실패 절 금지)", False),
        (0, "못 이기는 곳의 증명이 중심 명제의 절반", True),
    ],
    accent=LGREY, section="예비 슬라이드",
    notes="전부 사전등록 게이트라 원인·상금 수치화. 논문에선 독립 실패 목록이 아니라 승리의 경계 증명으로만 "
          "배치. 중심 문장이 '물리가 모르는 것만 채운다'인 이상, 못 이기는 곳의 증명이 논지의 절반.")

add_content(
    "[예비] Q: 짜집기 아닌가 / WaveSplat은 한계 지적받고 버린 것 아닌가",
    [
        (0, "인정: 부품 단위(beamforming·MMSE·CNN)는 전부 인용 대상", True),
        (0, "방법론 = 추정기 계보(사다리) — 각 화살표에 우리 진단·판정이 붙음", True),
        (1, "v1 파라메트릭 WaveSplat: 이산-성분 영역 승자 (1comp 0.44/0.63, OCEANS)", False),
        (1, "→ 한계를 우리가 수치 진단(greedy 유령·모델차수·광대역 취약)", False),
        (1, "v2 은행-MMSE (무학습 물리, 우리 설계): 광대역·실측 강건 — 레시피 0.96", False),
        (1, "→ 남은 갭(실측 오지정)을 또 진단 → v3 학습 슬롯(γ·σ)의 자리 확정", False),
        (0, "v1은 폐기 아님 — 레짐 지도: 준단색→파라메트릭 / 광대역→은행 / 오지정→+학습", True),
        (0, "조립 법칙도 방법 지식: \"net은 prior만\"+kp-정규화만 zero-shot (반대 조립은 죽음)", True),
        (0, "\"한계 있었잖아\" → \"네, 그 정량 진단이 다음 설계를 만든 것이 기여입니다\"", True),
    ],
    accent=LGREY, section="예비 슬라이드",
    notes="'학습 기반으로 변경'이 아님을 먼저 정정 — 전환은 2단계고 가운데(v2 은행-MMSE)가 무학습 물리이며 "
          "우리 설계다(무학습 레시피 0.96이 v2만으로 달성되는 게 증거). v1 한계는 남이 지적한 게 아니라 "
          "우리 병목 해부가 밝혔고 그 진단이 v2를 설계했다 — 사다리 전체가 방법론이고 각 단계에 판정 "
          "수치와 유효 영역이 있다. v1은 준단색 영역에서 여전히 승자(A-7 처방도 파라메트릭 폴리시 이식). "
          "그래도 '새 네트워크' 요구가 오면 방법 축 승격 안건으로 격상 — 요청 1이 그 자리.")

# (부록 핵심 수치 표는 슬라이드 md의 마지막 부록 표로 유지 — pptx 26장)

prs.save(OUT)
print("SAVED:", OUT)
print("slides:", len(prs.slides._sldIdLst))
