"""
fill_labmeeting_template_v2.py

User 가 직접 작업한 docs/labmeeting_20260513_박세용.pptx 의 빈 슬라이드를
사용자 톤으로 채우고 새 파일로 저장.

User tone (slide 4, 7, 8, 11, 12, 18, 19 분석):
- L0/L1/L2 indent
- 짧은 명사구 (1-2 줄/L2)
- 구분자 "-" (en-dash 가끔), em-dash 안 씀
- bold marker (*...*) 안 씀
- 한국어 + 영어 용어 inline

빈 슬라이드 (채울 곳):
- Slide 5: §1 Intro #2 — 도전 과제
- Slide 9: §2 — 응용 매트릭스 (table 위에 짧은 intro)
- Slide 13: §3 Dataset (body + wave-only figure, NO WaveSplat)
- Slide 14: §3 WaveSplat 알고리즘
- Slide 16: §4 Results + Mechanism
- Slide 20: §5 PD vs PD+FF (animation 아래 caption)
- Slide 22: Appendix A (table)
- Slide 23: Appendix B (table)
- Slide 24-26: Appendix C×3 (figures)

사용자 작성 보존: 4, 7, 8, 11, 12, 18, 19 — 건드리지 않음.

Output: docs/labmeeting_20260513_박세용_v2.pptx
"""
import os
import shutil
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'docs', 'labmeeting_20260513_박세용.pptx')
DST = os.path.join(ROOT, 'docs', 'labmeeting_20260513_박세용_v2.pptx')
DOCS = os.path.join(ROOT, 'docs')


# ── User-tone slide content ──

SLIDE_5_INTRO_CHALLENGES = [
    (0, '본 task 의 핵심 도전 과제'),
    (1, '희소·비균일 측정 (sparse, non-uniform observation)'),
    (2, 'LiDAR frame 당 valid 점 약 50개 (raydrop, shadowing 후)'),
    (2, '거리에 따른 점밀도 비균일 - 가까울수록 dense, 멀수록 sparse'),
    (2, 'Wave gauge / dense radar 와 다른 setup → 기존 방법 직접 적용 어려움'),
    (2, 'Patch coverage: 약 50–150 m (실측 확인)'),
    (1, '실시간 제약 (real-time constraint)'),
    (2, '선박 제어 루프 ~10 Hz → frame 당 100 ms 이내 필요'),
    (2, 'EnKF + HOS (Wang & Pan 2021/22): 비선형 model, 수백 ms 이상'),
    (2, 'EnKF + linear (Feng Wang 2025, Ne=300): 632 ms - 100 ms 초과'),
    (2, 'PFT-PINN (Ehlers 2025): instance 마다 ~30 분 학습 (RTX 3090)'),
    (2, 'FNO supervised (Ehlers 2023): 1.9 ms - 단 학습 데이터 3120개 필요'),
    (1, '2D directional sea state'),
    (2, '실해 파도: wind sea (짧은 파) + swell (긴 파) 여러 방향 혼재'),
    (2, 'Wavenumber 도 단순 scalar 아닌 vector (k_x, k_y) 추정 필요'),
    (2, '선행 연구 다수 1D 가정 (Klein 2020, Desmars 2023, FNO)'),
    (2, '본 연구는 2D directional 그대로 처리 - 차별점 중 하나'),
]

SLIDE_9_NICHE_INTRO = [
    (0, '응용 매트릭스 - 본 연구의 위치'),
    (1, '6가지 응용 요구사항으로 본 연구와 5개 baseline 비교'),
    (1, '본 연구 고유 영역: real-time + 2D directional + sparse LiDAR + 학습 데이터 0개'),
]

SLIDE_13_DATASET = [
    (0, 'Dataset - 110 episode 합성'),
    (1, 'Sea state 구성 (BF2 ~ BF7, 6 단계)'),
    (2, 'BF2 (Hs=0.3 m, Tp=4.0 s) - 잔잔'),
    (2, 'BF3 (Hs=0.6, Tp=5.5) / BF4 (Hs=1.2, Tp=7.0) - mild ~ moderate'),
    (2, 'BF5 (Hs=2.0, Tp=9.0) / BF6 (Hs=3.0, Tp=10.0)'),
    (2, 'BF7 (Hs=5.0, Tp=12.0) - severe'),
    (1, 'Variation 축'),
    (2, 'N components: 1, 2, 5, 15, 30 (단순 ~ 복잡 wave field)'),
    (2, 'Seeds: 0 ~ 4 (재현 가능)'),
    (2, 'Raydrop 모델: uniform / slope 두 가지'),
    (1, '에피소드별 정량'),
    (2, 'Duration 30 s, dt 0.1 s (10 Hz) → frame 301개'),
    (2, 'GT η field: 80×60 격자, 240 m × 200 m 영역'),
    (2, 'LiDAR scan: 76,000 ray → valid 점 ~50개 / frame (raydrop 후)'),
    (2, 'Patch coverage: 약 50–150 m (실측 확인)'),
    (1, '전체 용량'),
    (2, '110 episodes ≈ 1.2 GB (HDF5 format)'),
]

SLIDE_14_WAVESPLAT = [
    (0, 'WaveSplat - phase-resolved 추정 알고리즘'),
    (1, '발상 - 격자 대신 파라미터로 직접 표현'),
    (2, '실제 파도 = 5–30개 sine wave 의 합 (JONSWAP 분해)'),
    (2, '각 성분 (A, k_x, k_y, ω, φ) 5개 파라미터로 완전 결정'),
    (2, 'N=10 이면 변수 50개, LiDAR 50점으로 1:1 추정 가능'),
    (2, '격자 4,800개 vs 파라미터 50개 - 1/100 차원'),
    (1, 'Wave field 표현 + 분산 관계 hard constraint'),
    (2, 'η(x, y, t) = Σ A_n · cos(k_n·x − ω_n·t + φ_n)'),
    (2, 'ω(k) = √(g·|k|·tanh(|k|·d)) 강제 → ω 자동 결정'),
    (2, '자유도 5N → 4N (20% 감소)'),
    (2, '물리적으로 불가능한 (k, ω) 조합 자동 배제'),
    (1, '2단계 흐름 - Init / Track 분리'),
    (2, 'Init (1초 이내, 한 번): beamforming + coarse-to-fine + Adam refinement'),
    (2, '   - Search range k_x, k_y ∈ [-4, 4], 81×81 grid'),
    (2, '   - Coarse-to-fine schedule N: 3 → 8 → 15'),
    (2, 'Track (매 frame, 13 ms): k 고정, (A, φ) 만 Adam update'),
    (2, '   - Phase prediction: φ ← φ + ω·Δt (kinematic preview)'),
    (1, 'Adaptive N - 성분 수 동적 조정'),
    (2, 'Pruning (cumulative energy 90% 유지) + Growth (잔차 RMS > 0.05 m)'),
    (2, 'Safety rollback (RMS 1.3× 초과 증가 시 즉시 복구)'),
]

SLIDE_16_RESULTS = [
    (0, '실험 결과 및 핵심 발견'),
    (1, '12-episode benchmark (BF2 ~ BF5 sweep)'),
    (2, 'WaveSplat: 평균 nRMSE 0.211, 13.4 ms / frame'),
    (2, 'Wang2025 baseline (Ne=300, fuzzy R, Delaunay): 0.253, 632 ms'),
    (2, '정확도 16% 우위 + 속도 47배 빠름 (CPU only)'),
    (2, '13.4 ms = 초당 75 frame → 10 Hz 제어 충분'),
    (1, 'Episode-level 패턴'),
    (2, '단순 파도 (1–5 comp): WaveSplat 압도 - BF3_2comp 0.052 vs 0.250'),
    (2, '복잡 파도 (≥15 comp): 일부 역전 - BF4_15comp 0.302 vs 0.249'),
    (2, 'Reversal point ~N=15 - 향후 hybrid init 으로 보완 예정'),
    (1, 'Ablation - 두 메커니즘 시너지'),
    (2, 'A 기본 0.213 / B +multi-frame 0.208'),
    (2, 'C +dispersion (single frame) 0.226 - 오히려 악화!'),
    (2, 'D 결합 (=WaveSplat) 0.211 - 가장 안정'),
    (2, '교훈: 물리 법칙은 정규화 역할, multi-frame 과 함께 써야 효과'),
    (1, '핵심 메커니즘 두 가지'),
    (2, '분산 관계 → 자유도 감소 (5N → 4N, 20%) + 비물리적 답 배제'),
    (2, 'Multi-frame temporal coherence → wave 시간 진화로 spatial Nyquist 우회'),
]

SLIDE_20_PD_FF = [
    (0, 'PD vs PD + Feedforward'),
    (1, '같은 파도 시퀀스, 두 controller 비교 (위 영상)'),
    (1, 'aRMS: PD 0.090 m/s² → PD+FF 0.047 m/s² (약 48% 감소)'),
    (1, 'ISO 2631 ride comfort 기준에서 의미 있는 수준'),
    (1, 'WaveSplat 정보가 제어에 도움 - proof-of-concept'),
]


# ── Appendix tables (same content as before) ──

APPENDIX_A_HEADERS = ['#', 'Paper (Year)', 'Sensor / Data', '핵심 방법', '한계 / 우리 차별']
APPENDIX_A_ROWS = [
    ['1', 'Desmars 2018', 'LIDAR (proposed)\nSim only',
     'LIDAR + ICWM, on-grid variational', 'LIDAR concept proposal, 1D'],
    ['2', 'Desmars 2020', 'Wave gauge array\n실측 (wave tank)',
     'Wave gauge + dispersion correction', 'Wave gauge ≠ LiDAR, 1D'],
    ['3', 'Desmars 2023', 'X-band radar (modeled)\nSim only',
     'HOS-OCM, vessel optical', '1D, on-grid'],
    ['4', 'Wang & Pan 2021', 'X-band radar\nSim + 실측',
     'EnKF + HOS 첫 결합', 'Dense obs, 1D'],
    ['5', 'Wang & Pan 2022', 'Shipborne radar\n실측 + sim',
     'EnKF-HOS-C (current 동시)', '1D, dense obs'],
    ['6', 'Feng Wang 2025', 'Stereo camera (WASS)\n실측 (Acqua Alta)',
     'ETKF + Ne=300 + fuzzy R', '격자 한계, 632 ms (본 baseline)'],
    ['7', 'Hunt 2007', '기상학\nSim',
     'LETKF', 'Wave-specific 아님'],
    ['8', 'Klein 2020', 'Model 비교\nSim + semi-exp',
     'LWT/NLSE/HOSM 정량 비교', '새 알고리즘 아님'],
    ['9', 'Ehlers 2023', 'X-band radar\nSim only',
     'FNO supervised, 1.9 ms', '3120 sample 학습 필요, 1D'],
    ['10', 'Ehlers 2025', 'Wave gauge\n실측 (wave flume)',
     'PFT-PINN unsupervised', '~30 분/instance, 1D + t'],
    ['11', 'Yan 2025', 'Synthetic\nSim only',
     'NS + EnKF + POD', 'Offline only, 1D xz'],
    ['12', 'Tang 2013', '이론\n—',
     'Atomic norm off-grid CS', '1D, SDP 비용'],
    ['13', 'Möller-Trumbore 1997', 'Graphics\n—',
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


# ── Helpers ──

def find_body_placeholder(slide):
    """Find body placeholder (large text frame in middle/lower portion)."""
    candidates = []
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        t = shape.top / 914400 if shape.top else 0
        h = shape.height / 914400 if shape.height else 0
        if t > 0.5 and h > 4:
            candidates.append((shape, h))
    if candidates:
        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0]
    return None


def remove_decorative_groups(slide):
    """Remove decorative GROUP shapes that aren't placeholders (cleanup)."""
    for shape in list(slide.shapes):
        try:
            if not shape.is_placeholder and 'GROUP' in str(shape.shape_type):
                sp = shape._element
                sp.getparent().remove(sp)
        except Exception:
            pass


def get_or_create_body(slide, default_left=0.35, default_top=1.07,
                      default_width=8.96, default_height=5.68):
    """Return existing body placeholder, or create a new textbox if missing."""
    body = find_body_placeholder(slide)
    if body is not None:
        return body
    # No body placeholder — also remove decorative groups, then add textbox
    remove_decorative_groups(slide)
    box = slide.shapes.add_textbox(
        Inches(default_left), Inches(default_top),
        Inches(default_width), Inches(default_height)
    )
    return box


def find_title_placeholder(slide):
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


def is_body_empty(slide):
    """Is body placeholder empty (no real content)?"""
    body = find_body_placeholder(slide)
    if body is None:
        return True
    text = body.text_frame.text.strip()
    if not text or 'April xx' in text:
        return True
    return False


def fill_body_user_tone(text_frame, items, font_sizes=(14, 12, 11),
                       space_after=(3, 1, 0), bold_l0=False):
    """Fill body with (level, text) tuples in user's compact tone.
    No bold markers, hyphens for separators.
    """
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
        sz = font_sizes[min(level, len(font_sizes) - 1)]
        sp = space_after[min(level, len(space_after) - 1)]
        p.space_after = Pt(sp)
        for r in p.runs:
            r.font.size = Pt(sz)
            if level == 0 and bold_l0:
                r.font.bold = True


def resize_body(slide, top_in=None, height_in=None, left_in=None, width_in=None):
    """Resize body placeholder. Must set all 4 dims explicitly to avoid
    losing inherited values."""
    body = find_body_placeholder(slide)
    if body is None:
        return
    cur_left = body.left
    cur_top = body.top
    cur_width = body.width
    cur_height = body.height
    body.left = Inches(left_in) if left_in is not None else cur_left
    body.top = Inches(top_in) if top_in is not None else cur_top
    body.width = Inches(width_in) if width_in is not None else cur_width
    body.height = Inches(height_in) if height_in is not None else cur_height


def add_picture_aspect_fit(slide, fig_path, left_in, top_in, width_in, height_in):
    """Add picture preserving aspect ratio inside box."""
    from PIL import Image
    if not os.path.isabs(fig_path):
        fig_path = os.path.join(ROOT, fig_path)
    if not os.path.exists(fig_path):
        print(f'  [warn] missing: {fig_path}')
        return False
    try:
        with Image.open(fig_path) as im:
            iw, ih = im.size
        ratio = iw / ih
        box_ratio = width_in / height_in
        if ratio > box_ratio:
            w = width_in
            h = w / ratio
        else:
            h = height_in
            w = h * ratio
        cx = left_in + (width_in - w) / 2
        cy = top_in + (height_in - h) / 2
        slide.shapes.add_picture(fig_path, Inches(cx), Inches(cy),
                                 width=Inches(w), height=Inches(h))
        return True
    except Exception:
        slide.shapes.add_picture(fig_path, Inches(left_in), Inches(top_in),
                                 width=Inches(width_in))
        return True


def update_dates(prs, new_date='May 13, 2026'):
    for slide in prs.slides:
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for p in shape.text_frame.paragraphs:
                for r in p.runs:
                    if r.text and 'April xx' in r.text:
                        r.text = new_date


def fill_appendix_a_table(slide):
    """Slide 22 — fill 13 paper rows."""
    table = next((s.table for s in slide.shapes if s.has_table), None)
    if table is None:
        return
    n = len(table.rows)
    # Headers may already be there; fill rows
    # Check if headers row matches
    for c, h in enumerate(APPENDIX_A_HEADERS):
        if c < len(table.columns):
            cell = table.cell(0, c)
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = h
            for r in p.runs:
                r.font.bold = True
                r.font.size = Pt(10)
    for r_idx, row in enumerate(APPENDIX_A_ROWS):
        if r_idx + 1 >= n:
            break
        for c_idx, val in enumerate(row):
            if c_idx >= len(table.columns):
                break
            cell = table.cell(r_idx + 1, c_idx)
            tf = cell.text_frame
            tf.word_wrap = True
            lines = val.split('\n')
            p = tf.paragraphs[0]
            p.text = lines[0]
            for r in p.runs:
                r.font.size = Pt(8)
            for line in lines[1:]:
                np_ = tf.add_paragraph()
                np_.text = line
                for r in np_.runs:
                    r.font.size = Pt(8)


def fill_appendix_b_table(slide):
    """Slide 23 — fill dataset breakdown."""
    table = next((s.table for s in slide.shapes if s.has_table), None)
    if table is None:
        return
    n = len(table.rows)
    for c, h in enumerate(APPENDIX_B_HEADERS):
        if c < len(table.columns):
            cell = table.cell(0, c)
            tf = cell.text_frame
            p = tf.paragraphs[0]
            p.text = h
            for r in p.runs:
                r.font.bold = True
                r.font.size = Pt(11)
    for r_idx, row in enumerate(APPENDIX_B_ROWS):
        if r_idx + 1 >= n:
            break
        for c_idx, val in enumerate(row):
            if c_idx >= len(table.columns):
                break
            cell = table.cell(r_idx + 1, c_idx)
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = val
            for r in p.runs:
                r.font.size = Pt(10)


def replace_picture_in_slide(slide, fig_path):
    """Find existing picture and replace with new figure (same position/size)."""
    if not os.path.isabs(fig_path):
        fig_path = os.path.join(ROOT, fig_path)
    if not os.path.exists(fig_path):
        return False
    for shape in list(slide.shapes):
        if shape.shape_type and 'PICTURE' in str(shape.shape_type):
            l, t, w, h = shape.left, shape.top, shape.width, shape.height
            sp = shape._element
            sp.getparent().remove(sp)
            slide.shapes.add_picture(fig_path, l, t, width=w, height=h)
            return True
    return False


# ── Presenter notes (간결한 구어체) ──

NOTES = {
    5: (
        "본 task 가 왜 어려운지 세 가지로 풀어 말씀드릴게요.\n\n"
        "첫째 *측정 데이터 부족* — LiDAR 는 frame 당 valid 점이 50개 정도밖에 안 되고, 거리 "
        "에 따라 점밀도가 다르고, 파도 표면 매끄러움이나 shadowing 으로 또 점이 빠집니다. "
        "기존 wave gauge 나 dense radar 처럼 촘촘한 측정을 가정한 방법은 그대로 못 써요.\n\n"
        "둘째 *시간 부족* — 제어 루프가 10 Hz 라 frame 당 100 ms 안에 끝나야 합니다. 기존 "
        "방법들은 다 이걸 못 만족해요. EnKF + HOS 는 수백 ms 이상, Feng Wang 의 ETKF + linear "
        "도 632 ms, PFT-PINN 은 instance 마다 30 분 학습. FNO 만 빠르지만 학습 데이터 3120개 "
        "필요해요.\n\n"
        "셋째 *2D directional* — 실해는 wind sea + swell 이 여러 방향 혼재라 wavenumber 도 "
        "vector 추정이 필요합니다. 선행 연구 다수가 1D 가정으로 단순화하지만 우리는 2D 로 "
        "그대로 처리합니다."
    ),
    9: (
        "이 표가 우리 연구의 정확한 위치를 한눈에 보여줘요. 행은 6 가지 응용 요구사항, 열은 "
        "본 연구와 5 개 baseline 이고 노란색이 우리 WaveSplat 입니다.\n\n"
        "핵심은 *어떤 방법도 모든 ✓ 를 받지 못한다* 는 것 — 각 방법마다 강점과 적합 응용이 "
        "달라요. 우리만 *real-time + 2D directional + sparse LiDAR + 학습 데이터 0개* 의 4 "
        "가지를 동시 만족하는데, 이 조합을 만족하는 선행 연구는 없습니다.\n\n"
        "응용 영역을 둘로 나누면, real-time control 영역에는 우리와 FNO 가, design-time "
        "분석 영역에는 VOF-EnKF 와 PFT-PINN 이 들어갑니다. 두 영역은 경쟁이 아닌 보완 관계 "
        "예요. 정직한 한계도 표에 보여요 — 파도 파쇄는 VOF-EnKF 만, 유속장은 PINN/EnKF "
        "영역이고 우리는 응용 자체가 다릅니다."
    ),
    13: (
        "합성 dataset 의 구성을 말씀드릴게요. 총 110 개 episode 를 만들었습니다.\n\n"
        "Sea state 는 BF2 잔잔한 바다부터 BF7 거친 폭풍까지 6 단계, 파도 성분 수는 1 개 단순 "
        "한 파부터 30 개 복잡한 파까지, seed 5 개로 재현 가능하게, raydrop 모델도 두 가지를 "
        "섞었어요.\n\n"
        "Per-episode — 30 초 동안 10 Hz 로 simulation 해서 frame 301 개. GT η field 는 80×60 "
        "격자, 240 × 200 m 영역. LiDAR scan 은 76,000 ray 발사 후 raydrop 거치면 valid 점 "
        "50 개 정도가 남고요, 실제 patch coverage 는 50–150 m 정도예요.\n\n"
        "전체 용량은 1.2 GB, HDF5 format. 우측 그림은 실제 dataset 의 한 frame — 파도장과 "
        "그 위에 찍힌 LiDAR 점들 (빨간 점) 입니다. 이 sparse 한 점들로부터 파도장 전체를 "
        "복원하는 게 우리 task 예요."
    ),
    14: (
        "WaveSplat 알고리즘의 근본 발상부터 말씀드릴게요. *발상 전환* 이 핵심입니다.\n\n"
        "기존 격자 방법은 4800 개 격자점의 파고를 추정하려고 했어요. LiDAR 50 점으로는 "
        "정보가 부족하니까 부드러움 가정 같은 추가 제약이 필요했습니다. 그런데 JONSWAP 으로 "
        "분해해보면 실해 파도는 *5–30 개 sine wave 의 합* 으로 충분히 표현됩니다. 각 성분이 "
        "5 개 파라미터로 결정되니까 N=10 이면 변수 50 개 — LiDAR 50 점으로 *1:1 추정 가능* "
        "해져요.\n\n"
        "여기에 분산 관계 (ω 가 k 에서 자동 결정) 를 hard constraint 로 강제하면 *자유도가 "
        "5N → 4N 으로 20% 줄어들고* 비물리적 답이 자동 배제됩니다.\n\n"
        "추정은 2 단계로 나눴어요. *Init 단계* 는 처음 1 초 안에 한 번만 실행되고 "
        "(k_x, k_y) 를 정확히 결정합니다 — beamforming + coarse-to-fine + Adam 정밀화 순서. "
        "*Track 단계* 는 매 frame 13 ms 안에 끝나고 (A, φ) 만 update 합니다. 무거운 작업은 "
        "처음 한 번만, 가벼운 작업은 매 frame 실시간으로요.\n\n"
        "*Adaptive N* 은 성분 수를 동적으로 조정해요 — pruning, growth, safety rollback "
        "세 메커니즘이 작동합니다."
    ),
    16: (
        "이 슬라이드가 *핵심 결과 + 메커니즘* 을 한 화면에 정리합니다.\n\n"
        "*12-episode benchmark* — BF2~BF5 다양한 파도에서 우리 WaveSplat 은 평균 nRMSE "
        "0.211, 13.4 ms / frame. Wang2025 는 0.253, 632 ms. *정확도 16% 우위 + 47 배 빠름* "
        "이고 그것도 CPU only 입니다. 다만 episode 별로 보면 — 단순 파도 (1–5 성분) 에서는 "
        "압도적이지만 (BF3_2comp 0.052 vs 0.250), 복잡 파도 (15 성분 이상) 에서는 일부 "
        "역전됩니다 (BF4_15comp 0.302 vs 0.249).\n\n"
        "*Ablation 결과* 가 흥미로워요. dispersion 단독은 0.226 으로 *오히려 악화*, "
        "multi-frame 단독은 0.208 (개선되나 불안정), 둘 다 결합하면 0.211 로 *가장 안정* "
        "합니다. 즉 *분산 관계는 마법이 아닌 정규화 역할* — multi-frame 과 함께 써야 효과가 "
        "있어요.\n\n"
        "*핵심 메커니즘 두 가지* — (1) 분산 관계가 자유도를 5N→4N 으로 줄이고 비물리적 답 "
        "을 배제하는 정규화 역할, (2) Multi-frame 이 wave 자체의 시간 진화로 spatial Nyquist "
        "한계를 우회. 이 두 가지가 결합되어야 sparse LiDAR 로 phase-resolved 추정이 가능 "
        "합니다."
    ),
    20: (
        "*위 영상이 PD vs PD+FF 의 직접 비교* 입니다. 잠시 영상에 집중해 주세요.\n\n"
        "위쪽은 시간에 따른 파도 + vessel 위치, 아래쪽은 heave acceleration 의 time series "
        "입니다. 파란선이 PD-only, 빨간선이 PD + FF (WaveSplat 사용) 인데, 빨간선의 *진동이 "
        "명확히 작아요*.\n\n"
        "정량으로는 — aRMS 가 PD 0.090 m/s² 에서 PD+FF 0.047 m/s² 로 *약 48% 감소* 했습니다. "
        "ISO 2631 ride comfort 기준에서 의미 있는 수준이고, 승차감 향상으로 직결돼요.\n\n"
        "강조드리고 싶은 건 — 이건 *시험 적용* 결과로 *WaveSplat 정보가 제어에 실제로 도움* "
        "이 된다는 proof-of-concept 입니다. 실제 deployment 에는 controller robustness, "
        "sensor noise 등 추가 고려가 필요하고 후속 연구로 두었습니다."
    ),
}


def main():
    shutil.copy(SRC, DST)
    prs = Presentation(DST)

    print('[1] 빈 슬라이드 채우기 — user tone (compact, hyphen, no bold)')

    # Slide 5 — §1 #2 도전 과제 (full body, body placeholder 없으면 textbox 추가)
    print('  Slide 5: 도전 과제')
    body = get_or_create_body(prs.slides[4])
    fill_body_user_tone(body.text_frame, SLIDE_5_INTRO_CHALLENGES,
                        font_sizes=(13, 11, 10), space_after=(3, 1, 0))

    # Slide 9 — §2 응용 매트릭스 intro (top portion only, table preserved)
    print('  Slide 9: 응용 매트릭스 (table 위 짧은 intro)')
    resize_body(prs.slides[8], top_in=1.07, height_in=1.3)
    body = find_body_placeholder(prs.slides[8])
    if body is not None:
        fill_body_user_tone(body.text_frame, SLIDE_9_NICHE_INTRO,
                            font_sizes=(13, 11, 10), space_after=(2, 1, 0))

    # Slide 13 — §3 Dataset (body left + 2 stacked figures right)
    print('  Slide 13: Dataset + 2 wave+lidar figures (stacked)')
    resize_body(prs.slides[12], top_in=1.07, height_in=5.68,
                left_in=0.35, width_in=4.6)
    body = get_or_create_body(prs.slides[12], default_width=4.6)
    fill_body_user_tone(body.text_frame, SLIDE_13_DATASET,
                        font_sizes=(13, 11, 10), space_after=(2, 1, 0))
    # 2 stacked wave + lidar figures (NO WaveSplat results)
    # Top: BF3 1-component (단순)
    add_picture_aspect_fit(prs.slides[12],
                           'docs/wavesim_BF3_1c_wave_lidar.png',
                           left_in=5.0, top_in=1.20, width_in=4.7, height_in=2.6)
    # Bottom: BF4 5-component (복잡)
    add_picture_aspect_fit(prs.slides[12],
                           'docs/wavesim_BF4_5c_wave_lidar.png',
                           left_in=5.0, top_in=4.05, width_in=4.7, height_in=2.6)
    # Caption labels
    cap1 = prs.slides[12].shapes.add_textbox(
        Inches(5.0), Inches(3.85), Inches(4.7), Inches(0.18))
    cap1_tf = cap1.text_frame
    cap1_tf.paragraphs[0].text = 'BF3 / N=1 (단순 wave)'
    for r in cap1_tf.paragraphs[0].runs:
        r.font.size = Pt(9)
        r.font.italic = True
    cap2 = prs.slides[12].shapes.add_textbox(
        Inches(5.0), Inches(6.70), Inches(4.7), Inches(0.18))
    cap2_tf = cap2.text_frame
    cap2_tf.paragraphs[0].text = 'BF4 / N=5 (복잡 wave) — 빨간 점: LiDAR scan'
    for r in cap2_tf.paragraphs[0].runs:
        r.font.size = Pt(9)
        r.font.italic = True

    # Slide 14 — §3 WaveSplat algorithm (body left + pipeline figure right)
    print('  Slide 14: WaveSplat 알고리즘 + pipeline')
    resize_body(prs.slides[13], top_in=1.07, height_in=5.68,
                left_in=0.35, width_in=4.6)
    body = find_body_placeholder(prs.slides[13])
    if body is not None:
        fill_body_user_tone(body.text_frame, SLIDE_14_WAVESPLAT,
                            font_sizes=(13, 11, 10), space_after=(2, 1, 0))
    add_picture_aspect_fit(prs.slides[13],
                           'paper/figures/fig_pipeline.png',
                           left_in=5.0, top_in=1.5, width_in=4.7, height_in=4.5)

    # Slide 16 — §4 Results + Mechanism (body + figure right)
    print('  Slide 16: Results + Mechanism + scatter figure')
    resize_body(prs.slides[15], top_in=1.07, height_in=5.68,
                left_in=0.35, width_in=4.6)
    body = find_body_placeholder(prs.slides[15])
    if body is not None:
        fill_body_user_tone(body.text_frame, SLIDE_16_RESULTS,
                            font_sizes=(13, 11, 10), space_after=(2, 1, 0))
    add_picture_aspect_fit(prs.slides[15],
                           'docs/fig_complexity_scatter.png',
                           left_in=5.0, top_in=1.5, width_in=4.7, height_in=4.5)

    # Slide 20 — PD vs PD+FF caption (animation already there)
    print('  Slide 20: PD vs PD+FF caption (animation 위치 유지)')
    # Body 가 placeholder 인 경우 가능하면 caption 영역 확보
    body = find_body_placeholder(prs.slides[19])
    if body is not None:
        # animation 은 y=1.80~6.30 차지. caption 은 위 (y=1.07~1.75) 또는 아래 (y=6.35~7.0)
        # 위에 짧게 배치
        resize_body(prs.slides[19], top_in=1.07, height_in=0.65,
                    left_in=0.35, width_in=8.96)
        body = find_body_placeholder(prs.slides[19])
        if body is not None:
            fill_body_user_tone(body.text_frame, [SLIDE_20_PD_FF[0]],
                                font_sizes=(13,), space_after=(0,))
    # Replace existing animation with our hydrofoil_wave_animation.gif if needed
    # (user already has fig_hydrofoil_phase2 likely — keep as is)

    # Slide 22-26 — Appendix 사용자 원본 그대로 보존 (수정 안 함)
    print('  Slide 22-26: Appendix 원본 유지 (수정 안 함)')

    # Update dates
    print('[2] 날짜 May 13, 2026 으로 업데이트')
    update_dates(prs)

    # Add presenter notes for filled slides
    print('[3] 발표자 노트 추가 (구어체)')
    for slide_idx, note_text in NOTES.items():
        slide = prs.slides[slide_idx - 1]  # 1-indexed → 0-indexed
        notes_slide = slide.notes_slide
        tf = notes_slide.notes_text_frame
        tf.text = note_text

    prs.save(DST)
    print(f'\nSaved: {DST}')
    print(f'Total slides: {len(prs.slides)}')


if __name__ == '__main__':
    main()
