"""`T11-EXTACC` 규칙 단일출처. ⛔`PREREG.md` 가 정본이고 이 파일이 코드로 옮긴다.

★이 파일과 `analyze_t11.py` 는 **러너보다 먼저** 쓰였다(2026-09-04 규율).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
ZIP = os.path.abspath(os.path.join(ROOT, '..', 'data',
                                   'thirdparty_synth_zenodo17359600', 'Data.zip'))
WANT_BYTES = 12828148812

# ── 결과 전 고정 (PREREG §2·§3) ────────────────────────────────────────────
N_FILE, SKIP = 24, 36          # sorted() 의 37–60번째 (T1:1–12 · T1b:13–24 · T10:25–36)
LADDER = (12, 50, 200, 800)    # 프레임당 점 수
PRIMARY_DENSITY = 200
WIN_S = 40.0                   # 관측 창 (저들 dt=1.0 s ⟹ 40 프레임)
HS_SPLIT = 2.0                 # Hs 층 경계 (결과 전 고정)
ARMS = ('SHIP', 'BANK', 'ZERO')
THRESH_SKILL, THRESH_UNDECIDED = 17, 13   # ≥17 SKILL · 13–16 UNDECIDED · ≤12 NO-SKILL
SEED = 20260904
GT_CONST = 0.5                 # ★붙이는 상수 장 --- 참 표면 정보 0비트


def label(w):
    if w >= THRESH_SKILL:
        return 'EXTACC-SKILL-AT-200'
    if w >= THRESH_UNDECIDED:
        return 'EXTACC-UNDECIDED'
    return 'EXTACC-NO-SKILL'


def blocked_by(rows):
    """⛔**긴 이름부터** 접두 비교 --- `T1b` 개정문 §5 의 결함을 되풀이하지 않는다."""
    block = ('G-NOLEAK', 'G-ARM', 'G-SIZE', 'G-DISJOINT', 'G-DENSITY',
             'G-ZERO', 'G-N', 'G-ENV', 'G-BLOCK')
    order = sorted(block, key=len, reverse=True)
    out = []
    for name, ok, _ in rows:
        if ok:
            continue
        nm = name.replace('★', '').strip()
        for b in order:
            if nm.startswith(b):
                out.append(b)
                break
    return out
