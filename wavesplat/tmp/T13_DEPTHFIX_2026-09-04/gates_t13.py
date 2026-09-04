"""`T13-DEPTHFIX` 상수·판정 단일출처. ⛔`PREREG.md` 가 정본.

★관측·창·채점 규약은 `T11 gates_t11` 을 **임포트**한다(⛔손복사 금지 — 재발 결함 부류).
★게이트·분석기는 러너보다 **먼저** 쓴다.
"""
import json, os, sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
REPO = os.path.abspath(os.path.join(ROOT, '..'))
T11 = os.path.abspath(os.path.join(HERE, '..', 'T11_EXTACC_2026-09-03'))
for _p in (HERE, ROOT, os.path.join(ROOT, 'scripts'), T11):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import gates_t11 as G11                                           # noqa: E402  ★단일출처

# ── T11 에서 임포트하는 규약 (⛔손복사 0) ──────────────────────────────────
ZIP, WANT_BYTES = G11.ZIP, G11.WANT_BYTES
SEED, J0, WIN_S, DENSITY = G11.SEED, G11.J0, G11.WIN_S, G11.DENSITY
DEPTH, GT_CONST, RESCALE_S = G11.DEPTH, G11.GT_CONST, G11.RESCALE_S
N_MODE, HS_SPLIT = G11.N_MODE, G11.HS_SPLIT
AUDIT_SMOKE_IDX, SMOKE_FILES = G11.AUDIT_SMOKE_IDX, G11.SMOKE_FILES
SAMPLE_T11 = os.path.join(T11, 'sample_t11.json')
RESULTS_T11 = os.path.join(T11, 'results_t11.jsonl')
POP_RANGE = (60, 120)
FRESH_SEED_OFFSET = 100

# ── 팔 (PREREG §2) ────────────────────────────────────────────────────────
ARMS_ALL = ('ZERO', 'ORACLE', 'SHIP', 'FIX')          # 52파일 전부
ARMS_T11ONLY = ('RESCALE', 'RESCALE-FIX')             # T11-24 에서만
FIX_ARMS = ('FIX', 'RESCALE-FIX')                     # 맵 처치가 걸리는 팔
N_TOTAL, N_T11, N_FRESH = 52, 24, 28
N_ROWS = N_TOTAL * 4 + N_T11 * 2                      # 208 + 48 = 256

# ── 판정 (PREREG §4) ──────────────────────────────────────────────────────
THRESH_WIN = 33      # n=52, α=.05 단측, p(≥33|.5)=0.0352
THRESH_HURT = 19     # w<=19 (대칭 다리 — ★T11 과 달리 **양방향이 설계 전제**다)
THRESH_T11 = 17      # 층 (n=24) — 기술통계
THRESH_FRESH = 19    # 층 (n=28) — 기술통계
CTRL_BAND = (8, 16)  # 음성대조 귀무 예상 구간 (n=24, 확률 0.936)
PRED_Q = {'fix_ship': 0.65, 'fix_zero': 0.50, 'ctrl': 0.50}
MDE = {'q50': 0.6245, 'q80': 0.6795, 'q90': 0.707,
       'unrejectable_w': (27, 32), 'q80_t11': 0.760, 'q80_fresh': 0.7305}


def p_ge(k, n, q=0.5):
    return sum(comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(k, n + 1))


def sign_p(k, n):
    return p_ge(k, n, 0.5) if n else None


def archive_mats():
    import zipfile
    with zipfile.ZipFile(ZIP) as zf:
        return sorted(n for n in zf.namelist() if n.endswith('.mat'))


def sample():
    """★**집합 차로 계산**한다 — 손목록이 아니다 (PREREG §3).

    모집단 `mixed_data2`(인덱스 60–119) − 감사 스모크 8 = **52 전부**. 선택이 없다.
    """
    mats = archive_mats()
    pop = mats[POP_RANGE[0]:POP_RANGE[1]]
    exposed = {mats[i] for i in AUDIT_SMOKE_IDX if i < len(mats)}
    return sorted(set(pop) - exposed)


def layers():
    """`T11-24`(앵커·노출) / `FRESH-28`(두 팔 blind) 로 가른다."""
    t11 = json.load(open(SAMPLE_T11, encoding='utf-8'))
    s = sample()
    return list(t11), sorted(set(s) - set(t11))


def seed_of(block):
    """시드 규약 (PREREG §3) — 앵커 재현에 필수. ⛔T11·스모크와 서로소."""
    t11, fresh = layers()
    if block in t11:
        return SEED + t11.index(block) + 1              # T11 run_t11.py 규약
    return SEED + FRESH_SEED_OFFSET + fresh.index(block) + 1


def label(w_fix_ship, w_fix_zero):
    """5라벨이 9조합을 **전부** 덮는다 (PREREG §4). `UNDECIDED` 없음.

    ⛔`FIX-UNDETECTED` 를 *"수심은 원인이 아니다"*·*"교락은 무해하다"* 로 읽지 않는다 —
    **검정력의 진술**이다(`MDE['q80']` 병기 의무).
    ⛔`DEPTH-WAS-MASKING-SKILL` 을 *"T11 BAND-MISMATCH 철회"* 로 읽지 않는다.
    """
    helps = w_fix_ship >= THRESH_WIN
    hurts = w_fix_ship <= THRESH_HURT
    skill = w_fix_zero >= THRESH_WIN
    if hurts:
        return 'FIX-HURTS'
    if helps and skill:
        return 'DEPTH-WAS-MASKING-SKILL'
    if helps:
        return 'FIX-HELPS-NO-SKILL'
    return 'SKILL-FIX-UNDETECTED' if skill else 'FIX-UNDETECTED'


def label_probs(q_fs=None, q_fz=None):
    """⛔독립 근사 — PREREG §5 가 그 한계를 결과 전에 공시했다(양의 상관 ⟹ 대각이 더 크다)."""
    q_fs = PRED_Q['fix_ship'] if q_fs is None else q_fs
    q_fz = PRED_Q['fix_zero'] if q_fz is None else q_fz
    n = N_TOTAL
    ph = p_ge(THRESH_WIN, n, q_fs)
    pu = sum(comb(n, i) * q_fs ** i * (1 - q_fs) ** (n - i)
             for i in range(0, THRESH_HURT + 1))
    pm = 1.0 - ph - pu
    ps = p_ge(THRESH_WIN, n, q_fz)
    return {'FIX-HURTS': pu, 'DEPTH-WAS-MASKING-SKILL': ph * ps,
            'FIX-HELPS-NO-SKILL': ph * (1 - ps),
            'SKILL-FIX-UNDETECTED': pm * ps, 'FIX-UNDETECTED': pm * (1 - ps)}


def blocked_by(rows):
    """⛔**긴 이름부터** 접두 비교."""
    block = ('G-MAP', 'G-BMAPCALLS', 'G-ANCHOR-SHIP', 'G-ANCHOR-RESCALE', 'G-REPEAT',
             'G-DEVICE', 'G-OOD-UNTOUCHED', 'G-SAMPLE', 'G-POP', 'G-N', 'G-ENV',
             'G-ARM', 'G-ATOMS', 'G-NOLEAK', 'G-CAP', 'G-TORIGIN', 'G-RESCALE',
             'G-FINITE', 'G-DENSITY', 'G-CLAMP', 'G-WRAPPER', 'G-COV', 'G-BLOCK',
             'G-FREEZE')
    order = sorted(block, key=len, reverse=True)
    out = []
    for name, ok, _ in rows:
        if ok:
            continue
        nm = name.replace('★', '').replace('⚠', '').strip()
        for b in order:
            if nm.startswith(b):
                out.append(b)
                break
    return out
