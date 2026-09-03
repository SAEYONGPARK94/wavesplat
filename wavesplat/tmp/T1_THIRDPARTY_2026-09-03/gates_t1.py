"""`T1-THIRDPARTY` v2 규칙 단일출처. ⛔`PREREG.md` 가 정본이고 이 파일이 코드로 옮긴다."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
ZIP = os.path.abspath(os.path.join(ROOT, '..', 'data',
                                   'thirdparty_synth_zenodo17359600', 'Data.zip'))
SHARED = os.path.join(ROOT, 'tmp', '_shared')

# ── 결과 전 고정 (PREREG v2 §1) ─────────────────────────────────────────────
WANT_BYTES = 12828148812
N_BLOCK = 12
N_FRAME = 24
N_SUR = 32
SEED = 20260903
Q = 0.95                     # nearest-rank 백분위
LABEL_HI, LABEL_LO = 10, 5   # ≥10 → COUPLED · 5–9 → PARTIAL · ≤4 → INDIST
NULL_TOL = 1e-10             # G-NULL: |FFT| 최대 상대차
NULLMOMENT_TOL = 0.02        # G-NULLMOMENT: 대체표본 왜도 귀무의 |평균|
POWER_POS_MIN = 7            # G-POWER 양성 대조 ≥7/8 (우리 HOS)
POWER_NEG_MAX = 1            # G-POWER 음성 대조 ≤1/8 (우리 선형)


def label(w):
    if w >= LABEL_HI:
        return 'THIRDPARTY-PHASE-COUPLED'
    if w >= LABEL_LO:
        return 'THIRDPARTY-PARTIAL'
    return 'THIRDPARTY-PHASE-RANDOM-INDISTINGUISHABLE'


def blocked_by(rows):
    """★자가진단 — 차단 게이트 중 실패한 것의 **이름 접두**를 돌려준다.

    ⛔선례: `T5-ENS` 에서 이름 전체를 비교해 **항상 빈 리스트**를 냈다(게이트 14개가
    전부 장식이 될 뻔했다). 여기서는 접두 매칭으로 고친다.
    """
    block = ('G-SIZE', 'G-NULL', 'G-NULLMOMENT', 'G-POWER', 'G-SHARED',
             'G-N', 'G-NORUN', 'G-BLOCK')
    out = []
    for name, ok, _ in rows:
        if ok:
            continue
        for b in block:
            if name.replace('★', '').strip().startswith(b):
                out.append(b)
                break
    return out
