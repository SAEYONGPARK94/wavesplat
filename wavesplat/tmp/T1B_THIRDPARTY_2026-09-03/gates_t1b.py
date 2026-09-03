"""`T1b` 규칙 단일출처 — ⛔`T1` 의 규약을 **상속**한다(사본 재구현 금지)."""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
T1 = os.path.abspath(os.path.join(HERE, '..', 'T1_THIRDPARTY_2026-09-03'))
sys.path.insert(0, T1)
import gates_t1 as T                                                 # noqa: E402

ZIP, SHARED, ROOT = T.ZIP, T.SHARED, T.ROOT
WANT_BYTES, N_BLOCK, N_SUR, SEED, Q = T.WANT_BYTES, T.N_BLOCK, T.N_SUR, T.SEED, T.Q
NULL_TOL, NULLMOMENT_TOL = T.NULL_TOL, T.NULLMOMENT_TOL
POWER_POS_MIN, POWER_NEG_MAX = T.POWER_POS_MIN, T.POWER_NEG_MAX

# ── T1b 에서만 바뀌는 것 (결과 전 고정) ─────────────────────────────────────
N_FRAME = 502                # ★T1 은 24 였다
SKIP = 12                    # `sorted()` 의 앞 12개는 T1 이 썼다 ⟹ 13–24번째를 쓴다
LABEL_HI, LABEL_LO = 8, 4    # ≥8 → COUPLED · 4–7 → PARTIAL · ≤3 → INDIST


def label(w):
    if w >= LABEL_HI:
        return 'THIRDPARTY-PHASE-COUPLED'
    if w >= LABEL_LO:
        return 'THIRDPARTY-PARTIAL'
    return 'THIRDPARTY-PHASE-RANDOM-INDISTINGUISHABLE'


def blocked_by(rows):
    block = ('G-SIZE', 'G-NULL', 'G-NULLMOMENT', 'G-POWER', 'G-SHARED',
             'G-N', 'G-NORUN', 'G-BLOCK', 'G-DISJOINT')
    out = []
    for name, ok, _ in rows:
        if ok:
            continue
        for b in block:
            if name.replace('★', '').strip().startswith(b):
                out.append(b); break
    return out
