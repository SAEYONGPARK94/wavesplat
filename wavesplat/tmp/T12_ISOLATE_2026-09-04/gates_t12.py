"""`T12-ISOLATE` **판정 단일출처** — 분석면.

⛔`PREREG.md §4` 가 정본. ★상수는 `consts_t12.py`(측정면)에 있고 이 파일은 **판정만** 한다
  — `T11 AMENDMENT_3 §2` 가 겪은 「한 파일 안에서 측정면/분석면이 갈림」을 없앤다.
★게이트·분석기는 **러너보다 먼저** 쓴다(2026-09-04 규율).
"""
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import consts_t12 as C                                            # noqa: E402


def p_ge(k, n, q=0.5):
    return sum(comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(k, n + 1))


def sign_p(k, n):
    return p_ge(k, n, 0.5) if n else None


def label(w_primary, w_sec_max):
    """귀속 사다리 (PREREG §4.2) — **4라벨이 4조합을 전부 덮는다**. `UNDECIDED` 없음.

    ⛔`NO-SINGLE-AXIS-DETECTED` 는 **「미검출」**이다. *"인수분해되지 않는다"*·
    *"상호작용이다"*·*"어느 상수도 원인이 아니다"*·*"도달 불가"* 로 읽는 것을 **금지**한다.
    선례 = `M8b` 의 `GAP-NOT-FACTORIZABLE` 개명 · `T5-ENS` 의 *"「검출 불가」이지 폐쇄가 아니다"*.
    """
    p_win = w_primary >= C.THRESH_PRIMARY
    any_sec = w_sec_max >= C.THRESH_SEC
    if p_win and not any_sec:
        return 'PROPOSAL-SCALE-CARRIES'
    if p_win and any_sec:
        return 'MULTIPLE-AXES-CARRY'
    if any_sec:
        return 'OTHER-AXIS-CARRIES'
    return 'NO-SINGLE-AXIS-DETECTED'


def label_probs(q_map=None):
    """★등록 예측 하의 라벨 확률 — **계산한다**(⛔믿음 선언 금지, PREREG §5.2).

    `T11` v1 이 *"예측을 적고도 그 예측 하의 검정력을 안 쟀다"* 로 폐기된 그 항목이다.
    """
    q = dict(C.PRED_Q if q_map is None else q_map)
    n = C.N_PRIMARY
    a = p_ge(C.THRESH_PRIMARY, n, q[C.PRIMARY_ARM])
    no_sec = 1.0
    for arm in C.SEC8:
        no_sec *= (1.0 - p_ge(C.THRESH_SEC, n, q[arm]))
    b = 1.0 - no_sec                       # P(적어도 한 SEC 축이 문턱을 넘는다)
    return {'PROPOSAL-SCALE-CARRIES': a * (1 - b), 'MULTIPLE-AXES-CARRY': a * b,
            'OTHER-AXIS-CARRIES': (1 - a) * b, 'NO-SINGLE-AXIS-DETECTED': (1 - a) * (1 - b)}


def blocked_by(rows):
    """⛔**긴 이름부터** 접두 비교 (`G-NOLEAK` 이 `G-N` 보다 먼저 잡혀야 한다).

    ⛔`T11` 이 이 함수가 **항상 빈 리스트**를 내는 버그를 겪었다(차단 게이트 14개가 장식이
    될 뻔했다) ⟹ `G-BLOCK` 자가진단이 필수다.
    """
    block = ('G-KNOB', 'G-RESTORE', 'G-ANCHOR', 'G-EFFECT', 'G-PROPID',
             'G-RNGSTREAM', 'G-GROWFIRE', 'G-NOLEAK', 'G-ATOMS', 'G-ARM',
             'G-SHIPPIN', 'G-SAMPLE', 'G-HELDOUT', 'G-N', 'G-FINITE',
             'G-TORIGIN', 'G-DENSITY', 'G-RESCALE', 'G-SIZE', 'G-ENV',
             'G-FREEZE', 'G-BLOCK', 'G-COV')
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
