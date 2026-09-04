"""`T15-HOMEREG` 판정 단일출처 — 분석면. ⛔`PREREG.md §5` 가 정본.
★상수는 `consts_t15.py`(측정면)에 있고 이 파일은 **판정만** 한다."""
import os, sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import consts_t15 as C                                            # noqa: E402

LAYERS = ('crowded', 'sparse')
# ★MDE 는 **여기서도** 이름을 갖는다 — 손계산 값이 verdict 로 들어가는 경로를 한 곳으로 묶는다
#   (`check_prereg C1`). 값의 유일 정의는 `consts_t15.MDE` 이고 이 파일은 **재수출만** 한다.
MDE = C.MDE
Q80 = C.MDE['q80']              # ⟹ ★음성 결과 인용 시 **병기 의무**
UNREJECTABLE_W = C.MDE['unrejectable_w']


def p_le(k, n, q=0.5):
    return sum(comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(0, k + 1))


def p_ge(k, n, q=0.5):
    return sum(comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(k, n + 1))


def sign_p_hurt(w, n):
    """해침 다리의 단측 p (낮은 승수가 나쁘다)."""
    return p_le(w, n) if n else None


def judged_layers(W_ctrl):
    """양성대조가 복제된 층 = **처치 판정을 받는 층**. 실패 층은 제외하고 인쇄한다(PREREG §8)."""
    return [L for L in LAYERS if W_ctrl.get(L, (0, 0))[0] >= C.THRESH_HELP]


def label(W, W_ctrl, within_floor):
    """5라벨이 전 조합을 덮는다 (PREREG §5.1). `UNDECIDED` 없음. ★**문자열만** 돌려준다.

    ⛔`CHANGE-UNDETECTED`·`NO-HARM-*` 를 *"안전하다/무해하다"* 로 읽는 것을 **금지**한다 —
    **검정력의 진술**이다(`Q80 = 0.3185` 병기 의무).
    ★양성대조가 실패한 층은 **처치 판정에서 제외**하고 그 사실을 인쇄한다.
    """
    ok = judged_layers(W_ctrl)
    if not ok:
        return 'ANCHOR-DID-NOT-REPLICATE'
    if any(W[L][0] <= C.THRESH_HURT for L in ok):
        return 'HOME-REGRESSION'
    if all(W[L][0] >= C.THRESH_HELP for L in ok):
        return 'HOME-ALSO-IMPROVES'
    if all(within_floor.get(L) for L in ok):
        return 'NO-HARM-WITHIN-INTERVENTION-FLOOR'
    return 'CHANGE-UNDETECTED'


def label_probs(q=None, q_ctrl=None):
    """★등록 예측 하의 **다섯 라벨 전부**의 확률 — **계산한다**(⛔믿음 선언 금지, PREREG §6).

    양성대조도 모형에 넣는다(`q_ctrl`) — ⛔*"통과한다고 가정"* 은 `ANCHOR-DID-NOT-REPLICATE`
    를 확률 없는 라벨로 남기고, 그것이 `A3 §5` 가 이름 붙인 **음성 라벨 뭉개기** 부류다.
    ⚠단 `NO-HARM-WITHIN-INTERVENTION-FLOOR` 와 `CHANGE-UNDETECTED` 의 **분할은 q 가 모형하지
      않는다**(크기 비교다) ⟹ 둘 다 같은 버킷의 **상한**으로 보고한다. 그 사실을 여기 적는다.
    """
    q = C.PRED_Q if q is None else q
    q_ctrl = C.PRED_Q_CTRL if q_ctrl is None else q_ctrl
    n = C.N_LAYER
    ph = p_le(C.THRESH_HURT, n, q)          # 한 층이 해침을 낼 확률
    pw = p_ge(C.THRESH_HELP, n, q)          # 한 층이 이김을 낼 확률
    pc = p_ge(C.THRESH_HELP, n, q_ctrl)     # 한 층이 **양성대조를 복제**할 확률
    anchor = (1 - pc) ** 2
    reg = pc ** 2 * (1 - (1 - ph) ** 2) + 2 * pc * (1 - pc) * ph
    imp = pc ** 2 * pw ** 2 + 2 * pc * (1 - pc) * pw
    bucket = 1.0 - anchor - reg - imp        # 등가 판정이 가르는 몫 (⚠분할 미모형)
    return {'ANCHOR-DID-NOT-REPLICATE': anchor,
            'HOME-REGRESSION': reg,
            'HOME-ALSO-IMPROVES': imp,
            'NO-HARM-WITHIN-INTERVENTION-FLOOR': bucket,   # ★상한 (분할 미모형)
            'CHANGE-UNDETECTED': bucket,                   # ★상한 (분할 미모형)
            '_bucket_is_upper_bound_for_both': True,
            '_per_layer': {'hurt': ph, 'help': pw, 'ctrl_replicates': pc},
            '_q': q, '_q_ctrl': q_ctrl}


def blocked_by(rows):
    """⛔**긴 이름부터** 접두 비교."""
    block = ('G-TREATPIN', 'G-KNOB', 'G-RESTORE', 'G-FRAME', 'G-SIBLING', 'G-CORPUS',
             'G-CTRL', 'G-FLOOR', 'G-ARM', 'G-NOLEAK', 'G-ATOMS', 'G-CAP', 'G-FINITE',
             'G-N', 'G-DEPTH', 'G-ZERO', 'G-SEED', 'G-SMOKE', 'G-PROPID',
             'G-ENV', 'G-BLOCK', 'G-COV')
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
