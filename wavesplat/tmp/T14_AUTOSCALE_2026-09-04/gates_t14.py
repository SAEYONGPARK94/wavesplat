"""`T14-AUTOSCALE` 상수·판정 단일출처. ⛔`PREREG.md` 가 정본.
★관측·창·채점 규약은 `T11 gates_t11` 을 **임포트**한다(⛔손복사 금지)."""
import json, os, sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
T11 = os.path.abspath(os.path.join(HERE, '..', 'T11_EXTACC_2026-09-03'))
for _p in (HERE, ROOT, os.path.join(ROOT, 'scripts'), T11):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import gates_t11 as G11                                           # noqa: E402
import scale_rules_t14 as SR                                      # noqa: E402

ZIP, WANT_BYTES = G11.ZIP, G11.WANT_BYTES
J0, WIN_S, DENSITY = G11.J0, G11.WIN_S, G11.DENSITY
DEPTH, GT_CONST, N_MODE, HS_SPLIT = G11.DEPTH, G11.GT_CONST, G11.N_MODE, G11.HS_SPLIT
AUDIT_SMOKE_IDX, SMOKE_FILES = G11.AUDIT_SMOKE_IDX, G11.SMOKE_FILES
SAMPLE_T11 = os.path.join(T11, 'sample_t11.json')
SMOKE_T11 = os.path.join(T11, 'results_t11_SMOKE.jsonl')
POP_RANGE = (60, 120)
SEED14 = 20260914                       # ⛔앵커 시드(20260905/06)와 충돌하지 않게 고른 값
ANCHOR_SEEDS = {G11.SMOKE_FILES[0]: 20260905, G11.SMOKE_FILES[1]: 20260906}

# ── 팔 (PREREG §1) ────────────────────────────────────────────────────────
S_FIXED = {'SHIP': 1.0, 'FIX2': 2.0,
           'SW-1.414': 2 ** 0.5, 'SW-2.828': 2 ** 1.5, 'SW-4.000': 4.0}
T1_ARMS = ('ZERO', 'SHIP', 'FIX2', 'AUTO-AP', 'AUTO-BAND')
T2_ARMS = ('SW-1.414', 'SW-2.828', 'SW-4.000')
ANCHOR_ARMS = ('ZERO', 'SHIP', 'FIX2')
AUTO_ARMS = ('AUTO-AP', 'AUTO-BAND')
N_T1, N_T2, N_T3, N_ANC = 28, 12, 6, 2
N_ROWS = N_T1 * 5 + N_T2 * 3 + N_T3 * 5 + N_ANC * 3          # 140+36+30+6 = 212
N_RUNS = N_T1 * 4 + N_T2 * 3 + N_T3 * 4 + N_ANC * 2          # 112+36+24+4 = 176

# ── 판정 (PREREG §3) ──────────────────────────────────────────────────────
THRESH_PRIMARY = 20     # n=28, Bonferroni α/2 ⟹ P(W>=20|.5)=0.017849
THRESH_CTRL = 19        # 양성대조 FIX2<ZERO, 무보정 ⟹ P=0.043579
SIG_WORSE = 8           # w<=8 (P=0.017849) — 라벨 아님, 기록만
PRED_Q = {'ap': 0.85, 'band': 0.75, 'fix': 0.95}
MDE = {'primary_q50': 0.6941, 'primary_q80': 0.7630, 'primary_q90': 0.7958,
       'sec_q80': 0.7304, 'unrejectable_w': (15, 19)}
INVERT_TOL_REL = 1e-4   # max|P_implicit − P_explicit| <= 1e-4·σ4
KD_MIN = 3.141592653589793


def p_ge(k, n, q=0.5):
    return sum(comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(k, n + 1))


def sign_p(k, n):
    return p_ge(k, n, 0.5) if n else None


def archive_mats():
    import zipfile
    with zipfile.ZipFile(ZIP) as zf:
        return sorted(n for n in zf.namelist() if n.endswith('.mat'))


def tiers():
    """★층을 **집합 차·정렬 규칙으로 계산**한다 (⛔손목록 금지)."""
    mats = archive_mats()
    pop = mats[POP_RANGE[0]:POP_RANGE[1]]
    exposed = {mats[i] for i in AUDIT_SMOKE_IDX if i < len(mats)}
    t11 = set(json.load(open(SAMPLE_T11, encoding='utf-8')))
    t1 = sorted(set(pop) - t11 - exposed)                      # 28
    import re
    def ah(b):
        m = re.search(r'angle(\d+)h([\d.]+)\.mat', b)
        return int(m.group(1)), float(m.group(2))
    byang = {}
    for b in t1:
        byang.setdefault(ah(b)[0], []).append(b)
    t2 = sorted(b for a in byang                                # 각도별 Hs 오름차순 앞 2
                for b in sorted(byang[a], key=lambda z: ah(z)[1])[:2])
    t3 = sorted(n for n in mats if 'outofdistribution' in n)     # 6
    return t1, t2, t3, list(SMOKE_FILES)


def seed_of(block, tier):
    if tier == 'ANCHOR':
        return ANCHOR_SEEDS[block]
    t1, t2, t3, _ = tiers()
    order = t1 + t3
    return SEED14 + order.index(block) + 1


def label(w_ap, w_band, w_fix):
    """5라벨이 (F)×(a,b) 를 **전부** 덮는다. `UNDECIDED` 없음 (PREREG §3).

    ⛔`NO-AUTOSCALE-DETECTED` 는 **「미검출」**이다 — *"자동 선택은 불가능하다"*·
    *"도달 불가"* 로 읽는 것을 금지한다(선례 `T5-ENS`·`T11` 개정문 2 §1).
    ⛔`AUTOSCALE-WORKS-*` 를 *"우리가 저들보다 낫다"* 로 읽지 마라 — 외부 기준선 행이 0개다.
    """
    if w_fix < THRESH_CTRL:
        return 'ANCHOR-DID-NOT-REPLICATE'
    a = w_ap >= THRESH_PRIMARY
    b = w_band >= THRESH_PRIMARY
    if a and b:
        return 'AUTOSCALE-WORKS-BOTH-RULES'
    if a:
        return 'AUTOSCALE-WORKS-APERTURE'
    if b:
        return 'AUTOSCALE-WORKS-BAND'
    return 'NO-AUTOSCALE-DETECTED'


def label_probs(q_ap=None, q_band=None, q_fix=None):
    """⚠다리 독립 가정 — PREREG §3.1 이 그 한계를 **결과 전에** 공시했다."""
    q_ap = PRED_Q['ap'] if q_ap is None else q_ap
    q_band = PRED_Q['band'] if q_band is None else q_band
    q_fix = PRED_Q['fix'] if q_fix is None else q_fix
    pa = p_ge(THRESH_PRIMARY, N_T1, q_ap)
    pb = p_ge(THRESH_PRIMARY, N_T1, q_band)
    pf = p_ge(THRESH_CTRL, N_T1, q_fix)
    return {'AUTOSCALE-WORKS-BOTH-RULES': pf * pa * pb,
            'AUTOSCALE-WORKS-APERTURE': pf * pa * (1 - pb),
            'AUTOSCALE-WORKS-BAND': pf * (1 - pa) * pb,
            'NO-AUTOSCALE-DETECTED': pf * (1 - pa) * (1 - pb),
            'ANCHOR-DID-NOT-REPLICATE': 1 - pf}


def blocked_by(rows):
    """⛔**긴 이름부터** 접두 비교."""
    block = ('G-S-BLIND', 'G-S-SRC', 'G-S-READBACK', 'G-EQUIVAR', 'G-LTRAIN', 'G-KTRAIN',
             'G-SRC-PIN', 'G-INVERT', 'G-PURE', 'G-DEEP', 'G-SHIPID', 'G-ANCHOR-T11',
             'G-DEPLOY', 'G-NOLEAK', 'G-ATOMS', 'G-ARM', 'G-WRAPPER', 'G-CAP',
             'G-TORIGIN', 'G-DENSITY', 'G-SAMPLE', 'G-POP', 'G-N', 'G-FINITE',
             'G-SIZE', 'G-ENV', 'G-BLOCK', 'G-COV')
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
