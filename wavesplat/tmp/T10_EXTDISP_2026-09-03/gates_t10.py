"""`T10-EXTDISP` 규칙 단일출처. ⛔`PREREG.md` 가 정본이고 이 파일이 코드로 옮긴다."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
ZIP = os.path.abspath(os.path.join(ROOT, '..', 'data',
                                   'thirdparty_synth_zenodo17359600', 'Data.zip'))
WANT_BYTES = 12828148812

# ── 결과 전 고정 (PREREG §1·§2·§5) ─────────────────────────────────────────
N_FILE, SKIP = 12, 24        # `sorted()` 의 25–36번째 (T1: 1–12, T1b: 13–24)
KMIN, KMAX, NBIN = 0.012, 0.30, 24
MIN_RIDGE, MIN_MODES = 15, 8
LABEL_HI, LABEL_LO = 8, 3    # ≥8 → SHALLOW · 3–7 → MIXED · ≤2 → NOT-SHALLOW
POWER_SHALLOW = (10.0, 20.0, 30.0)   # 문턱 아래여야 한다
POWER_DEEP = 400.0                   # 문턱 위여야 한다


def threshold():
    """⛔**손으로 적지 않는다** --- 동결 귀무에서 되읽는다(`G-NULL-CAL`)."""
    d = json.load(open(os.path.join(HERE, 'null_t10.json'), encoding='utf-8'))
    return float(min(x['ratio'] for x in d['null']))


THRESH = threshold()


def label(w):
    if w >= LABEL_HI:
        return 'SHALLOW-DETECTED'
    if w >= LABEL_LO:
        return 'MIXED'
    return 'NOT-SHALLOW'


def blocked_by(rows):
    """★차단 게이트의 접두를 돌려준다.

    ⛔**긴 이름부터** 비교한다 --- `T1b` 개정문 §5 가 기록한 결함(`G-NULLMOMENT` 실패가
    `G-NULL` 로 보고됨)을 되풀이하지 않는다.
    """
    block = ('G-SIZE', 'G-NULL-CAL', 'G-POWER', 'G-RIDGE', 'G-BAND',
             'G-N', 'G-DISJOINT', 'G-NORUN', 'G-BLOCK')
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
