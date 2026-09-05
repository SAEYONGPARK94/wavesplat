"""`T16-BANDFIX` 판정 단일출처 — **분석면**. ⛔`PREREG.md` 가 정본이고 이 파일이 코드로 옮긴다.

★러너보다 **먼저** 쓰였다(2026-09-04 규율: 사전등록 → 분석기 → 게이트 → 러너 → 스모크 → 본실행).
⛔이 파일은 **판정만** 한다 — 규칙(`scale_rules_t16.py`)·러너는 측정면이다(`PREREG §9.1`).

★★**이 파일의 규율 세 가지** (전부 저장소가 대가를 치른 부류다)
1. ⛔**손타이핑 0.** 문턱·MDE·라벨 확률은 **계산한다**. `PREREG` 가 인쇄한 수는 `REGISTERED`
   에 **핀**으로만 들어가고 `registration_drift()` 가 계산값과 대조한다 ⟹ 어긋나면 분석기의
   `G-COV` 가 죽는다. (형제 배치가 **손타이핑 확률표로 두 번** 당했다.)
2. ⛔**배포·규칙 상수는 임포트한다.** `S_MIN/S_MAX/L_TRAIN/K_TRAIN` 은 `scale_rules_t14`,
   관측·창 규약은 `gates_t11`, 배포 상수는 `wavesplat.optimizer.*` 에서 온다.
3. ⛔**`(x or 기본값)` 금지** — `0.0` 은 falsy 다(`T14 AMENDMENT_1 §1`: 그 관용구가 **완벽함을
   위반으로** 셌다). 이 파일과 분석기는 전부 `is None` 으로 가른다.
"""
import json
import math
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
T11 = os.path.abspath(os.path.join(HERE, '..', 'T11_EXTACC_2026-09-03'))
T14 = os.path.abspath(os.path.join(HERE, '..', 'T14_AUTOSCALE_2026-09-04'))
for _p in (HERE, ROOT, os.path.join(ROOT, 'scripts'), T11, T14):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import gates_t11 as G11                                            # noqa: E402
import scale_rules_t14 as SR14                                     # noqa: E402

try:                        # ★측정면. 게이트를 **러너보다 먼저** 쓰므로 아직 없을 수 있다.
    import scale_rules_t16 as SR16                                 # noqa: E402
except ImportError:         # ⛔없으면 `G-RULEID` 가 FAIL 한다(분석기). 조용히 통과시키지 않는다.
    SR16 = None

# ── 관측·창·채점 규약 — ★`T11` 에서 **임포트**한다 (⛔손복사 금지, PREREG §2) ──────────
ZIP, WANT_BYTES = G11.ZIP, G11.WANT_BYTES
J0, WIN_S, DENSITY = G11.J0, G11.WIN_S, G11.DENSITY
DEPTH, GT_CONST = G11.DEPTH, G11.GT_CONST
AUDIT_SMOKE_IDX, SMOKE_FILES = G11.AUDIT_SMOKE_IDX, G11.SMOKE_FILES
SAMPLE_T11 = os.path.join(T11, 'sample_t11.json')
SMOKE_T11 = os.path.join(T11, 'results_t11_SMOKE.jsonl')
SAMPLE_T16 = os.path.join(HERE, 'sample_t16.json')

# ── 표본·시드 (PREREG §2) ────────────────────────────────────────────────────
POP_RANGE = (0, 60)             # `sorted()` 아카이브 인덱스 창 = `Data/mixed_data/` 전수
POP_PREFIX = 'Data/mixed_data/'
SEED16 = 20261300               # ⛔초안 20261100 은 `T15`(20261101–228)와 60/60 충돌이었다(R-1)
ANCHOR_SEEDS = {SMOKE_FILES[0]: 20260905, SMOKE_FILES[1]: 20260906}
# ★형제 배치 — `G-XBATCH`(성과 노출, 차단) 와 `SEC-BLIND`(속성 노출, 기술통계)의 출처.
#   ⛔`PREREG §2.1` 이 이름을 등록했다. 목록에 없는 배치를 조용히 더하지 마라.
XBATCH_OUTCOME = (('T11_EXTACC_2026-09-03', 'results_t11.jsonl'),
                  ('T12_ISOLATE_2026-09-04', 'results_t12.jsonl'),
                  ('T13_DEPTHFIX_2026-09-04', 'results_t13.jsonl'),
                  ('T14_AUTOSCALE_2026-09-04', 'results_t14.jsonl'))
PROPERTY_BATCH_DIRS = ('T1_THIRDPARTY_2026-09-03', 'T1B_THIRDPARTY_2026-09-03',
                       'T10_EXTDISP_2026-09-03')

# ── 팔 (PREREG §1) — ⛔`{}` 로 팔을 정의하지 않는다 ──────────────────────────
S_FIXED = {'SHIP': 1.0, 'FIX2': 2.0}
TIERP_ARMS = ('ZERO', 'SHIP', 'FIX2', 'AUTO-AP', 'AUTO-BAND', 'AUTO-BANDR')
ANCHOR_ARMS = ('ZERO', 'SHIP', 'FIX2')
AUTO_ARMS = ('AUTO-AP', 'AUTO-BAND', 'AUTO-BANDR')
CANDIDATES = ('AUTO-AP', 'AUTO-BANDR')     # ★`G-EQUIVAR` 가 **차단**하는 두 규칙
COMPARATORS = ('AUTO-BAND',)               # ⚠비교자 — `G-EQUIVAR-OLD` 는 **기록 전용**
BEAMFORM_ARMS = ('AUTO-BAND', 'AUTO-BANDR')   # ⛔`AUTO-AP` 는 빔포머를 **0회** 부른다(§8)
PRIMARY_ARM = 'AUTO-BANDR'
N_P, N_ANC = 60, 2
N_ROWS = N_P * len(TIERP_ARMS) + N_ANC * len(ANCHOR_ARMS)          # 360 + 6 = 366
N_RUNS = N_P * (len(TIERP_ARMS) - 1) + N_ANC * (len(ANCHOR_ARMS) - 1)   # 300 + 4 = 304

# ── 규칙 상수 — ★전부 **임포트** (PREREG §1 · REACHABILITY §3) ───────────────
S_MIN, S_MAX = SR14.S_MIN, SR14.S_MAX
L_TRAIN, K_TRAIN = SR14.L_TRAIN, SR14.K_TRAIN
N1 = 120                    # `bank_mmse.py:160`·`:163` 의 `n=120`
ZOOM = 3.5                  # `bank_mmse.py:163` 의 `3.5 * kp`
KLIM1_OLD = 2.0             # `bank_mmse.py:159-160` 의 **절대** 1단 격자
FLOOR = 1e-3                # `bank_mmse.py:161`·`:164` 의 `max(·, 1e-3)`
DK1_OLD = 2.0 * KLIM1_OLD / (N1 - 1)          # 0.033613 — 옛 1단 격자 간격
TWO_PI = 2.0 * math.pi
# ★규칙 호출 수 (§8, 결과 전 유도) = 2 빔포밍 팔 × 4 평가(본 1 + 등변 2 + blind 1) × 2 단
EVALS_PER_ARM = 4
STAGES_PER_EVAL = 2
BEAMFORM_CALLS_PER_ARM = EVALS_PER_ARM * STAGES_PER_EVAL           # 8
BEAMFORM_CALLS_PER_FILE = len(BEAMFORM_ARMS) * BEAMFORM_CALLS_PER_ARM   # 16

# ── 허용오차 (PREREG §5·§6.1·§12-2) ─────────────────────────────────────────
EQ_TOL = 1e-6               # ★`G-EQUIVAR` — `T14` 의 5.88% 보다 엄하다(§9.2-5)
INVERT_TOL_REL = 1e-4       # `max|P_implicit − P_explicit| <= 1e-4·σ4`
READBACK_TOL = 1e-9         # `G-S-READBACK`
KD_MIN = math.pi            # `G-DEEP`
TIE_REL = 1e-6              # ★동점 = **수치 동일**(개정 R-4). ⛔비트 동일 아님
INNER_MARGIN_REPORT = 2.0   # `G-INNER`(b) 보고 문턱 — ⛔막지 않는다
DRIVE_T1_LN = math.log(1.05)          # T1 «처치가 자보다 작다»
DRIVE_T2_SIGN = 55                    # T2 «방향이 섞였다» — 55/60 = 91.7%
GRID_SHAPE = [502, 128, 128]
GRID_DT = 1.0
# ★`G-GRID` — 한 파일 안 x/y 개구 차. ⛔**상대자**다(`AMENDMENT_2_PREDATA.md`).
#   사전등록 §6.2 는 절대 `1e-3` [m] 를 등록했는데 그 자는 **종횡비 질문에 붙은 절대 허용오차**라
#   차원이 어긋난다. 실측 아카이브 전체가 `|Δ| = 0.002 m = 3.23e-6 상대` 이고
#   `mixed_data` 와 `mixed_data2` 는 **두 축 모두 동일**하다 ⟹ 게이트의 등록 목적(기하 동형)은
#   **성립**하고 단위만 틀렸다. ⛔**이것은 완화다** — 사유·측정값·시점은 개정문 2 가 축자로 적는다.
LOBS_XY_REL = 1e-4          # 측정 3.23e-6 대비 약 31배 여유 (⛔관측에 맞춰 조율한 값이 아니다)
LOBS_SD_TOL = 1e-6          # `G-GRID` — 파일 간 `L_obs` 표준편차 (⛔이 자는 **안 건드렸다**)

# ── 다중성 (PREREG §3.1) ────────────────────────────────────────────────────
ALPHA = 0.05
FAMILY_SIZE = 2             # ★PRIMARY 가족 = **2 축**(`S`·`D`) ⟹ Bonferroni α/2
ALPHA_FAM = ALPHA / FAMILY_SIZE


# ── 이항 산수 ───────────────────────────────────────────────────────────────
def p_ge(k, n, q=0.5):
    return sum(comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(k, n + 1))


def p_le(k, n, q=0.5):
    return sum(comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(0, k + 1))


def sign_p(w, n):
    """이김 다리의 단측 p (⛔n=0 에서 죽지 않는다)."""
    return p_ge(w, n, 0.5) if n else None


def sign_p_hurt(w, n):
    """해침 쪽 단측 p — 라벨이 아니라 `SIG_WORSE` 기록용."""
    return p_le(w, n, 0.5) if n else None


def k_one_sided(n, alpha=ALPHA):
    """단측 문턱 `min{k : P(W>=k|n,.5) <= alpha}` (없으면 `None`)."""
    c = [k for k in range(n + 1) if p_ge(k, n, 0.5) <= alpha]
    return min(c) if c else None


def k_two_sided(n, alpha=ALPHA_FAM):
    """★양측 문턱 `k⁺(n) = min{k : P(W>=k|n,.5) <= alpha/2}` — **동점 제외 후** `n_eff` 로 부른다.

    ⛔`None` 이면 그 `n` 에서는 **어떤 자료로도** 양측 기각이 불가능하다 ⟹ `D-UNPOWERED`.
    ★존재 조건은 `0.5^n <= alpha/2` ⟺ `n >= 7`(alpha=0.05) — ⛔옛 문서의 *"n_eff <= 11"* 은
      **산술 오류**였고 개정 R-5 가 `n_eff <= 6` 으로 고쳤다. 이 함수가 그 사실의 유일출처다.
    """
    c = [k for k in range(n + 1) if p_ge(k, n, 0.5) <= alpha / 2.0]
    return min(c) if c else None


def mde_q(k, n, power):
    """문턱 `w>=k` 를 주어진 검정력으로 넘기는 **최소 q** (이분법, 단조성 이용)."""
    lo, hi = 0.5, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if p_ge(k, n, mid) < power:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ── 문턱 (PREREG §3.1 · §4) — ★전부 **계산한다** ────────────────────────────
THRESH_F = k_one_sided(N_P, ALPHA)                 # 양성대조 (무보정 단측)
THRESH_S = k_one_sided(N_P, ALPHA_FAM)             # ★축 S (Bonferroni 단측)
THRESH_SECE = k_one_sided(N_P, ALPHA)              # `SEC-E` (무보정 단측)
THRESH_D = k_two_sided(N_P)                        # ★축 D (Bonferroni 양측) = k⁺(60)
SIG_WORSE = N_P - THRESH_S                         # 라벨 아님 · 기록만
N_EFF_UNPOWERED = max(n for n in range(1, N_P + 1) if k_two_sided(n) is None)
N_BLIND = 24                                       # `SEC-BLIND` (미접촉 층, 기술통계)
THRESH_BLIND = k_one_sided(N_BLIND, ALPHA)


def _mde_table():
    """★MDE 를 **계산**한다 — ⛔`PREREG §4` 의 수를 손으로 옮기지 않는다."""
    out = {}
    for tag, k, n in (('S', THRESH_S, N_P), ('D', THRESH_D, N_P), ('F', THRESH_F, N_P),
                      ('SECE', THRESH_SECE, N_P), ('BLIND', THRESH_BLIND, N_BLIND)):
        for lab, pw in (('q50', 0.50), ('q80', 0.80), ('q90', 0.90)):
            out[f'{tag}_{lab}'] = round(mde_q(k, n, pw), 4)
    # ⛔기각 불가 대역 — 「과반이지만 문턱 미달」
    out['unrejectable_S'] = (N_P // 2 + 1, THRESH_S - 1)
    out['unrejectable_D'] = (N_P - THRESH_D + 1, THRESH_D - 1)
    return out


MDE = _mde_table()
Q80_S, Q80_D, Q80_BLIND = MDE['S_q80'], MDE['D_q80'], MDE['BLIND_q80']
# ⛔**세 값을 섞지 마라**(`M8b` 선례) — `S` 널이면 S_q80, `D` 널이면 D_q80, `SEC-BLIND` 는 따로.


def n_eff_thresholds(n_max=N_P):
    """동점 제외 대비 — `n_eff` 별 `k⁺` 와 q80 (PREREG §4 표를 **재생산**한다)."""
    out = {}
    for n in range(1, n_max + 1):
        k = k_two_sided(n)
        out[n] = None if k is None else {'k_plus': k, 'q80': round(mde_q(k, n, 0.80), 4),
                                         'p_one_tail': p_ge(k, n, 0.5)}
    return out


# ── 등록 예측 (PREREG §3.5) ─────────────────────────────────────────────────
PRED_Q = {'F': 0.95, 'S': 0.92, 'SEC-E': 0.92, 'D': 0.50}
# ⛔`q_D` 는 0.60 에서 **내렸다**(개정 R-2) — 옛 값은 다른 모집단(`T11` n=24 `mixed_data2`)에서
#   왔고 §9.2-3 이 그 비교를 스스로 금지한다. 0.50 은 *"정직한 동전던지기"* 가 아니라
#   ★**「처치가 자보다 작다」의 결과**다(§3.6-4 · `REACHABILITY §4.4`).
PRED_T = {'T1': 0.95, 'T2': 0.80}   # 결정적 통계량에 대한 예측 — ⚠이항이 아니다

# ── ⛔`PREREG` 가 인쇄한 수는 **핀**으로만 들어간다 ─────────────────────────
#   `registration_drift()` 가 위의 **계산값**과 대조하고, 분석기 `G-COV` 가 그것으로 죽는다.
#   ⟹ 문서와 코드가 갈리면 **배치가 라벨을 못 낸다**. (⛔반대로 문서를 코드에 맞춰 고치지 마라 —
#   사전등록은 결과 0행에서 동결됐다.)
REGISTERED = {
    'THRESH_F': 37, 'THRESH_S': 39, 'THRESH_SECE': 37, 'THRESH_D': 40,
    'SIG_WORSE': 21, 'N_EFF_UNPOWERED': 6, 'THRESH_BLIND': 17,
    'N_P': 60, 'N_ROWS': 366, 'N_RUNS': 304, 'BEAMFORM_CALLS_PER_FILE': 16,
    'S_q50': 0.6409, 'S_q80': 0.6915, 'S_q90': 0.7169,
    'D_q50': 0.6575, 'D_q80': 0.7074, 'D_q90': 0.7323,
    'F_q80': 0.6596, 'SECE_q80': 0.6596, 'BLIND_q80': 0.7597,
    'unrejectable_S': (31, 38), 'unrejectable_D': (21, 39),
    'S_MIN': 0.25, 'S_MAX': 4.0, 'L_TRAIN': 220.55, 'K_TRAIN': 0.06779,
    'DRIVE_T1_LN': 0.048790, 'DRIVE_T2_SIGN': 55, 'N1': 120, 'ZOOM': 3.5,
    'KLIM1_OLD': 2.0, 'FLOOR': 1e-3, 'TIE_REL': 1e-6,
}


def registration_drift():
    """★사전등록이 인쇄한 수 ↔ 이 파일이 계산한 수. 어긋난 항목의 목록(빈 목록 = 일치)."""
    live = {'THRESH_F': THRESH_F, 'THRESH_S': THRESH_S, 'THRESH_SECE': THRESH_SECE,
            'THRESH_D': THRESH_D, 'SIG_WORSE': SIG_WORSE,
            'N_EFF_UNPOWERED': N_EFF_UNPOWERED, 'THRESH_BLIND': THRESH_BLIND,
            'N_P': N_P, 'N_ROWS': N_ROWS, 'N_RUNS': N_RUNS,
            'BEAMFORM_CALLS_PER_FILE': BEAMFORM_CALLS_PER_FILE,
            'S_MIN': S_MIN, 'S_MAX': S_MAX, 'L_TRAIN': L_TRAIN, 'K_TRAIN': K_TRAIN,
            'DRIVE_T1_LN': round(DRIVE_T1_LN, 6), 'DRIVE_T2_SIGN': DRIVE_T2_SIGN,
            'N1': N1, 'ZOOM': ZOOM, 'KLIM1_OLD': KLIM1_OLD, 'FLOOR': FLOOR,
            'TIE_REL': TIE_REL}
    live.update({k: MDE[k] for k in ('S_q50', 'S_q80', 'S_q90', 'D_q50', 'D_q80', 'D_q90',
                                     'F_q80', 'SECE_q80', 'BLIND_q80',
                                     'unrejectable_S', 'unrejectable_D')})
    bad = []
    for k, want in REGISTERED.items():
        got = live.get(k)
        if isinstance(want, tuple):
            ok = tuple(got) == want if got is not None else False
        elif isinstance(want, float):
            ok = got is not None and abs(got - want) <= 1e-9 * max(1.0, abs(want))
        else:
            ok = got == want
        if not ok:
            bad.append(f'{k}: 등록 {want} vs 계산 {got}')
    return bad


# ── 표본 ────────────────────────────────────────────────────────────────────
def archive_mats():
    """아카이브의 `.mat` 을 `sorted()` 한 목록 (`T11 NOTE.md §8` 의 인덱스 규약)."""
    import zipfile
    with zipfile.ZipFile(ZIP) as zf:
        return sorted(n for n in zf.namelist() if n.endswith('.mat'))


def tiers():
    """★층은 **계산한다**(⛔손목록 금지). Tier-P = `sorted()` 인덱스 0–59 = `Data/mixed_data/` 전수."""
    mats = archive_mats()
    tier_p = sorted(mats[POP_RANGE[0]:POP_RANGE[1]])
    return tier_p, list(SMOKE_FILES)


def frozen_sample():
    """동결된 표본 파일(측정면). 없으면 `None` — `G-POP` 이 그것으로 FAIL 한다."""
    if not os.path.exists(SAMPLE_T16):
        return None
    return json.load(open(SAMPLE_T16, encoding='utf-8'))


def seed_of(block, tier, tier_p=None):
    """★관측 추첨 시드. Tier-P = `SEED16 + fi`(1-based, `sorted()` 순) · ANCHOR = `T11` 시드."""
    if tier == 'ANCHOR':
        return ANCHOR_SEEDS.get(block)
    tp = tiers()[0] if tier_p is None else tier_p
    return SEED16 + tp.index(block) + 1


def _archive_refs(path):
    """파일 안의 `Data/mixed_data/*.mat` 문자열 집합 (형제 산출물 스캔용)."""
    import re
    try:
        txt = open(path, encoding='utf-8', errors='replace').read()
    except OSError:
        return set()
    return set(re.findall(r'Data/mixed_data/[A-Za-z0-9._]+\.mat', txt))


def _archive_refs_any(path):
    """파일 안의 **아무** 아카이브 `.mat` 경로 집합 — `G-XBATCH` 의 양성 대조용."""
    import re
    try:
        txt = open(path, encoding='utf-8', errors='replace').read()
    except OSError:
        return set()
    return set(re.findall(r'Data/[A-Za-z0-9_]+/[A-Za-z0-9._]+\.mat', txt))


def xbatch_control():
    """★`G-XBATCH` 의 **양성 대조** (2026-09-05 신설 — `AMENDMENT_1 §13-1` 이 스스로 지적했다).

    `exposed_blocks()` 의 «Tier-P 교집합 0» 은 두 가지를 **구별하지 못한다**:
      (a) 형제가 `Data/mixed_data/` 를 정말로 안 건드렸다  ← 우리가 주장하려는 것
      (b) 그 산출물에 **블록 이름이 애초에 없다**          ← 그때 게이트는 **항진**이다
    ⟹ 각 형제 산출물이 아카이브 경로를 **하나 이상** 담고 있는지 센다.
    ⛔0건인 파일이 하나라도 있으면 `G-XBATCH` 는 FAIL 한다 — 「없음」을 「겹침 없음」으로
      읽지 않기 위해서다. ★이 대조는 게이트를 **엄하게만** 만든다.
    반환 = (파일별 참조 수, 0건 파일 목록).
    """
    base = os.path.dirname(HERE)
    counts, empty = {}, []
    for d, fn in XBATCH_OUTCOME:
        rel = os.path.join(d, fn)
        p = os.path.join(base, rel)
        n = len(_archive_refs_any(p)) if os.path.exists(p) else 0
        counts[rel] = n
        if n == 0:
            empty.append(rel)
    return counts, empty


def exposed_blocks():
    """★`mixed_data` 에 손댄 형제 배치의 **산출물을 직접 열어** 노출 집합을 만든다.

    반환 `(outcome, property_, missing)`:
      · `outcome`  = 추정기 팔이 **채점된** 블록 (`T11`~`T14` 동결 jsonl) ⟹ `G-XBATCH`(차단)
      · `property_`= 비선형성·분산관계 **속성만** 잰 블록 (`T1`·`T1b`·`T10`) ⟹ `SEC-BLIND`
      · `missing`  = 열지 못한 참조 파일 ⛔**부재를 통과로 세지 않는다**(`T11 A2 §7` 선례)
    ⛔형제 목록은 `PREREG §2.1` 이 등록한 것이다 — 조용히 더하거나 빼지 마라.
    ⛔이 배치 자신의 산출물은 스캔하지 않는다(그러면 blind 층이 스스로 사라진다).
    """
    import glob
    base = os.path.dirname(HERE)
    outcome, missing = set(), []
    for d, fn in XBATCH_OUTCOME:
        p = os.path.join(base, d, fn)
        if os.path.exists(p):
            outcome |= _archive_refs(p)
        else:
            missing.append(os.path.join(d, fn))
    prop = set()
    for d in PROPERTY_BATCH_DIRS:
        dd = os.path.join(base, d)
        if not os.path.isdir(dd):
            missing.append(d)
            continue
        for p in sorted(glob.glob(os.path.join(dd, '*.json'))
                        + glob.glob(os.path.join(dd, '*.jsonl'))):
            prop |= _archive_refs(p)
    return outcome, prop - outcome, missing


def blind_files(tier_p):
    """`SEC-BLIND` 의 **미접촉 24** — ⛔손목록이 아니라 형제 산출물 스캔의 여집합이다."""
    outcome, prop, _ = exposed_blocks()
    return sorted(set(tier_p) - outcome - prop)


def angle_hs(block):
    import re
    m = re.search(r'angle(\d+)h([\d.]+)\.mat', block)
    return (int(m.group(1)), float(m.group(2))) if m else (None, None)


def angle30_files(tier_p):
    """`SEC-BLIND-ANGLE30` — 교락 대조군(`angle ∈ {40,50,60}` 전 30파일, §2.1 · R-12)."""
    return sorted(b for b in tier_p if angle_hs(b)[0] in (40, 50, 60))


# ── 규칙이 소비자에게 넘겨야 하는 값 (분석기가 **독립 재계산**한다) ──────────
def klim1_new(l_obs):
    """★새 규칙의 1단 격자 반경 `0.5·(N1−1)·2π/L_obs` ⟹ `Δk₁ = 2π/L_obs`(Rayleigh 한계)."""
    return 0.5 * (N1 - 1) * (TWO_PI / l_obs)


def dk1_new(l_obs):
    return TWO_PI / l_obs


def klim1_expected(arm, l_obs):
    """팔별 1단 `klim` — ⛔러너 리터럴이 아니라 **행의 `L_obs` 에서 분석기가 다시 만든다**."""
    if arm == 'AUTO-BANDR':
        return klim1_new(l_obs)
    if arm == 'AUTO-BAND':
        return KLIM1_OLD
    return None                 # `AUTO-AP` 는 빔포머를 부르지 않는다


def dk1_expected(arm, l_obs):
    if arm == 'AUTO-BANDR':
        return dk1_new(l_obs)
    if arm == 'AUTO-BAND':
        return DK1_OLD
    return None


# ── 판정 사다리 (PREREG §3) — ★**2축** · `UNDECIDED` 없음 ───────────────────
S_LABELS = ('AUTOSCALE-BEATS-ZERO-BOTH-RULES',
            'AUTOSCALE-BEATS-ZERO-BANDR-ONLY',
            'NO-AUTOSCALE-SKILL-DETECTED')
D_LABELS = ('D-STAGE1-GRID-CHANGES-LANDING', 'D-STAGE1-GRID-WORSENS-LANDING',
            'D-TREATMENT-BELOW-SCALE', 'D-DIRECTION-MIXED',
            'D-NO-DIFFERENCE-DETECTED', 'D-UNPOWERED')
ANCHOR_LABEL = 'ANCHOR-DID-NOT-REPLICATE'


def label_S(w_s, w_e):
    """축 S (기술 축, 단측) — ⛔`SEC-E` 없이 인쇄 금지(§3.2 · §9.2-6b).

    ★`S` = `AUTO-BANDR < ZERO` 는 **처치를 판별하지 못한다**. 옛 규칙의 `s_band` 는 새 규칙과
    1% 차라 `SEC-E` 도 거의 확실히 발화하고, 손으로 고른 `FIX2` 도 발화한다 ⟹ `S` 가 말하는
    것은 「`s ∈ [1.2, 2.2]` 면 제로필드를 이긴다」이지 ***「밴드 수리가 이긴다」가 아니다***.
    ⛔`NO-AUTOSCALE-SKILL-DETECTED` 는 **미검출**이다 — *"자동 스케일은 불가능"*·*"도달 불가"*
    로 읽는 것을 금지하고 MDE `q80` 병기가 의무다(§3.4).
    """
    if w_s >= THRESH_S:
        if w_e >= THRESH_SECE:
            return 'AUTOSCALE-BEATS-ZERO-BOTH-RULES'
        return 'AUTOSCALE-BEATS-ZERO-BANDR-ONLY'        # ⚠`SEC-E` 무보정 ⟹ EXPLORATORY
    return 'NO-AUTOSCALE-SKILL-DETECTED'


def label_D(w_d, n_eff, t1, t2):
    """축 D (착지 축, 양측) — ★**널을 구조적 사실로 이름 붙인다**(§3.3).

    사전등록 발화 순서(⛔결과를 보고 순서·문턱을 바꾸지 않는다):
      0. `k⁺(n_eff)` 부재(= `n_eff <= 6`) → `D-UNPOWERED`
      1. `W >= k⁺`                        → `D-STAGE1-GRID-CHANGES-LANDING`
      2. `n_eff − W >= k⁺`                → `D-STAGE1-GRID-WORSENS-LANDING`
      3. 널 ∧ T1                          → `D-TREATMENT-BELOW-SCALE`
      4. 널 ∧ ¬T1 ∧ T2                    → `D-DIRECTION-MIXED`
      5. 널 ∧ ¬T1 ∧ ¬T2                   → `D-NO-DIFFERENCE-DETECTED`
    ⛔`D-TREATMENT-BELOW-SCALE` 를 `D-NO-DIFFERENCE-DETECTED` 로 접지 마라(`M8b` 선례) —
      **왜 널인지가 라벨의 내용**이다. ⛔*"차이가 없다"* 로도 읽지 마라.
    ⛔`D-UNPOWERED`·`D-*-DETECTED` 는 **미검출**이고, `D-TREATMENT-BELOW-SCALE`·
      `D-DIRECTION-MIXED` 는 **구조**다 — 셋 다 양성 주장이 아니다.
    ★2번 다리를 `W <= n_eff − k⁺` 가 아니라 `n_eff − W >= k⁺` 로 적는다: **반대 방향의
      이김 수**를 세는 것이 이 검정의 정직한 서술이고, 그래야 「양성 이름이 기각-실패 다리에
      붙었는가」를 보는 정적 검사가 이 다리를 옳게 읽는다.
    """
    kplus = k_two_sided(n_eff)
    if kplus is None:                        # ⛔어떤 자료로도 양측 기각 불가 (n_eff <= 6)
        return 'D-UNPOWERED'
    if w_d >= kplus:
        return 'D-STAGE1-GRID-CHANGES-LANDING'
    if (n_eff - w_d) >= kplus:
        return 'D-STAGE1-GRID-WORSENS-LANDING'
    if t1:
        return 'D-TREATMENT-BELOW-SCALE'
    if t2:
        return 'D-DIRECTION-MIXED'
    return 'D-NO-DIFFERENCE-DETECTED'


def label(w_f, w_s, w_e, w_d, n_eff, t1, t2):
    """★두 축을 합친 인쇄 라벨. ⛔양성대조 `F` 가 거짓이면 **다른 라벨을 인쇄하지 않는다**(R10).

    ⛔어떤 라벨도 *"우리 방법이 저들 방법보다 낫다"* 로 읽지 마라 — 외부 기준선 행이 **0개**다.
    `AUTOSCALE-BEATS-ZERO-*` 의 비교자는 **제로필드**이지 제3자 방법이 아니다(§9.2-1).
    """
    if w_f < THRESH_F:
        return ANCHOR_LABEL
    return f'{label_S(w_s, w_e)} ∧ {label_D(w_d, n_eff, t1, t2)}'


def label_probs(q=None, t=None, n=None):
    """★등록 예측 하의 **모든 라벨 확률** — ⛔손타이핑 금지, **계산한다**(§3.5 · §12-3).

    ⚠**종속 공시**: `S`·`D`·`F`·`SEC-E` 는 같은 60파일·같은 관측점을 공유하므로 독립이 아니다.
    아래는 곱셈으로 계산했고 ⛔**사후 검정력 주장에 쓰지 마라**.
    ★독립 가정이 **한 방향으로** 틀린다: 측정된 착지차가 1% 라 `S` 와 `SEC-E` 는 거의 공단조이고,
      그때 `AUTOSCALE-BEATS-ZERO-BANDR-ONLY` 의 확률은 **정확히 0** 이다 ⟹ 아래 값은 **상한**이다.
    """
    q = dict(PRED_Q) if q is None else dict(q)
    t = dict(PRED_T) if t is None else dict(t)
    n = N_P if n is None else n
    k_f, k_s, k_e = k_one_sided(n, ALPHA), k_one_sided(n, ALPHA_FAM), k_one_sided(n, ALPHA)
    k_d = k_two_sided(n)
    p_f = p_ge(k_f, n, q['F'])
    p_s = p_ge(k_s, n, q['S'])
    p_e = p_ge(k_e, n, q['SEC-E'])
    if k_d is None:                       # ⛔n 이 너무 작아 D 축이 원리적으로 판정 불가
        p_up = p_dn = 0.0
    else:
        p_up, p_dn = p_ge(k_d, n, q['D']), p_le(n - k_d, n, q['D'])
    p_null = 1.0 - p_up - p_dn
    s_ax = {'AUTOSCALE-BEATS-ZERO-BOTH-RULES': p_s * p_e,
            'AUTOSCALE-BEATS-ZERO-BANDR-ONLY': p_s * (1 - p_e),
            'NO-AUTOSCALE-SKILL-DETECTED': 1 - p_s}
    d_ax = {'D-STAGE1-GRID-CHANGES-LANDING': p_up,
            'D-STAGE1-GRID-WORSENS-LANDING': p_dn,
            'D-TREATMENT-BELOW-SCALE': p_null * t['T1'],
            'D-DIRECTION-MIXED': p_null * (1 - t['T1']) * t['T2'],
            'D-NO-DIFFERENCE-DETECTED': p_null * (1 - t['T1']) * (1 - t['T2']),
            'D-UNPOWERED': 0.0 if k_d is not None else 1.0}
    joint = {f'{a} ∧ {b}': p_f * pa * pb for a, pa in s_ax.items() for b, pb in d_ax.items()}
    joint[ANCHOR_LABEL] = 1 - p_f
    return {'q': q, 't': t, 'n': n,
            'thresholds': {'F': k_f, 'S': k_s, 'SEC-E': k_e, 'D': k_d},
            'P_F': p_f, 'P_S': p_s, 'P_SECE': p_e,
            'P_D_up': p_up, 'P_D_down': p_dn, 'P_D_null': p_null,
            'S_axis_given_F': s_ax, 'D_axis_given_F': d_ax,
            'joint': joint, 'joint_sum': sum(joint.values()),
            '_note': ('⚠다리 독립 가정 · S 와 SEC-E 는 공단조 ⟹ BANDR-ONLY 는 상한이다 · '
                      '⛔사후 검정력 주장에 쓰지 마라')}


def null_label_probs(n=None):
    """귀무 `q=0.5` 아래 오라벨률 (§3.5-5)."""
    return label_probs(q={'F': 0.5, 'S': 0.5, 'SEC-E': 0.5, 'D': 0.5}, n=n)


def ladder_table(n=N_P):
    """★사다리를 **전 조합에서** 돌려 본다 — `UNDECIDED` 구멍이 없음을 기계로 보인다.

    ⛔한 축이 지는 조합도 반드시 **라벨**을 낸다(`T15` 가 이 시험으로 「한 층이 지면 회귀 라벨」을
    잡았다). 반환 = `(설명, 라벨)` 목록.
    """
    kd = k_two_sided(n)
    rows = []
    for w_f in (THRESH_F - 1, THRESH_F):
        for w_s in (THRESH_S - 1, THRESH_S):
            for w_e in (THRESH_SECE - 1, THRESH_SECE):
                # `D` 축의 여섯 가지: 위쪽 · 아래쪽 · 널×(T1) · 널×(¬T1∧T2) · 널×(¬T1∧¬T2) · 무검정력
                for w_d, n_eff, t1, t2 in ((kd, n, True, True),
                                           (n - kd, n, True, True),
                                           (n // 2, n, True, True),
                                           (n // 2, n, False, True),
                                           (n // 2, n, False, False),
                                           (N_EFF_UNPOWERED, N_EFF_UNPOWERED, True, True)):
                    rows.append(({'w_F': w_f, 'w_S': w_s, 'w_SECE': w_e, 'w_D': w_d,
                                  'n_eff': n_eff, 'T1': t1, 'T2': t2},
                                 label(w_f, w_s, w_e, w_d, n_eff, t1, t2)))
    # ★경계 하나 더: `n_eff = 7` 에서는 `k⁺` 가 **존재한다**(⛔옛 문서의 "n_eff <= 11" 은 오류였다)
    rows.append(({'w_F': THRESH_F, 'w_S': THRESH_S, 'w_SECE': THRESH_SECE, 'w_D': 7,
                  'n_eff': N_EFF_UNPOWERED + 1, 'T1': True, 'T2': True},
                 label(THRESH_F, THRESH_S, THRESH_SECE, 7, N_EFF_UNPOWERED + 1, True, True)))
    return rows


# ── 차단 (PREREG §6) ────────────────────────────────────────────────────────
# ★**등록된 게이트 이름 전부**가 여기 있다 — 기록 전용도 포함한다.
#   ⛔그것이 「기록 전용이 막는다」는 뜻은 **아니다**: 기록 전용 게이트의 조건은 분석기에서
#   리터럴 `True` 라 **구조적으로 FAIL 할 수 없다** ⟹ `blocked_by` 가 그 이름을 볼 일이 없다.
#   이름을 넣어 두는 이유는 반대다 — 나중에 누가 조건을 붙이면 **조용히 장식이 되지 않는다**.
#   (`T5-ENS` 는 차단 게이트 14개가 이름 비교 결함으로 전부 장식이 될 뻔했다.)
GATES_BLOCKING = (
    # §6.1 규칙의 well-posedness
    'G-EQUIVAR', 'G-EQUIVAR-RAW', 'G-KLIM1', 'G-KLIM1-LIVE', 'G-FLOOR', 'G-EDGE',
    'G-INNER', 'G-S-BLIND', 'G-S-SRC', 'G-S-READBACK', 'G-RULEID', 'G-LTRAIN',
    'G-KTRAIN', 'G-SRC-PIN',
    # §6.2 표본·교차배치
    'G-XBATCH', 'G-POP', 'G-SAMPLE', 'G-GRID',
    # §5 역변환
    'G-INVERT', 'G-PURE', 'G-DEEP',
    # §6.3 T11·T14 계승
    'G-N', 'G-FINITE', 'G-SIZE', 'G-NOLEAK', 'G-ATOMS', 'G-CAP', 'G-ARM', 'G-WRAPPER',
    'G-SHIPID', 'G-ANCHOR-T11', 'G-DEPLOY', 'G-TORIGIN', 'G-DENSITY', 'G-ENV',
    'G-BLOCK', 'G-COV',
)
GATES_RECORD = ('G-EQUIVAR-OLD', 'G-BANDRES', 'G-SIBLING', 'G-TP',
                'G-SCLIP', 'G-SVAR', 'G-CLAMP', 'G-DRIVE')
GATE_NAMES = GATES_BLOCKING + GATES_RECORD


def blocked_by(rows):
    """⛔**긴 이름부터** 접두 비교.

    ★이 규약이 없으면 `G-EQUIVAR-RAW` 가 `G-EQUIVAR` 로, `G-NOLEAK` 이 `G-N` 으로 접힌다
    (`T5-ENS` 는 **이름 전체를 비교해** 차단 게이트 14개를 장식으로 만들 뻔했다).
    """
    order = sorted(set(GATE_NAMES), key=len, reverse=True)
    out = []
    for name, ok, _ in rows:
        if ok:
            continue
        nm = name.replace('★', '').replace('⚠', '').replace('⛔', '').strip()
        for b in order:
            if nm.startswith(b):
                out.append(b)
                break
    return out


def selfcheck_blocked_by(names=None):
    """★자가진단 — 등록된 **모든** 이름이 합성 위반 튜플로 실제로 잡히는지.

    반환 `(못 잡은 이름, 다른 이름으로 접힌 것)`. ⛔둘 다 비어야 한다(`G-BLOCK`).
    """
    names = list(GATE_NAMES) if names is None else list(names)
    missed, folded = [], []
    for nm in names:
        got = blocked_by([(f'★{nm} 합성 위반', False, '')])
        if not got:
            missed.append(nm)
        elif got[0] != nm:
            folded.append(f'{nm}→{got[0]}')
    if blocked_by([('★G-COV 합성 통과', True, '')]):
        missed.append('<통과 행이 차단됐다>')
    return missed, folded
